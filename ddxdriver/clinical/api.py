import asyncio
import json
import logging
import os
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

import tornado.httpserver
import tornado.ioloop
import tornado.web

from .auth import (
    AuthConfig,
    AuthIdentity,
    AuthenticationError,
    AuthorizationError,
    AuthProvider,
    authorize,
    build_auth_provider,
)
from .config import create_clinical_session, load_clinical_config
from .lifecycle import SessionLifecycleConfig
from .observability import (
    actor_reference,
    build_error_event,
    build_event,
    configure_observability,
    emit_event,
    request_id_from_header,
    resource_reference,
    session_reference,
)
from .persistence import (
    DEFAULT_OWNER_SUBJECT,
    ClinicalRepository,
    SessionArchivedError,
    SessionExpiredError,
    build_clinical_repository,
)
from .security import (
    SESSION_CREATE_PATH,
    RateLimiter,
    SecurityConfig,
    build_rate_limiter,
    client_rate_subject,
    has_json_content_type,
    identity_rate_subject,
    is_model_action,
    is_protected_api_path,
    origin_is_allowed,
    session_id_from_path,
    session_rate_subject,
    validate_case_id,
    validate_patient_id,
    validate_required_text,
)


CASE_STATUSES = frozenset({"draft", "ready", "active", "completed", "error"})


def missing_runtime_environment(config: dict) -> list[str]:
    """Return environment variables required by model classes in the active config."""
    model_classes: set[str] = set()
    for section in ("ddxdriver", "history_taking", "diagnosis", "rag"):
        model_cfg = config.get(section, {}).get("config", {}).get("model", {})
        class_name = model_cfg.get("class_name")
        if isinstance(class_name, str):
            model_classes.add(class_name)

    required: set[str] = set()
    if any(name.endswith(".OpenAIChat") for name in model_classes):
        required.add("OAI_KEY")
    if any(name.endswith(".OpenAIAzureChat") for name in model_classes):
        required.update({"OAI_KEY", "AZURE_ENDPOINT"})

    return sorted(name for name in required if not os.getenv(name))


class ClinicalSessionStore:
    """Resumable owner-scoped clinical sessions with a process-local hot cache."""

    def __init__(
        self,
        config: dict,
        repository: ClinicalRepository | None = None,
        lifecycle_config: SessionLifecycleConfig | None = None,
    ):
        self.config = config
        self.repository = repository or build_clinical_repository()
        self.lifecycle_config = lifecycle_config or SessionLifecycleConfig.from_env()
        self.sessions: dict[str, object] = {}
        self.session_owners: dict[str, str] = {}
        self.session_expires_at: dict[str, datetime | None] = {}
        self.locks: dict[str, asyncio.Lock] = {}
        self._load_lock = asyncio.Lock()
        self._last_cleanup = 0.0

    async def _maybe_cleanup(self) -> None:
        now = time.monotonic()
        if now - self._last_cleanup < self.lifecycle_config.cleanup_interval_seconds:
            return
        self._last_cleanup = now
        await asyncio.to_thread(self.repository.cleanup_expired)

    def _cache_expired(self, session_id: str) -> bool:
        expires_at = self.session_expires_at.get(session_id)
        return expires_at is not None and expires_at <= datetime.now(timezone.utc)

    def _drop_cache(self, session_id: str) -> None:
        self.sessions.pop(session_id, None)
        self.session_owners.pop(session_id, None)
        self.session_expires_at.pop(session_id, None)
        self.locks.pop(session_id, None)

    async def _validate_persisted_lifecycle(
        self,
        session_id: str,
        owner_subject: str,
    ) -> None:
        """Keep persisted expiry/archive state authoritative for hot-cache sessions.

        TTL is disabled by default. When enabled, a cheap repository read prevents
        a session restored after a process restart from receiving an accidental
        fresh in-memory TTL before the next successful mutation.
        """
        if self.lifecycle_config.ttl_hours == 0:
            return
        try:
            state = await asyncio.to_thread(
                self.repository.load,
                session_id,
                owner_subject=owner_subject,
            )
        except (SessionExpiredError, SessionArchivedError):
            self._drop_cache(session_id)
            raise
        if state is None:
            self._drop_cache(session_id)
            raise KeyError(session_id)

    async def create(
        self,
        patient_initial_info: str,
        patient_id=None,
        owner_subject: str = DEFAULT_OWNER_SUBJECT,
    ) -> tuple[str, object]:
        await self._maybe_cleanup()
        session_id = str(uuid.uuid4())
        session = await asyncio.to_thread(
            create_clinical_session,
            patient_initial_info,
            patient_id if patient_id is not None else session_id,
            self.config,
        )
        self.sessions[session_id] = session
        self.session_owners[session_id] = owner_subject
        self.locks[session_id] = asyncio.Lock()
        try:
            await self.persist(session_id, session, owner_subject=owner_subject)
        except Exception:
            self._drop_cache(session_id)
            raise
        return session_id, session

    async def get(
        self,
        session_id: str,
        owner_subject: str = DEFAULT_OWNER_SUBJECT,
    ):
        await self._maybe_cleanup()
        session = self.sessions.get(session_id)
        if session is not None:
            if self.session_owners.get(session_id) != owner_subject:
                raise KeyError(session_id)
            await self._validate_persisted_lifecycle(session_id, owner_subject)
            if self._cache_expired(session_id):
                self._drop_cache(session_id)
                await asyncio.to_thread(
                    self.repository.delete,
                    session_id,
                    owner_subject=owner_subject,
                )
                raise SessionExpiredError(session_id)
            return session

        async with self._load_lock:
            session = self.sessions.get(session_id)
            if session is not None:
                if self.session_owners.get(session_id) != owner_subject:
                    raise KeyError(session_id)
                await self._validate_persisted_lifecycle(session_id, owner_subject)
                return session

            state = await asyncio.to_thread(
                self.repository.load,
                session_id,
                owner_subject=owner_subject,
            )
            if state is None:
                raise KeyError(session_id)

            patient_state = state.get("patient")
            if not isinstance(patient_state, dict):
                raise ValueError("Persisted clinical session is missing patient state")
            patient_initial_info = patient_state.get("patient_initial_info")
            if not isinstance(patient_initial_info, str) or not patient_initial_info.strip():
                raise ValueError("Persisted clinical session has invalid patient initial information")
            patient_id = patient_state.get("patient_id", session_id)

            session = await asyncio.to_thread(
                create_clinical_session,
                patient_initial_info,
                patient_id,
                self.config,
            )
            restore = getattr(session, "restore_persistence_state", None)
            if not callable(restore):
                raise TypeError("Clinical session implementation cannot restore persisted state")
            await asyncio.to_thread(restore, state)

            self.sessions[session_id] = session
            self.session_owners[session_id] = owner_subject
            # The persisted repository remains authoritative until a mutation
            # refreshes the TTL via persist(). Do not manufacture a fresh cache TTL here.
            self.session_expires_at[session_id] = None
            self.locks.setdefault(session_id, asyncio.Lock())
            return session

    def lock(
        self,
        session_id: str,
        owner_subject: str = DEFAULT_OWNER_SUBJECT,
    ) -> asyncio.Lock:
        if session_id not in self.sessions or self.session_owners.get(session_id) != owner_subject:
            raise KeyError(session_id)
        return self.locks.setdefault(session_id, asyncio.Lock())

    async def persist(
        self,
        session_id: str,
        session,
        owner_subject: str = DEFAULT_OWNER_SUBJECT,
    ) -> None:
        serialize = getattr(session, "persistence_state", None)
        if not callable(serialize):
            return
        state = await asyncio.to_thread(serialize)
        expires_at = self.lifecycle_config.expires_at()
        await asyncio.to_thread(
            self.repository.save,
            session_id,
            state,
            owner_subject=owner_subject,
            expires_at=expires_at,
        )
        self.session_expires_at[session_id] = expires_at

    async def archive(
        self,
        session_id: str,
        owner_subject: str = DEFAULT_OWNER_SUBJECT,
    ) -> bool:
        archived = await asyncio.to_thread(
            self.repository.archive,
            session_id,
            owner_subject=owner_subject,
        )
        if archived:
            self._drop_cache(session_id)
        return archived

    async def delete(
        self,
        session_id: str,
        owner_subject: str = DEFAULT_OWNER_SUBJECT,
    ) -> bool:
        deleted = await asyncio.to_thread(
            self.repository.delete,
            session_id,
            owner_subject=owner_subject,
        )
        if deleted:
            self._drop_cache(session_id)
        return deleted


class BaseHandler(tornado.web.RequestHandler):
    def initialize(self, store: ClinicalSessionStore):
        self.store = store
        self.request_id = request_id_from_header(self.request.headers.get("X-Request-ID"))
        self._request_started = time.monotonic()
        self.identity: AuthIdentity | None = None
        self.set_header("X-Request-ID", self.request_id)

    @property
    def security_config(self) -> SecurityConfig:
        return self.application.settings["meddx_security_config"]

    @property
    def rate_limiter(self) -> RateLimiter:
        return self.application.settings["meddx_rate_limiter"]

    @property
    def auth_provider(self) -> AuthProvider:
        return self.application.settings["meddx_auth_provider"]

    @property
    def auth_config(self) -> AuthConfig:
        return self.application.settings["meddx_auth_config"]

    @property
    def owner_subject(self) -> str:
        return self.identity.subject if self.identity is not None else DEFAULT_OWNER_SUBJECT

    def set_default_headers(self):
        security_config = self.application.settings.get("meddx_security_config")
        if not isinstance(security_config, SecurityConfig):
            security_config = SecurityConfig.from_env()

        origin = self.request.headers.get("Origin")
        if "*" in security_config.allowed_origins:
            self.set_header("Access-Control-Allow-Origin", "*")
        elif origin in security_config.allowed_origins:
            self.set_header("Access-Control-Allow-Origin", origin)
            self.set_header("Vary", "Origin")

        self.set_header(
            "Access-Control-Allow-Headers",
            "Authorization, Content-Type, X-Request-ID",
        )
        self.set_header("Access-Control-Allow-Methods", "GET,POST,PUT,DELETE,OPTIONS")
        self.set_header(
            "Access-Control-Expose-Headers",
            "X-Request-ID, Retry-After, X-RateLimit-Limit, X-RateLimit-Remaining",
        )
        self.set_header("Access-Control-Max-Age", "600")
        self.set_header("Cache-Control", "no-store, max-age=0")
        self.set_header("Pragma", "no-cache")
        self.set_header("X-Content-Type-Options", "nosniff")
        self.set_header("X-Frame-Options", "DENY")
        self.set_header("Referrer-Policy", "no-referrer")
        self.set_header("Permissions-Policy", "camera=(), microphone=(), geolocation=()")
        self.set_header(
            "Content-Security-Policy",
            "default-src 'none'; frame-ancestors 'none'; base-uri 'none'",
        )
        self.set_header("Content-Type", "application/json")

    async def prepare(self):
        origin = self.request.headers.get("Origin")
        if not origin_is_allowed(origin, self.security_config):
            await self.audit_event("request.origin", "rejected")
            self.write_api_error(403, "Origin is not allowed")
            return

        if len(self.request.body) > self.security_config.max_body_bytes:
            await self.audit_event("request.body", "rejected", body_too_large=True)
            self.write_api_error(413, "Request body is too large")
            return

        if self.request.method == "OPTIONS":
            return

        if is_protected_api_path(self.request.path):
            try:
                self.identity = self.auth_provider.authenticate(self.request.headers)
                authorize(self.identity, self.auth_config.required_roles)
            except AuthenticationError:
                await self.audit_event("auth.authenticate", "rejected")
                self.set_header("WWW-Authenticate", 'Bearer realm="MEDDxAgent"')
                self.write_api_error(401, "Authentication required")
                return
            except AuthorizationError:
                await self.audit_event("auth.authorize", "rejected")
                self.write_api_error(403, "Not authorized for this resource")
                return

            general_subject = (
                identity_rate_subject(self.identity.subject)
                if self.identity.authenticated
                else client_rate_subject(self.request.remote_ip)
            )
            general_decision = self.rate_limiter.check(
                general_subject,
                self.security_config.requests_per_minute,
            )
            if not general_decision.allowed:
                await self.reject_rate_limit("request", general_decision.retry_after_seconds)
                return

        if self.request.method == "POST" and self.request.path == SESSION_CREATE_PATH:
            create_subject = (
                identity_rate_subject(self.owner_subject) + ":session-create"
                if self.identity and self.identity.authenticated
                else client_rate_subject(self.request.remote_ip) + ":session-create"
            )
            create_decision = self.rate_limiter.check(
                create_subject,
                self.security_config.session_creates_per_minute,
            )
            if not create_decision.allowed:
                await self.reject_rate_limit(
                    "session-create",
                    create_decision.retry_after_seconds,
                    limit=create_decision.limit,
                )
                return

        if is_model_action(self.request.path, self.request.method):
            session_id = session_id_from_path(self.request.path)
            if session_id:
                model_decision = self.rate_limiter.check(
                    session_rate_subject(session_id) + ":model-action",
                    self.security_config.model_actions_per_minute,
                )
                if not model_decision.allowed:
                    await self.reject_rate_limit(
                        "model-action",
                        model_decision.retry_after_seconds,
                        session_id=session_id,
                        limit=model_decision.limit,
                    )
                    return

    async def reject_rate_limit(
        self,
        scope: str,
        retry_after_seconds: int,
        *,
        session_id: str | None = None,
        limit: int | None = None,
    ) -> None:
        self.set_header("Retry-After", str(max(1, retry_after_seconds)))
        if limit is not None:
            self.set_header("X-RateLimit-Limit", str(limit))
        self.set_header("X-RateLimit-Remaining", "0")
        await self.audit_event(
            "request.rate_limit",
            "rejected",
            session_id,
            rate_limit_scope=scope,
        )
        self.write_api_error(429, "Too many requests. Please retry shortly.")

    def on_finish(self):
        if not hasattr(self, "_request_started"):
            return
        duration_ms = round((time.monotonic() - self._request_started) * 1000, 1)
        level = logging.DEBUG if isinstance(self, HealthHandler) else logging.INFO
        emit_event(
            build_event(
                "http.request",
                request_id=self.request_id,
                actor_ref=actor_reference(self.identity.subject) if self.identity else None,
                handler=type(self).__name__,
                method=self.request.method,
                status=self.get_status(),
                duration_ms=duration_ms,
                outcome="success" if self.get_status() < 400 else "error",
            ),
            level=level,
        )

    def options(self, *args, **kwargs):
        self.set_status(204)
        self.finish()

    def json_body(self) -> dict:
        if not self.request.body:
            return {}
        if not has_json_content_type(self.request.headers.get("Content-Type")):
            raise ValueError("Content-Type must be application/json")
        try:
            payload = json.loads(self.request.body.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError("Request body must be valid JSON") from exc
        if not isinstance(payload, dict):
            raise ValueError("Request body must be a JSON object")
        return payload

    def write_api_error(self, status: int, message: str):
        self.set_status(status)
        self.finish({"error": message})

    async def require_authenticated_identity(self) -> bool:
        if self.identity is not None and self.identity.authenticated:
            return True
        await self.audit_event("auth.required", "rejected")
        self.set_header("WWW-Authenticate", 'Bearer realm="MEDDxAgent"')
        self.write_api_error(401, "Authenticated account required")
        return False

    def monitor_error(self, operation: str, error: BaseException, session_id: str | None = None):
        emit_event(
            build_error_event(
                "clinical.error",
                error,
                request_id=self.request_id,
                actor_ref=actor_reference(self.identity.subject) if self.identity else None,
                operation=operation,
                session_ref=session_reference(session_id),
            ),
            level=logging.ERROR,
        )

    async def audit_event(
        self,
        action: str,
        outcome: str,
        session_id: str | None = None,
        **metadata,
    ) -> None:
        level = (
            logging.ERROR
            if outcome == "error"
            else logging.WARNING
            if outcome in {"rejected", "backend_not_ready"}
            else logging.INFO
        )
        emit_event(
            build_event(
                "clinical.audit",
                request_id=self.request_id,
                actor_ref=actor_reference(self.identity.subject) if self.identity else None,
                action=action,
                outcome=outcome,
                session_ref=session_reference(session_id),
                **metadata,
            ),
            level=level,
        )

    async def run_session_action(
        self,
        session_id: str,
        action,
        audit_action: str,
        persist: bool = True,
    ):
        if len(session_id) > 128:
            await self.audit_event(audit_action, "not_found")
            self.write_api_error(404, "Clinical session not found")
            return None

        try:
            session = await self.store.get(session_id, owner_subject=self.owner_subject)
            lock = self.store.lock(session_id, owner_subject=self.owner_subject)
        except SessionExpiredError:
            await self.audit_event(audit_action, "expired", session_id)
            self.write_api_error(410, "Clinical session expired")
            return None
        except SessionArchivedError:
            await self.audit_event(audit_action, "archived", session_id)
            self.write_api_error(410, "Clinical session archived")
            return None
        except KeyError:
            await self.audit_event(audit_action, "not_found", session_id)
            self.write_api_error(404, "Clinical session not found")
            return None
        except (ValueError, TypeError) as exc:
            self.monitor_error(audit_action, exc, session_id)
            await self.audit_event(
                audit_action,
                "error",
                session_id,
                error_type=type(exc).__name__,
            )
            self.write_api_error(500, "Stored clinical session could not be restored")
            return None

        try:
            async with lock:
                result = await asyncio.to_thread(action, session)
                if persist:
                    await self.store.persist(
                        session_id,
                        session,
                        owner_subject=self.owner_subject,
                    )
                await self.audit_event(audit_action, "success", session_id)
                return result
        except (ValueError, RuntimeError, TypeError) as exc:
            await self.audit_event(
                audit_action,
                "rejected",
                session_id,
                error_type=type(exc).__name__,
            )
            self.write_api_error(400, str(exc))
            return None
        except Exception as exc:
            self.monitor_error(audit_action, exc, session_id)
            await self.audit_event(
                audit_action,
                "error",
                session_id,
                error_type=type(exc).__name__,
            )
            self.write_api_error(500, "MEDDxAgent request failed")
            return None


class HealthHandler(BaseHandler):
    async def get(self):
        self.finish({"status": "ok"})


class ReadinessHandler(BaseHandler):
    async def get(self):
        missing = missing_runtime_environment(self.store.config)
        if missing:
            self.set_status(503)
            self.finish({"status": "not_ready", "missing_environment": missing})
            return
        self.finish({"status": "ready"})


class SessionsHandler(BaseHandler):
    async def post(self):
        missing = missing_runtime_environment(self.store.config)
        if missing:
            await self.audit_event("session.create", "backend_not_ready")
            self.set_status(503)
            self.finish(
                {
                    "error": "MEDDxAgent backend is not ready",
                    "missing_environment": missing,
                }
            )
            return

        try:
            payload = self.json_body()
            patient_initial_info = validate_required_text(
                payload.get("patient_initial_info", ""),
                "patient_initial_info",
                self.security_config.max_patient_info_chars,
            )
            patient_id = validate_patient_id(
                payload.get("patient_id"),
                self.security_config.max_patient_id_chars,
            )
            session_id, session = await self.store.create(
                patient_initial_info,
                patient_id,
                owner_subject=self.owner_subject,
            )
        except (ValueError, TypeError) as exc:
            await self.audit_event(
                "session.create",
                "rejected",
                error_type=type(exc).__name__,
            )
            self.write_api_error(400, str(exc))
            return
        except Exception as exc:
            self.monitor_error("session.create", exc)
            await self.audit_event(
                "session.create",
                "error",
                error_type=type(exc).__name__,
            )
            self.write_api_error(500, "Unable to create MEDDxAgent clinical session")
            return

        await self.audit_event("session.create", "success", session_id)
        self.set_status(201)
        self.finish({"session_id": session_id, **session.snapshot()})


class SessionHandler(BaseHandler):
    async def get(self, session_id: str):
        result = await self.run_session_action(
            session_id,
            lambda session: session.snapshot(),
            audit_action="session.read",
            persist=False,
        )
        if result is not None:
            self.finish({"session_id": session_id, **result})

    async def delete(self, session_id: str):
        deleted = await self.store.delete(session_id, owner_subject=self.owner_subject)
        if not deleted:
            await self.audit_event("session.delete", "not_found", session_id)
            self.write_api_error(404, "Clinical session not found")
            return
        await self.audit_event("session.delete", "success", session_id)
        self.set_status(204)
        self.finish()


class SessionArchiveHandler(BaseHandler):
    async def post(self, session_id: str):
        archived = await self.store.archive(session_id, owner_subject=self.owner_subject)
        if not archived:
            await self.audit_event("session.archive", "not_found", session_id)
            self.write_api_error(404, "Clinical session not found")
            return
        await self.audit_event("session.archive", "success", session_id)
        self.finish({"status": "archived"})


class ContextHandler(BaseHandler):
    async def post(self, session_id: str):
        try:
            payload = self.json_body()
            patient_initial_info = validate_required_text(
                payload.get("patient_initial_info", ""),
                "patient_initial_info",
                self.security_config.max_patient_info_chars,
            )
        except (ValueError, TypeError) as exc:
            await self.audit_event(
                "context.update",
                "rejected",
                session_id,
                error_type=type(exc).__name__,
            )
            self.write_api_error(400, str(exc))
            return

        def update_context(session):
            session.update_patient_initial_info(patient_initial_info)
            return session.snapshot()

        result = await self.run_session_action(
            session_id,
            update_context,
            audit_action="context.update",
        )
        if result is not None:
            self.finish({"session_id": session_id, **result})


class NextQuestionHandler(BaseHandler):
    async def post(self, session_id: str):
        def generate(session):
            question = session.next_question()
            return question, session.snapshot()

        result = await self.run_session_action(
            session_id,
            generate,
            audit_action="history.question.generate",
        )
        if result is not None:
            question, snapshot = result
            self.finish({"session_id": session_id, "question": question, **snapshot})


class AnswerHandler(BaseHandler):
    async def post(self, session_id: str):
        try:
            payload = self.json_body()
            answer = validate_required_text(
                payload.get("answer", ""),
                "answer",
                self.security_config.max_answer_chars,
            )
        except (ValueError, TypeError) as exc:
            await self.audit_event(
                "history.answer.submit",
                "rejected",
                session_id,
                error_type=type(exc).__name__,
            )
            self.write_api_error(400, str(exc))
            return

        def submit(session):
            session.submit_answer(answer)
            return session.snapshot()

        result = await self.run_session_action(
            session_id,
            submit,
            audit_action="history.answer.submit",
        )
        if result is not None:
            self.finish({"session_id": session_id, **result})


class FinishHistoryHandler(BaseHandler):
    async def post(self, session_id: str):
        def finish_history(session):
            session.finish_history()
            return session.snapshot()

        result = await self.run_session_action(
            session_id,
            finish_history,
            audit_action="history.finish",
        )
        if result is not None:
            self.finish({"session_id": session_id, **result})


class RunHandler(BaseHandler):
    async def post(self, session_id: str):
        def run(session):
            session.run()
            return session.snapshot()

        result = await self.run_session_action(
            session_id,
            run,
            audit_action="diagnosis.run",
        )
        if result is not None:
            self.finish({"session_id": session_id, **result})


def _validated_case_payload(payload: dict, path_case_id: str, max_case_id_chars: int) -> dict:
    case_id = validate_case_id(payload.get("id"), max_case_id_chars)
    if case_id != path_case_id:
        raise ValueError("Case payload id must match the request path")
    status = payload.get("status")
    if status not in CASE_STATUSES:
        raise ValueError("Case payload has an invalid status")
    patient = payload.get("patient")
    workflow = payload.get("workflow")
    if not isinstance(patient, dict) or not isinstance(workflow, dict):
        raise ValueError("Case payload must include patient and workflow objects")
    return payload


class CasesHandler(BaseHandler):
    async def get(self):
        if not await self.require_authenticated_identity():
            return
        try:
            limit = int(self.get_query_argument("limit", "100"))
            offset = int(self.get_query_argument("offset", "0"))
            cases = await asyncio.to_thread(
                self.store.repository.list_cases,
                self.owner_subject,
                limit=limit,
                offset=offset,
            )
        except ValueError as exc:
            self.write_api_error(400, str(exc))
            return
        except Exception as exc:
            self.monitor_error("case.list", exc)
            self.write_api_error(500, "Unable to load cases")
            return
        await self.audit_event("case.list", "success", count=len(cases))
        self.finish({"cases": cases})


class CaseHandler(BaseHandler):
    async def get(self, case_id: str):
        if not await self.require_authenticated_identity():
            return
        try:
            case_id = validate_case_id(case_id, self.security_config.max_case_id_chars)
            case = await asyncio.to_thread(
                self.store.repository.load_case,
                self.owner_subject,
                case_id,
            )
        except ValueError as exc:
            self.write_api_error(400, str(exc))
            return
        except Exception as exc:
            self.monitor_error("case.read", exc)
            self.write_api_error(500, "Unable to load case")
            return
        if case is None:
            await self.audit_event("case.read", "not_found", case_ref=resource_reference(case_id))
            self.write_api_error(404, "Case not found")
            return
        await self.audit_event("case.read", "success", case_ref=resource_reference(case_id))
        self.finish({"case": case})

    async def put(self, case_id: str):
        if not await self.require_authenticated_identity():
            return
        try:
            case_id = validate_case_id(case_id, self.security_config.max_case_id_chars)
            payload = _validated_case_payload(
                self.json_body(),
                case_id,
                self.security_config.max_case_id_chars,
            )
            case = await asyncio.to_thread(
                self.store.repository.save_case,
                self.owner_subject,
                case_id,
                payload,
            )
        except (ValueError, TypeError) as exc:
            await self.audit_event(
                "case.save",
                "rejected",
                case_ref=resource_reference(case_id),
                error_type=type(exc).__name__,
            )
            self.write_api_error(400, str(exc))
            return
        except Exception as exc:
            self.monitor_error("case.save", exc)
            self.write_api_error(500, "Unable to save case")
            return
        await self.audit_event("case.save", "success", case_ref=resource_reference(case_id))
        self.finish({"case": case})

    async def delete(self, case_id: str):
        if not await self.require_authenticated_identity():
            return
        try:
            case_id = validate_case_id(case_id, self.security_config.max_case_id_chars)
            deleted = await asyncio.to_thread(
                self.store.repository.delete_case,
                self.owner_subject,
                case_id,
            )
        except ValueError as exc:
            self.write_api_error(400, str(exc))
            return
        except Exception as exc:
            self.monitor_error("case.delete", exc)
            self.write_api_error(500, "Unable to delete case")
            return
        if not deleted:
            self.write_api_error(404, "Case not found")
            return
        await self.audit_event("case.delete", "success", case_ref=resource_reference(case_id))
        self.set_status(204)
        self.finish()


class CaseArchiveHandler(BaseHandler):
    async def post(self, case_id: str):
        if not await self.require_authenticated_identity():
            return
        try:
            case_id = validate_case_id(case_id, self.security_config.max_case_id_chars)
            archived = await asyncio.to_thread(
                self.store.repository.archive_case,
                self.owner_subject,
                case_id,
            )
        except ValueError as exc:
            self.write_api_error(400, str(exc))
            return
        except Exception as exc:
            self.monitor_error("case.archive", exc)
            self.write_api_error(500, "Unable to archive case")
            return
        if not archived:
            self.write_api_error(404, "Case not found")
            return
        await self.audit_event("case.archive", "success", case_ref=resource_reference(case_id))
        self.finish({"status": "archived"})


def make_app(
    config_path: str | Path | None = None,
    session_repository: ClinicalRepository | None = None,
    security_config: SecurityConfig | None = None,
    rate_limiter: RateLimiter | None = None,
    auth_config: AuthConfig | None = None,
    auth_provider: AuthProvider | None = None,
    lifecycle_config: SessionLifecycleConfig | None = None,
) -> tornado.web.Application:
    configure_observability()
    config = load_clinical_config(config_path)
    security_config = security_config or SecurityConfig.from_env()
    repository = session_repository or build_clinical_repository()
    auth_config = auth_config or (auth_provider.config if auth_provider else AuthConfig.from_env())
    auth_provider = auth_provider or build_auth_provider(auth_config)
    rate_limiter = rate_limiter or build_rate_limiter(repository, security_config)
    lifecycle_config = lifecycle_config or SessionLifecycleConfig.from_env()
    store = ClinicalSessionStore(
        config,
        repository=repository,
        lifecycle_config=lifecycle_config,
    )
    routes = [
        (r"/api/v1/health", HealthHandler, {"store": store}),
        (r"/api/v1/ready", ReadinessHandler, {"store": store}),
        (r"/api/v1/clinical/sessions", SessionsHandler, {"store": store}),
        (r"/api/v1/clinical/sessions/([^/]+)", SessionHandler, {"store": store}),
        (
            r"/api/v1/clinical/sessions/([^/]+)/archive",
            SessionArchiveHandler,
            {"store": store},
        ),
        (r"/api/v1/clinical/sessions/([^/]+)/context", ContextHandler, {"store": store}),
        (r"/api/v1/clinical/sessions/([^/]+)/question", NextQuestionHandler, {"store": store}),
        (r"/api/v1/clinical/sessions/([^/]+)/answer", AnswerHandler, {"store": store}),
        (
            r"/api/v1/clinical/sessions/([^/]+)/history/finish",
            FinishHistoryHandler,
            {"store": store},
        ),
        (r"/api/v1/clinical/sessions/([^/]+)/run", RunHandler, {"store": store}),
        (r"/api/v1/cases", CasesHandler, {"store": store}),
        (r"/api/v1/cases/([^/]+)", CaseHandler, {"store": store}),
        (r"/api/v1/cases/([^/]+)/archive", CaseArchiveHandler, {"store": store}),
    ]
    return tornado.web.Application(
        routes,
        debug=False,
        autoreload=False,
        serve_traceback=False,
        meddx_security_config=security_config,
        meddx_rate_limiter=rate_limiter,
        meddx_auth_config=auth_config,
        meddx_auth_provider=auth_provider,
    )


def main():
    configure_observability()
    config_path = os.getenv("MEDDX_CLINICAL_CONFIG")
    port = int(os.getenv("PORT", "8000"))
    security_config = SecurityConfig.from_env()
    auth_config = AuthConfig.from_env()
    lifecycle_config = SessionLifecycleConfig.from_env()
    app = make_app(
        config_path,
        security_config=security_config,
        auth_config=auth_config,
        lifecycle_config=lifecycle_config,
    )
    server = tornado.httpserver.HTTPServer(
        app,
        xheaders=False,
        decompress_request=False,
        max_body_size=security_config.max_body_bytes,
        max_header_size=security_config.max_header_bytes,
        body_timeout=security_config.body_timeout_seconds,
        idle_connection_timeout=security_config.idle_connection_timeout_seconds,
    )
    server.listen(port)
    emit_event(
        build_event(
            "service.started",
            port=port,
            auth_mode=auth_config.mode,
            rate_limit_backend=security_config.rate_limit_backend,
            max_body_bytes=security_config.max_body_bytes,
        )
    )
    tornado.ioloop.IOLoop.current().start()


if __name__ == "__main__":
    main()