import asyncio
import json
import logging
import os
import time
import uuid
from pathlib import Path

import tornado.httpserver
import tornado.ioloop
import tornado.web

from .config import create_clinical_session, load_clinical_config
from .observability import (
    build_error_event,
    build_event,
    configure_observability,
    emit_event,
    request_id_from_header,
    session_reference,
)
from .persistence import SQLiteClinicalSessionRepository
from .security import (
    CLINICAL_API_PREFIX,
    SESSION_CREATE_PATH,
    SecurityConfig,
    SlidingWindowRateLimiter,
    client_rate_subject,
    has_json_content_type,
    is_model_action,
    origin_is_allowed,
    session_id_from_path,
    session_rate_subject,
    validate_patient_id,
    validate_required_text,
)


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
    """Resumable clinical session storage with an in-process hot cache.

    Session state is serialized to SQLite after each successful mutation. Runtime
    model/agent objects remain in memory while the process is alive and are rebuilt
    from persisted state when a session is requested after a process restart.
    """

    def __init__(self, config: dict, repository: SQLiteClinicalSessionRepository | None = None):
        self.config = config
        self.repository = repository or SQLiteClinicalSessionRepository()
        self.sessions: dict[str, object] = {}
        self.locks: dict[str, asyncio.Lock] = {}
        self._load_lock = asyncio.Lock()

    async def create(self, patient_initial_info: str, patient_id=None) -> tuple[str, object]:
        session_id = str(uuid.uuid4())
        session = await asyncio.to_thread(
            create_clinical_session,
            patient_initial_info,
            patient_id if patient_id is not None else session_id,
            self.config,
        )
        self.sessions[session_id] = session
        self.locks[session_id] = asyncio.Lock()
        try:
            await self.persist(session_id, session)
        except Exception:
            self.sessions.pop(session_id, None)
            self.locks.pop(session_id, None)
            raise
        return session_id, session

    async def get(self, session_id: str):
        session = self.sessions.get(session_id)
        if session is not None:
            return session

        async with self._load_lock:
            session = self.sessions.get(session_id)
            if session is not None:
                return session

            state = await asyncio.to_thread(self.repository.load, session_id)
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
            self.locks.setdefault(session_id, asyncio.Lock())
            return session

    def lock(self, session_id: str) -> asyncio.Lock:
        if session_id not in self.sessions:
            raise KeyError(session_id)
        return self.locks.setdefault(session_id, asyncio.Lock())

    async def persist(self, session_id: str, session) -> None:
        serialize = getattr(session, "persistence_state", None)
        if not callable(serialize):
            return
        state = await asyncio.to_thread(serialize)
        await asyncio.to_thread(self.repository.save, session_id, state)


class BaseHandler(tornado.web.RequestHandler):
    def initialize(self, store: ClinicalSessionStore):
        self.store = store
        self.request_id = request_id_from_header(self.request.headers.get("X-Request-ID"))
        self._request_started = time.monotonic()
        self.set_header("X-Request-ID", self.request_id)

    @property
    def security_config(self) -> SecurityConfig:
        return self.application.settings["meddx_security_config"]

    @property
    def rate_limiter(self) -> SlidingWindowRateLimiter:
        return self.application.settings["meddx_rate_limiter"]

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

        self.set_header("Access-Control-Allow-Headers", "Content-Type, X-Request-ID")
        self.set_header("Access-Control-Allow-Methods", "GET,POST,OPTIONS")
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

        if self.request.method == "OPTIONS" or not self.request.path.startswith(
            CLINICAL_API_PREFIX
        ):
            return

        general_decision = self.rate_limiter.check(
            client_rate_subject(self.request.remote_ip),
            self.security_config.requests_per_minute,
        )
        if not general_decision.allowed:
            await self.reject_rate_limit("client", general_decision.retry_after_seconds)
            return

        if self.request.method == "POST" and self.request.path == SESSION_CREATE_PATH:
            create_decision = self.rate_limiter.check(
                client_rate_subject(self.request.remote_ip) + ":session-create",
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

    def monitor_error(self, operation: str, error: BaseException, session_id: str | None = None):
        emit_event(
            build_error_event(
                "clinical.error",
                error,
                request_id=self.request_id,
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
            session = await self.store.get(session_id)
            lock = self.store.lock(session_id)
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
                    await self.store.persist(session_id, session)
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
            session_id, session = await self.store.create(patient_initial_info, patient_id)
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


def make_app(
    config_path: str | Path | None = None,
    session_repository: SQLiteClinicalSessionRepository | None = None,
    security_config: SecurityConfig | None = None,
    rate_limiter: SlidingWindowRateLimiter | None = None,
) -> tornado.web.Application:
    configure_observability()
    config = load_clinical_config(config_path)
    security_config = security_config or SecurityConfig.from_env()
    rate_limiter = rate_limiter or SlidingWindowRateLimiter(
        max_keys=security_config.max_rate_limit_keys
    )
    store = ClinicalSessionStore(config, repository=session_repository)
    routes = [
        (r"/api/v1/health", HealthHandler, {"store": store}),
        (r"/api/v1/ready", ReadinessHandler, {"store": store}),
        (r"/api/v1/clinical/sessions", SessionsHandler, {"store": store}),
        (r"/api/v1/clinical/sessions/([^/]+)", SessionHandler, {"store": store}),
        (r"/api/v1/clinical/sessions/([^/]+)/context", ContextHandler, {"store": store}),
        (r"/api/v1/clinical/sessions/([^/]+)/question", NextQuestionHandler, {"store": store}),
        (r"/api/v1/clinical/sessions/([^/]+)/answer", AnswerHandler, {"store": store}),
        (r"/api/v1/clinical/sessions/([^/]+)/history/finish", FinishHistoryHandler, {"store": store}),
        (r"/api/v1/clinical/sessions/([^/]+)/run", RunHandler, {"store": store}),
    ]
    return tornado.web.Application(
        routes,
        debug=False,
        autoreload=False,
        serve_traceback=False,
        meddx_security_config=security_config,
        meddx_rate_limiter=rate_limiter,
    )


def main():
    configure_observability()
    config_path = os.getenv("MEDDX_CLINICAL_CONFIG")
    port = int(os.getenv("PORT", "8000"))
    security_config = SecurityConfig.from_env()
    app = make_app(config_path, security_config=security_config)
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
            rate_limiting=True,
            max_body_bytes=security_config.max_body_bytes,
        )
    )
    tornado.ioloop.IOLoop.current().start()


if __name__ == "__main__":
    main()
