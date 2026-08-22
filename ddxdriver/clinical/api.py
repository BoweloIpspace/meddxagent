import asyncio
import json
import logging
import os
import time
import uuid
from pathlib import Path

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

    def set_default_headers(self):
        allowed = {
            origin.strip()
            for origin in os.getenv(
                "MEDDX_ALLOWED_ORIGINS",
                "http://localhost:5173,https://meddxagentfrontend.vercel.app",
            ).split(",")
            if origin.strip()
        }
        origin = self.request.headers.get("Origin")
        if "*" in allowed:
            self.set_header("Access-Control-Allow-Origin", "*")
        elif origin in allowed:
            self.set_header("Access-Control-Allow-Origin", origin)
            self.set_header("Vary", "Origin")
        self.set_header("Access-Control-Allow-Headers", "Content-Type, X-Request-ID")
        self.set_header("Access-Control-Allow-Methods", "GET,POST,OPTIONS")
        self.set_header("Access-Control-Expose-Headers", "X-Request-ID")
        self.set_header("Content-Type", "application/json")

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
        try:
            payload = json.loads(self.request.body.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError("Request body must be valid JSON") from exc
        if not isinstance(payload, dict):
            raise ValueError("Request body must be a JSON object")
        return payload

    def write_api_error(self, status: int, message: str):
        self.set_status(status)
        self.finish({"error": message, "request_id": self.request_id})

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
                    "request_id": self.request_id,
                }
            )
            return

        try:
            payload = self.json_body()
            patient_initial_info = payload.get("patient_initial_info", "")
            if not isinstance(patient_initial_info, str) or not patient_initial_info.strip():
                raise ValueError("patient_initial_info is required")
            patient_id = payload.get("patient_id")
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
            patient_initial_info = payload.get("patient_initial_info", "")
            if not isinstance(patient_initial_info, str) or not patient_initial_info.strip():
                raise ValueError("patient_initial_info is required")
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
            answer = payload.get("answer", "")
            if not isinstance(answer, str) or not answer.strip():
                raise ValueError("answer is required")
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
) -> tornado.web.Application:
    configure_observability()
    config = load_clinical_config(config_path)
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
    return tornado.web.Application(routes)


def main():
    configure_observability()
    config_path = os.getenv("MEDDX_CLINICAL_CONFIG")
    port = int(os.getenv("PORT", "8000"))
    app = make_app(config_path)
    app.listen(port)
    emit_event(build_event("service.started", port=port))
    tornado.ioloop.IOLoop.current().start()


if __name__ == "__main__":
    main()
