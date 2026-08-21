import asyncio
import json
import os
import sys
import uuid
from pathlib import Path

import tornado.ioloop
import tornado.web

from .config import create_clinical_session, load_clinical_config


class ClinicalSessionStore:
    """Process-local session storage for the first frontend integration.

    Persistence is intentionally outside this first integration slice. The API
    contract can stay the same when durable storage is added later.
    """

    def __init__(self, config: dict):
        self.config = config
        self.sessions = {}
        self.locks: dict[str, asyncio.Lock] = {}

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
        return session_id, session

    def get(self, session_id: str):
        session = self.sessions.get(session_id)
        if session is None:
            raise KeyError(session_id)
        return session

    def lock(self, session_id: str) -> asyncio.Lock:
        if session_id not in self.locks:
            raise KeyError(session_id)
        return self.locks[session_id]


class BaseHandler(tornado.web.RequestHandler):
    def initialize(self, store: ClinicalSessionStore):
        self.store = store

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
        self.set_header("Access-Control-Allow-Headers", "Content-Type")
        self.set_header("Access-Control-Allow-Methods", "GET,POST,OPTIONS")
        self.set_header("Content-Type", "application/json")

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
        self.finish({"error": message})

    async def run_session_action(self, session_id: str, action):
        try:
            session = self.store.get(session_id)
            lock = self.store.lock(session_id)
        except KeyError:
            self.write_api_error(404, "Clinical session not found")
            return None

        try:
            async with lock:
                return await asyncio.to_thread(action, session)
        except (ValueError, RuntimeError, TypeError) as exc:
            self.write_api_error(400, str(exc))
            return None
        except Exception:
            self.log_exception(*sys.exc_info())
            self.write_api_error(500, "MEDDxAgent request failed")
            return None


class HealthHandler(BaseHandler):
    async def get(self):
        self.finish({"status": "ok"})


class SessionsHandler(BaseHandler):
    async def post(self):
        try:
            payload = self.json_body()
            patient_initial_info = payload.get("patient_initial_info", "")
            if not isinstance(patient_initial_info, str) or not patient_initial_info.strip():
                raise ValueError("patient_initial_info is required")
            patient_id = payload.get("patient_id")
            session_id, session = await self.store.create(patient_initial_info, patient_id)
        except (ValueError, TypeError) as exc:
            self.write_api_error(400, str(exc))
            return
        except Exception:
            self.log_exception(*sys.exc_info())
            self.write_api_error(500, "Unable to create MEDDxAgent clinical session")
            return

        self.set_status(201)
        self.finish({"session_id": session_id, **session.snapshot()})


class SessionHandler(BaseHandler):
    async def get(self, session_id: str):
        result = await self.run_session_action(session_id, lambda session: session.snapshot())
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
            self.write_api_error(400, str(exc))
            return

        def update_context(session):
            session.update_patient_initial_info(patient_initial_info)
            return session.snapshot()

        result = await self.run_session_action(session_id, update_context)
        if result is not None:
            self.finish({"session_id": session_id, **result})


class NextQuestionHandler(BaseHandler):
    async def post(self, session_id: str):
        def generate(session):
            question = session.next_question()
            return question, session.snapshot()

        result = await self.run_session_action(session_id, generate)
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
            self.write_api_error(400, str(exc))
            return

        def submit(session):
            session.submit_answer(answer)
            return session.snapshot()

        result = await self.run_session_action(session_id, submit)
        if result is not None:
            self.finish({"session_id": session_id, **result})


class FinishHistoryHandler(BaseHandler):
    async def post(self, session_id: str):
        def finish_history(session):
            session.finish_history()
            return session.snapshot()

        result = await self.run_session_action(session_id, finish_history)
        if result is not None:
            self.finish({"session_id": session_id, **result})


class RunHandler(BaseHandler):
    async def post(self, session_id: str):
        def run(session):
            session.run()
            return session.snapshot()

        result = await self.run_session_action(session_id, run)
        if result is not None:
            self.finish({"session_id": session_id, **result})


def make_app(config_path: str | Path | None = None) -> tornado.web.Application:
    config = load_clinical_config(config_path)
    store = ClinicalSessionStore(config)
    routes = [
        (r"/api/v1/health", HealthHandler, {"store": store}),
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
    config_path = os.getenv("MEDDX_CLINICAL_CONFIG")
    port = int(os.getenv("PORT", "8000"))
    app = make_app(config_path)
    app.listen(port)
    print(f"MEDDxAgent clinical API listening on port {port}")
    tornado.ioloop.IOLoop.current().start()


if __name__ == "__main__":
    main()
