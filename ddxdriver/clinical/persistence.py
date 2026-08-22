import json
import os
import sqlite3
from datetime import datetime, timezone
from pathlib import Path


DEFAULT_SESSION_DB_NAME = "clinical_sessions.sqlite3"


def resolve_session_db_path() -> Path:
    """Resolve the SQLite path used for resumable clinical sessions.

    ``MEDDX_SESSION_DB_PATH`` wins when supplied. Otherwise use a writable path
    under the service account's home directory. If a persistent volume is later
    mounted, pointing the environment variable at that volume requires no code change.
    """
    configured = os.getenv("MEDDX_SESSION_DB_PATH")
    if configured:
        return Path(configured).expanduser()
    return Path.home() / ".meddxagent" / DEFAULT_SESSION_DB_NAME


class SQLiteClinicalSessionRepository:
    """Small durable repository for serialized clinical session state."""

    def __init__(self, path: str | Path | None = None):
        self.path = Path(path) if path is not None else resolve_session_db_path()
        self.path = self.path.expanduser()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, timeout=10)
        connection.execute("PRAGMA busy_timeout = 10000")
        connection.execute("PRAGMA journal_mode = WAL")
        return connection

    def _initialize(self) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS clinical_sessions (
                    session_id TEXT PRIMARY KEY,
                    state_json TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
                """
            )

    def save(self, session_id: str, state: dict) -> None:
        if not isinstance(session_id, str) or not session_id:
            raise ValueError("Clinical session id must be a non-empty string")
        if not isinstance(state, dict):
            raise TypeError("Clinical session state must be a mapping")

        serialized = json.dumps(state, ensure_ascii=False, separators=(",", ":"))
        now = datetime.now(timezone.utc).isoformat()
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO clinical_sessions (session_id, state_json, created_at, updated_at)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(session_id) DO UPDATE SET
                    state_json = excluded.state_json,
                    updated_at = excluded.updated_at
                """,
                (session_id, serialized, now, now),
            )

    def load(self, session_id: str) -> dict | None:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT state_json FROM clinical_sessions WHERE session_id = ?",
                (session_id,),
            ).fetchone()

        if row is None:
            return None
        state = json.loads(row[0])
        if not isinstance(state, dict):
            raise ValueError("Persisted clinical session state is invalid")
        return state

    def delete(self, session_id: str) -> None:
        with self._connect() as connection:
            connection.execute(
                "DELETE FROM clinical_sessions WHERE session_id = ?",
                (session_id,),
            )
