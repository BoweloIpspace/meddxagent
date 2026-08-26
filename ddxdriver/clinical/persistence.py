from __future__ import annotations

import json
import os
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Protocol
from urllib.parse import urlparse


DEFAULT_SESSION_DB_NAME = "clinical_sessions.sqlite3"
DEFAULT_OWNER_SUBJECT = "anonymous"


class SessionExpiredError(KeyError):
    pass


class SessionArchivedError(KeyError):
    pass


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _to_iso(value: datetime | None) -> str | None:
    if value is None:
        return None
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc).isoformat()


def _from_iso(value: str | None) -> datetime | None:
    if not value:
        return None
    parsed = datetime.fromisoformat(value)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _validate_identifier(value: str, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return value


def resolve_session_db_path() -> Path:
    """Resolve the SQLite path used when no production database URL is configured."""
    configured = os.getenv("MEDDX_SESSION_DB_PATH")
    if configured:
        return Path(configured).expanduser()
    return Path.home() / ".meddxagent" / DEFAULT_SESSION_DB_NAME


def configured_database_url() -> str | None:
    return os.getenv("MEDDX_DATABASE_URL") or os.getenv("DATABASE_URL") or None


class ClinicalRepository(Protocol):
    is_distributed: bool

    def save(
        self,
        session_id: str,
        state: dict,
        *,
        owner_subject: str = DEFAULT_OWNER_SUBJECT,
        expires_at: datetime | None = None,
    ) -> None:
        ...

    def load(
        self,
        session_id: str,
        *,
        owner_subject: str | None = None,
    ) -> dict | None:
        ...

    def delete(self, session_id: str, *, owner_subject: str | None = None) -> bool:
        ...

    def archive(self, session_id: str, *, owner_subject: str | None = None) -> bool:
        ...

    def cleanup_expired(self, *, now: datetime | None = None) -> int:
        ...

    def save_case(self, owner_subject: str, case_id: str, payload: dict) -> dict:
        ...

    def load_case(self, owner_subject: str, case_id: str) -> dict | None:
        ...

    def list_cases(
        self,
        owner_subject: str,
        *,
        limit: int = 100,
        offset: int = 0,
    ) -> list[dict]:
        ...

    def archive_case(self, owner_subject: str, case_id: str) -> bool:
        ...

    def delete_case(self, owner_subject: str, case_id: str) -> bool:
        ...

    def increment_rate_limit_window(
        self,
        subject: str,
        window_start: int,
        *,
        now: datetime | None = None,
    ) -> int:
        ...

    def cleanup_rate_limit_windows(self, before_window_start: int) -> int:
        ...


class SQLiteClinicalSessionRepository:
    """Durable SQLite repository for sessions, authenticated cases and shared counters.

    SQLite remains the zero-configuration fallback. A Postgres repository can be
    selected with ``MEDDX_DATABASE_URL``/``DATABASE_URL`` without changing callers.
    """

    is_distributed = False

    def __init__(self, path: str | Path | None = None):
        self.path = Path(path) if path is not None else resolve_session_db_path()
        self.path = self.path.expanduser()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, timeout=10)
        connection.execute("PRAGMA busy_timeout = 10000")
        connection.execute("PRAGMA journal_mode = WAL")
        connection.execute("PRAGMA foreign_keys = ON")
        return connection

    def _initialize(self) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS clinical_sessions (
                    session_id TEXT PRIMARY KEY,
                    owner_subject TEXT NOT NULL DEFAULT 'anonymous',
                    state_json TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    expires_at TEXT,
                    archived_at TEXT
                )
                """
            )
            self._ensure_sqlite_column(
                connection,
                "clinical_sessions",
                "owner_subject",
                "TEXT NOT NULL DEFAULT 'anonymous'",
            )
            self._ensure_sqlite_column(connection, "clinical_sessions", "expires_at", "TEXT")
            self._ensure_sqlite_column(connection, "clinical_sessions", "archived_at", "TEXT")
            connection.execute(
                "CREATE INDEX IF NOT EXISTS idx_clinical_sessions_expires_at "
                "ON clinical_sessions(expires_at)"
            )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS clinical_cases (
                    owner_subject TEXT NOT NULL,
                    case_id TEXT NOT NULL,
                    payload_json TEXT NOT NULL,
                    status TEXT NOT NULL,
                    engine_session_id TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    archived_at TEXT,
                    PRIMARY KEY (owner_subject, case_id)
                )
                """
            )
            connection.execute(
                "CREATE INDEX IF NOT EXISTS idx_clinical_cases_owner_updated "
                "ON clinical_cases(owner_subject, updated_at DESC)"
            )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS rate_limit_windows (
                    subject TEXT NOT NULL,
                    window_start INTEGER NOT NULL,
                    request_count INTEGER NOT NULL,
                    updated_at TEXT NOT NULL,
                    PRIMARY KEY (subject, window_start)
                )
                """
            )
            connection.execute(
                "CREATE INDEX IF NOT EXISTS idx_rate_limit_windows_start "
                "ON rate_limit_windows(window_start)"
            )

    @staticmethod
    def _ensure_sqlite_column(
        connection: sqlite3.Connection,
        table: str,
        column: str,
        definition: str,
    ) -> None:
        columns = {
            row[1]
            for row in connection.execute(f"PRAGMA table_info({table})").fetchall()
        }
        if column not in columns:
            connection.execute(f"ALTER TABLE {table} ADD COLUMN {column} {definition}")

    def save(
        self,
        session_id: str,
        state: dict,
        *,
        owner_subject: str = DEFAULT_OWNER_SUBJECT,
        expires_at: datetime | None = None,
    ) -> None:
        _validate_identifier(session_id, "Clinical session id")
        _validate_identifier(owner_subject, "Clinical session owner")
        if not isinstance(state, dict):
            raise TypeError("Clinical session state must be a mapping")

        serialized = json.dumps(state, ensure_ascii=False, separators=(",", ":"))
        now = _utc_now().isoformat()
        with self._connect() as connection:
            cursor = connection.execute(
                """
                INSERT INTO clinical_sessions (
                    session_id, owner_subject, state_json, created_at, updated_at, expires_at
                )
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(session_id) DO UPDATE SET
                    state_json = excluded.state_json,
                    updated_at = excluded.updated_at,
                    expires_at = excluded.expires_at
                WHERE clinical_sessions.owner_subject = excluded.owner_subject
                """,
                (session_id, owner_subject, serialized, now, now, _to_iso(expires_at)),
            )
            if cursor.rowcount == 0:
                raise PermissionError("Clinical session is owned by another identity")

    def load(
        self,
        session_id: str,
        *,
        owner_subject: str | None = None,
    ) -> dict | None:
        query = (
            "SELECT state_json, expires_at, archived_at FROM clinical_sessions "
            "WHERE session_id = ?"
        )
        params: list[object] = [session_id]
        if owner_subject is not None:
            query += " AND owner_subject = ?"
            params.append(owner_subject)

        with self._connect() as connection:
            row = connection.execute(query, params).fetchone()
        if row is None:
            return None

        expires_at = _from_iso(row[1])
        if expires_at is not None and expires_at <= _utc_now():
            raise SessionExpiredError(session_id)
        if row[2]:
            raise SessionArchivedError(session_id)

        state = json.loads(row[0])
        if not isinstance(state, dict):
            raise ValueError("Persisted clinical session state is invalid")
        return state

    def delete(self, session_id: str, *, owner_subject: str | None = None) -> bool:
        query = "DELETE FROM clinical_sessions WHERE session_id = ?"
        params: list[object] = [session_id]
        if owner_subject is not None:
            query += " AND owner_subject = ?"
            params.append(owner_subject)
        with self._connect() as connection:
            cursor = connection.execute(query, params)
            return cursor.rowcount > 0

    def archive(self, session_id: str, *, owner_subject: str | None = None) -> bool:
        query = "UPDATE clinical_sessions SET archived_at = ?, updated_at = ? WHERE session_id = ?"
        now = _utc_now().isoformat()
        params: list[object] = [now, now, session_id]
        if owner_subject is not None:
            query += " AND owner_subject = ?"
            params.append(owner_subject)
        query += " AND archived_at IS NULL"
        with self._connect() as connection:
            cursor = connection.execute(query, params)
            return cursor.rowcount > 0

    def cleanup_expired(self, *, now: datetime | None = None) -> int:
        cutoff = _to_iso(now or _utc_now())
        with self._connect() as connection:
            cursor = connection.execute(
                "DELETE FROM clinical_sessions WHERE expires_at IS NOT NULL AND expires_at <= ?",
                (cutoff,),
            )
            return cursor.rowcount

    def save_case(self, owner_subject: str, case_id: str, payload: dict) -> dict:
        _validate_identifier(owner_subject, "Case owner")
        _validate_identifier(case_id, "Case id")
        if not isinstance(payload, dict):
            raise TypeError("Case payload must be a mapping")
        status = payload.get("status")
        if not isinstance(status, str) or not status:
            raise ValueError("Case payload is missing status")
        engine_session_id = payload.get("engineSessionId")
        if engine_session_id is not None and not isinstance(engine_session_id, str):
            raise ValueError("engineSessionId must be a string when present")

        serialized = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
        now = _utc_now().isoformat()
        with self._connect() as connection:
            existing = connection.execute(
                "SELECT created_at FROM clinical_cases WHERE owner_subject = ? AND case_id = ?",
                (owner_subject, case_id),
            ).fetchone()
            created_at = existing[0] if existing else now
            connection.execute(
                """
                INSERT INTO clinical_cases (
                    owner_subject, case_id, payload_json, status, engine_session_id,
                    created_at, updated_at, archived_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, NULL)
                ON CONFLICT(owner_subject, case_id) DO UPDATE SET
                    payload_json = excluded.payload_json,
                    status = excluded.status,
                    engine_session_id = excluded.engine_session_id,
                    updated_at = excluded.updated_at,
                    archived_at = NULL
                """,
                (
                    owner_subject,
                    case_id,
                    serialized,
                    status,
                    engine_session_id,
                    created_at,
                    now,
                ),
            )
        return payload

    def load_case(self, owner_subject: str, case_id: str) -> dict | None:
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT payload_json FROM clinical_cases
                WHERE owner_subject = ? AND case_id = ? AND archived_at IS NULL
                """,
                (owner_subject, case_id),
            ).fetchone()
        if row is None:
            return None
        payload = json.loads(row[0])
        if not isinstance(payload, dict):
            raise ValueError("Persisted clinical case payload is invalid")
        return payload

    def list_cases(
        self,
        owner_subject: str,
        *,
        limit: int = 100,
        offset: int = 0,
    ) -> list[dict]:
        if limit < 1 or limit > 500:
            raise ValueError("Case list limit must be between 1 and 500")
        if offset < 0:
            raise ValueError("Case list offset must be zero or greater")
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT payload_json FROM clinical_cases
                WHERE owner_subject = ? AND archived_at IS NULL
                ORDER BY updated_at DESC, case_id DESC
                LIMIT ? OFFSET ?
                """,
                (owner_subject, limit, offset),
            ).fetchall()
        payloads = []
        for row in rows:
            payload = json.loads(row[0])
            if not isinstance(payload, dict):
                raise ValueError("Persisted clinical case payload is invalid")
            payloads.append(payload)
        return payloads

    def archive_case(self, owner_subject: str, case_id: str) -> bool:
        now = _utc_now().isoformat()
        with self._connect() as connection:
            cursor = connection.execute(
                """
                UPDATE clinical_cases SET archived_at = ?, updated_at = ?
                WHERE owner_subject = ? AND case_id = ? AND archived_at IS NULL
                """,
                (now, now, owner_subject, case_id),
            )
            return cursor.rowcount > 0

    def delete_case(self, owner_subject: str, case_id: str) -> bool:
        with self._connect() as connection:
            cursor = connection.execute(
                "DELETE FROM clinical_cases WHERE owner_subject = ? AND case_id = ?",
                (owner_subject, case_id),
            )
            return cursor.rowcount > 0

    def increment_rate_limit_window(
        self,
        subject: str,
        window_start: int,
        *,
        now: datetime | None = None,
    ) -> int:
        timestamp = _to_iso(now or _utc_now())
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                """
                SELECT request_count FROM rate_limit_windows
                WHERE subject = ? AND window_start = ?
                """,
                (subject, window_start),
            ).fetchone()
            count = (row[0] if row else 0) + 1
            connection.execute(
                """
                INSERT INTO rate_limit_windows (subject, window_start, request_count, updated_at)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(subject, window_start) DO UPDATE SET
                    request_count = excluded.request_count,
                    updated_at = excluded.updated_at
                """,
                (subject, window_start, count, timestamp),
            )
            return count

    def cleanup_rate_limit_windows(self, before_window_start: int) -> int:
        with self._connect() as connection:
            cursor = connection.execute(
                "DELETE FROM rate_limit_windows WHERE window_start < ?",
                (before_window_start,),
            )
            return cursor.rowcount


class PostgresClinicalRepository:
    """Postgres implementation suitable for shared multi-instance application state."""

    is_distributed = True

    def __init__(self, dsn: str):
        parsed = urlparse(dsn)
        if parsed.scheme not in {"postgres", "postgresql"}:
            raise ValueError("Postgres repository requires a postgres:// or postgresql:// URL")
        self.dsn = dsn
        self._initialize()

    def _connect(self):
        import psycopg

        return psycopg.connect(self.dsn)

    def _initialize(self) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS clinical_sessions (
                    session_id TEXT PRIMARY KEY,
                    owner_subject TEXT NOT NULL DEFAULT 'anonymous',
                    state_json JSONB NOT NULL,
                    created_at TIMESTAMPTZ NOT NULL,
                    updated_at TIMESTAMPTZ NOT NULL,
                    expires_at TIMESTAMPTZ,
                    archived_at TIMESTAMPTZ
                )
                """
            )
            connection.execute(
                "CREATE INDEX IF NOT EXISTS idx_clinical_sessions_expires_at "
                "ON clinical_sessions(expires_at)"
            )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS clinical_cases (
                    owner_subject TEXT NOT NULL,
                    case_id TEXT NOT NULL,
                    payload_json JSONB NOT NULL,
                    status TEXT NOT NULL,
                    engine_session_id TEXT,
                    created_at TIMESTAMPTZ NOT NULL,
                    updated_at TIMESTAMPTZ NOT NULL,
                    archived_at TIMESTAMPTZ,
                    PRIMARY KEY (owner_subject, case_id)
                )
                """
            )
            connection.execute(
                "CREATE INDEX IF NOT EXISTS idx_clinical_cases_owner_updated "
                "ON clinical_cases(owner_subject, updated_at DESC)"
            )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS rate_limit_windows (
                    subject TEXT NOT NULL,
                    window_start BIGINT NOT NULL,
                    request_count BIGINT NOT NULL,
                    updated_at TIMESTAMPTZ NOT NULL,
                    PRIMARY KEY (subject, window_start)
                )
                """
            )
            connection.execute(
                "CREATE INDEX IF NOT EXISTS idx_rate_limit_windows_start "
                "ON rate_limit_windows(window_start)"
            )

    def save(
        self,
        session_id: str,
        state: dict,
        *,
        owner_subject: str = DEFAULT_OWNER_SUBJECT,
        expires_at: datetime | None = None,
    ) -> None:
        from psycopg.types.json import Jsonb

        _validate_identifier(session_id, "Clinical session id")
        _validate_identifier(owner_subject, "Clinical session owner")
        if not isinstance(state, dict):
            raise TypeError("Clinical session state must be a mapping")
        now = _utc_now()
        with self._connect() as connection:
            row = connection.execute(
                """
                INSERT INTO clinical_sessions (
                    session_id, owner_subject, state_json, created_at, updated_at, expires_at
                ) VALUES (%s, %s, %s, %s, %s, %s)
                ON CONFLICT(session_id) DO UPDATE SET
                    state_json = EXCLUDED.state_json,
                    updated_at = EXCLUDED.updated_at,
                    expires_at = EXCLUDED.expires_at
                WHERE clinical_sessions.owner_subject = EXCLUDED.owner_subject
                RETURNING session_id
                """,
                (session_id, owner_subject, Jsonb(state), now, now, expires_at),
            ).fetchone()
            if row is None:
                raise PermissionError("Clinical session is owned by another identity")

    def load(
        self,
        session_id: str,
        *,
        owner_subject: str | None = None,
    ) -> dict | None:
        query = (
            "SELECT state_json, expires_at, archived_at FROM clinical_sessions "
            "WHERE session_id = %s"
        )
        params: list[object] = [session_id]
        if owner_subject is not None:
            query += " AND owner_subject = %s"
            params.append(owner_subject)
        with self._connect() as connection:
            row = connection.execute(query, params).fetchone()
        if row is None:
            return None
        expires_at = row[1]
        if expires_at is not None and expires_at <= _utc_now():
            raise SessionExpiredError(session_id)
        if row[2] is not None:
            raise SessionArchivedError(session_id)
        state = row[0]
        if isinstance(state, str):
            state = json.loads(state)
        if not isinstance(state, dict):
            raise ValueError("Persisted clinical session state is invalid")
        return state

    def delete(self, session_id: str, *, owner_subject: str | None = None) -> bool:
        query = "DELETE FROM clinical_sessions WHERE session_id = %s"
        params: list[object] = [session_id]
        if owner_subject is not None:
            query += " AND owner_subject = %s"
            params.append(owner_subject)
        with self._connect() as connection:
            cursor = connection.execute(query, params)
            return cursor.rowcount > 0

    def archive(self, session_id: str, *, owner_subject: str | None = None) -> bool:
        query = (
            "UPDATE clinical_sessions SET archived_at = %s, updated_at = %s "
            "WHERE session_id = %s"
        )
        now = _utc_now()
        params: list[object] = [now, now, session_id]
        if owner_subject is not None:
            query += " AND owner_subject = %s"
            params.append(owner_subject)
        query += " AND archived_at IS NULL"
        with self._connect() as connection:
            cursor = connection.execute(query, params)
            return cursor.rowcount > 0

    def cleanup_expired(self, *, now: datetime | None = None) -> int:
        with self._connect() as connection:
            cursor = connection.execute(
                "DELETE FROM clinical_sessions WHERE expires_at IS NOT NULL AND expires_at <= %s",
                (now or _utc_now(),),
            )
            return cursor.rowcount

    def save_case(self, owner_subject: str, case_id: str, payload: dict) -> dict:
        from psycopg.types.json import Jsonb

        _validate_identifier(owner_subject, "Case owner")
        _validate_identifier(case_id, "Case id")
        if not isinstance(payload, dict):
            raise TypeError("Case payload must be a mapping")
        status = payload.get("status")
        if not isinstance(status, str) or not status:
            raise ValueError("Case payload is missing status")
        engine_session_id = payload.get("engineSessionId")
        if engine_session_id is not None and not isinstance(engine_session_id, str):
            raise ValueError("engineSessionId must be a string when present")
        now = _utc_now()
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO clinical_cases (
                    owner_subject, case_id, payload_json, status, engine_session_id,
                    created_at, updated_at, archived_at
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, NULL)
                ON CONFLICT(owner_subject, case_id) DO UPDATE SET
                    payload_json = EXCLUDED.payload_json,
                    status = EXCLUDED.status,
                    engine_session_id = EXCLUDED.engine_session_id,
                    updated_at = EXCLUDED.updated_at,
                    archived_at = NULL
                """,
                (owner_subject, case_id, Jsonb(payload), status, engine_session_id, now, now),
            )
        return payload

    def load_case(self, owner_subject: str, case_id: str) -> dict | None:
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT payload_json FROM clinical_cases
                WHERE owner_subject = %s AND case_id = %s AND archived_at IS NULL
                """,
                (owner_subject, case_id),
            ).fetchone()
        if row is None:
            return None
        payload = row[0]
        if isinstance(payload, str):
            payload = json.loads(payload)
        if not isinstance(payload, dict):
            raise ValueError("Persisted clinical case payload is invalid")
        return payload

    def list_cases(
        self,
        owner_subject: str,
        *,
        limit: int = 100,
        offset: int = 0,
    ) -> list[dict]:
        if limit < 1 or limit > 500:
            raise ValueError("Case list limit must be between 1 and 500")
        if offset < 0:
            raise ValueError("Case list offset must be zero or greater")
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT payload_json FROM clinical_cases
                WHERE owner_subject = %s AND archived_at IS NULL
                ORDER BY updated_at DESC, case_id DESC
                LIMIT %s OFFSET %s
                """,
                (owner_subject, limit, offset),
            ).fetchall()
        payloads: list[dict] = []
        for row in rows:
            payload = row[0]
            if isinstance(payload, str):
                payload = json.loads(payload)
            if not isinstance(payload, dict):
                raise ValueError("Persisted clinical case payload is invalid")
            payloads.append(payload)
        return payloads

    def archive_case(self, owner_subject: str, case_id: str) -> bool:
        now = _utc_now()
        with self._connect() as connection:
            cursor = connection.execute(
                """
                UPDATE clinical_cases SET archived_at = %s, updated_at = %s
                WHERE owner_subject = %s AND case_id = %s AND archived_at IS NULL
                """,
                (now, now, owner_subject, case_id),
            )
            return cursor.rowcount > 0

    def delete_case(self, owner_subject: str, case_id: str) -> bool:
        with self._connect() as connection:
            cursor = connection.execute(
                "DELETE FROM clinical_cases WHERE owner_subject = %s AND case_id = %s",
                (owner_subject, case_id),
            )
            return cursor.rowcount > 0

    def increment_rate_limit_window(
        self,
        subject: str,
        window_start: int,
        *,
        now: datetime | None = None,
    ) -> int:
        with self._connect() as connection:
            row = connection.execute(
                """
                INSERT INTO rate_limit_windows (subject, window_start, request_count, updated_at)
                VALUES (%s, %s, 1, %s)
                ON CONFLICT(subject, window_start) DO UPDATE SET
                    request_count = rate_limit_windows.request_count + 1,
                    updated_at = EXCLUDED.updated_at
                RETURNING request_count
                """,
                (subject, window_start, now or _utc_now()),
            ).fetchone()
        return int(row[0])

    def cleanup_rate_limit_windows(self, before_window_start: int) -> int:
        with self._connect() as connection:
            cursor = connection.execute(
                "DELETE FROM rate_limit_windows WHERE window_start < %s",
                (before_window_start,),
            )
            return cursor.rowcount


def build_clinical_repository(
    database_url: str | None = None,
    *,
    sqlite_path: str | Path | None = None,
) -> ClinicalRepository:
    configured = database_url if database_url is not None else configured_database_url()
    if configured:
        parsed = urlparse(configured)
        if parsed.scheme in {"postgres", "postgresql"}:
            return PostgresClinicalRepository(configured)
        if parsed.scheme == "sqlite":
            path = parsed.path or parsed.netloc
            if not path:
                raise ValueError("SQLite database URL must include a path")
            return SQLiteClinicalSessionRepository(path)
        raise ValueError("MEDDX_DATABASE_URL/DATABASE_URL must use postgres, postgresql or sqlite")
    return SQLiteClinicalSessionRepository(sqlite_path)