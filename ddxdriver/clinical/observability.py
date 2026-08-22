import hashlib
import json
import logging
import os
import re
import sys
import traceback
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


LOGGER_NAME = "meddxagent.clinical"
REQUEST_ID_PATTERN = re.compile(r"^[A-Za-z0-9._:-]{1,64}$")


def _utc_timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def configure_observability() -> logging.Logger:
    """Configure one-line JSON logs suitable for container log collection.

    Logs intentionally contain operational metadata only. Clinical request bodies,
    patient text, generated questions, answers and diagnostic content are never
    added by this module.
    """
    logger = logging.getLogger(LOGGER_NAME)
    level_name = os.getenv("MEDDX_LOG_LEVEL", "INFO").upper()
    level = getattr(logging, level_name, logging.INFO)
    logger.setLevel(level)
    logger.propagate = False

    if not any(getattr(handler, "_meddx_structured", False) for handler in logger.handlers):
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(logging.Formatter("%(message)s"))
        handler._meddx_structured = True
        logger.addHandler(handler)

        audit_path = os.getenv("MEDDX_AUDIT_LOG_PATH")
        if audit_path:
            path = Path(audit_path).expanduser()
            path.parent.mkdir(parents=True, exist_ok=True)
            file_handler = logging.FileHandler(path, encoding="utf-8")
            file_handler.setFormatter(logging.Formatter("%(message)s"))
            file_handler._meddx_structured = True
            logger.addHandler(file_handler)

    return logger


def request_id_from_header(value: str | None) -> str:
    if isinstance(value, str) and REQUEST_ID_PATTERN.fullmatch(value):
        return value
    return str(uuid.uuid4())


def session_reference(session_id: str | None) -> str | None:
    """Return a stable non-reversible short reference for log correlation."""
    if not session_id:
        return None
    return hashlib.sha256(session_id.encode("utf-8")).hexdigest()[:16]


def build_event(event: str, **fields: Any) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "timestamp": _utc_timestamp(),
        "service": "meddx-clinical-api",
        "event": event,
    }
    for key, value in fields.items():
        if value is not None:
            payload[key] = value
    return payload


def build_error_event(
    event: str,
    error: BaseException,
    **fields: Any,
) -> dict[str, Any]:
    """Build an error event without serializing the exception message.

    Provider/model exceptions can contain request details. Keeping only the type
    and sanitized stack locations gives operators correlation data without copying
    clinical content into logs.
    """
    frames = traceback.extract_tb(error.__traceback__) if error.__traceback__ else []
    safe_trace = [
        {
            "file": Path(frame.filename).name,
            "line": frame.lineno,
            "function": frame.name,
        }
        for frame in frames[-12:]
    ]
    return build_event(
        event,
        error_type=type(error).__name__,
        traceback=safe_trace,
        **fields,
    )


def emit_event(payload: dict[str, Any], level: int = logging.INFO) -> None:
    logger = configure_observability()
    logger.log(level, json.dumps(payload, ensure_ascii=False, separators=(",", ":"), default=str))
