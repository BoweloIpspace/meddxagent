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

from .monitoring import configure_monitoring, dispatch_monitoring_event


LOGGER_NAME = "meddxagent.clinical"
REQUEST_ID_PATTERN = re.compile(r"^[A-Za-z0-9._:-]{1,64}$")
SENSITIVE_FIELD_NAMES = {
    "answer",
    "body",
    "diagnosis",
    "dialogue_history",
    "patient_initial_info",
    "patient_profile",
    "prompt",
    "question",
    "rag_content",
    "rationale",
    "result",
}


def _utc_timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def configure_observability() -> logging.Logger:
    """Configure privacy-filtered JSON logs and the optional monitoring sink."""
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

    # Validate monitoring configuration during startup rather than at first error.
    configure_monitoring()
    return logger


def request_id_from_header(value: str | None) -> str:
    if isinstance(value, str) and REQUEST_ID_PATTERN.fullmatch(value):
        return value
    return str(uuid.uuid4())


def _stable_reference(value: str | None, length: int = 16) -> str | None:
    if not value:
        return None
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:length]


def session_reference(session_id: str | None) -> str | None:
    """Return a stable non-reversible short reference for log correlation."""
    return _stable_reference(session_id)


def actor_reference(subject: str | None) -> str | None:
    """Pseudonymize an authenticated subject before it reaches logs/monitoring."""
    return _stable_reference(subject)


def resource_reference(resource_id: str | None) -> str | None:
    return _stable_reference(resource_id)


def build_event(event: str, **fields: Any) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "timestamp": _utc_timestamp(),
        "service": "meddx-clinical-api",
        "event": event,
    }
    for key, value in fields.items():
        if value is not None and key not in SENSITIVE_FIELD_NAMES:
            payload[key] = value
    return payload


def build_error_event(
    event: str,
    error: BaseException,
    **fields: Any,
) -> dict[str, Any]:
    """Build an error event without serializing the exception message."""
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
    dispatch_monitoring_event(payload, level)
