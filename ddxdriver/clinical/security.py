from __future__ import annotations

import hashlib
import math
import os
import re
import time
from collections import OrderedDict, deque
from dataclasses import dataclass
from typing import Callable


DEFAULT_ALLOWED_ORIGINS = frozenset(
    {
        "http://localhost:5173",
        "https://meddxagentfrontend.vercel.app",
    }
)
CLINICAL_API_PREFIX = "/api/v1/clinical/"
SESSION_CREATE_PATH = "/api/v1/clinical/sessions"
SESSION_PATH_PATTERN = re.compile(r"^/api/v1/clinical/sessions/([^/]+)(?:/|$)")
MODEL_ACTION_SUFFIXES = ("/question", "/history/finish", "/run")


def _env_int(name: str, default: int, *, minimum: int, maximum: int) -> int:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer") from exc
    if value < minimum or value > maximum:
        raise ValueError(f"{name} must be between {minimum} and {maximum}")
    return value


def _allowed_origins_from_env() -> frozenset[str]:
    raw = os.getenv("MEDDX_ALLOWED_ORIGINS")
    if raw is None:
        return DEFAULT_ALLOWED_ORIGINS
    return frozenset(origin.strip() for origin in raw.split(",") if origin.strip())


@dataclass(frozen=True)
class SecurityConfig:
    """Runtime security controls for the public clinical HTTP API.

    Limits are deliberately conservative but configurable. A rate limit of zero
    disables only that specific limiter, which is useful for isolated tests.
    """

    allowed_origins: frozenset[str] = DEFAULT_ALLOWED_ORIGINS
    requests_per_minute: int = 240
    session_creates_per_minute: int = 30
    model_actions_per_minute: int = 30
    max_rate_limit_keys: int = 10_000
    max_body_bytes: int = 65_536
    max_header_bytes: int = 16_384
    max_patient_info_chars: int = 32_000
    max_answer_chars: int = 8_000
    max_patient_id_chars: int = 128
    body_timeout_seconds: int = 30
    idle_connection_timeout_seconds: int = 60

    @classmethod
    def from_env(cls) -> "SecurityConfig":
        return cls(
            allowed_origins=_allowed_origins_from_env(),
            requests_per_minute=_env_int(
                "MEDDX_RATE_LIMIT_REQUESTS_PER_MINUTE", 240, minimum=0, maximum=100_000
            ),
            session_creates_per_minute=_env_int(
                "MEDDX_RATE_LIMIT_SESSION_CREATES_PER_MINUTE",
                30,
                minimum=0,
                maximum=100_000,
            ),
            model_actions_per_minute=_env_int(
                "MEDDX_RATE_LIMIT_MODEL_ACTIONS_PER_MINUTE",
                30,
                minimum=0,
                maximum=100_000,
            ),
            max_rate_limit_keys=_env_int(
                "MEDDX_RATE_LIMIT_MAX_KEYS", 10_000, minimum=100, maximum=1_000_000
            ),
            max_body_bytes=_env_int(
                "MEDDX_MAX_BODY_BYTES", 65_536, minimum=1_024, maximum=10_000_000
            ),
            max_header_bytes=_env_int(
                "MEDDX_MAX_HEADER_BYTES", 16_384, minimum=4_096, maximum=1_000_000
            ),
            max_patient_info_chars=_env_int(
                "MEDDX_MAX_PATIENT_INFO_CHARS", 32_000, minimum=1_000, maximum=1_000_000
            ),
            max_answer_chars=_env_int(
                "MEDDX_MAX_ANSWER_CHARS", 8_000, minimum=100, maximum=100_000
            ),
            max_patient_id_chars=_env_int(
                "MEDDX_MAX_PATIENT_ID_CHARS", 128, minimum=16, maximum=1_024
            ),
            body_timeout_seconds=_env_int(
                "MEDDX_BODY_TIMEOUT_SECONDS", 30, minimum=5, maximum=300
            ),
            idle_connection_timeout_seconds=_env_int(
                "MEDDX_IDLE_CONNECTION_TIMEOUT_SECONDS", 60, minimum=5, maximum=600
            ),
        )


@dataclass(frozen=True)
class RateLimitDecision:
    allowed: bool
    limit: int
    remaining: int
    retry_after_seconds: int


class SlidingWindowRateLimiter:
    """Small bounded in-process sliding-window limiter.

    This intentionally does not trust forwarding headers. The caller decides the
    subject key, and the API uses Tornado's transport-level ``remote_ip`` for the
    client-wide limit plus a hashed session id for expensive model actions.
    """

    def __init__(
        self,
        *,
        max_keys: int = 10_000,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if max_keys < 1:
            raise ValueError("max_keys must be positive")
        self.max_keys = max_keys
        self.clock = clock
        self._buckets: OrderedDict[str, deque[float]] = OrderedDict()

    def check(
        self,
        key: str,
        limit: int,
        *,
        window_seconds: int = 60,
    ) -> RateLimitDecision:
        if limit < 0:
            raise ValueError("rate limit cannot be negative")
        if window_seconds < 1:
            raise ValueError("window_seconds must be positive")
        if limit == 0:
            return RateLimitDecision(True, 0, 0, 0)

        now = self.clock()
        bucket = self._buckets.get(key)
        if bucket is None:
            if len(self._buckets) >= self.max_keys:
                self._buckets.popitem(last=False)
            bucket = deque()
            self._buckets[key] = bucket
        else:
            self._buckets.move_to_end(key)

        cutoff = now - window_seconds
        while bucket and bucket[0] <= cutoff:
            bucket.popleft()

        if len(bucket) >= limit:
            retry_after = max(1, math.ceil(window_seconds - (now - bucket[0])))
            return RateLimitDecision(False, limit, 0, retry_after)

        bucket.append(now)
        return RateLimitDecision(True, limit, max(0, limit - len(bucket)), 0)


def origin_is_allowed(origin: str | None, config: SecurityConfig) -> bool:
    if not origin:
        return True
    return "*" in config.allowed_origins or origin in config.allowed_origins


def session_id_from_path(path: str) -> str | None:
    match = SESSION_PATH_PATTERN.match(path)
    return match.group(1) if match else None


def _stable_subject(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:24]


def client_rate_subject(remote_ip: str | None) -> str:
    return "client:" + _stable_subject(remote_ip or "unknown")


def session_rate_subject(session_id: str) -> str:
    return "session:" + _stable_subject(session_id)


def is_model_action(path: str, method: str) -> bool:
    return method.upper() == "POST" and path.startswith(CLINICAL_API_PREFIX) and path.endswith(
        MODEL_ACTION_SUFFIXES
    )


def validate_required_text(value, field_name: str, max_chars: int) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} is required")
    if len(value) > max_chars:
        raise ValueError(f"{field_name} exceeds the maximum allowed length")
    return value


def validate_patient_id(value, max_chars: int):
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (str, int)):
        raise ValueError("patient_id must be a string or integer")
    if isinstance(value, str):
        if not value.strip():
            raise ValueError("patient_id must not be empty")
        if len(value) > max_chars:
            raise ValueError("patient_id exceeds the maximum allowed length")
    return value


def has_json_content_type(content_type: str | None) -> bool:
    if not content_type:
        return False
    return content_type.split(";", 1)[0].strip().lower() == "application/json"
