from __future__ import annotations

import hashlib
import math
import os
import re
import time
from collections import OrderedDict, deque
from dataclasses import dataclass
from typing import Callable, Protocol


DEFAULT_ALLOWED_ORIGINS = frozenset(
    {
        "http://localhost:5173",
        "https://meddxagentfrontend.vercel.app",
    }
)
CLINICAL_API_PREFIX = "/api/v1/clinical/"
CASE_API_PREFIX = "/api/v1/cases"
PROTECTED_API_PREFIXES = (CLINICAL_API_PREFIX, CASE_API_PREFIX)
SESSION_CREATE_PATH = "/api/v1/clinical/sessions"
SESSION_PATH_PATTERN = re.compile(r"^/api/v1/clinical/sessions/([^/]+)(?:/|$)")
MODEL_ACTION_SUFFIXES = ("/question", "/history/finish", "/run")
RATE_LIMIT_BACKENDS = frozenset({"auto", "memory", "database"})


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
    """Runtime security controls for the public application HTTP API."""

    allowed_origins: frozenset[str] = DEFAULT_ALLOWED_ORIGINS
    requests_per_minute: int = 240
    session_creates_per_minute: int = 30
    model_actions_per_minute: int = 30
    max_rate_limit_keys: int = 10_000
    rate_limit_backend: str = "auto"
    max_body_bytes: int = 65_536
    max_header_bytes: int = 16_384
    max_patient_info_chars: int = 32_000
    max_answer_chars: int = 8_000
    max_patient_id_chars: int = 128
    max_case_id_chars: int = 128
    body_timeout_seconds: int = 30
    idle_connection_timeout_seconds: int = 60

    @classmethod
    def from_env(cls) -> "SecurityConfig":
        rate_limit_backend = os.getenv("MEDDX_RATE_LIMIT_BACKEND", "auto").strip().lower()
        if rate_limit_backend not in RATE_LIMIT_BACKENDS:
            raise ValueError(
                "MEDDX_RATE_LIMIT_BACKEND must be one of: "
                + ", ".join(sorted(RATE_LIMIT_BACKENDS))
            )
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
            rate_limit_backend=rate_limit_backend,
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
            max_case_id_chars=_env_int(
                "MEDDX_MAX_CASE_ID_CHARS", 128, minimum=16, maximum=1_024
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


class RateLimiter(Protocol):
    def check(
        self,
        key: str,
        limit: int,
        *,
        window_seconds: int = 60,
    ) -> RateLimitDecision:
        ...


class SlidingWindowRateLimiter:
    """Small bounded in-process sliding-window limiter."""

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


class DatabaseWindowRateLimiter:
    """Database-coordinated fixed-window limiter for multi-instance deployments."""

    def __init__(self, repository, *, clock: Callable[[], float] = time.time) -> None:
        self.repository = repository
        self.clock = clock
        self._checks = 0

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
        window_start = int(now // window_seconds) * window_seconds
        count = self.repository.increment_rate_limit_window(key, window_start)
        self._checks += 1
        if self._checks % 100 == 0:
            self.repository.cleanup_rate_limit_windows(window_start - (window_seconds * 2))

        remaining = max(0, limit - count)
        if count > limit:
            retry_after = max(1, math.ceil(window_start + window_seconds - now))
            return RateLimitDecision(False, limit, 0, retry_after)
        return RateLimitDecision(True, limit, remaining, 0)


def build_rate_limiter(repository, config: SecurityConfig) -> RateLimiter:
    backend = config.rate_limit_backend
    if backend == "database" or (backend == "auto" and getattr(repository, "is_distributed", False)):
        return DatabaseWindowRateLimiter(repository)
    return SlidingWindowRateLimiter(max_keys=config.max_rate_limit_keys)


def origin_is_allowed(origin: str | None, config: SecurityConfig) -> bool:
    if not origin:
        return True
    return "*" in config.allowed_origins or origin in config.allowed_origins


def is_protected_api_path(path: str) -> bool:
    return any(path.startswith(prefix) for prefix in PROTECTED_API_PREFIXES)


def session_id_from_path(path: str) -> str | None:
    match = SESSION_PATH_PATTERN.match(path)
    return match.group(1) if match else None


def _stable_subject(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:24]


def client_rate_subject(remote_ip: str | None) -> str:
    return "client:" + _stable_subject(remote_ip or "unknown")


def identity_rate_subject(subject: str) -> str:
    return "identity:" + _stable_subject(subject)


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


def validate_case_id(value: str, max_chars: int) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("case_id must be a non-empty string")
    if len(value) > max_chars:
        raise ValueError("case_id exceeds the maximum allowed length")
    if "/" in value or "\\" in value or any(ord(character) < 32 for character in value):
        raise ValueError("case_id contains invalid characters")
    return value


def has_json_content_type(content_type: str | None) -> bool:
    if not content_type:
        return False
    return content_type.split(";", 1)[0].strip().lower() == "application/json"
