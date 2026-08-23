from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone


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


@dataclass(frozen=True)
class SessionLifecycleConfig:
    """Expiry and cleanup policy for persisted clinical sessions.

    A TTL of zero preserves the current no-expiry behavior. Production deployments
    can enable expiry without a code change once retention policy is decided.
    """

    ttl_hours: int = 0
    cleanup_interval_seconds: int = 300

    @classmethod
    def from_env(cls) -> "SessionLifecycleConfig":
        return cls(
            ttl_hours=_env_int("MEDDX_SESSION_TTL_HOURS", 0, minimum=0, maximum=24 * 365),
            cleanup_interval_seconds=_env_int(
                "MEDDX_SESSION_CLEANUP_INTERVAL_SECONDS",
                300,
                minimum=30,
                maximum=86_400,
            ),
        )

    def expires_at(self, now: datetime | None = None) -> datetime | None:
        if self.ttl_hours == 0:
            return None
        current = now or datetime.now(timezone.utc)
        return current + timedelta(hours=self.ttl_hours)
