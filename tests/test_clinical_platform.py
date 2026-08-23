from datetime import datetime, timedelta, timezone
from unittest.mock import patch

import pytest

from ddxdriver.clinical.lifecycle import SessionLifecycleConfig
from ddxdriver.clinical.persistence import (
    PostgresClinicalRepository,
    SQLiteClinicalSessionRepository,
    SessionArchivedError,
    SessionExpiredError,
    build_clinical_repository,
)
from ddxdriver.clinical.security import (
    DatabaseWindowRateLimiter,
    SecurityConfig,
    build_rate_limiter,
)


def _state(marker: str = "state") -> dict:
    return {
        "version": 1,
        "patient": {
            "patient_id": "CASE-1",
            "patient_initial_info": "Headache",
            "patient_profile": "",
        },
        "history": None,
        "result": None,
        "marker": marker,
    }


def test_session_repository_enforces_owner_expiry_archive_and_cleanup(tmp_path):
    repository = SQLiteClinicalSessionRepository(tmp_path / "clinical.sqlite3")

    repository.save("owned", _state(), owner_subject="user-a")
    assert repository.load("owned", owner_subject="user-a") == _state()
    assert repository.load("owned", owner_subject="user-b") is None

    repository.save(
        "expired",
        _state("expired"),
        owner_subject="user-a",
        expires_at=datetime.now(timezone.utc) - timedelta(seconds=1),
    )
    with pytest.raises(SessionExpiredError):
        repository.load("expired", owner_subject="user-a")

    repository.save("archived", _state("archived"), owner_subject="user-a")
    assert repository.archive("archived", owner_subject="user-a") is True
    with pytest.raises(SessionArchivedError):
        repository.load("archived", owner_subject="user-a")

    removed = repository.cleanup_expired()
    assert removed >= 1
    assert repository.load("expired", owner_subject="user-a") is None


def test_session_lifecycle_is_disabled_by_default_and_configurable(monkeypatch):
    monkeypatch.delenv("MEDDX_SESSION_TTL_HOURS", raising=False)
    config = SessionLifecycleConfig.from_env()
    assert config.ttl_hours == 0
    assert config.expires_at() is None

    monkeypatch.setenv("MEDDX_SESSION_TTL_HOURS", "24")
    configured = SessionLifecycleConfig.from_env()
    assert configured.ttl_hours == 24
    assert configured.expires_at() is not None


def test_database_rate_limit_is_shared_across_limiter_instances(tmp_path):
    database_path = tmp_path / "clinical.sqlite3"
    first_repository = SQLiteClinicalSessionRepository(database_path)
    second_repository = SQLiteClinicalSessionRepository(database_path)
    first = DatabaseWindowRateLimiter(first_repository, clock=lambda: 120.0)
    second = DatabaseWindowRateLimiter(second_repository, clock=lambda: 120.0)

    assert first.check("same-subject", 2).allowed is True
    assert second.check("same-subject", 2).allowed is True
    blocked = first.check("same-subject", 2)
    assert blocked.allowed is False
    assert blocked.remaining == 0
    assert blocked.retry_after_seconds == 60


def test_rate_limit_factory_uses_database_only_when_requested_or_distributed(tmp_path):
    sqlite_repository = SQLiteClinicalSessionRepository(tmp_path / "clinical.sqlite3")
    auto = build_rate_limiter(sqlite_repository, SecurityConfig(rate_limit_backend="auto"))
    assert not isinstance(auto, DatabaseWindowRateLimiter)

    database = build_rate_limiter(
        sqlite_repository,
        SecurityConfig(rate_limit_backend="database"),
    )
    assert isinstance(database, DatabaseWindowRateLimiter)

    class DistributedRepository:
        is_distributed = True

    distributed = build_rate_limiter(
        DistributedRepository(),
        SecurityConfig(rate_limit_backend="auto"),
    )
    assert isinstance(distributed, DatabaseWindowRateLimiter)


def test_repository_factory_preserves_sqlite_default_and_selects_postgres(tmp_path):
    sqlite_repository = build_clinical_repository(database_url=None, sqlite_path=tmp_path / "x.sqlite3")
    assert isinstance(sqlite_repository, SQLiteClinicalSessionRepository)

    with patch(
        "ddxdriver.clinical.persistence.PostgresClinicalRepository",
        autospec=True,
    ) as postgres:
        build_clinical_repository("postgresql://user:pass@example.test/db")
    postgres.assert_called_once_with("postgresql://user:pass@example.test/db")


def test_postgres_repository_rejects_non_postgres_urls_without_connecting():
    with pytest.raises(ValueError, match="postgres"):
        PostgresClinicalRepository("https://example.test/database")
