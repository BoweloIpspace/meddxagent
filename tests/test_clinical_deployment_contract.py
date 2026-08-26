import tomllib
from pathlib import Path

import pytest

from ddxdriver.clinical.api import missing_runtime_environment
from ddxdriver.clinical.auth import AuthConfig
from ddxdriver.clinical.config import load_clinical_config
from ddxdriver.clinical.lifecycle import SessionLifecycleConfig
from ddxdriver.clinical.monitoring import MonitoringConfig
from ddxdriver.clinical.security import SecurityConfig

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_default_production_environment_contract(monkeypatch):
    for name in (
        "MEDDX_ALLOWED_ORIGINS",
        "MEDDX_RATE_LIMIT_REQUESTS_PER_MINUTE",
        "MEDDX_RATE_LIMIT_MODEL_ACTIONS_PER_MINUTE",
        "MEDDX_RATE_LIMIT_SESSION_ACTIONS_PER_MINUTE",
        "MEDDX_RATE_LIMIT_SESSION_CREATES_PER_MINUTE",
        "MEDDX_RATE_LIMIT_BACKEND",
        "MEDDX_AUTH_MODE",
        "MEDDX_SESSION_TTL_HOURS",
        "MEDDX_SESSION_CLEANUP_INTERVAL_SECONDS",
        "MEDDX_MONITORING_WEBHOOK_URL",
        "MEDDX_MONITORING_MIN_LEVEL",
        "MEDDX_MONITORING_TIMEOUT_SECONDS",
        "MEDDX_MONITORING_QUEUE_SIZE",
    ):
        monkeypatch.delenv(name, raising=False)

    security = SecurityConfig.from_env()
    auth = AuthConfig.from_env()
    lifecycle = SessionLifecycleConfig.from_env()
    monitoring = MonitoringConfig.from_env()

    assert security.allowed_origins
    assert security.requests_per_minute > 0
    assert security.model_actions_per_minute > 0
    assert security.session_actions_per_minute > 0
    assert security.session_creates_per_minute > 0
    assert security.rate_limit_backend == "auto"
    assert auth.mode == "disabled"
    assert lifecycle.ttl_hours == 0
    assert lifecycle.cleanup_interval_seconds > 0
    assert monitoring.webhook_url is None


def test_runtime_readiness_requires_active_model_environment(monkeypatch):
    monkeypatch.delenv("OAI_KEY", raising=False)
    monkeypatch.delenv("AZURE_ENDPOINT", raising=False)

    missing = missing_runtime_environment(load_clinical_config())

    assert "OAI_KEY" in missing


def test_example_environment_documents_runtime_controls():
    env_example = (REPO_ROOT / ".env.example").read_text(encoding="utf-8")
    documented = {
        line.split("=", 1)[0].strip()
        for line in env_example.splitlines()
        if line.strip() and not line.lstrip().startswith("#") and "=" in line
    }
    expected = {
        "OAI_KEY",
        "MEDDX_HOST",
        "MEDDX_PORT",
        "MEDDX_CONFIG_PATH",
        "MEDDX_SESSION_DB_PATH",
        "MEDDX_DATABASE_URL",
        "MEDDX_ALLOWED_ORIGINS",
        "MEDDX_RATE_LIMIT_REQUESTS_PER_MINUTE",
        "MEDDX_RATE_LIMIT_MODEL_ACTIONS_PER_MINUTE",
        "MEDDX_RATE_LIMIT_SESSION_ACTIONS_PER_MINUTE",
        "MEDDX_RATE_LIMIT_SESSION_CREATES_PER_MINUTE",
        "MEDDX_RATE_LIMIT_BACKEND",
        "MEDDX_AUTH_MODE",
        "MEDDX_AUTH_REQUIRED_ROLES",
        "MEDDX_AUTH_SHARED_TOKEN",
        "MEDDX_AUTH_SHARED_SUBJECT",
        "MEDDX_AUTH_SHARED_ROLES",
        "MEDDX_AUTH_ISSUER",
        "MEDDX_AUTH_AUDIENCE",
        "MEDDX_AUTH_JWKS_URL",
        "MEDDX_AUTH_ROLES_CLAIM",
        "MEDDX_AUTH_JWT_ALGORITHMS",
        "MEDDX_SESSION_TTL_HOURS",
        "MEDDX_SESSION_CLEANUP_INTERVAL_SECONDS",
        "MEDDX_LOG_LEVEL",
        "MEDDX_AUDIT_LOG_PATH",
        "MEDDX_MONITORING_WEBHOOK_URL",
        "MEDDX_MONITORING_MIN_LEVEL",
        "MEDDX_MONITORING_TIMEOUT_SECONDS",
        "MEDDX_MONITORING_QUEUE_SIZE",
    }

    assert expected <= documented


def test_railway_config_keeps_health_check_and_platform_managed_port():
    config = tomllib.loads((REPO_ROOT / "railway.toml").read_text(encoding="utf-8"))

    assert config["build"]["builder"] == "DOCKERFILE"
    assert config["deploy"]["healthcheckPath"] == "/api/v1/ready"
    assert config["deploy"]["healthcheckTimeout"] == 300
    assert "PORT" not in config.get("deploy", {})


def test_invalid_platform_environment_fails_fast(monkeypatch):
    monkeypatch.setenv("MEDDX_RATE_LIMIT_REQUESTS_PER_MINUTE", "not-a-number")
    with pytest.raises(ValueError, match="must be an integer"):
        SecurityConfig.from_env()
