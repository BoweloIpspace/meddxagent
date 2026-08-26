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


def test_default_clinical_config_matches_production_engine_contract():
    config = load_clinical_config()

    assert config["ddxdriver"]["class_name"] == "ddxdriver.ddxdrivers.open_choice.OpenChoice"
    assert config["ddxdriver"]["config"]["available_agents"] == [
        "history_taking",
        "rag",
        "diagnosis",
    ]
    assert config["ddxdriver"]["config"]["max_turns"] == 5

    rag = config["rag"]
    assert rag["class_name"] == "ddxdriver.rag_agents.searchrag_standard.SearchRAGStandard"
    assert rag["config"]["corpus_name"] == "PubMed"
    assert rag["config"]["top_k_search"] == 2
    assert rag["config"]["max_keyword_searches"] == 3
    assert rag["config"]["model"]["config"]["model_name"] == "gpt-4o"

    assert config["history_taking"]["config"]["model"]["config"]["model_name"] == "gpt-4o"
    assert config["diagnosis"]["config"]["model"]["config"]["model_name"] == "gpt-4o"
    assert config["diagnosis"]["config"]["fewshot"] == {
        "type": "none",
        "num_shots": 0,
        "self_generated_fewshot_cot": False,
    }


def test_default_runtime_requires_openai_key_but_not_optional_ncbi_identity(monkeypatch):
    config = load_clinical_config()
    monkeypatch.delenv("OAI_KEY", raising=False)
    monkeypatch.delenv("NCBI_EMAIL", raising=False)
    monkeypatch.delenv("NCBI_API_KEY", raising=False)

    assert missing_runtime_environment(config) == ["OAI_KEY"]

    monkeypatch.setenv("OAI_KEY", "test-openai-key")
    assert missing_runtime_environment(config) == []


def test_env_example_documents_every_runtime_control_used_by_deployment_code():
    text = (REPO_ROOT / ".env.example").read_text(encoding="utf-8")
    documented = set()
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("#"):
            stripped = stripped[1:].strip()
        if "=" not in stripped:
            continue
        name = stripped.split("=", 1)[0].strip()
        if name:
            documented.add(name)

    expected = {
        "OAI_KEY",
        "NCBI_EMAIL",
        "NCBI_API_KEY",
        "MEDDX_ALLOWED_ORIGINS",
        "MEDDX_DATABASE_URL",
        "MEDDX_SESSION_DB_PATH",
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
        "MEDDX_MONITORING_TIMEOUT_SECONDS",
        "MEDDX_MONITORING_MIN_LEVEL",
        "MEDDX_RATE_LIMIT_BACKEND",
        "MEDDX_RATE_LIMIT_REQUESTS_PER_MINUTE",
        "MEDDX_RATE_LIMIT_SESSION_CREATES_PER_MINUTE",
        "MEDDX_RATE_LIMIT_MODEL_ACTIONS_PER_MINUTE",
        "MEDDX_RATE_LIMIT_MAX_KEYS",
        "MEDDX_MAX_BODY_BYTES",
        "MEDDX_MAX_HEADER_BYTES",
        "MEDDX_MAX_PATIENT_INFO_CHARS",
        "MEDDX_MAX_ANSWER_CHARS",
        "MEDDX_MAX_PATIENT_ID_CHARS",
        "MEDDX_MAX_CASE_ID_CHARS",
        "MEDDX_BODY_TIMEOUT_SECONDS",
        "MEDDX_IDLE_CONNECTION_TIMEOUT_SECONDS",
        "PORT",
        "MEDDX_CLINICAL_CONFIG",
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

    monkeypatch.setenv("MEDDX_RATE_LIMIT_REQUESTS_PER_MINUTE", "-1")
    with pytest.raises(ValueError, match="must be between"):
        SecurityConfig.from_env()

    monkeypatch.setenv("MEDDX_SESSION_TTL_HOURS", "-1")
    with pytest.raises(ValueError, match="must be between"):
        SessionLifecycleConfig.from_env()

    monkeypatch.setenv("MEDDX_AUTH_MODE", "unknown")
    with pytest.raises(ValueError, match="MEDDX_AUTH_MODE"):
        AuthConfig.from_env()

    monkeypatch.setenv("MEDDX_MONITORING_WEBHOOK_URL", "http://insecure.example/hook")
    with pytest.raises(ValueError, match="HTTPS"):
        MonitoringConfig.from_env()
