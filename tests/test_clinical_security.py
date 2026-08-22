import json
import os
import tempfile
from pathlib import Path
from unittest.mock import patch

from tornado.testing import AsyncHTTPTestCase

from ddxdriver.clinical.api import make_app
from ddxdriver.clinical.persistence import SQLiteClinicalSessionRepository
from ddxdriver.clinical.security import (
    SecurityConfig,
    SlidingWindowRateLimiter,
    client_rate_subject,
    session_id_from_path,
    validate_patient_id,
    validate_required_text,
)


def test_sliding_window_rate_limiter_allows_then_recovers():
    now = [100.0]
    limiter = SlidingWindowRateLimiter(max_keys=10, clock=lambda: now[0])

    first = limiter.check("client-a", 2)
    second = limiter.check("client-a", 2)
    blocked = limiter.check("client-a", 2)

    assert first.allowed is True
    assert first.remaining == 1
    assert second.allowed is True
    assert second.remaining == 0
    assert blocked.allowed is False
    assert blocked.retry_after_seconds == 60

    now[0] = 160.1
    recovered = limiter.check("client-a", 2)
    assert recovered.allowed is True
    assert recovered.remaining == 1


def test_rate_limiter_has_bounded_key_storage():
    limiter = SlidingWindowRateLimiter(max_keys=2, clock=lambda: 1.0)
    limiter.check("one", 1)
    limiter.check("two", 1)
    limiter.check("three", 1)

    assert len(limiter._buckets) == 2
    assert "one" not in limiter._buckets


def test_security_helpers_validate_bounded_inputs():
    assert validate_required_text("Headache", "patient_initial_info", 20) == "Headache"
    assert validate_patient_id("CASE-1", 20) == "CASE-1"
    assert validate_patient_id(123, 20) == 123
    assert session_id_from_path("/api/v1/clinical/sessions/abc/run") == "abc"
    assert client_rate_subject("203.0.113.10") == client_rate_subject("203.0.113.10")

    try:
        validate_required_text("x" * 21, "patient_initial_info", 20)
    except ValueError as exc:
        assert "maximum allowed length" in str(exc)
    else:
        raise AssertionError("expected oversized clinical text to be rejected")

    for invalid in ({"id": 1}, ["id"], True, ""):
        try:
            validate_patient_id(invalid, 20)
        except ValueError:
            pass
        else:
            raise AssertionError(f"expected invalid patient_id to be rejected: {invalid!r}")


def test_security_config_reads_rate_limit_environment():
    with patch.dict(
        os.environ,
        {
            "MEDDX_RATE_LIMIT_REQUESTS_PER_MINUTE": "321",
            "MEDDX_RATE_LIMIT_SESSION_CREATES_PER_MINUTE": "22",
            "MEDDX_RATE_LIMIT_MODEL_ACTIONS_PER_MINUTE": "11",
            "MEDDX_ALLOWED_ORIGINS": "https://one.example,https://two.example",
        },
        clear=False,
    ):
        config = SecurityConfig.from_env()

    assert config.requests_per_minute == 321
    assert config.session_creates_per_minute == 22
    assert config.model_actions_per_minute == 11
    assert config.allowed_origins == frozenset(
        {"https://one.example", "https://two.example"}
    )


class TestClinicalSecurityHttp(AsyncHTTPTestCase):
    def get_app(self):
        self._tmpdir = tempfile.TemporaryDirectory()
        repository = SQLiteClinicalSessionRepository(
            Path(self._tmpdir.name) / "clinical.sqlite3"
        )
        self.security_config = SecurityConfig(
            allowed_origins=frozenset({"https://allowed.example"}),
            requests_per_minute=2,
            session_creates_per_minute=1,
            model_actions_per_minute=1,
            max_rate_limit_keys=100,
            max_body_bytes=128,
            max_header_bytes=16_384,
            max_patient_info_chars=40,
            max_answer_chars=20,
            max_patient_id_chars=20,
        )
        return make_app(
            session_repository=repository,
            security_config=self.security_config,
        )

    def tearDown(self):
        super().tearDown()
        self._tmpdir.cleanup()

    def test_security_headers_are_present_and_health_is_not_rate_limited(self):
        for _ in range(4):
            response = self.fetch("/api/v1/health")
            assert response.code == 200

        assert response.headers.get("Cache-Control") == "no-store, max-age=0"
        assert response.headers.get("X-Content-Type-Options") == "nosniff"
        assert response.headers.get("X-Frame-Options") == "DENY"
        assert response.headers.get("Referrer-Policy") == "no-referrer"
        assert "frame-ancestors 'none'" in response.headers.get(
            "Content-Security-Policy", ""
        )

    def test_disallowed_browser_origin_is_rejected(self):
        response = self.fetch(
            "/api/v1/health",
            headers={"Origin": "https://attacker.example"},
        )

        assert response.code == 403
        assert json.loads(response.body) == {"error": "Origin is not allowed"}
        assert response.headers.get("Access-Control-Allow-Origin") is None

    def test_allowed_cors_preflight_still_works(self):
        response = self.fetch(
            "/api/v1/clinical/sessions",
            method="OPTIONS",
            headers={
                "Origin": "https://allowed.example",
                "Access-Control-Request-Method": "POST",
                "Access-Control-Request-Headers": "Content-Type",
            },
        )

        assert response.code == 204
        assert response.headers.get("Access-Control-Allow-Origin") == "https://allowed.example"
        assert "POST" in response.headers.get("Access-Control-Allow-Methods", "")

    def test_oversized_request_body_is_413_before_model_work(self):
        response = self.fetch(
            "/api/v1/clinical/sessions",
            method="POST",
            headers={"Content-Type": "application/json"},
            body=json.dumps({"patient_initial_info": "x" * 200}),
        )

        assert response.code == 413
        assert json.loads(response.body) == {"error": "Request body is too large"}

    def test_non_json_payload_is_rejected(self):
        with patch.dict(os.environ, {"OAI_KEY": "test-key"}, clear=False):
            response = self.fetch(
                "/api/v1/clinical/sessions",
                method="POST",
                headers={"Content-Type": "text/plain"},
                body='{"patient_initial_info":"Headache"}',
            )

        assert response.code == 400
        assert json.loads(response.body) == {
            "error": "Content-Type must be application/json"
        }

    def test_client_rate_limit_ignores_spoofed_forwarded_for(self):
        first = self.fetch(
            "/api/v1/clinical/sessions/missing-session",
            headers={"X-Forwarded-For": "198.51.100.10"},
        )
        second = self.fetch(
            "/api/v1/clinical/sessions/missing-session",
            headers={"X-Forwarded-For": "198.51.100.11"},
        )
        third = self.fetch(
            "/api/v1/clinical/sessions/missing-session",
            headers={"X-Forwarded-For": "198.51.100.12"},
        )

        assert first.code == 404
        assert second.code == 404
        assert third.code == 429
        assert json.loads(third.body) == {
            "error": "Too many requests. Please retry shortly."
        }
        assert int(third.headers.get("Retry-After", "0")) >= 1
        assert third.headers.get("X-RateLimit-Remaining") == "0"


class TestClinicalModelRateLimitHttp(AsyncHTTPTestCase):
    def get_app(self):
        self._tmpdir = tempfile.TemporaryDirectory()
        repository = SQLiteClinicalSessionRepository(
            Path(self._tmpdir.name) / "clinical.sqlite3"
        )
        config = SecurityConfig(
            requests_per_minute=100,
            session_creates_per_minute=100,
            model_actions_per_minute=1,
            max_rate_limit_keys=100,
        )
        return make_app(session_repository=repository, security_config=config)

    def tearDown(self):
        super().tearDown()
        self._tmpdir.cleanup()

    def test_model_actions_have_per_session_limit(self):
        first = self.fetch(
            "/api/v1/clinical/sessions/session-a/question",
            method="POST",
            body="",
        )
        second = self.fetch(
            "/api/v1/clinical/sessions/session-a/question",
            method="POST",
            body="",
        )
        different_session = self.fetch(
            "/api/v1/clinical/sessions/session-b/question",
            method="POST",
            body="",
        )

        assert first.code == 404
        assert second.code == 429
        assert second.headers.get("X-RateLimit-Limit") == "1"
        assert different_session.code == 404
