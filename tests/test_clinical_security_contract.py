import json
import tempfile
from pathlib import Path

from tornado.testing import AsyncHTTPTestCase

from ddxdriver.clinical.api import make_app
from ddxdriver.clinical.persistence import SQLiteClinicalSessionRepository
from ddxdriver.clinical.security import SecurityConfig


class TestProductionSecurityResponseContract(AsyncHTTPTestCase):
    def get_app(self):
        self._tmpdir = tempfile.TemporaryDirectory()
        repository = SQLiteClinicalSessionRepository(
            Path(self._tmpdir.name) / "clinical.sqlite3"
        )
        config = SecurityConfig(
            allowed_origins=frozenset({"https://meddxagentfrontend.vercel.app"}),
            requests_per_minute=1,
            session_creates_per_minute=100,
            model_actions_per_minute=100,
            max_rate_limit_keys=100,
        )
        return make_app(session_repository=repository, security_config=config)

    def tearDown(self):
        super().tearDown()
        self._tmpdir.cleanup()

    def test_rate_limited_browser_response_keeps_security_and_correlation_headers(self):
        headers = {
            "Origin": "https://meddxagentfrontend.vercel.app",
            "X-Request-ID": "security-contract-1",
        }

        first = self.fetch(
            "/api/v1/clinical/sessions/missing-session",
            headers=headers,
        )
        second = self.fetch(
            "/api/v1/clinical/sessions/missing-session",
            headers={**headers, "X-Request-ID": "security-contract-2"},
        )

        assert first.code == 404
        assert second.code == 429
        assert json.loads(second.body) == {
            "error": "Too many requests. Please retry shortly."
        }
        assert int(second.headers["Retry-After"]) >= 1
        assert second.headers["X-RateLimit-Remaining"] == "0"
        assert second.headers["X-Request-ID"] == "security-contract-2"
        assert second.headers["Access-Control-Allow-Origin"] == (
            "https://meddxagentfrontend.vercel.app"
        )

        exposed = second.headers.get("Access-Control-Expose-Headers", "")
        assert "X-Request-ID" in exposed
        assert "Retry-After" in exposed
        assert "X-RateLimit-Limit" in exposed
        assert "X-RateLimit-Remaining" in exposed

        assert second.headers["Cache-Control"] == "no-store, max-age=0"
        assert second.headers["Pragma"] == "no-cache"
        assert second.headers["X-Content-Type-Options"] == "nosniff"
        assert second.headers["X-Frame-Options"] == "DENY"
        assert second.headers["Referrer-Policy"] == "no-referrer"
        assert second.headers["Permissions-Policy"] == (
            "camera=(), microphone=(), geolocation=()"
        )
        assert "default-src 'none'" in second.headers["Content-Security-Policy"]
        assert "frame-ancestors 'none'" in second.headers["Content-Security-Policy"]

    def test_cors_preflight_does_not_consume_the_clinical_request_budget(self):
        preflight_headers = {
            "Origin": "https://meddxagentfrontend.vercel.app",
            "Access-Control-Request-Method": "GET",
            "Access-Control-Request-Headers": "X-Request-ID",
        }

        for _ in range(3):
            response = self.fetch(
                "/api/v1/clinical/sessions/missing-session",
                method="OPTIONS",
                headers=preflight_headers,
            )
            assert response.code == 204

        first_clinical_request = self.fetch(
            "/api/v1/clinical/sessions/missing-session",
            headers={"Origin": "https://meddxagentfrontend.vercel.app"},
        )
        assert first_clinical_request.code == 404
