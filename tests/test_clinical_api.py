import json

from tornado.testing import AsyncHTTPTestCase

from ddxdriver.clinical.api import make_app


class TestClinicalApi(AsyncHTTPTestCase):
    def get_app(self):
        return make_app()

    def test_health(self):
        response = self.fetch("/api/v1/health")
        assert response.code == 200
        assert json.loads(response.body) == {"status": "ok"}

    def test_missing_session_is_404(self):
        response = self.fetch("/api/v1/clinical/sessions/does-not-exist")
        assert response.code == 404
        assert json.loads(response.body) == {"error": "Clinical session not found"}
