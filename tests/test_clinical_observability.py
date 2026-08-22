import json
import tempfile
from pathlib import Path

from tornado.testing import AsyncHTTPTestCase

from ddxdriver.clinical.api import make_app
from ddxdriver.clinical.observability import (
    build_error_event,
    build_event,
    request_id_from_header,
    session_reference,
)
from ddxdriver.clinical.persistence import SQLiteClinicalSessionRepository


def test_event_builder_drops_clinical_content_fields():
    event = build_event(
        "clinical.audit",
        action="history.answer.submit",
        answer="secret patient answer",
        patient_initial_info="secret patient context",
        question="secret model question",
        request_id="req-1",
    )

    serialized = json.dumps(event)
    assert "secret patient answer" not in serialized
    assert "secret patient context" not in serialized
    assert "secret model question" not in serialized
    assert event["request_id"] == "req-1"


def test_error_event_excludes_exception_message_and_raw_session_id():
    raw_session_id = "clinical-session-secret"
    try:
        raise RuntimeError("provider failure containing sensitive clinical text")
    except RuntimeError as exc:
        event = build_error_event(
            "clinical.error",
            exc,
            operation="diagnosis.run",
            session_ref=session_reference(raw_session_id),
        )

    serialized = json.dumps(event)
    assert "sensitive clinical text" not in serialized
    assert raw_session_id not in serialized
    assert event["error_type"] == "RuntimeError"
    assert event["session_ref"] == session_reference(raw_session_id)


def test_request_id_validation_rejects_log_injection():
    assert request_id_from_header("trace-123") == "trace-123"
    generated = request_id_from_header("bad\nheader")
    assert generated != "bad\nheader"
    assert "\n" not in generated


class TestClinicalObservabilityHttp(AsyncHTTPTestCase):
    def get_app(self):
        self._tmpdir = tempfile.TemporaryDirectory()
        repository = SQLiteClinicalSessionRepository(
            Path(self._tmpdir.name) / "clinical.sqlite3"
        )
        return make_app(session_repository=repository)

    def tearDown(self):
        super().tearDown()
        self._tmpdir.cleanup()

    def test_error_response_carries_request_id(self):
        response = self.fetch(
            "/api/v1/clinical/sessions/missing-session",
            headers={"X-Request-ID": "trace-missing-1"},
        )

        assert response.code == 404
        body = json.loads(response.body)
        assert body["error"] == "Clinical session not found"
        assert body["request_id"] == "trace-missing-1"

    def test_request_id_is_exposed_as_response_header(self):
        response = self.fetch(
            "/api/v1/health",
            headers={"X-Request-ID": "trace-health-1"},
        )

        assert response.code == 200
        assert response.headers.get("X-Request-ID") == "trace-health-1"
