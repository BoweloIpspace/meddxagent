import json
import tempfile
from pathlib import Path

from tornado.testing import AsyncHTTPTestCase

from ddxdriver.clinical.api import make_app
from ddxdriver.clinical.auth import AUTH_SHARED_TOKEN, AuthConfig, AuthIdentity, AuthenticationError, bearer_token
from ddxdriver.clinical.persistence import SQLiteClinicalSessionRepository


class PaginationTokenProvider:
    def __init__(self):
        self.config = AuthConfig(mode=AUTH_SHARED_TOKEN, shared_token="unused")

    def authenticate(self, headers):
        if bearer_token(headers.get("Authorization")) == "pagination-user":
            return AuthIdentity(
                "pagination-user",
                frozenset({"clinician"}),
                True,
                "test",
            )
        raise AuthenticationError("invalid test token")


def _case(case_id: str) -> dict:
    return {
        "id": case_id,
        "patient": {"chiefComplaint": "Headache", "initialInformation": "One day"},
        "status": "draft",
        "createdAt": "2026-08-23T00:00:00Z",
        "updatedAt": "2026-08-23T00:00:00Z",
        "currentIteration": 0,
        "differential": [],
        "rationale": "",
        "dialogueHistory": "",
        "ragContent": "",
        "workflow": {
            "historyQuestions": [],
            "historySummary": {
                "positiveFindings": [],
                "negativeFindings": [],
                "riskFactors": [],
                "redFlags": [],
            },
            "examination": {},
            "investigations": [],
        },
    }


def test_sqlite_case_pages_cover_more_than_backend_page_limit_without_duplicates(tmp_path):
    repository = SQLiteClinicalSessionRepository(tmp_path / "clinical.sqlite3")
    for index in range(1201):
        repository.save_case("pagination-user", f"CASE-{index:04d}", _case(f"CASE-{index:04d}"))

    first = repository.list_cases("pagination-user", limit=500, offset=0)
    second = repository.list_cases("pagination-user", limit=500, offset=500)
    third = repository.list_cases("pagination-user", limit=500, offset=1000)

    combined = first + second + third
    assert len(first) == 500
    assert len(second) == 500
    assert len(third) == 201
    assert len(combined) == 1201
    assert len({case["id"] for case in combined}) == 1201


def test_sqlite_case_pagination_rejects_negative_offset(tmp_path):
    repository = SQLiteClinicalSessionRepository(tmp_path / "clinical.sqlite3")
    try:
        repository.list_cases("pagination-user", limit=100, offset=-1)
    except ValueError as exc:
        assert "offset" in str(exc).lower()
    else:
        raise AssertionError("Negative case offset must be rejected")


class TestCasePaginationApi(AsyncHTTPTestCase):
    def get_app(self):
        self._tmpdir = tempfile.TemporaryDirectory()
        self.repository = SQLiteClinicalSessionRepository(
            Path(self._tmpdir.name) / "clinical.sqlite3"
        )
        for index in range(7):
            case_id = f"API-{index}"
            self.repository.save_case("pagination-user", case_id, _case(case_id))
        provider = PaginationTokenProvider()
        return make_app(
            session_repository=self.repository,
            auth_config=provider.config,
            auth_provider=provider,
        )

    def tearDown(self):
        super().tearDown()
        self._tmpdir.cleanup()

    def test_cases_endpoint_accepts_limit_and_offset(self):
        response = self.fetch(
            "/api/v1/cases?limit=3&offset=3",
            headers={"Authorization": "Bearer pagination-user"},
        )
        assert response.code == 200
        payload = json.loads(response.body)
        assert len(payload["cases"]) == 3

        expected = self.repository.list_cases("pagination-user", limit=3, offset=3)
        assert [case["id"] for case in payload["cases"]] == [case["id"] for case in expected]

    def test_cases_endpoint_rejects_invalid_offset(self):
        response = self.fetch(
            "/api/v1/cases?limit=3&offset=-1",
            headers={"Authorization": "Bearer pagination-user"},
        )
        assert response.code == 400
        assert "offset" in json.loads(response.body)["error"].lower()
