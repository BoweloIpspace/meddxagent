import json
import os
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest
from tornado.testing import AsyncHTTPTestCase

from ddxdriver.clinical.api import make_app
from ddxdriver.clinical.auth import (
    AUTH_DISABLED,
    AUTH_OIDC_JWT,
    AUTH_SHARED_TOKEN,
    AuthConfig,
    AuthIdentity,
    AuthenticationError,
    DisabledAuthProvider,
    SharedTokenAuthProvider,
    authorize,
    bearer_token,
)
from ddxdriver.clinical.persistence import SQLiteClinicalSessionRepository


def test_disabled_auth_preserves_current_clinical_identity_without_claiming_authentication():
    config = AuthConfig(mode=AUTH_DISABLED)
    identity = DisabledAuthProvider(config).authenticate({})

    assert identity.subject == "anonymous"
    assert identity.authenticated is False
    assert identity.roles == frozenset({"clinician"})
    authorize(identity, frozenset({"clinician", "admin"}))


def test_shared_token_auth_uses_constant_bearer_contract():
    config = AuthConfig(
        mode=AUTH_SHARED_TOKEN,
        shared_token="top-secret",
        shared_subject="clinician-1",
        shared_roles=frozenset({"clinician"}),
    )
    provider = SharedTokenAuthProvider(config)

    identity = provider.authenticate({"Authorization": "Bearer top-secret"})
    assert identity == AuthIdentity(
        subject="clinician-1",
        roles=frozenset({"clinician"}),
        authenticated=True,
        auth_mode=AUTH_SHARED_TOKEN,
    )

    with pytest.raises(AuthenticationError):
        provider.authenticate({"Authorization": "Bearer wrong"})
    with pytest.raises(AuthenticationError):
        provider.authenticate({})


def test_bearer_parser_rejects_non_bearer_and_empty_tokens():
    assert bearer_token("Bearer abc") == "abc"
    assert bearer_token("bearer abc") == "abc"
    assert bearer_token("Basic abc") is None
    assert bearer_token("Bearer ") is None
    assert bearer_token(None) is None


def test_oidc_auth_config_fails_fast_when_connection_details_are_missing():
    with patch.dict(os.environ, {"MEDDX_AUTH_MODE": AUTH_OIDC_JWT}, clear=True):
        with pytest.raises(ValueError, match="MEDDX_AUTH_ISSUER"):
            AuthConfig.from_env()


class SubjectTokenProvider:
    def __init__(self):
        self.config = AuthConfig(mode=AUTH_SHARED_TOKEN, shared_token="unused")

    def authenticate(self, headers):
        token = bearer_token(headers.get("Authorization"))
        if token == "user-a":
            return AuthIdentity("user-a", frozenset({"clinician"}), True, "test")
        if token == "user-b":
            return AuthIdentity("user-b", frozenset({"clinician"}), True, "test")
        raise AuthenticationError("invalid test token")


class TestAuthenticatedCaseApi(AsyncHTTPTestCase):
    def get_app(self):
        self._tmpdir = tempfile.TemporaryDirectory()
        repository = SQLiteClinicalSessionRepository(
            Path(self._tmpdir.name) / "clinical.sqlite3"
        )
        self.repository = repository
        provider = SubjectTokenProvider()
        return make_app(
            session_repository=repository,
            auth_config=provider.config,
            auth_provider=provider,
        )

    def tearDown(self):
        super().tearDown()
        self._tmpdir.cleanup()

    @staticmethod
    def _case(case_id: str, complaint: str = "Headache") -> dict:
        return {
            "id": case_id,
            "patient": {
                "chiefComplaint": complaint,
                "initialInformation": "One day",
            },
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

    def test_server_cases_require_authentication_and_are_owner_scoped(self):
        case = self._case("CASE-A")
        unauthenticated = self.fetch("/api/v1/cases")
        assert unauthenticated.code == 401

        saved = self.fetch(
            "/api/v1/cases/CASE-A",
            method="PUT",
            headers={
                "Authorization": "Bearer user-a",
                "Content-Type": "application/json",
            },
            body=json.dumps(case),
        )
        assert saved.code == 200
        assert json.loads(saved.body)["case"]["id"] == "CASE-A"

        owner_list = self.fetch(
            "/api/v1/cases",
            headers={"Authorization": "Bearer user-a"},
        )
        assert owner_list.code == 200
        assert [item["id"] for item in json.loads(owner_list.body)["cases"]] == ["CASE-A"]

        other_list = self.fetch(
            "/api/v1/cases",
            headers={"Authorization": "Bearer user-b"},
        )
        assert other_list.code == 200
        assert json.loads(other_list.body) == {"cases": []}

        other_read = self.fetch(
            "/api/v1/cases/CASE-A",
            headers={"Authorization": "Bearer user-b"},
        )
        assert other_read.code == 404

    def test_case_archive_hides_record_and_delete_is_owner_scoped(self):
        case = self._case("CASE-LIFE")
        save = self.fetch(
            "/api/v1/cases/CASE-LIFE",
            method="PUT",
            headers={
                "Authorization": "Bearer user-a",
                "Content-Type": "application/json",
            },
            body=json.dumps(case),
        )
        assert save.code == 200

        archive = self.fetch(
            "/api/v1/cases/CASE-LIFE/archive",
            method="POST",
            headers={"Authorization": "Bearer user-a"},
            body="",
        )
        assert archive.code == 200
        assert json.loads(archive.body) == {"status": "archived"}

        hidden = self.fetch(
            "/api/v1/cases/CASE-LIFE",
            headers={"Authorization": "Bearer user-a"},
        )
        assert hidden.code == 404

        wrong_owner_delete = self.fetch(
            "/api/v1/cases/CASE-LIFE",
            method="DELETE",
            headers={"Authorization": "Bearer user-b"},
        )
        assert wrong_owner_delete.code == 404

        owner_delete = self.fetch(
            "/api/v1/cases/CASE-LIFE",
            method="DELETE",
            headers={"Authorization": "Bearer user-a"},
        )
        assert owner_delete.code == 204
