import json
import os
from unittest.mock import patch

from tornado.testing import AsyncHTTPTestCase

from ddxdriver.clinical.api import make_app


class FakeClinicalSession:
    def __init__(self, patient_initial_info: str, patient_id=None):
        self.patient_initial_info = patient_initial_info
        self.patient_id = patient_id
        self.patient_profile = ""
        self.history_complete = False
        self.pending_question = None
        self.history_turns = []
        self.dialogue_history = ""
        self.result = None

    def snapshot(self):
        return {
            "patient_initial_info": self.patient_initial_info,
            "patient_profile": self.patient_profile,
            "history_complete": self.history_complete,
            "pending_question": self.pending_question,
            "history_turns": self.history_turns,
            "dialogue_history": self.dialogue_history,
            "result": self.result,
        }

    def update_patient_initial_info(self, patient_initial_info: str):
        self.patient_initial_info = patient_initial_info
        if self.history_complete:
            self.patient_profile = self._build_profile()

    def next_question(self):
        if self.pending_question:
            raise RuntimeError("Submit the pending patient response before requesting another question")
        self.pending_question = "Any fever?"
        self.history_turns.append({"question": self.pending_question, "answer": ""})
        return self.pending_question

    def submit_answer(self, answer: str):
        if not self.pending_question:
            raise RuntimeError("No MEDDxAgent question is waiting for a patient response")
        self.history_turns[-1]["answer"] = answer
        self.pending_question = None
        self.dialogue_history = "Doctor: Any fever?\nPatient: " + answer + "\n"

    def _build_profile(self):
        return (
            "Clinical information available before history:\n"
            + self.patient_initial_info
            + "\n\nHistory-derived patient profile:\n- Reports fever"
        )

    def finish_history(self):
        if self.pending_question:
            raise RuntimeError("Cannot finish history while a question is waiting for a patient response")
        self.history_complete = True
        self.patient_profile = self._build_profile()
        return self.patient_profile

    def run(self):
        if not self.history_complete:
            raise RuntimeError("Finish the human-in-the-loop history before running retrieval/diagnosis")
        self.result = {
            "ranked_differential": ["Diagnosis A", "Diagnosis B"],
            "rationale": "engine rationale",
            "dialogue_history": self.dialogue_history,
            "rag_content": "retrieved evidence",
            "intermediate_differentials": [["Diagnosis A", "Diagnosis B"]],
        }
        return self.result


class TestClinicalApi(AsyncHTTPTestCase):
    def get_app(self):
        return make_app()

    def _create_fake_session(self) -> str:
        def fake_create(patient_initial_info, patient_id, config):
            return FakeClinicalSession(patient_initial_info, patient_id)

        with patch("ddxdriver.clinical.api.create_clinical_session", side_effect=fake_create):
            response = self.fetch(
                "/api/v1/clinical/sessions",
                method="POST",
                headers={"Content-Type": "application/json"},
                body=json.dumps({"patient_initial_info": "Chief complaint: Headache"}),
            )
        assert response.code == 201
        return json.loads(response.body)["session_id"]

    def test_health(self):
        response = self.fetch("/api/v1/health")
        assert response.code == 200
        assert json.loads(response.body) == {"status": "ok"}

    def test_production_origin_cors_and_preflight(self):
        origin = "https://meddxagentfrontend.vercel.app"
        response = self.fetch("/api/v1/health", headers={"Origin": origin})
        assert response.code == 200
        assert response.headers.get("Access-Control-Allow-Origin") == origin
        assert response.headers.get("Vary") == "Origin"

        preflight = self.fetch(
            "/api/v1/clinical/sessions",
            method="OPTIONS",
            headers={"Origin": origin, "Access-Control-Request-Method": "POST"},
        )
        assert preflight.code == 204
        assert preflight.headers.get("Access-Control-Allow-Origin") == origin
        assert "POST" in preflight.headers.get("Access-Control-Allow-Methods", "")

    def test_readiness_reports_missing_model_environment(self):
        with patch.dict(os.environ, {}, clear=True):
            response = self.fetch("/api/v1/ready")
        assert response.code == 503
        assert json.loads(response.body) == {
            "status": "not_ready",
            "missing_environment": ["OAI_KEY"],
        }

    def test_readiness_is_ready_when_model_environment_is_present(self):
        with patch.dict(os.environ, {"OAI_KEY": "test-key"}, clear=True):
            response = self.fetch("/api/v1/ready")
        assert response.code == 200
        assert json.loads(response.body) == {"status": "ready"}

    def test_missing_session_is_404(self):
        response = self.fetch("/api/v1/clinical/sessions/does-not-exist")
        assert response.code == 404
        assert json.loads(response.body) == {"error": "Clinical session not found"}

    def test_invalid_session_transitions_are_400(self):
        session_id = self._create_fake_session()

        answer = self.fetch(
            f"/api/v1/clinical/sessions/{session_id}/answer",
            method="POST",
            headers={"Content-Type": "application/json"},
            body=json.dumps({"answer": "Yes"}),
        )
        assert answer.code == 400
        assert json.loads(answer.body) == {
            "error": "No MEDDxAgent question is waiting for a patient response"
        }

        run = self.fetch(
            f"/api/v1/clinical/sessions/{session_id}/run",
            method="POST",
            body="",
        )
        assert run.code == 400
        assert json.loads(run.body) == {
            "error": "Finish the human-in-the-loop history before running retrieval/diagnosis"
        }

        question = self.fetch(
            f"/api/v1/clinical/sessions/{session_id}/question",
            method="POST",
            body="",
        )
        assert question.code == 200

        second_question = self.fetch(
            f"/api/v1/clinical/sessions/{session_id}/question",
            method="POST",
            body="",
        )
        assert second_question.code == 400
        assert json.loads(second_question.body) == {
            "error": "Submit the pending patient response before requesting another question"
        }

        finish = self.fetch(
            f"/api/v1/clinical/sessions/{session_id}/history/finish",
            method="POST",
            body="",
        )
        assert finish.code == 400
        assert json.loads(finish.body) == {
            "error": "Cannot finish history while a question is waiting for a patient response"
        }

    def test_full_session_contract_preserves_late_clinical_context(self):
        def fake_create(patient_initial_info, patient_id, config):
            return FakeClinicalSession(patient_initial_info, patient_id)

        with patch("ddxdriver.clinical.api.create_clinical_session", side_effect=fake_create):
            create_response = self.fetch(
                "/api/v1/clinical/sessions",
                method="POST",
                headers={"Content-Type": "application/json"},
                body=json.dumps(
                    {
                        "patient_id": "CASE-TEST",
                        "patient_initial_info": "Chief complaint: Headache",
                    }
                ),
            )

            assert create_response.code == 201
            created = json.loads(create_response.body)
            session_id = created["session_id"]
            assert created["patient_initial_info"] == "Chief complaint: Headache"
            assert created["history_turns"] == []

            question_response = self.fetch(
                f"/api/v1/clinical/sessions/{session_id}/question",
                method="POST",
                body="",
            )
            assert question_response.code == 200
            question = json.loads(question_response.body)
            assert question["question"] == "Any fever?"
            assert question["pending_question"] == "Any fever?"

            answer_response = self.fetch(
                f"/api/v1/clinical/sessions/{session_id}/answer",
                method="POST",
                headers={"Content-Type": "application/json"},
                body=json.dumps({"answer": "Yes"}),
            )
            assert answer_response.code == 200
            answered = json.loads(answer_response.body)
            assert answered["pending_question"] is None
            assert answered["history_turns"] == [{"question": "Any fever?", "answer": "Yes"}]

            final_context = (
                "Chief complaint: Headache\n"
                "Temperature: 39 C\n"
                "Investigation - CT head: Normal"
            )
            context_response = self.fetch(
                f"/api/v1/clinical/sessions/{session_id}/context",
                method="POST",
                headers={"Content-Type": "application/json"},
                body=json.dumps({"patient_initial_info": final_context}),
            )
            assert context_response.code == 200
            refreshed = json.loads(context_response.body)
            assert refreshed["patient_initial_info"] == final_context
            assert refreshed["history_turns"] == [{"question": "Any fever?", "answer": "Yes"}]

            finish_response = self.fetch(
                f"/api/v1/clinical/sessions/{session_id}/history/finish",
                method="POST",
                body="",
            )
            assert finish_response.code == 200
            finished = json.loads(finish_response.body)
            assert finished["history_complete"] is True
            assert "Temperature: 39 C" in finished["patient_profile"]
            assert "Investigation - CT head: Normal" in finished["patient_profile"]
            assert "History-derived patient profile" in finished["patient_profile"]

            run_response = self.fetch(
                f"/api/v1/clinical/sessions/{session_id}/run",
                method="POST",
                body="",
            )
            assert run_response.code == 200
            ran = json.loads(run_response.body)
            assert ran["result"]["ranked_differential"] == ["Diagnosis A", "Diagnosis B"]
            assert ran["result"]["dialogue_history"] == "Doctor: Any fever?\nPatient: Yes\n"

            get_response = self.fetch(f"/api/v1/clinical/sessions/{session_id}")
            assert get_response.code == 200
            stored = json.loads(get_response.body)
            assert stored["patient_initial_info"] == final_context
            assert stored["result"] == ran["result"]
