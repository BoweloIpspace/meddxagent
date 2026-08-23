import asyncio
import json
import os
import tempfile
from pathlib import Path
from unittest.mock import patch

from tornado.testing import AsyncHTTPTestCase

from ddxdriver.clinical.api import ClinicalSessionStore, make_app
from ddxdriver.clinical.persistence import SQLiteClinicalSessionRepository


class DeterministicClinicalSession:
    """Offline test double exercising the application contract, not clinical accuracy."""

    def __init__(self, patient_initial_info: str, patient_id=None):
        self.patient_initial_info = patient_initial_info
        self.patient_id = patient_id
        self.patient_profile = ""
        self.history_complete = False
        self.pending_question = None
        self.history_turns: list[dict[str, str]] = []
        self.dialogue_history = ""
        self.result = None
        self._questions = [
            "Where did the abdominal pain start and where is it now?",
            "Is the pain worse with movement or coughing?",
        ]

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

    def persistence_state(self):
        return {
            "version": 1,
            "patient": {
                "patient_id": self.patient_id,
                "patient_initial_info": self.patient_initial_info,
                "patient_profile": self.patient_profile,
            },
            "history": {
                "complete": self.history_complete,
                "pending_question": self.pending_question,
                "turns": self.history_turns,
                "dialogue_history": self.dialogue_history,
            },
            "result": self.result,
            "questions": self._questions,
        }

    def restore_persistence_state(self, state: dict):
        patient = state["patient"]
        history = state["history"]
        self.patient_id = patient["patient_id"]
        self.patient_initial_info = patient["patient_initial_info"]
        self.patient_profile = patient["patient_profile"]
        self.history_complete = history["complete"]
        self.pending_question = history["pending_question"]
        self.history_turns = history["turns"]
        self.dialogue_history = history["dialogue_history"]
        self.result = state["result"]
        self._questions = state["questions"]

    def update_patient_initial_info(self, patient_initial_info: str):
        self.patient_initial_info = patient_initial_info

    def next_question(self):
        if self.pending_question:
            raise RuntimeError("Submit the pending patient response before requesting another question")
        answered = len(self.history_turns)
        if answered >= len(self._questions):
            return None
        question = self._questions[answered]
        self.pending_question = question
        self.history_turns.append({"question": question, "answer": ""})
        return question

    def submit_answer(self, answer: str):
        if not self.pending_question:
            raise RuntimeError("No MEDDxAgent question is waiting for a patient response")
        self.history_turns[-1]["answer"] = answer
        self.pending_question = None
        self.dialogue_history = "".join(
            f"Doctor: {turn['question']}\nPatient: {turn['answer']}\n"
            for turn in self.history_turns
            if turn["answer"]
        )

    def finish_history(self):
        if self.pending_question:
            raise RuntimeError("Cannot finish history while a question is waiting for a patient response")
        self.history_complete = True
        self.patient_profile = (
            "Clinical information available before history:\n"
            + self.patient_initial_info
            + "\n\nHistory-derived patient profile:\n"
            + "\n".join(f"- {turn['answer']}" for turn in self.history_turns)
        )

    def run(self):
        if not self.history_complete:
            raise RuntimeError("Finish the human-in-the-loop history before running retrieval/diagnosis")
        self.result = {
            "ranked_differential": ["Synthetic diagnosis A", "Synthetic diagnosis B"],
            "rationale": "deterministic test rationale",
            "dialogue_history": self.dialogue_history,
            "rag_content": "deterministic retrieved evidence",
            "intermediate_differentials": [
                ["Synthetic diagnosis B", "Synthetic diagnosis A"],
                ["Synthetic diagnosis A", "Synthetic diagnosis B"],
            ],
        }


def _fake_create(patient_initial_info, patient_id, config):
    return DeterministicClinicalSession(patient_initial_info, patient_id)


class TestFullClinicalApplicationFlow(AsyncHTTPTestCase):
    def get_app(self):
        self._tmpdir = tempfile.TemporaryDirectory()
        self.database_path = Path(self._tmpdir.name) / "clinical.sqlite3"
        self.repository = SQLiteClinicalSessionRepository(self.database_path)
        return make_app(session_repository=self.repository)

    def tearDown(self):
        super().tearDown()
        self._tmpdir.cleanup()

    def test_full_consultation_flow_and_restart_recovery(self):
        initial = (
            "Age: 27\nSex: Male\nChief complaint: Abdominal pain\n"
            "Initial clinical information: 18 hours of worsening abdominal pain with nausea"
        )
        final_context = initial + "\nAbdominal examination: Focal right lower quadrant tenderness"

        with (
            patch.dict(os.environ, {"OAI_KEY": "test-key"}, clear=True),
            patch("ddxdriver.clinical.api.create_clinical_session", side_effect=_fake_create),
        ):
            created_response = self.fetch(
                "/api/v1/clinical/sessions",
                method="POST",
                headers={"Content-Type": "application/json"},
                body=json.dumps(
                    {"patient_id": "CASE-E2E", "patient_initial_info": initial}
                ),
            )
            assert created_response.code == 201
            session_id = json.loads(created_response.body)["session_id"]

            first_question = self.fetch(
                f"/api/v1/clinical/sessions/{session_id}/question",
                method="POST",
                body="",
            )
            assert first_question.code == 200
            assert "Where did" in json.loads(first_question.body)["question"]

            first_answer = self.fetch(
                f"/api/v1/clinical/sessions/{session_id}/answer",
                method="POST",
                headers={"Content-Type": "application/json"},
                body=json.dumps(
                    {"answer": "It started near the umbilicus and moved to the right lower abdomen."}
                ),
            )
            assert first_answer.code == 200

            second_question = self.fetch(
                f"/api/v1/clinical/sessions/{session_id}/question",
                method="POST",
                body="",
            )
            assert second_question.code == 200

            second_answer = self.fetch(
                f"/api/v1/clinical/sessions/{session_id}/answer",
                method="POST",
                headers={"Content-Type": "application/json"},
                body=json.dumps({"answer": "Yes, movement and coughing make it worse."}),
            )
            assert second_answer.code == 200

            context = self.fetch(
                f"/api/v1/clinical/sessions/{session_id}/context",
                method="POST",
                headers={"Content-Type": "application/json"},
                body=json.dumps({"patient_initial_info": final_context}),
            )
            assert context.code == 200

            finish = self.fetch(
                f"/api/v1/clinical/sessions/{session_id}/history/finish",
                method="POST",
                body="",
            )
            assert finish.code == 200
            assert json.loads(finish.body)["history_complete"] is True

            run = self.fetch(
                f"/api/v1/clinical/sessions/{session_id}/run",
                method="POST",
                body="",
            )
            assert run.code == 200
            result = json.loads(run.body)["result"]
            assert result["ranked_differential"] == [
                "Synthetic diagnosis A",
                "Synthetic diagnosis B",
            ]
            assert result["rag_content"] == "deterministic retrieved evidence"
            assert "confidence" not in result

            fetched = self.fetch(f"/api/v1/clinical/sessions/{session_id}")
            assert fetched.code == 200
            assert json.loads(fetched.body)["patient_initial_info"] == final_context

            restarted_store = ClinicalSessionStore(
                {},
                repository=SQLiteClinicalSessionRepository(self.database_path),
            )
            restored = asyncio.run(restarted_store.get(session_id))
            assert restored.snapshot()["result"] == result
            assert restored.snapshot()["history_turns"][-1]["answer"].startswith("Yes, movement")
