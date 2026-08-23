import asyncio
import json
import sqlite3
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from ddxdriver.clinical.api import ClinicalSessionStore
from ddxdriver.clinical.history import ClinicalHistorySession
from ddxdriver.clinical.persistence import SQLiteClinicalSessionRepository
from ddxdriver.clinical.results import ClinicalResult
from ddxdriver.clinical.session import ClinicalSession
from ddxdriver.utils import DialogueHistory, Patient


class FakePersistentSession:
    def __init__(self, patient_initial_info: str, patient_id=None):
        self.patient_initial_info = patient_initial_info
        self.patient_id = patient_id
        self.marker = "created"

    def persistence_state(self):
        return {
            "version": 1,
            "patient": {
                "patient_id": self.patient_id,
                "patient_initial_info": self.patient_initial_info,
                "patient_profile": "",
            },
            "history": None,
            "result": None,
            "marker": self.marker,
        }

    def restore_persistence_state(self, state: dict):
        self.patient_id = state["patient"]["patient_id"]
        self.patient_initial_info = state["patient"]["patient_initial_info"]
        self.marker = state["marker"]


def _fake_create(patient_initial_info, patient_id, config):
    return FakePersistentSession(patient_initial_info, patient_id)


def _history(patient: Patient, dialogue: list[tuple[str, str]] | None = None):
    dialogue_history = DialogueHistory()
    if dialogue:
        dialogue_history.add_dialogue(dialogue)

    history = ClinicalHistorySession.__new__(ClinicalHistorySession)
    history.patient = patient
    history.history_taking_agent = SimpleNamespace(dialogue_history=dialogue_history)
    history.complete = False
    history.derived_profile = ""
    history.conversation_goals = ""
    return history


def test_sqlite_repository_round_trips_json_state(tmp_path):
    repository = SQLiteClinicalSessionRepository(tmp_path / "clinical.sqlite3")
    state = {
        "version": 1,
        "patient": {
            "patient_id": "CASE-1",
            "patient_initial_info": "Headache",
            "patient_profile": "",
        },
        "history": None,
        "result": None,
    }

    repository.save("session-1", state)

    assert repository.load("session-1") == state
    assert repository.load("missing") is None


def test_sqlite_repository_latest_write_wins_and_delete_removes_state(tmp_path):
    repository = SQLiteClinicalSessionRepository(tmp_path / "clinical.sqlite3")
    first = {
        "version": 1,
        "patient": {
            "patient_id": "CASE-LATEST",
            "patient_initial_info": "Initial context",
            "patient_profile": "",
        },
        "history": None,
        "result": None,
        "marker": "first",
    }
    second = {**first, "marker": "second"}

    repository.save("session-latest", first)
    repository.save("session-latest", second)

    assert repository.load("session-latest")["marker"] == "second"

    repository.delete("session-latest")
    assert repository.load("session-latest") is None


def test_sqlite_repository_rejects_corrupt_json_instead_of_returning_partial_state(tmp_path):
    database_path = tmp_path / "clinical.sqlite3"
    repository = SQLiteClinicalSessionRepository(database_path)

    with sqlite3.connect(database_path) as connection:
        connection.execute(
            """
            INSERT INTO clinical_sessions (session_id, state_json, created_at, updated_at)
            VALUES (?, ?, ?, ?)
            """,
            ("corrupt-session", "{not-json", "now", "now"),
        )

    with pytest.raises(json.JSONDecodeError):
        repository.load("corrupt-session")


def test_clinical_session_state_restores_history_and_result():
    patient = Patient(
        patient_id="CASE-2",
        patient_initial_info="Chief complaint: Headache",
        patient_profile="Clinical information available before history:\nChief complaint: Headache",
    )
    session = ClinicalSession.__new__(ClinicalSession)
    session.patient = patient
    session.history = _history(
        patient,
        [("doctor", "Any fever?"), ("patient", "Yes")],
    )
    session.history.complete = True
    session.history.derived_profile = "- Reports fever"
    session.history.conversation_goals = "Clarify infection symptoms"
    session.driver = None
    session.result = ClinicalResult(
        ranked_differential=["Diagnosis A", "Diagnosis B"],
        rationale="engine rationale",
        dialogue_history="Doctor: Any fever?\nPatient: Yes\n",
        rag_content="retrieved evidence",
        intermediate_differentials=[["Diagnosis A", "Diagnosis B"]],
    )

    state = session.persistence_state()
    json.dumps(state)

    restored_patient = Patient(patient_id="CASE-2", patient_initial_info="placeholder")
    restored = ClinicalSession.__new__(ClinicalSession)
    restored.patient = restored_patient
    restored.history = _history(restored_patient)
    restored.driver = object()
    restored.result = None

    restored.restore_persistence_state(state)

    assert restored.patient.patient_initial_info == "Chief complaint: Headache"
    assert restored.patient.patient_profile == patient.patient_profile
    assert restored.history.dialogue_history == "Doctor: Any fever?\nPatient: Yes\n"
    assert restored.history.complete is True
    assert restored.history.derived_profile == "- Reports fever"
    assert restored.history.conversation_goals == "Clarify infection symptoms"
    assert restored.driver is None
    assert restored.result.to_dict() == session.result.to_dict()


def test_session_store_rehydrates_after_process_cache_is_lost(tmp_path):
    database_path = tmp_path / "clinical.sqlite3"
    first_repository = SQLiteClinicalSessionRepository(database_path)
    first_store = ClinicalSessionStore({}, repository=first_repository)

    with patch("ddxdriver.clinical.api.create_clinical_session", side_effect=_fake_create):
        session_id, session = asyncio.run(
            first_store.create("Chief complaint: Headache", patient_id="CASE-3")
        )

    session.marker = "persisted after mutation"
    asyncio.run(first_store.persist(session_id, session))

    second_repository = SQLiteClinicalSessionRepository(database_path)
    second_store = ClinicalSessionStore({}, repository=second_repository)
    with patch("ddxdriver.clinical.api.create_clinical_session", side_effect=_fake_create):
        restored = asyncio.run(second_store.get(session_id))

    assert restored.patient_id == "CASE-3"
    assert restored.patient_initial_info == "Chief complaint: Headache"
    assert restored.marker == "persisted after mutation"


def test_session_store_rehydrates_the_latest_of_multiple_persisted_mutations(tmp_path):
    database_path = tmp_path / "clinical.sqlite3"
    first_store = ClinicalSessionStore(
        {},
        repository=SQLiteClinicalSessionRepository(database_path),
    )

    with patch("ddxdriver.clinical.api.create_clinical_session", side_effect=_fake_create):
        session_id, session = asyncio.run(
            first_store.create("Chief complaint: Abdominal pain", patient_id="CASE-4")
        )

    session.marker = "after history"
    asyncio.run(first_store.persist(session_id, session))
    session.marker = "after investigations"
    asyncio.run(first_store.persist(session_id, session))

    restarted_store = ClinicalSessionStore(
        {},
        repository=SQLiteClinicalSessionRepository(database_path),
    )
    with patch("ddxdriver.clinical.api.create_clinical_session", side_effect=_fake_create):
        restored = asyncio.run(restarted_store.get(session_id))

    assert restored.marker == "after investigations"


def test_failed_initial_persist_does_not_leave_non_durable_hot_session(tmp_path):
    repository = SQLiteClinicalSessionRepository(tmp_path / "clinical.sqlite3")
    store = ClinicalSessionStore({}, repository=repository)

    with (
        patch("ddxdriver.clinical.api.create_clinical_session", side_effect=_fake_create),
        patch.object(repository, "save", side_effect=RuntimeError("disk unavailable")),
        pytest.raises(RuntimeError, match="disk unavailable"),
    ):
        asyncio.run(store.create("Chief complaint: Headache", patient_id="CASE-5"))

    assert store.sessions == {}
    assert store.locks == {}
