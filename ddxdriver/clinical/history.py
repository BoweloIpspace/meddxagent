from ddxdriver.ddxdrivers.utils import dialogue_to_patient_profile
from ddxdriver.history_taking_agents.llm_history_taking import LLMHistoryTaking
from ddxdriver.models.base import Model
from ddxdriver.utils import Patient

from .context import ClinicalContext


class ClinicalHistorySession:
    """Resumable MEDDxAgent history taking for a real patient response loop.

    The history-taking model still generates the doctor questions using the same
    prompts as the research implementation. The application supplies each patient
    answer explicitly instead of invoking the simulated PatientAgent.
    """

    def __init__(
        self,
        patient: Patient,
        history_taking_agent: LLMHistoryTaking,
        context: ClinicalContext,
        profile_model: Model,
        conversation_goals: str = "",
    ):
        if patient is None:
            raise ValueError("Clinical history requires a patient")
        if history_taking_agent is None:
            raise ValueError("Clinical history requires a history-taking agent")
        if profile_model is None:
            raise ValueError("Clinical history requires a model to build the patient profile")

        self.patient = patient
        self.history_taking_agent = history_taking_agent
        self.context = context
        self.profile_model = profile_model
        self.conversation_goals = conversation_goals
        self.complete = False

    def reset(self) -> None:
        self.history_taking_agent.dialogue_history.reset()
        self.patient.patient_profile = None
        self.complete = False

    @property
    def dialogue_history(self) -> str:
        return self.history_taking_agent.dialogue_history.format_dialogue_history()

    @property
    def turns(self) -> list[dict[str, str]]:
        turns: list[dict[str, str]] = []
        for role, content in self.history_taking_agent.dialogue_history.dialogue_history:
            if role == "doctor":
                turns.append({"question": content, "answer": ""})
            elif role == "patient" and turns:
                turns[-1]["answer"] = content
        return turns

    @property
    def pending_question(self) -> str | None:
        dialogue = self.history_taking_agent.dialogue_history.dialogue_history
        if not dialogue:
            return None
        role, content = dialogue[-1]
        return content if role == "doctor" else None

    @property
    def answered_questions(self) -> int:
        return sum(
            1
            for role, _ in self.history_taking_agent.dialogue_history.dialogue_history
            if role == "patient"
        )

    def next_question(self) -> str | None:
        """Generate the next MEDDxAgent question, or return None when history is done."""
        if self.complete:
            return None
        if self.pending_question is not None:
            raise RuntimeError("Submit the pending patient response before requesting another question")

        question = self.history_taking_agent.generate_question(
            patient_initial_info=self.patient.patient_initial_info,
            specialist_preface=self.context.SPECIALIST_PREFACE,
            conversation_goals=self.conversation_goals,
        )
        if question is None:
            self.complete = True
        return question

    def submit_answer(self, answer: str) -> None:
        """Attach a real patient answer to the pending MEDDxAgent question."""
        if self.complete:
            raise RuntimeError("History taking is already complete")
        self.history_taking_agent.record_patient_answer(answer)

    def finish(self) -> str:
        """Build the patient profile consumed by retrieval/diagnosis and return it.

        The research profile summarizer is optimized for antecedents/symptoms. In
        clinical mode the initial payload can also contain examination and test
        results, so it is retained verbatim alongside the history-derived profile.
        """
        if self.pending_question is not None:
            raise RuntimeError("Cannot finish history while a question is waiting for a patient response")

        derived_profile = ""
        if self.dialogue_history:
            derived_profile = dialogue_to_patient_profile(
                dialogue_history_text=self.dialogue_history,
                patient=self.patient,
                model=self.profile_model,
            )
            if not isinstance(derived_profile, str):
                raise TypeError("MEDDxAgent history-derived patient profile must be a string")

        profile_parts = [
            "Clinical information available before history:\n"
            + self.patient.patient_initial_info.strip()
        ]
        if derived_profile.strip():
            profile_parts.append("History-derived patient profile:\n" + derived_profile.strip())

        profile = "\n\n".join(profile_parts)
        self.patient.patient_profile = profile
        self.complete = True
        return profile
