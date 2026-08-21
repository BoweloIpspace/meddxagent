from typing import Dict

from ddxdriver.models import init_model
from ddxdriver.patient_agents import PatientAgent
from ddxdriver.benchmarks import Bench
from ddxdriver.utils import OutputDict

from .base import HistoryTaking
from .utils import get_history_taking_system_prompt, get_history_taking_user_prompt
from ddxdriver.logger import log


class LLMHistoryTaking(HistoryTaking):
    def __init__(self, history_taking_agent_cfg):
        super().__init__(history_taking_agent_cfg)
        self.model = init_model(
            self.config["model"]["class_name"], **self.config["model"]["config"]
        )

    def generate_question(
        self,
        patient_initial_info: str,
        specialist_preface: str = "You are a medical doctor.",
        conversation_goals: str = "",
    ) -> str | None:
        """Generate and record one doctor question without fabricating a patient answer.

        This is the application-facing seam for human-in-the-loop history taking. The
        existing ``__call__`` method still performs the original simulated multi-question
        loop used by research experiments.
        """
        answered_questions = sum(
            1 for role, _ in self.dialogue_history.dialogue_history if role == "patient"
        )
        if answered_questions >= self.max_questions:
            return None

        if self.dialogue_history.dialogue_history:
            last_role, _ = self.dialogue_history.dialogue_history[-1]
            if last_role == "doctor":
                raise RuntimeError(
                    "Cannot generate another history question before the current patient response is recorded"
                )

        system_prompt = get_history_taking_system_prompt(
            specialist_preface=specialist_preface,
        )
        user_prompt = get_history_taking_user_prompt(
            patient_initial_info=patient_initial_info,
            dialogue_history_text=self.dialogue_history.format_dialogue_history(),
            conversation_goals=conversation_goals,
        )
        question = self.model(system_prompt=system_prompt, user_prompt=user_prompt)
        log.info("Doctor: " + question + "\n")

        if question == "None":
            return None

        self.dialogue_history.add_dialogue(("doctor", question))
        return question

    def record_patient_answer(self, answer: str) -> None:
        """Record a real patient response for the currently pending doctor question."""
        if not isinstance(answer, str) or not answer.strip():
            raise ValueError("Patient answer must be a non-empty string")
        if not self.dialogue_history.dialogue_history:
            raise RuntimeError("Cannot record a patient answer before a doctor question")

        last_role, _ = self.dialogue_history.dialogue_history[-1]
        if last_role != "doctor":
            raise RuntimeError("There is no pending doctor question for this patient answer")

        self.dialogue_history.add_dialogue(("patient", answer))
        log.info("Patient: " + answer + "\n")

    def __call__(
        self,
        patient_agent: PatientAgent,
        bench: Bench,
        conversation_goals: str = "",
    ) -> Dict:
        specialist_preface = bench.SPECIALIST_PREFACE
        num_questions = 0
        while num_questions < self.max_questions:
            question = self.generate_question(
                patient_initial_info=patient_agent.patient.patient_initial_info,
                specialist_preface=specialist_preface,
                conversation_goals=conversation_goals,
            )
            if question is None:
                # Preserve the original research dialogue behavior, which recorded the
                # model's explicit stop token before ending the simulated conversation.
                self.dialogue_history.add_dialogue(("doctor", "None"))
                break

            answer = patient_agent(question=question)
            self.record_patient_answer(answer)
            num_questions += 1

        return {OutputDict.DIALOGUE_HISTORY: self.dialogue_history}
