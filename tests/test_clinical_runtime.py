from ddxdriver.clinical import ClinicalContext, ClinicalHistorySession, collect_clinical_result
from ddxdriver.history_taking_agents.llm_history_taking import LLMHistoryTaking
from ddxdriver.utils import DialogueHistory, OutputDict, Patient


class SequenceModel:
    def __init__(self, outputs):
        self.outputs = iter(outputs)

    def __call__(self, **kwargs):
        return next(self.outputs)


class FakePatientAgent:
    def __init__(self, patient, answers):
        self.patient = patient
        self.answers = iter(answers)

    def __call__(self, question: str) -> str:
        return next(self.answers)


class FakeInteractiveHistory:
    def __init__(self, questions):
        self.questions = iter(questions)
        self.dialogue_history = DialogueHistory()

    def generate_question(self, **kwargs):
        question = next(self.questions)
        if question is not None:
            self.dialogue_history.add_dialogue(("doctor", question))
        return question

    def record_patient_answer(self, answer: str):
        self.dialogue_history.add_dialogue(("patient", answer))


class FakeResultDriver:
    pred_ddxs = [["A", "B"], ["B", "A"]]

    def get_final_ddx(self):
        return self.pred_ddxs[-1]

    def get_final_ddx_rationale(self):
        return "engine rationale"

    def get_dialogue_history(self):
        return "driver dialogue"

    def get_final_rag_content(self):
        return "retrieved evidence"


def test_original_history_call_still_runs_simulated_loop():
    agent = LLMHistoryTaking.__new__(LLMHistoryTaking)
    agent.max_questions = 3
    agent.dialogue_history = DialogueHistory()
    agent.model = SequenceModel(["When did it start?", "None"])

    patient = Patient(patient_initial_info="Headache")
    patient_agent = FakePatientAgent(patient, ["Yesterday"])
    context = ClinicalContext()

    result = agent(patient_agent=patient_agent, bench=context)

    assert result[OutputDict.DIALOGUE_HISTORY].format_dialogue_history() == (
        "Doctor: When did it start?\n"
        "Patient: Yesterday\n"
        "Doctor: None\n"
    )


def test_clinical_history_waits_for_real_patient_answer_and_builds_profile():
    patient = Patient(patient_initial_info="Headache for one day")
    history = FakeInteractiveHistory(["Any fever?", None])
    profile_model = SequenceModel(["- Headache for one day\n- No fever"])
    session = ClinicalHistorySession(
        patient=patient,
        history_taking_agent=history,
        context=ClinicalContext(),
        profile_model=profile_model,
    )

    assert session.next_question() == "Any fever?"
    assert session.pending_question == "Any fever?"

    session.submit_answer("No")
    assert session.pending_question is None
    assert session.next_question() is None

    profile = session.finish()
    assert profile == "- Headache for one day\n- No fever"
    assert patient.patient_profile == profile
    assert session.dialogue_history == "Doctor: Any fever?\nPatient: No\n"


def test_clinical_context_does_not_return_benchmark_fewshot_cases():
    context = ClinicalContext(diagnosis_options=["Pneumonia"], ddx_length=5)
    assert context.SPECIALIST_PREFACE == "You are a medical doctor."
    assert context.DDX_LENGTH == 5
    assert context.get_fewshot(Patient(), {"type": "dynamic", "num_shots": 5}) == []


def test_clinical_result_contains_only_real_driver_outputs():
    result = collect_clinical_result(FakeResultDriver(), dialogue_history="real dialogue")
    assert result.to_dict() == {
        "ranked_differential": ["B", "A"],
        "rationale": "engine rationale",
        "dialogue_history": "real dialogue",
        "rag_content": "retrieved evidence",
        "intermediate_differentials": [["A", "B"], ["B", "A"]],
    }
