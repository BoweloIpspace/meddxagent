from copy import deepcopy

from ddxdriver.ddxdrivers import init_ddxdriver
from ddxdriver.diagnosis_agents import init_diagnosis_agent
from ddxdriver.history_taking_agents import init_history_taking_agent
from ddxdriver.models import init_model
from ddxdriver.rag_agents import init_rag_agent
from ddxdriver.utils import Agents, Patient

from .context import ClinicalContext
from .history import ClinicalHistorySession
from .results import ClinicalResult, collect_clinical_result


CLINICAL_SESSION_STATE_VERSION = 1


class ClinicalSession:
    """Small application adapter around the existing MEDDxAgent components.

    History taking is human-in-the-loop. After history is finalized, the same
    DDxDriver/RAG/diagnosis implementations are initialized with a benchmark-free
    clinical context and run on the resulting patient profile.
    """

    def __init__(
        self,
        patient_initial_info: str,
        ddxdriver_cfg: dict,
        diagnosis_agent_cfg: dict,
        history_taking_agent_cfg: dict | None = None,
        rag_agent_cfg: dict | None = None,
        context: ClinicalContext | None = None,
        patient_id=0,
    ):
        if not isinstance(patient_initial_info, str) or not patient_initial_info.strip():
            raise ValueError("Clinical session requires non-empty patient initial information")
        if not ddxdriver_cfg or not diagnosis_agent_cfg:
            raise ValueError("Clinical session requires DDxDriver and diagnosis agent configuration")

        self.patient = Patient(
            patient_id=patient_id,
            patient_initial_info=patient_initial_info.strip(),
            patient_profile="",
        )
        self.context = context or ClinicalContext()
        self.ddxdriver_cfg = deepcopy(ddxdriver_cfg)
        self.diagnosis_agent_cfg = deepcopy(diagnosis_agent_cfg)
        self.history_taking_agent_cfg = deepcopy(history_taking_agent_cfg)
        self.rag_agent_cfg = deepcopy(rag_agent_cfg)
        self.driver = None
        self.result: ClinicalResult | None = None

        self.history: ClinicalHistorySession | None = None
        if self.history_taking_agent_cfg:
            history_agent = init_history_taking_agent(
                self.history_taking_agent_cfg["class_name"],
                history_taking_agent_cfg=self.history_taking_agent_cfg["config"],
            )
            driver_model_cfg = self.ddxdriver_cfg["config"].get("model")
            if not driver_model_cfg:
                raise ValueError(
                    "Clinical history requires the DDxDriver model configuration used to build patient profiles"
                )
            profile_model = init_model(
                driver_model_cfg["class_name"],
                **driver_model_cfg["config"],
            )
            self.history = ClinicalHistorySession(
                patient=self.patient,
                history_taking_agent=history_agent,
                context=self.context,
                profile_model=profile_model,
            )
            self.history.reset()
        else:
            self.patient.patient_profile = self.patient.patient_initial_info

    @property
    def history_complete(self) -> bool:
        return self.history is None or self.history.complete

    def update_patient_initial_info(self, patient_initial_info: str) -> None:
        """Refresh the clinical payload as later exam/test findings become available."""
        if not isinstance(patient_initial_info, str) or not patient_initial_info.strip():
            raise ValueError("patient_initial_info must be a non-empty string")

        self.patient.patient_initial_info = patient_initial_info.strip()
        self.driver = None
        self.result = None

        if self.history is None:
            self.patient.patient_profile = self.patient.patient_initial_info
        elif self.patient.patient_profile:
            self.history.refresh_initial_context()

    def next_question(self) -> str | None:
        if self.history is None:
            return None
        return self.history.next_question()

    def submit_answer(self, answer: str) -> None:
        if self.history is None:
            raise RuntimeError("This clinical session was created without a history-taking agent")
        self.history.submit_answer(answer)

    def finish_history(self) -> str:
        if self.history is None:
            return self.patient.patient_profile
        return self.history.finish()

    def _clinical_driver_cfg(self) -> dict:
        cfg = deepcopy(self.ddxdriver_cfg)
        driver_config = cfg["config"]

        available_agents = [Agents.DIAGNOSIS.value]
        if self.rag_agent_cfg:
            available_agents.insert(0, Agents.RAG.value)
        driver_config["available_agents"] = available_agents

        if isinstance(driver_config.get("agent_order"), list):
            driver_config["agent_order"] = [
                agent
                for agent in driver_config["agent_order"]
                if agent in available_agents
            ]
            if not driver_config["agent_order"]:
                driver_config["agent_order"] = available_agents

        return cfg

    def _init_driver(self):
        diagnosis_agent = init_diagnosis_agent(
            self.diagnosis_agent_cfg["class_name"],
            diagnosis_agent_cfg=self.diagnosis_agent_cfg["config"],
        )
        rag_agent = (
            init_rag_agent(
                self.rag_agent_cfg["class_name"],
                self.rag_agent_cfg["config"],
            )
            if self.rag_agent_cfg
            else None
        )
        driver_cfg = self._clinical_driver_cfg()
        return init_ddxdriver(
            driver_cfg["class_name"],
            ddxdriver_cfg=driver_cfg["config"],
            bench=self.context,
            diagnosis_agent=diagnosis_agent,
            history_taking_agent=None,
            patient_agent=None,
            rag_agent=rag_agent,
        )

    def run(self) -> ClinicalResult:
        if self.history is not None and not self.history.complete:
            raise RuntimeError("Finish the human-in-the-loop history before running retrieval/diagnosis")
        if not isinstance(self.patient.patient_profile, str) or not self.patient.patient_profile:
            raise RuntimeError("Clinical session has no patient profile to diagnose")

        self.driver = self._init_driver()
        self.driver(self.patient)
        self.result = collect_clinical_result(
            self.driver,
            dialogue_history=self.history.dialogue_history if self.history else "",
        )
        return self.result

    def persistence_state(self) -> dict:
        """Return the minimum JSON-serializable state needed to resume this session.

        Runtime model/agent instances are deliberately not serialized. They are rebuilt
        from the clinical configuration and this state is replayed onto the fresh adapter.
        """
        history_state = None
        if self.history is not None:
            history_state = {
                "dialogue": [
                    [role, content]
                    for role, content in self.history.history_taking_agent.dialogue_history.dialogue_history
                ],
                "complete": self.history.complete,
                "derived_profile": self.history.derived_profile,
                "conversation_goals": self.history.conversation_goals,
            }

        return {
            "version": CLINICAL_SESSION_STATE_VERSION,
            "patient": {
                "patient_id": self.patient.patient_id,
                "patient_initial_info": self.patient.patient_initial_info,
                "patient_profile": self.patient.patient_profile or "",
            },
            "history": history_state,
            "result": self.result.to_dict() if self.result else None,
        }

    def restore_persistence_state(self, state: dict) -> None:
        """Restore a state produced by :meth:`persistence_state` onto this adapter."""
        if not isinstance(state, dict):
            raise TypeError("Clinical session persistence state must be a mapping")
        if state.get("version") != CLINICAL_SESSION_STATE_VERSION:
            raise ValueError("Unsupported clinical session persistence state version")

        patient_state = state.get("patient")
        if not isinstance(patient_state, dict):
            raise ValueError("Clinical session persistence state is missing patient data")

        patient_initial_info = patient_state.get("patient_initial_info")
        if not isinstance(patient_initial_info, str) or not patient_initial_info.strip():
            raise ValueError("Persisted clinical session has invalid patient initial information")

        self.patient.patient_id = patient_state.get("patient_id", self.patient.patient_id)
        self.patient.patient_initial_info = patient_initial_info.strip()
        patient_profile = patient_state.get("patient_profile", "")
        if patient_profile is None:
            patient_profile = ""
        if not isinstance(patient_profile, str):
            raise TypeError("Persisted clinical session patient profile must be a string")

        history_state = state.get("history")
        if history_state is not None:
            if self.history is None:
                raise ValueError("Persisted session requires a history-taking configuration")
            if not isinstance(history_state, dict):
                raise TypeError("Persisted clinical history state must be a mapping")

            dialogue = history_state.get("dialogue", [])
            if not isinstance(dialogue, list):
                raise TypeError("Persisted clinical history dialogue must be a list")

            self.history.history_taking_agent.dialogue_history.reset()
            for entry in dialogue:
                if not isinstance(entry, list) or len(entry) != 2:
                    raise ValueError("Persisted clinical history dialogue entry is invalid")
                role, content = entry
                if not isinstance(role, str) or not isinstance(content, str):
                    raise TypeError("Persisted clinical history dialogue values must be strings")
                self.history.history_taking_agent.dialogue_history.add_dialogue((role, content))

            self.history.complete = bool(history_state.get("complete", False))
            derived_profile = history_state.get("derived_profile", "")
            conversation_goals = history_state.get("conversation_goals", "")
            if not isinstance(derived_profile, str) or not isinstance(conversation_goals, str):
                raise TypeError("Persisted clinical history text fields must be strings")
            self.history.derived_profile = derived_profile
            self.history.conversation_goals = conversation_goals
        elif self.history is not None:
            self.history.reset()

        self.patient.patient_profile = patient_profile
        self.driver = None

        result_state = state.get("result")
        if result_state is None:
            self.result = None
        else:
            if not isinstance(result_state, dict):
                raise TypeError("Persisted clinical result must be a mapping")
            self.result = ClinicalResult(
                ranked_differential=deepcopy(result_state.get("ranked_differential", [])),
                rationale=result_state.get("rationale", ""),
                dialogue_history=result_state.get("dialogue_history", ""),
                rag_content=result_state.get("rag_content", ""),
                intermediate_differentials=deepcopy(
                    result_state.get("intermediate_differentials", [])
                ),
            )

    def snapshot(self) -> dict:
        return {
            "patient_initial_info": self.patient.patient_initial_info,
            "patient_profile": self.patient.patient_profile or "",
            "history_complete": self.history_complete,
            "pending_question": self.history.pending_question if self.history else None,
            "history_turns": self.history.turns if self.history else [],
            "dialogue_history": self.history.dialogue_history if self.history else "",
            "result": self.result.to_dict() if self.result else None,
        }
