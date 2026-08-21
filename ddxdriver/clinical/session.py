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
