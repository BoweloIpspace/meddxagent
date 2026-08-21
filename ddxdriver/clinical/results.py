from copy import deepcopy
from dataclasses import dataclass
from typing import List

from ddxdriver.ddxdrivers.base import DDxDriver


@dataclass
class ClinicalResult:
    ranked_differential: List[str]
    rationale: str
    dialogue_history: str
    rag_content: str
    intermediate_differentials: List[List[str]]

    def to_dict(self) -> dict:
        return {
            "ranked_differential": deepcopy(self.ranked_differential),
            "rationale": self.rationale,
            "dialogue_history": self.dialogue_history,
            "rag_content": self.rag_content,
            "intermediate_differentials": deepcopy(self.intermediate_differentials),
        }


def collect_clinical_result(
    driver: DDxDriver,
    dialogue_history: str = "",
) -> ClinicalResult:
    """Return only outputs MEDDxAgent actually produced for the clinical application."""
    if driver is None:
        raise ValueError("Cannot collect clinical result without a DDxDriver")

    return ClinicalResult(
        ranked_differential=deepcopy(driver.get_final_ddx()),
        rationale=driver.get_final_ddx_rationale(),
        dialogue_history=dialogue_history or driver.get_dialogue_history(),
        rag_content=driver.get_final_rag_content(),
        intermediate_differentials=deepcopy(driver.pred_ddxs),
    )
