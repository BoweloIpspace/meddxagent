from dataclasses import dataclass, field
from typing import List

from ddxdriver.utils import Patient


@dataclass
class ClinicalContext:
    """Minimal Bench-compatible context for real clinical runs.

    MEDDxAgent agents currently read a small amount of runtime metadata from the
    benchmark object. The application should not load benchmark patients or ground
    truth, so this adapter exposes only the fields the agents actually need.
    """

    specialist_preface: str = "You are a medical doctor."
    diagnosis_options: List[str] = field(default_factory=list)
    ddx_length: int | None = None

    @property
    def SPECIALIST_PREFACE(self) -> str:
        return self.specialist_preface

    @property
    def DDX_LENGTH(self) -> int | None:
        return self.ddx_length

    def get_fewshot(self, patient: Patient, fewshot_cfg: dict) -> List[Patient]:
        """Clinical mode deliberately supplies no benchmark ground-truth examples."""
        return []
