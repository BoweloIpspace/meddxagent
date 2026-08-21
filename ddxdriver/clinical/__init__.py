"""Application-facing helpers for running MEDDxAgent with real patient input.

The research runner remains unchanged. This package provides the smallest adapter
needed by a clinician-facing application: a benchmark-free context, resumable
history taking, and serialization of genuine DDxDriver outputs.
"""

from .context import ClinicalContext
from .history import ClinicalHistorySession
from .results import ClinicalResult, collect_clinical_result

__all__ = [
    "ClinicalContext",
    "ClinicalHistorySession",
    "ClinicalResult",
    "collect_clinical_result",
]
