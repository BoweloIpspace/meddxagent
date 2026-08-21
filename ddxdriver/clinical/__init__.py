"""Application-facing helpers for running MEDDxAgent with real patient input.

The research runner remains unchanged. This package provides the smallest adapter
needed by a clinician-facing application: a benchmark-free context, resumable
history taking, and serialization of genuine DDxDriver outputs.
"""

from .config import create_clinical_session, load_clinical_config
from .context import ClinicalContext
from .history import ClinicalHistorySession
from .results import ClinicalResult, collect_clinical_result
from .session import ClinicalSession

__all__ = [
    "ClinicalContext",
    "ClinicalHistorySession",
    "ClinicalResult",
    "ClinicalSession",
    "collect_clinical_result",
    "create_clinical_session",
    "load_clinical_config",
]
