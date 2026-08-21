from pathlib import Path

import yaml

from ddxdriver.utils import find_project_root

from .context import ClinicalContext
from .session import ClinicalSession


DEFAULT_CLINICAL_CONFIG = find_project_root() / "configs" / "clinical.yml"


def load_clinical_config(path: str | Path | None = None) -> dict:
    config_path = Path(path) if path else DEFAULT_CLINICAL_CONFIG
    if not config_path.exists():
        raise FileNotFoundError(f"Clinical MEDDxAgent config not found: {config_path}")

    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)

    if not isinstance(config, dict):
        raise ValueError("Clinical MEDDxAgent config must be a mapping")

    for required in ("ddxdriver", "diagnosis"):
        if not config.get(required):
            raise ValueError(f"Clinical MEDDxAgent config is missing '{required}'")

    return config


def create_clinical_session(
    patient_initial_info: str,
    patient_id=0,
    config: dict | None = None,
    config_path: str | Path | None = None,
) -> ClinicalSession:
    cfg = config or load_clinical_config(config_path)
    context_cfg = cfg.get("context", {})
    context = ClinicalContext(
        specialist_preface=context_cfg.get("specialist_preface", "You are a medical doctor."),
        diagnosis_options=context_cfg.get("diagnosis_options", []) or [],
        ddx_length=context_cfg.get("ddx_length"),
    )

    return ClinicalSession(
        patient_initial_info=patient_initial_info,
        patient_id=patient_id,
        context=context,
        ddxdriver_cfg=cfg["ddxdriver"],
        diagnosis_agent_cfg=cfg["diagnosis"],
        history_taking_agent_cfg=cfg.get("history_taking"),
        rag_agent_cfg=cfg.get("rag"),
    )
