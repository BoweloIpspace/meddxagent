import logging
from unittest.mock import patch

import pytest

from ddxdriver.clinical.monitoring import MonitoringConfig, _MonitoringDispatcher
from ddxdriver.clinical.observability import build_error_event


def test_monitoring_webhook_requires_https(monkeypatch):
    monkeypatch.setenv("MEDDX_MONITORING_WEBHOOK_URL", "http://example.test/hook")
    with pytest.raises(ValueError, match="HTTPS"):
        MonitoringConfig.from_env()


def test_monitoring_dispatches_sanitized_error_payload_without_blocking_request_path():
    try:
        raise RuntimeError("provider message containing patient information")
    except RuntimeError as error:
        payload = build_error_event(
            "clinical.error",
            error,
            operation="diagnosis.run",
            patient_initial_info="DO NOT SEND THIS",
            answer="DO NOT SEND THIS EITHER",
        )

    assert "patient_initial_info" not in payload
    assert "answer" not in payload
    assert "provider message" not in str(payload)
    assert payload["error_type"] == "RuntimeError"

    config = MonitoringConfig(
        webhook_url="https://monitoring.example.test/hook",
        timeout_seconds=1,
        minimum_level=logging.ERROR,
    )
    dispatcher = _MonitoringDispatcher(config)
    with patch("ddxdriver.clinical.monitoring._post_json") as post_json:
        dispatcher.submit(payload, logging.ERROR)
        dispatcher._queue.join()

    post_json.assert_called_once()
    delivered = post_json.call_args.args[1]
    assert delivered == payload
