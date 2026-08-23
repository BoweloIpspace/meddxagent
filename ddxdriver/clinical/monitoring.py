from __future__ import annotations

import json
import logging
import os
import queue
import threading
import urllib.parse
import urllib.request
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class MonitoringConfig:
    webhook_url: str | None = None
    timeout_seconds: float = 3.0
    minimum_level: int = logging.ERROR

    @classmethod
    def from_env(cls) -> "MonitoringConfig":
        webhook_url = os.getenv("MEDDX_MONITORING_WEBHOOK_URL") or None
        timeout_raw = os.getenv("MEDDX_MONITORING_TIMEOUT_SECONDS", "3")
        try:
            timeout_seconds = float(timeout_raw)
        except ValueError as exc:
            raise ValueError("MEDDX_MONITORING_TIMEOUT_SECONDS must be numeric") from exc
        if timeout_seconds <= 0 or timeout_seconds > 30:
            raise ValueError("MEDDX_MONITORING_TIMEOUT_SECONDS must be between 0 and 30")

        level_name = os.getenv("MEDDX_MONITORING_MIN_LEVEL", "ERROR").upper()
        minimum_level = getattr(logging, level_name, None)
        if not isinstance(minimum_level, int):
            raise ValueError("MEDDX_MONITORING_MIN_LEVEL must be a valid logging level")

        config = cls(
            webhook_url=webhook_url,
            timeout_seconds=timeout_seconds,
            minimum_level=minimum_level,
        )
        config.validate()
        return config

    def validate(self) -> None:
        if not self.webhook_url:
            return
        parsed = urllib.parse.urlparse(self.webhook_url)
        if parsed.scheme != "https" or not parsed.netloc:
            raise ValueError("MEDDX_MONITORING_WEBHOOK_URL must be an absolute HTTPS URL")


class _MonitoringDispatcher:
    """Bounded background dispatcher so monitoring never blocks clinical requests."""

    def __init__(self, config: MonitoringConfig, max_queue_size: int = 256):
        self.config = config
        self._queue: queue.Queue[dict[str, Any]] = queue.Queue(maxsize=max_queue_size)
        self._started = False
        self._start_lock = threading.Lock()

    def submit(self, payload: dict[str, Any], level: int) -> None:
        if not self.config.webhook_url or level < self.config.minimum_level:
            return
        self._ensure_started()
        try:
            self._queue.put_nowait(dict(payload))
        except queue.Full:
            # Primary structured logging remains authoritative. Monitoring is best-effort.
            return

    def _ensure_started(self) -> None:
        if self._started:
            return
        with self._start_lock:
            if self._started:
                return
            thread = threading.Thread(
                target=self._run,
                name="meddx-monitoring-dispatcher",
                daemon=True,
            )
            thread.start()
            self._started = True

    def _run(self) -> None:
        while True:
            payload = self._queue.get()
            try:
                _post_json(
                    self.config.webhook_url,
                    payload,
                    timeout_seconds=self.config.timeout_seconds,
                )
            except Exception:
                # Do not recursively log a monitoring-delivery failure through the same sink.
                pass
            finally:
                self._queue.task_done()


def _post_json(url: str | None, payload: dict[str, Any], *, timeout_seconds: float) -> None:
    if not url:
        return
    body = json.dumps(payload, ensure_ascii=False, separators=(",", ":"), default=str).encode(
        "utf-8"
    )
    request = urllib.request.Request(
        url,
        data=body,
        headers={
            "Content-Type": "application/json",
            "User-Agent": "meddx-clinical-api/1.0",
        },
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
        response.read(1)


_dispatcher: _MonitoringDispatcher | None = None
_dispatcher_lock = threading.Lock()


def configure_monitoring(config: MonitoringConfig | None = None) -> _MonitoringDispatcher:
    global _dispatcher
    if _dispatcher is not None:
        return _dispatcher
    with _dispatcher_lock:
        if _dispatcher is None:
            _dispatcher = _MonitoringDispatcher(config or MonitoringConfig.from_env())
    return _dispatcher


def dispatch_monitoring_event(payload: dict[str, Any], level: int) -> None:
    configure_monitoring().submit(payload, level)


def reset_monitoring_for_tests() -> None:
    global _dispatcher
    with _dispatcher_lock:
        _dispatcher = None
