"""Lekkie wątki pomocnicze wspierające tryb cloud."""

from __future__ import annotations

import json
import logging
import threading
from datetime import datetime, timezone
from typing import Mapping

from bot_core.observability.metrics import MetricsRegistry, get_global_metrics_registry


LOGGER = logging.getLogger(__name__)


class CloudOrchestrator:
    """Uruchamia okresowe zadania pomocnicze (AI, marketplace, alerty)."""

    def __init__(
        self,
        context,
        *,
        marketplace_refresh_interval: int = 900,
        retrain_poll_interval: int = 60,
        metrics_registry: MetricsRegistry | None = None,
    ) -> None:
        self._context = context
        self._marketplace_interval = max(0, int(marketplace_refresh_interval or 0))
        self._retrain_interval = max(0, int(retrain_poll_interval or 0))
        self._stop_event = threading.Event()
        self._threads: list[threading.Thread] = []
        self._metrics = (
            metrics_registry
            or getattr(context, "metrics_registry", None)
            or get_global_metrics_registry()
        )
        self._worker_status_metric = self._metrics.gauge(
            "bot_cloud_worker_status",
            "Status workerów CloudOrchestrator (1=zdrowy, 0=problem)",
        )
        self._worker_last_error_metric = self._metrics.gauge(
            "bot_cloud_worker_last_error",
            "Ostatni błąd workerów CloudOrchestrator (1=obecny)",
        )
        self._health_lock = threading.Lock()
        self._health: dict[str, object] = {
            "status": "stopped",
            "workers": {
                "retrain": {
                    "enabled": self._retrain_interval > 0,
                    "lastRunAt": None,
                    "lastError": None,
                    "interval": self._retrain_interval,
                },
                "marketplace": {
                    "enabled": self._marketplace_interval > 0,
                    "lastRunAt": None,
                    "lastError": None,
                    "interval": self._marketplace_interval,
                },
            },
        }

    def start(self) -> None:
        scheduler = getattr(self._context, "retrain_scheduler", None)
        if scheduler is not None and self._retrain_interval > 0:
            self._threads.append(
                threading.Thread(
                    target=self._poll_scheduler,
                    args=(scheduler,),
                    name="cloud-retrain",
                    daemon=True,
                )
            )

        if (
            self._marketplace_interval > 0
            and getattr(self._context, "marketplace_repository", None) is not None
            and hasattr(self._context, "reload_marketplace_presets")
        ):
            self._threads.append(
                threading.Thread(
                    target=self._refresh_marketplace,
                    name="cloud-marketplace",
                    daemon=True,
                )
            )

        for thread in self._threads:
            thread.start()
        self._set_health(status="running")

    def stop(self) -> None:
        self._stop_event.set()
        for thread in list(self._threads):
            thread.join(timeout=2.0)
        self._threads.clear()
        self._set_health(status="stopped")

    def _poll_scheduler(self, scheduler: object) -> None:
        while not self._stop_event.wait(self._retrain_interval or 60):
            self._run_scheduler_once(scheduler)

    def _refresh_marketplace(self) -> None:
        while not self._stop_event.wait(self._marketplace_interval or 300):
            self._refresh_marketplace_once()

    def _run_scheduler_once(self, scheduler: object) -> None:
        try:
            maybe_run = getattr(scheduler, "maybe_run", None)
            if callable(maybe_run):
                maybe_run()
                self._set_health(
                    workers={
                        "retrain": {
                            "lastRunAt": datetime.now(timezone.utc).isoformat(),
                            "lastError": None,
                        }
                    }
                )
        except Exception:  # pragma: no cover - diagnostyka
            self._set_health(
                workers={
                    "retrain": {
                        "lastError": "scheduler_failure",
                        "lastRunAt": datetime.now(timezone.utc).isoformat(),
                    }
                }
            )
            LOGGER.debug("Błąd podczas wywołania retraining scheduler", exc_info=True)

    def _refresh_marketplace_once(self) -> None:
        try:
            self._context.reload_marketplace_presets()
            self._set_health(
                workers={
                    "marketplace": {
                        "lastRunAt": datetime.now(timezone.utc).isoformat(),
                        "lastError": None,
                    }
                }
            )
        except Exception:  # pragma: no cover - diagnostyka
            self._set_health(
                workers={
                    "marketplace": {
                        "lastError": "refresh_failed",
                        "lastRunAt": datetime.now(timezone.utc).isoformat(),
                    }
                }
            )
            LOGGER.debug("Nie udało się odświeżyć presetów Marketplace", exc_info=True)

    def health_snapshot(self) -> dict[str, object]:
        with self._health_lock:
            return json.loads(json.dumps(self._health))

    def _set_health(self, *, status: str | None = None, workers: Mapping[str, Mapping[str, object]] | None = None) -> None:
        with self._health_lock:
            payload = dict(self._health)
            if status is not None:
                payload["status"] = status
            current_workers = dict(payload.get("workers") or {})
            if workers:
                for name, data in workers.items():
                    entry = dict(current_workers.get(name) or {})
                    entry.update(data)
                    current_workers[name] = entry
            payload["workers"] = current_workers
            payload["updatedAt"] = datetime.now(timezone.utc).isoformat()
            self._health = payload
            self._sync_metrics(payload)

    def _sync_metrics(self, snapshot: Mapping[str, object]) -> None:
        workers = snapshot.get("workers")
        if not isinstance(workers, Mapping):
            return

        for name, payload in workers.items():
            if not isinstance(payload, Mapping):
                continue
            enabled = bool(payload.get("enabled", False))
            last_error = payload.get("lastError")
            status_value = 1.0 if enabled and not last_error else 0.0
            self._worker_status_metric.set(status_value, labels={"worker": str(name)})
            self._worker_last_error_metric.set(
                1.0 if last_error else 0.0,
                labels={"worker": str(name), "error": str(last_error or "")},
            )


__all__ = ["CloudOrchestrator"]
