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
        health_hook=None,
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
        self._health_metric = self._metrics.gauge(
            "bot_cloud_health_status",
            "Aktualny status CloudOrchestrator (1=zdrowy, 0=problem)",
        )
        self._last_error_metric = self._metrics.gauge(
            "bot_cloud_last_error",
            "Ostatni błąd CloudOrchestrator (1=obecny)",
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
        self._health_hook = health_hook
        self._last_reported: tuple[object, object] | None = None
        healthy, last_error = self._compute_overall_health(self._health)
        self._health["_health"] = healthy
        self._health["_lastError"] = last_error
        self._sync_metrics(self._health)
        self._emit_health(self._health)

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

    def run_synthetic_probes(
        self,
        *,
        previous_snapshot: Mapping[str, object] | None = None,
        prometheus_ok: bool | None = None,
    ) -> dict[str, object]:
        """Zwraca wynik sondy DR łączącej bieżące `_health`/`_lastError` z poprzednim stanem.

        Synthetic probe służy do potwierdzania gotowości failover (multi-region
        Alertmanager/Prometheus) oraz do rehydratacji alertów `_lastError` w
        przypadku świeżo podniesionego procesu cloud (np. po przełączeniu
        control-plane). Gdy bieżący snapshot nie zawiera `_lastError`, a poprzedni
        tak, wynik sondy nadal zwróci informację o błędzie i oznaczy ją jako
        `rehydratedFromPrevious`.
        """

        current_snapshot = self.health_snapshot()
        current_error = current_snapshot.get("_lastError")
        previous_error = None
        if previous_snapshot:
            previous_error = previous_snapshot.get("_lastError")

        effective_error = current_error or previous_error
        rehydrated = bool(effective_error and not current_error and previous_error)
        health_ok = bool(current_snapshot.get("_health"))
        prometheus_healthy = prometheus_ok is True
        failover_ready = bool(health_ok and prometheus_healthy and not effective_error)

        return {
            "timestamp": current_snapshot.get("updatedAt"),
            "healthOk": health_ok,
            "prometheusOk": prometheus_ok,
            "failoverReady": failover_ready,
            "effectiveLastError": effective_error,
            "rehydratedFromPrevious": rehydrated,
            "snapshot": current_snapshot,
        }

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
            healthy, last_error = self._compute_overall_health(payload)
            payload["_health"] = healthy
            payload["_lastError"] = last_error
            self._health = payload
            self._sync_metrics(payload)
            self._emit_health(payload)

    def _sync_metrics(self, snapshot: Mapping[str, object]) -> None:
        workers = snapshot.get("workers")
        if isinstance(workers, Mapping):
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
        overall_health = 1.0 if snapshot.get("_health") else 0.0
        last_error = snapshot.get("_lastError")
        self._health_metric.set(overall_health)
        self._last_error_metric.set(
            1.0 if last_error else 0.0,
            labels={"error": str(last_error or "")},
        )

    def _compute_overall_health(self, payload: Mapping[str, object]) -> tuple[bool, str | None]:
        status = payload.get("status")
        workers = payload.get("workers")
        last_error: str | None = None
        if isinstance(workers, Mapping):
            for worker_payload in workers.values():
                if not isinstance(worker_payload, Mapping):
                    continue
                worker_error = worker_payload.get("lastError")
                if worker_error:
                    last_error = str(worker_error)
                    break
        healthy = bool(status == "running" and not last_error)
        if not healthy and last_error is None and isinstance(status, str):
            last_error = status
        return healthy, last_error

    def _emit_health(self, snapshot: Mapping[str, object]) -> None:
        current = (snapshot.get("_health"), snapshot.get("_lastError"))
        if self._last_reported != current:
            LOGGER.info("Cloud orchestrator health update", extra={"_health": current[0], "_lastError": current[1]})
            self._last_reported = current
        if callable(self._health_hook):
            try:
                self._health_hook(snapshot)
            except Exception:  # pragma: no cover - defensywne logowanie hooków
                LOGGER.debug("Health hook zgłosił wyjątek", exc_info=True)


__all__ = ["CloudOrchestrator"]
