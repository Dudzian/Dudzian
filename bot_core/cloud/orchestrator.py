"""Lekkie wątki pomocnicze wspierające tryb cloud."""

from __future__ import annotations

import logging
import threading


LOGGER = logging.getLogger(__name__)


class CloudOrchestrator:
    """Uruchamia okresowe zadania pomocnicze (AI, marketplace, alerty)."""

    def __init__(
        self,
        context,
        *,
        marketplace_refresh_interval: int = 900,
        retrain_poll_interval: int = 60,
    ) -> None:
        self._context = context
        self._marketplace_interval = max(0, int(marketplace_refresh_interval or 0))
        self._retrain_interval = max(0, int(retrain_poll_interval or 0))
        self._stop_event = threading.Event()
        self._threads: list[threading.Thread] = []

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

    def stop(self) -> None:
        self._stop_event.set()
        for thread in list(self._threads):
            thread.join(timeout=2.0)
        self._threads.clear()

    def _poll_scheduler(self, scheduler: object) -> None:
        while not self._stop_event.wait(self._retrain_interval or 60):
            try:
                maybe_run = getattr(scheduler, "maybe_run", None)
                if callable(maybe_run):
                    maybe_run()
            except Exception:  # pragma: no cover - diagnostyka
                LOGGER.debug("Błąd podczas wywołania retraining scheduler", exc_info=True)

    def _refresh_marketplace(self) -> None:
        while not self._stop_event.wait(self._marketplace_interval or 300):
            try:
                self._context.reload_marketplace_presets()
            except Exception:  # pragma: no cover - diagnostyka
                LOGGER.debug("Nie udało się odświeżyć presetów Marketplace", exc_info=True)


__all__ = ["CloudOrchestrator"]
