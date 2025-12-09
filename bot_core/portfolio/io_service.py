"""Warstwa I/O dla decyzji i eksportów portfolio."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Mapping

from bot_core.portfolio.allocation_exporter import (
    PortfolioAllocationDocument,
    export_allocations_for_governor_config,
)
from bot_core.portfolio.decision_log import PortfolioDecisionLog

LOGGER = logging.getLogger(__name__)


class PortfolioIOService:
    """Centralizuje zapisy decision logu i eksport alokacji."""

    def __init__(self, *, decision_log: PortfolioDecisionLog | None = None) -> None:
        self._decision_log = decision_log

    def record_portfolio_decision(
        self, decision: object, *, metadata: Mapping[str, object] | None = None
    ) -> None:
        if self._decision_log is None:
            return
        try:
            self._decision_log.record(decision, metadata=metadata)
        except Exception:  # pragma: no cover
            LOGGER.exception("Nie udało się zapisać wpisu decision logu portfela")

    def export_allocations(
        self,
        config: object,
        output_path: Path,
        *,
        governor_name: str | None = None,
        environment: str | None = None,
    ) -> PortfolioAllocationDocument:
        return export_allocations_for_governor_config(
            config,
            output_path,
            governor_name=governor_name,
            environment=environment,
        )


__all__ = ["PortfolioIOService"]
