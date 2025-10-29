"""Koordynator runtime spajający PortfolioGovernor z danymi schedulera."""
from __future__ import annotations

import json
import logging
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone

from bot_core.market_intel import MarketIntelSnapshot
from bot_core.observability.slo import SLOStatus
from bot_core.portfolio import PortfolioDecision, PortfolioGovernor
from bot_core.risk import StressOverrideRecommendation

_LOGGER = logging.getLogger(__name__)

AllocationProvider = Callable[[], tuple[float, Mapping[str, float]]]
MarketDataProvider = Callable[[], Mapping[str, MarketIntelSnapshot]]
SLOStatusProvider = Callable[[], Mapping[str, SLOStatus]]
StressOverrideProvider = Callable[[], Sequence[StressOverrideRecommendation]]
MetadataProvider = Callable[[], Mapping[str, object]]
Clock = Callable[[], datetime]


def _default_clock() -> datetime:
    return datetime.now(timezone.utc)


@dataclass(slots=True)
class PortfolioRuntimeCoordinator:
    """Odpowiada za cykliczną ewaluację PortfolioGovernora w runtime."""

    governor: PortfolioGovernor
    allocation_provider: AllocationProvider
    market_data_provider: MarketDataProvider
    stress_override_provider: StressOverrideProvider | None = None
    slo_status_provider: SLOStatusProvider | None = None
    metadata_provider: MetadataProvider | None = None
    clock: Clock = _default_clock
    capital_policy_listener: Callable[[PortfolioDecision], None] | None = None
    tco_report_provider: Callable[[], Mapping[str, object]] | None = None
    tco_report_consumer: Callable[[Mapping[str, object]], None] | None = None

    _last_decision: PortfolioDecision | None = None
    _last_run: datetime | None = None
    _last_tco_signature: str | None = None

    def evaluate(self, *, force: bool = False) -> PortfolioDecision | None:
        """Uruchamia ewaluację governora jeżeli minął czas cool-down."""

        now = self.clock()
        if (
            not force
            and self._last_run is not None
            and (now - self._last_run).total_seconds() < self.cooldown_seconds
        ):
            return self._last_decision

        try:
            portfolio_value, allocations = self.allocation_provider()
        except Exception:  # pragma: no cover - diagnostyka providerów
            _LOGGER.exception("PortfolioGovernor: błąd pobierania alokacji portfela")
            return self._last_decision

        try:
            market_data = self.market_data_provider()
        except Exception:  # pragma: no cover - diagnostyka providerów
            _LOGGER.exception("PortfolioGovernor: błąd pobierania danych Market Intel")
            market_data = {}

        slo_statuses: Mapping[str, SLOStatus] | None = None
        if self.slo_status_provider is not None:
            try:
                payload = self.slo_status_provider()
                slo_statuses = dict(payload)
            except Exception:  # pragma: no cover - diagnostyka providerów
                _LOGGER.exception("PortfolioGovernor: błąd pobierania statusów SLO")

        stress_overrides: Sequence[StressOverrideRecommendation] | None = None
        if self.stress_override_provider is not None:
            try:
                stress_overrides = tuple(self.stress_override_provider())
            except Exception:  # pragma: no cover - diagnostyka providerów
                _LOGGER.exception("PortfolioGovernor: błąd pobierania override'ów Stress Lab")

        metadata: Mapping[str, object] | None = None
        if self.metadata_provider is not None:
            try:
                metadata = dict(self.metadata_provider())
            except Exception:  # pragma: no cover - diagnostyka providerów
                _LOGGER.exception("PortfolioGovernor: błąd pobierania metadanych ewaluacji")

        decision = self.governor.evaluate(
            portfolio_value=float(portfolio_value),
            allocations=allocations,
            market_data=market_data,
            stress_overrides=stress_overrides,
            slo_statuses=slo_statuses,
            timestamp=now,
            log_context=metadata,
        )

        if decision is not None and self.capital_policy_listener is not None:
            try:
                self.capital_policy_listener(decision)
            except Exception:  # pragma: no cover - diagnostyka odbiorcy
                _LOGGER.exception("PortfolioRuntimeCoordinator: błąd aktualizacji polityki kapitału")

        if self.tco_report_provider is not None and self.tco_report_consumer is not None:
            try:
                report_payload = self.tco_report_provider()
            except Exception:  # pragma: no cover - diagnostyka providerów
                _LOGGER.exception("PortfolioRuntimeCoordinator: błąd pobierania raportu TCO")
            else:
                try:
                    signature = json.dumps(report_payload, sort_keys=True, default=str)
                except TypeError:
                    signature = None
                if signature is None or signature != self._last_tco_signature:
                    try:
                        self.tco_report_consumer(report_payload)
                    except Exception:  # pragma: no cover - diagnostyka odbiorcy
                        _LOGGER.exception("PortfolioRuntimeCoordinator: błąd aktualizacji kosztów TCO")
                    else:
                        self._last_tco_signature = signature

        self._last_decision = decision
        self._last_run = now
        return decision

    def set_capital_policy_listener(
        self, listener: Callable[[PortfolioDecision], None] | None
    ) -> None:
        self.capital_policy_listener = listener

    def set_tco_report_hooks(
        self,
        *,
        provider: Callable[[], Mapping[str, object]] | None,
        consumer: Callable[[Mapping[str, object]], None] | None,
    ) -> None:
        self.tco_report_provider = provider
        self.tco_report_consumer = consumer
        self._last_tco_signature = None

    @property
    def last_decision(self) -> PortfolioDecision | None:
        return self._last_decision

    @property
    def last_run(self) -> datetime | None:
        return self._last_run

    @property
    def cooldown_seconds(self) -> float:
        config = getattr(self.governor, "_config", None)
        if config is None:
            return 300.0
        value = getattr(config, "rebalance_cooldown_seconds", 300)
        try:
            return float(value)
        except (TypeError, ValueError):  # pragma: no cover - diagnostyka konfiguracji
            return 300.0


__all__ = ["PortfolioRuntimeCoordinator"]
