"""Bootstrapowanie execution layer dla pipeline runtime."""

from __future__ import annotations

import logging
from typing import Mapping

from bot_core.config.models import RuntimeExecutionSettings
from bot_core.execution import build_live_execution_service
from bot_core.execution.base import ExecutionService, PriceResolver
from bot_core.execution.paper import MarketMetadata, PaperTradingExecutionService
from bot_core.runtime.bootstrap import BootstrapContext

_LOGGER = logging.getLogger(__name__)


class ExecutionBootstrapper:
    """Buduje i konfiguruje execution stack bez ingerencji w logikę routera."""

    def build_paper_execution_service(
        self,
        markets: Mapping[str, MarketMetadata],
        paper_settings: Mapping[str, object],
        *,
        price_resolver: PriceResolver | None = None,
    ) -> PaperTradingExecutionService:
        return PaperTradingExecutionService(
            markets,
            initial_balances=paper_settings["initial_balances"],  # type: ignore[arg-type]
            maker_fee=float(paper_settings["maker_fee"]),
            taker_fee=float(paper_settings["taker_fee"]),
            slippage_bps=float(paper_settings["slippage_bps"]),
            ledger_directory=paper_settings["ledger_directory"],
            ledger_filename_pattern=str(paper_settings["ledger_filename_pattern"]),
            ledger_retention_days=paper_settings["ledger_retention_days"],  # type: ignore[arg-type]
            ledger_fsync=bool(paper_settings["ledger_fsync"]),
            price_resolver=price_resolver,
        )

    def bootstrap_execution_service(
        self,
        *,
        bootstrap_ctx: BootstrapContext,
        markets: Mapping[str, MarketMetadata],
        paper_settings: Mapping[str, object],
        runtime_settings: RuntimeExecutionSettings | None,
        execution_mode: str,
        price_resolver: PriceResolver | None = None,
    ) -> ExecutionService:
        """Zwraca usługę execution preferując instancję dostarczoną przez bootstrap."""

        context_service = getattr(bootstrap_ctx, "execution_service", None)

        if execution_mode == "live":
            if isinstance(context_service, ExecutionService) and not isinstance(
                context_service, PaperTradingExecutionService
            ):
                _LOGGER.debug("Execution bootstrap: użyto istniejącego serwisu live z kontekstu")
                return context_service
            settings = runtime_settings or RuntimeExecutionSettings()
            _LOGGER.debug("Execution bootstrap: budowanie LiveExecutionRouter")
            service = build_live_execution_service(
                bootstrap_ctx=bootstrap_ctx,
                environment=bootstrap_ctx.environment,
                runtime_settings=settings,
            )
            self._persist_bootstrap_service(
                bootstrap_ctx,
                service,
                error_message="Nie udało się zapisać LiveExecutionRouter w BootstrapContext",
            )
            return service

        if isinstance(context_service, PaperTradingExecutionService):
            _LOGGER.debug("Execution bootstrap: użyto istniejącego PaperTradingExecutionService")
            return context_service

        _LOGGER.debug("Execution bootstrap: budowanie PaperTradingExecutionService")
        service = self.build_paper_execution_service(
            markets,
            paper_settings,
            price_resolver=price_resolver,
        )
        self._persist_bootstrap_service(
            bootstrap_ctx,
            service,
            error_message="Nie udało się zapisać PaperTradingExecutionService w BootstrapContext",
        )
        return service

    @staticmethod
    def _persist_bootstrap_service(
        bootstrap_ctx: BootstrapContext,
        service: ExecutionService,
        *,
        error_message: str,
    ) -> None:
        try:
            bootstrap_ctx.execution_service = service
        except Exception:  # pragma: no cover - kontekst może być tylko-do-odczytu
            _LOGGER.debug(error_message, exc_info=True)
