"""Bootstrap warstwy risk/guardrails dla runtime pipeline."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Mapping

from core.monitoring import AsyncIOGuardrails

from bot_core.config.models import CoreConfig, RuntimeAppConfig
from bot_core.exchanges.base import ExchangeAdapterFactory
from bot_core.runtime.bootstrap import BootstrapContext, bootstrap_environment
from bot_core.runtime.multi_strategy_scheduler import MultiStrategyScheduler
from bot_core.runtime.scheduler import AsyncIOTaskQueue
from bot_core.security import SecretManager

_LOGGER = logging.getLogger(__name__)


class RiskBootstrapper:
    """Wydziela bootstrap warstwy ryzyka i guardrails bez zmiany logiki risk engine."""

    def bootstrap_context(
        self,
        *,
        environment_name: str,
        config_path: str | Path,
        secret_manager: SecretManager,
        adapter_factories: Mapping[str, ExchangeAdapterFactory] | None = None,
        risk_profile_name: str | None = None,
        core_config: CoreConfig | None = None,
    ) -> BootstrapContext:
        try:
            return bootstrap_environment(
                environment_name,
                config_path=config_path,
                secret_manager=secret_manager,
                adapter_factories=adapter_factories,
                risk_profile_name=risk_profile_name,
                core_config=core_config,
            )
        except Exception:
            _LOGGER.exception(
                "Risk bootstrap failed during bootstrap_environment (environment=%s, risk_profile=%s)",
                environment_name,
                risk_profile_name,
            )
            raise

    def bootstrap_io_guardrails(
        self,
        *,
        runtime_config: RuntimeAppConfig | None,
        bootstrap_ctx: BootstrapContext,
        environment_name: str,
    ) -> tuple[AsyncIOTaskQueue | None, AsyncIOGuardrails | None]:
        io_dispatcher: AsyncIOTaskQueue | None = None
        io_guardrails: AsyncIOGuardrails | None = None

        if runtime_config and runtime_config.io_queue is not None:
            io_config = runtime_config.io_queue
            log_directory = io_config.log_directory or "logs/guardrails"
            ui_alerts_path = getattr(bootstrap_ctx, "metrics_ui_alerts_path", None)
            ui_alerts_path = Path(ui_alerts_path) if ui_alerts_path else None
            io_guardrails = AsyncIOGuardrails(
                environment=environment_name,
                log_directory=Path(log_directory).expanduser(),
                rate_limit_warning_threshold=io_config.rate_limit_warning_seconds,
                timeout_warning_threshold=io_config.timeout_warning_seconds,
                ui_alerts_path=ui_alerts_path,
            )
            io_dispatcher = AsyncIOTaskQueue(
                default_max_concurrency=io_config.max_concurrency,
                default_burst=io_config.burst,
                event_listener=io_guardrails,
            )
            for name, limits in io_config.exchanges.items():
                io_dispatcher.configure_exchange(
                    name,
                    max_concurrency=limits.max_concurrency,
                    burst=limits.burst,
                )
            self._persist_io_dispatcher(bootstrap_ctx, io_dispatcher)
        else:
            self._persist_io_dispatcher(bootstrap_ctx, None)

        return io_dispatcher, io_guardrails

    @staticmethod
    def bind_scheduler_limits(
        scheduler: MultiStrategyScheduler,
        *,
        signal_limits: Mapping[str, Mapping[str, object]] | None,
    ) -> None:
        if signal_limits:
            scheduler.configure_signal_limits(signal_limits)

    @staticmethod
    def _persist_io_dispatcher(
        bootstrap_ctx: BootstrapContext,
        io_dispatcher: AsyncIOTaskQueue | None,
    ) -> None:
        try:
            bootstrap_ctx.io_dispatcher = io_dispatcher
        except Exception:  # pragma: no cover - kontekst może blokować zapisy
            _LOGGER.debug(
                "Nie udało się zarejestrować io_dispatcher w BootstrapContext",
                exc_info=True,
            )
