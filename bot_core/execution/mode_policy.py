"""Polityka wyboru trybu egzekucji (paper/live/auto)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Protocol

from bot_core.config.models import RuntimeExecutionSettings
from bot_core.exchanges.base import Environment as ExchangeEnvironment


@dataclass(frozen=True, slots=True)
class ExecutionModeContext:
    """Znormalizowany kontekst środowiska dla decyzji o trybie."""

    exchange_environment: ExchangeEnvironment
    offline_mode: bool


class ExecutionModePolicy(Protocol):
    """Kontrakt strategii wybierającej tryb egzekucji."""

    def decide(
        self,
        requested_mode: str,
        ctx: ExecutionModeContext,
        settings: RuntimeExecutionSettings,
    ) -> str | None:
        """Zwraca wybrany tryb lub ``None`` gdy polityka nie ma zastosowania."""


def _to_exchange_environment(value: Any) -> ExchangeEnvironment:
    if isinstance(value, ExchangeEnvironment):
        return value
    if isinstance(value, str):
        try:
            return ExchangeEnvironment(value.lower())
        except ValueError:
            return ExchangeEnvironment.PAPER
    return ExchangeEnvironment.PAPER


def build_mode_context(environment: Any) -> ExecutionModeContext:
    """Ekstrahuje potrzebne informacje ze struktury środowiska."""

    exchange_env = _to_exchange_environment(
        getattr(environment, "environment", ExchangeEnvironment.PAPER)
    )
    offline_mode = bool(getattr(environment, "offline_mode", False))
    return ExecutionModeContext(exchange_environment=exchange_env, offline_mode=offline_mode)


class OfflinePaperPolicy:
    """Wymusza tryb paper w trybie offline, jeśli włączono opcję w konfiguracji."""

    def decide(
        self,
        requested_mode: str,
        ctx: ExecutionModeContext,
        settings: RuntimeExecutionSettings,
    ) -> str | None:
        if settings.force_paper_when_offline and ctx.offline_mode:
            return "paper"
        return None


class PaperPolicy:
    """Zwraca tryb paper, gdy został jawnie wybrany lub gdy domyślny jest nieznany."""

    def decide(
        self,
        requested_mode: str,
        ctx: ExecutionModeContext,
        settings: RuntimeExecutionSettings,
    ) -> str | None:  # noqa: D401 - krótka odpowiedź
        return "paper" if requested_mode == "paper" else None


class LivePolicy:
    """Wymusza aktywną konfigurację live dla trybu live."""

    def decide(
        self,
        requested_mode: str,
        ctx: ExecutionModeContext,
        settings: RuntimeExecutionSettings,
    ) -> str | None:
        if requested_mode != "live":
            return None
        if settings.live is None or not settings.live.enabled:
            raise ValueError(
                "Konfiguracja runtime.execution.live nie jest aktywna, a wymuszono tryb live."
            )
        return "live"


class AutoPolicy:
    """Automatycznie wybiera live/paper na podstawie środowiska i konfiguracji."""

    def decide(
        self,
        requested_mode: str,
        ctx: ExecutionModeContext,
        settings: RuntimeExecutionSettings,
    ) -> str | None:
        if requested_mode != "auto":
            return None
        if ctx.exchange_environment in {ExchangeEnvironment.LIVE, ExchangeEnvironment.TESTNET}:
            if settings.live and settings.live.enabled:
                return "live"
        return "paper"


class FallbackPolicy:
    """Zapewnia, że zawsze zostanie zwrócony tryb paper."""

    def decide(
        self,
        requested_mode: str,
        ctx: ExecutionModeContext,
        settings: RuntimeExecutionSettings,
    ) -> str | None:  # noqa: D401 - krótka odpowiedź
        return "paper"


class ExecutionModeSelector:
    """Kompozytor polityk wybierających tryb egzekucji."""

    def __init__(self, policies: Iterable[ExecutionModePolicy] | None = None) -> None:
        self._policies = tuple(policies) if policies is not None else (
            OfflinePaperPolicy(),
            PaperPolicy(),
            LivePolicy(),
            AutoPolicy(),
            FallbackPolicy(),
        )

    @staticmethod
    def _normalize_mode(settings: RuntimeExecutionSettings) -> str:
        requested = (settings.default_mode or "paper").strip().lower()
        return requested if requested in {"paper", "live", "auto"} else "paper"

    def resolve(self, settings: RuntimeExecutionSettings | None, environment: Any) -> str:
        resolved_settings = settings or RuntimeExecutionSettings()
        requested_mode = self._normalize_mode(resolved_settings)
        ctx = build_mode_context(environment)
        for policy in self._policies:
            decision = policy.decide(requested_mode, ctx, resolved_settings)
            if decision:
                return decision
        return "paper"


DEFAULT_MODE_SELECTOR = ExecutionModeSelector()


__all__ = [
    "ExecutionModePolicy",
    "ExecutionModeSelector",
    "ExecutionModeContext",
    "build_mode_context",
    "DEFAULT_MODE_SELECTOR",
]
