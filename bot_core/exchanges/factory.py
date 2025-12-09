"""Fabryka adapterów giełdowych oparta na zunifikowanym kontrakcie."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

from bot_core.exchanges.base import (
    Environment,
    ExchangeAdapter,
    ExchangeAdapterFactory,
    ExchangeBackend,
    ExchangeCredentials,
)
from bot_core.exchanges.health import HealthCheck
from bot_core.exchanges.rate_limiter import RateLimitRule, normalize_rate_limit_rules


@dataclass(slots=True)
class ExchangeAdapterConfig:
    """Pojedynczy punkt konfiguracji adaptera."""

    name: str
    credentials: ExchangeCredentials
    factory: ExchangeAdapterFactory
    environment: Environment | None = None
    settings: Mapping[str, Any] = field(default_factory=dict)
    rate_limits: Sequence[RateLimitRule | Mapping[str, Any]] | None = None
    health_checks: Sequence[HealthCheck] | None = None
    endpoints: Mapping[str, Any] | None = None

    def merged_settings(self) -> dict[str, Any]:
        merged: dict[str, Any] = dict(self.settings or {})
        if self.rate_limits is not None:
            merged.setdefault(
                "rate_limit_rules",
                normalize_rate_limit_rules(self.rate_limits, default=()),
            )
        if self.health_checks is not None:
            merged.setdefault("health_checks", tuple(self.health_checks))
        if self.endpoints:
            merged.setdefault("endpoints", dict(self.endpoints))
        return merged


def build_exchange_adapter(config: ExchangeAdapterConfig) -> ExchangeBackend:
    """Tworzy adapter skonfigurowany wspólnym obiektem konfiguracji."""

    kwargs: dict[str, Any] = {"settings": config.merged_settings()}
    if config.environment is not None:
        kwargs["environment"] = config.environment

    adapter: ExchangeAdapter = config.factory(config.credentials, **kwargs)
    if hasattr(adapter, "name") and not getattr(adapter, "name"):
        adapter.name = config.name  # type: ignore[attr-defined]
    return adapter


__all__ = [
    "ExchangeAdapterConfig",
    "build_exchange_adapter",
]
