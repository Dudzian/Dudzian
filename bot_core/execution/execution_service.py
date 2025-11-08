"""Narzędzia do wyboru i budowy usług egzekucji na potrzeby runtime."""
from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Mapping, MutableMapping

from bot_core.config.models import RuntimeExecutionLiveSettings, RuntimeExecutionSettings
from bot_core.execution.base import ExecutionContext, ExecutionService
from bot_core.execution.live_router import LiveExecutionRouter, QoSConfig
from bot_core.exchanges.base import Environment as ExchangeEnvironment, ExchangeAdapter, OrderRequest

_LOGGER = logging.getLogger(__name__)

def resolve_execution_mode(
    settings: RuntimeExecutionSettings | None,
    environment: Any,
) -> str:
    """Określa efektywny tryb egzekucji dla wskazanego środowiska."""

    if settings is None:
        settings = RuntimeExecutionSettings()

    mode = (settings.default_mode or "paper").strip().lower()
    if mode not in {"paper", "live", "auto"}:
        mode = "paper"

    environment_value = getattr(environment, "environment", ExchangeEnvironment.PAPER)
    if isinstance(environment_value, str):
        try:
            env_enum = ExchangeEnvironment(environment_value.lower())
        except ValueError:
            env_enum = ExchangeEnvironment.PAPER
    elif isinstance(environment_value, ExchangeEnvironment):
        env_enum = environment_value
    else:
        env_enum = ExchangeEnvironment.PAPER

    offline_mode = bool(getattr(environment, "offline_mode", False))
    if settings.force_paper_when_offline and offline_mode:
        return "paper"

    if mode == "paper":
        return "paper"
    if mode == "live":
        if settings.live is None or not settings.live.enabled:
            raise ValueError(
                "Konfiguracja runtime.execution.live nie jest aktywna, a wymuszono tryb live."
            )
        return "live"

    # tryb auto
    if env_enum in {ExchangeEnvironment.LIVE, ExchangeEnvironment.TESTNET} and settings.live and settings.live.enabled:
        return "live"
    return "paper"


def _resolve_path(candidate: str | None, *, base: str | None = None) -> str | None:
    if not candidate:
        return None
    path = Path(candidate).expanduser()
    if not path.is_absolute() and base:
        path = Path(base).expanduser() / path
    path.parent.mkdir(parents=True, exist_ok=True)
    return str(path)


def _load_decision_log_key(
    settings: RuntimeExecutionLiveSettings,
    *,
    base: str | None = None,
) -> bytes | None:
    if settings.decision_log_key_value:
        value = settings.decision_log_key_value.strip()
        if value:
            return value.encode("utf-8")
    if settings.decision_log_key_env:
        env_value = os.environ.get(settings.decision_log_key_env)
        if env_value:
            return env_value.strip().encode("utf-8")
    if settings.decision_log_key_path:
        path = Path(settings.decision_log_key_path).expanduser()
        if not path.is_absolute() and base:
            path = Path(base).expanduser() / path
        try:
            content = path.read_bytes().strip()
        except FileNotFoundError:
            return None
        if content:
            return content
    return None


def _collect_adapters(bootstrap_ctx: Any) -> MutableMapping[str, ExchangeAdapter]:
    adapters: MutableMapping[str, ExchangeAdapter] = {}
    primary = getattr(bootstrap_ctx, "adapter", None)
    if isinstance(primary, ExchangeAdapter):
        adapters[str(getattr(primary, "name", "")) or getattr(primary, "exchange", "") or "primary"] = primary

    for attr_name in ("exchange_adapters", "adapters", "adapter_pool"):
        extra = getattr(bootstrap_ctx, attr_name, None)
        if isinstance(extra, Mapping):
            for name, adapter in extra.items():
                if isinstance(adapter, ExchangeAdapter):
                    adapters[str(name)] = adapter
    return adapters


def build_live_execution_service(
    *,
    bootstrap_ctx: Any,
    environment: Any,
    runtime_settings: RuntimeExecutionSettings,
) -> ExecutionService:
    """Tworzy router egzekucji live mapujący zlecenia na adaptery giełdowe."""

    live_cfg = runtime_settings.live
    if live_cfg is None or not live_cfg.enabled:
        raise ValueError("Sekcja runtime.execution.live musi być aktywna, aby uruchomić tryb live")

    adapters = _collect_adapters(bootstrap_ctx)
    environment_exchange = str(getattr(environment, "exchange", "")).strip()
    primary_adapter = getattr(bootstrap_ctx, "adapter", None)
    if environment_exchange and isinstance(primary_adapter, ExchangeAdapter):
        adapters.setdefault(environment_exchange, primary_adapter)

    if not adapters:
        raise RuntimeError("Brak zarejestrowanych adapterów giełdowych do trybu live")

    default_route = tuple(live_cfg.default_route or ())
    if not default_route:
        default_route = (environment_exchange or next(iter(adapters.keys())),)

    missing = [name for name in default_route if name not in adapters]
    if missing:
        raise RuntimeError(
            f"Konfiguracja live odwołuje się do nieznanych adapterów: {', '.join(missing)}"
        )

    overrides: dict[str, tuple[str, ...]] = {}
    for symbol, route in (live_cfg.route_overrides or {}).items():
        normalized = tuple(route)
        if not normalized:
            continue
        unknown = [name for name in normalized if name not in adapters]
        if unknown:
            raise RuntimeError(
                f"Override dla symbolu {symbol} zawiera nieznane giełdy: {', '.join(unknown)}"
            )
        overrides[str(symbol)] = normalized

    data_root = getattr(environment, "data_cache_path", None)
    decision_log_path = _resolve_path(live_cfg.decision_log_path, base=data_root)
    decision_log_key = _load_decision_log_key(live_cfg, base=data_root)

    router_kwargs: dict[str, Any] = {
        "adapters": adapters,
        "default_route": default_route,
        "route_overrides": overrides or None,
        "decision_log_path": decision_log_path,
        "decision_log_hmac_key": decision_log_key,
        "decision_log_key_id": live_cfg.decision_log_key_id,
        "decision_log_rotate_bytes": int(live_cfg.decision_log_rotate_bytes),
        "decision_log_keep": int(live_cfg.decision_log_keep),
    }

    latency = tuple(live_cfg.latency_histogram_buckets or ())
    if latency:
        router_kwargs["latency_buckets"] = latency

    metrics_registry = getattr(bootstrap_ctx, "metrics_registry", None)
    if metrics_registry is not None:
        router_kwargs["metrics_registry"] = metrics_registry

    qos_cfg = getattr(live_cfg, "qos", None)
    if qos_cfg is not None:
        per_exchange = {str(name): int(value) for name, value in qos_cfg.per_exchange_concurrency.items()}

        priority_key = qos_cfg.priority_metadata_key

        def _priority_resolver(request: OrderRequest, context: ExecutionContext) -> int:
            if priority_key:
                metadata_value = None
                if context.metadata:
                    metadata_value = context.metadata.get(priority_key)
                if metadata_value is None and getattr(request, "metadata", None):
                    metadata_value = request.metadata.get(priority_key)  # type: ignore[attr-defined]
                if metadata_value is not None:
                    try:
                        return int(metadata_value)
                    except (TypeError, ValueError):
                        _LOGGER.debug(
                            "Nie udało się sparsować priorytetu kolejki %s=%s",
                            priority_key,
                            metadata_value,
                            exc_info=True,
                        )
            return 0

        router_kwargs["qos"] = QoSConfig(
            max_queue_size=int(qos_cfg.max_queue_size),
            worker_concurrency=int(qos_cfg.worker_concurrency),
            per_exchange_concurrency=per_exchange,
            priority_resolver=_priority_resolver if priority_key else None,
            max_queue_wait_seconds=(
                float(qos_cfg.max_queue_wait_seconds)
                if qos_cfg.max_queue_wait_seconds is not None
                else None
            ),
        )

    io_dispatcher = getattr(bootstrap_ctx, "io_dispatcher", None)
    if io_dispatcher is not None:
        router_kwargs["io_dispatcher"] = io_dispatcher

    return LiveExecutionRouter(**router_kwargs)


__all__ = [
    "resolve_execution_mode",
    "build_live_execution_service",
]
