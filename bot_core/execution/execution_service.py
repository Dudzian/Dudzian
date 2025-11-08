"""Narzędzia do wyboru i budowy usług egzekucji na potrzeby runtime."""
from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Mapping, MutableMapping

from bot_core.config.models import RuntimeExecutionLiveSettings, RuntimeExecutionSettings
from bot_core.execution.base import ExecutionService
from bot_core.execution.live_router import LiveExecutionRouter
from bot_core.exchanges.base import Environment as ExchangeEnvironment, ExchangeAdapter
from bot_core.security.signing import build_transaction_signer_selector


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
    if env_enum is ExchangeEnvironment.LIVE and settings.live and settings.live.enabled:
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


def _normalize_signer_entry(config: Mapping[str, Any], *, base: str | None = None) -> dict[str, Any]:
    normalized: dict[str, Any] = {str(key): value for key, value in config.items()}
    key_path = normalized.get("key_path")
    if isinstance(key_path, (str, os.PathLike)):
        path = Path(key_path).expanduser()
        if base:
            base_path = Path(base).expanduser()
            if not path.is_absolute():
                path = base_path / path
        normalized["key_path"] = str(path)
    return normalized


def _normalize_signer_config(config: Mapping[str, Any], *, base: str | None = None) -> dict[str, Any]:
    result: dict[str, Any] = {}
    default_cfg = config.get("default")
    if isinstance(default_cfg, Mapping):
        result["default"] = _normalize_signer_entry(default_cfg, base=base)
    accounts_cfg = config.get("accounts")
    if isinstance(accounts_cfg, Mapping):
        result["accounts"] = {
            str(account_id): _normalize_signer_entry(account_cfg, base=base)
            for account_id, account_cfg in accounts_cfg.items()
            if isinstance(account_cfg, Mapping)
        }
    for key, value in config.items():
        if key in {"default", "accounts"}:
            continue
        if isinstance(value, Mapping):
            result[str(key)] = _normalize_signer_entry(value, base=base)
        else:
            result[str(key)] = value
    return result


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

    signer_selector = None
    signer_config = getattr(live_cfg, "signers", None)
    if isinstance(signer_config, Mapping):
        normalized_signers = _normalize_signer_config(signer_config, base=data_root)
        signer_selector = build_transaction_signer_selector(normalized_signers)
        if signer_selector is not None:
            router_kwargs["transaction_signers"] = signer_selector
            if _LOGGER.isEnabledFor(logging.DEBUG):
                audit_bundle = signer_selector.describe_audit_bundle()

                signers_info = audit_bundle.get("signers", {})
                for account_id, info in signers_info.items():
                    label = account_id if account_id is not None else "default"
                    _LOGGER.debug("Skonfigurowany podpisujący %s: %s", label, dict(info))

                key_index_summary = audit_bundle.get("key_index", {})
                for key_id, summary in key_index_summary.items():
                    _LOGGER.debug("Indeks key_id %s: %s", key_id, dict(summary))

                hardware_summary = audit_bundle.get("hardware_requirements", {})
                if hardware_summary:
                    _LOGGER.debug(
                        "Podsumowanie wymagań sprzętowych: %s",
                        dict(hardware_summary),
                    )

                issues = tuple(audit_bundle.get("issues", ()))
                if issues:
                    _LOGGER.debug(
                        "Wykryte problemy konfiguracji podpisów: %s",
                        [dict(issue) for issue in issues],
                    )
                else:
                    _LOGGER.debug("Konfiguracja podpisów nie zgłasza problemów audytowych.")

    license_capabilities = getattr(bootstrap_ctx, "license_capabilities", None)
    requires_hw_wallet = bool(
        getattr(license_capabilities, "require_hardware_wallet_for_outgoing", False)
    )
    if requires_hw_wallet:
        router_kwargs["require_hardware_wallet_for_withdrawals"] = True
        if signer_selector is None:
            raise RuntimeError(
                "Licencja wymaga portfela sprzętowego dla wypłat, ale w konfiguracji runtime.execution.live.signers "
                "nie zdefiniowano podpisującego."
            )
        missing_hw: list[str] = []
        for account_id, signer in signer_selector.iter_signers():
            if not getattr(signer, "requires_hardware", False):
                label = account_id if account_id is not None else "default"
                missing_hw.append(str(label))
        if missing_hw:
            raise RuntimeError(
                "Licencja wymaga portfela sprzętowego dla wypłat, jednak podpisy dla kont "
                f"{', '.join(sorted(missing_hw))} nie korzystają z urządzeń."
            )

    return LiveExecutionRouter(**router_kwargs)


__all__ = [
    "resolve_execution_mode",
    "build_live_execution_service",
]
