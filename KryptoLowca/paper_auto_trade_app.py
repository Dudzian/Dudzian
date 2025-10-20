"""Warstwa kompatybilności korzystająca z nowoczesnego launchera papierowego."""

from __future__ import annotations

import shlex

from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, Mapping, Optional
from types import MethodType

from bot_core.runtime.metadata import (
    RiskManagerSettings,
    derive_risk_manager_settings,
)

from KryptoLowca.auto_trader.paper import (
    DEFAULT_PAPER_BALANCE,
    DEFAULT_SYMBOL,
    HeadlessTradingStub as _ModernHeadlessTradingStub,
    PaperAutoTradeApp as _ModernPaperAutoTradeApp,
)
from KryptoLowca.risk_settings_loader import (
    DEFAULT_CORE_CONFIG_PATH,
    load_risk_settings_from_core,
)


def _settings_to_legacy_dict(
    settings: RiskManagerSettings | Mapping[str, Any] | None,
) -> Dict[str, Any]:
    """Konwertuje obiekt ustawień na słownik w stylu historycznego API."""

    if isinstance(settings, RiskManagerSettings):
        payload: Dict[str, Any] = dict(asdict(settings))
    elif isinstance(settings, Mapping):
        payload = dict(settings)
    else:
        payload = {}

    if "max_position_pct" not in payload and "max_risk_per_trade" in payload:
        try:
            payload["max_position_pct"] = float(payload["max_risk_per_trade"])
        except Exception:
            payload["max_position_pct"] = payload["max_risk_per_trade"]

    return payload


def _resolve_environment_name(core_cfg: Any, requested: str | None, profile_name: str | None) -> str | None:
    """Odwzorowuje nazwę środowiska zgodnie z logiką legacy loadera."""

    environments = getattr(core_cfg, "environments", {}) or {}
    if requested and requested in environments:
        return requested

    for env_name, env_cfg in environments.items():
        risk_profile = getattr(env_cfg, "risk_profile", None)
        if isinstance(risk_profile, str) and risk_profile == profile_name:
            return env_name

    if requested:
        return requested

    if environments:
        return next(iter(environments))
    return None


def _legacy_reload_risk_settings(
    modern: _ModernPaperAutoTradeApp,
    *,
    config_path: str | Path | None = None,
    environment: str | None = None,
) -> tuple[str | None, RiskManagerSettings, object | None]:
    """Wczytuje ustawienia ryzyka, imitując zachowanie historycznego modułu."""

    resolved_path = Path(
        config_path or modern.core_config_path or DEFAULT_CORE_CONFIG_PATH
    ).expanduser().resolve()
    env_hint = environment or modern.core_environment

    profile_name: str | None = None
    settings_payload: Mapping[str, Any] | None = None
    profile_cfg: object | None = None
    core_cfg: Any | None = None

    gui = getattr(modern, "gui", None)
    gui_loader = getattr(gui, "reload_risk_manager_settings", None)
    if callable(gui_loader):
        try:
            gui_profile, gui_settings, gui_profile_cfg = gui_loader(
                config_path=resolved_path,
                environment=env_hint,
            )
        except Exception:
            gui_profile = None
            gui_settings = None
            gui_profile_cfg = None
        else:
            if gui_profile:
                profile_name = gui_profile
            if gui_settings:
                settings_payload = dict(gui_settings)
            if gui_profile_cfg is not None:
                profile_cfg = gui_profile_cfg
            gui_core_path = getattr(gui, "core_config_path", None)
            if gui_core_path:
                resolved_path = Path(gui_core_path).expanduser().resolve()
            gui_env = getattr(gui, "core_environment", None)
            if gui_env:
                env_hint = gui_env

    if settings_payload is None:
        profile_name, settings_payload, profile_cfg, core_cfg = load_risk_settings_from_core(
            resolved_path,
            environment=env_hint,
        )

    effective_env = _resolve_environment_name(core_cfg, env_hint, profile_name)

    source_profile: object | None = profile_cfg if profile_cfg is not None else settings_payload
    derived_settings = derive_risk_manager_settings(
        source_profile,
        profile_name=profile_name,
        defaults=settings_payload,
    )

    modern.core_config_path = resolved_path
    modern.core_environment = effective_env

    if profile_name:
        modern.risk_profile_name = profile_name
    modern.risk_profile_config = profile_cfg
    modern.risk_manager_settings = derived_settings

    signature = modern._make_signature(
        derived_settings,
        modern.risk_profile_name,
        modern.risk_profile_config,
        environment=modern.core_environment,
    )
    changed = signature != getattr(modern, "_last_signature", None)

    modern.reload_count += 1
    modern.last_reload_at = datetime.utcnow()
    modern._last_signature = signature
    modern._risk_config_mtime = modern._get_risk_config_mtime()
    modern._watch_last_mtime = modern._risk_config_mtime

    if changed:
        modern._sync_headless_stub_settings()
        stub = modern.headless_stub
        if stub is modern.gui and hasattr(stub, "apply_risk_profile"):
            try:
                stub.apply_risk_profile(modern.risk_profile_name, derived_settings)
            except Exception:
                pass
        modern._notify_trader_of_risk_update(derived_settings, profile_cfg)
        modern._notify_listeners(derived_settings, profile_cfg)
        modern._update_bootstrap_context(derived_settings, profile_cfg)
        modern._update_watch_aliases()

    return modern.risk_profile_name, derived_settings, modern.risk_profile_config


def _legacy_reload_risk_profile(
    modern: _ModernPaperAutoTradeApp,
    profile: str | None = None,
) -> RiskManagerSettings:
    if profile:
        modern.risk_profile_name = profile
    _, settings, _ = _legacy_reload_risk_settings(
        modern,
        config_path=modern.core_config_path,
        environment=modern.core_environment,
    )
    return settings


def _legacy_handle_cli_command(modern: _ModernPaperAutoTradeApp, command: str) -> bool:
    raw = command.strip()
    if not raw:
        return False

    try:
        parts = shlex.split(raw)
    except ValueError:
        return False

    if not parts:
        return False

    key = parts[0].lower().replace("_", "-")
    if key not in {"reload-risk", "risk-reload", "reload-risk-config"}:
        return False

    env_override: str | None = None
    config_override: str | None = None
    tokens = iter(parts[1:])
    for token in tokens:
        lowered = token.lower()
        if lowered in {"--env", "-e"}:
            env_override = next(tokens, None)
            continue
        if lowered in {"--config", "-c", "--core"}:
            config_override = next(tokens, None)
            continue
        if lowered.startswith("env="):
            env_override = token.split("=", 1)[1] or env_override
            continue
        if lowered.startswith("config=") or lowered.startswith("core=") or lowered.startswith("path="):
            config_override = token.split("=", 1)[1] or config_override
            continue
        if env_override is None and token:
            env_override = token
            continue
        if config_override is None and token:
            config_override = token

    _legacy_reload_risk_settings(
        modern,
        config_path=config_override,
        environment=env_override,
    )
    return True


class HeadlessTradingStub(_ModernHeadlessTradingStub):
    """Rozszerzony stub zapewniający pola wymagane przez legacy testy."""

    def __init__(
        self,
        *,
        symbol: str = DEFAULT_SYMBOL,
        paper_balance: float = DEFAULT_PAPER_BALANCE,
        risk_profile_name: str | None = None,
        risk_manager_settings: RiskManagerSettings | None = None,
    ) -> None:
        super().__init__(
            symbol=symbol,
            paper_balance=paper_balance,
            risk_profile_name=risk_profile_name,
            risk_manager_settings=risk_manager_settings,
        )
        self.last_risk_settings: Dict[str, Any] | None = None
        self.last_profile_name: str | None = None
        self.last_updated_at: datetime | None = None
        self.update_count: int = 0

    # ---- Kompatybilność z historycznym API ----
    def _store_snapshot(
        self,
        settings: Mapping[str, Any] | RiskManagerSettings | None,
        profile_name: str | None,
    ) -> None:
        payload = _settings_to_legacy_dict(settings)
        self.last_risk_settings = payload
        self.last_profile_name = profile_name
        self.last_updated_at = datetime.utcnow()
        self.update_count += 1

    def apply_risk_profile(
        self,
        name: str | None,
        settings: RiskManagerSettings | None,
    ) -> None:  # type: ignore[override]
        super().apply_risk_profile(name, settings)
        self._store_snapshot(settings, name)

    def update_risk_limits(
        self,
        settings: Mapping[str, Any],
        *,
        profile_name: str | None = None,
    ) -> None:
        derived: RiskManagerSettings | None = None
        try:
            derived = derive_risk_manager_settings(
                settings,
                profile_name=profile_name,
                defaults=self.risk_manager_settings,
            )
        except Exception:
            derived = None

        if derived is not None:
            super().apply_risk_profile(profile_name, derived)
            self._store_snapshot(derived, profile_name)
            return

        self._store_snapshot(settings, profile_name)


class PaperAutoTradeApp:
    """Adapter delegujący logikę do nowoczesnego ``PaperAutoTradeApp``."""

    def __init__(
        self,
        *,
        gui: object | None = None,
        headless_stub: HeadlessTradingStub | None = None,
        core_config_path: str | Path | None = None,
        core_environment: str | None = None,
        symbol: str = DEFAULT_SYMBOL,
        enable_gui: bool = True,
        use_dummy_feed: bool = True,
        paper_balance: float = DEFAULT_PAPER_BALANCE,
        risk_profile: str | None = None,
        risk_profile_name: str | None = None,
        bootstrap_context: object | None = None,
        execution_service: object | None = None,
    ) -> None:
        stub = headless_stub or HeadlessTradingStub(
            symbol=symbol,
            paper_balance=paper_balance,
        )

        kwargs: Dict[str, Any] = {}
        if bootstrap_context is not None:
            kwargs["bootstrap_context"] = bootstrap_context
        if execution_service is not None:
            kwargs["execution_service"] = execution_service

        effective_profile = risk_profile or risk_profile_name

        self._modern = _ModernPaperAutoTradeApp(
            symbol=symbol,
            enable_gui=enable_gui,
            use_dummy_feed=use_dummy_feed,
            paper_balance=paper_balance,
            core_config_path=core_config_path,
            core_environment=core_environment,
            risk_profile=effective_profile,
            gui=gui,
            headless_stub=stub,
            **kwargs,
        )
        self._modern.reload_risk_settings = MethodType(_legacy_reload_risk_settings, self._modern)
        self._modern.reload_risk_profile = MethodType(_legacy_reload_risk_profile, self._modern)
        self._modern.handle_cli_command = MethodType(_legacy_handle_cli_command, self._modern)
        self.gui = self._modern.gui
        self.headless_stub = stub
        self._listener_bridge: Dict[
            Callable[[Mapping[str, Any], Optional[str], object | None], None],
            Callable[[RiskManagerSettings, Optional[str], object | None], None],
        ] = {}

    # ---- Delegacje ----
    def __getattr__(self, item: str) -> Any:
        return getattr(self._modern, item)

    # ---- Legacy API ----
    def add_listener(
        self,
        listener: Callable[[Mapping[str, Any], Optional[str], object | None], None],
    ) -> None:
        if not callable(listener):
            raise TypeError("Oczekiwano wywoływalnego callbacku")

        def _bridge(
            settings: RiskManagerSettings,
            profile_name: Optional[str],
            profile_cfg: object | None,
        ) -> None:
            listener(_settings_to_legacy_dict(settings), profile_name, profile_cfg)

        self._listener_bridge[listener] = _bridge
        self._modern.add_listener(_bridge)

    def reload_risk_settings(
        self,
        *,
        config_path: str | Any | None = None,
        environment: str | None = None,
    ) -> tuple[str | None, Dict[str, Any], object | None]:
        profile, settings, profile_cfg = self._modern.reload_risk_settings(
            config_path=config_path,
            environment=environment,
        )
        return profile, _settings_to_legacy_dict(settings), profile_cfg

    def handle_cli_command(self, command: str) -> bool:
        return self._modern.handle_cli_command(command)

    def start_auto_reload(self, interval: float = 2.0) -> None:
        self._modern.start_auto_reload(interval)

    def stop_auto_reload(self, timeout: float | None = 1.0) -> None:
        self._modern.stop_auto_reload(timeout=timeout)

    def start(self) -> None:
        self._modern.start()

    def stop(self) -> None:
        self._modern.stop()


__all__ = ["HeadlessTradingStub", "PaperAutoTradeApp"]

