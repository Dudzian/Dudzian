"""Helpers for loading risk manager settings from the shared core.yaml config."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Tuple

from bot_core.config import CoreConfig, RiskProfileConfig, load_core_config

PACKAGE_ROOT = Path(__file__).resolve().parent
DEFAULT_CORE_CONFIG_PATH = (PACKAGE_ROOT.parent / "config" / "core.yaml").resolve()


def _pick_environment(config: CoreConfig, environment: str | None) -> Tuple[str | None, RiskProfileConfig | None]:
    """Return the selected environment name and associated risk profile config."""

    env_name: str | None = None
    profile_name: str | None = None

    if environment:
        candidate = environment.strip()
        if candidate:
            env_name = candidate if candidate in config.environments else None
    if env_name is None and config.environments:
        env_name = next(iter(config.environments))

    if env_name is not None:
        env_cfg = config.environments.get(env_name)
        if env_cfg is not None:
            profile_name = env_cfg.risk_profile

    profile_cfg = config.risk_profiles.get(profile_name, None) if profile_name else None
    if profile_cfg is None and config.risk_profiles:
        profile_name, profile_cfg = next(iter(config.risk_profiles.items()))

    return profile_name, profile_cfg


def risk_profile_to_adapter_settings(profile: RiskProfileConfig) -> Mapping[str, Any]:
    """Translate :class:`RiskProfileConfig` into a RiskManager adapter payload."""

    max_risk_per_trade = float(profile.max_position_pct)
    max_daily_loss = float(profile.max_daily_loss_pct)
    hard_drawdown = float(profile.hard_drawdown_pct)

    settings: dict[str, Any] = {
        "max_risk_per_trade": max_risk_per_trade,
        "max_portfolio_risk": hard_drawdown if hard_drawdown > 0 else max_daily_loss,
        "max_daily_loss_pct": max_daily_loss,
        "max_drawdown_pct": hard_drawdown,
        "max_positions": int(profile.max_open_positions),
        "max_leverage": float(profile.max_leverage),
        "target_volatility": float(profile.target_volatility),
        "stop_loss_atr_multiple": float(profile.stop_loss_atr_multiple),
        "risk_profile_name": profile.name,
    }
    return settings


def load_risk_settings_from_core(
    config_path: str | Path | None = None,
    *,
    environment: str | None = None,
) -> tuple[str, Mapping[str, Any], RiskProfileConfig | None, CoreConfig]:
    """Load risk settings from ``core.yaml``.

    Parameters
    ----------
    config_path:
        Optional path to ``core.yaml``. When ``None`` the default repository path is used.
    environment:
        Optional name of the environment whose risk profile should be loaded. When omitted the
        first environment defined in the config is selected.

    Returns
    -------
    tuple
        A tuple ``(profile_name, settings_dict, profile_config, core_config)``. ``settings_dict``
        is a mapping ready to be passed to :class:`~KryptoLowca.managers.risk_manager_adapter.RiskManager`.
    """

    target_path = Path(config_path or DEFAULT_CORE_CONFIG_PATH).expanduser().resolve()
    core_config = load_core_config(target_path)

    profile_name, profile_cfg = _pick_environment(core_config, environment)
    if profile_cfg is None:
        return profile_name or "", {}, None, core_config

    settings = risk_profile_to_adapter_settings(profile_cfg)
    return profile_name or profile_cfg.name, settings, profile_cfg, core_config


__all__ = [
    "DEFAULT_CORE_CONFIG_PATH",
    "load_risk_settings_from_core",
    "risk_profile_to_adapter_settings",
]
