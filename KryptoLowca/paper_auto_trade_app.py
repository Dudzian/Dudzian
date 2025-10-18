"""Helper application for managing paper auto-trading with optional GUI support."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

from KryptoLowca.risk_settings_loader import (
    DEFAULT_CORE_CONFIG_PATH,
    load_risk_settings_from_core,
)


@dataclass
class HeadlessTradingStub:
    """A lightweight stub used in tests to observe risk limit updates."""

    last_risk_settings: Dict[str, Any] | None = None
    last_profile_name: Optional[str] = None
    last_updated_at: Optional[datetime] = None
    update_count: int = 0

    def update_risk_limits(self, settings: Mapping[str, Any], *, profile_name: Optional[str] = None) -> None:
        self.last_risk_settings = dict(settings)
        self.last_profile_name = profile_name
        self.last_updated_at = datetime.utcnow()
        self.update_count += 1


@dataclass
class PaperAutoTradeApp:
    """Orchestrates risk settings reloads for the paper auto-trading stack."""

    gui: Any | None = None
    headless_stub: HeadlessTradingStub = field(default_factory=HeadlessTradingStub)
    core_config_path: Path = field(default_factory=lambda: Path(DEFAULT_CORE_CONFIG_PATH))
    core_environment: Optional[str] = None

    risk_profile_name: Optional[str] = None
    risk_manager_settings: Dict[str, Any] = field(default_factory=dict)
    risk_manager_config: Any | None = None

    def _load_risk_settings(
        self,
        *,
        environment: Optional[str] = None,
    ) -> tuple[str, Dict[str, Any], Any | None, Optional[str]]:
        env_hint = environment or self.core_environment
        if self.gui is not None:
            profile_name, settings, profile_cfg = self.gui.reload_risk_manager_settings(
                environment=env_hint
            )
            return profile_name, dict(settings), profile_cfg, env_hint

        profile_name, settings, profile_cfg, core_cfg = load_risk_settings_from_core(
            self.core_config_path,
            environment=env_hint,
        )
        if env_hint and env_hint in core_cfg.environments:
            actual_env = env_hint
        elif core_cfg.environments:
            actual_env = next(iter(core_cfg.environments))
        else:
            actual_env = None
        return profile_name, dict(settings), profile_cfg, actual_env

    def reload_risk_settings(self, *, environment: Optional[str] = None) -> tuple[str, Dict[str, Any], Any | None]:
        profile_name, settings, profile_cfg, actual_env = self._load_risk_settings(environment=environment)
        if environment:
            self.core_environment = actual_env or environment
        elif actual_env is not None:
            self.core_environment = actual_env

        if settings:
            self.headless_stub.update_risk_limits(settings, profile_name=profile_name)

        self.risk_profile_name = profile_name
        self.risk_manager_settings = dict(settings)
        self.risk_manager_config = profile_cfg
        return profile_name, dict(settings), profile_cfg

    def handle_cli_command(self, command: str) -> bool:
        normalized = command.strip().lower().replace("_", "-")
        if normalized in {"reload-risk", "risk-reload", "reload-risk-config"}:
            self.reload_risk_settings()
            return True
        return False


__all__ = ["HeadlessTradingStub", "PaperAutoTradeApp"]
