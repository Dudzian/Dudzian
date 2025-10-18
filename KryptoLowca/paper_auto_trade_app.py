"""Helper application for managing paper auto-trading with optional GUI support."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
import json
from pathlib import Path
import shlex
import threading
from typing import Any, Callable, Dict, List, Mapping, Optional

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

    def update_risk_limits(
        self,
        settings: Mapping[str, Any],
        *,
        profile_name: Optional[str] = None,
    ) -> None:
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
    _listeners: List[Callable[[Mapping[str, Any], Optional[str], Any | None], None]] = field(default_factory=list)
    _watch_thread: threading.Thread | None = field(init=False, default=None)
    _watch_stop_event: threading.Event | None = field(init=False, default=None)
    _watch_interval: float = field(init=False, default=2.0)
    _watch_last_mtime: float | None = field(init=False, default=None)
    _last_signature: str | None = field(init=False, default=None)
    last_reload_at: datetime | None = field(init=False, default=None)
    reload_count: int = field(init=False, default=0)

    def __post_init__(self) -> None:
        self.core_config_path = Path(self.core_config_path).expanduser().resolve()

    def add_listener(
        self,
        listener: Callable[[Mapping[str, Any], Optional[str], Any | None], None],
    ) -> None:
        """Register a callback notified whenever risk limits are refreshed."""

        self._listeners.append(listener)

    def _load_risk_settings(
        self,
        *,
        config_path: Optional[Path | str] = None,
        environment: Optional[str] = None,
    ) -> tuple[str, Dict[str, Any], Any | None, Optional[str]]:
        if config_path is not None:
            self.core_config_path = Path(config_path).expanduser().resolve()

        env_hint = environment or self.core_environment
        if self.gui is not None:
            profile_name, settings, profile_cfg = self.gui.reload_risk_manager_settings(
                config_path=self.core_config_path,
                environment=env_hint,
            )

            gui_core_path = getattr(self.gui, "core_config_path", None)
            if gui_core_path is not None:
                self.core_config_path = Path(gui_core_path).expanduser().resolve()

            actual_env = environment or getattr(self.gui, "core_environment", None) or env_hint
            return profile_name, dict(settings), profile_cfg, actual_env

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

    def reload_risk_settings(
        self,
        *,
        config_path: Optional[Path | str] = None,
        environment: Optional[str] = None,
    ) -> tuple[str, Dict[str, Any], Any | None]:
        profile_name, settings, profile_cfg, actual_env = self._load_risk_settings(
            config_path=config_path,
            environment=environment,
        )
        if environment:
            self.core_environment = actual_env or environment
        elif actual_env is not None:
            self.core_environment = actual_env

        signature = self._make_signature(settings, profile_name, actual_env)
        changed = signature != self._last_signature

        if settings and changed:
            self.headless_stub.update_risk_limits(settings, profile_name=profile_name)

        self.risk_profile_name = profile_name
        self.risk_manager_settings = dict(settings)
        self.risk_manager_config = profile_cfg
        self.last_reload_at = datetime.utcnow()
        self.reload_count += 1

        self._watch_last_mtime = self._stat_core_config_mtime()
        self._last_signature = signature

        if settings and changed:
            for listener in list(self._listeners):
                try:
                    listener(dict(settings), profile_name, profile_cfg)
                except Exception as exc:
                    # Listeners should never break reload flow; log via print to remain dependency free.
                    print(
                        f"[PaperAutoTradeApp] listener failed: {exc!r}",
                        flush=True,
                    )
        return profile_name, dict(settings), profile_cfg

    @staticmethod
    def _make_signature(
        settings: Mapping[str, Any],
        profile_name: Optional[str],
        environment: Optional[str],
    ) -> str:
        try:
            payload = json.dumps(settings, sort_keys=True, default=str)
        except TypeError:
            payload = repr(settings)
        return "|".join(
            part or ""
            for part in (
                profile_name,
                environment,
                payload,
            )
        )

    def handle_cli_command(self, command: str) -> bool:
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

        env_override: Optional[str] = None
        config_override: Optional[str] = None

        tokens = iter(parts[1:])
        for token in tokens:
            lowered = token.lower()
            if lowered in {"--env", "-e"}:
                try:
                    env_override = next(tokens)
                except StopIteration:
                    env_override = None
                continue
            if lowered in {"--config", "-c", "--core"}:
                try:
                    config_override = next(tokens)
                except StopIteration:
                    config_override = None
                continue
            if any(lowered.startswith(prefix) for prefix in ("env=", "--env=", "environment=")):
                env_override = token.split("=", 1)[1] or None
                continue
            if any(lowered.startswith(prefix) for prefix in ("config=", "--config=", "core=", "path=")):
                config_override = token.split("=", 1)[1] or None
                continue
            if env_override is None:
                env_override = token

        self.reload_risk_settings(config_path=config_override, environment=env_override)
        return True

    def _stat_core_config_mtime(self) -> float | None:
        try:
            return self.core_config_path.stat().st_mtime
        except FileNotFoundError:
            return None

    def start_auto_reload(self, interval: float = 2.0) -> None:
        """Start background watcher that reloads risk settings on ``core.yaml`` changes."""

        if interval <= 0:
            raise ValueError("interval must be positive")

        self._watch_interval = float(interval)
        if self._watch_thread and self._watch_thread.is_alive():
            self._watch_last_mtime = self._stat_core_config_mtime()
            return

        stop_event = threading.Event()
        self._watch_stop_event = stop_event
        self._watch_last_mtime = self._stat_core_config_mtime()

        def _worker() -> None:
            while not stop_event.wait(self._watch_interval):
                mtime = self._stat_core_config_mtime()
                if mtime is None:
                    if self._watch_last_mtime is not None:
                        # When the file disappears (e.g. during an atomic rewrite) reset the
                        # cached timestamp so that the next reappearance triggers a reload.
                        self._watch_last_mtime = None
                    continue
                last = self._watch_last_mtime
                if last is not None and mtime <= last:
                    continue
                self._watch_last_mtime = mtime
                try:
                    self.reload_risk_settings()
                except Exception as exc:
                    print(
                        f"[PaperAutoTradeApp] auto-reload failed: {exc!r}",
                        flush=True,
                    )

        thread = threading.Thread(target=_worker, name="core-config-watcher", daemon=True)
        self._watch_thread = thread
        thread.start()

    def stop_auto_reload(self, timeout: float | None = 1.0) -> None:
        """Stop the background watcher if it is running."""

        stop_event = self._watch_stop_event
        thread = self._watch_thread
        if not stop_event or not thread:
            return

        stop_event.set()
        thread.join(timeout=timeout)
        self._watch_stop_event = None
        self._watch_thread = None


__all__ = ["HeadlessTradingStub", "PaperAutoTradeApp"]
