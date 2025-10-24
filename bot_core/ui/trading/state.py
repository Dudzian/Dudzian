"""Definicje stanu aplikacji Trading GUI."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Generic, Optional, TypeVar

from bot_core.runtime.paths import DesktopAppPaths
from bot_core.runtime.metadata import RuntimeEntrypointMetadata, RiskManagerSettings
from bot_core.security.capabilities import LicenseCapabilities
from bot_core.security.guards import CapabilityGuard


T = TypeVar("T")


class UiVar(Generic[T]):
    """Minimalny kontener wartości imitujący interfejs ``tk.Variable``."""

    __slots__ = ("_value",)

    def __init__(self, value: T | None = None) -> None:
        self._value: T | None = value

    def get(self) -> T | None:
        return self._value

    def set(self, value: T | None) -> None:
        self._value = value

    def __repr__(self) -> str:  # pragma: no cover - debug helper
        return f"UiVar({self._value!r})"


class UiStringVar(UiVar[str]):
    """Zmienne tekstowe używane przez interfejsy UI niezależnie od toolkitu."""


class UiDoubleVar(UiVar[float]):
    """Zmienne liczb zmiennoprzecinkowych."""


class UiBooleanVar(UiVar[bool]):
    """Zmienne logiczne."""


@dataclass
class AppState:
    """Stan aplikacji utrzymywany w interfejsie tradingowym."""

    paths: DesktopAppPaths
    runtime_metadata: Optional[RuntimeEntrypointMetadata]
    symbol: UiStringVar
    network: UiStringVar
    mode: UiStringVar
    timeframe: UiStringVar
    fraction: UiDoubleVar
    paper_balance: UiStringVar
    account_balance: UiStringVar
    status: UiStringVar
    running: bool = False
    open_positions: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    risk_profile_name: str | None = None
    risk_profile_config: Any | None = None
    risk_manager_config: Dict[str, Any] | None = None
    risk_manager_settings: RiskManagerSettings | None = None
    risk_profile_label: UiStringVar | None = None
    risk_limits_label: UiStringVar | None = None
    risk_notional_label: UiStringVar | None = None
    market_symbol: UiStringVar | None = None
    market_price: UiStringVar | None = None
    license_capabilities: LicenseCapabilities | None = None
    capability_guard: CapabilityGuard | None = None
    license_summary: UiStringVar | None = None
    license_notice: UiStringVar | None = None
    license_path: str | None = None
    market_intel_label: UiStringVar | None = None
    market_intel_summary: str = "Market intel: —"
    market_intel_history_label: UiStringVar | None = None
    market_intel_history: list[str] = field(default_factory=list)
    market_intel_history_display: str = "Brak historii market intel"
    market_intel_auto_save: UiBooleanVar | None = None
    market_intel_history_destination: str | None = None
    market_intel_history_destination_display: str = "Plik historii: domyślny"
    market_intel_history_path_label: UiStringVar | None = None
    notify_error: Callable[[str, str], None] | None = None


__all__ = [
    "AppState",
    "UiVar",
    "UiStringVar",
    "UiDoubleVar",
    "UiBooleanVar",
]
