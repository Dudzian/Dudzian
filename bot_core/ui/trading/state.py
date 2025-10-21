"""Definicje stanu aplikacji Trading GUI."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import tkinter as tk

from bot_core.runtime.paths import DesktopAppPaths
from bot_core.runtime.metadata import RuntimeEntrypointMetadata, RiskManagerSettings
from bot_core.security.capabilities import LicenseCapabilities
from bot_core.security.guards import CapabilityGuard


@dataclass
class AppState:
    """Stan aplikacji utrzymywany w interfejsie tradingowym."""

    paths: DesktopAppPaths
    runtime_metadata: Optional[RuntimeEntrypointMetadata]
    symbol: tk.StringVar
    network: tk.StringVar
    mode: tk.StringVar
    timeframe: tk.StringVar
    fraction: tk.DoubleVar
    paper_balance: tk.StringVar
    account_balance: tk.StringVar
    status: tk.StringVar
    running: bool = False
    open_positions: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    risk_profile_name: str | None = None
    risk_profile_config: Any | None = None
    risk_manager_config: Dict[str, Any] | None = None
    risk_manager_settings: RiskManagerSettings | None = None
    risk_profile_label: tk.StringVar | None = None
    risk_limits_label: tk.StringVar | None = None
    risk_notional_label: tk.StringVar | None = None
    market_symbol: tk.StringVar | None = None
    market_price: tk.StringVar | None = None
    license_capabilities: LicenseCapabilities | None = None
    capability_guard: CapabilityGuard | None = None
    license_summary: tk.StringVar | None = None
    license_notice: tk.StringVar | None = None
    license_path: str | None = None
    market_intel_label: tk.StringVar | None = None
    market_intel_summary: str = "Market intel: —"
    market_intel_history_label: tk.StringVar | None = None
    market_intel_history: list[str] = field(default_factory=list)
    market_intel_history_display: str = "Brak historii market intel"
    market_intel_auto_save: tk.BooleanVar | None = None
    market_intel_history_destination: str | None = None
    market_intel_history_destination_display: str = "Plik historii: domyślny"
    market_intel_history_path_label: tk.StringVar | None = None


__all__ = ["AppState"]
