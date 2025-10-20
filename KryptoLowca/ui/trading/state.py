"""Definicje stanu aplikacji Trading GUI."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import tkinter as tk

from bot_core.runtime.paths import DesktopAppPaths
from bot_core.runtime.metadata import RuntimeEntrypointMetadata, RiskManagerSettings


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
