"""Kompatybilna nakładka na nowy moduł GUI w ``KryptoLowca.ui.trading``."""

from __future__ import annotations

# Zachowujemy minimalne zależności – nowe moduły samodzielnie dbają o konfigurację
# środowiska (ścieżki repo, logging itp.).

from KryptoLowca.ui.trading.app import TradingGUI, main
from KryptoLowca.ui.trading.controller import TradingSessionController
from KryptoLowca.ui.trading.state import AppState
from KryptoLowca.ui.trading.view import TradingView

__all__ = [
    "AppState",
    "TradingGUI",
    "TradingSessionController",
    "TradingView",
    "main",
]
