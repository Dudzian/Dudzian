# managers/risk_manager_adapter.py
# -*- coding: utf-8 -*-
from __future__ import annotations
from typing import Any, Dict
from KryptoLowca.risk_management import create_risk_manager  # istniejący moduł

class RiskManager:
    def __init__(self, config: Dict[str, Any]):
        self.risk_mgr = create_risk_manager(config)

    def calculate_position_size(self, symbol: str, signal: float, market_data, portfolio) -> float:
        # Zostawiamy szczegóły algorytmu w risk_management – tutaj tylko adapter.
        return self.risk_mgr.calculate_position_size(symbol, signal, market_data, portfolio)
