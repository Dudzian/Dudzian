# managers/risk_manager_adapter.py
# -*- coding: utf-8 -*-
from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import pandas as pd

from KryptoLowca.risk_management import create_risk_manager  # istniejący moduł


class RiskManager:
    """Adapter zapewniający jednolity kontrakt dla GUI/TradingEngine."""

    def __init__(self, config: Dict[str, Any]):
        self.risk_mgr = create_risk_manager(config)
        self._last_details: Optional[Dict[str, Any]] = None

    def calculate_position_size(
        self,
        symbol: str,
        signal: Any,
        market_data: Any,
        portfolio: Optional[Dict[str, Any]] = None,
        *,
        return_details: bool = False,
    ) -> float | Tuple[float, Dict[str, Any]]:
        """
        Zwraca rekomendowaną frakcję kapitału (0..1).

        Adapter normalizuje sygnał/market_data do formatu oczekiwanego przez
        `risk_management.RiskManagement.calculate_position_size`.
        """

        portfolio_ctx = portfolio or {}

        if isinstance(signal, dict):
            signal_payload = dict(signal)
        else:
            try:
                strength = float(signal)
            except Exception:
                strength = 0.0
            signal_payload = {
                "symbol": symbol,
                "strength": abs(strength),
                "confidence": min(1.0, abs(strength) / 100.0 if strength else 0.5),
                "direction": "LONG" if strength >= 0 else "SHORT",
                "prediction": strength,
            }

        if isinstance(market_data, pd.DataFrame):
            market_df = market_data
        elif isinstance(market_data, dict):
            if "df" in market_data and isinstance(market_data["df"], pd.DataFrame):
                market_df = market_data["df"]
            else:
                price = market_data.get("price")
                market_df = pd.DataFrame({"close": [float(price or 0.0)]})
        else:
            market_df = pd.DataFrame({"close": [0.0]})

        sizing = self.risk_mgr.calculate_position_size(symbol, signal_payload, market_df, portfolio_ctx)

        details: Dict[str, Any] = {}
        recommended = 0.0

        if hasattr(sizing, "recommended_size"):
            recommended = float(getattr(sizing, "recommended_size", 0.0))
            details = {
                "recommended_size": recommended,
                "max_allowed_size": float(getattr(sizing, "max_allowed_size", recommended)),
                "kelly_size": float(getattr(sizing, "kelly_size", recommended)),
                "risk_adjusted_size": float(getattr(sizing, "risk_adjusted_size", recommended)),
                "confidence_level": float(getattr(sizing, "confidence_level", 0.0)),
                "reasoning": getattr(sizing, "reasoning", ""),
            }
        elif isinstance(sizing, dict):
            try:
                recommended = float(sizing.get("recommended_size", sizing.get("size", 0.0)))
            except Exception:
                recommended = 0.0
            details = dict(sizing)
            details.setdefault("recommended_size", recommended)
        else:
            try:
                recommended = float(sizing)
            except Exception:
                recommended = 0.0

        recommended = max(0.0, min(1.0, recommended))
        if not details:
            details = {"recommended_size": recommended}
        else:
            details["recommended_size"] = recommended

        self._last_details = details
        if return_details:
            return recommended, details
        return recommended

    def last_position_details(self) -> Optional[Dict[str, Any]]:
        """Zwróć ostatnie szczegóły kalkulacji wielkości pozycji."""

        return self._last_details
