# managers/risk_manager_adapter.py
# -*- coding: utf-8 -*-
from __future__ import annotations

import asyncio
import logging
from typing import Any, Dict, Optional, Tuple

import pandas as pd

from KryptoLowca.alerts import AlertSeverity, emit_alert
from KryptoLowca.risk_management import create_risk_manager  # istniejący moduł


logger = logging.getLogger(__name__)
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s"))
    logger.addHandler(handler)
logger.setLevel(logging.INFO)


class RiskManager:
    """Adapter zapewniający jednolity kontrakt dla GUI/TradingEngine."""

    def __init__(self, config: Dict[str, Any], *, db_manager: Optional[Any] = None, mode: str = "paper"):
        self.risk_mgr = create_risk_manager(config)
        self._db_manager = db_manager
        self._mode = mode or str(config.get("mode", "paper"))
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
        self._log_risk_snapshot(symbol, recommended, details)
        self._maybe_emit_alert(symbol, recommended, details)
        if return_details:
            return recommended, details
        return recommended

    def last_position_details(self) -> Optional[Dict[str, Any]]:
        """Zwróć ostatnie szczegóły kalkulacji wielkości pozycji."""

        return self._last_details

    def set_mode(self, mode: str) -> None:
        """Ustaw tryb pracy (paper/live) używany przy logowaniu limitów."""

        if mode:
            self._mode = mode

    # ------------------------------ helpers ---------------------------------
    def _log_risk_snapshot(self, symbol: str, recommended: float, details: Dict[str, Any]) -> None:
        if not self._db_manager:
            return

        snapshot = {
            "symbol": symbol,
            "max_fraction": float(details.get("max_allowed_size", 0.0)),
            "recommended_size": float(recommended),
            "mode": self._mode,
            "details": details,
        }

        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            sync = getattr(getattr(self._db_manager, "sync", None), "log_risk_limit", None)
            if callable(sync):
                try:
                    sync(snapshot)
                except Exception:  # pragma: no cover - logujemy i kontynuujemy
                    logger.exception("Nie udało się zapisać limitu ryzyka (sync)")
            return

        log_method = getattr(self._db_manager, "log_risk_limit", None)
        if callable(log_method):
            try:
                if asyncio.iscoroutinefunction(log_method):
                    loop.create_task(log_method(snapshot))
                else:
                    log_method(snapshot)
            except Exception:  # pragma: no cover
                logger.exception("Nie udało się zapisać limitu ryzyka (async)")
        else:
            sync = getattr(getattr(self._db_manager, "sync", None), "log_risk_limit", None)
            if callable(sync):
                try:
                    loop.run_in_executor(None, sync, snapshot)
                except Exception:  # pragma: no cover
                    logger.exception("Nie udało się zapisać limitu ryzyka (executor)")

    def _maybe_emit_alert(self, symbol: str, recommended: float, details: Dict[str, Any]) -> None:
        context = dict(details)
        context.setdefault("symbol", symbol)
        context.setdefault("mode", self._mode)

        if recommended <= 0.0:
            emit_alert(
                f"RiskManager zablokował ekspozycję na {symbol} (rekomendacja 0%).",
                severity=AlertSeverity.WARNING,
                source="risk",
                context=context,
            )
            return

        max_allowed = float(details.get("max_allowed_size", 1.0))
        if max_allowed > 0 and recommended >= max_allowed * 0.99:
            emit_alert(
                f"Rekomendowana wielkość pozycji dla {symbol} osiąga limit ryzyka.",
                severity=AlertSeverity.WARNING,
                source="risk",
                context=context,
            )
        elif float(details.get("confidence_level", 1.0)) < 0.2:
            emit_alert(
                f"Niska pewność kalkulacji pozycji dla {symbol}.",
                severity=AlertSeverity.INFO,
                source="risk",
                context=context,
            )

        reasoning = str(details.get("reasoning", ""))
        if "error" in reasoning.lower():
            emit_alert(
                f"RiskManager zgłosił problem podczas kalkulacji dla {symbol}.",
                severity=AlertSeverity.ERROR,
                source="risk",
                context=context,
            )
