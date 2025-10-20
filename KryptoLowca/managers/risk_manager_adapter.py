"""Adapter integrujący frontendową warstwę z silnikiem ryzyka ``bot_core``.

Nowa implementacja korzysta z ``ThresholdRiskEngine`` zamiast historycznego
``RiskManagement``. Adapter dba o translację sygnałów strategii oraz kontekstu
portfela do modeli danych oczekiwanych przez silnik (`OrderRequest`,
`AccountSnapshot`) i udostępnia kompatybilny interfejs wykorzystywany przez GUI
oraz kontroler CLI.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Dict, Iterable, Mapping, MutableMapping, Optional, Tuple

import numpy as np
import pandas as pd

from bot_core.alerts import AlertSeverity, emit_alert
from bot_core.config.models import RiskProfileConfig
from bot_core.exchanges.base import AccountSnapshot, OrderRequest
from bot_core.risk.base import RiskProfile, RiskRepository
from bot_core.risk.engine import (
    InMemoryRiskRepository,
    PositionState,
    ThresholdRiskEngine,
)
from bot_core.risk.events import RiskDecisionLog
from bot_core.risk.factory import build_risk_profile_from_config


_LOGGER = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers – translacja konfiguracji i danych wejściowych
# ---------------------------------------------------------------------------


def _as_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _as_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _pct(value: float | int | None, *, default: float = 0.0) -> float:
    if value is None:
        return default
    numeric = _as_float(value, default)
    if numeric > 1:
        numeric /= 100.0
    return max(0.0, numeric)


def _guess_profile_name(config: Mapping[str, Any] | None) -> str:
    if not config:
        return "manual"
    for key in ("risk_profile_name", "profile_name", "name"):
        candidate = config.get(key)
        if isinstance(candidate, str) and candidate.strip():
            return candidate.strip()
    return "manual"


def _resolve_profile_config(config: Mapping[str, Any] | None) -> RiskProfileConfig:
    """Konstruuje ``RiskProfileConfig`` z luźnego słownika ustawień GUI."""

    name = _guess_profile_name(config)
    cfg = config or {}

    return RiskProfileConfig(
        name=name,
        max_daily_loss_pct=_pct(
            cfg.get("max_daily_loss_pct") or cfg.get("max_daily_loss"), default=0.1
        ),
        max_position_pct=_pct(
            cfg.get("max_risk_per_trade") or cfg.get("max_position_pct"), default=0.02
        ),
        target_volatility=_pct(cfg.get("target_volatility"), default=0.0),
        max_leverage=max(1.0, _as_float(cfg.get("max_leverage"), 3.0)),
        stop_loss_atr_multiple=max(
            0.5,
            _as_float(cfg.get("stop_loss_atr_multiple") or cfg.get("stop_loss_atr"), 2.0),
        ),
        max_open_positions=max(1, _as_int(cfg.get("max_positions"), 5)),
        hard_drawdown_pct=_pct(
            cfg.get("max_drawdown_pct")
            or cfg.get("hard_drawdown_pct")
            or cfg.get("emergency_stop_drawdown"),
            default=0.2,
        ),
    )


def _latest_price(market: Any) -> float:
    if isinstance(market, pd.DataFrame) and not market.empty:
        if "close" in market.columns:
            return _as_float(market["close"].iloc[-1], 0.0)
        numeric_cols = market.select_dtypes(include=[np.number]).columns
        if len(numeric_cols):
            return _as_float(market[numeric_cols[0]].iloc[-1], 0.0)
    if isinstance(market, Mapping):
        if "price" in market:
            return _as_float(market.get("price"), 0.0)
        if "close" in market:
            return _as_float(market.get("close"), 0.0)
    try:
        return float(market)
    except Exception:
        return 0.0


def _extract_atr(market: Any) -> float | None:
    if isinstance(market, pd.DataFrame) and not market.empty:
        for column in ("atr", "ATR", "atr14", "ATR14"):
            if column in market.columns:
                value = market[column].iloc[-1]
                candidate = _as_float(value, -1.0)
                if candidate > 0:
                    return candidate
        if {"high", "low", "close"}.issubset(market.columns):
            highs = market["high"].to_numpy()
            lows = market["low"].to_numpy()
            closes = market["close"].to_numpy()
            if len(highs) >= 2:
                tr = np.maximum.reduce(
                    [
                        highs[1:] - lows[1:],
                        np.abs(highs[1:] - closes[:-1]),
                        np.abs(lows[1:] - closes[:-1]),
                    ]
                )
                if np.any(np.isfinite(tr)):
                    return float(np.nanmean(tr))
    if isinstance(market, Mapping):
        for key in ("atr", "ATR", "avg_true_range"):
            if key in market:
                candidate = _as_float(market.get(key), -1.0)
                if candidate > 0:
                    return candidate
    return None


def _normalize_signal(symbol: str, signal: Any) -> Mapping[str, Any]:
    if isinstance(signal, Mapping):
        payload = {str(k): v for k, v in signal.items()}
        payload.setdefault("symbol", symbol)
        if "direction" not in payload:
            strength = _as_float(payload.get("strength"), 0.0)
            payload["direction"] = "LONG" if strength >= 0 else "SHORT"
        return payload

    strength = _as_float(signal, 0.0)
    magnitude = min(1.0, abs(strength))
    return {
        "symbol": symbol,
        "strength": magnitude,
        "confidence": min(1.0, magnitude),
        "direction": "LONG" if strength >= 0 else "SHORT",
        "prediction": strength,
        "target_fraction": magnitude,
    }


def _portfolio_positions(portfolio: Mapping[str, Any] | None) -> Iterable[Mapping[str, Any]]:
    if not portfolio:
        return ()
    positions = portfolio.get("positions") if isinstance(portfolio, Mapping) else None
    if isinstance(positions, Mapping):
        return [dict({"symbol": key, **value}) for key, value in positions.items()]
    if isinstance(positions, Iterable):
        return [dict(pos) for pos in positions]
    return ()


def _portfolio_equity(portfolio: Mapping[str, Any] | None) -> Tuple[float, float, float]:
    if not isinstance(portfolio, Mapping):
        return 0.0, 0.0, 0.0

    candidates = (
        portfolio.get("equity"),
        portfolio.get("capital"),
        portfolio.get("cash"),
        portfolio.get("balance"),
    )
    total_equity = 0.0
    for value in candidates:
        total_equity = _as_float(value, -1.0)
        if total_equity > 0:
            break
    if total_equity <= 0:
        total_equity = 0.0

    available_margin = _as_float(
        portfolio.get("available_margin") or portfolio.get("free_margin"), total_equity
    )
    maintenance_margin = _as_float(
        portfolio.get("maintenance_margin") or portfolio.get("used_margin"), 0.0
    )
    return total_equity, available_margin, maintenance_margin


def _position_notional(position: Mapping[str, Any]) -> float:
    if "notional" in position:
        return _as_float(position.get("notional"), 0.0)
    qty = _as_float(position.get("qty") or position.get("quantity") or position.get("size"), 0.0)
    price = _as_float(position.get("avg_entry") or position.get("entry_price"), 0.0)
    return abs(qty) * price


# ---------------------------------------------------------------------------
# Repozytorium w pamięci rozszerzone o seedowanie stanu na potrzeby GUI
# ---------------------------------------------------------------------------


class AdapterRiskRepository(InMemoryRiskRepository):
    """Repozytorium pamięciowe z możliwością ręcznej inicjalizacji stanu."""

    def seed(self, profile: str, state: Mapping[str, Any]) -> None:
        self._storage[str(profile)] = dict(state)


# ---------------------------------------------------------------------------
# Adapter publiczny wykorzystywany przez GUI / CLI
# ---------------------------------------------------------------------------


class RiskManager:
    """Adapter kompatybilny z historycznym API ``calculate_position_size``."""

    def __init__(
        self,
        config: Mapping[str, Any] | None,
        *,
        db_manager: Any | None = None,
        mode: str = "paper",
        profile_name: str | None = None,
        decision_log: RiskDecisionLog | None = None,
        repository: RiskRepository | None = None,
    ) -> None:
        profile_cfg = _resolve_profile_config(config)
        if profile_name:
            profile_cfg = RiskProfileConfig(
                name=profile_name,
                max_daily_loss_pct=profile_cfg.max_daily_loss_pct,
                max_position_pct=profile_cfg.max_position_pct,
                target_volatility=profile_cfg.target_volatility,
                max_leverage=profile_cfg.max_leverage,
                stop_loss_atr_multiple=profile_cfg.stop_loss_atr_multiple,
                max_open_positions=profile_cfg.max_open_positions,
                hard_drawdown_pct=profile_cfg.hard_drawdown_pct,
            )

        self._profile_config = profile_cfg
        self._profile: RiskProfile = build_risk_profile_from_config(profile_cfg)
        self._mode = mode or "paper"
        self._db_manager = db_manager
        self._decision_log = (
            decision_log if decision_log is not None else RiskDecisionLog(max_entries=200)
        )
        self._repository: RiskRepository = (
            repository if repository is not None else AdapterRiskRepository()
        )
        self._engine = ThresholdRiskEngine(
            repository=self._repository,
            decision_log=self._decision_log,
        )
        self._engine.register_profile(self._profile)
        self._last_details: Mapping[str, Any] | None = None

    # ------------------------------------------------------------------
    def calculate_position_size(
        self,
        symbol: str,
        signal: Any,
        market_data: Any,
        portfolio: Optional[Mapping[str, Any]] = None,
        *,
        return_details: bool = False,
    ) -> float | Tuple[float, Mapping[str, Any]]:
        normalized_symbol = str(symbol or "").upper()
        signal_payload = _normalize_signal(normalized_symbol, signal)
        price = max(0.0, _latest_price(market_data))
        atr_value = _extract_atr(market_data)
        total_equity, available_margin, maintenance_margin = _portfolio_equity(portfolio)
        direction = str(signal_payload.get("direction", "LONG")).upper()
        entering_long = direction != "SHORT"

        if total_equity <= 0 or price <= 0:
            reason = "Brak danych o kapitale lub cenie."
            _LOGGER.warning(
                "RiskManager.calculate_position_size: %s (kapitał=%s, cena=%s).",
                reason,
                total_equity,
                price,
            )
            if self._decision_log is not None:
                self._decision_log.record(
                    profile=self._profile.name,
                    symbol=normalized_symbol,
                    side="buy" if entering_long else "sell",
                    quantity=0.0,
                    price=price if price > 0 else None,
                    notional=None,
                    allowed=False,
                    reason=reason,
                    metadata={"mode": self._mode},
                )
            emit_alert(
                reason,
                severity=AlertSeverity.WARNING,
                source="risk",
                context={
                    "profile": self._profile.name,
                    "symbol": normalized_symbol,
                    "mode": self._mode,
                },
            )
            details = {
                "allowed": False,
                "reason": reason,
                "recommended_size": 0.0,
            }
            self._log_risk_snapshot(normalized_symbol, 0.0, details)
            self._last_details = details
            return (0.0, details) if return_details else 0.0

        # direction już znormalizowany powyżej
        base_fraction = _as_float(
            signal_payload.get("target_fraction")
            or signal_payload.get("recommended_size")
            or signal_payload.get("size"),
            0.0,
        )
        if not base_fraction:
            strength = _as_float(signal_payload.get("strength"), 0.0)
            confidence = _as_float(signal_payload.get("confidence"), 0.0)
            base_fraction = min(1.0, max(abs(strength), confidence))

        max_fraction = self._profile.max_position_exposure()
        fraction = max(0.0, min(max_fraction, base_fraction))

        quantity = fraction * total_equity / price

        stop_multiple = self._profile.stop_loss_atr_multiple()
        stop_price = None
        if atr_value and atr_value > 0 and stop_multiple > 0:
            distance = atr_value * stop_multiple
            stop_price = price - distance if entering_long else price + distance

        order_request = OrderRequest(
            symbol=normalized_symbol,
            side="buy" if entering_long else "sell",
            quantity=max(0.0, quantity),
            order_type="market",
            price=price,
            stop_price=stop_price,
            atr=atr_value,
            metadata=signal_payload,
        )

        account_snapshot = AccountSnapshot(
            balances={normalized_symbol: total_equity},
            total_equity=total_equity,
            available_margin=max(0.0, available_margin),
            maintenance_margin=max(0.0, maintenance_margin),
        )

        self._sync_state(portfolio, account_snapshot)

        result = self._engine.apply_pre_trade_checks(
            order_request,
            account=account_snapshot,
            profile_name=self._profile.name,
        )

        recommended_fraction = fraction
        if not result.allowed and result.adjustments:
            max_quantity = result.adjustments.get("max_quantity") if result.adjustments else None
            if max_quantity is not None and price > 0 and total_equity > 0:
                recommended_fraction = max(0.0, min(1.0, _as_float(max_quantity) * price / total_equity))
            else:
                recommended_fraction = 0.0
        elif not result.allowed:
            recommended_fraction = 0.0

        details: MutableMapping[str, Any] = {
            "allowed": result.allowed,
            "reason": result.reason,
            "recommended_size": recommended_fraction,
            "adjustments": dict(result.adjustments or {}),
            "metadata": dict(result.metadata or {}),
        }

        self._emit_alert(normalized_symbol, result, recommended_fraction)
        self._log_risk_snapshot(normalized_symbol, recommended_fraction, details)
        self._last_details = details

        if self._decision_log is not None:
            try:
                notional = recommended_fraction * total_equity
                quantity = notional / price if price > 0 else 0.0
            except Exception:
                notional = None
                quantity = 0.0

            adjustments_payload = None
            if result.adjustments:
                try:
                    adjustments_payload = {
                        str(key): (
                            float(value)
                            if isinstance(value, (int, float)) and not isinstance(value, bool)
                            else value
                        )
                        for key, value in result.adjustments.items()
                    }
                except Exception:
                    adjustments_payload = None

            metadata: Dict[str, Any] = {
                "mode": self._mode,
                "source": "risk_manager_adapter",
                "recommended_fraction": recommended_fraction,
            }
            if result.metadata:
                try:
                    metadata["engine_metadata"] = dict(result.metadata)
                except Exception:
                    metadata["engine_metadata"] = result.metadata
            metadata["details"] = dict(details)

            try:
                self._decision_log.record(
                    profile=self._profile.name,
                    symbol=normalized_symbol,
                    side="buy" if entering_long else "sell",
                    quantity=max(0.0, float(quantity)),
                    price=price if price > 0 else None,
                    notional=notional,
                    allowed=bool(result.allowed),
                    reason=result.reason,
                    adjustments=adjustments_payload,
                    metadata=metadata,
                )
            except Exception:  # pragma: no cover - defensywne logowanie
                _LOGGER.debug("RiskDecisionLog record failed", exc_info=True)

        if return_details:
            return recommended_fraction, details
        return recommended_fraction

    # ------------------------------------------------------------------
    def last_position_details(self) -> Mapping[str, Any] | None:
        return self._last_details

    # ------------------------------------------------------------------
    def latest_guard_state(self) -> Mapping[str, Any] | None:
        """Zwraca zrzut stanu strażnika ryzyka, jeśli dostępny."""

        try:
            return self._engine.snapshot_state(self._profile.name)
        except Exception:  # pragma: no cover - defensywne logowanie
            _LOGGER.debug(
                "RiskManager.latest_guard_state: nie udało się pobrać stanu profilu %s",
                self._profile.name,
                exc_info=True,
            )
            return None

    # ------------------------------------------------------------------
    def _sync_state(
        self,
        portfolio: Mapping[str, Any] | None,
        snapshot: AccountSnapshot,
    ) -> None:
        state = self._engine._states.get(self._profile.name)  # type: ignore[attr-defined]
        if state is None:
            return

        if state.start_of_day_equity <= 0:
            state.start_of_day_equity = snapshot.total_equity
        state.last_equity = snapshot.total_equity
        if state.peak_equity <= 0:
            state.peak_equity = snapshot.total_equity

        positions: Dict[str, PositionState] = {}
        for raw in _portfolio_positions(portfolio):
            symbol = str(raw.get("symbol") or raw.get("pair") or "").upper()
            if not symbol:
                continue
            side = str(raw.get("side") or raw.get("direction") or "long")
            notional = _position_notional(raw)
            if notional <= 0:
                continue
            positions[symbol] = PositionState(side=side, notional=notional)

        state.positions = positions
        self._engine._persist_state(self._profile.name)  # type: ignore[attr-defined]

    # ------------------------------------------------------------------
    def _emit_alert(
        self,
        symbol: str,
        result: Any,
        recommended_fraction: float,
    ) -> None:
        context: Dict[str, Any] = {
            "profile": self._profile.name,
            "symbol": symbol,
            "mode": self._mode,
            "recommended_fraction": recommended_fraction,
        }
        if result.adjustments:
            context["adjustments"] = dict(result.adjustments)
        if result.metadata:
            context["metadata"] = dict(result.metadata)

        if result.allowed:
            if recommended_fraction >= self._profile.max_position_exposure() * 0.95:
                emit_alert(
                    "Wielkość pozycji zbliża się do maksymalnego limitu profilu.",
                    severity=AlertSeverity.WARNING,
                    source="risk",
                    context=context,
                )
        else:
            message = result.reason or "Zlecenie zablokowane przez silnik ryzyka."
            emit_alert(
                message,
                severity=AlertSeverity.WARNING,
                source="risk",
                context=context,
            )

    # ------------------------------------------------------------------
    def _log_risk_snapshot(
        self,
        symbol: str,
        recommended_fraction: float,
        details: Mapping[str, Any],
    ) -> None:
        if not self._db_manager:
            return

        snapshot: Dict[str, Any] = {
            "symbol": symbol,
            "max_fraction": float(self._profile.max_position_exposure()),
            "recommended_size": float(max(0.0, min(1.0, recommended_fraction))),
            "mode": self._mode,
            "details": {
                "profile": self._profile.name,
                "mode": self._mode,
                **{str(key): value for key, value in details.items()},
            },
        }

        log_method = getattr(self._db_manager, "log_risk_limit", None)
        if callable(log_method):
            try:
                loop = asyncio.get_running_loop()
            except RuntimeError:
                try:
                    if asyncio.iscoroutinefunction(log_method):
                        asyncio.run(log_method(snapshot))
                    else:
                        log_method(snapshot)
                except Exception:  # pragma: no cover - defensywne logowanie
                    _LOGGER.exception("Nie udało się zapisać limitu ryzyka (direct)")
            else:
                try:
                    if asyncio.iscoroutinefunction(log_method):
                        loop.create_task(log_method(snapshot))
                    else:
                        loop.call_soon_threadsafe(log_method, snapshot)
                except Exception:  # pragma: no cover - defensywne logowanie
                    _LOGGER.exception("Nie udało się zapisać limitu ryzyka (async)")
            return

        sync_adapter = getattr(getattr(self._db_manager, "sync", None), "log_risk_limit", None)
        if callable(sync_adapter):
            try:
                sync_adapter(snapshot)
            except Exception:  # pragma: no cover - defensywne logowanie
                _LOGGER.exception("Nie udało się zapisać limitu ryzyka (sync)")


# Zgodność wsteczna – historyczne GUI importowało klasę ``RiskManagerAdapter``.
RiskManagerAdapter = RiskManager

__all__ = ["RiskManager", "RiskManagerAdapter"]

