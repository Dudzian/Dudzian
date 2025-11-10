"""Zaawansowane narzędzia zarządzania ryzykiem niezależne od warstwy archiwalnej."""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from bot_core.alerts import AlertSeverity, emit_alert
from bot_core.observability.pandas_warnings import capture_pandas_warnings

__all__ = [
    "RiskLevel",
    "RiskMetrics",
    "PositionSizing",
    "VolatilityEstimator",
    "CorrelationAnalyzer",
    "RiskManagement",
    "AccountExposure",
    "AccountRiskReport",
    "MultiAccountRiskSnapshot",
    "MultiAccountRiskManager",
    "create_risk_manager",
    "backtest_risk_strategy",
    "calculate_optimal_leverage",
]


_LOGGER = logging.getLogger(__name__)
if not _LOGGER.handlers:
    _handler = logging.StreamHandler()
    _handler.setFormatter(
        logging.Formatter("[%(asctime)s] %(name)s - %(levelname)s - %(message)s")
    )
    _LOGGER.addHandler(_handler)
_LOGGER.setLevel(logging.INFO)


class RiskLevel(Enum):
    """Skala poziomów ryzyka używana w raportach."""

    VERY_LOW = 1
    LOW = 2
    MEDIUM = 3
    HIGH = 4
    VERY_HIGH = 5


@dataclass(slots=True)
class RiskMetrics:
    """Pakiet metryk opisujących stan ryzyka portfela."""

    var_95: float
    expected_shortfall: float
    max_drawdown_risk: float
    correlation_risk: float
    liquidity_risk: float
    overall_risk_score: float
    risk_level: RiskLevel


@dataclass(slots=True)
class PositionSizing:
    """Wyniki kalkulacji wielkości pozycji."""

    recommended_size: float
    max_allowed_size: float
    kelly_size: float
    risk_adjusted_size: float
    confidence_level: float
    reasoning: str


@dataclass(slots=True)
class AccountExposure:
    """Ekspozycja pojedynczego konta używana do agregacji ryzyka."""

    account_id: str
    portfolio: Mapping[str, Mapping[str, Any]]
    equity: float
    params: Mapping[str, Any] | None = None


@dataclass(slots=True)
class AccountRiskReport:
    """Raport ryzyka dla konta."""

    account_id: str
    equity: float
    metrics: RiskMetrics
    dynamic_limit: float


@dataclass(slots=True)
class MultiAccountRiskSnapshot:
    """Zbiorczy widok metryk multi-account."""

    aggregate_metrics: RiskMetrics
    accounts: tuple[AccountRiskReport, ...]
    symbol_limits: Mapping[str, float]


class VolatilityEstimator:
    """Szacunek zmienności przy użyciu EWMA / uproszczonego GARCH."""

    def __init__(self, lookback_periods: int = 252) -> None:
        self.lookback = lookback_periods
        self.logger = logging.getLogger(__name__)

    def ewma_volatility(
        self, returns: pd.Series, lambda_param: float = 0.94
    ) -> float:
        try:
            with capture_pandas_warnings(
                self.logger, component="risk.volatility.ewma"
            ):
                if returns is None or len(returns) < 2:
                    return 0.02
                if len(returns) < 10:
                    return float(returns.std()) if len(returns) > 1 else 0.02

                weights = np.array(
                    [(1 - lambda_param) * (lambda_param**i) for i in range(len(returns))]
                )[::-1]
                weights = weights / weights.sum()
                weighted_variance = np.sum(weights * (returns**2))
                return float(np.sqrt(weighted_variance * 252))
        except Exception as exc:  # pragma: no cover - log i fallback
            self.logger.error("Error calculating EWMA volatility: %s", exc)
            return 0.02

    def garch_volatility(self, returns: pd.Series) -> float:
        try:
            with capture_pandas_warnings(
                self.logger, component="risk.volatility.garch"
            ):
                if returns is None or len(returns) < 50:
                    empty = returns if returns is not None else pd.Series([], dtype=float)
                    return self.ewma_volatility(empty)

                omega, alpha, beta = 1e-6, 0.1, 0.85
                variances = [float(returns.var())]
                limit = min(len(returns), 100)
                for idx in range(1, limit):
                    new_var = (
                        omega
                        + alpha * (returns.iloc[idx - 1] ** 2)
                        + beta * variances[-1]
                    )
                    variances.append(float(new_var))
                return float(np.sqrt(variances[-1] * 252))
        except Exception as exc:  # pragma: no cover - log i fallback
            self.logger.error("Error calculating GARCH volatility: %s", exc)
            empty = returns if returns is not None else pd.Series([], dtype=float)
            return self.ewma_volatility(empty)

    def realized_volatility(self, returns: pd.Series) -> float:
        try:
            with capture_pandas_warnings(
                self.logger, component="risk.volatility.realized"
            ):
                if returns is None or len(returns) == 0:
                    return 0.02
                return float(returns.std() * np.sqrt(252))
        except Exception:  # pragma: no cover - fallback
            return 0.02


class CorrelationAnalyzer:
    """Narzędzia do analizy korelacji między instrumentami."""

    def __init__(self) -> None:
        self.logger = logging.getLogger(__name__)
        self.correlation_cache: Dict[str, float] = {}

    def calculate_dynamic_correlation(
        self, returns1: pd.Series, returns2: pd.Series, window: int = 60
    ) -> float:
        try:
            with capture_pandas_warnings(
                self.logger, component="risk.correlation.dynamic"
            ):
                if returns1 is None or returns2 is None:
                    return 0.0
                if len(returns1) < window or len(returns2) < window:
                    return 0.0
                aligned = pd.concat([returns1, returns2], axis=1).dropna()
                if len(aligned) < window:
                    return 0.0
                rolling_corr = aligned.iloc[:, 0].rolling(window).corr(aligned.iloc[:, 1])
                value = rolling_corr.iloc[-1]
                return float(value) if not pd.isna(value) else 0.0
        except Exception as exc:  # pragma: no cover - log i fallback
            self.logger.error("Error calculating dynamic correlation: %s", exc)
            return 0.0

    def portfolio_correlation_risk(
        self, returns_dict: Mapping[str, pd.Series]
    ) -> float:
        try:
            with capture_pandas_warnings(
                self.logger, component="risk.correlation.portfolio"
            ):
                symbols = list(returns_dict.keys())
                if len(symbols) < 2:
                    return 0.0

                correlations: List[float] = []
                for idx, sym_a in enumerate(symbols):
                    for sym_b in symbols[idx + 1 :]:
                        corr = self.calculate_dynamic_correlation(
                            returns_dict[sym_a], returns_dict[sym_b]
                        )
                        correlations.append(abs(float(corr)))

                avg_corr = float(np.mean(correlations)) if correlations else 0.0
                return min(1.0, max(0.0, avg_corr))
        except Exception as exc:  # pragma: no cover - log i fallback
            self.logger.error("Error calculating portfolio correlation risk: %s", exc)
            return 0.5


class RiskManagement:
    """Zaawansowane zarządzanie ryzykiem (sizing, alerty, raporty)."""

    def __init__(self, params: Optional[Mapping[str, Any]] = None) -> None:
        self.params = dict(params or {})
        self.logger = logging.getLogger(__name__)

        self.max_risk_per_trade = float(self.params.get("max_risk_per_trade", 0.02))
        self.max_portfolio_risk = float(self.params.get("max_portfolio_risk", 0.10))
        self.max_correlation_risk = float(self.params.get("max_correlation_risk", 0.7))
        self.max_sector_concentration = float(
            self.params.get("max_sector_concentration", 0.3)
        )

        self.confidence_level = float(self.params.get("confidence_level", 0.95))
        self.lookback_period = int(self.params.get("lookback_period", 252))
        self.max_positions = int(self.params.get("max_positions", 10))
        self.emergency_stop_drawdown = float(
            self.params.get("emergency_stop_drawdown", 0.15)
        )

        self.min_stop_loss = float(self.params.get("min_stop_loss", 0.005))
        self.max_stop_loss = float(self.params.get("max_stop_loss", 0.05))
        self.use_dynamic_stops = bool(self.params.get("use_dynamic_stops", True))

        self.volatility_estimator = VolatilityEstimator(self.lookback_period)
        self.correlation_analyzer = CorrelationAnalyzer()

        self.current_positions: Dict[str, Dict[str, Any]] = {}
        self.historical_returns: Dict[str, pd.Series] = {}
        self.portfolio_value_history: List[float] = []

        self._validate_parameters()

    def _emit_alert(
        self,
        message: str,
        *,
        severity: AlertSeverity = AlertSeverity.WARNING,
        context: Optional[Mapping[str, Any]] = None,
    ) -> None:
        try:
            emit_alert(message, severity=severity, source="risk", context=dict(context or {}))
        except Exception:  # pragma: no cover - alert nie może zatrzymać silnika
            self.logger.exception("Failed to emit risk alert")

    def _validate_parameters(self) -> None:
        if self.max_portfolio_risk <= self.max_risk_per_trade:
            raise ValueError(
                "max_portfolio_risk must be greater than max_risk_per_trade"
            )
        if not 0.5 <= self.confidence_level <= 0.99:
            raise ValueError("confidence_level must be between 0.5 and 0.99")
        if self.max_positions < 1:
            raise ValueError("max_positions must be at least 1")

    # ------------------------------------------------------------------
    # Position sizing
    # ------------------------------------------------------------------
    def calculate_position_size(
        self,
        symbol: str,
        signal_data: Mapping[str, Any],
        market_data: pd.DataFrame,
        current_portfolio: Mapping[str, Mapping[str, Any]],
    ) -> PositionSizing:
        try:
            with capture_pandas_warnings(
                self.logger, component="risk.position_sizing"
            ):
                signal_strength = float(signal_data.get("strength", 0.5))
                signal_confidence = float(signal_data.get("confidence", 0.5))

                returns = (
                    market_data["close"].pct_change().dropna()
                    if "close" in market_data
                    else pd.Series([], dtype=float)
                )

                kelly_size = self._calculate_kelly_criterion(returns, signal_confidence)
                volatility = self.volatility_estimator.ewma_volatility(returns)
                vol_adjusted_size = self._volatility_adjusted_size(
                    volatility, signal_strength
                )
                risk_parity_size = self._risk_parity_sizing(current_portfolio, volatility)
                portfolio_heat = self._calculate_portfolio_heat(current_portfolio)
                heat_adjusted_size = self._heat_adjusted_sizing(portfolio_heat)

                if portfolio_heat > 0.8:
                    self._emit_alert(
                        "Wysokie 'portfolio heat' "
                        f"({portfolio_heat:.2%}) – ograniczam ekspozycję.",
                        severity=AlertSeverity.WARNING,
                        context={
                            "portfolio_heat": float(portfolio_heat),
                            "symbol": symbol,
                        },
                    )

                correlation_adjustment = self._correlation_adjustment(
                    symbol, current_portfolio
                )

                sizes = [
                    kelly_size,
                    vol_adjusted_size,
                    risk_parity_size,
                    heat_adjusted_size,
                ]
                weights = [0.3, 0.3, 0.2, 0.2]
                combined_size = float(sum(s * w for s, w in zip(sizes, weights)))
                final_size = combined_size * float(correlation_adjustment)

                max_allowed = float(self._calculate_max_position_size(current_portfolio))
                recommended_size = float(max(0.0, min(final_size, max_allowed)))

                if recommended_size == 0.0:
                    self._emit_alert(
                        "Rekomendacja risk engine: brak nowej pozycji "
                        f"na {symbol} (0%).",
                        severity=AlertSeverity.WARNING,
                        context={
                            "symbol": symbol,
                            "portfolio_heat": float(portfolio_heat),
                            "max_allowed": max_allowed,
                            "correlation_adjustment": float(correlation_adjustment),
                        },
                    )

                size_variance = float(np.var(sizes))
                confidence = float(max(0.1, 1.0 - size_variance))
                if confidence < 0.2:
                    self._emit_alert(
                        f"Bardzo niska pewność sizingu dla {symbol} ({confidence:.2f}).",
                        severity=AlertSeverity.INFO,
                        context={
                            "symbol": symbol,
                            "confidence": confidence,
                            "size_variance": size_variance,
                        },
                    )

                reasoning = self._generate_sizing_reasoning(
                    kelly_size,
                    vol_adjusted_size,
                    risk_parity_size,
                    heat_adjusted_size,
                    float(correlation_adjustment),
                    float(portfolio_heat),
                )

                return PositionSizing(
                    recommended_size=recommended_size,
                    max_allowed_size=max_allowed,
                    kelly_size=kelly_size,
                    risk_adjusted_size=final_size,
                    confidence_level=confidence,
                    reasoning=reasoning,
                )
        except Exception as exc:  # pragma: no cover - log i fallback
            self.logger.error("Error calculating position size: %s", exc)
            self._emit_alert(
                f"Błąd kalkulacji wielkości pozycji dla {symbol}: {exc}",
                severity=AlertSeverity.ERROR,
                context={"symbol": symbol},
            )
            return PositionSizing(
                recommended_size=0.01,
                max_allowed_size=0.02,
                kelly_size=0.01,
                risk_adjusted_size=0.01,
                confidence_level=0.1,
                reasoning=f"Error in calculation: {exc}",
            )

    def _calculate_kelly_criterion(
        self, returns: pd.Series, signal_confidence: float
    ) -> float:
        try:
            if returns is None or len(returns) < 20:
                return 0.01
            positive = returns[returns > 0]
            negative = returns[returns < 0]
            if len(positive) == 0 or len(negative) == 0:
                return 0.01
            historical_win_rate = len(positive) / len(returns)
            probability = (historical_win_rate + signal_confidence) / 2.0
            avg_win = float(positive.mean())
            avg_loss = float(abs(negative.mean()))
            if avg_loss <= 0:
                return 0.01
            b_ratio = avg_win / avg_loss
            kelly = (b_ratio * probability - (1 - probability)) / max(b_ratio, 1e-9)
            safe_kelly = max(0.0, min(kelly * 0.25, 0.05))
            return float(safe_kelly)
        except Exception as exc:  # pragma: no cover - log i fallback
            self.logger.error("Error in Kelly calculation: %s", exc)
            return 0.01

    def _volatility_adjusted_size(
        self, volatility: float, signal_strength: float
    ) -> float:
        try:
            target_vol = 0.15
            if volatility <= 0:
                return 0.01
            base_size = target_vol / volatility
            strength_multiplier = 0.5 + (signal_strength * 0.5)
            adjusted = base_size * strength_multiplier
            return float(max(0.005, min(adjusted, 0.05)))
        except Exception as exc:  # pragma: no cover - log i fallback
            self.logger.error("Error in volatility adjustment: %s", exc)
            return 0.01

    def _risk_parity_sizing(
        self, current_portfolio: Mapping[str, Mapping[str, Any]], asset_volatility: float
    ) -> float:
        try:
            if not current_portfolio:
                return 0.02
            existing_vols = [
                float(position.get("volatility", math.nan))
                for position in current_portfolio.values()
                if "volatility" in position
            ]
            existing_vols = [vol for vol in existing_vols if not math.isnan(vol)]
            avg_existing_vol = float(np.mean(existing_vols)) if existing_vols else 0.2
            if asset_volatility <= 0:
                return 0.01
            rp = (avg_existing_vol / asset_volatility) * 0.02
            return float(max(0.005, min(rp, 0.04)))
        except Exception as exc:  # pragma: no cover - log i fallback
            self.logger.error("Error in risk parity sizing: %s", exc)
            return 0.01

    def _calculate_portfolio_heat(
        self, current_portfolio: Mapping[str, Mapping[str, Any]]
    ) -> float:
        try:
            if not current_portfolio:
                return 0.0
            total_risk = 0.0
            for position in current_portfolio.values():
                size = float(position.get("size", 0.0))
                volatility = float(position.get("volatility", 0.2))
                total_risk += size * volatility
            return float(min(1.0, max(0.0, total_risk)))
        except Exception as exc:  # pragma: no cover - log i fallback
            self.logger.error("Error calculating portfolio heat: %s", exc)
            return 0.5

    def _heat_adjusted_sizing(self, portfolio_heat: float) -> float:
        try:
            heat_factor = max(0.1, 1.0 - float(portfolio_heat))
            return float(0.02 * heat_factor)
        except Exception:  # pragma: no cover - fallback
            return 0.01

    def _correlation_adjustment(
        self, symbol: str, current_portfolio: Mapping[str, Mapping[str, Any]]
    ) -> float:
        try:
            if not current_portfolio or symbol not in self.historical_returns:
                return 1.0
            sym_ret = self.historical_returns.get(symbol)
            if sym_ret is None or len(sym_ret) < 2:
                return 1.0
            correlations: List[float] = []
            for existing_symbol in current_portfolio.keys():
                if existing_symbol == symbol:
                    continue
                ex_ret = self.historical_returns.get(existing_symbol)
                if ex_ret is None or len(ex_ret) < 2:
                    continue
                corr = self.correlation_analyzer.calculate_dynamic_correlation(
                    sym_ret, ex_ret
                )
                correlations.append(abs(float(corr)))
            if not correlations:
                return 1.0
            avg_corr = float(np.mean(correlations))
            return float(max(0.3, 1.0 - avg_corr))
        except Exception as exc:  # pragma: no cover - log i fallback
            self.logger.error("Error in correlation adjustment: %s", exc)
            return 1.0

    def _calculate_max_position_size(
        self, current_portfolio: Mapping[str, Mapping[str, Any]]
    ) -> float:
        try:
            num_positions = len(current_portfolio) if current_portfolio else 0
            max_single = min(
                self.max_risk_per_trade,
                self.max_portfolio_risk / max(1, num_positions + 1),
                1.0 / max(self.max_positions, 5),
            )
            return float(max(0.005, max_single))
        except Exception as exc:  # pragma: no cover - log i fallback
            self.logger.error("Error calculating max position size: %s", exc)
            return 0.01

    # ------------------------------------------------------------------
    # Metryki ryzyka
    # ------------------------------------------------------------------
    def calculate_risk_metrics(
        self,
        portfolio_data: Mapping[str, Mapping[str, Any]],
        market_data: Mapping[str, pd.DataFrame],
    ) -> RiskMetrics:
        try:
            with capture_pandas_warnings(self.logger, component="risk.metrics"):
                var_95 = float(
                    self._calculate_var(portfolio_data, market_data, confidence=0.95)
                )
                expected_shortfall = float(
                    self._calculate_expected_shortfall(portfolio_data, market_data)
                )
                max_dd_risk = float(self._calculate_drawdown_risk(portfolio_data))
                correlation_risk = float(
                    self._calculate_correlation_risk(portfolio_data, market_data)
                )
                liquidity_risk = float(
                    self._calculate_liquidity_risk(portfolio_data, market_data)
                )

                overall_score = (
                    var_95 * 0.3
                    + expected_shortfall * 0.25
                    + max_dd_risk * 0.2
                    + correlation_risk * 0.15
                    + liquidity_risk * 0.1
                )
                overall_score = float(min(1.0, max(0.0, overall_score)))

                if overall_score < 0.2:
                    level = RiskLevel.VERY_LOW
                elif overall_score < 0.4:
                    level = RiskLevel.LOW
                elif overall_score < 0.6:
                    level = RiskLevel.MEDIUM
                elif overall_score < 0.8:
                    level = RiskLevel.HIGH
                else:
                    level = RiskLevel.VERY_HIGH

                if level.value >= RiskLevel.HIGH.value:
                    self._emit_alert(
                        "Podwyższone ryzyko portfela",
                        severity=(
                            AlertSeverity.WARNING
                            if level == RiskLevel.HIGH
                            else AlertSeverity.CRITICAL
                        ),
                        context={
                            "overall_risk_score": overall_score,
                            "risk_level": level.name,
                            "var_95": var_95,
                            "expected_shortfall": expected_shortfall,
                        },
                    )

                return RiskMetrics(
                    var_95=var_95,
                    expected_shortfall=expected_shortfall,
                    max_drawdown_risk=max_dd_risk,
                    correlation_risk=correlation_risk,
                    liquidity_risk=liquidity_risk,
                    overall_risk_score=overall_score,
                    risk_level=level,
                )
        except Exception as exc:  # pragma: no cover - log i fallback
            self.logger.error("Error calculating risk metrics: %s", exc)
        return RiskMetrics(
            var_95=0.05,
            expected_shortfall=0.08,
            max_drawdown_risk=0.1,
            correlation_risk=0.5,
            liquidity_risk=0.3,
            overall_risk_score=0.5,
            risk_level=RiskLevel.MEDIUM,
        )


class MultiAccountRiskManager:
    """Agreguje metryki ryzyka dla wielu kont i wyznacza limity dynamiczne."""

    def __init__(self, *, base_params: Mapping[str, Any] | None = None) -> None:
        self._base_params = dict(base_params or {})
        self._managers: Dict[str, RiskManagement] = {}
        self.logger = logging.getLogger(__name__)

    def _ensure_manager(
        self, account_id: str, params: Mapping[str, Any] | None
    ) -> RiskManagement:
        if account_id not in self._managers:
            merged = dict(self._base_params)
            if params:
                merged.update(params)
            self._managers[account_id] = RiskManagement(merged)
        return self._managers[account_id]

    def reset_account(self, account_id: str) -> None:
        self._managers.pop(account_id, None)

    def collect_snapshot(
        self,
        exposures: Sequence[AccountExposure],
        market_data: Mapping[str, pd.DataFrame],
    ) -> MultiAccountRiskSnapshot:
        if not exposures:
            base_manager = RiskManagement(self._base_params)
            aggregate_metrics = base_manager.calculate_risk_metrics({}, market_data)
            return MultiAccountRiskSnapshot(
                aggregate_metrics=aggregate_metrics,
                accounts=tuple(),
                symbol_limits={},
            )

        account_reports: list[AccountRiskReport] = []
        aggregated_portfolio: Dict[str, Dict[str, float]] = {}

        for exposure in exposures:
            manager = self._ensure_manager(exposure.account_id, exposure.params)
            metrics = manager.calculate_risk_metrics(exposure.portfolio, market_data)
            dynamic_limit = float(
                min(
                    manager.max_portfolio_risk,
                    max(0.01, metrics.var_95 * 1.5 + metrics.correlation_risk * 0.5),
                )
            )
            account_reports.append(
                AccountRiskReport(
                    account_id=exposure.account_id,
                    equity=float(exposure.equity),
                    metrics=metrics,
                    dynamic_limit=dynamic_limit,
                )
            )

            for symbol, position in exposure.portfolio.items():
                container = aggregated_portfolio.setdefault(symbol, {"size": 0.0, "notional": 0.0})
                size = float(position.get("size") or position.get("quantity") or 0.0)
                price = float(position.get("price") or position.get("last_price") or 0.0)
                notional = float(position.get("notional") or size * price)
                container["size"] += size
                container["notional"] += notional

        aggregate_manager = RiskManagement(self._base_params)
        aggregate_metrics = aggregate_manager.calculate_risk_metrics(aggregated_portfolio, market_data)

        symbol_limits: Dict[str, float] = {}
        for symbol, payload in aggregated_portfolio.items():
            base_limit = float(
                max(
                    0.01,
                    min(1.0, aggregate_metrics.var_95 * 1.2 + aggregate_metrics.correlation_risk * 0.6),
                )
            )
            symbol_limit = base_limit
            for report, exposure in zip(account_reports, exposures):
                if symbol in exposure.portfolio:
                    symbol_limit = max(symbol_limit, report.dynamic_limit)
            exposure_size = payload.get("size", 0.0)
            if exposure_size:
                symbol_limit = min(1.0, symbol_limit + abs(exposure_size) * 0.01)
            symbol_limits[symbol] = symbol_limit

        return MultiAccountRiskSnapshot(
            aggregate_metrics=aggregate_metrics,
            accounts=tuple(account_reports),
            symbol_limits=symbol_limits,
        )

    def _calculate_var(
        self,
        portfolio_data: Mapping[str, Mapping[str, Any]],
        market_data: Mapping[str, pd.DataFrame],
        confidence: float = 0.95,
    ) -> float:
        try:
            if not portfolio_data:
                return 0.0
            portfolio_returns: List[float] = []
            for symbol, position in portfolio_data.items():
                md = market_data.get(symbol)
                if isinstance(md, pd.DataFrame) and "close" in md:
                    rets = md["close"].pct_change().dropna()
                    if len(rets) > 10:
                        sim = np.random.choice(rets, size=1000, replace=True)
                        size = float(position.get("size", 0.0))
                        portfolio_returns.extend(list(sim * size))
            if not portfolio_returns:
                return 0.0
            percentile = (1.0 - confidence) * 100.0
            var = np.percentile(portfolio_returns, percentile)
            return float(abs(var))
        except Exception as exc:  # pragma: no cover - log i fallback
            self.logger.error("Error calculating VaR: %s", exc)
            return 0.05

    def _calculate_expected_shortfall(
        self,
        portfolio_data: Mapping[str, Mapping[str, Any]],
        market_data: Mapping[str, pd.DataFrame],
    ) -> float:
        try:
            var_95 = float(self._calculate_var(portfolio_data, market_data, 0.95))
            es = float(min(var_95 * 1.3, 0.5))
            return es
        except Exception as exc:  # pragma: no cover - log i fallback
            self.logger.error("Error calculating Expected Shortfall: %s", exc)
            return 0.08

    def _calculate_drawdown_risk(
        self, portfolio_data: Mapping[str, Mapping[str, Any]]
    ) -> float:
        try:
            if not self.portfolio_value_history or len(self.portfolio_value_history) < 10:
                return 0.1
            pv = np.array(self.portfolio_value_history, dtype=float)
            peak = np.maximum.accumulate(pv)
            dd = (peak - pv) / np.maximum(peak, 1e-9)
            max_dd = float(np.max(dd))
            return float(min(max_dd * 1.8, 0.5))
        except Exception as exc:  # pragma: no cover - log i fallback
            self.logger.error("Error calculating drawdown risk: %s", exc)
            return 0.1

    def _calculate_correlation_risk(
        self,
        portfolio_data: Mapping[str, Mapping[str, Any]],
        market_data: Mapping[str, pd.DataFrame],
    ) -> float:
        try:
            if len(portfolio_data) < 2:
                return 0.0

            returns_dict: Dict[str, pd.Series] = {}
            for symbol, position in portfolio_data.items():
                if symbol in self.historical_returns:
                    returns_dict[symbol] = self.historical_returns[symbol]
                    continue
                md = market_data.get(symbol)
                if isinstance(md, pd.DataFrame) and "close" in md:
                    returns_dict[symbol] = md["close"].pct_change().dropna().tail(252)

            if len(returns_dict) < 2:
                return 0.0

            corr_risk = self.correlation_analyzer.portfolio_correlation_risk(returns_dict)
            if corr_risk > self.max_correlation_risk:
                self._emit_alert(
                    "Ryzyko korelacji przekracza dopuszczalny poziom",
                    severity=AlertSeverity.WARNING,
                    context={"correlation_risk": float(corr_risk)},
                )
            return float(corr_risk)
        except Exception as exc:  # pragma: no cover - log i fallback
            self.logger.error("Error calculating correlation risk: %s", exc)
            return 0.5

    def _calculate_liquidity_risk(
        self,
        portfolio_data: Mapping[str, Mapping[str, Any]],
        market_data: Mapping[str, pd.DataFrame],
    ) -> float:
        try:
            liquidity_scores: List[float] = []
            for symbol, position in portfolio_data.items():
                md = market_data.get(symbol)
                if not isinstance(md, pd.DataFrame) or "volume" not in md:
                    liquidity_scores.append(0.3)
                    continue
                avg_volume = float(md["volume"].tail(30).mean())
                size = float(position.get("size", 0.0))
                if avg_volume <= 0:
                    liquidity_scores.append(0.8)
                    continue
                volume_impact = size / max(avg_volume, 1e-9)
                liquidity_scores.append(min(1.0, volume_impact))
            return float(np.mean(liquidity_scores)) if liquidity_scores else 0.0
        except Exception as exc:  # pragma: no cover - log i fallback
            self.logger.error("Error calculating liquidity risk: %s", exc)
            return 0.3

    # ------------------------------------------------------------------
    # Limity pozycji i alerty
    # ------------------------------------------------------------------
    def check_position_limits(
        self,
        position: Mapping[str, Any],
        current_portfolio: Mapping[str, Mapping[str, Any]],
    ) -> Tuple[bool, str]:
        try:
            size = float(position.get("size", 0.0))
            if size <= 0:
                return False, "Position size must be positive"

            symbol = str(position.get("symbol", ""))
            sector = str(position.get("sector", ""))

            if size > self.max_risk_per_trade:
                return False, "Exceeds max risk per trade"

            total_portfolio_risk = float(
                self._calculate_portfolio_heat(current_portfolio)
            ) + size
            if total_portfolio_risk > self.max_portfolio_risk:
                return False, "Portfolio risk limit reached"

            if sector and current_portfolio:
                sector_exposure = sum(
                    float(pos.get("size", 0.0))
                    for pos in current_portfolio.values()
                    if pos.get("sector") == sector
                )
                if sector_exposure + size > self.max_sector_concentration:
                    return False, "Sector concentration limit"

            if len(current_portfolio) >= self.max_positions and symbol not in current_portfolio:
                return False, "Maximum number of positions reached"

            return True, "OK"
        except Exception as exc:  # pragma: no cover - log i fallback
            self.logger.error("Error checking position limits: %s", exc)
            return False, "Risk check error"

    def calculate_stop_loss(
        self, entry_price: float, atr: float, *, direction: str = "long"
    ) -> float:
        try:
            atr_component = atr if atr > 0 else entry_price * 0.01
            multiplier = 2.0 if self.use_dynamic_stops else 1.0
            raw_sl = entry_price - multiplier * atr_component
            if direction.lower() == "short":
                raw_sl = entry_price + multiplier * atr_component
            sl_distance = abs(raw_sl - entry_price) / max(entry_price, 1e-9)
            if sl_distance < self.min_stop_loss:
                sl_distance = self.min_stop_loss
            elif sl_distance > self.max_stop_loss:
                sl_distance = self.max_stop_loss
            if direction.lower() == "short":
                return entry_price + sl_distance * entry_price
            return entry_price - sl_distance * entry_price
        except Exception as exc:  # pragma: no cover - log i fallback
            self.logger.error("Error calculating stop loss: %s", exc)
            return entry_price * (0.98 if direction.lower() == "long" else 1.02)

    def update_trailing_stop(
        self, position: Dict[str, Any], current_price: float
    ) -> Dict[str, Any]:
        try:
            trailing_params = position.get("trailing_params")
            if not trailing_params:
                return position

            current_stop = float(position.get("stop_loss", current_price))
            position_side = str(position.get("side", "long")).lower()
            atr_multiplier = float(trailing_params.get("atr_multiplier", 2.0))
            update_threshold = float(trailing_params.get("update_threshold", 0.01))

            if position_side == "long":
                new_stop = current_price - (atr_multiplier * update_threshold)
                if new_stop > current_stop:
                    position["stop_loss"] = float(new_stop)
                    position["trailing_updated"] = True
                    self.logger.info(
                        "Trailing stop LONG: %.4f -> %.4f", current_stop, new_stop
                    )
            else:
                new_stop = current_price + (atr_multiplier * update_threshold)
                if new_stop < current_stop:
                    position["stop_loss"] = float(new_stop)
                    position["trailing_updated"] = True
                    self.logger.info(
                        "Trailing stop SHORT: %.4f -> %.4f", current_stop, new_stop
                    )

            return position
        except Exception as exc:  # pragma: no cover - log i fallback
            self.logger.error("Error updating trailing stop: %s", exc)
            return position

    # ------------------------------------------------------------------
    # Awaryjne kontrole i raportowanie
    # ------------------------------------------------------------------
    def emergency_risk_check(
        self,
        portfolio_value: float,
        initial_value: float,
        current_positions: Mapping[str, Mapping[str, Any]],
    ) -> Dict[str, Any]:
        try:
            current_drawdown = (
                (initial_value - portfolio_value) / initial_value
                if initial_value > 0
                else 0.0
            )
            actions: List[str] = []
            alerts: List[str] = []

            if current_drawdown > self.emergency_stop_drawdown:
                actions.append("EMERGENCY_STOP")
                alerts.append(
                    f"Emergency drawdown exceeded: {current_drawdown:.2%} > "
                    f"{self.emergency_stop_drawdown:.2%}"
                )
                self._emit_alert(
                    f"Drawdown {current_drawdown:.2%} przekracza limit awaryjny",
                    severity=AlertSeverity.CRITICAL,
                    context={"drawdown": float(current_drawdown)},
                )

            portfolio_heat = self._calculate_portfolio_heat(current_positions)
            if portfolio_heat > 0.8:
                actions.append("REDUCE_POSITIONS")
                alerts.append(f"Portfolio heat too high: {portfolio_heat:.2%}")
                self._emit_alert(
                    f"Przegrzanie portfela: heat {portfolio_heat:.2%}",
                    severity=AlertSeverity.WARNING,
                    context={"portfolio_heat": float(portfolio_heat)},
                )

            position_sizes = [
                float(pos.get("size", 0.0)) for pos in current_positions.values()
            ] if current_positions else []
            max_position = float(max(position_sizes)) if position_sizes else 0.0
            if max_position > 0.15:
                actions.append("REDUCE_LARGEST_POSITION")
                alerts.append(f"Position too large: {max_position:.2%}")
                self._emit_alert(
                    f"Pojedyncza pozycja {max_position:.2%} przekracza limit",
                    severity=AlertSeverity.WARNING,
                    context={"max_position": float(max_position)},
                )

            correlation_risk = self._calculate_correlation_risk(current_positions, {})
            if correlation_risk > 0.8:
                actions.append("DIVERSIFY_PORTFOLIO")
                alerts.append(f"High correlation risk: {correlation_risk:.2%}")
                self._emit_alert(
                    f"Wysoka korelacja portfela ({correlation_risk:.2%})",
                    severity=AlertSeverity.WARNING,
                    context={"correlation_risk": float(correlation_risk)},
                )

            return {
                "emergency_stop_required": "EMERGENCY_STOP" in actions,
                "actions_required": actions,
                "risk_alerts": alerts,
                "current_drawdown": float(current_drawdown),
                "portfolio_heat": float(portfolio_heat),
                "max_position_size": float(max_position),
                "correlation_risk": float(correlation_risk),
            }
        except Exception as exc:  # pragma: no cover - log i fallback
            self.logger.error("Error in emergency risk check: %s", exc)
            self._emit_alert(
                f"Błąd awaryjnej kontroli ryzyka: {exc}",
                severity=AlertSeverity.ERROR,
                context={"error": str(exc)},
            )
            return {
                "emergency_stop_required": True,
                "actions_required": ["EMERGENCY_STOP"],
                "risk_alerts": [f"Error in risk calculation: {exc}"],
                "current_drawdown": 0.5,
                "portfolio_heat": 1.0,
                "max_position_size": 1.0,
                "correlation_risk": 1.0,
            }

    def generate_risk_report(
        self,
        portfolio_data: Mapping[str, Mapping[str, Any]],
        market_data: Mapping[str, pd.DataFrame],
    ) -> str:
        try:
            metrics = self.calculate_risk_metrics(portfolio_data, market_data)
            report: List[str] = []
            report.append("=== RISK MANAGEMENT REPORT ===\n")
            report.append(f"🎯 Overall Risk Level: {metrics.risk_level.name}\n")
            report.append(f"📊 Risk Score: {metrics.overall_risk_score:.3f}\n\n")
            report.append("📈 Risk Metrics:\n")
            report.append(f"  • Value at Risk (95%): {metrics.var_95:.3f}\n")
            report.append(f"  • Expected Shortfall: {metrics.expected_shortfall:.3f}\n")
            report.append(f"  • Max Drawdown Risk: {metrics.max_drawdown_risk:.3f}\n")
            report.append(f"  • Correlation Risk: {metrics.correlation_risk:.3f}\n")
            report.append(f"  • Liquidity Risk: {metrics.liquidity_risk:.3f}\n\n")

            if portfolio_data:
                num_positions = len(portfolio_data)
                total_exposure = float(
                    sum(pos.get("size", 0.0) for pos in portfolio_data.values())
                )
                report.append("📋 Portfolio Overview:\n")
                report.append(f"  • Number of Positions: {num_positions}\n")
                report.append(f"  • Total Exposure: {total_exposure:.2%}\n")
                report.append(
                    f"  • Portfolio Heat: {self._calculate_portfolio_heat(portfolio_data):.3f}\n\n"
                )

            report.append("💡 Risk Recommendations:\n")
            if metrics.risk_level.value >= RiskLevel.HIGH.value:
                report.append("  ⚠️  REDUCE RISK: Consider closing some positions\n")
                report.append("  ⚠️  INCREASE DIVERSIFICATION: Add uncorrelated assets\n")
                report.append("  ⚠️  TIGHTEN STOPS: Reduce stop-loss distances\n")
            elif metrics.risk_level.value <= RiskLevel.LOW.value:
                report.append("  ✅ CURRENT RISK ACCEPTABLE: Monitor and maintain\n")
                report.append("  🔍 OPPORTUNITY: Consider increasing position sizes\n")
            else:
                report.append("  ⚖️  BALANCED RISK: Current approach appropriate\n")
                report.append("  👀 MONITOR: Watch for regime changes\n")

            if len(portfolio_data) > 1:
                report.append("\n🔗 Correlation Analysis:\n")
                report.append(
                    f"  • Average Correlation: {metrics.correlation_risk:.3f}\n"
                )
                if metrics.correlation_risk > 0.7:
                    report.append(
                        "  ⚠️  HIGH CORRELATION: Positions may move together\n"
                    )
                elif metrics.correlation_risk < 0.3:
                    report.append(
                        "  ✅ GOOD DIVERSIFICATION: Low correlation between positions\n"
                    )

            return "".join(report)
        except Exception as exc:  # pragma: no cover - log i fallback
            self.logger.error("Error generating risk report: %s", exc)
            return f"Error generating risk report: {exc}"

    def optimize_portfolio_risk(
        self,
        current_portfolio: Mapping[str, Mapping[str, Any]],
        target_risk: Optional[float] = None,
    ) -> Dict[str, Any]:
        try:
            if not current_portfolio:
                return {"optimization_needed": False, "suggestions": []}

            target = float(target_risk or (self.max_portfolio_risk * 0.8))
            current_risk = float(self._calculate_portfolio_heat(current_portfolio))
            suggestions: List[Dict[str, Any]] = []

            if current_risk > target:
                excess = current_risk - target
                positions_by_risk: List[Tuple[str, float, Mapping[str, Any]]] = []
                for symbol, pos in current_portfolio.items():
                    prisk = float(pos.get("size", 0.0)) * float(pos.get("volatility", 0.2))
                    positions_by_risk.append((symbol, prisk, pos))
                positions_by_risk.sort(key=lambda item: item[1], reverse=True)

                risk_to_reduce = float(excess)
                for symbol, prisk, pos in positions_by_risk:
                    if risk_to_reduce <= 0:
                        break
                    max_reduction = prisk * 0.5
                    reduction = min(risk_to_reduce, max_reduction)
                    suggested_size = max(
                        0.0,
                        float(pos.get("size", 0.0))
                        - reduction / float(pos.get("volatility", 0.2)),
                    )
                    suggestions.append(
                        {
                            "action": "reduce_position",
                            "symbol": symbol,
                            "current_size": float(pos.get("size", 0.0)),
                            "suggested_size": suggested_size,
                            "risk_reduction": float(reduction),
                            "reason": "Portfolio risk too high",
                        }
                    )
                    risk_to_reduce -= reduction

            elif current_risk < target * 0.5:
                available_risk = target - current_risk
                for symbol, pos in current_portfolio.items():
                    if available_risk <= 0:
                        break
                    cur_size = float(pos.get("size", 0.0))
                    volatility = float(pos.get("volatility", 0.2))
                    max_increase = float(available_risk / max(volatility, 1e-9))
                    suggested_increase = min(max_increase, cur_size * 0.5)
                    if suggested_increase > 0.01:
                        suggestions.append(
                            {
                                "action": "increase_position",
                                "symbol": symbol,
                                "current_size": cur_size,
                                "suggested_size": cur_size + suggested_increase,
                                "risk_increase": suggested_increase * volatility,
                                "reason": "Portfolio risk below target",
                            }
                        )
                        available_risk -= suggested_increase * volatility

            return {
                "optimization_needed": len(suggestions) > 0,
                "current_risk": current_risk,
                "target_risk": target,
                "suggestions": suggestions,
            }
        except Exception as exc:  # pragma: no cover - log i fallback
            self.logger.error("Error optimizing portfolio risk: %s", exc)
            return {"optimization_needed": False, "error": str(exc)}

    def update_portfolio_state(
        self,
        portfolio_value: float,
        positions: Mapping[str, Mapping[str, Any]],
        market_returns: Mapping[str, pd.Series],
    ) -> None:
        try:
            with capture_pandas_warnings(
                self.logger, component="risk.portfolio_state"
            ):
                self.portfolio_value_history.append(float(portfolio_value))
                if len(self.portfolio_value_history) > 252:
                    self.portfolio_value_history = self.portfolio_value_history[-252:]

                self.current_positions = dict(positions) if positions else {}

                for symbol, returns in (market_returns or {}).items():
                    if symbol not in self.historical_returns:
                        self.historical_returns[symbol] = returns
                    else:
                        combined = pd.concat([self.historical_returns[symbol], returns]).dropna()
                        self.historical_returns[symbol] = combined.tail(252)
                self.logger.debug(
                    "Updated portfolio state: value=%.2f, positions=%d",
                    portfolio_value,
                    len(self.current_positions),
                )
        except Exception as exc:  # pragma: no cover - log i fallback
            self.logger.error("Error updating portfolio state: %s", exc)

    def _generate_sizing_reasoning(
        self,
        kelly_size: float,
        vol_adjusted_size: float,
        risk_parity_size: float,
        heat_adjusted_size: float,
        correlation_adjustment: float,
        portfolio_heat: float,
    ) -> str:
        return (
            "Sizing components -> "
            f"Kelly: {kelly_size:.3f}, Vol: {vol_adjusted_size:.3f}, "
            f"RiskParity: {risk_parity_size:.3f}, Heat: {heat_adjusted_size:.3f}, "
            f"CorrAdj: {correlation_adjustment:.3f}, Heat: {portfolio_heat:.3f}"
        )


def create_risk_manager(config: Mapping[str, Any]) -> RiskManagement:
    try:
        return RiskManagement(config)
    except Exception as exc:  # pragma: no cover - log i fallback
        _LOGGER.exception("create_risk_manager failed")
        return RiskManagement({})


def backtest_risk_strategy(
    risk_manager: RiskManagement,
    historical_data: Mapping[str, pd.DataFrame],
    signals: Sequence[Mapping[str, Any]],
) -> Dict[str, Any]:
    try:
        with capture_pandas_warnings(_LOGGER, component="risk.backtest"):
            results: Dict[str, Any] = {
                "total_trades": 0,
                "risk_adjusted_trades": 0,
                "rejected_trades": 0,
                "max_portfolio_heat": 0.0,
                "risk_events": [],
            }
            portfolio: Dict[str, Dict[str, Any]] = {}

            for signal in signals or []:
                symbol = str(signal.get("symbol", "UNKNOWN"))
                md = historical_data.get(symbol)
                if not isinstance(md, pd.DataFrame):
                    continue

                sizing = risk_manager.calculate_position_size(
                    symbol, signal, md, portfolio
                )
                can_trade, reason = risk_manager.check_position_limits(
                    {
                        "symbol": symbol,
                        "size": sizing.recommended_size,
                        "volatility": 0.2,
                    },
                    portfolio,
                )
                results["total_trades"] += 1
                if can_trade and sizing.recommended_size > 0.005:
                    entry_price = (
                        signal.get("price")
                        if "price" in signal
                        else float(md["close"].iloc[-1]) if "close" in md and len(md) else 0.0
                    )
                    portfolio[symbol] = {
                        "size": sizing.recommended_size,
                        "entry_price": float(entry_price),
                        "volatility": 0.2,
                    }
                    results["risk_adjusted_trades"] += 1
                else:
                    results["rejected_trades"] += 1
                    results["risk_events"].append(
                        {
                            "symbol": symbol,
                            "reason": reason,
                            "suggested_size": sizing.recommended_size,
                        }
                    )

                heat = risk_manager._calculate_portfolio_heat(portfolio)
                results["max_portfolio_heat"] = max(
                    results["max_portfolio_heat"], float(heat)
                )

            results["risk_adjusted_ratio"] = (
                results["risk_adjusted_trades"] / results["total_trades"]
                if results["total_trades"] > 0
                else 0.0
            )
            return results
    except Exception as exc:  # pragma: no cover - log i fallback
        _LOGGER.exception("backtest_risk_strategy failed")
        return {"error": str(exc), "total_trades": 0}


def calculate_optimal_leverage(
    returns: pd.Series, target_volatility: float = 0.15
) -> float:
    try:
        with capture_pandas_warnings(_LOGGER, component="risk.leverage"):
            if returns is None or len(returns) < 20:
                return 1.0
            port_vol = float(returns.std() * np.sqrt(252))
            if port_vol <= 0:
                return 1.0
            lev = float(target_volatility / port_vol)
            return float(max(0.1, min(lev, 3.0)))
    except Exception:  # pragma: no cover - fallback
        return 1.0
