# risk_management.py
# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
from typing import Any, Dict, Tuple, Optional, List
import logging
from dataclasses import dataclass
from enum import Enum
import math

from bot_core.alerts import AlertSeverity, emit_alert


# --- logging ---
logger = logging.getLogger(__name__)
if not logger.handlers:
    _h = logging.StreamHandler()
    _f = logging.Formatter('[%(asctime)s] %(name)s - %(levelname)s - %(message)s')
    _h.setFormatter(_f)
    logger.addHandler(_h)
logger.setLevel(logging.INFO)


class RiskLevel(Enum):
    """Poziomy ryzyka"""
    VERY_LOW = 1
    LOW = 2
    MEDIUM = 3
    HIGH = 4
    VERY_HIGH = 5


@dataclass
class RiskMetrics:
    """Metryki ryzyka dla pozycji/portfela"""
    var_95: float                # Value at Risk 95%
    expected_shortfall: float    # Expected Shortfall (CVaR)
    max_drawdown_risk: float     # Ryzyko max drawdown
    correlation_risk: float      # Ryzyko korelacji
    liquidity_risk: float        # Ryzyko płynności
    overall_risk_score: float    # Ogólny scoring ryzyka 0..1
    risk_level: RiskLevel


@dataclass
class PositionSizing:
    """Wynik kalkulacji wielkości pozycji"""
    recommended_size: float
    max_allowed_size: float
    kelly_size: float
    risk_adjusted_size: float
    confidence_level: float
    reasoning: str


class VolatilityEstimator:
    """Zaawansowane szacowanie volatility (EWMA/GARCH uproszczony)"""

    def __init__(self, lookback_periods: int = 252):
        self.lookback = lookback_periods
        self.logger = logging.getLogger(__name__)

    def ewma_volatility(self, returns: pd.Series, lambda_param: float = 0.94) -> float:
        """Exponentially Weighted Moving Average volatility"""
        try:
            if returns is None or len(returns) < 2:
                return 0.02
            if len(returns) < 10:
                return float(returns.std()) if len(returns) > 1 else 0.02

            weights = np.array([(1 - lambda_param) * (lambda_param ** i)
                                for i in range(len(returns))])[::-1]
            weights = weights / weights.sum()
            weighted_variance = np.sum(weights * (returns ** 2))
            return float(np.sqrt(weighted_variance * 252))  # Annualizacja
        except Exception as e:
            self.logger.error(f"Error calculating EWMA volatility: {e}")
            return 0.02

    def garch_volatility(self, returns: pd.Series) -> float:
        """Uproszczona estymacja GARCH(1,1)"""
        try:
            if returns is None or len(returns) < 50:
                return self.ewma_volatility(returns if returns is not None else pd.Series([], dtype=float))
            omega, alpha, beta = 1e-6, 0.1, 0.85
            variances = [float(returns.var())]
            lim = min(len(returns), 100)
            for i in range(1, lim):
                new_var = omega + alpha * (returns.iloc[i - 1] ** 2) + beta * variances[-1]
                variances.append(float(new_var))
            return float(np.sqrt(variances[-1] * 252))
        except Exception as e:
            self.logger.error(f"Error calculating GARCH volatility: {e}")
            return self.ewma_volatility(returns if returns is not None else pd.Series([], dtype=float))

    def realized_volatility(self, returns: pd.Series) -> float:
        """Realized volatility (roczna)"""
        try:
            return float(returns.std() * np.sqrt(252)) if returns is not None and len(returns) else 0.02
        except Exception:
            return 0.02


class CorrelationAnalyzer:
    """Analizy korelacji między instrumentami"""

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.correlation_cache: Dict[str, float] = {}

    def calculate_dynamic_correlation(self, returns1: pd.Series, returns2: pd.Series,
                                      window: int = 60) -> float:
        """Dynamiczna korelacja z rolling window"""
        try:
            if returns1 is None or returns2 is None:
                return 0.0
            if len(returns1) < window or len(returns2) < window:
                return 0.0
            aligned = pd.concat([returns1, returns2], axis=1).dropna()
            if len(aligned) < window:
                return 0.0
            rolling_corr = aligned.iloc[:, 0].rolling(window).corr(aligned.iloc[:, 1])
            val = rolling_corr.iloc[-1]
            return float(val) if not pd.isna(val) else 0.0
        except Exception as e:
            self.logger.error(f"Error calculating dynamic correlation: {e}")
            return 0.0

    def portfolio_correlation_risk(self, returns_dict: Dict[str, pd.Series]) -> float:
        """Oblicza ryzyko korelacji dla całego portfela (0..1)"""
        try:
            symbols = list(returns_dict.keys())
            if len(symbols) < 2:
                return 0.0
            correlations: List[float] = []
            for i in range(len(symbols)):
                for j in range(i + 1, len(symbols)):
                    corr = self.calculate_dynamic_correlation(
                        returns_dict[symbols[i]], returns_dict[symbols[j]]
                    )
                    correlations.append(abs(float(corr)))
            avg_corr = float(np.mean(correlations)) if correlations else 0.0
            return min(1.0, max(0.0, avg_corr))
        except Exception as e:
            self.logger.error(f"Error calculating portfolio correlation risk: {e}")
            return 0.5


class RiskManagement:
    """Zaawansowane zarządzanie ryzykiem"""

    def __init__(self, params: Dict = None):
        self.params = params or {}
        self.logger = logging.getLogger(__name__)

        # Basic risk parameters
        self.max_risk_per_trade = float(self.params.get('max_risk_per_trade', 0.02))  # 2%
        self.max_portfolio_risk = float(self.params.get('max_portfolio_risk', 0.10))  # 10%
        self.max_correlation_risk = float(self.params.get('max_correlation_risk', 0.7))
        self.max_sector_concentration = float(self.params.get('max_sector_concentration', 0.3))

        # Advanced parameters
        self.confidence_level = float(self.params.get('confidence_level', 0.95))
        self.lookback_period = int(self.params.get('lookback_period', 252))
        self.max_positions = int(self.params.get('max_positions', 10))
        self.emergency_stop_drawdown = float(self.params.get('emergency_stop_drawdown', 0.15))

        # Stop loss parameters
        self.min_stop_loss = float(self.params.get('min_stop_loss', 0.005))  # 0.5%
        self.max_stop_loss = float(self.params.get('max_stop_loss', 0.05))   # 5%
        self.use_dynamic_stops = bool(self.params.get('use_dynamic_stops', True))

        # Initialize components
        self.volatility_estimator = VolatilityEstimator(self.lookback_period)
        self.correlation_analyzer = CorrelationAnalyzer()

        # Portfolio state
        self.current_positions: Dict[str, Dict] = {}
        self.historical_returns: Dict[str, pd.Series] = {}
        self.portfolio_value_history: List[float] = []

        # Validation
        self._validate_parameters()

    def _emit_alert(self, message: str, *, severity: AlertSeverity = AlertSeverity.WARNING, context: Optional[Dict[str, Any]] = None) -> None:
        try:
            emit_alert(message, severity=severity, source="risk", context=context)
        except Exception:  # pragma: no cover - alert nie może zatrzymać risk engine'u
            self.logger.exception("Failed to emit risk alert")

    # ---------- Walidacja ----------
    def _validate_parameters(self):
        """Waliduje parametry zarządzania ryzykiem"""
        if self.max_portfolio_risk <= self.max_risk_per_trade:
            raise ValueError("max_portfolio_risk must be greater than max_risk_per_trade")
        if not 0.5 <= self.confidence_level <= 0.99:
            raise ValueError("confidence_level must be between 0.5 and 0.99")
        if self.max_positions < 1:
            raise ValueError("max_positions must be at least 1")

    # ---------- Position sizing ----------
    def calculate_position_size(self, symbol: str, signal_data: Dict,
                                market_data: pd.DataFrame,
                                current_portfolio: Dict) -> PositionSizing:
        """Zaawansowane obliczanie wielkości pozycji (frakcja kapitału 0..1)"""
        try:
            signal_strength = float(signal_data.get('strength', 0.5))
            signal_confidence = float(signal_data.get('confidence', 0.5))

            returns = market_data['close'].pct_change().dropna() if 'close' in market_data else pd.Series([], dtype=float)

            # 1) Kelly (bezpieczny ułamek)
            kelly_size = self._calculate_kelly_criterion(returns, signal_confidence)

            # 2) Volatility-based
            volatility = self.volatility_estimator.ewma_volatility(returns)
            vol_adjusted_size = self._volatility_adjusted_size(volatility, signal_strength)

            # 3) Risk parity
            risk_parity_size = self._risk_parity_sizing(current_portfolio, volatility)

            # 4) Portfolio heat
            portfolio_heat = self._calculate_portfolio_heat(current_portfolio)
            heat_adjusted_size = self._heat_adjusted_sizing(portfolio_heat)
            if portfolio_heat > 0.8:
                self._emit_alert(
                    f"Wysokie 'portfolio heat' ({portfolio_heat:.2%}) – ograniczam ekspozycję.",
                    severity=AlertSeverity.WARNING,
                    context={"portfolio_heat": float(portfolio_heat), "symbol": symbol},
                )

            # 5) Korelacja
            correlation_adjustment = self._correlation_adjustment(symbol, current_portfolio)

            sizes = [kelly_size, vol_adjusted_size, risk_parity_size, heat_adjusted_size]
            weights = [0.3, 0.3, 0.2, 0.2]
            combined_size = float(sum(s * w for s, w in zip(sizes, weights)))
            final_size = combined_size * float(correlation_adjustment)

            # limity pozycji
            max_allowed = float(self._calculate_max_position_size(current_portfolio))
            recommended_size = float(max(0.0, min(final_size, max_allowed)))
            if recommended_size == 0.0:
                self._emit_alert(
                    f"Rekomendacja risk engine: brak nowej pozycji na {symbol} (0%).",
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
                kelly_size, vol_adjusted_size, risk_parity_size,
                heat_adjusted_size, float(correlation_adjustment), float(portfolio_heat)
            )

            return PositionSizing(
                recommended_size=recommended_size,
                max_allowed_size=max_allowed,
                kelly_size=kelly_size,
                risk_adjusted_size=final_size,
                confidence_level=confidence,
                reasoning=reasoning
            )

        except Exception as e:
            self.logger.error(f"Error calculating position size: {e}")
            self._emit_alert(
                f"Błąd kalkulacji wielkości pozycji dla {symbol}: {e}",
                severity=AlertSeverity.ERROR,
                context={"symbol": symbol},
            )
            return PositionSizing(
                recommended_size=0.01,
                max_allowed_size=0.02,
                kelly_size=0.01,
                risk_adjusted_size=0.01,
                confidence_level=0.1,
                reasoning=f"Error in calculation: {e}"
            )

    def _calculate_kelly_criterion(self, returns: pd.Series, signal_confidence: float) -> float:
        """Kelly Criterion (zabezpieczony ułamek)"""
        try:
            if returns is None or len(returns) < 20:
                return 0.01
            pos = returns[returns > 0]
            neg = returns[returns < 0]
            if len(pos) == 0 or len(neg) == 0:
                return 0.01
            historical_win_rate = len(pos) / len(returns)
            p = (historical_win_rate + signal_confidence) / 2.0
            avg_win = float(pos.mean())
            avg_loss = float(abs(neg.mean()))
            if avg_loss <= 0:
                return 0.01
            b = avg_win / avg_loss
            kelly = (b * p - (1 - p)) / max(b, 1e-9)
            safe_kelly = max(0.0, min(kelly * 0.25, 0.05))  # 25% Kelly, max 5%
            return float(safe_kelly)
        except Exception as e:
            self.logger.error(f"Error in Kelly calculation: {e}")
            return 0.01

    def _volatility_adjusted_size(self, volatility: float, signal_strength: float) -> float:
        """Rozmiar pozycji odwrotnie proporcjonalny do zmienności"""
        try:
            target_vol = 0.15
            if volatility <= 0:
                return 0.01
            base_size = target_vol / volatility
            strength_multiplier = 0.5 + (signal_strength * 0.5)  # 0.5..1.0
            adjusted = base_size * strength_multiplier
            return float(max(0.005, min(adjusted, 0.05)))
        except Exception as e:
            self.logger.error(f"Error in volatility adjustment: {e}")
            return 0.01

    def _risk_parity_sizing(self, current_portfolio: Dict, asset_volatility: float) -> float:
        """Risk parity: wyrównanie kontrybucji ryzyka"""
        try:
            if not current_portfolio:
                return 0.02
            existing_vols = [pos.get('volatility', None) for pos in current_portfolio.values() if 'volatility' in pos]
            avg_existing_vol = float(np.mean(existing_vols)) if existing_vols else 0.2
            if asset_volatility <= 0:
                return 0.01
            rp = (avg_existing_vol / asset_volatility) * 0.02
            return float(max(0.005, min(rp, 0.04)))
        except Exception as e:
            self.logger.error(f"Error in risk parity sizing: {e}")
            return 0.01

    def _calculate_portfolio_heat(self, current_portfolio: Dict) -> float:
        """Całkowite 'ciepło' portfela (orientacyjnie 0..1)"""
        try:
            if not current_portfolio:
                return 0.0
            total_risk = 0.0
            for position in current_portfolio.values():
                size = float(position.get('size', 0.0))
                vol = float(position.get('volatility', 0.2))
                total_risk += size * vol
            return float(min(1.0, max(0.0, total_risk)))
        except Exception as e:
            self.logger.error(f"Error calculating portfolio heat: {e}")
            return 0.5

    def _heat_adjusted_sizing(self, portfolio_heat: float) -> float:
        """Zmniejsza rozmiar wraz ze wzrostem 'heat'"""
        try:
            heat_factor = max(0.1, 1.0 - float(portfolio_heat))
            return float(0.02 * heat_factor)
        except Exception:
            return 0.01

    def _correlation_adjustment(self, symbol: str, current_portfolio: Dict) -> float:
        """Zmniejsza rozmiar, gdy nowa pozycja mocno skorelowana z istniejącymi"""
        try:
            if not current_portfolio or symbol not in self.historical_returns:
                return 1.0
            sym_ret = self.historical_returns.get(symbol)
            if sym_ret is None or len(sym_ret) < 2:
                return 1.0
            corrs: List[float] = []
            for ex_sym in current_portfolio.keys():
                if ex_sym == symbol:
                    continue
                ex_ret = self.historical_returns.get(ex_sym)
                if ex_ret is None or len(ex_ret) < 2:
                    continue
                c = self.correlation_analyzer.calculate_dynamic_correlation(sym_ret, ex_ret)
                corrs.append(abs(float(c)))
            if not corrs:
                return 1.0
            avg_corr = float(np.mean(corrs))
            return float(max(0.3, 1.0 - avg_corr))  # 0.3..1.0
        except Exception as e:
            self.logger.error(f"Error in correlation adjustment: {e}")
            return 1.0

    def _calculate_max_position_size(self, current_portfolio: Dict) -> float:
        """Maksymalna wielkość pozycji z ograniczeń dywersyfikacyjnych i ryzyka"""
        try:
            num_positions = len(current_portfolio) if current_portfolio else 0
            max_single = min(
                self.max_risk_per_trade,
                self.max_portfolio_risk / max(1, num_positions + 1),  # +1 zakłada nową pozycję
                1.0 / max(self.max_positions, 5)
            )
            return float(max(0.005, max_single))
        except Exception as e:
            self.logger.error(f"Error calculating max position size: {e}")
            return 0.01

    # ---------- Metryki ryzyka ----------
    def calculate_risk_metrics(self, portfolio_data: Dict, market_data: Dict) -> RiskMetrics:
        """Oblicza kompleksowe metryki ryzyka portfela"""
        try:
            var_95 = float(self._calculate_var(portfolio_data, market_data, confidence=0.95))
            expected_shortfall = float(self._calculate_expected_shortfall(portfolio_data, market_data))
            max_dd_risk = float(self._calculate_drawdown_risk(portfolio_data))
            correlation_risk = float(self._calculate_correlation_risk(portfolio_data, market_data))
            liquidity_risk = float(self._calculate_liquidity_risk(portfolio_data, market_data))

            overall_score = (
                var_95 * 0.3 +
                expected_shortfall * 0.25 +
                max_dd_risk * 0.2 +
                correlation_risk * 0.15 +
                liquidity_risk * 0.1
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
                    severity=AlertSeverity.WARNING if level == RiskLevel.HIGH else AlertSeverity.CRITICAL,
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
                risk_level=level
            )
        except Exception as e:
            self.logger.error(f"Error calculating risk metrics: {e}")
            return RiskMetrics(
                var_95=0.05, expected_shortfall=0.08, max_drawdown_risk=0.1,
                correlation_risk=0.5, liquidity_risk=0.3, overall_risk_score=0.5,
                risk_level=RiskLevel.MEDIUM
            )

    def _calculate_var(self, portfolio_data: Dict, market_data: Dict, confidence: float = 0.95) -> float:
        """Monte Carlo / historyczne VaR"""
        try:
            if not portfolio_data:
                return 0.0
            portfolio_returns: List[float] = []
            for symbol, position in portfolio_data.items():
                md = market_data.get(symbol)
                if isinstance(md, pd.DataFrame) and 'close' in md:
                    rets = md['close'].pct_change().dropna()
                    if len(rets) > 10:
                        sim = np.random.choice(rets, size=1000, replace=True)
                        size = float(position.get('size', 0.0))
                        portfolio_returns.extend(list(sim * size))
            if not portfolio_returns:
                return 0.0
            percentile = (1.0 - confidence) * 100.0
            var = np.percentile(portfolio_returns, percentile)
            return float(abs(var))
        except Exception as e:
            self.logger.error(f"Error calculating VaR: {e}")
            return 0.05

    def _calculate_expected_shortfall(self, portfolio_data: Dict, market_data: Dict) -> float:
        """Expected Shortfall (CVaR) – uproszczone"""
        try:
            var_95 = float(self._calculate_var(portfolio_data, market_data, 0.95))
            es = float(min(var_95 * 1.3, 0.5))
            return es
        except Exception as e:
            self.logger.error(f"Error calculating Expected Shortfall: {e}")
            return 0.08

    def _calculate_drawdown_risk(self, portfolio_data: Dict) -> float:
        """Oszacowanie ryzyka max drawdown (na podstawie historii wartości portfela)"""
        try:
            if not self.portfolio_value_history or len(self.portfolio_value_history) < 10:
                return 0.1
            pv = np.array(self.portfolio_value_history, dtype=float)
            peak = np.maximum.accumulate(pv)
            dd = (peak - pv) / np.maximum(peak, 1e-9)
            max_dd = float(np.max(dd))
            return float(min(max_dd * 1.8, 0.5))
        except Exception as e:
            self.logger.error(f"Error calculating drawdown risk: {e}")
            return 0.1

    def _calculate_correlation_risk(self, portfolio_data: Dict, market_data: Dict) -> float:
        """Ryzyko korelacji między pozycjami (0..1)"""
        try:
            if len(portfolio_data) < 2:
                return 0.0
            returns_dict: Dict[str, pd.Series] = {}
            for symbol in portfolio_data.keys():
                df = market_data.get(symbol)
                if isinstance(df, pd.DataFrame) and 'close' in df:
                    rets = df['close'].pct_change().dropna()
                    if len(rets) > 20:
                        returns_dict[symbol] = rets
            return float(self.correlation_analyzer.portfolio_correlation_risk(returns_dict))
        except Exception as e:
            self.logger.error(f"Error calculating correlation risk: {e}")
            return 0.5

    def _calculate_liquidity_risk(self, portfolio_data: Dict, market_data: Dict) -> float:
        """Ryzyko płynności (na bazie zmienności wolumenu)"""
        try:
            if not portfolio_data:
                return 0.0
            scores: List[float] = []
            for symbol, position in portfolio_data.items():
                df = market_data.get(symbol)
                if isinstance(df, pd.DataFrame) and 'volume' in df.columns:
                    vol20 = float(df['volume'].tail(20).mean()) if len(df) >= 20 else float(df['volume'].mean())
                    volstd = float(df['volume'].tail(20).std()) if len(df) >= 20 else float(df['volume'].std())
                    if vol20 > 0:
                        cv = volstd / vol20  # im większe CV, tym niższa płynność
                        score = float(min(1.0, max(0.0, cv)))
                    else:
                        score = 1.0
                    weight = float(position.get('size', 0.0))
                    scores.append(score * weight)
            return float(np.mean(scores)) if scores else 0.3
        except Exception as e:
            self.logger.error(f"Error calculating liquidity risk: {e}")
            return 0.3

    def _generate_sizing_reasoning(self, kelly: float, vol_adj: float, risk_parity: float,
                                   heat_adj: float, corr_adj: float, portfolio_heat: float) -> str:
        """Uzasadnienie wielkości pozycji (dla logów/raportów)"""
        parts = [
            f"Kelly: {kelly:.3f}",
            f"Vol-adj: {vol_adj:.3f}",
            f"Risk-parity: {risk_parity:.3f}",
            f"Heat-adj: {heat_adj:.3f}",
            f"Corr-factor: {corr_adj:.3f}",
            f"Portfolio heat: {portfolio_heat:.3f}"
        ]
        return "Position sizing → " + ", ".join(parts)

    # ---------- Limity pozycji ----------
    def check_position_limits(self, new_position: Dict, current_portfolio: Dict) -> Tuple[bool, str]:
        """Sprawdza czy nowa pozycja nie narusza limitów ryzyka"""
        try:
            if len(current_portfolio) >= self.max_positions:
                message = f"Maximum positions limit reached ({self.max_positions})"
                self._emit_alert(
                    message,
                    severity=AlertSeverity.WARNING,
                    context={"max_positions": self.max_positions},
                )
                return False, message

            current_risk = self._calculate_portfolio_heat(current_portfolio)
            new_pos_risk = float(new_position.get('size', 0.0)) * float(new_position.get('volatility', 0.2))
            total = current_risk + new_pos_risk
            if total > self.max_portfolio_risk:
                message = f"Portfolio risk limit exceeded: {total:.3f} > {self.max_portfolio_risk:.3f}"
                self._emit_alert(
                    message,
                    severity=AlertSeverity.CRITICAL,
                    context={"total_risk": total, "max_portfolio_risk": self.max_portfolio_risk},
                )
                return False, message

            symbol = new_position.get('symbol')
            if symbol and symbol in self.historical_returns:
                corr_factor = self._correlation_adjustment(symbol, current_portfolio)
                if corr_factor < 0.5:
                    message = f"High correlation risk detected (factor: {corr_factor:.3f})"
                    self._emit_alert(
                        message,
                        severity=AlertSeverity.WARNING,
                        context={"symbol": symbol, "correlation_factor": float(corr_factor)},
                    )
                    return False, message

            sector = new_position.get('sector')
            if sector:
                sector_exposure = sum(
                    pos.get('size', 0.0) for pos in current_portfolio.values() if pos.get('sector') == sector
                )
                new_sector_exposure = float(sector_exposure) + float(new_position.get('size', 0.0))
                if new_sector_exposure > self.max_sector_concentration:
                    message = (
                        f"Sector concentration limit exceeded: {new_sector_exposure:.3f} > {self.max_sector_concentration:.3f}"
                    )
                    self._emit_alert(
                        message,
                        severity=AlertSeverity.WARNING,
                        context={"sector": sector, "exposure": new_sector_exposure},
                    )
                    return False, message

            return True, "Position approved"
        except Exception as e:
            self.logger.error(f"Error checking position limits: {e}")
            self._emit_alert(
                f"Błąd walidacji limitów pozycji: {e}",
                severity=AlertSeverity.ERROR,
                context={"error": str(e)},
            )
            return False, f"Error in limit check: {e}"

    # ---------- Stopy i trailing ----------
    def calculate_dynamic_stops(self, symbol: str, entry_price: float,
                                market_data: pd.DataFrame, position_side: str) -> Dict:
        """Oblicza dynamiczne (ATR) lub statyczne stop lossy dla pozycji"""
        try:
            if not self.use_dynamic_stops:
                dist = float(self.max_risk_per_trade)
                stop_price = entry_price * (1 - dist) if position_side == 'long' else entry_price * (1 + dist)
                return {'stop_loss': stop_price, 'type': 'static', 'distance': dist}

            returns = market_data['close'].pct_change().dropna() if 'close' in market_data else pd.Series([], dtype=float)
            volatility = self.volatility_estimator.ewma_volatility(returns)

            if 'volatility' in market_data.columns and len(market_data) >= 20:
                atr = float(market_data['volatility'].tail(20).mean())
            else:
                high_low = market_data['high'] - market_data['low'] if all(c in market_data for c in ['high', 'low']) else pd.Series([], dtype=float)
                atr = float(high_low.tail(20).mean()) if len(high_low) else max(0.001, entry_price * 0.005)  # fallback

            atr_mult = 2.0 if volatility < 0.3 else 3.0
            stop_distance = float(atr_mult * atr / max(entry_price, 1e-9))
            stop_distance = float(max(self.min_stop_loss, min(stop_distance, self.max_stop_loss)))

            stop_price = entry_price - (atr_mult * atr) if position_side == 'long' else entry_price + (atr_mult * atr)

            trailing_params = {
                'initial_stop': stop_price,
                'atr_multiplier': atr_mult,
                'update_threshold': atr * 0.5  # aktualizacja po ruchu 0.5 ATR
            }
            return {
                'stop_loss': float(stop_price),
                'type': 'dynamic_atr',
                'distance': float(stop_distance),
                'atr_multiplier': float(atr_mult),
                'trailing_params': trailing_params
            }
        except Exception as e:
            self.logger.error(f"Error calculating dynamic stops: {e}")
            fb = 0.02
            stop_price = entry_price * (1 - fb) if position_side == 'long' else entry_price * (1 + fb)
            return {'stop_loss': float(stop_price), 'type': 'fallback', 'distance': float(fb)}

    def update_trailing_stop(self, position: Dict, current_price: float) -> Dict:
        """Aktualizuje trailing stop dla pozycji long/short (tylko w kierunku zysku)"""
        try:
            trailing_params = position.get('trailing_params')
            if not trailing_params:
                return position

            current_stop = float(position.get('stop_loss', current_price))
            position_side = position.get('side', 'long')
            atr_multiplier = float(trailing_params.get('atr_multiplier', 2.0))
            update_threshold = float(trailing_params.get('update_threshold', 0.01))

            # Dla uproszczenia 'update_threshold' traktujemy jako wartość w jednostkach ceny (np. ATR/2)
            if position_side == 'long':
                # Przesuwamy stop tylko w górę
                new_stop = current_price - (atr_multiplier * update_threshold)
                if new_stop > current_stop:
                    position['stop_loss'] = float(new_stop)
                    position['trailing_updated'] = True
                    self.logger.info(f"Trailing stop LONG: {current_stop:.4f} -> {new_stop:.4f}")
            else:
                # short: przesuwamy stop tylko w dół
                new_stop = current_price + (atr_multiplier * update_threshold)
                if new_stop < current_stop:
                    position['stop_loss'] = float(new_stop)
                    position['trailing_updated'] = True
                    self.logger.info(f"Trailing stop SHORT: {current_stop:.4f} -> {new_stop:.4f}")

            return position
        except Exception as e:
            self.logger.error(f"Error updating trailing stop: {e}")
            return position

    # ---------- Awaryjne kontrole ryzyka ----------
    def emergency_risk_check(self, portfolio_value: float, initial_value: float,
                             current_positions: Dict) -> Dict:
        """Sprawdzenie awaryjne ryzyka - czy zatrzymać trading / zredukować ekspozycję"""
        try:
            current_drawdown = ((initial_value - portfolio_value) / initial_value) if initial_value > 0 else 0.0
            actions: List[str] = []
            alerts: List[str] = []

            if current_drawdown > self.emergency_stop_drawdown:
                actions.append("EMERGENCY_STOP")
                alerts.append(f"Emergency drawdown exceeded: {current_drawdown:.2%} > {self.emergency_stop_drawdown:.2%}")
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

            position_sizes = [pos.get('size', 0.0) for pos in current_positions.values()] if current_positions else []
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
                'emergency_stop_required': "EMERGENCY_STOP" in actions,
                'actions_required': actions,
                'risk_alerts': alerts,
                'current_drawdown': float(current_drawdown),
                'portfolio_heat': float(portfolio_heat),
                'max_position_size': float(max_position),
                'correlation_risk': float(correlation_risk)
            }
        except Exception as e:
            self.logger.error(f"Error in emergency risk check: {e}")
            self._emit_alert(
                f"Błąd awaryjnej kontroli ryzyka: {e}",
                severity=AlertSeverity.ERROR,
                context={"error": str(e)},
            )
            return {
                'emergency_stop_required': True,
                'actions_required': ["EMERGENCY_STOP"],
                'risk_alerts': [f"Error in risk calculation: {e}"],
                'current_drawdown': 0.5,
                'portfolio_heat': 1.0,
                'max_position_size': 1.0,
                'correlation_risk': 1.0
            }

    # ---------- Raport ryzyka ----------
    def generate_risk_report(self, portfolio_data: Dict, market_data: Dict) -> str:
        """Generuje tekstowy raport ryzyka"""
        try:
            rm = self.calculate_risk_metrics(portfolio_data, market_data)
            report = []
            report.append("=== RISK MANAGEMENT REPORT ===\n")
            report.append(f"🎯 Overall Risk Level: {rm.risk_level.name}\n")
            report.append(f"📊 Risk Score: {rm.overall_risk_score:.3f}\n\n")
            report.append("📈 Risk Metrics:\n")
            report.append(f"  • Value at Risk (95%): {rm.var_95:.3f}\n")
            report.append(f"  • Expected Shortfall: {rm.expected_shortfall:.3f}\n")
            report.append(f"  • Max Drawdown Risk: {rm.max_drawdown_risk:.3f}\n")
            report.append(f"  • Correlation Risk: {rm.correlation_risk:.3f}\n")
            report.append(f"  • Liquidity Risk: {rm.liquidity_risk:.3f}\n\n")

            if portfolio_data:
                num_positions = len(portfolio_data)
                total_exposure = float(sum(pos.get('size', 0.0) for pos in portfolio_data.values()))
                report.append("📋 Portfolio Overview:\n")
                report.append(f"  • Number of Positions: {num_positions}\n")
                report.append(f"  • Total Exposure: {total_exposure:.2%}\n")
                report.append(f"  • Portfolio Heat: {self._calculate_portfolio_heat(portfolio_data):.3f}\n\n")

            report.append("💡 Risk Recommendations:\n")
            if rm.risk_level.value >= 4:
                report.append("  ⚠️  REDUCE RISK: Consider closing some positions\n")
                report.append("  ⚠️  INCREASE DIVERSIFICATION: Add uncorrelated assets\n")
                report.append("  ⚠️  TIGHTEN STOPS: Reduce stop-loss distances\n")
            elif rm.risk_level.value <= 2:
                report.append("  ✅ CURRENT RISK ACCEPTABLE: Monitor and maintain\n")
                report.append("  🔍 OPPORTUNITY: Consider increasing position sizes\n")
            else:
                report.append("  ⚖️  BALANCED RISK: Current approach appropriate\n")
                report.append("  👀 MONITOR: Watch for regime changes\n")

            if len(portfolio_data) > 1:
                report.append("\n🔗 Correlation Analysis:\n")
                report.append(f"  • Average Correlation: {rm.correlation_risk:.3f}\n")
                if rm.correlation_risk > 0.7:
                    report.append("  ⚠️  HIGH CORRELATION: Positions may move together\n")
                elif rm.correlation_risk < 0.3:
                    report.append("  ✅ GOOD DIVERSIFICATION: Low correlation between positions\n")

            return "".join(report)
        except Exception as e:
            self.logger.error(f"Error generating risk report: {e}")
            return f"Error generating risk report: {e}"

    # ---------- Optymalizacja ryzyka ----------
    def optimize_portfolio_risk(self, current_portfolio: Dict, target_risk: float = None) -> Dict:
        """Proponuje redukcje/zwiększenia pozycji aby zbliżyć się do target_risk"""
        try:
            if not current_portfolio:
                return {'optimization_needed': False, 'suggestions': []}

            target_risk = float(target_risk or (self.max_portfolio_risk * 0.8))
            current_risk = float(self._calculate_portfolio_heat(current_portfolio))
            suggestions: List[Dict] = []

            if current_risk > target_risk:
                excess = current_risk - target_risk
                positions_by_risk = []
                for symbol, pos in current_portfolio.items():
                    prisk = float(pos.get('size', 0.0)) * float(pos.get('volatility', 0.2))
                    positions_by_risk.append((symbol, prisk, pos))
                positions_by_risk.sort(key=lambda x: x[1], reverse=True)

                risk_to_reduce = float(excess)
                for symbol, prisk, pos in positions_by_risk:
                    if risk_to_reduce <= 0:
                        break
                    max_reduction = prisk * 0.5
                    reduction = min(risk_to_reduce, max_reduction)
                    suggested_size = max(0.0, float(pos.get('size', 0.0)) - reduction / float(pos.get('volatility', 0.2)))
                    suggestions.append({
                        'action': 'reduce_position',
                        'symbol': symbol,
                        'current_size': float(pos.get('size', 0.0)),
                        'suggested_size': suggested_size,
                        'risk_reduction': float(reduction),
                        'reason': 'Portfolio risk too high'
                    })
                    risk_to_reduce -= reduction

            elif current_risk < target_risk * 0.5:
                available_risk = target_risk - current_risk
                for symbol, pos in current_portfolio.items():
                    if available_risk <= 0:
                        break
                    cur_size = float(pos.get('size', 0.0))
                    vol = float(pos.get('volatility', 0.2))
                    max_increase = float(available_risk / max(vol, 1e-9))
                    suggested_increase = min(max_increase, cur_size * 0.5)
                    if suggested_increase > 0.01:
                        suggestions.append({
                            'action': 'increase_position',
                            'symbol': symbol,
                            'current_size': cur_size,
                            'suggested_size': cur_size + suggested_increase,
                            'risk_increase': suggested_increase * vol,
                            'reason': 'Portfolio risk below target'
                        })
                        available_risk -= suggested_increase * vol

            return {
                'optimization_needed': len(suggestions) > 0,
                'current_risk': current_risk,
                'target_risk': target_risk,
                'suggestions': suggestions
            }
        except Exception as e:
            self.logger.error(f"Error optimizing portfolio risk: {e}")
            return {'optimization_needed': False, 'error': str(e)}

    # ---------- Aktualizacja stanu ----------
    def update_portfolio_state(self, portfolio_value: float, positions: Dict, market_returns: Dict):
        """Aktualizuje historię wartości portfela i rolling zwroty dla analizy korelacji"""
        try:
            self.portfolio_value_history.append(float(portfolio_value))
            if len(self.portfolio_value_history) > 252:
                self.portfolio_value_history = self.portfolio_value_history[-252:]

            self.current_positions = dict(positions) if positions else {}

            for symbol, rets in (market_returns or {}).items():
                if symbol not in self.historical_returns:
                    self.historical_returns[symbol] = rets
                else:
                    combined = pd.concat([self.historical_returns[symbol], rets]).dropna()
                    self.historical_returns[symbol] = combined.tail(252)
            self.logger.debug(f"Updated portfolio state: value={portfolio_value:.2f}, positions={len(self.current_positions)}")
        except Exception as e:
            self.logger.error(f"Error updating portfolio state: {e}")


# ---------- Fabryka / API zgodne z adapterem ----------
def create_risk_manager(config: Dict) -> RiskManagement:
    """Factory function do tworzenia risk managera (API wymagane przez adapter)"""
    try:
        return RiskManagement(config)
    except Exception as e:
        logger.exception("create_risk_manager failed")
        # Bezpieczny fallback z domyślnymi parametrami
        return RiskManagement({})


# ---------- Backtest prostych zasad ryzyka ----------
def backtest_risk_strategy(risk_manager: RiskManagement, historical_data: Dict,
                           signals: List[Dict]) -> Dict:
    """Backtest strategii zarządzania ryzykiem (size/limity)"""
    try:
        results = {
            'total_trades': 0,
            'risk_adjusted_trades': 0,
            'rejected_trades': 0,
            'max_portfolio_heat': 0.0,
            'risk_events': []
        }
        portfolio: Dict[str, Dict] = {}
        for signal in signals or []:
            symbol = signal.get('symbol', 'UNKNOWN')
            md = historical_data.get(symbol)
            if not isinstance(md, pd.DataFrame):
                continue

            sizing = risk_manager.calculate_position_size(symbol, signal, md, portfolio)
            can_trade, reason = risk_manager.check_position_limits(
                {'symbol': symbol, 'size': sizing.recommended_size, 'volatility': 0.2},
                portfolio
            )
            results['total_trades'] += 1
            if can_trade and sizing.recommended_size > 0.005:
                portfolio[symbol] = {
                    'size': sizing.recommended_size,
                    'entry_price': signal.get('price', float(md['close'].iloc[-1]) if 'close' in md else 0.0),
                    'volatility': 0.2
                }
                results['risk_adjusted_trades'] += 1
            else:
                results['rejected_trades'] += 1
                results['risk_events'].append({'symbol': symbol, 'reason': reason, 'suggested_size': sizing.recommended_size})

            heat = risk_manager._calculate_portfolio_heat(portfolio)
            results['max_portfolio_heat'] = max(results['max_portfolio_heat'], float(heat))

        results['risk_adjusted_ratio'] = (
            results['risk_adjusted_trades'] / results['total_trades'] if results['total_trades'] > 0 else 0.0
        )
        return results
    except Exception as e:
        logger.exception("backtest_risk_strategy failed")
        return {'error': str(e), 'total_trades': 0}


# ---------- Dźwignia dla target volatility ----------
def calculate_optimal_leverage(returns: pd.Series, target_volatility: float = 0.15) -> float:
    """Oblicza optymalną dźwignię dla target volatility (prosty model)"""
    try:
        if returns is None or len(returns) < 20:
            return 1.0
        port_vol = float(returns.std() * np.sqrt(252))
        if port_vol <= 0:
            return 1.0
        lev = float(target_volatility / port_vol)
        return float(max(0.1, min(lev, 3.0)))  # 0.1..3.0
    except Exception:
        return 1.0
