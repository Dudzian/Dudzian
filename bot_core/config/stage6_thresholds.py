"""Stałe progów Stage6 oraz pomocnicza walidacja konfiguracji."""
from __future__ import annotations

from typing import Final

EXPECTED_MARKET_INTEL: Final[dict[str, object]] = {
    "default_weight": 1.15,
    "required_symbols": ("BTCUSDT", "ETHUSDT", "SOLUSDT"),
}

EXPECTED_PORTFOLIO_GOVERNOR: Final[dict[str, float]] = {
    "rebalance_interval_minutes": 20.0,
    "smoothing": 0.55,
    "default_baseline_weight": 0.28,
    "default_min_weight": 0.08,
    "default_max_weight": 0.50,
    "min_score_threshold": 0.08,
    "default_cost_bps": 5.0,
}

EXPECTED_PORTFOLIO_SCORING: Final[dict[str, float]] = {
    "alpha": 0.9,
    "cost": 1.3,
    "slo": 1.2,
    "risk": 0.8,
}

EXPECTED_STRATEGIES: Final[dict[str, tuple[float, float, float]]] = {
    "core_daily_trend": (0.22, 0.60, 1.25),
    "core_mean_reversion": (0.10, 0.45, 1.45),
    "core_volatility_target": (0.12, 0.40, 1.15),
    "core_cross_exchange": (0.05, 0.28, 1.30),
}

EXPECTED_STRESS_THRESHOLDS: Final[dict[str, float]] = {
    "max_liquidity_loss_pct": 0.6,
    "max_spread_increase_bps": 45.0,
    "max_volatility_increase_pct": 0.80,
    "max_sentiment_drawdown": 0.50,
    "max_funding_change_bps": 25.0,
    "max_latency_spike_ms": 150.0,
    "max_blackout_minutes": 40.0,
    "max_dispersion_bps": 60.0,
}

EXPECTED_BLACKOUT_OVERRIDES: Final[dict[str, float]] = {
    "max_latency_spike_ms": 180.0,
    "max_blackout_minutes": 55.0,
}


def collect_stage6_threshold_differences(config: object) -> list[str]:
    """Porównuje konfigurację Stage6 z progami warsztatowymi i zwraca listę różnic."""

    differences: list[str] = []

    market_intel = getattr(config, "market_intel", None)
    if market_intel is None:
        differences.append("Brak sekcji market_intel")
    else:
        expected_weight = EXPECTED_MARKET_INTEL["default_weight"]
        if abs(market_intel.default_weight - float(expected_weight)) > 1e-9:
            differences.append(
                f"market_intel.default_weight={market_intel.default_weight} (oczekiwano {expected_weight})"
            )

        required_symbols = tuple(market_intel.required_symbols or ())
        expected_symbols = EXPECTED_MARKET_INTEL["required_symbols"]
        if required_symbols != expected_symbols:
            differences.append(
                "market_intel.required_symbols="
                f"{required_symbols} (oczekiwano {expected_symbols})"
            )

    governor = getattr(config, "portfolio_governor", None)
    if governor is None:
        differences.append("Brak sekcji portfolio_governor")
    else:
        for field_name, expected in EXPECTED_PORTFOLIO_GOVERNOR.items():
            actual = getattr(governor, field_name, None)
            if actual is None or abs(actual - expected) > 1e-9:
                differences.append(
                    f"portfolio_governor.{field_name}={actual} (oczekiwano {expected})"
                )

        scoring = getattr(governor, "scoring", None)
        if scoring is None:
            differences.append("Brak sekcji portfolio_governor.scoring")
        else:
            for field_name, expected in EXPECTED_PORTFOLIO_SCORING.items():
                actual = getattr(scoring, field_name, None)
                if actual is None or abs(actual - expected) > 1e-9:
                    differences.append(
                        "portfolio_governor.scoring."
                        f"{field_name}={actual} (oczekiwano {expected})"
                    )

        strategies = getattr(governor, "strategies", {})
        for strategy_name, expected_values in EXPECTED_STRATEGIES.items():
            strategy = strategies.get(strategy_name)
            if strategy is None:
                differences.append(
                    f"Brak strategii portfolio_governor.strategies['{strategy_name}']"
                )
                continue

            expected_min, expected_max, expected_multiplier = expected_values
            if abs(strategy.min_weight - expected_min) > 1e-9:
                differences.append(
                    f"{strategy_name}.min_weight={strategy.min_weight} (oczekiwano {expected_min})"
                )
            if abs(strategy.max_weight - expected_max) > 1e-9:
                differences.append(
                    f"{strategy_name}.max_weight={strategy.max_weight} (oczekiwano {expected_max})"
                )
            if abs(strategy.max_signal_factor - expected_multiplier) > 1e-9:
                differences.append(
                    f"{strategy_name}.max_signal_factor={strategy.max_signal_factor} (oczekiwano {expected_multiplier})"
                )

    stress_lab = getattr(config, "stress_lab", None)
    if stress_lab is None:
        differences.append("Brak sekcji stress_lab")
    else:
        thresholds = getattr(stress_lab, "thresholds", None)
        if thresholds is None:
            differences.append("Brak sekcji stress_lab.thresholds")
        else:
            for field_name, expected in EXPECTED_STRESS_THRESHOLDS.items():
                actual = getattr(thresholds, field_name, None)
                if actual is None or abs(actual - expected) > 1e-9:
                    differences.append(
                        f"stress_lab.thresholds.{field_name}={actual} (oczekiwano {expected})"
                    )

        blackout = None
        for scenario in getattr(stress_lab, "scenarios", ()):  # pragma: no branch
            if getattr(scenario, "name", "") == "exchange_blackout_and_latency":
                blackout = scenario
                break

        if blackout is None:
            differences.append(
                "Brak scenariusza stress_lab.scenarios exchange_blackout_and_latency"
            )
        else:
            overrides = getattr(blackout, "threshold_overrides", None)
            if overrides is None:
                differences.append("Brak override'ów dla blackout scenario")
            else:
                for field_name, expected in EXPECTED_BLACKOUT_OVERRIDES.items():
                    actual = getattr(overrides, field_name, None)
                    if actual is None or abs(actual - expected) > 1e-9:
                        differences.append(
                            "stress_lab.scenarios['exchange_blackout_and_latency'].threshold_overrides."
                            f"{field_name}={actual} (oczekiwano {expected})"
                        )

    return differences
