"""Próbki danych dla symulacji portfolio_stress Stage6."""
from __future__ import annotations

from pathlib import Path

from bot_core.config.models import (
    PortfolioStressAssetShockConfig,
    PortfolioStressFactorShockConfig,
    PortfolioStressScenarioConfig,
)
from bot_core.risk.portfolio_stress import load_portfolio_stress_baseline, PortfolioStressBaseline

_SAMPLE_BASELINE_PATH = Path(__file__).with_name("portfolio_stress_baseline.json")


def load_sample_baseline() -> PortfolioStressBaseline:
    """Wczytuje przygotowany baseline portfela Stage6."""

    return load_portfolio_stress_baseline(_SAMPLE_BASELINE_PATH)


def build_sample_scenarios() -> tuple[PortfolioStressScenarioConfig, ...]:
    """Buduje zestaw scenariuszy zgodny z config/core.yaml."""

    return (
        PortfolioStressScenarioConfig(
            name="usd_liquidity_crunch",
            title="Kryzys płynności USD",
            description="Ucieczka płynności dolarowej i wzrost zmienności na parach bazowych.",
            horizon_days=5.0,
            probability=0.15,
            factors=(
                PortfolioStressFactorShockConfig(
                    factor="usd_liquidity",
                    return_pct=-0.18,
                    liquidity_haircut_pct=0.35,
                    notes="Zwinięcie płynności on/off-ramp USD.",
                ),
                PortfolioStressFactorShockConfig(
                    factor="funding_rates",
                    return_pct=-0.05,
                    notes="Silne ujemne fundingi na perpetuals.",
                ),
            ),
            assets=(
                PortfolioStressAssetShockConfig(
                    symbol="btc_usdt",
                    return_pct=-0.22,
                    liquidity_haircut_pct=0.25,
                ),
                PortfolioStressAssetShockConfig(
                    symbol="eth_usdt",
                    return_pct=-0.2,
                    liquidity_haircut_pct=0.2,
                ),
            ),
            tags=("liquidity", "systemic"),
            metadata={"stress_score": 0.82},
        ),
        PortfolioStressScenarioConfig(
            name="alt_season_boom_and_bust",
            title="Hossa/załamanie altcoinów",
            description="Dwufazowy scenariusz dynamicznych przepływów do altów i gwałtownej korekty.",
            horizon_days=10.0,
            probability=0.25,
            factors=(
                PortfolioStressFactorShockConfig(
                    factor="alt_beta",
                    return_pct=0.12,
                    notes="Wczesna faza rajdu altcoinów.",
                ),
                PortfolioStressFactorShockConfig(
                    factor="alt_liquidity",
                    return_pct=-0.16,
                    liquidity_haircut_pct=0.3,
                    notes="Gwałtowna realizacja zysków i drenowanie płynności.",
                ),
            ),
            assets=(
                PortfolioStressAssetShockConfig(
                    symbol="sol_usdt",
                    return_pct=-0.24,
                    liquidity_haircut_pct=0.28,
                ),
                PortfolioStressAssetShockConfig(
                    symbol="ada_usdt",
                    return_pct=-0.18,
                ),
                PortfolioStressAssetShockConfig(
                    symbol="dot_usdt",
                    return_pct=-0.17,
                ),
            ),
            tags=("altcoins", "dispersion"),
            metadata={"phases": 2, "comment": "Rotacja altów -> korekta"},
        ),
        PortfolioStressScenarioConfig(
            name="rates_regime_shift",
            title="Szok stóp procentowych",
            description="Skokowe podniesienie stóp i odpływ kapitału z ryzykownych aktywów.",
            horizon_days=7.0,
            probability=0.2,
            factors=(
                PortfolioStressFactorShockConfig(
                    factor="global_rates",
                    return_pct=-0.1,
                    notes="Dyskontowanie agresywnej ścieżki stóp.",
                ),
                PortfolioStressFactorShockConfig(
                    factor="usd_liquidity",
                    return_pct=-0.08,
                    liquidity_haircut_pct=0.18,
                ),
                PortfolioStressFactorShockConfig(
                    factor="vol_targeting",
                    return_pct=-0.05,
                ),
            ),
            assets=(
                PortfolioStressAssetShockConfig(
                    symbol="btc_usdt",
                    return_pct=-0.15,
                ),
                PortfolioStressAssetShockConfig(
                    symbol="eth_usdt",
                    return_pct=-0.17,
                ),
                PortfolioStressAssetShockConfig(
                    symbol="bnb_usdt",
                    return_pct=-0.14,
                ),
            ),
            cash_return_pct=0.01,
            tags=("macro", "rates"),
            metadata={"scenario_source": "fomc_minutes"},
        ),
    )


__all__ = ["load_sample_baseline", "build_sample_scenarios"]
