from __future__ import annotations

from pathlib import Path
import sys

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bot_core.risk.profiles.aggressive import AggressiveProfile
from bot_core.risk.profiles.balanced import BalancedProfile
from bot_core.risk.profiles.conservative import ConservativeProfile
from bot_core.risk.profiles.manual import ManualProfile
from bot_core.risk.simulation import (
    Candle,
    RiskSimulationRunner,
    SimulationSettings,
    run_simulations_from_config,
)


@pytest.fixture()
def bullish_candles() -> list[Candle]:
    candles: list[Candle] = []
    price = 10_000.0
    timestamp = 1_640_995_200_000  # 2022-01-01 UTC
    for _ in range(96):
        open_price = price
        close_price = price * 1.002
        high = close_price * 1.001
        low = open_price * 0.999
        candles.append(
            Candle(
                timestamp_ms=timestamp,
                open=open_price,
                high=high,
                low=low,
                close=close_price,
                volume=1_200.0,
            )
        )
        price = close_price
        timestamp += 3_600_000
    return candles


def test_simulation_runner_generates_results_without_breaches(bullish_candles: list[Candle]) -> None:
    profiles = [
        ConservativeProfile(),
        BalancedProfile(),
        AggressiveProfile(),
        ManualProfile(
            name="manual_lab",
            max_positions=5,
            max_leverage=1.0,
            drawdown_limit=0.5,
            daily_loss_limit=0.25,
            max_position_pct=0.0,
            target_volatility=0.2,
            stop_loss_atr_multiple=1.0,
        ),
    ]
    runner = RiskSimulationRunner(
        profiles=profiles,
        candles_by_symbol={"BTCUSDT": bullish_candles},
        settings=SimulationSettings(base_equity=50_000.0, max_bars=90),
    )
    report = runner.run()
    assert len(report.profiles) == len(profiles)
    for profile_result in report.profiles:
        assert profile_result.max_drawdown_pct >= 0.0
        for stress in profile_result.stress_tests:
            assert stress.name in {"flash_crash", "dry_liquidity", "latency_spike"}
        if profile_result.profile == "manual_lab":
            assert profile_result.final_equity == pytest.approx(50_000.0)
            assert profile_result.breaches == ()
        else:
            assert profile_result.final_equity > 50_000.0
            assert profile_result.breaches == ()


def test_flash_crash_detects_breach(bullish_candles: list[Candle]) -> None:
    crash_candles = list(bullish_candles)
    last = crash_candles[-1]
    crash_candles[-1] = Candle(
        timestamp_ms=last.timestamp_ms,
        open=last.open,
        high=last.high,
        low=last.close * 0.6,
        close=last.close * 0.65,
        volume=last.volume,
    )
    runner = RiskSimulationRunner(
        profiles=[ConservativeProfile()],
        candles_by_symbol={"BTCUSDT": crash_candles},
        settings=SimulationSettings(base_equity=25_000.0, max_bars=90),
    )
    report = runner.run()
    profile_result = report.profiles[0]
    flash_results = [item for item in profile_result.stress_tests if item.name == "flash_crash"]
    assert flash_results, "flash_crash stress test should be present"
    assert flash_results[0].status == "failed"
    assert "drawdown" in (flash_results[0].notes or "")


def test_run_simulation_from_config_with_synthetic_fallback(tmp_path: Path) -> None:
    config_path = tmp_path / "core.yaml"
    data_dir = tmp_path / "data"
    config = {
        "risk_profiles": {
            "conservative": {
                "max_daily_loss_pct": 0.02,
                "max_position_pct": 0.05,
                "target_volatility": 0.1,
                "max_leverage": 3.0,
                "stop_loss_atr_multiple": 1.0,
                "max_open_positions": 3,
                "hard_drawdown_pct": 0.1,
                "data_quality": {"max_gap_minutes": 1440, "min_ok_ratio": 0.9},
                "strategy_allocations": {},
                "instrument_buckets": ["core"],
            }
        },
        "environments": {
            "paper": {
                "exchange": "binance",
                "environment": "paper",
                "keychain_key": "paper",
                "data_cache_path": str(data_dir),
                "risk_profile": "conservative",
                "alert_channels": [],
            }
        },
    }
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    report = run_simulations_from_config(
        config_path=config_path,
        dataset_root=None,
        namespace="binance_spot",
        symbols=["BTCUSDT"],
        interval="1h",
        settings=SimulationSettings(base_equity=40_000.0, max_bars=48),
        synthetic_fallback=True,
    )
    assert report.synthetic_data is True
    assert report.profiles[0].sample_size > 0
    assert (tmp_path / "data").exists() is False
