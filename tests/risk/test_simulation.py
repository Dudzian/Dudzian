from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from bot_core.risk.engine import ThresholdRiskEngine
from bot_core.risk.simulation import (
    DEFAULT_PROFILES,
    DEFAULT_SMOKE_SCENARIOS,
    load_orders_from_parquet,
    write_default_smoke_scenarios,
)


ROOT = Path(__file__).resolve().parents[2]


def _write_orders_parquet(path: str) -> None:
    base = datetime(2024, 1, 1, 12, 0, 0)
    orders = [
        {
            "profile": "conservative",
            "timestamp": base.isoformat(),
            "symbol": "BTCUSDT",
            "side": "buy",
            "price": 10000.0,
            "quantity": 0.01,
            "total_equity": 10000.0,
            "available_margin": 8000.0,
            "maintenance_margin": 100.0,
            "atr": 150.0,
            "stop_price": 9850.0,
            "position_value": 100.0,
            "pnl": 25.0,
        },
        {
            "profile": "conservative",
            "timestamp": (base + timedelta(minutes=1)).isoformat(),
            "symbol": "BTCUSDT",
            "side": "buy",
            "price": 10000.0,
            "quantity": 0.1,
            "total_equity": 10000.0,
            "available_margin": 8000.0,
            "maintenance_margin": 100.0,
            "atr": 150.0,
            "stop_price": 9850.0,
            "position_value": 1000.0,
            "pnl": -10.0,
        },
        {
            "profile": "balanced",
            "timestamp": (base + timedelta(minutes=2)).isoformat(),
            "symbol": "ETHUSDT",
            "side": "buy",
            "price": 100.0,
            "quantity": 1.0,
            "total_equity": 20000.0,
            "available_margin": 15000.0,
            "maintenance_margin": 100.0,
            "atr": 5.0,
            "stop_price": 98.0,
            "position_value": 100.0,
            "pnl": 0.0,
        },
        {
            "profile": "aggressive",
            "timestamp": (base + timedelta(minutes=3)).isoformat(),
            "symbol": "SOLUSDT",
            "side": "buy",
            "price": 150.0,
            "quantity": 0.5,
            "total_equity": 5000.0,
            "available_margin": 4000.0,
            "maintenance_margin": 100.0,
            "atr": 8.0,
            "stop_price": 134.0,
            "position_value": 75.0,
            "pnl": 12.0,
        },
        {
            "profile": "manual",
            "timestamp": (base + timedelta(minutes=4)).isoformat(),
            "symbol": "ADAUSDT",
            "side": "buy",
            "price": 50.0,
            "quantity": 2.0,
            "total_equity": 10000.0,
            "available_margin": 9000.0,
            "maintenance_margin": 150.0,
            "atr": 4.0,
            "stop_price": 45.0,
            "position_value": 100.0,
            "pnl": 5.0,
        },
    ]
    table = pa.Table.from_pylist(orders)
    pq.write_table(table, path)


def test_run_simulations_from_parquet(tmp_path):
    parquet_path = tmp_path / "orders.parquet"
    _write_orders_parquet(parquet_path)
    manual_overrides = {
        "max_positions": 4,
        "max_leverage": 4.0,
        "drawdown_limit": 0.2,
        "daily_loss_limit": 0.05,
        "max_position_pct": 0.2,
        "target_volatility": 0.3,
        "stop_loss_atr_multiple": 1.0,
    }

    suite = ThresholdRiskEngine.run_simulations_from_parquet(
        parquet_path,
        manual_overrides=manual_overrides,
    )

    assert [scenario.profile for scenario in suite.scenarios] == list(DEFAULT_PROFILES)

    conservative = suite.scenarios[0]
    assert conservative.total_orders == 2
    assert conservative.accepted_orders == 1
    assert conservative.rejected_orders == 1
    assert any("ekspozycji" in reason for reason in conservative.rejection_reasons)

    balanced = suite.scenarios[1]
    assert balanced.total_orders == 1
    assert balanced.accepted_orders == 0
    assert balanced.rejected_orders == 1
    assert any("stop loss" in reason.lower() for reason in balanced.rejection_reasons)

    aggressive = suite.scenarios[2]
    assert aggressive.accepted_orders == 1
    assert aggressive.rejected_orders == 0

    manual = suite.scenarios[3]
    assert manual.accepted_orders == 1
    assert manual.rejected_orders == 0

    mapping = suite.to_mapping()
    assert "generated_at" in mapping
    assert len(mapping["scenarios"]) == len(DEFAULT_PROFILES)

    report_dir = tmp_path / "reports"
    output = ThresholdRiskEngine.generate_simulation_reports(
        parquet_path,
        output_dir=report_dir,
        manual_overrides=manual_overrides,
    )

    json_path = report_dir / "risk_simulation_report.json"
    pdf_path = report_dir / "risk_simulation_report.pdf"

    assert json_path.exists()
    assert pdf_path.exists()
    assert output["json_path"] == str(json_path)
    assert output["pdf_path"] == str(pdf_path)
    assert pdf_path.read_bytes().startswith(b"%PDF")


def test_write_default_smoke_scenarios(tmp_path):
    parquet_path = tmp_path / "default_smoke.parquet"
    generated = write_default_smoke_scenarios(parquet_path)
    assert generated.exists()

    orders = load_orders_from_parquet(generated)
    assert len(orders) == len(DEFAULT_SMOKE_SCENARIOS)
    assert [order.profile for order in orders] == [
        record["profile"] for record in DEFAULT_SMOKE_SCENARIOS
    ]
