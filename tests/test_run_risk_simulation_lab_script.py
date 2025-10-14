from __future__ import annotations

import json
from pathlib import Path
import sys

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_risk_simulation_lab import main as run_main


@pytest.fixture()
def simple_config(tmp_path: Path) -> Path:
    config_path = tmp_path / "core.yaml"
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
                "data_cache_path": str(tmp_path / "data"),
                "risk_profile": "conservative",
                "alert_channels": [],
            }
        },
    }
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    return config_path


def test_run_risk_simulation_lab_cli(tmp_path: Path, simple_config: Path, capsys: pytest.CaptureFixture[str]) -> None:
    output_dir = tmp_path / "output"
    exit_code = run_main(
        [
            "--config",
            str(simple_config),
            "--output-dir",
            str(output_dir),
            "--synthetic-fallback",
            "--print-summary",
        ]
    )
    assert exit_code == 0
    json_path = output_dir / "risk_simulation_report.json"
    pdf_path = output_dir / "risk_simulation_report.pdf"
    assert json_path.exists()
    assert pdf_path.exists()
    captured = capsys.readouterr()
    summary = json.loads(captured.out.strip())
    assert "breach_count" in summary
    assert summary["stress_failures"] == 0
    assert pdf_path.read_bytes().startswith(b"%PDF-1.4")


def test_cli_fail_on_breach(tmp_path: Path, simple_config: Path) -> None:
    # sztucznie wymusza naruszenie poprzez brak fallbacku i brak danych
    exit_code = run_main(
        [
            "--config",
            str(simple_config),
            "--output-dir",
            str(tmp_path / "output"),
            "--fail-on-breach",
        ]
    )
    # brak danych powoduje status błędu
    assert exit_code in {2, 3}
