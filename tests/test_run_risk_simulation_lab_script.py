from __future__ import annotations

import json
import logging
from pathlib import Path
import sys

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_risk_simulation_lab import (
    DEFAULT_CONFIG_PATH,
    DEFAULT_OUTPUT_DIR,
    _build_parser,
    main as run_main,
)


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


@pytest.fixture()
def stage5_config(tmp_path: Path) -> tuple[Path, Path]:
    config_path = tmp_path / "core.yaml"
    reports_dir = tmp_path / "reports"
    reports_dir.mkdir()
    tco_path = reports_dir / "tco_report.json"
    tco_payload = {
        "generated_at": "2024-05-01T12:00:00Z",
        "total": {"cost_bps": 6.5, "monthly_total": 1250.0},
        "metadata": {"events_count": 12},
    }
    tco_path.write_text(json.dumps(tco_payload), encoding="utf-8")
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
        "decision_engine": {
            "tco": {
                "reports": [f"reports/{tco_path.name}"],
                "require_at_startup": False,
            }
        },
    }
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    return config_path, tco_path


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


def test_cli_supports_stage2_profile_all(tmp_path: Path, simple_config: Path) -> None:
    output_dir = tmp_path / "stage2"
    exit_code = run_main(
        [
            "--config",
            str(simple_config),
            "--output-dir",
            str(output_dir),
            "--synthetic-fallback",
            "--profile",
            "all",
        ]
    )
    assert exit_code == 0
    payload = json.loads((output_dir / "risk_simulation_report.json").read_text(encoding="utf-8"))
    profiles = {entry["profile"] for entry in payload["profiles"]}
    assert profiles == {"conservative"}


def test_cli_stage5_latency_tco(tmp_path: Path, stage5_config: tuple[Path, Path]) -> None:
    config_path, tco_path = stage5_config
    output_dir = tmp_path / "stage5"
    exit_code = run_main(
        [
            "--config",
            str(config_path),
            "--output-dir",
            str(output_dir),
            "--synthetic-fallback",
            "--scenario",
            "latency_spike",
            "--include-tco",
        ]
    )
    assert exit_code == 0
    payload = json.loads((output_dir / "risk_simulation_report.json").read_text(encoding="utf-8"))
    stress_names = {test["name"] for test in payload["profiles"][0]["stress_tests"]}
    assert stress_names == {"latency_spike"}
    tco_section = payload.get("tco")
    assert tco_section is not None
    assert tco_section.get("status") == "ok"
    reports = tco_section.get("reports")
    assert isinstance(reports, list) and reports
    first_entry = reports[0]
    assert first_entry["status"] == "ok"
    assert Path(first_entry["path"]).resolve() == tco_path.resolve()
    summary = first_entry.get("summary", {})
    assert summary.get("cost_bps") == 6.5
    assert summary.get("monthly_total") == 1250.0


def test_parser_stage2_defaults() -> None:
    parser = _build_parser()
    args = parser.parse_args(["--profile", "all"])
    assert args.config == DEFAULT_CONFIG_PATH
    assert args.output_dir == DEFAULT_OUTPUT_DIR
    assert args.profile == ["all"]


def test_parser_stage5_defaults() -> None:
    parser = _build_parser()
    args = parser.parse_args(["--scenario", "latency_spike", "--include-tco"])
    assert args.config == DEFAULT_CONFIG_PATH
    assert args.output_dir == DEFAULT_OUTPUT_DIR
    assert args.scenario == ["latency_spike"]
    assert args.include_tco is True
