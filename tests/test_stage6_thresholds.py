"""Assercje konfiguracji Stage6 po warsztacie operatorów z 2024-06-07."""

from __future__ import annotations

import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import pytest

from bot_core.config.loader import load_core_config
from bot_core.config.stage6_thresholds import (
    EXPECTED_BLACKOUT_OVERRIDES,
    EXPECTED_MARKET_INTEL,
    EXPECTED_PORTFOLIO_GOVERNOR,
    EXPECTED_PORTFOLIO_SCORING,
    EXPECTED_STRESS_THRESHOLDS,
    EXPECTED_STRATEGIES,
    collect_stage6_threshold_differences,
)


@pytest.mark.parametrize("config_path", [Path("config/core.yaml")])
def test_stage6_thresholds_aligned_with_workshop(config_path: Path) -> None:
    """Zapewnia, że kluczowe progi Stage6 pozostają zgodne z ustaleniami warsztatowymi."""
    core_config = load_core_config(str(config_path))

    market_intel = getattr(core_config, "market_intel", None)
    assert market_intel is not None, "market_intel section missing"
    assert market_intel.default_weight == pytest.approx(
        EXPECTED_MARKET_INTEL["default_weight"]
    )
    assert tuple(market_intel.required_symbols or ()) == EXPECTED_MARKET_INTEL[
        "required_symbols"
    ]

    governor = getattr(core_config, "portfolio_governor", None)
    assert governor is not None, "portfolio_governor section missing"
    for field_name, expected in EXPECTED_PORTFOLIO_GOVERNOR.items():
        assert getattr(governor, field_name) == pytest.approx(expected)

    for field_name, expected in EXPECTED_PORTFOLIO_SCORING.items():
        assert getattr(governor.scoring, field_name) == pytest.approx(expected)

    strategies = governor.strategies
    assert "core_daily_trend" in strategies
    for strategy_name, expected_values in EXPECTED_STRATEGIES.items():
        assert strategy_name in strategies
        expected_min, expected_max, expected_multiplier = expected_values
        strategy = strategies[strategy_name]
        assert strategy.min_weight == pytest.approx(expected_min)
        assert strategy.max_weight == pytest.approx(expected_max)
        assert strategy.max_signal_factor == pytest.approx(expected_multiplier)

    stress_lab = getattr(core_config, "stress_lab", None)
    assert stress_lab is not None, "stress_lab section missing"
    thresholds = stress_lab.thresholds
    for field_name, expected in EXPECTED_STRESS_THRESHOLDS.items():
        assert getattr(thresholds, field_name) == pytest.approx(expected)

    blackout = None
    for scenario in stress_lab.scenarios:
        if getattr(scenario, "name", "") == "exchange_blackout_and_latency":
            blackout = scenario
            break
    assert blackout is not None, "exchange_blackout_and_latency scenario missing"
    overrides = getattr(blackout, "threshold_overrides", None)
    assert overrides is not None, "blackout scenario lacks overrides"
    for field_name, expected in EXPECTED_BLACKOUT_OVERRIDES.items():
        assert getattr(overrides, field_name) == pytest.approx(expected)

    # Obrona przed regresją w funkcji różnic używanej przez CLI.
    assert collect_stage6_threshold_differences(core_config) == []


def test_stage6_threshold_verifier_cli() -> None:
    """Uruchamia skrypt CLI i oczekuje sukcesu bez rozbieżności."""
    result = subprocess.run(
        [sys.executable, "scripts/verify_stage6_thresholds.py", "--config", "config/core.yaml"],
        check=False,
        capture_output=True,
        text=True,
    )

    stdout = result.stdout.strip()
    stderr = result.stderr.strip()

    assert result.returncode == 0, f"Skrypt zwrócił {result.returncode}:\nSTDOUT: {stdout}\nSTDERR: {stderr}"
    assert "✅" in stdout


def test_stage6_threshold_verifier_cli_json_report(tmp_path: Path) -> None:
    """Generuje raport JSON zgodny z oczekiwaniami audytu."""
    report_path = tmp_path / "stage6_thresholds.json"
    result = subprocess.run(
        [
            sys.executable,
            "scripts/verify_stage6_thresholds.py",
            "--config",
            "config/core.yaml",
            "--json-report",
            str(report_path),
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    stdout = result.stdout.strip()
    stderr = result.stderr.strip()

    assert result.returncode == 0, f"Skrypt zwrócił {result.returncode}:\nSTDOUT: {stdout}\nSTDERR: {stderr}"
    assert report_path.exists(), f"Brak raportu JSON: {report_path}"

    payload = json.loads(report_path.read_text(encoding="utf-8"))
    assert payload["status"] == "ok"
    assert payload["differences"] == []
    assert payload["config_path"].endswith("config/core.yaml")
    # ISO 8601 z suffiksem Z
    datetime.fromisoformat(payload["timestamp"].replace("Z", "+00:00"))


def test_stage6_threshold_verifier_cli_json_report_mismatch(tmp_path: Path) -> None:
    """Raport JSON zawiera rozbieżności, gdy konfiguracja odbiega od progów."""
    config_copy = tmp_path / "core.yaml"
    original = Path("config/core.yaml").read_text(encoding="utf-8")
    config_copy.write_text(original.replace("default_weight: 1.15", "default_weight: 1.42"), encoding="utf-8")

    observability_dir = tmp_path / "observability"
    observability_dir.mkdir(parents=True, exist_ok=True)
    (observability_dir / "slo.yml").write_text(
        Path("config/observability/slo.yml").read_text(encoding="utf-8"),
        encoding="utf-8",
    )

    report_path = tmp_path / "audit.json"
    result = subprocess.run(
        [
            sys.executable,
            "scripts/verify_stage6_thresholds.py",
            "--config",
            str(config_copy),
            "--json-report",
            str(report_path),
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    stdout = result.stdout.strip()
    stderr = result.stderr.strip()

    assert result.returncode == 1, f"Skrypt zwrócił {result.returncode}:\nSTDOUT: {stdout}\nSTDERR: {stderr}"
    assert "❌" in stdout
    assert report_path.exists()

    payload = json.loads(report_path.read_text(encoding="utf-8"))
    assert payload["status"] == "mismatch"
    assert payload["differences"]
    assert any("market_intel.default_weight" in diff for diff in payload["differences"])
