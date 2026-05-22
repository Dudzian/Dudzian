from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "scripts/controller_mock_preview.py"
SAFE_CONFIG = REPO_ROOT / "config/e2e/demo_paper.yml"


def _run(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(SCRIPT), *args],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )


def _mutated_config(tmp_path: Path, dotted_path: str, value: object) -> Path:
    payload = yaml.safe_load(SAFE_CONFIG.read_text(encoding="utf-8"))
    current = payload
    segments = dotted_path.split(".")
    for segment in segments[:-1]:
        current = current[segment]
    current[segments[-1]] = value
    path = tmp_path / f"mutated_{dotted_path.replace('.', '_')}.yml"
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return path


def test_controller_mock_preview_safe_demo_json() -> None:
    result = _run("--mode", "demo", "--config", str(SAFE_CONFIG), "--max-signals", "1", "--json")
    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert payload["status"] == "ok"
    assert payload["mode"] == "demo"
    assert payload["controller_backed_preview_started"] is True
    assert payload["synthetic_signals_processed"] >= 1
    assert payload["exchange_io"] == "disabled"
    assert payload["order_execution"] in {"mocked", "disabled", "mocked_or_disabled"}
    assert payload["api_keys_required"] is False
    assert payload["live_mode_allowed"] is False
    assert payload["real_orders_submitted"] is False
    assert payload["runtime_loop_started"] is False
    assert payload["issues"] == []


def test_controller_mock_preview_live_mode_blocked() -> None:
    result = _run("--mode", "live", "--config", str(SAFE_CONFIG), "--max-signals", "1", "--json")
    assert result.returncode == 2
    payload = json.loads(result.stdout)
    assert payload["status"] == "blocked"
    assert payload["reason"] == "controller_mock_preview_forbids_live_mode"


def test_controller_mock_preview_unsafe_config_blocked(tmp_path: Path) -> None:
    cases = (
        ("execution.live.enabled", True),
        ("trading.enable_live_mode", True),
        ("trading.enable_paper_mode", False),
        ("execution.force_paper_when_offline", False),
        ("execution.default_mode", "live"),
    )
    for dotted_path, value in cases:
        cfg = _mutated_config(tmp_path, dotted_path, value)
        result = _run("--mode", "demo", "--config", str(cfg), "--max-signals", "1", "--json")
        assert result.returncode == 2
        payload = json.loads(result.stdout)
        assert payload["status"] == "blocked"
        assert f"unsafe_config:{dotted_path}" in payload["issues"]


def test_controller_mock_preview_blocks_too_many_signals() -> None:
    result = _run("--mode", "demo", "--config", str(SAFE_CONFIG), "--max-signals", "999", "--json")
    assert result.returncode == 2
    payload = json.loads(result.stdout)
    assert payload["reason"] == "max_signals_out_of_bounds"


def test_controller_mock_preview_no_api_keys_required(monkeypatch) -> None:
    for key in ("API_KEY", "API_SECRET", "BINANCE_API_KEY", "BINANCE_API_SECRET"):
        monkeypatch.delenv(key, raising=False)
    result = _run("--mode", "demo", "--config", str(SAFE_CONFIG), "--max-signals", "1", "--json")
    assert result.returncode == 0
    payload = json.loads(result.stdout)
    assert payload["api_keys_required"] is False


def test_controller_mock_preview_output_is_json_parseable() -> None:
    result = _run("--mode", "demo", "--config", str(SAFE_CONFIG), "--max-signals", "1", "--json")
    assert result.returncode == 0
    payload = json.loads(result.stdout)
    assert payload["safety_contract_version"] == "controller_mock_preview.v1"


def test_controller_mock_preview_invalid_side_blocked() -> None:
    result = _run(
        "--mode",
        "demo",
        "--config",
        str(SAFE_CONFIG),
        "--max-signals",
        "1",
        "--side",
        "HOLD",
        "--json",
    )
    assert result.returncode == 2
    assert "invalid choice" in result.stderr


def test_controller_mock_preview_quantity_zero_blocked() -> None:
    result = _run(
        "--mode",
        "demo",
        "--config",
        str(SAFE_CONFIG),
        "--max-signals",
        "1",
        "--quantity",
        "0",
        "--json",
    )
    assert result.returncode == 2
    payload = json.loads(result.stdout)
    assert payload["reason"] == "controller_mock_preview_invalid_quantity"
    assert "invalid_quantity" in payload["issues"]


def test_controller_mock_preview_quantity_negative_blocked() -> None:
    result = _run(
        "--mode",
        "demo",
        "--config",
        str(SAFE_CONFIG),
        "--max-signals",
        "1",
        "--quantity",
        "-1",
        "--json",
    )
    assert result.returncode == 2
    payload = json.loads(result.stdout)
    assert payload["reason"] == "controller_mock_preview_invalid_quantity"
    assert "invalid_quantity" in payload["issues"]


def test_controller_mock_preview_blank_symbol_blocked() -> None:
    result = _run(
        "--mode",
        "demo",
        "--config",
        str(SAFE_CONFIG),
        "--max-signals",
        "1",
        "--symbol",
        "   ",
        "--json",
    )
    assert result.returncode == 2
    payload = json.loads(result.stdout)
    assert payload["reason"] == "controller_mock_preview_invalid_symbol"
    assert "invalid_symbol" in payload["issues"]
