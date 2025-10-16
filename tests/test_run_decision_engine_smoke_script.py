from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "run_decision_engine_smoke.py"
CONFIG_PATH = REPO_ROOT / "config" / "core.yaml"


def test_run_decision_engine_smoke(tmp_path: Path) -> None:
    risk_snapshot = {
        "balanced": {
            "start_of_day_equity": 100_000.0,
            "daily_realized_pnl": 0.0,
            "peak_equity": 105_000.0,
            "last_equity": 102_000.0,
            "positions": {
                "BTCUSDT": {"side": "long", "notional": 15_000.0},
                "ETHUSDT": {"side": "long", "notional": 8_000.0},
            },
        }
    }
    candidates = [
        {
            "strategy": "mean_reversion_alpha",
            "action": "enter",
            "risk_profile": "balanced",
            "symbol": "ADAUSDT",
            "notional": 4_000.0,
            "expected_return_bps": 16.0,
            "expected_probability": 0.9,
            "latency_ms": 120.0,
        }
    ]
    tco_report = {
        "strategies": {
            "mean_reversion_alpha": {
                "profiles": {"balanced": {"cost_bps": 4.0}},
                "total": {"cost_bps": 4.5},
            }
        },
        "total": {"cost_bps": 5.2},
    }

    risk_path = tmp_path / "risk.json"
    risk_path.write_text(json.dumps(risk_snapshot), encoding="utf-8")
    candidates_path = tmp_path / "candidates.json"
    candidates_path.write_text(json.dumps(candidates), encoding="utf-8")
    tco_path = tmp_path / "tco.json"
    tco_path.write_text(json.dumps(tco_report), encoding="utf-8")
    output_path = tmp_path / "decision_smoke.json"

    completed = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--config",
            str(CONFIG_PATH),
            "--risk-snapshot",
            str(risk_path),
            "--candidates",
            str(candidates_path),
            "--tco-report",
            str(tco_path),
            "--output",
            str(output_path),
        ],
        check=False,
        cwd=str(REPO_ROOT),
    )
    assert completed.returncode == 0

    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload["accepted"] >= 1
    assert payload["rejected"] == 0
    assert payload["stress_failures"] == 0
