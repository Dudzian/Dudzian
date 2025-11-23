from __future__ import annotations

import json
from types import SimpleNamespace
from pathlib import Path

from scripts import run_ai_governor_cycle


def test_cli_cycle_uses_runner_defaults(tmp_path: Path, monkeypatch) -> None:
    recorded = SimpleNamespace(decision=None)

    def _serialize(decisions):
        recorded.decision = decisions[0]
        return json.dumps([{"mode": decisions[0].mode}])

    monkeypatch.setattr(run_ai_governor_cycle, "_serialize", _serialize)

    exit_code = run_ai_governor_cycle.main(["--mode", "scalping", "--limit", "1"])

    assert exit_code == 0
    assert recorded.decision.mode == "scalping"


def test_cli_cycle_reads_custom_snapshot(tmp_path: Path) -> None:
    snapshot_file = tmp_path / "snapshot.json"
    snapshot_file.write_text(
        json.dumps(
            [
                {
                    "strategy": "grid_live",
                    "regime": "daily",
                    "hit_rate": 0.55,
                    "pnl": -12.0,
                    "sharpe": 0.1,
                    "observations": 3,
                }
            ]
        ),
        encoding="utf-8",
    )

    exit_code = run_ai_governor_cycle.main(
        ["--snapshot", str(snapshot_file), "--mode", "grid", "--limit", "4"]
    )

    assert exit_code == 0
