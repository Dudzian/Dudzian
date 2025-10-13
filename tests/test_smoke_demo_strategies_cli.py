from __future__ import annotations

from pathlib import Path

import scripts.smoke_demo_strategies as cli

ROOT = Path(__file__).resolve().parents[1]


def test_smoke_demo_cli_runs(tmp_path) -> None:
    manifest = ROOT / "data/backtests/normalized/manifest.yaml"
    config = ROOT / "config/core.yaml"

    exit_code = cli.main(
        [
            "--config",
            str(config),
            "--manifest",
            str(manifest),
            "--environment",
            "demo",
            "--cycles",
            "2",
        ]
    )
    assert exit_code == 0


def test_smoke_demo_run_demo_returns_summary() -> None:
    manifest = ROOT / "data/backtests/normalized/manifest.yaml"
    config = ROOT / "config/core.yaml"

    result = cli.run_demo(
        config_path=config,
        manifest_path=manifest,
        environment="demo",
        scheduler_name=None,
        cycles=2,
    )
    assert result.cycles == 2
    assert "cross_exchange_watch" in result.emitted_signals
    assert result.emitted_signals["cross_exchange_watch"] >= 0
    for payload in result.telemetry.values():
        assert "signals" in payload
        assert "latency_ms" in payload
