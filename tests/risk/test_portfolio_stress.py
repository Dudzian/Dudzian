import json
import os
import subprocess
import sys
from pathlib import Path

from stage6_samples.portfolio_stress import build_sample_scenarios, load_sample_baseline

from bot_core.risk.portfolio_stress import run_portfolio_stress
from bot_core.runtime.portfolio_inputs import load_portfolio_stress_summary


def test_run_portfolio_stress_builds_report() -> None:
    baseline = load_sample_baseline()
    scenarios = build_sample_scenarios()

    report = run_portfolio_stress(baseline, scenarios, report_metadata={"source": "unit-test"})

    assert report.baseline.portfolio_id == "stage6_core"
    assert len(report.scenarios) == len(scenarios)
    mapping = report.to_mapping()
    assert mapping["schema"] == "stage6.risk.portfolio_stress.report"
    assert mapping["metadata"]["source"] == "unit-test"
    summary = mapping["summary"]
    assert summary["scenario_count"] == len(scenarios)
    assert summary["max_drawdown_pct"] > 0
    assert summary["worst_scenario"]["name"] in {scenario.name for scenario in scenarios}
    assert summary["total_probability"] > 0
    assert summary["expected_pnl_usd"] != 0
    assert summary["var_95_return_pct"] <= 0
    assert summary["var_95_pnl_usd"] < 0
    assert summary["cvar_95_return_pct"] <= 0
    assert summary["cvar_95_pnl_usd"] < 0
    tag_aggregates = summary["tag_aggregates"]
    assert isinstance(tag_aggregates, list) and tag_aggregates
    tags_by_name = {entry["tag"]: entry for entry in tag_aggregates if "tag" in entry}
    liquidity_tag = tags_by_name["liquidity"]
    assert liquidity_tag["scenario_count"] == 1
    assert liquidity_tag["total_probability"] > 0
    assert liquidity_tag["worst_scenario"]["name"] == "usd_liquidity_crunch"

    liquidity = next(s for s in report.scenarios if s.scenario.name == "usd_liquidity_crunch")
    assert liquidity.total_return_pct < 0
    assert liquidity.drawdown_pct > 0
    assert any(pos.symbol == "btc_usdt" for pos in liquidity.positions)


def test_portfolio_stress_cli_generates_artifacts(tmp_path: Path) -> None:
    env = os.environ.copy()
    env["PYTHONPATH"] = "."

    json_path = tmp_path / "portfolio_stress.json"
    csv_path = tmp_path / "portfolio_stress.csv"
    sig_path = tmp_path / "portfolio_stress.sig"
    key_path = tmp_path / "portfolio_stress.key"
    key_path.write_bytes(b"stage6-secret")

    command = [
        sys.executable,
        "scripts/run_portfolio_stress.py",
        "--config",
        "config/core.yaml",
        "--baseline",
        "stage6_samples/portfolio_stress_baseline.json",
        "--output-json",
        str(json_path),
        "--output-csv",
        str(csv_path),
        "--signature-path",
        str(sig_path),
        "--signing-key-path",
        str(key_path),
        "--scenario",
        "usd_liquidity_crunch",
        "--scenario",
        "rates_regime_shift",
    ]

    subprocess.run(command, check=True, env=env)

    payload = json.loads(json_path.read_text(encoding="utf-8"))
    assert payload["schema"] == "stage6.risk.portfolio_stress.report"
    assert len(payload["scenarios"]) == 2
    assert payload["summary"]["scenario_count"] == 2
    assert "worst_scenario" in payload["summary"]
    assert "var_95_return_pct" in payload["summary"]
    assert "cvar_95_return_pct" in payload["summary"]
    tag_summary = payload["summary"]["tag_aggregates"]
    assert isinstance(tag_summary, list)
    assert any(entry.get("tag") == "liquidity" for entry in tag_summary)

    signature = json.loads(sig_path.read_text(encoding="utf-8"))
    assert signature["algorithm"] == "HMAC-SHA256"
    assert "value" in signature

    csv_lines = csv_path.read_text(encoding="utf-8").strip().splitlines()
    assert csv_lines[0].startswith("scenario,title,horizon_days")


def test_load_portfolio_stress_summary_includes_aggregates(tmp_path: Path) -> None:
    baseline = load_sample_baseline()
    scenarios = build_sample_scenarios()
    report = run_portfolio_stress(baseline, scenarios)
    path = tmp_path / "report.json"
    report.write_json(path)

    summary = load_portfolio_stress_summary(path)

    assert summary["scenario_count"] == len(scenarios)
    assert summary["summary"]["max_drawdown_pct"] == summary["max_drawdown_pct"]
    assert summary["summary"]["scenario_count"] == len(scenarios)
    assert summary["summary"]["total_probability"] > 0
    assert "worst_scenario" in summary
    assert "expected_pnl_usd" in summary["summary"]
    assert "var_95_return_pct" in summary["summary"]
    assert "cvar_95_return_pct" in summary["summary"]
    assert any(
        entry.get("tag") == "macro"
        for entry in summary["summary"].get("tag_aggregates", [])
    )
