import json
from pathlib import Path

from scripts.run_backtest_regression import run_backtest_regressions


def test_ci_harness_generates_audit_artifacts(tmp_path: Path) -> None:
    summaries = run_backtest_regressions(tmp_path)
    assert summaries, "Harness should produce at least one summary"
    summary_index = tmp_path / "guardrail_summary.json"
    assert summary_index.exists(), "Aggregate guardrail summary should be produced"
    index_payload = json.loads(summary_index.read_text(encoding="utf-8"))
    assert index_payload["total"] >= 1
    assert index_payload["blocked"] == 0
    assert index_payload["metrics_violations"] == {}
    assert index_payload["warnings"] == {}
    summary_path = tmp_path / "trend_following" / "summary.json"
    assert summary_path.exists()
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    assert payload["scenario"] == "trend_following"
    assert payload["guardrails"]["allowed"] is True
    assert "strategy_metadata" in payload
    metadata = payload["strategy_metadata"]
    assert metadata["risk_profile"] == "balanced"
    assert "available_data" in metadata
    available = set(metadata["available_data"])
    assert {"open", "high", "low", "close", "volume"}.issubset(available)
    assert "required_data_missing" not in metadata
    metrics = payload["metrics"]
    assert "omega_ratio" in metrics
    assert "sortino_ratio" in metrics
    assert "max_exposure_pct" in metrics
    guardrails = payload["guardrails"]
    assert guardrails["threshold_sources"]["max_drawdown_pct"] == "risk_profile:balanced"
    assert guardrails["thresholds"]["min_sortino_ratio"] >= 0
    assert guardrails["thresholds"]["min_omega_ratio"] >= 0
    assert guardrails["threshold_sources"]["min_sortino_ratio"] == "risk_profile:balanced"
    assert guardrails["threshold_sources"]["min_omega_ratio"] == "risk_profile:balanced"
    assert guardrails["observed"]["sortino_ratio"] >= guardrails["thresholds"]["min_sortino_ratio"]
    assert guardrails["observed"]["omega_ratio"] >= guardrails["thresholds"]["min_omega_ratio"]
    assert guardrails["thresholds"]["max_risk_of_ruin_pct"] >= 0
    assert guardrails["threshold_sources"]["max_risk_of_ruin_pct"] == "risk_profile:balanced"
    assert guardrails["observed"]["risk_of_ruin_pct"] <= guardrails["thresholds"]["max_risk_of_ruin_pct"]
    assert guardrails["thresholds"]["min_hit_ratio_pct"] >= 0
    assert guardrails["threshold_sources"]["min_hit_ratio_pct"] == "risk_profile:balanced"
    assert guardrails["observed"]["hit_ratio_pct"] >= guardrails["thresholds"]["min_hit_ratio_pct"]
    assert guardrails["violations"] == []
    html_path = tmp_path / "trend_following" / "report.html"
    assert html_path.exists()
    artefacts = payload["artefacts"]
    assert artefacts["html"].endswith("report.html")
