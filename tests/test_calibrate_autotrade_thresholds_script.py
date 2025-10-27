from __future__ import annotations

import csv
import json
import math
import subprocess
import sys
from datetime import datetime, timezone
from unittest.mock import patch
from pathlib import Path

import pytest

from bot_core.runtime.journal import TradingDecisionEvent

from scripts.calibrate_autotrade_thresholds import (
    _AMBIGUOUS_SYMBOL_MAPPING,
    _DEFAULT_GROUP_SAMPLE_LIMIT,
    _DEFAULT_GLOBAL_SAMPLE_LIMIT,
    _canonicalize_symbol_key,
    _build_symbol_map,
    _generate_report,
    _load_current_signal_thresholds,
    _load_autotrade_entries,
    _write_csv,
    StreamingMetricAggregator,
    _approximate_folded_normal_percentile,
)


def _write_journal(path: Path) -> None:
    events = [
        TradingDecisionEvent(
            event_type="ai_inference",
            timestamp=datetime(2023, 12, 31, 23, 0, tzinfo=timezone.utc),
            environment="paper",
            portfolio="core",
            risk_profile="balanced",
            symbol="BTCUSDT",
            primary_exchange="binance",
            strategy="trend_following",
            metadata={
                "signal_after_adjustment": "0.90",
                "signal_after_clamp": "0.88",
            },
        ),
        TradingDecisionEvent(
            event_type="ai_inference",
            timestamp=datetime(2024, 1, 1, 0, 0, tzinfo=timezone.utc),
            environment="paper",
            portfolio="core",
            risk_profile="balanced",
            symbol="BTCUSDT",
            primary_exchange="binance",
            strategy="trend_following",
            metadata={
                "signal_after_adjustment": "0.62",
                "signal_after_clamp": "0.55",
            },
        ),
        TradingDecisionEvent(
            event_type="ai_inference",
            timestamp=datetime(2024, 1, 1, 1, 15, tzinfo=timezone.utc),
            environment="paper",
            portfolio="core",
            risk_profile="balanced",
            symbol="BTCUSDT",
            primary_exchange="binance",
            strategy="trend_following",
            metadata={
                "signal_after_adjustment": "NaN",
                "signal_after_clamp": "Infinity",
            },
        ),
        TradingDecisionEvent(
            event_type="ai_inference",
            timestamp=datetime(2024, 1, 1, 1, 0, tzinfo=timezone.utc),
            environment="paper",
            portfolio="core",
            risk_profile="balanced",
            symbol="BTCUSDT",
            primary_exchange="binance",
            strategy="trend_following",
            metadata={
                "signal_after_adjustment": "0.48",
                "signal_after_clamp": "0.52",
            },
        ),
        TradingDecisionEvent(
            event_type="ai_inference",
            timestamp=datetime(2024, 1, 1, 2, 0, tzinfo=timezone.utc),
            environment="paper",
            portfolio="core",
            risk_profile="balanced",
            symbol="ETHUSDT",
            primary_exchange="kraken",
            strategy="mean_reversion",
            metadata={
                "signal_after_adjustment": "-0.30",
                "signal_after_clamp": "-0.28",
            },
        ),
        TradingDecisionEvent(
            event_type="risk_freeze",
            timestamp=datetime(2024, 1, 1, 3, 0, tzinfo=timezone.utc),
            environment="paper",
            portfolio="core",
            risk_profile="balanced",
            symbol="BTCUSDT",
            status="risk_freeze",
            primary_exchange="binance",
            strategy="trend_following",
            metadata={
                "reason": "manual_override",
                "frozen_for": "120",
            },
        ),
    ]
    with path.open("w", encoding="utf-8") as handle:
        for event in events:
            handle.write(json.dumps(event.as_dict()))
            handle.write("\n")


def _write_autotrade_export(path: Path) -> None:
    payload = {
        "version": 1,
        "entries": [
            {
                "timestamp": "2024-01-01T00:00:00Z",
                "decision": {
                    "details": {
                        "symbol": "BTCUSDT",
                        "primary_exchange": "binance",
                        "strategy": "trend_following",
                        "summary": {"risk_score": 0.72},
                    }
                },
            },
            {
                "timestamp": "2024-01-01T01:30:00Z",
                "decision": {
                    "details": {
                        "symbol": "BTCUSDT",
                        "primary_exchange": "binance",
                        "strategy": "trend_following",
                        "summary": {"risk_score": 0.65},
                    }
                },
            },
            {
                "timestamp": "2024-01-01T01:45:00Z",
                "decision": {
                    "details": {
                        "symbol": "BTCUSDT",
                        "primary_exchange": "binance",
                        "strategy": "trend_following",
                        "summary": {"risk_score": "NaN"},
                    }
                },
            },
            {
                "timestamp": "2024-01-01T02:00:00Z",
                "detail": {
                    "symbol": "ETHUSDT",
                    "primary_exchange": "kraken",
                    "strategy": "mean_reversion",
                    "summary": {"risk_score": 0.41},
                },
            },
            {
                "timestamp": "2024-01-01T03:30:00Z",
                "status": "auto_risk_freeze",
                "detail": {
                    "symbol": "BTCUSDT",
                    "reason": "risk_score_threshold",
                    "frozen_for": 180,
                    "primary_exchange": "binance",
                    "strategy": "trend_following",
                    "summary": {
                        "risk_score": 0.83,
                        "primary_exchange": "binance",
                        "strategy": "trend_following",
                    },
                },
            },
            {
                "timestamp": "2024-01-01T03:45:00Z",
                "status": "auto_risk_freeze",
                "detail": {
                    "symbol": "BTCUSDT",
                    "reason": "risk_score_threshold",
                    "frozen_for": "NaN",
                    "primary_exchange": "binance",
                    "strategy": "trend_following",
                    "summary": {"risk_score": 0.9},
                },
            },
        ],
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_script_generates_report(tmp_path: Path) -> None:
    journal_path = tmp_path / "journal.jsonl"
    _write_journal(journal_path)

    export_path = tmp_path / "autotrade.json"
    _write_autotrade_export(export_path)

    output_json = tmp_path / "report.json"
    output_csv = tmp_path / "report.csv"

    result = subprocess.run(
        [
            sys.executable,
            "scripts/calibrate_autotrade_thresholds.py",
            "--journal",
            str(journal_path),
            "--autotrade-export",
            str(export_path),
            "--percentiles",
            "0.5,0.9",
            "--suggestion-percentile",
            "0.9",
            "--since",
            "2024-01-01T00:30:00Z",
            "--until",
            "2024-01-01T04:00:00Z",
            "--current-threshold",
            "signal_after_adjustment=0.82,signal_after_clamp=0.78",
            "--output-json",
            str(output_json),
            "--output-csv",
            str(output_csv),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    assert "Zapisano raport JSON" in result.stdout
    assert output_json.exists()
    assert output_csv.exists()

    payload = json.loads(output_json.read_text(encoding="utf-8"))
    assert payload["schema"] == "stage6.autotrade.threshold_calibration"
    groups = {
        (entry["primary_exchange"], entry["strategy"]): entry for entry in payload["groups"]
    }
    trend_group = groups[("binance", "trend_following")]
    signal_stats = trend_group["metrics"]["signal_after_adjustment"]
    assert signal_stats["count"] == 1
    assert signal_stats["percentiles"]["p90"] >= 0.48
    assert signal_stats["suggested_threshold"] >= signal_stats["percentiles"]["p90"]
    assert signal_stats["current_threshold"] == 0.82
    assert all(math.isfinite(value) for value in signal_stats["percentiles"].values())

    signal_clamp_stats = trend_group["metrics"]["signal_after_clamp"]
    assert signal_clamp_stats["current_threshold"] == 0.78
    assert all(math.isfinite(value) for value in signal_clamp_stats["percentiles"].values())

    risk_stats = trend_group["metrics"]["risk_score"]
    assert risk_stats["count"] == 1
    assert risk_stats["current_threshold"] >= 0.7
    assert all(math.isfinite(value) for value in risk_stats["percentiles"].values())

    freeze_stats = trend_group["metrics"]["risk_freeze_duration"]
    assert freeze_stats["count"] == 2
    assert freeze_stats["max"] >= 180
    assert all(math.isfinite(value) for value in freeze_stats["percentiles"].values())

    freeze_summary = trend_group["freeze_summary"]
    assert freeze_summary["total"] == 3
    assert freeze_summary["auto"] == 2
    assert freeze_summary["manual"] == 1
    assert freeze_summary["omitted"] == 0
    reasons = {item["reason"]: item["count"] for item in freeze_summary["reasons"]}
    assert reasons["manual_override"] == 1
    assert reasons["risk_score_threshold"] == 2
    raw_freezes = trend_group["raw_freeze_events"]
    assert {entry["status"] for entry in raw_freezes} >= {"risk_freeze", "auto_risk_freeze"}
    assert trend_group["raw_freeze_events_truncated"] is False
    assert trend_group["raw_freeze_events_omitted"] == 0

    assert trend_group["raw_values_truncated"] is False
    assert all(count == 0 for count in trend_group["raw_values_omitted"].values())

    mean_rev_group = groups[("kraken", "mean_reversion")]
    mean_rev_risk = mean_rev_group["metrics"]["risk_score"]
    assert mean_rev_risk["count"] == 1

    global_summary = payload["global_summary"]
    assert global_summary["freeze_summary"]["total"] == 3
    assert global_summary["freeze_summary"]["omitted"] == 0
    global_signal = global_summary["metrics"]["signal_after_adjustment"]
    assert global_signal["count"] == 2
    assert global_signal["current_threshold"] == 0.82
    assert global_signal["sample_truncated"] is False
    assert global_signal["retained_samples"] == 2
    assert global_signal["omitted_samples"] == 0
    assert "raw_values" not in global_summary

    raw_values = trend_group["raw_values"]["signal_after_adjustment"]
    assert 0.62 not in raw_values
    assert all(math.isfinite(value) for value in raw_values)

    for metric_values in trend_group["raw_values"].values():
        assert all(math.isfinite(value) for value in metric_values)
    assert trend_group["raw_values_omitted"].get("signal_after_adjustment", 0) == 0

    with output_csv.open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)

    assert reader.fieldnames is not None
    for column in ("sample_truncated", "retained_samples", "omitted_samples"):
        assert column in reader.fieldnames, f"Brak kolumny {column} w eksporcie CSV"

    assert {row["metric"] for row in rows} >= {
        "signal_after_adjustment",
        "risk_score",
        "risk_freeze_duration",
    }
    freeze_rows = [
        row
        for row in rows
        if row["metric"] == "__freeze_summary__" and row["primary_exchange"] == "binance"
    ]
    assert freeze_rows, "Brak wiersza freeze_summary dla binance"
    freeze_row = freeze_rows[0]
    assert int(freeze_row["freeze_total"]) == 3
    assert int(freeze_row["freeze_auto"]) == 2
    assert int(freeze_row["freeze_manual"]) == 1
    assert int(freeze_row["freeze_omitted"]) == 0
    assert freeze_row["freeze_truncated"] in {"false", ""}
    assert freeze_row["freeze_status_counts"], "Oczekiwano rozbicia statusów w CSV"
    aggregated_freeze_rows = [
        row
        for row in rows
        if row["metric"] == "__freeze_summary__" and row["primary_exchange"] == "__all__"
    ]
    assert aggregated_freeze_rows, "Brak globalnego wiersza freeze_summary"
    assert int(aggregated_freeze_rows[0]["freeze_total"]) == 3

    global_metric_rows = [
        row
        for row in rows
        if row["primary_exchange"] == "__all__"
        and row["strategy"] == "__all__"
        and row["metric"] == "signal_after_adjustment"
    ]
    assert global_metric_rows, "Brak wiersza globalnego dla signal_after_adjustment"
    global_row = global_metric_rows[0]
    assert global_row["sample_truncated"] in {"false", ""}
    assert global_row["retained_samples"] == "2"
    assert global_row["omitted_samples"] == "0"

    for row in rows:
        if row["metric"] == "risk_score" and row["primary_exchange"] == "binance":
            assert float(row["current_threshold"]) >= 0.7
        if row["metric"] == "signal_after_adjustment" and row["primary_exchange"] == "binance":
            assert float(row["current_threshold"]) == 0.82
        if row["metric"] == "signal_after_clamp" and row["primary_exchange"] == "binance":
            assert float(row["current_threshold"]) == 0.78
    assert any(row["primary_exchange"] == "__all__" for row in rows)

    assert "raw_values" not in payload["global_summary"]


def test_script_accepts_cli_risk_score_threshold(tmp_path: Path) -> None:
    journal_path = tmp_path / "journal.jsonl"
    _write_journal(journal_path)

    export_path = tmp_path / "autotrade.json"
    _write_autotrade_export(export_path)

    output_json = tmp_path / "report.json"

    subprocess.run(
        [
            sys.executable,
            "scripts/calibrate_autotrade_thresholds.py",
            "--journal",
            str(journal_path),
            "--autotrade-export",
            str(export_path),
            "--percentiles",
            "0.5",
            "--suggestion-percentile",
            "0.5",
            "--current-threshold",
            "risk_score=0.72",
            "--output-json",
            str(output_json),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    payload = json.loads(output_json.read_text(encoding="utf-8"))
    trend_group = next(
        entry
        for entry in payload["groups"]
        if entry["primary_exchange"] == "binance" and entry["strategy"] == "trend_following"
    )
    risk_stats = trend_group["metrics"]["risk_score"]
    assert risk_stats["current_threshold"] == pytest.approx(0.72)
    sources = payload["sources"]
    assert sources["risk_score_override"] == pytest.approx(0.72)
    signal_sources = sources["current_signal_thresholds"]
    assert signal_sources["inline"]["risk_score"] == pytest.approx(0.72)
    assert "file_risk_score" not in signal_sources


def test_script_tracks_file_risk_score_threshold(tmp_path: Path) -> None:
    journal_path = tmp_path / "journal.jsonl"
    _write_journal(journal_path)

    export_path = tmp_path / "autotrade.json"
    _write_autotrade_export(export_path)

    thresholds_path = tmp_path / "current_thresholds.json"
    thresholds_path.write_text(json.dumps({"risk_score": {"value": 0.58}}), encoding="utf-8")

    output_json = tmp_path / "report.json"

    subprocess.run(
        [
            sys.executable,
            "scripts/calibrate_autotrade_thresholds.py",
            "--journal",
            str(journal_path),
            "--autotrade-export",
            str(export_path),
            "--percentiles",
            "0.5",
            "--suggestion-percentile",
            "0.5",
            "--current-threshold",
            str(thresholds_path),
            "--output-json",
            str(output_json),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    payload = json.loads(output_json.read_text(encoding="utf-8"))
    trend_group = next(
        entry
        for entry in payload["groups"]
        if entry["primary_exchange"] == "binance" and entry["strategy"] == "trend_following"
    )
    risk_stats = trend_group["metrics"]["risk_score"]
    assert risk_stats["current_threshold"] == pytest.approx(0.58)
    sources = payload["sources"]
    signal_sources = sources["current_signal_thresholds"]
    assert signal_sources["file_risk_score"] == pytest.approx(0.58)
    assert "risk_score_override" not in sources
    assert "inline" not in signal_sources or "risk_score" not in signal_sources.get("inline", {})


def test_script_limits_raw_freeze_events_via_cli(tmp_path: Path) -> None:
    journal_path = tmp_path / "journal.jsonl"
    _write_journal(journal_path)

    export_path = tmp_path / "autotrade.json"
    _write_autotrade_export(export_path)

    output_json = tmp_path / "report.json"

    subprocess.run(
        [
            sys.executable,
            "scripts/calibrate_autotrade_thresholds.py",
            "--journal",
            str(journal_path),
            "--autotrade-export",
            str(export_path),
            "--max-freeze-events",
            "1",
            "--output-json",
            str(output_json),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    payload = json.loads(output_json.read_text(encoding="utf-8"))
    trend_group = next(
        entry
        for entry in payload["groups"]
        if entry["primary_exchange"] == "binance" and entry["strategy"] == "trend_following"
    )

    assert len(trend_group["raw_freeze_events"]) == 1
    assert trend_group["freeze_summary"]["total"] == 3
    assert trend_group["freeze_summary"]["omitted"] == 2
    assert trend_group["raw_freeze_events_truncated"] is True
    assert trend_group["raw_freeze_events_omitted"] == 2
    assert payload["sources"]["max_freeze_events"] == 1
    assert payload["sources"]["raw_freeze_events_truncated_groups"] == 1
    assert payload["global_summary"]["freeze_summary"]["omitted"] == 2


def test_script_limits_raw_values_via_cli(tmp_path: Path) -> None:
    journal_path = tmp_path / "journal.jsonl"
    _write_journal(journal_path)

    export_path = tmp_path / "autotrade.json"
    _write_autotrade_export(export_path)

    output_json = tmp_path / "report.json"

    subprocess.run(
        [
            sys.executable,
            "scripts/calibrate_autotrade_thresholds.py",
            "--journal",
            str(journal_path),
            "--autotrade-export",
            str(export_path),
            "--max-raw-values",
            "1",
            "--output-json",
            str(output_json),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    payload = json.loads(output_json.read_text(encoding="utf-8"))
    trend_group = next(
        entry
        for entry in payload["groups"]
        if entry["primary_exchange"] == "binance" and entry["strategy"] == "trend_following"
    )

    trend_raw_values = trend_group["raw_values"]
    assert len(trend_raw_values["signal_after_adjustment"]) == 1
    assert trend_raw_values["signal_after_adjustment"][0] == pytest.approx(0.48)
    assert len(trend_raw_values["signal_after_clamp"]) == 1
    assert trend_raw_values["signal_after_clamp"][0] == pytest.approx(0.52)
    assert len(trend_raw_values["risk_freeze_duration"]) == 1
    assert trend_raw_values["risk_freeze_duration"][0] == pytest.approx(180.0)
    assert len(trend_raw_values["risk_score"]) == 1
    assert trend_raw_values["risk_score"][0] == pytest.approx(0.72)

    omitted_counts = trend_group["raw_values_omitted"]
    assert omitted_counts["signal_after_adjustment"] == 2
    assert omitted_counts["signal_after_clamp"] == 2
    assert omitted_counts["risk_score"] == 1
    assert omitted_counts["risk_freeze_duration"] == 1
    assert trend_group["raw_values_truncated"] is True

    sources = payload["sources"]
    assert sources["max_raw_values"] == 1
    assert sources["max_group_samples"] == _DEFAULT_GROUP_SAMPLE_LIMIT
    assert sources["raw_values_truncated_groups"] == 1
    assert sources["raw_values_omitted_total"] == 6
    assert sources["max_global_samples"] == _DEFAULT_GLOBAL_SAMPLE_LIMIT
    assert "global_samples_truncated_metrics" not in sources
    assert "global_samples_omitted_total" not in sources


def test_script_keeps_percentiles_when_raw_values_disabled(tmp_path: Path) -> None:
    journal_path = tmp_path / "journal.jsonl"
    _write_journal(journal_path)

    export_path = tmp_path / "autotrade.json"
    _write_autotrade_export(export_path)

    output_json = tmp_path / "report.json"

    subprocess.run(
        [
            sys.executable,
            "scripts/calibrate_autotrade_thresholds.py",
            "--journal",
            str(journal_path),
            "--autotrade-export",
            str(export_path),
            "--max-raw-values",
            "0",
            "--output-json",
            str(output_json),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    payload = json.loads(output_json.read_text(encoding="utf-8"))
    trend_group = next(
        entry
        for entry in payload["groups"]
        if entry["primary_exchange"] == "binance" and entry["strategy"] == "trend_following"
    )

    metrics = trend_group["metrics"]["signal_after_adjustment"]
    assert metrics["count"] == 3
    assert metrics["sample_truncated"] is False
    assert metrics["percentiles"]["p95"] is not None

    sources = payload["sources"]
    assert sources["max_raw_values"] == 0
    assert sources["max_group_samples"] == _DEFAULT_GROUP_SAMPLE_LIMIT


def test_script_estimates_global_percentiles_without_samples(tmp_path: Path) -> None:
    journal_path = tmp_path / "journal.jsonl"
    _write_journal(journal_path)

    export_path = tmp_path / "autotrade.json"
    _write_autotrade_export(export_path)

    output_json = tmp_path / "report.json"
    output_csv = tmp_path / "report.csv"

    subprocess.run(
        [
            sys.executable,
            "scripts/calibrate_autotrade_thresholds.py",
            "--journal",
            str(journal_path),
            "--autotrade-export",
            str(export_path),
            "--percentiles",
            "0.5,0.9",
            "--max-global-samples",
            "0",
            "--output-json",
            str(output_json),
            "--output-csv",
            str(output_csv),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    payload = json.loads(output_json.read_text(encoding="utf-8"))
    global_summary = payload["global_summary"]
    global_signal = global_summary["metrics"]["signal_after_adjustment"]

    assert global_signal["sample_truncated"] is True
    assert global_signal["retained_samples"] == 0
    assert global_signal["omitted_samples"] == global_signal["count"]

    expected_values = [0.90, 0.62, 0.48, -0.30]
    aggregator = StreamingMetricAggregator(sample_limit=0)
    aggregator.extend(expected_values)
    percentile_values = [0.5, 0.9]
    expected_percentiles = aggregator.statistics(percentile_values).payload["percentiles"]

    percentiles_payload = global_signal["percentiles"]
    for key, value in expected_percentiles.items():
        assert key in percentiles_payload
        assert percentiles_payload[key] == pytest.approx(value)
        assert -1.0 <= percentiles_payload[key] <= 1.0

    suggestion_percentile = payload["suggestion_percentile"]
    assert global_signal["suggested_threshold"] is not None
    assert -1.0 <= global_signal["suggested_threshold"] <= 1.0

    mean = global_signal["mean"]
    stddev = global_signal["stddev"]
    assert isinstance(mean, (int, float))
    assert isinstance(stddev, (int, float))

    folded_candidate = _approximate_folded_normal_percentile(
        mean, stddev, suggestion_percentile
    )
    bounds = [abs(value) for value in expected_values if math.isfinite(value)]
    lower_bound = min(bounds)
    upper_bound = max(bounds)
    folded_candidate = max(lower_bound, min(folded_candidate, upper_bound))
    assert global_signal["suggested_threshold"] == pytest.approx(folded_candidate)

    risk_summary = global_summary["metrics"]["risk_score"]
    for value in risk_summary["percentiles"].values():
        if value is not None:
            assert 0.0 <= value <= 1.0
    suggested_risk = risk_summary.get("suggested_threshold")
    if suggested_risk is not None:
        assert 0.0 <= suggested_risk <= 1.0

    sources = payload["sources"]
    assert sources["max_global_samples"] == 0
    truncated_metrics = sources.get("global_samples_truncated_metrics")
    assert truncated_metrics is not None and truncated_metrics >= 1
    omitted_total = sources.get("global_samples_omitted_total")
    assert omitted_total is not None and omitted_total >= global_signal["count"]

    with output_csv.open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)

    percentile_labels = payload["percentiles"]
    csv_row = next(
        row
        for row in rows
        if row["primary_exchange"] == "__all__" and row["metric"] == "signal_after_adjustment"
    )
    assert csv_row["sample_truncated"] == "true"
    assert int(csv_row["retained_samples"]) == 0
    assert int(csv_row["omitted_samples"]) == global_signal["count"]
    for label in percentile_labels:
        assert float(csv_row[label]) == pytest.approx(percentiles_payload[label])


def test_script_respects_group_sample_limit(tmp_path: Path) -> None:
    journal_path = tmp_path / "journal.jsonl"
    _write_journal(journal_path)

    export_path = tmp_path / "autotrade.json"
    _write_autotrade_export(export_path)

    output_json = tmp_path / "report.json"

    subprocess.run(
        [
            sys.executable,
            "scripts/calibrate_autotrade_thresholds.py",
            "--journal",
            str(journal_path),
            "--autotrade-export",
            str(export_path),
            "--max-group-samples",
            "1",
            "--output-json",
            str(output_json),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    payload = json.loads(output_json.read_text(encoding="utf-8"))
    trend_group = next(
        entry
        for entry in payload["groups"]
        if entry["primary_exchange"] == "binance" and entry["strategy"] == "trend_following"
    )

    metrics = trend_group["metrics"]["signal_after_adjustment"]
    assert metrics["sample_truncated"] is True
    assert metrics["retained_samples"] == 1
    assert metrics["omitted_samples"] == metrics["count"] - 1

    sources = payload["sources"]
    assert sources["max_group_samples"] == 1
    assert sources["group_samples_truncated_metrics"] >= 1


def test_streaming_metric_aggregator_clamps_approximations() -> None:
    aggregator = StreamingMetricAggregator(sample_limit=0, domain=(0.0, 1.0))
    aggregator.extend([0.94, 0.96, 0.97, 0.98])
    stats = aggregator.statistics([0.99])
    percentile_value = stats.payload["percentiles"]["p99"]
    assert percentile_value is not None
    assert 0.0 <= percentile_value <= 1.0
    assert percentile_value <= 0.98

    suggested = aggregator.suggest_threshold(0.999, statistics_result=stats)
    assert suggested is not None
    assert 0.0 <= suggested <= 1.0

    absolute_aggregator = StreamingMetricAggregator(sample_limit=0, domain=(-1.0, 1.0))
    absolute_aggregator.extend([-0.8, -0.6, 0.7, 0.65])
    absolute_stats = absolute_aggregator.statistics([0.999])
    absolute_threshold = absolute_aggregator.suggest_threshold(
        0.999,
        absolute=True,
        statistics_result=absolute_stats,
    )
    assert absolute_threshold is not None
    assert 0.0 <= absolute_threshold <= 1.0


def test_generate_report_includes_clamp_counters() -> None:
    report = _generate_report(
        journal_events=[
            {
                "event_type": "ai_inference",
                "primary_exchange": "binance",
                "strategy": "trend_following",
                "symbol": "BTCUSDT",
                "signal_after_adjustment": "1.4",
                "signal_after_clamp": "1.5",
            }
        ],
        autotrade_entries=[
            {
                "timestamp": "2024-01-01T00:00:00Z",
                "decision": {
                    "details": {
                        "symbol": "BTCUSDT",
                        "primary_exchange": "binance",
                        "strategy": "trend_following",
                        "summary": {"risk_score": 1.4},
                    }
                },
            }
        ],
        percentiles=[0.9],
        suggestion_percentile=0.9,
    )

    assert report["groups"], "Raport powinien zawierać co najmniej jedną grupę"
    group = report["groups"][0]

    adjustment_clamps = group["metrics"]["signal_after_adjustment"]["clamped_values"]
    clamp_clamps = group["metrics"]["signal_after_clamp"]["clamped_values"]

    assert adjustment_clamps["regular"] > 0
    assert adjustment_clamps["absolute"] > 0
    assert clamp_clamps["regular"] > 0
    assert clamp_clamps["absolute"] > 0

    risk_clamps = group["metrics"]["risk_score"]["clamped_values"]
    assert risk_clamps["regular"] > 0
    assert risk_clamps["absolute"] == 0

    sources = report["sources"].get("clamped_values")
    assert sources is not None
    assert sources["regular"] >= adjustment_clamps["regular"] + clamp_clamps["regular"]
    assert sources["absolute"] >= clamp_clamps["absolute"]
    assert sources.get("metrics_with_regular_clamp", 0) >= 1
    assert sources.get("metrics_with_absolute_clamp", 0) >= 1


def test_streaming_metric_aggregator_reuses_sorted_samples(monkeypatch: pytest.MonkeyPatch) -> None:
    aggregator = StreamingMetricAggregator(sample_limit=1000)
    for value in range(-500, 500):
        aggregator.add(float(value))

    call_counter = {"count": 0}
    original_sorted_samples = StreamingMetricAggregator._sorted_samples

    def counting_sorted_samples(self: StreamingMetricAggregator) -> list[float]:
        call_counter["count"] += 1
        return original_sorted_samples(self)

    monkeypatch.setattr(StreamingMetricAggregator, "_sorted_samples", counting_sorted_samples)

    stats_result = aggregator.statistics([0.9])
    assert call_counter["count"] == 1

    first_suggestion = aggregator.suggest_threshold(
        0.9,
        statistics_result=stats_result,
    )
    assert first_suggestion is not None
    assert call_counter["count"] == 1

    absolute_suggestion = aggregator.suggest_threshold(
        0.9,
        absolute=True,
        statistics_result=stats_result,
    )
    assert absolute_suggestion is not None
    assert call_counter["count"] == 1


def test_script_accepts_thresholds_from_json_file(tmp_path: Path) -> None:
    journal_path = tmp_path / "journal.jsonl"
    _write_journal(journal_path)

    export_path = tmp_path / "autotrade.json"
    _write_autotrade_export(export_path)

    thresholds_path = tmp_path / "thresholds.json"
    thresholds_payload = {
        "overrides": {
            "signal_after_adjustment": "0.81",
        }
    }
    thresholds_path.write_text(json.dumps(thresholds_payload), encoding="utf-8")

    output_json = tmp_path / "report.json"

    subprocess.run(
        [
            sys.executable,
            "scripts/calibrate_autotrade_thresholds.py",
            "--journal",
            str(journal_path),
            "--autotrade-export",
            str(export_path),
            "--current-threshold",
            str(thresholds_path),
            "--current-threshold",
            "signal_after_clamp=0.77",
            "--output-json",
            str(output_json),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    payload = json.loads(output_json.read_text(encoding="utf-8"))
    trend_group = next(
        entry
        for entry in payload["groups"]
        if entry["primary_exchange"] == "binance" and entry["strategy"] == "trend_following"
    )
    metrics = trend_group["metrics"]
    assert metrics["signal_after_adjustment"]["current_threshold"] == 0.81
    assert metrics["signal_after_clamp"]["current_threshold"] == 0.77


def test_script_normalizes_cli_threshold_keys(tmp_path: Path) -> None:
    journal_path = tmp_path / "journal.jsonl"
    _write_journal(journal_path)

    export_path = tmp_path / "autotrade.json"
    _write_autotrade_export(export_path)

    output_json = tmp_path / "report.json"

    subprocess.run(
        [
            sys.executable,
            "scripts/calibrate_autotrade_thresholds.py",
            "--journal",
            str(journal_path),
            "--autotrade-export",
            str(export_path),
            "--percentiles",
            "0.5",
            "--suggestion-percentile",
            "0.5",
            "--current-threshold",
            "Signal_After_Adjustment=0.8",
            "--current-threshold",
            "signal-after-clamp=0.7",
            "--output-json",
            str(output_json),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    payload = json.loads(output_json.read_text(encoding="utf-8"))
    trend_group = next(
        entry
        for entry in payload["groups"]
        if entry["primary_exchange"] == "binance" and entry["strategy"] == "trend_following"
    )
    signal_stats = trend_group["metrics"]["signal_after_adjustment"]
    assert signal_stats["current_threshold"] == pytest.approx(0.8)
    clamp_stats = trend_group["metrics"]["signal_after_clamp"]
    assert clamp_stats["current_threshold"] == pytest.approx(0.7)
    current_sources = payload["sources"]["current_signal_thresholds"]
    assert current_sources["inline"]["signal_after_adjustment"] == pytest.approx(0.8)
    assert current_sources["inline"]["signal_after_clamp"] == pytest.approx(0.7)


def test_load_current_thresholds_rejects_directory(tmp_path: Path) -> None:
    thresholds_dir = tmp_path / "thresholds"
    thresholds_dir.mkdir()

    with pytest.raises(SystemExit) as excinfo:
        _load_current_signal_thresholds([str(thresholds_dir)])

    assert "musi wskazywać plik" in str(excinfo.value)


def test_load_current_thresholds_errors_on_missing_file(tmp_path: Path) -> None:
    missing_path = tmp_path / "missing_thresholds.json"

    with pytest.raises(SystemExit) as excinfo:
        _load_current_signal_thresholds([str(missing_path)])

    assert "Ścieżka z progami nie istnieje" in str(excinfo.value)


def test_load_current_thresholds_supports_nested_structures(tmp_path: Path) -> None:
    payload = {
        "signals": {
            "signal_after_adjustment": {"current_threshold": "0.84"},
            "signal_after_clamp": {"threshold": 0.76},
        },
        "overrides": [
            {"metric": "signal_after_adjustment", "value": 0.85},
            {"name": "signal_after_clamp", "current": "0.74"},
        ],
        "legacy_signal_after_adjustment_threshold": 0.83,
    }

    thresholds_path = tmp_path / "nested_thresholds.json"
    thresholds_path.write_text(json.dumps(payload), encoding="utf-8")

    thresholds, risk_score_sources, metadata = _load_current_signal_thresholds(
        [str(thresholds_path)]
    )

    assert thresholds["signal_after_adjustment"] == pytest.approx(0.85)
    assert thresholds["signal_after_clamp"] == pytest.approx(0.74)
    assert risk_score_sources.from_files is None
    assert risk_score_sources.from_inline is None
    assert metadata["files"] == [str(thresholds_path)]
    assert metadata.get("inline") is None


def test_load_current_thresholds_tracks_inline_sources() -> None:
    thresholds, risk_score_sources, metadata = _load_current_signal_thresholds(
        ["signal_after_adjustment=0.81", "risk_score=0.66"]
    )

    assert thresholds == {"signal_after_adjustment": 0.81}
    assert risk_score_sources.from_inline == pytest.approx(0.66)
    assert risk_score_sources.from_files is None
    assert "files" not in metadata
    assert metadata["inline"] == {
        "signal_after_adjustment": pytest.approx(0.81),
        "risk_score": pytest.approx(0.66),
    }


def test_group_metrics_use_streaming_aggregators_for_large_inputs() -> None:
    sample_count = _DEFAULT_GROUP_SAMPLE_LIMIT + 200
    journal_events = [
        {
            "primary_exchange": "binance",
            "strategy": "trend_following",
            "signal_after_adjustment": 0.5,
            "signal_after_clamp": 0.5,
        }
        for _ in range(sample_count)
    ]

    report = _generate_report(
        journal_events=journal_events,
        autotrade_entries=[],
        percentiles=[0.5],
        suggestion_percentile=0.5,
        max_raw_values=0,
    )

    sources = report["sources"]
    assert sources["max_raw_values"] == 0
    assert sources["max_group_samples"] == _DEFAULT_GROUP_SAMPLE_LIMIT
    assert sources["group_samples_truncated_metrics"] >= 1
    assert sources["group_samples_omitted_total"] > 0
    group_entry = report["groups"][0]
    metrics = group_entry["metrics"]["signal_after_adjustment"]

    assert metrics["count"] == sample_count
    assert metrics["sample_truncated"] is True
    assert metrics["retained_samples"] == _DEFAULT_GROUP_SAMPLE_LIMIT
    assert metrics["omitted_samples"] == sample_count - _DEFAULT_GROUP_SAMPLE_LIMIT
    percentile_label = report["percentiles"][0]
    assert metrics["percentiles"][percentile_label] == pytest.approx(0.5)


def test_load_current_thresholds_without_sources() -> None:
    thresholds, risk_score_sources, metadata = _load_current_signal_thresholds(None)

    assert thresholds == {}
    assert risk_score_sources.from_inline is None
    assert risk_score_sources.from_files is None
    assert metadata == {}


def test_load_current_thresholds_rejects_non_finite_cli() -> None:
    with pytest.raises(SystemExit) as excinfo:
        _load_current_signal_thresholds(["signal_after_adjustment=NaN"])

    assert "niefinityczną" in str(excinfo.value)


def test_load_current_thresholds_rejects_non_finite_file(tmp_path: Path) -> None:
    payload = {
        "signal_after_adjustment": {
            "current_threshold": "Infinity",
        },
        "risk_score": {
            "value": "-Infinity",
        },
    }

    thresholds_path = tmp_path / "bad_thresholds.json"
    thresholds_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(SystemExit) as excinfo:
        _load_current_signal_thresholds([str(thresholds_path)])

    assert "niefinityczną" in str(excinfo.value)


def test_group_resolution_prefers_entry_metadata_over_symbol_map() -> None:
    journal_events = [
        {
            "symbol": "BTCUSDT",
            "primary_exchange": "binance",
            "strategy": "trend_following",
            "signal_after_adjustment": 0.5,
            "signal_after_clamp": 0.4,
        }
    ]

    entry_kraken = {
        "decision": {
            "details": {
                "symbol": "BTCUSDT",
                "primary_exchange": "kraken",
                "strategy": "mean_reversion",
                "summary": {"risk_score": 0.61},
            }
        }
    }
    entry_binance = {
        "decision": {
            "details": {
                "symbol": "BTCUSDT",
                "summary": {"risk_score": 0.73},
            },
            "primary_exchange": "binance",
            "strategy": "trend_following",
        }
    }

    for entries in ([entry_kraken, entry_binance], [entry_binance, entry_kraken]):
        report = _generate_report(
            journal_events=journal_events,
            autotrade_entries=list(entries),
            percentiles=[0.5],
            suggestion_percentile=0.5,
        )

        groups = {
            (group["primary_exchange"], group["strategy"]): group
            for group in report["groups"]
        }

        kraken_group = groups[("kraken", "mean_reversion")]
        binance_group = groups[("binance", "trend_following")]

        assert (
            kraken_group["metrics"]["risk_score"]["count"] == 1
        ), "Entry-specific metadata should override symbol_map fallback"
        assert (
            binance_group["metrics"]["risk_score"]["count"] == 1
        ), "Entries should remain grouped by their own metadata"


def test_group_resolution_merges_entry_and_summary_metadata_before_symbol_map() -> None:
    journal_events = [
        {
            "symbol": "BTCUSDT",
            "primary_exchange": "binance",
            "strategy": "trend_following",
        }
    ]

    autotrade_entries = [
        {
            "detail": {
                "symbol": "BTCUSDT",
                "primary_exchange": "kraken",
                "summary": {
                    "risk_score": 0.45,
                    "strategy": "mean_reversion",
                },
            }
        }
    ]

    with patch("scripts.calibrate_autotrade_thresholds.load_risk_thresholds", return_value={}):
        report = _generate_report(
            journal_events=journal_events,
            autotrade_entries=autotrade_entries,
            percentiles=[0.5],
            suggestion_percentile=0.5,
        )

    assert report["groups"]
    group = report["groups"][0]
    assert group["primary_exchange"] == "kraken"
    assert group["strategy"] == "mean_reversion"


def test_report_groups_routing_identifiers_case_insensitively() -> None:
    journal_events = [
        {
            "symbol": "BTCUSDT",
            "primary_exchange": "Binance",
            "strategy": "TREND_FOLLOWING",
            "signal_after_adjustment": 0.61,
            "signal_after_clamp": 0.6,
        },
        {
            "symbol": "BTCUSDT",
            "primary_exchange": "binance",
            "strategy": "trend_following",
            "signal_after_adjustment": 0.59,
            "signal_after_clamp": 0.58,
        },
    ]

    autotrade_entries = [
        {
            "decision": {
                "details": {
                    "symbol": "BTCUSDT",
                    "primary_exchange": "BINANCE",
                    "strategy": "Trend_Following",
                    "summary": {"risk_score": 0.67},
                }
            }
        }
    ]

    with patch("scripts.calibrate_autotrade_thresholds.load_risk_thresholds", return_value={}):
        report = _generate_report(
            journal_events=journal_events,
            autotrade_entries=autotrade_entries,
            percentiles=[0.5],
            suggestion_percentile=0.5,
        )

    assert len(report["groups"]) == 1
    group = report["groups"][0]
    assert group["primary_exchange"] == "Binance"
    assert group["strategy"] == "TREND_FOLLOWING"
    assert group["primary_exchange"].casefold() == "binance"
    assert group["strategy"].casefold() == "trend_following"
    assert group["metrics"]["signal_after_adjustment"]["count"] == 2
    assert group["metrics"]["risk_score"]["count"] == 1


def test_autotrade_entry_keeps_detail_routing_when_symbol_missing() -> None:
    journal_events = [
        {
            "primary_exchange": "kraken",
            "strategy": "mean_reversion",
        }
    ]
    autotrade_entries = [
        {
            "detail": {
                "symbol": "LTCUSDT",
                "primary_exchange": "kraken",
                "strategy": "mean_reversion",
                "summary": {
                    "risk_score": 0.51,
                },
            }
        }
    ]

    with patch("scripts.calibrate_autotrade_thresholds.load_risk_thresholds", return_value={}):
        report = _generate_report(
            journal_events=journal_events,
            autotrade_entries=autotrade_entries,
            percentiles=[0.5],
            suggestion_percentile=0.5,
        )

    groups = {
        (group["primary_exchange"], group["strategy"])
        for group in report["groups"]
    }
    assert ("kraken", "mean_reversion") in groups
    assert ("unknown", "unknown") not in groups


def test_autotrade_entry_reads_decision_level_routing_metadata() -> None:
    journal_events = []
    autotrade_entries = [
        {
            "decision": {
                "primary_exchange": "coinbase",
                "strategy": "momentum",
                "details": {
                    "symbol": "ADAUSD",
                    "summary": {
                        "risk_score": 0.23,
                    },
                },
            }
        }
    ]

    with patch("scripts.calibrate_autotrade_thresholds.load_risk_thresholds", return_value={}):
        report = _generate_report(
            journal_events=journal_events,
            autotrade_entries=autotrade_entries,
            percentiles=[0.5],
            suggestion_percentile=0.5,
        )

    assert report["groups"]
    group = report["groups"][0]
    assert group["primary_exchange"] == "coinbase"
    assert group["strategy"] == "momentum"


def test_autotrade_entry_reads_nested_routing_metadata() -> None:
    journal_events = [
        {
            "symbol": "ADAUSD",
            "primary_exchange": "binance",
            "strategy": "trend_following",
        }
    ]

    autotrade_entries = [
        {
            "detail": {
                "routing": {
                    "primary_exchange": "ftx",
                    "strategy": "grid_bot",
                },
                "symbol": "ADAUSD",
                "summary": {
                    "risk_score": 0.67,
                },
            }
        }
    ]

    with patch("scripts.calibrate_autotrade_thresholds.load_risk_thresholds", return_value={}):
        report = _generate_report(
            journal_events=journal_events,
            autotrade_entries=autotrade_entries,
            percentiles=[0.5],
            suggestion_percentile=0.5,
        )

    groups = {
        (group["primary_exchange"], group["strategy"]): group for group in report["groups"]
    }

    assert ("ftx", "grid_bot") in groups
    nested_group = groups[("ftx", "grid_bot")]
    assert nested_group["metrics"]["risk_score"]["count"] == 1


def test_generate_report_uses_custom_risk_threshold_path(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    autotrade_entries = [
        {
            "decision": {
                "details": {
                    "symbol": "BTCUSDT",
                    "primary_exchange": "binance",
                    "strategy": "trend_following",
                    "summary": {"risk_score": 0.61},
                }
            }
        }
    ]

    config_path = tmp_path / "risk_thresholds.yaml"
    config_path.write_text("auto_trader: {}\n", encoding="utf-8")

    calls: list[Path | None] = []

    def _fake_loader(*, config_path: Path | None = None):
        calls.append(config_path)
        return {"auto_trader": {"map_regime_to_signal": {"risk_score": 0.42}}}

    monkeypatch.setattr(
        "scripts.calibrate_autotrade_thresholds.load_risk_thresholds",
        _fake_loader,
    )

    report = _generate_report(
        journal_events=[],
        autotrade_entries=autotrade_entries,
        percentiles=[0.5],
        suggestion_percentile=0.5,
        risk_threshold_sources=[str(config_path)],
    )

    assert calls == [config_path]
    metrics = report["groups"][0]["metrics"]["risk_score"]
    assert metrics["current_threshold"] == pytest.approx(0.42)


def test_generate_report_merges_multiple_risk_threshold_paths(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    autotrade_entries = [
        {
            "decision": {
                "details": {
                    "symbol": "BTCUSDT",
                    "primary_exchange": "binance",
                    "strategy": "trend_following",
                    "summary": {"risk_score": 0.7},
                }
            }
        }
    ]

    first = tmp_path / "risk_a.yaml"
    second = tmp_path / "risk_b.yaml"
    first.write_text("{}", encoding="utf-8")
    second.write_text("{}", encoding="utf-8")

    responses = {first: 0.33, second: 0.88}
    calls: list[Path | None] = []

    def _fake_loader(*, config_path: Path | None = None):
        calls.append(config_path)
        return {"auto_trader": {"map_regime_to_signal": {"risk_score": responses[config_path]}}}

    monkeypatch.setattr(
        "scripts.calibrate_autotrade_thresholds.load_risk_thresholds",
        _fake_loader,
    )

    report = _generate_report(
        journal_events=[],
        autotrade_entries=autotrade_entries,
        percentiles=[0.5],
        suggestion_percentile=0.5,
        risk_threshold_sources=[str(first), str(second)],
    )

    assert calls == [first, second]
    metrics = report["groups"][0]["metrics"]["risk_score"]
    assert metrics["current_threshold"] == pytest.approx(0.88)


def test_cli_risk_score_override_beats_config(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    autotrade_entries = [
        {
            "decision": {
                "details": {
                    "symbol": "BTCUSDT",
                    "primary_exchange": "binance",
                    "strategy": "trend_following",
                    "summary": {"risk_score": 0.6},
                }
            }
        }
    ]

    override_path = tmp_path / "risk_thresholds.yaml"
    override_path.write_text("auto_trader: {}\n", encoding="utf-8")

    def _fake_loader(*, config_path: Path | None = None):
        assert config_path == override_path
        return {"auto_trader": {"map_regime_to_signal": {"risk_score": 0.9}}}

    monkeypatch.setattr(
        "scripts.calibrate_autotrade_thresholds.load_risk_thresholds",
        _fake_loader,
    )

    report = _generate_report(
        journal_events=[],
        autotrade_entries=autotrade_entries,
        percentiles=[0.5],
        suggestion_percentile=0.5,
        risk_threshold_sources=[str(override_path)],
        cli_risk_score=0.42,
    )

    metrics = report["groups"][0]["metrics"]["risk_score"]
    assert metrics["current_threshold"] == pytest.approx(0.42)
    assert report["sources"]["risk_score_override"] == pytest.approx(0.42)


def test_autotrade_entry_reads_routing_sequences() -> None:
    journal_events = [
        {
            "symbol": "ADAUSD",
            "primary_exchange": "binance",
            "strategy": "trend_following",
        }
    ]

    autotrade_entries = [
        {
            "detail": {
                "symbol": "ADAUSD",
                "routing": {
                    "legs": [
                        {
                            "route": {
                                "primary_exchange": "kraken",
                                "strategy": "mean_reversion",
                            }
                        }
                    ]
                },
                "summary": {
                    "risk_score": 0.51,
                },
            }
        }
    ]

    with patch("scripts.calibrate_autotrade_thresholds.load_risk_thresholds", return_value={}):
        report = _generate_report(
            journal_events=journal_events,
            autotrade_entries=autotrade_entries,
            percentiles=[0.5],
            suggestion_percentile=0.5,
        )

    groups = {
        (group["primary_exchange"], group["strategy"]): group for group in report["groups"]
    }

    assert ("kraken", "mean_reversion") in groups
    nested_group = groups[("kraken", "mean_reversion")]
    assert nested_group["metrics"]["risk_score"]["count"] == 1


def test_journal_freeze_event_preserves_event_metadata_over_symbol_map() -> None:
    journal_events = [
        {
            "event_type": "ai_inference",
            "symbol": "BTCUSDT",
            "primary_exchange": "binance",
            "strategy": "trend_following",
            "signal_after_adjustment": 0.73,
        },
        {
            "event_type": "risk_freeze",
            "symbol": "BTCUSDT",
            "primary_exchange": "kraken",
            "strategy": "mean_reversion",
            "status": "risk_freeze",
            "freeze_duration": 300,
        },
    ]

    with patch("scripts.calibrate_autotrade_thresholds.load_risk_thresholds", return_value={}):
        report = _generate_report(
            journal_events=journal_events,
            autotrade_entries=[],
            percentiles=[0.5],
            suggestion_percentile=0.5,
        )

    groups = {
        (group["primary_exchange"], group["strategy"]): group for group in report["groups"]
    }

    assert ("binance", "trend_following") in groups
    assert ("kraken", "mean_reversion") in groups

    assert groups[("kraken", "mean_reversion")]["freeze_summary"]["total"] == 1
    assert groups[("binance", "trend_following")]["freeze_summary"]["total"] == 0


def test_autotrade_entry_falls_back_to_summary_metadata() -> None:
    journal_events = []
    autotrade_entries = [
        {
            "detail": {
                "symbol": "SOLUSDT",
                "summary": {
                    "primary_exchange": "binance",
                    "strategy": "breakout",
                    "risk_score": 0.37,
                },
            }
        }
    ]

    with patch("scripts.calibrate_autotrade_thresholds.load_risk_thresholds", return_value={}):
        report = _generate_report(
            journal_events=journal_events,
            autotrade_entries=autotrade_entries,
            percentiles=[0.5],
            suggestion_percentile=0.5,
        )

    assert report["groups"]
    group = report["groups"][0]
    assert group["primary_exchange"] == "binance"
    assert group["strategy"] == "breakout"


def test_build_symbol_map_updates_incomplete_entries() -> None:
    events = [
        {"symbol": "ADAUSDT", "primary_exchange": None, "strategy": None},
        {"symbol": "ADAUSDT", "primary_exchange": "binance", "strategy": "trend"},
    ]

    mapping = _build_symbol_map(events)

    canonical_symbol = _canonicalize_symbol_key("ADAUSDT")
    assert canonical_symbol is not None
    assert mapping[canonical_symbol] == ("binance", "trend")


def test_build_symbol_map_combines_partial_details() -> None:
    events = [
        {"symbol": "DOGEUSDT", "primary_exchange": "binance", "strategy": None},
        {"symbol": "DOGEUSDT", "primary_exchange": None, "strategy": "momentum"},
    ]

    mapping = _build_symbol_map(events)

    canonical_symbol = _canonicalize_symbol_key("DOGEUSDT")
    assert canonical_symbol is not None
    assert mapping[canonical_symbol] == ("binance", "momentum")


def test_build_symbol_map_marks_ambiguous_when_conflicting_routing() -> None:
    events = [
        {
            "symbol": "XRPUSDT",
            "primary_exchange": "binance",
            "strategy": "scalping",
        },
        {
            "symbol": "XRPUSDT",
            "primary_exchange": "kraken",
            "strategy": "scalping",
        },
    ]

    mapping = _build_symbol_map(events)

    canonical_symbol = _canonicalize_symbol_key("XRPUSDT")
    assert canonical_symbol is not None
    assert mapping[canonical_symbol] == _AMBIGUOUS_SYMBOL_MAPPING


def test_generate_report_accepts_generators() -> None:
    journal_events = (
        {
            "primary_exchange": "Binance",
            "strategy": "Trend",
            "symbol": "BTCUSDT",
            "signal_after_adjustment": 0.6,
            "signal_after_clamp": 0.58,
        },
        {
            "primary_exchange": "Binance",
            "strategy": "Trend",
            "symbol": "BTCUSDT",
            "signal_after_adjustment": 0.4,
            "signal_after_clamp": 0.38,
        },
    )
    autotrade_entries = (
        {
            "timestamp": "2024-01-01T00:00:00Z",
            "detail": {
                "symbol": "btcusdt",
                "primary_exchange": "binance",
                "strategy": "trend",
                "summary": {"risk_score": 0.42},
            },
        },
    )

    report = _generate_report(
        journal_events=(dict(event) for event in journal_events),
        autotrade_entries=(dict(entry) for entry in autotrade_entries),
        percentiles=[0.5],
        suggestion_percentile=0.5,
    )

    sources = report["sources"]
    assert sources["journal_events"] == 2
    assert sources["autotrade_entries"] == 1
    assert "max_freeze_events" not in sources
    assert "max_raw_values" not in sources
    assert sources["max_global_samples"] == _DEFAULT_GLOBAL_SAMPLE_LIMIT
    assert "global_samples_truncated_metrics" not in sources
    assert report["groups"]
    trend_group = next(iter(report["groups"]))
    assert trend_group["metrics"]["signal_after_adjustment"]["count"] == 2


def test_generate_report_limits_raw_freeze_events() -> None:
    autotrade_entries = [
        {
            "timestamp": "2024-01-01T00:00:00Z",
            "detail": {
                "symbol": "BTCUSDT",
                "primary_exchange": "binance",
                "strategy": "trend_following",
                "summary": {"risk_score": 0.51},
            },
        },
        {
            "status": "auto_risk_freeze",
            "detail": {
                "symbol": "BTCUSDT",
                "primary_exchange": "binance",
                "strategy": "trend_following",
                "summary": {"risk_score": 0.71},
            },
        },
        {
            "status": "risk_freeze_manual",
            "detail": {
                "symbol": "BTCUSDT",
                "primary_exchange": "binance",
                "strategy": "trend_following",
                "summary": {"risk_score": 0.68},
            },
        },
        {
            "status": "auto_risk_freeze",
            "detail": {
                "symbol": "BTCUSDT",
                "primary_exchange": "binance",
                "strategy": "trend_following",
                "summary": {"risk_score": 0.92},
            },
        },
    ]

    with patch("scripts.calibrate_autotrade_thresholds.load_risk_thresholds", return_value={}):
        report = _generate_report(
            journal_events=[],
            autotrade_entries=autotrade_entries,
            percentiles=[0.5],
            suggestion_percentile=0.5,
            max_freeze_events=1,
        )

    assert report["groups"]
    group = report["groups"][0]
    assert group["freeze_summary"]["total"] == 3
    assert len(group["raw_freeze_events"]) == 1
    assert group["raw_freeze_events_truncated"] is True
    assert group["raw_freeze_events_omitted"] == 2
    assert report["sources"]["max_freeze_events"] == 1
    assert report["sources"]["raw_freeze_events_truncated_groups"] == 1


def test_generate_report_includes_freeze_only_groups() -> None:
    autotrade_entries = [
        {
            "timestamp": "2024-02-01T00:00:00Z",
            "status": "auto_risk_freeze",
            "detail": {
                "symbol": "ETHUSDT",
                "primary_exchange": "binance",
                "strategy": "trend_following",
                "summary": {"risk_score": 0.71},
            },
        },
        {
            "timestamp": "2024-02-01T01:00:00Z",
            "status": "risk_freeze_manual",
            "detail": {
                "symbol": "ETHUSDT",
                "primary_exchange": "binance",
                "strategy": "trend_following",
                "summary": {"risk_score": 0.69},
            },
        },
    ]

    with patch("scripts.calibrate_autotrade_thresholds.load_risk_thresholds", return_value={}):
        report = _generate_report(
            journal_events=[],
            autotrade_entries=autotrade_entries,
            percentiles=[0.5],
            suggestion_percentile=0.5,
        )

    groups = {
        (group["primary_exchange"], group["strategy"]): group for group in report["groups"]
    }

    key = ("binance", "trend_following")
    assert key in groups
    freeze_only_group = groups[key]
    assert freeze_only_group["metrics"] == {}
    assert freeze_only_group["freeze_summary"]["total"] == 2
    assert freeze_only_group["freeze_summary"]["omitted"] == 0
    assert len(freeze_only_group["raw_freeze_events"]) == 2
    assert freeze_only_group["raw_freeze_events_truncated"] is False
    assert freeze_only_group["raw_freeze_events_omitted"] == 0

    global_summary = report["global_summary"]["freeze_summary"]
    assert global_summary["total"] == 2
    assert global_summary["omitted"] == 0


def test_generate_report_limits_global_samples() -> None:
    journal_events = []
    for index in range(200):
        journal_events.append(
            {
                "primary_exchange": "binance",
                "strategy": "trend_following",
                "symbol": "BTCUSDT",
                "signal_after_adjustment": index / 100.0,
                "signal_after_clamp": index / 120.0,
            }
        )

    autotrade_entries = []
    for index in range(200):
        autotrade_entries.append(
            {
                "timestamp": f"2024-01-01T00:{index:02d}:00Z",
                "detail": {
                    "symbol": "BTCUSDT",
                    "primary_exchange": "binance",
                    "strategy": "trend_following",
                    "summary": {"risk_score": index / 200.0},
                },
            }
        )

    report = _generate_report(
        journal_events=journal_events,
        autotrade_entries=autotrade_entries,
        percentiles=[0.5, 0.9],
        suggestion_percentile=0.9,
        max_global_samples=10,
    )

    global_metrics = report["global_summary"]["metrics"]
    signal_stats = global_metrics["signal_after_adjustment"]
    assert signal_stats["count"] == 200
    assert signal_stats["sample_truncated"] is True
    assert signal_stats["retained_samples"] == 10
    assert signal_stats["omitted_samples"] == 190
    assert signal_stats["percentiles"]

    risk_stats = global_metrics["risk_score"]
    assert risk_stats["count"] == 200
    assert risk_stats["sample_truncated"] is True
    assert risk_stats["retained_samples"] == 10
    assert risk_stats["omitted_samples"] == 190

    sources = report["sources"]
    assert sources["max_global_samples"] == 10
    assert sources["global_samples_truncated_metrics"] >= 1
    assert sources["global_samples_omitted_total"] >= 190


def test_write_csv_includes_freeze_only_groups(tmp_path: Path) -> None:
    autotrade_entries = [
        {
            "timestamp": "2024-02-01T00:00:00Z",
            "status": "auto_risk_freeze",
            "detail": {
                "symbol": "ETHUSDT",
                "primary_exchange": "binance",
                "strategy": "trend_following",
                "summary": {"risk_score": 0.71},
            },
        },
        {
            "timestamp": "2024-02-01T01:00:00Z",
            "status": "risk_freeze_manual",
            "detail": {
                "symbol": "ETHUSDT",
                "primary_exchange": "binance",
                "strategy": "trend_following",
                "summary": {"risk_score": 0.69},
            },
        },
    ]

    with patch("scripts.calibrate_autotrade_thresholds.load_risk_thresholds", return_value={}):
        report = _generate_report(
            journal_events=[],
            autotrade_entries=autotrade_entries,
            percentiles=[0.5],
            suggestion_percentile=0.5,
        )

    csv_path = tmp_path / "freeze_only.csv"
    _write_csv(
        report["groups"],
        csv_path,
        percentiles=report["percentiles"],
        global_summary=report["global_summary"],
    )

    with csv_path.open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)

    assert reader.fieldnames is not None
    for column in ("sample_truncated", "retained_samples", "omitted_samples"):
        assert column in reader.fieldnames, f"Brak kolumny {column} w eksporcie CSV"

    freeze_rows = [
        row
        for row in rows
        if row["metric"] == "__freeze_summary__" and row["primary_exchange"] == "binance"
    ]
    assert freeze_rows, "CSV powinien zawierać wiersz dla blokad"
    freeze_row = freeze_rows[0]
    assert int(freeze_row["freeze_total"]) == 2
    assert int(freeze_row["freeze_auto"]) == 1
    assert int(freeze_row["freeze_manual"]) == 1
    assert freeze_row["freeze_status_counts"], "Oczekiwano rozbicia statusów"

    aggregated_freeze_rows = [
        row
        for row in rows
        if row["metric"] == "__freeze_summary__" and row["primary_exchange"] == "__all__"
    ]
    assert aggregated_freeze_rows, "CSV powinien zawierać wiersz zbiorczy"
    assert int(aggregated_freeze_rows[0]["freeze_total"]) == 2


def test_build_symbol_map_coalesces_case_variants() -> None:
    events = [
        {"symbol": "BtcUsdt", "primary_exchange": "binance", "strategy": "trend"},
        {"symbol": "BTCUSDT", "primary_exchange": None, "strategy": "TREND"},
    ]

    mapping = _build_symbol_map(events)

    canonical_symbol = _canonicalize_symbol_key("BTCUSDT")
    assert canonical_symbol is not None
    assert set(mapping) == {canonical_symbol}
    assert mapping[canonical_symbol] == ("binance", "trend")


def test_symbol_map_fallback_supplements_missing_metadata() -> None:
    journal_events = [
        {
            "symbol": "SOLUSDT",
            "primary_exchange": "binance",
        }
    ]

    autotrade_entries = [
        {
            "detail": {
                "symbol": "SOLUSDT",
                "summary": {
                    "risk_score": 0.42,
                },
            }
        }
    ]

    with patch("scripts.calibrate_autotrade_thresholds.load_risk_thresholds", return_value={}):
        report = _generate_report(
            journal_events=journal_events,
            autotrade_entries=autotrade_entries,
            percentiles=[0.5],
            suggestion_percentile=0.5,
        )

    groups = {
        (group["primary_exchange"], group["strategy"]): group for group in report["groups"]
    }

    assert ("binance", "unknown") in groups
    assert groups[("binance", "unknown")]["metrics"]["risk_score"]["count"] == 1


def test_symbol_map_matches_symbols_case_insensitively() -> None:
    journal_events = [
        {
            "symbol": "ethusdt",
            "primary_exchange": "binance",
            "strategy": "trend_following",
        }
    ]

    mapping = _build_symbol_map(journal_events)
    canonical_symbol = _canonicalize_symbol_key("ETHUSDT")
    assert canonical_symbol is not None
    assert mapping[canonical_symbol] == ("binance", "trend_following")

    autotrade_entries = [
        {
            "decision": {
                "details": {
                    "symbol": "ETHUSDT",
                    "summary": {"risk_score": 0.42},
                }
            }
        }
    ]

    with patch("scripts.calibrate_autotrade_thresholds.load_risk_thresholds", return_value={}):
        report = _generate_report(
            journal_events=journal_events,
            autotrade_entries=autotrade_entries,
            percentiles=[0.5],
            suggestion_percentile=0.5,
        )

    groups = {
        (group["primary_exchange"], group["strategy"]): group for group in report["groups"]
    }

    assert ("binance", "trend_following") in groups
    matched_group = groups[("binance", "trend_following")]
    assert matched_group["metrics"]["risk_score"]["count"] == 1


def test_unknown_routing_display_is_canonicalized() -> None:
    journal_events = [
        {
            "symbol": "ADAUSDT",
            "primary_exchange": "Unknown",
            "strategy": "UNKNOWN",
            "signal_after_adjustment": 0.51,
        }
    ]

    with patch("scripts.calibrate_autotrade_thresholds.load_risk_thresholds", return_value={}):
        report = _generate_report(
            journal_events=journal_events,
            autotrade_entries=[],
            percentiles=[0.5],
            suggestion_percentile=0.5,
        )

    assert report["groups"]
    group = report["groups"][0]
    assert group["primary_exchange"] == "unknown"
    assert group["strategy"] == "unknown"


def test_unknown_synonyms_are_normalized() -> None:
    journal_events = [
        {
            "symbol": "ADAUSDT",
            "primary_exchange": "N/A",
            "strategy": "None",
            "signal_after_adjustment": 0.48,
        }
    ]

    with patch("scripts.calibrate_autotrade_thresholds.load_risk_thresholds", return_value={}):
        report = _generate_report(
            journal_events=journal_events,
            autotrade_entries=[],
            percentiles=[0.5],
            suggestion_percentile=0.5,
        )

    assert report["groups"]
    group = report["groups"][0]
    assert group["primary_exchange"] == "unknown"
    assert group["strategy"] == "unknown"


def test_symbol_map_ambiguous_entry_keeps_unknown_routing() -> None:
    journal_events = [
        {
            "symbol": "SOLUSDT",
            "primary_exchange": "binance",
            "strategy": "breakout",
        },
        {
            "symbol": "SOLUSDT",
            "primary_exchange": "kraken",
            "strategy": "breakout",
        },
    ]

    autotrade_entries = [
        {
            "detail": {
                "symbol": "SOLUSDT",
                "summary": {
                    "risk_score": 0.38,
                },
            }
        }
    ]

    with patch("scripts.calibrate_autotrade_thresholds.load_risk_thresholds", return_value={}):
        report = _generate_report(
            journal_events=journal_events,
            autotrade_entries=autotrade_entries,
            percentiles=[0.5],
            suggestion_percentile=0.5,
        )

    groups = {
        (group["primary_exchange"], group["strategy"]): group for group in report["groups"]
    }

    assert set(groups) == {("unknown", "unknown")}
    assert groups[("unknown", "unknown")]["metrics"]["risk_score"]["count"] == 1


def test_autotrade_export_is_parsed_streamingly(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    export_path = tmp_path / "export.json"
    entries = [
        {"timestamp": "2024-01-01T00:00:00Z", "detail": {"symbol": "BTCUSDT"}},
        {"timestamp": "2024-01-01T00:05:00Z", "detail": {"symbol": "BTCUSDT"}},
        {"timestamp": "2024-01-01T00:10:00Z", "detail": {"symbol": "BTCUSDT"}},
        {"timestamp": "2024-01-01T00:15:00Z", "detail": {"symbol": "BTCUSDT"}},
    ]
    export_path.write_text(json.dumps({"entries": entries}), encoding="utf-8")

    read_lengths: list[int] = []
    original_open = Path.open

    class CountingHandle:
        def __init__(self, handle):  # type: ignore[no-untyped-def]
            self._handle = handle

        def read(self, size: int = -1) -> str:
            data = self._handle.read(size)
            read_lengths.append(len(data))
            return data

        def __getattr__(self, name: str):  # noqa: ANN204 - passthrough helper
            return getattr(self._handle, name)

        def __enter__(self):  # noqa: ANN204 - passthrough helper
            self._handle.__enter__()
            return self

        def __exit__(self, *exc):  # noqa: ANN204 - passthrough helper
            return self._handle.__exit__(*exc)

    def counting_open(self, *args, **kwargs):  # type: ignore[no-untyped-def]
        return CountingHandle(original_open(self, *args, **kwargs))

    monkeypatch.setattr("scripts.calibrate_autotrade_thresholds._STREAM_READ_SIZE", 16)
    monkeypatch.setattr(Path, "open", counting_open)

    loaded_entries = list(_load_autotrade_entries([str(export_path)]))
    assert len(loaded_entries) == len(entries)

    positive_reads = [length for length in read_lengths if length > 0]
    assert len(positive_reads) >= 2
    assert max(positive_reads) <= 16


def test_autotrade_export_with_bom(tmp_path: Path) -> None:
    export_path = tmp_path / "export_with_bom.json"
    payload = {"entries": [{"timestamp": "2024-01-01T00:00:00Z"}]}
    export_path.write_text("\ufeff" + json.dumps(payload), encoding="utf-8")

    entries = list(_load_autotrade_entries([str(export_path)]))

    assert len(entries) == 1
    assert entries[0]["timestamp"] == "2024-01-01T00:00:00Z"


def test_autotrade_export_rejects_extra_data_after_array(tmp_path: Path) -> None:
    export_path = tmp_path / "export_extra.json"
    payload = [{"timestamp": "2024-01-01T00:00:00Z"}]
    export_path.write_text(json.dumps(payload) + " {}", encoding="utf-8")

    with pytest.raises(SystemExit, match="dodatkowe dane po tablicy"):
        list(_load_autotrade_entries([str(export_path)]))
