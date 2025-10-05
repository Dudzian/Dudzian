from datetime import datetime, timezone

import pytest

from bot_core.data.ohlcv.coverage_check import (
    CoverageStatus,
    CoverageSummary,
    coerce_summary_mapping,
    compute_gap_statistics,
    compute_gap_statistics_by_interval,
    evaluate_summary_thresholds,
    summarize_coverage,
)
from bot_core.data.ohlcv.manifest_report import ManifestEntry


def _manifest_entry(
    *,
    symbol: str,
    interval: str,
    status: str,
    gap_minutes: float | None,
    threshold_minutes: int | None,
    row_count: int | None = None,
    timestamp: datetime | None = None,
) -> ManifestEntry:
    last_ts_ms = None
    last_ts_iso = None
    if timestamp is not None:
        last_ts_ms = int(timestamp.timestamp() * 1000)
        last_ts_iso = timestamp.astimezone(timezone.utc).isoformat()
    return ManifestEntry(
        symbol=symbol,
        interval=interval,
        row_count=row_count,
        last_timestamp_ms=last_ts_ms,
        last_timestamp_iso=last_ts_iso,
        gap_minutes=gap_minutes,
        threshold_minutes=threshold_minutes,
        status=status,
    )


def test_summarize_coverage_builds_metrics() -> None:
    entries = [
        CoverageStatus(
            symbol="BTCUSDT",
            interval="1d",
            manifest_entry=_manifest_entry(
                symbol="BTCUSDT",
                interval="1d",
                status="ok",
                gap_minutes=45.0,
                threshold_minutes=2880,
                row_count=365,
                timestamp=datetime(2024, 5, 1, tzinfo=timezone.utc),
            ),
            required_rows=360,
            issues=(),
        ),
        CoverageStatus(
            symbol="ETHUSDT",
            interval="1d",
            manifest_entry=_manifest_entry(
                symbol="ETHUSDT",
                interval="1d",
                status="warning",
                gap_minutes=240.0,
                threshold_minutes=180,
                row_count=355,
                timestamp=datetime(2024, 5, 2, tzinfo=timezone.utc),
            ),
            required_rows=360,
            issues=("manifest_status:warning", "insufficient_rows:355<360"),
        ),
        CoverageStatus(
            symbol="SOLUSDT",
            interval="1h",
            manifest_entry=_manifest_entry(
                symbol="SOLUSDT",
                interval="1h",
                status="missing_metadata",
                gap_minutes=None,
                threshold_minutes=120,
                row_count=None,
                timestamp=None,
            ),
            required_rows=720,
            issues=("manifest_status:missing_metadata", "missing_row_count"),
        ),
    ]

    summary = summarize_coverage(entries)

    assert isinstance(summary, CoverageSummary)
    assert summary.total == 3
    assert summary.ok == 1
    assert summary.error == 2
    assert summary.warning == 1
    assert summary.stale_entries == 1  # ETH ma lukę większą niż próg
    assert summary.ok_ratio == pytest.approx(1 / 3)
    assert summary.manifest_status_counts["warning"] == 1
    assert summary.issue_counts["manifest_status"] == 2
    assert summary.issue_counts["missing_row_count"] == 1
    assert summary.issue_examples["missing_row_count"] == "missing_row_count"
    assert summary.status == "error"

    payload = summary.to_mapping()
    assert payload["status"] == "error"
    assert payload["ok"] == 1
    assert payload["warning"] == 1
    assert payload["error"] == 2
    assert payload["stale_entries"] == 1
    assert payload["manifest_status_counts"]["missing_metadata"] == 1
    assert payload["issue_counts"]["insufficient_rows"] == 1
    assert payload["worst_gap"]["symbol"] == "ETHUSDT"
    assert payload["worst_gap"]["gap_minutes"] == pytest.approx(240.0)
    assert payload["worst_gap"]["manifest_status"] == "warning"
    assert "last_timestamp_iso" in payload["worst_gap"]


def test_summarize_coverage_handles_empty() -> None:
    summary = summarize_coverage([])

    assert summary.total == 0
    assert summary.ok == 0
    assert summary.error == 0
    assert summary.warning == 0
    assert summary.ok_ratio is None
    assert summary.status == "ok"
    assert summary.worst_gap is None

    payload = summary.to_mapping()
    assert payload["status"] == "ok"
    assert payload["ok_ratio"] is None
    assert payload["manifest_status_counts"] == {}
    assert payload["issue_counts"] == {}


def test_coerce_summary_mapping_handles_none() -> None:
    payload = coerce_summary_mapping(None)

    assert payload["status"] == "unknown"
    assert payload["total"] == 0
    assert payload["issue_counts"] == {}
    assert payload["worst_gap"] is None


def test_coerce_summary_mapping_completes_mapping() -> None:
    payload = coerce_summary_mapping({"status": "warning", "total": 2})

    assert payload["status"] == "warning"
    assert payload["total"] == 2
    assert payload["ok"] == 0  # uzupełnione domyślnie
    assert "manifest_status_counts" in payload


def test_coerce_summary_mapping_from_summary() -> None:
    statuses = [
        CoverageStatus(
            symbol="BTCUSDT",
            interval="1d",
            manifest_entry=_manifest_entry(
                symbol="BTCUSDT",
                interval="1d",
                status="ok",
                gap_minutes=0.0,
                threshold_minutes=120,
                row_count=10,
                timestamp=datetime(2024, 1, 1, tzinfo=timezone.utc),
            ),
            required_rows=10,
            issues=(),
        )
    ]

    payload = coerce_summary_mapping(summarize_coverage(statuses))

    assert payload["status"] == "ok"
    assert payload["total"] == 1
    assert payload["ok"] == 1


def test_evaluate_summary_thresholds_flags_max_gap() -> None:
    summary = summarize_coverage(
        [
            CoverageStatus(
                symbol="BTCUSDT",
                interval="1h",
                manifest_entry=_manifest_entry(
                    symbol="BTCUSDT",
                    interval="1h",
                    status="ok",
                    gap_minutes=180.0,
                    threshold_minutes=240,
                    row_count=720,
                    timestamp=datetime(2024, 1, 1, tzinfo=timezone.utc),
                ),
                required_rows=720,
                issues=(),
            )
        ]
    )

    result = evaluate_summary_thresholds(summary, max_gap_minutes=120.0)

    assert result.thresholds["max_gap_minutes"] == pytest.approx(120.0)
    assert "max_gap_exceeded:180.0>120.0" in result.issues
    assert result.observed["worst_gap_minutes"] == pytest.approx(180.0)


def test_evaluate_summary_thresholds_flags_ok_ratio() -> None:
    statuses = [
        CoverageStatus(
            symbol="BTCUSDT",
            interval="1d",
            manifest_entry=_manifest_entry(
                symbol="BTCUSDT",
                interval="1d",
                status="ok",
                gap_minutes=10.0,
                threshold_minutes=120,
                row_count=100,
                timestamp=datetime(2024, 1, 1, tzinfo=timezone.utc),
            ),
            required_rows=100,
            issues=(),
        ),
        CoverageStatus(
            symbol="ETHUSDT",
            interval="1d",
            manifest_entry=_manifest_entry(
                symbol="ETHUSDT",
                interval="1d",
                status="ok",
                gap_minutes=15.0,
                threshold_minutes=120,
                row_count=100,
                timestamp=datetime(2024, 1, 1, tzinfo=timezone.utc),
            ),
            required_rows=100,
            issues=(),
        ),
        CoverageStatus(
            symbol="SOLUSDT",
            interval="1d",
            manifest_entry=_manifest_entry(
                symbol="SOLUSDT",
                interval="1d",
                status="warning",
                gap_minutes=300.0,
                threshold_minutes=120,
                row_count=40,
                timestamp=datetime(2024, 1, 1, tzinfo=timezone.utc),
            ),
            required_rows=100,
            issues=("manifest_status:warning",),
        ),
        CoverageStatus(
            symbol="XRPUSDT",
            interval="1d",
            manifest_entry=_manifest_entry(
                symbol="XRPUSDT",
                interval="1d",
                status="error",
                gap_minutes=400.0,
                threshold_minutes=120,
                row_count=20,
                timestamp=datetime(2024, 1, 1, tzinfo=timezone.utc),
            ),
            required_rows=100,
            issues=("manifest_status:error",),
        ),
    ]

    summary = summarize_coverage(statuses)
    result = evaluate_summary_thresholds(summary, min_ok_ratio=0.9)

    assert result.thresholds["min_ok_ratio"] == pytest.approx(0.9)
    assert any(issue.startswith("ok_ratio_below_threshold:") for issue in result.issues)
    assert result.observed["ok_ratio"] == pytest.approx(summary.ok_ratio or 0.0)


def test_evaluate_summary_thresholds_handles_empty_manifest() -> None:
    summary = CoverageSummary(
        total=0,
        ok=0,
        error=0,
        warning=0,
        manifest_status_counts={},
        issue_counts={},
        issue_examples={},
        stale_entries=0,
        worst_gap=None,
    )

    result = evaluate_summary_thresholds(summary, min_ok_ratio=0.5)

    assert "manifest_empty_for_threshold" in result.issues
    assert result.thresholds["min_ok_ratio"] == pytest.approx(0.5)


def test_compute_gap_statistics_returns_percentiles() -> None:
    base = datetime(2024, 1, 1, tzinfo=timezone.utc)
    statuses = [
        CoverageStatus(
            symbol="BTCUSDT",
            interval="1d",
            manifest_entry=_manifest_entry(
                symbol="BTCUSDT",
                interval="1d",
                status="ok",
                gap_minutes=10.0,
                threshold_minutes=120,
                row_count=100,
                timestamp=base,
            ),
            required_rows=100,
            issues=(),
        ),
        CoverageStatus(
            symbol="ETHUSDT",
            interval="1d",
            manifest_entry=_manifest_entry(
                symbol="ETHUSDT",
                interval="1d",
                status="ok",
                gap_minutes=55.0,
                threshold_minutes=120,
                row_count=100,
                timestamp=base,
            ),
            required_rows=100,
            issues=(),
        ),
        CoverageStatus(
            symbol="SOLUSDT",
            interval="1h",
            manifest_entry=_manifest_entry(
                symbol="SOLUSDT",
                interval="1h",
                status="ok",
                gap_minutes=5.0,
                threshold_minutes=60,
                row_count=200,
                timestamp=base,
            ),
            required_rows=180,
            issues=(),
        ),
    ]

    stats = compute_gap_statistics(statuses)

    assert stats.total_entries == 3
    assert stats.with_gap_measurement == 3
    assert stats.min_gap_minutes == pytest.approx(5.0)
    assert stats.max_gap_minutes == pytest.approx(55.0)
    assert stats.median_gap_minutes == pytest.approx(10.0, rel=1e-6)
    assert stats.percentile_95_gap_minutes >= stats.median_gap_minutes
    mapping = stats.to_mapping()
    assert mapping["percentile_90_gap_minutes"] == pytest.approx(stats.percentile_90_gap_minutes)


def test_compute_gap_statistics_by_interval_groups_values() -> None:
    base = datetime(2024, 1, 1, tzinfo=timezone.utc)
    statuses = [
        CoverageStatus(
            symbol="BTCUSDT",
            interval="1d",
            manifest_entry=_manifest_entry(
                symbol="BTCUSDT",
                interval="1d",
                status="ok",
                gap_minutes=120.0,
                threshold_minutes=1440,
                row_count=100,
                timestamp=base,
            ),
            required_rows=100,
            issues=(),
        ),
        CoverageStatus(
            symbol="ETHUSDT",
            interval="1d",
            manifest_entry=_manifest_entry(
                symbol="ETHUSDT",
                interval="1d",
                status="ok",
                gap_minutes=240.0,
                threshold_minutes=1440,
                row_count=100,
                timestamp=base,
            ),
            required_rows=100,
            issues=(),
        ),
        CoverageStatus(
            symbol="SOLUSDT",
            interval="1h",
            manifest_entry=_manifest_entry(
                symbol="SOLUSDT",
                interval="1h",
                status="ok",
                gap_minutes=15.0,
                threshold_minutes=60,
                row_count=200,
                timestamp=base,
            ),
            required_rows=180,
            issues=(),
        ),
    ]

    stats_by_interval = compute_gap_statistics_by_interval(statuses)

    assert set(stats_by_interval.keys()) == {"1d", "1h"}
    daily_stats = stats_by_interval["1d"].to_mapping()
    assert daily_stats["with_gap_measurement"] == 2
    assert daily_stats["max_gap_minutes"] == pytest.approx(240.0)
    hourly_stats = stats_by_interval["1h"].to_mapping()
    assert hourly_stats["min_gap_minutes"] == pytest.approx(15.0)
    assert hourly_stats["total_entries"] == 1
