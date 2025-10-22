from datetime import datetime, timezone

import pytest

from bot_core.reporting.ui_bridge import ReportEntry, ReportFile, _build_dashboard_payload


def _entry(summary: dict[str, object] | None, identifier: str = "daily/report") -> ReportEntry:
    now = datetime.now(timezone.utc)
    return ReportEntry(
        identifier=identifier,
        category="daily",
        summary_path="/tmp/summary.json",
        summary=summary,
        summary_error=None,
        exports=[
            ReportFile(
                relative_path="export.csv",
                absolute_path="/tmp/export.csv",
                size=42,
                modified_at=now,
            )
        ],
        updated_at=now,
        total_size=42,
        export_count=1,
        created_at=now,
    )


def test_dashboard_payload_extracts_equity_and_heatmap() -> None:
    summary = {
        "equity_curve": [
            {"timestamp": "2024-01-01T00:00:00Z", "value": 100.0},
            {"timestamp": "2024-01-02T00:00:00Z", "value": 110.5},
        ],
        "asset_heatmap": {
            "BTC": {"value": 0.6},
            "ETH": 0.4,
        },
    }
    payload = _build_dashboard_payload([_entry(summary)])

    equity = payload["equity_curve"]
    assert len(equity) == 2
    assert equity[0]["value"] == pytest.approx(100.0)
    heatmap = {cell["asset"]: cell for cell in payload["asset_heatmap"]}
    assert heatmap["BTC"]["value"] == pytest.approx(0.6)
    assert heatmap["ETH"]["value"] == pytest.approx(0.4)


def test_dashboard_payload_merges_heatmap_sources() -> None:
    first = {
        "asset_heatmap": {
            "BTC": {"value": 0.4},
            "ETH": 0.2,
        }
    }
    second = {
        "asset_heatmap": {
            "BTC": {"value": -0.1},
        }
    }

    payload = _build_dashboard_payload([_entry(first, "daily/a"), _entry(second, "daily/b")])
    heatmap = {cell["asset"]: cell for cell in payload["asset_heatmap"]}
    assert heatmap["BTC"]["value"] == pytest.approx(0.3)
    assert len(heatmap["BTC"]["sources"]) == 2


def test_dashboard_payload_handles_missing_data() -> None:
    payload = _build_dashboard_payload([_entry(None)])
    assert payload["equity_curve"] == []
    assert payload["asset_heatmap"] == []


def test_dashboard_payload_downsamples_equity_and_keeps_last_point() -> None:
    summary = {
        "equity_curve": [
            {"timestamp": index, "value": float(index)}
            for index in range(1200)
        ]
    }
    payload = _build_dashboard_payload([_entry(summary)])
    points = payload["equity_curve"]

    assert len(points) <= 600
    assert points[-1]["timestamp"] == "1970-01-01T00:19:59+00:00"
    assert points[-1]["value"] == pytest.approx(1199.0)
