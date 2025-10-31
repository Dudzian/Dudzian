from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pytest

from core.reporting import DemoPaperReport


@pytest.fixture
def payload() -> dict[str, object]:
    return {
        "mode": "paper",
        "started_at": datetime(2024, 5, 1, 10, 0, tzinfo=timezone.utc).isoformat(),
        "finished_at": datetime(2024, 5, 1, 10, 5, tzinfo=timezone.utc).isoformat(),
        "status": "success",
        "entrypoint": "demo_desktop",
        "validation": {
            "environment": "DemoEnv",
            "expected_environment": "paper",
            "symbols": ["BTCUSDT"],
        },
        "errors": [],
        "warnings": ["Ograniczona konfiguracja konta"],
    }


def test_demo_paper_report_creates_markdown(tmp_path: Path, payload: dict[str, object]) -> None:
    decision_events = (
        {"event": "trade_executed", "symbol": "BTCUSDT"},
        {"event": "trade_executed", "symbol": "ETHUSDT"},
    )

    report = DemoPaperReport.from_payload(payload, decision_events=decision_events)

    assert report.kpi["decision_events"] == 2
    assert report.steps[0].status == "success"
    assert report.steps[1].details["required"] is True

    markdown_path = report.write_markdown(tmp_path)
    content = markdown_path.read_text(encoding="utf-8")

    assert "## Podsumowanie KPI" in content
    assert "Liczba zdarzeń tradingowych" in content
    assert "2" in content


def test_demo_paper_report_handles_missing_timestamps(payload: dict[str, object]) -> None:
    payload.pop("started_at", None)
    payload["finished_at"] = "invalid"

    report = DemoPaperReport.from_payload(payload, decision_events=())

    assert report.kpi["duration_seconds"] is None
    assert report.steps[0].status == "success"
