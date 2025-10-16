from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone

import pytest

from bot_core.observability.alert_overrides import AlertOverride
from bot_core.observability.dashboard_sync import (
    build_dashboard_annotations_payload,
    load_dashboard_definition,
    load_overrides_from_document,
)


def test_build_dashboard_annotations_payload_filters_inactive() -> None:
    reference = datetime(2024, 1, 1, 12, 0, tzinfo=timezone.utc)
    active = AlertOverride(
        alert="latency",
        status="breach",
        severity="critical",
        reason="Latency przekracza 200ms",
        indicator="latency",
        created_at=reference - timedelta(minutes=10),
        expires_at=reference + timedelta(minutes=50),
        tags=("stage6",),
        metadata={"error_budget_pct": 12.5},
    )
    expired = AlertOverride(
        alert="availability",
        status="warning",
        severity="warning",
        reason="Spadek dostępności",
        indicator="availability",
        created_at=reference - timedelta(hours=2),
        expires_at=reference - timedelta(minutes=1),
    )

    payload = build_dashboard_annotations_payload(
        [active, expired],
        reference=reference,
        dashboard_uid="stage6-resilience-ops",
        panel_id=1,
    )

    assert payload["schema"] == "stage6.observability.dashboard_annotations"
    assert payload["dashboard_uid"] == "stage6-resilience-ops"
    assert payload["panel_id"] == 1
    annotations = payload["annotations"]
    assert len(annotations) == 1
    annotation = annotations[0]
    assert annotation["dashboardUid"] == "stage6-resilience-ops"
    assert annotation["panelId"] == 1
    assert annotation["tags"] == [
        "override",
        "breach",
        "critical",
        "latency",
        "stage6",
    ]
    assert annotation["data"]["alert"] == "latency"
    assert annotation["time"] == int(active.created_at.timestamp() * 1000)
    assert annotation["timeEnd"] == int(active.expires_at.timestamp() * 1000)


def test_load_dashboard_definition(tmp_path) -> None:
    dashboard_path = tmp_path / "dashboard.json"
    dashboard_path.write_text(json.dumps({"uid": "stage6", "title": "Stage6"}), encoding="utf-8")
    definition = load_dashboard_definition(dashboard_path)
    assert definition.uid == "stage6"
    assert definition.payload["title"] == "Stage6"


def test_load_overrides_from_document_validates_schema() -> None:
    payload = {
        "schema": "stage6.observability.alert_overrides",
        "overrides": [
            {
                "alert": "latency",
                "status": "breach",
                "severity": "critical",
                "created_at": "2024-01-01T12:00:00Z",
            }
        ],
    }

    overrides = load_overrides_from_document(payload)
    assert len(overrides) == 1
    assert overrides[0].alert == "latency"


def test_load_overrides_from_document_requires_schema() -> None:
    with pytest.raises(ValueError):
        load_overrides_from_document({"schema": "invalid", "overrides": []})


def test_load_overrides_from_document_requires_iterable() -> None:
    with pytest.raises(ValueError):
        load_overrides_from_document({"schema": "stage6.observability.alert_overrides", "overrides": None})

