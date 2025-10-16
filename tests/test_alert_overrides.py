from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from bot_core.observability.alert_overrides import (
    AlertOverrideBuilder,
    AlertOverrideManager,
    load_overrides_document,
)
from bot_core.observability.slo import SLODefinition, SLOStatus


def _utcnow() -> datetime:
    return datetime.now(timezone.utc).replace(microsecond=0)


def test_builder_generates_override_with_metadata() -> None:
    definition = SLODefinition(
        name="latency",
        indicator="latency_p95",
        target=0.5,
        comparison="<=",
        warning_threshold=0.45,
        severity="critical",
        description="Latency budget",
        tags=("stage6", "latency"),
    )
    status = SLOStatus(
        name="latency",
        indicator="latency_p95",
        value=0.72,
        target=0.5,
        comparison="<=",
        status="breach",
        severity="critical",
        warning_threshold=0.45,
        error_budget_pct=0.44,
        window_start=_utcnow() - timedelta(hours=1),
        window_end=_utcnow(),
        sample_size=1440,
        reason="Latencja powyżej budżetu",
        metadata={"latency_p95": 0.72},
    )

    builder = AlertOverrideBuilder({"latency": definition})
    overrides = builder.build_from_statuses(
        {"latency": status},
        include_warning=True,
        default_ttl=timedelta(minutes=30),
        requested_by="SRE",
        source="slo_monitor",
        extra_tags=("noc",),
    )

    assert len(overrides) == 1
    override = overrides[0]
    assert override.alert == "latency"
    assert override.indicator == "latency_p95"
    assert override.severity == "critical"
    assert override.metadata["error_budget_pct"] == pytest.approx(0.44)
    assert set(override.tags) == {"stage6", "latency", "noc"}
    assert override.expires_at is not None
    assert override.expires_at - override.created_at == timedelta(minutes=30)


def test_manager_payload_and_loading_handles_expiry() -> None:
    definition = SLODefinition(
        name="throughput",
        indicator="orders_per_minute",
        target=120,
        comparison=">=",
        warning_threshold=150,
        severity="warning",
    )
    status = SLOStatus(
        name="throughput",
        indicator="orders_per_minute",
        value=130,
        target=120,
        comparison=">=",
        status="warning",
        severity="warning",
        warning_threshold=150,
        window_start=_utcnow() - timedelta(minutes=30),
        window_end=_utcnow(),
        sample_size=300,
        reason="Spadek throughput",
    )
    builder = AlertOverrideBuilder({"throughput": definition})
    overrides = builder.build_from_statuses(
        {"throughput": status},
        default_ttl=timedelta(minutes=1),
        include_warning=True,
        requested_by="Ops",
    )

    manager = AlertOverrideManager(overrides)
    reference = _utcnow()
    payload = manager.to_payload(reference=reference)
    assert payload["schema"].startswith("stage6.observability")
    assert payload["summary"]["active"] == 1
    annotations = payload["annotations"]
    assert annotations and annotations[0]["alert"] == "throughput"

    # dokument można ponownie załadować
    loaded = load_overrides_document(payload)
    assert len(loaded) == 1

    # po wygaśnięciu override nie jest aktywny
    manager.prune_expired(reference=reference + timedelta(minutes=2))
    assert manager.summary(reference=reference + timedelta(minutes=2))["active"] == 0
