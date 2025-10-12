from datetime import datetime, timezone

import pytest

from bot_core.observability.metrics import MetricsRegistry
from bot_core.runtime.risk_metrics import RiskMetricsExporter
from bot_core.runtime.risk_service import RiskExposure, RiskSnapshot


def _base_labels() -> dict[str, str]:
    return {"environment": "paper", "stage": "demo", "profile": "balanced"}


def _make_snapshot(*exposures: RiskExposure) -> RiskSnapshot:
    return RiskSnapshot(
        profile_name="balanced",
        portfolio_value=10_000.0,
        current_drawdown=0.03,
        daily_loss=0.01,
        used_leverage=1.25,
        exposures=exposures,
        generated_at=datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc),
        force_liquidation=False,
        metadata={
            "risk_profile_summary": {
                "name": "balanced",
                "severity_min": "warning",
                "extends_chain": ["conservative", "ops"],
            }
        },
    )


def test_risk_metrics_exporter_updates_gauges() -> None:
    registry = MetricsRegistry()
    exporter = RiskMetricsExporter(registry, environment="paper", stage="demo")

    exposures = (
        RiskExposure(code="daily_loss_pct", current=0.01, maximum=0.03, threshold=0.02),
        RiskExposure(code="active_positions", current=2.0, maximum=5.0, threshold=4.0),
    )
    snapshot = _make_snapshot(*exposures)

    exporter.observe(snapshot)

    base_labels = _base_labels()
    portfolio_value = registry.get("risk_portfolio_value").value(labels=base_labels)
    assert portfolio_value == pytest.approx(10_000.0)

    severity = registry.get("risk_profile_min_severity").value(labels=base_labels)
    assert severity == pytest.approx(3.0)

    chain_length = registry.get("risk_profile_extends_chain_length").value(labels=base_labels)
    assert chain_length == pytest.approx(2.0)

    timestamp = registry.get("risk_snapshot_generated_at").value(labels=base_labels)
    assert timestamp == pytest.approx(snapshot.generated_at.timestamp())

    loss_labels = dict(base_labels)
    loss_labels["limit"] = "daily_loss_pct"
    ratio = registry.get("risk_exposure_ratio").value(labels=loss_labels)
    assert ratio == pytest.approx(0.01 / 0.03)
    threshold = registry.get("risk_exposure_threshold").value(labels=loss_labels)
    assert threshold == pytest.approx(0.02)

    positions_labels = dict(base_labels)
    positions_labels["limit"] = "active_positions"
    current_positions = registry.get("risk_exposure_current").value(labels=positions_labels)
    assert current_positions == pytest.approx(2.0)


def test_risk_metrics_exporter_resets_missing_limits() -> None:
    registry = MetricsRegistry()
    exporter = RiskMetricsExporter(registry, environment="paper", stage="demo")

    first_snapshot = _make_snapshot(
        RiskExposure(code="daily_loss_pct", current=0.02, maximum=0.04, threshold=0.03)
    )
    exporter.observe(first_snapshot)

    second_snapshot = _make_snapshot(
        RiskExposure(code="active_positions", current=1.0, maximum=4.0, threshold=3.0)
    )
    exporter.observe(second_snapshot)

    base_labels = _base_labels()
    loss_labels = dict(base_labels)
    loss_labels["limit"] = "daily_loss_pct"
    assert registry.get("risk_exposure_current").value(labels=loss_labels) == pytest.approx(0.0)
    assert registry.get("risk_exposure_max").value(labels=loss_labels) == pytest.approx(0.0)
    assert registry.get("risk_exposure_ratio").value(labels=loss_labels) == pytest.approx(0.0)

    positions_labels = dict(base_labels)
    positions_labels["limit"] = "active_positions"
    assert registry.get("risk_exposure_current").value(labels=positions_labels) == pytest.approx(1.0)
    assert registry.get("risk_exposure_max").value(labels=positions_labels) == pytest.approx(4.0)
