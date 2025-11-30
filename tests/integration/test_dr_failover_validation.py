from __future__ import annotations

from bot_core.observability.dr_failover import (
    FailoverState,
    compare_last_error,
    evaluate_outcome,
)


def test_failover_comparison_covers_rehydration_and_rule_sync() -> None:
    before = FailoverState(last_error="primary_region_down", rules_digest="abc123", firing_alerts=2)
    after = FailoverState(last_error="primary_region_down", rules_digest="abc123", firing_alerts=2)

    comparison = compare_last_error(before, after)

    assert comparison.rehydrated is True
    assert comparison.regression is False
    assert comparison.cleared is False
    assert comparison.rules_in_sync is True
    assert comparison.unchanged is True
    assert comparison.alerts_missing is False


def test_failover_comparison_detects_regression_and_clearance() -> None:
    before = FailoverState(last_error=None, rules_digest="digest-old")
    after = FailoverState(last_error="new_error", rules_digest="digest-new")

    regression = compare_last_error(before, after)
    assert regression.regression is True
    assert regression.cleared is False
    assert regression.rules_in_sync is False
    assert regression.alerts_missing is False

    cleared = compare_last_error(after, before)
    assert cleared.cleared is True
    assert cleared.regression is False
    assert cleared.alerts_missing is False


def test_failover_comparison_marks_changed_last_error_as_regression() -> None:
    before = FailoverState(last_error="old_error", rules_digest="digest")
    after = FailoverState(last_error="new_error", rules_digest="digest")

    comparison = compare_last_error(before, after)

    assert comparison.regression is True
    assert comparison.rehydrated is False
    assert comparison.rules_in_sync is True


def test_failover_comparison_marks_missing_alerts_when_rehydrated() -> None:
    before = FailoverState(last_error="primary_region_down", rules_digest="abc123", firing_alerts=3)
    after = FailoverState(last_error="primary_region_down", rules_digest="abc123", firing_alerts=0)

    comparison = compare_last_error(before, after)

    assert comparison.rehydrated is True
    assert comparison.alerts_missing is True
    assert comparison.rules_in_sync is True
    assert comparison.regression is False


def test_evaluate_outcome_reports_failure_reasons() -> None:
    baseline = compare_last_error(
        FailoverState(last_error="primary_region_down", rules_digest="digest", firing_alerts=2),
        FailoverState(last_error="primary_region_down", rules_digest="digest", firing_alerts=2),
    )

    status, reasons = evaluate_outcome(baseline, before_ok=True, after_ok=True)
    assert status == "pass"
    assert reasons == []

    regression = compare_last_error(
        FailoverState(last_error=None, rules_digest="digestA"),
        FailoverState(last_error="new_error", rules_digest="digestB"),
    )

    status, reasons = evaluate_outcome(regression, before_ok=False, after_ok=True)
    assert status == "fail"
    assert reasons == ["regression_last_error", "rules_drift", "before_probe_failed"]

    status, reasons = evaluate_outcome(regression, before_ok=True, after_ok=False)
    assert status == "fail"
    assert reasons == ["regression_last_error", "rules_drift", "after_probe_failed"]
