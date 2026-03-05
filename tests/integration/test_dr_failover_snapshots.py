from __future__ import annotations

import json
from pathlib import Path

from bot_core.observability.dr_failover import compare_last_error, evaluate_outcome
from scripts.dr_failover_validation import _extract_state, _state_to_dict, _write_state_snapshot


def test_failover_snapshots_capture_rehydration_and_rule_sync(tmp_path: Path) -> None:
    before_payload = {
        "cloud": {"_lastError": "primary_region_down"},
        "prometheus": {"samples": {"rulesDigest": "digest-123"}},
        "alertmanager": {"firing": 2},
    }
    after_payload = {
        "cloud": {"_lastError": "primary_region_down"},
        "prometheus": {"samples": {"rulesDigest": "digest-123"}},
        "alertmanager": {"firing": 2},
    }

    before_state = _extract_state(before_payload)
    after_state = _extract_state(after_payload)

    prefix = tmp_path / "weekly_failover"
    before_path = _write_state_snapshot(
        prefix.with_name(f"{prefix.name}_before.json"), before_state
    )
    after_path = _write_state_snapshot(prefix.with_name(f"{prefix.name}_after.json"), after_state)

    comparison = compare_last_error(before_state, after_state)
    status, reasons = evaluate_outcome(comparison, before_ok=True, after_ok=True)

    assert status == "pass"
    assert reasons == []
    assert comparison.rehydrated is True
    assert comparison.rules_in_sync is True

    stored_before = json.loads(before_path.read_text())
    stored_after = json.loads(after_path.read_text())
    summary_states = {
        "before": _state_to_dict(before_state),
        "after": _state_to_dict(after_state),
    }

    assert stored_before["lastError"] == "primary_region_down"
    assert stored_after["lastError"] == "primary_region_down"
    assert stored_before["rulesDigest"] == stored_after["rulesDigest"] == "digest-123"
    assert stored_before["firingAlerts"] == stored_after["firingAlerts"] == 2
    assert summary_states == {
        "before": {
            "lastError": "primary_region_down",
            "rulesDigest": "digest-123",
            "firingAlerts": 2,
        },
        "after": {
            "lastError": "primary_region_down",
            "rulesDigest": "digest-123",
            "firingAlerts": 2,
        },
    }
