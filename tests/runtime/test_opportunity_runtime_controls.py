from bot_core.runtime.opportunity_runtime_controls import OpportunityRuntimeControls


def test_controls_update_increments_revision_and_normalizes_mode() -> None:
    controls = OpportunityRuntimeControls(
        opportunity_ai_enabled=True,
        manual_kill_switch=False,
        policy_mode="assist",
    )

    first = controls.snapshot()
    assert first.policy_mode == "assist"
    assert first.revision == 0

    second = controls.update(manual_kill_switch=True, policy_mode="LIVE")
    assert second.manual_kill_switch is True
    assert second.policy_mode == "live"
    assert second.revision == 1


def test_controls_ignore_invalid_mode_fallback_shadow() -> None:
    controls = OpportunityRuntimeControls(policy_mode="invalid")
    snapshot = controls.snapshot()
    assert snapshot.policy_mode == "shadow"
