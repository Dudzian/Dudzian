from ui.backend.runtime_service import RuntimeService


class _RunnerStub:
    def __init__(self) -> None:
        self.cycles = 0
        self.until_calls = 0

    def run_cycle(self, regime=None):  # noqa: ANN001 - interfejs testowy
        self.cycles += 1
        return {"mode": regime or "manual"}

    def run_until(self, **kwargs):  # noqa: ANN003 - interfejs testowy
        self.until_calls += 1
        self.cycles += kwargs.get("limit", 1)
        return ({"mode": "auto"},)

    def snapshot(self):
        return {
            "history": [
                {
                    "mode": "hedge",
                    "timestamp": "2024-01-01T00:00:00Z",
                    "decision": {"state": "executed"},
                }
            ]
        }


def test_manual_cycle_updates_snapshot_with_runner() -> None:
    runner = _RunnerStub()
    service = RuntimeService(
        decision_loader=lambda limit: [],
        ai_governor_loader=lambda: {},
        ai_runner_factory=lambda: runner,
    )

    assert service.executionMode == "manual"
    assert service.runManualCycle() is True
    assert runner.cycles >= 1
    assert service.aiGovernorSnapshot.get("history")


def test_auto_mode_respects_sla_guardrail_and_runner() -> None:
    runner = _RunnerStub()
    service = RuntimeService(
        decision_loader=lambda limit: [],
        ai_governor_loader=lambda: {},
        ai_runner_factory=lambda: runner,
    )

    service._feed_sla_report = {"sla_state": "critical"}
    assert service.setExecutionMode("auto") is False
    assert service.executionMode == "manual"
    assert runner.until_calls == 0
    assert service.lastOperatorAction.get("action") == "guardrail_block"

    service._feed_sla_report = {"sla_state": "ok"}
    assert service.setExecutionMode("auto") is True
    assert runner.until_calls == 1
    assert service.executionMode == "auto"


def test_exposure_guardrail_blocks_manual_cycle() -> None:
    runner = _RunnerStub()
    service = RuntimeService(
        decision_loader=lambda limit: [],
        ai_governor_loader=lambda: {},
        ai_runner_factory=lambda: runner,
    )
    service._risk_metrics = {"exposure": 0.9}

    assert service.runManualCycle() is False
    assert service.lastOperatorAction.get("action") == "guardrail_block"
    assert "ekspozycja" in service.lastOperatorAction.get("entry", {}).get("reason", "")
