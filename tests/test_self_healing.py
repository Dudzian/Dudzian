from __future__ import annotations

import json
from pathlib import Path

from bot_core.resilience.drill import FailoverDrillSummary, FailoverServiceResult
from bot_core.resilience.self_healing import (
    CompositeSelfHealingExecutor,
    SelfHealingAction,
    SelfHealingExecution,
    build_self_healing_plan,
    execute_self_healing_plan,
    default_self_healing_executor,
    load_self_healing_rules,
    summarize_self_healing_plan,
    write_self_healing_report,
    write_self_healing_signature,
)


def _make_summary() -> FailoverDrillSummary:
    service = FailoverServiceResult(
        name="risk-service",
        status="failed",
        max_rto_minutes=15,
        observed_rto_minutes=30,
        max_rpo_minutes=10,
        observed_rpo_minutes=5,
        missing_artifacts=("runbooks/*.md",),
        matched_artifacts=("runbooks/sre.md",),
        issues=("RTO 30 min przekracza limit 15 min",),
        metadata={"owner": "sre"},
    )
    return FailoverDrillSummary(
        drill_name="demo-drill",
        executed_at="2024-05-01T12:00:00Z",
        generated_at="2024-05-01T12:05:00Z",
        services=(service,),
        status="failed",
        counts={"total": 1, "ok": 0, "warning": 0, "failed": 1},
        metadata={"region": "local"},
        bundle_audit=None,
    )


def _make_action(module: str = "runtime.risk_service", **overrides) -> SelfHealingAction:
    metadata = overrides.pop("metadata", {"scope": "risk"})
    return SelfHealingAction(
        service=overrides.pop("service", "risk-service"),
        service_status=overrides.pop("service_status", "failed"),
        rule_name=overrides.pop("rule_name", "demo-rule"),
        module=module,
        command=overrides.pop("command", None),
        delay_seconds=overrides.pop("delay_seconds", 0.0),
        severity=overrides.pop("severity", "critical"),
        tags=overrides.pop("tags", ("resilience",)),
        metadata=metadata,
        reason=overrides.pop("reason", "status=failed"),
        issues=overrides.pop("issues", ("RTO 30 min przekracza limit 15 min",)),
    )


def test_load_rules_and_plan(tmp_path: Path) -> None:
    config_path = tmp_path / "self_heal.json"
    config_path.write_text(
        json.dumps(
            {
                "rules": [
                    {
                        "name": "risk-restart",
                        "service_pattern": "risk-*",
                        "statuses": ["failed", "warning"],
                        "severity": "critical",
                        "tags": ["resilience"],
                        "metadata": {"owner": "resilience-team"},
                        "actions": [
                            {
                                "module": "runtime.risk_service",
                                "command": ["python", "-c", "print('restart risk')"],
                                "delay_seconds": 0.0,
                                "tags": ["restart"],
                                "metadata": {"scope": "risk"},
                            }
                        ],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    rules = load_self_healing_rules(config_path)
    summary = _make_summary()
    plan = build_self_healing_plan(summary, rules)
    assert plan.drill_name == "demo-drill"
    assert len(plan.actions) == 1
    action = plan.actions[0]
    assert action.service == "risk-service"
    assert action.command == ("python", "-c", "print('restart risk')")
    assert "resilience" in action.tags
    assert action.metadata["owner"] == "resilience-team"


def test_execute_plan_and_sign(tmp_path: Path) -> None:
    config_path = tmp_path / "self_heal.json"
    config_path.write_text(
        json.dumps(
            {
                "rules": [
                    {
                        "service_pattern": "risk-service",
                        "actions": [
                            {
                                "module": "runtime.risk_service",
                                "command": ["python", "-c", "print('restart risk')"],
                            }
                        ],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    rules = load_self_healing_rules(config_path)
    summary = _make_summary()
    plan = build_self_healing_plan(summary, rules)

    class StubExecutor:
        def __init__(self) -> None:
            self.calls = 0

        def __call__(self, action):
            self.calls += 1
            return SelfHealingExecution(
                action=action,
                status="success",
                started_at="2024-05-01T12:10:00Z",
                completed_at="2024-05-01T12:10:01Z",
                exit_code=0,
                output="restart risk\n",
                error=None,
                notes=None,
            )

    executor = StubExecutor()
    report = execute_self_healing_plan(plan, executor, sleep=lambda _delay: None)
    assert executor.calls == len(plan.actions)
    assert report.mode == "execute"
    assert report.status == ("success" if plan.actions else "noop")
    assert report.actions[0].status == "success"

    plan_report = summarize_self_healing_plan(plan)
    assert plan_report.mode == "plan"
    assert plan_report.actions[0].status == "planned"

    output_path = tmp_path / "self_heal_report.json"
    payload = write_self_healing_report(report, output_path)
    assert payload["schema"] == "stage6.resilience.self_healing.report"
    key_path = tmp_path / "hmac.key"
    key_path.write_bytes(b"secret-key")
    signature_path = tmp_path / "self_heal_report.sig"
    signature = write_self_healing_signature(
        payload,
        signature_path,
        key=b"secret-key",
        key_id="local",
        target=output_path.name,
    )
    assert signature["schema"] == "stage6.resilience.self_healing.report.signature"


def test_composite_executor_uses_registered_handler() -> None:
    action = _make_action(module="custom.handler")
    calls: list[str] = []

    def handler(target: SelfHealingAction) -> SelfHealingExecution:
        calls.append(target.module)
        return SelfHealingExecution(
            action=target,
            status="success",
            started_at="2024-05-20T10:00:00Z",
            completed_at="2024-05-20T10:00:01Z",
            exit_code=0,
            output="handler executed",
            error=None,
        )

    executor = CompositeSelfHealingExecutor(handlers={"custom.handler": handler})
    result = executor(action)

    assert calls == ["custom.handler"]
    assert result.status == "success"


def test_composite_executor_falls_back_to_subprocess() -> None:
    action = _make_action(module="runtime.restart")
    invoked: list[str] = []

    def fallback(target: SelfHealingAction) -> SelfHealingExecution:
        invoked.append(target.module)
        return SelfHealingExecution(
            action=target,
            status="skipped",
            started_at="2024-05-20T10:05:00Z",
            completed_at="2024-05-20T10:05:00Z",
            exit_code=None,
            output=None,
            error=None,
        )

    executor = CompositeSelfHealingExecutor(subprocess_executor=fallback)
    result = executor(action)

    assert invoked == ["runtime.restart"]
    assert result.status == "skipped"


def test_composite_executor_traps_handler_exception() -> None:
    action = _make_action(module="custom.handler")

    def handler(_: SelfHealingAction) -> SelfHealingExecution:  # pragma: no cover - exception path
        raise RuntimeError("boom")

    executor = CompositeSelfHealingExecutor(handlers={"custom.handler": handler})
    result = executor(action)

    assert result.status == "error"
    assert result.notes == "handler_exception"
    assert result.error == "boom"


def test_default_executor_reports_missing_disable_multi_strategy(monkeypatch) -> None:
    from bot_core.resilience import self_healing as module

    monkeypatch.setattr(module, "_disable_multi_strategy", None, raising=False)
    executor = default_self_healing_executor()

    action = _make_action(module="scripts.disable_multi_strategy", metadata={"scope": "governor"})
    result = executor(action)

    assert result.status == "error"
    assert result.notes == "module_not_found"


def test_default_executor_invokes_disable_multi_strategy(monkeypatch) -> None:
    from bot_core.resilience import self_healing as module

    class StubDisable:
        def __init__(self) -> None:
            self.calls: list[list[str]] = []

        def run(self, args: list[str]) -> int:
            self.calls.append(list(args))
            return 0

    stub = StubDisable()
    monkeypatch.setattr(module, "_disable_multi_strategy", stub, raising=False)

    executor = default_self_healing_executor()
    action = _make_action(
        module="scripts.disable_multi_strategy",
        metadata={"scope": "portfolio", "requested_by": "ui", "duration_minutes": "15"},
    )

    result = executor(action)

    assert result.status == "success"
    assert result.output == "component=portfolio"
    assert stub.calls
    assert stub.calls[0][0:2] == ["--component", "portfolio"]
    assert "--requested-by" in stub.calls[0]

