import json
from pathlib import Path

from bot_core.resilience.drill import FailoverDrillSummary, FailoverServiceResult
from bot_core.resilience.self_healing import (
    SelfHealingExecution,
    build_self_healing_plan,
    execute_self_healing_plan,
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

