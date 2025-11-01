import asyncio
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import tests._pathbootstrap  # noqa: F401  # pylint: disable=unused-import

from core.compliance import ComplianceAuditResult, ComplianceFinding
from core.runtime.compliance_scheduler import (
    ComplianceAuditRunOutcome,
    ComplianceScheduleSettings,
    ComplianceScheduler,
)


def _result_template(passed: bool = False) -> ComplianceAuditResult:
    finding = ComplianceFinding(
        rule_id="KYC_MISSING_FIELDS",
        severity="high",
        message="Brak wymaganych pól",
        metadata={"missing_fields": ("address",)},
    )
    return ComplianceAuditResult(
        generated_at=datetime(2025, 1, 1, 8, 0, tzinfo=timezone.utc),
        passed=passed,
        findings=() if passed else (finding,),
        context_summary={"strategy": "demo"},
        config_path=Path("config/compliance/audit.yml"),
    )


def test_compliance_scheduler_runs_within_window() -> None:
    async def scenario() -> None:
        timestamps = [datetime(2025, 1, 1, 8, 0, tzinfo=timezone.utc)]
        events: list[Any] = []
        settings = ComplianceScheduleSettings.from_mapping(
            {
                "enabled": True,
                "interval_hours": 12,
                "window": {"start": "06:00", "end": "22:00"},
            }
        )
        scheduler = ComplianceScheduler(
            settings=settings,
            clock=lambda: timestamps[-1],
            event_publisher=events.append,
        )

        async def callback() -> ComplianceAuditResult:
            return _result_template(passed=False)

        outcome = await scheduler.run_once(callback)
        assert isinstance(outcome, ComplianceAuditRunOutcome)
        assert outcome.status == "completed"
        assert outcome.result is not None
        assert scheduler.last_run == timestamps[-1]
        assert scheduler.next_run is not None
        assert scheduler.next_run - scheduler.last_run == settings.interval
        assert events, "Powinno zostać opublikowane zdarzenie zakończenia audytu"
        completion = events[-1]
        assert completion.findings_total == 1
        assert completion.severity_breakdown.get("high") == 1

    asyncio.run(scenario())


def test_compliance_scheduler_respects_window() -> None:
    async def scenario() -> None:
        timestamps = [datetime(2025, 1, 1, 1, 0, tzinfo=timezone.utc)]
        settings = ComplianceScheduleSettings.from_mapping(
            {
                "enabled": True,
                "interval_hours": 12,
                "window": {"start": "06:00", "end": "22:00"},
            }
        )
        scheduler = ComplianceScheduler(settings=settings, clock=lambda: timestamps[-1])

        async def callback() -> ComplianceAuditResult:
            return _result_template()

        outcome = await scheduler.run_once(callback)
        assert outcome.status == "skipped"
        assert outcome.reason == "outside_window"
        assert scheduler.next_run is not None
        assert scheduler.next_run.time().hour == 6

    asyncio.run(scenario())


def test_compliance_scheduler_handles_disabled() -> None:
    async def scenario() -> None:
        settings = ComplianceScheduleSettings(enabled=False)
        scheduler = ComplianceScheduler(settings=settings)

        async def callback() -> ComplianceAuditResult:
            return _result_template(passed=True)

        outcome = await scheduler.run_once(callback)
        assert outcome.status == "disabled"
        assert outcome.result is None

    asyncio.run(scenario())
