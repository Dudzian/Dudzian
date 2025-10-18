from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

import pytest

from bot_core.runtime.tco_reporting import RuntimeTCOReporter


@pytest.fixture(name="fixed_clock")
def fixture_fixed_clock() -> Callable[[], datetime]:
    moment = datetime(2024, 1, 1, tzinfo=timezone.utc)

    def _clock() -> datetime:
        return moment

    return _clock


def _record_sample_fill(reporter: RuntimeTCOReporter) -> None:
    reporter.record_execution(
        strategy="alpha",
        risk_profile="balanced",
        instrument="BTC_USDT",
        exchange="binance",
        side="buy",
        quantity=1.5,
        executed_price=10000.0,
        reference_price=9995.0,
        commission=2.5,
        funding=0.0,
        other=0.0,
    )


def test_export_allows_clearing_events(tmp_path: Path, fixed_clock) -> None:
    reporter = RuntimeTCOReporter(
        output_dir=tmp_path,
        export_formats=("json",),
        clock=fixed_clock,
    )
    _record_sample_fill(reporter)
    assert reporter.events()

    artifacts = reporter.export(clear_events=True)

    assert artifacts is not None
    assert reporter.events() == ()
    assert reporter.last_export() == artifacts


def test_flush_respects_clear_after_export_flag(tmp_path: Path, fixed_clock) -> None:
    reporter = RuntimeTCOReporter(
        output_dir=tmp_path,
        export_formats=("json",),
        flush_events=2,
        clear_after_export=True,
        clock=fixed_clock,
    )

    _record_sample_fill(reporter)
    assert len(reporter.events()) == 1

    _record_sample_fill(reporter)

    assert reporter.events() == ()
    last_export = reporter.last_export()
    assert last_export is not None
    assert "json" in last_export
    assert last_export["json"].exists()


def test_manual_clear_events_resets_buffer(tmp_path: Path, fixed_clock) -> None:
    reporter = RuntimeTCOReporter(
        output_dir=tmp_path,
        export_formats=("json",),
        clock=fixed_clock,
    )
    _record_sample_fill(reporter)
    _record_sample_fill(reporter)
    assert len(reporter.events()) == 2

    reporter.clear_events()

    assert reporter.events() == ()


def test_json_export_contains_scheduler_summary(tmp_path: Path, fixed_clock) -> None:
    reporter = RuntimeTCOReporter(
        output_dir=tmp_path,
        export_formats=("json",),
        clock=fixed_clock,
    )
    reporter.record_execution(
        strategy="alpha",
        risk_profile="balanced",
        instrument="BTC_USDT",
        exchange="binance",
        side="buy",
        quantity=1.0,
        executed_price=10000.0,
        reference_price=10000.0,
        commission=1.0,
        metadata={"scheduler": "cron.daily"},
    )

    artifacts = reporter.export()

    assert artifacts is not None
    json_payload = json.loads(artifacts["json"].read_text(encoding="utf-8"))
    assert json_payload["metadata"]["scheduler_count"] == 1
    assert "cron.daily" in json_payload["schedulers"]
