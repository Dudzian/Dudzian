from __future__ import annotations

from pathlib import Path


def test_exchange_report_schedule_is_strict_and_fetches_real_snapshots() -> None:
    workflow = Path('.github/workflows/exchange-report.yml').read_text(encoding='utf-8')

    assert 'schedule)' in workflow
    assert 'echo "LONG_POLL_MODE=strict"' in workflow
    assert "if: github.event_name == 'schedule'" in workflow
    assert 'python scripts/fetch_long_poll_snapshots.py' in workflow
    assert '--skip-freshness' in workflow


def test_exchange_report_dispatch_remains_fixture_mode() -> None:
    workflow = Path('.github/workflows/exchange-report.yml').read_text(encoding='utf-8')

    assert 'workflow_dispatch)' in workflow
    assert 'echo "LONG_POLL_MODE=fixture"' in workflow
    assert '--allow-stale-long-poll' in workflow
