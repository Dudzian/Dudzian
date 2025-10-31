from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest


from bot_core.security.rotation import RotationRegistry


@pytest.fixture()
def registry_path(tmp_path: Path) -> Path:
    return tmp_path / "rotation.json"


def test_rotation_registry_persists_entries(registry_path: Path) -> None:
    registry = RotationRegistry(registry_path)
    timestamp = datetime(2024, 1, 1, tzinfo=timezone.utc)

    registry.mark_rotated("binance_paper", "trading", timestamp=timestamp)

    reloaded = RotationRegistry(registry_path)
    status = reloaded.status("binance_paper", "trading", now=timestamp + timedelta(days=1))

    assert status.last_rotated == timestamp
    assert pytest.approx(status.days_since_rotation or 0.0, rel=1e-6) == 1.0
    assert status.is_due is False


def test_rotation_registry_reports_missing_as_due(registry_path: Path) -> None:
    registry = RotationRegistry(registry_path)

    status = registry.status("kraken_live", "trading", now=datetime(2024, 5, 1, tzinfo=timezone.utc))

    assert status.last_rotated is None
    assert status.is_due is True
    assert status.is_overdue is True


def test_rotation_registry_detects_overdue(registry_path: Path) -> None:
    registry = RotationRegistry(registry_path)
    rotated_at = datetime(2023, 1, 1, tzinfo=timezone.utc)
    registry.mark_rotated("binance_live", "trading", timestamp=rotated_at)

    now = rotated_at + timedelta(days=120)
    status = registry.status("binance_live", "trading", interval_days=90.0, now=now)

    assert status.is_due is True
    assert status.is_overdue is True
    assert status.due_in_days < 0


def test_due_within_filters_entries(registry_path: Path) -> None:
    registry = RotationRegistry(registry_path)
    rotated_recently = datetime.now(timezone.utc) - timedelta(days=10)
    rotated_old = datetime.now(timezone.utc) - timedelta(days=89)

    registry.mark_rotated("binance_futures", "trading", timestamp=rotated_recently)
    registry.mark_rotated("kraken_paper", "trading", timestamp=rotated_old)

    statuses = list(
        registry.due_within(interval_days=90.0, warn_within_days=14.0, now=datetime.now(timezone.utc))
    )

    keys = {status.key for status in statuses}
    assert keys == {"kraken_paper"}

