"""Testy porównywania snapshotów strategii."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.analysis.compare_snapshots import compare_snapshots


def _write_snapshot(directory: Path, payload: dict[str, object], filename: str = "snapshot.json") -> None:
    directory.mkdir(parents=True, exist_ok=True)
    (directory / filename).write_text(json.dumps(payload), encoding="utf-8")


@pytest.fixture()
def sample_payload() -> dict[str, object]:
    return {
        "strategies": {
            "alpha": {"total_return": 0.11, "max_drawdown": 0.04, "trades": 42},
            "beta": {"total_return": 0.05, "max_drawdown": 0.02, "trades": 12},
        }
    }


def test_compare_snapshots_within_tolerance(tmp_path: Path, sample_payload: dict[str, object]) -> None:
    legacy_dir = tmp_path / "legacy"
    async_dir = tmp_path / "async"

    _write_snapshot(legacy_dir, sample_payload)

    async_payload = {
        "strategies": {
            "alpha": {"total_return": 0.112, "max_drawdown": 0.0405, "trades": 42},
            "beta": {"total_return": 0.051, "max_drawdown": 0.0205, "trades": 12},
        }
    }
    _write_snapshot(async_dir, async_payload)

    result = compare_snapshots(
        legacy_dir,
        async_dir,
        relative_tolerance=0.05,
        absolute_tolerance=1e-4,
    )

    assert result.is_within_tolerance
    assert not result.deviations


def test_compare_snapshots_detects_large_deviation(tmp_path: Path, sample_payload: dict[str, object]) -> None:
    legacy_dir = tmp_path / "legacy"
    async_dir = tmp_path / "async"

    _write_snapshot(legacy_dir, sample_payload)

    async_payload = {
        "strategies": {
            "alpha": {"total_return": 0.2, "max_drawdown": 0.04, "trades": 42},
            "beta": {"total_return": 0.05, "max_drawdown": 0.02, "trades": 12},
        }
    }
    _write_snapshot(async_dir, async_payload)

    result = compare_snapshots(
        legacy_dir,
        async_dir,
        relative_tolerance=0.05,
        absolute_tolerance=1e-4,
    )

    assert not result.is_within_tolerance
    assert result.deviations
    deviation = result.deviations[0]
    assert deviation.strategy == "alpha"
    assert deviation.metric == "total_return"
