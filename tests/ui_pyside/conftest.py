"""Shared PySide UI test hygiene fixtures."""

from __future__ import annotations

from pathlib import Path

import pytest


@pytest.fixture(autouse=True)
def ui_generated_artifacts_path(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep UI-generated diagnostics/layout artifacts out of tracked repo paths."""

    monkeypatch.setenv(
        "BOT_CORE_UI_FEED_LATENCY_PATH",
        str(tmp_path / "reports" / "ci" / "decision_feed_metrics.json"),
    )
    monkeypatch.setenv("BOT_CORE_UI_LAYOUTS_PATH", str(tmp_path / "var" / "ui_layouts.json"))
