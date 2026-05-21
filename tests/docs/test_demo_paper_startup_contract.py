from __future__ import annotations

from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
DEMO_PAPER_PATH = REPO_ROOT / "config/e2e/demo_paper.yml"


def test_demo_paper_config_exists() -> None:
    assert DEMO_PAPER_PATH.exists(), f"Brak pliku konfiguracji: {DEMO_PAPER_PATH}"


def test_demo_paper_config_enforces_safe_flags() -> None:
    payload = yaml.safe_load(DEMO_PAPER_PATH.read_text(encoding="utf-8"))

    trading = payload.get("trading") or {}
    execution = payload.get("execution") or {}
    live = execution.get("live") or {}

    assert trading.get("enable_paper_mode") is True
    assert trading.get("enable_live_mode") is False
    assert execution.get("default_mode") == "paper"
    assert execution.get("force_paper_when_offline") is True
    assert live.get("enabled") is False
