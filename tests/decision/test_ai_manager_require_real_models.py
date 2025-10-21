from __future__ import annotations

import os

import pytest

os.environ.setdefault("BOT_CORE_MINIMAL_EXCHANGES", "1")
os.environ.setdefault("BOT_CORE_MINIMAL_CORE", "1")
os.environ.setdefault("BOT_CORE_MINIMAL_DECISION", "1")

from bot_core.ai.manager import AIManager


def test_require_real_models_raises_when_repository_empty(tmp_path) -> None:
    ai_manager = AIManager(model_dir=tmp_path / "cache")
    with pytest.raises(RuntimeError):
        ai_manager.require_real_models()
