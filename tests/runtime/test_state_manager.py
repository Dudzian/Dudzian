from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pytest

from bot_core.runtime.state_manager import RuntimeStateError, RuntimeStateManager


def test_record_and_load_checkpoint(tmp_path: Path) -> None:
    manager = RuntimeStateManager(tmp_path)
    checkpoint = manager.record_checkpoint(
        entrypoint="demo_desktop",
        mode="demo",
        config_path="config/runtime.yaml",
        metadata={"foo": "bar"},
    )

    assert checkpoint.entrypoint == "demo_desktop"
    assert checkpoint.mode == "demo"
    assert checkpoint.metadata["foo"] == "bar"

    loaded = manager.load_checkpoint()
    assert loaded is not None
    assert loaded.entrypoint == "demo_desktop"
    assert loaded.mode == "demo"
    assert isinstance(loaded.created_at, datetime)
    assert loaded.created_at.tzinfo is not None


def test_require_checkpoint_validates_mode_and_entrypoint(tmp_path: Path) -> None:
    manager = RuntimeStateManager(tmp_path)
    manager.record_checkpoint(entrypoint="demo_desktop", mode="demo", config_path="config/runtime.yaml", metadata={})

    checkpoint = manager.require_checkpoint(target_mode="paper", entrypoint="demo_desktop")
    assert checkpoint.entrypoint == "demo_desktop"

    with pytest.raises(RuntimeStateError):
        manager.require_checkpoint(target_mode="paper", entrypoint="other")

    manager.record_checkpoint(entrypoint="demo_desktop", mode="paper", config_path="config/runtime.yaml", metadata={})
    with pytest.raises(RuntimeStateError):
        manager.require_checkpoint(target_mode="paper", entrypoint="demo_desktop")


def test_clear_removes_checkpoint(tmp_path: Path) -> None:
    manager = RuntimeStateManager(tmp_path)
    manager.record_checkpoint(entrypoint="demo_desktop", mode="demo", config_path="config/runtime.yaml", metadata={})
    manager.clear()
    assert manager.load_checkpoint() is None
