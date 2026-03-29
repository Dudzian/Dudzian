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


@pytest.mark.parametrize("target_mode", ["paper", "live"])
def test_require_checkpoint_accepts_demo_checkpoint_for_paper_and_live(
    tmp_path: Path, target_mode: str
) -> None:
    manager = RuntimeStateManager(tmp_path)
    manager.record_checkpoint(
        entrypoint="demo_desktop", mode="demo", config_path="config/runtime.yaml", metadata={}
    )

    checkpoint = manager.require_checkpoint(target_mode=target_mode, entrypoint="demo_desktop")
    assert checkpoint.entrypoint == "demo_desktop"


def test_require_checkpoint_rejects_missing_checkpoint(tmp_path: Path) -> None:
    manager = RuntimeStateManager(tmp_path)

    with pytest.raises(RuntimeStateError, match="Brak zapisanego checkpointu fazy demo"):
        manager.require_checkpoint(target_mode="paper", entrypoint="demo_desktop")


def test_require_checkpoint_rejects_mismatched_entrypoint(tmp_path: Path) -> None:
    manager = RuntimeStateManager(tmp_path)
    manager.record_checkpoint(
        entrypoint="demo_desktop", mode="demo", config_path="config/runtime.yaml", metadata={}
    )

    with pytest.raises(RuntimeStateError, match="innego punktu wejścia"):
        manager.require_checkpoint(target_mode="paper", entrypoint="other")


def test_require_checkpoint_rejects_invalid_source_mode(tmp_path: Path) -> None:
    manager = RuntimeStateManager(tmp_path)

    manager.record_checkpoint(
        entrypoint="demo_desktop", mode="paper", config_path="config/runtime.yaml", metadata={}
    )
    with pytest.raises(RuntimeStateError, match="wymagany: 'demo'"):
        manager.require_checkpoint(target_mode="paper", entrypoint="demo_desktop")


def test_clear_removes_checkpoint(tmp_path: Path) -> None:
    manager = RuntimeStateManager(tmp_path)
    manager.record_checkpoint(
        entrypoint="demo_desktop", mode="demo", config_path="config/runtime.yaml", metadata={}
    )
    manager.clear()
    assert manager.load_checkpoint() is None


def test_require_checkpoint_rejects_unsupported_target_mode(tmp_path: Path) -> None:
    manager = RuntimeStateManager(tmp_path)
    manager.record_checkpoint(
        entrypoint="demo_desktop", mode="demo", config_path="config/runtime.yaml", metadata={}
    )

    with pytest.raises(RuntimeStateError, match="Nieobsługiwane przejście"):
        manager.require_checkpoint(target_mode="staging", entrypoint="demo_desktop")


@pytest.mark.parametrize("target_mode", ["", "   "])
def test_require_checkpoint_rejects_empty_or_whitespace_target_mode(
    tmp_path: Path, target_mode: str
) -> None:
    manager = RuntimeStateManager(tmp_path)
    manager.record_checkpoint(
        entrypoint="demo_desktop", mode="demo", config_path="config/runtime.yaml", metadata={}
    )

    with pytest.raises(RuntimeStateError, match="Docelowy tryb checkpointu nie może być pusty"):
        manager.require_checkpoint(target_mode=target_mode, entrypoint="demo_desktop")
