from pathlib import Path

import pytest

from scripts.ci import bootstrap_locked_pip


def test_bootstrap_installs_exact_pip_pin_from_reviewed_lock(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    lock = tmp_path / "release.lock"
    lock.write_text("pip==24.3.1\n", encoding="utf-8")
    calls: list[tuple[list[str], bool]] = []
    monkeypatch.setattr(
        bootstrap_locked_pip.subprocess,
        "run",
        lambda command, check: calls.append((command, check)),
    )

    assert bootstrap_locked_pip.main([str(lock)]) == 0
    command, check = calls[0]
    assert check is True
    assert command[-2:] == ["--no-deps", "pip==24.3.1"]
    assert "--no-deps" in command


def test_bootstrap_fails_closed_without_pip_pin(tmp_path: Path) -> None:
    lock = tmp_path / "release.lock"
    lock.write_text("wheel==0.45.1\n", encoding="utf-8")
    with pytest.raises(SystemExit, match="no exact pip pin"):
        bootstrap_locked_pip.main([str(lock)])
