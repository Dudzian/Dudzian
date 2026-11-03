"""The disposable Windows proof must never remove an unowned CLI directory."""

from __future__ import annotations

import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

from tests.deployment import windows_postgresql_bootstrap_native as native


def marker(root: Path) -> None:
    (root / native.OWNERSHIP_MARKER).write_text(native.OWNERSHIP_CONTENT, encoding="ascii")


@pytest.mark.parametrize("kind", ("outside", "temporary_itself", "unmarked"))
def test_cleanup_rejects_existing_unowned_roots(tmp_path, monkeypatch, kind):
    temporary = tmp_path / "runner"
    temporary.mkdir()
    monkeypatch.setenv("RUNNER_TEMP", str(temporary))
    root = temporary if kind == "temporary_itself" else tmp_path / "outside"
    if kind == "unmarked":
        root = temporary / "unmarked"
    root.mkdir(exist_ok=True)
    sentinel = root / "keep"
    sentinel.write_text("unchanged")
    if kind == "outside":
        marker(root)
    with pytest.raises((native.ProofFailure, FileNotFoundError)):
        native.cleanup_owned_root(root)
    assert sentinel.read_text() == "unchanged"


def test_prepare_rejects_existing_root_without_modifying_it(tmp_path, monkeypatch):
    monkeypatch.setenv("RUNNER_TEMP", str(tmp_path))
    root = tmp_path / "existing"
    root.mkdir()
    (root / "keep").write_text("unchanged")
    with pytest.raises(native.ProofFailure):
        native.prepare(root, {})
    assert {path.name for path in root.iterdir()} == {"keep"}


def test_run_rejects_absent_marker_before_any_native_operation(tmp_path, monkeypatch):
    monkeypatch.setenv("RUNNER_TEMP", str(tmp_path))
    root = tmp_path / "unmarked"
    root.mkdir()
    (root / "keep").write_text("unchanged")
    with pytest.raises(FileNotFoundError):
        native.run(root, {})
    assert {path.name for path in root.iterdir()} == {"keep"}


def test_cleanup_accepts_only_marked_authored_temp(tmp_path, monkeypatch):
    monkeypatch.setenv("RUNNER_TEMP", str(tmp_path))
    root = tmp_path / "owned"
    root.mkdir()
    marker(root)
    (root / "disposable").write_text("fixture")
    native.cleanup_owned_root(root)
    assert not root.exists()


def test_cleanup_rejects_symlink_root_and_marker(tmp_path, monkeypatch):
    monkeypatch.setenv("RUNNER_TEMP", str(tmp_path))
    root = tmp_path / "owned"
    root.mkdir()
    marker(root)
    link = tmp_path / "link"
    try:
        link.symlink_to(root, target_is_directory=True)
    except OSError:
        pytest.skip("symlinks unavailable")
    with pytest.raises(native.ProofFailure):
        native.cleanup_owned_root(link)
    (root / native.OWNERSHIP_MARKER).unlink()
    (root / native.OWNERSHIP_MARKER).symlink_to(tmp_path / "other")
    with pytest.raises(native.ProofFailure):
        native.cleanup_owned_root(root)
    assert root.exists()


def test_non_windows_main_never_cleans_even_marked_existing_root(tmp_path, monkeypatch):
    monkeypatch.setenv("RUNNER_TEMP", str(tmp_path))
    monkeypatch.setattr(sys, "platform", "linux")
    root = tmp_path / "owned"
    root.mkdir()
    marker(root)
    (root / "keep").write_text("unchanged")
    report = tmp_path / "report.json"
    monkeypatch.setattr(sys, "argv", ["native", "--root", str(root), "--report", str(report)])
    assert native.main() == 1
    assert (root / "keep").read_text() == "unchanged"
    assert "temporary_root_removed" not in json.loads(report.read_text())


def test_job_setup_failure_never_cleans_marked_existing_root(tmp_path, monkeypatch):
    monkeypatch.setenv("RUNNER_TEMP", str(tmp_path))
    monkeypatch.setattr(sys, "platform", "win32")
    root = tmp_path / "owned"
    root.mkdir()
    marker(root)
    (root / "keep").write_text("unchanged")
    report = tmp_path / "report.json"
    monkeypatch.setattr(sys, "argv", ["native", "--root", str(root), "--report", str(report)])

    def fail_job(*_args):
        raise native.ProofFailure

    monkeypatch.setitem(sys.modules, "win32api", SimpleNamespace())
    monkeypatch.setitem(sys.modules, "win32job", SimpleNamespace(CreateJobObject=fail_job))
    assert native.main() == 1
    assert (root / "keep").read_text() == "unchanged"
    assert "temporary_root_removed" not in json.loads(report.read_text())
