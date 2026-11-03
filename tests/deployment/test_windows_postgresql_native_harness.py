"""The disposable Windows proof must never remove an unowned CLI directory."""

from __future__ import annotations

import json
import os
from pathlib import Path
import secrets
import sys
from types import SimpleNamespace
from xml.etree import ElementTree

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


@pytest.mark.parametrize(
    ("unexecuted", "child_exit"),
    ((None, 0), ("skipped", 0), ("error", 0), ("failure", 0), (None, 1)),
)
def test_native_credential_proof_refuses_skips_even_when_pytest_exits_zero(
    tmp_path, monkeypatch, unexecuted, child_exit
):
    monkeypatch.setenv("RUNNER_TEMP", str(tmp_path))
    monkeypatch.setenv("PGPASSWORD", secrets.token_urlsafe(12))
    ambient = os.environ["PGPASSWORD"]
    private_detail = secrets.token_urlsafe(24)
    root = tmp_path / "owned"
    root.mkdir()
    marker(root)

    def child(command, **kwargs):
        assert command[1:3] == ["-m", "pytest"]
        assert "--confcutdir=tests/deployment" in command
        assert command[command.index("-o") + 1] == "tmp_path_retention_policy=failed"
        assert command[command.index("--basetemp") + 1] == str(root / "credential-acl")
        assert kwargs["stdout"] is native.subprocess.DEVNULL
        assert kwargs["stderr"] is native.subprocess.DEVNULL
        assert kwargs["timeout"] == 90
        assert kwargs["env"]["CRYPTOHUNTER_RUN_WINDOWS_BOOTSTRAP_ACL_INTEGRATION"] == "1"
        assert kwargs["env"]["PYTEST_DISABLE_PLUGIN_AUTOLOAD"] == "1"
        ambient_preserved = kwargs["env"]["PGPASSWORD"] == ambient
        assert ambient_preserved
        # Contradictory success counters ensure a skipped testcase itself cannot
        # be mistaken for an executed proof, even with process exit status zero.
        document = ElementTree.Element(
            "testsuite", tests="2", failures="0", errors="0", skipped="0"
        )
        for value in (False, True):
            case = ElementTree.SubElement(
                document,
                "testcase",
                name=native.NATIVE_CREDENTIAL_ACL_TEST + f"[{value}]",
            )
            if value and unexecuted is not None:
                problem = ElementTree.SubElement(case, unexecuted)
                problem.text = (
                    private_detail
                    + "\ntests/deployment/test_windows_bootstrap_credentials.py:400: AssertionError"
                    + "\ntests/deployment/test_windows_bootstrap_credentials.py:484: PermissionError"
                )
        ElementTree.ElementTree(document).write(command[command.index("--junitxml") + 1])
        return SimpleNamespace(returncode=child_exit)

    monkeypatch.setattr(native.subprocess, "run", child)
    report = {}
    if unexecuted is None and child_exit == 0:
        native.qualify_native_credential_acl_crash(root, report)
        assert report["native_credential_acl_crash"] is True
    else:
        with pytest.raises(native.ProofFailure):
            native.qualify_native_credential_acl_crash(root, report)
        assert "native_credential_acl_crash" not in report
    assert report["native_acl_pytest_exit"] == child_exit
    assert private_detail not in json.dumps(report)
    if unexecuted in {"error", "failure"}:
        assert report["native_acl_crash_failure_type"] == "PERMISSION_ERROR"
        assert report["native_acl_crash_failure_line"] == 484
    parent_ambient_preserved = os.environ["PGPASSWORD"] == ambient
    assert parent_ambient_preserved
