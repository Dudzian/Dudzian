"""Synthetic observations only. Never inspect or modify a real installation."""

from copy import deepcopy
import hashlib
import json
import os
import sys
from pathlib import Path

import pytest

from deployment import stage9_stage41b_coexistence as q
from scripts import diagnose_stage9_stage41b_coexistence as cli

pytestmark = pytest.mark.skipif(sys.platform != "linux", reason="native Linux diagnostic")


def external():
    value = {
        "schema": 1,
        "installations": {commit: {"files": {}, "directories": ["."]} for commit in q.COMMITS},
    }
    raw = json.dumps(value).encode()
    return raw, hashlib.sha256(raw).hexdigest()


def test_external_digest_and_exact_two_commits_required():
    raw, pin = external()
    assert set(q.evidence(raw, pin)) == set(q.COMMITS)
    for changed, digest in [
        (raw + b" ", pin),
        (raw, "a" * 64),
        (b"{}", hashlib.sha256(b"{}").hexdigest()),
    ]:
        with pytest.raises(ValueError):
            q.evidence(changed, digest)


def test_duplicate_manifest_key_is_rejected():
    raw = b'{"schema":1,"schema":1,"installations":{}}'
    with pytest.raises(q.preparation.PreparationBlocked):
        q.evidence(raw, hashlib.sha256(raw).hexdigest())


@pytest.fixture
def installed(tmp_path, monkeypatch):
    root = tmp_path / q.COMMITS[0]
    root.mkdir()
    (root / "installation.json").write_text("synthetic receipt")
    (root / "payload").write_bytes(b"reviewed bytes")
    monkeypatch.setattr(q.preparation, "PREFIX", tmp_path)
    # Only fixture ownership is substituted. No real privileged verifier runs.
    monkeypatch.setattr(q.host, "_protected", lambda path, **kwargs: path.lstat())
    monkeypatch.setattr(q.preparation, "verify_installation", lambda *args: {})
    inventory = q._inventory(root)
    expected = {key: inventory[key] for key in ("files", "directories")}
    return root, expected


def test_matching_independent_inventory_reports_file_boundaries_only(installed):
    _, expected = installed
    report = q.installation(q.COMMITS[0], expected)
    assert report["integrity"] == report["provenance"] == "PASS"
    assert "qualification" not in report


def test_rewritten_file_and_local_receipt_cannot_replace_external_evidence(installed):
    root, expected = installed
    (root / "payload").write_bytes(b"replacement bytes")
    (root / "installation.json").write_text("locally updated digest")
    report = q.installation(q.COMMITS[0], expected)
    assert report["provenance"] == "BLOCKED"


def test_manifest_verifier_failure_never_passes(installed, monkeypatch):
    def refuse(*args):
        raise q.preparation.PreparationBlocked("MANIFEST_MISMATCH")

    monkeypatch.setattr(q.preparation, "verify_installation", refuse)
    assert q.installation(q.COMMITS[0], installed[1])["integrity"] == "UNCERTAIN"


def test_change_between_installation_walks_is_uncertain(installed, monkeypatch):
    root, expected = installed
    monkeypatch.setattr(
        q.preparation,
        "verify_installation",
        lambda *args: (root / "payload").write_bytes(b"changed during observation"),
    )
    report = q.installation(q.COMMITS[0], expected)
    assert report["integrity"] == "UNCERTAIN"


def test_nofollow_reader_rejects_symlink(installed):
    root, _ = installed
    (root / "link").symlink_to(root / "payload")
    with pytest.raises(OSError):
        q._read(root / "link")


@pytest.fixture
def proc(monkeypatch):
    items = [Path("/proc/1"), Path("/proc") / str(os.getpid())]
    monkeypatch.setattr(q, "_listing", lambda path: items)
    monkeypatch.setattr(q.host, "_procfs_visibility", lambda: True)
    monkeypatch.setattr(
        q,
        "_installation_metadata",
        lambda path: {"identities": [("file", (0, 456, 123))], "complete": True},
    )

    def proc_stat(pid):
        fields = ["S", *("0" for _ in range(49))]
        fields[19] = "123"
        return f"{pid} (worker) ".encode() + " ".join(fields).encode() + b"\n"

    snapshot = (
        (
            "1",
            proc_stat(1),
            b"Name: worker\nPid: 1\nTgid: 1\nUid: 123 123 123 123\n"
            b"Gid: 123 123 123 123\nGroups: 123\n",
            b"0::/unrelated",
            b"000-100 r--p 0 00:00 0\n",
            (("/other", 0, 1),),
        ),
    )

    def observe(path):
        row = list(snapshot[0])
        row[0] = path.name
        row[1] = proc_stat(path.name)
        row[2] = (
            row[2]
            .replace(b"Pid: 1\n", f"Pid: {path.name}\n".encode())
            .replace(b"Tgid: 1\n", f"Tgid: {path.name}\n".encode())
        )
        return (tuple(row),)

    monkeypatch.setattr(q, "_pid_snapshot", observe)
    return items, snapshot


def test_stable_visible_processes_cannot_qualify_host(proc):
    report = q.processes()
    assert report["stable_visible_snapshot"]
    assert report["state"] == "UNCERTAIN"
    assert report["full_confirmation"] is False


@pytest.mark.parametrize("reference", ["path", "inode", "maps", "cgroup"])
def test_old_files_detected_under_unrelated_uid_and_service(proc, monkeypatch, reference):
    _, original = proc
    row = list(original[0])
    if reference == "path":
        row[5] = ((str(q.preparation.PREFIX) + "/old (deleted)", 0, 1),)
    elif reference == "inode":
        row[5] = (("/renamed-alias", 123, 456),)
    elif reference == "maps":
        device = f"{os.major(123):x}:{os.minor(123):x}"
        row[4] = f"000-100 r--p 0 {device} 456 /renamed (deleted)\n".encode()
    else:
        row[3] = b"0::/system.slice/ch-stage41b-other"
    monkeypatch.setattr(q, "_pid_snapshot", lambda path: (tuple(row),))
    report = q.processes()
    assert report["state"] == "BLOCKED"
    assert report["matched_pids"]


@pytest.mark.parametrize("failure", ["denied", "vanished", "changed", "enumeration", "hidden"])
def test_incomplete_proc_is_uncertain(proc, monkeypatch, failure):
    if failure in {"denied", "vanished"}:

        def fail(path):
            raise (
                PermissionError("PRIVATE") if failure == "denied" else FileNotFoundError("PRIVATE")
            )

        monkeypatch.setattr(q, "_pid_snapshot", fail)
    elif failure == "changed":
        answers = iter([proc[1], proc[1], ()])
        monkeypatch.setattr(q, "_pid_snapshot", lambda path: next(answers))
    elif failure == "enumeration":
        answers = iter([proc[0], [Path("/proc/1")]])
        monkeypatch.setattr(q, "_listing", lambda path: next(answers))
    else:
        monkeypatch.setattr(q.host, "_procfs_visibility", lambda: False)
    report = q.processes()
    assert report["state"] == "UNCERTAIN"
    assert not report["full_confirmation"]
    assert "PRIVATE" not in str(report)


@pytest.fixture
def resource_host(monkeypatch):
    legacy = {"units": {"list-unit-files": {"names": []}}, "runtime_paths": {}}
    checks = {"nss:passwd:62000": True, "unit:inactive.service": True, "path:cgroup": True}
    monkeypatch.setattr(q.host, "_collisions", lambda plan: deepcopy(checks))
    monkeypatch.setattr(q.host, "_stage41b_diagnostics", lambda: deepcopy(legacy))
    monkeypatch.setattr(q.host, "_nss", lambda database: [])
    return checks, legacy


@pytest.mark.parametrize("boundary", ["nss:passwd:62000", "unit:inactive.service", "path:cgroup"])
def test_numeric_uid_inactive_unit_and_cgroup_collisions_block(resource_host, boundary):
    resource_host[0][boundary] = False
    assert q.resources({})["state"] == "BLOCKED"


def test_inactive_legacy_unit_blocks(resource_host):
    resource_host[1]["units"]["list-unit-files"]["names"] = [
        "cryptohunter-root-proof-signer.service"
    ]
    assert q.resources({})["state"] == "BLOCKED"


def test_numeric_legacy_aliases_are_visible(resource_host, monkeypatch):
    monkeypatch.setattr(
        q.host,
        "_nss",
        lambda database: [["chs41bold", "x", "62000", "62001"], ["renamed", "x", "62000", "62001"]]
        if database == "passwd"
        else [["alias", "x", "62001", ""]],
    )
    report = q.resources({})
    assert report["state"] == "BLOCKED"
    assert report["legacy_identity_aliases"]["passwd"] == ["chs41bold", "renamed"]
    assert report["legacy_identity_aliases"]["group"] == ["alias"]


def test_resource_change_or_unavailable_observation_is_uncertain(resource_host, monkeypatch):
    values = iter([{"uid": True}, {"uid": False}])
    monkeypatch.setattr(q.host, "_collisions", lambda plan: next(values))
    assert q.resources({})["state"] == "UNCERTAIN"


def test_even_positive_file_evidence_never_grants_exception(resource_host, proc, monkeypatch):
    monkeypatch.setattr(
        q,
        "installation",
        lambda commit, expected: {"commit": commit, "integrity": "PASS", "provenance": "PASS"},
    )
    raw, pin = external()
    report = q.diagnose({}, raw, pin)
    assert report["qualification"] == "BLOCKED"
    assert report["preflight_exception"] is False
    assert report["coexistence_authorized"] is False
    assert report["physical_test_status"] == "PHYSICAL_TEST_PENDING"
    assert report["status"] == "IN_PROGRESS"
    assert report["production_status"] == "PRODUCTION_NO_GO"


def test_root_cli_refuses_before_any_observation(monkeypatch, capsys):
    monkeypatch.setattr(cli.os, "geteuid", lambda: 0)
    monkeypatch.setattr(q, "diagnose", lambda *args: pytest.fail("privileged observation"))
    assert cli.main(["--run-id", "012345abcdef"]) == 2
    assert "PROTECTED_LOADER_REQUIRED" in capsys.readouterr().out


def test_real_snapshot_reads_all_thread_resources_without_executing(tmp_path):
    root = tmp_path / "123"
    task = root / "task/123"
    task.mkdir(parents=True)
    (task / "fd").mkdir()
    for name, raw in {
        "stat": b"123 (other) S " + b"0 " * 18 + b"123 " + b"0 " * 29 + b"0\n",
        "status": b"Name: other\nPid: 123\nTgid: 123\nUid: 999 999 999 999\n"
        b"Gid: 999 999 999 999\nGroups: 999\n",
        "cgroup": b"0::/other",
        "maps": b"0-10 r--p 0 00:00 0\n",
    }.items():
        (task / name).write_bytes(raw)
    payload = tmp_path / "old-library"
    payload.write_bytes(b"never execute")
    for name in ("exe", "cwd", "root", "fd/5"):
        (task / name).symlink_to(payload)
    for name in ("stat", "status", "cgroup", "maps", "exe", "cwd", "root", "fd"):
        (root / name).symlink_to(task / name)
    snapshot = q._pid_snapshot(root)
    assert len(snapshot[0][5]) == 4
    assert q._references(snapshot, {(payload.stat().st_dev, payload.stat().st_ino)})


def test_proc_read_limit_and_empty_metadata_are_uncertain_inputs(tmp_path, monkeypatch):
    path = tmp_path / "maps"
    path.write_bytes(b"")
    with pytest.raises(ValueError):
        q._proc_bytes(path)
    path.write_bytes(b"12345")
    monkeypatch.setattr(q, "LIMIT", 4)
    with pytest.raises(ValueError):
        q._proc_bytes(path)
    with pytest.raises(ValueError):
        q._read(path)


def test_read_detects_replacement_during_observation(tmp_path, monkeypatch):
    target = tmp_path / "file"
    target.write_bytes(b"old")
    read = q.os.read
    replaced = False

    def replace(descriptor, size):
        nonlocal replaced
        raw = read(descriptor, size)
        if not replaced:
            replacement = tmp_path / "replacement"
            replacement.write_bytes(b"new")
            replacement.replace(target)
            replaced = True
        return raw

    monkeypatch.setattr(q.os, "read", replace)
    with pytest.raises(ValueError, match="FILE_CHANGED"):
        q._read(target)


def test_unprivileged_cli_outputs_diagnostic_and_nonzero_status(monkeypatch, capsys):
    monkeypatch.setattr(cli.os, "geteuid", lambda: 1000)
    monkeypatch.setattr(q, "diagnose", lambda *args: {"qualification": "BLOCKED"})
    assert cli.main(["--run-id", "012345abcdef"]) == 2
    assert json.loads(capsys.readouterr().out)["qualification"] == "BLOCKED"


def test_unavailable_evidence_does_not_print_private_errors(monkeypatch, capsys, tmp_path):
    monkeypatch.setattr(cli.os, "geteuid", lambda: 1000)
    assert cli.main(["--run-id", "012345abcdef", "--evidence", str(tmp_path / "PRIVATE")]) == 2
    report = capsys.readouterr().out
    assert "PRIVATE" not in report
    assert "BLOCKED" in report
