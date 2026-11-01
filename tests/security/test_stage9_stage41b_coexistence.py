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


def test_cli_projects_closed_schema_without_nested_nss_or_unknown_fields(monkeypatch, capsys):
    secret = "PRIVATE_ACCOUNT_DATA_NEVER_PRINT"
    raw_report = {
        "qualification": "PASS",
        "coexistence_authorized": True,
        "preflight_exception": True,
        "production_status": "PRODUCTION_GO",
        "evidence_pin_verified": True,
        "blockers": [secret, {"unknown": secret}],
        "unknown": {"nested": [{"passwd": secret}, {"future_secret": secret}]},
        "resources": {
            "state": "BLOCKED",
            "reason": "RESOURCE_OBSERVATION_INCOMPLETE_OR_CHANGED",
            "legacy_identity_aliases": {"passwd": [secret], "group": [secret]},
            "source_cca_checks": {secret: False},
            "legacy": {"units": {"names": [secret]}, "future": [secret]},
            "reason_counts": {secret: 999999},
        },
        "process_observation": {
            "state": "UNCERTAIN",
            "reason": "PROCESS_OBSERVATION_INCOMPLETE_OR_CHANGED",
            "stable_visible_snapshot": False,
            "full_confirmation": True,
            "uncertainty": ["PROCFS_VISIBILITY_UNCERTAIN", secret, {"nested": secret}],
            "matched_pids": [secret],
            "checked_pid_count": secret,
            "installation_metadata": {"private": secret},
        },
        "installations": [
            {
                "commit": commit,
                "integrity": "PASS",
                "provenance": "BLOCKED",
                "reason": "INDEPENDENT_INVENTORY_MISMATCH",
                "files": {secret: secret},
                "directory_count": secret,
            }
            for commit in q.COMMITS
        ],
    }
    monkeypatch.setattr(cli.os, "geteuid", lambda: 1000)
    monkeypatch.setattr(q, "diagnose", lambda *args: raw_report)
    assert cli.main(["--run-id", "012345abcdef"]) == 2
    captured = capsys.readouterr()
    assert captured.err == ""
    assert secret not in captured.out
    report = json.loads(captured.out)
    assert report["qualification"] == "BLOCKED"
    assert report["production_status"] == "PRODUCTION_NO_GO"
    assert report["coexistence_authorized"] is False
    assert report["preflight_exception"] is False
    assert report["evidence_pin_verified"] is True
    assert report["blockers"] == [
        "NO_COMPLETE_INDEPENDENT_HOST_ISOLATION_PROOF",
        "NO_OTHER_INSTALLATIONS_GATE_RETAINED",
    ]
    assert "unknown" not in report
    assert report["resources"]["state"] == "BLOCKED"
    assert report["resources"]["reason"] == "RESOURCE_OBSERVATION_INCOMPLETE_OR_CHANGED"
    assert "legacy_identity_aliases" not in report["resources"]
    assert "legacy" not in report["resources"]
    assert report["process_observation"]["full_confirmation"] is False
    assert report["process_observation"]["uncertainty"] == ["PROCFS_VISIBILITY_UNCERTAIN"]
    assert report["installations"][0]["provenance"] == "BLOCKED"
    assert report["installations"][0]["reason"] == "INDEPENDENT_INVENTORY_MISMATCH"


@pytest.mark.parametrize("section", [None, [], "PRIVATE_SECRET", 123, False])
def test_unknown_report_shape_is_closed(section):
    projected = cli._safe_projection(section)
    assert projected["qualification"] == "BLOCKED"
    assert projected["reason"] == "DIAGNOSTIC_REPORT_SCHEMA_UNAVAILABLE"
    assert "PRIVATE_SECRET" not in json.dumps(projected)


@pytest.mark.parametrize(
    "payload", ["PRIVATE_SECRET", {"private": "PRIVATE_SECRET"}, ["PRIVATE_SECRET"]]
)
def test_sensitive_data_in_known_status_and_reason_fields_cannot_reach_output(payload):
    report = {
        "reason": payload,
        "evidence_pin_verified": payload,
        "resources": {"state": payload, "reason": payload},
        "process_observation": {
            "state": payload,
            "reason": payload,
            "stable_visible_snapshot": payload,
            "stable_process_identities": payload,
            "stable_installation_bindings": payload,
            "uncertainty": [payload],
        },
        "installations": [
            {"commit": commit, "integrity": payload, "provenance": payload, "reason": payload}
            for commit in q.COMMITS
        ],
    }
    projected = cli._safe_projection(report)
    assert "PRIVATE_SECRET" not in json.dumps(projected)
    assert projected["evidence_pin_verified"] is False
    assert projected["resources"]["state"] == "UNCERTAIN"
    assert projected["process_observation"]["uncertainty"] == []
    assert projected["process_observation"]["stable_visible_snapshot"] is False
    assert all(item["integrity"] == "UNCERTAIN" for item in projected["installations"])


def test_cli_does_not_print_observer_exception_data(monkeypatch, capsys):
    monkeypatch.setattr(cli.os, "geteuid", lambda: 1000)

    def fail(*args):
        raise RuntimeError("PRIVATE_EXCEPTION_DATA")

    monkeypatch.setattr(q, "diagnose", fail)
    assert cli.main(["--run-id", "012345abcdef"]) == 2
    captured = capsys.readouterr()
    assert captured.err == ""
    assert "PRIVATE_EXCEPTION_DATA" not in captured.out
    assert json.loads(captured.out)["reason"] == "DIAGNOSTIC_INPUT_OR_OBSERVATION_UNAVAILABLE"


def test_cli_does_not_print_import_exception_data(monkeypatch, capsys):
    import builtins

    original_import = builtins.__import__

    def import_module(name, *args, **kwargs):
        if name == "deployment.stage9_source_cca_test_package":
            raise ImportError("PRIVATE_IMPORT_ERROR")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(cli.os, "geteuid", lambda: 1000)
    monkeypatch.setattr(builtins, "__import__", import_module)
    assert cli.main(["--run-id", "012345abcdef"]) == 2
    captured = capsys.readouterr()
    assert captured.err == ""
    assert "PRIVATE_IMPORT_ERROR" not in captured.out
    assert json.loads(captured.out)["reason"] == "DIAGNOSTIC_INPUT_OR_OBSERVATION_UNAVAILABLE"


def test_cli_never_serializes_or_compares_unknown_objects(monkeypatch, capsys):
    class PrivateObject:
        def __str__(self):
            raise RuntimeError("PRIVATE_SERIALIZATION_DATA")

        def __repr__(self):
            raise RuntimeError("PRIVATE_REPRESENTATION_DATA")

        def __eq__(self, other):
            raise RuntimeError("PRIVATE_EQUALITY_DATA")

    private = PrivateObject()
    report = {
        "reason": private,
        "unknown": private,
        "resources": {"state": private, "reason": private, "legacy": private},
        "process_observation": {
            "state": private,
            "reason": private,
            "uncertainty": [private, {"nested": private}, "PROCESS_PERMISSION_DENIED"],
            "stable_visible_snapshot": private,
        },
        "installations": [
            {"commit": private, "integrity": private},
            {"commit": q.COMMITS[1], "integrity": private, "reason": private},
        ],
    }
    monkeypatch.setattr(cli.os, "geteuid", lambda: 1000)
    monkeypatch.setattr(q, "diagnose", lambda *args: report)
    assert cli.main(["--run-id", "012345abcdef"]) == 2
    captured = capsys.readouterr()
    assert captured.err == ""
    assert "PRIVATE_" not in captured.out
    projected = json.loads(captured.out)
    assert projected["qualification"] == "BLOCKED"
    assert projected["process_observation"]["uncertainty"] == ["PROCESS_PERMISSION_DENIED"]


def test_cli_projection_failure_does_not_print_exception_payload(monkeypatch, capsys):
    class PrivateKey:
        def __hash__(self):
            return hash("reason")

        def __eq__(self, other):
            raise RuntimeError("PRIVATE_KEY_COMPARISON_DATA")

    monkeypatch.setattr(cli.os, "geteuid", lambda: 1000)
    monkeypatch.setattr(q, "diagnose", lambda *args: {PrivateKey(): "private"})
    assert cli.main(["--run-id", "012345abcdef"]) == 2
    captured = capsys.readouterr()
    assert captured.err == ""
    assert "PRIVATE_" not in captured.out
    projected = json.loads(captured.out)
    assert projected["reason"] == "DIAGNOSTIC_INPUT_OR_OBSERVATION_UNAVAILABLE"


@pytest.mark.parametrize(
    "arguments",
    [
        ["--uid-base", "PRIVATE_ARGUMENT_VALUE"],
        ["--unknown-option", "PRIVATE_ARGUMENT_VALUE"],
        ["--run-id", "PRIVATE_ARGUMENT_VALUE", "--uid-base", "invalid"],
    ],
)
def test_cli_parse_errors_do_not_echo_argument_values(arguments, capsys):
    with pytest.raises(SystemExit) as caught:
        cli.main(arguments)
    assert caught.value.code == 2
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "PRIVATE_ARGUMENT_VALUE" not in captured.err
    assert "invalid" not in captured.err
    assert json.loads(captured.err)["reason"] == "DIAGNOSTIC_ARGUMENTS_INVALID"
