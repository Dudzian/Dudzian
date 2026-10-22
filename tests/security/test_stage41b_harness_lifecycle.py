"""Unprivileged orchestration regressions, never physical Stage 41B evidence."""

from __future__ import annotations

import base64
from copy import deepcopy
import json
from pathlib import Path
import signal
import stat
import subprocess
import sys
from types import SimpleNamespace

import pytest

from scripts import qualify_stage41b_signer_isolation as harness

pytestmark = pytest.mark.skipif(
    sys.platform != "linux", reason="Linux-only Stage 41B process and systemd qualification harness"
)

RUN_ID = "a" * 32


def unused_unit():
    value = {key: "" for key in harness.UNIT_FIELDS}
    value.update(LoadState="not-found", ActiveState="inactive", Transient="no", MainPID="0")
    return value


def owned_unit(role="root_signer", pid="1234"):
    name = harness._unit_name(RUN_ID, role)
    value = unused_unit()
    value.update(
        Id=name,
        LoadState="loaded",
        ActiveState="active",
        Transient="yes",
        FragmentPath=f"/run/systemd/transient/{name}",
        Description=f"Stage41B TEST ONLY {RUN_ID} {role}",
        InvocationID="b" * 32,
        MainPID=pid,
        ControlGroup=f"/system.slice/{name}",
    )
    return value


@pytest.fixture
def local_run(monkeypatch, tmp_path):
    """Redirect every owned fixture into pytest tmp; virtualize ownership only."""
    run = harness.PhysicalRun(RUN_ID, Path("/opt/reviewed"), Path("/opt/venv/bin/python"))
    run.base, run.hidden = tmp_path / "owner", tmp_path / "hidden"
    monkeypatch.setattr(harness, "_state", lambda run_id, role: tmp_path / ("state-" + role))
    original_lstat = Path.lstat

    def root_fixture_lstat(path):
        value = original_lstat(path)
        if path.is_relative_to(tmp_path):
            fields = list(value)
            fields[4] = 0
            return type(value)(fields)
        return value

    monkeypatch.setattr(Path, "lstat", root_fixture_lstat)
    monkeypatch.setattr(harness, "inspect_unit", lambda name: unused_unit())
    return run


def test_create_manifest_and_cleanup_remove_only_recorded_scope(local_run):
    local_run.create()
    manifest = json.loads((local_run.base / "owner.json").read_text())
    assert manifest == {
        "stage": "41B_TEST_ONLY",
        "run_id": RUN_ID,
        "units": [harness._unit_name(RUN_ID, role) for role in harness.ROLES],
    }
    assert len(local_run.owned_paths) == 7
    assert local_run.cleanup() is True
    assert not local_run.base.exists() and not local_run.hidden.exists()


def test_installed_test_unit_collision_precedes_filesystem_mutation(local_run, monkeypatch):
    monkeypatch.setattr(harness, "inspect_unit", lambda name: owned_unit())
    with pytest.raises(harness.QualificationBlocked, match="TEST_UNIT_COLLISION"):
        local_run.create()
    assert local_run.owned_paths == []
    assert not local_run.base.exists()


def test_existing_runtime_path_is_not_reused_or_removed(local_run):
    runtime = harness._state(RUN_ID, "root_signer")
    runtime.mkdir()
    sentinel = runtime / "unknown-origin"
    sentinel.write_text("retain")
    with pytest.raises(harness.QualificationBlocked, match="TEST_RUNTIME_COLLISION"):
        local_run.create()
    assert local_run.cleanup() is True
    assert sentinel.read_text() == "retain"


def test_partial_creation_cleans_known_path_and_retains_existing_foreign_scope(local_run):
    local_run.hidden.mkdir()
    sentinel = local_run.hidden / "unknown-origin"
    sentinel.write_text("retain")
    with pytest.raises(FileExistsError):
        local_run.create()
    assert local_run.cleanup() is True
    assert not local_run.base.exists()
    assert sentinel.read_text() == "retain"


def test_cleanup_refuses_symlink_replacement_and_preserves_target(local_run, tmp_path):
    local_run.create()
    witness = local_run.hidden / "root_signer/witness"
    witness.unlink()
    target = tmp_path / "unrelated-file"
    target.write_text("untouched")
    witness.symlink_to(target)
    assert local_run.cleanup() is False
    assert witness.is_symlink()
    assert target.read_text() == "untouched"


def test_cleanup_refuses_unrecorded_descendant(local_run):
    local_run.create()
    unknown = local_run.base / "unknown-origin"
    unknown.write_text("retain")
    assert local_run.cleanup() is False
    assert unknown.read_text() == "retain"


def test_cleanup_does_not_stop_replaced_or_unpinned_unit(local_run, monkeypatch):
    local_run.attempted = ["root_signer", "history_signer"]
    local_run.pins["root_signer"] = owned_unit()
    replacement = owned_unit()
    replacement["InvocationID"] = "c" * 32
    monkeypatch.setattr(harness, "inspect_unit", lambda name: replacement)
    commands = []
    monkeypatch.setattr(harness, "_command", lambda argv, **kwargs: commands.append(argv))
    assert local_run.cleanup() is False
    assert commands == []


def test_cleanup_stops_pinned_invocation_then_requires_unit_and_runtime_gone(
    local_run, monkeypatch
):
    local_run.create()
    local_run.attempted = ["root_signer"]
    local_run.pins["root_signer"] = owned_unit()
    snapshots = iter((owned_unit(), unused_unit()))
    monkeypatch.setattr(harness, "inspect_unit", lambda name: next(snapshots))
    commands = []
    monkeypatch.setattr(harness, "_command", lambda argv, **kwargs: commands.append((argv, kwargs)))
    assert local_run.cleanup() is True
    assert commands == [
        (["systemctl", "stop", harness._unit_name(RUN_ID, "root_signer")], {"timeout": 12})
    ]


def test_cleanup_failure_in_one_unit_still_cleans_recorded_files(local_run, monkeypatch):
    local_run.create()
    local_run.attempted = ["root_signer"]
    monkeypatch.setattr(
        harness,
        "inspect_unit",
        lambda name: (_ for _ in ()).throw(OSError("native secret in error")),
    )
    assert local_run.cleanup() is False
    assert not local_run.base.exists() and not local_run.hidden.exists()


def test_start_pins_each_service_before_native_socket_readiness(local_run, monkeypatch):
    seen = {}

    def inspect(name):
        seen[name] = seen.get(name, 0) + 1
        role = (
            "root_signer" if name == harness._unit_name(RUN_ID, "root_signer") else "history_signer"
        )
        return (
            unused_unit()
            if seen[name] == 1
            else owned_unit(role, "1234" if role == "root_signer" else "1235")
        )

    monkeypatch.setattr(harness, "inspect_unit", inspect)
    monkeypatch.setattr(Path, "exists", lambda path: path.name == "control.sock")
    commands = []
    monkeypatch.setattr(harness, "_command", lambda argv, **kwargs: commands.append(argv))
    local_run.start()
    assert tuple(local_run.pins) == harness.ROLES
    assert local_run.attempted == list(harness.ROLES)
    assert all(
        local_run.pins[role]["Id"] == harness._unit_name(RUN_ID, role) for role in harness.ROLES
    )
    assert len(commands) == 2
    assert all(command[0] == "systemd-run" for command in commands)


def test_start_failure_retains_pin_for_later_cleanup(local_run, monkeypatch):
    first = True

    def inspect(name):
        nonlocal first
        if first:
            first = False
            return unused_unit()
        value = owned_unit()
        value["ActiveState"] = "failed"
        return value

    monkeypatch.setattr(harness, "inspect_unit", inspect)
    monkeypatch.setattr(Path, "exists", lambda path: False)
    monkeypatch.setattr(harness, "_command", lambda *args, **kwargs: b"")
    with pytest.raises(harness.QualificationBlocked, match="TEST_SERVICE_START_FAILED"):
        local_run.start()
    assert local_run.pins["root_signer"]["MainPID"] == "1234"


def test_request_records_public_transcript_and_has_bounded_history(local_run, monkeypatch):
    monkeypatch.setattr(
        harness, "_request", lambda *args: ({"status": "PASS"}, b"unit public transcript")
    )
    assert local_run.request("root_signer", "TEST_STATUS") == {"status": "PASS"}
    assert local_run.transcripts == [b"unit public transcript"]
    local_run.transcripts *= 64
    with pytest.raises(harness.QualificationBlocked, match="TEST_TRANSCRIPT_LIMIT"):
        local_run.request("root_signer", "TEST_STATUS")


def test_live_statuses_requires_original_active_invocations(local_run, monkeypatch):
    local_run.pins = {role: owned_unit(role) for role in harness.ROLES}
    monkeypatch.setattr(
        harness,
        "inspect_unit",
        lambda name: owned_unit(
            "root_signer" if name == harness._unit_name(RUN_ID, "root_signer") else "history_signer"
        ),
    )
    monkeypatch.setattr(local_run, "request", lambda role, operation: {"role": role})
    validated = []
    monkeypatch.setattr(harness, "validate_status_pair", lambda *args: validated.append(args))
    assert local_run.live_statuses() == {role: {"role": role} for role in harness.ROLES}
    assert len(validated) == 1
    monkeypatch.setattr(harness, "inspect_unit", lambda name: unused_unit())
    with pytest.raises(harness.QualificationBlocked, match="TEST_SERVICE_NOT_LIVE"):
        local_run.live_statuses()


def test_scan_requires_both_role_receipts_and_includes_both_journals(local_run, monkeypatch):
    local_run.pins = {role: owned_unit(role) for role in harness.ROLES}
    local_run.pins["history_signer"]["InvocationID"] = "c" * 32
    commands = []
    monkeypatch.setattr(
        harness, "_command", lambda argv: commands.append(argv) or b"public unit journal"
    )
    local_run.transcripts = [b"public IPC output"]
    requests = []

    def request(role, operation, arguments):
        requests.append((role, operation, arguments))
        return {"status": "PASS", "reason": "NO_PRIVATE_MATERIAL_IN_ARTIFACTS"}

    monkeypatch.setattr(local_run, "request", request)
    assert local_run.scan() is True
    assert {row[0] for row in requests} == set(harness.ROLES)
    decoded = [base64.b64decode(item) for row in requests for item in row[2]["artifacts_b64"]]
    assert b"public IPC output" in decoded and b"public unit journal" in decoded
    assert len(commands) == 2
    assert {
        argument
        for command in commands
        for argument in command
        if argument.startswith("_SYSTEMD_INVOCATION_ID=")
    } == {"_SYSTEMD_INVOCATION_ID=" + pin["InvocationID"] for pin in local_run.pins.values()}
    assert all(command[0] == "journalctl" for command in commands)
    assert all(not argument.startswith("--unit") for command in commands for argument in command)
    monkeypatch.setattr(local_run, "request", lambda *args: {"status": "FAIL"})
    assert local_run.scan() is False


def test_scan_rejects_oversized_unscanned_artifact(local_run, monkeypatch):
    local_run.pins = {role: owned_unit(role) for role in harness.ROLES}
    monkeypatch.setattr(harness, "_command", lambda argv: b"")
    local_run.transcripts = [b"a" * (harness.MAX_ARTIFACT + 1)]
    with pytest.raises(harness.QualificationBlocked, match="TEST_ARTIFACT_LIMIT"):
        local_run.scan()


def test_inspect_unit_requires_complete_unique_known_properties(monkeypatch):
    valid = "\n".join(f"{key}={unused_unit()[key]}" for key in harness.UNIT_FIELDS).encode()
    monkeypatch.setattr(harness, "_command", lambda argv: valid)
    assert harness.inspect_unit("unit-fixture.service") == unused_unit()
    for raw in (valid + b"\nMainPID=0", valid + b"\nUnknown=yes", b"MainPID=0", b"invalid-line"):
        monkeypatch.setattr(harness, "_command", lambda argv, value=raw: value)
        with pytest.raises(harness.QualificationBlocked):
            harness.inspect_unit("unit-fixture.service")


def test_host_command_failures_and_unbounded_output_block(monkeypatch):
    def completed(code=0, stdout=b"", stderr=b""):
        monkeypatch.setattr(
            harness.subprocess,
            "run",
            lambda *args, **kwargs: SimpleNamespace(returncode=code, stdout=stdout, stderr=stderr),
        )

    completed(stdout=b"public")
    assert harness._command(["fixture"]) == b"public"
    completed(code=1, stderr=b"private native diagnostic")
    with pytest.raises(harness.QualificationBlocked, match="HOST_COMMAND_UNAVAILABLE"):
        harness._command(["fixture"])
    completed(stderr=b"x" * (harness.MAX_ARTIFACT + 1))
    with pytest.raises(harness.QualificationBlocked, match="HOST_OUTPUT_LIMIT"):
        harness._command(["fixture"])


@pytest.fixture
def ready_host(monkeypatch):
    monkeypatch.setattr(harness.sys, "platform", "linux")
    original_text = Path.read_text
    original_bytes = Path.read_bytes
    files = {"/etc/os-release": 'ID=ubuntu\nVERSION_ID="24.04"\n', "/proc/1/comm": "systemd\n"}
    monkeypatch.setattr(
        Path,
        "read_text",
        lambda path, **kwargs: files[str(path)]
        if str(path) in files
        else original_text(path, **kwargs),
    )
    monkeypatch.setattr(
        Path,
        "read_bytes",
        lambda path: b"non-secret host identity"
        if str(path) in {"/etc/machine-id", "/sys/devices/virtual/dmi/id/product_uuid"}
        else original_bytes(path),
    )
    monkeypatch.setattr(harness.os, "geteuid", lambda: 0)
    monkeypatch.setattr(harness.shutil, "which", lambda program: "/usr/bin/" + program)
    monkeypatch.setattr(
        harness.subprocess, "run", lambda *args, **kwargs: SimpleNamespace(returncode=1)
    )
    monkeypatch.setattr(harness, "_check_protected", lambda *args, **kwargs: None)
    monkeypatch.setattr(harness, "_trusted_code", lambda *args: None)
    inspected = []
    monkeypatch.setattr(
        harness, "inspect_unit", lambda name: inspected.append(name) or unused_unit()
    )
    return files, inspected


def test_preflight_observes_every_production_name_without_mutating_units(ready_host):
    _files, inspected = ready_host
    assert (
        harness.host_preflight(Path("/opt/reviewed"), Path("/opt/venv/bin/python"))["status"]
        == "PASS"
    )
    assert tuple(inspected) == harness.PRODUCTION_UNITS


def test_preflight_refuses_installed_production_units_even_inactive(ready_host, monkeypatch):
    value = unused_unit()
    value.update(LoadState="loaded", FragmentPath="/etc/systemd/system/production.service")
    monkeypatch.setattr(harness, "inspect_unit", lambda name: value)
    with pytest.raises(harness.QualificationBlocked, match="PRODUCTION_UNIT_COLLISION"):
        harness.host_preflight(Path("/opt/reviewed"), Path("/opt/venv/bin/python"))


def test_preflight_requires_physical_ubuntu_systemd_and_explicit_root_mode(ready_host, monkeypatch):
    files, _inspected = ready_host
    files["/etc/os-release"] = "ID=debian\nVERSION_ID=13\n"
    with pytest.raises(harness.QualificationBlocked, match="UBUNTU_24_04_REQUIRED"):
        harness.host_preflight(Path("/opt/reviewed"), Path("/opt/venv/bin/python"))
    files["/etc/os-release"] = "ID=ubuntu\nVERSION_ID=24.04\n"
    files["/proc/1/comm"] = "container-init\n"
    with pytest.raises(harness.QualificationBlocked, match="SYSTEMD_PID1_REQUIRED"):
        harness.host_preflight(Path("/opt/reviewed"), Path("/opt/venv/bin/python"))
    files["/proc/1/comm"] = "systemd\n"
    monkeypatch.setattr(harness.os, "geteuid", lambda: 1000)
    with pytest.raises(harness.QualificationBlocked, match="CONTROLLED_ROOT_HARNESS_REQUIRED"):
        harness.host_preflight(Path("/opt/reviewed"), Path("/opt/venv/bin/python"))
    monkeypatch.setattr(harness.os, "geteuid", lambda: 0)
    monkeypatch.setattr(
        harness.subprocess, "run", lambda *args, **kwargs: SimpleNamespace(returncode=0)
    )
    with pytest.raises(harness.QualificationBlocked, match="DEDICATED_PHYSICAL_HOST_REQUIRED"):
        harness.host_preflight(Path("/opt/reviewed"), Path("/opt/venv/bin/python"))


PROBE_REASONS = {
    "foreign_store_dac": "PERMISSION_DENIED",
    "foreign_keyring_files": "PERMISSION_DENIED",
    "foreign_mount": "NAMESPACE_PATH_ABSENT",
    "foreign_dbus": "PERMISSION_DENIED",
    "foreign_keyring": "FOREIGN_SECRET_ABSENT_FROM_OWN_NATIVE_BUS",
    "foreign_custody": "FOREIGN_REFERENCE_AND_IDENTITY_REJECTED",
    "foreign_custody_store": "FOREIGN_ORIGINAL_PROVIDER_STORE_REJECTED",
    "foreign_ipc": "PERMISSION_DENIED",
    "own_liveness": "ORIGINAL_CUSTODY_READ_AND_BUS_OWNER",
    "identity_binding": "ROLE_AND_CREDENTIAL_BINDING_REJECTED",
    "unauthorized_ipc": "UNPRIVILEGED_PEER_REJECTED",
}


@pytest.fixture
def fake_execution(monkeypatch):
    class FakeRun:
        run_id = RUN_ID
        hidden = Path("/unit-fixture/hidden")

        def __init__(self):
            self.events = []
            self.statuses = {role: {"role": role} for role in harness.ROLES}
            self.after = self.statuses
            self.probes = {
                role: {
                    "boundaries": {
                        name: harness.result("PASS", reason)
                        for name, reason in PROBE_REASONS.items()
                    }
                }
                for role in harness.ROLES
            }

        def create(self):
            self.events.append("create")

        def start(self):
            self.events.append("start")

        def live_statuses(self):
            self.events.append("live")
            return self.statuses if self.events.count("live") == 1 else self.after

        def request(self, role, operation, arguments=None):
            self.events.append((role, operation))
            return self.probes[role] if operation == "TEST_PROBE" else {"public": "signature"}

        def scan(self):
            self.events.append("scan")
            return True

    run = FakeRun()
    monkeypatch.setattr(harness, "_verify_signatures", lambda *args: None)
    monkeypatch.setattr(
        harness, "_check_protected", lambda path, **kwargs: run.events.append(("witness", path))
    )
    boundaries = {
        key: harness.result("BLOCKED", "UNIT_FIXTURE") for key in harness.REQUIRED_BOUNDARIES
    }
    return run, boundaries


def test_execution_requires_both_live_services_and_host_witness_before_probes(fake_execution):
    run, boundaries = fake_execution
    harness._execute(run, boundaries)
    assert boundaries["service_liveness"]["status"] == "PASS"
    assert run.events.index("live") < run.events.index(("root_signer", "TEST_PROBE"))
    assert run.events.index(("witness", run.hidden / "history_signer/witness")) < run.events.index(
        ("root_signer", "TEST_PROBE")
    )
    assert run.events.count("live") == 3
    assert all(boundaries[name]["status"] == "PASS" for name in PROBE_REASONS)


def test_changed_or_dead_native_service_after_probes_vetoes_liveness(fake_execution):
    run, boundaries = fake_execution
    run.after = deepcopy(run.statuses)
    run.after["history_signer"]["native_service_live"] = False
    with pytest.raises(harness.QualificationBlocked, match="LIVE_ROLE_FACTS_CHANGED"):
        harness._execute(run, boundaries)
    assert boundaries["service_liveness"]["status"] == "BLOCKED"
    assert "scan" not in run.events


def test_enoent_is_not_accepted_for_foreign_bus_or_store(fake_execution):
    run, boundaries = fake_execution
    for role in harness.ROLES:
        run.probes[role]["boundaries"]["foreign_dbus"] = harness.result(
            "PASS", "NAMESPACE_PATH_ABSENT"
        )
        run.probes[role]["boundaries"]["foreign_store_dac"] = harness.result(
            "PASS", "NAMESPACE_PATH_ABSENT"
        )
    harness._execute(run, boundaries)
    assert boundaries["foreign_dbus"]["status"] == "BLOCKED"
    assert boundaries["foreign_store_dac"]["status"] == "BLOCKED"


def test_missing_blocked_and_breached_probe_receipts_stay_distinct(fake_execution):
    run, boundaries = fake_execution
    run.probes["root_signer"]["boundaries"].pop("identity_binding")
    run.probes["history_signer"]["boundaries"]["foreign_dbus"] = harness.result(
        "BLOCKED", "SERVICE_UNAVAILABLE"
    )
    run.probes["history_signer"]["boundaries"]["foreign_keyring"] = harness.result(
        "FAIL", "ISOLATION_BREACH"
    )
    harness._execute(run, boundaries)
    assert boundaries["identity_binding"]["status"] == "BLOCKED"
    assert boundaries["foreign_dbus"]["status"] == "BLOCKED"
    assert boundaries["foreign_keyring"]["status"] == "FAIL"


@pytest.fixture
def main_run(monkeypatch):
    run = SimpleNamespace(
        run_id=RUN_ID,
        code_root=Path("/opt/reviewed"),
        python=Path("/opt/venv/bin/python"),
        owned_paths=[],
        attempted=[],
        pins={},
    )
    run.cleaned = False

    def cleanup():
        run.cleaned = True
        return True

    run.cleanup = cleanup
    monkeypatch.setattr(harness, "PhysicalRun", lambda *args: run)
    monkeypatch.setattr(
        harness, "host_preflight", lambda *args: harness.result("PASS", "UNIT_HOST_FIXTURE")
    )
    handlers = {}

    def install(number, handler):
        previous = handlers.get(number, signal.SIG_DFL)
        handlers[number] = handler
        return previous

    monkeypatch.setattr(harness.signal, "signal", install)
    return run, handlers


def test_check_mode_never_executes_or_claims_physical_pass(main_run, monkeypatch, capsys):
    run, _handlers = main_run
    monkeypatch.setattr(
        harness, "_execute", lambda *args: pytest.fail("read-only check executed physical controls")
    )
    assert harness.main(["--check"]) == 2
    report = json.loads(capsys.readouterr().out)
    assert report["status"] == "BLOCKED" and report["mode"] == "check"
    assert report["authorizes_sign_bind_enrollment"] is False
    assert report["stage9"] == "IN_PROGRESS / NO-GO"
    assert run.cleaned is False


def test_execute_error_cleans_scope_and_redacts_native_exception(main_run, monkeypatch, capsys):
    run, handlers = main_run

    def execute(_run, boundaries):
        run.attempted.append("root_signer")
        raise RuntimeError("native secret must not escape to logs")

    monkeypatch.setattr(harness, "_execute", execute)
    assert harness.main(["--execute"]) == 2
    report_text = capsys.readouterr().out
    report = json.loads(report_text)
    assert "native secret" not in report_text
    assert run.cleaned is True
    assert report["boundaries"]["cleanup"]["status"] == "PASS"
    assert all(value == signal.SIG_DFL for value in handlers.values())


def test_signal_interrupt_still_performs_owned_scope_cleanup(main_run, monkeypatch, capsys):
    run, handlers = main_run

    def execute(_run, boundaries):
        run.attempted.append("root_signer")
        handlers[signal.SIGTERM](signal.SIGTERM, None)

    monkeypatch.setattr(harness, "_execute", execute)
    assert harness.main(["--execute"]) == 2
    report = json.loads(capsys.readouterr().out)
    assert run.cleaned is True
    assert report["boundaries"]["service_liveness"]["reason"] == "PHYSICAL_RUN_INTERRUPTED"


@pytest.fixture
def host_status_fixture(monkeypatch):
    statuses, pins, process_fields, namespace_inodes = {}, {}, {}, {}
    process_roles, executables = {}, {}
    missing = set()
    for index, role in enumerate(harness.ROLES):
        uid, gid, pid = 61011 + index, 62011 + index, 1234 + index
        state = harness._state(RUN_ID, role)
        pins[role] = owned_unit(role, str(pid))
        for key in ("NoNewPrivileges", "PrivateMounts", "PrivateIPC", "PrivateNetwork"):
            pins[role][key] = "yes"
        namespaces = {
            name: 1000 * (index + 1) + offset for offset, name in enumerate(("mnt", "ipc", "net"))
        }
        statuses[role] = {
            "uid": uid,
            "gid": gid,
            "pid": pid,
            "role": role,
            "run_id": RUN_ID,
            "state": str(state),
            "socket": str(state / "control.sock"),
            "bus": str(state / "bus"),
            "native_service_live": True,
            "no_new_privs": True,
            "namespaces": namespaces,
            "metadata_path": str(state / "metadata.json"),
            "keyring_path": str(state / "home/keyrings"),
            "private_index_path": str(state / "secret-index.json"),
            "public_key_b64": role + "-public-key",
            "credential_id": role + "-credential",
            "secret_reference": role + "-secret-reference",
            "service_namespace": role + "-namespace",
            "bus_pid": 2234 + index,
            "keyring_pid": 3234 + index,
        }
        process_fields[pid] = {
            "Uid": " ".join([str(uid)] * 4),
            "Gid": " ".join([str(gid)] * 4),
            "NoNewPrivs": "1",
            "Groups": str(gid),
            "CapEff": "0",
            "CapPrm": "0",
            "CapBnd": "0",
            "CapAmb": "0",
        }
        for name, inode in namespaces.items():
            namespace_inodes[f"/proc/{pid}/ns/{name}"] = inode
        process_roles[pid] = role
        for key, executable in (
            ("bus_pid", "dbus-daemon"),
            ("keyring_pid", "gnome-keyring-daemon"),
        ):
            child_pid = statuses[role][key]
            process_fields[child_pid] = deepcopy(process_fields[pid])
            process_roles[child_pid] = role
            executables[f"/proc/{child_pid}/exe"] = executable
            for name, inode in namespaces.items():
                namespace_inodes[f"/proc/{child_pid}/ns/{name}"] = inode
    for name in ("mnt", "ipc", "net"):
        namespace_inodes[f"/proc/self/ns/{name}"] = 99999
    original_text, original_stat, original_lstat, original_resolve = (
        Path.read_text,
        Path.stat,
        Path.lstat,
        Path.resolve,
    )

    def read_text(path, **kwargs):
        for pid, role in process_roles.items():
            if str(path) == f"/proc/{pid}/status":
                return "\n".join(f"{key}:\t{value}" for key, value in process_fields[pid].items())
            if str(path) == f"/proc/{pid}/cgroup":
                return "0::" + pins[role]["ControlGroup"] + "\n"
        return original_text(path, **kwargs)

    def fake_stat(path, **kwargs):
        if str(path) in namespace_inodes:
            return SimpleNamespace(st_ino=namespace_inodes[str(path)])
        return original_stat(path, **kwargs)

    def fake_lstat(path):
        for role in harness.ROLES:
            if str(path).startswith(statuses[role]["state"]):
                return SimpleNamespace(st_mode=stat.S_IFDIR | 0o700, st_uid=statuses[role]["uid"])
        return original_lstat(path)

    monkeypatch.setattr(Path, "read_text", read_text)
    monkeypatch.setattr(Path, "stat", fake_stat)
    monkeypatch.setattr(Path, "lstat", fake_lstat)
    monkeypatch.setattr(
        Path,
        "resolve",
        lambda path, **kwargs: Path("/usr/bin") / executables[str(path)]
        if str(path) in executables
        else original_resolve(path, **kwargs),
    )
    monkeypatch.setattr(Path, "exists", lambda path: str(path) not in missing)
    monkeypatch.setattr(Path, "is_symlink", lambda path: False)
    return statuses, pins, process_fields, namespace_inodes, missing


def test_host_verifies_process_identity_namespaces_resources_and_systemd_properties(
    host_status_fixture,
):
    statuses, pins, _fields, _inodes, _missing = host_status_fixture
    harness.validate_status_pair(statuses, pins, RUN_ID)


def test_worker_claim_cannot_override_host_process_capabilities_or_groups(host_status_fixture):
    statuses, pins, fields, _inodes, _missing = host_status_fixture
    pid = statuses["root_signer"]["pid"]
    fields[pid]["CapBnd"] = "1"
    with pytest.raises(harness.QualificationFailed, match="HOST_PROCESS_IDENTITY_MISMATCH"):
        harness.validate_status_pair(statuses, pins, RUN_ID)
    fields[pid]["CapBnd"] = "0"
    fields[pid]["Groups"] += " 7777"
    with pytest.raises(harness.QualificationFailed, match="HOST_PROCESS_IDENTITY_MISMATCH"):
        harness.validate_status_pair(statuses, pins, RUN_ID)


def test_namespace_receipts_must_match_host_and_remain_separate_from_host(host_status_fixture):
    statuses, pins, _fields, inodes, _missing = host_status_fixture
    root = statuses["root_signer"]
    root["namespaces"]["net"] += 1
    with pytest.raises(harness.QualificationBlocked, match="HOST_NAMESPACE_MISMATCH"):
        harness.validate_status_pair(statuses, pins, RUN_ID)
    inodes[f"/proc/{root['pid']}/ns/net"] = root["namespaces"]["net"]
    inodes["/proc/self/ns/net"] = root["namespaces"]["net"]
    with pytest.raises(harness.QualificationFailed, match="HOST_NAMESPACE_SHARED"):
        harness.validate_status_pair(statuses, pins, RUN_ID)


def test_both_roles_must_have_distinct_host_confirmed_namespaces(host_status_fixture):
    statuses, pins, _fields, inodes, _missing = host_status_fixture
    root, history = statuses["root_signer"], statuses["history_signer"]
    history["namespaces"]["ipc"] = root["namespaces"]["ipc"]
    inodes[f"/proc/{history['pid']}/ns/ipc"] = root["namespaces"]["ipc"]
    for key in ("bus_pid", "keyring_pid"):
        inodes[f"/proc/{history[key]}/ns/ipc"] = root["namespaces"]["ipc"]
    with pytest.raises(harness.QualificationFailed, match="CROSS_ROLE_NAMESPACE_SHARED"):
        harness.validate_status_pair(statuses, pins, RUN_ID)


def test_missing_native_resource_or_live_service_vetoes_status_pair(host_status_fixture):
    statuses, pins, _fields, _inodes, missing = host_status_fixture
    missing.add(statuses["history_signer"]["bus"])
    with pytest.raises(harness.QualificationBlocked, match="NATIVE_RESOURCE_MISSING"):
        harness.validate_status_pair(statuses, pins, RUN_ID)
    missing.clear()
    statuses["history_signer"]["native_service_live"] = False
    with pytest.raises(harness.QualificationBlocked, match="INVALID_ROLE_LIVENESS_OR_IDENTITY"):
        harness.validate_status_pair(statuses, pins, RUN_ID)


def test_status_pair_requires_actual_nnp_and_configured_isolation_properties(host_status_fixture):
    statuses, pins, _fields, _inodes, _missing = host_status_fixture
    statuses["root_signer"]["no_new_privs"] = False
    with pytest.raises(harness.QualificationFailed, match="NO_NEW_PRIVILEGES_DISABLED"):
        harness.validate_status_pair(statuses, pins, RUN_ID)
    statuses["root_signer"]["no_new_privs"] = True
    pins["history_signer"]["PrivateIPC"] = "no"
    with pytest.raises(harness.QualificationFailed, match="SYSTEMD_ISOLATION_PROPERTY_MISSING"):
        harness.validate_status_pair(statuses, pins, RUN_ID)


def test_same_public_key_is_a_separation_failure_despite_distinct_processes(host_status_fixture):
    statuses, pins, _fields, _inodes, _missing = host_status_fixture
    statuses["history_signer"]["public_key_b64"] = statuses["root_signer"]["public_key_b64"]
    with pytest.raises(harness.QualificationFailed, match="CROSS_ROLE_IDENTITY_COLLISION"):
        harness.validate_status_pair(statuses, pins, RUN_ID)


def test_native_children_must_inherit_role_security_and_namespace(host_status_fixture):
    statuses, pins, fields, inodes, _missing = host_status_fixture
    child_pid = statuses["history_signer"]["keyring_pid"]
    fields[child_pid]["CapAmb"] = "1"
    with pytest.raises(harness.QualificationFailed, match="NATIVE_CHILD_SECURITY_MISMATCH"):
        harness.validate_status_pair(statuses, pins, RUN_ID)
    fields[child_pid]["CapAmb"] = "0"
    inodes[f"/proc/{child_pid}/ns/net"] += 1
    with pytest.raises(harness.QualificationFailed, match="NATIVE_CHILD_NAMESPACE_MISMATCH"):
        harness.validate_status_pair(statuses, pins, RUN_ID)


def test_missing_native_child_identity_cannot_establish_bus_liveness(host_status_fixture):
    statuses, pins, _fields, _inodes, _missing = host_status_fixture
    statuses["history_signer"]["bus_pid"] = True
    with pytest.raises(harness.QualificationBlocked, match="NATIVE_CHILD_IDENTITY_MISSING"):
        harness.validate_status_pair(statuses, pins, RUN_ID)
