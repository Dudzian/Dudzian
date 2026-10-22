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

    # The orchestration fixture never opens or signals a host process. Tests
    # below exercise the actual pidfd helpers through their class methods.
    def capture(role, snapshot):
        if int(snapshot["MainPID"]) > 1:
            run.process_handles[role] = 701 + harness.ROLES.index(role)
            run.process_identities[role] = _process_receipt(snapshot)

    original_close = harness.os.close
    monkeypatch.setattr(run, "_capture_process", capture)
    monkeypatch.setattr(run, "_terminate_process", lambda role: True)
    monkeypatch.setattr(harness, "_process_finished", lambda handle, timeout=0: False)
    monkeypatch.setattr(
        harness.os, "close", lambda handle: None if handle in {701, 702} else original_close(handle)
    )
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


def test_cleanup_terminates_pinned_process_then_requires_unit_and_runtime_gone(
    local_run, monkeypatch
):
    local_run.create()
    local_run.attempted = ["root_signer"]
    local_run.pins["root_signer"] = owned_unit()
    snapshots = iter((owned_unit(), unused_unit()))
    monkeypatch.setattr(harness, "inspect_unit", lambda name: next(snapshots))
    commands = []
    terminated = []
    monkeypatch.setattr(
        local_run, "_terminate_process", lambda role: terminated.append(role) or True
    )
    monkeypatch.setattr(harness, "_command", lambda argv, **kwargs: commands.append((argv, kwargs)))
    assert local_run.cleanup() is True
    assert commands == []
    assert terminated == ["root_signer"]


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
        harness, "_request", lambda *args, **kwargs: ({"status": "PASS"}, b"unit public transcript")
    )
    assert local_run.request("root_signer", "TEST_STATUS") == {"status": "PASS"}
    assert local_run.transcripts == [b"unit public transcript"]
    local_run.transcripts *= 64
    with pytest.raises(harness.QualificationBlocked, match="TEST_TRANSCRIPT_LIMIT"):
        local_run.request("root_signer", "TEST_STATUS")


def test_live_statuses_requires_original_active_invocations(local_run, monkeypatch):
    local_run.pins = {
        role: owned_unit(role, str(1234 + index)) for index, role in enumerate(harness.ROLES)
    }
    for role in harness.ROLES:
        local_run._capture_process(role, local_run.pins[role])
    monkeypatch.setattr(
        harness,
        "_process_identity",
        lambda pid: deepcopy(
            next(
                identity
                for identity in local_run.process_identities.values()
                if identity["pid"] == pid
            )
        ),
    )
    monkeypatch.setattr(
        harness,
        "inspect_unit",
        lambda name: deepcopy(
            local_run.pins[
                "root_signer"
                if name == harness._unit_name(RUN_ID, "root_signer")
                else "history_signer"
            ]
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
    monkeypatch.setattr(harness, "_check_protected", lambda *args, **kwargs: None)

    def completed(code=0, stdout=b"", stderr=b""):
        monkeypatch.setattr(
            harness.subprocess,
            "run",
            lambda *args, **kwargs: SimpleNamespace(returncode=code, stdout=stdout, stderr=stderr),
        )

    completed(stdout=b"public")
    assert harness._command(["systemctl"]) == b"public"
    completed(code=1, stderr=b"private native diagnostic")
    with pytest.raises(harness.QualificationBlocked, match="HOST_COMMAND_UNAVAILABLE"):
        harness._command(["systemctl"])
    completed(stderr=b"x" * (harness.MAX_ARTIFACT + 1))
    with pytest.raises(harness.QualificationBlocked, match="HOST_OUTPUT_LIMIT"):
        harness._command(["systemctl"])


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
    monkeypatch.setattr(harness, "_prepared_location", lambda *args: {"commit": "a" * 40})
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
    installation = Path("/opt/cryptohunter-stage41b") / ("a" * 40)
    run = SimpleNamespace(
        run_id=RUN_ID,
        code_root=installation / "source",
        python=installation / "venv/bin/python",
        owned_paths=[],
        attempted=[],
        pins={},
        cleanup_issues=[],
        process_identities={},
        base=Path("/run/cryptohunter-stage41b-" + RUN_ID),
        hidden=Path("/run/cryptohunter-stage41b-hidden-" + RUN_ID),
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


@pytest.mark.parametrize("collision", ["masked", "drop-in"])
def test_masked_or_dropin_test_name_prevents_any_creation(local_run, monkeypatch, collision):
    snapshot = unused_unit()
    if collision == "masked":
        snapshot["LoadState"] = "masked"
    else:
        snapshot["DropInPaths"] = "/etc/systemd/system/signer.service.d/override.conf"
    monkeypatch.setattr(harness, "inspect_unit", lambda _name: snapshot)
    with pytest.raises(harness.QualificationBlocked, match="TEST_UNIT_COLLISION"):
        local_run.create()
    assert local_run.owned_paths == []
    assert not local_run.base.exists()


def test_immediate_worker_exit_pins_zero_pid_invocation_for_cleanup(local_run, monkeypatch):
    manager = {}
    observations = {}

    def command(argv, **kwargs):
        if argv[0] == "systemd-run":
            name = next(item.removeprefix("--unit=") for item in argv if item.startswith("--unit="))
            manager[name] = owned_unit(pid="0")
            manager[name]["ActiveState"] = "failed"
        else:
            pytest.fail("an exited test worker must never be stopped by unit name")
        return b""

    def inspect(name):
        if name not in manager:
            return unused_unit()
        observations[name] = observations.get(name, 0) + 1
        return unused_unit() if observations[name] >= 3 else deepcopy(manager[name])

    monkeypatch.setattr(harness, "_command", command)
    monkeypatch.setattr(harness, "inspect_unit", inspect)
    with pytest.raises(harness.QualificationBlocked, match="TEST_SERVICE_START_FAILED"):
        local_run.start()
    assert local_run.pins["root_signer"]["MainPID"] == "0"
    assert local_run.cleanup() is True
    assert local_run.process_handles == {}


@pytest.mark.parametrize("failure", ["systemd-run-error", "systemd-run-timeout"])
def test_command_failure_after_manager_creation_recovers_owned_cleanup_receipt(
    local_run, monkeypatch, failure
):
    manager = {}
    observations = {}

    def command(argv, **kwargs):
        if argv[0] == "systemd-run":
            name = next(item.removeprefix("--unit=") for item in argv if item.startswith("--unit="))
            manager[name] = owned_unit(pid="0")
            manager[name]["ActiveState"] = "failed"
            if failure == "systemd-run-timeout":
                raise subprocess.TimeoutExpired(argv, 20)
            raise harness.QualificationBlocked("HOST_COMMAND_UNAVAILABLE")
        pytest.fail("failed StartTransientUnit must not be undone by unit name")

    def inspect(name):
        if name not in manager:
            return unused_unit()
        observations[name] = observations.get(name, 0) + 1
        return unused_unit() if observations[name] >= 3 else deepcopy(manager[name])

    monkeypatch.setattr(harness, "_command", command)
    monkeypatch.setattr(harness, "inspect_unit", inspect)
    with pytest.raises((harness.QualificationBlocked, subprocess.TimeoutExpired)):
        local_run.start()
    assert local_run.pins["root_signer"]["InvocationID"] == "b" * 32
    assert local_run.cleanup() is True
    assert local_run.process_handles == {}


def test_command_error_without_receipt_preserves_unproven_outcome(local_run, monkeypatch):
    commands = []

    def command(argv, **kwargs):
        commands.append(argv)
        raise harness.QualificationBlocked("HOST_COMMAND_UNAVAILABLE")

    monkeypatch.setattr(harness, "_command", command)
    ticks = iter(range(20))
    monkeypatch.setattr(harness.time, "monotonic", lambda: next(ticks))
    monkeypatch.setattr(harness.time, "sleep", lambda _duration: None)
    with pytest.raises(harness.QualificationBlocked, match="HOST_COMMAND_UNAVAILABLE"):
        local_run.start()
    assert local_run.pins == {}
    assert local_run.cleanup() is False
    assert any(issue["reason"] == "STARTUP_OUTCOME_UNPROVEN" for issue in local_run.cleanup_issues)
    assert len(commands) == 1 and commands[0][0] == "systemd-run"


def test_exit_before_invocation_is_observed_retains_unknown_unit(local_run, monkeypatch):
    before_start = True
    unknown = owned_unit(pid="0")
    unknown.update(InvocationID="", ActiveState="failed")
    stopped = []

    def inspect(_name):
        return unused_unit() if before_start else unknown

    def command(argv, **kwargs):
        nonlocal before_start
        before_start = False
        if argv[:2] == ["systemctl", "stop"]:
            stopped.append(argv[2])
        return b""

    monkeypatch.setattr(harness, "inspect_unit", inspect)
    monkeypatch.setattr(harness, "_command", command)
    with pytest.raises(harness.QualificationBlocked):
        local_run.start()
    assert local_run.pins == {}
    assert local_run.cleanup() is False
    assert stopped == []


@pytest.mark.parametrize("change", ["MainPID", "InvocationID"])
def test_start_rejects_pid_or_invocation_replacement_after_first_pin(
    local_run, monkeypatch, change
):
    launched, observations = False, 0

    def inspect(_name):
        nonlocal observations
        if not launched:
            return unused_unit()
        observations += 1
        value = owned_unit()
        if observations > 1:
            value[change] = "1235" if change == "MainPID" else "c" * 32
        return value

    def command(argv, **kwargs):
        nonlocal launched
        launched = True
        return b""

    monkeypatch.setattr(harness, "inspect_unit", inspect)
    monkeypatch.setattr(harness, "_command", command)
    monkeypatch.setattr(Path, "exists", lambda _path: False)
    with pytest.raises(harness.QualificationBlocked, match="TEST_UNIT_OR_PROCESS_REPLACED"):
        local_run.start()
    assert local_run.pins["root_signer"][change] == ("1234" if change == "MainPID" else "b" * 32)
    assert local_run.cleanup() is False


def test_one_live_signer_and_one_failed_signer_are_both_cleaned_by_receipt(local_run, monkeypatch):
    manager = {}
    signalled = []
    failed_observations = 0

    def command(argv, **kwargs):
        if argv[0] == "systemd-run":
            name = next(item.removeprefix("--unit=") for item in argv if item.startswith("--unit="))
            role = (
                "root_signer"
                if name == harness._unit_name(RUN_ID, "root_signer")
                else "history_signer"
            )
            manager[name] = owned_unit(role, "1234" if role == "root_signer" else "0")
            if role == "history_signer":
                manager[name]["ActiveState"] = "failed"
        else:
            pytest.fail("cleanup must not stop either signer by unit name")
        return b""

    def inspect(name):
        nonlocal failed_observations
        value = manager.get(name, unused_unit())
        if value.get("ActiveState") == "failed":
            failed_observations += 1
            if failed_observations >= 3:
                return unused_unit()
        return deepcopy(value)

    def terminate(role):
        signalled.append(role)
        manager[harness._unit_name(RUN_ID, role)] = unused_unit()
        return True

    monkeypatch.setattr(harness, "inspect_unit", inspect)
    monkeypatch.setattr(local_run, "_terminate_process", terminate)
    monkeypatch.setattr(harness, "_command", command)
    monkeypatch.setattr(
        Path,
        "exists",
        lambda path: "state-root_signer" in path.parts and path.name == "control.sock",
    )
    with pytest.raises(harness.QualificationBlocked, match="TEST_SERVICE_START_FAILED"):
        local_run.start()
    assert tuple(local_run.pins) == harness.ROLES
    assert local_run.cleanup() is True
    assert signalled == ["root_signer"]


def test_startup_timeout_keeps_original_receipt_and_cleans_it(local_run, monkeypatch):
    launched = False
    ticks = iter((0, 1, 25))
    signalled = []

    def inspect(_name):
        return owned_unit() if launched else unused_unit()

    def command(argv, **kwargs):
        nonlocal launched
        if argv[0] == "systemd-run":
            launched = True
        else:
            pytest.fail("startup timeout must not cause a name-based stop")
        return b""

    def terminate(role):
        nonlocal launched
        signalled.append(role)
        launched = False
        return True

    monkeypatch.setattr(harness, "inspect_unit", inspect)
    monkeypatch.setattr(harness, "_command", command)
    monkeypatch.setattr(local_run, "_terminate_process", terminate)
    monkeypatch.setattr(harness.time, "monotonic", lambda: next(ticks))
    monkeypatch.setattr(harness.time, "sleep", lambda _duration: None)
    monkeypatch.setattr(Path, "exists", lambda _path: False)
    with pytest.raises(harness.QualificationBlocked, match="TEST_SERVICE_START_TIMEOUT"):
        local_run.start()
    assert local_run.pins["root_signer"]["MainPID"] == "1234"
    monkeypatch.setattr(harness.time, "monotonic", lambda: 0)
    assert local_run.cleanup() is True
    assert signalled == ["root_signer"]


def test_exited_invocation_with_zero_pid_is_cleaned_without_requiring_live_worker(
    local_run, monkeypatch
):
    local_run.attempted = ["root_signer"]
    local_run.pins["root_signer"] = owned_unit()
    ended = owned_unit(pid="0")
    ended["ActiveState"] = "inactive"
    observations = iter((ended, unused_unit()))
    monkeypatch.setattr(harness, "inspect_unit", lambda _name: next(observations))
    commands = []
    monkeypatch.setattr(harness, "_command", lambda argv, **kwargs: commands.append(argv) or b"")
    assert local_run.cleanup() is True
    assert commands == []


def test_collected_unit_with_unknown_runtime_socket_is_retained_and_blocked(local_run):
    local_run.attempted = ["root_signer"]
    local_run.pins["root_signer"] = owned_unit()
    runtime = harness._state(RUN_ID, "root_signer")
    runtime.mkdir()
    unknown = runtime / "control.sock"
    unknown.write_bytes(b"not-a-owned-test-socket")
    assert local_run.cleanup() is False
    assert unknown.read_bytes() == b"not-a-owned-test-socket"


def test_cleanup_suppresses_repeated_interrupt_and_restores_all_original_handlers(
    main_run, monkeypatch, capsys
):
    run, handlers = main_run
    checks = []

    def execute(_run, _boundaries):
        run.attempted.append("root_signer")
        handlers[signal.SIGTERM](signal.SIGTERM, None)

    def cleanup():
        checks.append(
            all(
                handlers[number] == signal.SIG_IGN
                for number in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP)
            )
        )
        run.cleaned = True
        return True

    run.cleanup = cleanup
    monkeypatch.setattr(harness, "_execute", execute)
    assert harness.main(["--execute"]) == 2
    report = json.loads(capsys.readouterr().out)
    assert checks == [True]
    assert run.cleaned is True
    assert all(handler == signal.SIG_DFL for handler in handlers.values())
    assert report["boundaries"]["cleanup"]["status"] == "PASS"


def test_incomplete_cleanup_reports_exact_test_resources_and_never_passes(
    main_run, monkeypatch, capsys
):
    run, _handlers = main_run
    run.cleanup = lambda: False

    def execute(_run, _boundaries):
        run.attempted.append("root_signer")
        run.pins["root_signer"] = owned_unit()
        raise harness.QualificationBlocked("TEST_SERVICE_START_FAILED")

    monkeypatch.setattr(harness, "_execute", execute)
    assert harness.main(["--execute"]) == 2
    report = json.loads(capsys.readouterr().out)
    assert report["status"] == "BLOCKED"
    assert report["boundaries"]["cleanup"]["status"] == "BLOCKED"
    assert report["scope_paths_requiring_review"] == [str(run.base), str(run.hidden)]
    assert report["resources_requiring_review"][0]["unit"] == harness._unit_name(
        RUN_ID, "root_signer"
    )


def _process_receipt(snapshot):
    return {
        "pid": int(snapshot["MainPID"]),
        "start_time": 987654,
        "uid": 61011,
        "gid": 62011,
        "control_group": snapshot["ControlGroup"],
    }


def test_pidfd_capture_pins_process_starttime_and_exact_invocation(local_run, monkeypatch):
    snapshot = owned_unit()
    identity = _process_receipt(snapshot)
    opened, closed = [], []
    monkeypatch.setattr(
        harness.os, "pidfd_open", lambda pid, flags: opened.append((pid, flags)) or 701
    )
    monkeypatch.setattr(harness.os, "close", closed.append)
    monkeypatch.setattr(harness, "_process_identity", lambda pid: deepcopy(identity))
    monkeypatch.setattr(harness, "inspect_unit", lambda name: deepcopy(snapshot))
    harness.PhysicalRun._capture_process(local_run, "root_signer", snapshot)
    assert opened == [(1234, 0)]
    assert local_run.process_handles == {"root_signer": 701}
    assert local_run.process_identities == {"root_signer": identity}
    assert closed == []


@pytest.mark.parametrize("race", ["invocation", "starttime", "cgroup"])
def test_process_capture_closes_unproven_pidfd_on_identity_race(local_run, monkeypatch, race):
    snapshot = owned_unit()
    identity = _process_receipt(snapshot)
    changed = deepcopy(snapshot)
    observations = 0
    if race == "invocation":
        changed["InvocationID"] = "c" * 32
    if race == "cgroup":
        identity["control_group"] = "/system.slice/unknown.service"

    def read_identity(pid):
        nonlocal observations
        observations += 1
        result = deepcopy(identity)
        if race == "starttime" and observations > 1:
            result["start_time"] += 1
        return result

    closed = []
    monkeypatch.setattr(harness.os, "pidfd_open", lambda pid, flags: 701)
    monkeypatch.setattr(harness.os, "close", closed.append)
    monkeypatch.setattr(harness, "_process_identity", read_identity)
    monkeypatch.setattr(harness, "inspect_unit", lambda name: deepcopy(changed))
    with pytest.raises(harness.QualificationBlocked):
        harness.PhysicalRun._capture_process(local_run, "root_signer", snapshot)
    assert local_run.process_handles == {}
    assert local_run.process_identities == {}
    assert closed == [701]


def test_termination_targets_only_pinned_descriptor_and_never_unit_name(local_run, monkeypatch):
    snapshot = owned_unit()
    identity = _process_receipt(snapshot)
    local_run.pins["root_signer"] = snapshot
    local_run.process_handles["root_signer"] = 701
    local_run.process_identities["root_signer"] = identity
    readiness = iter((False, True))
    signals = []
    monkeypatch.setattr(harness, "_process_finished", lambda handle, timeout=0: next(readiness))
    monkeypatch.setattr(harness, "inspect_unit", lambda name: deepcopy(snapshot))
    monkeypatch.setattr(harness, "_process_identity", lambda pid: deepcopy(identity))
    monkeypatch.setattr(harness.signal, "pidfd_send_signal", lambda *args: signals.append(args))
    monkeypatch.setattr(
        harness,
        "_command",
        lambda *args, **kwargs: pytest.fail("cleanup must not signal by systemd unit name"),
    )
    assert harness.PhysicalRun._terminate_process(local_run, "root_signer") is True
    assert signals == [(701, signal.SIGTERM, None, 0)]


@pytest.mark.parametrize("race", ["invocation", "pid", "starttime"])
def test_cleanup_sends_no_signal_after_unit_or_pid_replacement(local_run, monkeypatch, race):
    snapshot = owned_unit()
    identity = _process_receipt(snapshot)
    local_run.pins["root_signer"] = deepcopy(snapshot)
    local_run.process_handles["root_signer"] = 701
    local_run.process_identities["root_signer"] = deepcopy(identity)
    if race == "invocation":
        snapshot["InvocationID"] = "c" * 32
    elif race == "pid":
        snapshot["MainPID"] = "5678"
    else:
        identity["start_time"] += 1
    monkeypatch.setattr(harness, "_process_finished", lambda handle, timeout=0: False)
    monkeypatch.setattr(harness, "inspect_unit", lambda name: deepcopy(snapshot))
    monkeypatch.setattr(harness, "_process_identity", lambda pid: deepcopy(identity))
    monkeypatch.setattr(
        harness.signal,
        "pidfd_send_signal",
        lambda *args: pytest.fail("ambiguous process must not be signalled"),
    )
    assert harness.PhysicalRun._terminate_process(local_run, "root_signer") is False
    assert local_run.cleanup_issues[-1]["reason"] == "PROCESS_OR_INVOCATION_CHANGED_NO_SIGNAL_SENT"


def test_pinned_process_timeout_is_reported_after_descriptor_only_escalation(
    local_run, monkeypatch
):
    snapshot = owned_unit()
    identity = _process_receipt(snapshot)
    local_run.pins["root_signer"] = snapshot
    local_run.process_handles["root_signer"] = 701
    local_run.process_identities["root_signer"] = identity
    signals = []
    monkeypatch.setattr(harness, "_process_finished", lambda handle, timeout=0: False)
    monkeypatch.setattr(harness, "inspect_unit", lambda name: deepcopy(snapshot))
    monkeypatch.setattr(harness, "_process_identity", lambda pid: deepcopy(identity))
    monkeypatch.setattr(harness.signal, "pidfd_send_signal", lambda *args: signals.append(args))
    assert harness.PhysicalRun._terminate_process(local_run, "root_signer") is False
    assert [item[1] for item in signals] == [signal.SIGTERM, signal.SIGKILL]
    assert all(item[0] == 701 for item in signals)
    assert local_run.cleanup_issues[-1]["reason"] == "PINNED_PROCESS_DID_NOT_EXIT"


def test_host_execution_uses_fixed_program_and_sanitized_environment(monkeypatch):
    seen = []
    monkeypatch.setattr(harness, "_check_protected", lambda *args, **kwargs: None)
    monkeypatch.setenv("LD_PRELOAD", "/unit-untrusted/loader.so")
    monkeypatch.setenv("SYSTEMD_HOST", "unit-untrusted-host")
    monkeypatch.setenv("PATH", "/unit-untrusted/bin")
    monkeypatch.setattr(
        harness.subprocess,
        "run",
        lambda argv, **kwargs: seen.append((argv, kwargs))
        or SimpleNamespace(returncode=0, stdout=b"public", stderr=b""),
    )
    assert harness._command(["systemctl", "show", "fixture.service"]) == b"public"
    argv, options = seen[0]
    assert argv[0] == "/usr/bin/systemctl"
    assert options["env"] == {
        "PATH": "/usr/sbin:/usr/bin:/sbin:/bin",
        "LANG": "C.UTF-8",
        "LC_ALL": "C.UTF-8",
    }
    assert "LD_PRELOAD" not in options["env"] and "SYSTEMD_HOST" not in options["env"]
    for argv in (["/unit-untrusted/systemctl"], ["sh"], []):
        with pytest.raises(harness.QualificationBlocked, match="HOST_PROGRAM_NOT_ALLOWLISTED"):
            harness._host_argv(argv)


def test_process_identity_records_starttime_despite_parentheses_in_comm(monkeypatch):
    suffix = ["S", *(["0"] * 18), "987654"]
    files = {
        "/proc/1234/status": "Uid:\t61011 61011 61011 61011\nGid:\t62011 62011 62011 62011\n",
        "/proc/1234/stat": "1234 (unit process with ) parentheses) " + " ".join(suffix),
        "/proc/1234/cgroup": "0::/system.slice/cryptohunter-root-proof-signer.service\n",
    }
    monkeypatch.setattr(Path, "read_text", lambda path, **kwargs: files[str(path)])
    assert harness._process_identity(1234) == {
        "pid": 1234,
        "start_time": 987654,
        "uid": 61011,
        "gid": 62011,
        "control_group": "/system.slice/cryptohunter-root-proof-signer.service",
    }
    files["/proc/1234/status"] = "Uid:\t0 0 0 0\nGid:\t62011 62011 62011 62011\n"
    with pytest.raises(harness.QualificationBlocked, match="PROCESS_IDENTITY_UNPROVEN"):
        harness._process_identity(1234)
    files["/proc/1234/status"] = "Uid:\t61011 61012 61011 61011\nGid:\t62011 62011 62011 62011\n"
    with pytest.raises(harness.QualificationBlocked, match="PROCESS_IDENTITY_UNPROVEN"):
        harness._process_identity(1234)
    files["/proc/1234/status"] = "Uid:\t61011 61011 61011 61011\nGid:\t62011 62011 62011 62011\n"
    files["/proc/1234/stat"] = "1234 (unit) S truncated"
    with pytest.raises(harness.QualificationBlocked, match="PROCESS_STARTTIME_UNPROVEN"):
        harness._process_identity(1234)


def test_prepared_checkout_and_interpreter_share_exact_committed_installation():
    installation = Path("/opt/cryptohunter-stage41b") / ("a" * 40)
    assert harness._prepared_location(
        installation / "source", installation / "venv/bin/python"
    ) == {"commit": "a" * 40, "installation": str(installation)}
    for source, interpreter in (
        (Path("/home/dudzian/CryptoHunter"), installation / "venv/bin/python"),
        (installation / "source", Path("/usr/bin/python3")),
        (Path("/opt/unreviewed") / ("a" * 40) / "source", installation / "venv/bin/python"),
        (
            Path("/opt/cryptohunter-stage41b/unpinned/source"),
            Path("/opt/cryptohunter-stage41b/unpinned/venv/bin/python"),
        ),
    ):
        with pytest.raises(
            harness.QualificationBlocked, match="PREPARED_QUALIFICATION_ENVIRONMENT_REQUIRED"
        ):
            harness._prepared_location(source, interpreter)


def test_readonly_cleanup_inspection_requires_all_resources_absent(monkeypatch):
    monkeypatch.setattr(harness, "inspect_unit", lambda name: unused_unit())
    monkeypatch.setattr(Path, "lstat", lambda path: (_ for _ in ()).throw(FileNotFoundError()))
    monkeypatch.setattr(
        harness,
        "_command",
        lambda *args, **kwargs: pytest.fail("read-only inspection must not execute mutations"),
    )
    report = harness.inspect_cleanup(RUN_ID)
    assert report["status"] == "PASS"
    assert report["read_only"] is True
    assert report["physical_qualification"] == "NOT_EXECUTED"
    assert report["production_private_control"] == "BLOCKED"
    assert report["authorizes_sign_bind_enrollment"] is False
    assert all(entry["reason"] == "UNIT_ABSENT" for entry in report["units"].values())
    assert all(entry["reason"] == "PATH_ABSENT" for entry in report["paths"])


def test_cleanup_inspection_retains_replacement_and_denied_metadata(monkeypatch):
    foreign = owned_unit()
    foreign["Description"] = "production signer"
    monkeypatch.setattr(harness, "inspect_unit", lambda name: foreign)
    monkeypatch.setattr(
        Path, "lstat", lambda path: (_ for _ in ()).throw(PermissionError("unit metadata denied"))
    )
    report = harness.inspect_cleanup(RUN_ID)
    assert report["status"] == "BLOCKED"
    assert all(
        entry["reason"] == "FOREIGN_OR_UNPROVEN_UNIT_NO_ACTION_TAKEN"
        for entry in report["units"].values()
    )
    assert all(entry["reason"] == "PATH_METADATA_UNAVAILABLE" for entry in report["paths"])


def test_cleanup_inspection_cli_never_constructs_or_executes_physical_run(monkeypatch, capsys):
    inspection = {"status": "PASS", "read_only": True, "physical_qualification": "NOT_EXECUTED"}
    monkeypatch.setattr(harness, "inspect_cleanup", lambda run_id: inspection)
    monkeypatch.setattr(
        harness,
        "PhysicalRun",
        lambda *args: pytest.fail("inspection must not create a physical run"),
    )
    assert harness.main(["--inspect-cleanup", "--run-id", RUN_ID]) == 0
    assert json.loads(capsys.readouterr().out) == inspection
    for arguments in (
        ["--inspect-cleanup"],
        ["--inspect-cleanup", "--run-id", "../untrusted"],
        ["--execute", "--run-id", RUN_ID],
    ):
        with pytest.raises(SystemExit) as rejected:
            harness.main(arguments)
        assert rejected.value.code == 2
