"""Hermetyczne testy odczytów natywnych; bez instalacji i zmian usług hosta."""

from __future__ import annotations

import copy
import hashlib
import json
import os
from pathlib import Path
import stat
import subprocess
import sys
from types import SimpleNamespace

import pytest

from deployment import linux_physical_native as native
from deployment.linux_physical_observations import ObservationError, UNIT_PROPERTIES
from tests.deployment.test_linux_physical_collector import (
    NEW_BOOT,
    OLD_BOOT,
    _snapshot,
    installation_fixture,
    observation_fixture,
)


@pytest.fixture
def driver(tmp_path):
    return native.NativeDriver(installation_fixture(tmp_path), "a" * 64)


def result(stdout="", returncode=0):
    return subprocess.CompletedProcess([], returncode, stdout=stdout, stderr="")


def test_state_creation_refuses_other_platform_before_mutation(driver, monkeypatch):
    monkeypatch.setattr(native.sys, "platform", "win32")
    monkeypatch.setattr(
        native, "_safe_parents", lambda *_: pytest.fail("Próba zmiany obcego hosta")
    )
    with pytest.raises(ObservationError, match="natywnego Linux"):
        driver.create_state()


def test_real_regular_file_is_bounded_and_observed(tmp_path, monkeypatch):
    path = tmp_path / "material"
    path.write_bytes(b"rzeczywiste bajty")
    assert native.read_regular(path, maximum=17) == b"rzeczywiste bajty"
    with pytest.raises(ObservationError):
        native.read_regular(path, maximum=3)
    monkeypatch.setattr(native, "_safe_parents", lambda _path, **_kwargs: None)
    observed = native.file_observation(path)
    assert observed["sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
    assert observed["uid"] == path.stat().st_uid
    assert observed["kind"] == "regular"


@pytest.mark.parametrize("kind", ["symlink", "directory", "fifo"])
def test_nonregular_input_refuses_without_blocking(tmp_path, kind):
    path = tmp_path / "material"
    if kind == "symlink":
        target = tmp_path / "target"
        target.write_bytes(b"bytes")
        path.symlink_to(target)
    elif kind == "directory":
        path.mkdir()
    else:
        os.mkfifo(path)
    with pytest.raises((OSError, ObservationError)):
        native.read_regular(path)


def test_read_refuses_file_changed_while_descriptor_is_held(tmp_path, monkeypatch):
    path = tmp_path / "material"
    path.write_bytes(b"known")
    actual = native.os.fstat
    calls = 0

    def changed(descriptor):
        nonlocal calls
        info = actual(descriptor)
        calls += 1
        return SimpleNamespace(
            st_dev=info.st_dev,
            st_ino=info.st_ino,
            st_size=info.st_size,
            st_mtime_ns=info.st_mtime_ns + (calls > 1),
            st_ctime_ns=info.st_ctime_ns,
            st_mode=info.st_mode,
            st_uid=info.st_uid,
        )

    monkeypatch.setattr(native.os, "fstat", changed)
    with pytest.raises(ObservationError, match="zmienił"):
        native.read_regular(path)


@pytest.mark.parametrize(
    "mode,uid", [(stat.S_IFDIR | 0o777, 0), (stat.S_IFDIR | 0o755, 12), (stat.S_IFLNK | 0o755, 0)]
)
def test_untrusted_parent_refuses(mode, uid, monkeypatch):
    monkeypatch.setattr(Path, "lstat", lambda _path: SimpleNamespace(st_mode=mode, st_uid=uid))
    with pytest.raises(ObservationError, match="katalog"):
        native._safe_parents(Path("/approved/material"))


def test_relative_parent_refuses():
    with pytest.raises(ObservationError, match="bezwzględna"):
        native._safe_parents(Path("relative/material"))


@pytest.mark.parametrize(
    "failure", [OSError("native"), subprocess.TimeoutExpired("tool", 2), UnicodeError("encoding")]
)
def test_command_failures_are_bounded_refusals(monkeypatch, failure):
    def fail(*_args, **_kwargs):
        raise failure

    monkeypatch.setattr(native.subprocess, "run", fail)
    with pytest.raises(ObservationError, match="wykonanie"):
        native.command(["/usr/bin/systemctl", "show", "known.service"], timeout=2)


def host_files():
    return {
        "/proc/1/comm": b"systemd\n",
        "/etc/machine-id": b"1" * 32 + b"\n",
        "/sys/class/dmi/id/product_uuid": b"11111111-2222-3333-4444-555555555555\n",
        "/proc/sys/kernel/random/boot_id": OLD_BOOT.encode(),
        "/usr/lib/os-release": b'ID=ubuntu\nVERSION_ID="24.04"\n',
    }


@pytest.mark.parametrize(
    "fault", [None, "not_root", "pid1", "os", "duplicate_os", "machine", "vm", "architecture"]
)
def test_host_identity_requires_native_physical_ubuntu(monkeypatch, fault):
    files = host_files()
    if fault == "pid1":
        files["/proc/1/comm"] = b"init"
    elif fault == "os":
        files["/usr/lib/os-release"] = b"ID=debian\nVERSION_ID=12\n"
    elif fault == "duplicate_os":
        files["/usr/lib/os-release"] += b"ID=ubuntu\n"
    elif fault == "machine":
        files["/etc/machine-id"] = b"unknown"
    monkeypatch.setattr(native.os, "geteuid", lambda: 1 if fault == "not_root" else 0)
    actual_is_dir = Path.is_dir
    monkeypatch.setattr(
        Path,
        "is_dir",
        lambda path: True if str(path) == "/run/systemd/system" else actual_is_dir(path),
    )
    monkeypatch.setattr(
        native, "read_regular", lambda path, **_kwargs: files.get(str(path), b"trusted-python")
    )
    monkeypatch.setattr(
        native, "command", lambda _args: result(returncode=0 if fault == "vm" else 1)
    )
    monkeypatch.setattr(
        native.platform, "machine", lambda: "aarch64" if fault == "architecture" else "x86_64"
    )
    if fault:
        with pytest.raises(ObservationError):
            native.host_identity()
    else:
        identity = native.host_identity()
        assert identity["virtualization"] == "none"
        assert identity["os_version"] == "24.04"
        assert identity["boot_id"] == OLD_BOOT
        assert identity["machine_id_sha256"] == native.sha256(b"1" * 32)


def process_stat(pid=4321, ticks=123, state="S"):
    return (
        f"{pid} (name with (parentheses)) ".encode()
        + " ".join([state] + ["0"] * 18 + [str(ticks)]).encode()
    )


@pytest.mark.parametrize(
    "raw",
    [
        b"1 bad",
        process_stat(pid=2),
        process_stat(state="Z"),
        process_stat(ticks=0),
        process_stat().replace(b"123", b"bad"),
    ],
)
def test_malformed_or_dead_process_identity_refuses(raw):
    with pytest.raises(ObservationError):
        native._start_ticks(raw, 4321)


@pytest.mark.parametrize("fault", [None, "reused_pid", "different_uids", "duplicate_status"])
def test_process_observation_binds_kernel_identity_and_executable(tmp_path, monkeypatch, fault):
    executable = tmp_path / "service"
    executable.write_bytes(b"trusted executable bytes")
    status = b"Uid:\t61001 61001 61001 61001\nGid:\t61001 61001 61001 61001\nGroups:\t61001\nNoNewPrivs:\t1\n"
    status += b"".join(
        name.encode() + b":\t0000000000000000\n"
        for name in ("CapInh", "CapPrm", "CapEff", "CapBnd", "CapAmb")
    )
    if fault == "different_uids":
        status = status.replace(b"61001 61001 61001 61001", b"61001 0 61001 61001", 1)
    elif fault == "duplicate_status":
        status += b"Uid:\t61001 61001 61001 61001\n"
    calls = 0

    def read(path, **_kwargs):
        nonlocal calls
        if path.name == "status":
            return status
        calls += 1
        return process_stat(ticks=124 if fault == "reused_pid" and calls > 1 else 123)

    actual_open = native.os.open
    monkeypatch.setattr(native, "read_regular", read)
    monkeypatch.setattr(
        native.os,
        "open",
        lambda path, flags: actual_open(
            executable if str(path) == "/proc/4321/exe" else path, flags
        ),
    )
    if fault:
        with pytest.raises(ObservationError):
            native.process_observation(4321)
    else:
        observed = native.process_observation(4321)
        assert observed["start_time_ticks"] == 123
        assert observed["uid"] == observed["gid"] == 61001
        assert observed["exe_sha256"] == native.sha256(executable.read_bytes())


@pytest.mark.parametrize(
    "stdout,code,expected",
    [
        ("", 1, None),
        ("cryptohunter-qualification\t1.0\tinstall ok installed", 0, "1.0"),
        ("invalid", 0, "error"),
        ("", 2, "error"),
    ],
)
def test_package_native_result_is_not_a_pass_declaration(
    driver, monkeypatch, stdout, code, expected
):
    monkeypatch.setattr(native, "command", lambda _args: result(stdout, code))
    if expected == "error":
        with pytest.raises(ObservationError):
            driver.package()
    elif expected is None:
        assert driver.package() is None
    else:
        assert driver.package()["version"] == expected


@pytest.mark.parametrize("fault", [None, "missing", "duplicate", "unknown", "command_error"])
def test_systemd_properties_require_exact_complete_native_inventory(driver, monkeypatch, fault):
    rows = [f"{key}=observed" for key in UNIT_PROPERTIES]
    if fault == "missing":
        rows.pop()
    elif fault == "duplicate":
        rows.append(rows[0])
    elif fault == "unknown":
        rows.append("Status=PASS")
    monkeypatch.setattr(
        native,
        "command",
        lambda _args: result("\n".join(rows), 1 if fault == "command_error" else 0),
    )
    if fault:
        with pytest.raises(ObservationError):
            driver.properties("ch-qual-runtime.service")
    else:
        assert set(driver.properties("ch-qual-runtime.service")) == set(UNIT_PROPERTIES)


@pytest.mark.parametrize("fault", [None, "empty", "multiple", "not_object", "command_error"])
def test_journal_records_boot_and_unit_without_message_secrets(driver, monkeypatch, fault):
    calls = []
    row = {
        "_BOOT_ID": OLD_BOOT.replace("-", ""),
        "UNIT": "ch-qual-runtime.service",
        "__REALTIME_TIMESTAMP": "12345",
    }
    raw = json.dumps(row)
    if fault == "empty":
        raw = ""
    elif fault == "multiple":
        raw += "\n" + raw
    elif fault == "not_object":
        raw = "[]"

    def run(args):
        calls.append(args)
        return result(raw, 1 if fault == "command_error" else 0)

    monkeypatch.setattr(native, "command", run)
    if fault:
        with pytest.raises(ObservationError):
            driver.journal("ch-qual-runtime.service", OLD_BOOT)
    else:
        assert driver.journal("ch-qual-runtime.service", OLD_BOOT) == {
            "boot_id": OLD_BOOT.replace("-", ""),
            "unit": "ch-qual-runtime.service",
            "realtime_usec": 12345,
        }
        assert "--boot=" + OLD_BOOT.replace("-", "") in calls[0]
        assert "--lines=1" in calls[0]
        assert all("MESSAGE" not in item for item in calls[0])


@pytest.mark.parametrize("fault", [None, "missing_identity", "missing_cluster"])
def test_postgresql_uses_only_scoped_readonly_peer_connection(driver, monkeypatch, fault):
    calls = []
    connect_args = []

    class Cursor:
        def __init__(self, rows=None, row=None):
            self.rows = rows
            self.row = row

        def fetchall(self):
            return self.rows

        def fetchone(self):
            return self.row

    class Connection:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def execute(self, sql, params=None):
            calls.append((sql, params))
            if "pg_settings" in sql:
                return Cursor(
                    rows=[("listen_addresses", ""), ("fsync", "on"), ("synchronous_commit", "on")]
                )
            if "system_user,current_user" in sql:
                return Cursor(
                    row=None
                    if fault == "missing_identity"
                    else ("peer:root", "postgres", "postgres", None, None, False)
                )
            if "pg_hba_file_rules" in sql:
                return Cursor(rows=[("local", ["all"], ["all"], None, None, "peer", None, None)])
            if "pg_ident_file_mappings" in sql:
                return Cursor(rows=[("ch", "root", "postgres", None)])
            assert "pg_control_system()" in sql
            return Cursor(row=None if fault == "missing_cluster" else ("123456789",))

    def connect(**kwargs):
        connect_args.append(kwargs)
        return Connection()

    monkeypatch.setitem(sys.modules, "psycopg", SimpleNamespace(connect=connect))
    if fault:
        with pytest.raises(ObservationError):
            driver.postgres()
    else:
        observed = driver.postgres()
        assert observed["system_user"] == "peer:root"
        assert observed["system_identifier"] == "123456789"
        assert len(observed["hba_sha256"]) == 64
    connection = connect_args[0]
    assert connection["host"] == driver.installation["postgres"]["socket"]
    assert connection["password"] is None and connection["passfile"] == "/dev/null"
    assert connection["connect_timeout"] == 5
    assert "default_transaction_read_only=on" in connection["options"]
    assert "statement_timeout=5000" in connection["options"]
    assert all(sql.startswith("SELECT ") for sql, _params in calls)


@pytest.mark.parametrize("fault", [None, "cgroup_substitution", "changed_systemd"])
def test_snapshot_binds_processes_files_and_cgroup(driver, monkeypatch, fault):
    expected = _snapshot(driver.installation, step=0)
    properties = {unit["name"]: unit["properties"] for unit in expected["units"]}
    count = {}

    def props(name):
        count[name] = count.get(name, 0) + 1
        value = copy.deepcopy(properties[name])
        if fault == "cgroup_substitution":
            value["ControlGroup"] = "/other.slice/foreign.service"
        elif fault == "changed_systemd" and count[name] > 1:
            value["MainPID"] = "99999"
        return value

    monkeypatch.setattr(driver, "properties", props)
    monkeypatch.setattr(driver, "package", lambda: expected["package"])
    monkeypatch.setattr(driver, "postgres", lambda: expected["postgres"])
    monkeypatch.setattr(
        driver,
        "journal",
        lambda name: next(unit["journal"] for unit in expected["units"] if unit["name"] == name),
    )
    monkeypatch.setattr(native, "_safe_parents", lambda _path, **_kwargs: None)
    monkeypatch.setattr(
        native,
        "process_observation",
        lambda pid: next(
            proc for unit in expected["units"] for proc in unit["processes"] if proc["pid"] == pid
        ),
    )
    file_rows = {
        row["path"]: row
        for row in [
            *expected["configuration"],
            expected["state"],
            *[unit[k] for unit in expected["units"] for k in ("unit_file", "executable")],
        ]
    }
    monkeypatch.setattr(native, "file_observation", lambda path: file_rows[str(path)])

    def read(path, **_kwargs):
        name = path.parent.name
        return (properties[name]["MainPID"] + "\n").encode()

    monkeypatch.setattr(native, "read_regular", read)
    actual_lstat = Path.lstat

    def lstat(path):
        if str(path) == "/var/log/journal":
            return SimpleNamespace(st_mode=stat.S_IFDIR | 0o755, st_uid=0)
        if str(path) == driver.installation["ipc"][0]["path"]:
            return SimpleNamespace(st_mode=stat.S_IFSOCK | 0o660, st_uid=61002, st_gid=61002)
        return actual_lstat(path)

    monkeypatch.setattr(Path, "lstat", lstat)
    if fault:
        with pytest.raises(ObservationError):
            driver.snapshot()
    else:
        assert driver.snapshot() == expected


@pytest.mark.parametrize("fault", [None, "orphan", "failed_stop"])
def test_lifecycle_records_real_command_order_and_observes_orphans(driver, monkeypatch, fault):
    initial = _snapshot(driver.installation, step=0)
    monkeypatch.setattr(driver, "snapshot", lambda: initial)
    monkeypatch.setattr(driver, "_wait_snapshot", lambda: initial)
    monkeypatch.setattr(
        driver, "properties", lambda _name: {"ActiveState": "inactive", "MainPID": "0"}
    )
    monkeypatch.setattr(native, "read_regular", lambda _path, **_kwargs: b"")
    alive = {proc["pid"]: proc for unit in initial["units"] for proc in unit["processes"]}

    def process(pid):
        if fault == "orphan":
            return alive[pid]
        raise FileNotFoundError

    monkeypatch.setattr(native, "process_observation", process)
    commands = []

    def run(args):
        commands.append(args)
        return result(returncode=1 if fault == "failed_stop" else 0)

    monkeypatch.setattr(native, "command", run)
    if fault == "failed_stop":
        with pytest.raises(ObservationError):
            driver.lifecycle()
        assert len(commands) == 1
    else:
        observed = driver.lifecycle()
        assert len(commands) == 8
        assert (
            commands[0][2] == "stop" and commands[2][2] == "start" and commands[4][2] == "restart"
        )
        assert commands[6][2:5] == ["kill", "--kill-whom=all", "--signal=SIGKILL"]
        assert bool(observed["stopped"][0]["old_processes_alive"]) is (fault == "orphan")


def test_readiness_retry_is_bounded_and_never_turns_failure_into_snapshot(driver, monkeypatch):
    attempts = 0

    def fail():
        nonlocal attempts
        attempts += 1
        raise ObservationError("not ready")

    monkeypatch.setattr(driver, "snapshot", fail)
    monkeypatch.setattr(native.time, "sleep", lambda _seconds: None)
    with pytest.raises(ObservationError, match="limicie"):
        driver._wait_snapshot()
    assert attempts == 30


def test_readiness_wait_returns_only_observed_running_processes(driver, monkeypatch):
    waiting = _snapshot(driver.installation, step=0)
    waiting["units"][0]["processes"] = []
    expected = _snapshot(driver.installation, step=1)
    values = iter((waiting, expected))
    monkeypatch.setattr(driver, "snapshot", lambda: next(values))
    monkeypatch.setattr(native.time, "sleep", lambda _seconds: None)
    assert driver._wait_snapshot() == expected


@pytest.mark.parametrize("phase,exit_code", [("update", 0), ("failed_update", 1)])
def test_update_collect_retains_native_exit_code_and_before_after(
    driver, monkeypatch, phase, exit_code
):
    before = _snapshot(driver.installation, step=0, updated=phase == "failed_update")
    after = _snapshot(driver.installation, step=1, updated=True)
    previous = [{"payload": {"observations": {"snapshot": before}}}]
    called = []
    monkeypatch.setattr(driver, "snapshot", lambda: before)
    monkeypatch.setattr(driver, "_wait_snapshot", lambda: after)
    monkeypatch.setattr(driver, "install", lambda kind: called.append(kind) or exit_code)
    assert driver.collect(phase, previous) == {
        "before": before,
        "install_exit_code": exit_code,
        "after": after,
    }
    assert called == [phase]


def test_post_reboot_collection_refuses_same_boot_before_snapshot(driver, monkeypatch):
    monkeypatch.setattr(native, "host_identity", lambda: {"boot_id": OLD_BOOT})
    monkeypatch.setattr(
        driver, "snapshot", lambda: pytest.fail("Snapshot nie może zastąpić restartu")
    )
    records = [{}, {}, {}, {"payload": {"host": {"boot_id": OLD_BOOT}}}]
    with pytest.raises(ObservationError, match="restartu"):
        driver.collect("post_reboot", records)


def test_post_reboot_collection_reads_logs_of_exact_previous_boot(driver, monkeypatch):
    monkeypatch.setattr(native, "host_identity", lambda: {"boot_id": NEW_BOOT})
    expected = _snapshot(driver.installation, step=1, boot=NEW_BOOT)
    monkeypatch.setattr(driver, "snapshot", lambda: expected)
    calls = []
    monkeypatch.setattr(
        driver, "journal", lambda name, boot: calls.append((name, boot)) or {"boot": boot}
    )
    records = [{}, {}, {}, {"payload": {"host": {"boot_id": OLD_BOOT}}}]
    observed = driver.collect("post_reboot", records)
    assert observed["snapshot"] == expected
    assert calls == [(unit["name"], OLD_BOOT) for unit in driver.installation["units"]]


@pytest.mark.parametrize("code", [0, 1])
def test_installer_uses_exclusive_verified_copy_and_removes_only_its_material(
    driver, tmp_path, monkeypatch, code
):
    source = Path(driver.installation["packages"]["update"]["path"])
    source.write_bytes(b"exact reviewed installer bytes")
    driver.installation["packages"]["update"]["sha256"] = native.sha256(source.read_bytes())
    staging = tmp_path / "private-staging"
    actual_path = Path

    def path(value):
        return (
            staging if str(value) == "/var/lib/cryptohunter-stage9-evidence" else actual_path(value)
        )

    monkeypatch.setattr(native, "Path", path)
    monkeypatch.setattr(native, "_safe_parents", lambda _path, **_kwargs: None)
    # Warstwa uprawnień ma osobne odmowy; tu czytamy rzeczywiste bajty bez root.
    read_regular = native.read_regular
    monkeypatch.setattr(native, "read_regular", lambda path, **_kwargs: read_regular(path))
    commands = []

    def run(args, **kwargs):
        commands.append((args, kwargs))
        material = actual_path(args[-1])
        assert material.parent == staging
        assert material != source
        assert material.read_bytes() == b"exact reviewed installer bytes"
        assert stat.S_IMODE(material.stat().st_mode) == 0o600
        return result(returncode=code)

    monkeypatch.setattr(native, "command", run)
    assert driver.install("update") == code
    assert commands[0][0][:2] == ["/usr/bin/dpkg", "--install"]
    assert commands[0][1]["timeout"] == 300
    assert source.read_bytes() == b"exact reviewed installer bytes"
    assert list(staging.iterdir()) == []


def test_installer_rejects_changed_artifact_before_command(driver, tmp_path, monkeypatch):
    path = Path(driver.installation["packages"]["baseline"]["path"])
    path.write_bytes(b"substituted")
    monkeypatch.setattr(native, "_safe_parents", lambda _path, **_kwargs: None)
    read_regular = native.read_regular
    monkeypatch.setattr(native, "read_regular", lambda path, **_kwargs: read_regular(path))
    monkeypatch.setattr(
        native,
        "command",
        lambda *_args, **_kwargs: pytest.fail("Nie wolno wykonać podmienionego pakietu"),
    )
    with pytest.raises(ObservationError, match="Podmieniony"):
        driver.install("baseline")


def test_state_is_exclusive_random_material_in_dedicated_location(driver, tmp_path, monkeypatch):
    driver.state_path = tmp_path / "sentinel"
    monkeypatch.setattr(native, "_safe_parents", lambda _path, **_kwargs: None)
    owners = []
    monkeypatch.setattr(native.os, "fchown", lambda _fd, uid, gid: owners.append((uid, gid)))
    driver.create_state()
    content = driver.state_path.read_bytes()
    assert len(content) == 32
    assert owners == [(61001, 61001)]
    assert stat.S_IMODE(driver.state_path.stat().st_mode) == 0o600
    with pytest.raises(FileExistsError):
        driver.create_state()
    assert driver.state_path.read_bytes() == content


def test_baseline_observes_absence_without_installing(driver, monkeypatch):
    clean_paths = set(driver.installation["clean_paths"])
    actual_lstat = Path.lstat

    def lstat(path):
        if str(path) in clean_paths:
            raise FileNotFoundError
        return actual_lstat(path)

    monkeypatch.setattr(Path, "lstat", lstat)
    monkeypatch.setattr(driver, "package", lambda: None)
    monkeypatch.setattr(driver, "properties", lambda _name: {"LoadState": "not-found"})
    assert driver.collect("baseline", []) == observation_fixture(driver.installation)[0]


@pytest.mark.parametrize("fault", [None, "dirty", "failed_install"])
def test_clean_install_rechecks_absence_before_mutation(driver, monkeypatch, fault):
    clean = observation_fixture(driver.installation)[0]
    current = copy.deepcopy(clean)
    if fault == "dirty":
        current["paths"][0]["kind"] = "present"
    monkeypatch.setattr(driver, "baseline", lambda: current)
    expected = _snapshot(driver.installation, step=0)
    monkeypatch.setattr(driver, "_wait_snapshot", lambda: expected)
    calls = []
    monkeypatch.setattr(
        driver,
        "install",
        lambda kind: calls.append(("install", kind)) or (1 if fault == "failed_install" else 0),
    )
    monkeypatch.setattr(driver, "create_state", lambda: calls.append(("state", "new")))
    previous = [{"payload": {"observations": clean}}]
    if fault:
        with pytest.raises(ObservationError):
            driver.collect("clean_install", previous)
        assert calls == ([] if fault == "dirty" else [("install", "baseline")])
    else:
        assert driver.collect("clean_install", previous) == {
            "install_exit_code": 0,
            "snapshot": expected,
        }
        assert calls == [("install", "baseline"), ("state", "new")]


def test_update_state_drift_refuses_before_installer(driver, monkeypatch):
    before = _snapshot(driver.installation, step=0)
    old = copy.deepcopy(before)
    old["state"]["sha256"] = "c" * 64
    monkeypatch.setattr(driver, "snapshot", lambda: before)
    monkeypatch.setattr(
        driver, "install", lambda *_args: pytest.fail("Nie wolno aktualizować obcego stanu")
    )
    with pytest.raises(ObservationError, match="zmienił"):
        driver.collect("update", [{"payload": {"observations": {"snapshot": old}}}])
