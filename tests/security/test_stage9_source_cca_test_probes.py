"""Read-only host probes and lifetime races, simulated without host mutation."""

from contextlib import nullcontext
import copy
import os
from pathlib import Path
import stat
import subprocess
import sys
from types import SimpleNamespace

import pytest

from deployment import stage9_source_cca_test_runtime as runtime
from deployment.stage9_source_cca_test_package import HarnessRefusal, plan

pytestmark = pytest.mark.skipif(
    sys.platform != "linux", reason="Linux-only NSS, procfs and pidfd observation probes"
)


@pytest.mark.parametrize(
    "mode,uid,gid,links,directory,code",
    [
        (stat.S_IFREG | 0o644, 1000, 0, 1, False, "UNPROTECTED_HOST_PATH"),
        (stat.S_IFREG | 0o644, 0, 1000, 1, False, "UNPROTECTED_HOST_PATH"),
        (stat.S_IFREG | 0o666, 0, 0, 1, False, "UNPROTECTED_HOST_PATH"),
        (stat.S_IFLNK | 0o777, 0, 0, 1, False, "UNPROTECTED_HOST_PATH"),
        (stat.S_IFREG | 0o644, 0, 0, 2, False, "UNSAFE_HARDLINK"),
        (stat.S_IFREG | 0o644, 0, 0, 1, True, "UNPROTECTED_HOST_PATH"),
    ],
)
def test_host_path_requires_root_ownership_type_and_single_link(
    monkeypatch, mode, uid, gid, links, directory, code
):
    monkeypatch.setattr(
        Path,
        "lstat",
        lambda _: SimpleNamespace(st_mode=mode, st_uid=uid, st_gid=gid, st_nlink=links),
    )
    with pytest.raises(HarnessRefusal, match=code):
        runtime._protected(Path("/simulated"), directory=directory)


@pytest.mark.parametrize("directory", [False, True])
def test_protected_host_path_valid_metadata_is_returned(monkeypatch, directory):
    metadata = SimpleNamespace(
        st_mode=(stat.S_IFDIR if directory else stat.S_IFREG) | 0o755,
        st_uid=0,
        st_gid=0,
        st_nlink=2 if directory else 1,
    )
    monkeypatch.setattr(Path, "lstat", lambda _: metadata)
    assert runtime._protected(Path("/simulated"), directory=directory) is metadata


def test_absence_requires_missing_entry_and_not_observation_error(monkeypatch, tmp_path):
    path = tmp_path / "entry"
    assert runtime._absent(path) is True
    path.write_text("foreign")
    assert runtime._absent(path) is False

    def denied(_):
        raise PermissionError("PRIVATE")

    monkeypatch.setattr(Path, "lstat", denied)
    with pytest.raises(HarnessRefusal, match="PATH_OBSERVATION_UNAVAILABLE"):
        runtime._absent(path)


def test_program_allowlist_rejects_caller_controlled_name_before_observation():
    with pytest.raises(HarnessRefusal, match="HOST_PROGRAM_NOT_ALLOWLISTED"):
        runtime._program("/tmp/unapproved")


def test_program_checks_resolved_file_and_both_ancestor_chains(monkeypatch, tmp_path):
    executable = tmp_path / "program"
    executable.write_bytes(b"inert fixture")
    probes = []
    monkeypatch.setitem(runtime._PROGRAMS, "test_program", executable)
    monkeypatch.setattr(
        runtime,
        "_protected",
        lambda path, *, directory: probes.append((path, directory)),
    )
    assert runtime._program("test_program") == executable
    assert probes[0] == (executable, False)
    assert (Path("/"), True) in probes
    assert (tmp_path, True) in probes


def test_program_rejects_unowned_symlink_in_alias_chain(monkeypatch, tmp_path):
    actual = tmp_path / "actual"
    actual.mkdir()
    (actual / "program").write_bytes(b"inert fixture")
    alias = tmp_path / "alias"
    alias.symlink_to(actual, target_is_directory=True)
    monkeypatch.setitem(runtime._PROGRAMS, "test_program", alias / "program")
    monkeypatch.setattr(runtime, "_protected", lambda *_args, **_kwargs: None)
    original = Path.lstat

    def lstat(path):
        metadata = original(path)
        if path == alias:
            values = list(metadata)
            values[4] = 1000
            return os.stat_result(values)
        return metadata

    monkeypatch.setattr(Path, "lstat", lstat)
    with pytest.raises(HarnessRefusal, match="UNPROTECTED_HOST_PATH"):
        runtime._program("test_program")


def test_program_missing_is_not_treated_as_available(monkeypatch, tmp_path):
    monkeypatch.setitem(runtime._PROGRAMS, "missing", tmp_path / "missing")
    with pytest.raises(FileNotFoundError):
        runtime._program("missing")


@pytest.fixture
def command(monkeypatch):
    calls = []
    answer = SimpleNamespace(returncode=0, stdout=b"safe output\n", stderr=b"private stderr")
    monkeypatch.setattr(runtime, "_program", lambda name: Path("/trusted") / name)

    def execute(arguments, **kwargs):
        calls.append((arguments, kwargs))
        return answer

    monkeypatch.setattr(runtime.subprocess, "run", execute)
    return answer, calls


def test_command_uses_fixed_environment_timeout_and_no_shell(command):
    answer, calls = command
    assert runtime._command("systemctl", "show", "own.service") == (0, "safe output\n")
    arguments, options = calls[0]
    assert arguments == ["/trusted/systemctl", "show", "own.service"]
    assert options == {
        "stdin": subprocess.DEVNULL,
        "stdout": subprocess.PIPE,
        "stderr": subprocess.PIPE,
        "check": False,
        "timeout": 15,
        "env": runtime._ENV,
    }
    assert "private stderr" not in runtime._command("systemctl")[1]


@pytest.mark.parametrize("field", ["stdout", "stderr"])
def test_command_rejects_oversized_output(command, monkeypatch, field):
    answer, _ = command
    monkeypatch.setattr(runtime, "_MAX_OUTPUT", 4)
    answer.stdout = answer.stderr = b""
    setattr(answer, field, b"12345")
    with pytest.raises(HarnessRefusal, match="HOST_OUTPUT_LIMIT"):
        runtime._command("getent")


@pytest.mark.parametrize(
    "name,missing,code,accepted",
    [
        ("getent", True, 2, True),
        ("getent", False, 2, False),
        ("systemctl", True, 2, False),
        ("getent", True, 1, False),
    ],
)
def test_command_only_allows_expected_getent_absence(command, name, missing, code, accepted):
    answer, _ = command
    answer.returncode = code
    answer.stdout = b""
    if accepted:
        assert runtime._command(name, missing=missing) == (2, "")
    else:
        with pytest.raises(HarnessRefusal, match="HOST_COMMAND_FAILED"):
            runtime._command(name, missing=missing)


def test_command_non_ascii_observation_is_closed_refusal(command):
    answer, _ = command
    answer.stdout = b"\xffPRIVATE"
    with pytest.raises(HarnessRefusal, match="HOST_OBSERVATION_INVALID") as refused:
        runtime._command("getent")
    assert "PRIVATE" not in str(refused.value)


@pytest.mark.parametrize(
    "database,key,response,expected",
    [
        ("passwd", "owned", (0, "owned:x:62000:62000:TEST_ONLY:/nonexistent:/sbin/nologin\n"), 1),
        ("group", "owned", (0, "owned:x:62000:\n"), 1),
        ("passwd", None, (0, ""), 0),
        ("group", "missing", (2, ""), 0),
    ],
)
def test_nss_validates_exact_database_shapes(monkeypatch, database, key, response, expected):
    calls = []

    def command(*args, **kwargs):
        calls.append((args, kwargs))
        return response

    monkeypatch.setattr(runtime, "_command", command)
    assert len(runtime._nss(database, key)) == expected
    assert calls == [(("getent", database) + ((key,) if key else ()), {"missing": key is not None})]


@pytest.mark.parametrize(
    "database,response",
    [
        ("passwd", (2, "unexpected data")),
        ("passwd", (0, "owned:x:62000:62000\n")),
        ("group", (0, ":x:62000:\n")),
        ("group", (0, "owned:x:NaN:\n")),
        ("group", (0, "")),
        ("group", (0, "owned:x:62000:\nother:x:62001:\n")),
    ],
)
def test_nss_malformed_or_ambiguous_key_lookup_refused(monkeypatch, database, response):
    monkeypatch.setattr(runtime, "_command", lambda *_args, **_kwargs: response)
    with pytest.raises(HarnessRefusal, match="NSS_OBSERVATION_INVALID"):
        runtime._nss(database, "owned")


@pytest.fixture
def proc(monkeypatch, tmp_path):
    directory = tmp_path / "proc"
    directory.mkdir()
    (directory / "nonnumeric").mkdir()
    original = Path.iterdir
    monkeypatch.setattr(
        Path,
        "iterdir",
        lambda path: original(directory) if path == Path("/proc") else original(path),
    )

    def process(pid, uids="0 0 0 0", executable=None, status=None):
        item = directory / str(pid)
        item.mkdir()
        if status is None:
            status = "Name:\tinert\nUid:\t" + uids + "\n"
        (item / "status").write_text(status, encoding="ascii")
        if executable is not None:
            (item / "exe").symlink_to(executable)
        return item

    monkeypatch.setattr(runtime.os, "getpid", lambda: 999)
    process(999)
    return process


def test_process_scan_checks_all_uid_fields_and_exact_installation_prefix(proc):
    resources = plan("012345abcdef")
    root = resources["root"]
    uid = resources["roles"][0]["uid"]
    proc(10, uids=f"0 {uid} 0 0")
    proc(11, executable=root + "/root_source")
    proc(12, executable=root)
    proc(13, executable=root + "-foreign/worker")
    proc(14)  # A kernel thread has no executable.
    assert sorted(runtime._processes(resources)) == [10, 11, 12]


@pytest.mark.parametrize("uids", ["", "0 0 0", "0 0 0 bad"])
def test_process_scan_incomplete_uid_metadata_never_proves_absence(proc, uids):
    proc(10, uids=uids)
    with pytest.raises(HarnessRefusal, match="PROCESS_OBSERVATION_UNAVAILABLE"):
        runtime._processes(plan("012345abcdef"))


def test_process_scan_vanished_process_is_ignored(proc):
    item = proc(10)
    (item / "status").unlink()
    assert runtime._processes(plan("012345abcdef")) == []


def test_process_scan_without_own_pid_never_proves_complete_absence(proc, monkeypatch):
    monkeypatch.setattr(runtime.os, "getpid", lambda: 123456)
    with pytest.raises(HarnessRefusal, match="PROCESS_OBSERVATION_UNAVAILABLE"):
        runtime._processes(plan("012345abcdef"))


def test_collisions_check_numeric_nss_aliases_units_and_all_resource_paths(monkeypatch):
    resources = plan("012345abcdef")
    role = resources["roles"][0]
    probes = []
    monkeypatch.setattr(
        runtime,
        "_protected",
        lambda path, *, directory: probes.append((str(path), directory)),
    )
    monkeypatch.setattr(
        runtime,
        "_nss",
        lambda database, key: [["foreign", "x", key]]
        if (database, key) == ("passwd", str(role["uid"]))
        else [],
    )
    monkeypatch.setattr(runtime, "inspect_unit", lambda name: {"unused": name != role["unit"]})
    monkeypatch.setattr(runtime, "unit_is_unused", lambda record: record["unused"])
    monkeypatch.setattr(runtime, "_absent", lambda path: str(path) != role["unit_path"] + ".d")
    result = runtime._collisions(resources)
    assert result["nss:passwd:" + str(role["uid"])] is False
    assert result["nss:passwd:" + role["principal"]] is True
    assert result["unit:" + role["unit"]] is False
    assert result["path:" + role["unit_path"] + ".d"] is False
    assert result["path:/run/credentials/" + role["unit"]] is True
    assert result["path:/sys/fs/cgroup" + role["control_group"]] is True
    assert probes == [
        ("/sys/fs/cgroup", True),
        ("/sys/fs/cgroup/cgroup.controllers", False),
    ]


@pytest.mark.parametrize("error", [PermissionError, UnicodeError, ValueError])
def test_process_scan_unreadable_metadata_blocks_absence(proc, monkeypatch, error):
    item = proc(10)
    original = Path.read_text

    def read(path, *args, **kwargs):
        if path == item / "status":
            raise error("PRIVATE details")
        return original(path, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", read)
    with pytest.raises(HarnessRefusal, match="PROCESS_OBSERVATION_UNAVAILABLE") as refused:
        runtime._processes(plan("012345abcdef"))
    assert "PRIVATE" not in str(refused.value)


@pytest.fixture
def physical_preflight(monkeypatch):
    reads = {
        "/proc/1/comm": "systemd\n",
        "/sys/class/dmi/id/sys_vendor": "Dell Inc.\n",
        "/sys/class/dmi/id/product_name": "Latitude 5490\n",
    }
    original_read = Path.read_text
    original_stat = os.stat
    original_is_file = Path.is_file
    original_glob = Path.glob
    monkeypatch.setattr(runtime.sys, "platform", "linux")
    monkeypatch.setattr(
        runtime.platform, "freedesktop_os_release", lambda: {"ID": "ubuntu", "VERSION_ID": "24.04"}
    )
    monkeypatch.setattr(runtime.platform, "release", lambda: "TEST_ONLY-kernel")
    monkeypatch.setattr(
        Path,
        "read_text",
        lambda path, *args, **kwargs: reads[str(path)]
        if str(path) in reads
        else original_read(path, *args, **kwargs),
    )
    monkeypatch.setattr(
        Path,
        "is_file",
        lambda path: True
        if str(path) == "/sys/fs/cgroup/cgroup.controllers"
        else original_is_file(path),
    )
    monkeypatch.setattr(
        Path,
        "glob",
        lambda path, pattern: []
        if str(path) in {"/run", "/run/systemd/system"}
        else original_glob(path, pattern),
    )
    monkeypatch.setattr(
        runtime.os,
        "stat",
        lambda path, *args, **kwargs: SimpleNamespace(st_ino=1)
        if str(path).startswith("/proc/self/ns/")
        else original_stat(path, *args, **kwargs),
    )
    monkeypatch.setattr(runtime, "_program", lambda name: Path("/trusted") / name)
    monkeypatch.setattr(runtime, "_protected", lambda *_args, **_kwargs: True)
    monkeypatch.setattr(runtime, "_collisions", lambda _: {"own_resources_unused": True})
    monkeypatch.setattr(runtime, "_nss", lambda *_args: [])
    monkeypatch.setattr(runtime, "_absent", lambda _: True)
    monkeypatch.setattr(runtime, "_command", lambda *_args: (0, ""))
    monkeypatch.setattr(runtime, "_processes", lambda _: [])
    monkeypatch.setattr(
        runtime.subprocess, "run", lambda *_args, **_kwargs: SimpleNamespace(returncode=1)
    )
    return reads


def test_physical_preflight_matches_only_fully_observed_read_only_host(physical_preflight):
    report = runtime.preflight(plan("012345abcdef"))
    assert report["ready"] is True
    assert report["read_only"] is True
    assert report["blockers"] == []
    assert report["kernel"] == "TEST_ONLY-kernel"
    assert report["checks"] and all(value is True for value in report["checks"].values())


@pytest.mark.parametrize(
    "name,path,value",
    [
        ("systemd_pid1", "/proc/1/comm", "python\n"),
        ("dell_latitude_5490", "/sys/class/dmi/id/sys_vendor", "Other\n"),
        ("dell_latitude_5490", "/sys/class/dmi/id/product_name", "Latitude 5500\n"),
    ],
)
def test_preflight_requires_exact_physical_identity(physical_preflight, name, path, value):
    physical_preflight[path] = value
    report = runtime.preflight(plan("012345abcdef"))
    assert report["ready"] is False
    assert name in report["blockers"]
    assert report["checks"]["no_test_processes"] is True


@pytest.mark.parametrize(
    "probe,expected",
    [
        ("virtual", "physical_host"),
        ("ubuntu", "ubuntu_24_04"),
        ("collisions", "own_resources_unused"),
        ("collisions_unreadable", "resource_observation"),
        ("old_accounts", "no_prior_test_accounts"),
        ("old_paths", "no_prior_test_paths"),
        ("other_installation", "no_other_installations"),
        ("other_units", "no_cryptohunter_units"),
        ("old_process", "no_test_processes"),
        ("program_unavailable", "program:useradd"),
        ("namespace_unavailable", "namespace:net"),
    ],
)
def test_preflight_failure_does_not_skip_remaining_probes(
    physical_preflight, monkeypatch, probe, expected
):
    if probe == "virtual":
        monkeypatch.setattr(
            runtime.subprocess, "run", lambda *_args, **_kwargs: SimpleNamespace(returncode=0)
        )
    elif probe == "ubuntu":
        monkeypatch.setattr(runtime.platform, "freedesktop_os_release", lambda: {"ID": "debian"})
    elif probe == "collisions":
        monkeypatch.setattr(runtime, "_collisions", lambda _: {"own_resources_unused": False})
    elif probe == "collisions_unreadable":

        def unavailable(_):
            raise PermissionError("PRIVATE")

        monkeypatch.setattr(runtime, "_collisions", unavailable)
    elif probe == "old_accounts":
        monkeypatch.setattr(runtime, "_nss", lambda *_args: [["chs9t_foreign"]])
    elif probe == "old_paths":
        monkeypatch.setattr(Path, "glob", lambda *_args: [Path("/run/ch-s9-test-foreign")])
    elif probe == "other_installation":
        monkeypatch.setattr(runtime, "_absent", lambda _: False)
    elif probe == "other_units":
        monkeypatch.setattr(
            runtime, "_command", lambda *_args: (0, "cryptohunter.service enabled\n")
        )
    elif probe == "old_process":
        monkeypatch.setattr(runtime, "_processes", lambda _: [123])
    elif probe == "program_unavailable":
        original = runtime._program

        def program(name):
            if name == "useradd":
                raise FileNotFoundError("PRIVATE")
            return original(name)

        monkeypatch.setattr(runtime, "_program", program)
    elif probe == "namespace_unavailable":
        original = runtime.os.stat

        def observation(path, *args, **kwargs):
            if str(path) == "/proc/self/ns/net":
                raise PermissionError("PRIVATE")
            return original(path, *args, **kwargs)

        monkeypatch.setattr(runtime.os, "stat", observation)
    report = runtime.preflight(plan("012345abcdef"))
    assert report["ready"] is False
    assert expected in report["blockers"]
    assert "no_test_processes" in report["checks"]
    assert "PRIVATE" not in str(report)


@pytest.fixture
def capture(monkeypatch):
    resources = plan("012345abcdef")
    run = runtime.PhysicalRun(
        {"plan": resources, "commit": "a" * 40, "files": {}, "package_sha256": "b" * 64}
    )
    role = resources["roles"][0]
    run.started.add(role["role"])
    observed = {
        "Id": role["unit"],
        "FragmentPath": role["unit_path"],
        "DropInPaths": "",
        "Transient": "no",
        "Description": "Stage9 SourceCCA TEST ONLY " + resources["run_id"] + " " + role["role"],
        "InvocationID": "c" * 32,
        "MainPID": "123",
    }
    lifetime = {"pid": 123, "start_time_ticks": 456, "control_group": role["control_group"]}
    closed = []
    monkeypatch.setattr(runtime.sys, "platform", "linux")
    monkeypatch.setattr(run, "assert_owned_file", lambda _: True)
    monkeypatch.setattr(runtime, "_defer_interrupts", nullcontext)
    monkeypatch.setattr(runtime, "inspect_unit", lambda _: copy.deepcopy(observed))
    monkeypatch.setattr(runtime, "_process_lifetime", lambda _: copy.deepcopy(lifetime))
    monkeypatch.setattr(runtime, "_process_finished", lambda *_args: False)
    monkeypatch.setattr(runtime.os, "pidfd_open", lambda *_args: 71)
    monkeypatch.setattr(runtime.os, "close", closed.append)
    return SimpleNamespace(run=run, role=role, observed=observed, lifetime=lifetime, closed=closed)


def test_capture_pins_successful_lifetime_and_never_reopens_pidfd(capture, monkeypatch):
    assert capture.run._capture(capture.role) is True
    assert capture.run.handles == {capture.role["role"]: 71}
    assert capture.run.lifetimes[capture.role["role"]] == capture.lifetime
    assert not capture.closed
    monkeypatch.setattr(
        runtime.os, "pidfd_open", lambda *_args: pytest.fail("must reuse pinned fd")
    )
    assert capture.run._capture(capture.role) is True


@pytest.mark.parametrize("pid", ["", "invalid", "0", "1"])
def test_capture_requires_real_mainpid_before_pidfd(capture, monkeypatch, pid):
    capture.observed["MainPID"] = pid
    monkeypatch.setattr(runtime.os, "pidfd_open", lambda *_args: pytest.fail("no process to pin"))
    assert capture.run._capture(capture.role) is False
    assert not capture.run.handles


def test_capture_unit_identity_mismatch_prevents_process_capture(capture):
    capture.observed["DropInPaths"] = "/foreign/drop-in"
    with pytest.raises(HarnessRefusal, match="UNIT_IDENTITY_CHANGED"):
        capture.run._capture(capture.role)
    assert not capture.run.handles and not capture.closed


def test_capture_closed_pidfd_is_not_retained(capture, monkeypatch):
    monkeypatch.setattr(runtime, "_process_finished", lambda *_args: True)
    with pytest.raises(HarnessRefusal, match="PROCESS_UNAVAILABLE"):
        capture.run._capture(capture.role)
    assert not capture.run.handles
    assert capture.closed == [71]


@pytest.mark.parametrize("race", ["lifetime", "unit", "cgroup", "exited"])
def test_capture_races_close_unpublished_descriptor(capture, monkeypatch, race):
    if race == "lifetime":
        values = iter([capture.lifetime, {**capture.lifetime, "start_time_ticks": 457}])
        monkeypatch.setattr(runtime, "_process_lifetime", lambda _: next(values))
    elif race == "unit":
        values = iter([capture.observed, {**capture.observed, "InvocationID": "d" * 32}])
        monkeypatch.setattr(runtime, "inspect_unit", lambda _: next(values))
    elif race == "cgroup":
        capture.lifetime["control_group"] = "/foreign.slice/other.service"
    else:
        values = iter([False, True])
        monkeypatch.setattr(runtime, "_process_finished", lambda *_args: next(values))
    with pytest.raises(HarnessRefusal, match="PROCESS_IDENTITY_CHANGED"):
        capture.run._capture(capture.role)
    assert capture.closed == [71]
    assert not capture.run.handles


def test_capture_journal_failure_retains_cleanup_authority(capture, monkeypatch):
    def unavailable():
        raise OSError("journal unavailable")

    monkeypatch.setattr(capture.run, "_journal", unavailable)
    with pytest.raises(OSError, match="journal unavailable"):
        capture.run._capture(capture.role)
    assert capture.run.handles == {capture.role["role"]: 71}
    assert not capture.closed


def test_capture_changed_retained_lifetime_is_refused_without_dropping_pin(capture):
    assert capture.run._capture(capture.role)
    capture.lifetime["start_time_ticks"] += 1
    with pytest.raises(HarnessRefusal, match="PROCESS_IDENTITY_CHANGED"):
        capture.run._capture(capture.role)
    assert capture.run.handles == {capture.role["role"]: 71}
    assert not capture.closed


def test_capture_changed_non_identity_unit_property_refused(capture):
    assert capture.run._capture(capture.role)
    capture.observed["ControlGroup"] = "/unexpected"
    with pytest.raises(HarnessRefusal, match="UNIT_IDENTITY_CHANGED"):
        capture.run._capture(capture.role)


@pytest.fixture
def start(capture, monkeypatch):
    run, role = capture.run, capture.role
    run.started.clear()
    run.package["files"]["manifest.json"] = b"synthetic manifest fixture"
    commands = []
    monkeypatch.setattr(run, "_capture", lambda _: True)
    monkeypatch.setattr(runtime, "_command", lambda *args: commands.append(args))
    monkeypatch.setattr(
        runtime,
        "parse_manifest",
        lambda _: SimpleNamespace(roles=[SimpleNamespace(role=role["role"])]),
    )
    monkeypatch.setattr(runtime, "_unit", lambda _: (123, {}))
    monkeypatch.setattr(
        runtime,
        "_process_identity",
        lambda _: {"uid": role["uid"], "gid": role["gid"]},
    )
    monkeypatch.setattr(runtime.time, "sleep", lambda _: None)
    capture.commands = commands
    return capture


def test_start_requires_real_unit_and_matching_process_credentials(start):
    start.run.start_role(start.role)
    assert start.commands == [("systemctl", "start", start.role["unit"])]
    assert not start.run.pending
    assert start.role["role"] in start.run.started


def test_start_command_failure_still_attempts_to_capture_possible_creation(start, monkeypatch):
    captures = []

    def fail(*_args):
        raise HarnessRefusal("HOST_COMMAND_FAILED")

    monkeypatch.setattr(runtime, "_command", fail)
    monkeypatch.setattr(start.run, "_capture", lambda _: captures.append(True))
    with pytest.raises(HarnessRefusal, match="HOST_COMMAND_FAILED"):
        start.run.start_role(start.role)
    assert captures == [True]
    assert start.run.pending == ["start:" + start.role["role"]]


def test_start_capture_failure_cannot_succeed_by_unit_name_only(start, monkeypatch):
    clock = iter([0, 11])
    monkeypatch.setattr(runtime.time, "monotonic", lambda: next(clock))
    monkeypatch.setattr(start.run, "_capture", lambda _: False)
    with pytest.raises(HarnessRefusal, match="SERVICE_START_UNPROVEN"):
        start.run.start_role(start.role)
    assert start.run.pending == ["start:" + start.role["role"]]


def test_start_retries_bounded_observation_failure_then_requires_success(start, monkeypatch):
    attempts = iter([False, OSError("temporary unavailable"), True])

    def capture(_):
        value = next(attempts)
        if isinstance(value, Exception):
            raise value
        return value

    clock = iter([0, 1, 2])
    monkeypatch.setattr(runtime.time, "monotonic", lambda: next(clock))
    monkeypatch.setattr(start.run, "_capture", capture)
    start.run.start_role(start.role)
    assert not start.run.pending


def test_start_wrong_actual_credentials_remain_unproven(start, monkeypatch):
    clock = iter([0, 1, 11])
    monkeypatch.setattr(runtime.time, "monotonic", lambda: next(clock))
    monkeypatch.setattr(runtime, "_process_identity", lambda _: {"uid": 0, "gid": 0})
    with pytest.raises(HarnessRefusal, match="SERVICE_START_UNPROVEN"):
        start.run.start_role(start.role)
    assert start.run.pending


def test_start_live_previous_lifetime_cannot_be_replaced(start):
    start.run.handles[start.role["role"]] = 71
    with pytest.raises(HarnessRefusal, match="PROCESS_IDENTITY_CHANGED"):
        start.run.start_role(start.role)
    assert not start.commands


def test_start_exited_previous_lifetime_clears_only_verified_pin(start, monkeypatch):
    name = start.role["role"]
    start.run.handles[name] = 71
    start.run.lifetimes[name] = start.lifetime
    start.run.pins[name] = copy.deepcopy(start.observed)
    monkeypatch.setattr(runtime, "_process_finished", lambda *_args: True)
    start.run.start_role(start.role)
    assert start.closed == [71]
    assert name not in start.run.handles
    assert name not in start.run.lifetimes
    assert name not in start.run.pins
    assert not start.run.pending
