"""S9-RPI-25: cgroup-v2 observations never establish production qualification."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from types import SimpleNamespace

import pytest

from deployment import stage9_native_linux_cgroup_membership_observation as probe


def _systemd(unit, account, *, pid=4321, group=None):
    if group is None:
        group = f"/system.slice/{unit}"
    return (
        f"Id={unit}\n"
        "LoadState=loaded\n"
        "ActiveState=active\n"
        f"User={account}\n"
        f"Group={account}\n"
        f"MainPID={pid}\n"
        f"ControlGroup={group}\n"
    )


def _runner(command, **kwargs):
    assert command[:4] == ("/usr/bin/systemctl", "--system", "show", "--no-pager")
    assert kwargs["timeout"] == 3
    assert kwargs["env"] == {"LC_ALL": "C", "PATH": "/usr/bin:/bin"}
    unit = command[-1]
    account = next(
        name for _, candidate, name in probe._PROPOSED_SYSTEMD_UNITS if candidate == unit
    )
    units = [candidate for _, candidate, _ in probe._PROPOSED_SYSTEMD_UNITS]
    return SimpleNamespace(
        returncode=0, stdout=_systemd(unit, account, pid=4321 + units.index(unit))
    )


def _report(process_reader):
    return probe.inspect_host(runner=_runner, process_reader=process_reader)


def _stat_bytes(pid, start):
    fields = ["S", *(str(i) for i in range(1, 19)), str(start)]
    return f"{pid} (worker) ".encode() + " ".join(fields).encode() + b"\n"


def _assert_blocked(report):
    assert report["status"] == report["production_emission"] == "BLOCKED"
    assert report["qualification"] == "NOT_QUALIFIED"
    assert "PRIVATE_BACKEND_MESSAGE" not in json.dumps(report, allow_nan=False)


def test_full_systemd_metadata_and_fixed_unit_are_required():
    _, unit, account = probe._PROPOSED_SYSTEMD_UNITS[0]
    assert probe._parse_unit_cgroup(_systemd(unit, account), unit=unit, account=account) == (
        4321,
        f"/system.slice/{unit}",
    )


@pytest.mark.parametrize(
    "control_group",
    [
        "/",
        "",
        "relative",
        "/system.slice/other.service",
        "/system.slice//cryptohunter-root-custody-source.service",
        "/system.slice/../cryptohunter-root-custody-source.service",
        "/system.slice/cryptohunter-root-custody-source.service/",
        "/system.slice/cryptohunter-root-custody-source.service\nInjected=true",
    ],
)
def test_invalid_systemd_control_group_rejected(control_group):
    _, unit, account = probe._PROPOSED_SYSTEMD_UNITS[0]
    raw = _systemd(unit, account, group=control_group)
    with pytest.raises(ValueError):
        probe._parse_unit_cgroup(raw, unit=unit, account=account)


def test_systemd_missing_duplicate_and_wrong_identity_rejected():
    _, unit, account = probe._PROPOSED_SYSTEMD_UNITS[0]
    raw = _systemd(unit, account)
    candidates = [
        raw.replace("MainPID=4321\n", ""),
        raw + "ControlGroup=/system.slice/x\n",
        raw.replace(f"User={account}", "User=root"),
        raw + "Extra=private\n",
        raw.replace("MainPID=4321", "MainPID=0"),
    ]
    for candidate in candidates:
        with pytest.raises(ValueError):
            probe._parse_unit_cgroup(candidate, unit=unit, account=account)


@pytest.mark.parametrize(
    "raw",
    [
        b"1:name=systemd:/system.slice/service\n",
        b"0::/system.slice/service\n1:cpu:/system.slice/service\n",
        b"0::relative\n",
        b"0::/\n",
        b"0::/system.slice//service\n",
        b"0::/system.slice/../service\n",
        b"0::/system.slice/service\r\n",
        b"0::/system.slice/service\x00\n",
        b"0::/system.slice/service",
        b"0::/system.slice/\xff\n",
        b"",
        b"x" * 8193,
    ],
)
def test_cgroup_v1_hybrid_invalid_or_oversized_data_rejected(raw):
    with pytest.raises(ValueError, match="MALFORMED_PROC_CGROUP_V2"):
        probe._parse_proc_cgroup_v2(raw)


def test_cgroup_v2_exact_and_component_boundary_children():
    group = "/system.slice/worker.service"
    assert probe._parse_proc_cgroup_v2(b"0::/system.slice/worker.service\n") == group
    assert probe._is_member(group, group)
    assert probe._is_member(group + "/worker", group)
    assert not probe._is_member(group + ".spoof", group)
    assert not probe._is_member(group + "-spoof", group)


@pytest.mark.skipif(sys.platform != "linux", reason="Synthetic Linux observations")
def test_three_roles_report_metadata_only_without_raw_paths():
    units = [unit for _, unit, _ in probe._PROPOSED_SYSTEMD_UNITS]
    accounts = {unit: name for _, unit, name in probe._PROPOSED_SYSTEMD_UNITS}

    def runner(command, **_kwargs):
        unit = command[-1]
        return SimpleNamespace(
            returncode=0,
            stdout=_systemd(unit, accounts[unit], pid=4321 + units.index(unit)),
        )

    def reader(pid):
        unit = units[pid - 4321]
        return pid, 12345 + pid, f"/system.slice/{unit}/worker"

    report = probe.inspect_host(runner=runner, process_reader=reader)
    assert not report["observed_failures"]
    assert len(report["process_observations"]) == 3
    assert all(
        row["observation"] == "CGROUP_MEMBERSHIP_METADATA_ONLY"
        for row in report["process_observations"].values()
    )
    assert report["status"] == report["production_emission"] == "BLOCKED"
    assert report["qualification"] == "NOT_QUALIFIED"
    assert report["snapshot"] == "SYNTHETIC_TEST_INPUT"
    assert "/system.slice/" not in json.dumps(report)


@pytest.mark.skipif(sys.platform != "linux", reason="Synthetic Linux refusal")
def test_prefix_spoof_and_wrong_unit_are_not_normalized_to_match():
    report = _report(
        lambda pid: (
            pid,
            12345,
            "/system.slice/cryptohunter-root-custody-source.service-spoof",
        )
    )
    assert len(report["observed_failures"]) == 3
    assert all(
        row["issues"] == ["CGROUP_MEMBERSHIP_NOT_MATCHED"]
        for row in report["process_observations"].values()
    )


@pytest.mark.skipif(sys.platform != "linux", reason="Synthetic Linux boundaries")
@pytest.mark.parametrize(
    "malformed",
    [
        None,
        [],
        [4321, 12345, "/system.slice/worker.service"],
        (4321, 12345),
        (True, 12345, "/system.slice/worker.service"),
        (4321, float("nan"), "/system.slice/worker.service"),
        (4321, 2**64, "/system.slice/worker.service"),
        (4321, 12345, "PRIVATE_KEY_SECRET"),
        (4321, 12345, "/system.slice/worker.service", "extra"),
    ],
)
def test_backend_identity_must_be_bounded_and_sanitized(malformed):
    report = _report(lambda _pid: malformed)
    assert all(
        row["issues"] == ["INVALID_PROC_CGROUP_IDENTITY"]
        for row in report["process_observations"].values()
    )
    assert "PRIVATE_KEY_SECRET" not in json.dumps(report)


@pytest.mark.skipif(sys.platform != "linux", reason="Synthetic backend refusal")
@pytest.mark.parametrize(
    "exception",
    [
        ValueError("PRIVATE_KEY_SECRET"),
        RuntimeError("PRIVATE_KEY_SECRET"),
        PermissionError("PRIVATE_KEY_SECRET"),
        subprocess.TimeoutExpired("secret", 3),
    ],
)
def test_backend_messages_never_leak(exception):
    def broken(_pid):
        raise exception

    report = _report(broken)
    assert len(report["observed_failures"]) == 3
    assert report["status"] == "BLOCKED"
    assert "PRIVATE_KEY_SECRET" not in json.dumps(report)


@pytest.mark.skipif(sys.platform != "linux", reason="Read-only Linux procfs fixture")
@pytest.mark.parametrize("drift", ["cgroup", "starttime"])
def test_procfs_double_read_refuses_changed_process_or_cgroup(monkeypatch, drift):
    pid = os.getpid()
    calls = {"stat": 0, "cgroup": 0}

    def reader(_fd, name):
        calls[name] += 1
        if name == "stat":
            ticks = 12345 + (int(drift == "starttime") if calls[name] == 2 else 0)
            return _stat_bytes(pid, ticks)
        group = "/system.slice/worker.service"
        if drift == "cgroup" and calls[name] == 2:
            group += "-changed"
        return f"0::{group}\n".encode()

    monkeypatch.setattr(probe, "_read_proc_file", reader)
    with pytest.raises(ValueError, match="PROCESS_CHANGED_DURING_OBSERVATION"):
        probe._read_process_cgroup(pid)
    assert calls == {"stat": 2, "cgroup": 2}


def test_non_linux_and_cli_remain_blocked(monkeypatch, capsys):
    monkeypatch.setattr(sys, "platform", "win32")
    assert probe.inspect_host()["observed_failures"] == ["NATIVE_LINUX_REQUIRED"]
    assert probe.main() == 2
    output = json.loads(capsys.readouterr().out)
    assert output["status"] == "BLOCKED"
    assert output["qualification"] == "NOT_QUALIFIED"
    assert output["production_emission"] == "BLOCKED"


@pytest.mark.skipif(sys.platform != "linux", reason="Synthetic cross-role identity")
def test_same_mainpid_in_two_roles_reports_cross_role_collision():
    units = [unit for _, unit, _ in probe._PROPOSED_SYSTEMD_UNITS]
    accounts = {unit: account for _, unit, account in probe._PROPOSED_SYSTEMD_UNITS}
    reads = 0

    def runner(command, **_kwargs):
        unit = command[-1]
        index = units.index(unit)
        pid = 4321 if index < 2 else 4323
        return SimpleNamespace(
            returncode=0,
            stdout=_systemd(unit, accounts[unit], pid=pid),
        )

    def reader(pid):
        nonlocal reads
        if pid == 4321:
            unit = units[reads]
            reads += 1
        else:
            unit = units[2]
        return pid, 12345, f"/system.slice/{unit}"

    report = probe.inspect_host(runner=runner, process_reader=reader)
    assert "root_source/history_source: SHARED_PROCESS_PID" in report["observed_failures"]
    assert report["status"] == report["production_emission"] == "BLOCKED"
    assert report["qualification"] == "NOT_QUALIFIED"


@pytest.mark.skipif(sys.platform != "linux", reason="Synthetic cross-role cgroup")
def test_nested_systemd_cgroups_report_cross_role_overlap():
    entries = list(probe._PROPOSED_SYSTEMD_UNITS)
    groups = {
        entries[0][1]: f"/system.slice/{entries[0][1]}",
        entries[1][1]: (f"/system.slice/{entries[0][1]}/workers/{entries[1][1]}"),
        entries[2][1]: f"/system.slice/{entries[2][1]}",
    }
    accounts = {unit: account for _, unit, account in entries}

    def runner(command, **_kwargs):
        unit = command[-1]
        pid = 4321 + list(groups).index(unit)
        return SimpleNamespace(
            returncode=0,
            stdout=_systemd(unit, accounts[unit], pid=pid, group=groups[unit]),
        )

    def reader(pid):
        unit = list(groups)[pid - 4321]
        return pid, 12345, groups[unit]

    report = probe.inspect_host(runner=runner, process_reader=reader)
    assert "root_source/history_source: CROSS_ROLE_CGROUP_OVERLAP" in report["observed_failures"]
    assert report["status"] == report["production_emission"] == "BLOCKED"


@pytest.mark.parametrize("separator", ["\r\n", "\v", "\f", "\x1c", "\x00", "\u2028"])
def test_systemd_control_separators_are_rejected(separator):
    _, unit, account = probe._PROPOSED_SYSTEMD_UNITS[0]
    raw = _systemd(unit, account).replace("\n", separator)
    with pytest.raises(ValueError, match="MALFORMED_SYSTEMD_CGROUP_METADATA"):
        probe._parse_unit_cgroup(raw, unit=unit, account=account)


@pytest.mark.parametrize("platform", ["win32", "darwin"])
def test_non_linux_never_calls_system_backends(monkeypatch, platform):
    monkeypatch.setattr(sys, "platform", platform)

    def forbidden(*_args, **_kwargs):
        pytest.fail("A non-Linux observation invoked a system backend")

    report = probe.inspect_host(runner=forbidden, process_reader=forbidden)
    _assert_blocked(report)
    assert report["observed_failures"] == ["NATIVE_LINUX_REQUIRED"]


@pytest.mark.skipif(sys.platform != "linux", reason="Synthetic malformed refusals")
@pytest.mark.parametrize("backend", ["runner", "process_reader"])
@pytest.mark.parametrize("tamper", ["missing", "nonstring", "unknown", "code-subclass", "subclass"])
def test_backend_cannot_inject_or_break_internal_refusal_codes(backend, tamper):
    class ForeignRefusal(probe._Refusal):
        def __str__(self):
            raise AssertionError("Backend exceptions must never be formatted")

    class ForeignCode(str):
        def __hash__(self):
            raise AssertionError("Backend code subclasses must never be hashed")

    error = probe._Refusal("SYSTEMD_UNAVAILABLE")
    if tamper == "missing":
        del error.code
    elif tamper == "nonstring":
        error.code = {"PRIVATE_BACKEND_MESSAGE": "secret"}
    elif tamper == "unknown":
        error.code = "PRIVATE_BACKEND_MESSAGE"
    elif tamper == "code-subclass":
        error.code = ForeignCode("SYSTEMD_UNAVAILABLE")
    else:
        error = ForeignRefusal("SYSTEMD_UNAVAILABLE")

    def broken(*_args, **_kwargs):
        raise error

    kwargs = {"runner": _runner, "process_reader": broken}
    kwargs[backend] = broken
    report = probe.inspect_host(**kwargs)
    _assert_blocked(report)
    assert all(
        row["issues"] == ["BACKEND_OBSERVATION_FAILED"]
        for row in report["process_observations"].values()
    )


@pytest.mark.skipif(sys.platform != "linux", reason="Synthetic malformed systemd responses")
@pytest.mark.parametrize(
    "response",
    [
        None,
        SimpleNamespace(stdout="PRIVATE_BACKEND_MESSAGE"),
        SimpleNamespace(returncode=None, stdout="PRIVATE_BACKEND_MESSAGE"),
        SimpleNamespace(returncode=True, stdout="PRIVATE_BACKEND_MESSAGE"),
        SimpleNamespace(returncode=0.0, stdout="PRIVATE_BACKEND_MESSAGE"),
        SimpleNamespace(returncode="0", stdout="PRIVATE_BACKEND_MESSAGE"),
        SimpleNamespace(returncode=0, stdout=None),
        SimpleNamespace(returncode=0, stdout=b"PRIVATE_BACKEND_MESSAGE"),
        SimpleNamespace(returncode=0, stdout="PRIVATE_BACKEND_MESSAGE" * 500),
    ],
)
def test_malformed_systemd_backend_responses_do_not_invoke_proc_or_leak(response):
    def runner(*_args, **_kwargs):
        return response

    def forbidden(_pid):
        pytest.fail("Malformed systemd metadata reached procfs")

    report = probe.inspect_host(runner=runner, process_reader=forbidden)
    _assert_blocked(report)
    assert len(report["observed_failures"]) == 3
    assert all(
        row["observation"] == "NOT_QUALIFIED" for row in report["process_observations"].values()
    )


@pytest.mark.skipif(sys.platform != "linux", reason="Synthetic systemd failure")
def test_systemd_failure_does_not_leak_stdout_or_stderr_or_invoke_proc():
    def runner(*_args, **_kwargs):
        return SimpleNamespace(
            returncode=1, stdout="PRIVATE_BACKEND_MESSAGE", stderr="PRIVATE_BACKEND_MESSAGE"
        )

    def forbidden(_pid):
        pytest.fail("An unsuccessful systemd response reached procfs")

    report = probe.inspect_host(runner=runner, process_reader=forbidden)
    _assert_blocked(report)
    assert all(
        row["issues"] == ["SYSTEMD_UNAVAILABLE"] for row in report["process_observations"].values()
    )


@pytest.mark.skipif(sys.platform != "linux", reason="Synthetic shared MainPID hints")
@pytest.mark.parametrize("unavailable", [False, True])
def test_shared_mainpid_is_reported_even_when_proc_membership_fails(unavailable):
    entries = list(probe._PROPOSED_SYSTEMD_UNITS)

    def runner(command, **_kwargs):
        unit = command[-1]
        index = next(i for i, row in enumerate(entries) if row[1] == unit)
        return SimpleNamespace(
            returncode=0,
            stdout=_systemd(unit, entries[index][2], pid=4321 if index < 2 else 4323),
        )

    def reader(pid):
        if unavailable:
            raise PermissionError("PRIVATE_BACKEND_MESSAGE")
        unit = entries[0 if pid == 4321 else 2][1]
        return pid, 12345, f"/system.slice/{unit}"

    report = probe.inspect_host(runner=runner, process_reader=reader)
    _assert_blocked(report)
    assert "root_source/history_source: SHARED_PROCESS_PID" in report["observed_failures"]
    expected = "PROCESS_OR_SYSTEMD_UNAVAILABLE" if unavailable else "CGROUP_MEMBERSHIP_NOT_MATCHED"
    assert report["process_observations"]["history_source"]["issues"] == [expected]


@pytest.mark.skipif(sys.platform != "linux", reason="Synthetic overlapping cgroup hints")
@pytest.mark.parametrize("failure", ["unavailable", "malformed", "mismatch"])
def test_nested_cgroups_are_reported_even_when_process_observation_fails(failure):
    entries = list(probe._PROPOSED_SYSTEMD_UNITS)
    groups = [f"/system.slice/{unit}" for _, unit, _ in entries]
    groups[1] = f"{groups[0]}/workers/{entries[1][1]}"

    def runner(command, **_kwargs):
        unit = command[-1]
        index = next(i for i, row in enumerate(entries) if row[1] == unit)
        return SimpleNamespace(
            returncode=0,
            stdout=_systemd(unit, entries[index][2], pid=4321 + index, group=groups[index]),
        )

    def reader(pid):
        if failure == "unavailable":
            raise PermissionError("PRIVATE_BACKEND_MESSAGE")
        if failure == "malformed":
            return None
        return pid, 12345, "/unrelated/PRIVATE_BACKEND_MESSAGE"

    report = probe.inspect_host(runner=runner, process_reader=reader)
    _assert_blocked(report)
    assert "root_source/history_source: CROSS_ROLE_CGROUP_OVERLAP" in report["observed_failures"]
    assert "/system.slice/" not in json.dumps(report)


@pytest.mark.skipif(sys.platform != "linux", reason="Read-only procfs descriptor fixture")
@pytest.mark.parametrize("failure", [None, "stat-parser", "cgroup-parser", "stat-io", "cgroup-io"])
def test_proc_directory_is_closed_after_success_and_failures(monkeypatch, failure):
    pid = os.getpid()
    opened = []

    def open_directory(path, flags):
        assert path == f"/proc/{pid}"
        assert flags & os.O_DIRECTORY
        assert flags & os.O_NOFOLLOW
        assert flags & os.O_CLOEXEC
        descriptor = os.open(path, flags)
        opened.append(descriptor)
        return descriptor

    def reader(descriptor, name):
        assert descriptor == opened[0]
        if failure == name + "-io":
            raise OSError("PRIVATE_BACKEND_MESSAGE")
        if failure == name + "-parser":
            return b"PRIVATE_BACKEND_MESSAGE"
        return _stat_bytes(pid, 12345) if name == "stat" else b"0::/system.slice/worker.service\n"

    monkeypatch.setattr(probe, "os", SimpleNamespace(**(vars(os) | {"open": open_directory})))
    monkeypatch.setattr(probe, "_read_proc_file", reader)
    if failure is None:
        assert probe._read_process_cgroup(pid) == (pid, 12345, "/system.slice/worker.service")
    else:
        with pytest.raises((OSError, ValueError)):
            probe._read_process_cgroup(pid)
    assert len(opened) == 1
    with pytest.raises(OSError):
        os.fstat(opened[0])


def test_cli_preserves_json_and_exit_2_when_default_backend_raises_bad_refusal(monkeypatch, capsys):
    monkeypatch.setattr(sys, "platform", "linux")
    error = probe._Refusal("SYSTEMD_UNAVAILABLE")
    del error.code

    def broken(*_args, **_kwargs):
        raise error

    monkeypatch.setattr(probe, "_run_systemctl", broken)
    assert probe.main() == 2
    report = json.loads(capsys.readouterr().out)
    _assert_blocked(report)
    assert report["snapshot"] == "UNTRUSTED_LIVE_OBSERVATION"
    assert all(
        row["issues"] == ["BACKEND_OBSERVATION_FAILED"]
        for row in report["process_observations"].values()
    )


@pytest.fixture
def tracked_children(monkeypatch):
    children = []
    real_popen = subprocess.Popen

    def popen(*args, **kwargs):
        assert kwargs["stdin"] == kwargs["stderr"] == subprocess.DEVNULL
        assert kwargs["stdout"] == subprocess.PIPE
        assert kwargs["close_fds"] is True
        child = real_popen(*args, **kwargs)
        children.append(child)
        return child

    monkeypatch.setattr(probe.subprocess, "Popen", popen)
    yield children
    assert len(children) == 1
    assert children[0].poll() is not None
    assert children[0].stdout.closed


@pytest.mark.skipif(sys.platform != "linux", reason="Synthetic Linux subprocess pipe")
@pytest.mark.parametrize("size", [0, 8192])
def test_systemd_runner_bounds_stdout_and_discards_stderr(tracked_children, size):
    command = (
        sys.executable,
        "-c",
        f"import os; os.write(2, b'PRIVATE_BACKEND_MESSAGE' * 10000); os.write(1, b'x' * {size})",
    )
    result = probe._run_systemctl(command, timeout=3)
    assert result.returncode == 0
    assert result.stdout == "x" * size
    assert result.stderr is None


@pytest.mark.skipif(sys.platform != "linux", reason="Synthetic Linux subprocess pipe")
def test_systemd_runner_refuses_oversized_stdout_and_reaps_child(tracked_children):
    command = (
        sys.executable,
        "-c",
        "import os, time; os.write(1, b'x' * 8193); time.sleep(30)",
    )
    with pytest.raises(ValueError, match="OVERSIZED_SYSTEMD_RESPONSE"):
        probe._run_systemctl(command, timeout=3)
    assert tracked_children[0].returncode != 0


@pytest.mark.skipif(sys.platform != "linux", reason="Synthetic Linux subprocess pipe")
@pytest.mark.parametrize("close_stdout", [False, True])
def test_systemd_runner_kills_reaps_and_closes_on_read_or_wait_timeout(
    tracked_children, close_stdout
):
    code = "import os, time; " + ("os.close(1); " if close_stdout else "") + "time.sleep(30)"
    command = (sys.executable, "-c", code)
    with pytest.raises(subprocess.TimeoutExpired):
        probe._run_systemctl(command, timeout=0.2)
    assert tracked_children[0].returncode != 0


@pytest.mark.skipif(sys.platform != "linux", reason="Synthetic Linux subprocess pipe")
def test_systemd_runner_closes_and_reaps_after_read_error(monkeypatch, tracked_children):
    command = (sys.executable, "-c", "import os, time; os.write(1, b'hello'); time.sleep(30)")

    def broken_read(_descriptor, _size):
        raise OSError("PRIVATE_BACKEND_MESSAGE")

    monkeypatch.setattr(probe, "os", SimpleNamespace(**(vars(os) | {"read": broken_read})))
    with pytest.raises(OSError):
        probe._run_systemctl(command, timeout=3)
    assert tracked_children[0].returncode != 0


@pytest.mark.skipif(sys.platform != "linux", reason="Synthetic Linux subprocess pipe")
def test_systemd_runner_closes_after_invalid_utf8(tracked_children):
    command = (sys.executable, "-c", "import os; os.write(1, b'PRIVATE_BACKEND_MESSAGE\\xff')")
    with pytest.raises(UnicodeDecodeError):
        probe._run_systemctl(command, timeout=3)


@pytest.mark.skipif(sys.platform != "linux", reason="Synthetic undecodable backend response")
@pytest.mark.parametrize("backend", ["runner", "process_reader"])
def test_decode_errors_emit_only_fixed_codes(backend):
    def broken(*_args, **_kwargs):
        raise UnicodeDecodeError("ascii", b"PRIVATE_BACKEND_MESSAGE\xff", 0, 1, "secret")

    kwargs = {"runner": _runner, "process_reader": broken}
    kwargs[backend] = broken
    report = probe.inspect_host(**kwargs)
    _assert_blocked(report)
    assert all(
        row["issues"] == ["MALFORMED_SYSTEMD_CGROUP_METADATA"]
        for row in report["process_observations"].values()
    )


def test_systemd_response_without_property_separator_is_rejected():
    _, unit, account = probe._PROPOSED_SYSTEMD_UNITS[0]
    raw = _systemd(unit, account) + "PRIVATE_BACKEND_MESSAGE\n"
    with pytest.raises(ValueError, match="MALFORMED_SYSTEMD_CGROUP_METADATA"):
        probe._parse_unit_cgroup(raw, unit=unit, account=account)


@pytest.mark.parametrize("platform", ["darwin", "win32"])
def test_direct_proc_reader_refuses_non_linux_before_open(monkeypatch, platform):
    monkeypatch.setattr(sys, "platform", platform)

    def forbidden(*_args, **_kwargs):
        pytest.fail("Non-Linux proc reader opened a descriptor")

    monkeypatch.setattr(probe, "os", SimpleNamespace(open=forbidden))
    with pytest.raises(ValueError, match="NATIVE_LINUX_REQUIRED"):
        probe._read_process_cgroup(4321)
