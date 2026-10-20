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
    return SimpleNamespace(returncode=0, stdout=_systemd(unit, account))


def _report(process_reader):
    return probe.inspect_host(runner=_runner, process_reader=process_reader)


def _stat_bytes(pid, start):
    fields = ["S", *(str(i) for i in range(1, 19)), str(start)]
    return f"{pid} (worker) ".encode() + " ".join(fields).encode() + b"\n"


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
        entries[1][1]: (
            f"/system.slice/{entries[0][1]}/workers/{entries[1][1]}"
        ),
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
    assert "root_source/history_source: CROSS_ROLE_CGROUP_OVERLAP" in report[
        "observed_failures"
    ]
    assert report["status"] == report["production_emission"] == "BLOCKED"
