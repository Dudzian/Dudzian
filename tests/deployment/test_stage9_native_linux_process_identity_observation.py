"""S9-RPI-24: process metadata diagnostics, never a qualified identity."""

from __future__ import annotations

import json
import sys
from types import SimpleNamespace

import pytest

from deployment import stage9_native_linux_process_identity_observation as probe


def _stat(pid=4321, start=12345, state="S"):
    fields = [state, *(str(i) for i in range(1, 19)), str(start)]
    return f"{pid} (worker ) name) ".encode() + " ".join(fields).encode() + b"\n"


def _status(pid=4321, uid=62001, gid=62001, *, groups="", caps="0000000000000000"):
    lines = [
        f"Name:\tworker",
        f"Pid:\t{pid}",
        f"Tgid:\t{pid}",
        f"Uid:\t{uid}\t{uid}\t{uid}\t{uid}",
        f"Gid:\t{gid}\t{gid}\t{gid}\t{gid}",
        f"Groups:\t{groups}",
        "NoNewPrivs:\t1",
    ]
    lines.extend(f"{field}:\t{caps}" for field in probe._CAP_FIELDS)
    return ("\n".join(lines) + "\n").encode()


def _unit(unit, account, pid=4321):
    return (
        f"Id={unit}\n"
        "LoadState=loaded\n"
        "ActiveState=active\n"
        f"User={account}\n"
        f"Group={account}\n"
        f"MainPID={pid}\n"
    )


def _identity(pid, uid, gid):
    return {
        "pid": pid,
        "start_time_ticks": 12345,
        "uid": uid,
        "gid": gid,
        "groups": [],
        "namespaces": {"mnt": uid + 50000, "ipc": uid + 60000, "net": uid + 70000},
    }


@pytest.mark.parametrize("raw", [b"broken", b"3 (x) Z " + b"1 " * 21])
def test_bad_proc_stat_is_not_accepted(raw):
    with pytest.raises(ValueError):
        probe._parse_stat(raw, 4321)


def test_proc_comm_with_parentheses_and_stable_starttime():
    assert probe._parse_stat(_stat(), 4321) == 12345
    with pytest.raises(ValueError, match="PID_REUSED_OR_CHANGED"):
        probe._parse_stat(_stat(), 4322)


def test_proc_status_distinct_nonprivileged_identity_and_empty_supplementary_groups():
    assert probe._parse_status(_status(), pid=4321, uid=62001, gid=62001) == {
        "uid": 62001,
        "gid": 62001,
        "groups": [],
    }


@pytest.mark.parametrize(
    ("content", "failure"),
    [
        (_status(uid=0), "PROCESS_UID_GID_NOT_MATCHED"),
        (_status(gid=0), "PROCESS_UID_GID_NOT_MATCHED"),
        (_status(groups="0 62001"), "PROCESS_GROUPS_NOT_ISOLATED"),
        (_status(groups="62002"), "PROCESS_GROUPS_NOT_ISOLATED"),
        (_status(caps="0000000000000001"), "CAPINH_NOT_EMPTY"),
        (_status().replace(b"NoNewPrivs:\t1", b"NoNewPrivs:\t0"), "NO_NEW_PRIVILEGES"),
        (_status().replace(b"Tgid:\t4321", b"Tgid:\t2"), "PROCESS_PID_NOT_MATCHED"),
        (_status().replace(b"Uid:", b"MissingUid:"), "INCOMPLETE_PROC_STATUS"),
        (_status() + b"Uid:\t62001\t62001\t62001\t62001\n", "DUPLICATE_PROC_STATUS"),
    ],
)
def test_proc_status_negative_cases(content, failure):
    with pytest.raises(ValueError, match=failure):
        probe._parse_status(content, pid=4321, uid=62001, gid=62001)


@pytest.mark.parametrize(
    ("mutation", "error"),
    [
        ("LoadState=missing\n", "LoadState_NOT_MATCHED"),
        ("User=wrong\n", "User_NOT_MATCHED"),
        ("Id=other\n", "Id_NOT_MATCHED"),
        ("MainPID=0\n", "INVALID_MAIN_PID"),
        ("Extra=value\n", "MALFORMED_SYSTEMD_RESPONSE"),
        ("MainPID=4321\n", "MALFORMED_SYSTEMD_RESPONSE"),
        ("\x00", "MALFORMED_SYSTEMD_RESPONSE"),
    ],
)
def test_systemd_pid_hint_rejects_mismatches(mutation, error):
    unit, account = probe._PROPOSED_SYSTEMD_UNITS[0][1:]
    original = _unit(unit, account)
    if mutation.startswith(("Extra=", "\x00")):
        observed = original + mutation
    elif mutation.startswith("MainPID=4321"):
        observed = original + mutation
    else:
        field = mutation.split("=", 1)[0]
        observed = "".join(
            mutation if line.startswith(field + "=") else line
            for line in original.splitlines(keepends=True)
        )
    with pytest.raises(ValueError, match=error):
        probe._parse_unit_hint(observed, unit=unit, account=account)


@pytest.mark.skipif(sys.platform != "linux", reason="Synthetic native Linux unit and proc checks")
def test_separate_three_role_observations_keep_production_blocked():
    accounts = {
        account: (62001 + i, 62001 + i, (62001 + i,))
        for i, (_role, _unit, account) in enumerate(probe._PROPOSED_SYSTEMD_UNITS)
    }
    calls = []

    def runner(command, **kwargs):
        unit = command[-1]
        role, _, account = next(
            entry for entry in probe._PROPOSED_SYSTEMD_UNITS if entry[1] == unit
        )
        calls.append((role, command, kwargs))
        return SimpleNamespace(returncode=0, stdout=_unit(unit, account, 4321 + len(calls)))

    observed = probe.inspect_host(
        runner=runner, accounts=accounts, process_reader=lambda pid, uid, gid: _identity(pid, uid, gid)
    )
    assert len(calls) == 3
    assert all(command[:3] == ("/usr/bin/systemctl", "--system", "show") for _, command, _ in calls)
    assert all(kwargs["timeout"] == 3 for _, _command, kwargs in calls)
    assert all(row["observation"] == "PROCESS_METADATA_ONLY" for row in observed["process_observations"].values())
    assert observed["observed_failures"] == []
    assert observed["snapshot"] == "SYNTHETIC_TEST_INPUT"
    assert observed["status"] == observed["production_emission"] == "BLOCKED"
    assert observed["qualification"] == "NOT_QUALIFIED"
    assert probe._BLOCKERS


@pytest.mark.skipif(sys.platform != "linux", reason="Synthetic process overlap check")
@pytest.mark.parametrize("shared", ["uid", "gid", "mnt", "ipc", "net"])
def test_shared_cross_role_identity_is_reported(shared):
    accounts = {
        account: (62001 + i, 62001 + i, (62001 + i,))
        for i, (_role, _unit, account) in enumerate(probe._PROPOSED_SYSTEMD_UNITS)
    }
    def runner(command, **_kwargs):
        unit = command[-1]
        _, _, account = next(row for row in probe._PROPOSED_SYSTEMD_UNITS if row[1] == unit)
        return SimpleNamespace(returncode=0, stdout=_unit(unit, account))

    def read(pid, uid, gid):
        result = _identity(pid, uid, gid)
        if uid == 62002:
            if shared == "uid":
                result["uid"] = 62001
            elif shared == "gid":
                result["gid"] = 62001
            else:
                result["namespaces"][shared] = 62001 + {"mnt": 50000, "ipc": 60000, "net": 70000}[shared]
        return result

    report = probe.inspect_host(runner=runner, process_reader=read, accounts=accounts)
    expected = "SHARED_UID_OR_GID" if shared in {"uid", "gid"} else "SHARED_PROCESS_NAMESPACE"
    assert f"root_source/history_source: {expected}" in report["observed_failures"]
    assert report["status"] == "BLOCKED"


@pytest.mark.skipif(sys.platform != "linux", reason="Synthetic missing identity")
def test_unavailable_or_malformed_process_is_not_normalized_to_pass():
    accounts = {
        account: (62001 + i, 62001 + i, (62001 + i,))
        for i, (_role, _unit, account) in enumerate(probe._PROPOSED_SYSTEMD_UNITS)
    }
    def unavailable(*args, **kwargs):
        raise PermissionError("private process path")

    report = probe.inspect_host(
        accounts=accounts,
        runner=lambda command, **kwargs: SimpleNamespace(
            returncode=0, stdout=_unit(command[-1], next(
                name for _role, unit, name in probe._PROPOSED_SYSTEMD_UNITS
                if unit == command[-1]
            ))
        ),
        process_reader=unavailable,
    )
    assert len(report["observed_failures"]) == 3
    assert all(
        row["issues"] == ["PROCESS_OR_SYSTEMD_UNAVAILABLE"]
        for row in report["process_observations"].values()
    )


def test_non_linux_is_explicitly_not_observed(monkeypatch):
    monkeypatch.setattr(sys, "platform", "win32")
    report = probe.inspect_host()
    assert report["observed_failures"] == ["NATIVE_LINUX_REQUIRED"]
    assert report["status"] == report["production_emission"] == "BLOCKED"
    assert report["snapshot"] == "UNTRUSTED_LIVE_OBSERVATION"


def test_cli_can_never_report_success(monkeypatch, capsys):
    monkeypatch.setattr(probe, "inspect_host", lambda: {"status": "BLOCKED"})
    assert probe.main() == 2
    assert json.loads(capsys.readouterr().out) == {"status": "BLOCKED"}
