"""S9-RPI-24: process metadata diagnostics, never a qualified identity."""

from __future__ import annotations

import json
import subprocess
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


def _accounts():
    return {
        account: (62001 + i, 62001 + i, (62001 + i,))
        for i, (_role, _unit, account) in enumerate(probe._PROPOSED_SYSTEMD_UNITS)
    }


def _runner(command, **_kwargs):
    unit = command[-1]
    _, _, account = next(row for row in probe._PROPOSED_SYSTEMD_UNITS if row[1] == unit)
    return SimpleNamespace(returncode=0, stdout=_unit(unit, account))


def _assert_refusal(report, code):
    assert report["status"] == report["production_emission"] == "BLOCKED"
    assert report["qualification"] == "NOT_QUALIFIED"
    assert report["snapshot"] == "SYNTHETIC_TEST_INPUT"
    assert report["blockers"] == list(probe._BLOCKERS)
    assert len(report["observed_failures"]) == 3
    for row in report["process_observations"].values():
        assert row["observation"] == "NOT_QUALIFIED"
        assert row["issues"] == [code]
        assert "identity" not in row
    json.dumps(report, allow_nan=False)


@pytest.mark.parametrize("raw", [b"broken", b"3 (x) Z " + b"1 " * 21])
def test_bad_proc_stat_is_not_accepted(raw):
    with pytest.raises(ValueError):
        probe._parse_stat(raw, 4321)


def test_proc_comm_with_parentheses_and_stable_starttime():
    assert probe._parse_stat(_stat(), 4321) == 12345
    assert probe._parse_stat(_stat(start=2**64 - 1), 4321) == 2**64 - 1
    with pytest.raises(ValueError, match="PID_REUSED_OR_CHANGED"):
        probe._parse_stat(_stat(), 4322)


@pytest.mark.parametrize("start", [0, -1, 2**64, "9" * 21, "+1", "1_2"])
def test_proc_stat_rejects_unbounded_or_nondecimal_start_time(start):
    with pytest.raises(ValueError, match="MALFORMED_PROC_STAT"):
        probe._parse_stat(_stat(start=start), 4321)


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
        runner=runner,
        accounts=accounts,
        process_reader=lambda pid, uid, gid: _identity(pid, uid, gid),
    )
    assert len(calls) == 3
    assert all(command[:3] == ("/usr/bin/systemctl", "--system", "show") for _, command, _ in calls)
    assert all(kwargs["timeout"] == 3 for _, _command, kwargs in calls)
    assert all(
        row["observation"] == "PROCESS_METADATA_ONLY"
        for row in observed["process_observations"].values()
    )
    assert observed["observed_failures"] == []
    assert observed["snapshot"] == "SYNTHETIC_TEST_INPUT"
    assert observed["status"] == observed["production_emission"] == "BLOCKED"
    assert observed["qualification"] == "NOT_QUALIFIED"
    assert probe._BLOCKERS


@pytest.mark.skipif(sys.platform != "linux", reason="Synthetic process overlap check")
@pytest.mark.parametrize("shared", ["uid", "gid", "mnt", "ipc", "net"])
def test_shared_cross_role_identity_is_reported(shared):
    accounts = _accounts()
    history_account = probe._PROPOSED_SYSTEMD_UNITS[1][2]
    if shared == "uid":
        accounts[history_account] = (62001, 62002, (62002,))
    elif shared == "gid":
        accounts[history_account] = (62002, 62001, (62001,))

    def runner(command, **_kwargs):
        unit = command[-1]
        _, _, account = next(row for row in probe._PROPOSED_SYSTEMD_UNITS if row[1] == unit)
        return SimpleNamespace(returncode=0, stdout=_unit(unit, account))

    def read(pid, uid, gid):
        result = _identity(pid, uid, gid)
        if uid == 62002 and shared in probe._NS_TYPES:
            result["namespaces"][shared] = (
                62001 + {"mnt": 50000, "ipc": 60000, "net": 70000}[shared]
            )
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
            returncode=0,
            stdout=_unit(
                command[-1],
                next(
                    name
                    for _role, unit, name in probe._PROPOSED_SYSTEMD_UNITS
                    if unit == command[-1]
                ),
            ),
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


@pytest.mark.skipif(sys.platform != "linux", reason="Read-only procfs self-fixture")
@pytest.mark.parametrize("drift", ["starttime", "credentials"])
def test_proc_identity_detects_non_atomic_snapshot_drift(monkeypatch, drift):
    import os

    pid = os.getpid()
    uid, gid = os.getuid(), os.getgid()
    counts = {"stat": 0, "status": 0}

    def read(_directory, name):
        counts[name] += 1
        if name == "stat":
            changed = drift == "starttime" and counts[name] == 2
            return _stat(pid, 12346 if changed else 12345)
        if drift == "credentials" and counts[name] == 2:
            return _status(pid, uid, gid, groups=str(gid))
        return _status(pid, uid, gid)

    monkeypatch.setattr(probe, "_read_proc_file", read)
    with pytest.raises(ValueError, match="PROCESS_CHANGED_DURING_OBSERVATION"):
        probe._read_proc_identity(pid, uid, gid)
    assert counts == {"stat": 2, "status": 2}


@pytest.mark.skipif(sys.platform != "linux", reason="Read-only procfs self-fixture")
def test_missing_namespace_observation_is_not_qualified(monkeypatch):
    import os

    pid = os.getpid()
    uid, gid = os.getuid(), os.getgid()

    monkeypatch.setattr(
        probe,
        "_read_proc_file",
        lambda _directory, name: _stat(pid) if name == "stat" else _status(pid, uid, gid),
    )
    monkeypatch.setattr(probe, "_NS_TYPES", ("not-a-real-namespace",))
    with pytest.raises(OSError):
        probe._read_proc_identity(pid, uid, gid)


class _UnprintableBackendError(ValueError):
    def __str__(self):
        raise AssertionError("Backend exceptions must never be formatted")


@pytest.mark.skipif(sys.platform != "linux", reason="Synthetic backend refusals")
@pytest.mark.parametrize("backend", ["runner", "process_reader"])
@pytest.mark.parametrize(
    ("error", "code"),
    [
        (ValueError("PRIVATE_BACKEND_MESSAGE"), "BACKEND_OBSERVATION_FAILED"),
        (ValueError("MALFORMED_PROC_STAT"), "BACKEND_OBSERVATION_FAILED"),
        (RuntimeError("PRIVATE_BACKEND_MESSAGE"), "BACKEND_OBSERVATION_FAILED"),
        (TypeError("PRIVATE_BACKEND_MESSAGE"), "BACKEND_OBSERVATION_FAILED"),
        (_UnprintableBackendError(), "BACKEND_OBSERVATION_FAILED"),
        (PermissionError("PRIVATE_BACKEND_MESSAGE"), "PROCESS_OR_SYSTEMD_UNAVAILABLE"),
        (TimeoutError("PRIVATE_BACKEND_MESSAGE"), "PROCESS_OR_SYSTEMD_UNAVAILABLE"),
        (
            subprocess.TimeoutExpired("PRIVATE_BACKEND_MESSAGE", 3),
            "PROCESS_OR_SYSTEMD_UNAVAILABLE",
        ),
    ],
    ids=[
        "value",
        "forged-code",
        "runtime",
        "type",
        "unprintable",
        "permission",
        "timeout",
        "subprocess-timeout",
    ],
)
def test_backend_errors_emit_only_fixed_refusal_codes(backend, error, code):
    def failing(*_args, **_kwargs):
        raise error

    kwargs = {"runner": _runner, "process_reader": _identity, "accounts": _accounts()}
    kwargs[backend] = failing
    report = probe.inspect_host(**kwargs)
    _assert_refusal(report, code)
    assert "PRIVATE_BACKEND_MESSAGE" not in json.dumps(report)


@pytest.mark.skipif(sys.platform != "linux", reason="Synthetic malformed refusal objects")
@pytest.mark.parametrize("tamper", ["missing", "nonstring", "unknown", "subclass"])
def test_backend_cannot_inject_or_break_internal_refusal_codes(tamper):
    class ForeignRefusal(probe._ObservationRefusal):
        def __str__(self):
            raise AssertionError("Backend exceptions must never be formatted")

    error = probe._ObservationRefusal("MALFORMED_PROC_STAT")
    if tamper == "missing":
        del error.code
    elif tamper == "nonstring":
        error.code = {"PRIVATE_BACKEND_MESSAGE": "secret"}
    elif tamper == "unknown":
        error.code = "PRIVATE_BACKEND_MESSAGE"
    else:
        error = ForeignRefusal("MALFORMED_PROC_STAT")

    def failing(*_args):
        raise error

    report = probe.inspect_host(runner=_runner, process_reader=failing, accounts=_accounts())
    _assert_refusal(report, "BACKEND_OBSERVATION_FAILED")
    assert "PRIVATE_BACKEND_MESSAGE" not in json.dumps(report)


@pytest.mark.skipif(sys.platform != "linux", reason="Synthetic backend refusals")
def test_systemd_decode_error_does_not_emit_undecodable_contents():
    def failing(*_args, **_kwargs):
        raise UnicodeDecodeError("ascii", b"PRIVATE_BACKEND_MESSAGE\xff", 23, 24, "secret")

    report = probe.inspect_host(runner=failing, process_reader=_identity, accounts=_accounts())
    _assert_refusal(report, "MALFORMED_SYSTEMD_RESPONSE")
    assert "PRIVATE_BACKEND_MESSAGE" not in json.dumps(report)


@pytest.mark.skipif(sys.platform != "linux", reason="Synthetic systemd response validation")
@pytest.mark.parametrize(
    ("response", "code"),
    [
        (None, "BACKEND_OBSERVATION_FAILED"),
        (SimpleNamespace(returncode=0), "BACKEND_OBSERVATION_FAILED"),
        (
            SimpleNamespace(returncode=False, stdout="PRIVATE_BACKEND_MESSAGE"),
            "MALFORMED_SYSTEMD_RESPONSE",
        ),
        (
            SimpleNamespace(returncode="0", stdout="PRIVATE_BACKEND_MESSAGE"),
            "MALFORMED_SYSTEMD_RESPONSE",
        ),
        (
            SimpleNamespace(returncode=0, stdout=b"PRIVATE_BACKEND_MESSAGE"),
            "MALFORMED_SYSTEMD_RESPONSE",
        ),
        (SimpleNamespace(returncode=1, stdout="PRIVATE_BACKEND_MESSAGE"), "SYSTEMD_UNAVAILABLE"),
    ],
)
def test_invalid_systemd_backend_responses_are_not_included_in_report(response, code):
    report = probe.inspect_host(
        runner=lambda *_args, **_kwargs: response,
        process_reader=_identity,
        accounts=_accounts(),
    )
    _assert_refusal(report, code)
    assert "PRIVATE_BACKEND_MESSAGE" not in json.dumps(report)


class _IntSubclass(int):
    pass


class _DictSubclass(dict):
    pass


class _ListSubclass(list):
    pass


@pytest.mark.skipif(sys.platform != "linux", reason="Synthetic process-reader validation")
@pytest.mark.parametrize("field", ["pid", "start_time_ticks", "uid", "gid"])
@pytest.mark.parametrize(
    "value",
    [
        None,
        True,
        1.0,
        float("nan"),
        float("inf"),
        "PRIVATE_BACKEND_MESSAGE",
        _IntSubclass(62001),
        0,
        -1,
        10**100,
    ],
    ids=[
        "none",
        "bool",
        "float",
        "nan",
        "infinity",
        "string",
        "int-subclass",
        "zero",
        "negative",
        "huge",
    ],
)
def test_process_reader_rejects_noninteger_or_unbounded_identity_fields(field, value):
    def malformed(pid, uid, gid):
        return {**_identity(pid, uid, gid), field: value}

    report = probe.inspect_host(runner=_runner, process_reader=malformed, accounts=_accounts())
    _assert_refusal(report, "INVALID_PROC_IDENTITY")
    assert "PRIVATE_BACKEND_MESSAGE" not in json.dumps(report)


@pytest.mark.skipif(sys.platform != "linux", reason="Synthetic process-reader validation")
@pytest.mark.parametrize(
    ("field", "value"),
    [("pid", 2**31), ("start_time_ticks", 2**64), ("uid", 2**32 - 1), ("gid", 2**32 - 1)],
)
def test_process_reader_rejects_values_above_kernel_metadata_bounds(field, value):
    def malformed(pid, uid, gid):
        return {**_identity(pid, uid, gid), field: value}

    _assert_refusal(
        probe.inspect_host(runner=_runner, process_reader=malformed, accounts=_accounts()),
        "INVALID_PROC_IDENTITY",
    )


@pytest.mark.skipif(sys.platform != "linux", reason="Synthetic process-reader validation")
@pytest.mark.parametrize(
    ("field", "code"),
    [
        ("pid", "PROCESS_PID_NOT_MATCHED"),
        ("uid", "PROCESS_UID_GID_NOT_MATCHED"),
        ("gid", "PROCESS_UID_GID_NOT_MATCHED"),
    ],
)
def test_process_reader_cannot_substitute_another_valid_identity(field, code):
    def mismatched(pid, uid, gid):
        identity = _identity(pid, uid, gid)
        identity[field] += 1
        return identity

    _assert_refusal(
        probe.inspect_host(runner=_runner, process_reader=mismatched, accounts=_accounts()),
        code,
    )


@pytest.mark.skipif(sys.platform != "linux", reason="Synthetic process-reader validation")
@pytest.mark.parametrize(
    "shape", ["none", "list", "subclass", "missing", "extra", "nonstring-key", "recursive"]
)
def test_process_reader_requires_a_closed_builtin_identity_record(shape):
    def malformed(pid, uid, gid):
        identity = _identity(pid, uid, gid)
        if shape == "none":
            return None
        if shape == "list":
            return list(identity.items())
        if shape == "subclass":
            return _DictSubclass(identity)
        if shape == "missing":
            del identity["namespaces"]
        elif shape == "extra":
            identity["PRIVATE_BACKEND_MESSAGE"] = "secret" * 10000
        elif shape == "nonstring-key":
            identity[1] = identity.pop("pid")
        else:
            identity["groups"] = identity
        return identity

    report = probe.inspect_host(runner=_runner, process_reader=malformed, accounts=_accounts())
    _assert_refusal(report, "INVALID_PROC_IDENTITY")
    assert "PRIVATE_BACKEND_MESSAGE" not in json.dumps(report)


@pytest.mark.skipif(sys.platform != "linux", reason="Synthetic process-reader validation")
@pytest.mark.parametrize(
    "shape", ["tuple", "set", "generator", "subclass", "many", "bool", "foreign", "recursive"]
)
def test_process_reader_requires_bounded_isolated_groups(shape):
    def malformed(pid, uid, gid):
        identity = _identity(pid, uid, gid)
        groups = {
            "tuple": (gid,),
            "set": {gid},
            "generator": iter([gid]),
            "subclass": _ListSubclass([gid]),
            "many": [gid] * 10000,
            "bool": [True],
            "foreign": [gid + 100],
            "recursive": [],
        }[shape]
        if shape == "recursive":
            groups.append(groups)
        identity["groups"] = groups
        return identity

    code = "PROCESS_GROUPS_NOT_ISOLATED" if shape == "foreign" else "INVALID_PROC_IDENTITY"
    _assert_refusal(
        probe.inspect_host(runner=_runner, process_reader=malformed, accounts=_accounts()),
        code,
    )


@pytest.mark.skipif(sys.platform != "linux", reason="Synthetic process-reader validation")
@pytest.mark.parametrize(
    "shape", ["none", "subclass", "missing", "extra", "string", "bool", "zero", "negative", "huge"]
)
def test_process_reader_requires_three_bounded_namespace_inodes(shape):
    def malformed(pid, uid, gid):
        identity = _identity(pid, uid, gid)
        namespaces = identity["namespaces"]
        if shape == "none":
            identity["namespaces"] = None
        elif shape == "subclass":
            identity["namespaces"] = _DictSubclass(namespaces)
        elif shape == "missing":
            del namespaces["net"]
        elif shape == "extra":
            namespaces["PRIVATE_BACKEND_MESSAGE"] = "secret"
        else:
            namespaces["mnt"] = {
                "string": "PRIVATE_BACKEND_MESSAGE",
                "bool": True,
                "zero": 0,
                "negative": -1,
                "huge": 2**64,
            }[shape]
        return identity

    report = probe.inspect_host(runner=_runner, process_reader=malformed, accounts=_accounts())
    _assert_refusal(report, "INVALID_PROC_IDENTITY")
    assert "PRIVATE_BACKEND_MESSAGE" not in json.dumps(report)


@pytest.mark.skipif(sys.platform != "linux", reason="Synthetic process-reader validation")
def test_validated_identity_is_a_fresh_bounded_snapshot():
    originals = []

    def reader(pid, uid, gid):
        identity = _identity(pid, uid, gid)
        identity["start_time_ticks"] = 2**64 - 1
        identity["groups"] = [gid]
        identity["namespaces"] = dict.fromkeys(probe._NS_TYPES, 2**64 - 1)
        originals.append(identity)
        return identity

    report = probe.inspect_host(runner=_runner, process_reader=reader, accounts=_accounts())
    for original, row in zip(originals, report["process_observations"].values(), strict=True):
        identity = row["identity"]
        assert identity == original
        assert identity is not original
        assert identity["groups"] is not original["groups"]
        assert identity["namespaces"] is not original["namespaces"]
        original["groups"].append("PRIVATE_BACKEND_MESSAGE")
        original["namespaces"]["mnt"] = "PRIVATE_BACKEND_MESSAGE"
        original["pid"] = "PRIVATE_BACKEND_MESSAGE"
    assert "PRIVATE_BACKEND_MESSAGE" not in json.dumps(report, allow_nan=False)
    assert report["status"] == report["production_emission"] == "BLOCKED"
    assert report["qualification"] == "NOT_QUALIFIED"


@pytest.mark.skipif(sys.platform != "linux", reason="Synthetic NSS refusals")
@pytest.mark.parametrize(
    "record",
    [
        (),
        (62001,),
        (62001, 62001),
        (62001, 62001, (), 3),
        "PRIVATE_BACKEND_MESSAGE",
        (True, 62001, (62001,)),
        (62001, 2**32 - 1, (62001,)),
    ],
)
def test_malformed_account_records_produce_fixed_refusals(record):
    accounts = dict.fromkeys((row[2] for row in probe._PROPOSED_SYSTEMD_UNITS), record)
    report = probe.inspect_host(runner=_runner, process_reader=_identity, accounts=accounts)
    _assert_refusal(report, "INVALID_NSS_IDENTITY")
    assert "PRIVATE_BACKEND_MESSAGE" not in json.dumps(report)


@pytest.mark.skipif(sys.platform != "linux", reason="Synthetic NSS refusals")
def test_nss_backend_error_does_not_leak_or_invoke_exception_string(monkeypatch):
    def failing():
        raise _UnprintableBackendError()

    monkeypatch.setattr(probe, "_native_accounts", failing)
    report = probe.inspect_host(runner=_runner, process_reader=_identity)
    _assert_refusal(report, "NSS_LOOKUP_UNAVAILABLE")


def test_cli_preserves_refusal_invariants_when_native_backend_fails(monkeypatch, capsys):
    monkeypatch.setattr(sys, "platform", "linux")

    def failing():
        raise _UnprintableBackendError()

    monkeypatch.setattr(probe, "_native_accounts", failing)
    assert probe.main() == 2
    report = json.loads(capsys.readouterr().out)
    assert report["status"] == report["production_emission"] == "BLOCKED"
    assert report["qualification"] == "NOT_QUALIFIED"
    assert report["snapshot"] == "UNTRUSTED_LIVE_OBSERVATION"
    assert all(
        row["issues"] == ["NSS_LOOKUP_UNAVAILABLE"]
        for row in report["process_observations"].values()
    )


@pytest.mark.skipif(sys.platform != "linux", reason="Read-only procfs namespace fixture")
@pytest.mark.parametrize("changed_namespace", ["mnt", "ipc", "net", None])
def test_namespace_inodes_are_rechecked_without_claiming_atomicity(monkeypatch, changed_namespace):
    import os
    import stat

    pid = os.getpid()
    uid, gid = os.getuid(), os.getgid()
    calls = dict.fromkeys(probe._NS_TYPES, 0)
    monkeypatch.setattr(
        probe,
        "_read_proc_file",
        lambda _directory, name: _stat(pid) if name == "stat" else _status(pid, uid, gid),
    )

    def namespace_stat(path, *, dir_fd):
        assert isinstance(dir_fd, int)
        name = path.removeprefix("ns/")
        calls[name] += 1
        changed = name == changed_namespace and calls[name] == 2
        return SimpleNamespace(st_mode=stat.S_IFREG, st_ino=1000 + int(changed))

    monkeypatch.setattr(probe, "os", SimpleNamespace(**(vars(os) | {"stat": namespace_stat})))
    if changed_namespace is None:
        identity = probe._read_proc_identity(pid, uid, gid)
        assert identity["namespaces"] == dict.fromkeys(probe._NS_TYPES, 1000)
    else:
        with pytest.raises(ValueError, match="PROCESS_CHANGED_DURING_OBSERVATION"):
            probe._read_proc_identity(pid, uid, gid)
    assert calls == dict.fromkeys(probe._NS_TYPES, 2)
