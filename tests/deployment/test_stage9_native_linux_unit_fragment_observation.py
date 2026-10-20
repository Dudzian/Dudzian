"""S9-RPI-26: only unpinned fragment digests, always production blocked."""

from __future__ import annotations

import hashlib
import json
import os
import stat
import sys
from types import SimpleNamespace

import pytest

from deployment import stage9_native_linux_unit_fragment_observation as probe


def _systemd(unit, account, *, pid=4321, fragment=None, dropins=""):
    if fragment is None:
        fragment = f"/etc/systemd/system/{unit}"
    return (
        f"Id={unit}\n"
        "LoadState=loaded\n"
        "ActiveState=active\n"
        f"User={account}\n"
        f"Group={account}\n"
        f"MainPID={pid}\n"
        f"FragmentPath={fragment}\n"
        f"DropInPaths={dropins}\n"
    )


def _runner(cmd, **kwargs):
    assert cmd[:4] == ("/usr/bin/systemctl", "--system", "show", "--no-pager")
    assert kwargs["timeout"] == 3
    assert kwargs["env"] == {"LC_ALL": "C", "PATH": "/usr/bin:/bin"}
    unit = cmd[-1]
    account = next(
        principal for _, name, principal in probe._PROPOSED_SYSTEMD_UNITS if name == unit
    )
    return SimpleNamespace(returncode=0, stdout=_systemd(unit, account))


def _fragment(_path):
    return {"sha256": "ab" * 32, "size": 1024}


def _report(*, runner=None, reader=None):
    return probe.inspect_host(
        runner=_runner if runner is None else runner,
        fragment_reader=_fragment if reader is None else reader,
    )


def _blocked(report):
    assert report["status"] == report["production_emission"] == "BLOCKED"
    assert report["qualification"] == "NOT_QUALIFIED"
    assert report["topology"] == "PROPOSED_NOT_APPROVED"
    assert "PRIVATE_UNIT_SECRET" not in json.dumps(report, allow_nan=False)


def test_fixed_systemd_fragment_paths_and_pid_hints():
    for _, unit, account in probe._PROPOSED_SYSTEMD_UNITS:
        for directory in probe._APPROVED_OBSERVATION_DIRS:
            path = f"{directory}/{unit}"
            assert probe._read_unit_hint(
                _systemd(unit, account, fragment=path), unit=unit, account=account
            ) == (4321, path)


@pytest.mark.parametrize(
    "fragment",
    [
        "",
        "/",
        "relative",
        "/tmp/custody-unit.service",
        "/etc/systemd/system/../cryptohunter-root-custody-source.service",
        "/etc/systemd/system/cryptohunter-root-custody-source.service/extra",
        "/etc/systemd/system/cryptohunter-root-custody-source.service.bak",
        "/etc/systemd/system//cryptohunter-root-custody-source.service",
        "/etc/systemd/system/cryptohunter-root-custody-source.service\x00",
    ],
)
def test_unapproved_fragment_path_fails_closed(fragment):
    _, unit, account = probe._PROPOSED_SYSTEMD_UNITS[0]
    with pytest.raises(ValueError):
        probe._read_unit_hint(
            _systemd(unit, account, fragment=fragment), unit=unit, account=account
        )


@pytest.mark.parametrize("dropin", [
    "/etc/systemd/system/worker.service.d/override.conf",
    "[]",
    " PRIVATE_UNIT_SECRET ",
])
def test_unapproved_dropins_fail_closed(dropin):
    _, unit, account = probe._PROPOSED_SYSTEMD_UNITS[0]
    with pytest.raises(ValueError, match="UNEXPECTED_SYSTEMD_DROPINS"):
        probe._read_unit_hint(
            _systemd(unit, account, dropins=dropin), unit=unit, account=account
        )


def test_duplicate_missing_extra_and_malformed_systemd_fields():
    _, unit, account = probe._PROPOSED_SYSTEMD_UNITS[0]
    source = _systemd(unit, account)
    candidates = (
        source + "DropInPaths=\n",
        source.replace("FragmentPath=/etc/systemd/system/" + unit + "\n", ""),
        source + "Unexpected=PRIVATE_UNIT_SECRET\n",
        source.replace("MainPID=4321", "MainPID=0"),
        source.replace("User=" + account, "User=root"),
        source.replace("\n", "\r\n"),
        source.replace("\n", "\v"),
        source.replace("\n", "\u2028"),
        source.removesuffix("\n"),
        source + "PRIVATE_UNIT_SECRET\n",
        source + "x" * 8193,
    )
    for candidate in candidates:
        with pytest.raises(ValueError):
            probe._read_unit_hint(candidate, unit=unit, account=account)


@pytest.mark.skipif(sys.platform != "linux", reason="Synthetic Linux-only service inventory")
def test_all_three_observations_are_unpinned_digests_never_qualified():
    seen = []

    def reader(path):
        seen.append(path)
        return {"sha256": "ab" * 32, "size": 1024}

    report = _report(reader=reader)
    _blocked(report)
    assert report["snapshot"] == "SYNTHETIC_TEST_INPUT"
    assert len(seen) == 3
    assert not report["observed_failures"]
    assert all(row["observation"] == "UNPINNED_FRAGMENT_DIGEST_ONLY"
               for row in report["fragment_observations"].values())
    assert "/etc/systemd/" not in json.dumps(report)


@pytest.mark.skipif(sys.platform != "linux", reason="Synthetic backend shape")
@pytest.mark.parametrize(
    "value",
    [
        None,
        [],
        {"sha256": "aa" * 32},
        {"sha256": "aa" * 32, "size": 1, "extra": "PRIVATE_UNIT_SECRET"},
        {"sha256": "AA" * 32, "size": 1},
        {"sha256": "a" * 63, "size": 1},
        {"sha256": "g" * 64, "size": 1},
        {"sha256": "a" * 64, "size": True},
        {"sha256": "a" * 64, "size": float("nan")},
        {"sha256": "a" * 64, "size": 0},
        {"sha256": "a" * 64, "size": 131073},
        {"sha256": "a" * 64, "size": "PRIVATE_UNIT_SECRET"},
    ],
)
def test_fragment_reader_malformed_values_are_sanitized(value):
    report = _report(reader=lambda _path: value)
    _blocked(report)
    assert all(
        row["issues"] == ["FRAGMENT_READER_INVALID"]
        for row in report["fragment_observations"].values()
    )


@pytest.mark.skipif(sys.platform != "linux", reason="Synthetic backend errors")
@pytest.mark.parametrize("backend", ["runner", "reader"])
@pytest.mark.parametrize(
    "error",
    [
        ValueError("PRIVATE_UNIT_SECRET"),
        RuntimeError("PRIVATE_UNIT_SECRET"),
        PermissionError("PRIVATE_UNIT_SECRET"),
    ],
)
def test_backend_errors_do_not_leak(backend, error):
    def broken(*_args, **_kwargs):
        raise error

    kwargs = {"runner": _runner, "reader": _fragment}
    kwargs[backend] = broken
    report = _report(**kwargs)
    _blocked(report)
    assert len(report["observed_failures"]) == 3


@pytest.mark.skipif(sys.platform != "linux", reason="Synthetic invalid systemd response")
def test_failed_systemd_does_not_read_fragment():
    def runner(*_args, **_kwargs):
        return SimpleNamespace(returncode=1, stdout="PRIVATE_UNIT_SECRET")

    def forbidden(_path):
        pytest.fail("Unexpected fragment access")

    report = _report(runner=runner, reader=forbidden)
    _blocked(report)
    assert all(row["issues"] == ["SYSTEMD_UNAVAILABLE"]
               for row in report["fragment_observations"].values())


@pytest.mark.parametrize("platform", ["win32", "darwin"])
def test_nonlinux_skips_backends_and_cli_never_returns_pass(monkeypatch, capsys, platform):
    monkeypatch.setattr(sys, "platform", platform)
    def forbidden(*_args, **_kwargs):
        pytest.fail("Non-Linux backend invoked")
    report = probe.inspect_host(runner=forbidden, fragment_reader=forbidden)
    _blocked(report)
    assert report["observed_failures"] == ["NATIVE_LINUX_REQUIRED"]
    assert probe.main() == 2
    output = json.loads(capsys.readouterr().out)
    _blocked(output)


def _root_stat_from(actual, *, uid=0, gid=0, mode=None, size=None):
    return SimpleNamespace(
        st_mode=actual.st_mode if mode is None else mode,
        st_uid=uid,
        st_gid=gid,
        st_dev=actual.st_dev,
        st_ino=actual.st_ino,
        st_nlink=actual.st_nlink,
        st_size=actual.st_size if size is None else size,
        st_ctime_ns=actual.st_ctime_ns,
        st_mtime_ns=actual.st_mtime_ns,
    )


@pytest.mark.skipif(sys.platform != "linux", reason="Read-only Linux fd fixture")
def test_fragment_digest_is_bounded_and_descriptor_is_closed(tmp_path, monkeypatch):
    path = tmp_path / "worker.service"
    payload = b"[Unit]\nDescription=PRIVATE_UNIT_SECRET\n"
    path.write_bytes(payload)
    real_fstat = os.fstat
    handles = []
    original_open = os.open
    def captured_open(name, flags):
        assert flags & os.O_NOFOLLOW
        assert flags & os.O_NONBLOCK
        descriptor = original_open(name, flags)
        handles.append(descriptor)
        return descriptor
    monkeypatch.setattr(
        probe,
        "os",
        SimpleNamespace(
            **(vars(os) | {
                "fstat": lambda fd: _root_stat_from(real_fstat(fd)),
                "open": captured_open,
            })
        ),
    )
    result = probe._hash_regular_fragment(str(path))
    assert result == {"sha256": hashlib.sha256(payload).hexdigest(), "size": len(payload)}
    with pytest.raises(OSError):
        real_fstat(handles[0])


@pytest.mark.skipif(sys.platform != "linux", reason="Read-only Linux fd fixture")
def test_fragment_refuses_oversize_and_closes_descriptor(tmp_path, monkeypatch):
    path = tmp_path / "oversize.service"
    path.write_bytes(b"x" * (probe._MAX_FRAGMENT_BYTES + 1))
    with pytest.raises(ValueError, match="FRAGMENT_"):
        probe._hash_regular_fragment(str(path))


def test_fragment_path_policy_rejects_unrelated_paths_before_stat(monkeypatch):
    def forbidden(*_args, **_kwargs):
        pytest.fail("Unapproved path triggered filesystem I/O")
    monkeypatch.setattr(probe, "os", SimpleNamespace(lstat=forbidden))
    with pytest.raises(ValueError, match="FRAGMENT_PATH_NOT_ALLOWED"):
        probe._read_fragment("/tmp/PRIVATE_UNIT_SECRET")


def test_fragment_safe_metadata_rejects_foreign_or_writable():
    fake = SimpleNamespace(
        st_mode=stat.S_IFREG | 0o644, st_uid=0, st_gid=0, st_nlink=1
    )
    assert probe._safe_file_metadata(fake)
    fake.st_uid = 1
    assert not probe._safe_file_metadata(fake)
    fake.st_uid = 0
    fake.st_mode = stat.S_IFREG | 0o666
    assert not probe._safe_file_metadata(fake)
    fake.st_mode = stat.S_IFLNK | 0o777
    assert not probe._safe_file_metadata(fake)
