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


@pytest.mark.parametrize(
    "dropin",
    [
        "/etc/systemd/system/worker.service.d/override.conf",
        "[]",
        " PRIVATE_UNIT_SECRET ",
    ],
)
def test_unapproved_dropins_fail_closed(dropin):
    _, unit, account = probe._PROPOSED_SYSTEMD_UNITS[0]
    with pytest.raises(ValueError, match="UNEXPECTED_SYSTEMD_DROPINS"):
        probe._read_unit_hint(_systemd(unit, account, dropins=dropin), unit=unit, account=account)


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
    assert all(
        row["observation"] == "UNPINNED_FRAGMENT_DIGEST_ONLY"
        for row in report["fragment_observations"].values()
    )
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
    assert all(
        row["issues"] == ["SYSTEMD_UNAVAILABLE"] for row in report["fragment_observations"].values()
    )


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


_IDENTITY_REFUSAL = "FRAGMENT_CHANGED_DURING_OBSERVATION"


def _root_stat_from(actual, **changes):
    fields = {
        name: getattr(actual, name)
        for name in (
            "st_mode",
            "st_dev",
            "st_ino",
            "st_nlink",
            "st_size",
            "st_ctime_ns",
            "st_mtime_ns",
        )
    }
    return SimpleNamespace(**(fields | {"st_uid": 0, "st_gid": 0} | changes))


@pytest.fixture
def fragment_fs(tmp_path, monkeypatch):
    """Real relative syscalls in a synthetic root, never the host's systemd tree."""
    if sys.platform != "linux":
        pytest.skip("Read-only Linux descriptor fixture")
    root = tmp_path / "root"
    parent = root / "etc/systemd/system"
    parent.mkdir(parents=True)
    _, unit, _account = probe._PROPOSED_SYSTEMD_UNITS[0]
    leaf = parent / unit
    payload = b"[Unit]\nDescription=PRIVATE_UNIT_SECRET\n"
    leaf.write_bytes(payload)
    leaf.chmod(0o644)
    opened, closed, reads, calls = {}, [], [], []
    hooks = {}

    def event(operation, name, *args):
        calls.append((operation, name))
        hook = hooks.get(operation)
        if hook is not None:
            return hook(name, *args)
        return None

    def lstat(name, *, dir_fd=None):
        event("lstat", name, dir_fd)
        actual = os.lstat(root if name == "/" else name, dir_fd=dir_fd)
        result = _root_stat_from(actual)
        hook = hooks.get("lstat_result")
        return result if hook is None else hook(name, result)

    def open_file(name, flags, *, dir_fd=None):
        assert flags & os.O_NOFOLLOW
        assert flags & os.O_CLOEXEC
        assert flags & os.O_NONBLOCK
        if name != unit:
            assert flags & os.O_DIRECTORY
        if name != "/":
            assert dir_fd in opened
            assert "/" not in name
        event("open", name, dir_fd)
        fd = os.open(root if name == "/" else name, flags, dir_fd=dir_fd)
        opened[fd] = name
        hook = hooks.get("opened")
        if hook is not None:
            hook(name, fd)
        return fd

    def fstat(fd):
        name = opened[fd]
        event("fstat", name, fd)
        result = _root_stat_from(os.fstat(fd))
        hook = hooks.get("fstat_result")
        return result if hook is None else hook(name, result)

    def read(fd, count):
        name = opened[fd]
        assert name == unit
        assert 0 < count <= 8192
        event("read", name, fd)
        data = os.read(fd, count)
        reads.append((fd, count, data))
        return data

    def close(fd):
        name = opened[fd]
        os.close(fd)
        closed.append(fd)
        event("close", name, fd)

    fake_os = SimpleNamespace(
        **(
            vars(os)
            | {
                "lstat": lstat,
                "open": open_file,
                "fstat": fstat,
                "read": read,
                "close": close,
            }
        )
    )
    monkeypatch.setattr(probe, "os", fake_os)
    fixture = SimpleNamespace(
        root=root,
        parent=parent,
        leaf=leaf,
        unit=unit,
        payload=payload,
        path=f"/etc/systemd/system/{unit}",
        opened=opened,
        closed=closed,
        reads=reads,
        calls=calls,
        hooks=hooks,
    )
    yield fixture
    # Failure paths must release every descriptor, including all pinned parents.
    assert len(closed) == len(opened)
    assert set(closed) == set(opened)
    for fd in opened:
        with pytest.raises(OSError):
            os.fstat(fd)


def test_fragment_digest_uses_one_open_and_one_forward_read(fragment_fs):
    fs = fragment_fs
    result = probe._read_fragment(fs.path)
    assert result == {
        "sha256": hashlib.sha256(fs.payload).hexdigest(),
        "size": len(fs.payload),
    }
    assert list(fs.opened.values()) == ["/", "etc", "systemd", "system", fs.unit]
    assert [data for _fd, _count, data in fs.reads] == [fs.payload, b""]
    assert len({fd for fd, _count, _data in fs.reads}) == 1
    assert fs.closed == list(reversed(fs.opened))
    assert "PRIVATE_UNIT_SECRET" not in json.dumps(result)


@pytest.mark.parametrize("size", [0, 131073])
def test_fragment_refuses_empty_or_oversize_before_read(fragment_fs, size):
    fs = fragment_fs
    fs.leaf.write_bytes(b"x" * size)
    with pytest.raises(ValueError, match="FRAGMENT_TOO_LARGE_OR_EMPTY"):
        probe._read_fragment(fs.path)
    assert not fs.reads


def test_fragment_accepts_exact_bound_without_reopening(fragment_fs):
    fs = fragment_fs
    payload = b"x" * probe._MAX_FRAGMENT_BYTES
    fs.leaf.write_bytes(payload)
    assert probe._read_fragment(fs.path) == {
        "sha256": hashlib.sha256(payload).hexdigest(),
        "size": len(payload),
    }
    assert sum(len(data) for _fd, _count, data in fs.reads) == probe._MAX_FRAGMENT_BYTES
    assert fs.reads[-1][1:] == (1, b"")
    assert list(fs.opened.values()).count(fs.unit) == 1


def test_fragment_growth_cannot_read_beyond_limit(fragment_fs):
    fs = fragment_fs
    fs.leaf.write_bytes(b"x" * probe._MAX_FRAGMENT_BYTES)
    grown = False

    def grow(name, _fd):
        nonlocal grown
        if not grown:
            grown = True
            with fs.leaf.open("ab") as stream:
                stream.write(b"y" * 8192)

    fs.hooks["read"] = grow
    with pytest.raises(ValueError, match="FRAGMENT_TOO_LARGE_OR_EMPTY"):
        probe._read_fragment(fs.path)
    assert sum(len(data) for _fd, _count, data in fs.reads) == probe._MAX_FRAGMENT_BYTES + 1


def test_fragment_file_aba_replacement_is_rejected_before_any_read(fragment_fs):
    fs = fragment_fs
    saved = fs.parent / "original"
    other = fs.parent / "replacement"
    other.write_bytes(fs.payload)
    other.chmod(0o644)
    before = fs.leaf.stat()
    os.utime(other, ns=(before.st_atime_ns, before.st_mtime_ns))

    def replace_with_b(name, _parent_fd):
        if name == fs.unit:
            fs.leaf.rename(saved)
            other.rename(fs.leaf)

    def restore_a(name, fd):
        if name == fs.unit:
            # The path is A again, but the open descriptor still refers to B.
            fs.leaf.rename(other)
            saved.rename(fs.leaf)
            assert os.fstat(fd).st_ino != fs.leaf.stat().st_ino

    fs.hooks.update(open=replace_with_b, opened=restore_a)
    with pytest.raises(ValueError, match=_IDENTITY_REFUSAL):
        probe._read_fragment(fs.path)
    assert fs.leaf.stat().st_ino == before.st_ino
    assert not fs.reads


@pytest.mark.parametrize(
    "field,value",
    [
        ("st_dev", 987654321),
        ("st_ino", 987654321),
        ("st_mode", stat.S_IFREG | 0o640),
        ("st_mode", stat.S_IFDIR | 0o644),
        ("st_uid", 1),
        ("st_gid", 1),
        ("st_nlink", 2),
        ("st_size", 1),
        ("st_ctime_ns", 1),
        ("st_mtime_ns", 1),
    ],
)
@pytest.mark.parametrize("component", ["file", "directory"])
def test_lstat_fstat_metadata_mismatch_prevents_read(fragment_fs, field, value, component):
    fs = fragment_fs
    target = fs.unit if component == "file" else "system"

    def mismatch(name, result):
        if name == target:
            changed = value if getattr(result, field) != value else value + 1
            setattr(result, field, changed)
        return result

    fs.hooks["fstat_result"] = mismatch
    with pytest.raises(ValueError, match=_IDENTITY_REFUSAL):
        probe._read_fragment(fs.path)
    assert not fs.reads


@pytest.mark.parametrize("phase", ["before_open", "after_open", "during_read", "aba"])
def test_parent_directory_replacement_cannot_redirect_fragment(fragment_fs, phase):
    fs = fragment_fs
    original = fs.parent.with_name("original")
    replacement = fs.parent.with_name("replacement")
    replacement.mkdir()
    (replacement / fs.unit).write_bytes(b"PRIVATE_UNIT_SECRET_B")
    (replacement / fs.unit).chmod(0o644)
    before = fs.parent.stat()

    def swap():
        fs.parent.rename(original)
        replacement.rename(fs.parent)

    def before_open(name, _parent_fd):
        if name == "system" and phase in {"before_open", "aba"}:
            swap()

    def after_open(name, _fd):
        if name == fs.unit and phase == "after_open":
            # The whole parent chain is pinned before the namespace changes.
            swap()
        elif name == "system" and phase == "aba":
            fs.parent.rename(replacement)
            original.rename(fs.parent)

    def during_read(_name, _fd):
        if phase == "during_read" and not fs.reads:
            swap()

    fs.hooks.update(open=before_open, opened=after_open, read=during_read)
    with pytest.raises(ValueError, match=_IDENTITY_REFUSAL):
        probe._read_fragment(fs.path)
    if phase in {"after_open", "during_read"}:
        assert [data for _fd, _count, data in fs.reads] == [fs.payload, b""]
    else:
        assert not fs.reads
    if phase == "aba":
        assert fs.parent.stat().st_ino == before.st_ino


@pytest.mark.parametrize("component", ["file", "directory"])
@pytest.mark.parametrize("phase", ["before_lstat", "before_open"])
def test_fragment_refuses_symlinks_without_read(fragment_fs, component, phase):
    fs = fragment_fs
    target = fs.leaf if component == "file" else fs.parent
    name = fs.unit if component == "file" else "system"
    saved = target.with_name("symlink-target")

    def replace(candidate, *_args):
        if candidate == name:
            target.rename(saved)
            target.symlink_to(saved.name, target_is_directory=component == "directory")

    if phase == "before_lstat":
        target.rename(saved)
        target.symlink_to(saved.name, target_is_directory=component == "directory")
    else:
        fs.hooks["open"] = replace
    with pytest.raises(ValueError, match=_IDENTITY_REFUSAL):
        probe._read_fragment(fs.path)
    assert not fs.reads


@pytest.mark.parametrize(
    "field,value",
    [("st_ino", 987654321), ("st_ctime_ns", 1), ("st_mtime_ns", 1), ("st_size", 1)],
)
def test_fragment_refuses_metadata_change_during_read(fragment_fs, field, value):
    fs = fragment_fs

    def mismatch_after_read(name, result):
        if name == fs.unit and fs.reads:
            setattr(result, field, value)
        return result

    fs.hooks["fstat_result"] = mismatch_after_read
    with pytest.raises(ValueError, match=_IDENTITY_REFUSAL):
        probe._read_fragment(fs.path)
    assert fs.reads


def test_fragment_refuses_replaced_path_after_read(fragment_fs):
    fs = fragment_fs
    replacement = fs.parent / "replacement"
    replacement.write_bytes(fs.payload)

    def replace_after_first_read(name, _fd):
        if fs.reads:
            replacement.replace(fs.leaf)
            del fs.hooks["read"]

    fs.hooks["read"] = replace_after_first_read
    with pytest.raises(ValueError, match=_IDENTITY_REFUSAL):
        probe._read_fragment(fs.path)


@pytest.mark.parametrize("component", ["file", "directory"])
def test_fragment_rechecks_parent_name_bindings_after_read(fragment_fs, component):
    fs = fragment_fs
    target = fs.unit if component == "file" else "system"
    observations = 0

    def changed_binding(name, result):
        nonlocal observations
        if name == target:
            observations += 1
            if observations == 2:
                result.st_ino += 1
        return result

    fs.hooks["lstat_result"] = changed_binding
    with pytest.raises(ValueError, match=_IDENTITY_REFUSAL):
        probe._read_fragment(fs.path)
    assert observations == 2
    assert [data for _fd, _count, data in fs.reads] == [fs.payload, b""]


@pytest.mark.parametrize(
    "component,changes,code",
    [
        ("directory", {"st_uid": 1}, "FRAGMENT_PATH_METADATA_NOT_OBSERVABLE"),
        ("directory", {"st_mode": stat.S_IFDIR | 0o777}, "FRAGMENT_PATH_METADATA_NOT_OBSERVABLE"),
        ("directory", {"st_mode": stat.S_IFREG | 0o644}, "FRAGMENT_PATH_METADATA_NOT_OBSERVABLE"),
        ("file", {"st_uid": 1}, "FRAGMENT_NOT_REGULAR_OR_SAFE"),
        ("file", {"st_gid": 1}, "FRAGMENT_NOT_REGULAR_OR_SAFE"),
        ("file", {"st_nlink": 2}, "FRAGMENT_NOT_REGULAR_OR_SAFE"),
        ("file", {"st_mode": stat.S_IFREG | 0o666}, "FRAGMENT_NOT_REGULAR_OR_SAFE"),
        ("file", {"st_mode": stat.S_IFIFO | 0o644}, "FRAGMENT_NOT_REGULAR_OR_SAFE"),
    ],
)
def test_unsafe_initial_metadata_is_refused_before_open(fragment_fs, component, changes, code):
    fs = fragment_fs
    target = fs.unit if component == "file" else "system"

    def unsafe(name, result):
        if name == target:
            vars(result).update(changes)
        return result

    fs.hooks["lstat_result"] = unsafe
    with pytest.raises(ValueError, match=code):
        probe._read_fragment(fs.path)
    assert target not in fs.opened.values()
    assert not fs.reads


@pytest.mark.parametrize(
    "operation,component,occurrence",
    [
        ("lstat", "/", 1),
        ("open", "/", 1),
        ("fstat", "/", 1),
        ("lstat", "etc", 1),
        ("open", "etc", 1),
        ("fstat", "etc", 1),
        ("lstat", "file", 1),
        ("open", "file", 1),
        ("fstat", "file", 1),
        ("read", "file", 1),
        ("fstat", "file", 2),
        ("lstat", "file", 2),
        ("lstat", "etc", 2),
        ("fstat", "etc", 2),
        ("close", "file", 1),
        ("close", "system", 1),
        ("close", "/", 1),
    ],
)
def test_fragment_io_errors_use_fixed_refusal_and_close_all_fds(
    fragment_fs,
    operation,
    component,
    occurrence,
):
    fs = fragment_fs
    target = fs.unit if component == "file" else component
    count = 0

    def fail(name, *_args):
        nonlocal count
        if name == target:
            count += 1
            if count == occurrence:
                raise OSError("PRIVATE_UNIT_SECRET: filesystem I/O failed")

    fs.hooks[operation] = fail
    with pytest.raises(ValueError, match=f"^{_IDENTITY_REFUSAL}$"):
        probe._read_fragment(fs.path)
    assert count >= occurrence


def test_fragment_path_policy_rejects_unrelated_paths_before_stat(monkeypatch):
    def forbidden(*_args, **_kwargs):
        pytest.fail("Unapproved path triggered filesystem I/O")

    monkeypatch.setattr(probe, "os", SimpleNamespace(lstat=forbidden, open=forbidden))
    with pytest.raises(ValueError, match="FRAGMENT_PATH_NOT_ALLOWED"):
        probe._read_fragment("/tmp/PRIVATE_UNIT_SECRET")


@pytest.mark.parametrize(
    "changes",
    [
        {"st_uid": 1},
        {"st_gid": 1},
        {"st_nlink": 2},
        {"st_mode": stat.S_IFREG | 0o666},
        {"st_mode": stat.S_IFLNK | 0o777},
        {"st_mode": stat.S_IFIFO | 0o644},
    ],
)
def test_fragment_safe_metadata_rejects_foreign_or_writable(changes):
    fields = {"st_mode": stat.S_IFREG | 0o644, "st_uid": 0, "st_gid": 0, "st_nlink": 1}
    assert probe._safe_file_metadata(SimpleNamespace(**fields))
    assert not probe._safe_file_metadata(SimpleNamespace(**(fields | changes)))


@pytest.mark.skipif(sys.platform != "linux", reason="Synthetic Linux-only service inventory")
@pytest.mark.parametrize("success", [True, False])
def test_cli_stays_blocked_with_exit_two_for_digest_or_refusal(monkeypatch, capsys, success):
    def reader(_path):
        if success:
            return _fragment(_path)
        raise probe._Refusal(_IDENTITY_REFUSAL)

    original_inspect = probe.inspect_host
    monkeypatch.setattr(
        probe,
        "inspect_host",
        lambda: original_inspect(runner=_runner, fragment_reader=reader),
    )
    assert probe.main() == 2
    report = json.loads(capsys.readouterr().out)
    _blocked(report)
    if success:
        assert not report["observed_failures"]
    else:
        assert all(
            row["issues"] == [_IDENTITY_REFUSAL] for row in report["fragment_observations"].values()
        )
        assert "fragment_sha256" not in json.dumps(report)
