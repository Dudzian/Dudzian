"""Dell-like private artifacts and partial procfs observations; no privileged operations."""

from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import stat
import sys
from types import SimpleNamespace

import pytest

from deployment import stage9_stage41b_coexistence as q
from scripts import diagnose_stage9_stage41b_coexistence as cli

pytestmark = pytest.mark.skipif(sys.platform != "linux", reason="native Linux observation")

_PATH_METHODS = {name: getattr(Path, name) for name in ("stat", "iterdir", "open")}
_OS_METHODS = {name: getattr(os, name) for name in ("stat", "readlink")}


class LocalPaths:
    """Faults apply only to delegated paths from this one synthetic filesystem."""

    def __init__(self):
        self.denied_stat = set()
        self.denied_open = set()
        self.partial_directories = {}
        self.ordered_directories = {}
        self.content_reads = []

    def path(self, path):
        return LocalPath(path, self)


class LocalPath:
    def __init__(self, path, access):
        self._path = path
        self._access = access

    def __fspath__(self):
        return str(self._path)

    def __str__(self):
        return str(self._path)

    def __hash__(self):
        return hash(self._path)

    def __eq__(self, other):
        return self._path == (other._path if isinstance(other, LocalPath) else other)

    def __lt__(self, other):
        return self._path < (other._path if isinstance(other, LocalPath) else other)

    def __truediv__(self, child):
        return self._access.path(self._path / child)

    def __getattr__(self, name):
        return getattr(self._path, name)

    @property
    def parents(self):
        return tuple(self._access.path(parent) for parent in self._path.parents)

    def stat(self, *args, **kwargs):
        if self._path in self._access.denied_stat:
            raise PermissionError("PRIVATE_SOURCE_VALUE")
        return self._path.stat(*args, **kwargs)

    def open(self, *args, **kwargs):
        if self._path in self._access.denied_open:
            self._access.content_reads.append(self._path)
            raise PermissionError("PRIVATE_CONTENT_DENIED")
        return self._path.open(*args, **kwargs)

    def read_bytes(self):
        with self.open("rb") as stream:
            return stream.read()

    def read_text(self, encoding=None, errors=None):
        with self.open("r", encoding=encoding, errors=errors) as stream:
            return stream.read()

    def iterdir(self):
        if self._path in self._access.ordered_directories:
            for child in self._access.ordered_directories[self._path]:
                yield self._access.path(child)
            return
        if self._path in self._access.partial_directories:
            for child in self._access.partial_directories[self._path]:
                yield self._access.path(child)
            raise PermissionError("PRIVATE_DIRECTORY_CONTENT")
        for child in self._path.iterdir():
            yield self._access.path(child)


def assert_global_filesystem_methods_unchanged():
    for name, implementation in _PATH_METHODS.items():
        assert getattr(Path, name) is implementation
    for name, implementation in _OS_METHODS.items():
        assert getattr(os, name) is implementation


def proc_stat(pid, *, start=123, flags=0, comm="worker", cpu=0):
    # /proc/PID/stat has 52 documented fields, with comm allowed to contain ')'.
    fields = ["0"] * 50
    fields[0] = "S"
    fields[6] = str(flags)
    fields[11] = str(cpu)
    fields[12] = str(cpu)
    fields[19] = str(start)
    return f"{pid} ({comm}) ".encode() + " ".join(fields).encode() + b"\n"


def proc_status(pid):
    return (
        f"Name:	worker\nPid:	{pid}\nTgid:	{pid}\n"
        "Uid:	1000 1000 1000 1000\nGid:	1000 1000 1000 1000\nGroups:	1000\n"
    ).encode()


def snapshot(pid, *, start=123, target="/unrelated", device=0, inode=1, cpu=0):
    raw_stat = proc_stat(pid, start=start, cpu=cpu)
    return (
        (
            str(pid),
            raw_stat,
            proc_status(pid),
            b"0::/unrelated.service\n",
            b"0-10 r--p 0 00:00 0\n",
            ((target, device, inode),),
        ),
    )


@pytest.fixture
def visible_proc(monkeypatch):
    own = os.getpid()
    assert own != 1
    items = [Path("/proc/1"), Path("/proc") / str(own)]
    original_listing = q._listing
    monkeypatch.setattr(
        q,
        "_listing",
        lambda path: list(items) if path == Path("/proc") else original_listing(path),
    )
    monkeypatch.setattr(q.host, "_procfs_visibility", lambda: True)
    metadata = {"identities": [("/old", (stat.S_IFREG | 0o600, 456, 123))], "complete": True}
    monkeypatch.setattr(q, "_installation_metadata", lambda path: deepcopy(metadata))
    return SimpleNamespace(own=own, items=items, metadata=metadata)


@pytest.mark.parametrize("reference_pass", [1, 2])
def test_unreadable_pid_one_does_not_hide_later_self_reference(
    visible_proc, monkeypatch, reference_pass
):
    calls = []
    own_reads = 0

    def observe(item):
        nonlocal own_reads
        calls.append(int(item.name))
        if item.name == "1":
            raise PermissionError("PRIVATE")
        own_reads += 1
        return snapshot(
            visible_proc.own,
            target=str(q.preparation.PREFIX) + "/old (deleted)"
            if own_reads == reference_pass
            else "/unrelated",
        )

    monkeypatch.setattr(q, "_pid_snapshot", observe)
    report = q.processes()
    assert visible_proc.own in calls
    assert report["checked_pid_count"] == 1
    assert report["unavailable_pid_count"] == 1
    assert report["matched_pid_count"] == 1
    assert report["matched_pids"] == [visible_proc.own]
    assert report["state"] == "BLOCKED"
    assert report["full_confirmation"] is False
    assert report["stable_visible_snapshot"] is False
    assert "PRIVATE" not in str(report)


@pytest.mark.parametrize("failure", [PermissionError, FileNotFoundError, ValueError])
def test_single_pid_failure_does_not_abort_available_nonmatching_process(
    visible_proc, monkeypatch, failure
):
    observed = []

    def observe(item):
        observed.append(int(item.name))
        if item.name == "1":
            raise failure("PRIVATE")
        return snapshot(visible_proc.own)

    monkeypatch.setattr(q, "_pid_snapshot", observe)
    report = q.processes()
    assert visible_proc.own in observed
    assert report["checked_pid_count"] == 1
    assert report["matched_pid_count"] == 0
    if failure is FileNotFoundError:
        assert report["unstable_pid_count"] == 1
    else:
        assert report["unavailable_pid_count"] == 1
    assert report["state"] == "UNCERTAIN"
    assert report["full_confirmation"] is False
    assert not report["stable_visible_snapshot"]


@pytest.mark.parametrize("failure", [PermissionError, FileNotFoundError, ValueError])
def test_positive_first_pass_survives_failed_second_pass(visible_proc, monkeypatch, failure):
    reads = {}

    def observe(item):
        pid = int(item.name)
        reads[pid] = reads.get(pid, 0) + 1
        if pid == visible_proc.own and reads[pid] > 1:
            raise failure("PRIVATE")
        return snapshot(
            pid, target=str(q.preparation.PREFIX) + "/old" if pid == visible_proc.own else "/other"
        )

    monkeypatch.setattr(q, "_pid_snapshot", observe)
    report = q.processes()
    assert report["matched_pids"] == [visible_proc.own]
    assert report["matched_pid_count"] == 1
    assert report["state"] == "BLOCKED"
    assert not report["stable_visible_snapshot"]
    assert not report["full_confirmation"]


def test_pid_starttime_change_is_unstable_and_reference_remains_blocking(visible_proc, monkeypatch):
    own_reads = 0

    def observe(item):
        nonlocal own_reads
        pid = int(item.name)
        if pid == visible_proc.own:
            own_reads += 1
        return snapshot(
            pid, start=own_reads if pid == visible_proc.own else 123, device=123, inode=456
        )

    monkeypatch.setattr(q, "_pid_snapshot", observe)
    report = q.processes()
    assert report["unstable_pid_count"] == 1
    assert report["checked_pid_count"] == 2
    assert report["matched_pid_count"] == 2
    assert report["state"] == "BLOCKED"
    assert not report["stable_visible_snapshot"]


def test_inode_metadata_change_does_not_erase_observed_reference(visible_proc, monkeypatch):
    calls = 0

    def metadata(path):
        nonlocal calls
        calls += 1
        result = deepcopy(visible_proc.metadata)
        if calls > len(q.COMMITS):
            result["identities"][0] = ("/old", (stat.S_IFREG | 0o600, 999, 123))
        return result

    monkeypatch.setattr(q, "_installation_metadata", metadata)
    monkeypatch.setattr(
        q, "_pid_snapshot", lambda item: snapshot(int(item.name), device=123, inode=456)
    )
    report = q.processes()
    assert report["matched_pid_count"] == 2
    assert report["state"] == "BLOCKED"
    assert not report["stable_visible_snapshot"]
    assert report["full_confirmation"] is False


def test_partial_metadata_still_allows_path_matches(visible_proc, monkeypatch):
    monkeypatch.setattr(
        q, "_installation_metadata", lambda path: {"identities": [], "complete": False}
    )
    monkeypatch.setattr(
        q,
        "_pid_snapshot",
        lambda item: snapshot(int(item.name), target=str(q.preparation.PREFIX) + "/old"),
    )
    report = q.processes()
    assert report["checked_pid_count"] == 2
    assert report["matched_pid_count"] == 2
    assert report["state"] == "BLOCKED"
    assert not report["stable_visible_snapshot"]


@pytest.fixture
def dell_installations(tmp_path, monkeypatch):
    """Model root ownership/protected ancestry; fixture modes and denials are real."""
    prefix = tmp_path / "protected-opt"
    prefix.mkdir(mode=0o755)
    prefix.chmod(0o755)
    private = set()
    for commit in q.COMMITS:
        root = prefix / commit
        root.mkdir(mode=0o755)
        root.chmod(0o755)
        for name in ("installation.json", "bundle.json", "private-artifact.whl"):
            path = root / name
            path.write_bytes(b"private bytes; never read by process observer")
            path.chmod(0o600)
            private.add(path)
        public = root / "public-runtime"
        public.write_bytes(b"public bytes; do not execute")
        public.chmod(0o644)
    access = LocalPaths()
    access.denied_open.update(private)
    scoped_prefix = access.path(prefix)
    monkeypatch.setattr(q.preparation, "PREFIX", scoped_prefix)

    def protected(path, *, directory):
        info = path.lstat()
        mode = info.st_mode
        if path == prefix or prefix in path.parents:
            assert (stat.S_ISDIR if directory else stat.S_ISREG)(mode)
            assert not mode & 0o7022
        else:
            # This fixture lives below user-owned pytest temp directories.
            mode = stat.S_IFDIR | 0o755
        return SimpleNamespace(
            st_mode=mode,
            st_ino=info.st_ino,
            st_dev=info.st_dev,
            st_nlink=info.st_nlink,
            st_uid=0,
            st_gid=0,
            st_size=info.st_size,
            st_mtime_ns=info.st_mtime_ns,
            st_ctime_ns=info.st_ctime_ns,
        )

    monkeypatch.setattr(q.host, "_protected", protected)
    original_read = q._read
    content_reads = access.content_reads

    def read(path):
        if path in private:
            content_reads.append(path)
            raise PermissionError("PRIVATE_CONTENT_DENIED")
        return original_read(path)

    monkeypatch.setattr(q, "_read", read)
    return SimpleNamespace(
        prefix=scoped_prefix, private=private, content_reads=content_reads, access=access
    )


def test_metadata_collects_private_file_inodes_without_content_reads(dell_installations):
    for commit in q.COMMITS:
        root = dell_installations.prefix / commit
        report = q._installation_metadata(root)
        assert report["complete"] is True
        records = dict(report["identities"])
        for path in dell_installations.private:
            if path.parent == root:
                assert records[str(path)][1:3] == (path.stat().st_ino, path.stat().st_dev)
                assert records[str(path)][4:6] == (0, 0)
                assert stat.S_IMODE(records[str(path)][0]) == 0o600
        assert stat.S_IMODE(records[str(root)][0]) == 0o755
    assert not dell_installations.content_reads


def test_private_fixture_open_denial_does_not_patch_global_filesystem_methods(
    dell_installations, tmp_path
):
    private = next(iter(dell_installations.private))
    scoped_private = dell_installations.access.path(private)
    assert_global_filesystem_methods_unchanged()
    for read in (
        lambda: scoped_private.open("rb"),
        scoped_private.read_bytes,
        scoped_private.read_text,
    ):
        with pytest.raises(PermissionError, match="PRIVATE_CONTENT_DENIED"):
            read()
    assert dell_installations.content_reads == [private] * 3
    outside = tmp_path / "outside"
    outside.mkdir()
    public = outside / "public"
    public.write_bytes(b"outside fixture scope")
    assert public.stat().st_size == len(b"outside fixture scope")
    assert list(outside.iterdir()) == [public]
    with public.open("rb") as stream:
        assert stream.read() == b"outside fixture scope"
    assert_global_filesystem_methods_unchanged()


def test_unprivileged_cli_uses_metadata_despite_private_denial_and_pid_one_failure(
    dell_installations, tmp_path, monkeypatch, capsys
):
    assert os.geteuid() != 0
    own = os.getpid()
    proc_items = [Path("/proc/1"), Path("/proc") / str(own)]
    original_listing = q._listing
    monkeypatch.setattr(
        q, "_listing", lambda path: proc_items if path == Path("/proc") else original_listing(path)
    )
    monkeypatch.setattr(q.host, "_procfs_visibility", lambda: True)
    private = next(iter(dell_installations.private))
    identity = private.stat()

    def observe(item):
        if item.name == "1":
            raise PermissionError("PRIVATE_PID")
        # A renamed alias tests inode correlation independent of pathname/UID.
        return snapshot(own, target="/renamed-alias", device=identity.st_dev, inode=identity.st_ino)

    monkeypatch.setattr(q, "_pid_snapshot", observe)
    monkeypatch.setattr(q, "resources", lambda plan: {"state": "UNCERTAIN"})
    raw = json.dumps(
        {
            "schema": 1,
            "installations": {commit: {"files": {}, "directories": ["."]} for commit in q.COMMITS},
        }
    ).encode()
    evidence = tmp_path / "external-fixture.json"
    evidence.write_bytes(raw)
    assert (
        cli.main(
            [
                "--run-id",
                "012345abcdef",
                "--evidence",
                str(evidence),
                "--evidence-sha256",
                hashlib.sha256(raw).hexdigest(),
            ]
        )
        == 2
    )
    report = json.loads(capsys.readouterr().out)
    assert all(
        item["integrity"] == "UNCERTAIN" and item["provenance"] != "PASS"
        for item in report["installations"]
    )
    processes = report["process_observation"]
    assert processes["matched_pids"] == [own]
    assert processes["checked_pid_count"] == 1
    assert processes["unavailable_pid_count"] == 1
    assert processes["matched_pid_count"] == 1
    assert processes["state"] == "BLOCKED"
    assert not processes["full_confirmation"]
    assert report["qualification"] == "BLOCKED"
    assert report["preflight_exception"] is False
    assert report["physical_test_status"] == "PHYSICAL_TEST_PENDING"
    assert report["status"] == "IN_PROGRESS"
    assert report["production_status"] == "PRODUCTION_NO_GO"


def test_changed_process_file_inode_is_unstable_without_losing_match(visible_proc, monkeypatch):
    own_reads = 0

    def observe(item):
        nonlocal own_reads
        pid = int(item.name)
        if pid == visible_proc.own:
            own_reads += 1
            return snapshot(pid, device=123, inode=456 if own_reads == 1 else 999)
        return snapshot(pid)

    monkeypatch.setattr(q, "_pid_snapshot", observe)
    report = q.processes()
    assert report["unstable_pid_count"] == 1
    assert report["matched_pids"] == [visible_proc.own]
    assert report["state"] == "BLOCKED"
    assert not report["stable_visible_snapshot"]


def test_metadata_helper_failure_does_not_prevent_path_matching(visible_proc, monkeypatch):
    def denied(path):
        raise PermissionError("PRIVATE_METADATA")

    monkeypatch.setattr(q, "_installation_metadata", denied)
    monkeypatch.setattr(
        q,
        "_pid_snapshot",
        lambda item: snapshot(int(item.name), target=str(q.preparation.PREFIX) + "/old"),
    )
    report = q.processes()
    assert report["checked_pid_count"] == 2
    assert report["matched_pid_count"] == 2
    assert report["state"] == "BLOCKED"
    assert not report["stable_visible_snapshot"]
    assert "PRIVATE_METADATA" not in str(report)


def test_positive_path_is_not_discarded_by_unparseable_maps():
    row = list(snapshot(os.getpid(), target=str(q.preparation.PREFIX) + "/old")[0])
    row[4] = b"invalid maps observation"
    assert q._references((tuple(row),), set())


def test_metadata_denial_retains_accessible_sibling_identities(dell_installations, monkeypatch):
    root = dell_installations.prefix / q.COMMITS[0]
    original = q.host._protected
    denied = root / "bundle.json"

    def protected(path, *, directory):
        if path == denied:
            raise PermissionError("PRIVATE_METADATA")
        return original(path, directory=directory)

    monkeypatch.setattr(q.host, "_protected", protected)
    report = q._installation_metadata(root)
    assert not report["complete"]
    records = dict(report["identities"])
    assert str(denied) not in records
    assert str(root / "installation.json") in records
    assert str(root / "public-runtime") in records
    assert not dell_installations.content_reads


def test_process_scan_never_invokes_full_content_inventory(visible_proc, monkeypatch):
    monkeypatch.setattr(q, "_inventory", lambda path: pytest.fail("content inventory invoked"))
    monkeypatch.setattr(q, "_pid_snapshot", lambda item: snapshot(int(item.name)))
    report = q.processes()
    assert report["checked_pid_count"] == 2
    assert report["stable_visible_snapshot"] is True
    assert report["state"] == "UNCERTAIN"
    assert report["installation_metadata"]["integrity_verified"] is False
    assert report["installation_metadata"]["provenance_verified"] is False


@pytest.fixture
def physical_proc(tmp_path, monkeypatch):
    """Synthetic procfs files, independent read failures, no physical qualification."""
    own = os.getpid()
    proc_root = tmp_path / "proc"
    payload = tmp_path / "unrelated-payload"
    payload.write_bytes(b"never executed")
    metadata = {"identities": [], "complete": True}
    access = LocalPaths()

    def task(pid, tid=None, *, flags=0, maps=b"0-10 r--p 0 00:00 0\n", comm="worker"):
        tid = pid if tid is None else tid
        root = proc_root / str(pid)
        task_path = root / "task" / str(tid)
        task_path.mkdir(parents=True)
        directory = root if tid == pid else task_path
        (directory / "fd").mkdir()
        for name, raw in {
            "stat": proc_stat(tid, flags=flags, comm=comm),
            "status": (
                f"Name:\tworker\nState:\tS (sleeping)\nTgid:\t{pid}\nPid:\t{tid}\n"
                "Uid:\t1000 1000 1000 1000\nGid:\t1000 1000 1000 1000\n"
                "Groups:\t1000\nThreads:\t1\n"
            ).encode(),
            "cgroup": b"0::/unrelated.service\n",
            "maps": maps,
        }.items():
            (directory / name).write_bytes(raw)
        for name in ("exe", "cwd", "root", "fd/5", "fd/6"):
            (directory / name).symlink_to(payload)
        if tid == pid:
            for name in ("stat", "status", "cgroup", "maps", "exe", "cwd", "root", "fd"):
                (task_path / name).symlink_to(directory / name)
        return directory

    first = task(1)
    own_task = task(own)
    original_listing = q._listing
    monkeypatch.setattr(
        q,
        "_listing",
        lambda path: list(access.path(proc_root).iterdir())
        if path == Path("/proc")
        else original_listing(path),
    )
    monkeypatch.setattr(q.host, "_procfs_visibility", lambda: True)
    monkeypatch.setattr(q, "_installation_metadata", lambda path: deepcopy(metadata))
    return SimpleNamespace(
        own=own,
        first=first,
        task=own_task,
        root=proc_root,
        payload=payload,
        add_task=task,
        metadata=metadata,
        access=access,
    )


def deny_proc_source(monkeypatch, paths, error=PermissionError):
    original = q._proc_bytes

    def read(path, **kwargs):
        if path in paths:
            raise error("PRIVATE_SOURCE_VALUE")
        return original(path, **kwargs)

    monkeypatch.setattr(q, "_proc_bytes", read)


def assert_incomplete(report):
    assert report["full_confirmation"] is False
    assert report["stable_visible_snapshot"] is False
    assert report["unavailable_pid_count"] or report["unstable_pid_count"]
    assert "PRIVATE_SOURCE_VALUE" not in str(report)


@pytest.mark.parametrize("reference", ["fd", "cgroup"])
def test_denied_maps_does_not_suppress_independent_positive(physical_proc, monkeypatch, reference):
    deny_proc_source(monkeypatch, {physical_proc.task / "maps"})
    if reference == "cgroup":
        (physical_proc.task / "cgroup").write_bytes(b"0::/system.slice/ch-stage41b-physical\n")
    else:
        info = physical_proc.payload.stat()
        physical_proc.metadata["identities"] = [("/alias", (0, info.st_ino, info.st_dev))]
    report = q.processes()
    assert physical_proc.own in report["matched_pids"]
    assert report["state"] == "BLOCKED"
    assert_incomplete(report)


def test_denied_executable_name_keeps_independent_inode_match(physical_proc, monkeypatch):
    # Only exe has this inode; fd/cwd/root cannot provide the positive.
    executable = physical_proc.payload.with_name("renamed-executable")
    executable.write_bytes(b"never executed")
    (physical_proc.task / "exe").unlink()
    (physical_proc.task / "exe").symlink_to(executable)
    info = executable.stat()
    physical_proc.metadata["identities"] = [("/alias", (0, info.st_ino, info.st_dev))]
    original = q._proc_readlink

    def readlink(path):
        if path == physical_proc.task / "exe":
            raise PermissionError("PRIVATE_SOURCE_VALUE")
        return original(path)

    monkeypatch.setattr(q, "_proc_readlink", readlink)
    assert_global_filesystem_methods_unchanged()
    report = q.processes()
    assert report["matched_pids"] == [physical_proc.own]
    assert report["state"] == "BLOCKED"
    assert_incomplete(report)
    assert_global_filesystem_methods_unchanged()


def test_disappearing_descriptor_keeps_prior_positive(physical_proc, monkeypatch):
    first_fd = physical_proc.task / "fd/5"
    first_fd.unlink()
    first_fd.symlink_to(str(q.preparation.PREFIX) + "/removed (deleted)")
    original = q._proc_readlink

    def readlink(path):
        if path == physical_proc.task / "fd/6":
            raise FileNotFoundError("PRIVATE_SOURCE_VALUE")
        return original(path)

    monkeypatch.setattr(q, "_proc_readlink", readlink)
    report = q.processes()
    assert physical_proc.own in report["matched_pids"]
    assert report["state"] == "BLOCKED"
    assert_incomplete(report)


def test_one_unavailable_thread_does_not_hide_later_thread(physical_proc, monkeypatch):
    later = physical_proc.add_task(physical_proc.own, physical_proc.own + 1)
    (later / "cgroup").write_bytes(b"0::/cryptohunter-stage41b-fixture\n")
    deny_proc_source(monkeypatch, {physical_proc.task / "stat", physical_proc.task / "maps"})
    report = q.processes()
    assert physical_proc.own in report["matched_pids"]
    assert report["state"] == "BLOCKED"
    assert report["observed_thread_count"] >= 3
    assert_incomplete(report)


def test_malformed_maps_line_does_not_hide_later_inode(physical_proc):
    device = 123
    physical_proc.metadata["identities"] = [("/alias", (0, 456, device))]
    (physical_proc.task / "maps").write_bytes(
        b"malformed PRIVATE_MAP_PAYLOAD\n"
        + f"0-10 r--p 0 {os.major(device):x}:{os.minor(device):x} 456 /renamed\n".encode()
    )
    report = q.processes()
    assert report["matched_pids"] == [physical_proc.own]
    assert report["state"] == "BLOCKED"
    assert "PRIVATE_MAP_PAYLOAD" not in str(report)
    assert_incomplete(report)


def test_malformed_thread_stat_does_not_hide_other_sources(physical_proc):
    (physical_proc.task / "stat").write_bytes(b"PRIVATE_INVALID_STAT")
    (physical_proc.task / "cgroup").write_bytes(b"0::/ch-stage41b-fixture\n")
    report = q.processes()
    assert physical_proc.own in report["matched_pids"]
    assert report["state"] == "BLOCKED"
    assert "PRIVATE_INVALID_STAT" not in str(report)
    assert_incomplete(report)


def test_denied_every_source_never_claims_absence(physical_proc, monkeypatch):
    deny_proc_source(monkeypatch, set(physical_proc.task.glob("*")))
    original = q._proc_readlink

    def denied(path):
        if physical_proc.task in path.parents:
            raise PermissionError("PRIVATE_SOURCE_VALUE")
        return original(path)

    monkeypatch.setattr(q, "_proc_readlink", denied)
    report = q.processes()
    assert report["state"] == "UNCERTAIN"
    assert report["matched_pid_count"] == 0
    assert_incomplete(report)


def test_cpu_counters_change_without_identity_or_reference_drift(visible_proc, monkeypatch):
    reads = {}

    def observe(item):
        pid = int(item.name)
        reads[pid] = reads.get(pid, 0) + 1
        return snapshot(pid, cpu=reads[pid] * 100)

    monkeypatch.setattr(q, "_pid_snapshot", observe)
    report = q.processes()
    assert report["unstable_pid_count"] == 0
    assert report["stable_visible_snapshot"] is True
    assert report["state"] == "UNCERTAIN"
    assert report["full_confirmation"] is False


@pytest.mark.parametrize("security_field", ["Uid", "Gid", "Groups"])
def test_credentials_change_is_reference_drift(visible_proc, monkeypatch, security_field):
    reads = {}

    def observe(item):
        pid = int(item.name)
        reads[pid] = reads.get(pid, 0) + 1
        row = list(snapshot(pid)[0])
        value = str(1000 + reads[pid])
        row[2] = (
            b"\n".join(
                f"{security_field}: ".encode()
                + " ".join([value] * (1 if security_field == "Groups" else 4)).encode()
                if line.startswith(security_field.encode() + b":")
                else line
                for line in row[2].splitlines()
            )
            + b"\n"
        )
        return (tuple(row),)

    monkeypatch.setattr(q, "_pid_snapshot", observe)
    report = q.processes()
    assert report["unstable_pid_count"] == 2
    assert report["state"] == "UNCERTAIN"
    assert not report["stable_visible_snapshot"]
    assert not report["full_confirmation"]


@pytest.mark.parametrize("comm", ["worker", "kworker/0:1"])
def test_regular_empty_maps_is_incomplete(physical_proc, comm):
    (physical_proc.task / "stat").write_bytes(proc_stat(physical_proc.own, comm=comm))
    (physical_proc.task / "maps").write_bytes(b"")
    report = q.processes()
    assert report["state"] == "UNCERTAIN"
    assert report["matched_pid_count"] == 0
    assert report["kernel_thread_pid_count"] == 0
    assert report["empty_kernel_maps_pid_count"] == 0
    assert_incomplete(report)


@pytest.mark.parametrize("comm", ["kworker/u8:1", "work (queue) with spaces)"])
def test_valid_kernel_thread_empty_maps_is_counted_separately(physical_proc, comm):
    (physical_proc.task / "stat").write_bytes(
        proc_stat(physical_proc.own, flags=0x00200000, comm=comm)
    )
    (physical_proc.task / "maps").write_bytes(b"")
    report = q.processes()
    assert report["kernel_thread_pid_count"] == 1
    assert report["empty_kernel_maps_pid_count"] == 1
    assert report["matched_pid_count"] == 0
    assert report["state"] == "UNCERTAIN"
    assert report["full_confirmation"] is False


@pytest.mark.parametrize(
    "raw",
    [
        b"PRIVATE_MALFORMED_STAT",
        b"1 (kworker) S 0 0\n",
        proc_stat(1, flags=0x00200000),
        proc_stat(999999, flags=0x00200000),
        proc_stat(os.getpid()).replace(b" S ", b" ? ", 1),
        proc_stat(os.getpid(), flags=-1),
        proc_stat(os.getpid(), flags=1 << 32),
        proc_stat(os.getpid(), start=-1),
        proc_stat(os.getpid(), start=1 << 64),
        proc_stat(os.getpid(), flags=0x00200000).replace(b"2097152", b"not-a-flag", 1),
        proc_stat(os.getpid(), flags=0x00200000).rsplit(b" ", 1)[0],
    ],
)
def test_invalid_stat_never_establishes_kernel_identity(physical_proc, raw):
    # The observed PID is the current process, so even valid PID 1 data is invalid here.
    (physical_proc.task / "stat").write_bytes(raw)
    (physical_proc.task / "maps").write_bytes(b"")
    report = q.processes()
    assert report["kernel_thread_pid_count"] == 0
    assert report["empty_kernel_maps_pid_count"] == 0
    assert report["state"] == "UNCERTAIN"
    assert "PRIVATE_MALFORMED_STAT" not in str(report)
    assert_incomplete(report)


def test_kernel_flag_must_be_consistent_between_identity_reads(physical_proc, monkeypatch):
    original = q._proc_bytes
    reads = 0

    def changed(path, **kwargs):
        nonlocal reads
        if path == physical_proc.task / "stat":
            reads += 1
            return proc_stat(physical_proc.own, flags=0x00200000 if reads % 2 else 0)
        return original(path, **kwargs)

    monkeypatch.setattr(q, "_proc_bytes", changed)
    (physical_proc.task / "maps").write_bytes(b"")
    report = q.processes()
    assert report["kernel_thread_pid_count"] == 0
    assert report["empty_kernel_maps_pid_count"] == 0
    assert_incomplete(report)


def test_local_starttime_change_keeps_detected_binding(physical_proc, monkeypatch):
    original = q._proc_bytes
    reads = 0

    def changed(path, **kwargs):
        nonlocal reads
        if path == physical_proc.task / "stat":
            reads += 1
            return proc_stat(physical_proc.own, start=reads)
        return original(path, **kwargs)

    monkeypatch.setattr(q, "_proc_bytes", changed)
    (physical_proc.task / "cgroup").write_bytes(b"0::/ch-stage41b-fixture\n")
    report = q.processes()
    assert physical_proc.own in report["matched_pids"]
    assert report["state"] == "BLOCKED"
    assert report["identity_unstable_pid_count"] >= 1
    assert_incomplete(report)


def test_process_link_permissions_change_remains_uncertain(physical_proc, monkeypatch):
    original = q._pid_snapshot
    reads = 0

    def observe(item):
        nonlocal reads
        if item.name == str(physical_proc.own):
            reads += 1
            if reads == 2:
                physical_proc.payload.chmod(0o600)
        return original(item)

    monkeypatch.setattr(q, "_pid_snapshot", observe)
    report = q.processes()
    assert physical_proc.own in report["unstable_pids"]
    assert report["bindings_unstable_pid_count"] >= 1
    assert report["state"] == "UNCERTAIN"
    assert not report["stable_visible_snapshot"]
    assert report["full_confirmation"] is False


def test_denied_link_inode_keeps_positive_pathname(physical_proc, monkeypatch):
    link = physical_proc.task / "fd/5"
    link.unlink()
    link.symlink_to(str(q.preparation.PREFIX) + "/private-library (deleted)")
    physical_proc.access.denied_stat.add(link)
    report = q.processes()
    assert physical_proc.own in report["matched_pids"]
    assert report["state"] == "BLOCKED"
    assert_incomplete(report)
    assert report["reason_counts"]["FD_PERMISSION_DENIED"] > 0


def test_malformed_first_thread_maps_keeps_later_thread_descriptor(physical_proc):
    (physical_proc.task / "maps").write_bytes(b"PRIVATE_INVALID_MAPS")
    later = physical_proc.add_task(physical_proc.own, physical_proc.own + 1)
    link = later / "fd/5"
    link.unlink()
    link.symlink_to(str(q.preparation.PREFIX) + "/deleted-runtime")
    report = q.processes()
    assert physical_proc.own in report["matched_pids"]
    assert report["state"] == "BLOCKED"
    assert "PRIVATE_INVALID_MAPS" not in str(report)
    assert_incomplete(report)


def test_oversized_maps_keeps_other_source_positive(physical_proc, monkeypatch):
    monkeypatch.setattr(q, "LIMIT", 4096)
    (physical_proc.task / "maps").write_bytes(b"PRIVATE_OVERSIZED_MAPS" * 4096)
    (physical_proc.task / "cgroup").write_bytes(b"0::/ch-stage41b-fixture\n")
    report = q.processes()
    assert physical_proc.own in report["matched_pids"]
    assert report["state"] == "BLOCKED"
    assert "PRIVATE_OVERSIZED_MAPS" not in str(report)
    assert_incomplete(report)


@pytest.mark.parametrize("directory", ["fd", "task"])
def test_partial_directory_iteration_retains_available_reference(
    physical_proc, monkeypatch, directory
):
    if directory == "fd":
        observed_directory = physical_proc.task / "fd"
        link = observed_directory / "5"
        retained = [link]
    else:
        later = physical_proc.add_task(physical_proc.own, physical_proc.own + 1)
        observed_directory = physical_proc.task / "task"
        link = later / "fd/5"
        retained = [observed_directory / str(physical_proc.own), later]
    link.unlink()
    link.symlink_to(str(q.preparation.PREFIX) + "/retained-runtime")
    physical_proc.access.partial_directories[observed_directory] = retained
    report = q.processes()
    assert physical_proc.own in report["matched_pids"]
    assert report["state"] == "BLOCKED"
    assert "PRIVATE_DIRECTORY_CONTENT" not in str(report)
    assert_incomplete(report)


@pytest.mark.parametrize("operation", ["stat", "fd", "task"])
def test_active_proc_faults_do_not_patch_global_filesystem_methods(
    physical_proc, tmp_path, operation
):
    link = physical_proc.task / "fd/5"
    if operation == "stat":
        physical_proc.access.denied_stat.add(link)
    elif operation == "fd":
        physical_proc.access.partial_directories[physical_proc.task / "fd"] = [link]
    else:
        later = physical_proc.add_task(physical_proc.own, physical_proc.own + 1)
        physical_proc.access.partial_directories[physical_proc.task / "task"] = [
            physical_proc.task / "task" / str(physical_proc.own),
            later,
        ]
        link = later / "fd/5"
    link.unlink()
    link.symlink_to(str(q.preparation.PREFIX) + "/retained-runtime")
    outside = tmp_path / "outside-proc"
    outside.mkdir()
    public = outside / "public"
    public.write_bytes(b"outside proc scope")
    assert_global_filesystem_methods_unchanged()
    report = q.processes()
    assert physical_proc.own in report["matched_pids"]
    assert report["state"] == "BLOCKED"
    assert_incomplete(report)
    source = "THREAD" if operation == "task" else "FD"
    assert report["reason_counts"][source + "_PERMISSION_DENIED"] > 0
    assert public.stat().st_size == len(b"outside proc scope")
    assert list(outside.iterdir()) == [public]
    with public.open("rb") as stream:
        assert stream.read() == b"outside proc scope"
    assert_global_filesystem_methods_unchanged()


def test_proc_readlink_preserves_raw_symlink_target(tmp_path):
    raw_target = "../directory//child/./runtime/"
    link = tmp_path / "symlink"
    link.symlink_to(raw_target)
    assert q._proc_readlink(link) == raw_target
    assert q._proc_readlink(LocalPaths().path(link)) == raw_target
    assert_global_filesystem_methods_unchanged()


def test_descriptor_enumeration_limit_retains_first_positive(physical_proc, monkeypatch):
    link = physical_proc.task / "fd/5"
    link.unlink()
    link.symlink_to(str(q.preparation.PREFIX) + "/limited-runtime")
    monkeypatch.setattr(q, "FD_LIMIT", 1)
    physical_proc.access.ordered_directories[physical_proc.task / "fd"] = sorted(
        (physical_proc.task / "fd").iterdir()
    )
    report = q.processes()
    assert physical_proc.own in report["matched_pids"]
    assert report["reason_counts"]["FD_LIMIT"] > 0
    assert report["state"] == "BLOCKED"
    assert_incomplete(report)


@pytest.mark.parametrize(
    "failure", ["PERMISSION_DENIED", "NOT_FOUND", "EMPTY", "MALFORMED", "LIMIT"]
)
def test_source_failure_categories_have_bounded_redacted_counts(
    physical_proc, monkeypatch, failure
):
    monkeypatch.setattr(q, "_REASON_LIMIT", 1)
    path = physical_proc.task / "maps"
    if failure in {"PERMISSION_DENIED", "NOT_FOUND"}:
        deny_proc_source(
            monkeypatch,
            {path},
            PermissionError if failure == "PERMISSION_DENIED" else FileNotFoundError,
        )
    elif failure == "LIMIT":
        monkeypatch.setattr(q, "LIMIT", 4096)
        path.write_bytes(b"PRIVATE_LIMITED_MAPS" * 4096)
    else:
        path.write_bytes(b"" if failure == "EMPTY" else b"PRIVATE_MALFORMED_MAPS")
    report = q.processes()
    assert report["reason_counts"]["MAPS_" + failure] == 1
    assert all(0 < count <= 1 for count in report["reason_counts"].values())
    assert "PRIVATE_" not in str(report)
    assert report["state"] == "UNCERTAIN"
    assert_incomplete(report)


def test_kernel_flag_drift_between_sweeps_is_identity_change(visible_proc, monkeypatch):
    reads = {}

    def observe(item):
        pid = int(item.name)
        reads[pid] = reads.get(pid, 0) + 1
        row = list(snapshot(pid)[0])
        row[1] = proc_stat(pid, flags=0x00200000 if reads[pid] == 1 else 0)
        return (tuple(row),)

    monkeypatch.setattr(q, "_pid_snapshot", observe)
    report = q.processes()
    assert report["identity_unstable_pid_count"] == 2
    assert report["stable_process_identities"] is False
    assert report["stable_visible_snapshot"] is False
    assert report["state"] == "UNCERTAIN"
    assert report["full_confirmation"] is False


@pytest.mark.parametrize("source", ["maps", "link"])
def test_malformed_link_does_not_hide_other_available_reference(source):
    row = list(snapshot(os.getpid())[0])
    inode = (os.makedev(1, 2), 456)
    row[5] = ((None,), ("/unrelated", *inode)) if source == "link" else ((None,),)
    if source == "maps":
        row[4] = b"0-10 r--p 0 01:02 456 /renamed\n"
    assert q._references((tuple(row),), {inode})


@pytest.mark.parametrize("missing", ["Pid", "Tgid", "Uid", "Gid", "Groups"])
def test_incomplete_status_never_establishes_stability(physical_proc, missing):
    path = physical_proc.task / "status"
    path.write_bytes(
        b"\n".join(
            line
            for line in path.read_bytes().splitlines()
            if not line.startswith(missing.encode() + b":")
        )
        + b"\n"
    )
    report = q.processes()
    assert report["state"] == "UNCERTAIN"
    assert report["stable_process_identities"] is False
    assert report["stable_installation_bindings"] is False
    assert report["reason_counts"]["STATUS_MALFORMED"] > 0
    assert_incomplete(report)


def test_failed_process_enumeration_cannot_report_stability(visible_proc, monkeypatch):
    original = q._listing

    def denied(path):
        if path == Path("/proc"):
            raise PermissionError("PRIVATE_PROCESS_LIST")
        return original(path)

    monkeypatch.setattr(q, "_listing", denied)
    report = q.processes()
    assert report["checked_pid_count"] == 0
    assert report["stable_process_identities"] is False
    assert report["stable_installation_bindings"] is False
    assert report["stable_visible_snapshot"] is False
    assert report["state"] == "UNCERTAIN"
    assert report["full_confirmation"] is False
    assert "PRIVATE_PROCESS_LIST" not in str(report)


@pytest.mark.parametrize("source", ["stat", "status"])
def test_well_formed_wrong_pid_is_identity_change_and_keeps_positive(physical_proc, source):
    if source == "stat":
        (physical_proc.task / "stat").write_bytes(proc_stat(physical_proc.own + 1000))
    else:
        path = physical_proc.task / "status"
        path.write_bytes(
            path.read_bytes().replace(
                f"Pid:\t{physical_proc.own}\n".encode(),
                f"Pid:\t{physical_proc.own + 1000}\n".encode(),
            )
        )
    (physical_proc.task / "cgroup").write_bytes(b"0::/ch-stage41b-fixture\n")
    report = q.processes()
    assert physical_proc.own in report["matched_pids"]
    assert report["identity_unstable_pid_count"] == 1
    assert report["reason_counts"][source.upper() + "_IDENTITY_CHANGED"] > 0
    assert report["state"] == "BLOCKED"
    assert_incomplete(report)


@pytest.mark.parametrize("field", ["Uid", "Gid", "Groups"])
@pytest.mark.parametrize("invalid", ["4294967296", "-1", "invalid"])
def test_malformed_credential_numbers_never_establish_stability(physical_proc, field, invalid):
    path = physical_proc.task / "status"
    lines = path.read_bytes().splitlines()
    replacement = invalid if field == "Groups" else " ".join([invalid] * 4)
    path.write_bytes(
        b"\n".join(
            field.encode() + b":\t" + replacement.encode()
            if line.startswith(field.encode() + b":")
            else line
            for line in lines
        )
        + b"\n"
    )
    report = q.processes()
    assert report["reason_counts"]["STATUS_MALFORMED"] > 0
    assert report["stable_process_identities"] is False
    assert report["state"] == "UNCERTAIN"
    assert_incomplete(report)


def test_late_installation_metadata_keeps_earlier_observed_inode(visible_proc, monkeypatch):
    calls = 0

    def metadata(path):
        nonlocal calls
        calls += 1
        if calls <= len(q.COMMITS):
            return {"identities": [], "complete": False}
        return {"identities": [("/late-alias", (0, 777, 123))], "complete": True}

    monkeypatch.setattr(q, "_installation_metadata", metadata)
    monkeypatch.setattr(
        q,
        "_pid_snapshot",
        lambda item: snapshot(
            int(item.name), device=123, inode=777 if int(item.name) == visible_proc.own else 1
        ),
    )
    report = q.processes()
    assert report["matched_pids"] == [visible_proc.own]
    assert report["state"] == "BLOCKED"
    assert report["stable_visible_snapshot"] is False
    assert report["installation_metadata"]["integrity_verified"] is False
    assert report["full_confirmation"] is False


def test_limited_inode_cache_keeps_known_positive_after_limit():
    row = list(snapshot(os.getpid())[0])
    row[5] = (("/first", 123, 1), ("/second", 123, 2), ("/matched", 123, 3))
    retained = set()
    matched, issues = q._reference_observation((tuple(row),), {(123, 3)}, retained, 1)
    assert matched
    assert len(retained) == 1
    assert issues["PROCESS_INODE_LIMIT"] > 0


def test_large_pid_budget_does_not_hide_later_pid(physical_proc, monkeypatch):
    monkeypatch.setattr(q, "PROCESS_LIMIT", 8192)
    (physical_proc.first / "maps").write_bytes(b"0-10 r--p 0 00:00 0\n" * 4096)
    (physical_proc.task / "cgroup").write_bytes(b"0::/ch-stage41b-later\n")
    report = q.processes()
    assert report["checked_pid_count"] == report["observed_pid_count"] == 2
    assert report["per_pid_limit_count"] == 1
    assert report["sweep_budget_limit_count"] == report["skipped_pid_count"] == 0
    assert report["matched_pids"] == [physical_proc.own]
    assert report["state"] == "BLOCKED"
    assert report["reason_counts"]["PROCESS_PID_LIMIT"] > 0
    assert "PROCESS_SWEEP_LIMIT" not in report["reason_counts"]
    assert_incomplete(report)


@pytest.mark.parametrize("reference_pass", [1, 2])
def test_per_pid_limit_retains_positive_from_either_sweep(
    physical_proc, monkeypatch, reference_pass
):
    monkeypatch.setattr(q, "PROCESS_LIMIT", 8192)
    (physical_proc.task / "maps").write_bytes(b"0-10 r--p 0 00:00 0\n" * 4096)
    original = q._proc_bytes
    reads = 0

    def cgroup(path, **kwargs):
        nonlocal reads
        if path == physical_proc.task / "cgroup":
            reads += 1
            if reads == reference_pass:
                return b"0::/ch-stage41b-retained\n"
        return original(path, **kwargs)

    monkeypatch.setattr(q, "_proc_bytes", cgroup)
    report = q.processes()
    assert report["matched_pids"] == [physical_proc.own]
    assert report["per_pid_limit_count"] == 1
    assert report["state"] == "BLOCKED"
    assert_incomplete(report)


def test_sweep_budget_reports_every_skipped_pid(physical_proc, monkeypatch):
    monkeypatch.setattr(q, "SWEEP_LIMIT", 1)
    (physical_proc.task / "cgroup").write_bytes(b"0::/ch-stage41b-unobserved\n")
    report = q.processes()
    assert report["sweep_budget_limit_count"] == report["skipped_pid_count"] == 2
    assert report["per_pid_limit_count"] == 0
    assert report["observed_pid_count"] == report["checked_pid_count"] == 0
    assert set(report["unavailable_pids"]) == {1, physical_proc.own}
    assert report["reason_counts"]["PROCESS_SWEEP_LIMIT"] == 4
    assert report["state"] == "UNCERTAIN"
    assert report["stable_process_identities"] is False
    assert report["stable_installation_bindings"] is False
    assert_incomplete(report)


@pytest.mark.parametrize("reference_pass", [1, 2])
def test_sweep_crossing_keeps_positive_before_discarding_fingerprint(
    physical_proc, monkeypatch, reference_pass
):
    monkeypatch.setattr(q, "SWEEP_LIMIT", q._PROCESS_RESERVATION)
    physical_proc.access.ordered_directories[physical_proc.root] = sorted(
        physical_proc.root.iterdir()
    )
    original = q._proc_bytes
    reads = 0

    def cgroup(path, **kwargs):
        nonlocal reads
        if path == physical_proc.first / "cgroup":
            reads += 1
            if reads == reference_pass:
                return b"0::/ch-stage41b-retained\n"
        return original(path, **kwargs)

    monkeypatch.setattr(q, "_proc_bytes", cgroup)
    report = q.processes()
    assert report["matched_pids"] == [1]
    assert report["checked_pid_count"] == 1
    assert report["skipped_pid_count"] == 1
    assert report["sweep_budget_limit_count"] == 2
    assert report["per_pid_limit_count"] == 0
    assert report["state"] == "BLOCKED"
    assert report["reason_counts"]["PROCESS_SWEEP_LIMIT"] == 4
    assert_incomplete(report)


def test_limit_before_reference_remains_explicit_uncertainty(physical_proc, monkeypatch):
    monkeypatch.setattr(q, "PROCESS_LIMIT", 1)
    (physical_proc.task / "cgroup").write_bytes(b"0::/ch-stage41b-unobserved\n")
    report = q.processes()
    assert report["matched_pid_count"] == 0
    assert report["state"] == "UNCERTAIN"
    assert report["per_pid_limit_count"] == 2
    assert report["stable_process_identities"] is False
    assert report["stable_installation_bindings"] is False
    assert_incomplete(report)


def test_many_descriptors_are_bounded_and_keep_prior_positive(physical_proc, monkeypatch):
    monkeypatch.setattr(q, "FD_LIMIT", 16)
    directory = physical_proc.task / "fd"
    for child in directory.iterdir():
        child.unlink()
    for number in range(256):
        (directory / f"{number:04}").symlink_to(
            str(q.preparation.PREFIX) + "/PRIVATE_FD_SECRET"
            if number == 0
            else physical_proc.payload
        )
    physical_proc.access.ordered_directories[directory] = sorted(directory.iterdir())
    reads = []
    original = q._proc_readlink

    def readlink(path):
        if directory in path.parents:
            reads.append(path.name)
        return original(path)

    monkeypatch.setattr(q, "_proc_readlink", readlink)
    assert_global_filesystem_methods_unchanged()
    report = q.processes()
    assert len(reads) == 32
    assert len(set(reads)) == 16
    assert report["per_pid_limit_count"] == 1
    assert report["reason_counts"]["FD_LIMIT"] > 0
    assert report["state"] == "BLOCKED"
    assert report["matched_pids"] == [physical_proc.own]
    assert "PRIVATE_FD_SECRET" not in json.dumps(report)
    assert_incomplete(report)
    assert_global_filesystem_methods_unchanged()


def test_large_pid_population_does_not_spend_raw_traffic_as_retained_memory(
    visible_proc, monkeypatch, record_property
):
    import tracemalloc

    visible_proc.items[:] = [Path("/proc/1"), Path("/proc") / str(visible_proc.own)] + [
        Path("/proc") / str(pid) for pid in range(100000, 100766) if pid != visible_proc.own
    ]
    population = len(visible_proc.items)
    raw_size = 96 * 1024
    assert population * raw_size > q.LIMIT
    reads = {}

    def observe(item):
        pid = int(item.name)
        reads[pid] = reads.get(pid, 0) + 1
        row = list(snapshot(pid)[0])
        row[3] = b"0::/" + b"x" * raw_size + b"\n"
        return (tuple(row),)

    monkeypatch.setattr(q, "_pid_snapshot", observe)
    tracemalloc.start()
    try:
        report = q.processes()
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    record_property("peak_traced_bytes", peak)
    record_property("synthetic_raw_bytes_per_sweep", population * raw_size)
    record_property("synthetic_pid_count", population)
    assert len(reads) == population
    assert set(reads.values()) == {2}
    assert report["checked_pid_count"] == population
    assert report["skipped_pid_count"] == report["sweep_budget_limit_count"] == 0
    assert report["stable_visible_snapshot"] is True
    assert report["state"] == "UNCERTAIN"
    assert report["full_confirmation"] is False
    # Inputs cumulatively exceed 64 MiB per pass, but only compact results live
    # between PIDs. This generous bound detects retaining raw snapshots.
    assert peak < 16 * 1024 * 1024


def test_process_payload_is_never_represented_for_matching_or_fingerprints(
    visible_proc, monkeypatch
):
    class RawBytes(bytes):
        def __repr__(self):
            pytest.fail("raw observation repr invoked")

    def observe(item):
        row = list(snapshot(int(item.name))[0])
        row[3] = RawBytes(b"0::/ch-stage41b-PRIVATE_CGROUP\n")
        row[4] = RawBytes(b"0-10 r--p 0 00:00 0 /PRIVATE_MAP\n")
        return (tuple(row),)

    monkeypatch.setattr(q, "_pid_snapshot", observe)
    report = q.processes()
    assert report["matched_pid_count"] == 2
    assert report["state"] == "BLOCKED"
    assert "PRIVATE_" not in json.dumps(report)
    assert_global_filesystem_methods_unchanged()


def test_link_only_partial_observation_counts_survive_later_hash_failure(
    physical_proc, monkeypatch
):
    deny_proc_source(
        monkeypatch,
        {physical_proc.task / source for source in ("stat", "status", "cgroup", "maps")},
    )
    executable = physical_proc.task / "exe"
    executable.unlink()
    executable.symlink_to(str(q.preparation.PREFIX) + "/PRIVATE_LINK_ONLY")

    def failed_digest(value):
        raise RuntimeError("PRIVATE_DIGEST_ERROR")

    monkeypatch.setattr(q, "_digest", failed_digest)
    report = q.processes()
    assert report["matched_pids"] == [physical_proc.own]
    assert report["state"] == "BLOCKED"
    assert report["checked_pid_count"] == report["observed_pid_count"] == 2
    assert report["observed_thread_count"] == 2
    assert report["reason_counts"]["PROCESS_IO_ERROR"] > 0
    assert "PRIVATE_" not in json.dumps(report)
    assert_incomplete(report)
    assert_global_filesystem_methods_unchanged()


def late_inode_metadata(monkeypatch, identities):
    calls = []

    def metadata(path):
        calls.append(path)
        if len(calls) <= len(q.COMMITS):
            return {"identities": [], "complete": False}
        return {
            "identities": [
                ("/PRIVATE_LATE_ALIAS", (0, inode, device)) for device, inode in identities
            ],
            "complete": True,
        }

    monkeypatch.setattr(q, "_installation_metadata", metadata)
    return calls


@pytest.mark.parametrize("reference_source", ["maps", "link"])
@pytest.mark.parametrize("reference_pass", [1, 2])
def test_late_inode_correlation_survives_full_fingerprint_budget(
    physical_proc, monkeypatch, reference_source, reference_pass
):
    physical_proc.access.ordered_directories[physical_proc.root] = sorted(
        physical_proc.root.iterdir()
    )
    target_file = physical_proc.payload.with_name("PRIVATE_LATE_INODE")
    target_file.write_bytes(b"never execute")
    info = target_file.stat()
    target = info.st_dev, info.st_ino
    metadata_calls = late_inode_metadata(monkeypatch, [target])
    # The one inode and PID holder fit exactly; the full thread fingerprint does not.
    minimal = q._PROCESS_RESERVATION + q._INODE_RESERVATION
    monkeypatch.setattr(q, "SWEEP_LIMIT", minimal)
    assert minimal + q._THREAD_RESERVATION > q.SWEEP_LIMIT
    original_snapshot = q._pid_snapshot
    reads = 0

    def observe(item):
        nonlocal reads
        if item.name == "1":
            reads += 1
            if reference_source == "maps":
                maps = (
                    (
                        f"0-10 r--p 0 {os.major(target[0]):x}:{os.minor(target[0]):x} "
                        f"{target[1]} /PRIVATE_RENAMED_ALIAS\n"
                    ).encode()
                    if reads == reference_pass
                    else b"0-10 r--p 0 00:00 0\n"
                )
                (physical_proc.first / "maps").write_bytes(maps)
            else:
                # exe is the first inode-bearing source when maps has inode zero.
                executable = physical_proc.first / "exe"
                executable.unlink()
                executable.symlink_to(
                    target_file if reads == reference_pass else physical_proc.payload
                )
        return original_snapshot(item)

    monkeypatch.setattr(q, "_pid_snapshot", observe)
    original_reference = q._reference_observation
    captured = []

    def reference(snapshot, inodes, observed=None, maximum=q.INODE_LIMIT):
        answer = original_reference(snapshot, inodes, observed, maximum)
        if snapshot.item.name == "1":
            captured.append(set(observed or ()))
        return answer

    monkeypatch.setattr(q, "_reference_observation", reference)
    monkeypatch.setattr(q, "resources", lambda plan: {"state": "UNCERTAIN"})
    diagnosis = q.diagnose({}, b"{}", "0" * 64)
    report = diagnosis["process_observation"]
    assert target in captured[reference_pass - 1]
    assert len(metadata_calls) == 2 * len(q.COMMITS)
    assert report["state"] == "BLOCKED"
    assert report["matched_pids"] == [1]
    assert report["matched_pid_count"] == 1
    assert report["reason_counts"]["PROCESS_SWEEP_LIMIT"] > 0
    assert report["sweep_budget_limit_count"] == 2
    assert report["per_pid_limit_count"] == 0
    assert report["full_confirmation"] is False
    assert diagnosis["qualification"] == "BLOCKED"
    assert diagnosis["coexistence_authorized"] is False
    assert diagnosis["preflight_exception"] is False
    assert "PRIVATE_" not in json.dumps(diagnosis)
    assert_incomplete(report)
    assert_global_filesystem_methods_unchanged()


def test_late_inode_correlation_without_minimal_budget_is_explicitly_incomplete(
    physical_proc, monkeypatch
):
    physical_proc.access.ordered_directories[physical_proc.root] = sorted(
        physical_proc.root.iterdir()
    )
    target = 123, 777
    metadata_calls = late_inode_metadata(monkeypatch, [target])
    (physical_proc.first / "maps").write_bytes(
        f"0-10 r--p 0 {os.major(target[0]):x}:{os.minor(target[0]):x} {target[1]} /PRIVATE_ALIAS\n".encode()
    )
    # One byte below a PID holder plus one inode must not create free cache space.
    monkeypatch.setattr(q, "SWEEP_LIMIT", q._PROCESS_RESERVATION + q._INODE_RESERVATION - 1)
    monkeypatch.setattr(q, "resources", lambda plan: {"state": "UNCERTAIN"})
    diagnosis = q.diagnose({}, b"{}", "0" * 64)
    report = diagnosis["process_observation"]
    assert len(metadata_calls) == 2 * len(q.COMMITS)
    assert report["state"] == "UNCERTAIN"
    assert report["matched_pid_count"] == 0
    assert report["checked_pid_count"] >= 1
    assert report["reason_counts"]["PROCESS_INODE_LIMIT"] > 0
    assert report["reason_counts"]["PROCESS_SWEEP_LIMIT"] > 0
    assert report["full_confirmation"] is False
    assert report["stable_installation_bindings"] is False
    assert diagnosis["qualification"] == "BLOCKED"
    assert "PRIVATE_" not in json.dumps(diagnosis)
    assert_incomplete(report)
    assert_global_filesystem_methods_unchanged()


@pytest.mark.parametrize("missing_bytes", [0, 1])
def test_minimal_late_inode_records_share_the_sweep_budget(
    physical_proc, monkeypatch, missing_bytes
):
    physical_proc.access.ordered_directories[physical_proc.root] = sorted(
        physical_proc.root.iterdir()
    )
    # The first full fingerprint fits. Its reservation must still be deducted
    # before admitting the second PID's minimal correlation record.
    first_info = physical_proc.payload.stat()
    target_file = physical_proc.payload.with_name("PRIVATE_SECOND_PID_INODE")
    target_file.write_bytes(b"never execute")
    second_info = target_file.stat()
    for name in ("exe", "cwd", "root", "fd/5", "fd/6"):
        path = physical_proc.task / name
        path.unlink()
        path.symlink_to(target_file)
    targets = [(first_info.st_dev, first_info.st_ino), (second_info.st_dev, second_info.st_ino)]
    late_inode_metadata(monkeypatch, targets)
    minimal = q._PROCESS_RESERVATION + q._INODE_RESERVATION
    monkeypatch.setattr(q, "SWEEP_LIMIT", 2 * minimal + q._THREAD_RESERVATION - missing_bytes)
    assert minimal + q._THREAD_RESERVATION < q.SWEEP_LIMIT
    assert 2 * (minimal + q._THREAD_RESERVATION) > q.SWEEP_LIMIT
    report = q.processes()
    assert report["checked_pid_count"] == 2
    assert report["observed_thread_count"] == 2
    assert report["sweep_budget_limit_count"] == 1
    assert report["skipped_pid_count"] == 0
    assert report["per_pid_limit_count"] == 0
    assert report["matched_pids"] == ([1] if missing_bytes else sorted([1, physical_proc.own]))
    assert report["state"] == "BLOCKED"
    if missing_bytes:
        assert report["reason_counts"]["PROCESS_INODE_LIMIT"] > 0
    assert_incomplete(report)
    assert_global_filesystem_methods_unchanged()


def test_late_inode_record_survives_failed_fingerprinting_and_stays_budgeted(
    physical_proc, monkeypatch
):
    physical_proc.access.ordered_directories[physical_proc.root] = sorted(
        physical_proc.root.iterdir()
    )
    info = physical_proc.payload.stat()
    late_inode_metadata(monkeypatch, [(info.st_dev, info.st_ino)])
    # Only the first correlation holder and its inode fit. A later exception
    # must neither erase that holder nor let the next PID reuse its reservation.
    monkeypatch.setattr(q, "SWEEP_LIMIT", q._PROCESS_RESERVATION + q._INODE_RESERVATION)

    def failed(snapshot, pid):
        raise RuntimeError("PRIVATE_FINGERPRINT_FAILURE")

    monkeypatch.setattr(q, "_fingerprints", failed)
    report = q.processes()
    assert report["state"] == "BLOCKED"
    assert report["matched_pids"] == [1]
    assert report["skipped_pid_count"] == 1
    assert report["reason_counts"]["PROCESS_IO_ERROR"] == 2
    assert report["full_confirmation"] is False
    assert "PRIVATE_" not in json.dumps(report)
    assert_incomplete(report)
    assert_global_filesystem_methods_unchanged()
