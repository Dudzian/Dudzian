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


def snapshot(pid, *, start=123, target="/unrelated", device=0, inode=1):
    raw_stat = f"{pid} (worker) S ".encode() + b"0 " * 18 + f"{start} 0\n".encode()
    return (
        (
            str(pid),
            raw_stat,
            b"Uid: 1000 1000 1000 1000\n",
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
    monkeypatch.setattr(q.preparation, "PREFIX", prefix)

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
    content_reads = []

    def read(path):
        if path in private:
            content_reads.append(path)
            raise PermissionError("PRIVATE_CONTENT_DENIED")
        return original_read(path)

    monkeypatch.setattr(q, "_read", read)
    original_open = Path.open

    def guarded_open(path, *args, **kwargs):
        if path in private:
            content_reads.append(path)
            raise PermissionError("PRIVATE_CONTENT_DENIED")
        return original_open(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", guarded_open)
    return SimpleNamespace(prefix=prefix, private=private, content_reads=content_reads)


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
