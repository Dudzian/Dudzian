"""Streaming procfs boundary regressions on delegated synthetic paths only."""

import io
import json
import os
import sys
import tracemalloc

import pytest

from deployment import stage9_stage41b_coexistence as q
from tests.security.test_stage9_stage41b_process_observation import (
    LocalPath,
    assert_global_filesystem_methods_unchanged,
    physical_proc as physical_proc,
)

pytestmark = pytest.mark.skipif(sys.platform != "linux", reason="native Linux observation")

PRIVATE_MARKER = b"PRIVATE_STREAM_OBSERVATION_VALUE"


class TrackedStream(io.BytesIO):
    """Reject an unbounded read without modifying any filesystem implementation."""

    def __init__(self, payload, requests, *, chunk_size=None, fail_after=None):
        super().__init__(payload)
        self.requests = requests
        self.chunk_size = chunk_size
        self.fail_after = fail_after
        self.calls = 0

    def read(self, size=-1):
        assert 0 < size <= q.PROC_READ_LIMIT
        self.requests.append(size)
        if self.fail_after is not None and self.calls >= self.fail_after:
            raise OSError(PRIVATE_MARKER.decode())
        self.calls += 1
        return super().read(min(size, self.chunk_size) if self.chunk_size else size)


def scripted_streams(physical_proc, monkeypatch, sources):
    requests = []
    access = physical_proc.access

    class StreamPath(LocalPath):
        def open(self, *args, **kwargs):
            if self._path in sources:
                assert args == ("rb",)
                payload, options = sources[self._path]
                return TrackedStream(payload, requests, **options)
            return super().open(*args, **kwargs)

    monkeypatch.setattr(access, "path", lambda path: StreamPath(path, access))
    return requests


def map_line(*, inode=0, name=b"/unrelated", device=b"01:02"):
    return b"0-10 r--p 0 " + device + b" " + str(inode).encode() + b" " + name + b"\n"


def assert_redacted_incomplete(report):
    assert report["full_confirmation"] is False
    assert report["stable_visible_snapshot"] is False
    assert report["stable_installation_bindings"] is False
    assert PRIVATE_MARKER.decode() not in json.dumps(report)
    assert_global_filesystem_methods_unchanged()


@pytest.mark.parametrize("reference", ["pathname", "inode"])
def test_maps_reference_split_across_short_reads(physical_proc, monkeypatch, reference):
    target = (os.makedev(1, 2), 345)
    physical_proc.metadata["identities"] = [("/known-alias", (0, target[1], target[0]))]
    payload = map_line(
        inode=target[1] if reference == "inode" else 0,
        name=str(q.preparation.PREFIX).encode() + b"/old " + PRIVATE_MARKER
        if reference == "pathname"
        else b"/renamed-alias " + PRIVATE_MARKER,
    )
    requests = scripted_streams(
        physical_proc,
        monkeypatch,
        {physical_proc.task / "maps": (payload, {"chunk_size": 7})},
    )
    report = q.processes()
    assert report["matched_pids"] == [physical_proc.own]
    assert report["state"] == "BLOCKED"
    assert report["stable_visible_snapshot"] is True
    assert len(requests) > 2
    assert max(requests) <= q.PROC_READ_LIMIT
    assert PRIVATE_MARKER.decode() not in json.dumps(report)
    assert_global_filesystem_methods_unchanged()


def test_overlong_maps_line_keeps_later_valid_inode(physical_proc, monkeypatch):
    target = (os.makedev(1, 2), 345)
    physical_proc.metadata["identities"] = [("/known-alias", (0, target[1], target[0]))]
    payload = map_line(name=PRIVATE_MARKER + b"x" * (q.PROC_READ_LIMIT + 17)) + map_line(
        inode=target[1]
    )
    scripted_streams(physical_proc, monkeypatch, {physical_proc.task / "maps": (payload, {})})
    report = q.processes()
    assert report["matched_pids"] == [physical_proc.own]
    assert report["state"] == "BLOCKED"
    assert report["reason_counts"]["MAPS_LINE_LIMIT"] > 0
    assert report["per_pid_limit_count"] == 1
    assert_redacted_incomplete(report)


def test_overlong_maps_pathname_crosses_block_boundary(physical_proc, monkeypatch):
    prefix = str(q.preparation.PREFIX).encode()
    header = b"0-10 r--p 0 00:00 0 /"
    padding = b"x" * (q.PROC_READ_LIMIT - len(header) - len(prefix) // 2)
    payload = header + padding + prefix + b"/" + PRIVATE_MARKER + b"\n"
    scripted_streams(physical_proc, monkeypatch, {physical_proc.task / "maps": (payload, {})})
    report = q.processes()
    assert report["matched_pids"] == [physical_proc.own]
    assert report["state"] == "BLOCKED"
    assert report["reason_counts"]["MAPS_LINE_LIMIT"] > 0
    assert_redacted_incomplete(report)


def test_known_maps_inode_matches_after_retained_cache_is_full(physical_proc, monkeypatch):
    target = (os.makedev(1, 2), 345)
    payload = map_line(inode=111) + map_line(inode=222) + map_line(inode=target[1])
    scripted_streams(physical_proc, monkeypatch, {physical_proc.task / "maps": (payload, {})})
    observation = q._pid_snapshot(physical_proc.access.path(physical_proc.task))
    retained = set()
    matched, issues = q._reference_observation(observation, {target}, retained, 1)
    assert matched
    assert len(retained) == 1
    assert target not in retained
    assert issues["PROCESS_INODE_LIMIT"] > 0
    fingerprint, _ = q._fingerprints(observation, physical_proc.own)
    assert fingerprint["observed"]
    assert_global_filesystem_methods_unchanged()


def test_maps_positive_survives_later_stream_io_failure(physical_proc, monkeypatch):
    target = (os.makedev(1, 2), 345)
    physical_proc.metadata["identities"] = [("/known-alias", (0, target[1], target[0]))]
    first_record = map_line(inode=target[1])
    scripted_streams(
        physical_proc,
        monkeypatch,
        {
            physical_proc.task / "maps": (
                first_record + PRIVATE_MARKER * 8,
                {"chunk_size": len(first_record), "fail_after": 1},
            )
        },
    )
    report = q.processes()
    assert report["matched_pids"] == [physical_proc.own]
    assert report["matched_pid_count"] == 1
    assert report["state"] == "BLOCKED"
    assert report["reason_counts"]["MAPS_IO_ERROR"] == 2
    assert_redacted_incomplete(report)


def test_many_threads_stream_shared_maps_without_retaining_raw_payloads(
    physical_proc, monkeypatch, record_property
):
    directories = [physical_proc.task]
    directories.extend(
        physical_proc.add_task(physical_proc.own, physical_proc.own + offset)
        for offset in range(1, 49)
    )
    payload = map_line(name=b"/shared-library " + PRIVATE_MARKER) * 4096
    scripted_streams(
        physical_proc,
        monkeypatch,
        {directory / "maps": (payload, {}) for directory in directories},
    )
    observation = q._pid_snapshot(physical_proc.access.path(physical_proc.task))
    tracemalloc.start()
    try:
        matched, issues = q._reference_observation(observation, set())
        retained_bytes, peak_bytes = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    record_property("peak_traced_bytes", peak_bytes)
    record_property("retained_traced_bytes", retained_bytes)
    record_property("raw_observed_bytes", len(payload) * len(directories))
    assert not matched
    assert not issues
    assert len(observation) == len(directories)
    assert len(observation.fingerprint["identities"]) == len(directories)
    # Observe over 12 MiB while permitting generous Python/allocator overhead.
    assert len(payload) * len(directories) > 12 * 1024 * 1024
    assert peak_bytes < len(payload) * 8
    assert retained_bytes < len(payload) * 3
    assert all(row[1:5] == (b"", b"", b"", b"") for row in observation)
    assert PRIVATE_MARKER not in json.dumps(observation.fingerprint).encode()
    assert_global_filesystem_methods_unchanged()


@pytest.mark.parametrize("consume", [False, True])
def test_proc_reader_bounds_every_read_and_preserves_exact_prefix(monkeypatch, consume):
    monkeypatch.setattr(q, "PROC_READ_LIMIT", 1024)
    payload = PRIVATE_MARKER * 2048
    requests = []

    class Source:
        def open(self, mode):
            assert mode == "rb"
            return TrackedStream(payload, requests)

    observed = []
    with pytest.raises(q._ProcReadError) as caught:
        q._proc_bytes(
            Source(),
            maximum=4099,
            consume=observed.append if consume else None,
        )
    assert caught.value.reason == "LIMIT"
    assert max(requests) <= 1024
    if consume:
        assert b"".join(observed) == payload[:4099]
        assert caught.value.partial == b""
    else:
        assert caught.value.partial == payload[:1024]
    assert str(caught.value) == "LIMIT"
    assert_global_filesystem_methods_unchanged()


@pytest.mark.parametrize("failure", ["io", "limit"])
def test_truncated_maps_inode_digits_never_invent_a_positive(physical_proc, monkeypatch, failure):
    target = (os.makedev(1, 2), 123)
    physical_proc.metadata["identities"] = [("/known-alias", (0, target[1], target[0]))]
    payload = map_line(inode=1234)
    cutoff = payload.index(b"1234") + len(b"123")
    options = {"chunk_size": cutoff, "fail_after": 1} if failure == "io" else {}
    scripted_streams(physical_proc, monkeypatch, {physical_proc.task / "maps": (payload, options)})
    if failure == "limit":
        initial_bytes = sum(
            (physical_proc.task / name).stat().st_size for name in ("stat", "status", "cgroup")
        )
        monkeypatch.setattr(q, "PROCESS_LIMIT", initial_bytes + cutoff)
    report = q.processes()
    assert report["matched_pids"] == []
    assert report["state"] == "UNCERTAIN"
    assert report["reason_counts"]["MAPS_IO_ERROR" if failure == "io" else "MAPS_LIMIT"] > 0
    assert_redacted_incomplete(report)


@pytest.mark.parametrize("stage", ["descriptor", "thread"])
def test_positive_cgroup_survives_later_unexpected_fingerprint_failure(
    physical_proc, monkeypatch, stage
):
    (physical_proc.task / "cgroup").write_bytes(b"0::/ch-stage41b-" + PRIVATE_MARKER + b"\n")

    def failed_digest(value):
        raise RuntimeError(PRIVATE_MARKER.decode())

    if stage == "descriptor":
        monkeypatch.setattr(q, "_digest", failed_digest)
    else:
        original = q._hash_update

        def hash_update(digest, value):
            if isinstance(value, tuple) and len(value) == 7 and value[0] == str(physical_proc.own):
                raise RuntimeError(PRIVATE_MARKER.decode())
            return original(digest, value)

        monkeypatch.setattr(q, "_hash_update", hash_update)
    report = q.processes()
    assert report["matched_pids"] == [physical_proc.own]
    assert report["state"] == "BLOCKED"
    assert report["observed_pid_count"] == 2
    assert report["reason_counts"]["PROCESS_IO_ERROR"] > 0
    assert_redacted_incomplete(report)


def test_metadata_io_failure_preserves_bounded_partial_bytes():
    payload = b"0::/ch-stage41b-" + PRIVATE_MARKER + b"\n"
    requests = []
    cutoff = len(b"0::/ch-stage41b-")

    class Source:
        def open(self, mode):
            assert mode == "rb"
            return TrackedStream(payload, requests, chunk_size=cutoff, fail_after=1)

    with pytest.raises(q._ProcReadError) as caught:
        q._proc_bytes(Source())
    assert caught.value.reason == "IO_ERROR"
    assert caught.value.partial == payload[:cutoff]
    assert str(caught.value) == "IO_ERROR"
    assert max(requests) <= q.PROC_READ_LIMIT
    assert_global_filesystem_methods_unchanged()


def test_partial_cgroup_positive_survives_stream_io_failure(physical_proc, monkeypatch):
    payload = b"0::/ch-stage41b-" + PRIVATE_MARKER + b"\n"
    scripted_streams(
        physical_proc,
        monkeypatch,
        {
            physical_proc.task / "cgroup": (
                payload,
                {"chunk_size": len(b"0::/ch-stage41b-"), "fail_after": 1},
            )
        },
    )
    report = q.processes()
    assert report["matched_pids"] == [physical_proc.own]
    assert report["state"] == "BLOCKED"
    assert report["observed_pid_count"] == 2
    assert report["reason_counts"]["CGROUP_IO_ERROR"] == 2
    assert_redacted_incomplete(report)


def test_many_descriptor_targets_are_hashed_without_retaining_raw_strings(
    physical_proc, monkeypatch, record_property
):
    count = 8192
    fd_directory = physical_proc.task / "fd"
    payload_info = physical_proc.payload.stat()
    target_prefix = "/unrelated/" + PRIVATE_MARKER.decode() + "x" * 4000
    raw_target_bytes = count * len(target_prefix) + sum(len(str(number)) for number in range(count))
    access = physical_proc.access
    original_readlink = q._proc_readlink
    # Construct synthetic paths with the fixture, before tracing the observer.
    # CPython interns pathlib names; resizing that process-global table during
    # a large CI suite must not be attributed to retained process observations.
    descriptor_paths = tuple(fd_directory / str(number) for number in range(count))
    for path in descriptor_paths:
        # pathlib parses parts lazily, so construction alone does not intern them.
        _ = path.parts

    class GeneratedDescriptors(LocalPath):
        def iterdir(self):
            if self._path == fd_directory:
                for path in descriptor_paths:
                    yield access.path(path)
                return
            yield from super().iterdir()

        def stat(self, *args, **kwargs):
            if self._path.parent == fd_directory:
                return payload_info
            return super().stat(*args, **kwargs)

    def readlink(path):
        if path.parent == fd_directory:
            return target_prefix + path.name
        return original_readlink(path)

    monkeypatch.setattr(access, "path", lambda path: GeneratedDescriptors(path, access))
    monkeypatch.setattr(q, "_proc_readlink", readlink)
    observation = q._pid_snapshot(access.path(physical_proc.task))
    tracemalloc.start()
    try:
        matched, issues = q._reference_observation(observation, set())
        retained_bytes, peak_bytes = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    record_property("peak_traced_bytes", peak_bytes)
    record_property("retained_traced_bytes", retained_bytes)
    record_property("raw_observed_bytes", raw_target_bytes)
    record_property("descriptor_count", count)
    assert not matched
    assert not issues
    assert len(observation[0][5]) == count + 3
    assert len(observation.inodes) == 1
    assert peak_bytes < raw_target_bytes // 4
    assert retained_bytes < raw_target_bytes // 8
    assert PRIVATE_MARKER not in json.dumps(observation.fingerprint).encode()
    assert_global_filesystem_methods_unchanged()
