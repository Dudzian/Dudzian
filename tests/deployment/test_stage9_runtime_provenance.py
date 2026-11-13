"""Read-only provenance collection cannot manufacture an authenticated PASS."""

import hashlib
import json
import os
import stat
from types import SimpleNamespace

import pytest

from scripts import stage9_runtime_provenance as provenance


@pytest.fixture
def manifest_data_reader(monkeypatch):
    """Test data validation on Windows without emulating native POSIX FD security.

    Linux keeps exercising the real nofollow reader in these tests. Windows
    substitutes only the trusted file-read boundary for inventory-data tests.
    """
    if os.name != "posix":

        def read_test_bytes(path, limit):
            raw = path.read_bytes()
            assert len(raw) <= limit
            return raw, path.stat()

        monkeypatch.setattr(provenance, "_read", read_test_bytes)


@pytest.mark.parametrize(
    "name,expected",
    [
        (
            "/usr/lib/python3.12/__pycache__/example.cpython-312.pyc",
            "/usr/lib/python3.12/example.py",
        ),
        ("/usr/lib/python3.12/example.pyc", None),
    ],
)
def test_pyc_source_uses_explicit_posix_inventory_paths(name, expected):
    assert provenance.pyc_source(name) == expected


def test_manifest_digest_is_checked_before_json_interpretation(
    tmp_path, monkeypatch, manifest_data_reader
):
    path = tmp_path / "manifest"
    path.write_bytes(b"not json")
    monkeypatch.setattr(provenance, "_json", lambda raw: pytest.fail("unverified JSON parsed"))
    with pytest.raises(provenance.ProvenanceError, match="VERIFIED_MANIFEST_DIGEST_MISMATCH"):
        provenance.load_inventory(path, "0" * 64)


def test_duplicate_manifest_key_refused(tmp_path, manifest_data_reader):
    path = tmp_path / "manifest"
    raw = b'{"external_files":{},"external_files":{}}'
    path.write_bytes(raw)
    with pytest.raises(provenance.ProvenanceError, match="DUPLICATE_JSON_KEY"):
        provenance.load_inventory(path, hashlib.sha256(raw).hexdigest())


def test_file_inventory_requires_full_integrity_record(tmp_path, manifest_data_reader):
    path = tmp_path / "manifest"
    raw = b'{"external_files":{"/file":{"kind":"file","mode":420}}}'
    path.write_bytes(raw)
    with pytest.raises(provenance.ProvenanceError, match="EXTERNAL_RECORD_INVALID"):
        provenance.load_inventory(path, hashlib.sha256(raw).hexdigest())


def test_valid_inventory_bound_to_verified_digest(tmp_path, manifest_data_reader):
    path = tmp_path / "manifest"
    inventory = {"/usr/lib/python3.12": {"kind": "directory", "mode": 493}}
    raw = json.dumps({"external_files": inventory}).encode()
    path.write_bytes(raw)
    assert provenance.load_inventory(path, hashlib.sha256(raw).hexdigest()) == inventory


@pytest.mark.skipif(os.name != "posix", reason="Native nofollow/fstat race test requires POSIX")
def test_file_mutation_during_read_refused(tmp_path, monkeypatch):
    path = tmp_path / "file"
    path.write_bytes(b"original")
    original_fstat = provenance.os.fstat
    calls = 0

    def fstat(fd):
        nonlocal calls
        calls += 1
        if calls == 2:
            path.write_bytes(b"changed data")
        return original_fstat(fd)

    monkeypatch.setattr(provenance.os, "fstat", fstat)
    with pytest.raises(provenance.ProvenanceError, match="OBJECT_CHANGED_DURING_READ"):
        provenance.observe(path)


@pytest.mark.parametrize("name", ["/../etc/passwd", "/tmp/*", "/tmp//file", "relative"])
def test_noncanonical_inventory_path_refused(tmp_path, name, manifest_data_reader):
    path = tmp_path / "manifest"
    raw = json.dumps({"external_files": {name: {"kind": "directory", "mode": 493}}}).encode()
    path.write_bytes(raw)
    with pytest.raises(provenance.ProvenanceError, match="EXTERNAL_RECORD_INVALID"):
        provenance.load_inventory(path, hashlib.sha256(raw).hexdigest())


@pytest.mark.skipif(os.name != "posix", reason="Native POSIX file observation and pyc paths")
def test_pyc_observed_as_data_without_execution_and_no_authenticated_pass(tmp_path, monkeypatch):
    path = tmp_path / "__pycache__" / "source.cpython-312.pyc"
    path.parent.mkdir()
    raw = b"invalid bytecode; must never execute"
    path.write_bytes(raw)
    expected = {
        "kind": "file",
        "mode": stat.S_IMODE(path.stat().st_mode),
        "size": len(raw),
        "sha256": hashlib.sha256(raw).hexdigest(),
    }
    paths = []
    monkeypatch.setattr(provenance, "package_observations", lambda p: paths.extend(p) or {})
    before = path.stat()
    report = provenance.collect({str(path): expected})
    after = path.stat()
    assert report["status"] == "BLOCKED"
    assert report["external_count"] == report["pyc_count"] == 1
    assert report["rows"][0]["observed"]["pyc_provenance"] == "UNRESOLVED_GENERATED_CODE"
    assert str(tmp_path / "source.py") in paths
    assert path.read_bytes() == raw
    assert (before.st_mtime_ns, before.st_mode) == (after.st_mtime_ns, after.st_mode)


def test_content_mismatch_is_failure(tmp_path, monkeypatch, manifest_data_reader):
    path = tmp_path / "object"
    path.write_bytes(b"unexpected")
    monkeypatch.setattr(provenance, "package_observations", lambda p: {})
    report = provenance.collect({str(path): {"kind": "file", "sha256": "0" * 64}})
    assert report["status"] == "FAIL"
    assert report["mismatched_or_unreadable"] == [str(path)]
    assert report["rows"][0]["observed"]["sha256"] == hashlib.sha256(b"unexpected").hexdigest()


@pytest.mark.skipif(os.name != "posix", reason="Linux native object types")
def test_symlink_and_fifo_never_opened_as_regular_code(tmp_path):
    link = tmp_path / "link"
    link.symlink_to(tmp_path / "absent")
    fifo = tmp_path / "fifo"
    os.mkfifo(fifo)
    assert provenance.observe(link)["kind"] == "symlink"
    assert provenance.observe(fifo)["kind"] == "special"


def test_collector_command_retains_blocked_exit_code(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(provenance, "load_inventory", lambda *args: {})
    monkeypatch.setattr(
        provenance,
        "collect",
        lambda _: {
            "external_count": 1352,
            "pyc_count": 573,
            "status": "BLOCKED",
        },
    )
    assert (
        provenance.main(["--manifest", str(tmp_path), "--verified-manifest-sha256", "a" * 64]) == 2
    )
    assert json.loads(capsys.readouterr().out)["status"] == "BLOCKED"


def test_exact_audit_counts_cannot_be_adjusted_to_pass(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(provenance, "load_inventory", lambda *args: {})
    monkeypatch.setattr(
        provenance,
        "collect",
        lambda _: {
            "external_count": 1351,
            "pyc_count": 573,
            "status": "BLOCKED",
        },
    )
    assert (
        provenance.main(["--manifest", str(tmp_path), "--verified-manifest-sha256", "a" * 64]) == 1
    )
    assert json.loads(capsys.readouterr().out)["audit_count_mismatch"] is True


def test_dpkg_only_queries_and_cannot_authenticate_packages(monkeypatch):
    calls = []

    def run(argv, **kwargs):
        calls.append((argv, kwargs))
        return SimpleNamespace(returncode=0, stdout=b"libc6:amd64\t2.39-0ubuntu8.6\tamd64\tii\n")

    monkeypatch.setattr(provenance.subprocess, "run", run)
    result = provenance.package_observations(["/usr/lib/libc.so.6"])
    assert result["authenticated"] is False
    assert "2.39-0ubuntu8.6" in result["versions_tsv"]
    assert calls[0][0][:2] == ["/usr/bin/dpkg-query", "--show"]
    assert calls[1][0] == ["/usr/bin/dpkg-query", "--search", "--", "/usr/lib/libc.so.6"]
    assert calls[0][1]["env"] == provenance.SAFE_ENV


def test_dpkg_failure_refused(monkeypatch):
    monkeypatch.setattr(
        provenance.subprocess,
        "run",
        lambda *a, **k: SimpleNamespace(
            returncode=2,
            stdout=b"",
        ),
    )
    with pytest.raises(provenance.ProvenanceError, match="DPKG_QUERY_FAILED"):
        provenance.package_observations(["/usr/lib/libc.so.6"])
