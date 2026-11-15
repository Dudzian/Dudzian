"""Trusted-host synthetic compilation only: exact target runtime is NOT_TESTED.

Reference bytes are assembled separately from builtin compile + marshal.dumps;
no fixture executes its source or PYC and no marshal.loads is used. These tests
exercise the mechanism; they produce no V5c3 execution observation or admission.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import marshal
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from deployment import stage9_observer_compile_worker_v5c3 as worker

pytestmark = [pytest.mark.security, pytest.mark.unit]
_SOURCE = b"assert False, 'never execute this module'\nvalue = 37\n"


def _sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _reference(source: bytes, *, filename: str = "/module.py", optimization: int = 0) -> bytes:
    header = importlib.util.MAGIC_NUMBER + (3).to_bytes(4, "little")
    header += importlib.util.source_hash(source)
    code = compile(source, filename, "exec", dont_inherit=True, optimize=optimization)
    return header + marshal.dumps(code)


def _request(source: bytes = _SOURCE, **changes: Any) -> dict[str, Any]:
    result = {
        "profile": worker.REQUEST_PROFILE,
        "run_id": "synthetic-host-mechanism-only",
        "nonce": "12" * 32,
        "source_sha256": _sha(source),
        "source_size": len(source),
        "source_hash_hex": importlib.util.source_hash(source).hex(),
        "expected_pyc_sha256": _sha(_reference(source)),
        "co_filename": "/module.py",
        "optimization": 0,
        "invalidation": "checked_hash",
    }
    result.update(changes)
    return result


def _compile(
    directory: Path,
    request: dict[str, Any] | None = None,
    *,
    source: bytes = _SOURCE,
    expected_magic: bytes = importlib.util.MAGIC_NUMBER,
) -> dict[str, Any]:
    input_path = directory / "source.py"
    input_path.write_bytes(source)
    return worker._compile_one(
        json.dumps(_request(source) if request is None else request).encode(),
        source_path=input_path,
        output_path=directory / "reconstructed.pyc",
        expected_magic=expected_magic,
    )


def test_trusted_host_synthetic_compile_measures_real_bytes_without_running_source(
    tmp_path: Path,
) -> None:
    report = _compile(tmp_path)
    output = (tmp_path / "reconstructed.pyc").read_bytes()
    assert output == _reference(_SOURCE)
    assert report["output_sha256"] == _sha(output)
    assert report["output_size"] == len(output)
    assert report["comparison"] == "MATCH"
    assert report["flags"] == 3
    assert report["source_hash_hex"] == importlib.util.source_hash(_SOURCE).hex()
    assert report["origin"] == "GUEST_REPORTED"
    assert report["run_id"] == "synthetic-host-mechanism-only"


@pytest.mark.parametrize(
    "changes",
    [
        {"expected_pyc_sha256": "ab" * 32},
        {"source_hash_hex": "12" * 8},
        {"co_filename": "/other/module.py"},
        {"optimization": 1},
        {"optimization": 2},
    ],
)
def test_independently_pinned_output_rejects_hash_filename_optimization(
    tmp_path: Path, changes: dict[str, Any]
) -> None:
    report = _compile(tmp_path, _request(**changes))
    assert report["comparison"] == "RECONSTRUCTION_MISMATCH"
    assert report["origin"] == "GUEST_REPORTED"
    assert (tmp_path / "reconstructed.pyc").is_file()


def test_wrong_magic_is_a_measured_mismatch(tmp_path: Path) -> None:
    report = _compile(tmp_path, expected_magic=b"FAIL")
    assert report["comparison"] == "RECONSTRUCTION_MISMATCH"


@pytest.mark.parametrize(
    "changes",
    [
        {"profile": "wrong"},
        {"run_id": ""},
        {"nonce": ""},
        {"source_size": True},
        {"optimization": True},
        {"optimization": -1},
        {"optimization": 3},
        {"source_size": worker.MAX_SOURCE_BYTES + 1},
        {"invalidation": "timestamp"},
        {"invalidation": "unchecked_hash"},
        {"co_filename": "relative.py"},
        {"co_filename": "/a/../module.py"},
        {"co_filename": "/a/./module.py"},
        {"co_filename": "/a//module.py"},
        {"co_filename": "/a\\module.py"},
        {"co_filename": "/a\x00module.py"},
        {"extra": "substitution"},
    ],
)
def test_invalid_metadata_refused_before_output(tmp_path: Path, changes: dict[str, Any]) -> None:
    with pytest.raises(worker.WorkerRefusal, match="REQUEST_REJECTED"):
        _compile(tmp_path, _request(**changes))
    assert not (tmp_path / "reconstructed.pyc").exists()


@pytest.mark.parametrize("raw", [b"[]", b"null", b"{}", b"\xff", b"[", b"x" * 16385])
def test_invalid_or_oversized_request_refused(raw: bytes) -> None:
    with pytest.raises(worker.WorkerRefusal, match="REQUEST_REJECTED"):
        worker._request(raw)


def test_duplicate_key_substitution_refused() -> None:
    raw = json.dumps(_request()).encode()
    duplicate = b'{"optimization":2,' + raw[1:]
    with pytest.raises(worker.WorkerRefusal, match="REQUEST_REJECTED"):
        worker._request(duplicate)


def test_source_hash_and_size_substitution_refused(tmp_path: Path) -> None:
    for changes in ({"source_sha256": "ab" * 32}, {"source_size": 1}):
        with pytest.raises(worker.WorkerRefusal, match="SOURCE_MISMATCH"):
            _compile(tmp_path, _request(**changes))
    assert not (tmp_path / "reconstructed.pyc").exists()


def test_encoding_bytes_are_preserved_and_substitution_is_refused(tmp_path: Path) -> None:
    latin1 = b"# coding: latin-1\nword = 'caf\xe9'\n"
    report = _compile(tmp_path, source=latin1)
    assert report["comparison"] == "MATCH"
    assert (tmp_path / "reconstructed.pyc").read_bytes() == _reference(latin1)
    (tmp_path / "reconstructed.pyc").unlink()
    with pytest.raises(worker.WorkerRefusal, match="SOURCE_MISMATCH"):
        _compile(tmp_path, _request(latin1), source=latin1.replace(b"\xe9", b"e"))


def test_missing_source_and_existing_output_are_refused(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        worker._compile_one(
            json.dumps(_request()).encode(),
            source_path=tmp_path / "missing.py",
            output_path=tmp_path / "reconstructed.pyc",
            expected_magic=importlib.util.MAGIC_NUMBER,
        )
    output = tmp_path / "reconstructed.pyc"
    output.write_bytes(b"existing forensic bytes")
    with pytest.raises(worker.WorkerRefusal, match="OUTPUT_PATH_REJECTED"):
        _compile(tmp_path)
    assert output.read_bytes() == b"existing forensic bytes"


def test_source_symlink_is_refused(tmp_path: Path) -> None:
    target = tmp_path / "actual.py"
    target.write_bytes(_SOURCE)
    link = tmp_path / "source.py"
    try:
        link.symlink_to(target)
    except OSError:
        pytest.skip("symlink creation unavailable on this platform")
    with pytest.raises(worker.WorkerRefusal, match="PATH_REJECTED"):
        worker._compile_one(
            json.dumps(_request()).encode(),
            source_path=link,
            output_path=tmp_path / "reconstructed.pyc",
            expected_magic=importlib.util.MAGIC_NUMBER,
        )


def test_forced_compiler_failure_has_no_positive_report(tmp_path: Path, monkeypatch: Any) -> None:
    def fail(*args: Any, **kwargs: Any) -> None:
        raise OSError("synthetic process boundary failure")

    monkeypatch.setattr(worker.py_compile, "compile", fail)
    with pytest.raises(OSError):
        _compile(tmp_path)
    assert not (tmp_path / "reconstructed.pyc").exists()


def test_cli_wrong_interpreter_refuses_without_reading_input(monkeypatch: Any, capsys: Any) -> None:
    monkeypatch.setattr(worker.sys, "argv", [worker.TOOL_PATH])
    monkeypatch.setattr(
        worker.sys, "flags", SimpleNamespace(isolated=1, no_site=1, dont_write_bytecode=1)
    )
    monkeypatch.setattr(worker.sys, "version_info", (3, 11, 99))
    assert worker.main() == 2
    assert json.loads(capsys.readouterr().out) == {
        "profile": worker.REPORT_PROFILE,
        "origin": "GUEST_REPORTED",
        "comparison": "REFUSED",
        "reason": "INTERPRETER_REJECTED",
    }


def test_cli_has_no_path_or_runtime_bypass(monkeypatch: Any, capsys: Any) -> None:
    monkeypatch.setattr(worker.sys, "argv", [worker.TOOL_PATH, "--synthetic-bypass"])
    assert worker.main() == 2
    assert json.loads(capsys.readouterr().out)["reason"] == "ARGUMENTS_REJECTED"


@pytest.mark.parametrize("interruption", [InterruptedError, KeyboardInterrupt])
def test_cli_interruption_produces_only_a_failed_guest_report(
    monkeypatch: Any, capsys: Any, interruption: type[BaseException]
) -> None:
    def fail() -> None:
        raise interruption()

    monkeypatch.setattr(worker, "_runtime_guard", fail)
    assert worker.main() == 2
    report = json.loads(capsys.readouterr().out)
    assert report["reason"] == "OPERATION_INTERRUPTED"
    assert report["comparison"] == "REFUSED"
    assert report["origin"] == "GUEST_REPORTED"
