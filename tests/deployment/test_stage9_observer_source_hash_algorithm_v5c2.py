"""Static independent distribution-source vectors, no target interpreter calls."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

from deployment import stage9_observer_source_hash_algorithm_v5c2 as algorithm
from deployment.stage9_observer_contract import ContractRefusal

pytestmark = [pytest.mark.security, pytest.mark.unit]

_FIXTURES = Path(__file__).parents[1] / "fixtures/stage9_observer/pyc_source_hash_v5c2"
_VECTORS = json.loads((_FIXTURES / "golden_vectors.json").read_text())["vectors"]


def _bytes(row: dict[str, Any]) -> bytes:
    if "source_hex" in row:
        return bytes.fromhex(row["source_hex"])
    return bytes.fromhex(row["source_repeat_hex"]) * row["repeat_count"]


@pytest.mark.parametrize("row", _VECTORS, ids=[row["id"] for row in _VECTORS])
def test_static_exact_distribution_c_reference_vectors(row: dict[str, Any]) -> None:
    raw = _bytes(row)
    assert len(raw) == row["source_size"]
    assert hashlib.sha256(raw).hexdigest() == row["source_sha256"]
    result = algorithm.independent_source_hash(
        raw, algorithm.ALGORITHM_ID, magic=bytes.fromhex(row["magic_hex"])
    )
    assert type(result) is bytes
    assert len(result) == 8
    assert result.hex() == row["expected_hex"]


def test_reference_provenance_and_material_identities_are_pinned() -> None:
    materials = dict(algorithm.ALGORITHM_MATERIALS)
    raw = (_FIXTURES / "source_materials.json").read_bytes()
    assert hashlib.sha256(raw).hexdigest() == materials["source_materials.json"]
    manifest = json.loads(raw)
    assert manifest["version"] == "3.12.3-1ubuntu0.17"
    assert manifest["independent_acceptance"] == "UNVERIFIED"
    assert manifest["exact_target_interpreter_validation"] == "NOT_TESTED"
    assert len(manifest["active_patches"]) == 82
    assert all(patch["apply_exit"] == 0 for patch in manifest["active_patches"])
    for row in manifest["source_archives"] + manifest["packaging"]:
        assert materials[row["identifier"]] == row["sha256"]
    for row in manifest["patched_files"]:
        assert materials["patched/" + row["identifier"]] == row["sha256"]
    reference = (_FIXTURES / "cpython_siphash13_reference.c").read_bytes()
    assert hashlib.sha256(reference).hexdigest() == materials["cpython_siphash13_reference.c"]
    vectors = json.loads((_FIXTURES / "golden_vectors.json").read_bytes())
    assert vectors["reference_source_sha256"] == manifest["reference"]["sha256"]
    assert vectors["source_materials_sha256"] == materials["source_materials.json"]


def test_full_magic_key_and_little_endian_output_are_observable() -> None:
    # Both expected values come from the retained independent C reference.
    raw = b"value = 37\n"
    normal = bytes.fromhex("0bc3e3a8ab4ff6bb")
    changed = bytes.fromhex("10bdb62d8b319586")
    assert algorithm.independent_source_hash(raw, algorithm.ALGORITHM_ID) == normal
    assert (
        algorithm.independent_source_hash(
            raw, algorithm.ALGORITHM_ID, magic=bytes.fromhex("cc0d0d0a")
        )
        == changed
    )
    assert normal != normal[::-1]
    assert algorithm.independent_source_hash(
        raw, algorithm.ALGORITHM_ID, magic=b"\xcb\x0d\0\0"
    ) not in (
        normal,
        normal[::-1],
    )
    assert algorithm.independent_source_hash(
        raw, algorithm.ALGORITHM_ID, magic=algorithm.MAGIC[::-1]
    ) not in (
        normal,
        normal[::-1],
    )


@pytest.mark.parametrize(
    "identifier",
    ["", "siphash24", "siphash13", "cpython-3.11", "sha256", None, True],
)
def test_unapproved_algorithm_identifier_fails_closed(identifier: Any) -> None:
    with pytest.raises(ContractRefusal, match="UNSUPPORTED_SOURCE"):
        algorithm.independent_source_hash(b"", identifier)


@pytest.mark.parametrize("raw", [None, "x", bytearray(b"x"), memoryview(b"x"), 1])
def test_requires_exact_immutable_source_bytes(raw: Any) -> None:
    with pytest.raises(ContractRefusal, match="FORMAT_REJECTED"):
        algorithm.independent_source_hash(raw, algorithm.ALGORITHM_ID)


@pytest.mark.parametrize("magic", [b"", b"abc", b"abcde", bytearray(b"abcd"), "abcd", None])
def test_magic_requires_four_immutable_bytes(magic: Any) -> None:
    with pytest.raises(ContractRefusal, match="FORMAT_REJECTED"):
        algorithm.independent_source_hash(b"x", algorithm.ALGORITHM_ID, magic=magic)


@pytest.mark.parametrize("limit", [-1, True, 1.0, 64 * 1024 * 1024 + 1, None])
def test_source_limit_cannot_expand_profile_or_use_ambiguous_types(limit: Any) -> None:
    with pytest.raises(ContractRefusal, match="LIMIT_EXCEEDED"):
        algorithm.independent_source_hash(b"", algorithm.ALGORITHM_ID, max_source_bytes=limit)


def test_source_limit_is_enforced_before_any_hash_work() -> None:
    with pytest.raises(ContractRefusal, match="LIMIT_EXCEEDED"):
        algorithm.independent_source_hash(b"x", algorithm.ALGORITHM_ID, max_source_bytes=0)
    assert (
        algorithm.independent_source_hash(b"", algorithm.ALGORITHM_ID, max_source_bytes=0).hex()
        == "baf18eedc9387a25"
    )


def test_cooperative_cancellation_propagates_without_returning_hash() -> None:
    calls = 0

    def cancel() -> None:
        nonlocal calls
        calls += 1
        if calls == 3:
            raise RuntimeError("INTERRUPTED")

    with pytest.raises(RuntimeError, match="INTERRUPTED"):
        algorithm.independent_source_hash(b"x" * 8192, algorithm.ALGORITHM_ID, check=cancel)
    assert calls == 3


def test_primitive_enforces_deadline_without_caller_callback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    times = iter([100.0, 100.0, 161.0])
    monkeypatch.setattr(algorithm.time, "monotonic", lambda: next(times))
    with pytest.raises(ContractRefusal, match="LIMIT_EXCEEDED"):
        algorithm.independent_source_hash(b"", algorithm.ALGORITHM_ID)


@pytest.mark.parametrize("elapsed", [-1.0, 60.0, float("nan"), float("inf")])
def test_invalid_or_expired_clock_fails_closed(
    monkeypatch: pytest.MonkeyPatch, elapsed: float
) -> None:
    times = iter([0.0, elapsed])
    monkeypatch.setattr(algorithm.time, "monotonic", lambda: next(times))
    with pytest.raises(ContractRefusal, match="LIMIT_EXCEEDED"):
        algorithm.independent_source_hash(b"", algorithm.ALGORITHM_ID)


def test_primitive_does_not_call_host_hash_or_execution_facilities() -> None:
    source = Path(algorithm.__file__).read_text()
    for prohibited in (
        "import importlib",
        "import _imp",
        "marshal.loads(",
        "compile(",
        "exec(",
        "eval(",
        "subprocess",
    ):
        assert prohibited not in source
