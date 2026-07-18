from __future__ import annotations

import copy
import json
from typing import Any

from ui.pyside_app import preview_block_p_closure_audit as closure
from ui.pyside_app import preview_block_p_desktop_exe_build_readiness_read_model as read_model


class EqualityBomb:
    equality_calls = 0

    def __eq__(self, other: object) -> bool:
        type(self).equality_calls += 1
        raise RuntimeError("EqualityBomb equality must not be called")


class HashBomb:
    hash_calls = 0

    def __hash__(self) -> int:
        type(self).hash_calls += 1
        raise RuntimeError("HashBomb hashing must not be called")


class UnhashableEqual:
    __hash__ = None  # type: ignore[assignment]

    def __eq__(self, other: object) -> bool:
        raise RuntimeError("UnhashableEqual equality must not be called")


class StrSubclass(str):
    pass


class BombKey(str):
    equality_calls = 0
    armed: bool

    def __new__(cls, value: str) -> "BombKey":
        instance = super().__new__(cls, value)
        instance.armed = False
        return instance

    def __eq__(self, other: object) -> bool:
        if self.armed:
            type(self).equality_calls += 1
            raise RuntimeError("BombKey equality must not be called")
        return super().__eq__(other)

    __hash__ = str.__hash__


def _reset_sentinel_counters() -> None:
    EqualityBomb.equality_calls = 0
    HashBomb.hash_calls = 0
    BombKey.equality_calls = 0


def _assert_sentinel_counters_zero() -> None:
    assert EqualityBomb.equality_calls == 0
    assert HashBomb.hash_calls == 0
    assert BombKey.equality_calls == 0


def test_real_source_builder_called_once_and_nominal(monkeypatch: Any) -> None:
    calls = 0
    source = read_model.build_preview_block_p_desktop_exe_build_readiness_read_model()
    before = copy.deepcopy(source)

    def wrapper() -> dict[str, Any]:
        nonlocal calls
        calls += 1
        return source

    monkeypatch.setattr(
        closure, "build_preview_block_p_desktop_exe_build_readiness_read_model", wrapper
    )
    payload = closure.build_preview_block_p_closure_audit()
    assert calls == 1
    assert source == before
    assert payload["source_18_7_accepted"] is True
    assert closure._integrity(payload) is True
    json.dumps(payload)


def test_closure_inventory_counts_capabilities_authorizations_non_execution() -> None:
    payload = closure._nominal()
    assert [r["step"] for r in payload["stage_audit_rows"]] == [f"18.{i}" for i in range(8)]
    assert [r["title"] for r in payload["stage_audit_rows"]] == [
        "BLOCK P DESKTOP EXE PACKAGING ENTRY CONTRACT",
        "BLOCK P DESKTOP EXE PACKAGING SOURCE INVENTORY",
        "BLOCK P DESKTOP EXE PACKAGING INVENTORY MATRIX",
        "BLOCK P DESKTOP EXE PACKAGING CONTRACT",
        "BLOCK P DESKTOP EXE PACKAGING READ MODEL",
        "BLOCK P DESKTOP EXE BUILD READINESS MATRIX",
        "BLOCK P DESKTOP EXE BUILD READINESS CONTRACT",
        "BLOCK P DESKTOP EXE BUILD READINESS READ MODEL",
    ]
    for row in payload["stage_audit_rows"]:
        assert row["source_only"] is True
        assert row["build_performed"] is False
        assert row["runtime_started"] is False
    summary = payload["closure_summary"]
    assert (
        summary["readiness_clause_count"],
        summary["acceptance_rule_count"],
        summary["unique_requirement_count"],
        summary["unique_blocker_count"],
        summary["unique_evidence_count"],
    ) == (17, 6, 8, 12, 12)
    for key in [
        "satisfied_readiness_clause_count",
        "satisfied_acceptance_rule_count",
        "observed_readiness_count",
        "validated_readiness_count",
        "satisfied_readiness_count",
        "ready_readiness_count",
    ]:
        assert summary[key] == 0
    assert set(payload["capability_audit"]["capability_state"].values()) == {"blocked"}
    assert payload["capability_audit"]["capability_state_modified"] is False
    for key, value in payload["authorization_audit"].items():
        assert value is False
    for key, value in payload["non_execution_audit"].items():
        assert value is (key in {"source_read", "closure_audit_built"})
    assert payload["closure_decision"]["block_p_source_only_design_closed"] is True
    assert payload["closure_decision"]["physical_build_completed"] is False


def _blocked_from(source: dict[str, Any], monkeypatch: Any) -> dict[str, Any]:
    monkeypatch.setattr(
        closure, "build_preview_block_p_desktop_exe_build_readiness_read_model", lambda: source
    )
    return closure.build_preview_block_p_closure_audit()


def test_source_tampering_returns_canonical_blocked(monkeypatch: Any) -> None:
    source = read_model.build_preview_block_p_desktop_exe_build_readiness_read_model()
    cases = []
    for key in [
        "source_18_6_accepted",
        "read_model_artifact_complete",
        "ready_for_block_p_8",
        "integrity_valid",
    ]:
        tampered = copy.deepcopy(source)
        tampered[key] = False
        cases.append(tampered)
    tampered = copy.deepcopy(source)
    tampered["status"] = "wrong"
    cases.append(tampered)
    tampered = copy.deepcopy(source)
    tampered["extra"] = True
    cases.append(tampered)
    tampered = copy.deepcopy(source)
    tampered.pop("status")
    cases.append(tampered)
    tampered = {k: source[k] for k in reversed(list(source))}
    cases.append(tampered)
    for case in cases:
        assert _blocked_from(case, monkeypatch) == closure._blocked()


def test_canonical_blocked_18_7_source_returns_canonical_blocked_18_8(monkeypatch: Any) -> None:
    assert _blocked_from(read_model._blocked(), monkeypatch) == closure._blocked()


def test_canonical_blocked_18_8_integrity() -> None:
    blocked = closure._blocked()
    assert blocked == closure._blocked()
    json.dumps(blocked)
    assert closure._integrity(blocked) is True
    extra = copy.deepcopy(blocked)
    extra["extra"] = False
    assert closure._integrity(extra) is False
    for key in closure.BLOCKED_FIELDS:
        missing = copy.deepcopy(blocked)
        missing.pop(key)
        assert closure._integrity(missing) is False
    assert closure._integrity({k: blocked[k] for k in reversed(list(blocked))}) is False
    for key in [
        "packaging_authorized",
        "build_authorized",
        "artifact_creation_authorized",
        "release_authorized",
        "runtime_authorized",
        "orders_authorized",
    ]:
        changed = copy.deepcopy(blocked)
        changed[key] = True
        assert closure._integrity(changed) is False


def test_exact_plain_cycles_and_depth() -> None:
    a: list[Any] = ["x"]
    b: list[Any] = ["x"]
    a.append(a)
    b.append(b)
    assert closure._exact_plain(a, b) is True
    c: list[Any] = ["y"]
    c.append(c)
    assert closure._exact_plain(a, c) is False
    da: dict[str, Any] = {"x": 1}
    db: dict[str, Any] = {"x": 1}
    da["self"] = da
    db["self"] = db
    assert closure._exact_plain(da, db) is True
    dc: dict[str, Any] = {"x": 2}
    dc["self"] = dc
    assert closure._exact_plain(da, dc) is False
    left: dict[str, Any] = {"v": 0}
    right: dict[str, Any] = {"v": 0}
    for _ in range(1500):
        left = {"n": left}
        right = {"n": right}
    assert closure._exact_plain(left, right) is True
    bad: dict[str, Any] = {"v": 0}
    cur_bad = bad
    for _ in range(1499):
        child: dict[str, Any] = {"n": cur_bad}
        cur_bad = child
    bad = {"n": cur_bad}
    cur = bad
    for _ in range(1500):
        cur = cur["n"]
    cur["v"] = 1
    assert closure._exact_plain(left, bad) is False


def test_local_source_top_level_schema_is_frozen_independent(monkeypatch: Any) -> None:
    assert tuple(read_model.TOP_LEVEL_FIELDS) == closure.SOURCE_18_7_TOP_LEVEL_FIELDS
    assert type(closure.SOURCE_18_7_TOP_LEVEL_FIELDS) is tuple
    assert closure.SOURCE_18_7_TOP_LEVEL_FIELDS is not read_model.TOP_LEVEL_FIELDS
    frozen = closure.SOURCE_18_7_TOP_LEVEL_FIELDS
    monkeypatch.setattr(read_model, "TOP_LEVEL_FIELDS", ["tampered"])
    assert closure.SOURCE_18_7_TOP_LEVEL_FIELDS == frozen


def _fresh_source_with_value(path: tuple[str, ...], value: object) -> dict[str, Any]:
    source = read_model.build_preview_block_p_desktop_exe_build_readiness_read_model()
    current: Any = source
    for part in path[:-1]:
        current = current[part]
    current[path[-1]] = value
    return source


def test_custom_scalar_source_values_fail_closed_without_custom_equality_or_hash(
    monkeypatch: Any,
) -> None:
    source = read_model.build_preview_block_p_desktop_exe_build_readiness_read_model()
    first_capability = next(
        iter(source["capability_read_model_state"]["contract_capability_state"])
    )
    cases: list[tuple[tuple[str, ...], object]] = []
    for factory in (EqualityBomb, UnhashableEqual, HashBomb):
        cases.extend(
            [
                (("schema_version",), factory()),
                (("build_readiness_read_model_summary", "readiness_clause_count"), factory()),
                (
                    (
                        "capability_read_model_state",
                        "contract_capability_state",
                        first_capability,
                    ),
                    factory(),
                ),
            ]
        )
    cases.extend(
        [
            (("schema_version",), StrSubclass(source["schema_version"])),
            (
                ("build_readiness_read_model_summary", "readiness_clause_count"),
                StrSubclass("17"),
            ),
            (
                ("capability_read_model_state", "contract_capability_state", first_capability),
                StrSubclass("blocked"),
            ),
        ]
    )

    for path, value in cases:
        _reset_sentinel_counters()
        tampered = _fresh_source_with_value(path, value)
        monkeypatch.setattr(
            closure,
            "build_preview_block_p_desktop_exe_build_readiness_read_model",
            lambda: tampered,
        )
        payload = closure.build_preview_block_p_closure_audit()
        assert payload == closure._blocked()
        assert closure._integrity(payload) is True
        json.dumps(payload)
        _assert_sentinel_counters_zero()


def test_bomb_key_root_returns_blocked_without_custom_equality_or_hash(monkeypatch: Any) -> None:
    _reset_sentinel_counters()
    source = read_model.build_preview_block_p_desktop_exe_build_readiness_read_model()
    target_key = "schema_version"
    bomb_key = BombKey(target_key)
    items = list(source.items())
    source.clear()
    for key, value in items:
        source[bomb_key if key == target_key else key] = value
    bomb_key.armed = True
    monkeypatch.setattr(
        closure, "build_preview_block_p_desktop_exe_build_readiness_read_model", lambda: source
    )
    assert closure.build_preview_block_p_closure_audit() == closure._blocked()
    _assert_sentinel_counters_zero()


def test_upstream_integrity_exception_fails_closed(monkeypatch: Any) -> None:
    source = read_model.build_preview_block_p_desktop_exe_build_readiness_read_model()

    def boom(payload: object) -> bool:
        raise RuntimeError("integrity boom")

    monkeypatch.setattr(closure, "_integrity_18_7", boom)
    monkeypatch.setattr(
        closure, "build_preview_block_p_desktop_exe_build_readiness_read_model", lambda: source
    )
    assert closure._source_accepted(source) is False
    assert closure.build_preview_block_p_closure_audit() == closure._blocked()


def test_upstream_builder_exception_fails_closed(monkeypatch: Any) -> None:
    def boom() -> dict[str, Any]:
        raise RuntimeError("builder boom")

    monkeypatch.setattr(
        closure, "build_preview_block_p_desktop_exe_build_readiness_read_model", boom
    )
    payload = closure.build_preview_block_p_closure_audit()
    assert payload == closure._blocked()
    assert closure._integrity(payload) is True
    json.dumps(payload)
