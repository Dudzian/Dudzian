from __future__ import annotations

import copy
import json
from typing import Any

import pytest

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


PathPart = str | int
Path = tuple[PathPart, ...]


class ListSubclass(list[object]):
    pass


class DictSubclass(dict[str, object]):
    pass


def _get_path(root: object, path: Path) -> object:
    current = root
    for part in path:
        if type(part) is str:
            assert type(current) is dict
            current = current[part]
        else:
            assert type(part) is int
            assert type(current) is list
            current = current[part]
    return current


def _set_path(root: object, path: Path, value: object) -> None:
    assert path
    parent = _get_path(root, path[:-1])
    final = path[-1]
    if type(final) is str:
        assert type(parent) is dict
        parent[final] = value
    else:
        assert type(final) is int
        assert type(parent) is list
        parent[final] = value


def _replace_path(root: object, path: Path, replacement: object) -> object:
    if not path:
        return replacement
    _set_path(root, path, replacement)
    return root


def _leaf_paths(root: object) -> list[Path]:
    paths: list[Path] = []
    pending: list[tuple[object, Path]] = [(root, ())]
    while pending:
        value, path = pending.pop()
        if type(value) is dict:
            for key, child in reversed(list(value.items())):
                assert type(key) is str
                pending.append((child, path + (key,)))
        elif type(value) is list:
            for index in range(len(value) - 1, -1, -1):
                pending.append((value[index], path + (index,)))
        else:
            assert type(value) in (str, bool, int, type(None))
            paths.append(path)
    return paths


def _container_paths(root: object) -> tuple[list[Path], list[Path]]:
    dict_paths: list[Path] = []
    list_paths: list[Path] = []
    pending: list[tuple[object, Path]] = [(root, ())]
    while pending:
        value, path = pending.pop()
        if type(value) is dict:
            dict_paths.append(path)
            for key, child in reversed(list(value.items())):
                assert type(key) is str
                pending.append((child, path + (key,)))
        elif type(value) is list:
            list_paths.append(path)
            for index in range(len(value) - 1, -1, -1):
                pending.append((value[index], path + (index,)))
    return dict_paths, list_paths


def _path_id(path: Path) -> str:
    if not path:
        return "<root>"
    return ".".join(str(part) for part in path)


SOURCE_LEAF_PATHS = _leaf_paths(closure._trusted_source_template())
NOMINAL_LEAF_PATHS = _leaf_paths(closure._nominal())
BLOCKED_LEAF_PATHS = _leaf_paths(closure._blocked())

SOURCE_EXACT_VALUE_PATHS = tuple(SOURCE_LEAF_PATHS)
SOURCE_EXACT_TYPE_PATHS = tuple(SOURCE_LEAF_PATHS)

NOMINAL_EXACT_VALUE_PATHS = tuple(NOMINAL_LEAF_PATHS)
NOMINAL_EXACT_TYPE_PATHS = tuple(NOMINAL_LEAF_PATHS)

BLOCKED_EXACT_VALUE_PATHS = tuple(BLOCKED_LEAF_PATHS)
BLOCKED_EXACT_TYPE_PATHS = tuple(BLOCKED_LEAF_PATHS)

SOURCE_DICT_PATHS, SOURCE_LIST_PATHS = _container_paths(closure._trusted_source_template())
NOMINAL_DICT_PATHS, NOMINAL_LIST_PATHS = _container_paths(closure._nominal())
BLOCKED_DICT_PATHS, BLOCKED_LIST_PATHS = _container_paths(closure._blocked())


def _replace_key(mapping: dict[str, Any], key: str) -> BombKey:
    bomb_key = BombKey(key)
    items = list(mapping.items())
    mapping.clear()
    for item_key, value in items:
        mapping[bomb_key if item_key == key else item_key] = value
    bomb_key.armed = True
    return bomb_key


def _wrong_scalar(value: object) -> object:
    if type(value) is bool:
        result: object = not value
    elif type(value) is int:
        result = value + 1
    elif type(value) is str:
        result = value + "_tampered"
    else:
        raise AssertionError("None requires dedicated handling")
    assert type(result) is type(value)
    assert result != value
    return result


def _wrong_type_scalar(value: object) -> object:
    if type(value) is bool:
        result: object = 0
    elif type(value) is int:
        result = True
    elif type(value) is str:
        result = StrSubclass(value)
    else:
        result = EqualityBomb()
    assert type(result) is not type(value)
    return result


def _assert_scalar_paths(
    name: str,
    factory: Any,
    paths: list[Path],
    expected_count: int,
) -> None:
    payload = factory()
    assert len(paths) == expected_count
    assert len(paths) == len(set(paths)), name
    for path in paths:
        leaf = _get_path(payload, path)
        assert leaf is not None, _path_id(path)
        assert type(leaf) in (str, bool, int), _path_id(path)
        assert _get_path(factory(), path) == leaf, _path_id(path)


def test_scalar_path_counts_are_canonical() -> None:
    _assert_scalar_paths("source", closure._trusted_source_template, SOURCE_LEAF_PATHS, 742)
    _assert_scalar_paths("nominal", closure._nominal, NOMINAL_LEAF_PATHS, 289)
    _assert_scalar_paths("blocked", closure._blocked, BLOCKED_LEAF_PATHS, 16)

    assert all(
        _get_path(closure._trusted_source_template(), path) is not None
        for path in SOURCE_LEAF_PATHS
    )
    assert all(_get_path(closure._nominal(), path) is not None for path in NOMINAL_LEAF_PATHS)
    assert all(_get_path(closure._blocked(), path) is not None for path in BLOCKED_LEAF_PATHS)


def _assert_source_scalar_mutation_blocks(
    monkeypatch: pytest.MonkeyPatch,
    path: Path,
    replacement: object,
) -> None:
    source = closure._trusted_source_template()
    canonical = closure._trusted_source_template()

    original = _get_path(source, path)
    _set_path(source, path, replacement)

    assert type(replacement) is not type(original) or replacement != original
    assert closure._exact_plain(source, canonical) is False
    assert closure._source_accepted(source) is False

    calls = 0

    def builder() -> dict[str, Any]:
        nonlocal calls
        calls += 1
        return source

    monkeypatch.setattr(
        closure,
        "build_preview_block_p_desktop_exe_build_readiness_read_model",
        builder,
    )

    payload = closure.build_preview_block_p_closure_audit()

    assert calls == 1
    assert payload == closure._blocked()
    assert closure._integrity(payload) is True
    assert json.dumps(payload)


@pytest.mark.parametrize("path", SOURCE_EXACT_VALUE_PATHS, ids=_path_id)
def test_source_exact_value_scalar_matrix_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
    path: Path,
) -> None:
    original = _get_path(closure._trusted_source_template(), path)

    assert type(original) in (str, bool, int)

    replacement = _wrong_scalar(original)

    assert type(replacement) is type(original)
    assert replacement != original

    _assert_source_scalar_mutation_blocks(monkeypatch, path, replacement)


@pytest.mark.parametrize("path", NOMINAL_EXACT_VALUE_PATHS, ids=_path_id)
def test_nominal_exact_value_scalar_matrix_is_rejected(path: Path) -> None:
    payload = closure._nominal()
    original = _get_path(payload, path)

    assert type(original) in (str, bool, int)

    replacement = _wrong_scalar(original)

    assert type(replacement) is type(original)
    assert replacement != original

    _set_path(payload, path, replacement)

    assert closure._integrity(payload) is False


@pytest.mark.parametrize("path", BLOCKED_EXACT_VALUE_PATHS, ids=_path_id)
def test_blocked_exact_value_scalar_matrix_is_rejected(path: Path) -> None:
    payload = closure._blocked()
    original = _get_path(payload, path)

    assert type(original) in (str, bool, int)

    replacement = _wrong_scalar(original)

    assert type(replacement) is type(original)
    assert replacement != original

    _set_path(payload, path, replacement)

    assert closure._integrity(payload) is False


@pytest.mark.parametrize("path", SOURCE_EXACT_TYPE_PATHS, ids=_path_id)
def test_source_exact_type_scalar_matrix_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
    path: Path,
) -> None:
    _reset_sentinel_counters()

    original = _get_path(closure._trusted_source_template(), path)
    replacement = _wrong_type_scalar(original)

    assert type(replacement) is not type(original)

    _assert_source_scalar_mutation_blocks(monkeypatch, path, replacement)

    _assert_sentinel_counters_zero()


@pytest.mark.parametrize("path", NOMINAL_EXACT_TYPE_PATHS, ids=_path_id)
def test_nominal_exact_type_scalar_matrix_is_rejected(path: Path) -> None:
    _reset_sentinel_counters()

    payload = closure._nominal()
    original = _get_path(payload, path)
    replacement = _wrong_type_scalar(original)

    assert type(replacement) is not type(original)

    _set_path(payload, path, replacement)

    assert closure._integrity(payload) is False
    _assert_sentinel_counters_zero()


@pytest.mark.parametrize("path", BLOCKED_EXACT_TYPE_PATHS, ids=_path_id)
def test_blocked_exact_type_scalar_matrix_is_rejected(path: Path) -> None:
    _reset_sentinel_counters()

    payload = closure._blocked()
    original = _get_path(payload, path)
    replacement = _wrong_type_scalar(original)

    assert type(replacement) is not type(original)

    _set_path(payload, path, replacement)

    assert closure._integrity(payload) is False
    _assert_sentinel_counters_zero()


def test_exhaustive_scalar_path_coverage_is_complete() -> None:
    assert set(SOURCE_EXACT_VALUE_PATHS) == set(SOURCE_LEAF_PATHS)
    assert set(SOURCE_EXACT_TYPE_PATHS) == set(SOURCE_LEAF_PATHS)

    assert set(NOMINAL_EXACT_VALUE_PATHS) == set(NOMINAL_LEAF_PATHS)
    assert set(NOMINAL_EXACT_TYPE_PATHS) == set(NOMINAL_LEAF_PATHS)

    assert set(BLOCKED_EXACT_VALUE_PATHS) == set(BLOCKED_LEAF_PATHS)
    assert set(BLOCKED_EXACT_TYPE_PATHS) == set(BLOCKED_LEAF_PATHS)

    assert (
        len(SOURCE_EXACT_VALUE_PATHS)
        + len(SOURCE_EXACT_TYPE_PATHS)
        + len(NOMINAL_EXACT_VALUE_PATHS)
        + len(NOMINAL_EXACT_TYPE_PATHS)
        + len(BLOCKED_EXACT_VALUE_PATHS)
        + len(BLOCKED_EXACT_TYPE_PATHS)
        == 2094
    )


def test_trusted_source_template_matches_real_18_7_builder_once(monkeypatch: Any) -> None:
    calls = 0
    real = read_model.build_preview_block_p_desktop_exe_build_readiness_read_model

    def counted() -> dict[str, Any]:
        nonlocal calls
        calls += 1
        return real()

    monkeypatch.setattr(
        read_model, "build_preview_block_p_desktop_exe_build_readiness_read_model", counted
    )
    source = read_model.build_preview_block_p_desktop_exe_build_readiness_read_model()
    snapshot = copy.deepcopy(source)
    trusted = closure._trusted_source_template()
    assert closure._exact_plain(source, trusted) is True
    assert closure._source_accepted(source) is True
    assert source == snapshot
    assert calls == 1
    assert len(source["readiness_clause_read_rows"]) == 17
    assert len(source["acceptance_rule_read_rows"]) == 6
    summary = source["build_readiness_read_model_summary"]
    assert (
        summary["unique_requirement_count"],
        summary["unique_blocker_count"],
        summary["unique_evidence_count"],
    ) == (8, 12, 12)
    for section in source["capability_read_model_state"].values():
        assert set(section.values()) == {"blocked"}
    assert summary["ready_readiness_count"] == 0
    assert summary["build_authorized"] is False


def test_no_upstream_private_factories_required(monkeypatch: Any) -> None:
    calls = 0

    def boom() -> dict[str, Any]:
        nonlocal calls
        calls += 1
        raise RuntimeError("private factory must not be used")

    monkeypatch.setattr(read_model, "_nominal", boom)
    monkeypatch.setattr(read_model, "_blocked", boom)
    assert closure._trusted_source_template()
    assert closure._nominal()
    assert closure._blocked()
    assert closure._integrity(closure._nominal()) is True
    assert closure._integrity(closure._blocked()) is True
    assert calls == 0


def _assert_path_inventory(
    name: str, payload: dict[str, Any], leaves: list[Path], dicts: list[Path], lists: list[Path]
) -> int:
    assert len(leaves) == len(set(leaves)), name
    assert len(dicts) == len(set(dicts)), name
    assert len(lists) == len(set(lists)), name
    assert () in dicts
    for path in leaves:
        assert type(_get_path(payload, path)) in (str, bool, int, type(None)), _path_id(path)
    for path in dicts:
        assert type(_get_path(payload, path)) is dict, _path_id(path)
    for path in lists:
        assert type(_get_path(payload, path)) is list, _path_id(path)
    return sum(1 for path in leaves if _get_path(payload, path) is None)


def test_path_inventory_counts_and_round_trip() -> None:
    counts = {
        "source": (
            len(SOURCE_LEAF_PATHS),
            len(SOURCE_DICT_PATHS),
            len(SOURCE_LIST_PATHS),
            _assert_path_inventory(
                "source",
                closure._trusted_source_template(),
                SOURCE_LEAF_PATHS,
                SOURCE_DICT_PATHS,
                SOURCE_LIST_PATHS,
            ),
        ),
        "nominal": (
            len(NOMINAL_LEAF_PATHS),
            len(NOMINAL_DICT_PATHS),
            len(NOMINAL_LIST_PATHS),
            _assert_path_inventory(
                "nominal",
                closure._nominal(),
                NOMINAL_LEAF_PATHS,
                NOMINAL_DICT_PATHS,
                NOMINAL_LIST_PATHS,
            ),
        ),
        "blocked": (
            len(BLOCKED_LEAF_PATHS),
            len(BLOCKED_DICT_PATHS),
            len(BLOCKED_LIST_PATHS),
            _assert_path_inventory(
                "blocked",
                closure._blocked(),
                BLOCKED_LEAF_PATHS,
                BLOCKED_DICT_PATHS,
                BLOCKED_LIST_PATHS,
            ),
        ),
    }
    assert counts == {
        "source": (742, 36, 58, 0),
        "nominal": (289, 35, 4, 0),
        "blocked": (16, 1, 2, 0),
    }


def test_closure_schema_completeness_and_list_paths() -> None:
    source = closure._trusted_source_template()
    nominal = closure._nominal()
    blocked = closure._blocked()
    assert len(source["readiness_clause_read_rows"]) == 17
    assert len(source["acceptance_rule_read_rows"]) == 6
    assert len(nominal["stage_audit_rows"]) == 8
    assert len(nominal["closure_findings"]) == 14
    assert type(nominal["capability_audit"]["capability_state"]) is dict
    assert type(nominal["authorization_audit"]) is dict
    assert type(nominal["non_execution_audit"]) is dict
    assert nominal["future_steps"] == []
    assert blocked["stage_audit_rows"] == []
    assert blocked["closure_findings"] == []
    assert ("readiness_clause_read_rows",) in SOURCE_LIST_PATHS
    assert ("acceptance_rule_read_rows",) in SOURCE_LIST_PATHS
    assert ("future_steps",) in SOURCE_LIST_PATHS
    assert ("stage_audit_rows",) in NOMINAL_LIST_PATHS
    assert ("closure_findings",) in NOMINAL_LIST_PATHS
    assert ("future_steps",) in NOMINAL_LIST_PATHS
    assert ("stage_audit_rows",) in BLOCKED_LIST_PATHS
    assert ("closure_findings",) in BLOCKED_LIST_PATHS


def test_representative_bomb_key_paths(monkeypatch: Any) -> None:
    source_cases = [
        ((), "schema_version"),
        (("readiness_clause_read_rows", 0), "readiness_id"),
        (("capability_read_model_state", "contract_capability_state"), "build"),
    ]
    for path, key in source_cases:
        _reset_sentinel_counters()
        source = closure._trusted_source_template()
        _replace_key(_get_path(source, path), key)  # type: ignore[arg-type]
        assert closure._source_accepted(source) is False
        calls = 0

        def wrapper() -> dict[str, Any]:
            nonlocal calls
            calls += 1
            return source

        monkeypatch.setattr(
            closure, "build_preview_block_p_desktop_exe_build_readiness_read_model", wrapper
        )
        payload = closure.build_preview_block_p_closure_audit()
        assert calls == 1
        assert payload == closure._blocked()
        assert closure._integrity(payload) is True
        json.dumps(payload)
        _assert_sentinel_counters_zero()
    for factory, cases in (
        (
            closure._nominal,
            [((), "schema_version"), (("closure_summary",), "source_18_7_accepted")],
        ),
        (closure._blocked, [((), "schema_version")]),
    ):
        for path, key in cases:
            _reset_sentinel_counters()
            payload = factory()
            _replace_key(_get_path(payload, path), key)  # type: ignore[arg-type]
            assert closure._integrity(payload) is False
            _assert_sentinel_counters_zero()


def test_representative_custom_values(monkeypatch: Any) -> None:
    source_paths = [
        ("readiness_clause_read_rows", 0, "source_requirement_ids"),
        ("build_readiness_read_model_summary", "readiness_clause_count"),
        ("capability_read_model_state", "contract_capability_state", "build"),
    ]
    nominal_paths = [("stage_audit_rows", 0, "step"), ("closure_summary", "source_18_7_accepted")]
    blocked_paths = [("status",)]
    for cls in (EqualityBomb, UnhashableEqual, HashBomb):
        for path in source_paths:
            _reset_sentinel_counters()
            source = closure._trusted_source_template()
            _set_path(source, path, cls())
            monkeypatch.setattr(
                closure,
                "build_preview_block_p_desktop_exe_build_readiness_read_model",
                lambda: source,
            )
            assert closure.build_preview_block_p_closure_audit() == closure._blocked()
            _assert_sentinel_counters_zero()
        for path in nominal_paths:
            _reset_sentinel_counters()
            payload = closure._nominal()
            _set_path(payload, path, cls())
            assert closure._integrity(payload) is False
            _assert_sentinel_counters_zero()
        for path in blocked_paths:
            _reset_sentinel_counters()
            payload = closure._blocked()
            _set_path(payload, path, cls())
            assert closure._integrity(payload) is False
            _assert_sentinel_counters_zero()


def test_exact_subclass_rejection(monkeypatch: Any) -> None:
    cases = [
        ("schema_version", StrSubclass("x")),
        (
            "readiness_clause_read_rows",
            ListSubclass(closure._trusted_source_template()["readiness_clause_read_rows"]),
        ),
        (
            "build_readiness_read_model_summary",
            DictSubclass(closure._trusted_source_template()["build_readiness_read_model_summary"]),
        ),
    ]
    for key, value in cases:
        source = closure._trusted_source_template()
        source[key] = value
        monkeypatch.setattr(
            closure, "build_preview_block_p_desktop_exe_build_readiness_read_model", lambda: source
        )
        assert closure.build_preview_block_p_closure_audit() == closure._blocked()
    source_root = DictSubclass(closure._trusted_source_template())
    monkeypatch.setattr(
        closure, "build_preview_block_p_desktop_exe_build_readiness_read_model", lambda: source_root
    )
    assert closure.build_preview_block_p_closure_audit() == closure._blocked()
    for factory in (closure._nominal, closure._blocked):
        for key in (
            "schema_version",
            "stage_audit_rows" if factory is closure._nominal else "closure_findings",
        ):
            payload = factory()
            payload[key] = (
                StrSubclass("x") if key == "schema_version" else ListSubclass(payload[key])
            )
            assert closure._integrity(payload) is False
        payload = factory()
        assert closure._integrity(DictSubclass(payload)) is False


def test_wrong_scalar_helpers() -> None:
    assert _wrong_scalar(True) is False
    assert _wrong_scalar(1) == 2
    assert _wrong_scalar("x") == "x_tampered"
    assert type(_wrong_type_scalar(True)) is int
    assert type(_wrong_type_scalar(1)) is bool
    assert type(_wrong_type_scalar("x")) is StrSubclass
    assert type(_wrong_type_scalar(None)) is EqualityBomb


def _assert_real_source_matches_trusted_after_upstream_restore() -> None:
    real_source = read_model.build_preview_block_p_desktop_exe_build_readiness_read_model()
    assert closure._exact_plain(real_source, closure._trusted_source_template()) is True
    assert closure._source_accepted(real_source) is True


def test_production_does_not_expose_upstream_mutable_schema_aliases() -> None:
    for name in (
        "SOURCE_ROWS",
        "SOURCE_CAPABILITY_FIELDS",
        "SOURCE_CONTRACT_BOUNDARY_FIELDS",
        "SOURCE_TOP_LEVEL_FIELDS",
    ):
        assert not hasattr(closure, name)


def test_trusted_template_isolation_from_mutable_source_rows() -> None:
    trusted_before = closure._trusted_source_template()
    upstream_before = copy.deepcopy(read_model.SOURCE_ROWS)
    try:
        read_model.SOURCE_ROWS[0]["source_requirement_ids"][0] = StrSubclass(
            read_model.SOURCE_ROWS[0]["source_requirement_ids"][0]
        )
        assert closure._exact_plain(closure._trusted_source_template(), trusted_before) is True
    finally:
        read_model.SOURCE_ROWS[:] = upstream_before
    _assert_real_source_matches_trusted_after_upstream_restore()


def test_trusted_template_isolation_from_mutable_capability_fields() -> None:
    trusted_before = closure._trusted_source_template()
    upstream_before = copy.deepcopy(read_model.CAPABILITY_FIELDS)
    try:
        read_model.CAPABILITY_FIELDS.append(StrSubclass("tampered_capability"))
        read_model.CAPABILITY_FIELDS[0], read_model.CAPABILITY_FIELDS[-1] = (
            read_model.CAPABILITY_FIELDS[-1],
            read_model.CAPABILITY_FIELDS[0],
        )
        assert closure._exact_plain(closure._trusted_source_template(), trusted_before) is True
    finally:
        read_model.CAPABILITY_FIELDS[:] = upstream_before
    _assert_real_source_matches_trusted_after_upstream_restore()


def test_trusted_template_isolation_from_mutable_boundary_fields() -> None:
    trusted_before = closure._trusted_source_template()
    upstream_before = copy.deepcopy(read_model.SOURCE_CONTRACT_BOUNDARY_FIELDS)
    try:
        read_model.SOURCE_CONTRACT_BOUNDARY_FIELDS.append(StrSubclass("tampered_boundary"))
        read_model.SOURCE_CONTRACT_BOUNDARY_FIELDS[0] = StrSubclass(
            read_model.SOURCE_CONTRACT_BOUNDARY_FIELDS[0]
        )
        assert closure._exact_plain(closure._trusted_source_template(), trusted_before) is True
    finally:
        read_model.SOURCE_CONTRACT_BOUNDARY_FIELDS[:] = upstream_before
    _assert_real_source_matches_trusted_after_upstream_restore()


def test_trusted_template_isolation_from_mutable_source_top_level_fields() -> None:
    trusted_before = closure._trusted_source_template()
    upstream_before = copy.deepcopy(read_model.SOURCE_TOP_LEVEL_FIELDS)
    try:
        read_model.SOURCE_TOP_LEVEL_FIELDS.append(StrSubclass("tampered_top_level"))
        read_model.SOURCE_TOP_LEVEL_FIELDS[0], read_model.SOURCE_TOP_LEVEL_FIELDS[-1] = (
            read_model.SOURCE_TOP_LEVEL_FIELDS[-1],
            read_model.SOURCE_TOP_LEVEL_FIELDS[0],
        )
        assert closure._exact_plain(closure._trusted_source_template(), trusted_before) is True
    finally:
        read_model.SOURCE_TOP_LEVEL_FIELDS[:] = upstream_before
    _assert_real_source_matches_trusted_after_upstream_restore()


def test_trusted_source_template_returns_independent_fresh_plain_data() -> None:
    first = closure._trusted_source_template()
    second = closure._trusted_source_template()
    assert id(first) != id(second)
    assert id(first["readiness_clause_read_rows"]) != id(second["readiness_clause_read_rows"])
    assert id(first["readiness_clause_read_rows"][0]) != id(second["readiness_clause_read_rows"][0])
    for key in ("source_requirement_ids", "source_blocker_ids", "required_evidence_ids"):
        assert id(first["readiness_clause_read_rows"][0][key]) != id(
            second["readiness_clause_read_rows"][0][key]
        )
    for key in closure.SOURCE_18_7_CAPABILITY_SECTION_FIELDS:
        assert id(first["capability_read_model_state"][key]) != id(
            second["capability_read_model_state"][key]
        )
    assert id(first["read_model_boundaries"]) != id(second["read_model_boundaries"])
    assert id(first["source_boundaries"]) != id(second["source_boundaries"])
    assert id(
        first["block_p_desktop_exe_build_readiness_contract_reference"]["source_top_level_fields"]
    ) != id(
        second["block_p_desktop_exe_build_readiness_contract_reference"]["source_top_level_fields"]
    )
    assert id(first["acceptance_rule_read_rows"][0]["required_contract_clause_ids"]) != id(
        second["acceptance_rule_read_rows"][0]["required_contract_clause_ids"]
    )
    assert id(first["acceptance_rule_read_rows"][1]["required_evidence_ids"]) != id(
        second["acceptance_rule_read_rows"][1]["required_evidence_ids"]
    )
    assert id(first["acceptance_rule_read_rows"][2]["required_blocker_ids"]) != id(
        second["acceptance_rule_read_rows"][2]["required_blocker_ids"]
    )
    specs_before = copy.deepcopy(closure.SOURCE_18_7_ROW_SPECS)
    first["readiness_clause_read_rows"][0]["source_requirement_ids"].append("tampered")
    first["capability_read_model_state"]["contract_capability_state"]["build"] = "tampered"
    first["read_model_boundaries"]["source_only"] = False
    first["block_p_desktop_exe_build_readiness_contract_reference"][
        "source_top_level_fields"
    ].append("tampered")
    first["acceptance_rule_read_rows"][0]["required_contract_clause_ids"].append("tampered")
    first["acceptance_rule_read_rows"][1]["required_evidence_ids"].append("tampered")
    first["acceptance_rule_read_rows"][2]["required_blocker_ids"].append("tampered")
    assert closure._exact_plain(second, closure._trusted_source_template()) is True
    assert specs_before == closure.SOURCE_18_7_ROW_SPECS


def _assert_recursively_immutable_spec(value: object) -> None:
    if type(value) in (str, bool, int, type(None)):
        return
    assert type(value) is tuple
    for item in value:
        _assert_recursively_immutable_spec(item)


def test_source_template_specs_are_recursively_immutable() -> None:
    for spec in (
        closure.SOURCE_18_7_TOP_LEVEL_FIELDS,
        closure.SOURCE_18_7_REQUIRED_TRUE_FIELDS,
        closure.SOURCE_18_7_SUMMARY_ZERO_COUNT_FIELDS,
        closure.SOURCE_18_7_SUMMARY_FALSE_AUTHORIZATION_FIELDS,
        closure.SOURCE_18_7_DECISION_FALSE_FIELDS,
        closure.SOURCE_18_7_REQUIRED_TRUE_BOUNDARY_FIELDS,
        closure.SOURCE_18_7_CAPABILITY_SECTION_FIELDS,
        closure.SOURCE_18_7_ROW_SPECS,
        closure.SOURCE_18_7_CAPABILITY_FIELDS,
        closure.SOURCE_18_6_TOP_LEVEL_FIELDS,
        closure.SOURCE_18_7_SOURCE_BOUNDARY_VALUES,
        closure.SOURCE_18_7_READ_MODEL_BOUNDARY_VALUES,
    ):
        _assert_recursively_immutable_spec(spec)
    assert type(closure.SOURCE_18_7_REQUIREMENT_IDS_INDEX) is int
    assert type(closure.SOURCE_18_7_BLOCKER_IDS_INDEX) is int
    assert type(closure.SOURCE_18_7_EVIDENCE_IDS_INDEX) is int


def test_source_template_has_no_mutable_row_field_lookup_map() -> None:
    assert not hasattr(closure, "SOURCE_18_7_ROW_FIELD_INDEXES")
    assert type(closure.SOURCE_18_7_ROW_SPECS) is tuple
    assert type(closure.SOURCE_18_7_CAPABILITY_FIELDS) is tuple
    assert type(closure.SOURCE_18_6_TOP_LEVEL_FIELDS) is tuple
    assert type(closure.SOURCE_18_7_SOURCE_BOUNDARY_VALUES) is tuple
    assert type(closure.SOURCE_18_7_READ_MODEL_BOUNDARY_VALUES) is tuple
