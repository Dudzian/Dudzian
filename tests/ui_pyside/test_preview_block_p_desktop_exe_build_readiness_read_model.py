from __future__ import annotations

import copy
import json
from collections import Counter
from collections.abc import Callable
from typing import Any, cast

import pytest

from ui.pyside_app import (
    preview_block_p_desktop_exe_build_readiness_contract as source_contract_module,
)
from ui.pyside_app import preview_block_p_desktop_exe_build_readiness_read_model as read_model
from ui.pyside_app.preview_block_p_desktop_exe_build_readiness_contract import (
    build_preview_block_p_desktop_exe_build_readiness_contract,
)


PathPart = str | int
Path = tuple[PathPart, ...]


class HashBomb:
    hash_calls = 0

    def __hash__(self) -> int:
        type(self).hash_calls += 1
        raise RuntimeError("HashBomb hashing must not be called")


class EqualityBomb:
    equality_calls = 0

    def __eq__(self, other: object) -> bool:
        type(self).equality_calls += 1
        raise RuntimeError("EqualityBomb equality must not be called")


class UnhashableEqual:
    __hash__ = None  # type: ignore[assignment]

    def __eq__(self, other: object) -> bool:
        raise RuntimeError("UnhashableEqual equality must not be called")


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


class StrSubclass(str):
    pass


class ListSubclass(list[object]):
    pass


class DictSubclass(dict[str, object]):
    pass


def _reset_bomb_counters() -> None:
    BombKey.equality_calls = 0
    EqualityBomb.equality_calls = 0
    HashBomb.hash_calls = 0


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
            items = list(value.items())
            for key, child in reversed(items):
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
            items = list(value.items())
            for key, child in reversed(items):
                assert type(key) is str
                pending.append((child, path + (key,)))
        elif type(value) is list:
            list_paths.append(path)
            for index in range(len(value) - 1, -1, -1):
                pending.append((value[index], path + (index,)))
    return dict_paths, list_paths


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


def _path_id(path: Path) -> str:
    if not path:
        return "<root>"

    return ".".join(str(part) for part in path)


SOURCE_LEAF_PATHS = _leaf_paths(read_model._trusted_source_template())
NOMINAL_LEAF_PATHS = _leaf_paths(read_model._nominal())
BLOCKED_LEAF_PATHS = _leaf_paths(read_model._blocked())

SOURCE_EXACT_VALUE_PATHS = SOURCE_LEAF_PATHS
SOURCE_EXACT_TYPE_PATHS = SOURCE_LEAF_PATHS

NOMINAL_EXACT_VALUE_PATHS = NOMINAL_LEAF_PATHS
NOMINAL_EXACT_TYPE_PATHS = NOMINAL_LEAF_PATHS

BLOCKED_EXACT_VALUE_PATHS = BLOCKED_LEAF_PATHS
BLOCKED_EXACT_TYPE_PATHS = BLOCKED_LEAF_PATHS

SOURCE_DICT_PATHS, SOURCE_LIST_PATHS = _container_paths(read_model._trusted_source_template())
NOMINAL_DICT_PATHS, NOMINAL_LIST_PATHS = _container_paths(read_model._nominal())
BLOCKED_DICT_PATHS, BLOCKED_LIST_PATHS = _container_paths(read_model._blocked())


def _path_key_id(case: tuple[Path, str]) -> str:
    path, key = case
    return f"{_path_id(path)}::{key}"


def _path_index_id(case: tuple[Path, int]) -> str:
    path, index = case
    return f"{_path_id(path)}::{index}"


def _path_pair_id(case: tuple[Path, int, int]) -> str:
    path, left, right = case
    return f"{_path_id(path)}::{left}<->{right}"


def _dict_missing_cases(
    factory: Callable[[], dict[str, Any]], paths: list[Path]
) -> list[tuple[Path, str]]:
    payload = factory()
    cases: list[tuple[Path, str]] = []
    for path in paths:
        mapping = _get_path(payload, path)
        assert type(mapping) is dict
        for key in mapping:
            assert type(key) is str
            cases.append((path, key))
    return cases


def _list_missing_cases(
    factory: Callable[[], dict[str, Any]], paths: list[Path]
) -> list[tuple[Path, int]]:
    payload = factory()
    cases: list[tuple[Path, int]] = []
    for path in paths:
        values = _get_path(payload, path)
        assert type(values) is list
        cases.extend((path, index) for index in range(len(values)))
    return cases


def _reorderable_dict_paths(factory: Callable[[], dict[str, Any]], paths: list[Path]) -> list[Path]:
    payload = factory()
    result: list[Path] = []
    for path in paths:
        mapping = _get_path(payload, path)
        assert type(mapping) is dict
        if len(mapping) > 1:
            result.append(path)
    return result


def _first_distinct_pair(values: list[object]) -> tuple[int, int] | None:
    for left in range(len(values)):
        for right in range(left + 1, len(values)):
            if read_model._exact_plain(values[left], values[right]) is False:
                return left, right
    return None


def _reorderable_list_cases(
    factory: Callable[[], dict[str, Any]], paths: list[Path]
) -> list[tuple[Path, int, int]]:
    payload = factory()
    cases: list[tuple[Path, int, int]] = []
    for path in paths:
        values = _get_path(payload, path)
        assert type(values) is list
        pair = _first_distinct_pair(values)
        if pair is not None:
            cases.append((path, pair[0], pair[1]))
    return cases


def _nonempty_dict_paths(factory: Callable[[], dict[str, Any]], paths: list[Path]) -> list[Path]:
    payload = factory()
    result: list[Path] = []
    for path in paths:
        mapping = _get_path(payload, path)
        assert type(mapping) is dict
        if mapping:
            result.append(path)
    return result


def _empty_dict_count(factory: Callable[[], dict[str, Any]], paths: list[Path]) -> int:
    payload = factory()
    return sum(1 for path in paths if _get_path(payload, path) == {})


def _empty_list_count(factory: Callable[[], dict[str, Any]], paths: list[Path]) -> int:
    payload = factory()
    return sum(1 for path in paths if _get_path(payload, path) == [])


SOURCE_DICT_MISSING_CASES = _dict_missing_cases(
    read_model._trusted_source_template, SOURCE_DICT_PATHS
)
NOMINAL_DICT_MISSING_CASES = _dict_missing_cases(read_model._nominal, NOMINAL_DICT_PATHS)
BLOCKED_DICT_MISSING_CASES = _dict_missing_cases(read_model._blocked, BLOCKED_DICT_PATHS)
SOURCE_LIST_MISSING_CASES = _list_missing_cases(
    read_model._trusted_source_template, SOURCE_LIST_PATHS
)
NOMINAL_LIST_MISSING_CASES = _list_missing_cases(read_model._nominal, NOMINAL_LIST_PATHS)
BLOCKED_LIST_MISSING_CASES = _list_missing_cases(read_model._blocked, BLOCKED_LIST_PATHS)
SOURCE_REORDERABLE_DICT_PATHS = _reorderable_dict_paths(
    read_model._trusted_source_template, SOURCE_DICT_PATHS
)
NOMINAL_REORDERABLE_DICT_PATHS = _reorderable_dict_paths(read_model._nominal, NOMINAL_DICT_PATHS)
BLOCKED_REORDERABLE_DICT_PATHS = _reorderable_dict_paths(read_model._blocked, BLOCKED_DICT_PATHS)
SOURCE_REORDERABLE_LIST_CASES = _reorderable_list_cases(
    read_model._trusted_source_template, SOURCE_LIST_PATHS
)
NOMINAL_REORDERABLE_LIST_CASES = _reorderable_list_cases(read_model._nominal, NOMINAL_LIST_PATHS)
BLOCKED_REORDERABLE_LIST_CASES = _reorderable_list_cases(read_model._blocked, BLOCKED_LIST_PATHS)
SOURCE_BOMB_KEY_PATHS = _nonempty_dict_paths(read_model._trusted_source_template, SOURCE_DICT_PATHS)
NOMINAL_BOMB_KEY_PATHS = _nonempty_dict_paths(read_model._nominal, NOMINAL_DICT_PATHS)
BLOCKED_BOMB_KEY_PATHS = _nonempty_dict_paths(read_model._blocked, BLOCKED_DICT_PATHS)
EXHAUSTIVE_CONTAINER_MUTATION_CASE_COUNTS = {
    "source dict missing": len(SOURCE_DICT_MISSING_CASES),
    "source dict extra": len(SOURCE_DICT_PATHS),
    "source dict reorder": len(SOURCE_REORDERABLE_DICT_PATHS),
    "source DictSubclass": len(SOURCE_DICT_PATHS),
    "source BombKey": len(SOURCE_BOMB_KEY_PATHS),
    "nominal dict missing": len(NOMINAL_DICT_MISSING_CASES),
    "nominal dict extra": len(NOMINAL_DICT_PATHS),
    "nominal dict reorder": len(NOMINAL_REORDERABLE_DICT_PATHS),
    "nominal DictSubclass": len(NOMINAL_DICT_PATHS),
    "nominal BombKey": len(NOMINAL_BOMB_KEY_PATHS),
    "blocked dict missing": len(BLOCKED_DICT_MISSING_CASES),
    "blocked dict extra": len(BLOCKED_DICT_PATHS),
    "blocked dict reorder": len(BLOCKED_REORDERABLE_DICT_PATHS),
    "blocked DictSubclass": len(BLOCKED_DICT_PATHS),
    "blocked BombKey": len(BLOCKED_BOMB_KEY_PATHS),
    "source list missing": len(SOURCE_LIST_MISSING_CASES),
    "source list extra": len(SOURCE_LIST_PATHS),
    "source list reorder": len(SOURCE_REORDERABLE_LIST_CASES),
    "source ListSubclass": len(SOURCE_LIST_PATHS),
    "nominal list missing": len(NOMINAL_LIST_MISSING_CASES),
    "nominal list extra": len(NOMINAL_LIST_PATHS),
    "nominal list reorder": len(NOMINAL_REORDERABLE_LIST_CASES),
    "nominal ListSubclass": len(NOMINAL_LIST_PATHS),
    "blocked list missing": len(BLOCKED_LIST_MISSING_CASES),
    "blocked list extra": len(BLOCKED_LIST_PATHS),
    "blocked list reorder": len(BLOCKED_REORDERABLE_LIST_CASES),
    "blocked ListSubclass": len(BLOCKED_LIST_PATHS),
}
EXHAUSTIVE_CONTAINER_MUTATION_CASE_TOTAL = sum(EXHAUSTIVE_CONTAINER_MUTATION_CASE_COUNTS.values())
EXTRA_LIST_ITEM = "__unexpected_test_item__"


def test_does_not_call_upstream_private_nominal(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = 0

    def fail_private_nominal() -> dict[str, Any]:
        nonlocal calls
        calls += 1
        raise RuntimeError("18.7 must not call upstream private nominal factory")

    monkeypatch.setattr(source_contract_module, "_nominal", fail_private_nominal)
    trusted = read_model._trusted_source_template()
    nominal = read_model._nominal()
    blocked = read_model._blocked()

    assert trusted["status"] == read_model.SOURCE_STATUS
    assert read_model._integrity(nominal) is True
    assert read_model._integrity(blocked) is True
    assert calls == 0


def test_local_template_matches_real_18_6_builder_once_and_is_immutable() -> None:
    calls = 0
    sources: list[dict[str, Any]] = []

    def real_source_wrapper() -> dict[str, Any]:
        nonlocal calls
        calls += 1
        source = cast(dict[str, Any], build_preview_block_p_desktop_exe_build_readiness_contract())
        sources.append(source)
        return source

    real_source = real_source_wrapper()
    snapshot = copy.deepcopy(real_source)
    trusted = read_model._trusted_source_template()

    assert calls == 1
    assert read_model._exact_plain(real_source, trusted) is True
    assert read_model._source_accepted(real_source) is True
    assert real_source == snapshot


def test_public_builder_uses_real_18_6_builder_once_and_returns_nominal(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = 0
    snapshots: list[dict[str, Any]] = []

    def real_source_wrapper() -> dict[str, Any]:
        nonlocal calls
        calls += 1
        source = cast(dict[str, Any], build_preview_block_p_desktop_exe_build_readiness_contract())
        snapshots.append(copy.deepcopy(source))
        return source

    monkeypatch.setattr(
        read_model,
        "build_preview_block_p_desktop_exe_build_readiness_contract",
        real_source_wrapper,
    )
    payload = read_model.build_preview_block_p_desktop_exe_build_readiness_read_model()

    assert calls == 1
    assert len(snapshots) == 1
    assert read_model._exact_plain(snapshots[0], read_model._trusted_source_template()) is True
    assert read_model._source_accepted(snapshots[0]) is True
    assert payload["status"] == read_model.STATUS
    assert payload["source_18_6_accepted"] is True
    assert read_model._integrity(payload) is True
    assert json.dumps(payload)


def test_counts_preservation_decision_and_zero_authorization() -> None:
    source = build_preview_block_p_desktop_exe_build_readiness_contract()
    payload = read_model._nominal(source)
    summary = payload["build_readiness_read_model_summary"]
    assert len(payload["readiness_clause_read_rows"]) == 17
    assert len(payload["acceptance_rule_read_rows"]) == 6
    assert summary["unique_requirement_count"] == 8
    assert summary["unique_blocker_count"] == 12
    assert summary["unique_evidence_count"] == 12
    for key in (
        "satisfied_readiness_clause_count",
        "observed_readiness_count",
        "validated_readiness_count",
        "satisfied_readiness_count",
        "ready_readiness_count",
        "satisfied_acceptance_rule_count",
    ):
        assert summary[key] == 0
    for key in (
        "build_readiness_contract_satisfied",
        "build_ready",
        "packaging_authorized",
        "build_authorized",
        "artifact_creation_authorized",
        "release_authorized",
        "runtime_authorized",
        "orders_authorized",
    ):
        assert summary[key] is False
    decision = payload["fail_closed_readiness_decision_view"]
    assert decision["contract_satisfied"] is False
    assert decision["build_ready"] is False
    assert decision["only_source_only_18_8_handoff_allowed"] is True
    for key in decision:
        if key.endswith("by_18_6"):
            assert decision[key] is False


def test_rows_and_rules_preserve_source_order_and_links() -> None:
    source = build_preview_block_p_desktop_exe_build_readiness_contract()
    payload = read_model._nominal(source)
    assert [r["readiness_id"] for r in payload["readiness_clause_read_rows"]] == [
        r["readiness_id"] for r in source["build_readiness_contract_rows"]
    ]
    assert [r["contract_clause_id"] for r in payload["readiness_clause_read_rows"]] == [
        r["contract_clause_id"] for r in source["build_readiness_contract_rows"]
    ]
    for out, src in zip(
        payload["readiness_clause_read_rows"], source["build_readiness_contract_rows"], strict=True
    ):
        for key in ("source_requirement_ids", "source_blocker_ids", "required_evidence_ids"):
            assert out[key] == src[key]
        assert out["contract_clause_satisfied"] is False
        assert out["build_authorization_granted"] is False
    assert [r["rule_id"] for r in payload["acceptance_rule_read_rows"]] == [
        r["rule_id"] for r in source["build_readiness_acceptance_rules"]
    ]


def test_capability_maps_all_blocked() -> None:
    payload = read_model._nominal(build_preview_block_p_desktop_exe_build_readiness_contract())
    for mapping in payload["capability_read_model_state"].values():
        assert all(value == "blocked" for value in mapping.values())


@pytest.mark.parametrize(
    "mutate",
    [
        lambda s: s.__setitem__("source_18_5_accepted", False),
        lambda s: s.__setitem__("build_readiness_contract_artifact_complete", False),
        lambda s: s.__setitem__("ready_for_block_p_7", False),
        lambda s: s.__setitem__("integrity_valid", False),
        lambda s: s.__setitem__("status", "wrong"),
        lambda s: s.__setitem__("extra", False),
        lambda s: s.pop("integrity_valid"),
        lambda s: (
            s.clear()
            or s.update(dict(reversed(list(read_model._trusted_source_template().items()))))
        ),
    ],
)
def test_source_tampering_uses_public_builder_and_returns_canonical_blocked(
    mutate: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = 0
    source = copy.deepcopy(read_model._trusted_source_template())
    mutate(source)
    assert read_model._source_accepted(source) is False

    def builder() -> dict[str, Any]:
        nonlocal calls
        calls += 1
        return source

    monkeypatch.setattr(
        read_model, "build_preview_block_p_desktop_exe_build_readiness_contract", builder
    )
    payload = read_model.build_preview_block_p_desktop_exe_build_readiness_read_model()

    assert calls == 1
    assert payload == read_model._blocked()
    assert read_model._integrity(payload) is True
    assert json.dumps(payload)


def test_canonical_blocked_18_6_source_returns_canonical_blocked_18_7(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = 0
    blocked_18_6 = source_contract_module._blocked()
    assert read_model._source_accepted(blocked_18_6) is False

    def builder() -> dict[str, Any]:
        nonlocal calls
        calls += 1
        return blocked_18_6

    monkeypatch.setattr(
        read_model, "build_preview_block_p_desktop_exe_build_readiness_contract", builder
    )
    payload = read_model.build_preview_block_p_desktop_exe_build_readiness_read_model()

    assert calls == 1
    assert payload == read_model._blocked()
    assert read_model._integrity(payload) is True
    assert json.dumps(payload)


def test_canonical_blocked_exact_integrity_and_authorizations() -> None:
    blocked = read_model._blocked()
    assert blocked == read_model._blocked()
    assert read_model._integrity(blocked) is True
    assert json.dumps(blocked)
    extra = copy.deepcopy(blocked)
    extra["extra"] = False
    assert read_model._integrity(extra) is False
    for key in read_model.BLOCKED_FIELDS:
        missing = copy.deepcopy(blocked)
        missing.pop(key)
        assert read_model._integrity(missing) is False
    assert read_model._integrity(dict(reversed(list(blocked.items())))) is False
    for key in (
        "build_ready",
        "packaging_authorized",
        "build_authorized",
        "artifact_creation_authorized",
    ):
        changed = copy.deepcopy(blocked)
        changed[key] = True
        assert read_model._integrity(changed) is False


def test_comparator_cycles_and_deep_graphs() -> None:
    a: list[Any] = []
    b: list[Any] = []
    a.append(a)
    b.append(b)
    assert read_model._exact_plain(a, b) is True
    c: list[Any] = [False]
    c.append(c)
    assert read_model._exact_plain(c, b) is False
    da: dict[str, Any] = {}
    db: dict[str, Any] = {}
    da["self"] = da
    db["self"] = db
    assert read_model._exact_plain(da, db) is True
    dc: dict[str, Any] = {"self": None}
    dc["self"] = dc
    dc["x"] = False
    assert read_model._exact_plain(dc, db) is False
    left: dict[str, Any] = {"value": 0}
    right: dict[str, Any] = {"value": 0}
    for _ in range(1500):
        left = {"next": left}
        right = {"next": right}
    assert read_model._exact_plain(left, right) is True
    mismatch: dict[str, Any] = {"value": 1}
    for _ in range(1500):
        mismatch = {"next": mismatch}
    assert read_model._exact_plain(left, mismatch) is False


def _assert_zero_bomb_counters() -> None:
    assert BombKey.equality_calls == 0
    assert EqualityBomb.equality_calls == 0
    assert HashBomb.hash_calls == 0


def test_path_helpers_replace_without_mutating_canonical_factory_result() -> None:
    canonical = read_model._blocked()
    assert _replace_path(canonical, (), {"root": True}) == {"root": True}
    fresh = read_model._blocked()
    assert fresh == read_model._blocked()
    nested = read_model._nominal()
    path = ("readiness_clause_read_rows", 0, "read_model_row_id")
    original = _get_path(nested, path)
    assert type(original) is str
    assert _replace_path(nested, path, "changed") is nested
    assert _get_path(nested, path) == "changed"
    assert _get_path(read_model._nominal(), path) == original
    with pytest.raises(AssertionError):
        _set_path(nested, (), False)


def _assert_unique_paths(paths: list[Path]) -> None:
    assert len(paths) == len(set(paths))


def _assert_path_inventory(
    root: dict[str, Any],
    leaf_paths: list[Path],
    dict_paths: list[Path],
    list_paths: list[Path],
) -> None:
    _assert_unique_paths(leaf_paths)
    _assert_unique_paths(dict_paths)
    _assert_unique_paths(list_paths)
    assert () in dict_paths
    for path in leaf_paths:
        assert type(_get_path(root, path)) in (str, bool, int, type(None))
    for path in dict_paths:
        assert type(_get_path(root, path)) is dict
    for path in list_paths:
        assert type(_get_path(root, path)) is list


def test_path_enumeration_counts_and_round_trip() -> None:
    source = read_model._trusted_source_template()
    nominal = read_model._nominal()
    blocked = read_model._blocked()
    assert len(SOURCE_LEAF_PATHS) == 739
    assert len(SOURCE_DICT_PATHS) == 37
    assert len(SOURCE_LIST_PATHS) == 57
    assert len(NOMINAL_LEAF_PATHS) == 742
    assert len(NOMINAL_DICT_PATHS) == 36
    assert len(NOMINAL_LIST_PATHS) == 58
    assert len(BLOCKED_LEAF_PATHS) == 14
    assert len(BLOCKED_DICT_PATHS) == 1
    assert len(BLOCKED_LIST_PATHS) == 2
    _assert_path_inventory(source, SOURCE_LEAF_PATHS, SOURCE_DICT_PATHS, SOURCE_LIST_PATHS)
    _assert_path_inventory(nominal, NOMINAL_LEAF_PATHS, NOMINAL_DICT_PATHS, NOMINAL_LIST_PATHS)
    _assert_path_inventory(blocked, BLOCKED_LEAF_PATHS, BLOCKED_DICT_PATHS, BLOCKED_LIST_PATHS)


def _assert_scalar_paths_resolve_to_exact_builtins(
    root: dict[str, Any],
    leaf_paths: list[Path],
) -> int:
    _assert_unique_paths(leaf_paths)
    none_count = 0
    for path in leaf_paths:
        leaf = _get_path(root, path)
        assert type(leaf) in (str, bool, int, type(None))
        if leaf is None:
            none_count += 1
    return none_count


def test_scalar_path_counts_are_canonical() -> None:
    assert len(SOURCE_LEAF_PATHS) == 739
    assert len(NOMINAL_LEAF_PATHS) == 742
    assert len(BLOCKED_LEAF_PATHS) == 14

    assert (
        _assert_scalar_paths_resolve_to_exact_builtins(
            read_model._trusted_source_template(), SOURCE_LEAF_PATHS
        )
        == 0
    )
    assert (
        _assert_scalar_paths_resolve_to_exact_builtins(read_model._nominal(), NOMINAL_LEAF_PATHS)
        == 0
    )
    assert (
        _assert_scalar_paths_resolve_to_exact_builtins(read_model._blocked(), BLOCKED_LEAF_PATHS)
        == 0
    )

    assert all(
        _get_path(read_model._trusted_source_template(), path) is not None
        for path in SOURCE_LEAF_PATHS
    )
    assert all(_get_path(read_model._nominal(), path) is not None for path in NOMINAL_LEAF_PATHS)
    assert all(_get_path(read_model._blocked(), path) is not None for path in BLOCKED_LEAF_PATHS)


def _assert_source_scalar_mutation_blocks(
    monkeypatch: pytest.MonkeyPatch,
    path: Path,
    replacement: object,
) -> None:
    source = copy.deepcopy(read_model._trusted_source_template())
    snapshot = copy.deepcopy(source)

    original = _get_path(source, path)
    _set_path(source, path, replacement)

    assert type(replacement) is not type(original) or replacement != original
    assert read_model._exact_plain(source, snapshot) is False
    assert read_model._source_accepted(source) is False

    calls = 0

    def builder() -> dict[str, Any]:
        nonlocal calls
        calls += 1
        return source

    monkeypatch.setattr(
        read_model,
        "build_preview_block_p_desktop_exe_build_readiness_contract",
        builder,
    )

    payload = read_model.build_preview_block_p_desktop_exe_build_readiness_read_model()

    assert calls == 1
    assert payload == read_model._blocked()
    assert read_model._integrity(payload) is True
    assert json.dumps(payload)


@pytest.mark.parametrize(
    "path",
    SOURCE_EXACT_VALUE_PATHS,
    ids=_path_id,
)
def test_source_exact_value_scalar_matrix_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
    path: Path,
) -> None:
    original = _get_path(read_model._trusted_source_template(), path)

    assert type(original) in (str, bool, int)

    replacement = _wrong_scalar(original)

    assert type(replacement) is type(original)
    assert replacement != original

    _assert_source_scalar_mutation_blocks(
        monkeypatch,
        path,
        replacement,
    )


@pytest.mark.parametrize(
    "path",
    NOMINAL_EXACT_VALUE_PATHS,
    ids=_path_id,
)
def test_nominal_exact_value_scalar_matrix_is_rejected(path: Path) -> None:
    payload = copy.deepcopy(read_model._nominal())
    original = _get_path(payload, path)

    assert type(original) in (str, bool, int)

    replacement = _wrong_scalar(original)
    _set_path(payload, path, replacement)

    assert read_model._integrity(payload) is False


@pytest.mark.parametrize(
    "path",
    BLOCKED_EXACT_VALUE_PATHS,
    ids=_path_id,
)
def test_blocked_exact_value_scalar_matrix_is_rejected(path: Path) -> None:
    payload = copy.deepcopy(read_model._blocked())
    original = _get_path(payload, path)

    assert type(original) in (str, bool, int)

    replacement = _wrong_scalar(original)
    _set_path(payload, path, replacement)

    assert read_model._integrity(payload) is False


@pytest.mark.parametrize(
    "path",
    SOURCE_EXACT_TYPE_PATHS,
    ids=_path_id,
)
def test_source_exact_type_scalar_matrix_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
    path: Path,
) -> None:
    _reset_bomb_counters()

    original = _get_path(read_model._trusted_source_template(), path)
    replacement = _wrong_type_scalar(original)

    assert type(replacement) is not type(original)

    _assert_source_scalar_mutation_blocks(
        monkeypatch,
        path,
        replacement,
    )

    assert BombKey.equality_calls == 0
    assert EqualityBomb.equality_calls == 0
    assert HashBomb.hash_calls == 0


@pytest.mark.parametrize(
    "path",
    NOMINAL_EXACT_TYPE_PATHS,
    ids=_path_id,
)
def test_nominal_exact_type_scalar_matrix_is_rejected(path: Path) -> None:
    _reset_bomb_counters()
    payload = copy.deepcopy(read_model._nominal())
    original = _get_path(payload, path)
    replacement = _wrong_type_scalar(original)

    assert type(replacement) is not type(original)

    _set_path(payload, path, replacement)

    assert read_model._integrity(payload) is False
    assert BombKey.equality_calls == 0
    assert EqualityBomb.equality_calls == 0
    assert HashBomb.hash_calls == 0


@pytest.mark.parametrize(
    "path",
    BLOCKED_EXACT_TYPE_PATHS,
    ids=_path_id,
)
def test_blocked_exact_type_scalar_matrix_is_rejected(path: Path) -> None:
    _reset_bomb_counters()
    payload = copy.deepcopy(read_model._blocked())
    original = _get_path(payload, path)
    replacement = _wrong_type_scalar(original)

    assert type(replacement) is not type(original)

    _set_path(payload, path, replacement)

    assert read_model._integrity(payload) is False
    assert BombKey.equality_calls == 0
    assert EqualityBomb.equality_calls == 0
    assert HashBomb.hash_calls == 0


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
        == 2990
    )


def test_row_rule_and_link_list_path_completeness() -> None:
    source = read_model._trusted_source_template()
    nominal = read_model._nominal()
    assert len(_get_path(source, ("build_readiness_contract_rows",))) == 17
    assert len(_get_path(source, ("build_readiness_acceptance_rules",))) == 6
    assert len(_get_path(nominal, ("readiness_clause_read_rows",))) == 17
    assert len(_get_path(nominal, ("acceptance_rule_read_rows",))) == 6
    for path in (
        ("build_readiness_contract_rows",),
        ("build_readiness_acceptance_rules",),
        ("readiness_clause_read_rows",),
        ("acceptance_rule_read_rows",),
        ("future_steps",),
    ):
        root = source if path[0].startswith("build_") else nominal
        assert path in _container_paths(root)[1]
    source_link_paths = [
        path
        for path in SOURCE_LIST_PATHS
        if path[-1:]
        in (("source_requirement_ids",), ("source_blocker_ids",), ("required_evidence_ids",))
    ]
    nominal_link_paths = [
        path
        for path in NOMINAL_LIST_PATHS
        if path[-1:]
        in (("source_requirement_ids",), ("source_blocker_ids",), ("required_evidence_ids",))
    ]
    assert len(source_link_paths) == 52
    assert len(nominal_link_paths) == 52
    for row_index in range(17):
        for key in ("source_requirement_ids", "source_blocker_ids", "required_evidence_ids"):
            assert ("build_readiness_contract_rows", row_index, key) in source_link_paths
            assert ("readiness_clause_read_rows", row_index, key) in nominal_link_paths


@pytest.mark.parametrize(
    ("path", "key"),
    [
        ((), "schema_version"),
        (("build_readiness_contract_rows", 0), "readiness_id"),
    ],
)
def test_bomb_key_source_fails_closed_without_equality_or_hashing(
    path: Path, key: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    _reset_bomb_counters()
    source = copy.deepcopy(read_model._trusted_source_template())
    mapping = cast(dict[str, Any], _get_path(source, path))
    _replace_key(mapping, key)
    assert read_model._source_accepted(source) is False
    calls = 0

    def builder() -> dict[str, Any]:
        nonlocal calls
        calls += 1
        return source

    monkeypatch.setattr(
        read_model, "build_preview_block_p_desktop_exe_build_readiness_contract", builder
    )
    assert (
        read_model.build_preview_block_p_desktop_exe_build_readiness_read_model()
        == read_model._blocked()
    )
    assert calls == 1
    assert read_model._integrity(read_model._blocked()) is True
    _assert_zero_bomb_counters()


@pytest.mark.parametrize(
    ("factory", "path", "key"),
    [
        (read_model._nominal, (), "schema_version"),
        (read_model._nominal, ("build_readiness_read_model_summary",), "source_18_6_accepted"),
        (read_model._blocked, (), "schema_version"),
    ],
)
def test_bomb_key_integrity_rejects_without_equality_or_hashing(
    factory: Any, path: Path, key: str
) -> None:
    _reset_bomb_counters()
    payload = copy.deepcopy(factory())
    mapping = cast(dict[str, Any], _get_path(payload, path))
    _replace_key(mapping, key)
    assert read_model._integrity(payload) is False
    _assert_zero_bomb_counters()


@pytest.mark.parametrize("bomb", [EqualityBomb(), UnhashableEqual(), HashBomb()])
def test_custom_source_values_fail_closed_without_custom_equality_or_hashing(
    bomb: object, monkeypatch: pytest.MonkeyPatch
) -> None:
    _reset_bomb_counters()
    source = copy.deepcopy(read_model._trusted_source_template())
    _set_path(source, ("build_readiness_contract_rows", 0, "source_requirement_ids", 0), bomb)
    assert read_model._source_accepted(source) is False
    calls = 0

    def builder() -> dict[str, Any]:
        nonlocal calls
        calls += 1
        return source

    monkeypatch.setattr(
        read_model, "build_preview_block_p_desktop_exe_build_readiness_contract", builder
    )
    assert (
        read_model.build_preview_block_p_desktop_exe_build_readiness_read_model()
        == read_model._blocked()
    )
    assert calls == 1
    _assert_zero_bomb_counters()


@pytest.mark.parametrize("bomb", [EqualityBomb(), UnhashableEqual(), HashBomb()])
def test_custom_nominal_values_fail_integrity_without_custom_equality_or_hashing(
    bomb: object,
) -> None:
    _reset_bomb_counters()
    payload = read_model._nominal()
    _set_path(payload, ("readiness_clause_read_rows", 0, "source_requirement_ids", 0), bomb)
    assert read_model._integrity(payload) is False
    _assert_zero_bomb_counters()


@pytest.mark.parametrize(
    ("path", "replacement"),
    [
        (("schema_version",), StrSubclass(read_model.SOURCE_SCHEMA_VERSION)),
        (("build_readiness_contract_rows",), ListSubclass()),
        ((), DictSubclass()),
    ],
)
def test_exact_subclass_source_rejection(
    path: Path, replacement: object, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = copy.deepcopy(read_model._trusted_source_template())
    replacement = DictSubclass(source) if path == () else replacement
    source = cast(dict[str, Any], _replace_path(source, path, replacement))
    assert read_model._source_accepted(source) is False
    monkeypatch.setattr(
        read_model, "build_preview_block_p_desktop_exe_build_readiness_contract", lambda: source
    )
    assert (
        read_model.build_preview_block_p_desktop_exe_build_readiness_read_model()
        == read_model._blocked()
    )


@pytest.mark.parametrize(
    ("factory", "path", "replacement"),
    [
        (read_model._nominal, ("schema_version",), StrSubclass(read_model.SCHEMA_VERSION)),
        (read_model._nominal, ("readiness_clause_read_rows",), ListSubclass()),
        (read_model._nominal, (), DictSubclass()),
        (read_model._blocked, ("schema_version",), StrSubclass(read_model.SCHEMA_VERSION)),
        (read_model._blocked, ("readiness_clause_read_rows",), ListSubclass()),
        (read_model._blocked, (), DictSubclass()),
    ],
)
def test_exact_subclass_integrity_rejection(factory: Any, path: Path, replacement: object) -> None:
    payload = copy.deepcopy(factory())
    replacement = DictSubclass(payload) if path == () else replacement
    payload = cast(dict[str, Any], _replace_path(payload, path, replacement))
    assert read_model._integrity(payload) is False


def test_wrong_scalar_helpers_for_future_scalar_matrix() -> None:
    _reset_bomb_counters()
    assert _wrong_scalar(True) is False
    assert _wrong_scalar(1) == 2
    assert _wrong_scalar("x") == "x_tampered"
    with pytest.raises(AssertionError):
        _wrong_scalar(None)
    assert type(_wrong_type_scalar(True)) is int
    assert type(_wrong_type_scalar(1)) is bool
    assert type(_wrong_type_scalar("x")) is StrSubclass
    assert type(_wrong_type_scalar(None)) is EqualityBomb
    _assert_zero_bomb_counters()


def _assert_source_payload_blocks_once(
    monkeypatch: pytest.MonkeyPatch,
    source: object,
) -> None:
    assert read_model._source_accepted(source) is False
    calls = 0

    def builder() -> object:
        nonlocal calls
        calls += 1
        return source

    monkeypatch.setattr(
        read_model,
        "build_preview_block_p_desktop_exe_build_readiness_contract",
        builder,
    )
    payload = read_model.build_preview_block_p_desktop_exe_build_readiness_read_model()
    assert calls == 1
    assert payload == read_model._blocked()
    assert read_model._integrity(payload) is True
    assert json.dumps(payload)


def _extra_key(mapping: dict[str, object]) -> str:
    candidate = "unexpected_test_field"
    while candidate in mapping:
        candidate += "_x"
    assert type(candidate) is str
    return candidate


def _extra_list_item(values: list[object]) -> str:
    candidate = EXTRA_LIST_ITEM
    while any(read_model._exact_plain(candidate, value) for value in values):
        candidate += "_x"
    return candidate


def _fresh_mutable(factory: Callable[[], dict[str, Any]]) -> dict[str, Any]:
    return copy.deepcopy(factory())


def _mutate_dict_missing(
    factory: Callable[[], dict[str, Any]], path: Path, key: str
) -> dict[str, Any]:
    payload = _fresh_mutable(factory)
    mapping = _get_path(payload, path)
    assert type(mapping) is dict
    assert key in mapping
    original_length = len(mapping)
    del mapping[key]
    assert key not in mapping
    assert len(mapping) == original_length - 1
    return payload


def _mutate_dict_extra(factory: Callable[[], dict[str, Any]], path: Path) -> dict[str, Any]:
    payload = _fresh_mutable(factory)
    mapping = _get_path(payload, path)
    assert type(mapping) is dict
    key = _extra_key(mapping)
    original_length = len(mapping)
    mapping[key] = False
    assert key in mapping
    assert len(mapping) == original_length + 1
    return payload


def _mutate_dict_reorder(factory: Callable[[], dict[str, Any]], path: Path) -> dict[str, Any]:
    payload = _fresh_mutable(factory)
    mapping = _get_path(payload, path)
    assert type(mapping) is dict
    reordered = dict(reversed(list(mapping.items())))
    assert list(reordered) != list(mapping)
    payload = cast(dict[str, Any], _replace_path(payload, path, reordered))
    assert list(cast(dict[str, Any], _get_path(payload, path))) == list(reordered)
    return payload


def _mutate_dict_subclass(factory: Callable[[], dict[str, Any]], path: Path) -> dict[str, Any]:
    payload = _fresh_mutable(factory)
    mapping = _get_path(payload, path)
    assert type(mapping) is dict
    replacement = DictSubclass(mapping)
    assert type(replacement) is DictSubclass
    assert dict(replacement) == mapping
    payload = cast(dict[str, Any], _replace_path(payload, path, replacement))
    assert type(_get_path(payload, path)) is DictSubclass
    return payload


def _mutate_bomb_key(factory: Callable[[], dict[str, Any]], path: Path) -> dict[str, Any]:
    payload = _fresh_mutable(factory)
    mapping = _get_path(payload, path)
    assert type(mapping) is dict
    key = next(iter(mapping))
    assert type(key) is str
    _reset_bomb_counters()
    bomb_key = _replace_key(mapping, key)
    assert type(bomb_key) is BombKey
    return payload


def _mutate_list_missing(
    factory: Callable[[], dict[str, Any]], path: Path, index: int
) -> dict[str, Any]:
    payload = _fresh_mutable(factory)
    values = _get_path(payload, path)
    assert type(values) is list
    original_length = len(values)
    del values[index]
    assert len(values) == original_length - 1
    return payload


def _mutate_list_extra(factory: Callable[[], dict[str, Any]], path: Path) -> dict[str, Any]:
    payload = _fresh_mutable(factory)
    values = _get_path(payload, path)
    assert type(values) is list
    item = _extra_list_item(values)
    original_length = len(values)
    values.append(item)
    assert len(values) == original_length + 1
    assert read_model._exact_plain(values[-1], item) is True
    return payload


def _mutate_list_reorder(
    factory: Callable[[], dict[str, Any]], path: Path, left: int, right: int
) -> dict[str, Any]:
    payload = _fresh_mutable(factory)
    canonical = factory()
    values = _get_path(payload, path)
    canonical_values = _get_path(canonical, path)
    assert type(values) is list
    assert type(canonical_values) is list
    assert read_model._exact_plain(values[left], values[right]) is False
    values[left], values[right] = values[right], values[left]
    assert read_model._exact_plain(values, canonical_values) is False
    return payload


def _mutate_list_subclass(factory: Callable[[], dict[str, Any]], path: Path) -> dict[str, Any]:
    payload = _fresh_mutable(factory)
    values = _get_path(payload, path)
    assert type(values) is list
    replacement = ListSubclass(values)
    assert type(replacement) is ListSubclass
    assert list(replacement) == values
    payload = cast(dict[str, Any], _replace_path(payload, path, replacement))
    assert type(_get_path(payload, path)) is ListSubclass
    return payload


@pytest.mark.parametrize("case", SOURCE_DICT_MISSING_CASES, ids=_path_key_id)
def test_exhaustive_container_source_dict_missing_schema_matrix(
    monkeypatch: pytest.MonkeyPatch, case: tuple[Path, str]
) -> None:
    _assert_source_payload_blocks_once(
        monkeypatch, _mutate_dict_missing(read_model._trusted_source_template, *case)
    )


@pytest.mark.parametrize("case", NOMINAL_DICT_MISSING_CASES, ids=_path_key_id)
def test_exhaustive_container_nominal_dict_missing_schema_matrix(case: tuple[Path, str]) -> None:
    assert read_model._integrity(_mutate_dict_missing(read_model._nominal, *case)) is False


@pytest.mark.parametrize("case", BLOCKED_DICT_MISSING_CASES, ids=_path_key_id)
def test_exhaustive_container_blocked_dict_missing_schema_matrix(case: tuple[Path, str]) -> None:
    assert read_model._integrity(_mutate_dict_missing(read_model._blocked, *case)) is False


@pytest.mark.parametrize("path", SOURCE_DICT_PATHS, ids=_path_id)
def test_exhaustive_container_source_dict_extra_schema_matrix(
    monkeypatch: pytest.MonkeyPatch, path: Path
) -> None:
    _assert_source_payload_blocks_once(
        monkeypatch, _mutate_dict_extra(read_model._trusted_source_template, path)
    )


@pytest.mark.parametrize("path", NOMINAL_DICT_PATHS, ids=_path_id)
def test_exhaustive_container_nominal_dict_extra_schema_matrix(path: Path) -> None:
    assert read_model._integrity(_mutate_dict_extra(read_model._nominal, path)) is False


@pytest.mark.parametrize("path", BLOCKED_DICT_PATHS, ids=_path_id)
def test_exhaustive_container_blocked_dict_extra_schema_matrix(path: Path) -> None:
    assert read_model._integrity(_mutate_dict_extra(read_model._blocked, path)) is False


@pytest.mark.parametrize("path", SOURCE_REORDERABLE_DICT_PATHS, ids=_path_id)
def test_exhaustive_container_source_dict_reorder_schema_matrix(
    monkeypatch: pytest.MonkeyPatch, path: Path
) -> None:
    _assert_source_payload_blocks_once(
        monkeypatch, _mutate_dict_reorder(read_model._trusted_source_template, path)
    )


@pytest.mark.parametrize("path", NOMINAL_REORDERABLE_DICT_PATHS, ids=_path_id)
def test_exhaustive_container_nominal_dict_reorder_schema_matrix(path: Path) -> None:
    assert read_model._integrity(_mutate_dict_reorder(read_model._nominal, path)) is False


@pytest.mark.parametrize("path", BLOCKED_REORDERABLE_DICT_PATHS, ids=_path_id)
def test_exhaustive_container_blocked_dict_reorder_schema_matrix(path: Path) -> None:
    assert read_model._integrity(_mutate_dict_reorder(read_model._blocked, path)) is False


@pytest.mark.parametrize("path", SOURCE_DICT_PATHS, ids=_path_id)
def test_exhaustive_container_source_dict_subclass_schema_matrix(
    monkeypatch: pytest.MonkeyPatch, path: Path
) -> None:
    _assert_source_payload_blocks_once(
        monkeypatch, _mutate_dict_subclass(read_model._trusted_source_template, path)
    )


@pytest.mark.parametrize("path", NOMINAL_DICT_PATHS, ids=_path_id)
def test_exhaustive_container_nominal_dict_subclass_schema_matrix(path: Path) -> None:
    assert read_model._integrity(_mutate_dict_subclass(read_model._nominal, path)) is False


@pytest.mark.parametrize("path", BLOCKED_DICT_PATHS, ids=_path_id)
def test_exhaustive_container_blocked_dict_subclass_schema_matrix(path: Path) -> None:
    assert read_model._integrity(_mutate_dict_subclass(read_model._blocked, path)) is False


@pytest.mark.parametrize("path", SOURCE_BOMB_KEY_PATHS, ids=_path_id)
def test_exhaustive_container_source_bomb_key_schema_matrix(
    monkeypatch: pytest.MonkeyPatch, path: Path
) -> None:
    _assert_source_payload_blocks_once(
        monkeypatch, _mutate_bomb_key(read_model._trusted_source_template, path)
    )
    _assert_zero_bomb_counters()


@pytest.mark.parametrize("path", NOMINAL_BOMB_KEY_PATHS, ids=_path_id)
def test_exhaustive_container_nominal_bomb_key_schema_matrix(path: Path) -> None:
    assert read_model._integrity(_mutate_bomb_key(read_model._nominal, path)) is False
    _assert_zero_bomb_counters()


@pytest.mark.parametrize("path", BLOCKED_BOMB_KEY_PATHS, ids=_path_id)
def test_exhaustive_container_blocked_bomb_key_schema_matrix(path: Path) -> None:
    assert read_model._integrity(_mutate_bomb_key(read_model._blocked, path)) is False
    _assert_zero_bomb_counters()


@pytest.mark.parametrize("case", SOURCE_LIST_MISSING_CASES, ids=_path_index_id)
def test_exhaustive_container_source_list_missing_schema_matrix(
    monkeypatch: pytest.MonkeyPatch, case: tuple[Path, int]
) -> None:
    _assert_source_payload_blocks_once(
        monkeypatch, _mutate_list_missing(read_model._trusted_source_template, *case)
    )


@pytest.mark.parametrize("case", NOMINAL_LIST_MISSING_CASES, ids=_path_index_id)
def test_exhaustive_container_nominal_list_missing_schema_matrix(case: tuple[Path, int]) -> None:
    assert read_model._integrity(_mutate_list_missing(read_model._nominal, *case)) is False


@pytest.mark.parametrize("case", BLOCKED_LIST_MISSING_CASES, ids=_path_index_id)
def test_exhaustive_container_blocked_list_missing_schema_matrix(case: tuple[Path, int]) -> None:
    assert read_model._integrity(_mutate_list_missing(read_model._blocked, *case)) is False


@pytest.mark.parametrize("path", SOURCE_LIST_PATHS, ids=_path_id)
def test_exhaustive_container_source_list_extra_schema_matrix(
    monkeypatch: pytest.MonkeyPatch, path: Path
) -> None:
    _assert_source_payload_blocks_once(
        monkeypatch, _mutate_list_extra(read_model._trusted_source_template, path)
    )


@pytest.mark.parametrize("path", NOMINAL_LIST_PATHS, ids=_path_id)
def test_exhaustive_container_nominal_list_extra_schema_matrix(path: Path) -> None:
    assert read_model._integrity(_mutate_list_extra(read_model._nominal, path)) is False


@pytest.mark.parametrize("path", BLOCKED_LIST_PATHS, ids=_path_id)
def test_exhaustive_container_blocked_list_extra_schema_matrix(path: Path) -> None:
    assert read_model._integrity(_mutate_list_extra(read_model._blocked, path)) is False


@pytest.mark.parametrize("case", SOURCE_REORDERABLE_LIST_CASES, ids=_path_pair_id)
def test_exhaustive_container_source_list_reorder_schema_matrix(
    monkeypatch: pytest.MonkeyPatch, case: tuple[Path, int, int]
) -> None:
    _assert_source_payload_blocks_once(
        monkeypatch, _mutate_list_reorder(read_model._trusted_source_template, *case)
    )


@pytest.mark.parametrize("case", NOMINAL_REORDERABLE_LIST_CASES, ids=_path_pair_id)
def test_exhaustive_container_nominal_list_reorder_schema_matrix(
    case: tuple[Path, int, int],
) -> None:
    assert read_model._integrity(_mutate_list_reorder(read_model._nominal, *case)) is False


@pytest.mark.parametrize("case", BLOCKED_REORDERABLE_LIST_CASES, ids=_path_pair_id)
def test_exhaustive_container_blocked_list_reorder_schema_matrix(
    case: tuple[Path, int, int],
) -> None:
    assert read_model._integrity(_mutate_list_reorder(read_model._blocked, *case)) is False


@pytest.mark.parametrize("path", SOURCE_LIST_PATHS, ids=_path_id)
def test_exhaustive_container_source_list_subclass_schema_matrix(
    monkeypatch: pytest.MonkeyPatch, path: Path
) -> None:
    _assert_source_payload_blocks_once(
        monkeypatch, _mutate_list_subclass(read_model._trusted_source_template, path)
    )


@pytest.mark.parametrize("path", NOMINAL_LIST_PATHS, ids=_path_id)
def test_exhaustive_container_nominal_list_subclass_schema_matrix(path: Path) -> None:
    assert read_model._integrity(_mutate_list_subclass(read_model._nominal, path)) is False


@pytest.mark.parametrize("path", BLOCKED_LIST_PATHS, ids=_path_id)
def test_exhaustive_container_blocked_list_subclass_schema_matrix(path: Path) -> None:
    assert read_model._integrity(_mutate_list_subclass(read_model._blocked, path)) is False


def _field_counter(
    factory: Callable[[], dict[str, Any]], paths: list[Path]
) -> Counter[tuple[Path, str]]:
    payload = factory()
    result: Counter[tuple[Path, str]] = Counter()
    for path in paths:
        mapping = _get_path(payload, path)
        assert type(mapping) is dict
        for key in mapping:
            assert type(key) is str
            result[(path, key)] += 1
    return result


def _item_counter(
    factory: Callable[[], dict[str, Any]], paths: list[Path]
) -> Counter[tuple[Path, int]]:
    payload = factory()
    result: Counter[tuple[Path, int]] = Counter()
    for path in paths:
        values = _get_path(payload, path)
        assert type(values) is list
        for index in range(len(values)):
            result[(path, index)] += 1
    return result


def test_exhaustive_container_path_coverage_is_complete() -> None:
    assert Counter(SOURCE_DICT_PATHS) == Counter(
        _container_paths(read_model._trusted_source_template())[0]
    )
    assert Counter(NOMINAL_DICT_PATHS) == Counter(_container_paths(read_model._nominal())[0])
    assert Counter(BLOCKED_DICT_PATHS) == Counter(_container_paths(read_model._blocked())[0])
    assert Counter(SOURCE_LIST_PATHS) == Counter(
        _container_paths(read_model._trusted_source_template())[1]
    )
    assert Counter(NOMINAL_LIST_PATHS) == Counter(_container_paths(read_model._nominal())[1])
    assert Counter(BLOCKED_LIST_PATHS) == Counter(_container_paths(read_model._blocked())[1])
    assert Counter(SOURCE_BOMB_KEY_PATHS) == Counter(
        _nonempty_dict_paths(read_model._trusted_source_template, SOURCE_DICT_PATHS)
    )
    assert Counter(NOMINAL_BOMB_KEY_PATHS) == Counter(
        _nonempty_dict_paths(read_model._nominal, NOMINAL_DICT_PATHS)
    )
    assert Counter(BLOCKED_BOMB_KEY_PATHS) == Counter(
        _nonempty_dict_paths(read_model._blocked, BLOCKED_DICT_PATHS)
    )
    assert Counter(SOURCE_REORDERABLE_DICT_PATHS) == Counter(
        _reorderable_dict_paths(read_model._trusted_source_template, SOURCE_DICT_PATHS)
    )
    assert Counter(NOMINAL_REORDERABLE_DICT_PATHS) == Counter(
        _reorderable_dict_paths(read_model._nominal, NOMINAL_DICT_PATHS)
    )
    assert Counter(BLOCKED_REORDERABLE_DICT_PATHS) == Counter(
        _reorderable_dict_paths(read_model._blocked, BLOCKED_DICT_PATHS)
    )
    assert Counter(SOURCE_DICT_MISSING_CASES) == _field_counter(
        read_model._trusted_source_template, SOURCE_DICT_PATHS
    )
    assert Counter(NOMINAL_DICT_MISSING_CASES) == _field_counter(
        read_model._nominal, NOMINAL_DICT_PATHS
    )
    assert Counter(BLOCKED_DICT_MISSING_CASES) == _field_counter(
        read_model._blocked, BLOCKED_DICT_PATHS
    )
    assert Counter(SOURCE_REORDERABLE_LIST_CASES) == Counter(
        _reorderable_list_cases(read_model._trusted_source_template, SOURCE_LIST_PATHS)
    )
    assert Counter(NOMINAL_REORDERABLE_LIST_CASES) == Counter(
        _reorderable_list_cases(read_model._nominal, NOMINAL_LIST_PATHS)
    )
    assert Counter(BLOCKED_REORDERABLE_LIST_CASES) == Counter(
        _reorderable_list_cases(read_model._blocked, BLOCKED_LIST_PATHS)
    )
    assert Counter(SOURCE_LIST_MISSING_CASES) == _item_counter(
        read_model._trusted_source_template, SOURCE_LIST_PATHS
    )
    assert Counter(NOMINAL_LIST_MISSING_CASES) == _item_counter(
        read_model._nominal, NOMINAL_LIST_PATHS
    )
    assert Counter(BLOCKED_LIST_MISSING_CASES) == _item_counter(
        read_model._blocked, BLOCKED_LIST_PATHS
    )


def test_exhaustive_container_path_counts_and_case_report() -> None:
    assert len(SOURCE_DICT_PATHS) == 37
    assert len(SOURCE_LIST_PATHS) == 57
    assert len(NOMINAL_DICT_PATHS) == 36
    assert len(NOMINAL_LIST_PATHS) == 58
    assert len(BLOCKED_DICT_PATHS) == 1
    assert len(BLOCKED_LIST_PATHS) == 2
    assert _empty_dict_count(read_model._trusted_source_template, SOURCE_DICT_PATHS) == 0
    assert _empty_dict_count(read_model._nominal, NOMINAL_DICT_PATHS) == 0
    assert _empty_dict_count(read_model._blocked, BLOCKED_DICT_PATHS) == 0
    assert _empty_list_count(read_model._trusted_source_template, SOURCE_LIST_PATHS) == 0
    assert _empty_list_count(read_model._nominal, NOMINAL_LIST_PATHS) == 0
    assert _empty_list_count(read_model._blocked, BLOCKED_LIST_PATHS) == 2
    assert len(SOURCE_REORDERABLE_DICT_PATHS) == 37
    assert len(SOURCE_DICT_PATHS) - len(SOURCE_REORDERABLE_DICT_PATHS) == 0
    assert len(NOMINAL_REORDERABLE_DICT_PATHS) == 36
    assert len(NOMINAL_DICT_PATHS) - len(NOMINAL_REORDERABLE_DICT_PATHS) == 0
    assert len(BLOCKED_REORDERABLE_DICT_PATHS) == 1
    assert len(BLOCKED_DICT_PATHS) - len(BLOCKED_REORDERABLE_DICT_PATHS) == 0
    assert len(SOURCE_REORDERABLE_LIST_CASES) == 6
    assert len(SOURCE_LIST_PATHS) - len(SOURCE_REORDERABLE_LIST_CASES) == 51
    assert len(NOMINAL_REORDERABLE_LIST_CASES) == 6
    assert len(NOMINAL_LIST_PATHS) - len(NOMINAL_REORDERABLE_LIST_CASES) == 52
    assert len(BLOCKED_REORDERABLE_LIST_CASES) == 0
    assert len(BLOCKED_LIST_PATHS) - len(BLOCKED_REORDERABLE_LIST_CASES) == 2
    expected = {
        "source dict missing": 715,
        "source dict extra": 37,
        "source dict reorder": 37,
        "source DictSubclass": 37,
        "source BombKey": 37,
        "nominal dict missing": 694,
        "nominal dict extra": 36,
        "nominal dict reorder": 36,
        "nominal DictSubclass": 36,
        "nominal BombKey": 36,
        "blocked dict missing": 16,
        "blocked dict extra": 1,
        "blocked dict reorder": 1,
        "blocked DictSubclass": 1,
        "blocked BombKey": 1,
        "source list missing": 117,
        "source list extra": 57,
        "source list reorder": 6,
        "source ListSubclass": 57,
        "nominal list missing": 141,
        "nominal list extra": 58,
        "nominal list reorder": 6,
        "nominal ListSubclass": 58,
        "blocked list missing": 0,
        "blocked list extra": 2,
        "blocked list reorder": 0,
        "blocked ListSubclass": 2,
    }
    assert EXHAUSTIVE_CONTAINER_MUTATION_CASE_COUNTS == expected
    assert EXHAUSTIVE_CONTAINER_MUTATION_CASE_TOTAL == sum(expected.values()) == 2225
