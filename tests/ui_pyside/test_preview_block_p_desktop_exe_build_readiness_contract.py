from __future__ import annotations

import copy
import json
from collections import Counter
from collections.abc import Callable
from typing import Any, cast

import pytest

from ui.pyside_app import preview_block_p_desktop_exe_build_readiness_matrix as matrix
import ui.pyside_app.preview_block_p_desktop_exe_build_readiness_contract as contract
from ui.pyside_app.preview_block_p_desktop_exe_build_readiness_matrix import (
    build_preview_block_p_desktop_exe_build_readiness_matrix,
)

EXPECTED_DECISION_FIELDS = [
    "source_matrix_exists_and_accepted",
    "contract_definitions_complete",
    "contract_satisfied",
    "no_readiness_row_satisfied",
    "no_blocker_resolved",
    "no_evidence_collected",
    "no_evidence_validated",
    "build_ready",
    "packaging_authorized",
    "build_authorized",
    "artifact_creation_authorized",
    "release_authorized",
    "runtime_enabled",
    "orders_enabled",
    "only_source_only_18_7_handoff_allowed",
    "build_performed_by_18_6",
    "packaging_authorized_by_18_6",
    "build_authorized_by_18_6",
    "artifact_creation_authorized_by_18_6",
    "release_performed_by_18_6",
    "runtime_enabled_by_18_6",
    "orders_enabled_by_18_6",
]
EXPECTED_CONTRACT_BOUNDARY_FIELDS = [
    "reads_18_5_only",
    "source_only",
    "plain_data",
    "static_contract",
    "can_feed_only_18_7_build_readiness_read_model",
    "repo_rescan",
    "filesystem_scan",
    "environment_scan",
    "secret_file_read",
    "dependency_import",
    "dependency_resolution",
    "pyside_import",
    "qml_load",
    "qt_plugin_discovery",
    "entrypoint_selection",
    "entrypoint_validation",
    "evidence_collection",
    "evidence_validation",
    "blocker_resolution",
    "readiness_approval",
    "build_gate_approval",
    "build_readiness_grant",
    "packaging_authorization",
    "build_authorization",
    "spec_file_creation",
    "build_command_creation",
    "build_command_execution",
    "packaging_performed",
    "artifact_created",
    "artifact_scanned",
    "artifact_signed",
    "installer_created",
    "release_performed",
    "runtime_started",
    "orders_enabled",
    "network_opened",
    "credentials_read",
    "qml_bridge_gateway_controller_changed",
]
BY_18_6_FIELDS = [
    "build_performed_by_18_6",
    "packaging_authorized_by_18_6",
    "build_authorized_by_18_6",
    "artifact_creation_authorized_by_18_6",
    "release_performed_by_18_6",
    "runtime_enabled_by_18_6",
    "orders_enabled_by_18_6",
]


class HashBomb:
    hash_calls = 0

    def __hash__(self) -> int:
        type(self).hash_calls += 1
        raise RuntimeError


def test_real_18_5_builder_is_accepted(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = 0
    sources: list[dict[str, Any]] = []
    snapshots: list[dict[str, Any]] = []

    def builder() -> dict[str, Any]:
        nonlocal calls
        calls += 1
        source = cast(dict[str, Any], build_preview_block_p_desktop_exe_build_readiness_matrix())
        sources.append(source)
        snapshots.append(copy.deepcopy(source))
        return source

    monkeypatch.setattr(matrix, "READINESS_ROWS", copy.deepcopy(contract.SOURCE_ROWS))
    monkeypatch.setattr(
        contract, "build_preview_block_p_desktop_exe_build_readiness_matrix", builder
    )
    payload = contract.build_preview_block_p_desktop_exe_build_readiness_contract()

    assert calls == 1
    assert len(sources) == 1
    assert len(snapshots) == 1
    assert sources[0] == snapshots[0]
    assert contract._source_accepted(sources[0]) is True
    assert contract._integrity(payload) is True
    assert json.dumps(payload)
    assert payload["status"] == contract.STATUS
    assert payload["source_18_5_accepted"] is True
    assert payload["build_readiness_contract_artifact_complete"] is True
    assert payload["ready_for_block_p_7"] is True
    assert len(payload["build_readiness_contract_rows"]) == 17
    assert len(payload["build_readiness_acceptance_rules"]) == 6

    summary = payload["build_readiness_contract_summary"]
    assert summary["unique_requirement_count"] == 8
    assert summary["unique_blocker_count"] == 12
    assert summary["unique_evidence_count"] == 12
    assert summary["readiness_clause_count"] == 17
    assert summary["defined_readiness_clause_count"] == 17
    assert summary["satisfied_readiness_clause_count"] == 0
    assert summary["observed_readiness_count"] == 0
    assert summary["validated_readiness_count"] == 0
    assert summary["satisfied_readiness_count"] == 0
    assert summary["ready_readiness_count"] == 0
    assert summary["acceptance_rule_count"] == 6
    assert summary["satisfied_acceptance_rule_count"] == 0
    assert summary["packaging_authorized"] is False
    assert summary["build_authorized"] is False
    assert summary["artifact_creation_authorized"] is False
    assert summary["release_authorized"] is False
    assert summary["runtime_authorized"] is False
    assert summary["orders_authorized"] is False

    for row in payload["build_readiness_contract_rows"]:
        assert row["contract_clause_defined"] is True
        assert row["contract_clause_satisfied"] is False
        assert row["build_readiness_granted"] is False
        assert row["packaging_authorization_granted"] is False
        assert row["build_authorization_granted"] is False
        assert row["artifact_creation_authorization_granted"] is False

    capability_state = payload["capability_contract_state"]
    assert set(capability_state["source_capability_build_readiness_state"].values()) == {"blocked"}
    assert set(capability_state["contract_capability_state"].values()) == {"blocked"}


def test_nominal_decision_and_boundaries_have_canonical_order() -> None:
    payload = contract._nominal()
    decision = payload["fail_closed_build_readiness_contract_decision"]
    boundaries = payload["contract_boundaries"]

    assert list(decision) == EXPECTED_DECISION_FIELDS
    assert len(decision) == 22
    assert list(boundaries) == EXPECTED_CONTRACT_BOUNDARY_FIELDS
    assert len(boundaries) == 38
    for field in EXPECTED_CONTRACT_BOUNDARY_FIELDS[:5]:
        assert boundaries[field] is True
    for field in EXPECTED_CONTRACT_BOUNDARY_FIELDS[5:]:
        assert boundaries[field] is False
    for field in BY_18_6_FIELDS:
        assert decision[field] is False


def test_hash_bomb_fails_closed(monkeypatch: pytest.MonkeyPatch) -> None:
    source = copy.deepcopy(contract._source_template())
    source["readiness_rows"][0]["source_requirement_ids"][0] = HashBomb()
    HashBomb.hash_calls = 0
    assert contract._source_accepted(source) is False
    monkeypatch.setattr(
        contract, "build_preview_block_p_desktop_exe_build_readiness_matrix", lambda: source
    )
    payload = contract.build_preview_block_p_desktop_exe_build_readiness_contract()
    assert payload == contract._blocked()
    assert HashBomb.hash_calls == 0


def test_blocked_is_exact() -> None:
    blocked = contract._blocked()
    assert blocked == contract._blocked()
    assert contract._integrity(blocked) is True
    assert json.dumps(blocked)

    reordered = dict(reversed(list(blocked.items())))
    assert contract._integrity(reordered) is False

    blocked["extra"] = False
    assert contract._integrity(blocked) is False


@pytest.mark.parametrize(
    "mutate",
    [
        lambda source: source.__setitem__("ready_for_build_execution", True),
        lambda source: source.__setitem__("integrity_valid", False),
        lambda source: source.__setitem__("source_18_4_accepted", False),
        lambda source: source.__setitem__("status", "wrong"),
        lambda source: source.__setitem__("unexpected_top_level_field", False),
        lambda source: source.pop("integrity_valid"),
        lambda source: (
            source.clear()
            or source.update(dict(reversed(list(contract._source_template().items()))))
        ),
    ],
)
def test_source_top_level_tampering_returns_canonical_blocked(
    monkeypatch: pytest.MonkeyPatch, mutate: Any
) -> None:
    source = copy.deepcopy(contract._source_template())
    mutate(source)
    monkeypatch.setattr(
        contract, "build_preview_block_p_desktop_exe_build_readiness_matrix", lambda: source
    )
    payload = contract.build_preview_block_p_desktop_exe_build_readiness_contract()
    assert payload == contract._blocked()
    assert contract._integrity(payload) is True


def test_exact_plain_accepts_matching_cyclic_lists() -> None:
    actual: list[object] = []
    trusted: list[object] = []
    actual.append(actual)
    trusted.append(trusted)
    assert contract._exact_plain(actual, trusted) is True


def test_exact_plain_rejects_different_cyclic_lists() -> None:
    actual: list[object] = []
    trusted: list[object] = []
    actual.append(actual)
    trusted.extend([trusted, "extra"])
    assert contract._exact_plain(actual, trusted) is False


def test_exact_plain_accepts_matching_cyclic_dicts() -> None:
    actual: dict[str, Any] = {}
    trusted: dict[str, Any] = {}
    actual["self"] = actual
    trusted["self"] = trusted
    assert contract._exact_plain(actual, trusted) is True


def test_exact_plain_rejects_different_cyclic_dicts() -> None:
    actual: dict[str, Any] = {}
    trusted: dict[str, Any] = {}
    actual["self"] = actual
    trusted["self"] = trusted
    trusted["extra"] = False
    assert contract._exact_plain(actual, trusted) is False


def test_blocked_rejects_all_top_level_mutations() -> None:
    blocked = contract._blocked()
    assert blocked["source_18_5_accepted"] is False
    for field in blocked:
        payload = copy.deepcopy(blocked)
        payload.pop(field)
        assert contract._integrity(payload) is False
    for field in (
        "source_18_5_accepted",
        "build_readiness_contract_artifact_complete",
        "ready_for_block_p_7",
        "build_ready",
        "packaging_authorized",
        "build_authorized",
        "artifact_creation_authorized",
        "integrity_valid",
    ):
        payload = copy.deepcopy(blocked)
        payload[field] = not payload[field]
        assert contract._integrity(payload) is False


def test_main_sections_reject_missing_extra_and_reordered_fields() -> None:
    sections = [
        "block_p_desktop_exe_build_readiness_matrix_reference",
        "source_matrix_preservation",
        "build_readiness_contract_summary",
        "build_readiness_contract_principles",
        "capability_contract_state",
        "fail_closed_build_readiness_contract_decision",
        "non_execution_contract_evidence",
        "contract_boundaries",
        "source_boundaries",
    ]
    for name in sections:
        payload = copy.deepcopy(contract._nominal())
        section = payload[name]
        section.pop(next(iter(section)))
        assert contract._integrity(payload) is False
        payload = copy.deepcopy(contract._nominal())
        payload[name]["unexpected_test_field"] = False
        assert contract._integrity(payload) is False
        payload = copy.deepcopy(contract._nominal())
        payload[name] = dict(reversed(list(payload[name].items())))
        assert contract._integrity(payload) is False


PathPart = str | int
Path = tuple[PathPart, ...]


class BombKey(str):
    equality_calls = 0
    armed: bool

    def __new__(cls, value: str) -> BombKey:
        instance = super().__new__(cls, value)
        instance.armed = False
        return instance

    def __eq__(self, other: object) -> bool:
        if self.armed:
            type(self).equality_calls += 1
            raise RuntimeError("BombKey equality must not be called")
        return super().__eq__(other)

    __hash__ = str.__hash__


class EqualityBomb:
    equality_calls = 0

    def __eq__(self, other: object) -> bool:
        type(self).equality_calls += 1
        raise RuntimeError("EqualityBomb equality must not be called")


class UnhashableEqual:
    __hash__: Any = None

    def __eq__(self, other: object) -> bool:
        raise RuntimeError("UnhashableEqual equality must not be called")


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


def _wrong_scalar(value: object) -> object:
    result: object
    if type(value) is bool:
        result = not value
    elif type(value) is int:
        result = value + 1
    elif type(value) is str:
        result = value + "_tampered"
    else:
        raise AssertionError("None requires a dedicated mutation matrix")
    assert type(result) is type(value)
    assert result != value
    return result


def _wrong_type_scalar(value: object) -> object:
    if type(value) is bool:
        result: object = 0
    elif type(value) is int:
        result = True
    elif type(value) is str:
        result = StrSubclass(value + "_tampered")
    else:
        result = EqualityBomb()
    assert type(result) is not type(value)
    return result


SOURCE_LEAF_PATHS = _leaf_paths(contract._source_template())
NOMINAL_LEAF_PATHS = _leaf_paths(contract._nominal())
BLOCKED_LEAF_PATHS = _leaf_paths(contract._blocked())
SOURCE_EXACT_VALUE_PATHS = SOURCE_LEAF_PATHS
SOURCE_EXACT_TYPE_PATHS = SOURCE_LEAF_PATHS
NOMINAL_EXACT_VALUE_PATHS = NOMINAL_LEAF_PATHS
NOMINAL_EXACT_TYPE_PATHS = NOMINAL_LEAF_PATHS
BLOCKED_EXACT_VALUE_PATHS = BLOCKED_LEAF_PATHS
BLOCKED_EXACT_TYPE_PATHS = BLOCKED_LEAF_PATHS
SOURCE_DICT_PATHS, SOURCE_LIST_PATHS = _container_paths(contract._source_template())
NOMINAL_DICT_PATHS, NOMINAL_LIST_PATHS = _container_paths(contract._nominal())
BLOCKED_DICT_PATHS, BLOCKED_LIST_PATHS = _container_paths(contract._blocked())


def _replace_path(root: object, path: Path, replacement: object) -> object:
    if not path:
        return replacement
    _set_path(root, path, replacement)
    return root


def _path_id(path: Path) -> str:
    if not path:
        return "<root>"
    return ".".join(str(part) for part in path)


def _assert_source_payload_blocks_once(
    monkeypatch: pytest.MonkeyPatch,
    source: object,
) -> None:
    assert contract._source_accepted(source) is False

    calls = 0

    def builder() -> object:
        nonlocal calls
        calls += 1
        return source

    monkeypatch.setattr(
        contract,
        "build_preview_block_p_desktop_exe_build_readiness_matrix",
        builder,
    )
    payload = contract.build_preview_block_p_desktop_exe_build_readiness_contract()

    assert calls == 1
    assert payload == contract._blocked()
    assert contract._integrity(payload) is True
    assert json.dumps(payload)


def _assert_source_scalar_mutation_blocks(
    monkeypatch: pytest.MonkeyPatch,
    path: Path,
    replacement: object,
) -> None:
    source = copy.deepcopy(contract._source_template())
    snapshot = copy.deepcopy(source)
    _set_path(source, path, replacement)

    assert source != snapshot or contract._exact_plain(source, snapshot) is False
    _assert_source_payload_blocks_once(monkeypatch, source)


def _replace_key(mapping: dict[str, Any], key: str) -> BombKey:
    bomb_key = BombKey(key)
    items = list(mapping.items())
    mapping.clear()
    for item_key, value in items:
        mapping[bomb_key if item_key == key else item_key] = value
    bomb_key.armed = True
    return bomb_key


def _assert_source_blocks(monkeypatch: pytest.MonkeyPatch, source: object) -> dict[str, Any]:
    monkeypatch.setattr(
        contract, "build_preview_block_p_desktop_exe_build_readiness_matrix", lambda: source
    )
    payload = contract.build_preview_block_p_desktop_exe_build_readiness_contract()
    assert contract._exact_plain(payload, contract._blocked()) is True
    assert contract._integrity(payload) is True
    return cast(dict[str, Any], payload)


def _dict_missing_cases(
    factory: Callable[[], dict[str, Any]],
    paths: list[Path],
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
            if contract._exact_plain(values[left], values[right]) is False:
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


EXTRA_LIST_ITEM = "__unexpected_test_item__"


def _extra_key(mapping: dict[str, object]) -> str:
    candidate = "unexpected_test_field"
    while candidate in mapping:
        candidate += "_x"
    assert type(candidate) is str
    return candidate


def _extra_list_item(values: list[object]) -> str:
    candidate = EXTRA_LIST_ITEM
    while any(contract._exact_plain(candidate, value) for value in values):
        candidate += "_x"
    return candidate


SOURCE_DICT_MISSING_CASES = _dict_missing_cases(contract._source_template, SOURCE_DICT_PATHS)
NOMINAL_DICT_MISSING_CASES = _dict_missing_cases(contract._nominal, NOMINAL_DICT_PATHS)
BLOCKED_DICT_MISSING_CASES = _dict_missing_cases(contract._blocked, BLOCKED_DICT_PATHS)
SOURCE_LIST_MISSING_CASES = _list_missing_cases(contract._source_template, SOURCE_LIST_PATHS)
NOMINAL_LIST_MISSING_CASES = _list_missing_cases(contract._nominal, NOMINAL_LIST_PATHS)
BLOCKED_LIST_MISSING_CASES = _list_missing_cases(contract._blocked, BLOCKED_LIST_PATHS)
SOURCE_REORDERABLE_DICT_PATHS = _reorderable_dict_paths(
    contract._source_template, SOURCE_DICT_PATHS
)
NOMINAL_REORDERABLE_DICT_PATHS = _reorderable_dict_paths(contract._nominal, NOMINAL_DICT_PATHS)
BLOCKED_REORDERABLE_DICT_PATHS = _reorderable_dict_paths(contract._blocked, BLOCKED_DICT_PATHS)
SOURCE_REORDERABLE_LIST_CASES = _reorderable_list_cases(
    contract._source_template, SOURCE_LIST_PATHS
)
NOMINAL_REORDERABLE_LIST_CASES = _reorderable_list_cases(contract._nominal, NOMINAL_LIST_PATHS)
BLOCKED_REORDERABLE_LIST_CASES = _reorderable_list_cases(contract._blocked, BLOCKED_LIST_PATHS)
SOURCE_BOMB_KEY_PATHS = _nonempty_dict_paths(contract._source_template, SOURCE_DICT_PATHS)
NOMINAL_BOMB_KEY_PATHS = _nonempty_dict_paths(contract._nominal, NOMINAL_DICT_PATHS)
BLOCKED_BOMB_KEY_PATHS = _nonempty_dict_paths(contract._blocked, BLOCKED_DICT_PATHS)
EXHAUSTIVE_CONTAINER_MUTATION_CASE_COUNTS = {
    "source dict missing-field cases": len(SOURCE_DICT_MISSING_CASES),
    "source dict extra-field cases": len(SOURCE_DICT_PATHS),
    "source dict reorder cases": len(SOURCE_REORDERABLE_DICT_PATHS),
    "source DictSubclass cases": len(SOURCE_DICT_PATHS),
    "source BombKey cases": len(SOURCE_BOMB_KEY_PATHS),
    "nominal dict missing-field cases": len(NOMINAL_DICT_MISSING_CASES),
    "nominal dict extra-field cases": len(NOMINAL_DICT_PATHS),
    "nominal dict reorder cases": len(NOMINAL_REORDERABLE_DICT_PATHS),
    "nominal DictSubclass cases": len(NOMINAL_DICT_PATHS),
    "nominal BombKey cases": len(NOMINAL_BOMB_KEY_PATHS),
    "blocked dict missing-field cases": len(BLOCKED_DICT_MISSING_CASES),
    "blocked dict extra-field cases": len(BLOCKED_DICT_PATHS),
    "blocked dict reorder cases": len(BLOCKED_REORDERABLE_DICT_PATHS),
    "blocked DictSubclass cases": len(BLOCKED_DICT_PATHS),
    "blocked BombKey cases": len(BLOCKED_BOMB_KEY_PATHS),
    "source list missing-item cases": len(SOURCE_LIST_MISSING_CASES),
    "source list extra-item cases": len(SOURCE_LIST_PATHS),
    "source list reorder cases": len(SOURCE_REORDERABLE_LIST_CASES),
    "source ListSubclass cases": len(SOURCE_LIST_PATHS),
    "nominal list missing-item cases": len(NOMINAL_LIST_MISSING_CASES),
    "nominal list extra-item cases": len(NOMINAL_LIST_PATHS),
    "nominal list reorder cases": len(NOMINAL_REORDERABLE_LIST_CASES),
    "nominal ListSubclass cases": len(NOMINAL_LIST_PATHS),
    "blocked list missing-item cases": len(BLOCKED_LIST_MISSING_CASES),
    "blocked list extra-item cases": len(BLOCKED_LIST_PATHS),
    "blocked list reorder cases": len(BLOCKED_REORDERABLE_LIST_CASES),
    "blocked ListSubclass cases": len(BLOCKED_LIST_PATHS),
}
EXHAUSTIVE_CONTAINER_MUTATION_CASE_TOTAL = sum(EXHAUSTIVE_CONTAINER_MUTATION_CASE_COUNTS.values())


def test_bomb_keys_fail_closed_without_equality(monkeypatch: pytest.MonkeyPatch) -> None:
    source = copy.deepcopy(contract._source_template())
    _reset_bomb_counters()
    _replace_key(source, "schema_version")
    assert contract._source_accepted(source) is False
    _assert_source_blocks(monkeypatch, source)
    assert BombKey.equality_calls == 0

    source = copy.deepcopy(contract._source_template())
    _reset_bomb_counters()
    _replace_key(source["readiness_rows"][0], "readiness_id")
    assert contract._source_accepted(source) is False
    _assert_source_blocks(monkeypatch, source)
    assert BombKey.equality_calls == 0


@pytest.mark.parametrize(
    "payload_factory,path",
    [
        (contract._nominal, ("schema_version",)),
        (contract._nominal, ("contract_boundaries", "source_only")),
        (contract._blocked, ("schema_version",)),
    ],
)
def test_integrity_rejects_bomb_keys_without_equality(payload_factory: Any, path: Path) -> None:
    payload = payload_factory()
    parent = _get_path(payload, path[:-1])
    assert type(parent) is dict
    _reset_bomb_counters()
    assert type(path[-1]) is str
    _replace_key(parent, path[-1])
    assert contract._integrity(payload) is False
    assert BombKey.equality_calls == 0


@pytest.mark.parametrize("sentinel", [EqualityBomb, UnhashableEqual, HashBomb])
def test_custom_source_values_fail_closed(
    monkeypatch: pytest.MonkeyPatch, sentinel: type[object]
) -> None:
    source = copy.deepcopy(contract._source_template())
    _reset_bomb_counters()
    source["readiness_rows"][0]["source_requirement_ids"][0] = sentinel()
    assert contract._source_accepted(source) is False
    _assert_source_blocks(monkeypatch, source)
    assert EqualityBomb.equality_calls == 0
    assert HashBomb.hash_calls == 0


@pytest.mark.parametrize("sentinel", [EqualityBomb, UnhashableEqual, HashBomb])
def test_integrity_rejects_custom_nested_list_values(sentinel: type[object]) -> None:
    payload = contract._nominal()
    _reset_bomb_counters()
    payload["build_readiness_contract_rows"][0]["source_requirement_ids"][0] = sentinel()
    assert contract._integrity(payload) is False
    assert EqualityBomb.equality_calls == 0
    assert HashBomb.hash_calls == 0


@pytest.mark.parametrize(
    "payload_factory,path,replacement",
    [
        (contract._source_template, ("schema_version",), lambda value: StrSubclass(value)),
        (contract._source_template, ("readiness_rows",), ListSubclass),
        (contract._source_template, ("boundaries",), DictSubclass),
        (contract._nominal, ("schema_version",), lambda value: StrSubclass(value)),
        (contract._nominal, ("build_readiness_contract_rows",), ListSubclass),
        (contract._nominal, ("contract_boundaries",), DictSubclass),
        (contract._blocked, ("schema_version",), lambda value: StrSubclass(value)),
        (contract._blocked, ("build_readiness_contract_rows",), ListSubclass),
        (contract._blocked, (), DictSubclass),
    ],
)
def test_exact_subclasses_are_rejected(
    monkeypatch: pytest.MonkeyPatch, payload_factory: Any, path: Path, replacement: Any
) -> None:
    payload = payload_factory()
    mutated: object = replacement(_get_path(payload, path))
    if payload_factory is contract._source_template:
        _set_path(payload, path, mutated)
        _assert_source_blocks(monkeypatch, payload)
    elif path:
        _set_path(payload, path, mutated)
        assert contract._integrity(payload) is False
    else:
        assert contract._integrity(mutated) is False


def _deep_list(depth: int, leaf: object) -> list[object]:
    root: list[object] = []
    current = root
    for _ in range(depth):
        child: list[object] = []
        current.append(child)
        current = child
    current.append(leaf)
    return root


def test_exact_plain_handles_depth_1500_without_recursion() -> None:
    assert contract._exact_plain(_deep_list(1500, "leaf"), _deep_list(1500, "leaf")) is True
    assert contract._exact_plain(_deep_list(1500, "actual"), _deep_list(1500, "trusted")) is False


@pytest.mark.parametrize(
    "payload_factory", [contract._source_template, contract._nominal, contract._blocked]
)
def test_path_helpers_round_trip_and_enumeration(payload_factory: Any) -> None:
    payload = payload_factory()
    leaf_paths = _leaf_paths(payload)
    dict_paths, list_paths = _container_paths(payload)
    assert len(leaf_paths) == len(set(leaf_paths))
    assert len(dict_paths) == len(set(dict_paths))
    assert len(list_paths) == len(set(list_paths))
    for path in leaf_paths:
        value = _get_path(payload, path)
        assert type(value) in (str, bool, int, type(None))
    for path in dict_paths:
        assert type(_get_path(payload, path)) is dict
    for path in list_paths:
        assert type(_get_path(payload, path)) is list
    assert leaf_paths
    original = copy.deepcopy(payload)
    mutated = copy.deepcopy(payload)
    path = leaf_paths[0]
    _set_path(mutated, path, _wrong_scalar(_get_path(mutated, path)))
    assert _get_path(mutated, path) != _get_path(original, path)
    assert _get_path(payload, path) == _get_path(original, path)


def test_path_helpers_reject_empty_set_path_and_return_wrong_types() -> None:
    with pytest.raises(AssertionError):
        _set_path({}, (), "value")
    for value in (True, 7, "value"):
        assert type(_wrong_scalar(value)) is type(value)
        assert type(_wrong_type_scalar(value)) is not type(value)


def test_enumeration_completeness_for_source_and_nominal_payloads() -> None:
    source = contract._source_template()
    nominal = contract._nominal()
    source_dict_paths, source_list_paths = _container_paths(source)
    nominal_dict_paths, nominal_list_paths = _container_paths(nominal)
    assert len(source["readiness_rows"]) == 17
    assert (
        sum(
            1
            for row in source["readiness_rows"]
            for field in ("source_requirement_ids", "source_blocker_ids", "required_evidence_ids")
            if type(row[field]) is list
        )
        == 51
    )
    assert ("readiness_rows",) in source_list_paths
    for index in range(17):
        for field in ("source_requirement_ids", "source_blocker_ids", "required_evidence_ids"):
            assert ("readiness_rows", index, field) in source_list_paths
    assert len(nominal["build_readiness_contract_rows"]) == 17
    assert len(nominal["build_readiness_acceptance_rules"]) == 6
    for path in (
        ("build_readiness_contract_rows",),
        ("build_readiness_acceptance_rules",),
        ("future_steps",),
    ):
        assert path in nominal_list_paths
    for index in range(17):
        for field in ("source_requirement_ids", "source_blocker_ids", "required_evidence_ids"):
            assert ("build_readiness_contract_rows", index, field) in nominal_list_paths
    assert source_dict_paths and nominal_dict_paths


def test_scalar_path_counts_are_canonical() -> None:
    source = contract._source_template()
    nominal = contract._nominal()
    blocked = contract._blocked()

    assert len(SOURCE_LEAF_PATHS) == 257
    assert len(NOMINAL_LEAF_PATHS) == 739
    assert len(BLOCKED_LEAF_PATHS) == 14
    assert len(SOURCE_LEAF_PATHS) == len(set(SOURCE_LEAF_PATHS))
    assert len(NOMINAL_LEAF_PATHS) == len(set(NOMINAL_LEAF_PATHS))
    assert len(BLOCKED_LEAF_PATHS) == len(set(BLOCKED_LEAF_PATHS))
    for payload, paths in (
        (source, SOURCE_LEAF_PATHS),
        (nominal, NOMINAL_LEAF_PATHS),
        (blocked, BLOCKED_LEAF_PATHS),
    ):
        assert all(type(_get_path(payload, path)) in (str, bool, int, type(None)) for path in paths)
        assert all(_get_path(payload, path) is not None for path in paths)


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
        == 2020
    )


@pytest.mark.parametrize("path", SOURCE_EXACT_VALUE_PATHS, ids=_path_id)
def test_source_exact_value_scalar_matrix_fails_closed(
    monkeypatch: pytest.MonkeyPatch, path: Path
) -> None:
    original = _get_path(contract._source_template(), path)
    assert type(original) in (str, bool, int)
    replacement = _wrong_scalar(original)
    assert type(replacement) is type(original)
    assert replacement != original
    _assert_source_scalar_mutation_blocks(monkeypatch, path, replacement)


@pytest.mark.parametrize("path", NOMINAL_EXACT_VALUE_PATHS, ids=_path_id)
def test_nominal_exact_value_scalar_matrix_is_rejected(path: Path) -> None:
    payload = copy.deepcopy(contract._nominal())
    original = _get_path(payload, path)
    assert type(original) in (str, bool, int)
    _set_path(payload, path, _wrong_scalar(original))
    assert contract._integrity(payload) is False


@pytest.mark.parametrize("path", BLOCKED_EXACT_VALUE_PATHS, ids=_path_id)
def test_blocked_exact_value_scalar_matrix_is_rejected(path: Path) -> None:
    payload = copy.deepcopy(contract._blocked())
    original = _get_path(payload, path)
    assert type(original) in (str, bool, int)
    _set_path(payload, path, _wrong_scalar(original))
    assert contract._integrity(payload) is False


@pytest.mark.parametrize("path", SOURCE_EXACT_TYPE_PATHS, ids=_path_id)
def test_source_exact_type_scalar_matrix_fails_closed(
    monkeypatch: pytest.MonkeyPatch, path: Path
) -> None:
    _reset_bomb_counters()
    original = _get_path(contract._source_template(), path)
    replacement = _wrong_type_scalar(original)
    assert type(replacement) is not type(original)
    _assert_source_scalar_mutation_blocks(monkeypatch, path, replacement)
    assert BombKey.equality_calls == 0
    assert EqualityBomb.equality_calls == 0
    assert HashBomb.hash_calls == 0


@pytest.mark.parametrize("path", NOMINAL_EXACT_TYPE_PATHS, ids=_path_id)
def test_nominal_exact_type_scalar_matrix_is_rejected(path: Path) -> None:
    _reset_bomb_counters()
    payload = copy.deepcopy(contract._nominal())
    original = _get_path(payload, path)
    replacement = _wrong_type_scalar(original)
    assert type(replacement) is not type(original)
    _set_path(payload, path, replacement)
    assert contract._integrity(payload) is False
    assert BombKey.equality_calls == 0
    assert EqualityBomb.equality_calls == 0
    assert HashBomb.hash_calls == 0


@pytest.mark.parametrize("path", BLOCKED_EXACT_TYPE_PATHS, ids=_path_id)
def test_blocked_exact_type_scalar_matrix_is_rejected(path: Path) -> None:
    _reset_bomb_counters()
    payload = copy.deepcopy(contract._blocked())
    original = _get_path(payload, path)
    replacement = _wrong_type_scalar(original)
    assert type(replacement) is not type(original)
    _set_path(payload, path, replacement)
    assert contract._integrity(payload) is False
    assert BombKey.equality_calls == 0
    assert EqualityBomb.equality_calls == 0
    assert HashBomb.hash_calls == 0


def _mutate_dict_missing(
    factory: Callable[[], dict[str, Any]], path: Path, key: str
) -> dict[str, Any]:
    payload = copy.deepcopy(factory())
    mapping = _get_path(payload, path)
    assert type(mapping) is dict
    mapping.pop(key)
    assert key not in mapping
    return payload


def _mutate_dict_extra(factory: Callable[[], dict[str, Any]], path: Path) -> dict[str, Any]:
    payload = copy.deepcopy(factory())
    mapping = _get_path(payload, path)
    assert type(mapping) is dict
    key = _extra_key(mapping)
    mapping[key] = False
    assert key in mapping
    return payload


def _mutate_dict_reorder(factory: Callable[[], dict[str, Any]], path: Path) -> dict[str, Any]:
    payload = copy.deepcopy(factory())
    mapping = _get_path(payload, path)
    assert type(mapping) is dict
    reordered = dict(reversed(list(mapping.items())))
    assert list(reordered) != list(mapping)
    mutated = _replace_path(payload, path, reordered)
    assert type(mutated) is dict
    return mutated


def _mutate_dict_subclass(factory: Callable[[], dict[str, Any]], path: Path) -> object:
    payload = copy.deepcopy(factory())
    mapping = _get_path(payload, path)
    assert type(mapping) is dict
    replacement = DictSubclass(mapping)
    assert type(replacement) is DictSubclass
    return _replace_path(payload, path, replacement)


def _mutate_bomb_key(factory: Callable[[], dict[str, Any]], path: Path) -> object:
    payload = copy.deepcopy(factory())
    mapping = _get_path(payload, path)
    assert type(mapping) is dict
    first_key = next(iter(mapping))
    assert type(first_key) is str
    _reset_bomb_counters()
    _replace_key(mapping, first_key)
    return payload


def _mutate_list_missing(
    factory: Callable[[], dict[str, Any]], path: Path, index: int
) -> dict[str, Any]:
    payload = copy.deepcopy(factory())
    values = _get_path(payload, path)
    assert type(values) is list
    original_len = len(values)
    values.pop(index)
    assert len(values) == original_len - 1
    return payload


def _mutate_list_extra(factory: Callable[[], dict[str, Any]], path: Path) -> dict[str, Any]:
    payload = copy.deepcopy(factory())
    values = _get_path(payload, path)
    assert type(values) is list
    original_len = len(values)
    values.append(_extra_list_item(values))
    assert len(values) == original_len + 1
    return payload


def _mutate_list_reorder(
    factory: Callable[[], dict[str, Any]], path: Path, left: int, right: int
) -> dict[str, Any]:
    payload = copy.deepcopy(factory())
    canonical = factory()
    values = _get_path(payload, path)
    assert type(values) is list
    values[left], values[right] = values[right], values[left]
    assert contract._exact_plain(values, _get_path(canonical, path)) is False
    return payload


def _mutate_list_subclass(factory: Callable[[], dict[str, Any]], path: Path) -> object:
    payload = copy.deepcopy(factory())
    values = _get_path(payload, path)
    assert type(values) is list
    replacement = ListSubclass(values)
    assert type(replacement) is ListSubclass
    return _replace_path(payload, path, replacement)


def _assert_bomb_counters_zero() -> None:
    assert BombKey.equality_calls == 0
    assert EqualityBomb.equality_calls == 0
    assert HashBomb.hash_calls == 0


def test_container_path_counts_are_canonical() -> None:
    assert len(SOURCE_DICT_PATHS) == 20
    assert len(SOURCE_LIST_PATHS) == 52
    assert len(NOMINAL_DICT_PATHS) == 37
    assert len(NOMINAL_LIST_PATHS) == 57
    assert len(BLOCKED_DICT_PATHS) == 1
    assert len(BLOCKED_LIST_PATHS) == 2
    for payload, dict_paths, list_paths in (
        (contract._source_template(), SOURCE_DICT_PATHS, SOURCE_LIST_PATHS),
        (contract._nominal(), NOMINAL_DICT_PATHS, NOMINAL_LIST_PATHS),
        (contract._blocked(), BLOCKED_DICT_PATHS, BLOCKED_LIST_PATHS),
    ):
        assert () in dict_paths
        assert len(dict_paths) == len(set(dict_paths))
        assert len(list_paths) == len(set(list_paths))
        assert all(type(_get_path(payload, path)) is dict for path in dict_paths)
        assert all(type(_get_path(payload, path)) is list for path in list_paths)


def test_replace_path_handles_root_nested_and_preserves_factory_result() -> None:
    original = contract._source_template()
    root_replaced = _replace_path(copy.deepcopy(original), (), {"replacement": True})
    assert root_replaced == {"replacement": True}
    nested: dict[str, Any] = copy.deepcopy(original)
    returned = _replace_path(nested, ("boundaries",), {"replacement": False})
    assert returned is nested
    assert nested["boundaries"] == {"replacement": False}
    assert contract._source_template() == original


@pytest.mark.parametrize(
    "path,key",
    SOURCE_DICT_MISSING_CASES,
    ids=lambda case: _path_id(case) if isinstance(case, tuple) else str(case),
)
def test_source_dict_schema_matrix_missing_field_fails_closed(
    monkeypatch: pytest.MonkeyPatch, path: Path, key: str
) -> None:
    _assert_source_payload_blocks_once(
        monkeypatch, _mutate_dict_missing(contract._source_template, path, key)
    )


@pytest.mark.parametrize("path,key", NOMINAL_DICT_MISSING_CASES, ids=lambda c: str(c))
def test_nominal_dict_schema_matrix_missing_field_is_rejected(path: Path, key: str) -> None:
    assert contract._integrity(_mutate_dict_missing(contract._nominal, path, key)) is False


@pytest.mark.parametrize("path,key", BLOCKED_DICT_MISSING_CASES, ids=lambda c: str(c))
def test_blocked_dict_schema_matrix_missing_field_is_rejected(path: Path, key: str) -> None:
    assert contract._integrity(_mutate_dict_missing(contract._blocked, path, key)) is False


@pytest.mark.parametrize("path", SOURCE_DICT_PATHS, ids=_path_id)
def test_source_dict_schema_matrix_extra_field_fails_closed(
    monkeypatch: pytest.MonkeyPatch, path: Path
) -> None:
    _assert_source_payload_blocks_once(
        monkeypatch, _mutate_dict_extra(contract._source_template, path)
    )


@pytest.mark.parametrize("path", NOMINAL_DICT_PATHS, ids=_path_id)
def test_nominal_dict_schema_matrix_extra_field_is_rejected(path: Path) -> None:
    assert contract._integrity(_mutate_dict_extra(contract._nominal, path)) is False


@pytest.mark.parametrize("path", BLOCKED_DICT_PATHS, ids=_path_id)
def test_blocked_dict_schema_matrix_extra_field_is_rejected(path: Path) -> None:
    assert contract._integrity(_mutate_dict_extra(contract._blocked, path)) is False


@pytest.mark.parametrize("path", SOURCE_REORDERABLE_DICT_PATHS, ids=_path_id)
def test_source_dict_schema_matrix_reorder_fails_closed(
    monkeypatch: pytest.MonkeyPatch, path: Path
) -> None:
    _assert_source_payload_blocks_once(
        monkeypatch, _mutate_dict_reorder(contract._source_template, path)
    )


@pytest.mark.parametrize("path", NOMINAL_REORDERABLE_DICT_PATHS, ids=_path_id)
def test_nominal_dict_schema_matrix_reorder_is_rejected(path: Path) -> None:
    assert contract._integrity(_mutate_dict_reorder(contract._nominal, path)) is False


@pytest.mark.parametrize("path", BLOCKED_REORDERABLE_DICT_PATHS, ids=_path_id)
def test_blocked_dict_schema_matrix_reorder_is_rejected(path: Path) -> None:
    assert contract._integrity(_mutate_dict_reorder(contract._blocked, path)) is False


@pytest.mark.parametrize("path", SOURCE_DICT_PATHS, ids=_path_id)
def test_source_dict_schema_matrix_dict_subclass_fails_closed(
    monkeypatch: pytest.MonkeyPatch, path: Path
) -> None:
    _assert_source_payload_blocks_once(
        monkeypatch, _mutate_dict_subclass(contract._source_template, path)
    )


@pytest.mark.parametrize("path", NOMINAL_DICT_PATHS, ids=_path_id)
def test_nominal_dict_schema_matrix_dict_subclass_is_rejected(path: Path) -> None:
    assert contract._integrity(_mutate_dict_subclass(contract._nominal, path)) is False


@pytest.mark.parametrize("path", BLOCKED_DICT_PATHS, ids=_path_id)
def test_blocked_dict_schema_matrix_dict_subclass_is_rejected(path: Path) -> None:
    assert contract._integrity(_mutate_dict_subclass(contract._blocked, path)) is False


@pytest.mark.parametrize("path", SOURCE_BOMB_KEY_PATHS, ids=_path_id)
def test_source_dict_schema_matrix_bomb_key_fails_closed(
    monkeypatch: pytest.MonkeyPatch, path: Path
) -> None:
    _assert_source_payload_blocks_once(
        monkeypatch, _mutate_bomb_key(contract._source_template, path)
    )
    _assert_bomb_counters_zero()


@pytest.mark.parametrize("path", NOMINAL_BOMB_KEY_PATHS, ids=_path_id)
def test_nominal_dict_schema_matrix_bomb_key_is_rejected(path: Path) -> None:
    assert contract._integrity(_mutate_bomb_key(contract._nominal, path)) is False
    _assert_bomb_counters_zero()


@pytest.mark.parametrize("path", BLOCKED_BOMB_KEY_PATHS, ids=_path_id)
def test_blocked_dict_schema_matrix_bomb_key_is_rejected(path: Path) -> None:
    assert contract._integrity(_mutate_bomb_key(contract._blocked, path)) is False
    _assert_bomb_counters_zero()


@pytest.mark.parametrize("path,index", SOURCE_LIST_MISSING_CASES, ids=lambda c: str(c))
def test_source_list_schema_matrix_missing_item_fails_closed(
    monkeypatch: pytest.MonkeyPatch, path: Path, index: int
) -> None:
    _assert_source_payload_blocks_once(
        monkeypatch, _mutate_list_missing(contract._source_template, path, index)
    )


@pytest.mark.parametrize("path,index", NOMINAL_LIST_MISSING_CASES, ids=lambda c: str(c))
def test_nominal_list_schema_matrix_missing_item_is_rejected(path: Path, index: int) -> None:
    assert contract._integrity(_mutate_list_missing(contract._nominal, path, index)) is False


@pytest.mark.parametrize("path,index", BLOCKED_LIST_MISSING_CASES, ids=lambda c: str(c))
def test_blocked_list_schema_matrix_missing_item_is_rejected(path: Path, index: int) -> None:
    assert contract._integrity(_mutate_list_missing(contract._blocked, path, index)) is False


@pytest.mark.parametrize("path", SOURCE_LIST_PATHS, ids=_path_id)
def test_source_list_schema_matrix_extra_item_fails_closed(
    monkeypatch: pytest.MonkeyPatch, path: Path
) -> None:
    _assert_source_payload_blocks_once(
        monkeypatch, _mutate_list_extra(contract._source_template, path)
    )


@pytest.mark.parametrize("path", NOMINAL_LIST_PATHS, ids=_path_id)
def test_nominal_list_schema_matrix_extra_item_is_rejected(path: Path) -> None:
    assert contract._integrity(_mutate_list_extra(contract._nominal, path)) is False


@pytest.mark.parametrize("path", BLOCKED_LIST_PATHS, ids=_path_id)
def test_blocked_list_schema_matrix_extra_item_is_rejected(path: Path) -> None:
    assert contract._integrity(_mutate_list_extra(contract._blocked, path)) is False


@pytest.mark.parametrize("path,left,right", SOURCE_REORDERABLE_LIST_CASES, ids=lambda c: str(c))
def test_source_list_schema_matrix_reorder_fails_closed(
    monkeypatch: pytest.MonkeyPatch, path: Path, left: int, right: int
) -> None:
    _assert_source_payload_blocks_once(
        monkeypatch, _mutate_list_reorder(contract._source_template, path, left, right)
    )


@pytest.mark.parametrize("path,left,right", NOMINAL_REORDERABLE_LIST_CASES, ids=lambda c: str(c))
def test_nominal_list_schema_matrix_reorder_is_rejected(path: Path, left: int, right: int) -> None:
    assert contract._integrity(_mutate_list_reorder(contract._nominal, path, left, right)) is False


@pytest.mark.parametrize("path,left,right", BLOCKED_REORDERABLE_LIST_CASES, ids=lambda c: str(c))
def test_blocked_list_schema_matrix_reorder_is_rejected(path: Path, left: int, right: int) -> None:
    assert contract._integrity(_mutate_list_reorder(contract._blocked, path, left, right)) is False


@pytest.mark.parametrize("path", SOURCE_LIST_PATHS, ids=_path_id)
def test_source_list_schema_matrix_list_subclass_fails_closed(
    monkeypatch: pytest.MonkeyPatch, path: Path
) -> None:
    _assert_source_payload_blocks_once(
        monkeypatch, _mutate_list_subclass(contract._source_template, path)
    )


@pytest.mark.parametrize("path", NOMINAL_LIST_PATHS, ids=_path_id)
def test_nominal_list_schema_matrix_list_subclass_is_rejected(path: Path) -> None:
    assert contract._integrity(_mutate_list_subclass(contract._nominal, path)) is False


@pytest.mark.parametrize("path", BLOCKED_LIST_PATHS, ids=_path_id)
def test_blocked_list_schema_matrix_list_subclass_is_rejected(path: Path) -> None:
    assert contract._integrity(_mutate_list_subclass(contract._blocked, path)) is False


def test_exhaustive_container_dict_path_coverage_is_complete() -> None:
    for paths, missing, reorderable, bomb in (
        (
            SOURCE_DICT_PATHS,
            SOURCE_DICT_MISSING_CASES,
            SOURCE_REORDERABLE_DICT_PATHS,
            SOURCE_BOMB_KEY_PATHS,
        ),
        (
            NOMINAL_DICT_PATHS,
            NOMINAL_DICT_MISSING_CASES,
            NOMINAL_REORDERABLE_DICT_PATHS,
            NOMINAL_BOMB_KEY_PATHS,
        ),
        (
            BLOCKED_DICT_PATHS,
            BLOCKED_DICT_MISSING_CASES,
            BLOCKED_REORDERABLE_DICT_PATHS,
            BLOCKED_BOMB_KEY_PATHS,
        ),
    ):
        assert Counter(paths) == Counter(paths)
        assert set(reorderable).issubset(set(paths))
        assert set(bomb).issubset(set(paths))
        assert len(reorderable) == len(set(reorderable))
        assert len(bomb) == len(set(bomb))
        payload = (
            contract._source_template()
            if paths is SOURCE_DICT_PATHS
            else contract._nominal()
            if paths is NOMINAL_DICT_PATHS
            else contract._blocked()
        )
        expected_paths: list[Path] = []
        for path in paths:
            mapping = _get_path(payload, path)
            assert type(mapping) is dict
            expected_paths.extend(path for _key in mapping)
        assert Counter(path for path, _ in missing) == Counter(expected_paths)


def test_exhaustive_container_list_path_coverage_is_complete() -> None:
    for paths, missing, reorderable in (
        (SOURCE_LIST_PATHS, SOURCE_LIST_MISSING_CASES, SOURCE_REORDERABLE_LIST_CASES),
        (NOMINAL_LIST_PATHS, NOMINAL_LIST_MISSING_CASES, NOMINAL_REORDERABLE_LIST_CASES),
        (BLOCKED_LIST_PATHS, BLOCKED_LIST_MISSING_CASES, BLOCKED_REORDERABLE_LIST_CASES),
    ):
        assert len(paths) == len(set(paths))
        assert {case[0] for case in reorderable}.issubset(set(paths))
        assert len(reorderable) == len({case[0] for case in reorderable})
        payload = (
            contract._source_template()
            if paths is SOURCE_LIST_PATHS
            else contract._nominal()
            if paths is NOMINAL_LIST_PATHS
            else contract._blocked()
        )
        expected_cases: list[tuple[Path, int]] = []
        for path in paths:
            values = _get_path(payload, path)
            assert type(values) is list
            expected_cases.extend((path, index) for index in range(len(values)))
        assert Counter(missing) == Counter(expected_cases)


def test_exhaustive_container_reported_total_matches_case_constants() -> None:
    assert EXHAUSTIVE_CONTAINER_MUTATION_CASE_TOTAL == sum(
        EXHAUSTIVE_CONTAINER_MUTATION_CASE_COUNTS.values()
    )
