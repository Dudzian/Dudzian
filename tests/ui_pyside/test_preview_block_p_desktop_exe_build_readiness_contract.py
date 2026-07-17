from __future__ import annotations

import copy
import json
from typing import Any

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
        source = build_preview_block_p_desktop_exe_build_readiness_matrix()
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
    __hash__ = None

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
        result = StrSubclass(value)
    else:
        result = EqualityBomb()
    assert type(result) is not type(value)
    return result


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
    return payload


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
