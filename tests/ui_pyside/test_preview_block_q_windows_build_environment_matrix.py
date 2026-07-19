from __future__ import annotations

import ast
import copy
import importlib
import json
from pathlib import Path
from typing import Any

import pytest

import ui.pyside_app.preview_block_q_windows_build_environment_inventory as q191
import ui.pyside_app.preview_block_q_windows_build_environment_matrix as q192
from ui.pyside_app.preview_block_q_windows_build_environment_inventory import (
    build_preview_block_q_windows_build_environment_inventory,
)
from ui.pyside_app.preview_block_q_windows_build_environment_matrix import (
    AUTHORIZATION_FALSE_FIELDS,
    BLOCKED_STATUS,
    ENVIRONMENT_MATRIX_SPECS,
    MATRIX_ROW_FIELDS,
    SOURCE_19_1_INVENTORY_SPECS,
    TOP_LEVEL_FIELDS,
    _blocked,
    _canonical_blocked,
    _canonical_nominal,
    _environment_matrix_rows,
    _exact_plain,
    _integrity,
    _source_accepted,
    _trusted_source_inventory_rows,
    _trusted_source_stub,
    build_preview_block_q_windows_build_environment_matrix,
)


class EqualityBomb:
    calls = 0

    def __eq__(self, other: object) -> bool:
        type(self).calls += 1
        raise AssertionError("eq")


class HashBomb:
    calls = 0

    def __hash__(self) -> int:
        type(self).calls += 1
        raise AssertionError("hash")


class UnhashableEqual:
    calls = 0
    __hash__ = None

    def __eq__(self, other: object) -> bool:
        type(self).calls += 1
        raise AssertionError("eq")


class StrSubclass(str):
    pass


class ListSubclass(list[Any]):
    pass


class DictSubclass(dict[str, Any]):
    pass


class GetItemBombDict(dict[str, Any]):
    getitem_calls = 0

    def __getitem__(self, key: object) -> Any:
        type(self).getitem_calls += 1
        raise AssertionError("getitem")


class ArmedBombKey(str):
    hash_calls = 0
    equality_calls = 0

    def __new__(cls, value: str) -> ArmedBombKey:
        instance = super().__new__(cls, value)
        instance.armed = False
        return instance

    def __hash__(self) -> int:
        if self.armed:
            type(self).hash_calls += 1
            raise AssertionError("hash")
        return str.__hash__(self)

    def __eq__(self, other: object) -> bool:
        if self.armed:
            type(self).equality_calls += 1
            raise AssertionError("eq")
        return str.__eq__(self, other)


def _replace_key(mapping: dict[str, Any], key: str) -> ArmedBombKey:
    items = list(mapping.items())
    mapping.clear()
    bomb = ArmedBombKey(key)
    for item_key, value in items:
        mapping[bomb if item_key == key else item_key] = value
    bomb.armed = True
    return bomb


def _via_builder(monkeypatch: pytest.MonkeyPatch, source: Any) -> dict[str, Any]:
    monkeypatch.setattr(
        q192, "build_preview_block_q_windows_build_environment_inventory", lambda: source
    )
    return build_preview_block_q_windows_build_environment_matrix()


def _assert_blocked(payload: dict[str, Any]) -> None:
    assert payload == _canonical_blocked()
    assert payload["status"] == BLOCKED_STATUS
    assert _integrity(payload) is True


def test_nominal_builder_once_source_unchanged_order_counts_json_deterministic_integrity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = 0
    source = build_preview_block_q_windows_build_environment_inventory()
    before = copy.deepcopy(source)

    def fake() -> dict[str, Any]:
        nonlocal calls
        calls += 1
        return source

    monkeypatch.setattr(q192, "build_preview_block_q_windows_build_environment_inventory", fake)
    payload = build_preview_block_q_windows_build_environment_matrix()
    assert calls == 1
    assert source == before
    assert _source_accepted(source) is True
    assert list(payload) == list(TOP_LEVEL_FIELDS)
    rows = payload["environment_matrix_rows"]
    assert len(rows) == 11
    assert [row["matrix_id"] for row in rows] == [spec[0] for spec in ENVIRONMENT_MATRIX_SPECS]
    assert [row["inventory_id"] for row in rows] == [
        spec[0] for spec in SOURCE_19_1_INVENTORY_SPECS
    ]
    assert [row["requirement_field"] for row in rows] == [
        spec[1] for spec in SOURCE_19_1_INVENTORY_SPECS
    ]
    assert len(set(row["matrix_id"] for row in rows)) == 11
    assert len(set(row["inventory_id"] for row in rows)) == 11
    assert len(set(row["requirement_field"] for row in rows)) == 11
    assert all(list(row) == list(MATRIX_ROW_FIELDS) for row in rows)
    assert all(
        row["matrix_state"] == "blocked" and row["requirement_satisfied"] is False for row in rows
    )
    assert payload["environment_matrix_summary"] == {
        "matrix_row_count": 11,
        "required_count": 11,
        "satisfied_count": 0,
        "unsatisfied_count": 11,
        "ready_count": 0,
        "blocked_count": 11,
        "evidence_required_count": 11,
        "evidence_collected_count": 0,
        "evidence_validated_count": 0,
        "environment_observation_complete": False,
        "environment_matrix_definition_complete": True,
        "environment_build_ready": False,
    }
    assert payload["environment_build_ready"] is False
    assert payload["future_steps"] == [
        {
            "next_step": "FUNCTIONAL-PREVIEW-19.3",
            "next_step_title": "WINDOWS BUILD ENVIRONMENT CONTRACT",
            "source_only": True,
            "environment_observation_performed": False,
            "physical_build_performed": False,
        }
    ]
    for key in AUTHORIZATION_FALSE_FIELDS:
        assert payload["build_execution_authorization_state"][key] is False
    json.dumps(payload)
    json.dumps(payload)
    assert calls == 1
    assert _integrity(payload) is True
    assert calls == 1
    assert payload == _canonical_nominal()
    assert calls == 1
    assert payload == build_preview_block_q_windows_build_environment_matrix()
    assert calls == 2
    assert _integrity(payload) is True
    assert calls == 2


@pytest.mark.parametrize(
    "mutate",
    [
        lambda s: s.pop("schema_version"),
        lambda s: s.update({"extra": True}),
        lambda s: s.update({"schema_version": s.pop("schema_version")}),
        lambda s: s.__setitem__("schema_version", "wrong"),
        lambda s: s.__setitem__("source_19_0_accepted", False),
        lambda s: s.__setitem__("environment_inventory_artifact_complete", False),
        lambda s: s.__setitem__("environment_observation_complete", True),
        lambda s: s.__setitem__("ready_for_block_q_2", False),
        lambda s: s.__setitem__("next_step", "x"),
        lambda s: s.__setitem__("next_step_title", "x"),
        lambda s: s["environment_inventory_rows"].pop(),
        lambda s: s["environment_inventory_rows"].append(
            copy.deepcopy(s["environment_inventory_rows"][0])
        ),
        lambda s: s["environment_inventory_rows"].reverse(),
        lambda s: s["environment_inventory_rows"][0].pop("inventory_id"),
        lambda s: s["environment_inventory_rows"][0].update({"extra": True}),
        lambda s: s["environment_inventory_rows"][0].update(
            {"inventory_id": s["environment_inventory_rows"][0].pop("inventory_id")}
        ),
        lambda s: s["environment_inventory_rows"][1].__setitem__("inventory_id", "windows_host"),
        lambda s: s["environment_inventory_rows"][0].__setitem__("inventory_id", "wrong"),
        lambda s: s["environment_inventory_rows"][0].__setitem__("requirement_field", "wrong"),
        lambda s: s["environment_inventory_rows"][0].__setitem__("category", "wrong"),
        lambda s: s["environment_inventory_rows"][0].__setitem__("required", False),
        lambda s: s["environment_inventory_rows"][0].__setitem__("collection_status", "collected"),
        lambda s: s["environment_inventory_rows"][0].__setitem__("validation_status", "validated"),
        lambda s: s["environment_inventory_rows"][0].__setitem__("resolution_status", "resolved"),
        lambda s: s["environment_inventory_rows"][0].__setitem__("observed_value", "x"),
        lambda s: s["environment_inventory_summary"].__setitem__("inventory_row_count", 10),
        lambda s: s["environment_inventory_scope"].__setitem__("inventory_row_count", 10),
        lambda s: s["environment_inventory_scope"].__setitem__("source_only_inventory", False),
        lambda s: s["build_execution_authorization_state"].__setitem__(
            "environment_matrix_definition_authorized", False
        ),
        lambda s: s["build_execution_authorization_state"].__setitem__(
            "packaging_authorized", True
        ),
        lambda s: s["future_steps"].pop(),
        lambda s: s["future_steps"].append(copy.deepcopy(s["future_steps"][0])),
        lambda s: s["future_steps"][0].__setitem__("next_step", "x"),
        lambda s: s.__setitem__("integrity_valid", False),
    ],
)
def test_source_acceptance_mutations_fail_closed(
    monkeypatch: pytest.MonkeyPatch, mutate: Any
) -> None:
    source = build_preview_block_q_windows_build_environment_inventory()
    mutate(source)
    _assert_blocked(_via_builder(monkeypatch, source))


def test_canonical_nominal_and_integrity_do_not_call_upstream_builder(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = 0

    def throwing_builder() -> dict[str, Any]:
        nonlocal calls
        calls += 1
        raise AssertionError("canonical trust boundary must not call upstream")

    monkeypatch.setattr(
        q192,
        "build_preview_block_q_windows_build_environment_inventory",
        throwing_builder,
    )

    nominal = q192._canonical_nominal()

    assert nominal["source_19_1_accepted"] is True
    assert q192._integrity(nominal) is True
    assert q192._integrity(q192._blocked()) is True
    assert calls == 0


def test_upstream_tampering_independence() -> None:
    canonical = q192._canonical_nominal()
    source = build_preview_block_q_windows_build_environment_inventory()
    source["environment_inventory_rows"][0]["collection_status"] = "tampered"

    tampered_payload = q192._nominal(source)

    assert tampered_payload != canonical
    assert q192._integrity(tampered_payload) is False
    assert q192._canonical_nominal() == canonical


def _function_calls(function_name: str) -> set[str]:
    tree = ast.parse(Path(q192.__file__).read_text())
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == function_name:
            calls: set[str] = set()
            for child in ast.walk(node):
                if isinstance(child, ast.Call):
                    if isinstance(child.func, ast.Name):
                        calls.add(child.func.id)
                    elif isinstance(child.func, ast.Attribute):
                        calls.add(child.func.attr)
            return calls
    raise AssertionError(f"missing function {function_name}")


def test_ast_trust_boundary_guard() -> None:
    forbidden = {"build_preview_block_q_windows_build_environment_inventory", "_integrity_19_1"}
    for function_name in ("_trusted_source_stub", "_canonical_nominal", "_integrity"):
        assert _function_calls(function_name).isdisjoint(forbidden)


def test_fresh_trusted_source_stubs_and_rows() -> None:
    specs_before = q192.SOURCE_19_1_INVENTORY_SPECS
    first = _trusted_source_stub()
    second = _trusted_source_stub()
    assert first is not second
    assert first["environment_inventory_rows"] is not second["environment_inventory_rows"]
    assert first["environment_inventory_rows"][0] is not second["environment_inventory_rows"][0]
    first["environment_inventory_rows"][0]["collection_status"] = "tampered"
    assert second["environment_inventory_rows"][0]["collection_status"] == "not_collected"
    third = _trusted_source_stub()
    assert third["environment_inventory_rows"] == _trusted_source_inventory_rows()
    assert q192.SOURCE_19_1_INVENTORY_SPECS is specs_before
    assert type(q192.SOURCE_19_1_INVENTORY_SPECS) is tuple


@pytest.mark.parametrize(
    ("field", "canonical", "replacement"),
    [
        ("collection_status", "not_collected", EqualityBomb()),
        ("collection_status", "not_collected", StrSubclass("not_collected")),
        ("collection_status", "not_collected", 1),
        ("collection_status", "not_collected", None),
        ("validation_status", "not_validated", EqualityBomb()),
        ("validation_status", "not_validated", StrSubclass("not_validated")),
        ("validation_status", "not_validated", 1),
        ("validation_status", "not_validated", None),
        ("resolution_status", "blocked", EqualityBomb()),
        ("resolution_status", "blocked", StrSubclass("blocked")),
        ("resolution_status", "blocked", 1),
        ("resolution_status", "blocked", None),
        ("observed_value", "", EqualityBomb()),
        ("observed_value", "", StrSubclass("")),
        ("observed_value", "", 1),
        ("observed_value", "", None),
    ],
)
def test_source_row_string_scalars_exact_type_first_adversarial_paths(
    monkeypatch: pytest.MonkeyPatch, field: str, canonical: str, replacement: Any
) -> None:
    del canonical
    EqualityBomb.calls = 0
    calls = 0
    source = build_preview_block_q_windows_build_environment_inventory()
    source["environment_inventory_rows"][0][field] = replacement

    def fake() -> dict[str, Any]:
        nonlocal calls
        calls += 1
        return source

    monkeypatch.setattr(q192, "build_preview_block_q_windows_build_environment_inventory", fake)
    blocked = build_preview_block_q_windows_build_environment_matrix()
    _assert_blocked(blocked)
    json.dumps(blocked)
    assert EqualityBomb.calls == 0
    assert calls == 1


@pytest.mark.parametrize(
    ("field", "replacement"),
    [
        ("inventory_row_count", True),
        ("inventory_row_count", StrSubclass("11")),
        ("inventory_row_count", EqualityBomb()),
        ("inventory_defined", 1),
        ("source_only_inventory", 1),
    ],
)
def test_scope_scalars_exact_type_first_adversarial(
    monkeypatch: pytest.MonkeyPatch, field: str, replacement: Any
) -> None:
    EqualityBomb.calls = 0
    calls = 0
    source = build_preview_block_q_windows_build_environment_inventory()
    source["environment_inventory_scope"][field] = replacement

    def fake() -> dict[str, Any]:
        nonlocal calls
        calls += 1
        return source

    monkeypatch.setattr(q192, "build_preview_block_q_windows_build_environment_inventory", fake)
    blocked = build_preview_block_q_windows_build_environment_matrix()
    _assert_blocked(blocked)
    assert EqualityBomb.calls == 0
    assert calls == 1


def test_upstream_integrity_exception_fail_closed(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(q192, "_integrity_19_1", lambda source: (_ for _ in ()).throw(RuntimeError))
    _assert_blocked(
        _via_builder(monkeypatch, build_preview_block_q_windows_build_environment_inventory())
    )


def test_blocked_independent_fresh_json_deterministic_and_rejects_positive_mutations(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(q192, "_nominal", lambda source: (_ for _ in ()).throw(RuntimeError))
    _assert_blocked(build_preview_block_q_windows_build_environment_matrix())
    first = _blocked()
    second = _blocked()
    assert first == second
    assert first is not second
    assert first["environment_matrix_rows"] is not second["environment_matrix_rows"]
    assert first["environment_matrix_summary"]["matrix_row_count"] == 0
    assert not any(first["environment_matrix_scope"].values())
    json.dumps(first)
    assert _integrity(first) is True
    mutated = _blocked()
    mutated.pop("schema_version")
    assert _integrity(mutated) is False
    mutated = _blocked()
    mutated["extra"] = False
    assert _integrity(mutated) is False
    mutated = _blocked()
    mutated["schema_version"] = mutated.pop("schema_version")
    assert _integrity(mutated) is False
    mutated = _blocked()
    mutated["environment_matrix_scope"]["matrix_defined"] = True
    assert _integrity(mutated) is False


def test_matrix_specs_and_fresh_rows() -> None:
    assert type(ENVIRONMENT_MATRIX_SPECS) is tuple
    assert len(ENVIRONMENT_MATRIX_SPECS) == 11
    assert len({spec[0] for spec in ENVIRONMENT_MATRIX_SPECS}) == 11
    assert [spec[1:] for spec in ENVIRONMENT_MATRIX_SPECS] == list(SOURCE_19_1_INVENTORY_SPECS)
    source_rows = build_preview_block_q_windows_build_environment_inventory()[
        "environment_inventory_rows"
    ]
    one = _environment_matrix_rows(source_rows)
    two = _environment_matrix_rows(source_rows)
    assert one == two and one is not two and one[0] is not two[0]
    one[0]["matrix_state"] = "changed"
    assert two[0]["matrix_state"] == "blocked"


def test_comparator_cycles_and_depth() -> None:
    a: list[Any] = []
    b: list[Any] = []
    a.append(a)
    b.append(b)
    assert _exact_plain(a, b) is True
    ad: dict[str, Any] = {}
    bd: dict[str, Any] = {}
    ad["self"] = ad
    bd["self"] = bd
    assert _exact_plain(ad, bd) is True
    c: list[Any] = [1]
    c.append(c)
    assert _exact_plain(c, b) is False
    left: Any = 0
    right: Any = 0
    for _ in range(1500):
        left = [left]
        right = [right]
    assert _exact_plain(left, right) is True
    right[0] = 1
    assert _exact_plain(left, right) is False


@pytest.mark.parametrize(
    "mutate",
    [
        lambda s: _replace_key(s, "schema_version"),
        lambda s: _replace_key(s["environment_inventory_rows"][0], "inventory_id"),
        lambda s: _replace_key(s["environment_inventory_summary"], "inventory_row_count"),
        lambda s: _replace_key(s["environment_inventory_scope"], "inventory_defined"),
        lambda s: _replace_key(s["build_execution_authorization_state"], "packaging_authorized"),
        lambda s: _replace_key(s["future_steps"][0], "next_step"),
        lambda s: s["environment_inventory_rows"][0].__setitem__("inventory_id", EqualityBomb()),
        lambda s: s.__setitem__(
            "environment_inventory_summary", DictSubclass(s["environment_inventory_summary"])
        ),
        lambda s: s["block_q_19_0_entry_contract_reference"].__setitem__(
            "schema_version", StrSubclass("x")
        ),
        lambda s: s.__setitem__(
            "environment_inventory_rows", ListSubclass(s["environment_inventory_rows"])
        ),
        lambda s: s.__setitem__(
            "environment_inventory_summary", GetItemBombDict(s["environment_inventory_summary"])
        ),
    ],
)
def test_adversarial_source_mutations_fail_closed_without_custom_counters(
    monkeypatch: pytest.MonkeyPatch, mutate: Any
) -> None:
    EqualityBomb.calls = HashBomb.calls = UnhashableEqual.calls = 0
    ArmedBombKey.hash_calls = ArmedBombKey.equality_calls = GetItemBombDict.getitem_calls = 0
    source = build_preview_block_q_windows_build_environment_inventory()
    mutate(source)
    _assert_blocked(_via_builder(monkeypatch, source))
    assert EqualityBomb.calls == 0
    assert HashBomb.calls == 0
    assert UnhashableEqual.calls == 0
    assert ArmedBombKey.hash_calls == 0
    assert ArmedBombKey.equality_calls == 0
    assert GetItemBombDict.getitem_calls == 0


def test_collection_safety_reload_and_test_order_isolation(monkeypatch: pytest.MonkeyPatch) -> None:
    assert _integrity(build_preview_block_q_windows_build_environment_matrix()) is True
    monkeypatch.setattr(
        q191,
        "build_preview_block_q_windows_build_environment_inventory",
        lambda: (_ for _ in ()).throw(RuntimeError),
        raising=False,
    )
    reloaded = importlib.reload(q192)
    assert reloaded._integrity(reloaded._blocked()) is True
    canonical_calls = 0

    def counted_throwing_builder() -> dict[str, Any]:
        nonlocal canonical_calls
        canonical_calls += 1
        raise AssertionError("canonical must stay local")

    monkeypatch.setattr(
        reloaded,
        "build_preview_block_q_windows_build_environment_inventory",
        counted_throwing_builder,
    )
    nominal = reloaded._canonical_nominal()
    assert nominal["source_19_1_accepted"] is True
    assert reloaded._integrity(nominal) is True
    assert canonical_calls == 0
    monkeypatch.setattr(q191, "_nominal", lambda source: (_ for _ in ()).throw(RuntimeError))
    monkeypatch.setattr(q191, "_blocked", lambda: (_ for _ in ()).throw(RuntimeError))
    monkeypatch.setattr(q191, "_canonical_nominal", lambda: (_ for _ in ()).throw(RuntimeError))
    monkeypatch.setattr(q191, "_canonical_blocked", lambda: (_ for _ in ()).throw(RuntimeError))
    reloaded = importlib.reload(q192)
    assert reloaded._integrity(reloaded._blocked()) is True
    canonical_calls = 0

    def counted_throwing_builder() -> dict[str, Any]:
        nonlocal canonical_calls
        canonical_calls += 1
        raise AssertionError("canonical must stay local")

    monkeypatch.setattr(
        reloaded,
        "build_preview_block_q_windows_build_environment_inventory",
        counted_throwing_builder,
    )
    nominal = reloaded._canonical_nominal()
    assert nominal["source_19_1_accepted"] is True
    assert reloaded._integrity(nominal) is True
    assert canonical_calls == 0
    monkeypatch.undo()
    clean = importlib.reload(q192)
    assert (
        clean.build_preview_block_q_windows_build_environment_matrix()["source_19_1_accepted"]
        is True
    )
    assert (
        clean.build_preview_block_q_windows_build_environment_matrix()["source_19_1_accepted"]
        is True
    )
    monkeypatch.setattr(
        clean, "build_preview_block_q_windows_build_environment_inventory", lambda: {}
    )
    assert (
        clean.build_preview_block_q_windows_build_environment_matrix()["source_19_1_accepted"]
        is False
    )
    monkeypatch.undo()
    assert (
        importlib.reload(clean).build_preview_block_q_windows_build_environment_matrix()[
            "source_19_1_accepted"
        ]
        is True
    )


def test_canonical_nominal_fresh() -> None:
    one = _canonical_nominal()
    two = _canonical_nominal()
    assert one == two and one is not two
    one["environment_matrix_rows"][0]["matrix_state"] = "changed"
    assert two["environment_matrix_rows"][0]["matrix_state"] == "blocked"
