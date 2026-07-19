from __future__ import annotations

import ast
import copy
import importlib
import json
from pathlib import Path
from typing import Any

import pytest

from ui.pyside_app.preview_block_q_windows_desktop_build_execution_entry_contract import (
    build_preview_block_q_windows_desktop_build_execution_entry_contract,
)
from ui.pyside_app.preview_block_q_windows_build_environment_inventory import (
    AUTHORIZATION_FALSE_FIELDS,
    BLOCKED_STATUS,
    ENVIRONMENT_INVENTORY_SPECS,
    ENVIRONMENT_REQUIREMENT_TRUE_FIELDS,
    SOURCE_19_0_TOP_LEVEL_FIELDS,
    TOP_LEVEL_FIELDS,
    _blocked,
    _canonical_blocked,
    _canonical_nominal,
    _environment_inventory_rows,
    _exact_plain,
    _integrity,
    _source_accepted,
    build_preview_block_q_windows_build_environment_inventory,
)
import ui.pyside_app.preview_block_q_windows_build_environment_inventory as q191


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
        q191, "build_preview_block_q_windows_desktop_build_execution_entry_contract", lambda: source
    )
    return build_preview_block_q_windows_build_environment_inventory()


def _assert_blocked(payload: dict[str, Any]) -> None:
    assert payload == _canonical_blocked()
    assert payload["status"] == BLOCKED_STATUS
    assert _integrity(payload) is True


def test_nominal_builder_once_source_unchanged_order_counts_json_deterministic_integrity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = 0
    source = build_preview_block_q_windows_desktop_build_execution_entry_contract()
    before = copy.deepcopy(source)

    def fake() -> dict[str, Any]:
        nonlocal calls
        calls += 1
        return source

    monkeypatch.setattr(
        q191, "build_preview_block_q_windows_desktop_build_execution_entry_contract", fake
    )
    payload = build_preview_block_q_windows_build_environment_inventory()
    assert calls == 1
    assert source == before
    assert _source_accepted(source) is True
    assert list(payload) == list(TOP_LEVEL_FIELDS)
    assert [row["inventory_id"] for row in payload["environment_inventory_rows"]] == [
        spec[0] for spec in ENVIRONMENT_INVENTORY_SPECS
    ]
    assert len(set(row["inventory_id"] for row in payload["environment_inventory_rows"])) == 11
    assert all(
        list(row)
        == [
            "inventory_id",
            "requirement_field",
            "category",
            "required",
            "declared_by_19_0",
            "collection_status",
            "validation_status",
            "resolution_status",
            "observed_value",
            "source_only_definition",
        ]
        for row in payload["environment_inventory_rows"]
    )
    assert payload["environment_inventory_summary"] == {
        "inventory_row_count": 11,
        "required_count": 11,
        "collected_count": 0,
        "validated_count": 0,
        "resolved_count": 0,
        "blocked_count": 11,
        "environment_observation_complete": False,
        "inventory_definition_complete": True,
    }
    assert payload["ready_for_block_q_2"] is True
    assert payload["future_steps"] == [
        {
            "next_step": "FUNCTIONAL-PREVIEW-19.2",
            "next_step_title": "WINDOWS BUILD ENVIRONMENT MATRIX",
            "source_only": True,
            "environment_scan_performed": False,
            "physical_build_performed": False,
        }
    ]
    for key in AUTHORIZATION_FALSE_FIELDS:
        assert payload["build_execution_authorization_state"][key] is False
    json.dumps(payload)
    assert payload == build_preview_block_q_windows_build_environment_inventory()
    assert _integrity(payload) is True


@pytest.mark.parametrize(
    "mutate",
    [
        lambda s: s.pop("schema_version"),
        lambda s: s.update({"extra": True}),
        lambda s: s.update({"schema_version": s.pop("schema_version")}),
        lambda s: s.__setitem__("schema_version", "wrong"),
        lambda s: s.__setitem__(
            "block_q_windows_desktop_build_execution_entry_contract_kind", "wrong"
        ),
        lambda s: s.__setitem__("block", "P"),
        lambda s: s.__setitem__("step", "19.x"),
        lambda s: s.__setitem__("status", "wrong"),
        lambda s: s.__setitem__("source_18_8_accepted", False),
        lambda s: s.__setitem__("entry_contract_artifact_complete", False),
        lambda s: s.__setitem__("ready_for_block_q_1", False),
        lambda s: s.__setitem__("next_step", "x"),
        lambda s: s.__setitem__("next_step_title", "x"),
        lambda s: s["windows_build_environment_requirements"].__setitem__(
            "windows_host_required", False
        ),
        lambda s: s["windows_build_environment_requirements"].__setitem__(
            "qml_load_performed", True
        ),
        lambda s: s["build_execution_authorization_state"].__setitem__(
            "environment_inventory_authorized", False
        ),
        lambda s: s["build_execution_authorization_state"].__setitem__(
            "packaging_authorized", True
        ),
        lambda s: s["entry_contract_boundaries"].__setitem__("source_only", False),
        lambda s: s.__setitem__("future_steps", []),
        lambda s: s["future_steps"].append(dict(s["future_steps"][0])),
        lambda s: s["future_steps"][0].__setitem__("next_step", "x"),
        lambda s: s.__setitem__("integrity_valid", False),
    ],
)
def test_source_mutations_return_canonical_blocked(
    monkeypatch: pytest.MonkeyPatch, mutate: Any
) -> None:
    source = build_preview_block_q_windows_desktop_build_execution_entry_contract()
    mutate(source)
    _assert_blocked(_via_builder(monkeypatch, source))


def test_integrity_19_0_exception_blocks(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        q191, "_integrity_19_0", lambda source: (_ for _ in ()).throw(AssertionError("boom"))
    )
    _assert_blocked(
        _via_builder(
            monkeypatch, build_preview_block_q_windows_desktop_build_execution_entry_contract()
        )
    )


def test_canonical_blocked_fresh_json_deterministic_and_rejects_mutations() -> None:
    one = _blocked()
    two = _blocked()
    assert one == two
    assert id(one) != id(two)
    assert id(one["build_execution_authorization_state"]) != id(
        two["build_execution_authorization_state"]
    )
    json.dumps(one)
    assert _integrity(one) is True
    for mutation in (
        lambda p: p.pop("schema_version"),
        lambda p: p.update({"extra": True}),
        lambda p: p.update({"schema_version": p.pop("schema_version")}),
        lambda p: p["build_execution_authorization_state"].__setitem__("orders_authorized", True),
        lambda p: p["environment_inventory_scope"].__setitem__("inventory_row_count", 11),
        lambda p: p["environment_inventory_scope"].__setitem__("inventory_defined", True),
        lambda p: p["block_q_19_0_entry_contract_reference"].__setitem__(
            "source_18_8_accepted", True
        ),
        lambda p: p["source_entry_contract_preservation"].__setitem__(
            "preserves_19_0_payload", True
        ),
        lambda p: p["environment_inventory_summary"].__setitem__("inventory_row_count", 1),
        lambda p: p["environment_inventory_rows"].append({}),
        lambda p: p["future_steps"].append({}),
        lambda p: p["build_execution_authorization_state"].__setitem__(
            "environment_matrix_definition_authorized", True
        ),
    ):
        changed = _blocked()
        mutation(changed)
        assert _integrity(changed) is False


def test_blocked_factory_is_independent_from_nominal(monkeypatch: pytest.MonkeyPatch) -> None:
    def throwing_nominal(source: dict[str, Any]) -> dict[str, Any]:
        raise AssertionError("blocked must not call nominal")

    with monkeypatch.context() as scoped:
        scoped.setattr(q191, "_nominal", throwing_nominal)
        payload = q191._blocked()
        assert payload["status"] == q191.BLOCKED_STATUS
        assert payload["block_q_19_0_entry_contract_reference"] == {
            "schema_version": "",
            "kind": "",
            "block": "Q",
            "step": "19.0",
            "status": "",
            "source_18_8_accepted": False,
            "entry_contract_artifact_complete": False,
            "ready_for_block_q_1": False,
            "integrity_valid": False,
            "source_top_level_fields": [],
        }
        assert payload["environment_inventory_scope"]["inventory_row_count"] == 0

    assert q191._integrity(payload) is True


def test_public_builder_returns_blocked_when_nominal_exception(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = build_preview_block_q_windows_desktop_build_execution_entry_contract()
    calls = 0

    def fake_source() -> dict[str, Any]:
        nonlocal calls
        calls += 1
        return source

    def throwing_nominal(source_payload: dict[str, Any]) -> dict[str, Any]:
        raise AssertionError("nominal exception")

    monkeypatch.setattr(
        q191, "build_preview_block_q_windows_desktop_build_execution_entry_contract", fake_source
    )
    with monkeypatch.context() as scoped:
        scoped.setattr(q191, "_nominal", throwing_nominal)
        payload = q191.build_preview_block_q_windows_build_environment_inventory()
        assert calls == 1
        assert payload == q191._blocked()

    assert q191._integrity(payload) is True
    json.dumps(payload)


def test_nominal_and_blocked_cross_field_coherence() -> None:
    nominal = build_preview_block_q_windows_build_environment_inventory()
    rows = nominal["environment_inventory_rows"]
    summary = nominal["environment_inventory_summary"]
    scope = nominal["environment_inventory_scope"]
    assert len(rows) == 11
    assert summary["inventory_row_count"] == len(rows)
    assert scope["inventory_row_count"] == len(rows)
    assert summary["required_count"] == 11
    assert summary["blocked_count"] == 11
    assert scope["inventory_defined"] is True

    blocked = _blocked()
    rows = blocked["environment_inventory_rows"]
    summary = blocked["environment_inventory_summary"]
    scope = blocked["environment_inventory_scope"]
    assert rows == []
    assert summary["inventory_row_count"] == len(rows) == 0
    assert scope["inventory_row_count"] == len(rows) == 0
    assert summary["required_count"] == 0
    assert summary["blocked_count"] == 0
    assert scope["inventory_defined"] is False
    assert scope["source_only_inventory"] is False


def test_blocked_positive_claim_audit() -> None:
    payload = _blocked()
    ref = payload["block_q_19_0_entry_contract_reference"]
    for key in (
        "source_18_8_accepted",
        "entry_contract_artifact_complete",
        "ready_for_block_q_1",
        "integrity_valid",
    ):
        assert ref[key] is False
    assert ref["source_top_level_fields"] == []

    preservation = payload["source_entry_contract_preservation"]
    for key in (
        "preserves_19_0_payload",
        "preserves_19_0_environment_requirements",
        "preserves_19_0_zero_execution_authorizations",
        "preserves_19_0_non_execution_state",
        "preserves_source_only_handoff",
    ):
        assert preservation[key] is False

    scope = payload["environment_inventory_scope"]
    assert scope["inventory_defined"] is False
    assert scope["source_only_inventory"] is False

    auth = payload["build_execution_authorization_state"]
    assert auth["environment_matrix_definition_authorized"] is False
    assert auth["only_source_only_19_2_handoff_allowed"] is False
    assert all(value is False for value in auth.values())
    assert payload["environment_inventory_rows"] == []
    assert payload["future_steps"] == []


def test_blocked_freshness_all_nested_containers() -> None:
    one = _blocked()
    two = _blocked()
    assert id(one) != id(two)
    for key in (
        "block_q_19_0_entry_contract_reference",
        "source_entry_contract_preservation",
        "environment_inventory_scope",
        "environment_inventory_rows",
        "environment_inventory_summary",
        "build_execution_authorization_state",
        "non_execution_inventory_evidence",
        "inventory_boundaries",
        "source_boundaries",
        "future_steps",
    ):
        assert id(one[key]) != id(two[key])
    assert id(one["block_q_19_0_entry_contract_reference"]["source_top_level_fields"]) != id(
        two["block_q_19_0_entry_contract_reference"]["source_top_level_fields"]
    )


def test_inventory_specs_and_factories_are_fresh() -> None:
    assert type(ENVIRONMENT_INVENTORY_SPECS) is tuple
    assert len(ENVIRONMENT_INVENTORY_SPECS) == 11
    ids = [spec[0] for spec in ENVIRONMENT_INVENTORY_SPECS]
    reqs = [spec[1] for spec in ENVIRONMENT_INVENTORY_SPECS]
    assert len(set(ids)) == 11
    assert len(set(reqs)) == 11
    assert tuple(reqs) == ENVIRONMENT_REQUIREMENT_TRUE_FIELDS
    rows_one = _environment_inventory_rows()
    rows_two = _environment_inventory_rows()
    assert id(rows_one) != id(rows_two)
    assert id(rows_one[0]) != id(rows_two[0])
    rows_one[0]["required"] = False
    assert rows_two[0]["required"] is True
    payload_one = _canonical_nominal()
    payload_two = _canonical_nominal()
    payload_one["environment_inventory_rows"][0]["required"] = False
    assert payload_two["environment_inventory_rows"][0]["required"] is True


def _deep_list(depth: int, leaf: str) -> list[Any]:
    root: list[Any] = []
    cur = root
    for _ in range(depth):
        nxt: list[Any] = []
        cur.append(nxt)
        cur = nxt
    cur.append(leaf)
    return root


def test_comparator_cycles_and_depth_1500() -> None:
    a: list[Any] = []
    b: list[Any] = []
    a.append(a)
    b.append(b)
    assert _exact_plain(a, b) is True
    c: dict[str, Any] = {}
    d: dict[str, Any] = {}
    c["self"] = c
    d["self"] = d
    assert _exact_plain(c, d) is True
    d["x"] = 1
    assert _exact_plain(c, d) is False
    assert _exact_plain(_deep_list(1500, "ok"), _deep_list(1500, "ok")) is True
    assert _exact_plain(_deep_list(1500, "no"), _deep_list(1500, "ok")) is False


def test_adversarial_inputs_do_not_trigger_custom_counters(monkeypatch: pytest.MonkeyPatch) -> None:
    source = build_preview_block_q_windows_desktop_build_execution_entry_contract()
    _replace_key(source, "schema_version")
    assert _source_accepted(source) is False
    source = build_preview_block_q_windows_desktop_build_execution_entry_contract()
    _replace_key(source["windows_build_environment_requirements"], "windows_host_required")
    assert _via_builder(monkeypatch, source) == _canonical_blocked()
    source = build_preview_block_q_windows_desktop_build_execution_entry_contract()
    _replace_key(source["build_execution_authorization_state"], "packaging_authorized")
    assert _via_builder(monkeypatch, source) == _canonical_blocked()
    source = build_preview_block_q_windows_desktop_build_execution_entry_contract()
    _replace_key(source["future_steps"][0], "next_step")
    assert _via_builder(monkeypatch, source) == _canonical_blocked()
    bad = _canonical_nominal()
    bad["environment_inventory_rows"][0]["observed_value"] = EqualityBomb()
    assert _integrity(bad) is False
    assert _exact_plain(ListSubclass(), []) is False
    assert _exact_plain(DictSubclass(), {}) is False
    assert _source_accepted(GetItemBombDict(source)) is False
    ref_bad = _canonical_nominal()
    ref_bad["block_q_19_0_entry_contract_reference"]["step"] = StrSubclass("19.0")
    assert _integrity(ref_bad) is False
    assert EqualityBomb.calls == HashBomb.calls == UnhashableEqual.calls == 0
    assert GetItemBombDict.getitem_calls == 0
    assert ArmedBombKey.hash_calls == ArmedBombKey.equality_calls == 0


def test_collection_safety_and_test_order_isolation(monkeypatch: pytest.MonkeyPatch) -> None:
    import ui.pyside_app.preview_block_q_windows_desktop_build_execution_entry_contract as q190

    assert (
        build_preview_block_q_windows_build_environment_inventory()["source_19_0_accepted"] is True
    )
    with monkeypatch.context() as scoped:
        scoped.setattr(
            q190,
            "build_preview_block_q_windows_desktop_build_execution_entry_contract",
            lambda: (_ for _ in ()).throw(AssertionError("builder")),
        )
        scoped.setattr(
            q190, "_nominal", lambda source: (_ for _ in ()).throw(AssertionError("nominal"))
        )
        scoped.setattr(q190, "_blocked", lambda: (_ for _ in ()).throw(AssertionError("blocked")))
        scoped.setattr(
            q190, "_canonical_nominal", lambda: (_ for _ in ()).throw(AssertionError("canonical"))
        )
        scoped.setattr(
            q190, "_canonical_blocked", lambda: (_ for _ in ()).throw(AssertionError("canonical"))
        )
        importlib.reload(q191)
        tree = ast.parse(
            Path("ui/pyside_app/preview_block_q_windows_build_environment_inventory.py").read_text()
        )
        assert [
            node.value.func.id
            for node in tree.body
            if isinstance(node, ast.Assign | ast.AnnAssign)
            and isinstance(node.value, ast.Call)
            and isinstance(node.value.func, ast.Name)
            and node.value.func.id in {"_nominal", "_blocked"}
        ] == []
    clean = importlib.reload(q191)
    assert (
        clean.build_preview_block_q_windows_build_environment_inventory()["source_19_0_accepted"]
        is True
    )
    monkeypatch.setattr(
        clean,
        "build_preview_block_q_windows_desktop_build_execution_entry_contract",
        lambda: {"bad": True},
    )
    _assert_blocked(clean.build_preview_block_q_windows_build_environment_inventory())
    monkeypatch.undo()
    clean = importlib.reload(q191)
    assert (
        clean.build_preview_block_q_windows_build_environment_inventory()["source_19_0_accepted"]
        is True
    )
