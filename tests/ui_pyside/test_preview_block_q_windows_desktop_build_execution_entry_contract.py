from __future__ import annotations

import ast
import copy
import importlib
import json
from pathlib import Path
from typing import Any

import pytest

from ui.pyside_app.preview_block_p_closure_audit import build_preview_block_p_closure_audit
from ui.pyside_app.preview_block_q_windows_desktop_build_execution_entry_contract import (
    AUTHORIZATION_FALSE_FIELDS,
    BLOCKED_FIELDS,
    BLOCKED_STATUS,
    EVIDENCE_IDS,
    NEXT_STEP,
    NEXT_STEP_TITLE,
    SOURCE_18_8_IDENTITY_VALUES,
    SOURCE_18_8_TOP_LEVEL_FIELDS,
    TOP_LEVEL_FIELDS,
    _blocked,
    _canonical_blocked,
    _canonical_nominal,
    _exact_plain,
    _integrity,
    _nominal,
    _source_accepted,
    _trusted_source_stub,
    build_preview_block_q_windows_desktop_build_execution_entry_contract,
)
import ui.pyside_app.preview_block_q_windows_desktop_build_execution_entry_contract as q19


class EqualityBomb:
    calls = 0

    def __eq__(self, other: object) -> bool:
        type(self).calls += 1
        raise AssertionError("equality called")


class HashBomb:
    calls = 0

    def __hash__(self) -> int:
        type(self).calls += 1
        raise AssertionError("hash called")


class UnhashableEqual:
    calls = 0
    __hash__ = None

    def __eq__(self, other: object) -> bool:
        type(self).calls += 1
        raise AssertionError("equality called")


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
        raise AssertionError("custom __getitem__ must not be called")


class BombKey(str):
    calls = 0

    def __hash__(self) -> int:
        type(self).calls += 1
        raise AssertionError("hash called")

    def __eq__(self, other: object) -> bool:
        type(self).calls += 1
        raise AssertionError("eq called")


def _via_builder(monkeypatch: pytest.MonkeyPatch, source: Any) -> dict[str, Any]:
    monkeypatch.setattr(q19, "build_preview_block_p_closure_audit", lambda: source)
    return build_preview_block_q_windows_desktop_build_execution_entry_contract()


def _assert_nominal(payload: dict[str, Any]) -> None:
    assert payload["source_18_8_accepted"] is True
    assert payload["ready_for_block_q_1"] is True
    assert q19._integrity(payload) is True


def _assert_blocked(payload: dict[str, Any]) -> None:
    assert payload["source_18_8_accepted"] is False
    assert payload["ready_for_block_q_1"] is False
    assert q19._integrity(payload) is True


def _run_scoped_collection_safety_reload(monkeypatch: pytest.MonkeyPatch) -> None:
    import ui.pyside_app.preview_block_p_closure_audit as p18

    with monkeypatch.context() as scoped:
        scoped.setattr(
            p18,
            "build_preview_block_p_closure_audit",
            lambda: (_ for _ in ()).throw(AssertionError("builder")),
        )
        scoped.setattr(
            p18, "_nominal", lambda source: (_ for _ in ()).throw(AssertionError("nominal"))
        )
        scoped.setattr(p18, "_blocked", lambda: (_ for _ in ()).throw(AssertionError("blocked")))
        reloaded = importlib.reload(q19)
        assert reloaded.SOURCE_18_8_TOP_LEVEL_FIELDS == SOURCE_18_8_TOP_LEVEL_FIELDS
        _assert_no_module_level_canonical_factory_calls()

    clean = importlib.reload(q19)
    payload = clean.build_preview_block_q_windows_desktop_build_execution_entry_contract()
    assert payload["status"] == clean.STATUS
    _assert_nominal(payload)


def _assert_no_module_level_canonical_factory_calls() -> None:
    tree = ast.parse(
        Path(
            "ui/pyside_app/preview_block_q_windows_desktop_build_execution_entry_contract.py"
        ).read_text()
    )
    forbidden_calls = []
    for node in tree.body:
        if isinstance(node, ast.Assign | ast.AnnAssign):
            value = node.value
            if isinstance(value, ast.Call) and isinstance(value.func, ast.Name):
                if value.func.id in {"_nominal", "_blocked"}:
                    forbidden_calls.append(value.func.id)
    assert forbidden_calls == []


def test_production_has_no_mutable_canonical_payload_globals() -> None:
    assert not hasattr(q19, "_CANONICAL_NOMINAL")
    assert not hasattr(q19, "_CANONICAL_BLOCKED")
    for name in (
        "_CANONICAL_NOMINAL",
        "_CANONICAL_BLOCKED",
        "CANONICAL_NOMINAL",
        "CANONICAL_BLOCKED",
        "TRUSTED_NOMINAL_PAYLOAD",
        "TRUSTED_BLOCKED_PAYLOAD",
    ):
        assert not isinstance(getattr(q19, name, None), dict)


def test_fresh_canonical_factory_outputs_are_isolated() -> None:
    immutable_spec_before = SOURCE_18_8_IDENTITY_VALUES
    blocked_one = _blocked()
    blocked_two = _blocked()
    blocked_three = _canonical_blocked()
    assert id(blocked_one) != id(blocked_two) != id(blocked_three)
    for key in (
        "build_execution_authorization_state",
        "non_execution_entry_evidence",
        "entry_contract_boundaries",
    ):
        assert id(blocked_one[key]) != id(blocked_two[key])
    assert id(blocked_one["future_steps"]) != id(blocked_two["future_steps"])
    blocked_one["build_execution_authorization_state"]["packaging_authorized"] = True
    assert blocked_two["build_execution_authorization_state"]["packaging_authorized"] is False
    assert blocked_three["build_execution_authorization_state"]["packaging_authorized"] is False
    assert _integrity(blocked_two) is True
    assert _integrity(blocked_three) is True

    nominal_one = _nominal(_trusted_source_stub())
    nominal_two = _nominal(_trusted_source_stub())
    nominal_three = _canonical_nominal()
    assert id(nominal_one) != id(nominal_two) != id(nominal_three)
    assert id(nominal_one["block_p_closure_audit_reference"]["source_top_level_fields"]) != id(
        nominal_two["block_p_closure_audit_reference"]["source_top_level_fields"]
    )
    assert id(nominal_one["build_execution_evidence_requirements"]) != id(
        nominal_two["build_execution_evidence_requirements"]
    )
    assert id(nominal_one["build_execution_evidence_requirements"][0]) != id(
        nominal_two["build_execution_evidence_requirements"][0]
    )
    for key in (
        "build_execution_authorization_state",
        "non_execution_entry_evidence",
        "entry_contract_boundaries",
    ):
        assert id(nominal_one[key]) != id(nominal_two[key])
    assert id(nominal_one["future_steps"]) != id(nominal_two["future_steps"])
    assert id(nominal_one["future_steps"][0]) != id(nominal_two["future_steps"][0])
    nominal_one["block_p_closure_audit_reference"]["source_top_level_fields"].append("tampered")
    nominal_one["build_execution_evidence_requirements"][0]["collected"] = True
    nominal_one["future_steps"][0]["physical_build_performed"] = True
    assert nominal_two["block_p_closure_audit_reference"]["source_top_level_fields"] == list(
        SOURCE_18_8_TOP_LEVEL_FIELDS
    )
    assert nominal_two["build_execution_evidence_requirements"][0]["collected"] is False
    assert nominal_two["future_steps"][0]["physical_build_performed"] is False
    assert nominal_three["build_execution_evidence_requirements"][0]["collected"] is False
    assert _integrity(nominal_two) is True
    assert _integrity(nominal_three) is True
    assert SOURCE_18_8_IDENTITY_VALUES == immutable_spec_before


def test_legacy_canonical_name_contamination_does_not_affect_integrity_or_builder(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(q19, "_CANONICAL_NOMINAL", {"tampered": True}, raising=False)
    monkeypatch.setattr(q19, "_CANONICAL_BLOCKED", {"tampered": True}, raising=False)
    assert _integrity(_nominal(_trusted_source_stub())) is True
    assert _integrity(_blocked()) is True
    nominal = build_preview_block_q_windows_desktop_build_execution_entry_contract()
    assert nominal["source_18_8_accepted"] is True
    rejected_source = build_preview_block_p_closure_audit()
    rejected_source["closure_decision"]["physical_build_completed"] = True
    blocked = _via_builder(monkeypatch, rejected_source)
    assert blocked == _blocked()
    assert "tampered" not in json.dumps(nominal)
    assert "tampered" not in json.dumps(blocked)


def test_blocked_source_flow_calls_source_once_and_returns_fresh_blocked(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = build_preview_block_p_closure_audit()
    source["closure_summary"]["source_18_7_accepted"] = False
    calls = 0

    def fake_builder() -> dict[str, Any]:
        nonlocal calls
        calls += 1
        return source

    monkeypatch.setattr(q19, "build_preview_block_p_closure_audit", fake_builder)
    payload = build_preview_block_q_windows_desktop_build_execution_entry_contract()
    fresh_blocked = _blocked()
    assert calls == 1
    assert payload == fresh_blocked
    assert _integrity(payload) is True
    assert json.loads(json.dumps(payload)) == payload
    assert id(payload) != id(fresh_blocked)


def test_nominal_builder_calls_source_once_and_preserves_source(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = build_preview_block_p_closure_audit()
    before = copy.deepcopy(source)
    calls = 0

    def fake_builder() -> dict[str, Any]:
        nonlocal calls
        calls += 1
        return source

    monkeypatch.setattr(q19, "build_preview_block_p_closure_audit", fake_builder)
    payload = build_preview_block_q_windows_desktop_build_execution_entry_contract()
    assert calls == 1
    assert source == before
    assert _source_accepted(source) is True
    assert _integrity(payload) is True
    assert json.loads(json.dumps(payload)) == payload
    assert payload == build_preview_block_q_windows_desktop_build_execution_entry_contract()
    assert tuple(payload) == TOP_LEVEL_FIELDS
    assert payload["block"] == "Q"
    assert payload["step"] == "19.0"
    assert payload["ready_for_block_q_1"] is True
    assert payload["next_step"] == NEXT_STEP
    assert payload["next_step_title"] == NEXT_STEP_TITLE
    for section in (
        "block_q_scope_definition",
        "non_execution_entry_evidence",
        "entry_contract_boundaries",
    ):
        for key, value in payload[section].items():
            if any(
                token in key
                for token in ("build", "packaging", "artifact", "release", "runtime", "orders")
            ) and key not in {
                "windows_desktop_build_execution_path_defined",
                "future_explicit_build_evidence_collection",
                "future_explicit_build_authorization",
                "future_explicit_build_command",
                "future_explicit_artifact_validation",
                "future_explicit_windows_build_environment_inventory",
                "source_only",
                "can_feed_only_19_1_windows_build_environment_inventory",
            }:
                assert value is False
    assert len(payload["build_execution_evidence_requirements"]) == len(EVIDENCE_IDS) == 13


def test_capability_audit_exact_type_first_adversarial_containers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    GetItemBombDict.getitem_calls = 0
    source = build_preview_block_p_closure_audit()
    source["capability_audit"] = GetItemBombDict(source["capability_audit"])
    payload = _via_builder(monkeypatch, source)
    assert payload == _blocked()
    assert GetItemBombDict.getitem_calls == 0

    source = build_preview_block_p_closure_audit()
    source["capability_audit"] = DictSubclass(source["capability_audit"])
    assert _via_builder(monkeypatch, source) == _blocked()

    source = build_preview_block_p_closure_audit()
    source["capability_audit"]["capability_state"] = DictSubclass(
        source["capability_audit"]["capability_state"]
    )
    assert _via_builder(monkeypatch, source) == _blocked()


def test_stage_steps_exact_type_first_adversarial_values(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    EqualityBomb.calls = 0
    cases = []
    source = build_preview_block_p_closure_audit()
    source["stage_audit_rows"][0]["step"] = EqualityBomb()
    cases.append(source)

    source = build_preview_block_p_closure_audit()
    source["stage_audit_rows"][0]["step"] = StrSubclass("18.0")
    cases.append(source)

    source = build_preview_block_p_closure_audit()
    source["stage_audit_rows"][0]["step"] = 18
    cases.append(source)

    source = build_preview_block_p_closure_audit()
    source["stage_audit_rows"][0] = DictSubclass(source["stage_audit_rows"][0])
    cases.append(source)

    source = build_preview_block_p_closure_audit()
    source["stage_audit_rows"] = ListSubclass(source["stage_audit_rows"])
    cases.append(source)

    for source in cases:
        assert _via_builder(monkeypatch, source) == _blocked()
    assert EqualityBomb.calls == 0


def test_order_isolation_nominal_collection_reload_blocked_nominal(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _assert_nominal(build_preview_block_q_windows_desktop_build_execution_entry_contract())
    _run_scoped_collection_safety_reload(monkeypatch)
    _assert_nominal(q19.build_preview_block_q_windows_desktop_build_execution_entry_contract())

    blocked_source = build_preview_block_p_closure_audit()
    blocked_source["closure_summary"]["source_18_7_accepted"] = False
    blocked = _via_builder(monkeypatch, blocked_source)
    _assert_blocked(blocked)

    importlib.reload(q19)
    _assert_nominal(q19.build_preview_block_q_windows_desktop_build_execution_entry_contract())


def test_source_acceptance_rejects_mutations(monkeypatch: pytest.MonkeyPatch) -> None:
    assert _source_accepted(build_preview_block_p_closure_audit()) is True
    blocked_source = build_preview_block_p_closure_audit()
    blocked_source["source_18_7_accepted"] = False
    assert _source_accepted(blocked_source) is False
    cases: list[Any] = []
    src = build_preview_block_p_closure_audit()
    missing = copy.deepcopy(src)
    missing.pop("schema_version")
    cases.append(missing)
    extra = copy.deepcopy(src)
    extra["extra"] = True
    cases.append(extra)
    reordered = {"block": src["block"], **{k: v for k, v in src.items() if k != "block"}}
    cases.append(reordered)
    for key, value in (
        ("schema_version", "x"),
        ("block_p_closure_audit_kind", "x"),
        ("block", "Q"),
        ("step", "19.0"),
        ("status", "x"),
    ):
        m = copy.deepcopy(src)
        m[key] = value
        cases.append(m)
    for key in (
        "source_18_7_accepted",
        "closure_audit_complete",
        "block_p_source_only_design_closed",
    ):
        m = copy.deepcopy(src)
        m["closure_summary"][key] = False
        cases.append(m)
    m = copy.deepcopy(src)
    m["integrity_valid"] = False
    cases.append(m)
    m = copy.deepcopy(src)
    m["closure_decision"]["physical_build_completed"] = True
    cases.append(m)
    for key in AUTHORIZATION_FALSE_FIELDS:
        m = copy.deepcopy(src)
        m["authorization_audit"][key] = True
        cases.append(m)
    m = copy.deepcopy(src)
    m["capability_audit"]["capability_state"]["runtime"] = "ready"
    cases.append(m)
    m = copy.deepcopy(src)
    m["future_steps"] = ["x"]
    cases.append(m)
    for case in cases:
        payload = _via_builder(monkeypatch, case)
        assert tuple(payload) == BLOCKED_FIELDS
        assert payload["status"] == BLOCKED_STATUS
        assert _integrity(payload) is True


def test_builder_exceptions_return_blocked(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        q19, "build_preview_block_p_closure_audit", lambda: (_ for _ in ()).throw(RuntimeError("x"))
    )
    assert build_preview_block_q_windows_desktop_build_execution_entry_contract() == _blocked()
    monkeypatch.setattr(
        q19, "build_preview_block_p_closure_audit", build_preview_block_p_closure_audit
    )
    monkeypatch.setattr(
        q19, "_integrity_18_8", lambda source: (_ for _ in ()).throw(RuntimeError("x"))
    )
    assert build_preview_block_q_windows_desktop_build_execution_entry_contract() == _blocked()


def test_blocked_integrity_and_mutations_rejected() -> None:
    payload = _blocked()
    assert payload == _blocked()
    assert json.loads(json.dumps(payload)) == payload
    assert _integrity(payload) is True
    for mutate in (
        lambda p: p.pop("schema_version"),
        lambda p: p.__setitem__("extra", True),
        lambda p: {"block": p["block"], **{k: v for k, v in p.items() if k != "block"}},
        lambda p: p["build_execution_authorization_state"].__setitem__(
            "packaging_authorized", True
        ),
    ):
        p = copy.deepcopy(payload)
        result = mutate(p)
        assert _integrity(result if isinstance(result, dict) else p) is False


def test_exact_plain_cycles_and_depth() -> None:
    a: list[Any] = []
    b: list[Any] = []
    a.append(a)
    b.append(b)
    assert _exact_plain(a, b) is True
    c: list[Any] = [1]
    c.append(c)
    assert _exact_plain(c, b) is False
    d: dict[str, Any] = {}
    e: dict[str, Any] = {}
    d["x"] = d
    e["x"] = e
    assert _exact_plain(d, e) is True
    f: dict[str, Any] = {"x": 1}
    f["y"] = f
    assert _exact_plain(f, e) is False
    actual: Any = "leaf"
    expected: Any = "leaf"
    for _ in range(1500):
        actual = [actual]
        expected = [expected]
    assert _exact_plain(actual, expected) is True
    actual[0][0] = "bad"
    assert _exact_plain(actual, expected) is False


def test_adversarial_fail_closed_without_custom_calls(monkeypatch: pytest.MonkeyPatch) -> None:
    src = build_preview_block_p_closure_audit()
    variants = []
    root_key = copy.deepcopy(src)
    root_key[StrSubclass("schema_version")] = root_key.pop("schema_version")
    variants.append(root_key)
    root_scalar = copy.deepcopy(src)
    root_scalar["schema_version"] = StrSubclass(root_scalar["schema_version"])
    variants.append(root_scalar)
    nested_auth = copy.deepcopy(src)
    nested_auth["authorization_audit"]["build_authorized"] = EqualityBomb()
    variants.append(nested_auth)
    nested_cap = copy.deepcopy(src)
    nested_cap["capability_audit"]["capability_state"]["runtime"] = StrSubclass("blocked")
    variants.append(nested_cap)
    variants.append(DictSubclass(src))
    nested_dict = copy.deepcopy(src)
    nested_dict["authorization_audit"] = DictSubclass(nested_dict["authorization_audit"])
    variants.append(nested_dict)
    nested_list = copy.deepcopy(src)
    nested_list["stage_audit_rows"] = ListSubclass(nested_list["stage_audit_rows"])
    variants.append(nested_list)
    variants.extend([HashBomb(), UnhashableEqual()])
    assert _source_accepted([(BombKey("schema_version"), "x")]) is False
    for variant in variants:
        assert _via_builder(monkeypatch, variant)["status"] == BLOCKED_STATUS
    assert EqualityBomb.calls == 0
    assert HashBomb.calls == 0
    assert UnhashableEqual.calls == 0
    assert BombKey.calls == 0


def test_collection_safety_import_does_not_call_upstream_private_or_builder(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _run_scoped_collection_safety_reload(monkeypatch)
