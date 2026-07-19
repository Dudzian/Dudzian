from __future__ import annotations

import ast
import copy
import importlib
import json
from pathlib import Path
from typing import Any

import pytest

import ui.pyside_app.preview_block_q_windows_build_environment_contract as contract
import ui.pyside_app.preview_block_q_windows_build_environment_matrix as matrix


def _copy(value: Any) -> Any:
    return copy.deepcopy(value)


def _blocked_from(monkeypatch: pytest.MonkeyPatch, source: Any) -> dict[str, Any]:
    calls = {"count": 0}

    def fake() -> Any:
        calls["count"] += 1
        return source

    monkeypatch.setattr(contract, "build_preview_block_q_windows_build_environment_matrix", fake)
    payload = contract.build_preview_block_q_windows_build_environment_contract()
    assert calls["count"] == 1
    assert payload == contract._canonical_blocked()
    return payload


def test_nominal_contract_core_and_single_upstream_call(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = {"count": 0}
    source = matrix.build_preview_block_q_windows_build_environment_matrix()
    snapshot = _copy(source)

    def fake() -> dict[str, Any]:
        calls["count"] += 1
        return source

    monkeypatch.setattr(contract, "build_preview_block_q_windows_build_environment_matrix", fake)
    payload = contract.build_preview_block_q_windows_build_environment_contract()
    assert calls["count"] == 1
    assert source == snapshot
    assert contract._source_accepted(source) is True
    assert tuple(payload) == contract.TOP_LEVEL_FIELDS
    assert len(payload["environment_contract_rows"]) == 11
    assert [tuple(row) for row in payload["environment_contract_rows"]] == [
        contract.CONTRACT_ROW_FIELDS
    ] * 11
    assert [row["matrix_id"] for row in payload["environment_contract_rows"]] == [
        spec[0] for spec in contract.SOURCE_19_2_MATRIX_SPECS
    ]
    assert [row["contract_id"] for row in payload["environment_contract_rows"]] == [
        spec[0] for spec in contract.ENVIRONMENT_CONTRACT_SPECS
    ]
    assert len({row["contract_id"] for row in payload["environment_contract_rows"]}) == 11
    assert len({row["matrix_id"] for row in payload["environment_contract_rows"]}) == 11
    assert len({row["inventory_id"] for row in payload["environment_contract_rows"]}) == 11
    assert len({row["requirement_field"] for row in payload["environment_contract_rows"]}) == 11
    assert all(row["build_gate_open"] is False for row in payload["environment_contract_rows"])
    assert payload["environment_contract_summary"] == {
        "contract_row_count": 11,
        "required_count": 11,
        "satisfied_count": 0,
        "unsatisfied_count": 11,
        "open_gate_count": 0,
        "closed_gate_count": 11,
        "evidence_required_count": 11,
        "evidence_collected_count": 0,
        "evidence_validated_count": 0,
        "environment_observation_complete": False,
        "environment_contract_definition_complete": True,
        "environment_build_ready": False,
    }
    assert payload["source_matrix_preservation"]["preserves_19_2_payload"] is False
    assert payload["source_matrix_preservation"]["preserves_all_11_matrix_rows"] is True
    assert payload["source_matrix_preservation"]["preserves_matrix_order"] is True
    assert payload["source_matrix_preservation"]["preserves_one_to_one_mapping"] is True
    assert payload["source_matrix_preservation"]["source_matrix_reinterpreted"] is False
    assert payload["environment_build_ready"] is False
    assert payload["ready_for_block_q_4"] is True
    assert payload["next_step"] == "FUNCTIONAL-PREVIEW-19.4"
    assert all(
        value is False
        for key, value in payload["build_execution_authorization_state"].items()
        if key
        not in (
            "environment_read_model_definition_authorized",
            "only_source_only_19_4_handoff_allowed",
        )
    )
    json.dumps(payload, sort_keys=True)
    assert payload == contract._canonical_nominal()
    assert contract._integrity(payload) is True
    assert calls["count"] == 1
    assert contract.build_preview_block_q_windows_build_environment_contract() == payload
    assert calls["count"] == 2


@pytest.mark.parametrize(
    "mutate",
    [
        lambda s: s.pop("schema_version"),
        lambda s: s.__setitem__("extra", False),
        lambda s: s.__setitem__("schema_version", "wrong"),
        lambda s: s.__setitem__("source_19_1_accepted", False),
        lambda s: s.__setitem__("environment_matrix_artifact_complete", False),
        lambda s: s.__setitem__("environment_observation_complete", True),
        lambda s: s.__setitem__("environment_build_ready", True),
        lambda s: s.__setitem__("ready_for_block_q_3", False),
        lambda s: s.__setitem__("next_step", "bad"),
        lambda s: s.__setitem__("next_step_title", "bad"),
        lambda s: s["environment_matrix_rows"].pop(),
        lambda s: s["environment_matrix_rows"].append(_copy(s["environment_matrix_rows"][0])),
        lambda s: s["environment_matrix_rows"].reverse(),
        lambda s: s["environment_matrix_rows"][0].pop("matrix_id"),
        lambda s: s["environment_matrix_rows"][0].__setitem__("extra", False),
        lambda s: s["environment_matrix_rows"][0].__setitem__("matrix_id", "matrix_python_version"),
        lambda s: s["environment_matrix_rows"][0].__setitem__("inventory_id", "python_version"),
        lambda s: s["environment_matrix_rows"][0].__setitem__("requirement_field", "bad"),
        lambda s: s["environment_matrix_rows"][0].__setitem__("category", "bad"),
        lambda s: s["environment_matrix_rows"][0].__setitem__("required", 1),
        lambda s: s["environment_matrix_rows"][0].__setitem__("source_collection_status", 1),
        lambda s: s["environment_matrix_rows"][0].__setitem__("matrix_state", "ready"),
        lambda s: s["environment_matrix_rows"][0].__setitem__("blocker_code", "bad"),
        lambda s: s["environment_matrix_rows"][0].__setitem__("evidence_collected", True),
        lambda s: s["environment_matrix_summary"].__setitem__("matrix_row_count", 10),
        lambda s: s["environment_matrix_scope"].__setitem__("matrix_row_count", 10),
        lambda s: s["environment_matrix_scope"].__setitem__("source_only_matrix", False),
        lambda s: s["build_execution_authorization_state"].__setitem__(
            "environment_contract_definition_authorized", False
        ),
        lambda s: s["build_execution_authorization_state"].__setitem__(
            "packaging_authorized", True
        ),
        lambda s: s["future_steps"].pop(),
        lambda s: s["future_steps"].append(_copy(s["future_steps"][0])),
        lambda s: s["future_steps"][0].__setitem__("next_step", "bad"),
        lambda s: s.__setitem__("integrity_valid", False),
    ],
)
def test_source_acceptance_mutations_block(monkeypatch: pytest.MonkeyPatch, mutate: Any) -> None:
    source = matrix.build_preview_block_q_windows_build_environment_matrix()
    mutate(source)
    _blocked_from(monkeypatch, source)


def test_upstream_integrity_exception_blocks(monkeypatch: pytest.MonkeyPatch) -> None:
    source = matrix.build_preview_block_q_windows_build_environment_matrix()
    monkeypatch.setattr(
        contract, "_integrity_19_2", lambda _source: (_ for _ in ()).throw(RuntimeError)
    )
    _blocked_from(monkeypatch, source)


def test_local_trust_boundary_and_ast_guard(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        contract,
        "build_preview_block_q_windows_build_environment_matrix",
        lambda: (_ for _ in ()).throw(RuntimeError),
    )
    monkeypatch.setattr(
        contract, "_integrity_19_2", lambda _source: (_ for _ in ()).throw(RuntimeError)
    )
    nominal = contract._canonical_nominal()
    assert nominal["source_19_2_accepted"] is True
    assert contract._integrity(nominal) is True
    mutated_source = contract._trusted_source_stub()
    mutated_source["environment_matrix_rows"][0]["matrix_id"] = "mutated"
    assert contract._canonical_nominal() == nominal
    tree = ast.parse(Path(contract.__file__).read_text())
    guarded = {"_trusted_source_stub", "_canonical_nominal", "_integrity"}
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name in guarded:
            calls = [
                n.func.id
                for n in ast.walk(node)
                if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
            ]
            assert "build_preview_block_q_windows_build_environment_matrix" not in calls
            assert "_integrity_19_2" not in calls


def test_blocked_independent_fresh_and_integrity(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(contract, "_nominal", lambda _source: (_ for _ in ()).throw(RuntimeError))
    source = matrix.build_preview_block_q_windows_build_environment_matrix()
    payload = _blocked_from(monkeypatch, source)
    other = contract._blocked()
    assert payload == other
    assert payload is not other
    assert payload["environment_contract_rows"] is not other["environment_contract_rows"]
    assert payload["environment_contract_summary"]["contract_row_count"] == 0
    assert all(v is False for v in payload["source_matrix_preservation"].values())
    json.dumps(payload)
    assert contract._integrity(payload) is True
    changed = _copy(payload)
    changed["future_steps"] = [{}]
    assert contract._integrity(changed) is False


def test_contract_specs_and_fresh_rows() -> None:
    assert type(contract.ENVIRONMENT_CONTRACT_SPECS) is tuple
    assert len(contract.ENVIRONMENT_CONTRACT_SPECS) == 11
    assert [spec[1:] for spec in contract.ENVIRONMENT_CONTRACT_SPECS] == [
        spec for spec in contract.SOURCE_19_2_MATRIX_SPECS
    ]
    assert len({spec[0] for spec in contract.ENVIRONMENT_CONTRACT_SPECS}) == 11
    rows1 = contract._environment_contract_rows(contract._trusted_source_matrix_rows())
    rows2 = contract._environment_contract_rows(contract._trusted_source_matrix_rows())
    rows1[0]["contract_id"] = "mutated"
    assert rows2[0]["contract_id"] == "contract_windows_host"


def test_contract_rows_consume_passed_source_rows() -> None:
    source_rows = contract._trusted_source_matrix_rows()
    source_rows[0]["required"] = False
    source_rows[0]["matrix_state"] = "source_row_state_probe"
    source_rows[0]["blocker_code"] = "source_row_blocker_probe"
    source_rows[0]["evidence_required"] = False
    source_rows[0]["evidence_collected"] = True
    source_rows[0]["evidence_validated"] = True
    source_rows[0]["requirement_satisfied"] = True
    source_rows[0]["source_only_definition"] = False
    rows = contract._environment_contract_rows(source_rows)
    assert rows[0]["contract_id"] == "contract_windows_host"
    assert rows[0]["matrix_id"] == source_rows[0]["matrix_id"]
    assert rows[0]["inventory_id"] == source_rows[0]["inventory_id"]
    assert rows[0]["requirement_field"] == source_rows[0]["requirement_field"]
    assert rows[0]["category"] == source_rows[0]["category"]
    assert rows[0]["required"] is False
    assert rows[0]["source_matrix_state"] == "source_row_state_probe"
    assert rows[0]["source_blocker_code"] == "source_row_blocker_probe"
    assert rows[0]["source_evidence_required"] is False
    assert rows[0]["source_evidence_collected"] is True
    assert rows[0]["source_evidence_validated"] is True
    assert rows[0]["requirement_satisfied"] is True
    assert rows[0]["source_only_definition"] is False
    source_rows[0]["matrix_id"] = "mismatch"
    with pytest.raises(ValueError):
        contract._environment_contract_rows(source_rows)


def test_comparator_cycles_and_depth() -> None:
    a: list[Any] = []
    b: list[Any] = []
    a.append(a)
    b.append(b)
    assert contract._exact_plain(a, b) is True
    da: dict[str, Any] = {"x": None}
    db: dict[str, Any] = {"x": None}
    da["x"] = da
    db["x"] = db
    assert contract._exact_plain(da, db) is True
    c: list[Any] = [[]]
    c[0].append(c)
    assert contract._exact_plain(a, c) is False
    x_deepest: list[Any] = ["leaf"]
    y_deepest: list[Any] = ["leaf"]
    x: Any = x_deepest
    y: Any = y_deepest
    for _ in range(1500):
        x = [x]
        y = [y]
    assert contract._exact_plain(x, y) is True
    y_deepest[0] = "different_at_depth_1500"
    assert contract._exact_plain(x, y) is False


class EqualityBomb:
    calls = 0

    def __eq__(self, other: object) -> bool:
        type(self).calls += 1
        raise AssertionError


class HashBomb:
    calls = 0

    def __hash__(self) -> int:
        type(self).calls += 1
        raise AssertionError


class UnhashableEqual:
    calls = 0
    __hash__ = None

    def __eq__(self, other: object) -> bool:
        type(self).calls += 1
        raise AssertionError


class StrSubclass(str):
    calls = 0

    def __eq__(self, other: object) -> bool:
        type(self).calls += 1
        raise AssertionError


class ListSubclass(list[Any]):
    pass


class DictSubclass(dict[str, Any]):
    pass


class GetItemBombDict(dict[str, Any]):
    calls = 0

    def __getitem__(self, key: str) -> Any:
        type(self).calls += 1
        raise AssertionError


class ArmedBombKey(str):
    calls = 0
    __hash__ = str.__hash__

    def __eq__(self, other: object) -> bool:
        type(self).calls += 1
        raise AssertionError


def test_adversarial_fail_closed_without_custom_counters(monkeypatch: pytest.MonkeyPatch) -> None:
    for cls in (
        EqualityBomb,
        HashBomb,
        UnhashableEqual,
        StrSubclass,
        GetItemBombDict,
        ArmedBombKey,
    ):
        cls.calls = 0
    sources = []
    root_key = matrix.build_preview_block_q_windows_build_environment_matrix()
    root_key[ArmedBombKey("schema_version")] = root_key.pop("schema_version")
    sources.append(root_key)
    row_key = matrix.build_preview_block_q_windows_build_environment_matrix()
    row_key["environment_matrix_rows"][0][ArmedBombKey("matrix_id")] = row_key[
        "environment_matrix_rows"
    ][0].pop("matrix_id")
    sources.append(row_key)
    row_val = matrix.build_preview_block_q_windows_build_environment_matrix()
    row_val["environment_matrix_rows"][0]["matrix_id"] = StrSubclass("matrix_windows_host")
    sources.append(row_val)
    summary_key = matrix.build_preview_block_q_windows_build_environment_matrix()
    summary_key["environment_matrix_summary"][ArmedBombKey("matrix_row_count")] = summary_key[
        "environment_matrix_summary"
    ].pop("matrix_row_count")
    sources.append(summary_key)
    scope_key = matrix.build_preview_block_q_windows_build_environment_matrix()
    scope_key["environment_matrix_scope"][ArmedBombKey("matrix_defined")] = scope_key[
        "environment_matrix_scope"
    ].pop("matrix_defined")
    sources.append(scope_key)
    auth_key = matrix.build_preview_block_q_windows_build_environment_matrix()
    auth_key["build_execution_authorization_state"][ArmedBombKey("packaging_authorized")] = (
        auth_key["build_execution_authorization_state"].pop("packaging_authorized")
    )
    sources.append(auth_key)
    future_key = matrix.build_preview_block_q_windows_build_environment_matrix()
    future_key["future_steps"][0][ArmedBombKey("next_step")] = future_key["future_steps"][0].pop(
        "next_step"
    )
    sources.append(future_key)
    nested = matrix.build_preview_block_q_windows_build_environment_matrix()
    nested["environment_matrix_rows"] = ListSubclass(nested["environment_matrix_rows"])
    sources.append(nested)
    ref = matrix.build_preview_block_q_windows_build_environment_matrix()
    ref["schema_version"] = EqualityBomb()
    sources.append(ref)
    getitem = matrix.build_preview_block_q_windows_build_environment_matrix()
    getitem["environment_matrix_summary"] = GetItemBombDict(getitem["environment_matrix_summary"])
    sources.append(getitem)
    hash_value = matrix.build_preview_block_q_windows_build_environment_matrix()
    hash_value["environment_matrix_rows"][0]["matrix_id"] = HashBomb()
    sources.append(hash_value)
    unhashable_value = matrix.build_preview_block_q_windows_build_environment_matrix()
    unhashable_value["environment_matrix_rows"][0]["inventory_id"] = UnhashableEqual()
    sources.append(unhashable_value)
    dict_subclass = matrix.build_preview_block_q_windows_build_environment_matrix()
    dict_subclass["environment_matrix_rows"][0] = DictSubclass(
        dict_subclass["environment_matrix_rows"][0]
    )
    sources.append(dict_subclass)
    for source in sources:
        _blocked_from(monkeypatch, source)
    assert EqualityBomb.calls == 0
    assert HashBomb.calls == 0
    assert UnhashableEqual.calls == 0
    assert StrSubclass.calls == 0
    assert GetItemBombDict.calls == 0
    assert ArmedBombKey.calls == 0


def test_collection_safety_reload_and_order_isolation(monkeypatch: pytest.MonkeyPatch) -> None:
    reloaded = importlib.reload(contract)
    monkeypatch.setattr(
        reloaded,
        "build_preview_block_q_windows_build_environment_matrix",
        lambda: (_ for _ in ()).throw(RuntimeError),
    )
    assert reloaded._canonical_nominal()["source_19_2_accepted"] is True
    assert reloaded._integrity(reloaded._canonical_nominal()) is True
    clean = importlib.reload(reloaded)
    nominal = clean.build_preview_block_q_windows_build_environment_contract()
    monkeypatch.setattr(
        clean, "build_preview_block_q_windows_build_environment_matrix", lambda: {"bad": True}
    )
    blocked = clean.build_preview_block_q_windows_build_environment_contract()
    monkeypatch.setattr(
        clean,
        "build_preview_block_q_windows_build_environment_matrix",
        matrix.build_preview_block_q_windows_build_environment_matrix,
    )
    assert clean.build_preview_block_q_windows_build_environment_contract() == nominal
    assert blocked == clean._canonical_blocked()
