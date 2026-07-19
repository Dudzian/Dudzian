from __future__ import annotations
import ast, copy, importlib
from pathlib import Path
from typing import Any
import pytest
import ui.pyside_app.preview_block_q_windows_build_environment_contract as contract
import ui.pyside_app.preview_block_q_windows_build_environment_read_model as read_model


def _source() -> dict[str, Any]:
    return dict(contract.build_preview_block_q_windows_build_environment_contract())


def _built_from(monkeypatch: pytest.MonkeyPatch, source: Any) -> dict[str, Any]:
    calls = {"count": 0}

    def fake() -> Any:
        calls["count"] += 1
        return source

    monkeypatch.setattr(
        read_model, "build_preview_block_q_windows_build_environment_contract", fake
    )
    payload = read_model.build_preview_block_q_windows_build_environment_read_model()
    assert calls["count"] == 1
    return payload


def test_valid_source_nominal_handoff_blocked_single_upstream_call(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = _source()
    snap = copy.deepcopy(source)
    payload = _built_from(monkeypatch, source)
    assert source == snap
    assert tuple(payload) == read_model.TOP_LEVEL_FIELDS
    assert payload == read_model._canonical_nominal()
    assert payload != read_model._canonical_blocked()
    assert payload["source_19_3_accepted"] is True
    assert payload["environment_read_model_artifact_complete"] is True
    assert len(payload["environment_read_model_rows"]) == 11
    assert [r["contract_id"] for r in payload["environment_read_model_rows"]] == [
        r["contract_id"] for r in source["environment_contract_rows"]
    ]
    assert payload["next_step"] == ""
    assert payload["next_step_title"] == ""
    assert payload["environment_build_ready"] is False
    assert payload["environment_observation_complete"] is False
    assert payload["environment_read_model_summary"] == read_model._summary(True)
    assert payload["ready_for_block_q_5"] is False
    assert payload["future_steps"] == []
    assert payload["build_execution_authorization_state"]["next_step_contract_missing"] is True
    assert (
        payload["build_execution_authorization_state"]["source_only_next_step_authorized"] is False
    )
    assert read_model._integrity(payload) is True


def test_source_rejected_blocked_path(monkeypatch: pytest.MonkeyPatch) -> None:
    payload = _built_from(monkeypatch, {"bad": True})
    assert payload == read_model._canonical_blocked()
    assert payload["source_19_3_accepted"] is False
    assert payload["environment_read_model_rows"] == []
    assert payload["environment_read_model_artifact_complete"] is False
    assert payload["environment_read_model_summary"]["read_model_definition_complete"] is False


def test_canonical_nominal_and_blocked_are_distinct_and_fresh() -> None:
    nominal = read_model._canonical_nominal()
    blocked = read_model._canonical_blocked()
    other = read_model._canonical_nominal()
    assert nominal != blocked
    assert nominal is not other
    assert nominal["environment_read_model_rows"] is not other["environment_read_model_rows"]
    assert len(nominal["environment_read_model_rows"]) == 11
    assert blocked["environment_read_model_rows"] == []


def test_rows_consumes_source_rows_and_mismatch_cases() -> None:
    rows = read_model._contract_rows()
    rows[0]["source_only_definition"] = False
    with pytest.raises(ValueError):
        read_model._rows(rows)
    rows = read_model._contract_rows()
    built = read_model._rows(rows)
    assert built[0]["contract_id"] == rows[0]["contract_id"]
    assert built[0]["matrix_id"] == rows[0]["matrix_id"]
    assert built[0]["inventory_id"] == rows[0]["inventory_id"]
    assert built[0]["requirement_field"] == rows[0]["requirement_field"]
    assert built[0]["category"] == rows[0]["category"]
    for key in ("contract_id", "matrix_id", "inventory_id", "requirement_field", "category"):
        bad = read_model._contract_rows()
        bad[0][key] = "bad"
        with pytest.raises(ValueError):
            read_model._rows(bad)
    with pytest.raises(ValueError):
        read_model._rows(read_model._contract_rows()[:-1])
    with pytest.raises(ValueError):
        read_model._rows(tuple(read_model._contract_rows()))
    bad2 = read_model._contract_rows()
    bad2[0] = []  # type: ignore[call-overload]
    with pytest.raises(ValueError):
        read_model._rows(bad2)


def test_rows_rejects_all_previously_missing_field_mutations() -> None:
    for key, value in (
        ("source_matrix_state", "ready"),
        ("source_blocker_code", "wrong"),
        ("source_evidence_required", False),
        ("acceptance_rule", "wrong"),
        ("satisfaction_rule", "wrong"),
    ):
        rows = read_model._contract_rows()
        rows[0][key] = value
        with pytest.raises(ValueError):
            read_model._rows(rows)


def test_blocked_reference_uses_empty_source_top_level_fields() -> None:
    fields = read_model._canonical_blocked()["block_q_19_3_contract_reference"][
        "source_top_level_fields"
    ]
    assert fields == []
    assert "block_q_windows_build_environment_matrix_kind" not in fields
    assert "environment_matrix_rows" not in fields
    assert "matrix_boundaries" not in fields


@pytest.mark.parametrize(
    "mutate",
    [
        lambda s: s.pop("schema_version"),
        lambda s: s.__setitem__("extra", False),
        lambda s: s.__setitem__("schema_version", "bad"),
        lambda s: s.__setitem__("block", "bad"),
        lambda s: s.__setitem__("step", "bad"),
        lambda s: s.__setitem__("status", "bad"),
        lambda s: s.__setitem__("next_step", "bad"),
        lambda s: s.__setitem__("next_step_title", "bad"),
        lambda s: s.__setitem__("source_19_2_accepted", False),
        lambda s: s.__setitem__("environment_contract_artifact_complete", False),
        lambda s: s.__setitem__("environment_observation_complete", True),
        lambda s: s.__setitem__("environment_build_ready", True),
        lambda s: s.__setitem__("ready_for_block_q_4", False),
        lambda s: s["environment_contract_rows"].pop(),
        lambda s: s["environment_contract_rows"].append(
            copy.deepcopy(s["environment_contract_rows"][0])
        ),
        lambda s: s["environment_contract_rows"].reverse(),
        lambda s: s["environment_contract_rows"][0].__setitem__("contract_id", "bad"),
        lambda s: s["environment_contract_rows"][0].__setitem__("matrix_id", "bad"),
        lambda s: s["environment_contract_rows"][0].__setitem__("inventory_id", "bad"),
        lambda s: s["environment_contract_rows"][0].__setitem__("requirement_field", "bad"),
        lambda s: s["environment_contract_rows"][0].__setitem__("category", "bad"),
        lambda s: s["environment_contract_rows"][0].__setitem__("required", 1),
        lambda s: s["environment_contract_rows"][0].__setitem__("build_gate_open", True),
        lambda s: s["environment_contract_rows"][0].__setitem__("requirement_satisfied", True),
        lambda s: s["environment_contract_rows"][0].__setitem__("source_matrix_state", "bad"),
        lambda s: s["environment_contract_rows"][0].__setitem__("source_blocker_code", "bad"),
        lambda s: s["environment_contract_rows"][0].__setitem__("source_evidence_required", False),
        lambda s: s["environment_contract_rows"][0].__setitem__("acceptance_rule", "bad"),
        lambda s: s["environment_contract_rows"][0].__setitem__("satisfaction_rule", "bad"),
        lambda s: s["environment_contract_rows"][0].__setitem__("source_only_definition", False),
        lambda s: s["environment_contract_rows"][0].__setitem__("source_evidence_collected", True),
        lambda s: s["block_q_19_2_matrix_reference"].__setitem__("source_matrix_row_count", 10),
        lambda s: s["block_q_19_2_matrix_reference"]["source_top_level_fields"].pop(),
        lambda s: s["source_matrix_preservation"].__setitem__("preserves_matrix_order", False),
        lambda s: s["environment_contract_summary"].__setitem__("contract_row_count", 10),
        lambda s: s["environment_contract_scope"].__setitem__("contract_defined", False),
        lambda s: s["build_execution_authorization_state"].__setitem__(
            "packaging_authorized", True
        ),
        lambda s: s["non_execution_contract_evidence"].__setitem__("network_opened", True),
        lambda s: s["contract_boundaries"].__setitem__("packaging", True),
        lambda s: s["source_boundaries"].__setitem__("block_p_builders_read", True),
        lambda s: s["future_steps"][0].__setitem__("next_step", "bad"),
        lambda s: s.__setitem__("integrity_valid", False),
    ],
)
def test_source_mutations_fail_closed(monkeypatch: pytest.MonkeyPatch, mutate: Any) -> None:
    source = _source()
    mutate(source)
    assert _built_from(monkeypatch, source) == read_model._canonical_blocked()


def test_top_level_order_and_scalar_subclass_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    class StrSubclass(str):
        pass

    source = _source()
    reordered = {k: source[k] for k in reversed(list(source))}
    assert _built_from(monkeypatch, reordered) == read_model._canonical_blocked()
    source = _source()
    source["step"] = StrSubclass("19.3")
    assert _built_from(monkeypatch, source) == read_model._canonical_blocked()


def test_fail_closed_exceptions(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        read_model,
        "build_preview_block_q_windows_build_environment_contract",
        lambda: (_ for _ in ()).throw(RuntimeError),
    )
    assert (
        read_model.build_preview_block_q_windows_build_environment_read_model()
        == read_model._canonical_blocked()
    )
    source = _source()
    monkeypatch.setattr(
        read_model, "_integrity_19_3", lambda _s: (_ for _ in ()).throw(RuntimeError)
    )
    assert _built_from(monkeypatch, source) == read_model._canonical_blocked()


class EqualityBomb:
    eq_count = 0

    def __eq__(self, other: object) -> bool:
        EqualityBomb.eq_count += 1
        raise AssertionError


class HashBomb:
    hash_count = 0

    def __hash__(self) -> int:
        HashBomb.hash_count += 1
        raise AssertionError


class UnhashableEqual:
    eq_count = 0
    __hash__: Any = None

    def __eq__(self, other: object) -> bool:
        UnhashableEqual.eq_count += 1
        raise AssertionError


class StrSubclass(str):
    pass


class ListSubclass(list):
    pass


class DictSubclass(dict):
    pass


class GetItemBombDict(dict):
    getitem_count = 0

    def __getitem__(self, key: object) -> object:
        GetItemBombDict.getitem_count += 1
        raise AssertionError


class ArmedBombKey(str):
    eq_count = 0
    hash_count = 0
    armed = False

    def __eq__(self, other: object) -> bool:
        if ArmedBombKey.armed:
            ArmedBombKey.eq_count += 1
            raise AssertionError
        return str.__eq__(self, other)

    def __hash__(self) -> int:
        if ArmedBombKey.armed:
            ArmedBombKey.hash_count += 1
            raise AssertionError
        return str.__hash__(self)


def test_adversarial_cases_fail_closed_without_custom_methods(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cases = []
    s = _source()
    s[ArmedBombKey("x")] = False
    cases.append(s)
    s = _source()
    s["environment_contract_rows"][0][ArmedBombKey("x")] = False
    cases.append(s)
    s = _source()
    s["environment_contract_summary"][ArmedBombKey("x")] = False
    cases.append(s)
    s = _source()
    s["environment_contract_scope"][ArmedBombKey("x")] = False
    cases.append(s)
    s = _source()
    s["build_execution_authorization_state"][ArmedBombKey("x")] = False
    cases.append(s)
    s = _source()
    s["future_steps"][0][ArmedBombKey("x")] = False
    cases.append(s)
    s = _source()
    row = s["environment_contract_rows"][0]
    s["environment_contract_rows"][0] = {
        (ArmedBombKey(k) if k == "contract_id" else k): v for k, v in row.items()
    }
    cases.append(s)
    s = _source()
    preservation = s["source_matrix_preservation"]
    s["source_matrix_preservation"] = {
        (ArmedBombKey(k) if k == "preserves_matrix_order" else k): v
        for k, v in preservation.items()
    }
    cases.append(s)
    ArmedBombKey.armed = True
    s = _source()
    s["environment_contract_rows"] = ListSubclass(s["environment_contract_rows"])
    cases.append(s)
    s = _source()
    s["environment_contract_rows"][0] = DictSubclass(s["environment_contract_rows"][0])
    cases.append(s)
    s = _source()
    s["environment_contract_summary"] = GetItemBombDict(s["environment_contract_summary"])
    cases.append(s)
    for obj in (EqualityBomb(), HashBomb(), UnhashableEqual(), StrSubclass("19.3")):
        s = _source()
        s["step"] = obj
        cases.append(s)
    for s in cases:
        assert _built_from(monkeypatch, s) == read_model._canonical_blocked()
    ArmedBombKey.armed = False
    assert EqualityBomb.eq_count == 0 and HashBomb.hash_count == 0 and UnhashableEqual.eq_count == 0
    assert (
        GetItemBombDict.getitem_count == 0
        and ArmedBombKey.eq_count == 0
        and ArmedBombKey.hash_count == 0
    )


def test_trusted_source_template_is_complete_and_locally_accepted(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = read_model._trusted_source_stub()
    assert source["block_q_19_2_matrix_reference"]["source_top_level_fields"]
    monkeypatch.setattr(read_model, "_integrity_19_3", lambda _source: True)
    assert read_model._source_accepted(source) is True


def test_trust_boundary_ast_guard() -> None:
    tree = ast.parse(Path(read_model.__file__).read_text())
    guarded = {"_trusted_source_stub", "_canonical_nominal", "_canonical_blocked", "_integrity"}
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name in guarded:
            names = [
                n.func.id
                for n in ast.walk(node)
                if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
            ]
            assert "build_preview_block_q_windows_build_environment_contract" not in names
            assert "_integrity_19_3" not in names


def test_canonical_and_integrity_ignore_upstream_monkeypatch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        read_model,
        "build_preview_block_q_windows_build_environment_contract",
        lambda: (_ for _ in ()).throw(RuntimeError),
    )
    monkeypatch.setattr(
        read_model, "_integrity_19_3", lambda _s: (_ for _ in ()).throw(RuntimeError)
    )
    nominal = read_model._canonical_nominal()
    blocked = read_model._canonical_blocked()
    assert read_model._integrity(nominal) is True
    assert read_model._integrity(blocked) is True


def test_freshness_reload_order_isolation(monkeypatch: pytest.MonkeyPatch) -> None:
    a = read_model._canonical_blocked()
    b = read_model._canonical_blocked()
    assert (
        a == b
        and a is not b
        and a["environment_read_model_rows"] is not b["environment_read_model_rows"]
    )
    clean = importlib.reload(read_model)
    assert (
        clean.build_preview_block_q_windows_build_environment_read_model()
        == clean._canonical_nominal()
    )
    monkeypatch.setattr(
        clean, "build_preview_block_q_windows_build_environment_contract", lambda: {"bad": True}
    )
    assert (
        clean.build_preview_block_q_windows_build_environment_read_model()
        == clean._canonical_blocked()
    )
    monkeypatch.undo()
    assert (
        clean.build_preview_block_q_windows_build_environment_read_model()
        == clean._canonical_nominal()
    )
    assert contract.build_preview_block_q_windows_build_environment_contract()["step"] == "19.3"


def test_comparator_cycles_and_depth_1500() -> None:
    x: list[Any] = []
    x.append(x)
    y: list[Any] = []
    y.append(y)
    assert read_model._exact_plain(x, y) is True
    d1: dict[str, Any] = {}
    d1["self"] = d1
    d2: dict[str, Any] = {}
    d2["self"] = d2
    assert read_model._exact_plain(d1, d2) is True
    m: list[Any] = []
    m.append(m)
    n: list[Any] = [[]]
    assert read_model._exact_plain(m, n) is False
    left: list[Any] = []
    right: list[Any] = []
    cl = left
    cr = right
    for _ in range(1500):
        nl: list[Any] = []
        nr: list[Any] = []
        cl.append(nl)
        cr.append(nr)
        cl = nl
        cr = nr
    cl.append("end")
    cr.append("end")
    assert read_model._exact_plain(left, right) is True
    cr[0] = "bad"
    assert read_model._exact_plain(left, right) is False
