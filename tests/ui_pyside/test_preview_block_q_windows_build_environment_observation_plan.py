from __future__ import annotations

import ast
import copy
import importlib
from pathlib import Path
from typing import Any

import pytest
import yaml

import ui.pyside_app.preview_block_q_windows_build_environment_observation_plan as m
import ui.pyside_app.preview_block_q_windows_build_environment_read_model as m19

MANIFEST = (
    Path(__file__).resolve().parents[2]
    / "docs/roadmap/block_q_19_5_windows_build_environment_observation_plan.yaml"
)


def test_nominal_path_monkeypatched_upstream_once_and_yaml_rows(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source: dict[str, Any] = m19.build_preview_block_q_windows_build_environment_read_model()
    before = copy.deepcopy(source)
    calls = 0

    def fake() -> dict[str, Any]:
        nonlocal calls
        calls += 1
        return source

    monkeypatch.setattr(m, "build_preview_block_q_windows_build_environment_read_model", fake)
    payload = m.build_preview_block_q_windows_build_environment_observation_plan()
    manifest = yaml.safe_load(MANIFEST.read_text(encoding="utf-8"))
    assert calls == 1
    assert source == before
    assert list(payload) == list(m.TOP_LEVEL_FIELDS)
    assert payload["source_19_4_accepted"] is True
    assert payload["observation_plan_artifact_complete"] is True
    assert len(payload["observation_plan_rows"]) == 11
    assert payload["next_step"] == payload["next_step_title"] == ""
    assert payload["ready_for_block_q_6"] is False
    assert payload["next_step_contract_missing"] is True
    assert payload["future_steps"] == []
    assert payload["observation_plan_rows"] == manifest["observation_plan_rows"]
    for i, row in enumerate(payload["observation_plan_rows"]):
        src = source["environment_read_model_rows"][i]
        for field in (
            "read_model_id",
            "contract_id",
            "matrix_id",
            "inventory_id",
            "requirement_field",
            "category",
            "required",
            "source_only_definition",
        ):
            assert row[field] == src[field]
        assert row["observation_authorized"] is False
        assert row["observation_status"] == "not_performed"
        assert row["evidence_status"] == "not_collected"
    assert payload["observation_plan_summary"]["authorized_observation_count"] == 0
    assert payload["observation_plan_summary"]["performed_observation_count"] == 0
    assert payload["observation_plan_summary"]["evidence_collected_count"] == 0
    assert all(
        v is False
        for k, v in payload["observation_authorization_state"].items()
        if k.endswith("_authorized") and k != "observation_plan_definition_authorized"
    )
    assert m._integrity(payload) is True


def _blocked_from_source(monkeypatch: pytest.MonkeyPatch, source: Any) -> dict[str, Any]:
    monkeypatch.setattr(
        m, "build_preview_block_q_windows_build_environment_read_model", lambda: source
    )
    return m.build_preview_block_q_windows_build_environment_observation_plan()


def _assert_canonical_blocked(payload: dict[str, Any]) -> None:
    assert payload == m._canonical_blocked()
    assert payload["source_19_4_accepted"] is False


def _replace_same_position_key(
    source: dict[str, Any], target_key: str, replacement_key: str
) -> dict[str, Any]:
    return {replacement_key if key == target_key else key: value for key, value in source.items()}


def test_source_rejected_blocked_path(monkeypatch: pytest.MonkeyPatch) -> None:
    payload = _blocked_from_source(monkeypatch, {"bad": "source"})
    assert payload["source_19_4_accepted"] is False
    assert payload["observation_plan_artifact_complete"] is False
    assert payload["observation_plan_rows"] == []
    assert payload["observation_plan_summary"]["plan_definition_complete"] is False
    assert m._integrity(payload) is True


def test_production_authorization_schema_matches_yaml_forbidden_order() -> None:
    manifest = yaml.safe_load(MANIFEST.read_text(encoding="utf-8"))
    expected_keys = list(manifest["forbidden_authorizations"])
    for payload in (m._canonical_nominal(), m._canonical_blocked()):
        auth = payload["observation_authorization_state"]
        forbidden_part = {key: auth[key] for key in expected_keys}
        assert list(forbidden_part) == expected_keys
        assert list(forbidden_part) == list(m.AUTHORIZATION_FIELDS)
        assert all(value is False for value in forbidden_part.values())


@pytest.mark.parametrize(
    "mutate",
    [
        lambda s: s.pop("schema_version"),
        lambda s: s.__setitem__("extra", True),
        lambda s: s.__setitem__("schema_version", "bad"),
        lambda s: s.__setitem__("source_19_3_accepted", False),
        lambda s: s["environment_read_model_rows"].pop(),
        lambda s: s["environment_read_model_rows"].append(
            dict(s["environment_read_model_rows"][0])
        ),
        lambda s: s["environment_read_model_rows"].reverse(),
        lambda s: s["environment_read_model_rows"][0].__setitem__("read_model_id", "bad"),
        lambda s: s["environment_read_model_rows"][0].__setitem__("contract_id", "bad"),
        lambda s: s["environment_read_model_rows"][0].__setitem__("matrix_id", "bad"),
        lambda s: s["environment_read_model_rows"][0].__setitem__("inventory_id", "bad"),
        lambda s: s["environment_read_model_rows"][0].__setitem__("requirement_field", "bad"),
        lambda s: s["environment_read_model_rows"][0].__setitem__("category", "bad"),
        lambda s: s["environment_read_model_rows"][0].__setitem__("required", False),
        lambda s: s["environment_read_model_rows"][0].__setitem__("source_contract_state", "bad"),
        lambda s: s["environment_read_model_rows"][0].__setitem__(
            "source_requirement_satisfied", True
        ),
        lambda s: s["environment_read_model_rows"][0].__setitem__("source_build_gate_open", True),
        lambda s: s["environment_read_model_rows"][0].__setitem__("read_required", False),
        lambda s: s["environment_read_model_rows"][0].__setitem__("read_status", "done"),
        lambda s: s["environment_read_model_rows"][0].__setitem__("observed_value", "observed"),
        lambda s: s["environment_read_model_rows"][0].__setitem__("evidence_collected", True),
        lambda s: s["environment_read_model_rows"][0].__setitem__("evidence_validated", True),
        lambda s: s["environment_read_model_rows"][0].__setitem__("read_model_state", "ready"),
        lambda s: s["environment_read_model_rows"][0].__setitem__("source_only_definition", False),
        lambda s: s["environment_read_model_rows"].__setitem__(
            0,
            {
                key: s["environment_read_model_rows"][0][key]
                for key in reversed(list(s["environment_read_model_rows"][0]))
            },
        ),
        lambda s: s["environment_read_model_rows"][0].__setitem__("extra", True),
        lambda s: s["environment_read_model_rows"][0].pop("read_model_id"),
        lambda s: s["block_q_19_3_contract_reference"].__setitem__("step", "bad"),
        lambda s: s["source_contract_preservation"].__setitem__("preserves_read_model_order", True),
        lambda s: s["environment_read_model_scope"].__setitem__("read_model_defined", False),
        lambda s: s["environment_read_model_summary"].__setitem__("read_model_row_count", 10),
        lambda s: s["build_execution_authorization_state"].__setitem__("orders_authorized", True),
        lambda s: s["non_execution_read_model_evidence"].__setitem__("orders_performed", True),
        lambda s: s["read_model_boundaries"].__setitem__("orders", True),
        lambda s: s["source_boundaries"].__setitem__("source_step", "bad"),
        lambda s: s.__setitem__("future_steps", [{}]),
        lambda s: s.__setitem__("integrity_valid", False),
    ],
)
def test_source_mutations_rejected(monkeypatch: pytest.MonkeyPatch, mutate: Any) -> None:
    source = m19.build_preview_block_q_windows_build_environment_read_model()
    mutate(source)
    _assert_canonical_blocked(_blocked_from_source(monkeypatch, source))


def test_fail_closed_paths(monkeypatch: pytest.MonkeyPatch) -> None:
    def boom() -> dict[str, Any]:
        raise RuntimeError("boom")

    monkeypatch.setattr(m, "build_preview_block_q_windows_build_environment_read_model", boom)
    assert (
        m.build_preview_block_q_windows_build_environment_observation_plan()["source_19_4_accepted"]
        is False
    )
    monkeypatch.setattr(
        m,
        "build_preview_block_q_windows_build_environment_read_model",
        m19.build_preview_block_q_windows_build_environment_read_model,
    )
    monkeypatch.setattr(
        m, "_integrity_19_4", lambda _s: (_ for _ in ()).throw(RuntimeError("boom"))
    )
    assert (
        m.build_preview_block_q_windows_build_environment_observation_plan()["source_19_4_accepted"]
        is False
    )
    monkeypatch.setattr(m, "_integrity_19_4", m19._integrity)
    monkeypatch.setattr(m, "_rows", lambda _s: (_ for _ in ()).throw(RuntimeError("boom")))
    assert (
        m.build_preview_block_q_windows_build_environment_observation_plan()["source_19_4_accepted"]
        is False
    )


def test_freshness_and_reload_order() -> None:
    a = m._canonical_nominal()
    b = m._canonical_nominal()
    c = m._canonical_blocked()
    d = m._canonical_blocked()
    assert a == b and a is not b and a["observation_plan_rows"] is not b["observation_plan_rows"]
    assert c == d and c is not d and c["observation_plan_rows"] is not d["observation_plan_rows"]
    assert a != c
    assert m._integrity(a) and m._integrity(c) and m._integrity(m._canonical_nominal())


def test_clean_reload_nominal_blocked_nominal_and_upstream_order_isolation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clean = importlib.reload(m)
    assert clean.build_preview_block_q_windows_build_environment_observation_plan() == (
        clean._canonical_nominal()
    )
    monkeypatch.setattr(
        clean, "build_preview_block_q_windows_build_environment_read_model", lambda: {"bad": True}
    )
    assert clean.build_preview_block_q_windows_build_environment_observation_plan() == (
        clean._canonical_blocked()
    )
    monkeypatch.undo()
    assert clean.build_preview_block_q_windows_build_environment_observation_plan() == (
        clean._canonical_nominal()
    )
    assert m19._integrity(m19.build_preview_block_q_windows_build_environment_read_model()) is True


def test_ast_trust_boundary_guard() -> None:
    tree = ast.parse(Path(m.__file__).read_text(encoding="utf-8"))
    guarded = {"_trusted_source_19_4", "_canonical_nominal", "_canonical_blocked", "_integrity"}
    for node in [n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name in guarded]:
        calls = [c for c in ast.walk(node) if isinstance(c, ast.Call)]
        names = {getattr(c.func, "id", "") for c in calls} | {
            getattr(c.func, "attr", "") for c in calls
        }
        assert "build_preview_block_q_windows_build_environment_read_model" not in names
        assert "_integrity_19_4" not in names
        assert "safe_load" not in names
        assert "read_text" not in names
        assert "open" not in names


def test_comparator_cycles_depth_and_mismatch() -> None:
    a: list[Any] = []
    b: list[Any] = []
    a.append(a)
    b.append(b)
    assert m._exact_plain(a, b)
    da: dict[str, Any] = {}
    db: dict[str, Any] = {}
    da["self"] = da
    db["self"] = db
    assert m._exact_plain(da, db)
    assert not m._exact_plain(a, [[]])
    x: Any = []
    y: Any = []
    x_cursor = x
    y_cursor = y
    for _ in range(1500):
        new_x: list[Any] = []
        new_y: list[Any] = []
        x_cursor.append(new_x)
        y_cursor.append(new_y)
        x_cursor = new_x
        y_cursor = new_y
    x_cursor.append("leaf")
    y_cursor.append("leaf")
    assert m._exact_plain(x, y)
    y_cursor[0] = "bad"
    assert not m._exact_plain(x, y)


def test_adversarial_exact_type_first_and_same_position_keys(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    counters = {"eq": 0, "hash": 0, "getitem": 0}

    class EqualityBomb:
        def __eq__(self, other: object) -> bool:
            counters["eq"] += 1
            raise AssertionError

    class HashBomb:
        def __hash__(self) -> int:
            counters["hash"] += 1
            raise AssertionError

    class UnhashableEqual:
        __hash__ = None  # type: ignore[assignment]

        def __eq__(self, other: object) -> bool:
            counters["eq"] += 1
            raise AssertionError

    class StrSubclass(str):
        pass

    class ListSubclass(list):
        pass

    class DictSubclass(dict):
        pass

    class GetItemBombDict(dict):
        def __getitem__(self, key: object) -> object:
            counters["getitem"] += 1
            raise AssertionError

    class ArmedBombKey(str):
        armed = False
        eq_count = 0
        hash_count = 0

        def __eq__(self, other: object) -> bool:
            if type(self).armed:
                type(self).eq_count += 1
                raise AssertionError
            return super().__eq__(other)

        def __hash__(self) -> int:
            if type(self).armed:
                type(self).hash_count += 1
                raise AssertionError
            return super().__hash__()

    source = m19.build_preview_block_q_windows_build_environment_read_model()
    for bad in (
        EqualityBomb(),
        HashBomb(),
        UnhashableEqual(),
        StrSubclass("schema_version"),
        ListSubclass(),
        DictSubclass(),
        GetItemBombDict(),
    ):
        assert not m._exact_plain(bad, m._canonical_nominal())
    str_subclass_source = _replace_same_position_key(
        source, "schema_version", StrSubclass("schema_version")
    )
    _assert_canonical_blocked(_blocked_from_source(monkeypatch, str_subclass_source))

    for field, bad in (
        ("step", EqualityBomb()),
        ("status", HashBomb()),
        ("source_19_3_accepted", UnhashableEqual()),
        ("integrity_valid", EqualityBomb()),
    ):
        injected = copy.deepcopy(source)
        injected[field] = bad
        _assert_canonical_blocked(_blocked_from_source(monkeypatch, injected))
    container_cases = (
        ("environment_read_model_rows", ListSubclass(source["environment_read_model_rows"])),
        (
            "environment_read_model_summary",
            DictSubclass(source["environment_read_model_summary"]),
        ),
        ("environment_read_model_scope", GetItemBombDict(source["environment_read_model_scope"])),
    )
    for field, bad in container_cases:
        injected = copy.deepcopy(source)
        injected[field] = bad
        _assert_canonical_blocked(_blocked_from_source(monkeypatch, injected))
    row_injected = copy.deepcopy(source)
    row_injected["environment_read_model_rows"][0] = DictSubclass(
        row_injected["environment_read_model_rows"][0]
    )
    _assert_canonical_blocked(_blocked_from_source(monkeypatch, row_injected))

    top_armed = _replace_same_position_key(source, "schema_version", ArmedBombKey("schema_version"))
    row_armed = copy.deepcopy(source)
    row_armed["environment_read_model_rows"][0] = _replace_same_position_key(
        row_armed["environment_read_model_rows"][0],
        "read_model_id",
        ArmedBombKey("read_model_id"),
    )
    nested_armed = copy.deepcopy(source)
    nested_armed["environment_read_model_scope"] = _replace_same_position_key(
        nested_armed["environment_read_model_scope"],
        "read_model_defined",
        ArmedBombKey("read_model_defined"),
    )
    try:
        ArmedBombKey.armed = True
        for armed_source in (top_armed, row_armed, nested_armed):
            _assert_canonical_blocked(_blocked_from_source(monkeypatch, armed_source))
            assert ArmedBombKey.eq_count == 0
            assert ArmedBombKey.hash_count == 0
    finally:
        ArmedBombKey.armed = False
    assert counters == {"eq": 0, "hash": 0, "getitem": 0}
