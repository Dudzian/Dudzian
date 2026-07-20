from __future__ import annotations

import ast
import copy
import importlib
from pathlib import Path
from typing import Any, cast

import pytest
import yaml

import ui.pyside_app.preview_block_q_windows_build_environment_observation_execution_contract as prod
from ui.pyside_app.preview_block_q_windows_build_environment_observation_plan import (
    _integrity as integrity_19_5,
    build_preview_block_q_windows_build_environment_observation_plan,
)

ROOT = Path(__file__).resolve().parents[2]
MANIFEST = (
    ROOT / "docs/roadmap/block_q_19_6_windows_build_environment_observation_execution_contract.yaml"
)

TOP = list(prod.TOP_LEVEL_FIELDS)
ROW = list(prod.ROW_FIELDS)


def manifest() -> dict[str, Any]:
    data = yaml.safe_load(MANIFEST.read_text(encoding="utf-8"))
    assert isinstance(data, dict)
    return data


def test_manifest_regression_orders() -> None:
    data = manifest()
    assert list(data) == [
        "schema_version",
        "kind",
        "block",
        "step",
        "title",
        "goal",
        "source_step",
        "source_kind",
        "source_builder",
        "source_only",
        "contract_artifact_complete",
        "actual_environment_observation_performed",
        "current_stage_execution_authorized",
        "future_read_only_observation_authorized",
        "environment_build_ready",
        "next_step",
        "next_step_title",
        "next_step_contract_missing",
        "next_step_authorized",
        "source_acceptance",
        "global_authorization",
        "observation_execution_contract_rows",
        "summary",
        "boundaries",
    ]
    assert list(data["source_acceptance"]) == [
        "requires_nominal_19_5",
        "requires_11_observation_plan_rows",
        "requires_source_only_19_5",
        "requires_zero_performed_observations",
        "requires_zero_collected_evidence",
        "requires_environment_build_not_ready",
        "requires_next_step_contract_missing",
        "source_rejected_blocked_payload_required",
    ]
    assert list(data["global_authorization"]) == list(prod._manifest_global_authorization())
    assert list(data["summary"]) == list(prod._manifest_summary())
    assert list(data["boundaries"]) == list(prod._manifest_boundaries())


def test_no_mutable_global_trusted_payloads_and_factory_poisoning_regression() -> None:
    forbidden = {"SOURCE_19_5", "MANIFEST_ROWS", "GLOBAL_AUTHORIZATION", "SUMMARY", "BOUNDARIES"}
    for name in forbidden:
        assert not hasattr(prod, name)
    for name, value in vars(prod).items():
        if name.endswith(("SPEC", "SPECS", "FIELDS")) or name in {
            "SCHEMA_VERSION",
            "KIND",
            "BLOCK_ID",
            "STEP_ID",
            "TITLE",
        }:
            assert type(value) not in (dict, list, set)

    source = prod._trusted_source_19_5()
    source["observation_plan_rows"][0]["observation_plan_id"] = "poisoned"
    assert (
        prod._trusted_source_19_5()["observation_plan_rows"][0]["observation_plan_id"]
        == "observation_plan_windows_host"
    )
    rows = prod._manifest_rows()
    rows[0]["execution_contract_id"] = "poisoned"
    assert prod._manifest_rows()[0]["execution_contract_id"] == "execution_contract_windows_host"
    auth = prod._manifest_global_authorization()
    auth["future_read_only_observation_authorized"] = False
    assert prod._manifest_global_authorization()["future_read_only_observation_authorized"] is True
    summary = prod._manifest_summary()
    summary["execution_contract_row_count"] = 0
    assert prod._manifest_summary()["execution_contract_row_count"] == 11
    boundaries = prod._manifest_boundaries()
    boundaries["reads_19_5_only"] = False
    assert prod._manifest_boundaries()["reads_19_5_only"] is True
    assert (
        prod._canonical_nominal()
        == prod.build_preview_block_q_windows_build_environment_observation_execution_contract()
    )


def test_manifest_factories_are_fresh() -> None:
    factories = (
        prod._trusted_source_19_5,
        prod._manifest_rows,
        prod._manifest_global_authorization,
        prod._manifest_summary,
        prod._manifest_boundaries,
    )
    for factory in factories:
        a = factory()
        b = factory()
        assert a == b
        assert a is not b
    assert (
        prod._trusted_source_19_5()["observation_plan_rows"]
        is not prod._trusted_source_19_5()["observation_plan_rows"]
    )
    assert (
        prod._trusted_source_19_5()["observation_plan_rows"][0]
        is not prod._trusted_source_19_5()["observation_plan_rows"][0]
    )
    assert prod._manifest_rows()[0] is not prod._manifest_rows()[0]


def test_nominal_uses_upstream_once_consumes_rows_and_matches_yaml(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = build_preview_block_q_windows_build_environment_observation_plan()
    before = copy.deepcopy(source)
    calls = 0

    def upstream() -> dict[str, Any]:
        nonlocal calls
        calls += 1
        return cast(dict[str, Any], source)

    monkeypatch.setattr(
        prod, "build_preview_block_q_windows_build_environment_observation_plan", upstream
    )
    payload = prod.build_preview_block_q_windows_build_environment_observation_execution_contract()
    data = manifest()
    assert calls == 1
    assert source == before
    assert list(payload) == TOP
    assert payload == prod._canonical_nominal()
    assert prod._integrity(payload) is True
    assert payload["source_19_5_accepted"] is True
    assert payload["observation_execution_contract_artifact_complete"] is True
    assert len(payload["observation_execution_contract_rows"]) == 11
    assert (
        payload["observation_execution_contract_rows"]
        == data["observation_execution_contract_rows"]
    )
    for i, row in enumerate(payload["observation_execution_contract_rows"]):
        src = source["observation_plan_rows"][i]
        assert list(row) == ROW
        assert row["observation_plan_id"] == src["observation_plan_id"]
        assert row["read_model_id"] == src["read_model_id"]
        assert row["observation_target"] == src["observation_target"]
        assert row["source_only_definition"] == src["source_only_definition"]
    assert payload["observation_execution_contract_summary"] == data["summary"]
    assert payload["global_authorization"] == data["global_authorization"]
    assert payload["observation_execution_contract_boundaries"] == data["boundaries"]
    assert payload["observation_execution_contract_summary"]["future_read_authorized_count"] == 11
    assert payload["observation_execution_contract_summary"]["performed_observation_count"] == 0
    assert "19.7" not in repr(payload)


def test_blocked_semantics_and_fail_closed(monkeypatch: pytest.MonkeyPatch) -> None:
    blocked = prod._canonical_blocked()
    assert blocked["source_19_5_accepted"] is False
    assert blocked["observation_execution_contract_rows"] == []
    assert blocked["future_read_only_observation_authorized"] is False
    assert (
        blocked["observation_execution_contract_summary"]["contract_definition_complete"] is False
    )
    assert prod._integrity(blocked) is True
    monkeypatch.setattr(
        prod,
        "build_preview_block_q_windows_build_environment_observation_plan",
        lambda: (_ for _ in ()).throw(RuntimeError("x")),
    )
    assert (
        prod.build_preview_block_q_windows_build_environment_observation_execution_contract()
        == blocked
    )
    monkeypatch.setattr(
        prod,
        "build_preview_block_q_windows_build_environment_observation_plan",
        lambda: {"bad": True},
    )
    assert (
        prod.build_preview_block_q_windows_build_environment_observation_execution_contract()
        == blocked
    )
    monkeypatch.setattr(
        prod, "_integrity_19_5", lambda _source: (_ for _ in ()).throw(RuntimeError("x"))
    )
    monkeypatch.setattr(
        prod,
        "build_preview_block_q_windows_build_environment_observation_plan",
        build_preview_block_q_windows_build_environment_observation_plan,
    )
    assert (
        prod.build_preview_block_q_windows_build_environment_observation_execution_contract()
        == blocked
    )


@pytest.mark.parametrize("key", list(prod.SOURCE_TOP_LEVEL_FIELDS))
def test_source_top_level_mutations_rejected(monkeypatch: pytest.MonkeyPatch, key: str) -> None:
    source = build_preview_block_q_windows_build_environment_observation_plan()
    mutated = copy.deepcopy(source)
    if type(mutated[key]) is bool:
        mutated[key] = not mutated[key]
    elif type(mutated[key]) is str:
        mutated[key] = mutated[key] + "x"
    elif type(mutated[key]) is list:
        mutated[key] = list(reversed(mutated[key])) or ["unexpected"]
    else:
        mutated[key] = {"replaced": True}
    monkeypatch.setattr(
        prod, "build_preview_block_q_windows_build_environment_observation_plan", lambda: mutated
    )
    assert (
        prod.build_preview_block_q_windows_build_environment_observation_execution_contract()
        == prod._canonical_blocked()
    )


def test_row_mutations_and_order_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    source = build_preview_block_q_windows_build_environment_observation_plan()
    cases = []
    missing = copy.deepcopy(source)
    missing["observation_plan_rows"].pop()
    cases.append(missing)
    added = copy.deepcopy(source)
    added["observation_plan_rows"].append(copy.deepcopy(added["observation_plan_rows"][0]))
    cases.append(added)
    reordered = copy.deepcopy(source)
    reordered["observation_plan_rows"] = list(reversed(reordered["observation_plan_rows"]))
    cases.append(reordered)
    for field in source["observation_plan_rows"][0]:
        changed = copy.deepcopy(source)
        changed["observation_plan_rows"][0][field] = "x"
        cases.append(changed)
    for case in cases:
        monkeypatch.setattr(
            prod,
            "build_preview_block_q_windows_build_environment_observation_plan",
            lambda case=case: case,
        )
        assert (
            prod.build_preview_block_q_windows_build_environment_observation_execution_contract()
            == prod._canonical_blocked()
        )


def test_freshness_reload_and_upstream_integrity() -> None:
    a = prod._canonical_nominal()
    b = prod._canonical_nominal()
    c = prod._canonical_blocked()
    d = prod._canonical_blocked()
    assert (
        a == b
        and a is not b
        and a["observation_execution_contract_rows"] is not b["observation_execution_contract_rows"]
    )
    assert c == d and c is not d and a != c
    mod = importlib.reload(prod)
    assert mod._integrity(mod._canonical_nominal()) is True
    assert mod._integrity(mod._canonical_blocked()) is True
    assert mod._integrity(mod._canonical_nominal()) is True
    assert (
        integrity_19_5(build_preview_block_q_windows_build_environment_observation_plan()) is True
    )


def test_ast_guard_for_non_runtime_helpers() -> None:
    tree = ast.parse(Path(prod.__file__).read_text(encoding="utf-8"))
    guarded = {"_trusted_source_19_5", "_canonical_nominal", "_canonical_blocked", "_integrity"}
    forbidden = {
        "build_preview_block_q_windows_build_environment_observation_plan",
        "_integrity_19_5",
        "open",
        "Path",
    }
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name in guarded:
            text = ast.unparse(node)
            for name in forbidden | {"safe_load", "platform", "subprocess"}:
                assert name not in text


def test_comparator_cycles_depth_and_mismatch() -> None:
    a: list[Any] = []
    b: list[Any] = []
    a.append(a)
    b.append(b)
    assert prod._exact_plain(a, b) is True
    da: dict[str, Any] = {}
    db: dict[str, Any] = {}
    da["x"] = da
    db["x"] = db
    assert prod._exact_plain(da, db) is True
    assert prod._exact_plain(a, [b, b]) is False
    x: Any = ["leaf"]
    y: Any = ["leaf"]
    x_cursor = x
    y_cursor = y
    for _ in range(1499):
        child_x: list[Any] = ["leaf"]
        child_y: list[Any] = ["leaf"]
        x_cursor[0] = child_x
        y_cursor[0] = child_y
        x_cursor = child_x
        y_cursor = child_y
    assert prod._exact_plain(x, y) is True
    y_cursor[0] = "bad"
    assert prod._exact_plain(x, y) is False


def test_adversarial_exact_type_first_and_no_custom_calls(monkeypatch: pytest.MonkeyPatch) -> None:
    counters = {"eq": 0, "hash": 0, "getitem": 0}

    class EqualityBomb:
        def __eq__(self, other: object) -> bool:
            counters["eq"] += 1
            raise AssertionError

    class HashBomb:
        def __hash__(self) -> int:
            counters["hash"] += 1
            raise AssertionError

    class GetItemBombDict(dict):
        def __getitem__(self, key: object) -> object:
            counters["getitem"] += 1
            raise AssertionError

    class StrSubclass(str):
        pass

    class ListSubclass(list):
        pass

    class DictSubclass(dict):
        pass

    for bad in (
        EqualityBomb(),
        HashBomb(),
        StrSubclass("x"),
        ListSubclass(),
        DictSubclass(),
        GetItemBombDict(),
    ):
        assert prod._exact_plain(bad, prod._canonical_nominal()) is False
    source = build_preview_block_q_windows_build_environment_observation_plan()
    for bad in (
        EqualityBomb(),
        HashBomb(),
        StrSubclass("Q"),
        ListSubclass(source["observation_plan_rows"]),
        DictSubclass(source),
    ):
        mutated = copy.deepcopy(source)
        mutated["block"] = bad
        monkeypatch.setattr(
            prod,
            "build_preview_block_q_windows_build_environment_observation_plan",
            lambda mutated=mutated: mutated,
        )
        assert (
            prod.build_preview_block_q_windows_build_environment_observation_execution_contract()
            == prod._canonical_blocked()
        )
    assert counters == {"eq": 0, "hash": 0, "getitem": 0}


def _same_position_replace(mapping: dict[str, Any], target: str, new_key: str) -> dict[str, Any]:
    return {new_key if key == target else key: value for key, value in dict.items(mapping)}


def test_reload_real_nominal_blocked_nominal_and_poisoned_factory_outputs_do_not_affect_build(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    mod = importlib.reload(prod)
    assert (
        mod.build_preview_block_q_windows_build_environment_observation_execution_contract()
        == mod._canonical_nominal()
    )
    monkeypatch.setattr(
        mod,
        "build_preview_block_q_windows_build_environment_observation_plan",
        lambda: {"bad": True},
    )
    assert (
        mod.build_preview_block_q_windows_build_environment_observation_execution_contract()
        == mod._canonical_blocked()
    )
    monkeypatch.undo()
    assert (
        mod.build_preview_block_q_windows_build_environment_observation_execution_contract()
        == mod._canonical_nominal()
    )
    assert (
        integrity_19_5(build_preview_block_q_windows_build_environment_observation_plan()) is True
    )
    mod._trusted_source_19_5()["observation_plan_rows"][0]["observation_plan_id"] = "poisoned"
    mod._manifest_rows()[0]["execution_contract_id"] = "poisoned"
    assert (
        mod.build_preview_block_q_windows_build_environment_observation_execution_contract()
        == mod._canonical_nominal()
    )


def test_same_position_armed_bomb_keys_rejected_without_custom_calls(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class ArmedBombKey(str):
        armed = False
        eq_count = 0
        hash_count = 0

        def __eq__(self, other: object) -> bool:
            if type(self).armed:
                type(self).eq_count += 1
                raise AssertionError("armed eq")
            return str.__eq__(self, other)  # type: ignore[no-any-return]

        def __hash__(self) -> int:
            if type(self).armed:
                type(self).hash_count += 1
                raise AssertionError("armed hash")
            return str.__hash__(self)

    cases = []
    source = prod._trusted_source_19_5()
    cases.append(_same_position_replace(source, "schema_version", ArmedBombKey("schema_version")))
    row_source = prod._trusted_source_19_5()
    row_source["observation_plan_rows"][0] = _same_position_replace(
        row_source["observation_plan_rows"][0],
        "observation_plan_id",
        ArmedBombKey("observation_plan_id"),
    )
    cases.append(row_source)
    nested_source = prod._trusted_source_19_5()
    nested_source["observation_plan_scope"] = _same_position_replace(
        nested_source["observation_plan_scope"],
        "observation_plan_defined",
        ArmedBombKey("observation_plan_defined"),
    )
    cases.append(nested_source)
    try:
        ArmedBombKey.armed = True
        for case in cases:
            monkeypatch.setattr(
                prod,
                "build_preview_block_q_windows_build_environment_observation_plan",
                lambda case=case: case,
            )
            payload = prod.build_preview_block_q_windows_build_environment_observation_execution_contract()
            assert payload == prod._canonical_blocked()
            assert payload["source_19_5_accepted"] is False
            assert ArmedBombKey.eq_count == 0
            assert ArmedBombKey.hash_count == 0
    finally:
        ArmedBombKey.armed = False


def test_full_adversarial_source_injection_rejected_without_custom_calls(
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

    scalar_fields = ("step", "status", "source_19_4_accepted", "integrity_valid")
    bombs = (EqualityBomb(), HashBomb(), UnhashableEqual(), StrSubclass("19.5"))
    cases = []
    for field in scalar_fields:
        for bomb in bombs:
            source = prod._trusted_source_19_5()
            source[field] = bomb
            cases.append(source)
    source = prod._trusted_source_19_5()
    source["observation_plan_rows"] = ListSubclass(source["observation_plan_rows"])
    cases.append(source)
    source = prod._trusted_source_19_5()
    source["observation_plan_rows"][0] = DictSubclass(source["observation_plan_rows"][0])
    cases.append(source)
    source = prod._trusted_source_19_5()
    source["observation_plan_summary"] = DictSubclass(source["observation_plan_summary"])
    cases.append(source)
    source = prod._trusted_source_19_5()
    source["observation_plan_scope"] = GetItemBombDict(source["observation_plan_scope"])
    cases.append(source)
    cases.append(DictSubclass(prod._trusted_source_19_5()))
    for case in cases:
        monkeypatch.setattr(
            prod,
            "build_preview_block_q_windows_build_environment_observation_plan",
            lambda case=case: case,
        )
        assert (
            prod.build_preview_block_q_windows_build_environment_observation_execution_contract()
            == prod._canonical_blocked()
        )
    assert counters == {"eq": 0, "hash": 0, "getitem": 0}


def test_source_row_schema_order_edge_cases_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    cases = []
    missing = prod._trusted_source_19_5()
    missing["observation_plan_rows"][0].pop("observation_plan_id")
    cases.append(missing)
    extra = prod._trusted_source_19_5()
    extra["observation_plan_rows"][0]["extra"] = "x"
    cases.append(extra)
    reversed_keys = prod._trusted_source_19_5()
    reversed_keys["observation_plan_rows"][0] = dict(
        reversed(list(dict.items(reversed_keys["observation_plan_rows"][0])))
    )
    cases.append(reversed_keys)

    class StrSubclass(str):
        pass

    str_key = prod._trusted_source_19_5()
    str_key["observation_plan_rows"][0] = _same_position_replace(
        str_key["observation_plan_rows"][0],
        "observation_plan_id",
        StrSubclass("observation_plan_id"),
    )
    cases.append(str_key)
    for case in cases:
        monkeypatch.setattr(
            prod,
            "build_preview_block_q_windows_build_environment_observation_plan",
            lambda case=case: case,
        )
        assert (
            prod.build_preview_block_q_windows_build_environment_observation_execution_contract()
            == prod._canonical_blocked()
        )


def test_nominal_builder_exceptions_return_fresh_integral_blocked(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        prod, "_rows_from_source", lambda _source: (_ for _ in ()).throw(RuntimeError("rows"))
    )
    payload = prod.build_preview_block_q_windows_build_environment_observation_execution_contract()
    assert payload == prod._canonical_blocked()
    assert payload is not prod._canonical_blocked()
    monkeypatch.undo()
    assert prod._integrity(payload) is True


def test_full_nested_freshness_for_nominal_and_blocked() -> None:
    mutable_fields = (
        "block_q_19_5_observation_plan_reference",
        "source_observation_plan_preservation",
        "observation_execution_contract_scope",
        "observation_execution_contract_rows",
        "observation_execution_contract_summary",
        "global_authorization",
        "non_execution_observation_execution_contract_evidence",
        "observation_execution_contract_boundaries",
        "source_boundaries",
        "future_steps",
    )
    for factory in (prod._canonical_nominal, prod._canonical_blocked):
        a = factory()
        b = factory()
        for field in mutable_fields:
            assert a[field] == b[field]
            assert a[field] is not b[field]
        for left, right in zip(
            a["observation_execution_contract_rows"],
            b["observation_execution_contract_rows"],
            strict=True,
        ):
            assert left == right
            assert left is not right
