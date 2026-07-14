from __future__ import annotations

import ast
import copy
import json
from pathlib import Path
from typing import Any

import pytest

import ui.pyside_app.preview_block_o_closure_audit as audit
from ui.pyside_app.preview_block_o_execution_authorization_readiness_read_model import (
    build_preview_block_o_execution_authorization_readiness_read_model,
)

MODULE = Path("ui/pyside_app/preview_block_o_closure_audit.py")


def payload() -> dict[str, Any]:
    return audit.build_preview_block_o_closure_audit()


def assert_plain_payload(value: Any) -> None:
    stack = [value]
    while stack:
        item = stack.pop()
        if type(item) is dict:
            for key, child in item.items():
                assert type(key) is str
                stack.append(child)
        elif type(item) is list:
            stack.extend(item)
        else:
            assert type(item) in (str, int, bool) or item is None


def assert_blocked(p: dict[str, Any]) -> None:
    assert p["status"] == audit.BLOCKED_STATUS
    assert p["block_o_closure_audit_ready"] is False
    assert p["block_o_closed"] is False
    assert p["ready_for_next_explicit_block"] is False
    json.dumps(p)
    assert_plain_payload(p)


def patched(monkeypatch: pytest.MonkeyPatch, src: Any, *, compare: bool = True) -> dict[str, Any]:
    original = copy.deepcopy(src) if compare and type(src) is dict else None
    monkeypatch.setattr(
        audit,
        "build_preview_block_o_execution_authorization_readiness_read_model",
        lambda: src,
    )
    p = audit.build_preview_block_o_closure_audit()
    if compare and type(src) is dict:
        assert src == original
    return p


def test_expected_source_matches_current_17_7() -> None:
    assert (
        audit.EXPECTED_SOURCE
        == build_preview_block_o_execution_authorization_readiness_read_model()
    )


def test_identity_order_reference_and_json_serializable() -> None:
    p = payload()
    assert list(p) == audit.TOP_LEVEL_FIELDS
    assert p["schema_version"] == audit.SCHEMA_VERSION
    assert p["block_o_closure_audit_kind"] == audit.KIND
    assert p["block"] == "O"
    assert p["step"] == "17.8"
    assert p["block_o_closure_audit_ready"] is True
    assert p["block_o_closed"] is True
    assert p["ready_for_next_explicit_block"] is True
    ref = p["block_o_execution_authorization_readiness_read_model_reference"]
    assert (
        ref["source_block_o_execution_authorization_readiness_read_model_step"]
        == "FUNCTIONAL-PREVIEW-17.7"
    )
    assert ref["source_readiness_read_model_read_by_17_8"] is True
    assert ref["block_o_closure_audit_ready_by_17_8"] is True
    assert p["next_step"] == "FUTURE-EXPLICIT-BLOCK"
    assert p["next_step_title"] == "TO BE DEFINED BY A SEPARATE PROMPT"
    json.dumps(p)


def test_source_builder_called_exactly_once(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = 0

    def fake() -> dict[str, Any]:
        nonlocal calls
        calls += 1
        return copy.deepcopy(audit.EXPECTED_SOURCE)

    monkeypatch.setattr(
        audit, "build_preview_block_o_execution_authorization_readiness_read_model", fake
    )
    assert audit.build_preview_block_o_closure_audit()["block_o_closed"] is True
    assert calls == 1


def test_two_domain_closure_rows() -> None:
    rows = payload()["domain_closure_audit_rows"]
    assert [r["domain"] for r in rows] == ["packaging_release", "runtime_safety"]
    assert [r["source_capability_count"] for r in rows] == [22, 18]
    for r in rows:
        assert next(iter(r)) == "closure_row_id"
        assert r["audited_by_block_o_closure"] is True
        assert r["recalculated_by_block_o_closure"] is False
        assert r["domain_closed_in_block_o"] is True
        assert r["execution_ready_after_block_o_closure"] is False
        assert r["execution_authorized_after_block_o_closure"] is False
        assert r["closure_classification"] == "block_o_closed_execution_not_ready_unauthorized"
        assert (
            r["closure_result"] == f"{r['domain']}_block_o_closed_execution_not_ready_unauthorized"
        )


def test_seven_requirement_closure_rows() -> None:
    rows = payload()["requirement_closure_audit_rows"]
    assert len(rows) == 7
    assert [r["requirement_id"] for r in rows] == [
        r["requirement_id"]
        for r in audit.EXPECTED_SOURCE["requirement_authorization_readiness_contract_read_rows"]
    ]
    for r in rows:
        assert next(iter(r)) == "closure_row_id"
        assert r["closure_confirms_requirement_missing"] is True
        assert r["closure_confirms_requirement_not_ready"] is True
        assert r["requirement_closed_as_missing_in_block_o"] is True
        assert r["requirement_ready_after_block_o_closure"] is False
        assert r["execution_authorized_after_block_o_closure"] is False
        assert r["closure_classification"] == "requirement_closed_as_missing_execution_not_ready"


def test_invariant_closure_state() -> None:
    state = payload()["invariant_closure_audit_state"]
    assert state["source_readiness_read_model_preserved"] is True
    assert state["invariants_preserved_at_block_o_closure"] is True
    assert state["invariants_sufficient_for_execution"] is False
    assert state["execution_ready_after_block_o_closure"] is False


def test_exe_closure_state_preserves_lineage() -> None:
    state = payload()["exe_closure_audit_state"]
    assert state["desktop_exe_direction_preserved_at_block_o_closure"] is True
    assert state["block_o_closure_is_not_build_authorization"] is True
    assert state["block_o_closure_is_not_packaging_authorization"] is True
    assert state["block_o_closure_is_not_release_authorization"] is True
    assert state["build_ready_after_block_o_closure"] is False
    assert state["packaging_ready_after_block_o_closure"] is False
    assert state["release_ready_after_block_o_closure"] is False


def test_real_capability_closure_state() -> None:
    state = payload()["real_capability_closure_audit_state"]
    assert state["real_capability_status_inherited_from_17_7"] is True
    assert state["real_capability_status_modified_by_17_8"] is False
    assert state["real_capabilities_opened_by_17_8"] is False
    assert state["all_real_capabilities_blocked_at_block_o_closure"] is True
    assert all(v == "blocked" for v in state["real_capability_status"].values())


def test_fail_closed_closure_decision() -> None:
    decision = payload()["fail_closed_closure_decision"]
    assert decision["block_o_execution_authorization_readiness_read_model_in_17_7"] == "preserved"
    assert decision["block_o_closure_audit_in_17_8"] == "closed"
    assert decision["next_explicit_block"] == "allowed"
    assert decision["execution_readiness_granted_by_17_8"] is False
    assert decision["execution_authorization_granted_by_17_8"] is False


def test_summary_evidence_and_boundaries() -> None:
    p = payload()
    summary = p["closure_audit_summary"]
    assert summary["all_source_contract_conditions_false"] is True
    assert summary["all_source_readiness_grants_false"] is True
    assert summary["all_source_authorization_grants_false"] is True
    assert summary["all_closure_readiness_grants_false"] is True
    assert summary["all_domains_closed_in_block_o"] is True
    assert p["closure_evidence"]["source_accepted"] is True
    assert p["closure_boundaries"]["reads_17_7_only"] is True
    assert p["closure_boundaries"]["no_runtime_orders"] is True
    assert p["closure_boundaries"]["no_packaging_build_release"] is True


def collect_ids(value: Any) -> list[int]:
    ids: list[int] = []
    stack = [value]
    while stack:
        item = stack.pop()
        if type(item) is dict:
            ids.append(id(item))
            stack.extend(item.values())
        elif type(item) is list:
            ids.append(id(item))
            stack.extend(item)
    return ids


def test_nominal_payload_has_no_shared_mutable_containers() -> None:
    ids = collect_ids(payload())
    assert len(ids) == len(set(ids))


def test_independent_builder_calls_do_not_share_state() -> None:
    one = payload()
    two = payload()
    one["real_capability_closure_audit_state"]["real_capability_status"]["build"] = "opened"
    assert (
        two["real_capability_closure_audit_state"]["real_capability_status"]["build"] == "blocked"
    )
    assert one["fail_closed_closure_decision"]["real_capability_status"]["build"] == "blocked"
    assert (
        audit.EXPECTED_SOURCE["real_capability_authorization_readiness_contract_read_state"][
            "real_capability_status"
        ]["build"]
        == "blocked"
    )


def test_forbidden_raw_tokens_absent() -> None:
    text = MODULE.read_text()
    assert "create_order" not in text
    assert "fetch_balance" not in text
    assert "ccxt" not in text


@pytest.mark.parametrize(
    "mutation",
    [
        lambda s: s.pop("schema_version"),
        lambda s: s.__setitem__("extra", True),
        lambda s: s.__setitem__("schema_version", "bad"),
        lambda s: s.__setitem__("execution_authorization_readiness_read_model_ready", 1),
        lambda s: s.__setitem__("ready_for_block_o_8", 1.0),
        lambda s: s.__setitem__("readiness_read_model_summary", {"bad": True}),
        lambda s: s.__setitem__("domain_authorization_readiness_contract_read_rows", []),
        lambda s: s["domain_authorization_readiness_contract_read_rows"].reverse(),
        lambda s: s["requirement_authorization_readiness_contract_read_rows"].pop(),
        lambda s: s["requirement_authorization_readiness_contract_read_rows"][0].__setitem__(
            "closure_result", "bad"
        ),
        lambda s: s["invariant_authorization_readiness_contract_read_state"].__setitem__(
            "x", object()
        ),
        lambda s: s.__setitem__("exe_authorization_readiness_contract_read_state", []),
        lambda s: s["real_capability_authorization_readiness_contract_read_state"][
            "real_capability_status"
        ].__setitem__("build", "open"),
        lambda s: s.__setitem__("fail_closed_readiness_contract_read_decision", {}),
        lambda s: s.__setitem__("source_boundaries", {}),
        lambda s: s.__setitem__("future_steps", ["bad"]),
    ],
)
def test_malformed_sources_block_without_exception(
    monkeypatch: pytest.MonkeyPatch, mutation: Any
) -> None:
    src = copy.deepcopy(audit.EXPECTED_SOURCE)
    mutation(src)
    p = patched(monkeypatch, src, compare=False)
    assert_blocked(p)


def test_invalid_requirements_isolated(monkeypatch: pytest.MonkeyPatch) -> None:
    src = copy.deepcopy(audit.EXPECTED_SOURCE)
    src["requirement_authorization_readiness_contract_read_rows"] = []
    p = patched(monkeypatch, src)
    assert_blocked(p)
    assert [r["source_capability_count"] for r in p["domain_closure_audit_rows"]] == [22, 18]
    assert p["invariant_closure_audit_state"]["invariants_preserved_at_block_o_closure"] is True
    assert all(
        r["closure_classification"] == "source_invalid" for r in p["domain_closure_audit_rows"]
    )
    assert p["closure_audit_summary"]["all_source_contract_conditions_false"] is False


def deep_dict(depth: int) -> dict[str, Any]:
    root: dict[str, Any] = {}
    cur = root
    for _ in range(depth):
        nxt: dict[str, Any] = {}
        cur["x"] = nxt
        cur = nxt
    return root


def deep_list(depth: int) -> list[Any]:
    root: list[Any] = []
    cur = root
    for _ in range(depth):
        nxt: list[Any] = []
        cur.append(nxt)
        cur = nxt
    return root


def test_plain_json_depth_cycles_subclasses_and_shared_refs() -> None:
    deep_d = deep_dict(1500)
    deep_l = deep_list(1500)
    assert audit._all_plain_json(deep_d) is True
    assert audit._all_plain_json(deep_l) is True
    assert audit._all_plain_json(deep_d, max_depth=audit.MAX_DIAGNOSTIC_CONTAINER_DEPTH) is False
    assert audit._all_plain_json(deep_l, max_depth=audit.MAX_DIAGNOSTIC_CONTAINER_DEPTH) is False
    cyclic_dict: dict[str, Any] = {}
    cyclic_dict["self"] = cyclic_dict
    assert audit._all_plain_json(cyclic_dict) is False
    cyclic_list: list[Any] = []
    cyclic_list.append(cyclic_list)
    assert audit._all_plain_json(cyclic_list) is False
    shared: dict[str, Any] = {"v": []}
    assert audit._all_plain_json({"a": shared["v"], "b": shared["v"]}) is True
    assert audit._all_plain_json(1.5) is False

    class DictSubclass(dict):
        pass

    class ListSubclass(list):
        pass

    assert audit._all_plain_json(DictSubclass()) is False
    assert audit._all_plain_json(ListSubclass()) is False
    assert audit._all_plain_json({"x": (1,)}) is False
    assert audit._all_plain_json({"x": {1, 2}}) is False


class Bomb:
    equality_calls = 0
    armed = False

    def __init__(self, target: str) -> None:
        self.target = target

    def __hash__(self) -> int:
        return hash(self.target)

    def __eq__(self, other: object) -> bool:
        if type(self).armed:
            type(self).equality_calls += 1
            raise AssertionError("eq called")
        return False


class Lying(Bomb):
    equality_calls = 0
    armed = False
    __hash__ = Bomb.__hash__

    def __eq__(self, other: object) -> bool:
        if type(self).armed:
            type(self).equality_calls += 1
            return True
        return False


@pytest.mark.parametrize("key", ["schema_version", "readiness_read_model_summary"])
def test_bomb_lying_top_level_keys(monkeypatch: pytest.MonkeyPatch, key: str) -> None:
    for cls in (Bomb, Lying):
        cls.equality_calls = 0
        cls.armed = False
        src = copy.deepcopy(audit.EXPECTED_SOURCE)
        value = src.pop(key)
        src[cls(key)] = value
        cls.equality_calls = 0
        cls.armed = True
        try:
            p = patched(monkeypatch, src, compare=False)
        finally:
            cls.armed = False
        assert_blocked(p)
        assert cls.equality_calls == 0


@pytest.mark.parametrize(
    "section, field",
    [
        ("invariant_authorization_readiness_contract_read_state", "audited_by_block_o_closure"),
        (
            "exe_authorization_readiness_contract_read_state",
            "block_o_closure_is_not_build_authorization",
        ),
        ("source_boundaries", "can_close_block_o"),
        ("source_boundaries", "block_o_closed"),
        ("source_boundaries", "can_feed_next_explicit_block"),
    ],
)
def test_bomb_lying_nested_shadowing(
    monkeypatch: pytest.MonkeyPatch, section: str, field: str
) -> None:
    for cls in (Bomb, Lying):
        cls.equality_calls = 0
        cls.armed = False
        src = copy.deepcopy(audit.EXPECTED_SOURCE)
        original_keys = list(src[section].keys())
        src[section][field] = True
        src[section][cls(field)] = True
        assert len(src[section]) == len(original_keys) + 2
        cls.equality_calls = 0
        cls.armed = True
        try:
            p = patched(monkeypatch, src, compare=False)
        finally:
            cls.armed = False
        assert_blocked(p)
        assert cls.equality_calls == 0


def test_no_shadowing_direct_probes() -> None:
    src = copy.deepcopy(audit.EXPECTED_SOURCE)
    assert (
        audit._no_shadowing(
            src,
            "source_boundaries",
            ["source_block_o_execution_authorization_readiness_read_model"],
        )
        is True
    )
    src["source_boundaries"]["block_o_closed"] = False
    assert audit._no_shadowing(src, "source_boundaries", ["block_o_closed"]) is False
    src = copy.deepcopy(audit.EXPECTED_SOURCE)
    src["source_boundaries"][Bomb("block_o_closed")] = True
    assert audit._no_shadowing(src, "source_boundaries", ["block_o_closed"]) is True


def test_ast_guard() -> None:
    tree = ast.parse(MODULE.read_text())
    imports = [n for n in ast.walk(tree) if isinstance(n, ast.Import)]
    import_from = [n for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)]
    assert imports == []
    assert [n.module for n in import_from] == [
        "__future__",
        "typing",
        "ui.pyside_app.preview_block_o_execution_authorization_readiness_read_model",
    ]
    calls = [n.func for n in ast.walk(tree) if isinstance(n, ast.Call)]
    name_calls = {n.id for n in calls if isinstance(n, ast.Name)}
    attribute_calls = {n.attr for n in calls if isinstance(n, ast.Attribute)}
    assert name_calls == {
        "_all_plain_json",
        "_copy_plain",
        "_exact_plain_matches",
        "_no_shadowing",
        "_owned_fields_are_unshadowed",
        "_plain_dict_section",
        "_plain_list_section",
        "_real_status",
        "_safe_top_level_source",
        "_scalar_reference",
        "_section_valid",
        "_source_identity_valid",
        "_valids",
        "all",
        "bool",
        "build_preview_block_o_execution_authorization_readiness_read_model",
        "id",
        "len",
        "list",
        "reversed",
        "set",
        "type",
        "zip",
    }
    assert attribute_calls == {
        "add",
        "append",
        "discard",
        "get",
        "items",
        "keys",
        "pop",
        "update",
        "upper",
        "values",
    }
    assert (
        sum(
            1
            for n in calls
            if isinstance(n, ast.Name)
            and n.id == "build_preview_block_o_execution_authorization_readiness_read_model"
        )
        == 1
    )
    assert not any("17_6" in name or "17_5" in name for name in name_calls | attribute_calls)
    forbidden = {
        "open",
        "read",
        "write",
        "socket",
        "connect",
        "request",
        "urlopen",
        "getenv",
        "environ",
        "system",
        "popen",
        "run",
        "Popen",
        "git",
        "validate",
        "confirm",
        "grant",
        "gate",
        "runtime",
        "order",
        "package",
        "PyInstaller",
        "build",
        "artifact",
        "release",
        "qml",
        "bridge",
        "gateway",
        "controller",
        "eval",
        "exec",
        "compile",
        "input",
        "__import__",
        "getattr",
        "setattr",
    }
    assert forbidden.isdisjoint(name_calls)
    assert forbidden.isdisjoint(attribute_calls)


def test_float_source_blocks_without_float_in_output(monkeypatch: pytest.MonkeyPatch) -> None:
    src = copy.deepcopy(audit.EXPECTED_SOURCE)
    src["schema_version"] = 1.5
    p = patched(monkeypatch, src)
    assert_blocked(p)
    assert (
        p["block_o_execution_authorization_readiness_read_model_reference"]["schema_version"]
        is None
    )


@pytest.mark.parametrize(
    ("field", "replacement", "validity"),
    [
        ("source_boundaries", {}, "source_boundaries_valid"),
        ("non_execution_readiness_read_evidence", {}, "evidence_valid"),
        ("future_steps", ["bad"], "future_steps_valid"),
        ("readiness_read_model_boundaries", {}, "readiness_read_model_boundaries_valid"),
        ("fail_closed_readiness_contract_read_decision", {}, "fail_closed_valid"),
    ],
)
def test_global_section_invalidity_does_not_degrade_local_closure_facts(
    monkeypatch: pytest.MonkeyPatch, field: str, replacement: Any, validity: str
) -> None:
    src = copy.deepcopy(audit.EXPECTED_SOURCE)
    src[field] = replacement
    p = patched(monkeypatch, src)
    assert_blocked(p)
    assert p["closure_evidence"][validity] is False
    assert p["closure_audit_summary"]["block_o_scope_complete"] is False
    assert all(row["domain_closed_in_block_o"] is True for row in p["domain_closure_audit_rows"])
    assert all(
        row["closure_classification"] == "block_o_closed_execution_not_ready_unauthorized"
        for row in p["domain_closure_audit_rows"]
    )
    assert all(
        row["requirement_closed_as_missing_in_block_o"] is True
        for row in p["requirement_closure_audit_rows"]
    )
    assert p["invariant_closure_audit_state"]["block_o_invariant_scope_complete"] is True
    assert p["exe_closure_audit_state"]["exe_source_only_scope_complete"] is True
    assert p["closure_audit_summary"]["all_domains_closed_in_block_o"] is True
    assert p["closure_audit_summary"]["all_requirements_closed_as_missing"] is True


@pytest.mark.parametrize(
    ("field", "value_factory", "validity", "identity_probe"),
    [
        ("readiness_read_model_summary", lambda: {}, "summary_valid", "self"),
        ("domain_authorization_readiness_contract_read_rows", lambda: [], "domain_rows_valid", 0),
        (
            "invariant_authorization_readiness_contract_read_state",
            lambda: {},
            "invariant_valid",
            "self",
        ),
        ("source_boundaries", lambda: {}, "source_boundaries_valid", "self"),
    ],
)
def test_builder_level_cycles_block_without_mutating_source(
    monkeypatch: pytest.MonkeyPatch,
    field: str,
    value_factory: Any,
    validity: str,
    identity_probe: Any,
) -> None:
    src = copy.deepcopy(audit.EXPECTED_SOURCE)
    value = value_factory()
    if type(value) is dict:
        value[identity_probe] = value
        expected_keys = list(value.keys())
    else:
        value.append(value)
        expected_keys = None
    src[field] = value
    p = patched(monkeypatch, src, compare=False)
    assert_blocked(p)
    assert p["closure_evidence"][validity] is False
    assert src[field] is value
    if type(value) is dict:
        assert list(value.keys()) == expected_keys
        assert value[identity_probe] is value
    else:
        assert value[0] is value


@pytest.mark.parametrize(
    ("field", "value", "validity"),
    [
        ("readiness_read_model_summary", deep_dict(1500), "summary_valid"),
        (
            "invariant_authorization_readiness_contract_read_state",
            deep_dict(1500),
            "invariant_valid",
        ),
        ("exe_authorization_readiness_contract_read_state", deep_dict(1500), "exe_valid"),
        ("schema_version", deep_dict(1500), "summary_valid"),
        ("source_boundaries", deep_dict(1500), "source_boundaries_valid"),
        ("domain_authorization_readiness_contract_read_rows", deep_list(1500), "domain_rows_valid"),
    ],
)
def test_builder_level_deep_sources_block_without_mutating_source(
    monkeypatch: pytest.MonkeyPatch, field: str, value: Any, validity: str
) -> None:
    src = copy.deepcopy(audit.EXPECTED_SOURCE)
    src[field] = value
    p = patched(monkeypatch, src, compare=False)
    assert_blocked(p)
    assert src[field] is value
    if field == "schema_version":
        assert p["block_o_execution_authorization_readiness_read_model_reference"][field] is None
    else:
        assert p["closure_evidence"][validity] is False


COMBINED_SHADOWING_TARGETS = [
    (
        "invariant_authorization_readiness_contract_read_state",
        "audited_by_block_o_closure",
        "invariant_valid",
        "exe_valid",
    ),
    (
        "exe_authorization_readiness_contract_read_state",
        "block_o_closure_is_not_build_authorization",
        "exe_valid",
        "invariant_valid",
    ),
    (
        "source_boundaries",
        "source_block_o_execution_authorization_readiness_read_model",
        "source_boundaries_valid",
        "invariant_valid",
    ),
    (
        "source_boundaries",
        "readiness_read_model_source_preserved",
        "source_boundaries_valid",
        "invariant_valid",
    ),
    ("source_boundaries", "can_close_block_o", "source_boundaries_valid", "invariant_valid"),
    ("source_boundaries", "block_o_closed", "source_boundaries_valid", "invariant_valid"),
    (
        "source_boundaries",
        "can_feed_next_explicit_block",
        "source_boundaries_valid",
        "invariant_valid",
    ),
]


def craft_custom_first_shadowing_section(
    src: dict[str, Any], section: str, field: str, cls: type[Bomb]
) -> tuple[dict[Any, Any], list[Any], list[Any], Bomb]:
    original = src[section]
    original_items = list(original.items())
    removed_items = original_items[-2:]
    assert all(key != field for key, _value in removed_items)
    _removed_key, removed_value = removed_items[-1]
    custom_key = cls(field)
    crafted: dict[Any, Any] = {custom_key: removed_value}
    for key, value in original_items[:-2]:
        crafted[key] = value
    crafted[field] = True
    src[section] = crafted
    keys_before = list(crafted.keys())
    values_snapshot = copy.deepcopy(list(crafted.values()))
    assert len(crafted) == len(original)
    assert next(iter(crafted)) is custom_key
    return crafted, keys_before, values_snapshot, custom_key


def assert_section_not_mutated(
    src: dict[str, Any],
    section: str,
    section_object: dict[Any, Any],
    keys_before: list[Any],
    values_snapshot: list[Any],
    custom_key_object: Bomb,
) -> None:
    assert src[section] is section_object
    current_keys = list(src[section].keys())
    current_values = list(src[section].values())
    assert len(current_keys) == len(keys_before)
    assert len(current_values) == len(values_snapshot)
    assert current_keys[0] is custom_key_object
    assert keys_before[0] is custom_key_object
    for index in range(1, len(current_keys)):
        assert type(current_keys[index]) is str
        assert type(keys_before[index]) is str
        assert current_keys[index] == keys_before[index]
    assert current_values == values_snapshot
    assert next(iter(src[section])) is custom_key_object


def test_section_not_mutated_detects_invariant_nested_list_mutation() -> None:
    src = copy.deepcopy(audit.EXPECTED_SOURCE)
    section_object, keys_before, values_snapshot, custom_key = craft_custom_first_shadowing_section(
        src,
        "invariant_authorization_readiness_contract_read_state",
        "audited_by_block_o_closure",
        Bomb,
    )
    nested_list = next(value for value in section_object.values() if type(value) is list)
    nested_list.append({"mutation_sentinel": True})
    with pytest.raises(AssertionError):
        assert_section_not_mutated(
            src,
            "invariant_authorization_readiness_contract_read_state",
            section_object,
            keys_before,
            values_snapshot,
            custom_key,
        )


def test_section_not_mutated_detects_source_boundaries_nested_dict_mutation() -> None:
    src = copy.deepcopy(audit.EXPECTED_SOURCE)
    section_object, keys_before, values_snapshot, custom_key = craft_custom_first_shadowing_section(
        src,
        "source_boundaries",
        "can_close_block_o",
        Bomb,
    )
    nested_dict = next(value for value in section_object.values() if type(value) is dict)
    nested_dict["mutation_sentinel"] = True
    with pytest.raises(AssertionError):
        assert_section_not_mutated(
            src,
            "source_boundaries",
            section_object,
            keys_before,
            values_snapshot,
            custom_key,
        )


@pytest.mark.parametrize(
    ("section", "field", "validity", "independent_validity"), COMBINED_SHADOWING_TARGETS
)
def test_no_shadowing_direct_custom_first_matrix(
    section: str, field: str, validity: str, independent_validity: str
) -> None:
    _ = validity, independent_validity
    for cls in (Bomb, Lying):
        cls.equality_calls = 0
        cls.armed = False
        src = copy.deepcopy(audit.EXPECTED_SOURCE)
        section_object, keys_before, values_before, custom_key = (
            craft_custom_first_shadowing_section(src, section, field, cls)
        )
        cls.equality_calls = 0
        cls.armed = True
        try:
            result = audit._no_shadowing(src, section, [field])
        finally:
            cls.armed = False
        assert result is False
        assert cls.equality_calls == 0
        assert_section_not_mutated(
            src, section, section_object, keys_before, values_before, custom_key
        )


@pytest.mark.parametrize(
    ("section", "field", "validity", "independent_validity"), COMBINED_SHADOWING_TARGETS
)
def test_combined_bomb_lying_collision_and_exact_string_shadowing(
    monkeypatch: pytest.MonkeyPatch,
    section: str,
    field: str,
    validity: str,
    independent_validity: str,
) -> None:
    for cls in (Bomb, Lying):
        cls.equality_calls = 0
        cls.armed = False
        src = copy.deepcopy(audit.EXPECTED_SOURCE)
        section_object, keys_before, values_before, custom_key = (
            craft_custom_first_shadowing_section(src, section, field, cls)
        )
        cls.equality_calls = 0
        cls.armed = True
        try:
            result = audit._no_shadowing(src, section, [field])
            p = patched(monkeypatch, src, compare=False)
        finally:
            cls.armed = False
        assert result is False
        assert cls.equality_calls == 0
        assert_blocked(p)
        assert p["closure_evidence"][validity] is False
        assert p["closure_evidence"][independent_validity] is True
        assert_section_not_mutated(
            src, section, section_object, keys_before, values_before, custom_key
        )


SOURCE_SCALAR_FIELDS = [
    "schema_version",
    "block_o_execution_authorization_readiness_read_model_kind",
    "block",
    "step",
    "execution_authorization_readiness_read_model_status",
    "execution_authorization_readiness_read_model_decision",
    "execution_authorization_readiness_read_model_ready",
    "ready_for_block_o_8",
    "next_step",
    "next_step_title",
    "status",
]


def changed_scalar_values(field: str) -> list[Any]:
    expected = audit.SOURCE_IDENTITY_EXPECTED[field]
    if type(expected) is bool:
        return [not expected, 1]
    return [f"changed_{field}"]


@pytest.mark.parametrize("field", SOURCE_SCALAR_FIELDS)
def test_top_level_scalar_identity_matrix_blocks_exact_mutations(
    monkeypatch: pytest.MonkeyPatch, field: str
) -> None:
    for changed in changed_scalar_values(field):
        src = copy.deepcopy(audit.EXPECTED_SOURCE)
        src[field] = changed
        p = patched(monkeypatch, src)
        assert_blocked(p)
        assert p["closure_evidence"]["source_accepted"] is False


def test_changed_source_status_blocks_closure(monkeypatch: pytest.MonkeyPatch) -> None:
    src = copy.deepcopy(audit.EXPECTED_SOURCE)
    src["execution_authorization_readiness_read_model_status"] = "changed_status"
    p = patched(monkeypatch, src)
    assert_blocked(p)
    assert p["closure_evidence"]["source_accepted"] is False


def test_changed_source_decision_blocks_closure(monkeypatch: pytest.MonkeyPatch) -> None:
    src = copy.deepcopy(audit.EXPECTED_SOURCE)
    src["execution_authorization_readiness_read_model_decision"] = "CHANGED_DECISION"
    p = patched(monkeypatch, src)
    assert_blocked(p)
    assert p["closure_evidence"]["source_accepted"] is False
