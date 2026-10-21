"""S9-RPI-38: offline plan never implies successful physical execution."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from deployment import (
    stage9_native_linux_test_execution_plan as planner,
    stage9_native_linux_test_syscall_outcome_classifier as outcomes,
)


def _valid():
    return planner.build_execution_plan()


def _blocked(result):
    assert set(result) == {
        "schema",
        "status",
        "qualification",
        "production_emission",
        "topology",
        "snapshot",
        "blockers",
        "plan",
        "counts",
    }
    assert result["schema"] == planner._SCHEMA
    assert result["status"] == "BLOCKED"
    assert result["qualification"] == "NOT_QUALIFIED"
    assert result["production_emission"] == "BLOCKED"
    assert result["topology"] == "PROPOSED_NOT_APPROVED"
    assert result["blockers"]
    assert result["counts"]["expected_total"] == 108
    assert result["counts"]["expected_owner_controls"] == 18
    assert result["counts"]["expected_cross_role"] == 90


def _invalid(record):
    result = planner.inspect_execution_plan(record)
    _blocked(result)
    assert result["plan"] == {
        "observation": "NOT_QUALIFIED",
        "reason": "INVALID_TEST_EXECUTION_PLAN",
    }
    assert result["snapshot"] == "NOT_QUALIFIED"
    assert result["counts"]["accepted_owner_controls"] == 0
    assert result["counts"]["accepted_cross_role"] == 0
    assert result["counts"]["missing"] == 108
    return result


def test_canonical_plan_has_exactly_eighteen_controls_then_ninety_denials():
    plan = _valid()
    assert set(plan) == {"schema", "snapshot", "steps"}
    assert plan["schema"] == planner._SCHEMA
    assert plan["snapshot"] == "SYNTHETIC_TEST_PLAN"
    steps = plan["steps"]
    assert len(steps) == 108
    assert [row["id"] for row in steps[:18]] == [f"OWN-{i:03d}" for i in range(1, 19)]
    assert [row["id"] for row in steps[18:]] == [f"CROSS-{i:03d}" for i in range(1, 91)]
    assert all(row["phase"] == "OWNER_CONTROL_FIRST" for row in steps[:18])
    assert all(row["phase"] == "CROSS_ROLE_AFTER_OWNER" for row in steps[18:])
    assert len({row["id"] for row in steps}) == 108


def test_each_negative_case_requires_matching_owner_positive_control():
    steps = _valid()["steps"]
    positives = {
        (row["actor"], row["operation"]): row
        for row in steps
        if row["phase"] == "OWNER_CONTROL_FIRST"
    }
    assert len(positives) == 18
    for row in steps[18:]:
        assert row["actor"] != row["owner"]
        assert row["expected"] == "EACCES_OR_EPERM"
        assert row["owner_control"] == positives[(row["owner"], row["operation"])]["id"]
    for row in steps[:18]:
        assert row["actor"] == row["owner"]
        assert row["expected"] == "SUCCESS"
        assert row["owner_control"] == "SELF"


def test_plan_matches_previously_approved_closed_108_case_inventory():
    steps = _valid()["steps"]
    cases = {(row["actor"], row["owner"], row["operation"]) for row in steps}
    assert cases == set(outcomes.expected_cases())
    assert len(cases) == 108
    assert set(row["operation"] for row in steps) == {
        "read",
        "write",
        "traverse",
    }


def test_complete_valid_plan_is_still_untrusted_and_blocked():
    result = planner.inspect_execution_plan(_valid())
    _blocked(result)
    assert result["snapshot"] == "SYNTHETIC_TEST_PLAN"
    assert result["plan"] == {
        "observation": "UNTRUSTED_OFFLINE_TEST_PLAN_ONLY",
        "reason": "NO_NATIVE_EXECUTION_OR_INDEPENDENT_HOST_EVIDENCE",
    }
    assert result["counts"]["accepted_owner_controls"] == 18
    assert result["counts"]["accepted_cross_role"] == 90
    assert result["counts"]["missing"] == 0


def test_default_no_input_is_not_qualified():
    result = planner.inspect_execution_plan()
    _blocked(result)
    assert result["plan"]["reason"] == "NO_TEST_EXECUTION_PLAN"
    assert result["snapshot"] == "NOT_QUALIFIED"
    assert result["counts"]["missing"] == 108


@pytest.mark.parametrize("n", [0, 1, 17, 18, 19, 89, 107])
def test_valid_canonical_prefix_is_incomplete_not_successful(n):
    record = _valid()
    record["steps"] = record["steps"][:n]
    result = planner.inspect_execution_plan(record)
    _blocked(result)
    assert result["plan"] == {
        "observation": "NOT_QUALIFIED",
        "reason": "INCOMPLETE_TEST_EXECUTION_PLAN",
    }
    assert result["counts"]["missing"] == 108 - n
    assert result["counts"]["accepted_owner_controls"] == min(n, 18)
    assert result["counts"]["accepted_cross_role"] == max(0, n - 18)


@pytest.mark.parametrize("first,second", [(0, 1), (17, 18), (18, 19), (40, 100)])
def test_reordered_steps_or_cross_first_are_invalid(first, second):
    record = _valid()
    record["steps"][first], record["steps"][second] = (
        record["steps"][second],
        record["steps"][first],
    )
    _invalid(record)


@pytest.mark.parametrize("index", [0, 17, 18, 107])
@pytest.mark.parametrize(
    "field,value",
    [
        ("phase", "EXECUTED"),
        ("id", "PRIVATE_SECRET"),
        ("actor", "UNKNOWN"),
        ("owner", "UNKNOWN"),
        ("operation", "execute"),
        ("expected", "TRUSTED_PASS"),
        ("owner_control", "OWN-999"),
        ("expected", True),
        ("actor", None),
    ],
)
def test_any_tampered_step_is_rejected_without_partial_credit(index, field, value):
    record = _valid()
    record["steps"][index][field] = value
    _invalid(record)


def test_cross_role_cannot_claim_self_approval():
    record = _valid()
    record["steps"][18]["owner_control"] = "SELF"
    _invalid(record)


@pytest.mark.parametrize("index", [0, 17, 18, 107])
def test_hidden_extra_field_in_any_position_is_rejected(index):
    record = _valid()
    record["steps"][index]["host_path"] = "/tmp/PRIVATE_SECRET"
    output = _invalid(record)
    assert "PRIVATE_SECRET" not in json.dumps(output)


@pytest.mark.parametrize(
    "mutation",
    [
        lambda p: p.update(schema="OTHER"),
        lambda p: p.update(schema=None),
        lambda p: p.update(snapshot="UNTRUSTED_LIVE_OBSERVATION"),
        lambda p: p.update(snapshot=None),
        lambda p: p.update(secret="PRIVATE_SECRET"),
        lambda p: p.pop("steps"),
        lambda p: p.pop("schema"),
        lambda p: p.pop("snapshot"),
        lambda p: p.update(steps=None),
        lambda p: p.update(steps=()),
        lambda p: p.update(steps="PRIVATE_SECRET"),
        lambda p: p["steps"].append(copy.deepcopy(p["steps"][0])),
    ],
)
def test_invalid_outer_schema_and_oversized_plan_rejected(mutation):
    plan = _valid()
    mutation(plan)
    _invalid(plan)


@pytest.mark.parametrize(
    "bad",
    [True, 2, (), [], "PRIVATE_SECRET", {"private": "PRIVATE_SECRET"}],
)
def test_non_mapping_inputs_fail_closed(bad):
    result = _invalid(bad)
    assert "PRIVATE_SECRET" not in json.dumps(result)


@pytest.mark.parametrize("replacement", [True, None, 4, (), [], "PRIVATE_SECRET"])
def test_non_mapping_steps_are_rejected(replacement):
    record = _valid()
    record["steps"][40] = replacement
    _invalid(record)


@pytest.mark.parametrize("field", list(planner._EXPECTED_FIELDS))
def test_missing_step_field_is_rejected(field):
    record = _valid()
    record["steps"][90].pop(field)
    _invalid(record)


def test_repeated_getter_returns_independent_mutable_plans():
    first = _valid()
    second = _valid()
    expected = copy.deepcopy(second)
    first["steps"][1]["expected"] = "PRIVATE_SECRET"
    first["steps"].pop()
    assert second == expected
    assert _valid() == expected


def test_canonical_descriptors_cannot_be_mutated_through_plan_or_directly():
    canonical = [dict(row) for row in planner._STEPS]
    plan = _valid()
    plan["steps"][0]["expected"] = "PRIVATE_SECRET"
    plan["steps"].clear()
    assert [dict(row) for row in planner._STEPS] == canonical
    assert _valid()["steps"] == canonical
    with pytest.raises(TypeError):
        planner._STEPS[0]["expected"] = "PRIVATE_SECRET"
    with pytest.raises(TypeError):
        planner._STEPS[0] = {}
    assert [dict(row) for row in planner._STEPS] == canonical


@pytest.mark.parametrize("optimize", [0, 1, 2])
@pytest.mark.parametrize("damage", ["missing_cross_case", "wrong_split", "duplicate_cross_case"])
def test_invalid_catalog_prevents_module_initialization_even_with_optimization(
    monkeypatch, optimize, damage
):
    cases = outcomes.expected_cases()
    if damage == "missing_cross_case":
        # Retain all six roles, but remove one of the 90 cross-role cases.
        broken = cases[1:]
    elif damage == "duplicate_cross_case":
        # Replace one negative case with a duplicate while preserving 18/90.
        broken = cases[:1] + cases[:1] + cases[2:]
        assert len(broken) == 108
        assert sum(actor != owner for actor, owner, _ in broken) == 90
        assert sum(actor == owner for actor, owner, _ in broken) == 18
        assert len(set(broken)) == 107
    else:
        # Keep 108 generated steps, with an invalid 15/93 split. This
        # catches guards that only check the total number of steps.
        roles = tuple(dict.fromkeys(actor for actor, _, _ in cases))[:5]
        cross = tuple(
            case for case in cases if case[0] in roles and case[1] in roles and case[0] != case[1]
        )
        broken = cross + cross[:33]
        assert len(broken) == 93
    monkeypatch.setattr(outcomes, "expected_cases", lambda: broken)
    source = Path(planner.__file__).read_text(encoding="utf-8")
    code = compile(source, planner.__file__, "exec", optimize=optimize)
    namespace = {"__name__": planner.__name__}
    with pytest.raises(RuntimeError, match="Invalid canonical TEST execution plan catalog"):
        exec(code, namespace)
    assert "build_execution_plan" not in namespace
    assert "inspect_execution_plan" not in namespace


def test_cli_emits_sanitized_summary_and_returns_two(capsys):
    assert planner.main() == 2
    report = json.loads(capsys.readouterr().out)
    _blocked(report)
    assert report["plan"]["observation"] == "UNTRUSTED_OFFLINE_TEST_PLAN_ONLY"
    assert "root_source" not in json.dumps(report)
    assert "PRIVATE_SECRET" not in json.dumps(report)
