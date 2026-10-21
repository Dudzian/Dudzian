"""S9-RPI-35: declared syscall outcomes are never effective denial proof."""

from __future__ import annotations

import copy
import json

import pytest

from deployment import stage9_native_linux_test_syscall_outcome_classifier as adapter


def _full(snapshot="SYNTHETIC_TEST_INPUT"):
    return {
        "schema": adapter._SCHEMA,
        "snapshot": snapshot,
        "cases": [
            {
                "actor": actor,
                "owner": owner,
                "operation": operation,
                "outcome": "DENIED_EACCES" if actor != owner else "SUCCESS",
            }
            for actor, owner, operation in adapter.expected_cases()
        ],
    }


def _blocked(report):
    assert report["schema"] == adapter._SCHEMA
    assert report["status"] == "BLOCKED"
    assert report["qualification"] == "NOT_QUALIFIED"
    assert report["production_emission"] == "BLOCKED"
    assert report["topology"] == "PROPOSED_NOT_APPROVED"
    assert len(report["blockers"]) > 0


def _reject(record):
    report = adapter.inspect_declared_outcomes(record)
    _blocked(report)
    assert report["classification"] == {
        "observation": "NOT_QUALIFIED",
        "reason": "INVALID_TEST_REPORT",
    }
    assert report["counts"]["cross_denied"] == 0
    assert report["counts"]["owner_success"] == 0
    return report


def test_fixture_plan_contains_exact_ninety_denials_and_eighteen_owner_controls():
    cases = adapter.expected_cases()
    assert len(cases) == 108
    assert len(set(cases)) == 108
    cross = [case for case in cases if case[0] != case[1]]
    owner = [case for case in cases if case[0] == case[1]]
    assert len(cross) == 90
    assert len(owner) == 18
    assert tuple(cross) == adapter._CROSS_CASES
    assert {role for role, _, _ in owner} == set(adapter._ROLES)
    assert set(operation for _, _, operation in owner) == {"read", "write", "traverse"}


def test_no_input_has_closed_default_without_host_observation():
    report = adapter.inspect_declared_outcomes()
    _blocked(report)
    assert report["snapshot"] == "NOT_QUALIFIED"
    assert report["classification"]["reason"] == "NO_TEST_OBSERVATIONS"
    assert report["counts"] == {
        "cross_expected": 90,
        "cross_denied": 0,
        "cross_allowed": 0,
        "cross_inconclusive": 0,
        "cross_missing": 90,
        "owner_expected": 18,
        "owner_success": 0,
        "owner_failed": 0,
        "owner_missing": 18,
    }


@pytest.mark.parametrize(
    "snapshot", ["SYNTHETIC_TEST_INPUT", "UNTRUSTED_LIVE_OBSERVATION"]
)
def test_complete_declared_denials_and_positive_controls_remain_untrusted(snapshot):
    report = adapter.inspect_declared_outcomes(_full(snapshot))
    _blocked(report)
    assert report["snapshot"] == snapshot
    assert report["classification"] == {
        "observation": "UNTRUSTED_DECLARED_DENIAL_WITH_OWNER_CONTROLS_ONLY",
        "reason": "NO_INDEPENDENT_KERNEL_EVIDENCE",
    }
    assert report["denial_matrix"] == {
        "observation": "UNTRUSTED_DECLARED_DENIALS_ONLY",
        "reason": "NO_INDEPENDENT_HOST_ATTESTATION",
    }
    assert report["counts"]["cross_denied"] == 90
    assert report["counts"]["owner_success"] == 18


@pytest.mark.parametrize("errno_result", ["DENIED_EACCES", "DENIED_EPERM"])
def test_denial_labels_are_only_untrusted_claims(errno_result):
    record = _full()
    for case in record["cases"]:
        if case["actor"] != case["owner"]:
            case["outcome"] = errno_result
    result = adapter.inspect_declared_outcomes(record)
    _blocked(result)
    assert result["classification"]["observation"].startswith("UNTRUSTED_DECLARED")
    assert result["counts"]["cross_denied"] == 90


@pytest.mark.parametrize("outcome", [
    "SUCCESS",
    "ENOENT",
    "OTHER_FAILURE",
    "NOT_ATTEMPTED",
])
@pytest.mark.parametrize("index", [0, 45, 89])
def test_cross_role_success_and_non_permission_errors_do_not_count_as_denial(
    outcome, index
):
    record = _full()
    record["cases"][index]["outcome"] = outcome
    result = adapter.inspect_declared_outcomes(record)
    _blocked(result)
    assert result["counts"]["cross_denied"] == 89
    if outcome == "SUCCESS":
        assert result["classification"]["reason"] == "DECLARED_CROSS_ROLE_ACCESS"
        assert result["counts"]["cross_allowed"] == 1
        assert result["denial_matrix"]["reason"] == "DECLARED_CROSS_ROLE_ACCESS"
    else:
        assert (
            result["classification"]["reason"]
            == "CROSS_ROLE_DENIAL_INCOMPLETE_OR_INCONCLUSIVE"
        )
        assert result["counts"]["cross_inconclusive"] == 1
        assert result["denial_matrix"]["reason"] == "INCOMPLETE_TEST_DENIAL_MATRIX"


@pytest.mark.parametrize("outcome", [
    "DENIED_EACCES",
    "DENIED_EPERM",
    "ENOENT",
    "OTHER_FAILURE",
    "NOT_ATTEMPTED",
])
def test_owner_positive_control_failure_blocks_complete_denial(outcome):
    record = _full()
    record["cases"][90]["outcome"] = outcome
    result = adapter.inspect_declared_outcomes(record)
    _blocked(result)
    assert result["classification"]["reason"] == "OWNER_CONTROL_FAILED_OR_INCONCLUSIVE"
    assert result["counts"]["owner_success"] == 17
    assert result["counts"]["owner_failed"] == 1


@pytest.mark.parametrize("position", [0, 89, 90, 107])
def test_missing_any_negative_or_owner_control_never_produces_complete_label(position):
    record = _full()
    del record["cases"][position]
    report = adapter.inspect_declared_outcomes(record)
    _blocked(report)
    assert report["classification"]["observation"] == "NOT_QUALIFIED"
    assert report["classification"]["reason"] == (
        "OWNER_CONTROL_INCOMPLETE" if position >= 90
        else "CROSS_ROLE_DENIAL_INCOMPLETE_OR_INCONCLUSIVE"
    )


def test_cross_role_success_has_priority_over_failed_owner_control():
    record = _full()
    record["cases"][3]["outcome"] = "SUCCESS"
    record["cases"][95]["outcome"] = "ENOENT"
    result = adapter.inspect_declared_outcomes(record)
    _blocked(result)
    assert result["classification"]["reason"] == "DECLARED_CROSS_ROLE_ACCESS"


def test_multiple_incomplete_failure_types_stay_blocked():
    record = _full()
    record["cases"][2]["outcome"] = "ENOENT"
    record["cases"][93]["outcome"] = "NOT_ATTEMPTED"
    del record["cases"][-1]
    result = adapter.inspect_declared_outcomes(record)
    _blocked(result)
    assert result["classification"]["reason"] == "OWNER_CONTROL_FAILED_OR_INCONCLUSIVE"
    assert result["counts"]["cross_inconclusive"] == 1
    assert result["counts"]["owner_missing"] == 1


@pytest.mark.parametrize(
    "bad",
    [4, True, [], "PRIVATE_SECRET", {"secret": "PRIVATE_SECRET"}, (1, 2)],
)
def test_non_record_input_is_rejected_without_data_exposure(bad):
    report = _reject(bad)
    assert "PRIVATE_SECRET" not in json.dumps(report)


@pytest.mark.parametrize(
    "mutate",
    [
        lambda r: r.update(schema="WRONG"),
        lambda r: r.update(schema=None),
        lambda r: r.update(snapshot="TRUSTED_LIVE"),
        lambda r: r.update(snapshot=42),
        lambda r: r.update(extra="PRIVATE_SECRET"),
        lambda r: r.pop("schema"),
        lambda r: r.pop("snapshot"),
        lambda r: r.pop("cases"),
        lambda r: r.update(cases={}),
        lambda r: r.update(cases=None),
        lambda r: r.update(cases="PRIVATE_SECRET"),
        lambda r: r["cases"].append(dict(r["cases"][0])),
    ],
)
def test_invalid_header_and_oversized_report_are_rejected(mutate):
    record = _full()
    mutate(record)
    _reject(record)


@pytest.mark.parametrize(
    "change",
    [
        {"actor": "OTHER"},
        {"actor": True},
        {"actor": None},
        {"owner": "OTHER"},
        {"owner": 42},
        {"operation": "execute"},
        {"operation": None},
        {"outcome": "DENIED_ENOENT"},
        {"outcome": "PASS"},
        {"outcome": False},
        {"outcome": None},
        {"private": "PRIVATE_SECRET"},
    ],
)
def test_invalid_case_shape_or_outcome_rejected(change):
    record = _full()
    record["cases"][23].update(change)
    _reject(record)


@pytest.mark.parametrize("field", ["actor", "owner", "operation", "outcome"])
def test_missing_required_cell_field_is_rejected(field):
    record = _full()
    del record["cases"][0][field]
    _reject(record)


@pytest.mark.parametrize("cell", [None, 5, False, (), [], "PRIVATE_SECRET"])
def test_non_mapping_cell_is_rejected(cell):
    record = _full()
    record["cases"][0] = cell
    _reject(record)


def test_duplicate_case_refuses_even_when_second_outcome_disagrees():
    record = _full()
    record["cases"][1] = dict(record["cases"][0], outcome="SUCCESS")
    _reject(record)


def test_unordered_complete_case_list_has_same_classification():
    record = _full()
    record["cases"].reverse()
    result = adapter.inspect_declared_outcomes(record)
    _blocked(result)
    assert result["counts"]["cross_denied"] == 90
    assert result["classification"]["observation"].startswith("UNTRUSTED_DECLARED")


def test_no_positive_controls_does_not_qualify_even_if_all_cross_claims_denied():
    record = _full()
    record["cases"] = record["cases"][:90]
    result = adapter.inspect_declared_outcomes(record)
    _blocked(result)
    assert result["denial_matrix"]["observation"] == "UNTRUSTED_DECLARED_DENIALS_ONLY"
    assert result["classification"]["reason"] == "OWNER_CONTROL_INCOMPLETE"
    assert result["counts"]["owner_missing"] == 18


def test_invalid_report_cannot_reflect_partial_case_counts_or_private_data():
    record = _full()
    record["cases"][66]["outcome"] = "PRIVATE_SIGNING_KEY"
    report = _reject(record)
    payload = json.dumps(report)
    assert "PRIVATE_SIGNING_KEY" not in payload
    assert "root_source" not in payload


def test_result_does_not_alias_input():
    record = _full()
    report = adapter.inspect_declared_outcomes(record)
    original = copy.deepcopy(report)
    record["cases"][0]["outcome"] = "SUCCESS"
    record["snapshot"] = "OTHER"
    assert report == original


def test_command_prints_no_observations_and_returns_two(capsys):
    assert adapter.main() == 2
    output = json.loads(capsys.readouterr().out)
    _blocked(output)
    assert output["classification"]["reason"] == "NO_TEST_OBSERVATIONS"
