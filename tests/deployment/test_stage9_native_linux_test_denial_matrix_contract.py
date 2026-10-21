"""S9-RPI-34: declared denial outcomes cannot authorize production."""

from __future__ import annotations

import copy
import json

import pytest

from deployment import stage9_native_linux_test_denial_matrix_contract as contract


def _complete(snapshot="SYNTHETIC_TEST_INPUT"):
    return {
        "schema": contract._SCHEMA,
        "snapshot": snapshot,
        "cells": [
            {
                "actor": actor,
                "owner": owner,
                "operation": operation,
                "observation": "CLAIMED_DENIED",
            }
            for actor, owner, operation in contract.expected_cases()
        ],
    }


def _blocked(report):
    assert report["schema"] == contract._SCHEMA
    assert report["status"] == "BLOCKED"
    assert report["qualification"] == "NOT_QUALIFIED"
    assert report["production_emission"] == "BLOCKED"
    assert report["topology"] == "PROPOSED_NOT_APPROVED"
    assert report["blockers"]
    assert set(report) == {
        "schema",
        "status",
        "qualification",
        "production_emission",
        "topology",
        "snapshot",
        "blockers",
        "matrix",
        "counts",
    }


def _rejected(record):
    report = contract.inspect_declared_matrix(record)
    _blocked(report)
    assert report["matrix"]["observation"] == "NOT_QUALIFIED"
    assert report["matrix"]["reason"] == "INVALID_TEST_REPORT"
    assert report["counts"]["claimed_denied"] == 0
    return report


def test_closed_matrix_has_six_roles_and_ninety_exclusive_denial_cells():
    cases = contract.expected_cases()
    assert len(cases) == 90
    assert len(set(cases)) == 90
    assert set(role for role, _owner, _operation in cases) == {
        "root_source",
        "history_source",
        "cca",
        "root_signer",
        "history_signer",
        "issuer",
    }
    assert all(actor != owner for actor, owner, _operation in cases)
    assert set(operation for _actor, _owner, operation in cases) == {
        "read",
        "write",
        "traverse",
    }
    assert cases == contract.expected_cases()


def test_missing_report_is_blocked_and_does_not_inspect_host():
    report = contract.inspect_declared_matrix()
    _blocked(report)
    assert report["snapshot"] == "NOT_QUALIFIED"
    assert report["matrix"] == {
        "observation": "NOT_QUALIFIED",
        "reason": "NO_TEST_OBSERVATIONS",
    }
    assert report["counts"] == {
        "expected": 90,
        "claimed_denied": 0,
        "claimed_allowed": 0,
        "not_observed": 0,
        "missing": 90,
    }


@pytest.mark.parametrize("snapshot", ["SYNTHETIC_TEST_INPUT", "UNTRUSTED_LIVE_OBSERVATION"])
def test_complete_declared_denials_never_grant_production_authority(snapshot):
    report = contract.inspect_declared_matrix(_complete(snapshot))
    _blocked(report)
    assert report["snapshot"] == snapshot
    assert report["matrix"] == {
        "observation": "UNTRUSTED_DECLARED_DENIALS_ONLY",
        "reason": "NO_INDEPENDENT_HOST_ATTESTATION",
    }
    assert report["counts"] == {
        "expected": 90,
        "claimed_denied": 90,
        "claimed_allowed": 0,
        "not_observed": 0,
        "missing": 0,
    }


@pytest.mark.parametrize(
    "observation,reason",
    [
        ("CLAIMED_ALLOWED", "DECLARED_CROSS_ROLE_ACCESS"),
        ("NOT_OBSERVED", "INCOMPLETE_TEST_DENIAL_MATRIX"),
    ],
)
@pytest.mark.parametrize("index", [0, 44, 89])
def test_single_failure_at_any_matrix_position_blocks_summary(observation, reason, index):
    record = _complete()
    record["cells"][index]["observation"] = observation
    result = contract.inspect_declared_matrix(record)
    _blocked(result)
    assert result["matrix"] == {"observation": "NOT_QUALIFIED", "reason": reason}
    assert result["counts"]["claimed_denied"] == 89
    assert result["counts"]["claimed_allowed"] == (observation == "CLAIMED_ALLOWED")
    assert result["counts"]["not_observed"] == (observation == "NOT_OBSERVED")


@pytest.mark.parametrize("index", [0, 34, 89])
def test_one_missing_cell_is_incomplete_not_a_clean_denial(index):
    record = _complete()
    del record["cells"][index]
    result = contract.inspect_declared_matrix(record)
    _blocked(result)
    assert result["matrix"]["reason"] == "INCOMPLETE_TEST_DENIAL_MATRIX"
    assert result["counts"]["missing"] == 1
    assert result["counts"]["claimed_denied"] == 89


def test_allowed_cell_takes_priority_over_missing_or_unobserved():
    record = _complete()
    record["cells"][0]["observation"] = "CLAIMED_ALLOWED"
    record["cells"][1]["observation"] = "NOT_OBSERVED"
    del record["cells"][2]
    result = contract.inspect_declared_matrix(record)
    _blocked(result)
    assert result["matrix"]["reason"] == "DECLARED_CROSS_ROLE_ACCESS"
    assert result["counts"]["claimed_denied"] == 87
    assert result["counts"]["missing"] == 1


@pytest.mark.parametrize(
    "replacement",
    [3, True, (), "PRIVATE_INPUT", ["PRIVATE_INPUT"], {"private": True}],
)
def test_refuse_non_mapping_and_unexpected_input_shapes(replacement):
    _rejected(replacement)


@pytest.mark.parametrize(
    "mutate",
    [
        lambda r: r.update(schema="WRONG"),
        lambda r: r.update(schema=None),
        lambda r: r.update(snapshot="READY"),
        lambda r: r.update(snapshot=None),
        lambda r: r.update(snapshot=True),
        lambda r: r.update(private="PRIVATE_INPUT"),
        lambda r: r.pop("cells"),
        lambda r: r.pop("snapshot"),
        lambda r: r.update(cells="PRIVATE_INPUT"),
        lambda r: r.update(cells=None),
        lambda r: r.update(cells=()),
        lambda r: r["cells"].append(r["cells"][0].copy()),
    ],
)
def test_refuse_invalid_envelope_and_oversized_matrix(mutate):
    record = _complete()
    mutate(record)
    _rejected(record)


@pytest.mark.parametrize(
    "mutation",
    [
        {"actor": "OTHER"},
        {"actor": None},
        {"actor": True},
        {"owner": "OTHER"},
        {"owner": None},
        {"owner": True},
        {"operation": "execute"},
        {"operation": None},
        {"observation": "GO"},
        {"observation": "DENIED"},
        {"observation": True},
        {"observation": None},
        {"extra": "PRIVATE_INPUT"},
    ],
)
def test_refuse_unknown_roles_operations_values_and_private_fields(mutation):
    record = _complete()
    record["cells"][0].update(mutation)
    _rejected(record)


@pytest.mark.parametrize("replacement", [[], None, "PRIVATE_INPUT", 42, True, ()])
def test_refuse_non_object_cells(replacement):
    record = _complete()
    record["cells"][0] = replacement
    _rejected(record)


def test_refuse_actor_equal_to_resource_owner():
    record = _complete()
    record["cells"][0]["owner"] = record["cells"][0]["actor"]
    _rejected(record)


@pytest.mark.parametrize("field", ["actor", "owner", "operation", "observation"])
def test_refuse_missing_cell_fields(field):
    record = _complete()
    record["cells"][0].pop(field)
    _rejected(record)


def test_refuse_duplicate_entries_even_with_different_claims():
    record = _complete()
    record["cells"][1] = dict(record["cells"][0], observation="CLAIMED_ALLOWED")
    _rejected(record)


def test_reorder_all_valid_rows_without_changing_semantics():
    record = _complete()
    record["cells"].reverse()
    result = contract.inspect_declared_matrix(record)
    _blocked(result)
    assert result["matrix"]["observation"] == "UNTRUSTED_DECLARED_DENIALS_ONLY"


def test_empty_cells_are_incomplete_and_not_qualified():
    record = _complete()
    record["cells"] = []
    result = contract.inspect_declared_matrix(record)
    _blocked(result)
    assert result["matrix"]["observation"] == "NOT_QUALIFIED"
    assert result["counts"]["missing"] == 90


def test_rejected_sensitive_input_is_not_ever_reflected_in_output():
    record = _complete()
    record["cells"][7]["observation"] = "PRIVATE_SIGNING_KEY"
    record["cells"][8]["private_secret"] = "PRIVATE_SIGNING_KEY"
    result = contract.inspect_declared_matrix(record)
    _blocked(result)
    assert "PRIVATE_SIGNING_KEY" not in json.dumps(result)
    assert "root_source" not in json.dumps(result)


def test_result_does_not_alias_mutable_caller_input():
    record = _complete()
    result = contract.inspect_declared_matrix(record)
    original = copy.deepcopy(result)
    record["cells"][0]["observation"] = "CLAIMED_ALLOWED"
    record["snapshot"] = "UNKNOWN"
    assert result == original


def test_cli_always_returns_two_and_only_blocked_fixed_json(capsys):
    assert contract.main() == 2
    report = json.loads(capsys.readouterr().out)
    _blocked(report)
    assert report["counts"]["missing"] == 90
    assert report["snapshot"] == "NOT_QUALIFIED"
