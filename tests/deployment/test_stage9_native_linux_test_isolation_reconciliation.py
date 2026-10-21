"""S9-RPI-37: no concatenation of forged TEST claims can qualify a host."""

from __future__ import annotations

import copy
import json

import pytest

from deployment import stage9_native_linux_test_fixture_readiness as fixture
from deployment import stage9_native_linux_test_isolation_reconciliation as reconcile
from deployment import stage9_native_linux_test_syscall_outcome_classifier as outcomes


def _fixture(snapshot="SYNTHETIC_TEST_INPUT"):
    return {
        "schema": fixture._SCHEMA,
        "snapshot": snapshot,
        "roles": [
            {
                "role": role,
                "uid": 66000 + i,
                "gid": 67000 + i,
                "effective_gids": [67000 + i],
                "directory_owner_uid": 66000 + i,
                "directory_owner_gid": 67000 + i,
                "directory_mode": 0o700,
                "file_owner_uid": 66000 + i,
                "file_owner_gid": 67000 + i,
                "file_mode": 0o600,
                "symlink_free_claim": True,
                "acl_absent_claim": True,
                "unprivileged_claim": True,
                "private_parent_chain_claim": True,
            }
            for i, role in enumerate(fixture.expected_roles())
        ],
    }


def _outcomes(snapshot="SYNTHETIC_TEST_INPUT"):
    return {
        "schema": outcomes._SCHEMA,
        "snapshot": snapshot,
        "cases": [
            {
                "actor": actor,
                "owner": owner,
                "operation": operation,
                "outcome": "DENIED_EACCES" if actor != owner else "SUCCESS",
            }
            for actor, owner, operation in outcomes.expected_cases()
        ],
    }


def _record(snapshot="SYNTHETIC_TEST_INPUT"):
    return {
        "schema": reconcile._SCHEMA,
        "fixture_session_ref": "a" * 32,
        "outcomes_session_ref": "a" * 32,
        "fixture": _fixture(snapshot),
        "outcomes": _outcomes(snapshot),
    }


def _blocked(report):
    assert report["schema"] == reconcile._SCHEMA
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
        "reconciliation",
        "counts",
    }


def _invalid(record):
    report = reconcile.inspect_declared_handoff(record)
    _blocked(report)
    assert report["reconciliation"] == {
        "observation": "NOT_QUALIFIED",
        "reason": "INVALID_TEST_HANDOFF",
    }
    assert report["snapshot"] == "NOT_QUALIFIED"
    assert report["counts"]["roles_declared"] == 0
    assert report["counts"]["cross_denied"] == 0
    return report


def test_without_input_remains_blocked_with_no_host_access():
    report = reconcile.inspect_declared_handoff()
    _blocked(report)
    assert report["reconciliation"]["reason"] == "NO_TEST_HANDOFF"
    assert report["counts"]["roles_expected"] == 6
    assert report["counts"]["roles_missing"] == 6
    assert report["counts"]["cross_expected"] == 90
    assert report["counts"]["owner_expected"] == 18


@pytest.mark.parametrize(
    "snapshot", ["SYNTHETIC_TEST_INPUT", "UNTRUSTED_LIVE_OBSERVATION"]
)
def test_matching_complete_declarations_remain_untrusted(snapshot):
    report = reconcile.inspect_declared_handoff(_record(snapshot))
    _blocked(report)
    assert report["snapshot"] == snapshot
    assert report["reconciliation"] == {
        "observation": "UNTRUSTED_RECONCILED_TEST_DECLARATIONS_ONLY",
        "reason": "NO_INDEPENDENT_LINUX_EVIDENCE",
    }
    assert report["counts"]["roles_declared"] == 6
    assert report["counts"]["cross_denied"] == 90
    assert report["counts"]["owner_success"] == 18
    assert report["counts"]["cross_allowed"] == 0


def test_declared_snapshot_mismatch_refuses_without_leaking_partial_counts():
    record = _record()
    record["outcomes"]["snapshot"] = "UNTRUSTED_LIVE_OBSERVATION"
    report = reconcile.inspect_declared_handoff(record)
    _blocked(report)
    assert report["reconciliation"]["reason"] == "SNAPSHOT_DECLARATIONS_DISAGREE"
    assert report["snapshot"] == "NOT_QUALIFIED"
    assert report["counts"]["cross_denied"] == 0
    assert report["counts"]["roles_declared"] == 0


def test_session_reference_mismatch_refuses_even_matching_complete_reports():
    record = _record()
    record["outcomes_session_ref"] = "b" * 32
    result = reconcile.inspect_declared_handoff(record)
    _blocked(result)
    assert result["reconciliation"]["reason"] == "SESSION_DECLARATIONS_DISAGREE"
    assert result["counts"]["roles_declared"] == 0
    assert result["counts"]["cross_denied"] == 0


@pytest.mark.parametrize("field", ["fixture_session_ref", "outcomes_session_ref"])
@pytest.mark.parametrize("bad", [None, True, 42, "", "0" * 32, "PRIVATE_SECRET", "A" * 32])
def test_invalid_or_unbounded_session_labels_are_refused(field, bad):
    record = _record()
    record[field] = bad
    result = _invalid(record)
    assert "PRIVATE_SECRET" not in json.dumps(result)


@pytest.mark.parametrize(
    "mutation",
    [
        lambda r: r["outcomes"]["cases"][0].update(outcome="SUCCESS"),
        lambda r: r["outcomes"]["cases"][44].update(outcome="SUCCESS"),
        lambda r: r["outcomes"]["cases"][89].update(outcome="SUCCESS"),
    ],
)
def test_cross_role_success_has_highest_refusal_priority(mutation):
    record = _record()
    record["fixture"]["roles"][0]["directory_mode"] = 0o755
    mutation(record)
    report = reconcile.inspect_declared_handoff(record)
    _blocked(report)
    assert report["reconciliation"]["reason"] == "DECLARED_CROSS_ROLE_ACCESS"
    assert report["counts"]["cross_allowed"] == 1
    assert report["counts"]["fixture_unsafe"] == 1


@pytest.mark.parametrize(
    "mutation",
    [
        lambda r: r["fixture"]["roles"][0].update(file_mode=0o644),
        lambda r: r["fixture"]["roles"][0].update(acl_absent_claim=False),
        lambda r: r["fixture"]["roles"].pop(),
        lambda r: r["fixture"]["roles"][0]["effective_gids"].append(67001),
    ],
)
def test_fixture_gap_blocks_even_with_108_complete_claims(mutation):
    record = _record()
    mutation(record)
    report = reconcile.inspect_declared_handoff(record)
    _blocked(report)
    assert report["reconciliation"]["reason"] == "TEST_FIXTURE_NOT_COMPLETE"
    assert report["counts"]["cross_denied"] == 90
    assert report["counts"]["owner_success"] == 18


@pytest.mark.parametrize("bad_result", ["ENOENT", "OTHER_FAILURE", "NOT_ATTEMPTED"])
def test_missing_target_and_unattempted_cross_cases_refuse(bad_result):
    record = _record()
    record["outcomes"]["cases"][3]["outcome"] = bad_result
    report = reconcile.inspect_declared_handoff(record)
    _blocked(report)
    assert report["reconciliation"]["reason"] == "SYSCALL_OUTCOMES_NOT_COMPLETE"
    assert report["counts"]["cross_inconclusive"] == 1
    assert report["counts"]["cross_denied"] == 89


@pytest.mark.parametrize("bad_result", ["ENOENT", "DENIED_EACCES", "NOT_ATTEMPTED"])
def test_failed_positive_owner_control_refuses(bad_result):
    record = _record()
    record["outcomes"]["cases"][90]["outcome"] = bad_result
    report = reconcile.inspect_declared_handoff(record)
    _blocked(report)
    assert report["reconciliation"]["reason"] == "SYSCALL_OUTCOMES_NOT_COMPLETE"
    assert report["counts"]["owner_success"] == 17
    assert report["counts"]["owner_failed"] == 1


def test_missing_owner_control_refuses_even_when_every_negative_is_denied():
    record = _record()
    record["outcomes"]["cases"] = record["outcomes"]["cases"][:90]
    report = reconcile.inspect_declared_handoff(record)
    _blocked(report)
    assert report["reconciliation"]["reason"] == "SYSCALL_OUTCOMES_NOT_COMPLETE"
    assert report["counts"]["owner_missing"] == 18


@pytest.mark.parametrize(
    "invalid_value",
    [0, True, (), [], "PRIVATE_SECRET", {"secret": "PRIVATE_SECRET"}],
)
def test_invalid_outer_container_refuses_without_echo(invalid_value):
    report = _invalid(invalid_value)
    assert "PRIVATE_SECRET" not in json.dumps(report)


@pytest.mark.parametrize(
    "mutate",
    [
        lambda r: r.update(schema="WRONG"),
        lambda r: r.update(schema=None),
        lambda r: r.update(private="PRIVATE_SECRET"),
        lambda r: r.pop("fixture"),
        lambda r: r.pop("outcomes"),
        lambda r: r.pop("fixture_session_ref"),
        lambda r: r.pop("outcomes_session_ref"),
        lambda r: r.update(fixture=None),
        lambda r: r.update(outcomes=[]),
        lambda r: r.update(outcomes="PRIVATE_SECRET"),
    ],
)
def test_wrong_outer_envelope_is_rejected(mutate):
    record = _record()
    mutate(record)
    _invalid(record)


@pytest.mark.parametrize(
    "mutate",
    [
        lambda r: r["fixture"].update(schema="WRONG"),
        lambda r: r["fixture"]["roles"][1].update(role="FAKE_ROLE"),
        lambda r: r["fixture"]["roles"][0].update(host_path="/PRIVATE_SECRET"),
        lambda r: r["outcomes"].update(schema="WRONG"),
        lambda r: r["outcomes"]["cases"][12].update(outcome="PRIVATE_SECRET"),
        lambda r: r["outcomes"]["cases"][12].update(extra="PRIVATE_SECRET"),
        lambda r: r["outcomes"]["cases"].append(dict(r["outcomes"]["cases"][0])),
    ],
)
def test_invalid_nested_envelopes_are_rejected_without_partial_reports(mutate):
    record = _record()
    mutate(record)
    report = _invalid(record)
    text = json.dumps(report)
    assert "PRIVATE_SECRET" not in text
    assert "/PRIVATE_SECRET" not in text


def test_no_host_private_data_or_raw_names_are_emitted():
    record = _record()
    record["fixture"]["roles"][0]["uid"] = 123456789
    record["fixture"]["roles"][0]["directory_owner_uid"] = 123456789
    record["fixture"]["roles"][0]["file_owner_uid"] = 123456789
    result = reconcile.inspect_declared_handoff(record)
    _blocked(result)
    output = json.dumps(result)
    assert "123456789" not in output
    assert "root_source" not in output
    assert "history_source" not in output
    assert "ch_root_custody_source" not in output


def test_output_does_not_alias_mutable_caller_inputs():
    record = _record()
    report = reconcile.inspect_declared_handoff(record)
    original = copy.deepcopy(report)
    record["outcomes"]["cases"][0]["outcome"] = "SUCCESS"
    record["fixture"]["roles"][0]["uid"] = 0
    assert report == original


def test_no_input_cli_prints_fixed_blockers_and_exit_two(capsys):
    assert reconcile.main() == 2
    report = json.loads(capsys.readouterr().out)
    _blocked(report)
    assert report["reconciliation"]["reason"] == "NO_TEST_HANDOFF"
