"""S9-RPI-36: closed TEST fixture layout declarations cannot grant Linux trust."""

from __future__ import annotations

import copy
import json

import pytest

from deployment import stage9_native_linux_test_fixture_readiness as fixture


def _complete(snapshot="SYNTHETIC_TEST_INPUT"):
    roles = []
    for index, role in enumerate(fixture.expected_roles()):
        uid = 66000 + index
        gid = 67000 + index
        roles.append(
            {
                "role": role,
                "uid": uid,
                "gid": gid,
                "effective_gids": [gid],
                "directory_owner_uid": uid,
                "directory_owner_gid": gid,
                "directory_mode": 0o700,
                "file_owner_uid": uid,
                "file_owner_gid": gid,
                "file_mode": 0o600,
                "symlink_free_claim": True,
                "acl_absent_claim": True,
                "unprivileged_claim": True,
                "private_parent_chain_claim": True,
            }
        )
    return {
        "schema": fixture._SCHEMA,
        "snapshot": snapshot,
        "roles": roles,
    }


def _assert_blocked(result):
    assert set(result) == {
        "schema",
        "status",
        "qualification",
        "production_emission",
        "topology",
        "snapshot",
        "blockers",
        "fixture",
        "counts",
    }
    assert result["schema"] == fixture._SCHEMA
    assert result["status"] == "BLOCKED"
    assert result["qualification"] == "NOT_QUALIFIED"
    assert result["production_emission"] == "BLOCKED"
    assert result["topology"] == "PROPOSED_NOT_APPROVED"
    assert result["blockers"]
    assert result["counts"]["planned_cross_role_cases"] == 90
    assert result["counts"]["planned_owner_controls"] == 18


def _invalid(record):
    result = fixture.inspect_declared_fixture(record)
    _assert_blocked(result)
    assert result["fixture"] == {
        "observation": "NOT_QUALIFIED",
        "reason": "INVALID_TEST_FIXTURE",
    }
    assert result["counts"]["declared_roles"] == 0
    return result


def test_role_inventory_matches_existing_six_role_contract():
    assert fixture.expected_roles() == (
        "root_source",
        "history_source",
        "cca",
        "root_signer",
        "history_signer",
        "issuer",
    )
    assert len(set(fixture.expected_roles())) == 6


def test_no_input_returns_fixed_blocked_plan_and_no_host_access():
    result = fixture.inspect_declared_fixture()
    _assert_blocked(result)
    assert result["snapshot"] == "NOT_QUALIFIED"
    assert result["fixture"]["reason"] == "NO_TEST_FIXTURE"
    assert result["counts"]["expected_roles"] == 6
    assert result["counts"]["declared_roles"] == 0
    assert result["counts"]["missing_roles"] == 6


@pytest.mark.parametrize(
    "snapshot", ["SYNTHETIC_TEST_INPUT", "UNTRUSTED_LIVE_OBSERVATION"]
)
def test_complete_six_role_layout_is_only_untrusted_not_production_ready(snapshot):
    result = fixture.inspect_declared_fixture(_complete(snapshot))
    _assert_blocked(result)
    assert result["snapshot"] == snapshot
    assert result["fixture"] == {
        "observation": "UNTRUSTED_TEST_LAYOUT_DECLARATIONS_ONLY",
        "reason": "NO_INDEPENDENT_PROCESS_OR_FILESYSTEM_EVIDENCE",
    }
    assert result["counts"]["declared_roles"] == 6
    assert result["counts"]["missing_roles"] == 0
    assert result["counts"]["unsafe_roles"] == 0
    assert result["counts"]["identity_collisions"] == 0


@pytest.mark.parametrize("index", [0, 2, 5])
def test_missing_role_is_incomplete(index):
    record = _complete()
    del record["roles"][index]
    result = fixture.inspect_declared_fixture(record)
    _assert_blocked(result)
    assert result["fixture"]["reason"] == "INCOMPLETE_TEST_FIXTURE"
    assert result["counts"]["missing_roles"] == 1


def test_empty_declaration_is_incomplete_but_valid_shape():
    record = _complete()
    record["roles"] = []
    result = fixture.inspect_declared_fixture(record)
    _assert_blocked(result)
    assert result["fixture"]["reason"] == "INCOMPLETE_TEST_FIXTURE"
    assert result["counts"]["missing_roles"] == 6


@pytest.mark.parametrize("field", ["uid", "gid"])
def test_cross_role_numeric_identity_collision_refuses(field):
    record = _complete()
    record["roles"][1][field] = record["roles"][0][field]
    if field == "gid":
        record["roles"][1]["effective_gids"] = [record["roles"][1]["gid"]]
        record["roles"][1]["directory_owner_gid"] = record["roles"][1]["gid"]
        record["roles"][1]["file_owner_gid"] = record["roles"][1]["gid"]
    else:
        record["roles"][1]["directory_owner_uid"] = record["roles"][1]["uid"]
        record["roles"][1]["file_owner_uid"] = record["roles"][1]["uid"]
    result = fixture.inspect_declared_fixture(record)
    _assert_blocked(result)
    assert result["fixture"]["reason"] == "DECLARED_ROLE_IDENTITY_COLLISION"
    assert result["counts"]["identity_collisions"] == 1


@pytest.mark.parametrize(
    "field,value",
    [
        ("uid", 0),
        ("uid", -1),
        ("uid", 2**32 - 1),
        ("gid", 0),
        ("directory_owner_uid", 42),
        ("directory_owner_gid", 42),
        ("file_owner_uid", 42),
        ("file_owner_gid", 42),
        ("directory_mode", 0o755),
        ("directory_mode", 0o777),
        ("file_mode", 0o644),
        ("file_mode", 0o666),
        ("symlink_free_claim", False),
        ("acl_absent_claim", False),
        ("unprivileged_claim", False),
        ("private_parent_chain_claim", False),
    ],
)
def test_unsafe_owner_permissions_privilege_or_path_claim_refuses(field, value):
    record = _complete()
    record["roles"][0][field] = value
    result = fixture.inspect_declared_fixture(record)
    _assert_blocked(result)
    assert result["fixture"]["reason"] == "DECLARED_TEST_RESOURCE_OR_PRIVILEGE_GAP"
    assert result["counts"]["unsafe_roles"] == 1


def test_extra_supplementary_gid_is_a_boundary_gap_not_clean_fixture():
    record = _complete()
    record["roles"][1]["effective_gids"].append(record["roles"][0]["gid"])
    result = fixture.inspect_declared_fixture(record)
    _assert_blocked(result)
    assert result["fixture"]["reason"] == "DECLARED_TEST_RESOURCE_OR_PRIVILEGE_GAP"
    assert result["counts"]["unsafe_roles"] == 1
    assert result["counts"]["identity_collisions"] == 1


def test_boundary_gap_priority_over_collision_and_missing_role():
    record = _complete()
    record["roles"][0]["acl_absent_claim"] = False
    record["roles"][1]["uid"] = record["roles"][0]["uid"]
    del record["roles"][5]
    result = fixture.inspect_declared_fixture(record)
    _assert_blocked(result)
    assert result["fixture"]["reason"] == "DECLARED_TEST_RESOURCE_OR_PRIVILEGE_GAP"
    assert result["counts"]["identity_collisions"] == 1
    assert result["counts"]["missing_roles"] == 1


@pytest.mark.parametrize(
    "bad",
    [12, True, [], "PRIVATE_SECRET", {"private": "PRIVATE_SECRET"}, ()],
)
def test_refuse_non_mapping_records(bad):
    result = _invalid(bad)
    assert "PRIVATE_SECRET" not in json.dumps(result)


@pytest.mark.parametrize(
    "mutate",
    [
        lambda r: r.update(schema="WRONG"),
        lambda r: r.update(schema=None),
        lambda r: r.update(snapshot="QUALIFIED"),
        lambda r: r.update(snapshot=None),
        lambda r: r.update(snapshot=True),
        lambda r: r.update(private="PRIVATE_SECRET"),
        lambda r: r.pop("schema"),
        lambda r: r.pop("snapshot"),
        lambda r: r.pop("roles"),
        lambda r: r.update(roles=()),
        lambda r: r.update(roles="PRIVATE_SECRET"),
        lambda r: r.update(roles=None),
        lambda r: r["roles"].append(copy.deepcopy(r["roles"][0])),
    ],
)
def test_refuse_invalid_envelope_duplicate_and_oversized_rows(mutate):
    record = _complete()
    mutate(record)
    _invalid(record)


@pytest.mark.parametrize(
    "field",
    [
        "role",
        "uid",
        "gid",
        "effective_gids",
        "directory_owner_uid",
        "directory_owner_gid",
        "directory_mode",
        "file_owner_uid",
        "file_owner_gid",
        "file_mode",
        "symlink_free_claim",
        "acl_absent_claim",
        "unprivileged_claim",
        "private_parent_chain_claim",
    ],
)
def test_missing_row_field_is_rejected_without_partial_counts(field):
    record = _complete()
    del record["roles"][0][field]
    _invalid(record)


@pytest.mark.parametrize(
    "field,value",
    [
        ("role", "UNKNOWN_ROLE"),
        ("role", 123),
        ("role", None),
        ("uid", True),
        ("uid", "PRIVATE_SECRET"),
        ("gid", 42.5),
        ("directory_owner_uid", None),
        ("directory_mode", True),
        ("file_mode", "0600"),
        ("acl_absent_claim", 1),
        ("unprivileged_claim", "true"),
        ("effective_gids", None),
        ("effective_gids", (67000,)),
        ("effective_gids", []),
        ("effective_gids", [True]),
        ("effective_gids", [67000, 67000]),
        ("effective_gids", [67000, "PRIVATE_SECRET"]),
        ("effective_gids", list(range(17))),
    ],
)
def test_malformed_role_fields_and_group_list_are_rejected(field, value):
    record = _complete()
    record["roles"][0][field] = value
    _invalid(record)


@pytest.mark.parametrize("bad", [None, 0, True, [], (), "PRIVATE_SECRET"])
def test_non_dict_role_entry_is_rejected(bad):
    record = _complete()
    record["roles"][0] = bad
    _invalid(record)


def test_duplicate_role_label_is_rejected_even_with_different_numeric_identity():
    record = _complete()
    record["roles"][1]["role"] = record["roles"][0]["role"]
    _invalid(record)


def test_unknown_field_with_private_data_refuses_without_echoing_values():
    record = _complete()
    record["roles"][0]["host_home"] = "/home/PRIVATE_SECRET"
    result = _invalid(record)
    payload = json.dumps(result)
    assert "PRIVATE_SECRET" not in payload
    assert "/home/" not in payload
    assert "root_source" not in payload


def test_reordered_valid_roles_do_not_change_results():
    record = _complete()
    record["roles"].reverse()
    result = fixture.inspect_declared_fixture(record)
    _assert_blocked(result)
    assert result["fixture"]["observation"] == "UNTRUSTED_TEST_LAYOUT_DECLARATIONS_ONLY"


def test_return_value_does_not_share_references_with_caller():
    record = _complete()
    result = fixture.inspect_declared_fixture(record)
    original = copy.deepcopy(result)
    record["roles"][0]["uid"] = 0
    record["roles"][0]["effective_gids"].append(42)
    record["snapshot"] = "READY"
    assert result == original


def test_default_command_emits_only_blocked_labels_and_exits_two(capsys):
    assert fixture.main() == 2
    result = json.loads(capsys.readouterr().out)
    _assert_blocked(result)
    assert result["fixture"]["reason"] == "NO_TEST_FIXTURE"
