"""Closed synthetic installation expectations never confer production authority."""

from __future__ import annotations

from dataclasses import FrozenInstanceError
import hashlib
import json

import pytest

from deployment import stage9_trusted_installation_contract as contract


def _manifest() -> dict:
    roles = []
    for index, role in enumerate(("root_source", "history_source", "cca"), 1):
        unit = f"synthetic-{role.replace('_', '-')}.service"
        roles.append(
            {
                "role": role,
                "installation_identity": f"synthetic-{role}",
                "unit": unit,
                "principal": f"synthetic_{role}",
                "uid": 2000 + index,
                "gid": 3000 + index,
                "supplementary_groups": [],
                "unit_file": {
                    "path": f"/etc/systemd/system/{unit}",
                    "sha256": "ab" * 32,
                    "uid": 0,
                    "gid": 0,
                    "mode": 0o644,
                },
                "executable": {
                    "path": f"/opt/synthetic/{role}",
                    "sha256": "cd" * 32,
                    "uid": 0,
                    "gid": 0,
                    "mode": 0o755,
                },
                "control_group": f"/system.slice/{unit}",
            }
        )
    return {
        "schema": contract.SCHEMA,
        "version": contract.VERSION,
        "installation_id": "synthetic-test-installation",
        "roles": roles,
    }


def _raw(value: dict) -> bytes:
    return json.dumps(value, allow_nan=False).encode("utf-8")


def test_exact_synthetic_manifest_has_three_immutable_distinct_roles():
    raw = _raw(_manifest())
    expected = contract.parse_manifest(raw)
    assert expected.installation_id == "synthetic-test-installation"
    assert {item.role for item in expected.roles} == contract.REQUIRED_ROLES
    assert len({item.uid for item in expected.roles}) == 3
    assert len({item.gid for item in expected.roles}) == 3
    assert all(item.supplementary_groups == () for item in expected.roles)
    assert expected.sha256 == hashlib.sha256(raw).hexdigest()
    assert contract.manifest_sha256(raw) == expected.sha256
    with pytest.raises(FrozenInstanceError):
        expected.installation_id = "other"


@pytest.mark.parametrize("raw", [None, b"", "{}", bytearray(b"{}"), b"\xff", b"{", b"[]"])
def test_missing_or_malformed_manifest_is_refused(raw):
    with pytest.raises(contract.InstallationRefusal):
        contract.parse_manifest(raw)


@pytest.mark.parametrize("raw", [b"null", b"true", b"1", b"NaN", b"Infinity", b"-Infinity"])
def test_non_object_and_nonfinite_json_is_refused(raw):
    with pytest.raises(contract.InstallationRefusal):
        contract.parse_manifest(raw)


def test_manifest_size_is_bounded_before_decoding():
    with pytest.raises(contract.InstallationRefusal, match="MANIFEST_TOO_LARGE"):
        contract.parse_manifest(b" " * (contract.MAX_MANIFEST_BYTES + 1))


@pytest.mark.parametrize("field,value", [("schema", "future"), ("version", 2), ("version", True)])
def test_unknown_or_ambiguous_schema_version_is_refused(field, value):
    document = _manifest()
    document[field] = value
    with pytest.raises(contract.InstallationRefusal, match="MANIFEST_SCHEMA_UNSUPPORTED"):
        contract.parse_manifest(_raw(document))


@pytest.mark.parametrize("scope", ["manifest", "role", "unit", "executable"])
def test_contract_rejects_unknown_fields_including_self_approval(scope):
    document = _manifest()
    target = document
    if scope != "manifest":
        target = document["roles"][0]
    if scope in {"unit", "executable"}:
        target = target["unit_file" if scope == "unit" else "executable"]
    target["independently_qualified"] = True
    with pytest.raises(contract.InstallationRefusal, match="MANIFEST_FIELDS_INVALID"):
        contract.parse_manifest(_raw(document))


def test_duplicate_json_fields_are_refused_even_if_values_match():
    raw = _raw(_manifest()).replace(b'"version": 1', b'"version": 1, "version": 1')
    with pytest.raises(contract.InstallationRefusal, match="MANIFEST_IDENTIFIER_DUPLICATE"):
        contract.parse_manifest(raw)


def test_nested_duplicate_json_fields_are_refused():
    raw = _raw(_manifest()).replace(b'"uid": 0', b'"uid": 0, "uid": 0', 1)
    with pytest.raises(contract.InstallationRefusal, match="MANIFEST_IDENTIFIER_DUPLICATE"):
        contract.parse_manifest(raw)


@pytest.mark.parametrize("roles", [[], None, {}, "root_source", ["root_source"], [1, 2, 3, 4]])
def test_manifest_requires_exactly_three_role_records(roles):
    document = _manifest()
    document["roles"] = roles
    with pytest.raises(contract.InstallationRefusal):
        contract.parse_manifest(_raw(document))


def test_unknown_role_is_not_an_alias_for_an_authority():
    document = _manifest()
    document["roles"][0]["role"] = "root_signer"
    with pytest.raises(contract.InstallationRefusal, match="MANIFEST_ROLE_UNKNOWN"):
        contract.parse_manifest(_raw(document))


@pytest.mark.parametrize("field", ["role", "installation_identity", "principal", "unit"])
def test_duplicate_role_identifiers_are_refused(field):
    document = _manifest()
    first, second = document["roles"][:2]
    second[field] = first[field]
    if field == "unit":
        second["unit_file"]["path"] = first["unit_file"]["path"]
        second["control_group"] = first["control_group"]
    with pytest.raises(contract.InstallationRefusal, match="MANIFEST_IDENTIFIER_DUPLICATE"):
        contract.parse_manifest(_raw(document))


@pytest.mark.parametrize("field", ["uid", "gid"])
def test_shared_primary_identity_is_refused(field):
    document = _manifest()
    document["roles"][1][field] = document["roles"][0][field]
    with pytest.raises(contract.InstallationRefusal, match="MANIFEST_ROLE_SEPARATION_REQUIRED"):
        contract.parse_manifest(_raw(document))


@pytest.mark.parametrize("field", ["uid", "gid"])
@pytest.mark.parametrize("value", [0, -1, True, "2001", 2**32 - 1])
def test_role_numeric_identity_requires_bounded_positive_exact_integer(field, value):
    document = _manifest()
    document["roles"][0][field] = value
    with pytest.raises(contract.InstallationRefusal, match="MANIFEST_PRINCIPAL_INVALID"):
        contract.parse_manifest(_raw(document))


@pytest.mark.parametrize("groups", [[3001], [4000], [4000, 4000], None, {}, ""])
def test_supplementary_membership_is_forbidden_even_if_manifest_lists_it(groups):
    document = _manifest()
    document["roles"][0]["supplementary_groups"] = groups
    with pytest.raises(
        contract.InstallationRefusal, match="MANIFEST_SUPPLEMENTARY_GROUPS_FORBIDDEN"
    ):
        contract.parse_manifest(_raw(document))


@pytest.mark.parametrize(
    "path",
    [
        "relative/executable",
        "/opt//synthetic/root_source",
        "/opt/./synthetic/root_source",
        "/opt/synthetic/../root_source",
        "/opt/synthetic/root_source/",
        "/opt/synthetic/root source",
        "/opt/synthetic/root_source\x00",
        "/opt/synthetic/ümlaut",
        "/opt/synthetic/root*source",
        "/opt/synthetic/" + "a" * 256,
    ],
)
def test_executable_path_aliases_and_unsafe_spellings_are_refused(path):
    document = _manifest()
    document["roles"][0]["executable"]["path"] = path
    with pytest.raises(contract.InstallationRefusal, match="MANIFEST_PATH_INVALID"):
        contract.parse_manifest(_raw(document))


@pytest.mark.parametrize(
    "field,value", [("uid", 1000), ("gid", 1000), ("mode", 0o777), ("mode", 0o4755)]
)
def test_unprotected_file_ownership_and_permissions_are_refused(field, value):
    document = _manifest()
    document["roles"][0]["executable"][field] = value
    with pytest.raises(contract.InstallationRefusal, match="MANIFEST_FILE_POLICY_INVALID"):
        contract.parse_manifest(_raw(document))


@pytest.mark.parametrize("digest", ["AA" * 32, "a" * 63, "g" * 64, "", None])
def test_sha256_requires_exact_lowercase_hex(digest):
    document = _manifest()
    document["roles"][0]["executable"]["sha256"] = digest
    with pytest.raises(contract.InstallationRefusal, match="MANIFEST_FILE_POLICY_INVALID"):
        contract.parse_manifest(_raw(document))


def test_unit_file_cannot_be_executable_or_mapped_to_another_unit():
    for field, value in (("mode", 0o755), ("path", "/etc/systemd/system/other.service")):
        document = _manifest()
        document["roles"][0]["unit_file"][field] = value
        with pytest.raises(contract.InstallationRefusal):
            contract.parse_manifest(_raw(document))


def test_executable_requires_an_executable_mode():
    document = _manifest()
    document["roles"][0]["executable"]["mode"] = 0o644
    with pytest.raises(contract.InstallationRefusal, match="MANIFEST_FILE_POLICY_INVALID"):
        contract.parse_manifest(_raw(document))


def test_no_two_roles_can_name_the_same_executable_object_path():
    document = _manifest()
    document["roles"][1]["executable"]["path"] = document["roles"][0]["executable"]["path"]
    with pytest.raises(contract.InstallationRefusal, match="MANIFEST_IDENTIFIER_DUPLICATE"):
        contract.parse_manifest(_raw(document))


@pytest.mark.parametrize(
    "unit", ["alias@.service", "OTHER.service", "other.socket", "../other.service"]
)
def test_units_have_one_unambiguous_service_name(unit):
    document = _manifest()
    document["roles"][0]["unit"] = unit
    with pytest.raises(contract.InstallationRefusal, match="MANIFEST_FIELDS_INVALID"):
        contract.parse_manifest(_raw(document))


def test_cgroup_requires_the_expected_unit_and_no_cross_role_overlap():
    document = _manifest()
    document["roles"][0]["control_group"] = "/system.slice/unrelated.service"
    with pytest.raises(contract.InstallationRefusal, match="MANIFEST_CGROUP_INVALID"):
        contract.parse_manifest(_raw(document))
    document = _manifest()
    first, second = document["roles"][:2]
    second["control_group"] = first["control_group"] + "/" + second["unit"]
    with pytest.raises(contract.InstallationRefusal, match="MANIFEST_CGROUP_INVALID"):
        contract.parse_manifest(_raw(document))


def test_exception_messages_cannot_disclose_arbitrary_backend_data():
    refusal = contract.InstallationRefusal("PRIVATE_CUSTODY_PATH")
    assert refusal.code == "BACKEND_OBSERVATION_FAILED"
    assert str(refusal) == "BACKEND_OBSERVATION_FAILED"


@pytest.mark.parametrize(
    "identifier", ["", None, "../installation", "two installations", "a" * 129]
)
def test_installation_identity_cannot_be_empty_or_ambiguous(identifier):
    document = _manifest()
    document["installation_id"] = identifier
    with pytest.raises(contract.InstallationRefusal, match="MANIFEST_FIELDS_INVALID"):
        contract.parse_manifest(_raw(document))


@pytest.mark.parametrize("principal", ["root", "Root", "service@host", "", None])
def test_superuser_or_noncanonical_account_name_is_refused(principal):
    document = _manifest()
    document["roles"][0]["principal"] = principal
    with pytest.raises(contract.InstallationRefusal, match="MANIFEST_PRINCIPAL_INVALID"):
        contract.parse_manifest(_raw(document))
