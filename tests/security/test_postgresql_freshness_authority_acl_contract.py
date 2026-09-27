"""Versioned, exact ACL contract regressions for FreshnessAuthority relations."""

import pytest

from bot_core.postgresql_freshness_authority import (
    PostgreSQLFreshnessAuthorityProvisioning,
    _expected_relation_acl,
    _relation_acl_difference,
)


CONFIG = PostgreSQLFreshnessAuthorityProvisioning()
TABLES = {
    "metadata",
    "key_material_role_bindings",
    "credentials",
    "key_lifecycle_history",
    "authority_lineages",
    "authority_generation_heads",
    "prepared_verifications",
    "authoritative_documents",
    "decisions",
    "finalization_receipts",
}


def _maintain_grantees(version: int) -> set[tuple[str, str]]:
    return {
        (relation, grantee)
        for relation, grantee, privilege, grantable in _expected_relation_acl(CONFIG, version)
        if privilege == "MAINTAIN" and not grantable
    }


def test_pg16_and_pg17_relation_acl_differ_only_by_owner_maintain():
    pg16 = _expected_relation_acl(CONFIG, 160000)
    pg17 = _expected_relation_acl(CONFIG, 170000)

    assert not _maintain_grantees(160000)
    assert pg17 - pg16 == {(table, CONFIG.schema_owner_role, "MAINTAIN", False) for table in TABLES}
    assert not pg16 - pg17


def test_pg17_maintain_is_never_granted_to_non_owner_roles_or_sequence():
    assert _maintain_grantees(170000) == {(table, CONFIG.schema_owner_role) for table in TABLES}
    assert all(
        privilege != "MAINTAIN"
        for relation, grantee, privilege, grantable in _expected_relation_acl(CONFIG, 170000)
        if grantee
        in {
            CONFIG.function_owner_role,
            CONFIG.reader_role,
            CONFIG.runtime_role,
            CONFIG.verifier_role,
            CONFIG.admin_role,
        }
    )


@pytest.mark.parametrize(
    ("version", "mutation", "label"),
    [
        (
            170000,
            lambda acl: acl - {("metadata", CONFIG.schema_owner_role, "MAINTAIN", False)},
            "missing ACL tuples=[('metadata', 'freshness_schema_owner', 'MAINTAIN', False)]",
        ),
        (
            160000,
            lambda acl: acl | {("metadata", CONFIG.schema_owner_role, "MAINTAIN", False)},
            "unexpected ACL tuples=[('metadata', 'freshness_schema_owner', 'MAINTAIN', False)]",
        ),
        (
            170000,
            lambda acl: acl | {("metadata", CONFIG.runtime_role, "MAINTAIN", False)},
            "unexpected ACL tuples=[('metadata', 'freshness_runtime', 'MAINTAIN', False)]",
        ),
    ],
)
def test_missing_or_unexpected_privilege_remains_an_exact_acl_failure(version, mutation, label):
    expected = _expected_relation_acl(CONFIG, version)
    actual = mutation(expected)

    assert actual != expected
    assert label in _relation_acl_difference(actual, expected)


def test_acl_difference_diagnostic_is_bounded():
    expected = {
        (f"table_{number:02}", CONFIG.schema_owner_role, "SELECT", False) for number in range(25)
    }
    diagnostic = _relation_acl_difference(set(), expected)

    assert "... (5 more)" in diagnostic
    assert "table_19" in diagnostic
    assert "table_20" not in diagnostic
    assert "unexpected ACL tuples=[]" in diagnostic
