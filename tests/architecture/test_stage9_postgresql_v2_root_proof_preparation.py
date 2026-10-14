"""Preparation V2 preserves the non-authorizing Stage 9 architecture boundary."""

from __future__ import annotations

import ast
import re
from dataclasses import fields
from itertools import product
from pathlib import Path

import pytest

from bot_core.licensing.cha_root_proof_attempt_reservation import (
    RootProofAttemptReservationError,
    resolve_root_proof_issuance_authorization_v2,
)
from bot_core.postgresql_atomic_root_proof_commit import (
    AtomicRootProofCommitConflict,
    prepare_atomic_root_proof_commit,
)
from bot_core.postgresql_atomic_root_proof_schema import PostgreSQLAtomicRootProofProvisioning
from bot_core.postgresql_authenticated_issuer_history import PostgreSQLAuthenticatedIssuerHistory
from bot_core.postgresql_root_proof_issuance_authority import (
    PostgreSQLRootProofIssuanceAuthority,
    ProductionLocalIssuanceAuthorityError,
)
from bot_core.postgresql_root_proof_issuance_authority_v2 import (
    PostgreSQLEntitlementRegistryProviderV2,
    PostgreSQLRootProofIssuanceAuthorityV2,
)
from bot_core.postgresql_root_proof_preparation_qualification import (
    PostgreSQLRootProofPreparationV2Deployment,
    _sources,
)
from bot_core.postgresql_root_proof_preparation_v2 import (
    PostgreSQLAuthenticatedIssuerHistoryV2,
    PostgreSQLRootProofRetentionV2,
    PostgreSQLRootProofSigningIntentV2,
)
from bot_core.postgresql_root_proof_retention import (
    PostgreSQLRootProofRetention,
    RootProofRetentionNotIssued,
)
from bot_core.postgresql_root_proof_retention_schema import _INDEXES as RETENTION_INDEXES
from bot_core.postgresql_root_proof_signing_intent import (
    PostgreSQLRootProofSigningIntent,
    RootProofSigningIntentNotIssued,
)
from bot_core.postgresql_root_proof_signing_intent_schema import _INDEXES as INTENT_INDEXES

ROOT = Path(__file__).resolve().parents[2]
PREPARATION = ROOT / "bot_core/postgresql_root_proof_preparation_v2.py"
QUALIFICATION = ROOT / "bot_core/postgresql_root_proof_preparation_qualification.py"
AUTHORIZATION = ROOT / "bot_core/licensing/cha_root_proof_attempt_reservation.py"
FINAL_OPERATIONS = frozenset(
    {
        "approve_commit",
        "commit_root_proof",
        "append_bind",
        "atomic_bind",
        "append_record",
        "atomic_append_record",
        "retain_head",
        "atomic_retain_head",
        "advance_checkpoint",
        "compare_and_swap_bind",
        "sign_root_proof",
        "sign_history_head",
    }
)


def _assert_preparation_has_no_final_operation(source: str) -> None:
    """Check executable Python/SQL, including nested methods and literal queries."""
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            name = (
                node.func.attr
                if isinstance(node.func, ast.Attribute)
                else node.func.id
                if isinstance(node.func, ast.Name)
                else ""
            )
            assert name not in FINAL_OPERATIONS, f"preparation invokes {name}"
        elif isinstance(node, ast.Constant) and isinstance(node.value, str):
            for name in FINAL_OPERATIONS:
                assert re.search(r"\." + name + r"\s*\(", node.value) is None, (
                    f"preparation executes final SQL operation {name}"
                )


def test_v2_preparation_invokes_neither_signing_nor_final_authority():
    _assert_preparation_has_no_final_operation(PREPARATION.read_text(encoding="utf-8"))


@pytest.mark.parametrize("operation", sorted(FINAL_OPERATIONS))
def test_preparation_boundary_rejects_an_added_final_operation(operation):
    with pytest.raises(AssertionError, match="preparation invokes"):
        _assert_preparation_has_no_final_operation(f"provider.{operation}()")
    with pytest.raises(AssertionError, match="preparation executes final SQL"):
        _assert_preparation_has_no_final_operation(
            f"connection.execute('SELECT authority.{operation}(%s)')"
        )


@pytest.mark.parametrize(
    "implementation,error",
    (
        (PostgreSQLRootProofSigningIntentV2, RootProofSigningIntentNotIssued),
        (PostgreSQLRootProofRetentionV2, RootProofRetentionNotIssued),
    ),
)
def test_preparation_cannot_report_issued_even_without_an_available_database(implementation, error):
    # The refusal must precede any connection, credential or detached DTO read.
    port = object.__new__(implementation)
    with pytest.raises(error):
        port.require_issued()


def test_public_entitlement_preparation_port_has_no_bind_capability():
    port = object.__new__(PostgreSQLEntitlementRegistryProviderV2)
    assert port.capabilities.compare_and_swap is False
    with pytest.raises(ProductionLocalIssuanceAuthorityError, match="NO_BIND_AUTHORITY"):
        port.compare_and_swap_bind(None)


def test_v2_history_port_exposes_reads_without_history_mutation_or_signing():
    methods = {
        name
        for name, value in vars(PostgreSQLAuthenticatedIssuerHistoryV2).items()
        if callable(value) and not name.startswith("_")
    }
    assert {"retained_history", "current_record"} <= methods
    assert not methods & FINAL_OPERATIONS


def test_public_preparation_pin_contains_no_writer_or_bootstrap_credentials():
    names = {field.name for field in fields(PostgreSQLRootProofPreparationV2Deployment)}
    assert not names & {"connection", "writer_connection", "admin_connection", "bootstrap", "dsn"}


def test_production_selector_still_constructs_only_original_public_authority():
    source = ROOT / "bot_core/postgresql_root_proof_issuance_authority.py"
    tree = ast.parse(source.read_text(encoding="utf-8"))
    configured = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "_compose_runtime_authority"
    )
    constructors = {
        node.func.id
        for node in ast.walk(configured)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    }
    assert "PostgreSQLRootProofIssuanceAuthority" in constructors
    assert "PostgreSQLRootProofIssuanceAuthorityV2" not in constructors


@pytest.mark.parametrize(
    "relative_path,qualifier",
    (
        ("bot_core/postgresql_root_proof_signing_intent.py", "qualify_signing_intent_schema"),
        ("bot_core/postgresql_root_proof_retention.py", "qualify_retention_schema"),
    ),
)
def test_legacy_preparation_still_invokes_its_exact_legacy_qualifier(relative_path, qualifier):
    tree = ast.parse((ROOT / relative_path).read_text(encoding="utf-8"))
    methods = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == "_qualify"
    ]
    assert len(methods) == 1
    assert any(
        isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == qualifier
        for node in ast.walk(methods[0])
    )


def test_legacy_unique_constraints_still_reject_two_attempts_of_one_entitlement():
    intent_unique = {
        keys for table, _, keys, primary in INTENT_INDEXES if table == "intents" and not primary
    }
    retention_unique = {
        keys for table, _, keys, primary in RETENTION_INDEXES if table == "prepared" and not primary
    }
    assert intent_unique == {
        ("attempt_key",),
        ("root_proof_key",),
        ("operation_key",),
        ("entitlement_key",),
        ("account_key",),
    }
    assert retention_unique == {
        ("attempt_key",),
        ("root_proof_key",),
        ("scope_bytes", "operation_key"),
        ("scope_bytes", "entitlement_key"),
        ("scope_bytes", "account_key"),
    }


def test_preparation_sql_facade_only_reads_and_locks_original_enrolled_signer():
    config = PostgreSQLAtomicRootProofProvisioning(
        "atomic",
        "atomic_owner",
        "atomic_writer",
        "atomic_admin",
        "entitlement",
        "intent",
        "retention",
        "history",
        "trust",
        "isolated_instance",
        "requester",
        "claimant",
    )
    roles = tuple(
        (name, name + "_runtime")
        for name in (
            "entitlement",
            "intent",
            "retention",
            "history",
            "requester",
            "claimant",
        )
    )
    sources = _sources(config, "preparation", roles)
    assert set(sources) == {"qualify_preparation_v2()", "lock_active_signer_v2(bytea)"}
    for _, _, source in sources.values():
        assert (
            re.search(r"\b(INSERT|UPDATE|DELETE|TRUNCATE|ALTER|CREATE|GRANT|REVOKE)\b", source)
            is None
        )
    fence_source = sources["lock_active_signer_v2(bytea)"][2]
    assert "FOR SHARE" in fence_source
    assert "lifecycle_credentials" in fence_source
    assert "original ACTIVE signer fence required" in fence_source


def _function_source(path: Path, name: str) -> str:
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source)
    function = next(
        node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == name
    )
    result = ast.get_source_segment(source, function)
    assert result is not None
    return result


def _assert_no_sql_mutation(source: str) -> None:
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            assert (
                re.match(
                    r"\s*(INSERT|UPDATE|DELETE|TRUNCATE|ALTER|CREATE|GRANT|REVOKE)\b", node.value
                )
                is None
            ), "reconstruction contains SQL mutation"


def test_preparation_pin_reconstruction_is_explicit_read_only_without_catalog_repair():
    source = _function_source(QUALIFICATION, "requalify_root_proof_preparation_v2")
    _assert_no_sql_mutation(source)
    _assert_preparation_has_no_final_operation(source)
    assert "SERIALIZABLE READ ONLY" in source


@pytest.mark.parametrize(
    "operation", ("INSERT", "UPDATE", "DELETE", "TRUNCATE", "ALTER", "CREATE", "GRANT", "REVOKE")
)
def test_reconstruction_contract_rejects_an_added_sql_mutation(operation):
    with pytest.raises(AssertionError, match="SQL mutation"):
        _assert_no_sql_mutation(f"connection.execute('{operation} catalog')")


def test_explicit_v2_authorization_does_not_install_or_consult_production_selector():
    source = _function_source(AUTHORIZATION, "resolve_root_proof_issuance_authorization_v2")
    calls = {
        node.func.id
        for node in ast.walk(ast.parse(source))
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    }
    assert "_issuance_authority_provider" not in calls
    assert "_configured_root_proof_issuance_authority" not in calls
    _assert_preparation_has_no_final_operation(source)
    with pytest.raises(RootProofAttemptReservationError, match="EXACT_V2_PREPARATION_AUTHORITY"):
        resolve_root_proof_issuance_authorization_v2(None, None)


@pytest.mark.parametrize(
    "path",
    (
        PREPARATION,
        QUALIFICATION,
        ROOT / "bot_core/postgresql_root_proof_issuance_authority_v2.py",
    ),
)
def test_v2_ports_have_no_qualification_or_environment_bypass(path):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    forbidden = {"trust_v2", "skip_qualification", "bypass_qualification", "test_mode"}
    for node in ast.walk(tree):
        if isinstance(node, ast.arg):
            assert node.arg not in forbidden
        elif isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            assert not (
                isinstance(node.func.value, ast.Name)
                and node.func.value.id == "os"
                and node.func.attr == "getenv"
            )
        elif isinstance(node, ast.Attribute):
            assert not (
                isinstance(node.value, ast.Name)
                and node.value.id == "os"
                and node.attr == "environ"
            )


@pytest.mark.parametrize(
    "versions", [values for values in product((0, 1), repeat=4) if len(set(values)) == 2]
)
def test_mixed_legacy_and_v2_commit_preparation_ports_refuse_before_database_access(versions):
    implementations = (
        (PostgreSQLRootProofSigningIntent, PostgreSQLRootProofSigningIntentV2),
        (PostgreSQLRootProofRetention, PostgreSQLRootProofRetentionV2),
        (PostgreSQLRootProofIssuanceAuthority, PostgreSQLRootProofIssuanceAuthorityV2),
        (PostgreSQLAuthenticatedIssuerHistory, PostgreSQLAuthenticatedIssuerHistoryV2),
    )
    ports = [
        object.__new__(implementations[index][version]) for index, version in enumerate(versions)
    ]
    with pytest.raises(AtomicRootProofCommitConflict, match="GENUINE_PREPARATION_AUTHORITIES"):
        prepare_atomic_root_proof_commit(
            None,
            None,
            signing_intent=ports[0],
            retention=ports[1],
            issuer_authority=ports[2],
            history=ports[3],
        )
