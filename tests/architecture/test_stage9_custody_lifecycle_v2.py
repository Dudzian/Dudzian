"""Lifecycle administration cannot qualify or activate the public issuer."""

from __future__ import annotations

import ast
import json
from pathlib import Path
from typing import Any

import pytest

ROOT = Path(__file__).parents[2]
FROZEN = ROOT / "docs/architecture/cryptohunter_product_architecture"
PUBLIC_ENTRY_POINTS = (
    "bot_core.root_proof_issuer_runtime",
    "bot_core.root_proof_issuer_orchestration",
    "bot_core.root_proof_issuer_recovery",
)
INTERNAL_ADMIN_AUTHORITIES = {
    "bot_core.postgresql_custody_lifecycle",
    "bot_core.postgresql_custody_lifecycle_schema",
    "bot_core.postgresql_custody_lifecycle_qualification",
    "bot_core.local_signing_lifecycle_evidence",
}


def _dependencies(module: str) -> set[str]:
    path = ROOT.joinpath(*module.split(".")).with_suffix(".py")
    if not path.is_file():
        path = ROOT.joinpath(*module.split("."), "__init__.py")
        if not path.is_file():
            return set()
    package = module.split(".") if path.name == "__init__.py" else module.split(".")[:-1]
    dependencies = {".".join(package[:count]) for count in range(1, len(package) + 1)}
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if isinstance(node, ast.Import):
            dependencies.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                prefix = ".".join(
                    package[: len(package) - node.level + 1] + (node.module or "").split(".")
                ).rstrip(".")
            else:
                prefix = node.module or ""
            dependencies.add(prefix)
            dependencies.update(f"{prefix}.{alias.name}" for alias in node.names)
    return {
        dependency
        for dependency in dependencies
        if dependency == "bot_core" or dependency.startswith("bot_core.")
    }


def _contract(stem: str) -> dict[str, Any]:
    return json.loads((FROZEN / f"{stem}_contract.json").read_text(encoding="utf-8"))


@pytest.mark.parametrize("entry_point", PUBLIC_ENTRY_POINTS)
def test_public_issuer_has_no_lifecycle_admin_import_path(entry_point: str) -> None:
    pending = [entry_point]
    visited: set[str] = set()
    while pending:
        module = pending.pop()
        if module in visited:
            continue
        visited.add(module)
        assert module not in INTERNAL_ADMIN_AUTHORITIES, (
            f"public issuer entry point {entry_point} reaches lifecycle admin authority {module}"
        )
        pending.extend(_dependencies(module) - visited)


def test_stage9_still_cannot_issue_or_commit_account_genesis() -> None:
    status = json.loads((ROOT / "deployment/stage9_current_status.json").read_text())
    assert status["semantic_root_proof_issuer_runtime"] == "NOT_IMPLEMENTED / BLOCKED"
    assert status["root_proof"] == "NOT_ISSUED"
    assert status["account_genesis"] == "INITIAL_BINDING_ONLY"
    assert status["prepared"] == "NOT_STARTED"
    assert status["root_proof_admission"] == "NOT_STARTED"
    assert status["production_provisioning_ready"] is False
    assert status["windows_production_ready"] == "NOT_READY"


def test_frozen_lifecycle_forbids_reactivation_and_preserves_authority_separation() -> None:
    contract = _contract("m05_account_genesis_root_proof_issuer_production_substrate_selection")
    lifecycle = contract["credential_lifecycle"]
    assert lifecycle["planned_retirement_transition"] == "ACTIVE -> VERIFY_ONLY"
    assert lifecycle["compromise_transitions"] == ["ACTIVE -> REVOKED", "VERIFY_ONLY -> REVOKED"]
    assert lifecycle["REVOKED_is_terminal"] is True
    assert lifecycle["REVOKED_to_ACTIVE_allowed"] is False
    assert lifecycle["REVOKED_to_VERIFY_ONLY_allowed"] is False
    assert lifecycle["separately_reviewed_rehabilitation_protocol_exists"] is False
    for state in ("VERIFY_ONLY", "REVOKED"):
        assert lifecycle["states"][state]["new_role_authorized_operations"] is False
    assert (
        contract["authority_separation"][
            "separate_semantic_owners_schemas_roles_credentials_namespaces_transactions_and_lifecycles"
        ]
        is True
    )


def test_completion_mac_is_not_account_genesis_freshness_or_production_qualification() -> None:
    physical = _contract("m05_account_genesis_physical_persistence_crash_atomicity")
    freshness = physical["freshness_ownership_boundary"]
    assert freshness["valid_local_HMAC_establishes_freshness"] is False
    assert freshness["coordinator_may_unilaterally_advance_authoritative_generation"] is False
    assert (
        physical["key_lifecycle_interaction"]["authority_commit_and_key_rotation_same_operation"]
        is False
    )

    readiness = _contract("m05_account_genesis_root_proof_issuer_implementation_readiness")
    assert readiness["implementation_gate"]["non_production_implementation_mode"] == "NONE_FROZEN"
    assert readiness["production_M05_gate"]["current_production_M05_available"] is False
    assert readiness["status_preservation"]["RootProofIssuer_implementation_allowed"] is False
    issuer = _contract("m05_account_genesis_independent_root_proof_issuer")
    assert (
        issuer["implementation_status"]["RootProofIssuer_implementation_allowed_by_this_artifact"]
        is False
    )


def test_v2_deployment_explicitly_withholds_production_readiness() -> None:
    from bot_core import postgresql_custody_lifecycle as lifecycle

    assert lifecycle.PROTOCOL_VERSION == 2
    assert lifecycle.ATOMIC_ISSUER_READY == "NOT_READY"
    assert lifecycle.PRODUCTION_LIFECYCLE_ISOLATION == "BLOCKED"
