"""Cross-layer refusals while production transaction authority is unresolved.

These tests exercise real input guards and unconditional production blockers.
Detached claims and isolated-test results never qualify an installed deployment,
an issuance attempt, or a physical PostgreSQL transaction.
"""

from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import pytest

from bot_core import (
    postgresql_atomic_root_proof_commit as isolated,
    postgresql_atomic_root_proof_transaction as transaction,
    production_root_lifecycle as production,
    root_proof_issuance_reconciliation as reconciliation,
    root_proof_issuer_orchestration as orchestration,
)
from bot_core.authenticated_issuer_history import ReconciliationOutcome


def _detached_claim(kind):
    if kind == "readiness_flags":
        return {
            "state": "READY",
            "profile": "PRODUCTION_LOCAL",
            "production_ready": True,
            "authorizes_operations": True,
            "same_connection": True,
            "atomic_global_snapshot": True,
        }
    if kind == "rewritten_orchestration_report":
        return replace(
            orchestration.orchestrate_root_proof_issuance(object()),
            disposition="ISSUED",
            outcome=ReconciliationOutcome.EXACT_COMMITTED,
            root_proof_issuance_available=True,
            authorizes_operations=True,
            atomic_global_snapshot=True,
            reasons=(),
            blocked_operations=(),
        )
    if kind == "isolated_committed_decision":
        return isolated.AtomicRootProofCommitDecision(
            {"state": "COMMITTED", "authorizes_operations": True},
            b"caller-payload",
            b"s" * 64,
            b"caller-envelope",
        )
    if kind == "transaction_owner":
        return transaction.AtomicRootProofTransactionOwner(object(), object())
    if kind == "isolated_commit_authority":
        return object.__new__(isolated.PostgreSQLAtomicRootProofCommit)
    raise AssertionError("unknown detached claim fixture")


_CLAIMS = (
    "readiness_flags",
    "rewritten_orchestration_report",
    "isolated_committed_decision",
    "transaction_owner",
    "isolated_commit_authority",
)


def _unexpected_authority_use(*_args, **_kwargs):
    pytest.fail("detached caller claims reached an authority or writable port")


@pytest.mark.parametrize("kind", _CLAIMS)
def test_production_commit_refuses_cross_layer_claims_before_any_authority_use(kind, monkeypatch):
    claim = _detached_claim(kind)
    boundary = object.__new__(production.ProductionRootLifecycleBoundary)
    monkeypatch.setattr(
        production.ProductionRootLifecycleBoundary, "_current", _unexpected_authority_use
    )
    monkeypatch.setattr(isolated.psycopg, "connect", _unexpected_authority_use)
    with pytest.raises(
        production.ProductionRootLifecycleError,
        match="PRODUCTION_BIND_BLOCKED_FROZEN_TRANSACTION_CONFLICT_AND_UNQUALIFIED_ATTEMPT_INGRESS",
    ):
        boundary.commit(
            claim,
            transaction_owner=claim,
            preparation=claim,
            approval=claim,
            approved=True,
            contract_status="CONTRACTUALLY_APPROVED",
            production_ready=True,
            authorizes_operations=True,
        )


@pytest.mark.parametrize("kind", _CLAIMS)
@pytest.mark.parametrize(
    "entrypoint",
    [
        orchestration.orchestrate_root_proof_issuance,
        reconciliation.reconcile_root_proof_issuance_attempt,
    ],
)
def test_public_issuance_readers_refuse_cross_layer_claims_before_composition(
    kind, entrypoint, monkeypatch
):
    claim = _detached_claim(kind)
    monkeypatch.setattr(
        reconciliation, "_configured_reconciliation_runtime", _unexpected_authority_use
    )
    monkeypatch.setattr(isolated.psycopg, "connect", _unexpected_authority_use)
    result = entrypoint(claim)
    assert result.disposition == "REFUSED"
    assert result.outcome is ReconciliationOutcome.CONFLICT
    assert result.authorizes_operations is False
    assert result.atomic_global_snapshot is False
    assert result.observations == ()


@pytest.mark.parametrize("kind", _CLAIMS)
def test_internal_orchestration_cannot_promote_claims_via_exact_runtime_dto(kind, monkeypatch):
    claim = _detached_claim(kind)
    runtime = reconciliation._ReconciliationRuntime(claim, claim, claim, (claim,))
    monkeypatch.setattr(reconciliation, "_capture", _unexpected_authority_use)
    monkeypatch.setattr(isolated.psycopg, "connect", _unexpected_authority_use)
    result = orchestration._orchestrate_selected_authorities(object(), runtime)
    assert result.disposition == "REFUSED"
    assert result.authorizes_operations is False
    assert result.root_proof_issuance_available is False
    assert result.observations == ()
    assert "ATOMIC_SIGNED_PROOF_BIND_HISTORY_AUTHORITY_UNAVAILABLE" in result.reasons
    assert "SEMANTIC_ISSUER_SIGNING_COMPOSITION_UNAVAILABLE" in result.reasons


@pytest.mark.parametrize("connection", [None, object(), SimpleNamespace(closed=False)])
def test_production_fencing_helper_requires_a_live_connection_before_qualification(
    connection, monkeypatch
):
    boundary = object.__new__(production.ProductionRootLifecycleBoundary)
    boundary._role = "issuer"
    installed = SimpleNamespace(
        pin=SimpleNamespace(provisioning=SimpleNamespace(issuer_role="production_issuer"))
    )
    monkeypatch.setattr(
        production.ProductionRootLifecycleBoundary, "_current", lambda _self: installed
    )
    monkeypatch.setattr(
        production.ProductionRootLifecycleBoundary, "qualify", _unexpected_authority_use
    )
    with pytest.raises(
        transaction.AtomicRootProofTransactionError, match="OPEN_FINAL_TRANSACTION_REQUIRED"
    ):
        boundary.assert_active_fences(connection, [])
