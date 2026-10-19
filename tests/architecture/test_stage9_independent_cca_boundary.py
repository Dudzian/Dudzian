"""Synthetic CCA evidence cannot become a production capability or ingress."""

from __future__ import annotations

import ast
import base64
from functools import lru_cache
from pathlib import Path

import pytest

import bot_core.independent_cca_decision as cca
import bot_core.local_signing_lifecycle_evidence as lifecycle
import bot_core.production_signer_ipc as signer_ipc
from bot_core.licensing.cha_root_proof_signed_attempt import (
    SignedIssuanceAttemptError,
    sign_root_proof_issuance_attempt,
)
from bot_core.licensing.pdsa_enrollment_authorization import (
    issue_production_pdsa_enrollment_authorization,
)
from bot_core.licensing.production_pre_enrollment import ProductionPreEnrollmentError
from bot_core.local_signing_custody import LocalSigningLifecycleError
from bot_core.production_root_lifecycle import (
    ProductionRootLifecycleBoundary,
    ProductionRootLifecycleError,
)
from bot_core.root_proof_issuer_orchestration import orchestrate_root_proof_issuance
from tests.architecture.test_stage9_atomic_root_proof_commit_boundary import (
    _repository_dependencies,
)
from tests.security._cca_decision_fixtures import build_cca_case
from tests.security.test_independent_custody_completion_qualification import (
    completion as completion,
)

ROOT = Path(__file__).resolve().parents[2]
TEST_LOG = "bot_core.test_only_cca_decision_log"
VERIFIER = "bot_core.independent_cca_decision"
PRODUCTION_ENTRY_POINTS = (
    "bot_core.production_root_lifecycle",
    "bot_core.production_signer_custody",
    "bot_core.production_signer_ipc",
    "bot_core.root_proof_issuer_runtime",
    "bot_core.root_proof_issuer_orchestration",
    "bot_core.root_proof_issuer_recovery",
    "bot_core.licensing.production_pre_enrollment",
    "bot_core.licensing.pdsa_enrollment_authorization",
)


@pytest.fixture
def cca_case(completion):
    return build_cca_case(completion.descriptor, completion.journal)


@pytest.fixture(params=["decision", "report", "envelope"])
def cca_artifact(cca_case, request):
    report = cca.verify_independent_cca_decision(
        cca_case.decision, policy_loader=lambda: cca_case.policy
    )
    assert report.signature == "MATCHED_EXPLICIT_POLICY"
    assert report.source_signature == "MATCHED_EXPLICIT_POLICY"
    assert report.status == "NOT_QUALIFIED"
    assert report.production_emission == "BLOCKED"
    return {
        "decision": cca_case.decision,
        "report": report,
        "envelope": cca_case.envelope,
    }[request.param]


def test_valid_synthetic_cca_evidence_cannot_publish_or_bind(cca_artifact):
    boundary = object.__new__(ProductionRootLifecycleBoundary)
    with pytest.raises(
        ProductionRootLifecycleError,
        match="^PRODUCTION_BIND_BLOCKED_FROZEN_TRANSACTION_CONFLICT_AND_UNQUALIFIED_ATTEMPT_INGRESS$",
    ):
        boundary.commit(cca_artifact)
    with pytest.raises(
        ProductionRootLifecycleError,
        match="^INDEPENDENT_CUSTODY_COMPLETION_AUTHORITY_REQUIRED$",
    ):
        boundary.publish_administrator_completion(
            None, "qualification-op-1", "a" * 64, cca_artifact
        )


def test_valid_synthetic_cca_evidence_cannot_mutate_custody(cca_case, cca_artifact):
    with pytest.raises(LocalSigningLifecycleError, match="genuine durable"):
        with lifecycle.verified_durable_signing_lifecycle_completion(
            cca_artifact,
            operation_descriptor=base64.b64decode(
                cca_case.payload["originals"]["operation_descriptor_b64"]
            ),
        ):
            pytest.fail("CCA diagnostic became a custody mutation capability")


def test_valid_synthetic_cca_evidence_cannot_open_issuer_sign_or_bind(cca_artifact):
    with pytest.raises(SignedIssuanceAttemptError, match="^VERIFIED_ATTEMPT_RESERVATION_REQUIRED$"):
        sign_root_proof_issuance_attempt(cca_artifact)
    result = orchestrate_root_proof_issuance(cca_artifact)
    assert result.disposition == "REFUSED"
    assert result.authorizes_operations is False
    assert result.root_proof_issuance_available is False
    assert {"ROOT_PROOF_SIGN", "REGISTRY_BIND", "CHA_FINALIZE"}.issubset(result.blocked_operations)


def test_valid_synthetic_cca_evidence_cannot_open_enrollment(cca_artifact):
    with pytest.raises(
        ProductionPreEnrollmentError,
        match="^AUTHENTICATED_PRODUCTION_PRE_ENROLLMENT_REQUIRED$",
    ):
        issue_production_pdsa_enrollment_authorization(cca_artifact)


@pytest.mark.parametrize("role", ["root_signer", "history_signer"])
def test_cca_report_does_not_add_sign_to_closed_production_signer_protocol(
    cca_artifact, role, tmp_path
):
    policy = signer_ipc.SignerSocketPolicy(
        role, str(tmp_path / "signer.sock"), 1001, 2001, 1002, 2002, "a" * 64
    )
    with pytest.raises(
        signer_ipc.ProductionSignerIPCError,
        match="^EXACT_DIAGNOSTIC_SIGNER_REQUEST_REQUIRED$",
    ):
        signer_ipc._request(
            {
                "version": 1,
                "operation": "SIGN",
                "role": role,
                "installation_sha256": "a" * 64,
                "nonce": "b" * 64,
                "cca_report": cca_artifact,
            },
            policy,
        )


@lru_cache(maxsize=None)
def _dependencies(module):
    return _repository_dependencies(module)


@pytest.mark.parametrize("entry_point", PRODUCTION_ENTRY_POINTS)
def test_production_import_graph_cannot_reach_test_cca_or_qualification_ingress(entry_point):
    pending, visited = [entry_point], set()
    while pending:
        module = pending.pop()
        if module in visited:
            continue
        visited.add(module)
        assert module not in {TEST_LOG, VERIFIER}, (
            f"production entry point {entry_point} reaches TEST/qualification CCA: {module}"
        )
        pending.extend(_dependencies(module) - visited)


def test_cca_verifier_uses_concrete_imports_and_cannot_reach_test_writer():
    path = ROOT / "bot_core/independent_cca_decision.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    assert not any(
        isinstance(node, ast.ImportFrom) and node.module == "bot_core" for node in ast.walk(tree)
    )
    pending, visited = [VERIFIER], set()
    while pending:
        module = pending.pop()
        if module in visited:
            continue
        visited.add(module)
        assert module != TEST_LOG, "qualification verifier reaches TEST CCA writer"
        pending.extend(_dependencies(module) - visited)
