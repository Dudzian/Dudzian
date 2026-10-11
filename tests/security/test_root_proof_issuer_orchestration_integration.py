"""Publiczna orkiestracja z całą genuine lineage InitialBinding Stage 9."""

import pytest

from bot_core import root_proof_issuer_orchestration as orchestration
from bot_core.authenticated_issuer_history import ReconciliationOutcome
from bot_core.root_proof_issuer_recovery import RootProofRecoveryPoint
from tests.security import test_root_proof_issuer_runtime as preflight_tests
from tests.security.test_root_proof_issuance_reconciliation import _history_boundary
from tests.security.test_root_proof_issuer_orchestration import (
    _durable_snapshot,
    _no_grant,
    _prohibit_mutations,
)

native_keyring = preflight_tests.native_keyring
wide = preflight_tests.wide
challenge_harness = preflight_tests.challenge_harness
integration = preflight_tests.integration
package = preflight_tests.package
lppi = preflight_tests.lppi
active = preflight_tests.active
committed = preflight_tests.committed
cha = preflight_tests.cha
reserved = preflight_tests.reserved
installed_signed_attempt = preflight_tests.installed_signed_attempt

pytestmark = pytest.mark.external_postgresql


def test_public_api_authenticates_genuine_binding_and_preserves_all_durable_authorities(
    installed_signed_attempt, monkeypatch
):
    with _history_boundary(installed_signed_attempt, monkeypatch) as authorities:
        before = _durable_snapshot(authorities)
        calls = _prohibit_mutations(authorities, monkeypatch)
        first = orchestration.orchestrate_root_proof_issuance(authorities.value.binding)
        assert first.disposition == "EXECUTION_BLOCKED"
        assert (
            first.admission
            is orchestration.RootProofAdmissionDecision.LIVE_ELIGIBLE_EXECUTION_BLOCKED
        )
        assert first.recovery.recovery_point is RootProofRecoveryPoint.BEFORE_FIRST_MUTATION
        assert first.recovery.outcome is ReconciliationOutcome.NOT_FOUND
        assert len(first.observations) == 3
        assert first.observations[0] == first.observations[1] == first.observations[2]
        assert (
            dict(first.observations[0].identifiers)["issuance_attempt_id"]
            == authorities.value.current.reservation.issuance_attempt_id
        )
        _no_grant(first)
        assert calls == []
        assert _durable_snapshot(authorities) == before
