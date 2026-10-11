"""Public API z autentyczną pełną lineage InitialBinding i czterema magazynami.

Ten osobny przypadek zachowuje wszystkie wcześniejsze weryfikatory Stage 9.
Szybkie testy silnika porównawczego są w sąsiednim module.
"""

import pytest

from bot_core import root_proof_issuance_reconciliation as reconciliation
from bot_core.authenticated_issuer_history import ReconciliationOutcome
from tests.security import test_root_proof_issuer_runtime as preflight_tests
from tests.security.test_root_proof_issuance_reconciliation import _history_boundary, _no_grant

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


def test_public_api_reads_genuine_initial_binding_and_durable_attempt(
    installed_signed_attempt, monkeypatch
):
    with _history_boundary(installed_signed_attempt, monkeypatch) as authorities:
        before = preflight_tests._database_dump(authorities.value.path)
        report = reconciliation.reconcile_root_proof_issuance_attempt(authorities.value.binding)
        assert report.outcome is ReconciliationOutcome.NOT_FOUND
        assert report.disposition == "PRE_ISSUANCE_OBSERVED"
        assert len(report.observations) == 2
        _no_grant(report)
        assert preflight_tests._database_dump(authorities.value.path) == before
