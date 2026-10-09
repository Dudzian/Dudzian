"""Genuine public PostgreSQL authorities bound to separate protected private custody."""

from pathlib import Path
from types import SimpleNamespace

import pytest

from bot_core import cha_root_proof_signing_custody as custody
from bot_core.cha_attempt_store import AttemptState
from bot_core.cha_issuance_request import IssuanceSigningRole
from bot_core.licensing import (
    cha_root_proof_attempt_reservation as boundary,
    cha_root_proof_signed_attempt as signed,
)
from bot_core.postgresql_preaccount_credentials import CredentialLifecycle
from tests.security import (
    test_local_signing_custody as native_custody_tests,
    test_postgresql_preaccount_credentials_integration as registries,
    test_postgresql_root_proof_issuance_authority as genuine,
)
from tests.security._local_signing_platform import requires_native_custody_locking

native_keyring = native_custody_tests.native_keyring

wide = genuine.wide
challenge_harness = genuine.challenge_harness
integration = genuine.integration
package = genuine.package
lppi = genuine.lppi
active = genuine.active
committed = genuine.committed
cha = genuine.cha
reserved = genuine.reserved


@pytest.mark.external_postgresql
@requires_native_custody_locking
def test_genuine_registry_private_binding_through_guarded_initial_binding(
    reserved,
    native_keyring,
    tmp_path: Path,
    monkeypatch,
):
    _, binding, _, retained = reserved
    trust = retained["pdsa_trust_domain"]
    principal = "deployment-provisioning-principal"
    monkeypatch.setattr(boundary, "_attempt_store_path", lambda: tmp_path / "attempts.sqlite3")
    auth_scope = SimpleNamespace(
        environment="PRODUCTION", trust_domain=trust, provisioning_principal_id=principal
    )
    paths = signed._custody_directories(auth_scope)
    admins = tuple(
        custody.OfflineIssuanceCustodyAdministrator(
            path,
            role=role,
            trust_domain=trust,
            principal="CryptoHunterAccountAuthority"
            if role is IssuanceSigningRole.REQUESTER
            else principal,
        )
        for path, role in zip(paths, IssuanceSigningRole, strict=True)
    )
    drafts = tuple(
        admin.stage(key_id=key, key_version=1)
        for admin, key in zip(admins, ("requester-key", "claimant-key"), strict=True)
    )
    requester, claimant = signed._installed_custody(auth_scope)
    for signer in (requester, claimant):
        with pytest.raises(ValueError, match="ACTIVE"):
            signer.identity()
    with genuine._installed_authorities(
        monkeypatch,
        trust,
        requester_public=bytes.fromhex(drafts[0].public_key_hex),
        claimant_public=bytes.fromhex(drafts[1].public_key_hex),
    ) as (pair, _, _):
        # Real exact PostgreSQL types, ACL/durability/lifecycle qualification and
        # public material checks. No provisioning-verifier test substitution.
        admins[0].activate(pair.requester)
        admins[1].activate(pair.claimant)
        authorization = boundary.resolve_root_proof_issuance_authorization(binding)
        attempt = boundary.reserve_root_proof_issuance_attempt(binding, authorization)
        attempt_id = attempt.issuance_attempt_id
        result = signed.sign_root_proof_issuance_attempt(attempt)
        assert result.state is AttemptState.SIGNED_IMMUTABLE_DURABLE_NOT_SENT
        assert result.issuance_attempt_id == attempt_id
        assert (
            signed.resume_root_proof_issuance_attempt(binding, authorization).identity
            == result.identity
        )
        assert requester.identity().public_key_hex != claimant.identity().public_key_hex
        pair.claimant_admin.transition_lifecycle(
            principal_id=principal, lifecycle=CredentialLifecycle.VERIFY_ONLY, expected_revision=1
        )
        with pytest.raises(Exception) as rejected:
            _ = result.identity
        assert rejected.type.__name__ in {
            "RootProofAttemptReservationError",
            "CredentialResolutionError",
            "ProductionLocalIssuanceAuthorityError",
        }


@pytest.mark.external_postgresql
@requires_native_custody_locking
@pytest.mark.parametrize("role", list(IssuanceSigningRole))
def test_genuine_registry_mismatch_cannot_activate_staged_private_key(
    role,
    native_keyring,
    tmp_path,
    monkeypatch,
):
    with registries._isolated_pair() as pair:
        requester = role is IssuanceSigningRole.REQUESTER
        port, admin_port = (
            (pair.requester, pair.requester_admin)
            if requester
            else (pair.claimant, pair.claimant_admin)
        )
        principal = "CryptoHunterAccountAuthority" if requester else "deployment-principal"
        admin = custody.OfflineIssuanceCustodyAdministrator(
            tmp_path, role=role, trust_domain=registries.TRUST_DOMAIN, principal=principal
        )
        draft = admin.stage(key_id="key-one", key_version=1)
        assert draft.lifecycle == "STAGED"
        admin_port.provision_credential(
            principal_id=principal,
            credential_id="credential-one",
            key_id="key-one",
            key_version=1,
            public_key=b"Z" * 32,
        )
        with pytest.raises(custody.LocalSigningCustodyError, match="binding mismatch"):
            admin.activate(port)
        runtime = (
            custody.LocalCHARequesterSigningCustody(tmp_path, trust_domain=registries.TRUST_DOMAIN)
            if requester
            else custody.LocalPreaccountClaimantAuthorizationCustody(
                tmp_path, trust_domain=registries.TRUST_DOMAIN, provisioning_principal=principal
            )
        )
        with pytest.raises(ValueError, match="ACTIVE"):
            runtime.identity()
