"""Read-only semantic preflight against genuine PostgreSQL public authorities.

The private custody fixtures exercise production adapters and native filesystem
locking with a controlled OS-keyring test seam. They do not provision deployment
credentials or provide evidence of production enrollment or issuer availability.
"""

from __future__ import annotations

import copy
import hashlib
import sqlite3
from dataclasses import FrozenInstanceError, asdict, replace
from pathlib import Path
from types import SimpleNamespace

import psycopg
import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from psycopg import sql
from psycopg.types.json import Jsonb

from bot_core import cha_root_proof_signing_custody as custody, root_proof_issuer_runtime as runtime
from bot_core.cha_attempt_store import AttemptState
from bot_core.cha_issuance_request import (
    CLAIMANT_DOMAIN,
    REQUESTER_DOMAIN,
    IssuanceSigningRole,
    encode_signature,
    request_reference,
)
from bot_core.entitlement_registry_contract import (
    AdminOutcome,
    EntitlementLifecycle,
    RevokeEntitlementRequest,
    SupersedeEntitlementRequest,
    UnboundBinding,
    admin_predecessor_for,
)
from bot_core.licensing import (
    cha_root_proof_attempt_reservation as boundary,
    cha_root_proof_signed_attempt as signed,
)
from bot_core.licensing.canonical import canonical_json_bytes, parse_canonical
from bot_core.postgresql_preaccount_credentials import CredentialLifecycle
from bot_core.root_proof_issuer_substrate import public_key_material_identity
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


@pytest.fixture
def installed_signed_attempt(reserved, native_keyring, tmp_path: Path, monkeypatch):
    """Generate actual distinct Ed25519 keys, public rows and a durable attempt."""

    _, binding, _, retained = reserved
    trust = retained["pdsa_trust_domain"]
    principal = "deployment-provisioning-principal"
    path = tmp_path / "attempts.sqlite3"
    monkeypatch.setattr(boundary, "_attempt_store_path", lambda: path)
    scope = SimpleNamespace(
        environment="PRODUCTION", trust_domain=trust, provisioning_principal_id=principal
    )
    admins = tuple(
        custody.OfflineIssuanceCustodyAdministrator(
            directory,
            role=role,
            trust_domain=trust,
            principal="CryptoHunterAccountAuthority"
            if role is IssuanceSigningRole.REQUESTER
            else principal,
        )
        for directory, role in zip(
            signed._custody_directories(scope), IssuanceSigningRole, strict=True
        )
    )
    drafts = tuple(
        admin.stage(key_id=key, key_version=1)
        for admin, key in zip(admins, ("requester-key", "claimant-key"), strict=True)
    )
    with genuine._installed_authorities(
        monkeypatch,
        trust,
        requester_public=bytes.fromhex(drafts[0].public_key_hex),
        claimant_public=bytes.fromhex(drafts[1].public_key_hex),
    ) as (pair, subject, entitlement_admin):
        admins[0].activate(pair.requester)
        admins[1].activate(pair.claimant)
        authorization = boundary.resolve_root_proof_issuance_authorization(binding)
        reservation = boundary.reserve_root_proof_issuance_attempt(binding, authorization)
        opaque = signed.sign_root_proof_issuance_attempt(reservation)
        current, retained_request = signed._signed_snapshot(opaque)
        authority = boundary._issuance_authority_provider()
        yield SimpleNamespace(
            binding=binding,
            authorization=authorization,
            reservation=reservation,
            opaque=opaque,
            current=current,
            retained=retained_request,
            authority=authority,
            pair=pair,
            subject=subject,
            entitlement_admin=entitlement_admin,
            path=path,
            trust=trust,
            principal=principal,
        )


def _database_dump(path: Path) -> tuple[str, ...]:
    with sqlite3.connect(path) as connection:
        return tuple(connection.iterdump())


def _reject_mutations(*args, **kwargs):
    pytest.fail("semantic preflight invoked a mutable issuance operation")


@pytest.mark.external_postgresql
@requires_native_custody_locking
def test_real_semantic_preflight_reports_exact_current_tuple_without_issuance(
    installed_signed_attempt, monkeypatch
):
    value = installed_signed_attempt
    before = _database_dump(value.path)
    entitlement = value.authority.entitlement_registry.authoritative_state(value.subject)
    requester = value.pair.requester.active_requester_credential("CryptoHunterAccountAuthority")
    claimant = value.pair.claimant.resolve_claimant(value.principal)
    monkeypatch.setattr(signed, "execute_local_issuance", _reject_mutations)
    monkeypatch.setattr(custody._RuntimeCustody, "_sign", _reject_mutations)
    monkeypatch.setattr(
        type(value.authority.entitlement_registry), "compare_and_swap_bind", _reject_mutations
    )

    report = runtime.preflight_root_proof_issuance_attempt(value.opaque)
    assert type(report) is runtime.RootProofIssuancePreflight
    auth = value.authorization.authorization
    identity = value.current.identity
    assert asdict(report) == {
        "disposition": "VERIFIED_NOT_AUTHORIZED_TO_ISSUE",
        "security_profile": "PRODUCTION_LOCAL",
        "environment": "PRODUCTION",
        "trust_domain": value.trust,
        "logical_operation_id": auth.logical_operation_id,
        "account_id": auth.account_id,
        "issuance_attempt_id": identity.issuance_attempt_id,
        "request_reference": value.retained.reference,
        "request_digest_sha256": hashlib.sha256(value.retained.canonical_bytes).hexdigest(),
        "initial_binding_reference": auth.initial_binding_reference,
        "initial_binding_digest_sha256": auth.initial_binding_digest_sha256,
        "entitlement_id": auth.bootstrap_entitlement_id,
        "entitlement_generation": 1,
        "entitlement_registry_revision": 1,
        "requester_registry_revision": 1,
        "claimant_registry_revision": 1,
        "attempt_identity_digest_sha256": identity.digest_sha256,
    }
    assert runtime.preflight_root_proof_issuance_attempt(value.opaque) == report
    with pytest.raises(FrozenInstanceError):
        report.environment = "TEST"
    assert _database_dump(value.path) == before
    assert value.authority.entitlement_registry.authoritative_state(value.subject) == entitlement
    assert (
        value.pair.requester.active_requester_credential("CryptoHunterAccountAuthority")
        == requester
    )
    assert value.pair.claimant.resolve_claimant(value.principal) == claimant
    assert entitlement.state.lifecycle is EntitlementLifecycle.ACTIVE
    assert type(entitlement.state.binding) is UnboundBinding
    assert value.opaque.state is AttemptState.SIGNED_IMMUTABLE_DURABLE_NOT_SENT
    with pytest.raises(signed.SignedIssuanceAttemptError):
        signed.require_verified_signed_immutable_root_proof_issuance_attempt(report)
    with pytest.raises(boundary.RootProofAttemptReservationError):
        boundary.require_verified_root_proof_issuance_authorization(report)
    for untrusted in (report, copy.copy(report), asdict(report), identity, value.reservation):
        with pytest.raises(runtime.RootProofIssuerRuntimeError):
            runtime.preflight_root_proof_issuance_attempt(untrusted)


@pytest.mark.parametrize(
    "untrusted",
    [
        None,
        {},
        b"signed-request",
        object(),
        object.__new__(signed.VerifiedSignedImmutableRootProofIssuanceAttempt),
    ],
)
def test_raw_or_forged_input_fails_before_authority_composition(untrusted, monkeypatch):
    monkeypatch.setattr(boundary, "_issuance_authority_provider", _reject_mutations)
    with pytest.raises(runtime.RootProofIssuerRuntimeError):
        runtime.preflight_root_proof_issuance_attempt(untrusted)


def test_upstream_failure_does_not_disclose_database_or_custody_secrets(monkeypatch):
    def unavailable(value):
        raise RuntimeError("database password=TEST_ONLY-secret deployment-private-selector")

    monkeypatch.setattr(runtime, "_read_issuer_preflight_source", unavailable)
    with pytest.raises(runtime.RootProofIssuerRuntimeError) as failure:
        runtime.preflight_root_proof_issuance_attempt(object())
    assert str(failure.value) == "PREFLIGHT_AUTHORITY_OR_REQUEST_UNAVAILABLE"
    assert failure.value.__cause__ is None
    assert failure.value.__suppress_context__


@pytest.mark.external_postgresql
@requires_native_custody_locking
@pytest.mark.parametrize("timing", ["before_preflight", "after_verification"])
def test_store_owner_replay_cannot_reuse_a_signed_capability(
    installed_signed_attempt, monkeypatch, timing
):
    value = installed_signed_attempt
    before = _database_dump(value.path)
    alternate = value.path.with_name("other-owner.sqlite3")
    if timing == "before_preflight":
        monkeypatch.setattr(boundary, "_attempt_store_path", lambda: alternate)
    else:

        def cut(name):
            if name == "REQUEST_VERIFIED":
                monkeypatch.setattr(boundary, "_attempt_store_path", lambda: alternate)

        monkeypatch.setattr(runtime, "_cut", cut)
    with pytest.raises(runtime.RootProofIssuerRuntimeError):
        runtime.preflight_root_proof_issuance_attempt(value.opaque)
    assert _database_dump(value.path) == before
    assert not alternate.exists()


@pytest.mark.external_postgresql
@requires_native_custody_locking
@pytest.mark.parametrize(
    "mutation",
    ["request", "reference", "requester_signature", "claimant_signature", "mixed_signatures"],
)
def test_durable_request_or_signature_corruption_fails_closed(installed_signed_attempt, mutation):
    value = installed_signed_attempt
    table = "issuance_requests" if mutation in {"request", "reference"} else "signature_checkpoints"
    with sqlite3.connect(value.path) as connection:
        connection.execute(f"DROP TRIGGER {table}_immutable_update")
        if mutation == "request":
            connection.execute("UPDATE issuance_requests SET canonical_bytes=?", (b"{}",))
        elif mutation == "reference":
            connection.execute(
                "UPDATE issuance_requests SET reference=?", ("immutable:req:sha256:" + "a" * 64,)
            )
        elif mutation == "mixed_signatures":
            connection.execute(
                "UPDATE signature_checkpoints SET signature=? WHERE role=?",
                (value.retained.claimant[1], IssuanceSigningRole.REQUESTER.value),
            )
        else:
            role = (
                IssuanceSigningRole.REQUESTER
                if mutation == "requester_signature"
                else IssuanceSigningRole.CLAIMANT
            )
            connection.execute(
                "UPDATE signature_checkpoints SET signature=? WHERE role=?",
                (encode_signature(b"x" * 64), role.value),
            )
        connection.execute(
            f"CREATE TRIGGER {table}_immutable_update BEFORE UPDATE ON {table} "
            f"BEGIN SELECT RAISE(ABORT,'immutable {table}'); END"
        )
    damaged = _database_dump(value.path)
    with pytest.raises(runtime.RootProofIssuerRuntimeError):
        runtime.preflight_root_proof_issuance_attempt(value.opaque)
    assert _database_dump(value.path) == damaged


def _change_authority(value, mutation):
    if mutation in {"requester_inactive", "claimant_inactive"}:
        requester = mutation.startswith("requester")
        admin = value.pair.requester_admin if requester else value.pair.claimant_admin
        principal = "CryptoHunterAccountAuthority" if requester else value.principal
        admin.transition_lifecycle(
            principal_id=principal,
            lifecycle=CredentialLifecycle.VERIFY_ONLY,
            expected_revision=1,
        )
    elif mutation in {"requester_rotation", "claimant_rotation"}:
        requester = mutation.startswith("requester")
        admin = value.pair.requester_admin if requester else value.pair.claimant_admin
        principal = "CryptoHunterAccountAuthority" if requester else value.principal
        prefix = "requester" if requester else "claimant"
        admin.rotate_credential(
            principal_id=principal,
            credential_id=prefix + "-credential-next",
            key_id=prefix + "-key-next",
            key_version=2,
            public_key=b"N" * 32,
            expected_revision=1,
        )
    else:
        original = value.authority.entitlement_registry.authoritative_state(value.subject).state
        if mutation == "entitlement_revoked":
            result = value.entitlement_admin.revoke_entitlement(
                RevokeEntitlementRequest(admin_predecessor_for(original))
            )
            assert result.outcome is AdminOutcome.COMMITTED
        elif mutation == "entitlement_superseded":
            result = value.entitlement_admin.supersede_entitlement(
                SupersedeEntitlementRequest(
                    admin_predecessor_for(original),
                    replace(original.identity, entitlement_generation=2),
                    original.provenance,
                )
            )
            assert result.outcome is AdminOutcome.COMMITTED
        elif mutation == "entitlement_bound_corruption":
            # Damaged BOUND authority evidence is a negative control, not an
            # issuer-authenticated BIND decision or a legal enrollment.
            with psycopg.connect(registries.BASE_DSN, autocommit=True) as connection:
                connection.execute(
                    sql.SQL(
                        "UPDATE {}.history SET state=jsonb_set(state,%s,%s), "
                        "state_integrity=md5(jsonb_set(state,%s,%s)::text) "
                        "WHERE lookup_handle=%s"
                    ).format(sql.Identifier(value.entitlement_admin._schema)),
                    (
                        ["binding"],
                        Jsonb({"kind": "BOUND"}),
                        ["binding"],
                        Jsonb({"kind": "BOUND"}),
                        value.subject.lookup_handle,
                    ),
                )
        else:
            raise AssertionError("unknown authority mutation")


@pytest.mark.external_postgresql
@requires_native_custody_locking
@pytest.mark.parametrize(
    "mutation",
    [
        "requester_inactive",
        "claimant_inactive",
        "requester_rotation",
        "claimant_rotation",
        "entitlement_revoked",
        "entitlement_superseded",
        "entitlement_bound_corruption",
    ],
)
def test_signed_attempt_cannot_replay_after_live_authority_changes(
    installed_signed_attempt, mutation
):
    value = installed_signed_attempt
    before = _database_dump(value.path)
    _change_authority(value, mutation)
    with pytest.raises(runtime.RootProofIssuerRuntimeError):
        runtime.preflight_root_proof_issuance_attempt(value.opaque)
    assert _database_dump(value.path) == before


@pytest.mark.external_postgresql
@requires_native_custody_locking
@pytest.mark.parametrize(
    "mutation", ["requester_inactive", "claimant_rotation", "entitlement_revoked"]
)
def test_authority_change_after_verification_is_revalidated(
    installed_signed_attempt, monkeypatch, mutation
):
    value = installed_signed_attempt
    seen = []

    def cut(name):
        if name == "REQUEST_VERIFIED":
            seen.append(name)
            _change_authority(value, mutation)

    monkeypatch.setattr(runtime, "_cut", cut)
    before = _database_dump(value.path)
    with pytest.raises(runtime.RootProofIssuerRuntimeError):
        runtime.preflight_root_proof_issuance_attempt(value.opaque)
    assert seen == ["REQUEST_VERIFIED"]
    assert _database_dump(value.path) == before


def _test_private_pair(value):
    """Adversarial signatures use only freshly generated test custody keys.

    Access stays in this negative-control helper, never the runtime. Reading
    real deployment secrets is outside this test's isolated custody fixture.
    """
    private = []
    for directory, role, principal in zip(
        signed._custody_directories(value.authorization.authorization),
        IssuanceSigningRole,
        ("CryptoHunterAccountAuthority", value.principal),
        strict=True,
    ):
        record = custody._read(directory, role, value.trust, principal)
        private.append(custody._private(directory, record))
    return tuple(private)


def _self_consistent_request(value, raw, requester_signature, claimant_signature, requester=None):
    retained = replace(
        value.retained,
        canonical_bytes=raw,
        reference=request_reference(raw),
        digest=hashlib.sha256(raw).hexdigest(),
        requester=(requester or value.retained.requester[0], requester_signature),
        claimant=(value.retained.claimant[0], claimant_signature),
    )
    identity = replace(
        value.current.identity,
        root_proof_issuance_request_signed_payload_digest_sha256=retained.digest,
        root_proof_issuance_request_canonical_bytes_reference=retained.reference,
        requester_signature_base64url=requester_signature,
        claimant_authorization_signature_base64url=claimant_signature,
    )
    current = replace(
        value.current, identity=identity, immutable_attempt_digest_sha256=identity.digest_sha256
    )
    return current, retained


@pytest.mark.external_postgresql
@requires_native_custody_locking
def test_independent_verifier_rejects_every_replayed_field_even_with_valid_signatures(
    installed_signed_attempt,
):
    value = installed_signed_attempt
    source, current, retained = signed._read_issuer_preflight_source(value.opaque)
    assert runtime._verify_request(source, current, retained) == (1, 1, 1)
    private = _test_private_pair(value)
    original = parse_canonical(retained.canonical_bytes)
    candidates = []
    for field, original_value in original.items():
        changed = dict(original)
        changed[field] = (
            original_value + 1 if type(original_value) is int else original_value + "-replay"
        )
        candidates.append(changed)
    candidates.append({**original, "caller_claimed_authorization": "READY"})
    candidates.append({key: content for key, content in original.items() if key != "account_id"})
    for payload in candidates:
        raw = canonical_json_bytes(payload)
        signatures = tuple(
            encode_signature(key.sign(role.domain.encode("ascii") + b"\x00" + raw))
            for key, role in zip(private, IssuanceSigningRole, strict=True)
        )
        changed_current, changed_retained = _self_consistent_request(value, raw, *signatures)
        # Both retained signatures and the immutable identity are internally
        # consistent. The exact current operation/namespace/scope still wins.
        for checkpoint in (changed_retained.requester, changed_retained.claimant):
            if payload.keys() == original.keys():
                checkpoint[0].verify(raw, checkpoint[1])
        with pytest.raises(runtime.RootProofIssuerRuntimeError, match="EXACT_IMMUTABLE_REQUEST"):
            runtime._verify_request(source, changed_current, changed_retained)


@pytest.mark.external_postgresql
@requires_native_custody_locking
def test_independent_verifier_rejects_cross_domain_signatures_and_unregistered_public_key(
    installed_signed_attempt,
):
    value = installed_signed_attempt
    source, _, retained = signed._read_issuer_preflight_source(value.opaque)
    raw = retained.canonical_bytes
    requester_private, claimant_private = _test_private_pair(value)
    cross_requester = encode_signature(
        requester_private.sign(CLAIMANT_DOMAIN.encode("ascii") + b"\x00" + raw)
    )
    cross_claimant = encode_signature(
        claimant_private.sign(REQUESTER_DOMAIN.encode("ascii") + b"\x00" + raw)
    )
    for requester_signature, claimant_signature in (
        (cross_requester, retained.claimant[1]),
        (retained.requester[1], cross_claimant),
        (retained.claimant[1], retained.requester[1]),
        (encode_signature(b"x" * 64), retained.claimant[1]),
    ):
        current, changed = _self_consistent_request(
            value, raw, requester_signature, claimant_signature
        )
        with pytest.raises(ValueError, match="signature verification failed"):
            runtime._verify_request(source, current, changed)

    rogue_private = Ed25519PrivateKey.generate()
    rogue_public = rogue_private.public_key().public_bytes(
        serialization.Encoding.Raw, serialization.PublicFormat.Raw
    )
    rogue_requester = replace(
        retained.requester[0],
        public_key_hex=rogue_public.hex(),
        key_material_identity=public_key_material_identity(rogue_public),
    )
    rogue_signature = encode_signature(
        rogue_private.sign(REQUESTER_DOMAIN.encode("ascii") + b"\x00" + raw)
    )
    rogue_requester.verify(raw, rogue_signature)
    current, changed = _self_consistent_request(
        value, raw, rogue_signature, retained.claimant[1], requester=rogue_requester
    )
    with pytest.raises(runtime.RootProofIssuerRuntimeError, match="EXACT_ACTIVE_PUBLIC_SIGNER"):
        runtime._verify_request(source, current, changed)
