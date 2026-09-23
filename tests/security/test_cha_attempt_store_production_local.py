from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import base64
from dataclasses import asdict, replace
from pathlib import Path
import json
import sqlite3

import pytest

import bot_core.cha_attempt_store as cha
from bot_core.cha_attempt_store import (
    AttemptAuthorization,
    AttemptConflictError,
    AttemptCorruptError,
    AttemptIdentity,
    AttemptNotFoundError,
    AttemptSchemaUnsupportedError,
    AttemptState,
    AuthoritativeUnboundEvidence,
    RecoveryResolution,
    SQLiteCHAAttemptStore,
)
from bot_core.root_proof_issuer_substrate import ProviderRole, SecurityProfile
from bot_core.root_proof_issuer_substrate import RootProofIssuerCompositionGate


def authorization(**changes: object) -> AttemptAuthorization:
    values: dict[str, object] = {
        "environment": "PRODUCTION_LOCAL",
        "trust_domain": "td-production",
        "product_scope": "CryptoHunter",
        "logical_operation_id": "ago_operation",
        "account_id": "acct_018f3e70-7b5c-7c21-8b9a-0123456789ab",
        "canonical_genesis_request_fingerprint_sha256": "1" * 64,
        "initial_binding_reference": "ib:authority:1",
        "initial_binding_digest_sha256": "2" * 64,
        "bootstrap_entitlement_id": "ent_bootstrap",
        "entitlement_generation": 1,
        "requester_principal_id": "CryptoHunterAccountAuthority",
        "requester_credential_role": "ACCOUNT_GENESIS_ROOT_PROOF_ISSUANCE_REQUESTER_V1",
        "requester_key_id": "rpr_local_1",
        "requester_key_version": 1,
        "provisioning_principal_id": "prv_local_1",
        "claimant_key_id": "clm_local_1",
        "claimant_key_version": 1,
    }
    values.update(changes)
    return AttemptAuthorization(**values)  # type: ignore[arg-type]


def identity(auth: AttemptAuthorization, attempt_id: str) -> AttemptIdentity:
    signature = base64.urlsafe_b64encode(bytes(range(64))).rstrip(b"=").decode()
    return AttemptIdentity(
        auth,
        attempt_id,
        "3" * 64,
        "immutable:req:1",
        signature,
        signature,
        "CRYPTOHUNTER_ACCOUNT_GENESIS_ROOT_PROOF_ISSUANCE_REQUEST_V1/JCS-SHA256-Ed25519-v1",
        "CRYPTOHUNTER_ACCOUNT_GENESIS_ROOT_PROOF_ENTITLEMENT_CLAIM_V1/JCS-SHA256-Ed25519-v1",
    )


def evidence(
    attempt_id: str, auth: AttemptAuthorization | None = None
) -> AuthoritativeUnboundEvidence:
    auth = auth or authorization()
    return AuthoritativeUnboundEvidence(
        "2",
        auth.environment,
        auth.trust_domain,
        auth.product_scope,
        "issuer-authority-local",
        "issuer-registry-local",
        auth.bootstrap_entitlement_id,
        auth.entitlement_generation,
        auth.logical_operation_id,
        auth.account_id,
        auth.canonical_genesis_request_fingerprint_sha256,
        attempt_id,
        auth.initial_binding_reference,
        auth.initial_binding_digest_sha256,
        "AUTHORITATIVELY_UNBOUND",
        "registry-state-7",
        "revision-7",
        "issuer:evidence:7",
        "4" * 64,
        "ROOT_PROOF_RECONCILIATION_V1",
    )


def open_store(path: Path) -> SQLiteCHAAttemptStore:
    return SQLiteCHAAttemptStore(path, "td-production")


def tamper_metadata(db: sqlite3.Connection, assignment: str) -> None:
    db.execute("DROP TRIGGER store_metadata_immutable_update")
    db.execute(f"UPDATE store_metadata SET {assignment}")  # noqa: S608 - fixed test literals
    db.execute(
        "CREATE TRIGGER store_metadata_immutable_update BEFORE UPDATE ON store_metadata "
        "BEGIN SELECT RAISE(ABORT,'immutable metadata'); END"
    )


def test_creates_dedicated_store_with_identity_permissions_and_effective_pragmas(
    tmp_path: Path,
) -> None:
    path = (tmp_path / "cha" / "attempts.sqlite3").resolve()
    with open_store(path) as store:
        assert path.is_file()
        assert store.identity.role is ProviderRole.CHA_ATTEMPT_STORE
        assert store.identity.security.profile is SecurityProfile.PRODUCTION_LOCAL
        assert store.effective_pragmas == ("wal", 2, 1)
        assert store.credential_identities() == ()
    assert path.stat().st_mode & 0o777 == 0o600


def test_reopen_accepts_identity_and_rejects_trust_domain_mismatch(tmp_path: Path) -> None:
    path = (tmp_path / "attempts.db").resolve()
    open_store(path).close()
    open_store(path).close()
    with pytest.raises(AttemptConflictError):
        SQLiteCHAAttemptStore(path, "different-domain")


def test_persisted_profile_mismatch_is_rejected(tmp_path: Path) -> None:
    path = (tmp_path / "attempts.db").resolve()
    store = open_store(path)
    store.close()
    db = sqlite3.connect(path)
    tamper_metadata(db, "profile='TEST'")
    db.commit()
    db.close()
    with pytest.raises(AttemptConflictError):
        open_store(path)


def test_transaction_a_is_atomic_idempotent_and_survives_reopen(tmp_path: Path) -> None:
    path = (tmp_path / "attempts.db").resolve()
    auth = authorization()
    with open_store(path) as store:
        first = store.reserve_or_resolve_attempt_id(auth)
        assert store.reserve_or_resolve_attempt_id(auth) == first
    with open_store(path) as reopened:
        assert reopened.attempt(auth.logical_operation_id) == first


def test_transaction_a_rolls_back_both_records_on_pointer_failure(tmp_path: Path) -> None:
    path = (tmp_path / "attempts.db").resolve()
    auth = authorization()
    with open_store(path) as store:
        store._connection.execute(  # noqa: SLF001 - deterministic transaction fault
            "CREATE TRIGGER fail_current BEFORE INSERT ON current_attempts BEGIN SELECT RAISE(ABORT,'fault'); END"
        )
        with pytest.raises(AttemptConflictError):
            store.reserve_or_resolve_attempt_id(auth)
        assert store._connection.execute("SELECT count(*) FROM reservations").fetchone() == (0,)  # noqa: SLF001
        with pytest.raises(AttemptNotFoundError):
            store.attempt(auth.logical_operation_id)


def test_transaction_b_is_atomic_and_immutable_then_survives_reopen(tmp_path: Path) -> None:
    path = (tmp_path / "attempts.db").resolve()
    auth = authorization()
    with open_store(path) as store:
        reserved = store.reserve_or_resolve_attempt_id(auth)
        final = store.finalize_attempt(
            identity(auth, reserved.reservation.issuance_attempt_id), expected_fence=1
        )
        assert final.fence == 2 and final.identity is not None
        with pytest.raises(sqlite3.IntegrityError, match="immutable attempt"):
            store._connection.execute("UPDATE immutable_attempts SET digest=?", ("9" * 64,))  # noqa: SLF001
    with open_store(path) as reopened:
        assert reopened.attempt(auth.logical_operation_id) == final


def test_transaction_b_rolls_back_attempt_and_pointer_on_cas_fault(tmp_path: Path) -> None:
    path = (tmp_path / "attempts.db").resolve()
    auth = authorization()
    with open_store(path) as store:
        current = store.reserve_or_resolve_attempt_id(auth)
        with pytest.raises(AttemptConflictError):
            store.finalize_attempt(
                identity(auth, current.reservation.issuance_attempt_id), expected_fence=99
            )
        assert store._connection.execute("SELECT count(*) FROM immutable_attempts").fetchone() == (
            0,
        )  # noqa: SLF001
        assert store.attempt(auth.logical_operation_id).fence == 1


def test_replacement_preserves_predecessor_and_exactly_one_fence_wins(tmp_path: Path) -> None:
    path = (tmp_path / "attempts.db").resolve()
    auth = authorization()
    with open_store(path) as store:
        old = store.reserve_or_resolve_attempt_id(auth)
        old = store.finalize_attempt(
            identity(auth, old.reservation.issuance_attempt_id), expected_fence=1
        )
        successor = store.replace_after_authoritative_unbound(
            auth, evidence(old.reservation.issuance_attempt_id), expected_fence=2
        )
        assert successor.reservation.issuance_attempt_id != old.reservation.issuance_attempt_id
        assert store._connection.execute("SELECT count(*) FROM reservations").fetchone() == (2,)  # noqa: SLF001
        with pytest.raises(AttemptConflictError):
            store.replace_after_authoritative_unbound(
                replace(auth, requester_key_version=3),
                evidence(old.reservation.issuance_attempt_id),
                expected_fence=2,
            )


def test_two_connections_have_one_concurrent_replacement_winner(tmp_path: Path) -> None:
    path = (tmp_path / "attempts.db").resolve()
    auth = authorization()
    with open_store(path) as store:
        old = store.reserve_or_resolve_attempt_id(auth)
        old = store.finalize_attempt(
            identity(auth, old.reservation.issuance_attempt_id), expected_fence=1
        )

    def compete(version: int) -> str:
        try:
            with open_store(path) as competitor:
                competitor.replace_after_authoritative_unbound(
                    replace(auth, requester_key_version=version),
                    evidence(old.reservation.issuance_attempt_id),
                    expected_fence=2,
                )
            return "winner"
        except AttemptConflictError:
            return "conflict"

    with ThreadPoolExecutor(max_workers=2) as pool:
        outcomes = list(pool.map(compete, (2, 3)))
    assert sorted(outcomes) == ["conflict", "winner"]


def test_recovery_resolution_is_atomic_fenced_and_not_found_is_not_unbound(tmp_path: Path) -> None:
    path = (tmp_path / "attempts.db").resolve()
    auth = authorization()
    with open_store(path) as store:
        current = store.reserve_or_resolve_attempt_id(auth)
        current = store.finalize_attempt(
            identity(auth, current.reservation.issuance_attempt_id), expected_fence=1
        )
        unknown = RecoveryResolution(
            current.reservation.issuance_attempt_id,
            AttemptState.MAY_HAVE_BEEN_SENT_OUTCOME_UNKNOWN,
            "issuer:authenticated:unknown",
            "5" * 64,
        )
        resolved = store.record_recovery_resolution(
            auth.logical_operation_id, unknown, expected_fence=2
        )
        assert resolved.state is AttemptState.MAY_HAVE_BEEN_SENT_OUTCOME_UNKNOWN
        with pytest.raises(TypeError, match="exact AttemptState"):
            RecoveryResolution(  # type: ignore[arg-type]
                current.reservation.issuance_attempt_id, "NOT_FOUND", "missing", "6" * 64
            )
        replay = store.record_recovery_resolution(
            auth.logical_operation_id, unknown, expected_fence=2
        )
        assert replay == resolved


def test_conflicting_retry_cannot_create_a_second_attempt(tmp_path: Path) -> None:
    with open_store((tmp_path / "attempts.db").resolve()) as store:
        store.reserve_or_resolve_attempt_id(authorization())
        with pytest.raises(AttemptConflictError):
            store.reserve_or_resolve_attempt_id(
                authorization(account_id="acct_018f3e70-7b5c-7c21-8b9a-1123456789ab")
            )
        assert store._connection.execute("SELECT count(*) FROM reservations").fetchone() == (1,)  # noqa: SLF001


def test_unknown_schema_fails_closed_without_repair(tmp_path: Path) -> None:
    path = (tmp_path / "attempts.db").resolve()
    store = open_store(path)
    store.close()
    db = sqlite3.connect(path)
    tamper_metadata(db, "schema_version=999")
    db.commit()
    db.close()
    with pytest.raises(AttemptSchemaUnsupportedError):
        open_store(path)


def test_malformed_state_fails_closed(tmp_path: Path) -> None:
    path = (tmp_path / "attempts.db").resolve()
    auth = authorization()
    store = open_store(path)
    store.reserve_or_resolve_attempt_id(auth)
    store.close()
    db = sqlite3.connect(path)
    db.execute("UPDATE current_attempts SET state='INVALID'")
    db.commit()
    db.close()
    with pytest.raises(AttemptCorruptError):
        open_store(path)


def test_current_pointer_to_missing_record_fails_closed(tmp_path: Path) -> None:
    path = (tmp_path / "attempts.db").resolve()
    auth = authorization()
    store = open_store(path)
    store.reserve_or_resolve_attempt_id(auth)
    store.close()
    db = sqlite3.connect(path)
    db.execute("PRAGMA foreign_keys=OFF")
    db.execute("DROP TRIGGER reservations_immutable_delete")
    db.execute("DELETE FROM reservations")
    db.execute(
        "CREATE TRIGGER reservations_immutable_delete BEFORE DELETE ON reservations "
        "BEGIN SELECT RAISE(ABORT,'immutable reservation'); END"
    )
    db.commit()
    db.close()
    with pytest.raises(AttemptCorruptError):
        open_store(path)


def test_store_has_no_account_mint_or_issuer_registry_write_dependency(tmp_path: Path) -> None:
    with open_store((tmp_path / "attempts.db").resolve()) as store:
        assert not hasattr(store, "mint_account_id")
        assert not hasattr(store, "compare_and_swap_bind")
        assert store.credential_identities() == ()


def test_concrete_store_qualifies_through_composition_gate(tmp_path: Path) -> None:
    from tests.security.test_root_proof_issuer_substrate import providers

    items = providers(SecurityProfile.PRODUCTION_LOCAL, trust_domain="td-production")
    with open_store((tmp_path / "attempts.db").resolve()) as store:
        items[
            next(
                i
                for i, item in enumerate(items)
                if item.identity.role is ProviderRole.CHA_ATTEMPT_STORE
            )
        ] = store  # type: ignore[assignment]
        assert RootProofIssuerCompositionGate().qualify(store.identity.security, items).qualified


def test_replacement_exact_retry_after_reopen_uses_same_successor_and_no_duplicates(
    tmp_path: Path,
) -> None:
    path = (tmp_path / "attempts.db").resolve()
    auth = authorization()
    with open_store(path) as store:
        old = store.reserve_or_resolve_attempt_id(auth)
        old = store.finalize_attempt(
            identity(auth, old.reservation.issuance_attempt_id), expected_fence=1
        )
        committed = store.replace_after_authoritative_unbound(
            auth, evidence(old.reservation.issuance_attempt_id), expected_fence=2
        )
        counts = tuple(
            store._connection.execute(f"SELECT count(*) FROM {table}").fetchone()[0]  # noqa: S608, SLF001
            for table in ("reservations", "replacement_relations", "attempt_transitions")
        )
    with open_store(path) as reopened:
        replay = reopened.replace_after_authoritative_unbound(
            auth, evidence(old.reservation.issuance_attempt_id), expected_fence=2
        )
        assert replay.reservation.issuance_attempt_id == committed.reservation.issuance_attempt_id
        assert counts == tuple(
            reopened._connection.execute(f"SELECT count(*) FROM {table}").fetchone()[0]  # noqa: S608, SLF001
            for table in ("reservations", "replacement_relations", "attempt_transitions")
        )


def test_replacement_conflicting_retry_is_rejected(tmp_path: Path) -> None:
    auth = authorization()
    with open_store((tmp_path / "attempts.db").resolve()) as store:
        old = store.reserve_or_resolve_attempt_id(auth)
        old = store.finalize_attempt(
            identity(auth, old.reservation.issuance_attempt_id), expected_fence=1
        )
        store.replace_after_authoritative_unbound(
            auth, evidence(old.reservation.issuance_attempt_id), expected_fence=2
        )
        conflicting = replace(
            evidence(old.reservation.issuance_attempt_id),
            authority_authenticated_evidence_digest_sha256="a" * 64,
        )
        with pytest.raises(AttemptConflictError):
            store.replace_after_authoritative_unbound(auth, conflicting, expected_fence=2)


def test_recovery_exact_retry_after_reopen_has_no_duplicate_transition(tmp_path: Path) -> None:
    path = (tmp_path / "attempts.db").resolve()
    auth = authorization()
    with open_store(path) as store:
        current = store.reserve_or_resolve_attempt_id(auth)
        current = store.finalize_attempt(
            identity(auth, current.reservation.issuance_attempt_id), expected_fence=1
        )
        resolution = RecoveryResolution(
            current.reservation.issuance_attempt_id,
            AttemptState.EXACT_BOUND_RECOVERED,
            "issuer:bound:1",
            "7" * 64,
        )
        committed = store.record_recovery_resolution(
            auth.logical_operation_id, resolution, expected_fence=2
        )
        count = store._connection.execute("SELECT count(*) FROM attempt_transitions").fetchone()[0]  # noqa: SLF001
    with open_store(path) as reopened:
        assert (
            reopened.record_recovery_resolution(
                auth.logical_operation_id, resolution, expected_fence=2
            )
            == committed
        )
        assert reopened._connection.execute(
            "SELECT count(*) FROM attempt_transitions"
        ).fetchone() == (count,)  # noqa: SLF001
        with pytest.raises(AttemptConflictError):
            reopened.record_recovery_resolution(
                auth.logical_operation_id,
                replace(resolution, authenticated_digest_sha256="8" * 64),
                expected_fence=2,
            )


def test_replacement_and_transition_history_reject_update_and_delete(tmp_path: Path) -> None:
    auth = authorization()
    with open_store((tmp_path / "attempts.db").resolve()) as store:
        old = store.reserve_or_resolve_attempt_id(auth)
        old = store.finalize_attempt(
            identity(auth, old.reservation.issuance_attempt_id), expected_fence=1
        )
        store.replace_after_authoritative_unbound(
            auth, evidence(old.reservation.issuance_attempt_id), expected_fence=2
        )
        for statement in (
            "UPDATE replacement_relations SET evidence_digest='x'",
            "DELETE FROM replacement_relations",
            "UPDATE attempt_transitions SET state='x'",
            "DELETE FROM attempt_transitions",
        ):
            with pytest.raises(sqlite3.IntegrityError, match="immutable"):
                store._connection.execute(statement)  # noqa: SLF001


def test_missing_security_trigger_is_rejected_on_reopen(tmp_path: Path) -> None:
    path = (tmp_path / "attempts.db").resolve()
    store = open_store(path)
    store._connection.execute("DROP TRIGGER transitions_immutable_delete")  # noqa: SLF001
    store.close()
    with pytest.raises(AttemptCorruptError, match="schema object"):
        open_store(path)


@pytest.mark.parametrize(
    "account_id",
    [
        "018f3e70-7b5c-7c21-8b9a-0123456789ab",
        "acct_018f3e70-7b5c-4c21-8b9a-0123456789ab",
        "acct_018F3e70-7b5c-7c21-8b9a-0123456789ab",
        "acct_018f3e70-7b5c-7c21-7b9a-0123456789ab",
        "acct_not-a-uuid",
    ],
)
def test_account_id_requires_canonical_lowercase_uuid7(account_id: str) -> None:
    with pytest.raises(ValueError, match="account_id"):
        authorization(account_id=account_id)


@pytest.mark.parametrize("signature", ["AQID", "=" * 86, "a+" * 43, "a" * 87])
def test_signature_requires_canonical_unpadded_64_byte_base64url(signature: str) -> None:
    auth = authorization()
    attempt_id = "rpa_018f3e70-7b5a-7c21-8b9a-0123456789ab"
    with pytest.raises(ValueError, match="signature"):
        replace(identity(auth, attempt_id), requester_signature_base64url=signature)


def test_persisted_authorization_and_idempotency_tampering_fails_closed(tmp_path: Path) -> None:
    path = (tmp_path / "attempts.db").resolve()
    auth = authorization()
    store = open_store(path)
    current = store.reserve_or_resolve_attempt_id(auth)
    store.close()
    db = sqlite3.connect(path)
    db.execute("DROP TRIGGER reservations_immutable_update")
    db.execute(
        "UPDATE reservations SET idempotency_key=? WHERE attempt_id=?",
        ("9" * 64, current.reservation.issuance_attempt_id),
    )
    db.execute(
        "CREATE TRIGGER reservations_immutable_update BEFORE UPDATE ON reservations "
        "BEGIN SELECT RAISE(ABORT,'immutable reservation'); END"
    )
    db.commit()
    db.close()
    with pytest.raises(AttemptCorruptError, match="inconsistent"):
        open_store(path)


def test_persisted_malformed_attempt_id_fails_closed(tmp_path: Path) -> None:
    path = (tmp_path / "attempts.db").resolve()
    auth = authorization()
    store = open_store(path)
    store.reserve_or_resolve_attempt_id(auth)
    store.close()
    db = sqlite3.connect(path)
    db.execute("PRAGMA foreign_keys=OFF")
    db.execute("UPDATE current_attempts SET attempt_id='rpa_invalid'")
    db.commit()
    db.close()
    with pytest.raises(AttemptCorruptError):
        open_store(path)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("account_id", "acct_malformed"),
        ("logical_operation_id", "ago_different"),
        ("product_scope", "OtherProduct"),
    ],
)
def test_persisted_authorization_tampering_fails_closed(
    tmp_path: Path, field: str, value: str
) -> None:
    path = (tmp_path / "attempts.db").resolve()
    auth = authorization()
    store = open_store(path)
    current = store.reserve_or_resolve_attempt_id(auth)
    store.close()
    db = sqlite3.connect(path)
    raw = db.execute(
        "SELECT authorization_json FROM reservations WHERE attempt_id=?",
        (current.reservation.issuance_attempt_id,),
    ).fetchone()[0]
    payload = json.loads(bytes(raw))
    payload[field] = value
    db.execute("DROP TRIGGER reservations_immutable_update")
    db.execute(
        "UPDATE reservations SET authorization_json=? WHERE attempt_id=?",
        (
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode(),
            current.reservation.issuance_attempt_id,
        ),
    )
    db.execute(
        "CREATE TRIGGER reservations_immutable_update BEFORE UPDATE ON reservations "
        "BEGIN SELECT RAISE(ABORT,'immutable reservation'); END"
    )
    db.commit()
    db.close()
    with pytest.raises(AttemptCorruptError):
        open_store(path)


def test_persisted_malformed_signature_fails_closed(tmp_path: Path) -> None:
    path = (tmp_path / "attempts.db").resolve()
    auth = authorization()
    store = open_store(path)
    current = store.reserve_or_resolve_attempt_id(auth)
    current = store.finalize_attempt(
        identity(auth, current.reservation.issuance_attempt_id), expected_fence=1
    )
    store.close()
    db = sqlite3.connect(path)
    raw = db.execute(
        "SELECT identity_json FROM immutable_attempts WHERE attempt_id=?",
        (current.reservation.issuance_attempt_id,),
    ).fetchone()[0]
    payload = json.loads(bytes(raw))
    payload["requester_signature_base64url"] = "AQID"
    db.execute("DROP TRIGGER attempts_immutable_update")
    db.execute(
        "UPDATE immutable_attempts SET identity_json=? WHERE attempt_id=?",
        (
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode(),
            current.reservation.issuance_attempt_id,
        ),
    )
    db.execute(
        "CREATE TRIGGER attempts_immutable_update BEFORE UPDATE ON immutable_attempts "
        "BEGIN SELECT RAISE(ABORT,'immutable attempt'); END"
    )
    db.commit()
    db.close()
    with pytest.raises(AttemptCorruptError, match="immutable attempt"):
        open_store(path)


def test_unknown_then_bound_is_monotonic_append_only_and_replayable(tmp_path: Path) -> None:
    path = (tmp_path / "attempts.db").resolve()
    auth = authorization()
    with open_store(path) as store:
        current = store.reserve_or_resolve_attempt_id(auth)
        current = store.finalize_attempt(
            identity(auth, current.reservation.issuance_attempt_id), expected_fence=1
        )
        unknown = RecoveryResolution(
            current.reservation.issuance_attempt_id,
            AttemptState.MAY_HAVE_BEEN_SENT_OUTCOME_UNKNOWN,
            "issuer:unknown:1",
            "5" * 64,
        )
        unknown_result = store.record_recovery_resolution(
            auth.logical_operation_id, unknown, expected_fence=2
        )
        assert unknown_result.fence == 3
    with open_store(path) as reopened:
        unknown_replay = reopened.record_recovery_resolution(
            auth.logical_operation_id, unknown, expected_fence=2
        )
        assert unknown_replay.fence == 3
        assert reopened._connection.execute(
            "SELECT count(*) FROM recovery_resolutions"
        ).fetchone() == (1,)  # noqa: SLF001
        bound = replace(
            unknown,
            outcome=AttemptState.EXACT_BOUND_RECOVERED,
            authenticated_reference="issuer:bound:2",
            authenticated_digest_sha256="6" * 64,
        )
        bound_result = reopened.record_recovery_resolution(
            auth.logical_operation_id, bound, expected_fence=3
        )
        assert bound_result.state is AttemptState.EXACT_BOUND_RECOVERED
        assert bound_result.fence == 4
        counts = (
            reopened._connection.execute("SELECT count(*) FROM recovery_resolutions").fetchone(),  # noqa: SLF001
            reopened._connection.execute(
                "SELECT count(*) FROM attempt_transitions WHERE state IN (?,?)",
                (
                    AttemptState.MAY_HAVE_BEEN_SENT_OUTCOME_UNKNOWN.value,
                    AttemptState.EXACT_BOUND_RECOVERED.value,
                ),
            ).fetchone(),  # noqa: SLF001
        )
        assert counts == ((2,), (2,))
        assert (
            reopened.record_recovery_resolution(auth.logical_operation_id, bound, expected_fence=3)
            == bound_result
        )
        assert reopened._connection.execute(
            "SELECT count(*) FROM recovery_resolutions"
        ).fetchone() == (2,)  # noqa: SLF001
        with pytest.raises(AttemptConflictError, match="non-monotonic"):
            reopened.record_recovery_resolution(
                auth.logical_operation_id,
                replace(
                    unknown,
                    authenticated_reference="issuer:late-unknown",
                    authenticated_digest_sha256="7" * 64,
                ),
                expected_fence=4,
            )
        for statement in (
            "UPDATE recovery_resolutions SET outcome='x'",
            "DELETE FROM recovery_resolutions",
        ):
            with pytest.raises(sqlite3.IntegrityError, match="immutable recovery"):
                reopened._connection.execute(statement)  # noqa: SLF001


@pytest.mark.parametrize(
    "replacement_sql",
    [
        "CREATE TRIGGER transitions_immutable_delete BEFORE DELETE ON attempt_transitions "
        "WHEN 0 BEGIN SELECT RAISE(ABORT,'immutable transition'); END",
        "CREATE TRIGGER transitions_immutable_delete AFTER DELETE ON attempt_transitions "
        "BEGIN SELECT RAISE(ABORT,'immutable transition'); END",
    ],
)
def test_semantically_weakened_trigger_is_rejected_on_reopen(
    tmp_path: Path, replacement_sql: str
) -> None:
    path = (tmp_path / "attempts.db").resolve()
    store = open_store(path)
    store._connection.execute("DROP TRIGGER transitions_immutable_delete")  # noqa: SLF001
    store._connection.execute(replacement_sql)  # noqa: SLF001
    store.close()
    with pytest.raises(AttemptCorruptError, match="trigger is malformed"):
        open_store(path)


def test_missing_required_unique_index_is_rejected_on_reopen(tmp_path: Path) -> None:
    path = (tmp_path / "attempts.db").resolve()
    store = open_store(path)
    index_name = next(
        row[1]
        for row in store._connection.execute("PRAGMA index_list(reservations)")  # noqa: SLF001
        if tuple(
            item[2]
            for item in store._connection.execute(f"PRAGMA index_info({row[1]})")  # noqa: S608, SLF001
        )
        == ("decision_key",)
    )
    store._connection.execute("PRAGMA writable_schema=ON")  # noqa: SLF001
    store._connection.execute("DELETE FROM sqlite_master WHERE name=?", (index_name,))  # noqa: SLF001
    table_sql = store._connection.execute(  # noqa: SLF001
        "SELECT sql FROM sqlite_master WHERE type='table' AND name='reservations'"
    ).fetchone()[0]
    weakened_sql = table_sql.replace(
        "decision_key TEXT NOT NULL UNIQUE", "decision_key TEXT NOT NULL"
    )
    store._connection.execute(  # noqa: SLF001
        "UPDATE sqlite_master SET sql=? WHERE type='table' AND name='reservations'",
        (weakened_sql,),
    )
    schema_version = store._connection.execute("PRAGMA schema_version").fetchone()[0]  # noqa: SLF001
    store._connection.execute(f"PRAGMA schema_version={schema_version + 1}")  # noqa: S608, SLF001
    store._connection.execute("PRAGMA writable_schema=OFF")  # noqa: SLF001
    store._connection.commit()  # noqa: SLF001
    store.close()
    with pytest.raises(AttemptCorruptError, match="UNIQUE/PRIMARY KEY"):
        open_store(path)


@pytest.mark.parametrize(
    ("column", "value"),
    [
        ("reservation_state", AttemptState.EXACT_BOUND_RECOVERED.value),
        ("reservation_kind", "REPLACEMENT"),
        ("decision_key", "a" * 64),
    ],
)
def test_reservation_semantic_identity_tamper_is_rejected(
    tmp_path: Path, column: str, value: str
) -> None:
    path = (tmp_path / "attempts.db").resolve()
    auth = authorization()
    store = open_store(path)
    store.reserve_or_resolve_attempt_id(auth)
    store.close()
    db = sqlite3.connect(path)
    db.execute("DROP TRIGGER reservations_immutable_update")
    db.execute(f"UPDATE reservations SET {column}=?", (value,))  # noqa: S608
    db.execute(
        "CREATE TRIGGER reservations_immutable_update BEFORE UPDATE ON reservations "
        "BEGIN SELECT RAISE(ABORT,'immutable reservation'); END"
    )
    db.commit()
    db.close()
    with pytest.raises(AttemptCorruptError):
        open_store(path)


@pytest.mark.parametrize("column", ["evidence_digest", "decision_key"])
def test_replacement_history_identity_tamper_is_rejected(tmp_path: Path, column: str) -> None:
    path = (tmp_path / "attempts.db").resolve()
    auth = authorization()
    store = open_store(path)
    old = store.reserve_or_resolve_attempt_id(auth)
    old = store.finalize_attempt(
        identity(auth, old.reservation.issuance_attempt_id), expected_fence=1
    )
    successor = store.replace_after_authoritative_unbound(
        auth, evidence(old.reservation.issuance_attempt_id), expected_fence=2
    )
    store.close()
    db = sqlite3.connect(path)
    db.execute("DROP TRIGGER replacements_immutable_update")
    db.execute(f"UPDATE replacement_relations SET {column}=?", ("b" * 64,))  # noqa: S608
    db.execute(
        "CREATE TRIGGER replacements_immutable_update BEFORE UPDATE ON replacement_relations "
        "BEGIN SELECT RAISE(ABORT,'immutable replacement'); END"
    )
    db.commit()
    db.close()
    with pytest.raises(AttemptCorruptError):
        open_store(path)


def test_recovery_history_self_consistency_tamper_is_rejected(tmp_path: Path) -> None:
    path = (tmp_path / "attempts.db").resolve()
    auth = authorization()
    store = open_store(path)
    current = store.reserve_or_resolve_attempt_id(auth)
    current = store.finalize_attempt(
        identity(auth, current.reservation.issuance_attempt_id), expected_fence=1
    )
    store.record_recovery_resolution(
        auth.logical_operation_id,
        RecoveryResolution(
            current.reservation.issuance_attempt_id,
            AttemptState.MAY_HAVE_BEEN_SENT_OUTCOME_UNKNOWN,
            "issuer:unknown:tamper",
            "5" * 64,
        ),
        expected_fence=2,
    )
    store.close()
    db = sqlite3.connect(path)
    db.execute("DROP TRIGGER recoveries_immutable_update")
    db.execute(
        "UPDATE recovery_resolutions SET predecessor_state=?",
        (AttemptState.EXACT_BOUND_RECOVERED.value,),
    )
    db.execute(
        "CREATE TRIGGER recoveries_immutable_update BEFORE UPDATE ON recovery_resolutions "
        "BEGIN SELECT RAISE(ABORT,'immutable recovery'); END"
    )
    db.commit()
    db.close()
    with pytest.raises(AttemptCorruptError, match="progression"):
        open_store(path)


def test_bound_and_unsigned_attempts_cannot_be_replaced(tmp_path: Path) -> None:
    auth = authorization()
    with open_store((tmp_path / "unsigned.db").resolve()) as store:
        unsigned = store.reserve_or_resolve_attempt_id(auth)
        before = store._connection.total_changes  # noqa: SLF001
        with pytest.raises(AttemptConflictError, match="unsigned"):
            store.replace_after_authoritative_unbound(
                auth, evidence(unsigned.reservation.issuance_attempt_id), expected_fence=1
            )
        assert store._connection.total_changes == before  # noqa: SLF001

    with open_store((tmp_path / "bound.db").resolve()) as store:
        current = store.reserve_or_resolve_attempt_id(auth)
        current = store.finalize_attempt(
            identity(auth, current.reservation.issuance_attempt_id), expected_fence=1
        )
        current = store.record_recovery_resolution(
            auth.logical_operation_id,
            RecoveryResolution(
                current.reservation.issuance_attempt_id,
                AttemptState.EXACT_BOUND_RECOVERED,
                "issuer:bound:terminal",
                "a" * 64,
            ),
            expected_fence=2,
        )
        counts = tuple(
            store._connection.execute(f"SELECT count(*) FROM {table}").fetchone()[0]  # noqa: S608, SLF001
            for table in ("reservations", "replacement_relations", "attempt_transitions")
        )
        with pytest.raises(AttemptConflictError, match="BOUND"):
            store.replace_after_authoritative_unbound(
                auth, evidence(current.reservation.issuance_attempt_id), expected_fence=3
            )
        assert store.attempt(auth.logical_operation_id) == current
        assert counts == tuple(
            store._connection.execute(f"SELECT count(*) FROM {table}").fetchone()[0]  # noqa: S608, SLF001
            for table in ("reservations", "replacement_relations", "attempt_transitions")
        )


def test_historical_recovery_and_replacement_replays_return_current_descendant(
    tmp_path: Path,
) -> None:
    path = (tmp_path / "attempts.db").resolve()
    auth = authorization()
    with open_store(path) as store:
        attempt_a = store.reserve_or_resolve_attempt_id(auth)
        attempt_a = store.finalize_attempt(
            identity(auth, attempt_a.reservation.issuance_attempt_id), expected_fence=1
        )
        unknown = RecoveryResolution(
            attempt_a.reservation.issuance_attempt_id,
            AttemptState.MAY_HAVE_BEEN_SENT_OUTCOME_UNKNOWN,
            "issuer:unknown:historical",
            "b" * 64,
        )
        store.record_recovery_resolution(auth.logical_operation_id, unknown, expected_fence=2)
        attempt_b = store.replace_after_authoritative_unbound(
            auth, evidence(attempt_a.reservation.issuance_attempt_id), expected_fence=3
        )
        counts = store._connection.total_changes  # noqa: SLF001
        assert (
            store.record_recovery_resolution(auth.logical_operation_id, unknown, expected_fence=2)
            == attempt_b
        )
        assert store._connection.total_changes == counts  # noqa: SLF001

        attempt_b = store.finalize_attempt(
            identity(auth, attempt_b.reservation.issuance_attempt_id), expected_fence=4
        )
        attempt_c = store.replace_after_authoritative_unbound(
            auth, evidence(attempt_b.reservation.issuance_attempt_id), expected_fence=5
        )
        counts = store._connection.total_changes  # noqa: SLF001
        assert (
            store.replace_after_authoritative_unbound(
                auth, evidence(attempt_a.reservation.issuance_attempt_id), expected_fence=3
            )
            == attempt_c
        )
        assert store._connection.total_changes == counts  # noqa: SLF001


@pytest.mark.parametrize("mode", ["extra", "changed", "missing"])
def test_transition_history_semantics_are_validated_on_reopen(tmp_path: Path, mode: str) -> None:
    path = (tmp_path / "attempts.db").resolve()
    auth = authorization()
    store = open_store(path)
    current = store.reserve_or_resolve_attempt_id(auth)
    store.finalize_attempt(
        identity(auth, current.reservation.issuance_attempt_id), expected_fence=1
    )
    if mode == "extra":
        store._connection.execute(  # noqa: SLF001
            "INSERT INTO attempt_transitions(attempt_id,state,decision_key) VALUES(?,?,?)",
            (
                current.reservation.issuance_attempt_id,
                AttemptState.EXACT_BOUND_RECOVERED.value,
                "c" * 64,
            ),
        )
    else:
        trigger = (
            "transitions_immutable_update" if mode == "changed" else "transitions_immutable_delete"
        )
        store._connection.execute(f"DROP TRIGGER {trigger}")  # noqa: S608, SLF001
        statement = (
            "UPDATE attempt_transitions SET evidence_digest='dddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddd'"
            if mode == "changed"
            else "DELETE FROM attempt_transitions"
        )
        store._connection.execute(statement)  # noqa: SLF001
        operation = "UPDATE" if mode == "changed" else "DELETE"
        store._connection.execute(  # noqa: SLF001
            f"CREATE TRIGGER {trigger} BEFORE {operation} ON attempt_transitions "  # noqa: S608
            "BEGIN SELECT RAISE(ABORT,'immutable transition'); END"
        )
    store.close()
    with pytest.raises(AttemptCorruptError, match="transition authority history"):
        open_store(path)


@pytest.mark.parametrize(
    "sql",
    [
        "CREATE UNIQUE INDEX fake_decision_unique ON reservations(decision_key) WHERE 0",
        "CREATE UNIQUE INDEX extra_operation_unique ON reservations(operation_id)",
    ],
)
def test_partial_or_extra_unique_index_is_rejected(tmp_path: Path, sql: str) -> None:
    path = (tmp_path / "attempts.db").resolve()
    store = open_store(path)
    store._connection.execute(sql)  # noqa: SLF001
    store.close()
    with pytest.raises(AttemptCorruptError, match="UNIQUE"):
        open_store(path)


@pytest.mark.parametrize(
    "state",
    [
        AttemptState.EXACT_BOUND_RECOVERED,
        AttemptState.MAY_HAVE_BEEN_SENT_OUTCOME_UNKNOWN,
    ],
)
def test_current_state_without_immutable_decision_is_rejected(
    tmp_path: Path, state: AttemptState
) -> None:
    path = (tmp_path / "attempts.db").resolve()
    auth = authorization()
    store = open_store(path)
    current = store.reserve_or_resolve_attempt_id(auth)
    store.finalize_attempt(
        identity(auth, current.reservation.issuance_attempt_id), expected_fence=1
    )
    store.close()
    db = sqlite3.connect(path)
    db.execute("UPDATE current_attempts SET state=?", (state.value,))
    db.commit()
    db.close()
    with pytest.raises(AttemptCorruptError, match="current projection"):
        open_store(path)


def test_positive_but_wrong_fence_is_rejected(tmp_path: Path) -> None:
    path = (tmp_path / "attempts.db").resolve()
    auth = authorization()
    store = open_store(path)
    store.reserve_or_resolve_attempt_id(auth)
    store.close()
    db = sqlite3.connect(path)
    db.execute("UPDATE current_attempts SET fence=fence+100")
    db.commit()
    db.close()
    with pytest.raises(AttemptCorruptError, match="current projection"):
        open_store(path)


def test_current_pointer_rollback_from_successor_is_rejected(tmp_path: Path) -> None:
    path = (tmp_path / "attempts.db").resolve()
    auth = authorization()
    store = open_store(path)
    attempt_a = store.reserve_or_resolve_attempt_id(auth)
    attempt_a = store.finalize_attempt(
        identity(auth, attempt_a.reservation.issuance_attempt_id), expected_fence=1
    )
    store.replace_after_authoritative_unbound(
        auth, evidence(attempt_a.reservation.issuance_attempt_id), expected_fence=2
    )
    store.close()
    db = sqlite3.connect(path)
    db.execute(
        "UPDATE current_attempts SET attempt_id=?,state=?,digest_status='DEFINED',digest=?",
        (
            attempt_a.reservation.issuance_attempt_id,
            AttemptState.SIGNED_IMMUTABLE_DURABLE_NOT_SENT.value,
            attempt_a.immutable_attempt_digest_sha256,
        ),
    )
    db.commit()
    db.close()
    with pytest.raises(AttemptCorruptError, match="current projection"):
        open_store(path)


@pytest.mark.parametrize(
    "history", ["bound_replacement", "unsigned_replacement", "unsigned_recovery"]
)
def test_illegal_historical_state_machine_is_rejected(tmp_path: Path, history: str) -> None:
    path = (tmp_path / "attempts.db").resolve()
    auth = authorization()
    store = open_store(path)
    attempt_a = store.reserve_or_resolve_attempt_id(auth)
    attempt_a = store.finalize_attempt(
        identity(auth, attempt_a.reservation.issuance_attempt_id), expected_fence=1
    )
    if history == "unsigned_recovery":
        store.record_recovery_resolution(
            auth.logical_operation_id,
            RecoveryResolution(
                attempt_a.reservation.issuance_attempt_id,
                AttemptState.MAY_HAVE_BEEN_SENT_OUTCOME_UNKNOWN,
                "issuer:unknown:forged-history",
                "e" * 64,
            ),
            expected_fence=2,
        )
    else:
        store.replace_after_authoritative_unbound(
            auth, evidence(attempt_a.reservation.issuance_attempt_id), expected_fence=2
        )
        if history == "bound_replacement":
            resolution = RecoveryResolution(
                attempt_a.reservation.issuance_attempt_id,
                AttemptState.EXACT_BOUND_RECOVERED,
                "issuer:bound:forged-history",
                "f" * 64,
            )
            payload = {
                "issuance_attempt_id": resolution.issuance_attempt_id,
                "outcome": resolution.outcome.value,
                "authenticated_reference": resolution.authenticated_reference,
                "authenticated_digest_sha256": resolution.authenticated_digest_sha256,
            }
            key = cha._decision_key("RECOVERY", payload)
            store._connection.execute(  # noqa: SLF001
                "INSERT INTO recovery_resolutions(attempt_id,predecessor_state,outcome,decision_key,resolution_json) "
                "VALUES(?,?,?,?,?)",
                (
                    resolution.issuance_attempt_id,
                    AttemptState.SIGNED_IMMUTABLE_DURABLE_NOT_SENT.value,
                    resolution.outcome.value,
                    key,
                    cha._canonical(payload),
                ),
            )
            store._connection.execute(  # noqa: SLF001
                "INSERT INTO attempt_transitions(attempt_id,state,evidence_reference,evidence_digest,decision_key) "
                "VALUES(?,?,?,?,?)",
                (
                    resolution.issuance_attempt_id,
                    resolution.outcome.value,
                    resolution.authenticated_reference,
                    resolution.authenticated_digest_sha256,
                    key,
                ),
            )
            store._connection.execute("DROP TRIGGER transitions_immutable_update")  # noqa: SLF001
            store._connection.execute(
                "UPDATE attempt_transitions SET transition_id=99 WHERE transition_id=2"
            )  # noqa: SLF001
            store._connection.execute(
                "UPDATE attempt_transitions SET transition_id=2 WHERE transition_id=3"
            )  # noqa: SLF001
            store._connection.execute(
                "UPDATE attempt_transitions SET transition_id=3 WHERE transition_id=99"
            )  # noqa: SLF001
            store._connection.execute(  # noqa: SLF001
                "CREATE TRIGGER transitions_immutable_update BEFORE UPDATE ON attempt_transitions "
                "BEGIN SELECT RAISE(ABORT,'immutable transition'); END"
            )
    if history != "bound_replacement":
        store._connection.execute("DROP TRIGGER attempts_immutable_delete")  # noqa: SLF001
        store._connection.execute("DROP TRIGGER transitions_immutable_delete")  # noqa: SLF001
        store._connection.execute(  # noqa: SLF001
            "UPDATE current_attempts SET digest_status='NOT_YET_DEFINED',digest=NULL"
        )
        store._connection.execute("DELETE FROM immutable_attempts")  # noqa: SLF001
        store._connection.execute(  # noqa: SLF001
            "DELETE FROM attempt_transitions WHERE state=?",
            (AttemptState.SIGNED_IMMUTABLE_DURABLE_NOT_SENT.value,),
        )
        store._connection.execute(  # noqa: SLF001
            "CREATE TRIGGER attempts_immutable_delete BEFORE DELETE ON immutable_attempts "
            "BEGIN SELECT RAISE(ABORT,'immutable attempt'); END"
        )
        store._connection.execute(  # noqa: SLF001
            "CREATE TRIGGER transitions_immutable_delete BEFORE DELETE ON attempt_transitions "
            "BEGIN SELECT RAISE(ABORT,'immutable transition'); END"
        )
    store.close()
    with pytest.raises(AttemptCorruptError):
        open_store(path)


def test_orphan_reservation_is_rejected(tmp_path: Path) -> None:
    path = (tmp_path / "attempts.db").resolve()
    auth = authorization()
    store = open_store(path)
    store.reserve_or_resolve_attempt_id(auth)
    orphan = cha._new_rpa_id()
    orphan_auth = replace(auth, requester_key_version=99)
    store._connection.execute(  # noqa: SLF001
        "INSERT INTO reservations VALUES(?,?,?,?,?,?,?)",
        (
            orphan,
            auth.logical_operation_id,
            cha._idempotency_key(orphan_auth),
            cha._canonical(asdict(orphan_auth)),
            AttemptState.RESERVED_AWAITING_SIGNATURES.value,
            "INITIAL",
            cha._decision_key("INITIAL", asdict(orphan_auth)),
        ),
    )
    store.close()
    with pytest.raises(AttemptCorruptError):
        open_store(path)


@pytest.mark.parametrize("foreign_key", ["extra", "cascade"])
def test_exact_foreign_key_contract_is_enforced(tmp_path: Path, foreign_key: str) -> None:
    path = (tmp_path / "attempts.db").resolve()
    store = open_store(path)
    store._connection.execute("DROP TRIGGER transitions_immutable_update")  # noqa: SLF001
    store._connection.execute("DROP TRIGGER transitions_immutable_delete")  # noqa: SLF001
    store._connection.execute("ALTER TABLE attempt_transitions RENAME TO old_transitions")  # noqa: SLF001
    clause = (
        ", FOREIGN KEY(state) REFERENCES reservations(attempt_id)" if foreign_key == "extra" else ""
    )
    action = " ON DELETE CASCADE" if foreign_key == "cascade" else ""
    store._connection.execute(  # noqa: SLF001
        "CREATE TABLE attempt_transitions(transition_id INTEGER PRIMARY KEY,"
        f"attempt_id TEXT NOT NULL REFERENCES reservations(attempt_id){action},"
        "state TEXT NOT NULL,evidence_reference TEXT,evidence_digest TEXT,"
        f"decision_key TEXT NOT NULL UNIQUE{clause})"
    )
    store._connection.execute("DROP TABLE old_transitions")  # noqa: SLF001
    for operation in ("update", "delete"):
        store._connection.execute(  # noqa: SLF001
            f"CREATE TRIGGER transitions_immutable_{operation} BEFORE {operation.upper()} "
            "ON attempt_transitions BEGIN SELECT RAISE(ABORT,'immutable transition'); END"
        )
    store.close()
    with pytest.raises(AttemptCorruptError, match="foreign key contract"):
        open_store(path)


@pytest.mark.parametrize(
    "forged_state",
    [
        AttemptState.EXACT_BOUND_RECOVERED,
        AttemptState.MAY_HAVE_BEEN_SENT_OUTCOME_UNKNOWN,
    ],
)
def test_already_open_store_rejects_externally_forged_current_state(
    tmp_path: Path, forged_state: AttemptState
) -> None:
    path = (tmp_path / "attempts.db").resolve()
    auth = authorization()
    with open_store(path) as store:
        current = store.reserve_or_resolve_attempt_id(auth)
        store.finalize_attempt(
            identity(auth, current.reservation.issuance_attempt_id), expected_fence=1
        )
        external = sqlite3.connect(path)
        external.execute("UPDATE current_attempts SET state=?", (forged_state.value,))
        external.commit()
        external.close()
        with pytest.raises(AttemptCorruptError, match="current projection"):
            store.attempt(auth.logical_operation_id)


def test_already_open_store_rejects_external_fence_and_pointer_rollback(tmp_path: Path) -> None:
    auth = authorization()
    fence_path = (tmp_path / "fence.db").resolve()
    with open_store(fence_path) as store:
        store.reserve_or_resolve_attempt_id(auth)
        external = sqlite3.connect(fence_path)
        external.execute("UPDATE current_attempts SET fence=fence+100")
        external.commit()
        external.close()
        with pytest.raises(AttemptCorruptError, match="current projection"):
            store.attempt(auth.logical_operation_id)

    pointer_path = (tmp_path / "pointer.db").resolve()
    with open_store(pointer_path) as store:
        attempt_a = store.reserve_or_resolve_attempt_id(auth)
        attempt_a = store.finalize_attempt(
            identity(auth, attempt_a.reservation.issuance_attempt_id), expected_fence=1
        )
        store.replace_after_authoritative_unbound(
            auth, evidence(attempt_a.reservation.issuance_attempt_id), expected_fence=2
        )
        external = sqlite3.connect(pointer_path)
        external.execute(
            "UPDATE current_attempts SET attempt_id=?,state=?,digest_status='DEFINED',digest=?",
            (
                attempt_a.reservation.issuance_attempt_id,
                AttemptState.SIGNED_IMMUTABLE_DURABLE_NOT_SENT.value,
                attempt_a.immutable_attempt_digest_sha256,
            ),
        )
        external.commit()
        external.close()
        with pytest.raises(AttemptCorruptError, match="current projection"):
            store.attempt(auth.logical_operation_id)


def test_transaction_c_rejects_projection_tamper_before_side_effects(tmp_path: Path) -> None:
    path = (tmp_path / "attempts.db").resolve()
    auth = authorization()
    with open_store(path) as store:
        current = store.reserve_or_resolve_attempt_id(auth)
        current = store.finalize_attempt(
            identity(auth, current.reservation.issuance_attempt_id), expected_fence=1
        )
        external = sqlite3.connect(path)
        external.execute(
            "UPDATE current_attempts SET state=?", (AttemptState.EXACT_BOUND_RECOVERED.value,)
        )
        external.commit()
        external.close()
        counts = tuple(
            store._connection.execute(f"SELECT count(*) FROM {table}").fetchone()[0]  # noqa: S608, SLF001
            for table in ("reservations", "replacement_relations", "attempt_transitions")
        )
        with pytest.raises(AttemptCorruptError, match="current projection"):
            store.replace_after_authoritative_unbound(
                auth, evidence(current.reservation.issuance_attempt_id), expected_fence=2
            )
        assert counts == tuple(
            store._connection.execute(f"SELECT count(*) FROM {table}").fetchone()[0]  # noqa: S608, SLF001
            for table in ("reservations", "replacement_relations", "attempt_transitions")
        )


@pytest.mark.parametrize(
    "changes",
    [
        {"account_id": "acct_018f3e70-7b5c-7c21-8b9a-0123456789ac"},
        {"canonical_genesis_request_fingerprint_sha256": "9" * 64},
        {"initial_binding_reference": "ib:authority:other"},
        {"initial_binding_digest_sha256": "8" * 64},
    ],
)
def test_replacement_cannot_rebind_persisted_predecessor(
    tmp_path: Path, changes: dict[str, object]
) -> None:
    auth = authorization()
    replacement_auth = replace(auth, **changes)
    with open_store((tmp_path / "attempts.db").resolve()) as store:
        current = store.reserve_or_resolve_attempt_id(auth)
        current = store.finalize_attempt(
            identity(auth, current.reservation.issuance_attempt_id), expected_fence=1
        )
        before = store._connection.total_changes  # noqa: SLF001
        with pytest.raises(AttemptConflictError, match="persisted predecessor"):
            store.replace_after_authoritative_unbound(
                replacement_auth,
                evidence(current.reservation.issuance_attempt_id, replacement_auth),
                expected_fence=2,
            )
        assert store._connection.total_changes == before  # noqa: SLF001


def test_replacement_preserves_credential_rotation_with_stable_binding(tmp_path: Path) -> None:
    auth = authorization()
    rotated = replace(auth, requester_key_version=2, claimant_key_version=2)
    with open_store((tmp_path / "attempts.db").resolve()) as store:
        current = store.reserve_or_resolve_attempt_id(auth)
        current = store.finalize_attempt(
            identity(auth, current.reservation.issuance_attempt_id), expected_fence=1
        )
        successor = store.replace_after_authoritative_unbound(
            rotated,
            evidence(current.reservation.issuance_attempt_id, rotated),
            expected_fence=2,
        )
        assert successor.reservation.authorization == rotated


def test_historical_authorization_must_match_store_trust_domain(tmp_path: Path) -> None:
    path = (tmp_path / "attempts.db").resolve()
    auth = authorization()
    store = open_store(path)
    store.reserve_or_resolve_attempt_id(auth)
    store.close()
    db = sqlite3.connect(path)
    raw = db.execute("SELECT authorization_json FROM reservations").fetchone()[0]
    payload = json.loads(bytes(raw))
    payload["trust_domain"] = "different-domain"
    db.execute("DROP TRIGGER reservations_immutable_update")
    db.execute(
        "UPDATE reservations SET authorization_json=?",
        (json.dumps(payload, sort_keys=True, separators=(",", ":")).encode(),),
    )
    db.execute(
        "CREATE TRIGGER reservations_immutable_update BEFORE UPDATE ON reservations "
        "BEGIN SELECT RAISE(ABORT,'immutable reservation'); END"
    )
    db.commit()
    db.close()
    with pytest.raises(AttemptCorruptError):
        open_store(path)


def test_authorization_subclass_is_rejected_before_semantic_reads(tmp_path: Path) -> None:
    class StatefulAuthorization(AttemptAuthorization):
        reads = 0

        def __getattribute__(self, name: str) -> object:
            if name in AttemptAuthorization.__dataclass_fields__:
                type(self).reads += 1
            return super().__getattribute__(name)

    auth = authorization()
    malicious = StatefulAuthorization(**asdict(auth))
    StatefulAuthorization.reads = 0
    with open_store((tmp_path / "attempts.db").resolve()) as store:
        before = store._connection.total_changes  # noqa: SLF001
        with pytest.raises(TypeError, match="exact AttemptAuthorization"):
            store.reserve_or_resolve_attempt_id(malicious)
        assert StatefulAuthorization.reads == 0
        assert store._connection.total_changes == before  # noqa: SLF001
        assert (
            store.reserve_or_resolve_attempt_id(auth).state
            is AttemptState.RESERVED_AWAITING_SIGNATURES
        )


def test_identity_override_is_rejected_before_method_or_property_use(tmp_path: Path) -> None:
    class StatefulIdentity(AttemptIdentity):
        calls = 0

        def payload(self) -> dict[str, str | int]:
            type(self).calls += 1
            return super().payload()

        @property
        def digest_sha256(self) -> str:
            type(self).calls += 1
            return super().digest_sha256

    auth = authorization()
    with open_store((tmp_path / "attempts.db").resolve()) as store:
        current = store.reserve_or_resolve_attempt_id(auth)
        valid = identity(auth, current.reservation.issuance_attempt_id)
        malicious = StatefulIdentity(
            valid.authorization,
            valid.issuance_attempt_id,
            valid.root_proof_issuance_request_signed_payload_digest_sha256,
            valid.root_proof_issuance_request_canonical_bytes_reference,
            valid.requester_signature_base64url,
            valid.claimant_authorization_signature_base64url,
            valid.issuance_request_domain_and_profile_version,
            valid.claimant_authorization_domain_and_profile_version,
        )
        before = store._connection.total_changes  # noqa: SLF001
        with pytest.raises(TypeError, match="exact AttemptIdentity"):
            store.finalize_attempt(malicious, expected_fence=1)
        assert StatefulIdentity.calls == 0
        assert store._connection.total_changes == before  # noqa: SLF001


def test_stateful_evidence_and_recovery_subclasses_are_rejected_without_reads(
    tmp_path: Path,
) -> None:
    class StatefulEvidence(AuthoritativeUnboundEvidence):
        reads = 0

        def __getattribute__(self, name: str) -> object:
            if name in AuthoritativeUnboundEvidence.__dataclass_fields__:
                type(self).reads += 1
            return super().__getattribute__(name)

    class StatefulRecovery(RecoveryResolution):
        reads = 0

        def __getattribute__(self, name: str) -> object:
            if name in RecoveryResolution.__dataclass_fields__:
                type(self).reads += 1
            return super().__getattribute__(name)

    auth = authorization()
    with open_store((tmp_path / "attempts.db").resolve()) as store:
        current = store.reserve_or_resolve_attempt_id(auth)
        current = store.finalize_attempt(
            identity(auth, current.reservation.issuance_attempt_id), expected_fence=1
        )
        valid_evidence = evidence(current.reservation.issuance_attempt_id)
        malicious_evidence = StatefulEvidence(**asdict(valid_evidence))
        valid_recovery = RecoveryResolution(
            current.reservation.issuance_attempt_id,
            AttemptState.MAY_HAVE_BEEN_SENT_OUTCOME_UNKNOWN,
            "issuer:stateful:unknown",
            "7" * 64,
        )
        malicious_recovery = StatefulRecovery(
            valid_recovery.issuance_attempt_id,
            valid_recovery.outcome,
            valid_recovery.authenticated_reference,
            valid_recovery.authenticated_digest_sha256,
        )
        StatefulEvidence.reads = StatefulRecovery.reads = 0
        before = store._connection.total_changes  # noqa: SLF001
        with pytest.raises(TypeError, match="exact AuthoritativeUnboundEvidence"):
            store.replace_after_authoritative_unbound(auth, malicious_evidence, expected_fence=2)
        with pytest.raises(TypeError, match="exact RecoveryResolution"):
            store.record_recovery_resolution(
                auth.logical_operation_id, malicious_recovery, expected_fence=2
            )
        assert StatefulEvidence.reads == StatefulRecovery.reads == 0
        assert store._connection.total_changes == before  # noqa: SLF001
        assert store.attempt(auth.logical_operation_id) == current


@pytest.mark.parametrize("kind", ["authorization", "identity", "evidence", "recovery"])
def test_fabricated_exact_value_objects_fail_closed(tmp_path: Path, kind: str) -> None:
    path = (tmp_path / f"{kind}.db").resolve()
    auth = authorization()
    with open_store(path) as store:
        if kind == "authorization":
            fabricated = object.__new__(AttemptAuthorization)
            object.__setattr__(fabricated, "environment", 7)
            call = lambda: store.reserve_or_resolve_attempt_id(fabricated)  # type: ignore[arg-type]
        else:
            current = store.reserve_or_resolve_attempt_id(auth)
            current = store.finalize_attempt(
                identity(auth, current.reservation.issuance_attempt_id), expected_fence=1
            )
            if kind == "identity":
                fabricated = object.__new__(AttemptIdentity)
                object.__setattr__(fabricated, "authorization", "not-an-authorization")
                call = lambda: store.finalize_attempt(fabricated, expected_fence=2)  # type: ignore[arg-type]
            elif kind == "evidence":
                fabricated = object.__new__(AuthoritativeUnboundEvidence)
                object.__setattr__(fabricated, "outcome", "NOT_FOUND")
                call = lambda: store.replace_after_authoritative_unbound(  # type: ignore[arg-type]
                    auth, fabricated, expected_fence=2
                )
            else:
                fabricated = object.__new__(RecoveryResolution)
                object.__setattr__(fabricated, "outcome", "EXACT_BOUND_RECOVERED")
                call = lambda: store.record_recovery_resolution(  # type: ignore[arg-type]
                    auth.logical_operation_id, fabricated, expected_fence=2
                )
        before = store._connection.total_changes  # noqa: SLF001
        with pytest.raises(ValueError, match="malformed"):
            call()
        assert store._connection.total_changes == before  # noqa: SLF001


@pytest.mark.parametrize("bad_fence", [True, False, 1.0, "1", 0, -1])
@pytest.mark.parametrize("transaction", ["finalize", "replace", "recover"])
def test_security_sensitive_transactions_require_exact_positive_fence(
    tmp_path: Path, transaction: str, bad_fence: object
) -> None:
    path = (tmp_path / f"{transaction}-{bad_fence!r}.db").resolve()
    auth = authorization()
    with open_store(path) as store:
        current = store.reserve_or_resolve_attempt_id(auth)
        if transaction == "finalize":
            call = lambda: store.finalize_attempt(  # type: ignore[arg-type]
                identity(auth, current.reservation.issuance_attempt_id),
                expected_fence=bad_fence,
            )
        else:
            current = store.finalize_attempt(
                identity(auth, current.reservation.issuance_attempt_id), expected_fence=1
            )
            if transaction == "replace":
                call = lambda: store.replace_after_authoritative_unbound(  # type: ignore[arg-type]
                    auth,
                    evidence(current.reservation.issuance_attempt_id),
                    expected_fence=bad_fence,
                )
            else:
                resolution = RecoveryResolution(
                    current.reservation.issuance_attempt_id,
                    AttemptState.MAY_HAVE_BEEN_SENT_OUTCOME_UNKNOWN,
                    "issuer:fence:unknown",
                    "6" * 64,
                )
                call = lambda: store.record_recovery_resolution(  # type: ignore[arg-type]
                    auth.logical_operation_id, resolution, expected_fence=bad_fence
                )
        before = store._connection.total_changes  # noqa: SLF001
        error = TypeError if type(bad_fence) is not int else ValueError
        with pytest.raises(error, match="expected_fence"):
            call()
        assert store._connection.total_changes == before  # noqa: SLF001


@pytest.mark.parametrize("operation_id", [None, 7, True, "", "   "])
def test_operation_id_public_boundaries_require_exact_nonempty_string(
    tmp_path: Path, operation_id: object
) -> None:
    resolution = object.__new__(RecoveryResolution)
    with open_store((tmp_path / f"operation-{operation_id!r}.db").resolve()) as store:
        error = TypeError if type(operation_id) is not str else ValueError
        with pytest.raises(error, match="operation_id"):
            store.attempt(operation_id)  # type: ignore[arg-type]
        with pytest.raises(error, match="operation_id"):
            store.record_recovery_resolution(  # type: ignore[arg-type]
                operation_id, resolution, expected_fence=1
            )


def test_product_scope_is_required_and_changes_idempotency_and_attempt_digest() -> None:
    with pytest.raises(TypeError):
        AttemptAuthorization(
            **{
                key: value
                for key, value in asdict(authorization()).items()
                if key != "product_scope"
            }
        )
    auth_a = authorization(product_scope="ProductA")
    auth_b = replace(auth_a, product_scope="ProductB")
    assert cha._idempotency_key(auth_a) != cha._idempotency_key(auth_b)  # noqa: SLF001
    attempt_id = "rpa_018f3e70-7b5a-7c21-8b9a-0123456789ab"
    assert identity(auth_a, attempt_id).digest_sha256 != identity(auth_b, attempt_id).digest_sha256
    with pytest.raises(ValueError):
        replace(auth_a, product_scope=" ")
    with pytest.raises(TypeError):
        AuthoritativeUnboundEvidence(
            **{
                key: value
                for key, value in asdict(evidence(attempt_id)).items()
                if key != "product_scope"
            }
        )


@pytest.mark.parametrize("evidence_product", ["ProductA", "ProductB"])
def test_cross_product_replacement_has_zero_side_effects(
    tmp_path: Path, evidence_product: str
) -> None:
    auth_a = authorization(product_scope="ProductA")
    auth_b = replace(auth_a, product_scope="ProductB")
    with open_store((tmp_path / f"cross-{evidence_product}.db").resolve()) as store:
        current = store.reserve_or_resolve_attempt_id(auth_a)
        current = store.finalize_attempt(
            identity(auth_a, current.reservation.issuance_attempt_id), expected_fence=1
        )
        before = store._connection.total_changes  # noqa: SLF001
        counts = tuple(
            store._connection.execute(f"SELECT count(*) FROM {table}").fetchone()[0]  # noqa: S608, SLF001
            for table in ("reservations", "replacement_relations", "attempt_transitions")
        )
        with pytest.raises(AttemptConflictError):
            store.replace_after_authoritative_unbound(
                auth_b,
                evidence(
                    current.reservation.issuance_attempt_id,
                    auth_a if evidence_product == "ProductA" else auth_b,
                ),
                expected_fence=2,
            )
        assert store._connection.total_changes == before  # noqa: SLF001
        assert counts == tuple(
            store._connection.execute(f"SELECT count(*) FROM {table}").fetchone()[0]  # noqa: S608, SLF001
            for table in ("reservations", "replacement_relations", "attempt_transitions")
        )
        assert store.attempt(auth_a.logical_operation_id) == current


@pytest.mark.parametrize("column", ["evidence_json", "authorization_json"])
def test_persisted_replacement_product_tamper_fails_closed(tmp_path: Path, column: str) -> None:
    path = (tmp_path / f"tamper-{column}.db").resolve()
    auth = authorization(product_scope="ProductA")
    with open_store(path) as store:
        old = store.reserve_or_resolve_attempt_id(auth)
        old = store.finalize_attempt(
            identity(auth, old.reservation.issuance_attempt_id), expected_fence=1
        )
        store.replace_after_authoritative_unbound(
            auth, evidence(old.reservation.issuance_attempt_id, auth), expected_fence=2
        )
    db = sqlite3.connect(path)
    raw = db.execute(f"SELECT {column} FROM replacement_relations").fetchone()[0]  # noqa: S608
    payload = json.loads(bytes(raw))
    payload["product_scope"] = "ProductB"
    db.execute("DROP TRIGGER replacements_immutable_update")
    db.execute(
        f"UPDATE replacement_relations SET {column}=?",  # noqa: S608
        (json.dumps(payload, sort_keys=True, separators=(",", ":")).encode(),),
    )
    db.commit()
    db.close()
    with pytest.raises(AttemptCorruptError):
        open_store(path)


def test_product_scope_survives_replacement_restart_and_old_schema_fails_closed(
    tmp_path: Path,
) -> None:
    path = (tmp_path / "restart-product.db").resolve()
    auth = authorization(product_scope="CryptoHunter")
    with open_store(path) as store:
        old = store.reserve_or_resolve_attempt_id(auth)
        old = store.finalize_attempt(
            identity(auth, old.reservation.issuance_attempt_id), expected_fence=1
        )
        successor = store.replace_after_authoritative_unbound(
            auth, evidence(old.reservation.issuance_attempt_id, auth), expected_fence=2
        )
    with open_store(path) as reopened:
        restored = reopened.attempt(auth.logical_operation_id)
        assert restored == successor
        assert restored.reservation.authorization.product_scope == "CryptoHunter"
        raw = reopened._connection.execute(  # noqa: SLF001
            "SELECT evidence_json FROM replacement_relations"
        ).fetchone()[0]
        assert json.loads(bytes(raw))["product_scope"] == "CryptoHunter"
    db = sqlite3.connect(path)
    tamper_metadata(db, "schema_version=3")
    db.commit()
    db.close()
    with pytest.raises(AttemptSchemaUnsupportedError):
        open_store(path)


def test_cross_product_retry_and_immutable_identity_product_tamper_fail_closed(
    tmp_path: Path,
) -> None:
    path = (tmp_path / "identity-product.db").resolve()
    auth = authorization(product_scope="ProductA")
    with open_store(path) as store:
        reserved = store.reserve_or_resolve_attempt_id(auth)
        with pytest.raises(AttemptConflictError):
            store.reserve_or_resolve_attempt_id(replace(auth, product_scope="ProductB"))
        finalized = store.finalize_attempt(
            identity(auth, reserved.reservation.issuance_attempt_id), expected_fence=1
        )
    db = sqlite3.connect(path)
    raw = db.execute(
        "SELECT identity_json FROM immutable_attempts WHERE attempt_id=?",
        (finalized.reservation.issuance_attempt_id,),
    ).fetchone()[0]
    payload = json.loads(bytes(raw))
    payload["product_scope"] = "ProductB"
    db.execute("DROP TRIGGER attempts_immutable_update")
    db.execute(
        "UPDATE immutable_attempts SET identity_json=? WHERE attempt_id=?",
        (
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode(),
            finalized.reservation.issuance_attempt_id,
        ),
    )
    db.commit()
    db.close()
    with pytest.raises(AttemptCorruptError):
        open_store(path)
