from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import base64
from dataclasses import replace
from pathlib import Path
import json
import sqlite3

import pytest

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


def evidence(attempt_id: str, auth: AttemptAuthorization | None = None) -> AuthoritativeUnboundEvidence:
    auth = auth or authorization()
    return AuthoritativeUnboundEvidence(
        "1",
        auth.environment,
        auth.trust_domain,
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


def test_creates_dedicated_store_with_identity_permissions_and_effective_pragmas(tmp_path: Path) -> None:
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
        final = store.finalize_attempt(identity(auth, reserved.reservation.issuance_attempt_id), expected_fence=1)
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
            store.finalize_attempt(identity(auth, current.reservation.issuance_attempt_id), expected_fence=99)
        assert store._connection.execute("SELECT count(*) FROM immutable_attempts").fetchone() == (0,)  # noqa: SLF001
        assert store.attempt(auth.logical_operation_id).fence == 1


def test_replacement_preserves_predecessor_and_exactly_one_fence_wins(tmp_path: Path) -> None:
    path = (tmp_path / "attempts.db").resolve()
    auth = authorization()
    with open_store(path) as store:
        old = store.reserve_or_resolve_attempt_id(auth)
        successor = store.replace_after_authoritative_unbound(auth, evidence(old.reservation.issuance_attempt_id), expected_fence=1)
        assert successor.reservation.issuance_attempt_id != old.reservation.issuance_attempt_id
        assert store._connection.execute("SELECT count(*) FROM reservations").fetchone() == (2,)  # noqa: SLF001
        with pytest.raises(AttemptConflictError):
            store.replace_after_authoritative_unbound(
                replace(auth, requester_key_version=3), evidence(old.reservation.issuance_attempt_id), expected_fence=1
            )


def test_two_connections_have_one_concurrent_replacement_winner(tmp_path: Path) -> None:
    path = (tmp_path / "attempts.db").resolve()
    auth = authorization()
    with open_store(path) as store:
        old = store.reserve_or_resolve_attempt_id(auth)

    def compete(version: int) -> str:
        try:
            with open_store(path) as competitor:
                competitor.replace_after_authoritative_unbound(
                    replace(auth, requester_key_version=version),
                    evidence(old.reservation.issuance_attempt_id),
                    expected_fence=1,
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
        resolved = store.record_recovery_resolution(auth.logical_operation_id, unknown, expected_fence=2)
        assert resolved.state is AttemptState.MAY_HAVE_BEEN_SENT_OUTCOME_UNKNOWN
        with pytest.raises(TypeError, match="exact AttemptState"):
            RecoveryResolution(  # type: ignore[arg-type]
                current.reservation.issuance_attempt_id, "NOT_FOUND", "missing", "6" * 64
            )
        replay = store.record_recovery_resolution(auth.logical_operation_id, unknown, expected_fence=2)
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
    with open_store(path) as reopened, pytest.raises(AttemptCorruptError):
        reopened.attempt(auth.logical_operation_id)


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
    with open_store(path) as reopened, pytest.raises(AttemptCorruptError, match="target is missing"):
        reopened.attempt(auth.logical_operation_id)


def test_store_has_no_account_mint_or_issuer_registry_write_dependency(tmp_path: Path) -> None:
    with open_store((tmp_path / "attempts.db").resolve()) as store:
        assert not hasattr(store, "mint_account_id")
        assert not hasattr(store, "compare_and_swap_bind")
        assert store.credential_identities() == ()


def test_concrete_store_qualifies_through_composition_gate(tmp_path: Path) -> None:
    from tests.security.test_root_proof_issuer_substrate import providers

    items = providers(SecurityProfile.PRODUCTION_LOCAL, trust_domain="td-production")
    with open_store((tmp_path / "attempts.db").resolve()) as store:
        items[next(i for i, item in enumerate(items) if item.identity.role is ProviderRole.CHA_ATTEMPT_STORE)] = store  # type: ignore[assignment]
        assert RootProofIssuerCompositionGate().qualify(store.identity.security, items).qualified


def test_replacement_exact_retry_after_reopen_uses_same_successor_and_no_duplicates(tmp_path: Path) -> None:
    path = (tmp_path / "attempts.db").resolve()
    auth = authorization()
    with open_store(path) as store:
        old = store.reserve_or_resolve_attempt_id(auth)
        committed = store.replace_after_authoritative_unbound(
            auth, evidence(old.reservation.issuance_attempt_id), expected_fence=1
        )
        counts = tuple(
            store._connection.execute(f"SELECT count(*) FROM {table}").fetchone()[0]  # noqa: S608, SLF001
            for table in ("reservations", "replacement_relations", "attempt_transitions")
        )
    with open_store(path) as reopened:
        replay = reopened.replace_after_authoritative_unbound(
            auth, evidence(old.reservation.issuance_attempt_id), expected_fence=1
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
        store.replace_after_authoritative_unbound(
            auth, evidence(old.reservation.issuance_attempt_id), expected_fence=1
        )
        conflicting = replace(
            evidence(old.reservation.issuance_attempt_id),
            authority_authenticated_evidence_digest_sha256="a" * 64,
        )
        with pytest.raises(AttemptConflictError):
            store.replace_after_authoritative_unbound(auth, conflicting, expected_fence=1)


def test_recovery_exact_retry_after_reopen_has_no_duplicate_transition(tmp_path: Path) -> None:
    path = (tmp_path / "attempts.db").resolve()
    auth = authorization()
    with open_store(path) as store:
        current = store.reserve_or_resolve_attempt_id(auth)
        current = store.finalize_attempt(identity(auth, current.reservation.issuance_attempt_id), expected_fence=1)
        resolution = RecoveryResolution(
            current.reservation.issuance_attempt_id,
            AttemptState.EXACT_BOUND_RECOVERED,
            "issuer:bound:1",
            "7" * 64,
        )
        committed = store.record_recovery_resolution(auth.logical_operation_id, resolution, expected_fence=2)
        count = store._connection.execute("SELECT count(*) FROM attempt_transitions").fetchone()[0]  # noqa: SLF001
    with open_store(path) as reopened:
        assert reopened.record_recovery_resolution(
            auth.logical_operation_id, resolution, expected_fence=2
        ) == committed
        assert reopened._connection.execute("SELECT count(*) FROM attempt_transitions").fetchone() == (count,)  # noqa: SLF001
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
        store.replace_after_authoritative_unbound(
            auth, evidence(old.reservation.issuance_attempt_id), expected_fence=1
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
    db.execute("UPDATE reservations SET idempotency_key=? WHERE attempt_id=?", ("9" * 64, current.reservation.issuance_attempt_id))
    db.execute(
        "CREATE TRIGGER reservations_immutable_update BEFORE UPDATE ON reservations "
        "BEGIN SELECT RAISE(ABORT,'immutable reservation'); END"
    )
    db.commit()
    db.close()
    with open_store(path) as reopened, pytest.raises(AttemptCorruptError, match="inconsistent"):
        reopened.attempt(auth.logical_operation_id)


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
    with open_store(path) as reopened, pytest.raises(AttemptCorruptError):
        reopened.attempt(auth.logical_operation_id)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("account_id", "acct_malformed"),
        ("logical_operation_id", "ago_different"),
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
        (json.dumps(payload, sort_keys=True, separators=(",", ":")).encode(), current.reservation.issuance_attempt_id),
    )
    db.execute(
        "CREATE TRIGGER reservations_immutable_update BEFORE UPDATE ON reservations "
        "BEGIN SELECT RAISE(ABORT,'immutable reservation'); END"
    )
    db.commit()
    db.close()
    with open_store(path) as reopened, pytest.raises(AttemptCorruptError):
        reopened.attempt(auth.logical_operation_id)


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
        (json.dumps(payload, sort_keys=True, separators=(",", ":")).encode(), current.reservation.issuance_attempt_id),
    )
    db.execute(
        "CREATE TRIGGER attempts_immutable_update BEFORE UPDATE ON immutable_attempts "
        "BEGIN SELECT RAISE(ABORT,'immutable attempt'); END"
    )
    db.commit()
    db.close()
    with open_store(path) as reopened, pytest.raises(AttemptCorruptError, match="immutable attempt"):
        reopened.attempt(auth.logical_operation_id)


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
        assert reopened._connection.execute("SELECT count(*) FROM recovery_resolutions").fetchone() == (1,)  # noqa: SLF001
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
        assert reopened.record_recovery_resolution(
            auth.logical_operation_id, bound, expected_fence=3
        ) == bound_result
        assert reopened._connection.execute("SELECT count(*) FROM recovery_resolutions").fetchone() == (2,)  # noqa: SLF001
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
    with open_store(path) as reopened, pytest.raises(AttemptCorruptError):
        reopened.attempt(auth.logical_operation_id)


@pytest.mark.parametrize("column", ["evidence_digest", "decision_key"])
def test_replacement_history_identity_tamper_is_rejected(
    tmp_path: Path, column: str
) -> None:
    path = (tmp_path / "attempts.db").resolve()
    auth = authorization()
    store = open_store(path)
    old = store.reserve_or_resolve_attempt_id(auth)
    successor = store.replace_after_authoritative_unbound(
        auth, evidence(old.reservation.issuance_attempt_id), expected_fence=1
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
    with open_store(path) as reopened, pytest.raises(AttemptCorruptError):
        reopened.attempt(successor.reservation.authorization.logical_operation_id)


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
    with open_store(path) as reopened, pytest.raises(AttemptCorruptError, match="progression"):
        reopened.attempt(auth.logical_operation_id)
