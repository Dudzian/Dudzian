"""Reconciled Stage 9 persistence semantics; these tuples confer no authority."""

import hashlib
import inspect
import json
import sqlite3
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, replace
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

import bot_core.cha_attempt_store as cha
import bot_core.uuid7 as uuid7
from bot_core.cha_attempt_store import (
    AttemptAuthorization,
    AttemptConflictError,
    AttemptCorruptError,
    AttemptNotFoundError,
    AttemptSchemaUnsupportedError,
    AttemptState,
    AuthoritativeUnboundEvidence,
    SQLiteCHAAttemptStore,
)
from bot_core.licensing.canonical import canonical_json_bytes
from bot_core.root_proof_issuer_substrate import SecurityProfile


def authorization(**changes: object) -> AttemptAuthorization:
    values = {
        "environment": "PRODUCTION",
        "trust_domain": "td-production",
        "product_scope": "CryptoHunter",
        "logical_operation_id": "ago_018f3e70-7b5a-7c21-8b9a-0123456789ab",
        "account_id": "acct_018f3e70-7b5c-7c21-8b9a-0123456789ab",
        "canonical_genesis_request_fingerprint_sha256": "1" * 64,
        "initial_binding_reference": "initial-binding-v1:" + "2" * 64,
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
        "reservation_identity": "ibr_" + "a" * 64,
        "reservation_relation": "EXACT_OPERATION_ACCOUNT",
        "initial_binding_sha256": "3" * 64,
        "authorization_evidence_sha256": "4" * 64,
    }
    values.update(changes)
    return AttemptAuthorization(**values)


def open_store(path: Path) -> SQLiteCHAAttemptStore:
    return SQLiteCHAAttemptStore(path, "td-production")


def test_semantic_environment_is_independent_of_provider_security_profile(tmp_path: Path) -> None:
    auth = authorization()
    with open_store(tmp_path / "attempts.db") as store:
        assert store.identity.security.profile is SecurityProfile.PRODUCTION_LOCAL
        retained = store.reserve_or_resolve_attempt_id(auth)
        assert retained.reservation.authorization.environment == "PRODUCTION"
        assert retained.state is AttemptState.RESERVED_AWAITING_SIGNATURES
        assert retained.identity is None
        assert retained.immutable_attempt_digest_sha256 is None
        with pytest.raises(AttemptConflictError, match="trust_domain"):
            store.reserve_or_resolve_attempt_id(replace(auth, trust_domain="other"))
    with open_store(tmp_path / "attempts.db") as restarted:
        assert restarted.attempt(auth.logical_operation_id) == retained
        assert restarted.reserve_or_resolve_attempt_id(auth) == retained


@pytest.mark.parametrize("environment", ["PRODUCTION_LOCAL", "TEST", "production"])
def test_stage9_environment_cannot_be_replaced_by_deployment_profile(environment: str) -> None:
    with pytest.raises(ValueError, match="semantic environment"):
        authorization(environment=environment)


@pytest.mark.parametrize(
    "operation_id",
    [
        "ago_operation",
        "ago_018F3E70-7b5a-7c21-8b9a-0123456789ab",
        "ago_018f3e70-7b5a-4c21-8b9a-0123456789ab",
        "ago_018f3e70-7b5a-7c21-7b9a-0123456789ab",
        "ago_018f3e70-7b5a-7c21-8b9a-0123456789ab ",
    ],
)
def test_stage9_operation_id_requires_exact_canonical_uuid7(operation_id: str) -> None:
    with pytest.raises(ValueError, match="logical_operation_id"):
        authorization(logical_operation_id=operation_id)


@pytest.mark.parametrize(
    "field",
    [
        "reservation_identity",
        "reservation_relation",
        "initial_binding_sha256",
        "authorization_evidence_sha256",
    ],
)
def test_stage9_binding_tuple_cannot_be_partial(field: str) -> None:
    with pytest.raises(ValueError, match="present together"):
        authorization(**{field: None})


@pytest.mark.parametrize(
    "change",
    [
        {"reservation_identity": "res_018f3e70-7b5a-7c21-8b9a-0123456789ab"},
        {"reservation_identity": "ibr_" + "A" * 64},
        {"reservation_identity": "ibr_" + "a" * 63},
        {"reservation_identity": "ibr_" + "a" * 64 + " "},
        {"reservation_relation": "CALLER_SELECTED"},
        {"initial_binding_reference": "initial-binding-v1:" + "5" * 64},
        {"initial_binding_reference": "ib:authority:1"},
        {"initial_binding_reference": "initial-binding-v1:" + "2" * 64 + " "},
    ],
)
def test_stage9_relation_and_reference_require_exact_frozen_representations(
    change: dict[str, object],
) -> None:
    with pytest.raises(ValueError):
        authorization(**change)


@pytest.mark.parametrize(
    "change",
    [
        {"reservation_identity": "ibr_" + "b" * 64},
        {"initial_binding_sha256": "5" * 64},
        {"authorization_evidence_sha256": "5" * 64},
        {"canonical_genesis_request_fingerprint_sha256": "5" * 64},
        {
            "initial_binding_digest_sha256": "5" * 64,
            "initial_binding_reference": "initial-binding-v1:" + "5" * 64,
        },
        {"requester_key_version": 2},
        {"account_id": "acct_018f3e70-7b5c-7c21-8b9a-0123456789ac"},
    ],
)
def test_stage9_tuple_changes_conflict_without_second_reservation(
    tmp_path: Path, change: dict[str, object]
) -> None:
    auth = authorization()
    with open_store(tmp_path / "attempts.db") as store:
        retained = store.reserve_or_resolve_attempt_id(auth)
        with pytest.raises(AttemptConflictError, match="incompatible"):
            store.reserve_or_resolve_attempt_id(replace(auth, **change))
        assert store.attempt(auth.logical_operation_id) == retained
        assert store._connection.execute("SELECT count(*) FROM reservations").fetchone() == (1,)


def test_stage9_idempotency_uses_shared_canonical_bytes_and_domain_separation(
    tmp_path: Path,
) -> None:
    auth = authorization(provisioning_principal_id="Provisioning Żółw 😀 \u000f")
    expected = canonical_json_bytes(asdict(auth))
    with open_store(tmp_path / "attempts.db") as store:
        current = store.reserve_or_resolve_attempt_id(auth)
        raw = store._connection.execute("SELECT authorization_json FROM reservations").fetchone()[0]
        assert raw == expected
        assert "Żółw 😀".encode() in raw
        assert b"\\u000f" in raw
        assert (
            current.reservation.exact_idempotency_key
            == hashlib.sha256(
                b"CRYPTOHUNTER_CHA_ATTEMPT_RESERVATION_IDEMPOTENCY_V1\x00" + expected
            ).hexdigest()
        )
        assert current.reservation.exact_idempotency_key != hashlib.sha256(expected).hexdigest()


@pytest.mark.parametrize(
    "tamper",
    ["unknown", "missing", "whitespace", "duplicates", "text", "identity", "reference"],
)
def test_stage9_loader_rejects_unknown_missing_or_noncanonical_persisted_fields(
    tmp_path: Path, tamper: str
) -> None:
    path = tmp_path / "attempts.db"
    auth = authorization()
    with open_store(path) as store:
        store.reserve_or_resolve_attempt_id(auth)
    db = sqlite3.connect(path)
    raw = db.execute("SELECT authorization_json FROM reservations").fetchone()[0]
    payload = json.loads(raw)
    if tamper == "unknown":
        payload["caller_selected_identity"] = "forged"
        changed = canonical_json_bytes(payload)
    elif tamper == "missing":
        payload.pop("authorization_evidence_sha256")
        changed = canonical_json_bytes(payload)
    elif tamper == "whitespace":
        changed = b" " + raw
    elif tamper == "duplicates":
        changed = b'{"environment":"PRODUCTION",' + raw[1:]
    elif tamper == "identity":
        payload["reservation_identity"] = "res_caller-selected"
        changed = canonical_json_bytes(payload)
    elif tamper == "reference":
        payload["initial_binding_reference"] = "ib:caller-selected"
        changed = canonical_json_bytes(payload)
    else:
        changed = raw.decode()
    db.execute("DROP TRIGGER reservations_immutable_update")
    db.execute("UPDATE reservations SET authorization_json=?", (changed,))
    db.execute(
        "CREATE TRIGGER reservations_immutable_update BEFORE UPDATE ON reservations "
        "BEGIN SELECT RAISE(ABORT,'immutable reservation'); END"
    )
    db.commit()
    db.close()
    with pytest.raises(AttemptCorruptError, match="malformed"):
        open_store(path)


def test_schema_four_is_explicitly_unsupported_without_migration(tmp_path: Path) -> None:
    path = tmp_path / "attempts.db"
    auth = authorization()
    with open_store(path) as store:
        retained = store.reserve_or_resolve_attempt_id(auth)
        assert store._connection.execute(
            "SELECT schema_version FROM store_metadata"
        ).fetchone() == (6,)
    db = sqlite3.connect(path)
    db.execute("DROP TRIGGER store_metadata_immutable_update")
    db.execute("UPDATE store_metadata SET schema_version=4")
    db.execute(
        "CREATE TRIGGER store_metadata_immutable_update BEFORE UPDATE ON store_metadata "
        "BEGIN SELECT RAISE(ABORT,'immutable metadata'); END"
    )
    db.commit()
    db.close()
    with pytest.raises(AttemptSchemaUnsupportedError, match="unsupported"):
        open_store(path)
    db = sqlite3.connect(path)
    assert db.execute("SELECT schema_version FROM store_metadata").fetchone() == (4,)
    assert db.execute("SELECT attempt_id FROM current_attempts").fetchone() == (
        retained.reservation.issuance_attempt_id,
    )
    db.close()


def test_rpa_mint_captures_one_internal_utc_instant_and_uses_two_csprng_fields(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    instant = datetime(2026, 10, 8, 12, 34, 56, 999999, tzinfo=timezone.utc)
    clocks = []
    random_widths = []

    class Clock:
        @staticmethod
        def now(tz: object) -> datetime:
            clocks.append(tz)
            return instant

    def random_bits(width: int) -> int:
        random_widths.append(width)
        return (1 << width) - 1

    monkeypatch.setattr(cha, "datetime", Clock)
    monkeypatch.setattr(uuid7.secrets, "randbits", random_bits)
    attempt_id = cha._new_rpa_id()
    parsed = uuid.UUID(attempt_id.removeprefix("rpa_"))
    assert clocks == [timezone.utc]
    assert random_widths == [12, 62]
    assert parsed.version == 7
    assert parsed.variant == uuid.RFC_4122
    assert parsed.int >> 80 == uuid7.reservation_epoch_milliseconds(instant)
    assert (parsed.int >> 64) & ((1 << 12) - 1) == (1 << 12) - 1
    assert parsed.int & ((1 << 62) - 1) == (1 << 62) - 1
    assert str(parsed) == attempt_id.removeprefix("rpa_")
    assert not inspect.signature(cha._new_rpa_id).parameters
    assert set(
        inspect.signature(SQLiteCHAAttemptStore.reserve_or_resolve_attempt_id).parameters
    ) == {
        "self",
        "auth",
    }


@pytest.mark.parametrize("millis", [0, (1 << 48) - 1])
def test_shared_rpa_uuid7_supports_entire_uint48_range(millis: int) -> None:
    parsed = uuid.UUID(uuid7.mint_uuid7("rpa_", millis).removeprefix("rpa_"))
    assert parsed.int >> 80 == millis
    assert parsed.version == 7
    assert parsed.variant == uuid.RFC_4122


@pytest.mark.parametrize("millis", [-1, 1 << 48, True, 1.5])
def test_shared_rpa_uuid7_rejects_out_of_range_without_masking(millis: object) -> None:
    with pytest.raises(uuid7.UUID7Error):
        uuid7.mint_uuid7("rpa_", millis)


@pytest.mark.parametrize(
    ("instant", "expected_millis"),
    [
        (datetime(1970, 1, 1, 0, 0, 0, 999999, tzinfo=timezone.utc), 999),
        (datetime.max.replace(tzinfo=timezone.utc), 253_402_300_799_999),
    ],
)
def test_rpa_internal_timestamp_floors_milliseconds_without_float_rounding(
    monkeypatch: pytest.MonkeyPatch, instant: datetime, expected_millis: int
) -> None:
    class Clock:
        @staticmethod
        def now(tz: object) -> datetime:
            return instant

    monkeypatch.setattr(cha, "datetime", Clock)
    parsed = uuid.UUID(cha._new_rpa_id().removeprefix("rpa_"))
    assert parsed.int >> 80 == expected_millis


def test_invalid_internal_clock_rolls_back_and_publishes_no_attempt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class Clock:
        @staticmethod
        def now(tz: object) -> datetime:
            return datetime(1970, 1, 1, tzinfo=timezone.utc) - timedelta(microseconds=1)

    auth = authorization()
    with open_store(tmp_path / "attempts.db") as store:
        with monkeypatch.context() as patch:
            patch.setattr(cha, "datetime", Clock)
            with pytest.raises(uuid7.UUID7Error):
                store.reserve_or_resolve_attempt_id(auth)
        assert not store._connection.in_transaction
        assert store._connection.execute("SELECT count(*) FROM reservations").fetchone() == (0,)
        with pytest.raises(AttemptNotFoundError):
            store.attempt(auth.logical_operation_id)
        assert (
            store.reserve_or_resolve_attempt_id(auth).state
            is AttemptState.RESERVED_AWAITING_SIGNATURES
        )


def test_stage9_pointer_failure_rolls_back_reservation_and_retry_can_succeed(
    tmp_path: Path,
) -> None:
    auth = authorization()
    with open_store(tmp_path / "attempts.db") as store:
        store._connection.execute(
            "CREATE TRIGGER fail_current BEFORE INSERT ON current_attempts "
            "BEGIN SELECT RAISE(ABORT,'fault'); END"
        )
        with pytest.raises(AttemptConflictError):
            store.reserve_or_resolve_attempt_id(auth)
        assert store._connection.execute("SELECT count(*) FROM reservations").fetchone() == (0,)
        assert store._connection.execute("SELECT count(*) FROM current_attempts").fetchone() == (0,)
        store._connection.execute("DROP TRIGGER fail_current")
        assert store.reserve_or_resolve_attempt_id(auth).fence == 1


def test_stage9_concurrent_reservation_retains_one_attempt_without_mutex(tmp_path: Path) -> None:
    path = tmp_path / "attempts.db"
    auth = authorization()
    with open_store(path):
        pass

    def reserve(_: int) -> str:
        with open_store(path) as competitor:
            return competitor.reserve_or_resolve_attempt_id(auth).reservation.issuance_attempt_id

    with ThreadPoolExecutor(max_workers=2) as pool:
        outcomes = list(pool.map(reserve, (1, 2)))
    assert outcomes[0] == outcomes[1]
    with open_store(path) as store:
        assert store._connection.execute("SELECT count(*) FROM reservations").fetchone() == (1,)
        assert store._connection.execute("SELECT count(*) FROM current_attempts").fetchone() == (1,)


def test_stage9_lost_response_restart_and_load_never_mint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "attempts.db"
    auth = authorization()
    with open_store(path) as store:
        retained = store.reserve_or_resolve_attempt_id(auth)

    def no_mint() -> str:
        pytest.fail("read, restart, or retry tried to mint an attempt")

    monkeypatch.setattr(cha, "_new_rpa_id", no_mint)
    with open_store(path) as store:
        assert store.attempt(auth.logical_operation_id) == retained
        assert store.reserve_or_resolve_attempt_id(auth) == retained


def test_legacy_reconciliation_evidence_cannot_replace_stage9_reservation(tmp_path: Path) -> None:
    auth = authorization()
    with open_store(tmp_path / "attempts.db") as store:
        retained = store.reserve_or_resolve_attempt_id(auth)
        evidence = AuthoritativeUnboundEvidence(
            schema_version="2",
            environment=auth.environment,
            trust_domain=auth.trust_domain,
            product_scope=auth.product_scope,
            issuer_authority_identity="issuer-authority-local",
            issuer_registry_identity="issuer-registry-local",
            bootstrap_entitlement_id=auth.bootstrap_entitlement_id,
            entitlement_generation=auth.entitlement_generation,
            logical_operation_id=auth.logical_operation_id,
            account_id=auth.account_id,
            canonical_genesis_request_fingerprint_sha256=auth.canonical_genesis_request_fingerprint_sha256,
            old_issuance_attempt_id=retained.reservation.issuance_attempt_id,
            initial_binding_reference=auth.initial_binding_reference,
            initial_binding_digest_sha256=auth.initial_binding_digest_sha256,
            outcome="AUTHORITATIVELY_UNBOUND",
            authoritative_state_identity="registry-state-1",
            authoritative_state_revision="revision-1",
            authority_authenticated_evidence_reference="issuer:evidence:1",
            authority_authenticated_evidence_digest_sha256="5" * 64,
            verification_profile_version="ROOT_PROOF_RECONCILIATION_V1",
        )
        before = store._connection.total_changes
        with pytest.raises(AttemptConflictError, match="persisted predecessor"):
            store.replace_after_authoritative_unbound(auth, evidence, expected_fence=1)
        assert store._connection.total_changes == before
        assert store.attempt(auth.logical_operation_id) == retained
        assert store._connection.execute("SELECT count(*) FROM reservations").fetchone() == (1,)
