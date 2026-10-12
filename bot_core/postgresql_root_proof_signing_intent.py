"""Wewnętrzna, nieautoryzująca rezerwacja Root Proof przed podpisaniem.

Poświadczenia zapisu należą wyłącznie do zaufanego issuera, nigdy do runtime
CHA. SQL nie dowodzi wykonania weryfikacji Python. Brak kwalifikowanej izolacji
tego autorytetu we wdrożeniu pozostawia publiczną orkiestrację zablokowaną.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Any, Literal, NoReturn

import psycopg
from psycopg import sql
from psycopg.errors import DataCorrupted, UniqueViolation

from bot_core.authenticated_issuer_history import ReconciliationOutcome
from bot_core.cha_attempt_signatures import RetainedIssuanceRequest
from bot_core.cha_attempt_store import AttemptAuthorization
from bot_core.cha_issuance_request import IssuanceSignerIdentity
from bot_core.licensing.canonical import parse_canonical
from bot_core.licensing.cha_account_reservation import VerifiedAccountGenesisInitialBinding
from bot_core.licensing.cha_root_proof_signed_attempt import (
    _read_issuer_preflight_source,
    _signed_attempt_lock_selector,
)
from bot_core.local_signing_custody import (
    LocalRootProofSigningProvider,
    SigningCredentialSnapshot,
    SigningKeyLifecycle,
)
from bot_core.postgresql_entitlement_registry import PostgreSQLConnectionConfig
from bot_core.postgresql_root_proof_issuance_authority import PostgreSQLRootProofIssuanceAuthority
from bot_core.postgresql_root_proof_retention import (
    ROOT_PROOF_SIGNING_DOMAIN,
    SIGNED_PAYLOAD_FIELDS,
    RootProofRetentionScope,
    _canonical,
    _exact_inputs,
    _historical_verification_unavailable,
    _request_data,
    _scope_data,
    _snapshot_data,
    _snapshot_from_data,
    _validate_snapshot,
)
from bot_core.postgresql_root_proof_signing_intent_schema import qualify_signing_intent_schema
from bot_core.root_proof_issuance_reconciliation import _failure
from bot_core.root_proof_issuer_runtime import RootProofIssuerRuntimeError, _verify_request
from bot_core.uuid7 import mint_uuid7, reservation_epoch_milliseconds

_STORAGE_SCHEMA = "CryptoHunter.Stage9.RootProofSigningIntentV1"
_IDENTIFIER = re.compile(r"[a-z_][a-z0-9_]{0,62}\Z")
_ROOT_ID = re.compile(r"rpf_[0-9a-f]{8}-[0-9a-f]{4}-7[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}\Z")


class RootProofSigningIntentError(RuntimeError):
    """Odmowa bez ujawniania DSN ani sekretów."""


class RootProofSigningIntentConflict(RootProofSigningIntentError):
    """Niezgodna tożsamość żądania, zakresu, credentials lub lifecycle."""


class RootProofSigningIntentCorrupt(RootProofSigningIntentError):
    """Niespójne zachowane bajty lub dowody autorytetu."""


class RootProofSigningIntentUnavailable(RootProofSigningIntentError):
    """Nie można przeprowadzić kwalifikowanego odczytu."""


class RootProofSigningIntentIndeterminate(RootProofSigningIntentError):
    """Zapis mógł zostać zatwierdzony; wymagany nowy autorytatywny odczyt."""


class RootProofSigningIntentNotIssued(RootProofSigningIntentError):
    """Rezerwacja nie jest podpisem, STAGED ani emisją."""


def _persistence_cut(name: str) -> None:
    """Punkt testowego przerwania przed i po COMMIT."""


def _new_root_proof_id() -> str:
    return mint_uuid7("rpf_", reservation_epoch_milliseconds(datetime.now(timezone.utc)))


def _payload(
    scope: RootProofRetentionScope,
    auth: AttemptAuthorization,
    request: RetainedIssuanceRequest,
    snapshot: SigningCredentialSnapshot,
    root_id: str,
) -> bytes:
    if type(root_id) is not str or _ROOT_ID.fullmatch(root_id) is None:
        raise ValueError("ISSUER_UUID7_REQUIRED")
    fields = {
        "schema_version": "1",
        "environment": scope.environment,
        "trust_domain": scope.trust_domain,
        "issuer_id": scope.issuer_id,
        "issuer_key_id": snapshot.credential_identity.credential_identity,
        "issuer_key_version": snapshot.key_version,
        "root_proof_id": root_id,
        "intended_action": "ACCOUNT_GENESIS_BOOTSTRAP",
        "product_scope": scope.product_scope,
        "account_id": auth.account_id,
        "logical_operation_id": auth.logical_operation_id,
        "canonical_genesis_request_fingerprint_sha256": (
            auth.canonical_genesis_request_fingerprint_sha256
        ),
        "bootstrap_entitlement_id": auth.bootstrap_entitlement_id,
        "entitlement_generation": auth.entitlement_generation,
        "authorization_constraints": {"max_logical_operations": 1, "max_account_ids": 1},
        "issuance_attempt_id": request.attempt_id,
    }
    if set(fields) != set(SIGNED_PAYLOAD_FIELDS):
        raise ValueError("FROZEN_SIGNED_PAYLOAD_FIELDS_REQUIRED")
    if any(
        checkpoint is None or checkpoint[0].key_material_identity == snapshot.key_material_identity
        for checkpoint in (request.requester, request.claimant)
    ):
        raise ValueError("ISSUER_REQUESTER_CLAIMANT_KEY_ALIAS_FORBIDDEN")
    return _canonical(fields)


def _envelope(
    scope: RootProofRetentionScope,
    auth: AttemptAuthorization,
    request: RetainedIssuanceRequest,
    snapshot: SigningCredentialSnapshot,
    root_id: str,
    payload: bytes,
) -> bytes:
    return _canonical(
        {
            "schema_version": "1",
            "storage_schema": _STORAGE_SCHEMA,
            "state": "RESERVED",
            "scope": _scope_data(scope),
            "authorization": asdict(auth),
            "retained_request": _request_data(request),
            "credential_snapshot": _snapshot_data(snapshot),
            "signed_payload_fields": list(SIGNED_PAYLOAD_FIELDS),
            "signing_domain_hex": ROOT_PROOF_SIGNING_DOMAIN.hex(),
            "canonical_payload_sha256": hashlib.sha256(payload).hexdigest(),
            "root_proof_id": root_id,
            "issuance_attempt_id": request.attempt_id,
            "logical_operation_id": auth.logical_operation_id,
            "account_id": auth.account_id,
            "bootstrap_entitlement_id": auth.bootstrap_entitlement_id,
            "entitlement_generation": auth.entitlement_generation,
        }
    )


def _stored_request(value: Any) -> RetainedIssuanceRequest:
    def checkpoint(name: str) -> tuple[IssuanceSignerIdentity, str]:
        item = value[name]
        return (
            IssuanceSignerIdentity.from_bytes(_canonical(item["identity"])),
            item["signature_base64url"],
        )

    return RetainedIssuanceRequest(
        value["attempt_id"],
        bytes.fromhex(value["canonical_bytes_hex"]),
        value["reference"],
        value["digest"],
        checkpoint("requester"),
        checkpoint("claimant"),
    )


@dataclass(frozen=True, slots=True)
class RootProofSigningIntent:
    """Odłączony, niezmienny wynik rezerwacji; nie jest capability."""

    scope: RootProofRetentionScope
    authorization: AttemptAuthorization
    retained_request: RetainedIssuanceRequest
    canonical_envelope: bytes
    canonical_payload: bytes
    credential_snapshot: SigningCredentialSnapshot
    root_proof_id: str
    state: Literal["RESERVED"] = "RESERVED"
    authorizes_operations: Literal[False] = False

    def __post_init__(self) -> None:
        if self.state != "RESERVED" or self.authorizes_operations is not False:
            raise TypeError("RESERVATION_NEVER_AUTHORIZES_OPERATIONS")

    @property
    def bytes_to_sign(self) -> bytes:
        """Dokładna zamrożona domena z separatorem NUL i przyszły payload."""
        return ROOT_PROOF_SIGNING_DOMAIN + self.canonical_payload


class PostgreSQLRootProofSigningIntent:
    """Port wyłącznie zaufanej kompozycji niezależnego issuera.

    Caller wybiera tylko genuine binding i genuine signed attempt. Połączenie
    issuer_writer musi być niedostępne dla niezaufanego runtime. Nie istnieje
    jeszcze kwalifikowana produkcyjna kompozycja tego portu.
    """

    __slots__ = ("__connection", "__schema", "__scope", "__signer", "__authority")
    __connection: PostgreSQLConnectionConfig
    __schema: str
    __scope: RootProofRetentionScope
    __signer: LocalRootProofSigningProvider
    __authority: PostgreSQLRootProofIssuanceAuthority

    def __init__(
        self,
        connection: PostgreSQLConnectionConfig,
        *,
        schema: str,
        scope: RootProofRetentionScope,
        verification_authority: LocalRootProofSigningProvider,
        issuer_authority: PostgreSQLRootProofIssuanceAuthority,
    ) -> None:
        if (
            type(connection) is not PostgreSQLConnectionConfig
            or type(schema) is not str
            or _IDENTIFIER.fullmatch(schema) is None
            or type(scope) is not RootProofRetentionScope
            or type(verification_authority) is not LocalRootProofSigningProvider
            or type(issuer_authority) is not PostgreSQLRootProofIssuanceAuthority
        ):
            raise TypeError("EXACT_INTERNAL_SIGNING_INTENT_COMPOSITION_REQUIRED")
        for name, value in (
            ("connection", PostgreSQLConnectionConfig(connection.dsn)),
            ("schema", schema),
            ("scope", RootProofRetentionScope(**asdict(scope))),
            ("signer", verification_authority),
            ("authority", issuer_authority),
        ):
            object.__setattr__(self, "_PostgreSQLRootProofSigningIntent__" + name, value)
        try:
            _validate_snapshot(self.__signer.credential_snapshot(), self.__scope)
            subject = issuer_authority._entitlement_subject
            if (
                subject.environment != scope.environment
                or subject.trust_domain != scope.trust_domain
            ):
                raise ValueError("PINNED_ISSUER_SCOPE_REQUIRED")
            issuer_authority.requalify()
            with self._connect() as conn, conn.transaction():
                self._qualify(conn)
        except Exception:
            raise RootProofSigningIntentUnavailable(
                "SIGNING_INTENT_COMPOSITION_UNAVAILABLE"
            ) from None

    def __init_subclass__(cls) -> None:
        raise TypeError("SIGNING_INTENT_PROVIDER_CANNOT_BE_SUBCLASSED")

    def __setattr__(self, name: str, value: object) -> None:
        raise TypeError("SIGNING_INTENT_COMPOSITION_IS_IMMUTABLE")

    def __repr__(self) -> str:
        return f"PostgreSQLRootProofSigningIntent(connection=<redacted>, schema={self.__schema!r})"

    def _connect(self) -> psycopg.Connection[Any]:
        conn = psycopg.connect(self.__connection.dsn, autocommit=True)
        try:
            conn.execute("SET search_path=pg_catalog")
        except Exception:
            conn.close()
            raise
        return conn

    def _qualify(self, conn: psycopg.Connection[Any]) -> None:
        qualify_signing_intent_schema(conn, self.__schema, self.__scope.trust_domain)

    def _selector(self, binding: object, signed_attempt: object) -> str:
        try:
            if type(binding) is not VerifiedAccountGenesisInitialBinding:
                raise TypeError("GENUINE_INITIAL_BINDING_REQUIRED")
            return _signed_attempt_lock_selector(signed_attempt)
        except Exception:
            raise RootProofSigningIntentConflict("GENUINE_LOCK_SELECTOR_REQUIRED") from None

    def _lock_attempt(self, conn: psycopg.Connection[Any], selector: str) -> None:
        # Selector wybiera wyłącznie mutex. Nie jest dowodem tożsamości, braku
        # rekordu ani uprawnieniem; pełna weryfikacja następuje pod blokadą.
        # Kolizja skrótu jedynie serializuje dodatkowe próby. Constraints SQL
        # pozostają autorytatywnym fencingiem innych tożsamości.
        lock_key = int.from_bytes(
            hashlib.sha256(b"RootProofSigningIntentV1\x00" + selector.encode()).digest()[:8],
            "big",
            signed=True,
        )
        conn.execute("SELECT pg_catalog.pg_advisory_xact_lock(%s)", (lock_key,))

    def _inputs(
        self, binding: object, signed_attempt: object
    ) -> tuple[AttemptAuthorization, RetainedIssuanceRequest]:
        try:
            if type(binding) is not VerifiedAccountGenesisInitialBinding:
                raise RootProofSigningIntentConflict("GENUINE_INITIAL_BINDING_REQUIRED")
            # Źródło ponownie weryfikuje własny binding przez autorytatywny
            # _exact_binding_authorization, a później wymaga tej samej tożsamości
            # obiektu. Sam exact type nie stanowi dowodu autentyczności.
            source, current, retained = _read_issuer_preflight_source(signed_attempt)
            if source.binding is not binding or source.provider is not self.__authority:
                raise RootProofSigningIntentConflict("EXACT_PINNED_BINDING_AND_AUTHORITY_REQUIRED")
            _verify_request(source, current, retained)
            if (
                source.authorization.environment != self.__scope.environment
                or source.authorization.trust_domain != self.__scope.trust_domain
                or source.authorization.product_scope != self.__scope.product_scope
            ):
                raise RootProofSigningIntentConflict("EXACT_SIGNING_INTENT_SCOPE_REQUIRED")
            return _exact_inputs(self.__scope, source.authorization, retained)
        except RootProofSigningIntentError:
            raise
        except RootProofIssuerRuntimeError as error:
            # Wyłącznie te jawne warunki bezpośrednio odczytanego _verify_request
            # dowodzą konfliktu. Ogólna odmowa portu nie dowodzi jego przyczyny.
            if str(error) in {
                "EXACT_PUBLIC_ISSUER_AUTHORITY_REQUIRED",
                "EXACT_CURRENT_SIGNED_ATTEMPT_REQUIRED",
                "EXACT_IMMUTABLE_REQUEST_REQUIRED",
                "EXACT_IMMUTABLE_ATTEMPT_IDENTITY_REQUIRED",
                "EXACT_ACTIVE_UNBOUND_ENTITLEMENT_REQUIRED",
                "EXACT_ACTIVE_PUBLIC_SIGNER_REQUIRED",
                "REQUESTER_CLAIMANT_ALIAS",
            }:
                raise RootProofSigningIntentConflict("LIVE_ATTEMPT_IDENTITY_CONFLICT") from None
            raise RootProofSigningIntentIndeterminate(
                "LIVE_ATTEMPT_REVERIFICATION_REFUSED"
            ) from None
        except Exception as error:
            outcome = _failure("SIGNING_INTENT_INPUT", error).outcome
            error_type = {
                ReconciliationOutcome.CONFLICT: RootProofSigningIntentConflict,
                ReconciliationOutcome.CORRUPT: RootProofSigningIntentCorrupt,
                ReconciliationOutcome.INDETERMINATE: RootProofSigningIntentIndeterminate,
            }.get(outcome, RootProofSigningIntentUnavailable)
            raise error_type("GENUINE_SIGNED_ATTEMPT_REVERIFICATION_REFUSED") from None

    def _snapshot(self) -> SigningCredentialSnapshot:
        try:
            snapshot = self.__signer.credential_snapshot()
            _validate_snapshot(snapshot, self.__scope)
        except Exception as error:
            if _historical_verification_unavailable(error):
                raise RootProofSigningIntentUnavailable("PINNED_SIGNER_UNAVAILABLE") from None
            raise RootProofSigningIntentCorrupt("PINNED_SIGNER_IDENTITY_REJECTED") from None
        if snapshot.lifecycle_state is not SigningKeyLifecycle.ACTIVE:
            raise RootProofSigningIntentConflict("SIGNING_INTENT_REQUIRES_EXACT_ACTIVE_SIGNER")
        return snapshot

    def _read_row(self, conn: psycopg.Connection[Any], request: RetainedIssuanceRequest) -> object:
        rows = conn.execute(
            sql.SQL("SELECT envelope,payload FROM {}.read_intent(%s,%s)").format(
                sql.Identifier(self.__schema)
            ),
            (_canonical(_scope_data(self.__scope)), _canonical(request.attempt_id)),
        ).fetchall()
        if not rows:
            return None
        if len(rows) != 1:
            raise RootProofSigningIntentCorrupt("SIGNING_INTENT_CARDINALITY_INVALID")
        return rows[0]

    def _restore(
        self, row: object, auth: AttemptAuthorization, request: RetainedIssuanceRequest
    ) -> RootProofSigningIntent:
        try:
            if type(row) is not tuple or len(row) != 2 or any(type(v) is not bytes for v in row):
                raise ValueError("EXACT_RETAINED_BYTEA_REQUIRED")
            envelope, payload = row
            document = parse_canonical(envelope)
            scope = RootProofRetentionScope(
                **{**document["scope"], "security_profile": self.__scope.security_profile}
            )
            stored_auth, stored_request = _exact_inputs(
                scope,
                AttemptAuthorization(**document["authorization"]),
                _stored_request(document["retained_request"]),
            )
            snapshot = _snapshot_from_data(document["credential_snapshot"])
            _validate_snapshot(snapshot, scope)
            if snapshot.lifecycle_state is not SigningKeyLifecycle.ACTIVE:
                raise ValueError("ORIGINAL_ACTIVE_SNAPSHOT_REQUIRED")
            root_id = document["root_proof_id"]
            if payload != _payload(
                scope, stored_auth, stored_request, snapshot, root_id
            ) or envelope != _envelope(
                scope, stored_auth, stored_request, snapshot, root_id, payload
            ):
                raise ValueError("IMMUTABLE_SIGNING_INTENT_BYTES_MISMATCH")
        except Exception:
            raise RootProofSigningIntentCorrupt("RETAINED_SIGNING_INTENT_REJECTED") from None
        if scope != self.__scope or stored_auth != auth or stored_request != request:
            raise RootProofSigningIntentConflict("EXACT_RETRY_INPUT_IDENTITY_REQUIRED")
        if self._snapshot() != snapshot:
            raise RootProofSigningIntentConflict("EXACT_RETRY_PINNED_CREDENTIAL_REQUIRED")
        return RootProofSigningIntent(scope, auth, request, envelope, payload, snapshot, root_id)

    def read_intent(self, binding: object, signed_attempt: object) -> RootProofSigningIntent | None:
        """Autorytatywny odczyt rezerwacji; None nie dowodzi braku wcześniejszej emisji."""
        selector = self._selector(binding, signed_attempt)
        try:
            with self._connect() as conn, conn.transaction():
                self._qualify(conn)
                self._lock_attempt(conn, selector)
                auth, request = self._inputs(binding, signed_attempt)
                if request.attempt_id != selector:
                    raise RootProofSigningIntentConflict("VERIFIED_ATTEMPT_LOCK_SELECTOR_MISMATCH")
                row = self._read_row(conn, request)
                intent = None if row is None else self._restore(row, auth, request)
                if self._inputs(binding, signed_attempt) != (auth, request):
                    raise RootProofSigningIntentConflict("LIVE_ATTEMPT_CHANGED_DURING_READ")
                return intent
        except RootProofSigningIntentError:
            raise
        except DataCorrupted:
            raise RootProofSigningIntentCorrupt("RETAINED_SIGNING_INTENT_KEYS_REJECTED") from None
        except Exception:
            raise RootProofSigningIntentUnavailable("SIGNING_INTENT_READ_UNAVAILABLE") from None

    def reserve(self, binding: object, signed_attempt: object) -> RootProofSigningIntent:
        """Rezerwuj wewnątrz transakcji issuera; żaden argument nie proponuje rpf_."""
        selector = self._selector(binding, signed_attempt)
        write_attempted = False
        try:
            with self._connect() as conn, conn.transaction():
                self._qualify(conn)
                self._lock_attempt(conn, selector)
                auth, request = self._inputs(binding, signed_attempt)
                if request.attempt_id != selector:
                    raise RootProofSigningIntentConflict("VERIFIED_ATTEMPT_LOCK_SELECTOR_MISMATCH")
                if self._inputs(binding, signed_attempt) != (auth, request):
                    raise RootProofSigningIntentConflict("LIVE_ATTEMPT_CHANGED_BEFORE_RESERVATION")
                row = self._read_row(conn, request)
                if row is not None:
                    retained = self._restore(row, auth, request)
                    if self._inputs(binding, signed_attempt) != (auth, request):
                        raise RootProofSigningIntentConflict("LIVE_ATTEMPT_CHANGED_DURING_RETRY")
                    return retained
                snapshot = self._snapshot()
                root_id = _new_root_proof_id()
                payload = _payload(self.__scope, auth, request, snapshot, root_id)
                envelope = _envelope(self.__scope, auth, request, snapshot, root_id, payload)
                write_attempted = True
                rows = conn.execute(
                    sql.SQL(
                        "SELECT envelope,payload FROM {}.reserve_intent(%s,%s,%s,%s,%s,%s,%s)"
                    ).format(sql.Identifier(self.__schema)),
                    (
                        _canonical(_scope_data(self.__scope)),
                        _canonical(request.attempt_id),
                        _canonical(auth.logical_operation_id),
                        _canonical(
                            {
                                "bootstrap_entitlement_id": auth.bootstrap_entitlement_id,
                                "entitlement_generation": auth.entitlement_generation,
                            }
                        ),
                        _canonical(root_id),
                        envelope,
                        payload,
                    ),
                ).fetchall()
                if len(rows) != 1:
                    raise RootProofSigningIntentCorrupt("SIGNING_INTENT_WRITE_CARDINALITY_INVALID")
                intent = self._restore(rows[0], auth, request)
                if (
                    self._inputs(binding, signed_attempt) != (auth, request)
                    or self._snapshot() != snapshot
                ):
                    raise RootProofSigningIntentConflict("LIVE_IDENTITIES_CHANGED_BEFORE_COMMIT")
                _persistence_cut("BEFORE_COMMIT")
            _persistence_cut("AFTER_COMMIT")
            return intent
        except RootProofSigningIntentError:
            raise
        except DataCorrupted:
            raise RootProofSigningIntentCorrupt("RETAINED_SIGNING_INTENT_KEYS_REJECTED") from None
        except UniqueViolation:
            raise RootProofSigningIntentConflict(
                "DURABLE_SIGNING_INTENT_IDENTITY_CONFLICT"
            ) from None
        except Exception:
            if write_attempted:
                raise RootProofSigningIntentIndeterminate(
                    "SIGNING_INTENT_WRITE_OUTCOME_UNKNOWN"
                ) from None
            raise RootProofSigningIntentUnavailable("SIGNING_INTENT_WRITE_UNAVAILABLE") from None

    def require_issued(self, *args: object, **kwargs: object) -> NoReturn:
        raise RootProofSigningIntentNotIssued("RESERVED_INTENT_IS_NOT_ISSUED")
