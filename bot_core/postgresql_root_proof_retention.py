"""Trwałe zachowanie podpisanego kandydata Root Proof bez decyzji o emisji.

To wewnętrzny port zaufanej kompozycji: przyjmuje już podpisane bajty i pełne
fakty żądania, lecz żadnemu DTO nie nadaje uprawnień. PostgreSQL zachowuje tylko
STAGED. Nie ma tu podpisywania, BIND, dopisywania historii ani przejścia do
COMMITTED. Poprawnego, uprzywilejowanego cofnięcia całej bazy nie wykrywa ten
port; wymagałoby to niezależnego, uwierzytelnionego checkpointu.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import asdict, dataclass
from typing import Any, Literal, NoReturn, cast

import psycopg
from psycopg import sql
from psycopg.errors import UniqueViolation

from bot_core.cha_attempt_signatures import RetainedIssuanceRequest
from bot_core.cha_attempt_store import AttemptAuthorization
from bot_core.cha_issuance_request import (
    IssuanceSignerIdentity,
    IssuanceSigningRole,
    encode_signature,
    request_bytes,
    request_reference,
)
from bot_core.licensing.canonical import canonical_json_bytes, exact, parse_canonical
from bot_core.local_signing_custody import (
    LocalRootProofSigningProvider,
    SigningCredentialSnapshot,
    SigningKeyLifecycle,
)
from bot_core.postgresql_entitlement_registry import PostgreSQLConnectionConfig
from bot_core.postgresql_root_proof_retention_schema import qualify_retention_schema
from bot_core.root_proof_issuer_substrate import (
    CredentialRoleIdentity,
    CredentialSemanticRole,
    ProviderIdentity,
    ProviderRole,
    SecurityProfile,
    SecurityProfileIdentity,
    public_key_material_identity,
)

ROOT_PROOF_SIGNING_DOMAIN = b"CRYPTOHUNTER_ACCOUNT_GENESIS_ROOT_PROOF_V1\x00"
SIGNED_PAYLOAD_FIELDS = (
    "schema_version",
    "environment",
    "trust_domain",
    "issuer_id",
    "issuer_key_id",
    "issuer_key_version",
    "root_proof_id",
    "intended_action",
    "product_scope",
    "account_id",
    "logical_operation_id",
    "canonical_genesis_request_fingerprint_sha256",
    "bootstrap_entitlement_id",
    "entitlement_generation",
    "authorization_constraints",
    "issuance_attempt_id",
)
_STORAGE_SCHEMA = "CryptoHunter.Stage9.PreparedRootProofRetentionV1"
_SAFE_IDENTIFIER = re.compile(r"[a-z_][a-z0-9_]{0,62}\Z")
_ROOT_PROOF_ID = re.compile(
    r"rpf_[0-9a-f]{8}-[0-9a-f]{4}-7[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}\Z"
)
_ATTEMPT_ID = re.compile(
    r"rpa_[0-9a-f]{8}-[0-9a-f]{4}-7[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}\Z"
)


class RootProofRetentionError(RuntimeError):
    """Jawna odmowa portu zachowania; wyjątek nie zawiera danych połączenia."""


class RootProofRetentionUnavailable(RootProofRetentionError):
    """Odczyt lub kwalifikacja nie pozwala ustalić stanu autorytetu."""


class RootProofRetentionIndeterminate(RootProofRetentionError):
    """Rozpoczęty zapis lub jego commit może mieć nieznany wynik."""


class RootProofRetentionConflict(RootProofRetentionError):
    """Kandydat albo ponowienie narusza dokładną zachowaną tożsamość."""


class RootProofRetentionCorrupt(RootProofRetentionError):
    """Zachowane bajty, podpis lub ich pochodzenie nie przechodzą weryfikacji."""


class RootProofRetentionNotIssued(RootProofRetentionError):
    """Brak uwierzytelnionej, atomowej decyzji o emisji; STAGED jej nie zastępuje."""


def _persistence_cut(name: str) -> None:
    """Punkt testowego przerwania bezpośrednio przed lub po commicie zapisu."""


@dataclass(frozen=True, slots=True)
class RootProofRetentionScope:
    """Pełny, dokładny zakres instalacji; sama konfiguracja nie nadaje grantu."""

    security_profile: SecurityProfile
    environment: str
    trust_domain: str
    product_scope: str
    issuer_id: str

    def __post_init__(self) -> None:
        if (
            type(self.security_profile) is not SecurityProfile
            or self.security_profile is not SecurityProfile.PRODUCTION_LOCAL
            or type(self.environment) is not str
            or self.environment != "PRODUCTION"
            or any(
                type(value) is not str or not value.strip()
                for value in (self.trust_domain, self.product_scope, self.issuer_id)
            )
        ):
            raise TypeError("EXACT_PRODUCTION_LOCAL_RETENTION_SCOPE_REQUIRED")


@dataclass(frozen=True, slots=True)
class PreparedRootProof:
    """Odłączone oryginalne dowody STAGED; nigdy dowód emisji ani capability."""

    scope: RootProofRetentionScope
    authorization: AttemptAuthorization
    retained_request: RetainedIssuanceRequest
    canonical_envelope: bytes
    canonical_payload: bytes
    signature: bytes
    credential_snapshot: SigningCredentialSnapshot
    root_proof_id: str
    state: Literal["STAGED"] = "STAGED"
    authorizes_operations: Literal[False] = False

    def __post_init__(self) -> None:
        if (
            type(self.state) is not str
            or self.state != "STAGED"
            or self.authorizes_operations is not False
        ):
            raise TypeError("STAGED_RETENTION_NEVER_AUTHORIZES_OPERATIONS")


def _canonical(value: Any) -> bytes:
    return cast(bytes, canonical_json_bytes(value))


def _scope_data(scope: RootProofRetentionScope) -> dict[str, str]:
    return {
        "security_profile": scope.security_profile.value,
        "environment": scope.environment,
        "trust_domain": scope.trust_domain,
        "product_scope": scope.product_scope,
        "issuer_id": scope.issuer_id,
    }


def _request_data(request: RetainedIssuanceRequest) -> dict[str, Any]:
    def checkpoint(value: tuple[IssuanceSignerIdentity, str] | None) -> dict[str, Any]:
        if value is None:
            raise ValueError("BOTH_RETAINED_REQUEST_SIGNATURES_REQUIRED")
        return {"identity": value[0].payload(), "signature_base64url": value[1]}

    return {
        "attempt_id": request.attempt_id,
        "canonical_bytes_hex": request.canonical_bytes.hex(),
        "reference": request.reference,
        "digest": request.digest,
        "requester": checkpoint(request.requester),
        "claimant": checkpoint(request.claimant),
    }


def _snapshot_data(snapshot: SigningCredentialSnapshot) -> dict[str, Any]:
    return {
        "provider_identity": {
            "role": snapshot.provider_identity.role.value,
            "security": {
                "profile": snapshot.provider_identity.security.profile.value,
                "trust_domain": snapshot.provider_identity.security.trust_domain,
            },
            "provider_namespace": snapshot.provider_identity.provider_namespace,
        },
        "credential_identity": {
            **asdict(snapshot.credential_identity),
            "semantic_role": snapshot.credential_identity.semantic_role.value,
        },
        "key_version": snapshot.key_version,
        "lifecycle_generation": snapshot.lifecycle_generation,
        "lifecycle_state": snapshot.lifecycle_state.value,
        "public_key_hex": snapshot.public_key.hex(),
        "key_material_identity": snapshot.key_material_identity,
    }


def _snapshot_from_data(value: object) -> SigningCredentialSnapshot:
    if type(value) is not dict:
        raise ValueError("EXACT_PUBLIC_SIGNER_SNAPSHOT_REQUIRED")
    exact(
        value,
        {
            "provider_identity",
            "credential_identity",
            "key_version",
            "lifecycle_generation",
            "lifecycle_state",
            "public_key_hex",
            "key_material_identity",
        },
        "credential_snapshot",
    )
    provider = value["provider_identity"]
    credential = value["credential_identity"]
    if type(provider) is not dict or type(credential) is not dict:
        raise ValueError("EXACT_PUBLIC_SIGNER_IDENTITIES_REQUIRED")
    exact(provider, {"role", "security", "provider_namespace"}, "provider_identity")
    security = provider["security"]
    if type(security) is not dict:
        raise ValueError("EXACT_SIGNER_SECURITY_SCOPE_REQUIRED")
    exact(security, {"profile", "trust_domain"}, "security")
    exact(credential, set(CredentialRoleIdentity.__dataclass_fields__), "credential_identity")
    return SigningCredentialSnapshot(
        ProviderIdentity(
            ProviderRole(provider["role"]),
            SecurityProfileIdentity(SecurityProfile(security["profile"]), security["trust_domain"]),
            provider["provider_namespace"],
        ),
        CredentialRoleIdentity(
            **{**credential, "semantic_role": CredentialSemanticRole(credential["semantic_role"])}
        ),
        value["key_version"],
        value["lifecycle_generation"],
        SigningKeyLifecycle(value["lifecycle_state"]),
        bytes.fromhex(value["public_key_hex"]),
        value["key_material_identity"],
    )


def _exact_inputs(
    scope: RootProofRetentionScope,
    authorization: AttemptAuthorization,
    retained_request: RetainedIssuanceRequest,
) -> tuple[AttemptAuthorization, RetainedIssuanceRequest]:
    if (
        type(authorization) is not AttemptAuthorization
        or type(retained_request) is not RetainedIssuanceRequest
    ):
        raise ValueError("EXACT_RETAINED_AUTHORIZATION_AND_REQUEST_REQUIRED")
    # Ponowne konstruktory wykrywają również mutację zamrożonych DTO przez object.__setattr__.
    auth = AttemptAuthorization(**asdict(authorization))
    if (
        auth.environment != scope.environment
        or auth.trust_domain != scope.trust_domain
        or auth.product_scope != scope.product_scope
        or auth.reservation_identity is None
        or type(retained_request.attempt_id) is not str
        or _ATTEMPT_ID.fullmatch(retained_request.attempt_id) is None
        or type(retained_request.canonical_bytes) is not bytes
        or retained_request.canonical_bytes != request_bytes(auth, retained_request.attempt_id)
        or type(retained_request.reference) is not str
        or retained_request.reference != request_reference(retained_request.canonical_bytes)
        or type(retained_request.digest) is not str
        or retained_request.digest != hashlib.sha256(retained_request.canonical_bytes).hexdigest()
    ):
        raise ValueError("EXACT_PRODUCTION_AUTHORIZATION_REQUEST_BINDING_REQUIRED")
    checkpoints: list[tuple[IssuanceSignerIdentity, str]] = []
    for value, role in (
        (retained_request.requester, IssuanceSigningRole.REQUESTER),
        (retained_request.claimant, IssuanceSigningRole.CLAIMANT),
    ):
        if (
            type(value) is not tuple
            or len(value) != 2
            or type(value[0]) is not IssuanceSignerIdentity
        ):
            raise ValueError("EXACT_DISTINCT_RETAINED_SIGNERS_REQUIRED")
        signer = IssuanceSignerIdentity.from_bytes(_canonical(value[0].payload()))
        if signer.role is not role:
            raise ValueError("RETAINED_SIGNER_ROLE_MISMATCH")
        signer.require_authorization(auth)
        signer.verify(retained_request.canonical_bytes, value[1])
        checkpoints.append((signer, value[1]))
    if checkpoints[0][0].key_material_identity == checkpoints[1][0].key_material_identity:
        raise ValueError("REQUESTER_CLAIMANT_KEY_ALIAS_FORBIDDEN")
    request = RetainedIssuanceRequest(
        retained_request.attempt_id,
        bytes(retained_request.canonical_bytes),
        retained_request.reference,
        retained_request.digest,
        checkpoints[0],
        checkpoints[1],
    )
    # Wymusza także dokładny zakres liczb interoperacyjnego JCS.
    _canonical(asdict(auth))
    _canonical(_request_data(request))
    return auth, request


def _validate_snapshot(snapshot: SigningCredentialSnapshot, scope: RootProofRetentionScope) -> None:
    if type(snapshot) is not SigningCredentialSnapshot:
        raise ValueError("EXACT_PINNED_SIGNER_SNAPSHOT_REQUIRED")
    restored = _snapshot_from_data(_snapshot_data(snapshot))
    provider, credential = restored.provider_identity, restored.credential_identity
    if (
        restored != snapshot
        or provider.role is not ProviderRole.ROOT_PROOF_SIGNING
        or provider.security != SecurityProfileIdentity(scope.security_profile, scope.trust_domain)
        or credential.semantic_role is not CredentialSemanticRole.ROOT_PROOF_ISSUER_SIGNING
        or credential.provider_namespace != provider.provider_namespace
        or credential.key_handle_or_version is None
        or credential.key_material_identity != restored.key_material_identity
        or restored.key_material_identity != public_key_material_identity(restored.public_key)
        # Dokładny local_signing_custody dopuszcza ACTIVE wyłącznie w generacji 1.
        # verify_historical pomija generację, aby legalne przejście VERIFY_ONLY
        # nie zmieniało dawnych dowodów; nie uprawnia to do wymyślenia generacji.
        or restored.lifecycle_generation
        not in {
            SigningKeyLifecycle.ACTIVE: {1},
            SigningKeyLifecycle.VERIFY_ONLY: {2},
            SigningKeyLifecycle.REVOKED: {2, 3},
        }[restored.lifecycle_state]
    ):
        raise ValueError("ROOT_PROOF_SIGNER_PUBLIC_IDENTITY_MISMATCH")
    _canonical(_snapshot_data(restored))


def _validate_payload(
    raw: bytes,
    signature: bytes,
    scope: RootProofRetentionScope,
    auth: AttemptAuthorization,
    request: RetainedIssuanceRequest,
    snapshot: SigningCredentialSnapshot,
) -> dict[str, Any]:
    if type(raw) is not bytes or type(signature) is not bytes or len(signature) != 64:
        raise ValueError("EXACT_CANONICAL_PAYLOAD_AND_ED25519_SIGNATURE_REQUIRED")
    payload = parse_canonical(raw)
    exact(payload, set(SIGNED_PAYLOAD_FIELDS), "root_proof_signed_payload")
    expected: dict[str, Any] = {
        "schema_version": "1",
        "environment": scope.environment,
        "trust_domain": scope.trust_domain,
        "issuer_id": scope.issuer_id,
        "issuer_key_id": snapshot.credential_identity.credential_identity,
        "issuer_key_version": snapshot.key_version,
        "intended_action": "ACCOUNT_GENESIS_BOOTSTRAP",
        "product_scope": scope.product_scope,
        "account_id": auth.account_id,
        "logical_operation_id": auth.logical_operation_id,
        "canonical_genesis_request_fingerprint_sha256": auth.canonical_genesis_request_fingerprint_sha256,
        "bootstrap_entitlement_id": auth.bootstrap_entitlement_id,
        "entitlement_generation": auth.entitlement_generation,
        "authorization_constraints": {"max_logical_operations": 1, "max_account_ids": 1},
        "issuance_attempt_id": request.attempt_id,
    }
    root_proof_id = payload["root_proof_id"]
    if (
        type(root_proof_id) is not str
        or _ROOT_PROOF_ID.fullmatch(root_proof_id) is None
        or _canonical({name: payload[name] for name in expected}) != _canonical(expected)
    ):
        raise ValueError("FROZEN_ROOT_PROOF_EXACT_TUPLE_REQUIRED")
    for checkpoint in (request.requester, request.claimant):
        if (
            checkpoint is None
            or checkpoint[0].key_material_identity == snapshot.key_material_identity
        ):
            raise ValueError("ISSUER_REQUESTER_CLAIMANT_KEY_ALIAS_FORBIDDEN")
    return cast(dict[str, Any], payload)


def _envelope(
    scope: RootProofRetentionScope,
    auth: AttemptAuthorization,
    request: RetainedIssuanceRequest,
    payload: bytes,
    signature: bytes,
    snapshot: SigningCredentialSnapshot,
    fields: dict[str, Any],
) -> bytes:
    return _canonical(
        {
            "schema_version": "1",
            "storage_schema": _STORAGE_SCHEMA,
            "state": "STAGED",
            "scope": _scope_data(scope),
            "authorization": asdict(auth),
            "retained_request": _request_data(request),
            "credential_snapshot": _snapshot_data(snapshot),
            "signed_payload_fields": list(SIGNED_PAYLOAD_FIELDS),
            "canonical_payload_sha256": hashlib.sha256(payload).hexdigest(),
            "signature_ed25519_base64url": encode_signature(signature),
            "signing_domain": ROOT_PROOF_SIGNING_DOMAIN[:-1].decode("ascii"),
            "root_proof_id": fields["root_proof_id"],
            "issuance_attempt_id": request.attempt_id,
            "logical_operation_id": auth.logical_operation_id,
            "account_id": auth.account_id,
            "bootstrap_entitlement_id": auth.bootstrap_entitlement_id,
            "entitlement_generation": auth.entitlement_generation,
        }
    )


class PostgreSQLRootProofRetention:
    """Dokładny runtime PostgreSQL z przypiętym, rzeczywistym weryfikatorem custody."""

    __slots__ = ("__connection", "__schema", "__scope", "__verification_authority")
    __connection: PostgreSQLConnectionConfig
    __schema: str
    __scope: RootProofRetentionScope
    __verification_authority: LocalRootProofSigningProvider

    def __init__(
        self,
        connection: PostgreSQLConnectionConfig,
        *,
        schema: str,
        scope: RootProofRetentionScope,
        verification_authority: LocalRootProofSigningProvider,
    ) -> None:
        if (
            type(connection) is not PostgreSQLConnectionConfig
            or type(schema) is not str
            or _SAFE_IDENTIFIER.fullmatch(schema) is None
            or type(scope) is not RootProofRetentionScope
            or type(verification_authority) is not LocalRootProofSigningProvider
        ):
            raise TypeError("EXACT_RETENTION_COMPOSITION_REQUIRED")
        detached_scope = RootProofRetentionScope(**asdict(scope))
        object.__setattr__(
            self,
            "_PostgreSQLRootProofRetention__connection",
            PostgreSQLConnectionConfig(connection.dsn),
        )
        object.__setattr__(self, "_PostgreSQLRootProofRetention__schema", schema)
        object.__setattr__(self, "_PostgreSQLRootProofRetention__scope", detached_scope)
        object.__setattr__(
            self, "_PostgreSQLRootProofRetention__verification_authority", verification_authority
        )
        try:
            _validate_snapshot(verification_authority.credential_snapshot(), detached_scope)
            with self._connect() as conn, conn.transaction():
                self._qualify(conn)
        except Exception:
            raise RootProofRetentionUnavailable("RETENTION_COMPOSITION_UNAVAILABLE") from None

    def __init_subclass__(cls) -> None:
        raise TypeError("RETENTION_PROVIDER_CANNOT_BE_SUBCLASSED")

    def __setattr__(self, name: str, value: object) -> None:
        raise TypeError("RETENTION_COMPOSITION_IS_IMMUTABLE")

    def __repr__(self) -> str:
        return f"PostgreSQLRootProofRetention(connection=<redacted>, schema={self.__schema!r})"

    def _connect(self) -> psycopg.Connection[Any]:
        conn = psycopg.connect(self.__connection.dsn, autocommit=True)
        try:
            conn.execute("SET search_path=pg_catalog")
        except Exception:
            conn.close()
            raise
        return conn

    def _qualify(self, conn: psycopg.Connection[Any]) -> None:
        qualify_retention_schema(conn, self.__schema, self.__scope.trust_domain)

    def _inputs(
        self, authorization: AttemptAuthorization, retained_request: RetainedIssuanceRequest
    ) -> tuple[AttemptAuthorization, RetainedIssuanceRequest]:
        try:
            return _exact_inputs(self.__scope, authorization, retained_request)
        except Exception:
            raise RootProofRetentionConflict("RETENTION_INPUT_BINDING_REJECTED") from None

    def _read_row(self, conn: psycopg.Connection[Any], request: RetainedIssuanceRequest) -> object:
        rows = conn.execute(
            sql.SQL("SELECT envelope,payload,signature FROM {}.read_prepared(%s,%s)").format(
                sql.Identifier(self.__schema)
            ),
            (_canonical(_scope_data(self.__scope)), _canonical(request.attempt_id)),
        ).fetchall()
        if not rows:
            return None
        if len(rows) != 1:
            raise RootProofRetentionCorrupt("RETENTION_READ_CARDINALITY_INVALID")
        return rows[0]

    def _restore(
        self, row: object, auth: AttemptAuthorization, request: RetainedIssuanceRequest
    ) -> PreparedRootProof:
        try:
            if (
                type(row) is not tuple
                or len(row) != 3
                or any(type(value) is not bytes for value in row)
            ):
                raise ValueError("EXACT_STORED_BYTEA_REQUIRED")
            envelope, payload, signature = row
            document = parse_canonical(envelope)
            snapshot = _snapshot_from_data(document["credential_snapshot"])
            _validate_snapshot(snapshot, self.__scope)
            if snapshot.lifecycle_state is not SigningKeyLifecycle.ACTIVE:
                raise ValueError("ORIGINAL_ACTIVE_SIGNING_SNAPSHOT_REQUIRED")
            fields = _validate_payload(payload, signature, self.__scope, auth, request, snapshot)
            expected = _envelope(self.__scope, auth, request, payload, signature, snapshot, fields)
            if envelope != expected:
                raise ValueError("FULL_ORIGINAL_RETENTION_ENVELOPE_MISMATCH")
            if (
                self.__verification_authority.verify_historical(
                    ROOT_PROOF_SIGNING_DOMAIN + payload, signature, snapshot
                )
                is not True
            ):
                raise ValueError("PINNED_ROOT_PROOF_SIGNATURE_INVALID")
            return PreparedRootProof(
                RootProofRetentionScope(**asdict(self.__scope)),
                auth,
                request,
                bytes(envelope),
                bytes(payload),
                bytes(signature),
                snapshot,
                fields["root_proof_id"],
            )
        except Exception:
            raise RootProofRetentionCorrupt("RETAINED_ROOT_PROOF_EVIDENCE_REJECTED") from None

    def read_prepared(
        self, *, authorization: AttemptAuthorization, retained_request: RetainedIssuanceRequest
    ) -> PreparedRootProof | None:
        """Odczytaj i zweryfikuj dokładne oryginalne STAGED; brak wiersza to tylko brak."""
        auth, request = self._inputs(authorization, retained_request)
        try:
            with self._connect() as conn, conn.transaction():
                self._qualify(conn)
                row = self._read_row(conn, request)
                return None if row is None else self._restore(row, auth, request)
        except RootProofRetentionError:
            raise
        except Exception:
            raise RootProofRetentionUnavailable("RETENTION_READ_UNAVAILABLE") from None

    def retain_prepared(
        self,
        *,
        authorization: AttemptAuthorization,
        retained_request: RetainedIssuanceRequest,
        canonical_payload: bytes,
        signature: bytes,
    ) -> PreparedRootProof:
        """Zachowaj istniejący podpis lub zwróć identyczny oryginał bez ponownej emisji."""
        auth, request = self._inputs(authorization, retained_request)
        write_attempted = False
        try:
            with self._connect() as conn, conn.transaction():
                self._qualify(conn)
                row = self._read_row(conn, request)
                if row is not None:
                    prepared = self._restore(row, auth, request)
                    if (
                        type(canonical_payload) is not bytes
                        or type(signature) is not bytes
                        or canonical_payload != prepared.canonical_payload
                        or signature != prepared.signature
                    ):
                        raise RootProofRetentionConflict("EXACT_RETRY_BYTES_REQUIRED")
                    return prepared
                try:
                    snapshot = self.__verification_authority.credential_snapshot()
                    _validate_snapshot(snapshot, self.__scope)
                    if snapshot.lifecycle_state is not SigningKeyLifecycle.ACTIVE:
                        raise ValueError("NEW_RETENTION_REQUIRES_ACTIVE_SIGNER")
                    fields = _validate_payload(
                        canonical_payload, signature, self.__scope, auth, request, snapshot
                    )
                    if (
                        self.__verification_authority.verify_historical(
                            ROOT_PROOF_SIGNING_DOMAIN + canonical_payload, signature, snapshot
                        )
                        is not True
                    ):
                        raise ValueError("PINNED_ROOT_PROOF_SIGNATURE_INVALID")
                    if self.__verification_authority.credential_snapshot() != snapshot:
                        raise ValueError("NEW_RETENTION_SIGNER_SNAPSHOT_CHANGED")
                    envelope = _envelope(
                        self.__scope, auth, request, canonical_payload, signature, snapshot, fields
                    )
                except Exception:
                    raise RootProofRetentionConflict("NEW_ROOT_PROOF_CANDIDATE_REJECTED") from None
                write_attempted = True
                rows = conn.execute(
                    sql.SQL(
                        "SELECT envelope,payload,signature FROM {}.retain_prepared(%s,%s,%s,%s,%s,%s,%s,%s)"
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
                        _canonical(fields["root_proof_id"]),
                        envelope,
                        canonical_payload,
                        signature,
                    ),
                ).fetchall()
                if len(rows) != 1:
                    raise RootProofRetentionCorrupt("RETENTION_WRITE_CARDINALITY_INVALID")
                prepared = self._restore(rows[0], auth, request)
                if (
                    prepared.canonical_envelope != envelope
                    or prepared.canonical_payload != canonical_payload
                    or prepared.signature != signature
                ):
                    raise RootProofRetentionConflict("EXACT_RETRY_ENVELOPE_REQUIRED")
                _persistence_cut("before_commit")
            _persistence_cut("after_commit")
            return prepared
        except RootProofRetentionError:
            raise
        except UniqueViolation:
            raise RootProofRetentionConflict("RETENTION_IDENTITY_ALREADY_FENCED") from None
        except Exception:
            if write_attempted:
                raise RootProofRetentionIndeterminate("RETENTION_WRITE_OUTCOME_UNKNOWN") from None
            raise RootProofRetentionUnavailable("RETENTION_WRITE_AUTHORITY_UNAVAILABLE") from None

    def require_issued(
        self, *, authorization: AttemptAuthorization, retained_request: RetainedIssuanceRequest
    ) -> NoReturn:
        """Odmów sukcesu emisji nawet dla poprawnego, trwale zachowanego STAGED."""
        self.read_prepared(authorization=authorization, retained_request=retained_request)
        raise RootProofRetentionNotIssued("ATOMIC_SIGNED_PROOF_BIND_HISTORY_AUTHORITY_UNAVAILABLE")
