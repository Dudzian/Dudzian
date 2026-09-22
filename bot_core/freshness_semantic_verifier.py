"""Isolated semantic verifier for AccountGenesis freshness candidates.

This component deliberately has one operation: turn three independently
authenticated, canonical protocol objects into a one-time PostgreSQL
preparation.  In particular it is not a general purpose signature service.
The credential resolver and database preparer are construction-time trusted
dependencies and are never selected by an IPC caller.
"""

from __future__ import annotations

from dataclasses import dataclass
import base64
import hashlib
import os
import re
import socket
import struct
from typing import Protocol

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey
import psycopg

from bot_core.postgresql_freshness_authority import (
    DOCUMENT_FIELDS,
    DOCUMENT_PAYLOAD_FIELDS,
    FINALIZATION_ROLE,
    PREPARATION_FIELDS,
    PROPOSER_ROLE,
    RECEIPT_FIELDS,
    canonical_json_bytes,
    complete_semantic_head_digest,
    normalize_complete_semantic_head_set,
    parse_canonical_json,
)

PROPOSER_AUTHENTICATION_FIELDS = frozenset({
    "schema_version", "proposer_identity", "environment", "trust_domain",
    "proposer_key_id", "proposer_key_version",
    "signed_or_authenticated_document_digest", "authentication_tag_or_signature",
})
PROPOSER_DOMAIN = b"cryptohunter.account-genesis.cha-proposer-authentication.v1\x00"
DOCUMENT_AUTHENTICATION_DOMAIN = (
    b"cryptohunter.account-genesis.freshness-document-authentication.v1\x00"
)
RECEIPT_AUTHENTICATION_DOMAIN = (
    b"cryptohunter.account-genesis.freshness-finalization-receipt-authentication.v1\x00"
)
DOCUMENT_DIGEST_DOMAIN = b"cryptohunter.account-genesis.freshness-document-digest.v1\x00"
ORIGINAL_DECISION_DOMAIN = b"cryptohunter.account-genesis.original-decision-identity.v1\x00"
PREPARATION_DOMAIN = b"cryptohunter.account-genesis.prepared-crypto-verification.v1\x00"
_HEX = re.compile(r"[0-9a-f]{64}\Z")


class FreshnessSemanticVerificationError(ValueError):
    """A candidate failed closed before a preparation was committed."""


class PreparationOutcomeUnknown(RuntimeError):
    """The connection was lost while commit outcome could no longer be known."""


@dataclass(frozen=True, slots=True)
class RetainedVerificationCredential:
    """Immutable credential facts resolved from the retained trust source."""

    environment: str
    trust_domain: str
    authority_id: str
    credential_id: str
    semantic_identity: str
    semantic_role: str
    key_id: str
    key_version: int
    lifecycle_generation: int
    public_key: bytes

    @property
    def public_key_material_identity(self) -> str:
        return hashlib.sha256(self.public_key).hexdigest()


class RetainedCredentialResolver(Protocol):
    def resolve(self, *, environment: str, trust_domain: str, authority_id: str,
                semantic_role: str, key_id: str, key_version: int
                ) -> RetainedVerificationCredential: ...

    def predecessor_complete_head_digest(
        self, *, environment: str, trust_domain: str, authority_id: str,
        generation: int, document_digest: str,
    ) -> str: ...


class PreparationWriter(Protocol):
    def prepare(self, preparation: bytes, document: bytes, receipt: bytes) -> str: ...


@dataclass(frozen=True, slots=True)
class PostgreSQLVerifierConnection:
    """Non-secret, fixed local peer-auth connection coordinates."""

    socket_directory: str
    port: int
    database: str = "freshness_gate"

    def __post_init__(self) -> None:
        if (type(self.socket_directory) is not str or
                not os.path.isabs(self.socket_directory)):
            raise ValueError("socket_directory must be an absolute Unix socket directory")
        if type(self.port) is not int or not 1 <= self.port <= 65535:
            raise ValueError("port must be an exact TCP/Unix port number")
        if type(self.database) is not str or re.fullmatch(r"[a-z_][a-z0-9_]{0,62}", self.database) is None:
            raise ValueError("database must be a safe fixed identifier")

    def connect(self) -> psycopg.Connection[object]:
        connection = psycopg.connect(
            host=self.socket_directory, port=self.port, dbname=self.database,
            user="freshness_crypto_verifier", sslmode="disable",
        )
        facts = connection.execute(
            "SELECT session_user,current_user,inet_client_addr(),inet_server_addr(),"
            "coalesce((SELECT ssl FROM pg_catalog.pg_stat_ssl "
            "WHERE pid=pg_catalog.pg_backend_pid()),false)"
        ).fetchone()
        if facts != ("freshness_crypto_verifier", "freshness_crypto_verifier", None, None, False):
            connection.close()
            raise FreshnessSemanticVerificationError("non-peer or wrong database session")
        return connection


@dataclass(frozen=True, slots=True)
class PostgreSQLRetainedCredentialResolver:
    """Reviewed read-only authority adapter; it has no raw table SELECT grants."""

    connection: PostgreSQLVerifierConnection

    def resolve(self, *, environment: str, trust_domain: str, authority_id: str,
                semantic_role: str, key_id: str, key_version: int
                ) -> RetainedVerificationCredential:
        with self.connection.connect() as database:
            row = database.execute(
                "SELECT * FROM freshness_authority.resolve_verification_credential(%s,%s,%s,%s,%s,%s)",
                (environment, trust_domain, authority_id, semantic_role, key_id, key_version),
            ).fetchone()
        if row is None:
            raise FreshnessSemanticVerificationError("unknown retained credential")
        credential_id, semantic_identity, lifecycle_generation, public_key = row
        return RetainedVerificationCredential(
            environment, trust_domain, authority_id, str(credential_id),
            str(semantic_identity), semantic_role, key_id, key_version,
            int(lifecycle_generation), bytes(public_key),
        )

    def predecessor_complete_head_digest(
        self, *, environment: str, trust_domain: str, authority_id: str,
        generation: int, document_digest: str,
    ) -> str:
        with self.connection.connect() as database:
            row = database.execute(
                "SELECT freshness_authority.resolve_predecessor_head(%s,%s,%s,%s,%s)",
                (environment, trust_domain, authority_id, generation, document_digest),
            ).fetchone()
        if row is None or type(row[0]) is not str:
            raise FreshnessSemanticVerificationError("unknown exact predecessor")
        return row[0]


@dataclass(frozen=True, slots=True)
class PostgreSQLPreparationWriter:
    """The sole database mutation operation available to the verifier process."""

    connection: PostgreSQLVerifierConnection

    def prepare(self, preparation: bytes, document: bytes, receipt: bytes) -> str:
        # No SET ROLE and no DML: kernel peer authentication establishes session_user.
        database = self.connection.connect()
        try:
            value = database.execute(
                "SELECT freshness_authority.prepare_verified_freshness_candidate(%s,%s,%s)",
                (preparation, document, receipt),
            ).fetchone()
            if value is None or type(value[0]) is not str:
                raise FreshnessSemanticVerificationError("preparation returned no identity")
            try:
                database.commit()
            except psycopg.OperationalError as exc:
                raise PreparationOutcomeUnknown(
                    "preparation commit outcome unknown; exact retry required"
                ) from exc
            return value[0]
        finally:
            database.close()


def _object(value: object, fields: frozenset[str], name: str) -> dict[str, object]:
    if type(value) is not dict or set(value) != fields:
        raise FreshnessSemanticVerificationError(f"invalid {name} schema")
    return value


def _string(value: object, name: str) -> str:
    if type(value) is not str or not value:
        raise FreshnessSemanticVerificationError(f"{name} must be a non-empty string")
    return value


def _integer(value: object, name: str, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum or value > 2**53 - 1:
        raise FreshnessSemanticVerificationError(f"{name} must be an exact safe integer")
    return value


def _digest(value: object, name: str) -> str:
    text = _string(value, name)
    if _HEX.fullmatch(text) is None:
        raise FreshnessSemanticVerificationError(f"{name} must be lowercase SHA-256 hex")
    return text


def _signature(value: object, name: str) -> bytes:
    text = _string(value, name)
    if "=" in text or re.fullmatch(r"[A-Za-z0-9_-]{86}", text) is None:
        raise FreshnessSemanticVerificationError(f"invalid {name} encoding")
    try:
        decoded = base64.urlsafe_b64decode(text + "==")
    except Exception as exc:
        raise FreshnessSemanticVerificationError(f"invalid {name} encoding") from exc
    if (len(decoded) != 64 or
            base64.urlsafe_b64encode(decoded).rstrip(b"=").decode("ascii") != text):
        raise FreshnessSemanticVerificationError(f"noncanonical {name}")
    return decoded


def _verify(public_key: bytes, signature: bytes, preimage: bytes, name: str) -> None:
    if type(public_key) is not bytes or len(public_key) != 32:
        raise FreshnessSemanticVerificationError(f"invalid retained {name} public key")
    try:
        Ed25519PublicKey.from_public_bytes(public_key).verify(signature, preimage)
    except (ValueError, InvalidSignature) as exc:
        raise FreshnessSemanticVerificationError(f"invalid {name} signature") from exc


@dataclass(frozen=True, slots=True)
class FreshnessSemanticVerifier:
    credentials: RetainedCredentialResolver
    preparations: PreparationWriter
    verifier_authority_identity: str
    verifier_authority_version: int = 1

    def __post_init__(self) -> None:
        _string(self.verifier_authority_identity, "verifier_authority_identity")
        _integer(self.verifier_authority_version, "verifier_authority_version", 1)

    def verify_and_prepare(self, proposer_authentication: bytes,
                           authoritative_document: bytes,
                           finalization_receipt: bytes) -> str:
        """Verify the exact candidate and return only its opaque preparation id."""
        try:
            proposer = _object(parse_canonical_json(proposer_authentication),
                               PROPOSER_AUTHENTICATION_FIELDS, "proposer authentication")
            document = _object(parse_canonical_json(authoritative_document),
                               DOCUMENT_FIELDS, "authoritative document")
            receipt = _object(parse_canonical_json(finalization_receipt),
                              RECEIPT_FIELDS, "receipt")
            payload = _object(document["payload"], DOCUMENT_PAYLOAD_FIELDS,
                              "document payload")
            return self._verified_prepare(proposer, document, payload, receipt,
                                          authoritative_document, finalization_receipt)
        except FreshnessSemanticVerificationError:
            raise
        except (TypeError, ValueError, UnicodeError, KeyError) as exc:
            raise FreshnessSemanticVerificationError("invalid canonical candidate") from exc

    def _verified_prepare(self, proposer: dict[str, object], document: dict[str, object],
                          payload: dict[str, object], receipt: dict[str, object],
                          document_bytes: bytes, receipt_bytes: bytes) -> str:
        if (_integer(proposer["schema_version"], "proposer schema version", 1) != 1 or
                _integer(payload["schema_version"], "document schema version", 1) != 1 or
                _integer(receipt["schema_version"], "receipt schema version", 1) != 1):
            raise FreshnessSemanticVerificationError("wrong schema version")
        environment = _string(payload["environment"], "environment")
        trust_domain = _string(payload["trust_domain"], "trust_domain")
        authority_id = _string(payload["authority_id"], "authority_id")
        if environment != "PRODUCTION":
            raise FreshnessSemanticVerificationError("TEST artifacts are forbidden")
        generation = _integer(payload["generation"], "generation", 1)
        predecessor_generation = _integer(payload["predecessor_generation"], "predecessor_generation")
        if generation != predecessor_generation + 1:
            raise FreshnessSemanticVerificationError("generation is not exact successor")
        predecessor_digest = _digest(payload["predecessor_document_digest"], "predecessor digest")
        heads = normalize_complete_semantic_head_set(payload["complete_semantic_head_set"])
        if heads != payload["complete_semantic_head_set"]:
            raise FreshnessSemanticVerificationError("semantic head set is not normalized")
        document_digest = hashlib.sha256(
            DOCUMENT_DIGEST_DOMAIN + canonical_json_bytes(payload)
        ).hexdigest()
        if _digest(document["document_digest"], "document digest") != document_digest:
            raise FreshnessSemanticVerificationError("document digest mismatch")

        proposer_key_id = _string(proposer["proposer_key_id"], "proposer key id")
        proposer_key_version = _integer(proposer["proposer_key_version"], "proposer key version", 1)
        proposer_identity = _string(proposer["proposer_identity"], "proposer identity")
        if (proposer["environment"] != environment or proposer["trust_domain"] != trust_domain or
                proposer["signed_or_authenticated_document_digest"] != document_digest):
            raise FreshnessSemanticVerificationError("proposer binding mismatch")
        proposer_credential = self.credentials.resolve(
            environment=environment, trust_domain=trust_domain, authority_id=authority_id,
            semantic_role=PROPOSER_ROLE, key_id=proposer_key_id,
            key_version=proposer_key_version,
        )
        self._credential(proposer_credential, environment, trust_domain, authority_id,
                         PROPOSER_ROLE, proposer_key_id, proposer_key_version)
        if proposer_credential.semantic_identity != proposer_identity:
            raise FreshnessSemanticVerificationError("proposer identity laundering")
        proposer_unsigned = {k: v for k, v in proposer.items() if k != "authentication_tag_or_signature"}
        proposer_preimage = PROPOSER_DOMAIN + canonical_json_bytes(proposer_unsigned)
        _verify(proposer_credential.public_key,
                _signature(proposer["authentication_tag_or_signature"], "proposer signature"),
                proposer_preimage, "proposer")

        finalization_key_id = _string(payload["freshness_authority_key_id"], "finalization key id")
        finalization_key_version = _integer(payload["freshness_authority_key_version"],
                                            "finalization key version", 1)
        finalization_credential = self.credentials.resolve(
            environment=environment, trust_domain=trust_domain, authority_id=authority_id,
            semantic_role=FINALIZATION_ROLE, key_id=finalization_key_id,
            key_version=finalization_key_version,
        )
        self._credential(finalization_credential, environment, trust_domain, authority_id,
                         FINALIZATION_ROLE, finalization_key_id, finalization_key_version)
        if proposer_credential.public_key == finalization_credential.public_key:
            raise FreshnessSemanticVerificationError("cross-role key material reuse")
        document_unsigned = {"payload": payload, "document_digest": document_digest}
        document_preimage = DOCUMENT_AUTHENTICATION_DOMAIN + canonical_json_bytes(document_unsigned)
        _verify(finalization_credential.public_key,
                _signature(document["authentication_tag_or_signature"], "document signature"),
                document_preimage, "document")

        head_digest = complete_semantic_head_digest(heads)
        predecessor_head_digest = _digest(
            self.credentials.predecessor_complete_head_digest(
                environment=environment, trust_domain=trust_domain,
                authority_id=authority_id, generation=predecessor_generation,
                document_digest=predecessor_digest,
            ),
            "resolved predecessor complete head digest",
        )
        self._receipt_bindings(receipt, environment, trust_domain, authority_id,
                               predecessor_generation, predecessor_digest, generation,
                               document_digest, head_digest, finalization_key_id,
                               finalization_key_version, payload["finalization_request_id"])
        receipt_unsigned = {k: v for k, v in receipt.items() if k != "authentication_tag_or_signature"}
        receipt_preimage = RECEIPT_AUTHENTICATION_DOMAIN + canonical_json_bytes(receipt_unsigned)
        _verify(finalization_credential.public_key,
                _signature(receipt["authentication_tag_or_signature"], "receipt signature"),
                receipt_preimage, "receipt")

        original_material = {k: receipt[k] for k in (
            "environment", "trust_domain", "authority_id", "exact_predecessor_generation",
            "exact_predecessor_document_digest", "accepted_generation",
            "accepted_document_digest", "complete_semantic_head_digest", "finalization_request_id",
        )}
        original_identity = hashlib.sha256(
            ORIGINAL_DECISION_DOMAIN + canonical_json_bytes(original_material)
        ).hexdigest()
        receipt_digest = hashlib.sha256(receipt_bytes).hexdigest()
        preparation_without_id: dict[str, object] = {
            "schema_version": 1, "security_profile": "PRODUCTION_LOCAL",
            "environment": environment, "trust_domain": trust_domain,
            "authority_id": authority_id, "operation_type": "FULL_AUTHORITATIVE_DOCUMENT",
            "expected_predecessor_generation": predecessor_generation,
            "expected_predecessor_document_digest": predecessor_digest,
            "expected_predecessor_complete_semantic_head_digest": predecessor_head_digest,
            "proposed_document_digest": document_digest,
            "original_decision_identity": original_identity,
            "proposer_identity": proposer_identity,
            "proposer_credential_role_identity": proposer_credential.credential_id,
            "proposer_key_version": proposer_key_version,
            "proposer_lifecycle_generation_observed_for_key_binding_only": proposer_credential.lifecycle_generation,
            "proposer_public_key_material_identity": proposer_credential.public_key_material_identity,
            "proposer_authentication_digest": hashlib.sha256(proposer_preimage).hexdigest(),
            "finalization_credential_role_identity": finalization_credential.credential_id,
            "finalization_key_version": finalization_key_version,
            "finalization_lifecycle_generation_observed_for_key_binding_only": finalization_credential.lifecycle_generation,
            "finalization_public_key_material_identity": finalization_credential.public_key_material_identity,
            "authoritative_document_authentication_digest": hashlib.sha256(document_preimage).hexdigest(),
            "receipt_id": receipt["receipt_id"], "receipt_canonical_digest": receipt_digest,
            "finalization_request_id": receipt["finalization_request_id"],
            "verifier_authority_identity": self.verifier_authority_identity,
            "verifier_authority_version": self.verifier_authority_version,
        }
        # Internal identity commits to every immutable evidence dimension and
        # avoids cyclicity by hashing the exact preparation object without its
        # own preparation_id member.
        preparation_id = hashlib.sha256(
            PREPARATION_DOMAIN + canonical_json_bytes(preparation_without_id)
        ).hexdigest()
        preparation = {**preparation_without_id, "preparation_id": preparation_id}
        if set(preparation) != PREPARATION_FIELDS:
            raise AssertionError("internal preparation schema drift")
        return self.preparations.prepare(canonical_json_bytes(preparation),
                                         document_bytes, receipt_bytes)

    @staticmethod
    def _credential(value: RetainedVerificationCredential, environment: str,
                    trust_domain: str, authority_id: str, role: str,
                    key_id: str, key_version: int) -> None:
        if (type(value) is not RetainedVerificationCredential or
                (value.environment, value.trust_domain, value.authority_id,
                 value.semantic_role, value.key_id, value.key_version) !=
                (environment, trust_domain, authority_id, role, key_id, key_version) or
                type(value.lifecycle_generation) is not int or value.lifecycle_generation < 1 or
                type(value.public_key) is not bytes or len(value.public_key) != 32):
            raise FreshnessSemanticVerificationError("retained credential mismatch")

    @staticmethod
    def _receipt_bindings(receipt: dict[str, object], environment: str,
                          trust_domain: str, authority_id: str,
                          predecessor_generation: int, predecessor_digest: str,
                          generation: int, document_digest: str, head_digest: str,
                          key_id: str, key_version: int, request_id: object) -> None:
        expected = {
            "environment": environment, "trust_domain": trust_domain,
            "authority_id": authority_id,
            "exact_predecessor_generation": predecessor_generation,
            "exact_predecessor_document_digest": predecessor_digest,
            "accepted_generation": generation, "accepted_document_digest": document_digest,
            "complete_semantic_head_digest": head_digest,
            "finalization_request_id": request_id,
            "freshness_authority_key_id": key_id,
            "freshness_authority_key_version": key_version,
        }
        _integer(receipt["exact_predecessor_generation"],
                 "receipt predecessor generation")
        _integer(receipt["accepted_generation"], "receipt accepted generation", 1)
        _integer(receipt["freshness_authority_key_version"],
                 "receipt finalization key version", 1)
        if any(type(receipt[k]) is not type(v) or receipt[k] != v
               for k, v in expected.items()):
            raise FreshnessSemanticVerificationError("document/receipt selector mismatch")
        _string(receipt["receipt_id"], "receipt id")
        _string(request_id, "finalization request id")


def serve_local_unix_socket(verifier: FreshnessSemanticVerifier, socket_path: str,
                            *, allowed_peer_uid: int, stop_after: int | None = None) -> None:
    """Serve the single verifier operation over a length-framed Unix socket.

    Deployment must start this executable component as
    ``os_freshness_crypto_verifier`` and own the containing directory.  There
    is intentionally no TCP listener, method selector, signing operation, key
    selector, provider selector, or database-role input.
    """
    if type(allowed_peer_uid) is not int or allowed_peer_uid < 0:
        raise ValueError("allowed_peer_uid must be an exact uid")
    path = os.fspath(socket_path)
    if os.path.exists(path):
        os.unlink(path)
    handled = 0
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as listener:
        listener.bind(path)
        os.chmod(path, 0o660)
        listener.listen(8)
        while stop_after is None or handled < stop_after:
            connection, _ = listener.accept()
            with connection:
                _, peer_uid, _ = struct.unpack("3i", connection.getsockopt(
                    socket.SOL_SOCKET, socket.SO_PEERCRED, struct.calcsize("3i")))
                if peer_uid != allowed_peer_uid:
                    continue
                try:
                    header = _receive_exact(connection, 4)
                    length = struct.unpack("!I", header)[0]
                    if length < 2 or length > 4 * 1024 * 1024:
                        raise FreshnessSemanticVerificationError("invalid IPC frame length")
                    request = parse_canonical_json(_receive_exact(connection, length))
                    request = _object(request, frozenset({
                        "proposer_authentication", "authoritative_document",
                        "finalization_receipt",
                    }), "IPC request")
                    inputs = tuple(_decode_ipc_bytes(request[name], name) for name in (
                        "proposer_authentication", "authoritative_document",
                        "finalization_receipt",
                    ))
                    preparation_id = verifier.verify_and_prepare(*inputs)
                    response = canonical_json_bytes({"preparation_id": preparation_id})
                except Exception:
                    # The wire response discloses no crypto/parser/credential oracle detail.
                    response = canonical_json_bytes({"error": "REJECTED"})
                connection.sendall(struct.pack("!I", len(response)) + response)
                handled += 1
    os.unlink(path)


def _receive_exact(connection: socket.socket, count: int) -> bytes:
    result = bytearray()
    while len(result) < count:
        part = connection.recv(count - len(result))
        if not part:
            raise FreshnessSemanticVerificationError("truncated IPC frame")
        result.extend(part)
    return bytes(result)


def _decode_ipc_bytes(value: object, name: str) -> bytes:
    text = _string(value, name)
    if "=" in text or re.fullmatch(r"[A-Za-z0-9_-]+", text) is None:
        raise FreshnessSemanticVerificationError("invalid IPC byte encoding")
    try:
        result = base64.urlsafe_b64decode(text + "=" * (-len(text) % 4))
    except Exception as exc:
        raise FreshnessSemanticVerificationError("invalid IPC byte encoding") from exc
    if base64.urlsafe_b64encode(result).rstrip(b"=").decode() != text:
        raise FreshnessSemanticVerificationError("noncanonical IPC byte encoding")
    return result
