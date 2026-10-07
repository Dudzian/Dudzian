"""Fixed issuer-side RPC boundary to the independently operated PDSA signers.

This module contains no private keys, signer implementation, caller-selectable
endpoint, or production substitute. An installed root-owned Unix socket proxies
the off-host quorum service. Its deployment must authenticate the issuer's peer
credentials and durably reserve each payload digest before signing. A retry must
return the retained exact reply; service restart must not forget reservations.

The exact authorized three-key set, two-signature threshold and Ed25519 message
prevent quorum substitution. The service chooses any two available authorized
keys and durably retains that selection and exact reply before responding. These
client checks do not prove remote retention: deployment/qualification must
establish the remote idempotency guarantee. The client verifies every returned
signature and fails closed when the service or current authority is absent.
"""

from __future__ import annotations

import hashlib
import os
import socket
import stat
import struct
import sys
import time
from dataclasses import dataclass
from pathlib import Path

from cryptography.exceptions import InvalidSignature

from bot_core.licensing.canonical import canonical_json_bytes, parse_canonical
from bot_core.licensing.external_provisioning import PACKAGE_FIELDS, PACKAGE_SCHEMA, PDSA_DOMAIN
from bot_core.licensing.pre_enrollment import PDSA_TRUST_DOMAIN
from bot_core.licensing.product_profile import PRODUCTION_PRODUCT_PROFILE
from deployment.production_enrollment_issuer import (
    ISSUER_SERVICE_USER,
    ProductionEnrollmentIssuerContext,
    ProductionEnrollmentIssuerError,
    _require_protected_ancestors,
    require_production_enrollment_issuer,
)
from deployment.windows_stage9_production_trust import require_current_production_trust_context

SIGNING_SOCKET = Path("/run/cryptohunter/pdsa-signing/quorum.sock")
REQUEST_SCHEMA = "CryptoHunter.Stage9.PDSAQuorumSigningRequest.v1"
REPLY_SCHEMA = "CryptoHunter.Stage9.PDSAQuorumSigningReply.v1"
MAX_PAYLOAD_BYTES = 16_384
MAX_REQUEST_BYTES = 36_864
MAX_REPLY_BYTES = 4_096
RPC_TIMEOUT_SECONDS = 5.0
_SIGNATURE_FIELDS = {"key_id", "algorithm", "signature_hex"}
_REPLY_FIELDS = {"schema_version", "reservation_id", "signatures"}


@dataclass(frozen=True, slots=True)
class _SocketIdentity:
    device: int
    inode: int
    uid: int
    gid: int
    mode: int


def _installed_signing_socket_identity() -> _SocketIdentity:
    """Inspect the fixed deployment, including the issuer's OS principal."""
    if sys.platform != "linux":
        raise ProductionEnrollmentIssuerError("PRODUCTION_PDSA_SIGNING_DEPLOYMENT_REQUIRED")
    import pwd

    try:
        principal = pwd.getpwnam(ISSUER_SERVICE_USER)
        if (
            principal.pw_uid == 0
            or os.geteuid() != principal.pw_uid
            or os.getegid() != principal.pw_gid
        ):
            raise ProductionEnrollmentIssuerError("PRODUCTION_PDSA_SIGNING_DEPLOYMENT_REQUIRED")
        _require_protected_ancestors(SIGNING_SOCKET.parent)
        metadata = SIGNING_SOCKET.lstat()
        if (
            not SIGNING_SOCKET.is_absolute()
            or SIGNING_SOCKET.resolve(strict=True) != SIGNING_SOCKET
            or not stat.S_ISSOCK(metadata.st_mode)
            or metadata.st_uid != 0
            or metadata.st_gid != principal.pw_gid
            or stat.S_IMODE(metadata.st_mode) != 0o660
            or metadata.st_nlink != 1
        ):
            raise ProductionEnrollmentIssuerError("PRODUCTION_PDSA_SIGNING_DEPLOYMENT_REQUIRED")
        return _SocketIdentity(
            metadata.st_dev,
            metadata.st_ino,
            metadata.st_uid,
            metadata.st_gid,
            stat.S_IMODE(metadata.st_mode),
        )
    except (KeyError, OSError) as exc:
        raise ProductionEnrollmentIssuerError(
            "PRODUCTION_PDSA_SIGNING_DEPLOYMENT_REQUIRED"
        ) from exc


def _set_deadline(connection: socket.socket, deadline: float) -> None:
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        raise ProductionEnrollmentIssuerError("PRODUCTION_PDSA_SIGNING_UNAVAILABLE")
    # Floating-point cancellation around large monotonic clock values can make
    # the computed remainder infinitesimally larger than the configured budget.
    # Clamp explicitly so a socket timeout never exceeds the fixed RPC bound.
    connection.settimeout(min(remaining, RPC_TIMEOUT_SECONDS))


def _read_exact(connection: socket.socket, size: int, deadline: float) -> bytes:
    chunks: list[bytes] = []
    remaining = size
    while remaining:
        _set_deadline(connection, deadline)
        chunk = connection.recv(remaining)
        if not chunk:
            raise ProductionEnrollmentIssuerError("PRODUCTION_PDSA_SIGNING_REPLY_INVALID")
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def _exchange_request(request_raw: bytes) -> bytes:
    """Single bounded RPC; peer authentication precedes sending any request."""
    identity = _installed_signing_socket_identity()
    if not 0 < len(request_raw) <= MAX_REQUEST_BYTES:
        raise ProductionEnrollmentIssuerError("PRODUCTION_PDSA_SIGNING_REQUEST_INVALID")
    deadline = time.monotonic() + RPC_TIMEOUT_SECONDS
    try:
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as connection:
            _set_deadline(connection, deadline)
            connection.connect(os.fspath(SIGNING_SOCKET))
            credentials = connection.getsockopt(socket.SOL_SOCKET, socket.SO_PEERCRED, 12)
            _pid, peer_uid, _peer_gid = struct.unpack("3i", credentials)
            if peer_uid != 0:
                raise ProductionEnrollmentIssuerError("PRODUCTION_PDSA_SIGNING_PEER_REQUIRED")
            if _installed_signing_socket_identity() != identity:
                raise ProductionEnrollmentIssuerError("PRODUCTION_PDSA_SIGNING_SOURCE_CHANGED")
            _set_deadline(connection, deadline)
            connection.sendall(struct.pack("!I", len(request_raw)) + request_raw)
            reply_size = struct.unpack("!I", _read_exact(connection, 4, deadline))[0]
            if not 0 < reply_size <= MAX_REPLY_BYTES:
                raise ProductionEnrollmentIssuerError("PRODUCTION_PDSA_SIGNING_REPLY_INVALID")
            reply_raw = _read_exact(connection, reply_size, deadline)
            if _installed_signing_socket_identity() != identity:
                raise ProductionEnrollmentIssuerError("PRODUCTION_PDSA_SIGNING_SOURCE_CHANGED")
            return reply_raw
    except (OSError, struct.error) as exc:
        raise ProductionEnrollmentIssuerError("PRODUCTION_PDSA_SIGNING_UNAVAILABLE") from exc


def _sign_enrollment_authorization(
    issuer: ProductionEnrollmentIssuerContext, payload_raw: bytes
) -> list[dict[str, str]]:
    authority = require_production_enrollment_issuer(issuer)
    trust = require_current_production_trust_context(authority.trust)
    if type(payload_raw) is not bytes or not 0 < len(payload_raw) <= MAX_PAYLOAD_BYTES:
        raise ProductionEnrollmentIssuerError("PRODUCTION_PDSA_SIGNING_PAYLOAD_INVALID")
    try:
        payload = parse_canonical(payload_raw)
    except (ValueError, TypeError, RecursionError) as exc:
        raise ProductionEnrollmentIssuerError("PRODUCTION_PDSA_SIGNING_PAYLOAD_INVALID") from exc
    if (
        set(payload) != PACKAGE_FIELDS
        or payload.get("schema_version") != PACKAGE_SCHEMA
        or payload.get("environment") != "PRODUCTION"
        or payload.get("pdsa_trust_domain") != PDSA_TRUST_DOMAIN
        or payload.get("product_profile") != PRODUCTION_PRODUCT_PROFILE
        or payload.get("release_policy_digest_sha256") != trust.release_payload_digest
        or type(payload.get("release_policy_generation")) is not int
        or payload.get("release_policy_generation") != trust.release_version
    ):
        raise ProductionEnrollmentIssuerError("PRODUCTION_PDSA_SIGNING_PAYLOAD_INVALID")
    authorized_signer_ids = sorted(trust.pdsa_keys)
    # TEST_ONLY keys are never a fallback in installed production composition.
    if len(authorized_signer_ids) != 3 or any(
        key_id.startswith("TEST_ONLY") for key_id in authorized_signer_ids
    ):
        raise ProductionEnrollmentIssuerError("PRODUCTION_PDSA_SIGNING_AUTHORITY_REQUIRED")
    from bot_core.licensing.pdsa_enrollment_authorization import (
        PDSAAuthorizationError,
        _require_signing_reservation,
    )

    try:
        _require_signing_reservation(authority, payload_raw)
    except PDSAAuthorizationError as exc:
        raise ProductionEnrollmentIssuerError(
            "PRODUCTION_PDSA_SIGNING_RESERVATION_REQUIRED"
        ) from exc
    digest = hashlib.sha256(payload_raw).hexdigest()
    request_raw = canonical_json_bytes(
        {
            "schema_version": REQUEST_SCHEMA,
            "reservation_id": digest,
            "payload_canonical_hex": payload_raw.hex(),
            "signature_domain": PDSA_DOMAIN[:-1].decode("ascii"),
            "authorized_signer_ids": authorized_signer_ids,
            "required_threshold": 2,
        }
    )
    reply_raw = _exchange_request(request_raw)
    try:
        reply = parse_canonical(reply_raw)
        signatures = reply.get("signatures")
        if (
            set(reply) != _REPLY_FIELDS
            or reply.get("schema_version") != REPLY_SCHEMA
            or reply.get("reservation_id") != digest
            or type(signatures) is not list
            or len(signatures) != 2
        ):
            raise ValueError("reply schema or reservation mismatch")
        message = PDSA_DOMAIN + bytes.fromhex(digest)
        result: list[dict[str, str]] = []
        selected_signer_ids: list[str] = []
        for record in signatures:
            if (
                type(record) is not dict
                or set(record) != _SIGNATURE_FIELDS
                or any(type(value) is not str for value in record.values())
                or record["key_id"] not in authorized_signer_ids
                or record["algorithm"] != "Ed25519"
                or len(record["signature_hex"]) != 128
                or any(character not in "0123456789abcdef" for character in record["signature_hex"])
            ):
                raise ValueError("noncanonical quorum record")
            key_id = record["key_id"]
            if selected_signer_ids and key_id <= selected_signer_ids[-1]:
                raise ValueError("quorum signer IDs must be distinct and sorted")
            trust.pdsa_keys[key_id].verify(bytes.fromhex(record["signature_hex"]), message)
            selected_signer_ids.append(key_id)
            result.append(dict(record))
    except (ValueError, TypeError, KeyError, RecursionError, InvalidSignature) as exc:
        raise ProductionEnrollmentIssuerError("PRODUCTION_PDSA_SIGNING_REPLY_INVALID") from exc
    require_production_enrollment_issuer(authority, context=trust)
    require_current_production_trust_context(trust)
    return result
