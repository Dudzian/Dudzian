"""Closed Stage 9 request and retained signing evidence; DTOs grant no authority."""

from __future__ import annotations

import base64
import binascii
import hashlib
from dataclasses import asdict, dataclass
from enum import Enum
from typing import TYPE_CHECKING, cast

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey

from bot_core.licensing.canonical import canonical_json_bytes, parse_canonical
from bot_core.root_proof_issuer_substrate import public_key_material_identity

if TYPE_CHECKING:
    from bot_core.cha_attempt_store import AttemptAuthorization

ISSUER_TARGET_NAMESPACE = "CryptoHunter.Stage9.RootProofIssuer.V1"
REQUESTER_DOMAIN = "CRYPTOHUNTER_ACCOUNT_GENESIS_ROOT_PROOF_ISSUANCE_REQUEST_V1"
CLAIMANT_DOMAIN = "CRYPTOHUNTER_ACCOUNT_GENESIS_ROOT_PROOF_ENTITLEMENT_CLAIM_V1"
REQUESTER_PROFILE = REQUESTER_DOMAIN + "/JCS-SHA256-Ed25519-v1"
CLAIMANT_PROFILE = CLAIMANT_DOMAIN + "/JCS-SHA256-Ed25519-v1"
REQUEST_FIELDS = (
    "schema_version",
    "environment",
    "trust_domain",
    "issuer_target_namespace",
    "entitlement_id",
    "entitlement_generation",
    "logical_operation_id",
    "account_id",
    "canonical_genesis_request_fingerprint_sha256",
    "initial_binding_reference",
    "initial_binding_digest_sha256",
    "requester_id",
    "requester_key_id",
    "requester_key_version",
    "provisioning_principal_id",
    "claimant_key_id",
    "claimant_key_version",
    "issuance_attempt_id",
)


class IssuanceSigningRole(str, Enum):
    REQUESTER = "CHA_ROOT_PROOF_ISSUANCE_REQUESTER_CUSTODY"
    CLAIMANT = "PREACCOUNT_ROOT_PROOF_ENTITLEMENT_CLAIMANT_CUSTODY"

    @property
    def domain(self) -> str:
        return REQUESTER_DOMAIN if self is IssuanceSigningRole.REQUESTER else CLAIMANT_DOMAIN

    @property
    def profile(self) -> str:
        return self.domain + "/JCS-SHA256-Ed25519-v1"

    @property
    def semantic_role(self) -> str:
        return (
            "ACCOUNT_GENESIS_ROOT_PROOF_ISSUANCE_REQUESTER_V1"
            if self is IssuanceSigningRole.REQUESTER
            else "ACCOUNT_GENESIS_ROOT_PROOF_CLAIMANT_V1"
        )


def request_bytes(auth: AttemptAuthorization, attempt_id: str) -> bytes:
    from bot_core.cha_attempt_store import _RPA_ID, AttemptAuthorization, _snapshot_authorization

    if type(auth) is not AttemptAuthorization:
        raise TypeError("exact authorization required")
    auth = _snapshot_authorization(auth)
    if type(attempt_id) is not str or _RPA_ID.fullmatch(attempt_id) is None:
        raise ValueError("exact reserved rpa required")
    if auth.requester_principal_id != "CryptoHunterAccountAuthority":
        raise ValueError("genuine requester principal required")
    if auth.provisioning_principal_id in (auth.account_id, auth.logical_operation_id):
        raise ValueError("pre-account claimant required")
    payload = {
        "schema_version": "1",
        "environment": auth.environment,
        "trust_domain": auth.trust_domain,
        "issuer_target_namespace": ISSUER_TARGET_NAMESPACE,
        "entitlement_id": auth.bootstrap_entitlement_id,
        "entitlement_generation": auth.entitlement_generation,
        "logical_operation_id": auth.logical_operation_id,
        "account_id": auth.account_id,
        "canonical_genesis_request_fingerprint_sha256": (
            auth.canonical_genesis_request_fingerprint_sha256
        ),
        "initial_binding_reference": auth.initial_binding_reference,
        "initial_binding_digest_sha256": auth.initial_binding_digest_sha256,
        "requester_id": auth.requester_principal_id,
        "requester_key_id": auth.requester_key_id,
        "requester_key_version": auth.requester_key_version,
        "provisioning_principal_id": auth.provisioning_principal_id,
        "claimant_key_id": auth.claimant_key_id,
        "claimant_key_version": auth.claimant_key_version,
        "issuance_attempt_id": attempt_id,
    }
    return cast(bytes, canonical_json_bytes(payload))


def request_reference(raw: bytes) -> str:
    if type(raw) is not bytes:
        raise TypeError("exact canonical bytes required")
    parse_canonical(raw)
    return "immutable:req:sha256:" + hashlib.sha256(raw).hexdigest()


def signature_bytes(encoded: str) -> bytes:
    try:
        if type(encoded) is not str or "=" in encoded:
            raise ValueError
        raw = base64.b64decode(encoded + "==", altchars=b"-_", validate=True)
        if len(raw) != 64 or encode_signature(raw) != encoded:
            raise ValueError
        return raw
    except (ValueError, binascii.Error) as exc:
        raise ValueError("canonical unpadded 64-byte Ed25519 signature required") from exc


def encode_signature(raw: bytes) -> str:
    if type(raw) is not bytes or len(raw) != 64:
        raise ValueError("exact 64-byte signature required")
    return base64.urlsafe_b64encode(raw).rstrip(b"=").decode("ascii")


@dataclass(frozen=True, slots=True)
class IssuanceSignerIdentity:
    role: IssuanceSigningRole
    principal: str
    semantic_role: str
    environment: str
    trust_domain: str
    key_id: str
    key_version: int
    public_key_hex: str
    key_material_identity: str
    service_namespace: str
    credential_namespace: str
    lifecycle_namespace: str
    key_handle: str
    lifecycle: str
    lifecycle_generation: int

    def __post_init__(self) -> None:
        if type(self.role) is not IssuanceSigningRole:
            raise TypeError("exact signing role required")
        for name, value in asdict(self).items():
            if name == "role":
                continue
            if name in ("key_version", "lifecycle_generation"):
                if type(value) is not int or not 1 <= value <= 9_007_199_254_740_991:
                    raise ValueError("positive interoperable identity version required")
            elif type(value) is not str or not value.strip():
                raise ValueError("exact signer identity text required")
        public = bytes.fromhex(self.public_key_hex)
        if (
            len(public) != 32
            or public.hex() != self.public_key_hex
            or public_key_material_identity(public) != self.key_material_identity
            or self.environment != "PRODUCTION"
            or self.lifecycle != "ACTIVE"
            or self.semantic_role != self.role.semantic_role
            or (
                self.role is IssuanceSigningRole.REQUESTER
                and self.principal != "CryptoHunterAccountAuthority"
            )
        ):
            raise ValueError("exact ACTIVE production signing identity required")

    def payload(self) -> dict:
        return {**asdict(self), "domain": self.role.domain, "profile": self.role.profile}

    @classmethod
    def from_bytes(cls, raw: bytes) -> IssuanceSignerIdentity:
        data = parse_canonical(raw)
        data["role"] = IssuanceSigningRole(data["role"])
        if data.pop("domain") != data["role"].domain or data.pop("profile") != data["role"].profile:
            raise ValueError("exact retained signing domain/profile required")
        return cls(**data)

    def require_authorization(self, auth: AttemptAuthorization) -> None:
        requester = self.role is IssuanceSigningRole.REQUESTER
        if (
            self.environment != auth.environment
            or self.trust_domain != auth.trust_domain
            or self.principal
            != (auth.requester_principal_id if requester else auth.provisioning_principal_id)
            or self.key_id != (auth.requester_key_id if requester else auth.claimant_key_id)
            or self.key_version
            != (auth.requester_key_version if requester else auth.claimant_key_version)
        ):
            raise ValueError("signer does not match reserved authorization")

    def verify(self, raw: bytes, signature: str) -> None:
        parse_canonical(raw)
        if set(parse_canonical(raw)) != set(REQUEST_FIELDS):
            raise ValueError("closed request field set required")
        try:
            Ed25519PublicKey.from_public_bytes(bytes.fromhex(self.public_key_hex)).verify(
                signature_bytes(signature), self.role.domain.encode("ascii") + b"\x00" + raw
            )
        except InvalidSignature as exc:
            raise ValueError("retained signature verification failed") from exc
