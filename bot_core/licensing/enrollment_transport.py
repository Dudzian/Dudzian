"""Bounded delivery contract for OFFLINE_FILE and a future ONLINE_API adapter.

Envelopes carry exact existing public protocol bytes. Parsing, content identifiers,
parent links and delivery receipts confer no authority. Installed PDSA and LPPI
boundaries must independently verify signatures, custody, freshness and state.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, cast

from .activation_request import ActivationRequestV1
from .canonical import canonical_json_bytes, parse_canonical
from .device_enrollment import TPMPublicProjectionV1
from .enrollment_file_io import read_artifact_file, write_artifact_file
from .pdsa_enrollment_authorization import PDSAEnrollmentAuthorizationPackageV1
from .pdsa_enrollment_challenge import PDSAEnrollmentChallengeV1
from .pre_enrollment import PreEnrollmentRequestV1
from .production_tpm_custody import (
    K_PSA_PROFILE,
    PROJECTION_PROFILE,
    PROJECTION_SOURCE_PROFILE,
    ProductionPreEnrollmentKeyCustodyEvidenceV1,
    parse_production_tpm_endorsement,
)
from .tpm_attestation import (
    TPMEnrollmentChallengeResponseV1,
    TPMEnrollmentChallengeV1,
    TPMEnrollmentRequestV1,
    _validate_activation_projection_binding,
)

SCHEMA_VERSION = "CryptoHunter.EnrollmentArtifactDelivery.v1"
MAX_ENVELOPE_BYTES = 2_097_152
_HEX = re.compile(r"[0-9a-f]+\Z")
_ID = re.compile(r"[0-9a-f]{64}\Z")
DEVICE_FIELDS = frozenset({"activation_request", "tpm_request"})
CHALLENGE_FIELDS = DEVICE_FIELDS | {"pdsa_challenge", "endorsement", "tpm_challenge"}
PRE_ENROLLMENT_FIELDS = CHALLENGE_FIELDS | {
    "tpm_response",
    "request",
    "request_signature",
    "custody_evidence",
}
STAGE_FIELDS = {
    "DEVICE_REQUEST": DEVICE_FIELDS,
    "TPM_CHALLENGE": CHALLENGE_FIELDS,
    "PRE_ENROLLMENT": PRE_ENROLLMENT_FIELDS,
    "PDSA_PACKAGE": frozenset({"package"}),
}
_LIMITS = {
    "activation_request": 32_768,
    "tpm_request": 131_072,
    "pdsa_challenge": 16_384,
    "endorsement": 131_072,
    "tpm_challenge": 16_384,
    "tpm_response": 131_072,
    "request": 16_384,
    "request_signature": 72,
    "custody_evidence": 131_072,
    "package": 32_768,
}


class EnrollmentTransportError(ValueError):
    """Malformed public delivery, never an authorization decision."""


def _identifier(value: object) -> str:
    if type(value) is not str or _ID.fullmatch(value) is None:
        raise EnrollmentTransportError("INVALID_DELIVERY_ID")
    return value


def _public_syntax(artifacts: dict[str, bytes]) -> None:
    for name, raw in artifacts.items():
        if name != "request_signature":
            document = parse_canonical(raw)
            # Historical public parsers compare version == 1. Require an exact
            # integer here too, so true and 1.0 cannot alias protocol version 1.
            if "version" in document and type(document["version"]) is not int:
                raise EnrollmentTransportError("INVALID_ARTIFACT_VERSION_TYPE")
    if "activation_request" in artifacts:
        activation = ActivationRequestV1.from_mapping(
            parse_canonical(artifacts["activation_request"])
        )
        request = TPMEnrollmentRequestV1.from_canonical_bytes(artifacts["tpm_request"])
        req = request.document
        projection = TPMPublicProjectionV1.verify(req["public_projection"])
        public = projection.document
        if (
            type(public["version"]) is not int
            or public["evidence_profile"] != PROJECTION_PROFILE
            or public["source"]
            != {"substrate": "Windows-TBS", "profile": PROJECTION_SOURCE_PROFILE}
            or public["k_psa"]["algorithm_profile"] != K_PSA_PROFILE
        ):
            raise EnrollmentTransportError("PRODUCTION_PROJECTION_REQUIRED")
        _validate_activation_projection_binding(activation, projection)
        if (
            req["activation_request_id"] != activation.document["request_id"]
            or req["activation_request_digest"]
            != hashlib.sha256(activation.canonical_bytes).hexdigest()
            or req["installation_id"] != activation.document["installation_id"]
            or req["release_policy_digest"]
            != activation.document["release"]["release_policy_digest"]
            or req["requested_entitlements"] != activation.document["requested_entitlements"]
        ):
            raise EnrollmentTransportError("DEVICE_REQUEST_BINDING_MISMATCH")
    if "pdsa_challenge" in artifacts:
        PDSAEnrollmentChallengeV1.from_canonical_bytes(artifacts["pdsa_challenge"])
        parse_production_tpm_endorsement(artifacts["endorsement"])
        challenge = TPMEnrollmentChallengeV1.from_canonical_bytes(artifacts["tpm_challenge"])
        if (
            challenge.document["request_id"] != req["request_id"]
            or challenge.document["public_projection_id"] != public["evidence_id"]
        ):
            raise EnrollmentTransportError("TPM_CHALLENGE_BINDING_MISMATCH")
    if "tpm_response" in artifacts:
        response = TPMEnrollmentChallengeResponseV1.from_canonical_bytes(artifacts["tpm_response"])
        if any(
            response.document[field] != challenge.document[field]
            for field in ("request_id", "challenge_id", "issuer_nonce_hex")
        ):
            raise EnrollmentTransportError("TPM_RESPONSE_BINDING_MISMATCH")
        pre = PreEnrollmentRequestV1.from_canonical_bytes(artifacts["request"])
        pre.verify_signature(artifacts["request_signature"])
        pre.compare_tpm_exchange_bindings(
            activation_request_raw=artifacts["activation_request"],
            request_raw=artifacts["tpm_request"],
            challenge_raw=artifacts["tpm_challenge"],
            response_raw=artifacts["tpm_response"],
        )
        ProductionPreEnrollmentKeyCustodyEvidenceV1.from_canonical_bytes(
            artifacts["custody_evidence"]
        )
    if "package" in artifacts:
        PDSAEnrollmentAuthorizationPackageV1.from_canonical_bytes(artifacts["package"])


@dataclass(frozen=True, init=False)
class ArtifactEnvelope:
    """Immutable exact-byte public delivery, with a content-derived identifier."""

    canonical_bytes: bytes

    def __init__(self, *_: object, **__: object) -> None:
        raise TypeError("use create() or from_canonical_bytes()")

    @classmethod
    def create(
        cls, kind: str, artifacts: dict[str, bytes], *, parent_id: str | None = None
    ) -> ArtifactEnvelope:
        if type(artifacts) is not dict or any(type(raw) is not bytes for raw in artifacts.values()):
            raise EnrollmentTransportError("INVALID_ARTIFACT_BYTES")
        return cls.from_canonical_bytes(
            canonical_json_bytes(
                {
                    "schema_version": SCHEMA_VERSION,
                    "kind": kind,
                    "parent_id": parent_id,
                    "artifacts": {name: raw.hex() for name, raw in artifacts.items()},
                }
            )
        )

    @classmethod
    def from_canonical_bytes(cls, raw: bytes) -> ArtifactEnvelope:
        try:
            if type(raw) is not bytes or not 0 < len(raw) <= MAX_ENVELOPE_BYTES:
                raise EnrollmentTransportError("DELIVERY_SIZE_REJECTED")
            value = parse_canonical(raw)
            if set(value) != {"schema_version", "kind", "parent_id", "artifacts"}:
                raise EnrollmentTransportError("DELIVERY_SCHEMA_MISMATCH")
            kind, encoded = value["kind"], value["artifacts"]
            if (
                value["schema_version"] != SCHEMA_VERSION
                or type(kind) is not str
                or kind not in STAGE_FIELDS
                or type(encoded) is not dict
                or set(encoded) != STAGE_FIELDS[kind]
            ):
                raise EnrollmentTransportError("DELIVERY_SCHEMA_MISMATCH")
            parent = value["parent_id"]
            if kind == "DEVICE_REQUEST":
                if parent is not None:
                    raise EnrollmentTransportError("UNEXPECTED_DELIVERY_PARENT")
            else:
                _identifier(parent)
            artifacts = {}
            for name, encoded_raw in encoded.items():
                if (
                    type(encoded_raw) is not str
                    or not 0 < len(encoded_raw) <= 2 * _LIMITS[name]
                    or len(encoded_raw) % 2
                    or _HEX.fullmatch(encoded_raw) is None
                ):
                    raise EnrollmentTransportError("INVALID_ARTIFACT_ENCODING_OR_SIZE")
                artifacts[name] = bytes.fromhex(encoded_raw)
            _public_syntax(artifacts)
            if kind in {"TPM_CHALLENGE", "PRE_ENROLLMENT"}:
                device = cls.create(
                    "DEVICE_REQUEST", {name: artifacts[name] for name in DEVICE_FIELDS}
                )
                expected = device.request_id
                if kind == "PRE_ENROLLMENT":
                    expected = cls.create(
                        "TPM_CHALLENGE",
                        {name: artifacts[name] for name in CHALLENGE_FIELDS},
                        parent_id=device.request_id,
                    ).request_id
                if parent != expected:
                    raise EnrollmentTransportError("DELIVERY_PARENT_MISMATCH")
        except EnrollmentTransportError:
            raise
        except (
            ValueError,
            TypeError,
            KeyError,
            AttributeError,
            RecursionError,
            OverflowError,
        ) as exc:
            raise EnrollmentTransportError("INVALID_DELIVERY_ARTIFACT") from exc
        result = object.__new__(cls)
        object.__setattr__(result, "canonical_bytes", raw)
        return result

    @property
    def request_id(self) -> str:
        return hashlib.sha256(self.canonical_bytes).hexdigest()

    @property
    def kind(self) -> str:
        return cast(str, parse_canonical(self.canonical_bytes)["kind"])

    @property
    def parent_id(self) -> str | None:
        return cast(str | None, parse_canonical(self.canonical_bytes)["parent_id"])

    @property
    def artifacts(self) -> dict[str, bytes]:
        return {
            name: bytes.fromhex(raw)
            for name, raw in parse_canonical(self.canonical_bytes)["artifacts"].items()
        }

    def preview(self) -> dict[str, object]:
        return {
            "request_id": self.request_id,
            "kind": self.kind,
            "parent_id": self.parent_id,
            "artifacts_sha256": {
                name: hashlib.sha256(raw).hexdigest() for name, raw in self.artifacts.items()
            },
            "authorization": "NOT_ESTABLISHED_BY_TRANSPORT",
        }


class ArtifactDelivery(Protocol):
    """Delivery only. ONLINE_API can implement this unchanged byte contract."""

    def deliver(self, envelope: ArtifactEnvelope) -> str: ...

    def receive(self, request_id: str) -> ArtifactEnvelope: ...


@dataclass(frozen=True)
class OfflineFileDelivery:
    directory: Path

    def deliver(self, envelope: ArtifactEnvelope) -> str:
        checked = ArtifactEnvelope.from_canonical_bytes(envelope.canonical_bytes)
        write_artifact_file(
            self.directory / (checked.request_id + ".json"), checked.canonical_bytes
        )
        return checked.request_id

    def receive(self, request_id: str) -> ArtifactEnvelope:
        identity = _identifier(request_id)
        envelope = ArtifactEnvelope.from_canonical_bytes(
            read_artifact_file(self.directory / (identity + ".json"))
        )
        if envelope.request_id != identity:
            raise EnrollmentTransportError("DELIVERY_ID_MISMATCH")
        return envelope
