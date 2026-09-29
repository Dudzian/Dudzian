"""PDSAEnrollmentPackageV1 construction; every returned field is signed."""

from __future__ import annotations

import hashlib
from typing import Any

from .activation_request import (
    ActivationRequestV1,
    EnrollmentDecisionV1,
    validate_activation_request_identity,
)
from .canonical import digest

DOMAIN = b"CryptoHunter.Licensing.PDSAEnrollmentPackageV1\x00"


def derive_enrollment_id(request_id: str, license_id: str) -> str:
    return hashlib.sha256(f"{request_id}:{license_id}".encode()).hexdigest()


def unsigned_payload(
    request: ActivationRequestV1,
    decision: EnrollmentDecisionV1,
    key_set_digest: str,
    signer_ids: list[str],
    threshold: int,
) -> dict[str, Any]:
    request_document, decision_document = request.document, decision.document
    validate_activation_request_identity(request_document)
    if decision_document["request_id"] != request_document["request_id"]:
        raise ValueError("decision request mismatch")
    requested = request_document["requested_entitlements"]
    if (decision_document["product"], decision_document["edition"]) != (
        requested["product"],
        requested["edition"],
    ):
        raise ValueError("decision changes requested product")
    if not set(decision_document["features"]).issubset(requested["requested_features"]):
        raise ValueError("unrequested feature")
    return {
        "schema": "PDSAEnrollmentPackageV1",
        "version": 1,
        "enrollment_id": derive_enrollment_id(
            request_document["request_id"], decision_document["license_id"]
        ),
        "issued_at_utc": decision_document["issued_at"],
        "subject": {
            "installation_id": request_document["installation_id"],
            "device_id": request_document["device"]["device_id"],
        },
        "k_psa": {
            "name": request_document["k_psa"]["name"],
            "public_area_digest": digest(request_document["k_psa"]["public_area"]),
        },
        "tpm": {
            key: request_document["tpm"][key]
            for key in ("ek_public_digest", "ak_public_digest", "evidence_profile")
        },
        "release_binding": request_document["release"],
        "license": {
            key: decision_document[key]
            for key in (
                "license_id",
                "product",
                "edition",
                "features",
                "issued_at",
                "expires_at",
                "renewal_after",
            )
        },
        "authority": {
            "pdsa_key_set_digest": key_set_digest,
            "threshold": threshold,
            "signer_key_ids": signer_ids,
        },
        "request_binding": {
            "activation_request_digest": digest(request_document),
            "request_id": request_document["request_id"],
        },
    }
