"""Transport abstractions and deliberately TEST_ONLY 2-of-3 authority."""

from __future__ import annotations

import hashlib
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Mapping

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from .activation_request import ActivationRequestV1, EnrollmentDecisionV1
from .canonical import canonical_json_bytes, digest
from .enrollment import DOMAIN, unsigned_payload
from .tpm_attestation import (
    PendingChallengeStore,
    ProductionTPMAttestationVerifier,
    TPMEnrollmentChallengeV1,
    TestOnlyTPMAttestationVerifier,
)

PRODUCTION_PATH_MARKER = "cryptohunter-production-authority"
_TEST_ONLY_PROVENANCE = "TEST_ONLY"


def public_authority_projection(public_keys: Mapping[str, str]) -> list[dict[str, str]]:
    return [
        {"algorithm": "Ed25519", "key_id": key_id, "public_key_hex": public_keys[key_id]}
        for key_id in sorted(public_keys)
    ]


class EnrollmentAuthority(ABC):
    @abstractmethod
    def issue(
        self, request: ActivationRequestV1, decision: EnrollmentDecisionV1
    ) -> dict[str, Any]: ...


class EnrollmentTransport(ABC):
    """Future HTTP boundary; implementations exchange these exact contracts only."""

    @abstractmethod
    def enroll(self, request: ActivationRequestV1) -> dict[str, Any]: ...


class TestOnlyPDSAAuthority:
    """Non-production signer constructible only through explicit TEST_ONLY factories."""

    __test__ = False

    def __init__(self, *_: object, **__: object) -> None:
        raise TypeError("use deterministic_fixture() or from_test_material(TEST_ONLY, ...)")

    @classmethod
    def from_test_material(
        cls,
        provenance: str,
        keys: Mapping[str, Ed25519PrivateKey],
        *,
        material_path: str | Path | None = None,
    ) -> "TestOnlyPDSAAuthority":
        if provenance != _TEST_ONLY_PROVENANCE:
            raise ValueError("TEST_ONLY provenance is required")
        if material_path is not None and PRODUCTION_PATH_MARKER in str(Path(material_path)).lower():
            raise ValueError(
                "STOP — PRODUCTION PDSA CEREMONY MATERIAL NOT ACTIVATED FOR LICENSING."
            )
        ordered = dict(sorted(keys.items()))
        if len(ordered) != 3 or any(not key_id.startswith("TEST_ONLY_") for key_id in ordered):
            raise ValueError("TEST_ONLY PDSA requires a three-key TEST_ONLY-labelled set")
        instance = object.__new__(cls)
        instance._keys = ordered
        instance.public_keys = {
            key_id: key.public_key()
            .public_bytes(serialization.Encoding.Raw, serialization.PublicFormat.Raw)
            .hex()
            for key_id, key in ordered.items()
        }
        instance.key_set_digest = digest(public_authority_projection(instance.public_keys))
        return instance

    @classmethod
    def deterministic_fixture(cls) -> "TestOnlyPDSAAuthority":
        keys = {
            f"TEST_ONLY_PDSA_{index}": Ed25519PrivateKey.from_private_bytes(
                hashlib.sha256(f"CryptoHunter TEST_ONLY PDSA {index}".encode()).digest()
            )
            for index in range(1, 4)
        }
        return cls.from_test_material(_TEST_ONLY_PROVENANCE, keys)

    def issue(self, request: ActivationRequestV1, decision: EnrollmentDecisionV1) -> dict[str, Any]:
        if request.document["tpm"]["evidence_profile"].startswith("WindowsTPM2-TBS-"):
            raise ValueError("ATTESTED_EXCHANGE_ISSUER_REQUIRED")
        signer_ids = list(self._keys)[:2]
        payload = unsigned_payload(request, decision, self.key_set_digest, signer_ids, 2)
        return self._sign_unsigned_payload(payload, signer_ids)

    def _sign_unsigned_payload(
        self, payload: dict[str, Any], signer_ids: list[str]
    ) -> dict[str, Any]:
        message = DOMAIN + hashlib.sha256(canonical_json_bytes(payload)).digest()
        result = dict(payload)
        result["signature_block"] = [
            {
                "algorithm": "Ed25519",
                "key_id": key_id,
                "authority_key_set_digest": self.key_set_digest,
                "signature": self._keys[key_id].sign(message).hex(),
            }
            for key_id in signer_ids
        ]
        return result


class OfflineEnrollmentAuthority(EnrollmentAuthority):
    def __init__(self, signer: TestOnlyPDSAAuthority) -> None:
        self.signer = signer

    def issue(self, request: ActivationRequestV1, decision: EnrollmentDecisionV1) -> dict[str, Any]:
        return self.signer.issue(request, decision)


class MockOnlineEnrollmentAuthority(OfflineEnrollmentAuthority):
    """Test double proving online automation cannot alter signed semantics."""


class AttestedEnrollmentIssuer:
    """Consequential boundary: always reverifies canonical exchange bytes before signing."""

    def __init__(self, signer: TestOnlyPDSAAuthority) -> None:
        self._signer = signer
        self._verifier = ProductionTPMAttestationVerifier()
        self.last_exchange_reference: str | None = None

    def issue(
        self,
        *,
        activation_request_raw: bytes,
        decision: EnrollmentDecisionV1,
        enrollment_request_raw: bytes,
        challenge_raw: bytes,
        response_raw: bytes,
        pending: PendingChallengeStore,
        expected_release_policy_digest: str,
    ) -> dict[str, Any]:
        request = _canonical_activation_request(activation_request_raw)
        if decision.document["request_id"] != request.document["request_id"]:
            raise ValueError("DECISION_ACTIVATION_REQUEST_MISMATCH")
        signer_ids = list(self._signer._keys)[:2]
        payload = unsigned_payload(request, decision, self._signer.key_set_digest, signer_ids, 2)
        verified = self._verifier.verify(
            activation_request_raw,
            enrollment_request_raw,
            challenge_raw,
            response_raw,
            pending=pending,
            expected_release_policy_digest=expected_release_policy_digest,
        )
        package = self._signer._sign_unsigned_payload(payload, signer_ids)
        challenge = TPMEnrollmentChallengeV1.from_canonical_bytes(challenge_raw)
        pending.consume(challenge)
        self.last_exchange_reference = verified.exchange_reference
        return package


class TestOnlyAttestedEnrollmentIssuer(AttestedEnrollmentIssuer):
    """Dedicated harness for exchange/signing regressions; never a production verifier."""

    __test__ = False

    def __init__(self, signer: TestOnlyPDSAAuthority) -> None:
        self._signer = signer
        self._verifier = TestOnlyTPMAttestationVerifier()
        self.last_exchange_reference = None


def _canonical_activation_request(raw: bytes) -> ActivationRequestV1:
    import json

    value = json.loads(raw)
    if canonical_json_bytes(value) != raw:
        raise ValueError("noncanonical activation request")
    return ActivationRequestV1.from_mapping(value)
