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
        signer_ids = list(self._keys)[:2]
        payload = unsigned_payload(request, decision, self.key_set_digest, signer_ids, 2)
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
