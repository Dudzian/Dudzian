"""Strict, fail-closed and network-independent enrollment verification."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Callable

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey

from .activation_request import (
    ID,
    ActivationRequestV1,
    require_digest,
    require_string_set,
    require_text,
    require_timestamp,
    validate_activation_request_identity,
)
from .authority import public_authority_projection
from .canonical import canonical_json_bytes, digest, exact, parse_canonical
from .enrollment import DOMAIN, derive_enrollment_id

K_PSA_NAME = re.compile(r"^000b[0-9a-f]{64}$")
SIGNATURE = re.compile(r"^[0-9a-f]{128}$")
ED25519_PUBLIC_KEY = re.compile(r"^[0-9a-f]{64}$")


@dataclass(frozen=True)
class LocalIdentityV1:
    installation_id: str
    device_id: str
    k_psa_name: str
    k_psa_public_area: Any
    ek_public_digest: str
    ak_public_digest: str
    evidence_profile: str
    release_policy_digest: str
    release_policy_version: int


@dataclass(frozen=True, init=False)
class VerifiedEnrollmentV1:
    enrollment_id: str
    license_id: str
    features: tuple[str, ...]
    expires_at: str | None
    canonical_bytes: bytes

    def __init__(self, *_: object, **__: object) -> None:
        raise TypeError("VerifiedEnrollmentV1 is created only by verification")

    @classmethod
    def _create(
        cls,
        enrollment_id: str,
        license_id: str,
        features: tuple[str, ...],
        expires_at: str | None,
        canonical: bytes,
    ) -> "VerifiedEnrollmentV1":
        instance = object.__new__(cls)
        object.__setattr__(instance, "enrollment_id", enrollment_id)
        object.__setattr__(instance, "license_id", license_id)
        object.__setattr__(instance, "features", features)
        object.__setattr__(instance, "expires_at", expires_at)
        object.__setattr__(instance, "canonical_bytes", canonical)
        return instance

    @property
    def package(self) -> dict[str, Any]:
        """Return a defensive projection; verified canonical bytes cannot be mutated."""
        value = json.loads(self.canonical_bytes)
        assert isinstance(value, dict)
        return value


class EnrollmentVerificationError(ValueError):
    def __init__(self, code: str) -> None:
        self.code = code
        super().__init__(code)


def _fail(code: str) -> None:
    raise EnrollmentVerificationError(code)


def _validate_authority_profile(trusted_public_keys: dict[str, str]) -> None:
    """Enforce the complete PDSA 2-of-3 public authority profile."""
    if len(trusted_public_keys) != 3:
        _fail("AUTHORITY_PROFILE_MISMATCH")
    public_key_bytes: list[bytes] = []
    for key_id, public_key_hex in trusted_public_keys.items():
        if not isinstance(key_id, str) or not ID.fullmatch(key_id):
            _fail("AUTHORITY_PROFILE_MISMATCH")
        if not isinstance(public_key_hex, str) or not ED25519_PUBLIC_KEY.fullmatch(public_key_hex):
            _fail("AUTHORITY_PROFILE_MISMATCH")
        try:
            raw_key = bytes.fromhex(public_key_hex)
            Ed25519PublicKey.from_public_bytes(raw_key)
        except ValueError:
            _fail("AUTHORITY_PROFILE_MISMATCH")
        public_key_bytes.append(raw_key)
    if len(set(public_key_bytes)) != 3:
        _fail("AUTHORITY_PROFILE_MISMATCH")


def _validate_schema(package: dict[str, Any]) -> None:
    try:
        exact(
            package,
            {
                "schema",
                "version",
                "enrollment_id",
                "issued_at_utc",
                "subject",
                "k_psa",
                "tpm",
                "release_binding",
                "license",
                "authority",
                "request_binding",
                "signature_block",
            },
            "package",
        )
        exact(package["subject"], {"installation_id", "device_id"}, "subject")
        exact(package["k_psa"], {"name", "public_area_digest"}, "k_psa")
        exact(package["tpm"], {"ek_public_digest", "ak_public_digest", "evidence_profile"}, "tpm")
        exact(
            package["release_binding"],
            {"release_policy_digest", "release_policy_version"},
            "release_binding",
        )
        exact(
            package["license"],
            {
                "license_id",
                "product",
                "edition",
                "features",
                "issued_at",
                "expires_at",
                "renewal_after",
            },
            "license",
        )
        exact(
            package["authority"],
            {"pdsa_key_set_digest", "threshold", "signer_key_ids"},
            "authority",
        )
        exact(
            package["request_binding"],
            {"activation_request_digest", "request_id"},
            "request_binding",
        )
        if not isinstance(package["signature_block"], list):
            raise ValueError("signature_block")
        for signature in package["signature_block"]:
            exact(
                signature,
                {"algorithm", "key_id", "authority_key_set_digest", "signature"},
                "signature",
            )
        if package["schema"] != "PDSAEnrollmentPackageV1" or package["version"] != 1:
            raise ValueError("schema")
        require_digest(package["enrollment_id"], "enrollment_id")
        require_timestamp(package["issued_at_utc"], "issued_at_utc")
        for name in ("installation_id", "device_id"):
            require_text(package["subject"][name], name)
        if not isinstance(package["k_psa"]["name"], str) or not K_PSA_NAME.fullmatch(
            package["k_psa"]["name"]
        ):
            raise ValueError("K_PSA.Name")
        require_digest(package["k_psa"]["public_area_digest"], "public_area_digest")
        require_digest(package["tpm"]["ek_public_digest"], "ek_public_digest")
        require_digest(package["tpm"]["ak_public_digest"], "ak_public_digest")
        require_text(package["tpm"]["evidence_profile"], "evidence_profile")
        require_digest(package["release_binding"]["release_policy_digest"], "release digest")
        version = package["release_binding"]["release_policy_version"]
        if not isinstance(version, int) or isinstance(version, bool):
            raise ValueError("release version")
        license_document = package["license"]
        for name in ("license_id", "product", "edition"):
            require_text(license_document[name], name)
        require_string_set(license_document["features"], "features")
        require_timestamp(license_document["issued_at"], "issued_at")
        require_timestamp(license_document["expires_at"], "expires_at", optional=True)
        require_timestamp(license_document["renewal_after"], "renewal_after", optional=True)
        authority = package["authority"]
        require_digest(authority["pdsa_key_set_digest"], "PDSA digest")
        if not isinstance(authority["threshold"], int) or isinstance(authority["threshold"], bool):
            raise ValueError("threshold")
        require_string_set(authority["signer_key_ids"], "signer_key_ids")
        require_digest(package["request_binding"]["activation_request_digest"], "request digest")
        require_digest(package["request_binding"]["request_id"], "request_id")
        for signature in package["signature_block"]:
            if signature["algorithm"] != "Ed25519":
                raise ValueError("algorithm")
            require_text(signature["key_id"], "key_id")
            require_digest(signature["authority_key_set_digest"], "authority digest")
            if not isinstance(signature["signature"], str) or not SIGNATURE.fullmatch(
                signature["signature"]
            ):
                raise ValueError("signature")
    except (KeyError, TypeError, ValueError):
        _fail("INVALID_SCHEMA")


def verify_pdsa_enrollment_package(
    raw: bytes,
    *,
    request: ActivationRequestV1,
    identity: LocalIdentityV1,
    trusted_public_keys: dict[str, str],
    expected_key_set_digest: str,
    trusted_time: Callable[[], datetime] | None = None,
) -> VerifiedEnrollmentV1:
    try:
        package = parse_canonical(raw)
    except ValueError:
        _fail("NONCANONICAL_OR_INVALID_SCHEMA")
    try:
        expiry_candidate = package["license"]["expires_at"]
        require_timestamp(expiry_candidate, "expires_at", optional=True)
    except (KeyError, TypeError):
        pass  # General strict schema validation below provides the stable schema code.
    except ValueError:
        _fail("INVALID_EXPIRY")
    _validate_schema(package)
    try:
        validate_activation_request_identity(request.document)
    except ValueError:
        _fail("REQUEST_BINDING_MISMATCH")

    unsigned = dict(package)
    signatures = unsigned.pop("signature_block")
    authority = package["authority"]
    _validate_authority_profile(trusted_public_keys)
    derived_key_set_digest = digest(public_authority_projection(trusted_public_keys))
    if (
        derived_key_set_digest != expected_key_set_digest
        or authority["pdsa_key_set_digest"] != derived_key_set_digest
    ):
        _fail("AUTHORITY_MISMATCH")
    if authority["threshold"] != 2:
        _fail("AUTHORITY_MISMATCH")
    signer_ids = [signature["key_id"] for signature in signatures]
    if len(signer_ids) != len(set(signer_ids)):
        _fail("DUPLICATE_SIGNER_ID")
    if sorted(authority["signer_key_ids"]) != sorted(signer_ids):
        _fail("SIGNER_IDS_MISMATCH")
    message = DOMAIN + hashlib.sha256(canonical_json_bytes(unsigned)).digest()
    valid = 0
    for signature in signatures:
        key_id = signature["key_id"]
        if key_id not in trusted_public_keys:
            _fail("UNKNOWN_SIGNER")
        if signature["authority_key_set_digest"] != derived_key_set_digest:
            _fail("AUTHORITY_MISMATCH")
        try:
            Ed25519PublicKey.from_public_bytes(bytes.fromhex(trusted_public_keys[key_id])).verify(
                bytes.fromhex(signature["signature"]), message
            )
        except (ValueError, InvalidSignature):
            _fail("SIGNATURE_INVALID")
        valid += 1
    if valid < authority["threshold"]:
        _fail("THRESHOLD_NOT_MET")

    request_document = request.document
    expected_binding = {
        "activation_request_digest": digest(request_document),
        "request_id": request_document["request_id"],
    }
    if package["request_binding"] != expected_binding:
        _fail("REQUEST_BINDING_MISMATCH")
    expected_enrollment_id = derive_enrollment_id(
        request_document["request_id"], package["license"]["license_id"]
    )
    if package["enrollment_id"] != expected_enrollment_id:
        _fail("ENROLLMENT_ID_MISMATCH")
    if package["issued_at_utc"] != package["license"]["issued_at"]:
        _fail("ISSUED_AT_MISMATCH")
    if package["subject"]["installation_id"] != identity.installation_id:
        _fail("INSTALLATION_MISMATCH")
    if package["subject"]["device_id"] != identity.device_id:
        _fail("DEVICE_MISMATCH")
    if package["k_psa"]["name"] != identity.k_psa_name:
        _fail("K_PSA_NAME_MISMATCH")
    if package["k_psa"]["public_area_digest"] != digest(identity.k_psa_public_area):
        _fail("K_PSA_PUBLIC_AREA_MISMATCH")
    expected_tpm = {
        "ek_public_digest": identity.ek_public_digest,
        "ak_public_digest": identity.ak_public_digest,
        "evidence_profile": identity.evidence_profile,
    }
    if package["tpm"] != expected_tpm:
        _fail("TPM_EVIDENCE_MISMATCH")
    expected_release = {
        "release_policy_digest": identity.release_policy_digest,
        "release_policy_version": identity.release_policy_version,
    }
    if package["release_binding"] != expected_release:
        _fail("RELEASE_POLICY_MISMATCH")

    expires_at = package["license"]["expires_at"]
    if expires_at is not None:
        try:
            expiry = datetime.strptime(expires_at, "%Y-%m-%dT%H:%M:%SZ").replace(
                tzinfo=timezone.utc
            )
        except (TypeError, ValueError):
            _fail("INVALID_EXPIRY")
        if trusted_time is None:
            _fail("TRUSTED_FRESHNESS_REQUIRED")
        try:
            now = trusted_time()
            if now.tzinfo is None:
                _fail("INVALID_FRESHNESS")
            # Canonical timestamps are UTC Z; normalize the parsed value explicitly.
            if now.astimezone(timezone.utc) >= expiry:
                _fail("EXPIRED")
        except EnrollmentVerificationError:
            raise
        except (TypeError, ValueError):
            _fail("INVALID_FRESHNESS")
    return VerifiedEnrollmentV1._create(
        package["enrollment_id"],
        package["license"]["license_id"],
        tuple(package["license"]["features"]),
        expires_at,
        raw,
    )
