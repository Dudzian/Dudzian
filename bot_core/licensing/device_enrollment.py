"""Platform-neutral construction and verification of public TPM enrollment bundles."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

from .activation_request import ActivationRequestV1
from .canonical import canonical_json_bytes, digest, exact

PRODUCTION_STOP = "STOP — PRODUCTION CEREMONY NOT COMPLETE FOR DEVICE ENROLLMENT."
NAME_STOP = "STOP — K_PSA TPM NAME VERIFICATION FAILED."
SECRET_MARKERS = (
    b"BEGIN PRIVATE KEY",
    b"BEGIN ENCRYPTED PRIVATE KEY",
    b"PRIVATE KEY",
    b"private_scalar",
    b"password",
    b"passphrase",
    b"private_blob",
)


def _name(public_hex: str) -> str:
    raw = bytes.fromhex(public_hex)
    if raw.hex() != public_hex:
        raise ValueError("noncanonical TPMT_PUBLIC")
    return "000b" + hashlib.sha256(raw).hexdigest()


def _body(value: Mapping[str, Any]) -> dict[str, Any]:
    result = dict(value)
    result.pop("evidence_id", None)
    return result


@dataclass(frozen=True, init=False)
class TPMPublicProjectionV1:
    """Immutable public projection accepted only after an independent Name check."""

    canonical_bytes: bytes

    def __init__(self, *_: object, **__: object) -> None:
        raise TypeError("use verify() or from_canonical_bytes()")

    @classmethod
    def _new(cls, raw: bytes) -> "TPMPublicProjectionV1":
        item = object.__new__(cls)
        object.__setattr__(item, "canonical_bytes", raw)
        return item

    @property
    def document(self) -> dict[str, Any]:
        value = json.loads(self.canonical_bytes)
        assert isinstance(value, dict)
        return value

    @property
    def evidence_reference(self) -> str:
        return self.document["evidence_id"]

    @classmethod
    def verify(cls, document: Mapping[str, Any]) -> "TPMPublicProjectionV1":
        value = json.loads(canonical_json_bytes(dict(document)))
        exact(
            value,
            {
                "schema",
                "version",
                "evidence_profile",
                "k_psa",
                "ek",
                "ak",
                "tpm",
                "source",
                "evidence_id",
            },
            "TPM evidence",
        )
        if value["schema"] != "WindowsTPMPublicProjectionV1" or value["version"] != 1:
            raise ValueError("unsupported TPM evidence")
        exact(
            value["k_psa"],
            {"public_area", "name", "algorithm_profile", "creation_hash"},
            "k_psa",
        )
        exact(value["k_psa"]["public_area"], {"hex"}, "TPMT_PUBLIC")
        for role in ("ek", "ak"):
            fields = {"public_area", "name", "public_digest"}
            if role == "ek":
                fields |= {
                    "manufacturer_certificate",
                    "manufacturer_certificate_digest",
                }
            exact(value[role], fields, role)
            exact(value[role]["public_area"], {"hex"}, f"{role} TPMT_PUBLIC")
        exact(value["tpm"], {"manufacturer", "model"}, "tpm")
        exact(value["source"], {"substrate", "profile"}, "source")
        public_hex = value["k_psa"]["public_area"]["hex"]
        try:
            computed = _name(public_hex)
        except (TypeError, ValueError) as exc:
            raise ValueError("invalid TPMT_PUBLIC") from exc
        if value["k_psa"]["name"] != computed:
            raise ValueError(NAME_STOP)
        _hex_fields = [value["k_psa"]["creation_hash"]]
        for role in ("ek", "ak"):
            public = bytes.fromhex(value[role]["public_area"]["hex"])
            if value[role]["name"] != "000b" + hashlib.sha256(public).hexdigest():
                raise ValueError(f"{role} TPM Name mismatch")
            if value[role]["public_digest"] != hashlib.sha256(public).hexdigest():
                raise ValueError(f"{role} public digest mismatch")
            _hex_fields.append(value[role]["public_digest"])
        if value["ek"]["manufacturer_certificate"] not in {
            "AVAILABLE",
            "NOT_AVAILABLE",
        }:
            raise ValueError("invalid EK manufacturer certificate status")
        certificate_digest = value["ek"]["manufacturer_certificate_digest"]
        if (certificate_digest is None) != (
            value["ek"]["manufacturer_certificate"] == "NOT_AVAILABLE"
        ):
            raise ValueError("EK certificate status/digest mismatch")
        if certificate_digest is not None:
            _hex_fields.append(certificate_digest)
        for field in _hex_fields:
            if not isinstance(field, str) or len(field) != 64:
                raise ValueError("invalid public digest")
            bytes.fromhex(field)
        expected = digest(_body(value))
        if value["evidence_id"] != expected:
            raise ValueError("evidence_id mismatch")
        return cls._new(canonical_json_bytes(value))

    @classmethod
    def from_canonical_bytes(cls, raw: bytes) -> "TPMPublicProjectionV1":
        value = json.loads(raw)
        if canonical_json_bytes(value) != raw:
            raise ValueError("noncanonical evidence")
        return cls.verify(value)


def make_evidence(
    *,
    public_area_hex: str,
    returned_name: str,
    creation_hash: str,
    ek_public_area_hex: str,
    ek_name: str,
    ak_public_area_hex: str,
    ak_name: str,
    algorithm_profile: str,
    evidence_profile: str,
    substrate_profile: str,
    manufacturer: str | None = None,
    model: str | None = None,
    ek_certificate_digest: str | None = None,
) -> TPMPublicProjectionV1:
    if returned_name != _name(public_area_hex):
        raise ValueError(NAME_STOP)
    body = {
        "schema": "WindowsTPMPublicProjectionV1",
        "version": 1,
        "evidence_profile": evidence_profile,
        "k_psa": {
            "public_area": {"hex": public_area_hex},
            "name": returned_name,
            "algorithm_profile": algorithm_profile,
            "creation_hash": creation_hash,
        },
        "ek": {
            "public_area": {"hex": ek_public_area_hex},
            "name": ek_name,
            "public_digest": hashlib.sha256(
                bytes.fromhex(ek_public_area_hex)
            ).hexdigest(),
            "manufacturer_certificate": (
                "AVAILABLE" if ek_certificate_digest is not None else "NOT_AVAILABLE"
            ),
            "manufacturer_certificate_digest": ek_certificate_digest,
        },
        "ak": {
            "public_area": {"hex": ak_public_area_hex},
            "name": ak_name,
            "public_digest": hashlib.sha256(
                bytes.fromhex(ak_public_area_hex)
            ).hexdigest(),
        },
        "tpm": {"manufacturer": manufacturer, "model": model},
        "source": {"substrate": "Windows-TBS", "profile": substrate_profile},
    }
    return TPMPublicProjectionV1.verify({**body, "evidence_id": digest(body)})


def derive_device_id(evidence: TPMPublicProjectionV1) -> str:
    value = evidence.document
    identity = {
        "k_psa_name": value["k_psa"]["name"],
        "ek_public_digest": value["ek"]["public_digest"],
        "ak_public_digest": value["ak"]["public_digest"],
    }
    return digest(identity)


def build_activation_request(
    *,
    evidence: TPMPublicProjectionV1,
    release_policy_digest: str,
    release_policy_version: int,
    requested_entitlements: dict[str, Any],
    installation_id: str,
    architecture: str,
    environment: str,
    production_trust_context: object | None = None,
    created_at_utc: str | None = None,
    nonce: str | None = None,
) -> ActivationRequestV1:
    if environment == "PRODUCTION":
        from deployment.windows_stage9_production_trust import (
            require_verified_production_trust_context,
        )

        try:
            production_trust_context = require_verified_production_trust_context(
                production_trust_context
            )
        except RuntimeError as exc:
            raise RuntimeError(PRODUCTION_STOP) from exc
        if (
            release_policy_digest != production_trust_context.release_payload_digest
            or release_policy_version != production_trust_context.release_version
        ):
            raise RuntimeError("PRODUCTION_RELEASE_POLICY_MISMATCH")
    if environment not in ("TEST_ONLY", "PRODUCTION"):
        raise ValueError("environment must be TEST_ONLY or PRODUCTION")
    value = evidence.document
    return ActivationRequestV1.create(
        created_at_utc=created_at_utc
        or datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        installation_id=installation_id,
        device={
            "device_id": derive_device_id(evidence),
            "platform": "Windows",
            "architecture": architecture,
        },
        tpm={
            "evidence_profile": value["evidence_profile"],
            "ek_public_digest": value["ek"]["public_digest"],
            "ak_public_digest": value["ak"]["public_digest"],
            "evidence_reference": evidence.evidence_reference,
            "manufacturer": value["tpm"]["manufacturer"],
            "model": value["tpm"]["model"],
        },
        k_psa={
            key: value["k_psa"][key]
            for key in ("public_area", "name", "algorithm_profile")
        },
        release={
            "release_policy_digest": release_policy_digest,
            "release_policy_version": release_policy_version,
        },
        requested_entitlements=requested_entitlements,
        nonce=nonce,
    )


def verify_activation_request_bundle(
    request_raw: bytes,
    evidence_raw: bytes,
    *,
    environment: str = "TEST_ONLY",
    allowed_release: tuple[str, int] | None = None,
    production_trust_context: object | None = None,
) -> tuple[ActivationRequestV1, TPMPublicProjectionV1]:
    request_value = json.loads(request_raw)
    if canonical_json_bytes(request_value) != request_raw:
        raise ValueError("noncanonical activation request")
    request = ActivationRequestV1.from_mapping(request_value)
    evidence = TPMPublicProjectionV1.from_canonical_bytes(evidence_raw)
    req, ev = request.document, evidence.document
    if environment == "TEST_ONLY":
        if allowed_release is None or production_trust_context is not None:
            raise ValueError("TEST_ONLY verification requires only allowed_release")
        release = allowed_release
    elif environment == "PRODUCTION":
        from deployment.windows_stage9_production_trust import (
            require_verified_production_trust_context,
        )

        if allowed_release is not None:
            raise ValueError("caller-supplied production release is forbidden")
        context = require_verified_production_trust_context(production_trust_context)
        release = (context.release_payload_digest, context.release_version)
    else:
        raise ValueError("environment must be TEST_ONLY or PRODUCTION")
    expected = build_activation_request(
        evidence=evidence,
        release_policy_digest=release[0],
        release_policy_version=release[1],
        requested_entitlements=req["requested_entitlements"],
        installation_id=req["installation_id"],
        architecture=req["device"]["architecture"],
        environment=environment,
        production_trust_context=production_trust_context,
        created_at_utc=req["created_at_utc"],
        nonce=req["nonce"],
    ).document
    for field in ("device", "tpm", "k_psa", "release"):
        if req[field] != expected[field]:
            raise ValueError(f"bundle {field} binding mismatch")
    return request, evidence


def export_bundle(
    output: Path,
    request: ActivationRequestV1,
    evidence: TPMPublicProjectionV1,
    *,
    environment: str = "TEST_ONLY",
    allowed_release: tuple[str, int] | None = None,
    production_trust_context: object | None = None,
) -> Path:
    verify_activation_request_bundle(
        request.canonical_bytes,
        evidence.canonical_bytes,
        environment=environment,
        allowed_release=allowed_release,
        production_trust_context=production_trust_context,
    )
    folder = (
        output
        / f"CryptoHunter-Activation-Request-{request.document['request_id'][:12]}"
    )
    folder.mkdir(parents=True, exist_ok=False)
    files = {
        "activation-request.json": request.canonical_bytes,
        "tpm-evidence.json": evidence.canonical_bytes,
    }
    for name, raw in files.items():
        lowered = raw.lower()
        if any(marker.lower() in lowered for marker in SECRET_MARKERS):
            raise ValueError("private material marker in public bundle")
        (folder / name).write_bytes(raw)
    verify_activation_request_bundle(
        (folder / "activation-request.json").read_bytes(),
        (folder / "tpm-evidence.json").read_bytes(),
        environment=environment,
        allowed_release=allowed_release,
        production_trust_context=production_trust_context,
    )
    return folder
