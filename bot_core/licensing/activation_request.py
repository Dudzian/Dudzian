"""Immutable canonical activation-request and operator-decision contracts."""

from __future__ import annotations

import json
import re
import secrets
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Mapping

from .canonical import canonical_json_bytes, digest, exact

HEX64 = re.compile(r"^[0-9a-f]{64}$")
ID = re.compile(r"^[A-Za-z0-9._-]{1,128}$")
UTC_TIMESTAMP = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z$")


def require_text(value: Any, name: str, *, identifier: bool = False) -> str:
    if not isinstance(value, str) or not value or (identifier and not ID.fullmatch(value)):
        raise ValueError(f"invalid {name}")
    return value


def require_digest(value: Any, name: str) -> str:
    if not isinstance(value, str) or not HEX64.fullmatch(value):
        raise ValueError(f"invalid {name}")
    return value


def require_timestamp(value: Any, name: str, *, optional: bool = False) -> str | None:
    if value is None and optional:
        return None
    if not isinstance(value, str) or not UTC_TIMESTAMP.fullmatch(value):
        raise ValueError(f"invalid {name}: expected YYYY-MM-DDTHH:MM:SSZ")
    try:
        datetime.strptime(value, "%Y-%m-%dT%H:%M:%SZ")
    except ValueError as exc:
        raise ValueError(f"invalid {name}") from exc
    return value


def require_string_set(value: Any, name: str) -> tuple[str, ...]:
    if not isinstance(value, list) or any(not isinstance(item, str) or not item for item in value):
        raise ValueError(f"invalid {name}")
    if value != sorted(set(value)):
        raise ValueError(f"{name} must be a sorted unique canonical set")
    return tuple(value)


def _projection(source: bytes) -> dict[str, Any]:
    value = json.loads(source)
    assert isinstance(value, dict)
    return value


def validate_activation_request_identity(document: Mapping[str, Any]) -> None:
    body = dict(document)
    request_id = body.pop("request_id", None)
    require_digest(request_id, "request_id")
    if digest(body) != request_id:
        raise ValueError("request_id mismatch")


@dataclass(frozen=True, init=False)
class ActivationRequestV1:
    canonical_bytes: bytes

    def __init__(self, *_: object, **__: object) -> None:
        raise TypeError("use create() or from_mapping()")

    @classmethod
    def _from_canonical(cls, canonical: bytes) -> "ActivationRequestV1":
        instance = object.__new__(cls)
        object.__setattr__(instance, "canonical_bytes", canonical)
        return instance

    @property
    def document(self) -> dict[str, Any]:
        """Return a defensive deep projection; canonical bytes remain authoritative."""
        return _projection(self.canonical_bytes)

    @classmethod
    def create(
        cls,
        *,
        created_at_utc: str,
        installation_id: str,
        device: dict[str, Any],
        tpm: dict[str, Any],
        k_psa: dict[str, Any],
        release: dict[str, Any],
        requested_entitlements: dict[str, Any],
        nonce: str | None = None,
    ) -> "ActivationRequestV1":
        body = {
            "schema": "CryptoHunterActivationRequestV1",
            "version": 1,
            "request_id": "pending",
            "created_at_utc": created_at_utc,
            "installation_id": installation_id,
            "device": device,
            "tpm": tpm,
            "k_psa": k_psa,
            "release": release,
            "requested_entitlements": requested_entitlements,
            "nonce": nonce or secrets.token_hex(32),
        }
        identity = dict(body)
        identity.pop("request_id")
        body["request_id"] = digest(identity)
        return cls.from_mapping(body)

    @classmethod
    def from_mapping(cls, document: Mapping[str, Any]) -> "ActivationRequestV1":
        # Canonical round-trip is the defensive deep copy and immutable source of truth.
        canonical = canonical_json_bytes(dict(document))
        value = _projection(canonical)
        exact(
            value,
            {
                "schema",
                "version",
                "request_id",
                "created_at_utc",
                "installation_id",
                "device",
                "tpm",
                "k_psa",
                "release",
                "requested_entitlements",
                "nonce",
            },
            "request",
        )
        if value["schema"] != "CryptoHunterActivationRequestV1" or value["version"] != 1:
            raise ValueError("unsupported activation request")
        exact(value["device"], {"device_id", "platform", "architecture"}, "device")
        exact(
            value["tpm"],
            {
                "evidence_profile",
                "ek_public_digest",
                "ak_public_digest",
                "evidence_reference",
                "manufacturer",
                "model",
            },
            "tpm",
        )
        exact(value["k_psa"], {"public_area", "name", "algorithm_profile"}, "k_psa")
        exact(value["release"], {"release_policy_digest", "release_policy_version"}, "release")
        exact(
            value["requested_entitlements"],
            {"product", "edition", "requested_features"},
            "entitlements",
        )
        require_timestamp(value["created_at_utc"], "created_at_utc")
        for name in ("installation_id", "nonce"):
            require_text(value[name], name)
        for name in ("device_id", "platform", "architecture"):
            require_text(value["device"][name], name)
        for name in ("evidence_profile", "evidence_reference"):
            require_text(value["tpm"][name], name)
        for name in ("manufacturer", "model"):
            if value["tpm"][name] is not None:
                require_text(value["tpm"][name], name)
        require_digest(value["tpm"]["ek_public_digest"], "ek_public_digest")
        require_digest(value["tpm"]["ak_public_digest"], "ak_public_digest")
        require_text(value["k_psa"]["name"], "K_PSA.Name")
        require_text(value["k_psa"]["algorithm_profile"], "algorithm_profile")
        require_digest(value["release"]["release_policy_digest"], "release_policy_digest")
        if not isinstance(value["release"]["release_policy_version"], int) or isinstance(
            value["release"]["release_policy_version"], bool
        ):
            raise ValueError("invalid release_policy_version")
        require_text(value["requested_entitlements"]["product"], "product")
        require_text(value["requested_entitlements"]["edition"], "edition")
        require_string_set(
            value["requested_entitlements"]["requested_features"], "requested_features"
        )
        validate_activation_request_identity(value)
        return cls._from_canonical(canonical)


@dataclass(frozen=True, init=False)
class EnrollmentDecisionV1:
    canonical_bytes: bytes

    def __init__(self, *_: object, **__: object) -> None:
        raise TypeError("use from_mapping()")

    @classmethod
    def _from_canonical(cls, canonical: bytes) -> "EnrollmentDecisionV1":
        instance = object.__new__(cls)
        object.__setattr__(instance, "canonical_bytes", canonical)
        return instance

    @property
    def document(self) -> dict[str, Any]:
        return _projection(self.canonical_bytes)

    @classmethod
    def from_mapping(cls, document: Mapping[str, Any]) -> "EnrollmentDecisionV1":
        canonical = canonical_json_bytes(dict(document))
        value = _projection(canonical)
        exact(
            value,
            {
                "schema",
                "version",
                "request_id",
                "license_id",
                "product",
                "edition",
                "features",
                "issued_at",
                "expires_at",
                "renewal_after",
                "approval_status",
                "operator_note",
            },
            "decision",
        )
        if value["schema"] != "EnrollmentDecisionV1" or value["version"] != 1:
            raise ValueError("unsupported decision")
        require_digest(value["request_id"], "request_id")
        for name in ("license_id", "product", "edition"):
            require_text(value[name], name, identifier=name == "license_id")
        require_string_set(value["features"], "features")
        require_timestamp(value["issued_at"], "issued_at")
        require_timestamp(value["expires_at"], "expires_at", optional=True)
        require_timestamp(value["renewal_after"], "renewal_after", optional=True)
        if value["approval_status"] != "APPROVED":
            raise ValueError("request was not approved")
        if value["operator_note"] is not None and not isinstance(value["operator_note"], str):
            raise ValueError("invalid operator_note")
        return cls._from_canonical(canonical)
