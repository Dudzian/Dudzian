"""Tożsamość wydania zatwierdzana niezależnie od producenta dowodów hosta.

Certyfikat identyfikuje materiały i wymagane kontrakty produktu. Nie jest
dowodem wykonania tych kontraktów ani zgodą na produkcyjną kwalifikację.
"""

from __future__ import annotations

import base64
import binascii
import hashlib
import re
from typing import Any, cast

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey

PRODUCT_ID = "urn:cryptohunter:product"
RELEASE_CONTRACT = "cryptohunter.linux.release.v1"
SECURITY_PROFILE = "cryptohunter.linux.stage9.v1"
REQUIRED_COMPONENT_CONTRACTS = {
    "runtime": (
        "cryptohunter.runtime.readonly-health.v1",
        "cryptohunter.ipc.client-authentication.v1",
        "cryptohunter.postgresql.runtime-peer-and-denials.v1",
    ),
    "verifier": (
        "cryptohunter.verifier.readonly-health.v1",
        "cryptohunter.ipc.server-authentication.v1",
        "cryptohunter.postgresql.verifier-peer-and-denials.v1",
    ),
}
STATEMENT_FIELDS = {
    "schema_version",
    "product_id",
    "contract",
    "security_profile",
    "repository",
    "source_revision",
    "release_id",
    "installation_profile_sha256",
    "artifacts",
    "components",
}


class ReleaseIdentityError(ValueError):
    """Brak niezależnego potwierdzenia tożsamości właściwego produktu."""


def _object(value: Any, fields: set[str]) -> dict[str, Any]:
    if type(value) is not dict or set(value) != fields:
        raise ReleaseIdentityError("Niekompletna tożsamość wydania CryptoHunter")
    return cast(dict[str, Any], value)


def _hash(value: Any, length: int = 64) -> bool:
    return type(value) is str and re.fullmatch(rf"[0-9a-f]{{{length}}}", value) is not None


def _decode(value: Any, length: int) -> bytes:
    if type(value) is not str:
        raise ReleaseIdentityError("Niepoprawny klucz lub podpis autorytetu wydania")
    try:
        raw = base64.b64decode(value, validate=True)
    except (ValueError, binascii.Error) as exc:
        raise ReleaseIdentityError("Niepoprawny klucz lub podpis autorytetu wydania") from exc
    if len(raw) != length or base64.b64encode(raw).decode("ascii") != value:
        raise ReleaseIdentityError("Niepoprawny klucz lub podpis autorytetu wydania")
    return raw


def validate_release_authorities(value: Any, *, producer_keys: set[str]) -> None:
    """Sprawdź zatwierdzony rejestr, bez ustanawiania zaufania do jego źródła."""
    if type(value) is not list or len(value) > 32:
        raise ReleaseIdentityError("Nieprawidłowy rejestr autorytetów wydania")
    seen: set[str] = set()
    for entry in value:
        key = _object(entry, {"key_id", "public_key"})
        raw = _decode(key["public_key"], 32)
        identifier = hashlib.sha256(raw).hexdigest()
        if key["key_id"] != identifier or identifier in seen or key["public_key"] in producer_keys:
            raise ReleaseIdentityError("Autorytet wydania nie jest niezależny od producenta hosta")
        seen.add(identifier)


def validate_release_statement(envelope: Any, installation: dict[str, Any]) -> dict[str, Any]:
    """Sprawdź wymagany profil; nazwy plików i jednostek pochodzą z wydania."""
    from deployment.linux_physical_evidence import digest

    certificate = _object(envelope, {"key_id", "payload", "signature"})
    statement = _object(certificate["payload"], STATEMENT_FIELDS)
    if (
        not _hash(certificate["key_id"])
        or type(statement["schema_version"]) is not int
        or statement["schema_version"] != 1
        or statement["product_id"] != PRODUCT_ID
        or statement["contract"] != RELEASE_CONTRACT
        or statement["security_profile"] != SECURITY_PROFILE
        or statement["repository"] != "Dudzian/Dudzian"
        or not _hash(statement["source_revision"], 40)
        or type(statement["release_id"]) is not str
        or re.fullmatch(r"[0-9][A-Za-z0-9.+:~_-]{0,99}", statement["release_id"]) is None
        or not _hash(statement["installation_profile_sha256"])
        or digest({k: v for k, v in installation.items() if k != "release_identity"})
        != statement["installation_profile_sha256"]
    ):
        raise ReleaseIdentityError(
            "Wydanie nie odpowiada wymaganemu produktowi, rewizji lub profilowi"
        )
    _decode(certificate["signature"], 64)
    artifacts = _object(statement["artifacts"], {"baseline", "update", "failed_update"})
    for phase, artifact in artifacts.items():
        _object(artifact, {"sha256", "version"})
        expected = installation["packages"][phase]
        if artifact != {"sha256": expected["sha256"], "version": expected["version"]}:
            raise ReleaseIdentityError(
                "Arbitralny pakiet nie jest artefaktem zatwierdzonego wydania"
            )
    components = statement["components"]
    units = installation["units"]
    if type(components) is not list or len(components) != len(units):
        raise ReleaseIdentityError("Niepełny inwentarz komponentów CryptoHunter")
    expected_components = []
    for unit in units:
        role = unit["role"]
        if role not in REQUIRED_COMPONENT_CONTRACTS:
            raise ReleaseIdentityError("Nieobsługiwana rola komponentu wydania")
        expected_components.append(
            {
                "role": role,
                "unit": unit["name"],
                "executable_sha256": unit["executable_sha256"],
                "update_executable_sha256": unit["update_executable_sha256"],
                "contracts": list(REQUIRED_COMPONENT_CONTRACTS[role]),
            }
        )
    if {entry["role"] for entry in expected_components} != set(REQUIRED_COMPONENT_CONTRACTS):
        raise ReleaseIdentityError("Brak wymaganej roli produktu")
    if components != expected_components:
        raise ReleaseIdentityError("Niezgodne programy lub kontrakty funkcjonalne komponentów")
    return statement


def verify_release_identity(
    installation: dict[str, Any],
    policy: dict[str, Any],
    *,
    source_revision: str,
    materials: dict[str, Any],
) -> dict[str, Any]:
    """Wymagaj oddzielnego autorytetu; podpis producenta hosta nie wystarcza."""
    from deployment.linux_physical_evidence import canonical_bytes

    certificate = _object(installation.get("release_identity"), {"key_id", "payload", "signature"})
    statement = validate_release_statement(certificate, installation)
    authorities = policy.get("release_authorities", [])
    producer_keys = {host["public_key"] for host in policy.get("hosts", [])}
    validate_release_authorities(authorities, producer_keys=producer_keys)
    selected = [item for item in authorities if item["key_id"] == certificate["key_id"]]
    if len(selected) != 1:
        raise ReleaseIdentityError("Brak niezależnie zatwierdzonego autorytetu wydania")
    key = Ed25519PublicKey.from_public_bytes(_decode(selected[0]["public_key"], 32))
    try:
        key.verify(_decode(certificate["signature"], 64), canonical_bytes(statement))
    except InvalidSignature as exc:
        raise ReleaseIdentityError("Nieprawidłowy podpis tożsamości wydania") from exc
    if (
        statement["source_revision"] != source_revision
        or materials["source_revision"] != source_revision
    ):
        raise ReleaseIdentityError("Rewizja wydania różni się od kwalifikowanej")
    for phase, field in (
        ("baseline", "artifact_sha256"),
        ("update", "update_artifact_sha256"),
        ("failed_update", "failed_update_artifact_sha256"),
    ):
        if statement["artifacts"][phase]["sha256"] != materials[field]:
            raise ReleaseIdentityError("Artefakt wydania różni się od materiałów fizycznego testu")
    return statement
