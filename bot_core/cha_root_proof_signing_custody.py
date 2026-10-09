"""Dedicated local requester and deployment claimant custody, separate from issuer ports.

Only the offline administrator stages keys and changes lifecycle. Runtime ports
derive public material from protected custody and expose one domain-bound operation.
The protected host administrator and native keyring remain trusted. No distributed
transaction is claimed across keyring, metadata and PostgreSQL.
"""

from __future__ import annotations

import base64
import hashlib
import os
import stat
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import cast

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from bot_core.cha_issuance_request import (
    ISSUER_TARGET_NAMESPACE,
    REQUEST_FIELDS,
    IssuanceSignerIdentity,
    IssuanceSigningRole,
    encode_signature,
)
from bot_core.licensing.canonical import canonical_json_bytes, parse_canonical
from bot_core.local_signing_custody import (
    LocalSigningCustodyError,
    _custody_lock,
    _exact_path_snapshot,
    _replace_record,
    _validate_directory,
    _write_new_record,
)
from bot_core.root_proof_issuer_substrate import public_key_material_identity
from bot_core.security.keyring_storage import KeyringSecretStorage

_SCHEMA = "CryptoHunter.Stage9.IssuanceSigningCustodyV1"
_IDENTITY_FIELDS = set(IssuanceSignerIdentity.__dataclass_fields__)


@dataclass(frozen=True, slots=True)
class IssuanceCustodyPublicRecord:
    """Offline ceremony facts, including the actual lifecycle; no authority."""

    role: IssuanceSigningRole
    principal: str
    environment: str
    trust_domain: str
    key_id: str
    key_version: int
    public_key_hex: str
    key_material_identity: str
    lifecycle: str


def _public_record(value: dict) -> IssuanceCustodyPublicRecord:
    return IssuanceCustodyPublicRecord(
        IssuanceSigningRole(value["role"]),
        value["principal"],
        value["environment"],
        value["trust_domain"],
        value["key_id"],
        value["key_version"],
        value["public_key_hex"],
        value["key_material_identity"],
        value["lifecycle"],
    )


def _scope(role: IssuanceSigningRole, trust: str, principal: str) -> str:
    if type(role) is not IssuanceSigningRole or any(
        type(value) is not str or not value.strip() for value in (trust, principal)
    ):
        raise LocalSigningCustodyError("exact production custody scope required")
    digest = hashlib.sha256(
        canonical_json_bytes(
            {
                "schema": _SCHEMA,
                "environment": "PRODUCTION",
                "trust_domain": trust,
                "role": role.value,
                "principal": principal,
            }
        )
    ).hexdigest()
    return "dudzian.stage9." + role.value.lower() + "." + digest


def _storage(directory: Path, role: IssuanceSigningRole, trust: str, principal: str):
    service = _scope(role, trust, principal)
    return KeyringSecretStorage(service_name=service, index_path=directory / "keyring-index.json")


def _read(directory: Path, role: IssuanceSigningRole, trust: str, principal: str) -> dict:
    descriptor = os.open(directory / "custody.json", os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    with os.fdopen(descriptor, "rb") as stream:
        info = os.fstat(stream.fileno())
        if not stat.S_ISREG(info.st_mode):
            raise LocalSigningCustodyError("unsafe issuance custody metadata")
        if sys.platform != "win32":
            if os.name == "posix" and (
                info.st_uid != os.geteuid() or stat.S_IMODE(info.st_mode) & 0o077
            ):
                raise LocalSigningCustodyError("unsafe issuance custody metadata")
        raw = stream.read()
    value = parse_canonical(raw)
    if (
        set(value) != _IDENTITY_FIELDS | {"schema", "secret_reference"}
        or value.pop("schema") != _SCHEMA
    ):
        raise LocalSigningCustodyError("invalid issuance custody metadata")
    service = _scope(role, trust, principal)
    if (
        value["role"] != role.value
        or value["principal"] != principal
        or value["environment"] != "PRODUCTION"
        or value["trust_domain"] != trust
        or value["service_namespace"] != service
        or value["credential_namespace"] != service + ".credentials"
        or value["lifecycle_namespace"] != service + ".lifecycle"
        or value["semantic_role"] != role.semantic_role
        or value["key_handle"] != service + ".key:" + value["key_material_identity"]
        or value["secret_reference"] != service + ".secret:" + value["key_material_identity"]
        or value["lifecycle"] not in ("STAGED", "ACTIVE", "VERIFY_ONLY", "REVOKED")
        or type(value["lifecycle_generation"]) is not int
        or value["lifecycle_generation"]
        not in {
            "STAGED": {1},
            "ACTIVE": {2},
            "VERIFY_ONLY": {3},
            "REVOKED": {3, 4},
        }[value["lifecycle"]]
    ):
        raise LocalSigningCustodyError("issuance custody identity mismatch")
    # Validate the complete identity even while offline lifecycle is non-ACTIVE.
    _identity(value, active_validation=True)
    return cast(dict, value)


def _identity(value: dict, *, active_validation: bool = False) -> IssuanceSignerIdentity:
    data = {name: value[name] for name in _IDENTITY_FIELDS}
    data["role"] = IssuanceSigningRole(data["role"])
    if active_validation:
        data["lifecycle"] = "ACTIVE"
    return IssuanceSignerIdentity(**data)


def _private(directory: Path, value: dict) -> Ed25519PrivateKey:
    try:
        store = _storage(
            directory, IssuanceSigningRole(value["role"]), value["trust_domain"], value["principal"]
        )
        if type(store) is not KeyringSecretStorage:
            raise ValueError("exact protected storage required")
        encoded = store.get_secret(value["secret_reference"])
        if type(encoded) is not str:
            raise ValueError("protected key missing")
        seed = base64.b64decode(encoded, validate=True)
        if len(seed) != 32:
            raise ValueError("invalid protected key")
        private = Ed25519PrivateKey.from_private_bytes(seed)
        public = private.public_key().public_bytes(
            serialization.Encoding.Raw, serialization.PublicFormat.Raw
        )
        if (
            public.hex() != value["public_key_hex"]
            or public_key_material_identity(public) != value["key_material_identity"]
        ):
            raise ValueError("private/public mismatch")
        return private
    except Exception:
        raise LocalSigningCustodyError("issuance custody material unavailable/corrupt") from None


class OfflineIssuanceCustodyAdministrator:
    """Admin-only ceremony: stage private key, register public key, then activate."""

    def __init__(
        self, directory: Path, *, role: IssuanceSigningRole, trust_domain: str, principal: str
    ) -> None:
        self._directory = _validate_directory(directory, create=True)
        self._role, self._trust, self._principal = role, trust_domain, principal
        _scope(role, trust_domain, principal)

    def stage(self, *, key_id: str, key_version: int) -> IssuanceCustodyPublicRecord:
        """Return public ceremony facts only. STAGED material cannot authorize."""
        service = _scope(self._role, self._trust, self._principal)
        with _custody_lock(self._directory, exclusive=True):
            path = self._directory / "custody.json"
            if path.exists():
                value = _read(self._directory, self._role, self._trust, self._principal)
                _private(self._directory, value)
                if (value["key_id"], value["key_version"]) != (key_id, key_version):
                    raise LocalSigningCustodyError("custody already stages a different identity")
                return _public_record(value)
            private = Ed25519PrivateKey.generate()
            public = private.public_key().public_bytes(
                serialization.Encoding.Raw, serialization.PublicFormat.Raw
            )
            material = public_key_material_identity(public)
            identity = IssuanceSignerIdentity(
                self._role,
                self._principal,
                self._role.semantic_role,
                "PRODUCTION",
                self._trust,
                key_id,
                key_version,
                public.hex(),
                material,
                service,
                service + ".credentials",
                service + ".lifecycle",
                service + ".key:" + material,
                "ACTIVE",
                2,
            )
            value = {
                **asdict(identity),
                "schema": _SCHEMA,
                "secret_reference": service + ".secret:" + material,
                "lifecycle": "STAGED",
                "lifecycle_generation": 1,
            }
            # Persist the public draft before writing the keyring. A crash here
            # leaves an unusable draft: retries never generate/substitute a key.
            if not _write_new_record(path, canonical_json_bytes(value)):
                raise LocalSigningCustodyError("custody provisioning conflict")
            seed = private.private_bytes(
                serialization.Encoding.Raw,
                serialization.PrivateFormat.Raw,
                serialization.NoEncryption(),
            )
            _storage(self._directory, self._role, self._trust, self._principal).set_secret(
                value["secret_reference"], base64.b64encode(seed).decode("ascii")
            )
            _private(self._directory, value)
            return _public_record(value)

    def activate(self, registry: object) -> IssuanceSignerIdentity:
        with _custody_lock(self._directory, exclusive=True):
            value = _read(self._directory, self._role, self._trust, self._principal)
            _private(self._directory, value)
            _require_registry(value, registry)
            if value["lifecycle"] == "STAGED":
                value["lifecycle"], value["lifecycle_generation"] = "ACTIVE", 2
                _replace_record(
                    self._directory / "custody.json",
                    canonical_json_bytes({"schema": _SCHEMA, **value}),
                )
            return _identity(value)

    def transition_lifecycle(self, state: str) -> None:
        with _custody_lock(self._directory, exclusive=True):
            value = _read(self._directory, self._role, self._trust, self._principal)
            if state not in {"ACTIVE": {"VERIFY_ONLY", "REVOKED"}, "VERIFY_ONLY": {"REVOKED"}}.get(
                value["lifecycle"], set()
            ):
                raise LocalSigningCustodyError("illegal issuance custody lifecycle transition")
            value["lifecycle"] = state
            value["lifecycle_generation"] += 1
            _replace_record(
                self._directory / "custody.json", canonical_json_bytes({"schema": _SCHEMA, **value})
            )


def _require_registry(value: dict, registry: object) -> None:
    from bot_core.postgresql_preaccount_credentials import (
        PostgreSQLClaimantIdentityRegistryProvider,
        PostgreSQLRequesterCredentialRegistryProvider,
    )

    role = IssuanceSigningRole(value["role"])
    expected = (
        PostgreSQLRequesterCredentialRegistryProvider
        if role is IssuanceSigningRole.REQUESTER
        else PostgreSQLClaimantIdentityRegistryProvider
    )
    if type(registry) is not expected:
        raise LocalSigningCustodyError("genuine public registry required")
    registry._qualify()
    record = registry._current(value["principal"])
    if (
        record.lifecycle.value != "ACTIVE"
        or record.principal_id != value["principal"]
        or record.environment != value["environment"]
        or record.trust_domain != value["trust_domain"]
        or record.key_id != value["key_id"]
        or record.key_version != value["key_version"]
        or record.credential_role != role.semantic_role
        or record.semantic_role.value
        != (
            "ROOT_PROOF_REQUESTER"
            if role is IssuanceSigningRole.REQUESTER
            else "ROOT_PROOF_CLAIMANT"
        )
        or record.public_key.hex() != value["public_key_hex"]
        or public_key_material_identity(record.public_key) != value["key_material_identity"]
        or registry.public_key(record.key_id) != record.public_key
    ):
        raise LocalSigningCustodyError("registry/private custody binding mismatch")


class _RuntimeCustody:
    _directory: Path
    _role: IssuanceSigningRole
    _trust: str
    _principal: str
    __slots__ = ("_directory", "_role", "_trust", "_principal")

    def __init__(
        self, directory: Path, role: IssuanceSigningRole, trust: str, principal: str
    ) -> None:
        object.__setattr__(
            self, "_directory", _exact_path_snapshot(directory, label="issuance custody")
        )
        object.__setattr__(self, "_role", role)
        object.__setattr__(self, "_trust", trust)
        object.__setattr__(self, "_principal", principal)

    def __setattr__(self, name: str, value: object) -> None:
        raise TypeError("runtime issuance custody is immutable")

    def identity(self) -> IssuanceSignerIdentity:
        directory = _validate_directory(self._directory, create=False)
        with _custody_lock(directory, exclusive=False):
            value = _read(directory, self._role, self._trust, self._principal)
            _private(directory, value)
            return _identity(value)

    def _sign(self, raw: bytes, expected: IssuanceSignerIdentity) -> str:
        directory = _validate_directory(self._directory, create=False)
        with _custody_lock(directory, exclusive=False):
            value = _read(directory, self._role, self._trust, self._principal)
            private = _private(directory, value)
            identity = _identity(value)
            payload = parse_canonical(raw)
            requester = self._role is IssuanceSigningRole.REQUESTER
            if (
                identity != expected
                or set(payload) != set(REQUEST_FIELDS)
                or payload["schema_version"] != "1"
                or payload["issuer_target_namespace"] != ISSUER_TARGET_NAMESPACE
                or payload["environment"] != identity.environment
                or payload["trust_domain"] != identity.trust_domain
                or payload["requester_id" if requester else "provisioning_principal_id"]
                != identity.principal
                or payload["requester_key_id" if requester else "claimant_key_id"]
                != identity.key_id
                or payload["requester_key_version" if requester else "claimant_key_version"]
                != identity.key_version
                or (
                    not requester
                    and identity.principal
                    in (payload["account_id"], payload["logical_operation_id"])
                )
            ):
                raise LocalSigningCustodyError("exact reserved signing request required")
            return encode_signature(private.sign(self._role.domain.encode("ascii") + b"\x00" + raw))


class LocalCHARequesterSigningCustody(_RuntimeCustody):
    __slots__ = ()

    def __init__(self, directory: Path, *, trust_domain: str) -> None:
        super().__init__(
            directory, IssuanceSigningRole.REQUESTER, trust_domain, "CryptoHunterAccountAuthority"
        )

    def __init_subclass__(cls) -> None:
        raise TypeError("requester custody cannot be subclassed")

    def sign_issuance_request(self, raw: bytes, expected: IssuanceSignerIdentity) -> str:
        return self._sign(raw, expected)


class LocalPreaccountClaimantAuthorizationCustody(_RuntimeCustody):
    __slots__ = ()

    def __init__(self, directory: Path, *, trust_domain: str, provisioning_principal: str) -> None:
        super().__init__(
            directory, IssuanceSigningRole.CLAIMANT, trust_domain, provisioning_principal
        )

    def __init_subclass__(cls) -> None:
        raise TypeError("claimant custody cannot be subclassed")

    def authorize_entitlement_claim(self, raw: bytes, expected: IssuanceSignerIdentity) -> str:
        return self._sign(raw, expected)
