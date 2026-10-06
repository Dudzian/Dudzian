"""Exact native successor key for the production LPPI lifecycle.

The lifecycle owner durably reserves creation before calling this adapter. This
module owns native handles and local qualification only; it does not issue LPPI
authority or adopt an unreserved key. TPM origin is verified independently from
CertifyCreation by the production custody verifier.
"""

from __future__ import annotations

import ctypes
import hashlib
import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, cast
from weakref import WeakKeyDictionary

from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric import ec, utils

from bot_core.licensing.pre_enrollment import validate_public_key
from deployment import windows_cng_pre_enrollment as cng

if TYPE_CHECKING:
    from bot_core.licensing.lppi_package_acceptance import VerifiedLPPIClientPackageAcceptance
    from deployment.windows_production_lppi_authority import LPPIAuthorityKeyReservationDescriptor

PROVIDER = cng.PROVIDER
KEY_NAME = "CryptoHunter.Stage9.Production.LPPI.Authority.v1"
ALGORITHM = cng.ALGORITHM
QUALIFICATION_PROFILE = "WINDOWS_PLATFORM_CRYPTO_PROVIDER_TPM_ECDSA_P256_V1"


class WindowsLPPIAuthorityKeyError(RuntimeError):
    """An exact native identity or authorized lifecycle could not be established."""


class _AuthorityNCryptAPI(cng._NCryptAPI):
    """Reuse only generic ABI primitives; override both fixed-name operations."""

    def open_key(self, provider: int) -> int | None:
        handle = cng._HANDLE()
        status = self.dll.NCryptOpenKey(
            provider, ctypes.byref(handle), KEY_NAME, 0, cng.MACHINE_KEY
        )
        if status & 0xFFFFFFFF == cng.NTE_BAD_KEYSET:
            return None
        cng._check(status, "OPEN_LPPI_AUTHORITY_KEY")
        if not handle.value:
            raise WindowsLPPIAuthorityKeyError("EMPTY_AUTHORITY_KEY_HANDLE")
        return int(handle.value)

    def create_key(self, provider: int) -> int:
        handle = cng._HANDLE()
        cng._check(
            self.dll.NCryptCreatePersistedKey(
                provider, ctypes.byref(handle), ALGORITHM, KEY_NAME, 0, cng.MACHINE_KEY
            ),
            "CREATE_LPPI_AUTHORITY_KEY",
        )
        if not handle.value:
            raise WindowsLPPIAuthorityKeyError("EMPTY_AUTHORITY_KEY_HANDLE")
        try:
            for name, value in (("Export Policy", 0), ("Key Usage", cng.SIGNING_ONLY)):
                number = cng._DWORD(value)
                cng._check(
                    self.dll.NCryptSetProperty(
                        handle.value, name, ctypes.byref(number), ctypes.sizeof(number), 0
                    ),
                    "SET_LPPI_AUTHORITY_KEY_PROFILE",
                )
            cng._check(self.dll.NCryptFinalizeKey(handle.value, 0), "FINALIZE_LPPI_AUTHORITY_KEY")
        except BaseException:
            self.free(handle.value)
            raise
        return int(handle.value)


def _load_native() -> _AuthorityNCryptAPI:
    return _AuthorityNCryptAPI()


def _require_acceptance(value: object) -> VerifiedLPPIClientPackageAcceptance:
    from bot_core.licensing.lppi_package_acceptance import require_verified_lppi_package_acceptance

    return require_verified_lppi_package_acceptance(value)


def _reservation(accepted: object, reservation: object) -> LPPIAuthorityKeyReservationDescriptor:
    from deployment.windows_production_lppi_authority import require_lppi_authority_key_reservation

    return require_lppi_authority_key_reservation(reservation, accepted)


def _require_descriptor(
    accepted: VerifiedLPPIClientPackageAcceptance, descriptor: LPPIAuthorityKeyReservationDescriptor
) -> None:
    public = descriptor.pre_enrollment_public_key
    unique = descriptor.pre_enrollment_unique_name
    if (
        type(public) is not bytes
        or public != accepted.pre_enrollment_public_key_bytes
        or type(unique) is not str
        or unique != accepted.pre_enrollment_key_unique_name
        or type(descriptor.reservation_id) is not str
        or not descriptor.reservation_id
        or descriptor.mode not in {"create", "recover", "reconcile"}
    ):
        raise WindowsLPPIAuthorityKeyError("EXACT_ACCEPTED_LIFECYCLE_BINDING_REQUIRED")
    validate_public_key(public)
    expected_public = descriptor.expected_authority_public_key
    expected_unique = descriptor.expected_authority_unique_name
    if descriptor.mode in {"create", "reconcile"}:
        if expected_public is not None or expected_unique is not None:
            raise WindowsLPPIAuthorityKeyError("INVALID_AUTHORITY_CREATION_RESERVATION")
    elif type(expected_public) is not bytes or type(expected_unique) is not str:
        raise WindowsLPPIAuthorityKeyError("EXACT_RETAINED_AUTHORITY_IDENTITY_REQUIRED")
    else:
        validate_public_key(expected_public)


def _qualify(native: _AuthorityNCryptAPI, key: int) -> tuple[bytes, str, str]:
    raw_provider = native.property(key, "Provider Handle")
    if len(raw_provider) != ctypes.sizeof(cng._HANDLE):
        raise WindowsLPPIAuthorityKeyError("INVALID_AUTHORITY_PROVIDER_HANDLE_PROPERTY")
    provider = int.from_bytes(raw_provider, "little")
    if not provider:
        raise WindowsLPPIAuthorityKeyError("EMPTY_AUTHORITY_PROVIDER_HANDLE_PROPERTY")
    try:
        implementation = native.number(provider, "Impl Type")
        if (
            native.text(provider, "Name") != PROVIDER
            or not implementation & cng.HARDWARE
            or implementation & cng.SOFTWARE
        ):
            raise WindowsLPPIAuthorityKeyError("AUTHORITY_PROVIDER_PROFILE_REJECTED")
    finally:
        # NCrypt Provider Handle returns an owned reference.
        native.free(provider)
    if (
        native.text(key, "Name") != KEY_NAME
        or native.text(key, "Algorithm Name") != ALGORITHM
        or native.text(key, "Algorithm Group") != "ECDSA"
        or native.number(key, "Length") != 256
        or native.number(key, "Key Type") != cng.MACHINE_KEY
        or native.number(key, "Key Usage") != cng.SIGNING_ONLY
        or native.number(key, "Export Policy") != 0
        or native.export_allowed(key)
    ):
        raise WindowsLPPIAuthorityKeyError("AUTHORITY_KEY_PROFILE_REJECTED")
    unique = native.text(key, "Unique Name")
    if not re.fullmatch(r"[A-Za-z0-9._\\:-]{1,256}", unique):
        raise WindowsLPPIAuthorityKeyError("UNSAFE_AUTHORITY_KEY_UNIQUE_NAME")
    blob = native.public_blob(key)
    return cng._sec1_from_public_blob(blob), unique, hashlib.sha256(blob).hexdigest()


def _require_identity(
    public: bytes, unique: str, descriptor: LPPIAuthorityKeyReservationDescriptor
) -> None:
    if (
        KEY_NAME == cng.KEY_NAME
        or public == descriptor.pre_enrollment_public_key
        or unique == descriptor.pre_enrollment_unique_name
    ):
        raise WindowsLPPIAuthorityKeyError("SUCCESSOR_AUTHORITY_KEY_MUST_BE_DISTINCT")
    expected = descriptor.expected_authority_public_key
    if expected is not None and (
        public != expected or unique != descriptor.expected_authority_unique_name
    ):
        raise WindowsLPPIAuthorityKeyError("RETAINED_AUTHORITY_IDENTITY_MISMATCH")


@dataclass(frozen=True)
class _QualifiedSnapshot:
    native: _AuthorityNCryptAPI
    dll: object
    provider: int
    key: int
    public: bytes
    unique_name: str
    public_blob_sha256: str
    accepted: object
    reservation: object
    reservation_id: str


_ISSUED_AUTHORITY_KEYS: WeakKeyDictionary[WindowsLPPIAuthorityKey, _QualifiedSnapshot] = (
    WeakKeyDictionary()
)


class WindowsLPPIAuthorityKey:
    """Factory-issued native handle, never a caller-supplied signing callback."""

    __slots__ = (
        "_native",
        "_provider",
        "_key",
        "_public",
        "_unique_name",
        "_public_blob_sha256",
        "__weakref__",
    )

    _native: _AuthorityNCryptAPI
    _provider: int
    _key: int
    _public: bytes
    _unique_name: str
    _public_blob_sha256: str

    def __init__(self) -> None:
        raise TypeError("use open_or_create_production_lppi_authority_key()")

    @property
    def public_key_bytes(self) -> bytes:
        return _qualified_snapshot(self).public

    @property
    def public_blob_sha256(self) -> str:
        return _qualified_snapshot(self).public_blob_sha256

    @property
    def key_unique_name(self) -> str:
        return _qualified_snapshot(self).unique_name

    @property
    def unique_name(self) -> str:
        return self.key_unique_name

    @property
    def provider_name(self) -> str:
        _qualified_snapshot(self)
        return cast(str, PROVIDER)

    @property
    def key_name(self) -> str:
        _qualified_snapshot(self)
        return KEY_NAME

    def sign_binding_pop(self, binding_raw: bytes) -> bytes:
        from bot_core.licensing.lppi_authority_key import authority_pop_signed_bytes

        require_verified_production_lppi_authority_key(self)
        if type(binding_raw) is not bytes:
            raise WindowsLPPIAuthorityKeyError("CANONICAL_AUTHORITY_BINDING_BYTES_REQUIRED")
        signed = authority_pop_signed_bytes(binding_raw)
        digest = hashlib.sha256(signed).digest()
        try:
            signature = self._native.sign_digest(self._key, digest)
            validate_public_key(self._public).verify(
                signature, digest, ec.ECDSA(utils.Prehashed(hashes.SHA256()))
            )
        except Exception as exc:
            raise WindowsLPPIAuthorityKeyError("AUTHORITY_BINDING_PROOF_FAILED") from exc
        return cast(bytes, signature)

    def close(self) -> None:
        _ISSUED_AUTHORITY_KEYS.pop(self, None)
        failures: list[Exception] = []
        for attribute in ("_key", "_provider"):
            handle = getattr(self, attribute, 0)
            if handle:
                setattr(self, attribute, 0)
                try:
                    self._native.free(handle)
                except Exception as exc:
                    failures.append(exc)
        if failures:
            raise WindowsLPPIAuthorityKeyError(
                "CLOSE_AUTHORITY_NATIVE_HANDLES_FAILED"
            ) from failures[0]

    def __enter__(self) -> WindowsLPPIAuthorityKey:
        require_verified_production_lppi_authority_key(self)
        return self

    def __exit__(self, *_: object) -> None:
        self.close()


# The qualified capability is the exact factory-issued native key, not a wrapper.
QualifiedLPPIAuthorityKey = WindowsLPPIAuthorityKey


def open_or_create_production_lppi_authority_key(
    accepted_package: object, reservation: object
) -> WindowsLPPIAuthorityKey:
    """Create only after durable reservation; recover only the fixed retained key."""
    accepted = _require_acceptance(accepted_package)
    descriptor = _reservation(accepted, reservation)
    _require_descriptor(accepted, descriptor)
    native = _load_native()
    if type(native) is not _AuthorityNCryptAPI:
        raise WindowsLPPIAuthorityKeyError("EXACT_AUTHORITY_NATIVE_BOUNDARY_REQUIRED")
    instance = object.__new__(WindowsLPPIAuthorityKey)
    instance._native, instance._provider, instance._key = native, 0, 0
    try:
        instance._provider = native.open_provider()
        instance._key = native.open_key(instance._provider) or 0
        if not instance._key:
            if descriptor.mode != "create":
                raise WindowsLPPIAuthorityKeyError("RETAINED_AUTHORITY_KEY_MISSING")
            instance._key = native.create_key(instance._provider)
            native.free(instance._key)
            instance._key = 0
            instance._key = native.open_key(instance._provider) or 0
            if not instance._key:
                raise WindowsLPPIAuthorityKeyError("CREATED_AUTHORITY_KEY_NOT_REOPENABLE")
        instance._public, instance._unique_name, instance._public_blob_sha256 = _qualify(
            native, instance._key
        )
        _require_identity(instance._public, instance._unique_name, descriptor)
        _ISSUED_AUTHORITY_KEYS[instance] = _QualifiedSnapshot(
            native,
            native.dll,
            instance._provider,
            instance._key,
            instance._public,
            instance._unique_name,
            instance._public_blob_sha256,
            accepted,
            reservation,
            descriptor.reservation_id,
        )
        return instance
    except BaseException:
        instance.close()
        raise


def reopen_production_lppi_authority_key(
    accepted_package: object, reservation: object
) -> WindowsLPPIAuthorityKey:
    """Require an exact retained identity; this path never creates a new key."""
    accepted = _require_acceptance(accepted_package)
    descriptor = _reservation(accepted, reservation)
    _require_descriptor(accepted, descriptor)
    if descriptor.mode != "recover":
        raise WindowsLPPIAuthorityKeyError("EXACT_RETAINED_AUTHORITY_IDENTITY_REQUIRED")
    return open_or_create_production_lppi_authority_key(accepted, reservation)


def probe_production_lppi_authority_key_presence(accepted_package: object) -> bool:
    """Read fixed-name presence before reservation without adopting any identity."""
    accepted = _require_acceptance(accepted_package)
    native = _load_native()
    if type(native) is not _AuthorityNCryptAPI:
        raise WindowsLPPIAuthorityKeyError("EXACT_AUTHORITY_NATIVE_BOUNDARY_REQUIRED")
    provider, key = 0, 0
    try:
        provider = native.open_provider()
        key = native.open_key(provider) or 0
        if key:
            public, unique, _ = _qualify(native, key)
            if (
                public == accepted.pre_enrollment_public_key_bytes
                or unique == accepted.pre_enrollment_key_unique_name
            ):
                raise WindowsLPPIAuthorityKeyError("SUCCESSOR_AUTHORITY_KEY_MUST_BE_DISTINCT")
        return bool(key)
    finally:
        try:
            if key:
                native.free(key)
        finally:
            if provider:
                native.free(provider)


def _qualified_snapshot(value: object) -> _QualifiedSnapshot:
    """Public identity projection only; it is never a current signing authority."""
    if type(value) is not WindowsLPPIAuthorityKey:
        raise WindowsLPPIAuthorityKeyError("QUALIFIED_PRODUCTION_LPPI_AUTHORITY_KEY_REQUIRED")
    try:
        snapshot = _ISSUED_AUTHORITY_KEYS.get(value)
        if (
            snapshot is None
            or type(value._native) is not _AuthorityNCryptAPI
            or value._native is not snapshot.native
            or value._native.dll is not snapshot.dll
            or type(value._provider) is not int
            or value._provider != snapshot.provider
            or snapshot.provider <= 0
            or type(value._key) is not int
            or value._key != snapshot.key
            or snapshot.key <= 0
            or type(value._public) is not bytes
            or value._public != snapshot.public
            or type(value._unique_name) is not str
            or value._unique_name != snapshot.unique_name
            or type(value._public_blob_sha256) is not str
            or value._public_blob_sha256 != snapshot.public_blob_sha256
        ):
            raise WindowsLPPIAuthorityKeyError("QUALIFIED_PRODUCTION_LPPI_AUTHORITY_KEY_REQUIRED")
        return snapshot
    except (AttributeError, TypeError) as exc:
        raise WindowsLPPIAuthorityKeyError(
            "QUALIFIED_PRODUCTION_LPPI_AUTHORITY_KEY_REQUIRED"
        ) from exc


def require_verified_production_lppi_authority_key(value: object) -> WindowsLPPIAuthorityKey:
    """Requalify all consequential uses, including creation, custody and signing."""
    snapshot = _qualified_snapshot(value)
    if type(value) is not WindowsLPPIAuthorityKey:
        raise WindowsLPPIAuthorityKeyError("QUALIFIED_PRODUCTION_LPPI_AUTHORITY_KEY_REQUIRED")
    try:
        accepted = _require_acceptance(snapshot.accepted)
        descriptor = _reservation(accepted, snapshot.reservation)
        _require_descriptor(accepted, descriptor)
        if descriptor.reservation_id != snapshot.reservation_id:
            raise WindowsLPPIAuthorityKeyError("AUTHORITY_LIFECYCLE_RESERVATION_CHANGED")
        public, unique, blob_digest = _qualify(value._native, value._key)
        if (public, unique, blob_digest) != (
            snapshot.public,
            snapshot.unique_name,
            snapshot.public_blob_sha256,
        ):
            raise WindowsLPPIAuthorityKeyError("AUTHORITY_NATIVE_IDENTITY_CHANGED")
        _require_identity(public, unique, descriptor)
    except Exception as exc:
        raise WindowsLPPIAuthorityKeyError(
            "QUALIFIED_PRODUCTION_LPPI_AUTHORITY_KEY_REQUIRED"
        ) from exc
    return value
