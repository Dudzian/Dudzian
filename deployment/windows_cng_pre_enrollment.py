"""Local Windows CNG qualification for the production pre-enrollment key.

This adapter does not authenticate a TPM enrollment exchange or a PDSA challenge.
Its public commitment records local identity continuity, not authority or TOFU
acceptance. The caller must keep the state directory in trusted machine storage.
TPM creation origin requires separate cryptographic attestation.
"""

from __future__ import annotations

import ctypes
import hashlib
import os
import re
import stat
import struct
import sys
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from weakref import WeakKeyDictionary

from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric import ec, utils

from bot_core.licensing.canonical import canonical_json_bytes, exact, parse_canonical
from bot_core.licensing.pre_enrollment import (
    PreEnrollmentRequestV1,
    public_key_fingerprint,
    validate_public_key,
)

PROVIDER = "Microsoft Platform Crypto Provider"
KEY_NAME = "CryptoHunter.Stage9.Production.PreEnrollment.v1"
ALGORITHM = "ECDSA_P256"
MACHINE_KEY = 0x20
SIGNING_ONLY = 0x02
HARDWARE = 0x01
SOFTWARE = 0x02
NTE_BAD_KEYSET = 0x80090016
NTE_EXISTS = 0x8009000F
NTE_BUFFER_TOO_SMALL = 0x80090028
P256_ORDER = 0xFFFFFFFF00000000FFFFFFFFFFFFFFFFBCE6FAADA7179E84F3B9CAC2FC632551
_STATE_NAME = "pre-enrollment-key-identity.json"
_LOCK_NAME = "pre-enrollment-key-identity.lock"
_HANDLE = ctypes.c_size_t
_DWORD = ctypes.c_uint32
_STATUS = ctypes.c_int32


class WindowsCNGPreEnrollmentError(RuntimeError):
    """A required local qualification or continuity check failed closed."""


def _check(status: int, operation: str) -> None:
    if status:
        raise WindowsCNGPreEnrollmentError(f"{operation}:0x{status & 0xFFFFFFFF:08X}")


def _sec1_from_public_blob(blob: bytes) -> bytes:
    if len(blob) != 72 or struct.unpack_from("<II", blob) != (0x31534345, 32):
        raise WindowsCNGPreEnrollmentError("INVALID_ECDSA_P256_PUBLIC_BLOB")
    public = b"\x04" + blob[8:]
    try:
        validate_public_key(public)
    except (TypeError, ValueError) as exc:
        raise WindowsCNGPreEnrollmentError("INVALID_ECDSA_P256_PUBLIC_POINT") from exc
    return public


class _NCryptAPI:
    """Fixed NCrypt ABI; no key import, deletion or private export entrypoint."""

    dll: Any

    def __init__(self) -> None:
        if sys.platform == "win32":
            # Load only the operating system's native boundary, not an application DLL.
            self.dll = ctypes.WinDLL("ncrypt.dll", use_last_error=True, winmode=0x800)
        else:
            raise WindowsCNGPreEnrollmentError("WINDOWS_REQUIRED")
        signatures = {
            "NCryptOpenStorageProvider": [ctypes.POINTER(_HANDLE), ctypes.c_wchar_p, _DWORD],
            "NCryptOpenKey": [
                _HANDLE,
                ctypes.POINTER(_HANDLE),
                ctypes.c_wchar_p,
                _DWORD,
                _DWORD,
            ],
            "NCryptCreatePersistedKey": [
                _HANDLE,
                ctypes.POINTER(_HANDLE),
                ctypes.c_wchar_p,
                ctypes.c_wchar_p,
                _DWORD,
                _DWORD,
            ],
            "NCryptSetProperty": [
                _HANDLE,
                ctypes.c_wchar_p,
                ctypes.c_void_p,
                _DWORD,
                _DWORD,
            ],
            "NCryptGetProperty": [
                _HANDLE,
                ctypes.c_wchar_p,
                ctypes.c_void_p,
                _DWORD,
                ctypes.POINTER(_DWORD),
                _DWORD,
            ],
            "NCryptFinalizeKey": [_HANDLE, _DWORD],
            "NCryptExportKey": [
                _HANDLE,
                _HANDLE,
                ctypes.c_wchar_p,
                ctypes.c_void_p,
                ctypes.c_void_p,
                _DWORD,
                ctypes.POINTER(_DWORD),
                _DWORD,
            ],
            "NCryptSignHash": [
                _HANDLE,
                ctypes.c_void_p,
                ctypes.c_void_p,
                _DWORD,
                ctypes.c_void_p,
                _DWORD,
                ctypes.POINTER(_DWORD),
                _DWORD,
            ],
            "NCryptFreeObject": [_HANDLE],
        }
        for name, arguments in signatures.items():
            function = getattr(self.dll, name)
            function.argtypes = arguments
            function.restype = _STATUS

    def open_provider(self) -> int:
        handle = _HANDLE()
        _check(
            self.dll.NCryptOpenStorageProvider(ctypes.byref(handle), PROVIDER, 0), "OPEN_PROVIDER"
        )
        if not handle.value:
            raise WindowsCNGPreEnrollmentError("EMPTY_PROVIDER_HANDLE")
        return handle.value

    def open_key(self, provider: int) -> int | None:
        handle = _HANDLE()
        status = self.dll.NCryptOpenKey(provider, ctypes.byref(handle), KEY_NAME, 0, MACHINE_KEY)
        if status & 0xFFFFFFFF == NTE_BAD_KEYSET:
            return None
        _check(status, "OPEN_KEY")
        if not handle.value:
            raise WindowsCNGPreEnrollmentError("EMPTY_KEY_HANDLE")
        return handle.value

    def create_key(self, provider: int) -> int:
        handle = _HANDLE()
        status = self.dll.NCryptCreatePersistedKey(
            provider,
            ctypes.byref(handle),
            ALGORITHM,
            KEY_NAME,
            0,
            MACHINE_KEY,
        )
        _check(status, "CREATE_KEY")
        if not handle.value:
            raise WindowsCNGPreEnrollmentError("EMPTY_KEY_HANDLE")
        try:
            for name, value in (("Export Policy", 0), ("Key Usage", SIGNING_ONLY)):
                number = _DWORD(value)
                _check(
                    self.dll.NCryptSetProperty(
                        handle.value,
                        name,
                        ctypes.byref(number),
                        ctypes.sizeof(number),
                        0,
                    ),
                    "SET_KEY_PROFILE",
                )
            _check(self.dll.NCryptFinalizeKey(handle.value, 0), "FINALIZE_KEY")
        except BaseException:
            self.free(handle.value)
            raise
        return handle.value

    def property(self, handle: int, name: str) -> bytes:
        size = _DWORD()
        status = self.dll.NCryptGetProperty(handle, name, None, 0, ctypes.byref(size), 0)
        if status & 0xFFFFFFFF not in (0, NTE_BUFFER_TOO_SMALL):
            _check(status, "READ_REQUIRED_PROPERTY")
        if not 0 < size.value <= 4096:
            raise WindowsCNGPreEnrollmentError("INVALID_PROPERTY_SIZE")
        buffer = (ctypes.c_ubyte * size.value)()
        capacity = size.value
        _check(
            self.dll.NCryptGetProperty(
                handle,
                name,
                buffer,
                capacity,
                ctypes.byref(size),
                0,
            ),
            "READ_REQUIRED_PROPERTY",
        )
        if size.value != capacity:
            raise WindowsCNGPreEnrollmentError("UNSTABLE_PROPERTY_SIZE")
        return bytes(buffer)

    def number(self, handle: int, name: str) -> int:
        raw = self.property(handle, name)
        if len(raw) != 4:
            raise WindowsCNGPreEnrollmentError("INVALID_NUMERIC_PROPERTY")
        return int(struct.unpack("<I", raw)[0])

    def export_allowed(self, handle: int) -> bool:
        """Read the PCP export permission; key origin requires TPM attestation."""
        # PCP_EXPORT_ALLOWED is a BOOLEAN, unlike the DWORD Export Policy.
        raw = self.property(handle, "PCP_EXPORT_ALLOWED")
        if raw not in (b"\0", b"\x01"):
            raise WindowsCNGPreEnrollmentError("INVALID_PCP_EXPORT_ALLOWED")
        return raw != b"\0"

    def text(self, handle: int, name: str) -> str:
        raw = self.property(handle, name)
        try:
            if len(raw) % 2 or not raw.endswith(b"\0\0"):
                raise ValueError("unterminated wide string")
            value = raw[:-2].decode("utf-16-le", errors="strict")
            if not value or "\0" in value:
                raise ValueError("invalid wide string")
            return value
        except (UnicodeError, ValueError) as exc:
            raise WindowsCNGPreEnrollmentError("INVALID_STRING_PROPERTY") from exc

    def public_blob(self, handle: int) -> bytes:
        size = _DWORD()
        _check(
            self.dll.NCryptExportKey(
                handle,
                0,
                "ECCPUBLICBLOB",
                None,
                None,
                0,
                ctypes.byref(size),
                0,
            ),
            "EXPORT_PUBLIC_KEY",
        )
        if size.value != 72:
            raise WindowsCNGPreEnrollmentError("INVALID_PUBLIC_BLOB_SIZE")
        buffer = (ctypes.c_ubyte * 72)()
        _check(
            self.dll.NCryptExportKey(
                handle,
                0,
                "ECCPUBLICBLOB",
                None,
                buffer,
                72,
                ctypes.byref(size),
                0,
            ),
            "EXPORT_PUBLIC_KEY",
        )
        if size.value != 72:
            raise WindowsCNGPreEnrollmentError("INVALID_PUBLIC_BLOB_SIZE")
        return bytes(buffer)

    def sign_digest(self, handle: int, digest: bytes) -> bytes:
        if len(digest) != 32:
            raise WindowsCNGPreEnrollmentError("INVALID_SIGNING_DIGEST")
        source = (ctypes.c_ubyte * 32).from_buffer_copy(digest)
        signature = (ctypes.c_ubyte * 64)()
        size = _DWORD()
        _check(
            self.dll.NCryptSignHash(
                handle,
                None,
                source,
                32,
                signature,
                64,
                ctypes.byref(size),
                0,
            ),
            "SIGN_REQUEST",
        )
        if size.value != 64:
            raise WindowsCNGPreEnrollmentError("INVALID_NATIVE_SIGNATURE")
        r = int.from_bytes(bytes(signature[:32]), "big")
        s = int.from_bytes(bytes(signature[32:]), "big")
        if not 0 < r < P256_ORDER or not 0 < s < P256_ORDER:
            raise WindowsCNGPreEnrollmentError("INVALID_NATIVE_SIGNATURE")
        return bytes(utils.encode_dss_signature(r, min(s, P256_ORDER - s)))

    def free(self, handle: int) -> None:
        _check(self.dll.NCryptFreeObject(handle), "CLOSE_NATIVE_HANDLE")


def _load_native() -> _NCryptAPI:
    return _NCryptAPI()


def _safe_path(path: Path, *, directory: bool = False) -> None:
    for component in (*reversed(path.parents), path):
        try:
            info = component.lstat()
        except FileNotFoundError:
            continue
        if stat.S_ISLNK(info.st_mode) or getattr(info, "st_file_attributes", 0) & 0x400:
            raise WindowsCNGPreEnrollmentError("REPARSE_OR_SYMLINK_STATE_PATH")
        if component == path:
            wanted = stat.S_ISDIR if directory else stat.S_ISREG
            if not wanted(info.st_mode) or (not directory and info.st_nlink > 1):
                raise WindowsCNGPreEnrollmentError("UNSAFE_STATE_PATH")


@contextmanager
def _state_lock(directory: Path) -> Iterator[Path]:
    directory = directory.absolute()
    _safe_path(directory, directory=True)
    directory.mkdir(parents=True, exist_ok=True)
    _safe_path(directory, directory=True)
    lock = directory / _LOCK_NAME
    _safe_path(lock)
    flags = os.O_RDWR | os.O_CREAT | getattr(os, "O_NOFOLLOW", 0)
    fd = os.open(lock, flags, 0o600)
    with os.fdopen(fd, "r+b") as stream:
        info = os.fstat(stream.fileno())
        if not stat.S_ISREG(info.st_mode) or info.st_nlink > 1:
            raise WindowsCNGPreEnrollmentError("UNSAFE_STATE_LOCK")
        if sys.platform == "win32":
            import msvcrt

            if info.st_size == 0:
                stream.write(b"\0")
                stream.flush()
            stream.seek(0)
            try:
                msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
            except OSError as exc:
                raise WindowsCNGPreEnrollmentError("IDENTITY_STATE_BUSY") from exc
            try:
                yield directory / _STATE_NAME
            finally:
                stream.seek(0)
                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            # Reachable only through a mocked native loader in cross-platform tests.
            import fcntl

            try:
                fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except OSError as exc:
                raise WindowsCNGPreEnrollmentError("IDENTITY_STATE_BUSY") from exc
            try:
                yield directory / _STATE_NAME
            finally:
                fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def _read_state(path: Path) -> dict[str, Any] | None:
    _safe_path(path)
    if not path.exists():
        return None
    try:
        if path.stat().st_size > 4096:
            raise ValueError("oversized state")
        value: dict[str, Any] = parse_canonical(path.read_bytes())
        exact(
            value,
            {
                "schema",
                "version",
                "provider",
                "key_name",
                "lifecycle",
                "public_key_fingerprint_sha256",
            },
            "pre-enrollment identity state",
        )
        if (
            value["schema"] != "WindowsCNGPreEnrollmentIdentityV1"
            or type(value["version"]) is not int
            or value["version"] != 1
            or value["provider"] != PROVIDER
            or value["key_name"] != KEY_NAME
            or value["lifecycle"] not in {"CREATION_RESERVED", "COMMITTED"}
        ):
            raise ValueError("invalid identity state")
        fingerprint = value["public_key_fingerprint_sha256"]
        if value["lifecycle"] == "CREATION_RESERVED":
            if fingerprint is not None:
                raise ValueError("reserved identity has fingerprint")
        elif not isinstance(fingerprint, str) or not re.fullmatch("[0-9a-f]{64}", fingerprint):
            raise ValueError("invalid identity fingerprint")
        return value
    except (ValueError, TypeError, OSError) as exc:
        raise WindowsCNGPreEnrollmentError("INVALID_IDENTITY_STATE") from exc


def _write_state(path: Path, *, fingerprint: str | None) -> None:
    _safe_path(path)
    value = {
        "schema": "WindowsCNGPreEnrollmentIdentityV1",
        "version": 1,
        "provider": PROVIDER,
        "key_name": KEY_NAME,
        "lifecycle": "CREATION_RESERVED" if fingerprint is None else "COMMITTED",
        "public_key_fingerprint_sha256": fingerprint,
    }
    temporary = path.with_name(path.name + "." + os.urandom(16).hex() + ".tmp")
    fd = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(canonical_json_bytes(value))
            stream.flush()
            os.fsync(stream.fileno())
        _safe_path(path)
        os.replace(temporary, path)
        if os.name != "nt":
            directory_fd = os.open(path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
    finally:
        temporary.unlink(missing_ok=True)


@dataclass(frozen=True)
class _VerifiedCNGKeySnapshot:
    native: _NCryptAPI
    dll: object
    provider: int
    key: int
    public: bytes
    unique_name: str
    state_directory: Path


_ISSUED_CNG_KEYS: WeakKeyDictionary[WindowsCNGPreEnrollmentKey, _VerifiedCNGKeySnapshot] = (
    WeakKeyDictionary()
)


class WindowsCNGPreEnrollmentKey:
    """Provider-owned key handle for qualification and request possession proof."""

    __slots__ = (
        "_native",
        "_provider",
        "_key",
        "_public",
        "_unique_name",
        "_state_directory",
        "__weakref__",
    )

    _native: _NCryptAPI
    _provider: int
    _key: int
    _public: bytes
    _unique_name: str
    _state_directory: Path

    def __init__(self) -> None:
        raise TypeError("use WindowsCNGPreEnrollmentKey.open_or_create()")

    @classmethod
    def open_or_create(cls, state_directory: Path) -> WindowsCNGPreEnrollmentKey:
        if cls is not WindowsCNGPreEnrollmentKey:
            raise TypeError("production CNG adapter cannot be substituted")
        native = _load_native()
        if type(native) is not _NCryptAPI:
            raise TypeError("production CNG adapter requires the exact native boundary")
        instance = object.__new__(cls)
        instance._native, instance._provider, instance._key = native, 0, 0
        try:
            instance._provider = native.open_provider()
            with _state_lock(state_directory) as state_path:
                state = _read_state(state_path)
                instance._key = native.open_key(instance._provider) or 0
                if state is None and instance._key:
                    raise WindowsCNGPreEnrollmentError("PREEXISTING_UNRESERVED_IDENTITY")
                if not instance._key:
                    if state is not None:
                        raise WindowsCNGPreEnrollmentError(
                            "PREVIOUS_IDENTITY_MISSING_RECONCILIATION_REQUIRED"
                        )
                    # Durable before any identity-generating native effect.
                    _write_state(state_path, fingerprint=None)
                    instance._key = native.create_key(instance._provider)
                    native.free(instance._key)
                    instance._key = 0
                    instance._key = native.open_key(instance._provider) or 0
                    if not instance._key:
                        raise WindowsCNGPreEnrollmentError("CREATED_IDENTITY_NOT_REOPENABLE")
                instance._public, instance._unique_name = instance._qualify()
                fingerprint = public_key_fingerprint(instance._public)
                if state is not None and state["lifecycle"] == "COMMITTED":
                    if state["public_key_fingerprint_sha256"] != fingerprint:
                        raise WindowsCNGPreEnrollmentError("COMMITTED_IDENTITY_MISMATCH")
                else:
                    # Commit only creation reserved by this lifecycle, including recovery.
                    _write_state(state_path, fingerprint=fingerprint)
                instance._state_directory = state_path.parent
            _ISSUED_CNG_KEYS[instance] = _VerifiedCNGKeySnapshot(
                instance._native,
                instance._native.dll,
                instance._provider,
                instance._key,
                instance._public,
                instance._unique_name,
                instance._state_directory,
            )
            return instance
        except BaseException:
            instance.close()
            raise

    @classmethod
    def open_existing(cls, state_directory: Path) -> WindowsCNGPreEnrollmentKey:
        """Open committed production identity without reserving or generating a key."""
        if cls is not WindowsCNGPreEnrollmentKey:
            raise TypeError("production CNG adapter cannot be substituted")
        native = _load_native()
        if type(native) is not _NCryptAPI:
            raise TypeError("production CNG adapter requires the exact native boundary")
        instance = object.__new__(cls)
        instance._native, instance._provider, instance._key = native, 0, 0
        try:
            instance._provider = native.open_provider()
            with _state_lock(state_directory) as state_path:
                state = _read_state(state_path)
                instance._key = native.open_key(instance._provider) or 0
                if state is None:
                    raise WindowsCNGPreEnrollmentError("RETAINED_PRE_ENROLLMENT_IDENTITY_REQUIRED")
                if state["lifecycle"] != "COMMITTED":
                    raise WindowsCNGPreEnrollmentError("COMMITTED_PRE_ENROLLMENT_IDENTITY_REQUIRED")
                if not instance._key:
                    raise WindowsCNGPreEnrollmentError(
                        "PREVIOUS_IDENTITY_MISSING_RECONCILIATION_REQUIRED"
                    )
                instance._public, instance._unique_name = instance._qualify()
                if state["public_key_fingerprint_sha256"] != public_key_fingerprint(
                    instance._public
                ):
                    raise WindowsCNGPreEnrollmentError("COMMITTED_IDENTITY_MISMATCH")
                instance._state_directory = state_path.parent
            _ISSUED_CNG_KEYS[instance] = _VerifiedCNGKeySnapshot(
                native,
                native.dll,
                instance._provider,
                instance._key,
                instance._public,
                instance._unique_name,
                instance._state_directory,
            )
            return require_verified_production_cng_key(instance)
        except BaseException:
            instance.close()
            raise

    def _qualify(self) -> tuple[bytes, str]:
        if not self._key:
            raise WindowsCNGPreEnrollmentError("KEY_HANDLE_CLOSED")
        native, key = self._native, self._key
        raw_provider = native.property(key, "Provider Handle")
        if len(raw_provider) != ctypes.sizeof(_HANDLE):
            raise WindowsCNGPreEnrollmentError("INVALID_PROVIDER_HANDLE_PROPERTY")
        provider = int.from_bytes(raw_provider, "little")
        if not provider:
            raise WindowsCNGPreEnrollmentError("EMPTY_PROVIDER_HANDLE_PROPERTY")
        try:
            implementation = native.number(provider, "Impl Type")
            if (
                native.text(provider, "Name") != PROVIDER
                or not implementation & HARDWARE
                or implementation & SOFTWARE
            ):
                raise WindowsCNGPreEnrollmentError("KEY_PROVIDER_PROFILE_REJECTED")
        finally:
            # Provider Handle returns an owned reference, not a borrowed handle.
            native.free(provider)
        if (
            native.text(key, "Algorithm Name") != ALGORITHM
            or native.text(key, "Algorithm Group") != "ECDSA"
            or native.number(key, "Length") != 256
            or native.number(key, "Key Usage") != SIGNING_ONLY
            or native.number(key, "Export Policy") != 0
            or native.export_allowed(key)
            or native.number(key, "Key Type") != MACHINE_KEY
            or native.text(key, "Name") != KEY_NAME
        ):
            raise WindowsCNGPreEnrollmentError("KEY_PROVIDER_PROFILE_REJECTED")
        unique = native.text(key, "Unique Name")
        if not re.fullmatch(r"[A-Za-z0-9._\\:-]{1,256}", unique):
            raise WindowsCNGPreEnrollmentError("UNSAFE_KEY_UNIQUE_NAME")
        return _sec1_from_public_blob(native.public_blob(key)), unique

    @property
    def public_key_bytes(self) -> bytes:
        if not self._key:
            raise WindowsCNGPreEnrollmentError("KEY_HANDLE_CLOSED")
        return self._public

    @property
    def public_evidence(self) -> dict[str, Any]:
        require_verified_production_cng_key(self)
        return {
            "schema": "WindowsCNGPreEnrollmentQualificationV1",
            "environment": "PRODUCTION",
            "qualification_scope": "LOCAL_CNG_PROVIDER_ONLY",
            "cng_provider_name": PROVIDER,
            "cng_key_name": KEY_NAME,
            "cng_key_unique_name": self._unique_name,
            "algorithm_profile": "ECDSA-P256-SHA256",
            "machine_scope": True,
            "signing_only": True,
            "private_export_policy": 0,
            "public_key_canonical_bytes": self.public_key_bytes.hex(),
            "public_key_fingerprint_sha256": public_key_fingerprint(self.public_key_bytes),
            "tpm_attestation": "NOT_VERIFIED",
            "legal_production_enrollment": "NOT_PERFORMED",
        }

    def sign_request(
        self,
        request: PreEnrollmentRequestV1,
        *,
        production_trust_context: object,
    ) -> bytes:
        if type(request) is not PreEnrollmentRequestV1:
            raise TypeError("exact canonical PreEnrollmentRequestV1 required")
        validated = PreEnrollmentRequestV1.from_canonical_bytes(request.canonical_bytes)
        validated.require_production_trust_binding(production_trust_context)
        require_verified_production_cng_key(self)
        document = validated.document
        current = self._public
        if current != self._public or (
            document["pre_enrollment_public_key_canonical_bytes"] != current.hex()
            or document["pre_enrollment_public_key_fingerprint_sha256"]
            != public_key_fingerprint(current)
        ):
            raise WindowsCNGPreEnrollmentError("REQUEST_KEY_IDENTITY_MISMATCH")
        digest = hashlib.sha256(validated.signing_bytes).digest()
        signature = self._native.sign_digest(self._key, digest)
        validate_public_key(current).verify(
            signature,
            digest,
            ec.ECDSA(utils.Prehashed(hashes.SHA256())),
        )
        return signature

    def sign_package_acceptance(
        self, challenge_raw: bytes, *, production_trust_context: object
    ) -> bytes:
        """Sign only the frozen package acceptance challenge with the retained key."""
        from bot_core.licensing.lppi_package_acceptance import (
            PACKAGE_ACCEPTANCE_DOMAIN,
            validate_package_acceptance_challenge,
        )
        from deployment.windows_stage9_production_trust import (
            require_current_production_trust_context,
        )

        require_current_production_trust_context(production_trust_context)
        require_verified_production_cng_key(self)
        challenge = validate_package_acceptance_challenge(challenge_raw)
        if challenge["pre_enrollment_public_key_fingerprint_sha256"] != public_key_fingerprint(
            self.public_key_bytes
        ):
            raise WindowsCNGPreEnrollmentError("PACKAGE_ACCEPTANCE_KEY_MISMATCH")
        return self._sign_lppi_bytes(
            PACKAGE_ACCEPTANCE_DOMAIN + hashlib.sha256(challenge_raw).digest()
        )

    def sign_authority_continuity(
        self, binding_raw: bytes, *, production_trust_context: object
    ) -> bytes:
        """Sign the exact initial successor binding with its pre-enrollment key."""
        from bot_core.licensing.lppi_package_acceptance import validate_continuity_payload
        from deployment.windows_stage9_production_trust import (
            require_current_production_trust_context,
        )

        require_current_production_trust_context(production_trust_context)
        require_verified_production_cng_key(self)
        binding = validate_continuity_payload(binding_raw)
        if binding["pre_enrollment_public_key_fingerprint_sha256"] != public_key_fingerprint(
            self.public_key_bytes
        ):
            raise WindowsCNGPreEnrollmentError("REJECT_CONTINUITY_SIGNATURE_KEY_MISMATCH")
        return self._sign_lppi_bytes(
            b"CryptoHunter.Stage9.LPPIAuthorityKeyContinuity.v1\x00"
            + hashlib.sha256(binding_raw).digest()
        )

    def _sign_lppi_bytes(self, signing_bytes: bytes) -> bytes:
        require_verified_production_cng_key(self)
        digest = hashlib.sha256(signing_bytes).digest()
        signature = self._native.sign_digest(self._key, digest)
        validate_public_key(self.public_key_bytes).verify(
            signature, digest, ec.ECDSA(utils.Prehashed(hashes.SHA256()))
        )
        return signature

    def close(self) -> None:
        _ISSUED_CNG_KEYS.pop(self, None)
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
            raise WindowsCNGPreEnrollmentError("CLOSE_NATIVE_HANDLES_FAILED") from failures[0]

    def __enter__(self) -> WindowsCNGPreEnrollmentKey:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()


def require_verified_production_cng_key(value: object) -> WindowsCNGPreEnrollmentKey:
    """Require factory-issued identity, current qualification and committed continuity."""
    if type(value) is not WindowsCNGPreEnrollmentKey:
        raise WindowsCNGPreEnrollmentError("VERIFIED_PRODUCTION_CNG_KEY_REQUIRED")
    try:
        snapshot = _ISSUED_CNG_KEYS.get(value)
        valid = (
            snapshot is not None
            and type(value._native) is _NCryptAPI
            and value._native is snapshot.native
            and value._native.dll is snapshot.dll
            and type(value._provider) is int
            and value._provider == snapshot.provider > 0
            and type(value._key) is int
            and value._key == snapshot.key > 0
            and type(value._public) is bytes
            and value._public == snapshot.public
            and type(value._unique_name) is str
            and value._unique_name == snapshot.unique_name
            and type(value._state_directory) is type(snapshot.state_directory)
            and value._state_directory == snapshot.state_directory
        )
        if not valid or snapshot is None:
            raise WindowsCNGPreEnrollmentError("VERIFIED_PRODUCTION_CNG_KEY_REQUIRED")
        with _state_lock(snapshot.state_directory) as state_path:
            state = _read_state(state_path)
            if (
                state is None
                or state["lifecycle"] != "COMMITTED"
                or state["public_key_fingerprint_sha256"] != public_key_fingerprint(snapshot.public)
            ):
                raise WindowsCNGPreEnrollmentError("VERIFIED_PRODUCTION_CNG_KEY_REQUIRED")
            public, unique = value._qualify()
            if public != snapshot.public or unique != snapshot.unique_name:
                raise WindowsCNGPreEnrollmentError("VERIFIED_PRODUCTION_CNG_KEY_REQUIRED")
    except (AttributeError, TypeError, ValueError, OSError, WindowsCNGPreEnrollmentError) as exc:
        raise WindowsCNGPreEnrollmentError("VERIFIED_PRODUCTION_CNG_KEY_REQUIRED") from exc
    return value
