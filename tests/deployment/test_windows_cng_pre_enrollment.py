"""TEST_ONLY NCrypt ABI simulator; no physical TPM or production key is used."""

from __future__ import annotations

import ctypes
import json
import struct
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import ec, utils

from bot_core.licensing.canonical import canonical_json_bytes
from bot_core.licensing.pre_enrollment import (
    ALGORITHM_PROFILE,
    PAYLOAD_FIELDS,
    PDSA_TRUST_DOMAIN,
    SCHEMA_VERSION,
    PreEnrollmentRequestV1,
    public_key_fingerprint,
)
from bot_core.licensing.product_profile import PRODUCT_NAME, PRODUCTION_PRODUCT_PROFILE
from deployment import windows_cng_pre_enrollment as adapter


def set_number(pointer: object, value: int, kind: object = ctypes.c_uint32) -> None:
    ctypes.cast(pointer, ctypes.POINTER(kind))[0] = value


def wide(value: str) -> bytes:
    return value.encode("utf-16-le") + b"\0\0"


class TestOnlyFunction:
    __test__ = False

    def __init__(self, function: object) -> None:
        self.function = function

    def __call__(self, *arguments: object) -> int:
        return self.function(*arguments)


class TestOnlyNCryptDLL:
    """Only tests monkeypatch the private loader; production exposes no injection."""

    __test__ = False
    environment = "TEST_ONLY"

    def __init__(self, *, existing: bool = False) -> None:
        self.private = ec.derive_private_key(71, ec.SECP256R1())
        self.persisted = existing
        self.calls: list[tuple] = []
        self.properties = {
            (11, "Name"): wide(adapter.PROVIDER),
            (11, "Impl Type"): struct.pack("<I", adapter.HARDWARE),
            (22, "Provider Handle"): (11).to_bytes(ctypes.sizeof(ctypes.c_size_t), "little"),
            (22, "Name"): wide(adapter.KEY_NAME),
            (22, "Unique Name"): wide("test-only-unique-name"),
            (22, "Algorithm Name"): wide(adapter.ALGORITHM),
            (22, "Algorithm Group"): wide("ECDSA"),
            (22, "Length"): struct.pack("<I", 256),
            (22, "Key Type"): struct.pack("<I", adapter.MACHINE_KEY),
            (22, "Key Usage"): struct.pack("<I", adapter.SIGNING_ONLY),
            (22, "Export Policy"): struct.pack("<I", 0),
            (22, "PCP_EXPORT_ALLOWED"): b"\0",
        }
        self.public_override: bytes | None = None
        self.open_error = 0
        self.create_error = 0
        self.finalize_error = 0
        self.finalize_lost_response = False
        self.property_error: str | None = None
        self.sign_error = 0
        self.signature_override: bytes | None = None
        self.reopen_missing = False
        for name in (
            "NCryptOpenStorageProvider",
            "NCryptOpenKey",
            "NCryptCreatePersistedKey",
            "NCryptSetProperty",
            "NCryptGetProperty",
            "NCryptFinalizeKey",
            "NCryptExportKey",
            "NCryptSignHash",
            "NCryptFreeObject",
        ):
            setattr(self, name, TestOnlyFunction(getattr(self, "_" + name)))

    def _NCryptOpenStorageProvider(self, output: object, provider: str, flags: int) -> int:
        self.calls.append(("provider", provider, flags))
        set_number(output, 11, ctypes.c_size_t)
        return 0

    def _NCryptOpenKey(
        self, provider: int, output: object, name: str, spec: int, flags: int
    ) -> int:
        self.calls.append(("open", name, flags))
        if self.open_error:
            return self.open_error
        if not self.persisted or self.reopen_missing:
            return adapter.NTE_BAD_KEYSET
        set_number(output, 22, ctypes.c_size_t)
        return 0

    def _NCryptCreatePersistedKey(
        self,
        provider: int,
        output: object,
        algorithm: str,
        name: str,
        spec: int,
        flags: int,
    ) -> int:
        self.calls.append(("create", algorithm, name, flags))
        if self.create_error:
            return self.create_error
        if self.persisted:
            return adapter.NTE_EXISTS
        set_number(output, 22, ctypes.c_size_t)
        return 0

    def _NCryptSetProperty(self, key: int, name: str, source: object, size: int, flags: int) -> int:
        raw = ctypes.string_at(source, size)
        self.calls.append(("set", name, raw, flags))
        self.properties[(key, name)] = raw
        return 0

    def _NCryptGetProperty(
        self,
        key: int,
        name: str,
        output: object,
        capacity: int,
        size: object,
        flags: int,
    ) -> int:
        self.calls.append(("property", key, name))
        if name == self.property_error or (key, name) not in self.properties:
            return 0x80090029
        raw = self.properties[(key, name)]
        set_number(size, len(raw))
        if output is not None:
            assert capacity >= len(raw)
            ctypes.memmove(output, raw, len(raw))
        return 0

    def _NCryptFinalizeKey(self, key: int, flags: int) -> int:
        self.calls.append(("finalize", flags))
        if not self.finalize_error or self.finalize_lost_response:
            self.persisted = True
        return self.finalize_error

    def _NCryptExportKey(
        self,
        key: int,
        other: int,
        kind: str,
        parameters: object,
        output: object,
        capacity: int,
        size: object,
        flags: int,
    ) -> int:
        self.calls.append(("export", kind))
        assert kind == "ECCPUBLICBLOB", "private export is forbidden even in the simulator"
        sec1 = self.private.public_key().public_bytes(
            serialization.Encoding.X962,
            serialization.PublicFormat.UncompressedPoint,
        )
        raw = self.public_override or struct.pack("<II", 0x31534345, 32) + sec1[1:]
        set_number(size, len(raw))
        if output is not None:
            assert capacity >= len(raw)
            ctypes.memmove(output, raw, len(raw))
        return 0

    def _NCryptSignHash(
        self,
        key: int,
        padding: object,
        digest: object,
        digest_size: int,
        output: object,
        capacity: int,
        size: object,
        flags: int,
    ) -> int:
        self.calls.append(("sign", ctypes.string_at(digest, digest_size)))
        if self.sign_error:
            return self.sign_error
        der = self.private.sign(
            ctypes.string_at(digest, digest_size),
            ec.ECDSA(utils.Prehashed(hashes.SHA256())),
        )
        r, s = utils.decode_dss_signature(der)
        # Deliberately native high-S: the adapter must normalize it.
        raw = self.signature_override or (
            r.to_bytes(32, "big") + max(s, adapter.P256_ORDER - s).to_bytes(32, "big")
        )
        set_number(size, len(raw))
        ctypes.memmove(output, raw, len(raw))
        return 0

    def _NCryptFreeObject(self, handle: int) -> int:
        self.calls.append(("free", handle))
        return 0


@pytest.fixture
def dll(monkeypatch: pytest.MonkeyPatch) -> TestOnlyNCryptDLL:
    simulation = TestOnlyNCryptDLL()
    native = object.__new__(adapter._NCryptAPI)
    native.dll = simulation
    monkeypatch.setattr(adapter, "_load_native", lambda: native)
    return simulation


def request_for(public: bytes) -> PreEnrollmentRequestV1:
    value = dict.fromkeys(PAYLOAD_FIELDS, "a" * 64)
    value.update(
        {
            "schema_version": SCHEMA_VERSION,
            "environment": "PRODUCTION",
            "product": PRODUCT_NAME,
            "product_profile": PRODUCTION_PRODUCT_PROFILE,
            "pdsa_trust_domain": PDSA_TRUST_DOMAIN,
            "pdsa_challenge_id": "pchal_0194d8bc-1234-7000-8000-0123456789ab",
            "pre_enrollment_public_key_algorithm_profile": ALGORITHM_PROFILE,
            "pre_enrollment_public_key_canonical_bytes": public.hex(),
            "pre_enrollment_public_key_fingerprint_sha256": public_key_fingerprint(public),
            "release_policy_generation": 1,
        }
    )
    return PreEnrollmentRequestV1.from_mapping(value)


@pytest.fixture
def test_only_trust(monkeypatch: pytest.MonkeyPatch) -> object:
    """Isolate native signing from canonical production trust loader tests."""
    token = object()

    def require_trust(_request: PreEnrollmentRequestV1, context: object) -> None:
        if context is not token:
            raise RuntimeError("TEST_ONLY_EXPECTED_TRUST_CAPABILITY_REQUIRED")

    monkeypatch.setattr(PreEnrollmentRequestV1, "require_production_trust_binding", require_trust)
    return token


def test_fixed_provider_generation_and_public_only_evidence(
    dll: TestOnlyNCryptDLL,
    tmp_path: Path,
) -> None:
    with adapter.WindowsCNGPreEnrollmentKey.open_or_create(tmp_path) as key:
        evidence = key.public_evidence
        assert len(key.public_key_bytes) == 65
        assert evidence["public_key_fingerprint_sha256"] == public_key_fingerprint(
            key.public_key_bytes
        )
        assert evidence["qualification_scope"] == "LOCAL_CNG_PROVIDER_ONLY"
        assert evidence["tpm_attestation"] == "NOT_VERIFIED"
        assert evidence["legal_production_enrollment"] == "NOT_PERFORMED"
        assert not any(
            marker in json.dumps(evidence) for marker in ("PRIVATE KEY", "private_blob", "password")
        )
    assert ("provider", adapter.PROVIDER, 0) in dll.calls
    assert ("create", "ECDSA_P256", adapter.KEY_NAME, adapter.MACHINE_KEY) in dll.calls
    assert ("set", "Export Policy", bytes(4), 0) in dll.calls
    assert ("set", "Key Usage", struct.pack("<I", 2), 0) in dll.calls
    assert all(call[1] == "ECCPUBLICBLOB" for call in dll.calls if call[0] == "export")
    assert dll.persisted  # Close releases handles; it never deletes a persisted key.


def test_retry_and_new_process_reuse_exact_identity(dll: TestOnlyNCryptDLL, tmp_path: Path) -> None:
    with adapter.WindowsCNGPreEnrollmentKey.open_or_create(tmp_path) as first:
        identity = first.public_key_bytes
    with adapter.WindowsCNGPreEnrollmentKey.open_or_create(tmp_path) as second:
        assert second.public_key_bytes == identity
    assert sum(call[0] == "create" for call in dll.calls) == 1
    state = json.loads((tmp_path / adapter._STATE_NAME).read_bytes())
    assert state["lifecycle"] == "COMMITTED"
    assert state["public_key_fingerprint_sha256"] == public_key_fingerprint(identity)


def test_existing_qualified_key_reused_locally_not_authority(
    dll: TestOnlyNCryptDLL,
    tmp_path: Path,
) -> None:
    dll.persisted = True
    with adapter.WindowsCNGPreEnrollmentKey.open_or_create(tmp_path) as key:
        assert key.public_evidence["tpm_attestation"] == "NOT_VERIFIED"
    assert not any(call[0] == "create" for call in dll.calls)


@pytest.mark.parametrize(
    ("handle", "property_name", "raw"),
    [
        (11, "Name", wide("Microsoft Software Key Storage Provider")),
        (11, "Impl Type", struct.pack("<I", adapter.SOFTWARE)),
        (11, "Impl Type", struct.pack("<I", adapter.HARDWARE | adapter.SOFTWARE)),
        (11, "Impl Type", bytes(4)),
        (22, "Algorithm Name", wide("RSA")),
        (22, "Algorithm Name", wide("Ed25519")),
        (22, "Algorithm Name", wide("ECDSA_P384")),
        (22, "Algorithm Group", wide("ECDH")),
        (22, "Length", struct.pack("<I", 384)),
        (22, "Key Usage", struct.pack("<I", 3)),
        (22, "Key Usage", bytes(4)),
        (22, "Export Policy", struct.pack("<I", 1)),
        (22, "Export Policy", struct.pack("<I", 8)),
        (22, "Key Type", bytes(4)),
        (22, "Name", wide("caller-substitute-key")),
        (22, "Unique Name", wide("PRIVATE KEY\nsecret")),
        (22, "PCP_EXPORT_ALLOWED", b"\x01"),
        (22, "PCP_EXPORT_ALLOWED", bytes(4)),
        (22, "Provider Handle", bytes(ctypes.sizeof(ctypes.c_size_t))),
    ],
)
def test_existing_wrong_profile_fails_without_replacement(
    dll: TestOnlyNCryptDLL,
    tmp_path: Path,
    handle: int,
    property_name: str,
    raw: bytes,
) -> None:
    dll.persisted = True
    dll.properties[(handle, property_name)] = raw
    with pytest.raises(adapter.WindowsCNGPreEnrollmentError):
        adapter.WindowsCNGPreEnrollmentKey.open_or_create(tmp_path)
    assert not any(call[0] == "create" for call in dll.calls)
    assert not (tmp_path / adapter._STATE_NAME).exists()


@pytest.mark.parametrize(
    "property_name",
    [
        "Provider Handle",
        "Impl Type",
        "Name",
        "Algorithm Name",
        "Length",
        "Key Usage",
        "Export Policy",
        "PCP_EXPORT_ALLOWED",
        "Key Type",
        "Unique Name",
    ],
)
def test_missing_required_property_fails_closed(
    dll: TestOnlyNCryptDLL,
    tmp_path: Path,
    property_name: str,
) -> None:
    dll.persisted = True
    dll.property_error = property_name
    with pytest.raises(adapter.WindowsCNGPreEnrollmentError, match="READ_REQUIRED_PROPERTY"):
        adapter.WindowsCNGPreEnrollmentKey.open_or_create(tmp_path)


@pytest.mark.parametrize(
    "blob",
    [
        b"",
        b"x" * 71,
        b"x" * 73,
        struct.pack("<II", 0x33534345, 32) + bytes(64),  # P384 magic with fake P256 size.
        struct.pack("<II", 0x31534345, 48) + bytes(64),
        struct.pack("<II", 0x31534345, 32) + bytes(64),  # Off curve.
    ],
)
def test_public_blob_rejects_wrong_curve_magic_size_and_point(blob: bytes) -> None:
    with pytest.raises(adapter.WindowsCNGPreEnrollmentError):
        adapter._sec1_from_public_blob(blob)


def test_public_blob_roundtrip_and_fingerprint(dll: TestOnlyNCryptDLL, tmp_path: Path) -> None:
    with adapter.WindowsCNGPreEnrollmentKey.open_or_create(tmp_path) as key:
        sec1 = key.public_key_bytes
        assert adapter._sec1_from_public_blob(struct.pack("<II", 0x31534345, 32) + sec1[1:]) == sec1
        assert request_for(sec1).document[
            "pre_enrollment_public_key_fingerprint_sha256"
        ] == public_key_fingerprint(sec1)


def test_provider_reference_is_freed(dll: TestOnlyNCryptDLL, tmp_path: Path) -> None:
    with adapter.WindowsCNGPreEnrollmentKey.open_or_create(tmp_path):
        assert ("free", 11) in dll.calls  # Owned Provider Handle property reference.
    assert sum(call == ("free", 11) for call in dll.calls) == 2  # Property plus initial provider.


def test_open_failure_never_falls_back_or_creates(dll: TestOnlyNCryptDLL, tmp_path: Path) -> None:
    dll.open_error = 0x80090010
    with pytest.raises(adapter.WindowsCNGPreEnrollmentError, match="OPEN_KEY"):
        adapter.WindowsCNGPreEnrollmentKey.open_or_create(tmp_path)
    assert not any(call[0] == "create" for call in dll.calls)
    assert not (tmp_path / adapter._STATE_NAME).exists()


def test_creation_failure_reserves_identity_and_requires_reconciliation(
    dll: TestOnlyNCryptDLL,
    tmp_path: Path,
) -> None:
    dll.create_error = 0x80090020
    with pytest.raises(adapter.WindowsCNGPreEnrollmentError, match="CREATE_KEY"):
        adapter.WindowsCNGPreEnrollmentKey.open_or_create(tmp_path)
    dll.create_error = 0
    with pytest.raises(adapter.WindowsCNGPreEnrollmentError, match="RECONCILIATION_REQUIRED"):
        adapter.WindowsCNGPreEnrollmentKey.open_or_create(tmp_path)
    assert sum(call[0] == "create" for call in dll.calls) == 1


def test_finalize_lost_response_reconciles_without_new_identity(
    dll: TestOnlyNCryptDLL,
    tmp_path: Path,
) -> None:
    dll.finalize_error, dll.finalize_lost_response = 0x80090020, True
    with pytest.raises(adapter.WindowsCNGPreEnrollmentError, match="FINALIZE_KEY"):
        adapter.WindowsCNGPreEnrollmentKey.open_or_create(tmp_path)
    assert dll.persisted
    dll.finalize_error = 0
    with adapter.WindowsCNGPreEnrollmentKey.open_or_create(tmp_path):
        pass
    assert sum(call[0] == "create" for call in dll.calls) == 1
    assert json.loads((tmp_path / adapter._STATE_NAME).read_bytes())["lifecycle"] == "COMMITTED"


def test_committed_missing_key_never_regenerated(dll: TestOnlyNCryptDLL, tmp_path: Path) -> None:
    with adapter.WindowsCNGPreEnrollmentKey.open_or_create(tmp_path):
        pass
    dll.persisted = False
    with pytest.raises(adapter.WindowsCNGPreEnrollmentError, match="RECONCILIATION_REQUIRED"):
        adapter.WindowsCNGPreEnrollmentKey.open_or_create(tmp_path)
    assert sum(call[0] == "create" for call in dll.calls) == 1


def test_committed_changed_identity_rejected(dll: TestOnlyNCryptDLL, tmp_path: Path) -> None:
    with adapter.WindowsCNGPreEnrollmentKey.open_or_create(tmp_path):
        pass
    original = (tmp_path / adapter._STATE_NAME).read_bytes()
    dll.private = ec.derive_private_key(72, ec.SECP256R1())
    with pytest.raises(adapter.WindowsCNGPreEnrollmentError, match="COMMITTED_IDENTITY_MISMATCH"):
        adapter.WindowsCNGPreEnrollmentKey.open_or_create(tmp_path)
    assert (tmp_path / adapter._STATE_NAME).read_bytes() == original
    assert sum(call[0] == "create" for call in dll.calls) == 1


def test_malformed_state_fail_closed(dll: TestOnlyNCryptDLL, tmp_path: Path) -> None:
    state = tmp_path / adapter._STATE_NAME
    state.write_bytes(b'{"schema":"TEST_ONLY"}')
    with pytest.raises(adapter.WindowsCNGPreEnrollmentError, match="INVALID_IDENTITY_STATE"):
        adapter.WindowsCNGPreEnrollmentKey.open_or_create(tmp_path)
    assert not any(call[0] == "create" for call in dll.calls)


def test_symlink_state_does_not_touch_target(
    dll: TestOnlyNCryptDLL,
    tmp_path: Path,
) -> None:
    target = tmp_path / "protected.json"
    target.write_bytes(b"unchanged")
    (tmp_path / adapter._STATE_NAME).symlink_to(target)
    with pytest.raises(adapter.WindowsCNGPreEnrollmentError, match="SYMLINK"):
        adapter.WindowsCNGPreEnrollmentKey.open_or_create(tmp_path)
    assert target.read_bytes() == b"unchanged"
    assert not any(call[0] == "create" for call in dll.calls)


def test_identity_reservation_is_durable_before_creation(
    dll: TestOnlyNCryptDLL,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = dll.NCryptCreatePersistedKey.function

    def inspect_reservation(*arguments: object) -> int:
        value = json.loads((tmp_path / adapter._STATE_NAME).read_bytes())
        assert value["lifecycle"] == "CREATION_RESERVED"
        assert value["public_key_fingerprint_sha256"] is None
        return original(*arguments)

    monkeypatch.setattr(dll.NCryptCreatePersistedKey, "function", inspect_reservation)
    with adapter.WindowsCNGPreEnrollmentKey.open_or_create(tmp_path):
        pass


def test_possession_signature_exact_request_and_low_s(
    dll: TestOnlyNCryptDLL,
    tmp_path: Path,
    test_only_trust: object,
) -> None:
    with adapter.WindowsCNGPreEnrollmentKey.open_or_create(tmp_path) as key:
        request = request_for(key.public_key_bytes)
        signature = key.sign_request(request, production_trust_context=test_only_trust)
        request.verify_signature(signature)
        r, s = utils.decode_dss_signature(signature)
        assert 0 < r < adapter.P256_ORDER and 0 < s <= adapter.P256_ORDER // 2


def test_wrong_request_key_cannot_be_signed(
    dll: TestOnlyNCryptDLL,
    tmp_path: Path,
    test_only_trust: object,
) -> None:
    with adapter.WindowsCNGPreEnrollmentKey.open_or_create(tmp_path) as key:
        other = (
            ec.derive_private_key(73, ec.SECP256R1())
            .public_key()
            .public_bytes(
                serialization.Encoding.X962,
                serialization.PublicFormat.UncompressedPoint,
            )
        )
        with pytest.raises(
            adapter.WindowsCNGPreEnrollmentError, match="REQUEST_KEY_IDENTITY_MISMATCH"
        ):
            key.sign_request(request_for(other), production_trust_context=test_only_trust)
        with pytest.raises(TypeError, match="exact canonical"):
            key.sign_request(
                SimpleNamespace(canonical_bytes=b"TEST_ONLY"),
                production_trust_context=test_only_trust,
            )
    assert not any(call[0] == "sign" for call in dll.calls)


def test_native_malformed_signature_rejected(
    dll: TestOnlyNCryptDLL,
    tmp_path: Path,
    test_only_trust: object,
) -> None:
    dll.signature_override = bytes(64)
    with adapter.WindowsCNGPreEnrollmentKey.open_or_create(tmp_path) as key:
        with pytest.raises(adapter.WindowsCNGPreEnrollmentError, match="INVALID_NATIVE_SIGNATURE"):
            key.sign_request(
                request_for(key.public_key_bytes), production_trust_context=test_only_trust
            )


@pytest.mark.parametrize("context", [None, object(), SimpleNamespace(environment="PRODUCTION")])
def test_unverified_production_trust_cannot_sign(
    dll: TestOnlyNCryptDLL,
    tmp_path: Path,
    context: object,
) -> None:
    from deployment.windows_stage9_production_trust import ProductionTrustUnavailable

    with adapter.WindowsCNGPreEnrollmentKey.open_or_create(tmp_path) as key:
        with pytest.raises(
            ProductionTrustUnavailable, match="VERIFIED_PRODUCTION_TRUST_CONTEXT_REQUIRED"
        ):
            key.sign_request(request_for(key.public_key_bytes), production_trust_context=context)
    assert not any(call[0] == "sign" for call in dll.calls)


def test_closed_handle_rejects_public_access(dll: TestOnlyNCryptDLL, tmp_path: Path) -> None:
    key = adapter.WindowsCNGPreEnrollmentKey.open_or_create(tmp_path)
    key.close()
    key.close()
    with pytest.raises(adapter.WindowsCNGPreEnrollmentError, match="KEY_HANDLE_CLOSED"):
        _ = key.public_key_bytes


def test_test_only_backend_and_constructor_rejected(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with pytest.raises(TypeError, match="open_or_create"):
        adapter.WindowsCNGPreEnrollmentKey()
    monkeypatch.setattr(adapter, "_load_native", lambda: TestOnlyNCryptDLL())
    with pytest.raises(TypeError, match="exact native boundary"):
        adapter.WindowsCNGPreEnrollmentKey.open_or_create(tmp_path)


def test_non_windows_native_loader_fails_closed() -> None:
    if adapter.os.name == "nt":
        pytest.skip("non-Windows qualification boundary")
    with pytest.raises(adapter.WindowsCNGPreEnrollmentError, match="WINDOWS_REQUIRED"):
        adapter._NCryptAPI()


@pytest.mark.skipif(sys.platform != "win32", reason="Windows-native DLL loading only")
def test_windows_native_ncrypt_abi_loads_without_creating_a_key() -> None:
    """Hosted Windows smoke test: resolve native functions; never touch a TPM key."""
    native = adapter._NCryptAPI()
    assert ctypes.sizeof(adapter._HANDLE) == ctypes.sizeof(ctypes.c_void_p)
    assert native.dll.NCryptOpenStorageProvider.restype is adapter._STATUS
    assert native.dll.NCryptCreatePersistedKey.argtypes[-1] is adapter._DWORD
    assert native.dll.NCryptSignHash.argtypes[0] is adapter._HANDLE
    assert not hasattr(native, "import_key")
    assert not hasattr(native, "export_private_key")
    assert not hasattr(native, "delete_key")


def test_only_public_state_is_written(dll: TestOnlyNCryptDLL, tmp_path: Path) -> None:
    with adapter.WindowsCNGPreEnrollmentKey.open_or_create(tmp_path):
        pass
    state = (tmp_path / adapter._STATE_NAME).read_bytes()
    assert canonical_json_bytes(json.loads(state)) == state
    assert "private" not in state.decode().lower()
    assert sorted(path.name for path in tmp_path.iterdir()) == sorted(
        [
            adapter._LOCK_NAME,
            adapter._STATE_NAME,
        ]
    )
