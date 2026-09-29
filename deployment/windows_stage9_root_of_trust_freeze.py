"""Fail-closed, public-only Stage-9 production freeze primitives."""

from __future__ import annotations

import hashlib
import json
import struct
import subprocess
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey

from deployment.windows_stage9_policy_material import (
    FREEZE_MANIFEST_SCHEMA,
    PDSA_PACKAGE_SCHEMA,
    RELEASE_SCHEMA,
    REVOCATION_SCHEMA,
    SIGNED_RELEASE_SCHEMA,
    PolicyVectorError,
    canonical_digest,
    canonical_json_bytes,
    validate_schema,
)

RELEASE_DOMAIN = b"CryptoHunter.Stage9.ReleasePolicyV1\x00"
ENROLLMENT_DOMAIN = b"CryptoHunter.Stage9.PDSAEnrollmentPackageV1\x00"
REVOCATION_DOMAIN = b"CryptoHunter.Stage9.RevocationStateV1\x00"
PENDING = "PENDING_CEREMONY"
ROOT_DERIVATION = "SHA256(zero32||TPM_CC_PolicyOR||normal_branch||recovery_branch); K_PSA.Name is per-device"


class _OpaqueVerified:
    __slots__ = ()

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        raise TypeError("verified authority contexts can only be created by their verifier")

    def __setattr__(self, name: str, value: Any) -> None:
        raise TypeError("verified authority contexts are immutable")


class TrustedRetainedRevocationHead:
    """Immutable head read from the custody-owned monotonic retained store."""

    __slots__ = (
        "sequence",
        "state_digest",
        "revoked_root_signer_ids",
        "revoked_pdsa_signer_ids",
        "serialized_record",
        "trust_domain",
        "custody_reader",
    )

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        raise TypeError("trusted retained heads come only from genesis or custody-store loading")

    def __setattr__(self, name: str, value: Any) -> None:
        raise TypeError("retained revocation head is immutable")


class ProductionRevocationHeadCustodyReader(ABC):
    """Trusted composition boundary backed by the custody-owned protected current-head store."""

    @abstractmethod
    def read_current_authenticated_record(self) -> bytes:
        """Return the current record after storage authentication and monotonic-head resolution."""


class VerifiedRevocationStateV1(_OpaqueVerified):
    __slots__ = (
        "serialized_envelope",
        "state_digest",
        "sequence",
        "previous_state_digest",
        "revoked_root_signer_ids",
        "revoked_pdsa_signer_ids",
        "effective_at",
        "root_key_set_digest",
        "root_key_ids",
        "root_threshold",
        "pinned_root",
        "retained_head",
    )


class VerifiedReleasePolicyV1(_OpaqueVerified):
    """Opaque, deeply immutable authority projection returned only by verification."""

    __slots__ = (
        "serialized_envelope",
        "purpose",
        "payload_digest",
        "release_policy_id",
        "release_version",
        "root_keys",
        "root_threshold",
        "pdsa_keys",
        "pdsa_key_set_digest",
        "pdsa_threshold",
        "valid_from",
        "valid_until",
        "verification_time",
        "recovery_public_digest",
        "recovery_name",
        "k_psa_profile",
        "policy_refs",
        "nv_template",
        "branch_order",
        "serialization_profile",
        "revocations",
        "pinned_root",
    )


class PinnedProductReleaseRootV1(_OpaqueVerified):
    """Externally pinned immutable Product Release Root authority context."""

    __slots__ = (
        "purpose",
        "environment",
        "algorithm",
        "encoding",
        "threshold",
        "keys",
        "key_set_digest",
        "canonical_key_set_bytes",
    )


class ParsedTpmtPublic:
    __slots__ = ("raw", "name_alg", "name", "digest")

    def __init__(self, raw: bytes, name_alg: int, name: str, digest: str) -> None:
        self.raw, self.name_alg, self.name, self.digest = raw, name_alg, name, digest


def source_revision(repo: Path | None = None) -> str:
    try:
        value = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=repo, check=True, capture_output=True, text=True
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "UNKNOWN"
    return value if len(value) in (40, 64) and all(c in "0123456789abcdef" for c in value) else "UNKNOWN"


def _deep_freeze(value: Any) -> Any:
    if isinstance(value, dict):
        return MappingProxyType({key: _deep_freeze(item) for key, item in value.items()})
    if isinstance(value, list):
        return tuple(_deep_freeze(item) for item in value)
    return value


def _deep_thaw(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: _deep_thaw(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_deep_thaw(item) for item in value]
    return value


def _verified_projection(kind: type[_OpaqueVerified], **values: Any) -> Any:
    instance = object.__new__(kind)
    for name, value in values.items():
        object.__setattr__(instance, name, value)
    return instance


def _require_context_type(value: Any, kind: type[_OpaqueVerified]) -> None:
    if not isinstance(value, kind):
        raise PolicyVectorError("wrong verified authority context type")


def load_pinned_product_release_root_v1(
    key_records: list[dict[str, str]], *, purpose: str, environment: str
) -> PinnedProductReleaseRootV1:
    """Load anchors from the external trusted configuration boundary."""
    if purpose not in ("PRODUCTION", "TEST_ONLY") or not environment:
        raise PolicyVectorError("invalid pinned-root purpose/environment")
    ids = [record.get("key_id") for record in key_records]
    public_keys = [record.get("public_key_hex") for record in key_records]
    if len(key_records) != 3 or ids != sorted(ids) or len(ids) != len(set(ids)) or len(public_keys) != len(set(public_keys)):
        raise PolicyVectorError("pinned Product Release Root must be an ordered unique 3-key set")
    for record in key_records:
        if record.get("algorithm") != "Ed25519" or record.get("encoding") != "RFC8032_RAW_32_BYTES_LOWER_HEX":
            raise PolicyVectorError("invalid pinned Product Release Root profile")
        _decode_public(record["public_key_hex"])
    canonical = {
        "schema": "CryptoHunter.ProductReleaseRootKeySetV1",
        "version": 1,
        "purpose": purpose,
        "environment": environment,
        "algorithm": "Ed25519",
        "encoding": "RFC8032_RAW_32_BYTES_LOWER_HEX",
        "threshold": 2,
        "keys": key_records,
    }
    canonical_bytes = canonical_json_bytes(canonical)
    return _verified_projection(
        PinnedProductReleaseRootV1,
        purpose=purpose,
        environment=environment,
        algorithm="Ed25519",
        encoding="RFC8032_RAW_32_BYTES_LOWER_HEX",
        threshold=2,
        keys=MappingProxyType(dict(zip(ids, public_keys, strict=True))),
        key_set_digest=hashlib.sha256(canonical_bytes).hexdigest(),
        canonical_key_set_bytes=canonical_bytes,
    )


def stage9_revocation_genesis_v1() -> TrustedRetainedRevocationHead:
    """Return the single pinned Stage-9 revocation genesis head."""
    record = {
        "schema": "CryptoHunter.Stage9RevocationRetainedHeadV1",
        "version": 1,
        "sequence": 0,
        "state_digest": "00" * 32,
        "revoked_root_signer_ids": [],
        "revoked_pdsa_signer_ids": [],
        "source": "PINNED_GENESIS",
    }
    serialized = canonical_json_bytes(record)
    return _retained_projection(record, serialized, trust_domain="PINNED_GENESIS", custody_reader=None)


def load_current_trusted_retained_revocation_head(
    custody_reader: ProductionRevocationHeadCustodyReader,
) -> TrustedRetainedRevocationHead:
    """Load only the current head selected and authenticated by the configured custody reader."""
    if not isinstance(custody_reader, ProductionRevocationHeadCustodyReader):
        raise PolicyVectorError("production retained head requires a configured custody reader")
    serialized_record = custody_reader.read_current_authenticated_record()
    record = _parse_retained_record(serialized_record, source="CUSTODY_AUTHENTICATED_STORE")
    return _retained_projection(
        record,
        serialized_record,
        trust_domain="PRODUCTION_CUSTODY_STORE",
        custody_reader=custody_reader,
    )


def make_test_only_retained_revocation_head(serialized_record: bytes) -> TrustedRetainedRevocationHead:
    """Create an isolated TEST_ONLY retained-head simulation; never valid for PRODUCTION roots."""
    record = _parse_retained_record(serialized_record, source="TEST_ONLY_CUSTODY_SIMULATION")
    return _retained_projection(
        record,
        serialized_record,
        trust_domain="TEST_ONLY_CUSTODY_SIMULATION",
        custody_reader=None,
    )


def _parse_retained_record(serialized_record: bytes, *, source: str) -> dict[str, Any]:
    try:
        record = json.loads(serialized_record)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise PolicyVectorError("invalid retained-head serialization") from exc
    if canonical_json_bytes(record) != serialized_record:
        raise PolicyVectorError("noncanonical retained-head serialization")
    if set(record) != {"schema", "version", "sequence", "state_digest", "revoked_root_signer_ids", "revoked_pdsa_signer_ids", "source"}:
        raise PolicyVectorError("wrong retained-head record shape")
    if record["schema"] != "CryptoHunter.Stage9RevocationRetainedHeadV1" or record["version"] != 1 or record["source"] != source:
        raise PolicyVectorError("wrong retained-head authority metadata")
    if not isinstance(record["sequence"], int) or record["sequence"] < 1:
        raise PolicyVectorError("invalid retained-head sequence")
    _hex(record["state_digest"], "retained state digest", 32)
    for field in ("revoked_root_signer_ids", "revoked_pdsa_signer_ids"):
        if record[field] != sorted(record[field]) or len(record[field]) != len(set(record[field])):
            raise PolicyVectorError("retained deny lists must be ordered and unique")
    return record


def _retained_projection(
    record: Mapping[str, Any],
    serialized: bytes,
    *,
    trust_domain: str,
    custody_reader: ProductionRevocationHeadCustodyReader | None,
) -> TrustedRetainedRevocationHead:
    instance = object.__new__(TrustedRetainedRevocationHead)
    object.__setattr__(instance, "sequence", record["sequence"])
    object.__setattr__(instance, "state_digest", record["state_digest"])
    object.__setattr__(instance, "revoked_root_signer_ids", frozenset(record["revoked_root_signer_ids"]))
    object.__setattr__(instance, "revoked_pdsa_signer_ids", frozenset(record["revoked_pdsa_signer_ids"]))
    object.__setattr__(instance, "serialized_record", bytes(serialized))
    object.__setattr__(instance, "trust_domain", trust_domain)
    object.__setattr__(instance, "custody_reader", custody_reader)
    return instance


def _hex(value: str, field: str, size: int | None = None) -> bytes:
    try:
        raw = bytes.fromhex(value)
    except (TypeError, ValueError) as exc:
        raise PolicyVectorError(f"invalid {field}") from exc
    if raw.hex() != value or (size is not None and len(raw) != size):
        raise PolicyVectorError(f"noncanonical {field}")
    return raw


def _word(data: bytes, pos: int, width: int) -> tuple[int, int]:
    if pos + width > len(data):
        raise PolicyVectorError("truncated TPMT_PUBLIC")
    return int.from_bytes(data[pos : pos + width], "big"), pos + width


def _blob(data: bytes, pos: int) -> tuple[bytes, int]:
    size, pos = _word(data, pos, 2)
    if pos + size > len(data):
        raise PolicyVectorError("malformed TPM2B in TPMT_PUBLIC")
    return data[pos : pos + size], pos + size


def validate_tpmt_public(public_hex: str, profile: Mapping[str, Any]) -> ParsedTpmtPublic:
    """Parse an exact ECC TPMT_PUBLIC and enforce the signed metadata profile."""
    raw = _hex(public_hex, "TPMT_PUBLIC")
    pos = 0
    key_type, pos = _word(raw, pos, 2)
    name_alg, pos = _word(raw, pos, 2)
    attributes, pos = _word(raw, pos, 4)
    auth_policy, pos = _blob(raw, pos)
    symmetric, pos = _word(raw, pos, 2)
    scheme, pos = _word(raw, pos, 2)
    scheme_hash, pos = _word(raw, pos, 2)
    curve, pos = _word(raw, pos, 2)
    kdf, pos = _word(raw, pos, 2)
    x, pos = _blob(raw, pos)
    y, pos = _blob(raw, pos)
    if pos != len(raw):
        raise PolicyVectorError("TPMT_PUBLIC trailing bytes")
    expected = (
        0x23,
        0x0B,
        int(profile["object_attributes"], 16),
        profile["auth_policy_size"],
        0x10,
        0x18,
        0x0B,
        0x03,
        0x10,
        32,
        32,
    )
    actual = (
        key_type,
        name_alg,
        attributes,
        len(auth_policy),
        symmetric,
        scheme,
        scheme_hash,
        curve,
        kdf,
        len(x),
        len(y),
    )
    metadata = (
        profile["type"],
        profile["name_algorithm"],
        profile["scheme"],
        profile["scheme_hash"],
        profile["curve"],
        profile["kdf"],
    )
    if metadata != (
        "TPM_ALG_ECC",
        "TPM_ALG_SHA256",
        "TPM_ALG_ECDSA",
        "TPM_ALG_SHA256",
        "TPM_ECC_NIST_P256",
        "TPM_ALG_NULL",
    ) or actual != expected:
        raise PolicyVectorError("TPMT_PUBLIC does not match signed key profile")
    if len(auth_policy) == 32 and auth_policy == bytes(32):
        raise PolicyVectorError("TPMT_PUBLIC policy authority authPolicy cannot be zero")
    digest = hashlib.sha256(raw).digest()
    return ParsedTpmtPublic(raw, name_alg, (struct.pack(">H", name_alg) + digest).hex(), digest.hex())


def _decode_public(value: str) -> Ed25519PublicKey:
    return Ed25519PublicKey.from_public_bytes(_hex(value, "Ed25519 public key", 32))


def _parse_time(value: str) -> datetime:
    try:
        if len(value) != 20 or not value.endswith("Z"):
            raise ValueError
        return datetime.strptime(value, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
    except (TypeError, ValueError) as exc:
        raise PolicyVectorError("timestamp must be canonical RFC3339 UTC seconds") from exc


def _verify_envelope(
    envelope: dict[str, Any],
    keys: Mapping[str, str],
    *,
    domain: bytes,
    required_threshold: int,
    revoked_signer_ids: frozenset[str],
) -> None:
    digest = canonical_digest(envelope["payload"])
    if envelope["payload_digest"] != digest.hex():
        raise PolicyVectorError("signed payload digest mismatch")
    if envelope["threshold"] != required_threshold or not 1 <= required_threshold <= len(keys):
        raise PolicyVectorError("signature threshold mismatch")
    signer_ids = [entry["signer_id"] for entry in envelope["signatures"]]
    if signer_ids != sorted(signer_ids) or len(signer_ids) != len(set(signer_ids)):
        raise PolicyVectorError("signatures must have unique signer IDs in lexical order")
    valid = 0
    for entry in envelope["signatures"]:
        signer_id = entry["signer_id"]
        if signer_id in revoked_signer_ids:
            raise PolicyVectorError("revoked signer")
        public_hex = keys.get(signer_id)
        if public_hex is None:
            raise PolicyVectorError("unknown signer")
        try:
            signature = _hex(entry["signature_hex"], "signature", 64)
            _decode_public(public_hex).verify(signature, domain + digest)
        except InvalidSignature as exc:
            raise PolicyVectorError("invalid signature") from exc
        valid += 1
    if valid < required_threshold:
        raise PolicyVectorError("insufficient signature threshold")


def _validate_pinned_root(root: PinnedProductReleaseRootV1) -> None:
    _require_context_type(root, PinnedProductReleaseRootV1)
    records = [
        {"key_id": key_id, "algorithm": root.algorithm, "encoding": root.encoding, "public_key_hex": public_key}
        for key_id, public_key in root.keys.items()
    ]
    rebuilt = load_pinned_product_release_root_v1(
        records, purpose=root.purpose, environment=root.environment
    )
    if rebuilt.key_set_digest != root.key_set_digest or rebuilt.canonical_key_set_bytes != root.canonical_key_set_bytes:
        raise PolicyVectorError("pinned Product Release Root projection mismatch")


def _validate_retained_head(
    head: TrustedRetainedRevocationHead, pinned_root: PinnedProductReleaseRootV1
) -> None:
    if not isinstance(head, TrustedRetainedRevocationHead):
        raise PolicyVectorError("untrusted retained revocation head")
    if head.sequence == 0:
        genesis = stage9_revocation_genesis_v1()
        fields = (
            "state_digest",
            "revoked_root_signer_ids",
            "revoked_pdsa_signer_ids",
            "serialized_record",
            "trust_domain",
        )
        if any(getattr(head, field, None) != getattr(genesis, field) for field in fields):
            raise PolicyVectorError("fake revocation genesis head")
        return
    if pinned_root.purpose == "PRODUCTION":
        if head.trust_domain != "PRODUCTION_CUSTODY_STORE" or not isinstance(
            head.custody_reader, ProductionRevocationHeadCustodyReader
        ):
            raise PolicyVectorError("production revocation requires custody-store retained head")
        loaded = load_current_trusted_retained_revocation_head(head.custody_reader)
    else:
        if head.trust_domain != "TEST_ONLY_CUSTODY_SIMULATION":
            raise PolicyVectorError("TEST_ONLY revocation requires isolated retained-head simulation")
        loaded = make_test_only_retained_revocation_head(head.serialized_record)
    fields = ("sequence", "state_digest", "revoked_root_signer_ids", "revoked_pdsa_signer_ids")
    if any(getattr(head, field, None) != getattr(loaded, field) for field in fields):
        raise PolicyVectorError("retained revocation projection mismatch")


def verify_revocation_state(
    serialized: bytes,
    pinned_root: PinnedProductReleaseRootV1,
    *,
    retained_head: TrustedRetainedRevocationHead,
    verification_time: datetime,
) -> VerifiedRevocationStateV1:
    """Verify the root-authorized append-only successor of a trusted retained head."""
    _validate_pinned_root(pinned_root)
    _validate_retained_head(retained_head, pinned_root)
    root_keys = pinned_root.keys
    try:
        envelope = json.loads(serialized)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise PolicyVectorError("invalid revocation serialization") from exc
    if canonical_json_bytes(envelope) != serialized:
        raise PolicyVectorError("noncanonical revocation serialization")
    validate_schema(envelope, REVOCATION_SCHEMA)
    payload = envelope["payload"]
    root_revoked = payload["revoked_root_signer_ids"]
    pdsa_revoked = payload["revoked_pdsa_signer_ids"]
    if root_revoked != sorted(root_revoked) or pdsa_revoked != sorted(pdsa_revoked):
        raise PolicyVectorError("revocation IDs must be lexically ordered")
    if not set(root_revoked).issubset(root_keys):
        raise PolicyVectorError("revocation state names unknown root signer")
    if payload["sequence"] != retained_head.sequence + 1:
        raise PolicyVectorError("revocation sequence is not the exact monotonic successor")
    if payload["previous_state_digest"] != retained_head.state_digest:
        raise PolicyVectorError("revocation previous-state digest mismatch")
    if not retained_head.revoked_root_signer_ids.issubset(root_revoked) or not retained_head.revoked_pdsa_signer_ids.issubset(pdsa_revoked):
        raise PolicyVectorError("revocation removal/unrevocation is forbidden")
    effective_at = _parse_time(payload["effective_at"])
    if verification_time.tzinfo is None or effective_at > verification_time.astimezone(timezone.utc):
        raise PolicyVectorError("revocation state is not yet effective")
    _verify_envelope(
        envelope,
        root_keys,
        domain=REVOCATION_DOMAIN,
        required_threshold=pinned_root.threshold,
        revoked_signer_ids=retained_head.revoked_root_signer_ids,
    )
    return _verified_projection(
        VerifiedRevocationStateV1,
        serialized_envelope=bytes(serialized),
        state_digest=hashlib.sha256(serialized).hexdigest(),
        sequence=payload["sequence"],
        previous_state_digest=payload["previous_state_digest"],
        revoked_root_signer_ids=frozenset(root_revoked),
        revoked_pdsa_signer_ids=frozenset(pdsa_revoked),
        effective_at=effective_at,
        root_key_set_digest=pinned_root.key_set_digest,
        root_key_ids=tuple(pinned_root.keys),
        root_threshold=pinned_root.threshold,
        pinned_root=pinned_root,
        retained_head=retained_head,
    )


def _pdsa_authority(payload: Mapping[str, Any]) -> tuple[dict[str, str], int]:
    entries = payload.get("pdsa_verification_key_set")
    contract = payload.get("production_contract")
    if not isinstance(entries, list) or not isinstance(contract, dict):
        raise PolicyVectorError("release lacks frozen PDSA authority")
    ids = [entry.get("key_id") for entry in entries]
    values = [entry.get("public_key_hex") for entry in entries]
    if ids != sorted(ids) or len(ids) != len(set(ids)) or len(values) != len(set(values)):
        raise PolicyVectorError("PDSA key set must be unique and lexically ordered")
    if ids != contract["pdsa_key_ids"]:
        raise PolicyVectorError("PDSA key IDs are not contract-bound")
    if payload["pdsa_verification_keys"] != values:
        raise PolicyVectorError("legacy PDSA public projection mismatch")
    threshold = contract["pdsa_threshold"]
    if not isinstance(threshold, int) or not 1 <= threshold <= len(entries):
        raise PolicyVectorError("invalid PDSA threshold")
    for entry in entries:
        if entry.get("algorithm") != "Ed25519" or entry.get("encoding") != "RFC8032_RAW_32_BYTES_LOWER_HEX":
            raise PolicyVectorError("invalid PDSA key profile")
        _decode_public(entry["public_key_hex"])
    return dict(zip(ids, values, strict=True)), threshold


def _release_root_key_set_digest(
    root: Mapping[str, Any], purpose: str, environment: str
) -> str:
    records = [
        {
            "key_id": key_id,
            "algorithm": root["algorithm"],
            "encoding": root["encoding"],
            "public_key_hex": public_key,
        }
        for key_id, public_key in zip(root["key_ids"], root["public_keys_hex"], strict=True)
    ]
    return load_pinned_product_release_root_v1(
        records, purpose=purpose, environment=environment
    ).key_set_digest


def _revalidate_revocation_context(value: VerifiedRevocationStateV1) -> VerifiedRevocationStateV1:
    _require_context_type(value, VerifiedRevocationStateV1)
    required = ("serialized_envelope", "pinned_root", "retained_head", "effective_at")
    if any(not hasattr(value, field) for field in required):
        raise PolicyVectorError("incomplete forged revocation context")
    rebuilt = verify_revocation_state(
        value.serialized_envelope,
        value.pinned_root,
        retained_head=value.retained_head,
        verification_time=value.effective_at,
    )
    fields = (
        "state_digest",
        "sequence",
        "previous_state_digest",
        "revoked_root_signer_ids",
        "revoked_pdsa_signer_ids",
        "root_key_set_digest",
        "root_key_ids",
        "root_threshold",
    )
    if any(getattr(value, field, None) != getattr(rebuilt, field) for field in fields):
        raise PolicyVectorError("revocation verified projection mismatch")
    return rebuilt


def verify_signed_release_policy(
    serialized: bytes,
    pinned_root: PinnedProductReleaseRootV1,
    *,
    verification_time: datetime,
    revocations: VerifiedRevocationStateV1,
) -> VerifiedReleasePolicyV1:
    """Build a PRODUCTION authority context; no caller controls purpose or quorum."""
    return _verify_signed_release_policy(
        serialized,
        pinned_root,
        verification_time=verification_time,
        revocations=revocations,
        expected_purpose="PRODUCTION",
    )


def verify_test_only_signed_release_policy(
    serialized: bytes,
    pinned_root: PinnedProductReleaseRootV1,
    *,
    verification_time: datetime,
    revocations: VerifiedRevocationStateV1,
) -> VerifiedReleasePolicyV1:
    """Exercise the identical ceremony path with explicitly TEST_ONLY authority."""
    return _verify_signed_release_policy(
        serialized,
        pinned_root,
        verification_time=verification_time,
        revocations=revocations,
        expected_purpose="TEST_ONLY",
    )


def _verify_signed_release_policy(
    serialized: bytes,
    pinned_root: PinnedProductReleaseRootV1,
    *,
    verification_time: datetime,
    revocations: VerifiedRevocationStateV1,
    expected_purpose: str,
) -> VerifiedReleasePolicyV1:
    _validate_pinned_root(pinned_root)
    _revalidate_revocation_context(revocations)
    root_keys = pinned_root.keys
    try:
        envelope = json.loads(serialized)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise PolicyVectorError("invalid signed release serialization") from exc
    if canonical_json_bytes(envelope) != serialized:
        raise PolicyVectorError("noncanonical signed release serialization")
    validate_schema(envelope, SIGNED_RELEASE_SCHEMA)
    payload = envelope["payload"]
    validate_schema(payload, RELEASE_SCHEMA)
    if payload["purpose"] != expected_purpose or pinned_root.purpose != expected_purpose:
        raise PolicyVectorError("release purpose does not match trust domain")
    root = payload["product_release_root"]
    root_ids = root.get("key_ids")
    if root_ids != sorted(root_keys) or [root_keys[key] for key in root_ids] != root["public_keys_hex"]:
        raise PolicyVectorError("Product Release Root key set is not exactly payload-bound")
    threshold = root["threshold"]
    release_root_digest = _release_root_key_set_digest(root, expected_purpose, pinned_root.environment)
    if release_root_digest != pinned_root.key_set_digest or revocations.root_key_set_digest != release_root_digest:
        raise PolicyVectorError("Product Release Root identity mismatch across release/revocation/pins")
    _verify_envelope(
        envelope,
        root_keys,
        domain=RELEASE_DOMAIN,
        required_threshold=threshold,
        revoked_signer_ids=revocations.revoked_root_signer_ids,
    )
    contract = payload.get("production_contract")
    if not isinstance(contract, dict):
        raise PolicyVectorError("signed release lacks production contract")
    valid_from, valid_until = _parse_time(contract["valid_from"]), _parse_time(contract["valid_until"])
    if verification_time.tzinfo is None:
        raise PolicyVectorError("verification_time must be timezone-aware")
    verification_time = verification_time.astimezone(timezone.utc)
    if not valid_from < valid_until or not valid_from <= verification_time < valid_until:
        raise PolicyVectorError("release validity window rejected")
    pdsa_keys, pdsa_threshold = _pdsa_authority(payload)
    if not revocations.revoked_root_signer_ids.issubset(root_keys) or not revocations.revoked_pdsa_signer_ids.issubset(pdsa_keys):
        raise PolicyVectorError("revocation state contains signer outside the signed release authority")
    recovery = validate_tpmt_public(payload["k_recovery"]["public_hex"], payload["k_recovery"])
    return _verified_projection(
        VerifiedReleasePolicyV1,
        serialized_envelope=bytes(serialized),
        purpose=payload["purpose"],
        payload_digest=envelope["payload_digest"],
        release_policy_id=payload["release_policy_id"],
        release_version=contract["release_version"],
        root_keys=MappingProxyType(dict(zip(root["key_ids"], root["public_keys_hex"], strict=True))),
        root_threshold=threshold,
        pdsa_keys=MappingProxyType(dict(pdsa_keys)),
        pdsa_key_set_digest=canonical_digest(payload["pdsa_verification_key_set"]).hex(),
        pdsa_threshold=pdsa_threshold,
        valid_from=valid_from,
        valid_until=valid_until,
        verification_time=verification_time,
        recovery_public_digest=recovery.digest,
        recovery_name=recovery.name,
        k_psa_profile=_deep_freeze(payload["k_psa_profile"]),
        policy_refs=_deep_freeze(payload["policy_refs"]),
        nv_template=_deep_freeze(payload["nv_template"]),
        branch_order=tuple(payload["branch_order"]),
        serialization_profile=payload["serialization_profile"],
        revocations=revocations,
        pinned_root=pinned_root,
    )


def _revalidate_release_context(value: VerifiedReleasePolicyV1) -> VerifiedReleasePolicyV1:
    _require_context_type(value, VerifiedReleasePolicyV1)
    required = (
        "serialized_envelope",
        "pinned_root",
        "verification_time",
        "revocations",
        "purpose",
    )
    if any(not hasattr(value, field) for field in required):
        raise PolicyVectorError("incomplete forged release context")
    rebuilt = _verify_signed_release_policy(
        value.serialized_envelope,
        value.pinned_root,
        verification_time=value.verification_time,
        revocations=value.revocations,
        expected_purpose=value.purpose,
    )
    fields = (
        "purpose",
        "payload_digest",
        "release_policy_id",
        "release_version",
        "root_threshold",
        "pdsa_key_set_digest",
        "pdsa_threshold",
        "valid_from",
        "valid_until",
        "verification_time",
        "recovery_public_digest",
        "recovery_name",
        "branch_order",
        "serialization_profile",
    )
    if any(getattr(value, field, None) != getattr(rebuilt, field) for field in fields):
        raise PolicyVectorError("release verified projection mismatch")
    if dict(value.root_keys) != dict(rebuilt.root_keys) or dict(value.pdsa_keys) != dict(rebuilt.pdsa_keys):
        raise PolicyVectorError("release verified key projection mismatch")
    for field in ("k_psa_profile", "policy_refs", "nv_template"):
        if _deep_thaw(getattr(value, field)) != _deep_thaw(getattr(rebuilt, field)):
            raise PolicyVectorError("release verified policy projection mismatch")
    return rebuilt


def verify_pdsa_enrollment_package(
    serialized: bytes,
    release: VerifiedReleasePolicyV1,
    *,
    device_identity_binding: str,
) -> dict[str, Any]:
    """Verify enrollment using only authority derived from a verified release."""
    release = _revalidate_release_context(release)
    try:
        package = json.loads(serialized)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise PolicyVectorError("invalid enrollment package serialization") from exc
    if canonical_json_bytes(package) != serialized:
        raise PolicyVectorError("noncanonical enrollment package serialization")
    validate_schema(package, PDSA_PACKAGE_SCHEMA)
    payload = package["payload"]
    if payload["purpose"] != release.purpose:
        raise PolicyVectorError("enrollment purpose mismatch")
    if payload["release_policy_digest"] != release.payload_digest:
        raise PolicyVectorError("wrong release digest")
    if payload["device_identity_binding"] != device_identity_binding:
        raise PolicyVectorError("wrong target/device binding")
    psa = validate_tpmt_public(payload["k_psa"]["tpmt_public_hex"], release.k_psa_profile)
    if payload["k_psa"]["name_hex"] != psa.name:
        raise PolicyVectorError("wrong K_PSA Name")
    expected = "TARGET_TPM_GENERATED" if payload["purpose"] == "PRODUCTION" else "TEST_FIXTURE"
    if payload["k_psa"]["provenance"] != expected:
        raise PolicyVectorError("K_PSA provenance does not match package purpose")
    _verify_envelope(
        package,
        release.pdsa_keys,
        domain=ENROLLMENT_DOMAIN,
        required_threshold=release.pdsa_threshold,
        revoked_signer_ids=release.revocations.revoked_pdsa_signer_ids,
    )
    return package


def _validated_source_revision(value: str) -> str:
    if value != "UNKNOWN" and (len(value) not in (40, 64) or any(c not in "0123456789abcdef" for c in value)):
        raise PolicyVectorError("invalid source_revision provenance")
    return value


def _manifest_base(
    *,
    policy_refs: Mapping[str, Any],
    nv_template: Mapping[str, Any],
    branch_order: tuple[str, ...] | list[str],
    serialization_profile: str,
    artifact_source_revision: str | None,
) -> dict[str, Any]:
    revision = source_revision(Path(__file__).resolve().parents[1]) if artifact_source_revision is None else artifact_source_revision
    return {
        "schema": "CryptoHunter.Stage9RootOfTrustFreezeManifestV1",
        "version": 1,
        "source_revision": _validated_source_revision(revision),
        "serialization_profile": serialization_profile,
        "policy_refs": _deep_thaw(policy_refs),
        "nv_template": _deep_thaw(nv_template),
        "branch_order": list(branch_order),
        "root_policy_derivation": ROOT_DERIVATION,
        "supported_migration_version": 1,
        "tool": {"name": "deployment.windows_stage9_root_of_trust_freeze", "version": 1},
    }


def build_unprovisioned_freeze_manifest(
    release_template: dict[str, Any], *, artifact_source_revision: str | None = None
) -> dict[str, Any]:
    manifest = _expected_unprovisioned_manifest(release_template, artifact_source_revision)
    verify_freeze_manifest(manifest, release_template=release_template)
    return manifest


def _expected_unprovisioned_manifest(
    release_template: Mapping[str, Any], artifact_source_revision: str | None
) -> dict[str, Any]:
    manifest = _manifest_base(
        policy_refs=release_template["policy_refs"],
        nv_template=release_template["nv_template"],
        branch_order=release_template["branch_order"],
        serialization_profile=release_template["serialization_profile"],
        artifact_source_revision=artifact_source_revision,
    )
    manifest.update(
        status="PRODUCTION_ROOT_MATERIAL_NOT_PROVISIONED",
        release_policy_payload_digest=PENDING,
        signed_release_policy_digest=PENDING,
        revocation_state_digest=PENDING,
        revocation_sequence=PENDING,
        product_root_key_set_digest=PENDING,
        product_release_root={"material_status": "NOT_PROVISIONED", "algorithm": "Ed25519", "encoding": "RFC8032_RAW_32_BYTES_LOWER_HEX", "threshold": 2, "key_ids": [], "public_keys_hex": []},
        pdsa_key_set_digest=PENDING,
        k_recovery={"material_status": "NOT_PROVISIONED", "tpmt_public_digest": PENDING, "name": PENDING},
    )
    return manifest


def build_frozen_manifest(
    release: VerifiedReleasePolicyV1, *, artifact_source_revision: str | None = None
) -> dict[str, Any]:
    """Build a complete manifest exclusively from sealed verified artifacts."""
    release = _revalidate_release_context(release)
    manifest = _expected_frozen_manifest(release, artifact_source_revision)
    verify_freeze_manifest(manifest, verified_release=release)
    return manifest


def verify_freeze_manifest(
    manifest: dict[str, Any],
    *,
    release_template: dict[str, Any] | None = None,
    verified_release: VerifiedReleasePolicyV1 | None = None,
) -> None:
    validate_schema(manifest, FREEZE_MANIFEST_SCHEMA)
    revision = _validated_source_revision(manifest["source_revision"])
    if manifest["status"] == "PRODUCTION_ROOT_MATERIAL_NOT_PROVISIONED":
        if release_template is None or verified_release is not None:
            raise PolicyVectorError("unprovisioned manifest requires only a release template")
        if manifest != _expected_unprovisioned_manifest(release_template, revision):
            raise PolicyVectorError("unprovisioned manifest is not the exact pending state")
        return
    if verified_release is None or release_template is not None:
        raise PolicyVectorError("frozen manifest requires a verified release")
    verified_release = _revalidate_release_context(verified_release)
    expected = _expected_frozen_manifest(verified_release, revision)
    if manifest != expected:
        raise PolicyVectorError("frozen manifest does not match verified artifacts")
    if _contains_pending(manifest):
        raise PolicyVectorError("frozen manifest contains pending value")


def _expected_frozen_manifest(
    release: VerifiedReleasePolicyV1, artifact_source_revision: str | None
) -> dict[str, Any]:
    result = _manifest_base(
        policy_refs=release.policy_refs,
        nv_template=release.nv_template,
        branch_order=release.branch_order,
        serialization_profile=release.serialization_profile,
        artifact_source_revision=artifact_source_revision,
    )
    result.update(
        status="PRODUCTION_ROOT_OF_TRUST_FROZEN" if release.purpose == "PRODUCTION" else "TEST_ONLY_ROOT_OF_TRUST_FROZEN",
        release_policy_payload_digest=release.payload_digest,
        signed_release_policy_digest=hashlib.sha256(release.serialized_envelope).hexdigest(),
        revocation_state_digest=release.revocations.state_digest,
        revocation_sequence=release.revocations.sequence,
        product_root_key_set_digest=release.pinned_root.key_set_digest,
        product_release_root={"material_status": "PROVISIONED", "algorithm": "Ed25519", "encoding": "RFC8032_RAW_32_BYTES_LOWER_HEX", "threshold": release.root_threshold, "key_ids": list(release.root_keys), "public_keys_hex": list(release.root_keys.values())},
        pdsa_key_set_digest=release.pdsa_key_set_digest,
        k_recovery={"material_status": "PROVISIONED", "tpmt_public_digest": release.recovery_public_digest, "name": release.recovery_name},
    )
    return result

def _contains_pending(value: Any) -> bool:
    if value == PENDING:
        return True
    if isinstance(value, dict):
        return any(_contains_pending(item) for item in value.values())
    if isinstance(value, list):
        return any(_contains_pending(item) for item in value)
    return False


_MIGRATIONS = {
    "RELEASE_POLICY_UPDATE": ("PRODUCT_RELEASE_ROOT_QUORUM", "RELEASE_AND_NV_HEAD_ADVANCE", "SIGNED_SUCCESSOR_RELEASE", "OFFLINE_RELEASE_CEREMONY", True),
    "PRODUCT_ROOT_ROTATION": ("CURRENT_PRODUCT_RELEASE_ROOT_QUORUM", "RELEASE_AND_REVOCATION_HEAD_ADVANCE", "DUAL_ROOT_CEREMONY_RECORD", "OFFLINE_BREAK_GLASS_CEREMONY", True),
    "PDSA_ROTATION": ("PRODUCT_RELEASE_ROOT_QUORUM", "RELEASE_VERSION_ADVANCE", "SIGNED_RELEASE_AND_REVOCATIONS", "OFFLINE_RELEASE_CEREMONY", True),
    "K_RECOVERY_ROTATION": ("PRODUCT_RELEASE_ROOT_QUORUM", "RELEASE_VERSION_ADVANCE", "TPMT_PUBLIC_NAMES_AND_CEREMONY", "OFFLINE_RECOVERY_KEY_CEREMONY", True),
    "K_PSA_REPLACEMENT": ("PDSA_THRESHOLD_RECOVERY", "DEVICE_GENERATION_ADVANCE", "EK_AK_CREATION_AND_PACKAGE", "PDSA_RECOVERY_ENROLLMENT", True),
    "TPM_REPLACEMENT": ("PDSA_THRESHOLD_AND_OPERATOR_RECOVERY", "DEVICE_LINEAGE_ADVANCE", "LOSS_REPLACEMENT_AND_PACKAGE", "DEVICE_REPROVISION", True),
    "NV_RECREATION": ("RECOVERY_AUTHORITY_AND_PDSA_POLICY", "RETAINED_HEAD_ADVANCE", "NV_NAMES_COUNTER_CPHASH_APPROVALS", "OFFLINE_NV_RECOVERY", True),
    "DEVICE_REPROVISION": ("PDSA_THRESHOLD", "ENROLLMENT_LINEAGE_ADVANCE", "SIGNED_RECOVERY_LINEAGE", "PDSA_RECOVERY_ENROLLMENT", True),
    "SERIALIZATION_PROFILE_CHANGE": ("PRODUCT_RELEASE_ROOT_QUORUM", "SUPPORTED_PROFILE_SUCCESSOR", "COMPATIBILITY_APPROVAL_AND_VECTORS", "UPGRADE_VERIFIER_OUT_OF_BAND", False),
    "ROLLBACK_ATTEMPT": ("NONE", "NEVER", "ROLLBACK_REJECTION_EVENT", "RESTORE_CURRENT_EVIDENCE", False),
    "UNKNOWN_FUTURE_VERSION": ("NONE", "NEVER", "UNKNOWN_VERSION_REJECTION_EVENT", "UPGRADE_VERIFIER_OUT_OF_BAND", False),
}


def migration_allowed(record: Mapping[str, Any]) -> bool:
    required = {"case", "authority", "old_version", "new_version", "monotonic_evidence", "audit_evidence", "recovery_path"}
    if set(record) != required or record.get("case") not in _MIGRATIONS:
        return False
    authority, monotonic, audit, recovery, allowed = _MIGRATIONS[record["case"]]
    if not allowed:
        return False
    old, new = record["old_version"], record["new_version"]
    return (
        isinstance(old, int)
        and not isinstance(old, bool)
        and isinstance(new, int)
        and not isinstance(new, bool)
        and new == old + 1
        and record["authority"] == authority
        and record["monotonic_evidence"] == monotonic
        and record["audit_evidence"] == audit
        and record["recovery_path"] == recovery
    )
