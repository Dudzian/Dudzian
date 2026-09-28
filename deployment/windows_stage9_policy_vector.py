"""Pure generator for the per-device Stage-9 TPM policy vector."""

from __future__ import annotations

import hashlib
import struct
from typing import Any

from deployment.windows_stage9_policy_material import (
    ENROLLMENT_SCHEMA,
    RELEASE_SCHEMA,
    PolicyVectorError,
    canonical_digest,
    validate_schema,
)

TPM_ALG_SHA256, TPM_ALG_NULL, TPM_ALG_ECDSA, TPM_ALG_ECC = 0x000B, 0x0010, 0x0018, 0x0023
TPM_ECC_NIST_P256 = 0x0003
TPM_CC_NV_INCREMENT, TPM_CC_POLICY_NV = 0x134, 0x149
TPM_CC_POLICY_AUTHORIZE, TPM_CC_POLICY_COMMAND_CODE = 0x16A, 0x16C
TPM_CC_POLICY_CP_HASH, TPM_CC_POLICY_OR = 0x16E, 0x171
TPM_EO_EQ = 0
TPMA_NV_WRITTEN = 0x20000000
REQUIRED_NV_ATTRIBUTES, FORBIDDEN_NV_ATTRIBUTES = 0x02040018, 0x40000002
PSA_ATTRIBUTES, RECOVERY_ATTRIBUTES = 0x000400B2, 0x00040040
MAX_UINT64 = (1 << 64) - 1


def _u16(value: int) -> bytes:
    return struct.pack(">H", value)


def _u32(value: int) -> bytes:
    return struct.pack(">I", value)


def _h(*parts: bytes) -> bytes:
    return hashlib.sha256(b"".join(parts)).digest()


def _take_u16(data: bytes, offset: int) -> tuple[int, int]:
    if offset + 2 > len(data):
        raise PolicyVectorError("truncated TPMT_PUBLIC")
    return struct.unpack_from(">H", data, offset)[0], offset + 2


def _take_u32(data: bytes, offset: int) -> tuple[int, int]:
    if offset + 4 > len(data):
        raise PolicyVectorError("truncated TPMT_PUBLIC")
    return struct.unpack_from(">I", data, offset)[0], offset + 4


def _take_2b(data: bytes, offset: int) -> tuple[bytes, int]:
    size, offset = _take_u16(data, offset)
    if offset + size > len(data):
        raise PolicyVectorError("malformed TPM2B size")
    return data[offset : offset + size], offset + size


def _decode_hex(value: str, field: str, size: int | None = None) -> bytes:
    try:
        result = bytes.fromhex(value)
    except ValueError as exc:
        raise PolicyVectorError(f"invalid {field}") from exc
    if result.hex() != value or (size is not None and len(result) != size):
        raise PolicyVectorError(f"non-canonical {field}")
    return result


def _validate_public(key: dict[str, Any], *, psa: bool) -> bytes:
    raw = _decode_hex(key["public_hex"], "TPMT_PUBLIC")
    offset = 0
    key_type, offset = _take_u16(raw, offset)
    name_alg, offset = _take_u16(raw, offset)
    attrs, offset = _take_u32(raw, offset)
    auth_policy, offset = _take_2b(raw, offset)
    symmetric, offset = _take_u16(raw, offset)
    scheme, offset = _take_u16(raw, offset)
    scheme_hash, offset = _take_u16(raw, offset)
    curve, offset = _take_u16(raw, offset)
    kdf, offset = _take_u16(raw, offset)
    x, offset = _take_2b(raw, offset)
    y, offset = _take_2b(raw, offset)
    expected_attrs, expected_auth = (PSA_ATTRIBUTES, 32) if psa else (RECOVERY_ATTRIBUTES, 0)
    expected = (
        TPM_ALG_ECC,
        TPM_ALG_SHA256,
        expected_attrs,
        expected_auth,
        TPM_ALG_NULL,
        TPM_ALG_ECDSA,
        TPM_ALG_SHA256,
        TPM_ECC_NIST_P256,
        TPM_ALG_NULL,
        32,
        32,
    )
    actual = (
        key_type,
        name_alg,
        attrs,
        len(auth_policy),
        symmetric,
        scheme,
        scheme_hash,
        curve,
        kdf,
        len(x),
        len(y),
    )
    if actual != expected:
        raise PolicyVectorError("TPMT_PUBLIC does not match its canonical key profile")
    if offset != len(raw):
        raise PolicyVectorError("TPMT_PUBLIC has trailing bytes")
    return raw


def _qualify_sources(
    release: dict[str, Any], enrollment: dict[str, Any]
) -> tuple[bytes, bytes, int]:
    validate_schema(release, RELEASE_SCHEMA)
    validate_schema(enrollment, ENROLLMENT_SCHEMA)
    if release["purpose"] != enrollment["purpose"]:
        raise PolicyVectorError("release/enrollment purpose mismatch")
    expected_provenance = (
        "PRODUCTION_PROVISIONED" if release["purpose"] == "PRODUCTION" else "TEST_FIXTURE"
    )
    if (
        release["k_recovery"]["provenance"] != expected_provenance
        or enrollment["k_psa"]["provenance"] != expected_provenance
    ):
        raise PolicyVectorError("purpose/provenance separation violated")
    release_digest = canonical_digest(release)
    if (
        _decode_hex(enrollment["release_policy_digest"], "release policy digest", 32)
        != release_digest
    ):
        raise PolicyVectorError("enrollment is not bound to ReleasePolicyV1")
    refs = release["policy_refs"]
    for domain in ("normal", "recovery"):
        if _decode_hex(refs[f"{domain}_hex"], f"{domain} policyRef", 32) != _h(
            refs[f"{domain}_label"].encode("ascii")
        ):
            raise PolicyVectorError("policyRef must equal SHA256(ASCII label)")
    if release["branch_order"] != ["NORMAL", "BOOTSTRAP_RECOVERY"]:
        raise PolicyVectorError("wrong branch order")
    generation = int(enrollment["vector_generation"])
    if generation > MAX_UINT64:
        raise PolicyVectorError("vector generation exceeds UINT64")
    return (
        _validate_public(enrollment["k_psa"], psa=True),
        _validate_public(release["k_recovery"], psa=False),
        generation,
    )


def _name(public: bytes) -> bytes:
    return _u16(TPM_ALG_SHA256) + _h(public)


def _branch(name: bytes, ref: bytes) -> bytes:
    return _h(_h(bytes(32), _u32(TPM_CC_POLICY_AUTHORIZE), name), ref)


def _nv_public(index: int, attrs: int, policy: bytes) -> bytes:
    return _u32(index) + _u16(TPM_ALG_SHA256) + _u32(attrs) + _u16(32) + policy + _u16(8)


def _cp_hash(name: bytes) -> bytes:
    return _h(_u32(TPM_CC_NV_INCREMENT), name, name)


def _recovery_approved(cp_hash: bytes) -> bytes:
    return _h(
        _h(bytes(32), _u32(TPM_CC_POLICY_COMMAND_CODE), _u32(TPM_CC_NV_INCREMENT)),
        _u32(TPM_CC_POLICY_CP_HASH),
        cp_hash,
    )


def generate_policy_vector(release: dict[str, Any], enrollment: dict[str, Any]) -> dict[str, Any]:
    """Generate a byte-complete vector from separate release and PDSA-bound inputs."""
    psa_public, recovery_public, generation = _qualify_sources(release, enrollment)
    psa_name, recovery_name = _name(psa_public), _name(recovery_public)
    normal_ref = bytes.fromhex(release["policy_refs"]["normal_hex"])
    recovery_ref = bytes.fromhex(release["policy_refs"]["recovery_hex"])
    normal_branch, recovery_branch = (
        _branch(psa_name, normal_ref),
        _branch(recovery_name, recovery_ref),
    )
    if normal_branch == recovery_branch:
        raise PolicyVectorError("duplicate PolicyOR branches")
    root = _h(bytes(32), _u32(TPM_CC_POLICY_OR), normal_branch, recovery_branch)
    template = release["nv_template"]
    attrs = int(template["attributes"], 16)
    if attrs != REQUIRED_NV_ATTRIBUTES or attrs & FORBIDDEN_NV_ATTRIBUTES:
        raise PolicyVectorError("wrong NV template")
    index = int(template["index"], 16)
    pre_public, post_public = (
        _nv_public(index, attrs, root),
        _nv_public(index, attrs | TPMA_NV_WRITTEN, root),
    )
    pre_name, post_name = _name(pre_public), _name(post_public)
    pre_cp, post_cp = _cp_hash(pre_name), _cp_hash(post_name)
    args = _h(struct.pack(">Q", generation), _u16(0), _u16(TPM_EO_EQ))
    normal_approved = _h(
        _h(bytes(32), _u32(TPM_CC_POLICY_NV), args, post_name),
        _u32(TPM_CC_POLICY_COMMAND_CODE),
        _u32(TPM_CC_NV_INCREMENT),
    )
    return {
        "schema": "CryptoHunter.Stage9PolicyVectorV1",
        "source_release_policy_digest": canonical_digest(release).hex(),
        "source_enrollment_material_digest": canonical_digest(enrollment).hex(),
        "branch_order": release["branch_order"],
        "k_psa_name": psa_name.hex(),
        "k_recovery_name": recovery_name.hex(),
        "normal_policy_ref": normal_ref.hex(),
        "recovery_policy_ref": recovery_ref.hex(),
        "normal_branch_digest": normal_branch.hex(),
        "recovery_branch_digest": recovery_branch.hex(),
        "production_root_policy_digest": root.hex(),
        "nv_public_pre_write": pre_public.hex(),
        "nv_name_pre_write": pre_name.hex(),
        "nv_public_post_write": post_public.hex(),
        "nv_name_post_write": post_name.hex(),
        "bootstrap_increment_cp_hash": pre_cp.hex(),
        "post_write_recovery_increment_cp_hash": post_cp.hex(),
        "normal_generation": enrollment["vector_generation"],
        "normal_approved_policy": normal_approved.hex(),
        "bootstrap_approved_policy": _recovery_approved(pre_cp).hex(),
        "post_write_recovery_approved_policy": _recovery_approved(post_cp).hex(),
    }
