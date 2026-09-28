"""Independent, fail-closed verifier for Stage-9 policy vectors."""

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

A_SHA256, A_NULL, A_ECDSA, A_ECC, CURVE_P256 = 0xB, 0x10, 0x18, 0x23, 3
C_INCREMENT, C_PNV, C_PAUTH, C_PCODE, C_PCPHASH, C_POR = 0x134, 0x149, 0x16A, 0x16C, 0x16E, 0x171
WRITTEN, NV_ATTRS, FORBIDDEN = 0x20000000, 0x02040018, 0x40000002
PSA_ATTRS, RECOVERY_ATTRS = 0x000400B2, 0x00040040


def _h(value: bytes) -> bytes:
    return hashlib.sha256(value).digest()


def _u16(value: int) -> bytes:
    return struct.pack(">H", value)


def _u32(value: int) -> bytes:
    return struct.pack(">I", value)


def _hex(value: str, field: str, size: int | None = None) -> bytes:
    try:
        result = bytes.fromhex(value)
    except ValueError as exc:
        raise PolicyVectorError(f"invalid {field}") from exc
    if result.hex() != value or (size is not None and len(result) != size):
        raise PolicyVectorError(f"non-canonical {field}")
    return result


def _word(data: bytes, pos: int, width: int) -> tuple[int, int]:
    if pos + width > len(data):
        raise PolicyVectorError("truncated TPMT_PUBLIC")
    return int.from_bytes(data[pos : pos + width], "big"), pos + width


def _blob(data: bytes, pos: int) -> tuple[bytes, int]:
    size, pos = _word(data, pos, 2)
    if pos + size > len(data):
        raise PolicyVectorError("malformed TPMT_PUBLIC coordinate or authPolicy size")
    return data[pos : pos + size], pos + size


def _public(key: dict[str, Any], psa: bool) -> bytes:
    raw, pos = _hex(key["public_hex"], "TPMT_PUBLIC"), 0
    key_type, pos = _word(raw, pos, 2)
    name_alg, pos = _word(raw, pos, 2)
    attrs, pos = _word(raw, pos, 4)
    auth, pos = _blob(raw, pos)
    symmetric, pos = _word(raw, pos, 2)
    scheme, pos = _word(raw, pos, 2)
    scheme_hash, pos = _word(raw, pos, 2)
    curve, pos = _word(raw, pos, 2)
    kdf, pos = _word(raw, pos, 2)
    x, pos = _blob(raw, pos)
    y, pos = _blob(raw, pos)
    wanted = (
        A_ECC,
        A_SHA256,
        PSA_ATTRS if psa else RECOVERY_ATTRS,
        32 if psa else 0,
        A_NULL,
        A_ECDSA,
        A_SHA256,
        CURVE_P256,
        A_NULL,
        32,
        32,
    )
    found = (
        key_type,
        name_alg,
        attrs,
        len(auth),
        symmetric,
        scheme,
        scheme_hash,
        curve,
        kdf,
        len(x),
        len(y),
    )
    if found != wanted:
        raise PolicyVectorError("TPMT_PUBLIC profile mismatch")
    if pos != len(raw):
        raise PolicyVectorError("TPMT_PUBLIC trailing garbage")
    return raw


def _eq(vector: dict[str, Any], field: str, expected: bytes | str) -> None:
    actual = vector.get(field)
    expected_value = expected.hex() if isinstance(expected, bytes) else expected
    if actual != expected_value:
        raise PolicyVectorError(f"{field} mismatch")


def verify_policy_vector(
    release: dict[str, Any], enrollment: dict[str, Any], vector: dict[str, Any]
) -> None:
    """Verify source schemas, authority boundary and every byte-derived field."""
    validate_schema(release, RELEASE_SCHEMA)
    validate_schema(enrollment, ENROLLMENT_SCHEMA)
    if vector.get("schema") != "CryptoHunter.Stage9PolicyVectorV1":
        raise PolicyVectorError("wrong vector schema")
    if release["purpose"] != enrollment["purpose"]:
        raise PolicyVectorError("purpose mismatch")
    provenance = "PRODUCTION_PROVISIONED" if release["purpose"] == "PRODUCTION" else "TEST_FIXTURE"
    if (
        release["k_recovery"]["provenance"] != provenance
        or enrollment["k_psa"]["provenance"] != provenance
    ):
        raise PolicyVectorError("purpose/provenance mismatch")
    release_digest = canonical_digest(release)
    if _hex(enrollment["release_policy_digest"], "bound release digest", 32) != release_digest:
        raise PolicyVectorError("release binding mismatch")
    _eq(vector, "source_release_policy_digest", release_digest)
    _eq(vector, "source_enrollment_material_digest", canonical_digest(enrollment))
    if (
        release["branch_order"] != ["NORMAL", "BOOTSTRAP_RECOVERY"]
        or vector.get("branch_order") != release["branch_order"]
    ):
        raise PolicyVectorError("branch order mismatch")
    refs = release["policy_refs"]
    normal_ref, recovery_ref = (
        _hex(refs["normal_hex"], "normal ref", 32),
        _hex(refs["recovery_hex"], "recovery ref", 32),
    )
    if normal_ref != _h(refs["normal_label"].encode("ascii")) or recovery_ref != _h(
        refs["recovery_label"].encode("ascii")
    ):
        raise PolicyVectorError("policyRef derivation mismatch")
    psa_raw, recovery_raw = (
        _public(enrollment["k_psa"], True),
        _public(release["k_recovery"], False),
    )
    psa_name, recovery_name = _u16(A_SHA256) + _h(psa_raw), _u16(A_SHA256) + _h(recovery_raw)
    _eq(vector, "k_psa_name", psa_name)
    _eq(vector, "k_recovery_name", recovery_name)
    _eq(vector, "normal_policy_ref", normal_ref)
    _eq(vector, "recovery_policy_ref", recovery_ref)
    zero = bytes(32)
    normal_branch = _h(_h(zero + _u32(C_PAUTH) + psa_name) + normal_ref)
    recovery_branch = _h(_h(zero + _u32(C_PAUTH) + recovery_name) + recovery_ref)
    if normal_branch == recovery_branch:
        raise PolicyVectorError("duplicate branches")
    _eq(vector, "normal_branch_digest", normal_branch)
    _eq(vector, "recovery_branch_digest", recovery_branch)
    root = _h(zero + _u32(C_POR) + normal_branch + recovery_branch)
    _eq(vector, "production_root_policy_digest", root)
    template, attrs = release["nv_template"], int(release["nv_template"]["attributes"], 16)
    if (
        attrs != NV_ATTRS
        or attrs & FORBIDDEN
        or template["auth_value"] != "EMPTY"
        or template["data_size"] != 8
    ):
        raise PolicyVectorError("NV template mismatch")
    index = int(template["index"], 16)

    def area(written: bool) -> bytes:
        return (
            _u32(index)
            + _u16(A_SHA256)
            + _u32(attrs | (WRITTEN if written else 0))
            + _u16(32)
            + root
            + _u16(8)
        )

    pre_area, post_area = area(False), area(True)
    pre_name, post_name = _u16(A_SHA256) + _h(pre_area), _u16(A_SHA256) + _h(post_area)
    _eq(vector, "nv_public_pre_write", pre_area)
    _eq(vector, "nv_public_post_write", post_area)
    _eq(vector, "nv_name_pre_write", pre_name)
    _eq(vector, "nv_name_post_write", post_name)
    pre_cp, post_cp = (
        _h(_u32(C_INCREMENT) + pre_name + pre_name),
        _h(_u32(C_INCREMENT) + post_name + post_name),
    )
    _eq(vector, "bootstrap_increment_cp_hash", pre_cp)
    _eq(vector, "post_write_recovery_increment_cp_hash", post_cp)
    expected_generation = enrollment["vector_generation"]
    if vector.get("normal_generation") != expected_generation:
        raise PolicyVectorError("normal_generation source binding mismatch")
    generation = int(expected_generation)
    if generation > (1 << 64) - 1:
        raise PolicyVectorError("generation exceeds UINT64")
    args = _h(struct.pack(">Q", generation) + _u16(0) + _u16(0))
    normal_approved = _h(
        _h(zero + _u32(C_PNV) + args + post_name) + _u32(C_PCODE) + _u32(C_INCREMENT)
    )
    _eq(vector, "normal_approved_policy", normal_approved)

    def recovery(cp: bytes) -> bytes:
        return _h(_h(zero + _u32(C_PCODE) + _u32(C_INCREMENT)) + _u32(C_PCPHASH) + cp)

    _eq(vector, "bootstrap_approved_policy", recovery(pre_cp))
    _eq(vector, "post_write_recovery_approved_policy", recovery(post_cp))
