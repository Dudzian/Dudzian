"""Shared schema loading and canonical JSON for Stage-9 public policy material."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any

import jsonschema

ROOT = Path(__file__).resolve().parent
RELEASE_SCHEMA = ROOT / "stage9_release_policy_v1.schema.json"
ENROLLMENT_SCHEMA = ROOT / "stage9_enrollment_policy_material_v1.schema.json"
SIGNED_RELEASE_SCHEMA = ROOT / "stage9_signed_release_policy_v1.schema.json"
PDSA_PACKAGE_SCHEMA = ROOT / "stage9_pdsa_enrollment_package_v1.schema.json"
FREEZE_MANIFEST_SCHEMA = ROOT / "stage9_root_of_trust_freeze_manifest_v1.schema.json"
REVOCATION_SCHEMA = ROOT / "stage9_revocation_state_v1.schema.json"


class PolicyVectorError(ValueError):
    """Canonical policy source or derived vector is invalid."""


def validate_schema(document: dict[str, Any], schema_path: Path) -> None:
    try:
        schema = json.loads(schema_path.read_text(encoding="utf-8"))
        jsonschema.Draft202012Validator(schema).validate(document)
    except (OSError, json.JSONDecodeError, jsonschema.ValidationError) as exc:
        raise PolicyVectorError(f"source material schema validation failed: {exc}") from exc


def canonical_json_bytes(value: Any) -> bytes:
    """Encode the repository's strict canonical JSON subset.

    Objects use lexicographically sorted Unicode keys, strings use JSON escaping
    with UTF-8 output, and only safe integral JSON numbers are allowed. Security
    uint64 values are decimal strings in the schemas, avoiding IEEE-754 loss.
    """

    def encode(item: Any) -> str:
        if item is None:
            return "null"
        if item is True:
            return "true"
        if item is False:
            return "false"
        if isinstance(item, str):
            return json.dumps(item, ensure_ascii=False, separators=(",", ":"))
        if isinstance(item, int) and not isinstance(item, bool):
            if abs(item) > 9_007_199_254_740_991:
                raise PolicyVectorError("JSON integer exceeds the interoperable exact range")
            return str(item)
        if isinstance(item, float):
            if not math.isfinite(item):
                raise PolicyVectorError("non-finite JSON numbers are forbidden")
            raise PolicyVectorError("floating-point JSON numbers are forbidden")
        if isinstance(item, list):
            return "[" + ",".join(encode(element) for element in item) + "]"
        if isinstance(item, dict):
            if not all(isinstance(key, str) for key in item):
                raise PolicyVectorError("JSON object keys must be strings")
            return (
                "{" + ",".join(encode(key) + ":" + encode(item[key]) for key in sorted(item)) + "}"
            )
        raise PolicyVectorError(f"unsupported canonical JSON type: {type(item).__name__}")

    return encode(value).encode("utf-8")


def canonical_digest(document: dict[str, Any]) -> bytes:
    return hashlib.sha256(canonical_json_bytes(document)).digest()
