"""Shared schema loading and canonical JSON for Stage-9 public policy material."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import jsonschema
from deployment.canonical_json import canonical_json_bytes as _canonical_json_bytes

ROOT = Path(__file__).resolve().parent
RELEASE_SCHEMA = ROOT / "stage9_release_policy_v1.schema.json"
ENROLLMENT_SCHEMA = ROOT / "stage9_enrollment_policy_material_v1.schema.json"
SIGNED_RELEASE_SCHEMA = ROOT / "stage9_signed_release_policy_v1.schema.json"
PDSA_PACKAGE_SCHEMA = ROOT / "stage9_pdsa_enrollment_package_v1.schema.json"
FREEZE_MANIFEST_SCHEMA = ROOT / "stage9_root_of_trust_freeze_manifest_v1.schema.json"
REVOCATION_SCHEMA = ROOT / "stage9_revocation_state_v1.schema.json"


class PolicyVectorError(ValueError):
    """Canonical policy source or derived vector is invalid."""


def canonical_json_bytes(value: Any) -> bytes:
    """Encode using the dependency-free shared Stage-9 canonical profile."""
    try:
        return _canonical_json_bytes(value)
    except ValueError as exc:
        raise PolicyVectorError(str(exc)) from exc


def validate_schema(document: dict[str, Any], schema_path: Path) -> None:
    try:
        schema = json.loads(schema_path.read_text(encoding="utf-8"))
        jsonschema.Draft202012Validator(schema).validate(document)
    except (OSError, json.JSONDecodeError, jsonschema.ValidationError) as exc:
        raise PolicyVectorError(f"source material schema validation failed: {exc}") from exc


def canonical_digest(document: dict[str, Any]) -> bytes:
    return hashlib.sha256(canonical_json_bytes(document)).digest()
