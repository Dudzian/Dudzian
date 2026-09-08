"""Executable integrity guard for the frozen M0.2--M0.11 architecture baseline."""

from __future__ import annotations

import copy
import hashlib
import json
import re
from pathlib import Path
from typing import Any

import pytest

ROOT = Path(__file__).parents[2]
DOCS = ROOT / "docs/architecture/cryptohunter_product_architecture"
MANIFEST_PATH = DOCS / "architecture_baseline_freeze.json"
PROJECTION_PATH = DOCS / "architecture_baseline_freeze.md"
EXPECTED_MILESTONES = [f"M0.{number}" for number in range(2, 12)]
EXPECTED_SCHEMA_VERSION = "cryptohunter.architecture_baseline_freeze.v1"
EXPECTED_PURPOSE = "INTEGRITY_GUARD_ONLY_NOT_RUNTIME_AUTHORITY"
RECORD_FIELDS = {
    "milestone_id",
    "canonical_artifact",
    "canonical_schema_version",
    "sha256",
}
FORBIDDEN_AUTHORITY_FIELDS = {
    "accepted",
    "current",
    "approved",
    "authorized",
    "execution_allowed",
    "live_allowed",
}
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


def _load_json(path: Path) -> Any:
    return json.loads(path.read_bytes())


def _all_keys(value: Any) -> set[str]:
    if isinstance(value, dict):
        return set(value) | {key for item in value.values() for key in _all_keys(item)}
    if isinstance(value, list):
        return {key for item in value for key in _all_keys(item)}
    return set()


def _validate_manifest(manifest: dict[str, Any], root: Path = ROOT) -> None:
    assert set(manifest) == {
        "schema_version",
        "baseline_id",
        "status",
        "purpose",
        "milestones",
    }
    assert manifest["schema_version"] == EXPECTED_SCHEMA_VERSION
    assert manifest["baseline_id"] == "M0.2-M0.11"
    assert manifest["status"] == "FROZEN"
    assert manifest["purpose"] == EXPECTED_PURPOSE
    assert not (_all_keys(manifest) & FORBIDDEN_AUTHORITY_FIELDS)

    milestones = manifest["milestones"]
    assert isinstance(milestones, list)
    milestone_ids = [record["milestone_id"] for record in milestones]
    assert milestone_ids == EXPECTED_MILESTONES
    assert len(milestone_ids) == len(set(milestone_ids))

    resolved_root = root.resolve()
    for record in milestones:
        assert set(record) == RECORD_FIELDS
        assert isinstance(record["canonical_artifact"], str)
        artifact = (root / record["canonical_artifact"]).resolve()
        assert artifact.is_relative_to(resolved_root)
        assert artifact.is_file()
        raw = artifact.read_bytes()
        canonical = json.loads(raw)
        assert canonical["schema_version"] == record["canonical_schema_version"]
        assert isinstance(record["sha256"], str)
        assert SHA256_RE.fullmatch(record["sha256"])
        assert hashlib.sha256(raw).hexdigest() == record["sha256"]


def _render_markdown(manifest: dict[str, Any]) -> str:
    rows = "\n".join(
        f"| {record['milestone_id']} | `{record['canonical_artifact']}` | "
        f"`{record['canonical_schema_version']}` | `{record['sha256']}` |"
        for record in manifest["milestones"]
    )
    return f"""# CryptoHunter Architecture Baseline Freeze

## {manifest["baseline_id"]} — {manifest["status"]}

Cel: `{manifest["purpose"]}`.

| Milestone | Canonical artifact | Canonical schema version | SHA-256 |
| --- | --- | --- | --- |
{rows}

Baseline M0.2–M0.11 jest zamrożony. Zmiana któregokolwiek canonical JSON wymaga jawnej aktualizacji freeze manifestu.

Hash dowodzi wyłącznie integralności. Freeze manifest nie jest runtime authority.

Późniejsza implementacja produkcyjna ma implementować istniejące kontrakty, a nie reinterpretować je.
"""


def _manifest() -> dict[str, Any]:
    value = _load_json(MANIFEST_PATH)
    assert isinstance(value, dict)
    return value


def test_canonical_freeze_manifest_guards_all_artifacts() -> None:
    _validate_manifest(_manifest())


def test_markdown_is_deterministic_manifest_projection() -> None:
    assert PROJECTION_PATH.read_text(encoding="utf-8") == _render_markdown(_manifest())


@pytest.mark.parametrize("missing", EXPECTED_MILESTONES)
def test_manifest_fails_closed_for_each_missing_milestone(missing: str) -> None:
    manifest = _manifest()
    manifest["milestones"] = [
        record for record in manifest["milestones"] if record["milestone_id"] != missing
    ]
    with pytest.raises(AssertionError):
        _validate_manifest(manifest)


def test_manifest_fails_closed_for_extra_milestone() -> None:
    manifest = _manifest()
    extra = copy.deepcopy(manifest["milestones"][-1])
    extra["milestone_id"] = "M0.12"
    manifest["milestones"].append(extra)
    with pytest.raises(AssertionError):
        _validate_manifest(manifest)


def test_manifest_fails_closed_for_duplicate_milestone() -> None:
    manifest = _manifest()
    manifest["milestones"].append(copy.deepcopy(manifest["milestones"][-1]))
    with pytest.raises(AssertionError):
        _validate_manifest(manifest)


@pytest.mark.parametrize("field", sorted(FORBIDDEN_AUTHORITY_FIELDS))
def test_manifest_rejects_runtime_authority_fields(field: str) -> None:
    manifest = _manifest()
    manifest[field] = True
    with pytest.raises(AssertionError):
        _validate_manifest(manifest)


@pytest.mark.parametrize("bad_hash", ["a" * 63, "A" * 64, "g" * 64, 0])
def test_manifest_fails_closed_for_malformed_sha256(bad_hash: Any) -> None:
    manifest = _manifest()
    manifest["milestones"][0]["sha256"] = bad_hash
    with pytest.raises(AssertionError):
        _validate_manifest(manifest)


def test_manifest_fails_closed_for_missing_artifact() -> None:
    manifest = _manifest()
    manifest["milestones"][0]["canonical_artifact"] = "missing.json"
    with pytest.raises(AssertionError):
        _validate_manifest(manifest)


def test_manifest_fails_closed_for_changed_schema_version() -> None:
    manifest = _manifest()
    manifest["milestones"][0]["canonical_schema_version"] = "changed"
    with pytest.raises(AssertionError):
        _validate_manifest(manifest)


def test_manifest_fails_closed_for_changed_canonical_bytes(tmp_path: Path) -> None:
    manifest = _manifest()
    source = ROOT / manifest["milestones"][0]["canonical_artifact"]
    target = tmp_path / "canonical.json"
    target.write_bytes(source.read_bytes() + b"\n")
    manifest["milestones"][0]["canonical_artifact"] = target.name
    with pytest.raises(AssertionError):
        _validate_manifest(manifest, tmp_path)
