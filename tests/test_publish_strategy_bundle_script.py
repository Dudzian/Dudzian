from __future__ import annotations

import json
from pathlib import Path

import pytest


from bot_core.security.signing import build_hmac_signature
from scripts.publish_strategy_bundle import run as publish_bundle
from tests._json_helpers import read_jsonl
from tests._signing_helpers import write_random_hmac_key


ROOT = Path(__file__).resolve().parents[1]
def _sha256(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def test_publish_strategy_bundle_end_to_end(tmp_path: Path) -> None:
    staging_dir = tmp_path / "staging"
    release_root = tmp_path / "releases"
    decision_log = tmp_path / "audit" / "decision_log.jsonl"

    signing_key_path = tmp_path / "keys" / "bundle.key"
    decision_key_path = tmp_path / "keys" / "decision.key"
    decision_key = write_random_hmac_key(decision_key_path)
    write_random_hmac_key(signing_key_path)

    exit_code = publish_bundle(
        [
            "--version",
            "1.2.3",
            "--signing-key-path",
            str(signing_key_path),
            "--signing-key-id",
            "stage4-signing",
            "--staging-dir",
            str(staging_dir),
            "--release-dir",
            str(release_root),
            "--decision-log-path",
            str(decision_log),
            "--decision-log-hmac-key-file",
            str(decision_key_path),
            "--decision-log-key-id",
            "stage4-dl",
            "--decision-log-category",
            "release.stage4.strategy",
            "--decision-log-notes",
            "Nightly Stage4 bundle",
        ]
    )

    assert exit_code == 0

    release_dir = release_root / "1.2.3"
    assert release_dir.is_dir()

    archive = release_dir / "stage4-strategies-1.2.3.zip"
    manifest = release_dir / "stage4-strategies-1.2.3.manifest.json"
    signature = release_dir / "stage4-strategies-1.2.3.manifest.sig"
    metadata_path = release_dir / "metadata.json"

    for path in (archive, manifest, signature, metadata_path):
        assert path.exists()

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    assert metadata["version"] == "1.2.3"
    assert metadata["bundle_name"] == "stage4-strategies"
    bundle_digest = metadata["artifacts"][archive.name]["sha256"]
    assert bundle_digest == _sha256(archive)

    entries = read_jsonl(decision_log)
    assert len(entries) == 1
    entry = entries[0]
    signature_record = entry.pop("signature")
    expected_signature = build_hmac_signature(
        entry,
        key=decision_key,
        algorithm="HMAC-SHA256",
        key_id="stage4-dl",
    )
    assert signature_record == expected_signature
    assert entry["schema"] == "stage4.strategy_release"
    assert entry["version"] == "1.2.3"
    assert entry["category"] == "release.stage4.strategy"
    assert entry["notes"] == "Nightly Stage4 bundle"
    assert entry["artifacts"][archive.name]["sha256"] == bundle_digest

    with pytest.raises(ValueError):
        publish_bundle(
            [
                "--version",
                "1.2.3",
                "--signing-key-path",
                str(signing_key_path),
                "--staging-dir",
                str(staging_dir),
                "--release-dir",
                str(release_root),
            ]
        )
