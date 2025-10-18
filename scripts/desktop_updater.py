"""Minimal desktop updater used by the OEM installer."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import zipfile
from pathlib import Path

from bot_core.security.signing import build_hmac_signature, canonical_json_bytes


def _verify_signature(archive: Path, signature_path: Path, key: str) -> None:
    payload = {
        "archive": str(archive.resolve()),
        "size": archive.stat().st_size,
    }
    calculated = build_hmac_signature(key.encode("utf-8"), canonical_json_bytes(payload))
    expected = signature_path.read_text(encoding="utf-8").strip()
    if calculated.get("value") != expected:
        raise SystemExit("Updater signature verification failed")


def _extract(archive: Path, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive, "r") as handle:
        handle.extractall(destination)


def _load_manifest(manifest_path: Path) -> dict[str, object]:
    try:
        return json.loads(manifest_path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:  # pragma: no cover - CLI feedback
        raise SystemExit(f"Manifest not found: {manifest_path}") from exc


def run(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="OEM desktop updater")
    parser.add_argument("--archive", required=True, help="Path to the update archive (zip)")
    parser.add_argument("--manifest", required=True, help="Path to the JSON manifest")
    parser.add_argument("--signature", required=True, help="Path to the signature file")
    parser.add_argument("--key", required=True, help="HMAC secret used for signature verification")
    parser.add_argument("--target", required=True, help="Directory where files should be extracted")
    parser.add_argument("--backup", help="Optional directory for backups before update")
    args = parser.parse_args(argv)

    archive_path = Path(args.archive).expanduser()
    manifest_path = Path(args.manifest).expanduser()
    signature_path = Path(args.signature).expanduser()
    target_dir = Path(args.target).expanduser()

    manifest = _load_manifest(manifest_path)
    if args.backup:
        backup_dir = Path(args.backup).expanduser()
        backup_dir.mkdir(parents=True, exist_ok=True)
        if target_dir.exists():
            shutil.make_archive(str(backup_dir / "bot_shell_backup"), "zip", target_dir)

    _verify_signature(archive_path, signature_path, args.key)
    _extract(archive_path, target_dir)

    manifest_output = target_dir / "update_manifest.json"
    manifest_output.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(run())
