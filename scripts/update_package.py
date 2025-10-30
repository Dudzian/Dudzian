#!/usr/bin/env python3
"""Narzędzie CLI do budowy oraz walidacji pakietów aktualizacji offline."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any

from bot_core.security.fingerprint import decode_secret
from bot_core.security.signing import build_hmac_signature
from bot_core.security.update import verify_update_bundle
from bot_core.security.update_bundle_utils import (
    UpdateDescriptionError,
    describe_update_directory,
    hash_file,
)


def build_package(args: argparse.Namespace) -> int:
    output_dir = Path(args.output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest: dict[str, Any] = {
        "id": args.package_id,
        "version": args.version,
        "platform": args.platform,
        "runtime": args.runtime,
        "artifacts": [],
        "metadata": {},
    }

    payload_source = Path(args.payload).expanduser()
    if not payload_source.exists():
        raise FileNotFoundError(payload_source)
    payload_target = output_dir / payload_source.name
    shutil.copy2(payload_source, payload_target)
    manifest["artifacts"].append(
        {
            "path": payload_target.name,
            "sha384": hash_file(payload_target, "sha384"),
            "sha256": hash_file(payload_target, "sha256"),
            "size": payload_target.stat().st_size,
            "type": "full",
        }
    )

    if args.diff:
        diff_source = Path(args.diff).expanduser()
        if not diff_source.exists():
            raise FileNotFoundError(diff_source)
        diff_target = output_dir / diff_source.name
        shutil.copy2(diff_source, diff_target)
        entry: dict[str, Any] = {
            "path": diff_target.name,
            "sha384": hash_file(diff_target, "sha384"),
            "sha256": hash_file(diff_target, "sha256"),
            "size": diff_target.stat().st_size,
            "type": "diff",
        }
        if args.base_id:
            entry["base_id"] = args.base_id
        manifest["artifacts"].append(entry)

    if args.integrity_manifest:
        integrity_source = Path(args.integrity_manifest).expanduser()
        if not integrity_source.exists():
            raise FileNotFoundError(integrity_source)
        integrity_target = output_dir / integrity_source.name
        shutil.copy2(integrity_source, integrity_target)
        manifest["integrity_manifest"] = {
            "path": integrity_target.name,
            "sha256": hash_file(integrity_target, "sha256"),
        }

    if args.metadata:
        manifest["metadata"] = json.loads(args.metadata)

    if args.key:
        key_bytes = decode_secret(args.key)
        manifest["signature"] = build_hmac_signature(manifest, key=key_bytes, key_id=args.key_id)

    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"status": "ok", "manifest": str(manifest_path)}))
    return 0


def verify_package(args: argparse.Namespace) -> int:
    package_dir = Path(args.package_dir).expanduser()
    manifest_path = package_dir / "manifest.json"
    if not manifest_path.exists():
        print(json.dumps({"status": "error", "error": f"Brak manifestu {manifest_path}"}))
        return 1

    signature_path: Path | None = None
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)
        manifest_data = json.loads(manifest_path.read_text(encoding="utf-8"))
        signature_data = manifest_data.get("signature")
        if signature_data:
            signature_path = tmpdir_path / "signature.json"
            signature_path.write_text(json.dumps(signature_data, ensure_ascii=False), encoding="utf-8")

        key_bytes = decode_secret(args.key) if args.key else None
        result = verify_update_bundle(
            manifest_path=manifest_path,
            base_dir=package_dir,
            signature_path=signature_path,
            hmac_key=key_bytes,
            license_result=None,
        )
        if not result.is_successful:
            payload = {
                "status": "error",
                "errors": result.errors,
                "warnings": result.warnings,
            }
            print(json.dumps(payload, ensure_ascii=False))
            return 2

    payload = {
        "status": "ok",
        "artifacts": [artifact.path for artifact in result.manifest.artifacts],
    }
    print(json.dumps(payload, ensure_ascii=False))
    return 0


def scan_packages(args: argparse.Namespace) -> int:
    packages_dir = Path(args.packages_dir).expanduser()
    if not packages_dir.exists():
        print(
            json.dumps(
                {
                    "status": "error",
                    "error": f"Katalog pakietów {packages_dir} nie istnieje",
                },
                ensure_ascii=False,
            )
        )
        return 1

    results: list[dict[str, Any]] = []
    for entry in sorted(packages_dir.iterdir()):
        if not entry.is_dir():
            continue
        try:
            description = describe_update_directory(entry)
        except UpdateDescriptionError as exc:
            results.append(
                {
                    "status": "error",
                    "path": str(entry),
                    "error": str(exc),
                }
            )
            continue

        results.append(
            {
                "status": "ok",
                "path": str(entry),
                **description,
            }
        )

    payload = {"status": "ok", "packages": results}
    print(json.dumps(payload, ensure_ascii=False))
    return 0


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Buduje i weryfikuje pakiety aktualizacji offline")
    subparsers = parser.add_subparsers(dest="command", required=True)

    build_parser = subparsers.add_parser("build", help="Buduje manifest aktualizacji i podpisuje artefakty")
    build_parser.add_argument("--output-dir", required=True, help="Katalog docelowy pakietu")
    build_parser.add_argument("--package-id", required=True)
    build_parser.add_argument("--version", required=True)
    build_parser.add_argument("--platform", required=True)
    build_parser.add_argument("--runtime", required=True)
    build_parser.add_argument("--payload", required=True, help="Pełny pakiet TAR")
    build_parser.add_argument("--diff", help="Łatka różnicowa")
    build_parser.add_argument("--base-id", help="Identyfikator bazowej wersji dla łatki")
    build_parser.add_argument("--integrity-manifest", dest="integrity_manifest")
    build_parser.add_argument("--metadata", help="Dodatkowe metadane JSON")
    build_parser.add_argument("--key", help="Klucz HMAC do podpisu (hex/base64)")
    build_parser.add_argument("--key-id", help="Identyfikator klucza podpisującego")

    verify_parser = subparsers.add_parser("verify", help="Weryfikuje pakiet aktualizacji")
    verify_parser.add_argument("--package-dir", required=True)
    verify_parser.add_argument("--key", help="Klucz HMAC do weryfikacji (hex/base64)")

    scan_parser = subparsers.add_parser(
        "scan", help="Generuje opis wszystkich pakietów w katalogu"
    )
    scan_parser.add_argument("--packages-dir", required=True)

    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        if args.command == "build":
            return build_package(args)
        if args.command == "verify":
            return verify_package(args)
        if args.command == "scan":
            return scan_packages(args)
    except Exception as exc:  # pragma: no cover - błędy raportowane do stderr
        print(json.dumps({"status": "error", "error": str(exc)}), file=sys.stderr)
        return 1
    raise ValueError(f"Nieznane polecenie {args.command}")


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

