#!/usr/bin/env python3
"""Buduje katalog Marketplace na podstawie specyfikacji presetów."""
from __future__ import annotations

import argparse
import base64
import hashlib
import hmac
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from bot_core.config_marketplace.schema import (  # noqa: E402
    MarketplaceCatalog,
    MarketplacePackageMetadata,
)

MARKETPLACE_CLI = REPO_ROOT / "scripts" / "marketplace_cli.py"
DEFAULT_PRESETS_DIR = REPO_ROOT / "config" / "marketplace" / "presets"
DEFAULT_PACKAGES_DIR = REPO_ROOT / "config" / "marketplace" / "packages"
DEFAULT_CATALOG = REPO_ROOT / "config" / "marketplace" / "catalog.json"


def _load_document(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return yaml.safe_load(text)


def _iter_specs(root: Path) -> Iterable[Path]:
    for path in sorted(root.rglob("*.json")):
        yield path
    for path in sorted(root.rglob("*.yaml")):
        yield path
    for path in sorted(root.rglob("*.yml")):
        yield path


def _ensure_json_serializable(payload: dict[str, Any]) -> dict[str, Any]:
    def _convert(value: Any) -> Any:
        if isinstance(value, datetime):
            return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")
        if isinstance(value, dict):
            return {k: _convert(v) for k, v in value.items()}
        if isinstance(value, list):
            return [_convert(item) for item in value]
        return value

    return _convert(payload)  # type: ignore[return-value]


def _normalize_timestamp(value: str | None) -> datetime:
    if not value:
        return datetime(1970, 1, 1, tzinfo=timezone.utc)
    text = value.replace("Z", "+00:00")
    try:
        return datetime.fromisoformat(text).astimezone(timezone.utc)
    except ValueError:
        raise ValueError(f"Niepoprawny format czasu: {value}") from None


def _compute_hmac(payload: dict[str, Any], *, key: bytes, algorithm: str) -> str:
    normalized = algorithm.strip().lower()
    if normalized not in {"hmac-sha256", "sha256"}:
        raise ValueError(f"Nieobsługiwany algorytm podpisu katalogu: {algorithm}")
    body = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    digest = hmac.new(key, body.encode("utf-8"), hashlib.sha256).digest()
    return base64.b64encode(digest).decode("ascii")


def _package_spec(
    *,
    spec_path: Path,
    output_path: Path,
    private_key: Path,
    key_id: str,
    issuer: str | None,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    format_arg = output_path.suffix.lower().lstrip(".")
    if format_arg not in {"json", "yaml", "yml"}:
        format_arg = "json"
    cli_args = [
        str(MARKETPLACE_CLI),
        "package",
        str(spec_path),
        "--output",
        str(output_path),
        "--private-key",
        str(private_key),
        "--key-id",
        key_id,
        "--format",
        "yaml" if format_arg in {"yaml", "yml"} else "json",
    ]
    if issuer:
        cli_args.extend(["--issuer", issuer])
    subprocess.run([sys.executable, *cli_args], check=True)


def build_catalog(
    *,
    presets_dir: Path,
    packages_dir: Path,
    catalog_path: Path,
    private_key: Path,
    key_id: str,
    signing_keys: dict[str, bytes],
    issuer: str | None = None,
) -> MarketplaceCatalog:
    packages: list[MarketplacePackageMetadata] = []
    latest_release = datetime(1970, 1, 1, tzinfo=timezone.utc)
    for spec_path in _iter_specs(presets_dir):
        document = _load_document(spec_path)
        catalog_data = document.get("catalog")
        if not isinstance(catalog_data, dict):
            continue

        preset_payload = document.get("preset")
        if not isinstance(preset_payload, Mapping):
            raise ValueError(f"Spec {spec_path} nie zawiera sekcji preset")

        metadata_dict: dict[str, Any] = json.loads(json.dumps(catalog_data))
        distributions = metadata_dict.get("distribution", [])
        if not distributions:
            raise ValueError(f"Spec {spec_path} nie zawiera sekcji distribution")

        for artifact in distributions:
            uri = artifact.get("uri")
            if not isinstance(uri, str):
                raise ValueError(f"Artefakt w {spec_path} nie ma poprawnego pola uri")
            artifact_path = packages_dir.parent / uri
            _package_spec(
                spec_path=spec_path,
                output_path=artifact_path,
                private_key=private_key,
                key_id=key_id,
                issuer=issuer,
            )
            blob = artifact_path.read_bytes()
            artifact["size_bytes"] = len(blob)
            digest = hashlib.sha256(blob).hexdigest()
            artifact["integrity"] = {"algorithm": "sha256", "digest": digest}

            signature = artifact.get("signature") or {}
            key_name = signature.get("key_id")
            if key_name:
                key_bytes = signing_keys.get(str(key_name))
                if not key_bytes:
                    raise ValueError(
                        f"Brak klucza podpisu '{key_name}' wymaganego dla artefaktu {uri}"
                    )
                payload = {
                    "package_id": metadata_dict.get("package_id"),
                    "version": metadata_dict.get("version"),
                    "artifact": artifact.get("name"),
                    "uri": uri,
                    artifact["integrity"]["algorithm"].lower(): digest,
                }
                signature["value"] = _compute_hmac(
                    payload,
                    key=key_bytes,
                    algorithm=str(signature.get("algorithm", "HMAC-SHA256")),
                )
                artifact["signature"] = signature

        metadata_dict = _ensure_json_serializable(metadata_dict)
        package = MarketplacePackageMetadata.model_validate(metadata_dict)
        packages.append(package)
        if package.release_date:
            latest_release = max(latest_release, package.release_date.astimezone(timezone.utc))

    packages.sort(key=lambda item: item.package_id)
    generated_at = latest_release if packages else datetime.now(timezone.utc)
    catalog = MarketplaceCatalog(
        schema_version="1.1",
        generated_at=generated_at,
        packages=packages,
    )
    catalog_path.parent.mkdir(parents=True, exist_ok=True)
    catalog_path.write_text(
        catalog.model_dump_json(indent=2, by_alias=False),
        encoding="utf-8",
    )
    return catalog


def _load_signing_keys(entries: Iterable[str]) -> dict[str, bytes]:
    mapping: dict[str, bytes] = {}
    for entry in entries:
        key_id, _, location = entry.partition(":")
        if not key_id or not location:
            raise ValueError(
                "Oczekiwano formatu KEY_ID:/sciezka/do/klucza dla kluczy podpisu katalogu."
            )
        mapping[key_id] = Path(location).expanduser().read_bytes()
    return mapping


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--presets", default=str(DEFAULT_PRESETS_DIR), help="Katalog ze specyfikacjami presetów.")
    parser.add_argument("--packages", default=str(DEFAULT_PACKAGES_DIR), help="Katalog docelowy artefaktów Marketplace.")
    parser.add_argument("--catalog", default=str(DEFAULT_CATALOG), help="Ścieżka docelowa katalogu Marketplace.")
    parser.add_argument("--private-key", required=True, help="Klucz prywatny Ed25519 do podpisu presetów.")
    parser.add_argument("--key-id", required=True, help="Identyfikator klucza Ed25519 do podpisu presetów.")
    parser.add_argument("--issuer", help="Opcjonalny identyfikator wystawcy podpisu presetów.")
    parser.add_argument(
        "--signing-key",
        action="append",
        required=True,
        help="Klucz HMAC do podpisu artefaktów w formacie KEY_ID:/sciezka/do/klucza (można powtarzać).",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    presets_dir = Path(args.presets).resolve()
    packages_dir = Path(args.packages).resolve()
    catalog_path = Path(args.catalog).resolve()
    private_key = Path(args.private_key).expanduser().resolve()
    signing_keys = _load_signing_keys(args.signing_key)

    build_catalog(
        presets_dir=presets_dir,
        packages_dir=packages_dir,
        catalog_path=catalog_path,
        private_key=private_key,
        key_id=args.key_id,
        signing_keys=signing_keys,
        issuer=args.issuer,
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
