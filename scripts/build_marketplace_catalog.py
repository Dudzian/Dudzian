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
from typing import Any, Iterable, Mapping, Sequence

import yaml
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import ed25519

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from bot_core.config_marketplace.schema import (  # noqa: E402
    MarketplaceCatalog,
    MarketplacePackageMetadata,
)

MARKETPLACE_CLI_MODULE = "scripts.marketplace_cli"
DEFAULT_PRESETS_DIR = REPO_ROOT / "config" / "marketplace" / "presets"
DEFAULT_PACKAGES_DIR = REPO_ROOT / "config" / "marketplace" / "packages"
DEFAULT_CATALOG = REPO_ROOT / "config" / "marketplace" / "catalog.json"
DEFAULT_MARKDOWN = REPO_ROOT / "config" / "marketplace" / "catalog.md"


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


def _load_ed25519_private_key(path: Path) -> ed25519.Ed25519PrivateKey:
    data = path.read_bytes()
    try:
        return serialization.load_pem_private_key(data, password=None)
    except ValueError:
        try:
            return ed25519.Ed25519PrivateKey.from_private_bytes(base64.b64decode(data))
        except Exception as exc:  # noqa: BLE001
            raise ValueError(f"Nie udało się wczytać klucza Ed25519 z {path}") from exc


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
        "-m",
        MARKETPLACE_CLI_MODULE,
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


def _backfill_exchange_last_verified_at(metadata: dict[str, Any]) -> None:
    """Ensure certified/production exchange entries have last_verified_at.

    Uses the package approved_at timestamp (when present) so generated
    catalogs do not regress to null values when specs omit the field.
    """

    release = metadata.get("release") or {}
    approved_at = release.get("approved_at")
    if not approved_at:
        return
    entries = metadata.get("exchange_compatibility") or []
    if not isinstance(entries, list):
        return

    for entry in entries:
        if not isinstance(entry, dict):
            continue
        status = str(entry.get("status") or "").strip().lower()
        if status in {"certified", "production"} and not entry.get("last_verified_at"):
            entry["last_verified_at"] = approved_at


def _format_budget_range(values: Sequence[float]) -> str:
    if not values:
        return "—"
    minimum = min(values)
    maximum = max(values)
    if abs(maximum - minimum) < 1e-6:
        return f"${minimum:,.0f}"
    return f"${minimum:,.0f}–${maximum:,.0f}"


def _markdown_escape(value: str) -> str:
    return value.replace("|", "\\|")


def _spec_label(spec_path: Path, root: Path) -> str:
    try:
        return str(spec_path.relative_to(root))
    except ValueError:
        return str(spec_path)


def _build_markdown(catalog: MarketplaceCatalog) -> str:
    lines = ["# Marketplace Catalog", ""]
    lines.append(f"*Wygenerowano:* {catalog.generated_at.isoformat().replace('+00:00', 'Z')}")
    lines.append("")
    header = "| Paczka | Wersja | Persony | Ryzyko | Budżet | Podsumowanie |"
    separator = "| --- | --- | --- | --- | --- | --- |"
    lines.extend([header, separator])
    for package in sorted(catalog.packages, key=lambda entry: entry.package_id):
        personas = ", ".join(
            _markdown_escape(pref.persona)
            for pref in package.user_preferences
            if getattr(pref, "persona", "")
        ) or "—"
        risk_summary = ", ".join(
            sorted(
                {
                    pref.risk_target.strip()
                    for pref in package.user_preferences
                    if pref.risk_target
                }
            )
        ) or "—"
        budgets = [float(pref.recommended_budget) for pref in package.user_preferences if pref.recommended_budget]
        budget_summary = _format_budget_range(budgets)
        summary = _markdown_escape(package.summary or "—")
        lines.append(
            f"| {package.display_name} | {package.version} | {personas} | {risk_summary} | {budget_summary} | {summary} |"
        )
    lines.append("")
    return "\n".join(lines)


def _write_signature(
    path: Path,
    *,
    content: str,
    hmac_key_id: str | None,
    signing_keys: Mapping[str, bytes],
    ed25519_key: ed25519.Ed25519PrivateKey | None,
    ed25519_key_id: str | None,
    issuer: str | None,
) -> None:
    if not hmac_key_id and not (ed25519_key and ed25519_key_id):
        return
    digest = hashlib.sha256(content.encode("utf-8")).hexdigest()
    timestamp = datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")
    payload: dict[str, Any] = {"target": str(path), "sha256": digest}

    if hmac_key_id:
        key_bytes = signing_keys.get(hmac_key_id)
        if not key_bytes:
            raise ValueError(f"Brak klucza HMAC '{hmac_key_id}' wymaganego do podpisu katalogu")
        hmac_value = hmac.new(key_bytes, content.encode("utf-8"), hashlib.sha256).digest()
        payload["hmac"] = {
            "algorithm": "HMAC-SHA256",
            "key_id": hmac_key_id,
            "signed_at": timestamp,
            "value": base64.b64encode(hmac_value).decode("ascii"),
        }

    if ed25519_key and ed25519_key_id:
        signature = ed25519_key.sign(content.encode("utf-8"))
        payload["ed25519"] = {
            "algorithm": "ed25519",
            "key_id": ed25519_key_id,
            "issuer": issuer,
            "signed_at": timestamp,
            "value": base64.b64encode(signature).decode("ascii"),
            "public_key": base64.b64encode(
                ed25519_key.public_key().public_bytes(
                    encoding=serialization.Encoding.Raw,
                    format=serialization.PublicFormat.Raw,
                )
            ).decode("ascii"),
        }

    path.with_suffix(path.suffix + ".sig").write_text(json.dumps(payload, indent=2), encoding="utf-8")


def build_catalog(
    *,
    presets_dir: Path,
    packages_dir: Path,
    catalog_path: Path,
    markdown_path: Path,
    private_key: Path,
    key_id: str,
    signing_keys: dict[str, bytes],
    issuer: str | None = None,
    catalog_signature_key: str | None = None,
    catalog_ed25519_key: ed25519.Ed25519PrivateKey | None = None,
    catalog_ed25519_key_id: str | None = None,
) -> MarketplaceCatalog:
    packages: list[MarketplacePackageMetadata] = []
    package_sources: list[tuple[MarketplacePackageMetadata, Path]] = []
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

        _backfill_exchange_last_verified_at(metadata_dict)
        metadata_dict = _ensure_json_serializable(metadata_dict)
        package = MarketplacePackageMetadata.model_validate(metadata_dict)
        packages.append(package)
        package_sources.append((package, spec_path))
        if package.release_date:
            latest_release = max(latest_release, package.release_date.astimezone(timezone.utc))

    strategy_packages = [
        (package, spec_path)
        for package, spec_path in package_sources
        if package.user_preferences
    ]
    if len(strategy_packages) < 15:
        raise ValueError(
            "Katalog Marketplace musi zawierać co najmniej 15 strategii z metadanymi user_preferences"
        )
    missing_persona = [
        (package, spec_path)
        for package, spec_path in strategy_packages
        if any(not (getattr(pref, "persona", "") or "").strip() for pref in package.user_preferences)
    ]
    if missing_persona:
        formatted = ", ".join(
            f"{entry.package_id} ({_spec_label(spec_path, presets_dir)})"
            for entry, spec_path in missing_persona
        )
        raise ValueError(
            "Brak kompletnych metadanych person w strategiach: " + formatted
        )

    packages.sort(key=lambda item: item.package_id)
    generated_at = latest_release if packages else datetime.now(timezone.utc)
    catalog = MarketplaceCatalog(
        schema_version="1.1",
        generated_at=generated_at,
        packages=packages,
    )
    catalog_path.parent.mkdir(parents=True, exist_ok=True)
    catalog_json = catalog.model_dump_json(indent=2, by_alias=False)
    catalog_path.write_text(catalog_json, encoding="utf-8")
    _write_signature(
        catalog_path,
        content=catalog_json,
        hmac_key_id=catalog_signature_key,
        signing_keys=signing_keys,
        ed25519_key=catalog_ed25519_key,
        ed25519_key_id=catalog_ed25519_key_id,
        issuer=issuer,
    )

    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_body = _build_markdown(catalog)
    markdown_path.write_text(markdown_body, encoding="utf-8")
    _write_signature(
        markdown_path,
        content=markdown_body,
        hmac_key_id=catalog_signature_key,
        signing_keys=signing_keys,
        ed25519_key=catalog_ed25519_key,
        ed25519_key_id=catalog_ed25519_key_id,
        issuer=issuer,
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
    parser.add_argument(
        "--markdown",
        default=str(DEFAULT_MARKDOWN),
        help="Ścieżka docelowa katalogu Marketplace w formacie Markdown.",
    )
    parser.add_argument("--private-key", required=True, help="Klucz prywatny Ed25519 do podpisu presetów.")
    parser.add_argument("--key-id", required=True, help="Identyfikator klucza Ed25519 do podpisu presetów.")
    parser.add_argument("--issuer", help="Opcjonalny identyfikator wystawcy podpisu presetów.")
    parser.add_argument(
        "--signing-key",
        action="append",
        required=True,
        help="Klucz HMAC do podpisu artefaktów w formacie KEY_ID:/sciezka/do/klucza (można powtarzać).",
    )
    parser.add_argument(
        "--catalog-signature-key",
        help="Identyfikator klucza HMAC używanego do podpisu katalogu JSON i Markdown.",
    )
    parser.add_argument(
        "--catalog-ed25519-key",
        help="Klucz prywatny Ed25519 używany do podpisu katalogu JSON/Markdown (domyślnie jak --private-key).",
    )
    parser.add_argument(
        "--catalog-ed25519-key-id",
        help="Identyfikator klucza Ed25519 podpisującego katalog JSON/Markdown.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    presets_dir = Path(args.presets).resolve()
    packages_dir = Path(args.packages).resolve()
    catalog_path = Path(args.catalog).resolve()
    markdown_path = Path(args.markdown).resolve()
    private_key = Path(args.private_key).expanduser().resolve()
    catalog_ed25519_path = (
        Path(args.catalog_ed25519_key).expanduser().resolve()
        if args.catalog_ed25519_key
        else private_key
    )
    signing_keys = _load_signing_keys(args.signing_key)
    catalog_ed25519_key_id = args.catalog_ed25519_key_id or args.key_id
    catalog_ed25519_key = (
        _load_ed25519_private_key(catalog_ed25519_path) if catalog_ed25519_key_id else None
    )

    build_catalog(
        presets_dir=presets_dir,
        packages_dir=packages_dir,
        catalog_path=catalog_path,
        markdown_path=markdown_path,
        private_key=private_key,
        key_id=args.key_id,
        signing_keys=signing_keys,
        issuer=args.issuer,
        catalog_signature_key=args.catalog_signature_key,
        catalog_ed25519_key=catalog_ed25519_key,
        catalog_ed25519_key_id=catalog_ed25519_key_id,
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
