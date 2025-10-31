"""Buduje pakiet dystrybucyjny desktopowego bota z zaszyfrowaną licencją."""

from __future__ import annotations

import argparse
import json
import shutil
import tarfile
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Mapping, Sequence

from bot_core.security.license_store import LicenseStore, LicenseStoreError
from bot_core.security.signing import build_hmac_signature
from bot_core.security.fingerprint import decode_secret


@dataclass(slots=True)
class IncludeMapping:
    """Opis dodatkowego katalogu kopiowanego do paczki."""

    target: Path
    source: Path


class DesktopBuildError(RuntimeError):
    """Sygnalizuje problem podczas przygotowywania paczki instalacyjnej."""


DEFAULT_OUTPUT = Path("var/dist/installers")


def _copy_tree(source: Path, destination: Path) -> None:
    if not source.exists():
        raise DesktopBuildError(f"Katalog źródłowy {source} nie istnieje")
    if destination.exists():
        shutil.rmtree(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(source, destination)


def _hash_file(path: Path, algorithm: str = "sha384") -> str:
    import hashlib

    digest = hashlib.new(algorithm)
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _parse_include_entries(entries: Iterable[str]) -> list[IncludeMapping]:
    mappings: list[IncludeMapping] = []
    for entry in entries:
        alias, separator, raw_path = entry.partition("=")
        if separator != "=" or not alias:
            raise DesktopBuildError(
                f"Niepoprawny format include '{entry}'. Użyj składni docelowy=ścieżka.")
        alias_path = Path(alias.strip())
        if alias_path.is_absolute():
            raise DesktopBuildError("Ścieżka docelowa include nie może być absolutna")
        source_path = Path(raw_path.strip()).expanduser().resolve()
        mappings.append(IncludeMapping(target=alias_path, source=source_path))
    return mappings


def _embed_license(
    staging_root: Path,
    *,
    license_json: Path,
    fingerprint: str,
    output_name: str,
    hmac_key: str | None,
) -> Mapping[str, str]:
    try:
        payload = json.loads(license_json.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:  # pragma: no cover - błędne dane użytkownika
        raise DesktopBuildError(f"Plik licencji zawiera niepoprawny JSON: {exc}") from exc
    if not isinstance(payload, Mapping):
        raise DesktopBuildError("Licencja musi być obiektem JSON")

    license_id = str(payload.get("license_id") or payload.get("licenseId") or "").strip()
    if not license_id:
        raise DesktopBuildError("Licencja nie zawiera pola license_id")

    license_dir = staging_root / "resources" / "license"
    license_dir.mkdir(parents=True, exist_ok=True)
    store_path = license_dir / output_name

    store = LicenseStore(path=store_path, fingerprint_override=fingerprint)
    store_payload: dict[str, object] = {
        "licenses": {
            license_id: {
                "payload": payload,
                "status": "provisioned",
                "issues": [],
                "hardware": {},
                "provisioned_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            }
        }
    }
    try:
        store.save(store_payload)
    except LicenseStoreError as exc:  # pragma: no cover - błędny fingerprint
        raise DesktopBuildError(f"Nie udało się zaszyfrować magazynu licencji: {exc}") from exc

    digest = _hash_file(store_path)
    report_payload: dict[str, object] = {
        "generated_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "license_store": {
            "path": store_path.relative_to(staging_root).as_posix(),
            "sha384": digest,
            "license_id": license_id,
        },
    }
    if hmac_key:
        key_id, secret = _parse_signing_key(hmac_key)
        signature = build_hmac_signature(
            {
                "generated_at": report_payload["generated_at"],
                "license_store": report_payload["license_store"],
            },
            key=secret,
            key_id=key_id,
        )
        report_payload["signature"] = signature

    integrity_path = license_dir / "license_integrity.json"
    integrity_path.write_text(json.dumps(report_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return {
        "license_id": license_id,
        "store_path": store_path.relative_to(staging_root).as_posix(),
        "integrity_path": integrity_path.relative_to(staging_root).as_posix(),
    }


def _parse_signing_key(value: str) -> tuple[str | None, bytes]:
    key_id: str | None = None
    secret = value
    if "=" in value:
        key_id, secret = value.split("=", 1)
        key_id = key_id.strip() or None
    secret_bytes = decode_secret(secret.strip())
    return key_id, secret_bytes


def build_distribution(
    *,
    version: str,
    platform: str,
    runtime_dir: Path,
    ui_dir: Path | None,
    includes: Iterable[str],
    license_json: Path,
    license_fingerprint: str,
    output_dir: Path,
    bundle_name: str | None = None,
    license_output_name: str = "license_store.json",
    license_hmac_key: str | None = None,
    signing_key: str | None = None,
) -> Path:
    """Buduje archiwum tar.gz z kompletną dystrybucją desktopową."""

    runtime_dir = runtime_dir.expanduser().resolve()
    if not runtime_dir.exists():
        raise DesktopBuildError(f"Katalog runtime {runtime_dir} nie istnieje")

    ui_path = ui_dir.expanduser().resolve() if ui_dir else None
    if ui_path is not None and not ui_path.exists():
        raise DesktopBuildError(f"Katalog UI {ui_path} nie istnieje")

    include_mappings = _parse_include_entries(includes)

    output_dir = output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    bundle_id = bundle_name or f"bot-desktop-{version}-{platform}"

    staging_dir = Path(tempfile.mkdtemp(prefix="desktop_bundle_"))
    try:
        bundle_root = staging_dir / bundle_id
        bundle_root.mkdir(parents=True, exist_ok=True)

        _copy_tree(runtime_dir, bundle_root / "runtime")
        if ui_path is not None:
            _copy_tree(ui_path, bundle_root / "ui")
        for mapping in include_mappings:
            _copy_tree(mapping.source, bundle_root / mapping.target)

        license_info = _embed_license(
            bundle_root,
            license_json=license_json.expanduser().resolve(),
            fingerprint=license_fingerprint,
            output_name=license_output_name,
            hmac_key=license_hmac_key,
        )

        manifest_entries: list[dict[str, object]] = []
        for file_path in sorted(bundle_root.rglob("*")):
            if not file_path.is_file():
                continue
            relative = file_path.relative_to(bundle_root).as_posix()
            manifest_entries.append(
                {
                    "path": relative,
                    "sha384": _hash_file(file_path),
                    "size": file_path.stat().st_size,
                }
            )

        manifest: dict[str, object] = {
            "version": version,
            "platform": platform,
            "generated_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            "artifacts": manifest_entries,
            "license": license_info,
        }

        manifest_path = bundle_root / "manifest.json"
        manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

        if signing_key:
            key_id, secret = _parse_signing_key(signing_key)
            signature = build_hmac_signature(manifest, key=secret, key_id=key_id)
            (bundle_root / "manifest.sig").write_text(
                json.dumps(signature, ensure_ascii=False, indent=2), encoding="utf-8"
            )

        archive_path = output_dir / f"{bundle_id}.tar.gz"
        if archive_path.exists():
            archive_path.unlink()
        with tarfile.open(archive_path, "w:gz") as archive:
            archive.add(bundle_root, arcname=bundle_root.name)
    finally:
        shutil.rmtree(staging_dir, ignore_errors=True)

    return archive_path


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--version", required=True, help="Wersja pakietu")
    parser.add_argument("--platform", required=True, choices=["linux", "macos", "windows"], help="Docelowa platforma")
    parser.add_argument("--runtime-dir", required=True, help="Katalog z binarkami runtime")
    parser.add_argument("--ui-dir", help="Opcjonalny katalog z zasobami UI")
    parser.add_argument("--include", action="append", default=[], help="Dodatkowe katalogi w formacie docelowy=ścieżka")
    parser.add_argument("--license-json", required=True, help="Plik JSON z licencją do zaszyfrowania")
    parser.add_argument("--license-fingerprint", required=True, help="Fingerprint docelowego hosta")
    parser.add_argument("--license-output-name", default="license_store.json", help="Nazwa pliku magazynu licencji w paczce")
    parser.add_argument("--license-hmac-key", help="Klucz HMAC (opcjonalny) do podpisania metadanych licencji")
    parser.add_argument("--signing-key", help="Klucz HMAC (KEY_ID=wartość) do podpisu manifestu")
    parser.add_argument("--bundle-name", help="Własna nazwa katalogu w archiwum")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT), help="Katalog docelowy z archiwami")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)

    archive = build_distribution(
        version=args.version,
        platform=args.platform,
        runtime_dir=Path(args.runtime_dir),
        ui_dir=Path(args.ui_dir) if args.ui_dir else None,
        includes=args.include,
        license_json=Path(args.license_json),
        license_fingerprint=args.license_fingerprint,
        output_dir=Path(args.output_dir),
        bundle_name=args.bundle_name,
        license_output_name=args.license_output_name,
        license_hmac_key=args.license_hmac_key,
        signing_key=args.signing_key,
    )
    print(f"Zbudowano pakiet: {archive}")
    return 0


if __name__ == "__main__":  # pragma: no cover - wejście CLI
    raise SystemExit(main())
