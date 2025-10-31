"""Tworzenie podpisanych pakietów aktualizacji `.kbot`."""
from __future__ import annotations

import argparse
import json
import shutil
import tarfile
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping

from bot_core.security.signing import build_hmac_signature


def _hash_sha256(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _build_payload_archive(payload_dir: Path, target_path: Path) -> None:
    with tarfile.open(target_path, mode="w") as archive:
        for entry in sorted(payload_dir.rglob("*")):
            if entry.is_file():
                archive.add(entry, arcname=str(entry.relative_to(payload_dir)))


def _parse_metadata(values: list[str]) -> Mapping[str, object]:
    metadata: dict[str, object] = {}
    for item in values:
        if "=" not in item:
            raise argparse.ArgumentTypeError(
                f"Metadane muszą mieć format klucz=wartość (otrzymano: {item!r})"
            )
        key, value = item.split("=", 1)
        metadata[key.strip()] = value.strip()
    return metadata


def build_kbot_package(
    *,
    package_id: str,
    version: str,
    payload_dir: Path,
    output_path: Path,
    fingerprint: str | None = None,
    metadata: Mapping[str, object] | None = None,
    signing_key: bytes | None = None,
    signing_key_id: str | None = None,
) -> Path:
    """Tworzy podpisaną paczkę `.kbot` z katalogu źródłowego."""

    payload_dir = payload_dir.expanduser().resolve()
    if not payload_dir.exists() or not payload_dir.is_dir():
        raise FileNotFoundError(f"Katalog {payload_dir} nie istnieje lub nie jest katalogiem")

    output_path = output_path.expanduser()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    staging_dir = Path(tempfile.mkdtemp(prefix="kbot_build_"))
    try:
        payload_archive = staging_dir / "payload.tar"
        _build_payload_archive(payload_dir, payload_archive)

        artifacts = [
            {
                "path": "payload.tar",
                "size": payload_archive.stat().st_size,
                "sha256": _hash_sha256(payload_archive),
            }
        ]
        manifest: dict[str, object] = {
            "id": package_id,
            "version": version,
            "fingerprint": fingerprint,
            "created_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            "artifacts": artifacts,
        }
        if metadata:
            manifest["metadata"] = dict(metadata)

        manifest_path = staging_dir / "manifest.json"
        manifest_path.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

        signature_path = staging_dir / "manifest.sig"
        if signing_key is not None:
            signature = build_hmac_signature(manifest, key=signing_key, key_id=signing_key_id)
            signature_path.write_text(
                json.dumps(signature, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )

        with tarfile.open(output_path, mode="w:gz") as archive:
            archive.add(manifest_path, arcname="manifest.json")
            if signature_path.exists():
                archive.add(signature_path, arcname="manifest.sig")
            archive.add(payload_archive, arcname="payload.tar")

        return output_path
    finally:
        shutil.rmtree(staging_dir, ignore_errors=True)


def _load_signing_key(args: argparse.Namespace) -> tuple[bytes | None, str | None]:
    if args.signing_key and args.signing_key_file:
        raise SystemExit("Podaj klucz podpisu poprzez --signing-key lub --signing-key-file, nie oba naraz")
    if args.signing_key:
        return args.signing_key.encode("utf-8"), args.signing_key_id
    if args.signing_key_file:
        key_path = Path(args.signing_key_file).expanduser()
        return key_path.read_bytes().strip(), args.signing_key_id
    return None, None


def parse_arguments(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("payload", type=Path, help="Katalog z plikami aktualizacji")
    parser.add_argument("output", type=Path, help="Ścieżka do wynikowej paczki .kbot")
    parser.add_argument("--package-id", required=True, help="Identyfikator pakietu")
    parser.add_argument("--version", required=True, help="Wersja pakietu")
    parser.add_argument("--fingerprint", help="Opcjonalny fingerprint urządzenia")
    parser.add_argument("--metadata", nargs="*", default=[], help="Dodatkowe metadane w formacie klucz=wartość")
    parser.add_argument("--signing-key", help="Klucz HMAC wprost w wierszu poleceń")
    parser.add_argument("--signing-key-file", help="Plik zawierający klucz HMAC")
    parser.add_argument("--signing-key-id", help="Identyfikator klucza HMAC")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_arguments(argv)
    metadata = _parse_metadata(args.metadata) if args.metadata else {}
    signing_key, signing_key_id = _load_signing_key(args)
    try:
        build_kbot_package(
            package_id=args.package_id,
            version=args.version,
            payload_dir=args.payload,
            output_path=args.output,
            fingerprint=args.fingerprint,
            metadata=metadata,
            signing_key=signing_key,
            signing_key_id=signing_key_id,
        )
    except Exception as exc:  # pragma: no cover - logika CLI
        raise SystemExit(f"Nie udało się zbudować paczki: {exc}") from exc
    return 0


if __name__ == "__main__":  # pragma: no cover - punkt wejścia CLI
    raise SystemExit(main())


__all__ = ["build_kbot_package", "main", "parse_arguments"]
