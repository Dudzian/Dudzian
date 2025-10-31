"""Generuje pakiet aktualizacyjny wraz z łatką różnicową i walidacją integralności."""

from __future__ import annotations

import argparse
import json
import shutil
import tarfile
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

from bot_core.security.fingerprint import decode_secret
from bot_core.security.update import verify_update_bundle
from scripts import update_package


def _parse_signing_key(value: str | None) -> tuple[str | None, str | None]:
    if not value:
        return None, None
    if "=" in value:
        key_id, secret = value.split("=", 1)
        return key_id.strip() or None, secret.strip()
    return None, value.strip()


@dataclass(slots=True)
class DiffResult:
    diff_archive: Path | None
    deleted_files: list[str]


DEFAULT_OUTPUT = Path("var/dist/updates")


def _hash_file(path: Path, algorithm: str = "sha256") -> str:
    import hashlib

    digest = hashlib.new(algorithm)
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _collect_hashes(root: Path) -> dict[str, str]:
    hashes: dict[str, str] = {}
    for entry in sorted(root.rglob("*")):
        if entry.is_file():
            hashes[entry.relative_to(root).as_posix()] = _hash_file(entry)
    return hashes


def _make_tar(source: Path, target: Path) -> Path:
    if target.exists():
        target.unlink()
    with tarfile.open(target, "w:gz") as archive:
        archive.add(source, arcname=source.name)
    return target


def _build_diff_archive(base_dir: Path, target_dir: Path, output_path: Path) -> DiffResult:
    base_hashes = _collect_hashes(base_dir)
    target_hashes = _collect_hashes(target_dir)

    changed: list[str] = []
    for rel_path, digest in target_hashes.items():
        if base_hashes.get(rel_path) != digest:
            changed.append(rel_path)

    deleted = sorted(set(base_hashes) - set(target_hashes))

    if not changed and not deleted:
        return DiffResult(diff_archive=None, deleted_files=[])

    temp_dir = Path(tempfile.mkdtemp(prefix="update_diff_"))
    try:
        if changed:
            for rel_path in changed:
                source_path = target_dir / rel_path
                destination = temp_dir / rel_path
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source_path, destination)
        if deleted:
            deleted_file = temp_dir / "deleted_files.json"
            deleted_file.write_text(json.dumps(sorted(deleted), ensure_ascii=False, indent=2), encoding="utf-8")
        diff_archive = _make_tar(temp_dir, output_path)
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)

    return DiffResult(diff_archive=diff_archive, deleted_files=deleted)


def create_update_package(
    *,
    base_dir: Path | None,
    target_dir: Path,
    output_dir: Path,
    package_id: str,
    version: str,
    platform: str,
    runtime: str,
    base_id: str | None = None,
    signing_key: str | None = None,
    metadata: dict[str, object] | None = None,
) -> Path:
    target_dir = target_dir.expanduser().resolve()
    if not target_dir.exists():
        raise FileNotFoundError(target_dir)

    base_dir = base_dir.expanduser().resolve() if base_dir else None
    if base_dir is not None and not base_dir.exists():
        raise FileNotFoundError(base_dir)

    output_dir = output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    package_root = output_dir / f"{package_id}-{version}"
    package_root.mkdir(parents=True, exist_ok=True)

    staging_dir = Path(tempfile.mkdtemp(prefix="update_package_stage_"))
    try:
        full_archive = _make_tar(target_dir, staging_dir / f"{package_id}-{version}-full.tar.gz")

        diff_result = DiffResult(diff_archive=None, deleted_files=[])
        if base_dir is not None:
            diff_result = _build_diff_archive(
                base_dir,
                target_dir,
                staging_dir / f"{package_id}-{version}-diff.tar.gz",
            )

        manifest_metadata: dict[str, object] = metadata.copy() if metadata else {}
        if diff_result.deleted_files:
            manifest_metadata.setdefault("deleted_files", diff_result.deleted_files)

        key_id, raw_secret = _parse_signing_key(signing_key)

        build_args = argparse.Namespace(
            output_dir=str(package_root),
            package_id=package_id,
            version=version,
            platform=platform,
            runtime=runtime,
            payload=str(full_archive),
            diff=str(diff_result.diff_archive) if diff_result.diff_archive else None,
            base_id=base_id,
            integrity_manifest=None,
            metadata=json.dumps(manifest_metadata, ensure_ascii=False) if manifest_metadata else None,
            key=raw_secret,
            key_id=key_id,
        )

        update_package.build_package(build_args)

        verify_key: bytes | None = None
        if raw_secret:
            verify_key = decode_secret(raw_secret)

        result = verify_update_bundle(
            manifest_path=package_root / "manifest.json",
            base_dir=package_root,
            signature_path=None,
            hmac_key=verify_key,
            license_result=None,
        )
        if not result.is_successful:
            raise RuntimeError(f"Weryfikacja pakietu aktualizacji nie powiodła się: {result.errors}")
    finally:
        shutil.rmtree(staging_dir, ignore_errors=True)

    return package_root


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target-dir", required=True, help="Katalog docelowej wersji instalacji")
    parser.add_argument("--base-dir", help="Opcjonalna poprzednia instalacja do wygenerowania diffu")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT), help="Katalog na pakiety aktualizacji")
    parser.add_argument("--package-id", required=True)
    parser.add_argument("--version", required=True)
    parser.add_argument("--platform", required=True)
    parser.add_argument("--runtime", required=True)
    parser.add_argument("--base-id", help="Identyfikator wersji bazowej dla łatki diff")
    parser.add_argument("--signing-key", help="Klucz HMAC do podpisu manifestu (KEY_ID=wartość)")
    parser.add_argument("--metadata", action="append", default=[], help="Dodatkowe metadane w formacie klucz=wartość (JSON)")
    return parser


def _parse_metadata(entries: Iterable[str]) -> dict[str, object]:
    result: dict[str, object] = {}
    for entry in entries:
        key, sep, raw_value = entry.partition("=")
        if sep != "=" or not key:
            raise SystemExit(f"Metadane muszą mieć format klucz=wartość: {entry}")
        try:
            value = json.loads(raw_value)
        except json.JSONDecodeError:
            value = raw_value
        result[key] = value
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)

    package_root = create_update_package(
        base_dir=Path(args.base_dir) if args.base_dir else None,
        target_dir=Path(args.target_dir),
        output_dir=Path(args.output_dir),
        package_id=args.package_id,
        version=args.version,
        platform=args.platform,
        runtime=args.runtime,
        base_id=args.base_id,
        signing_key=args.signing_key,
        metadata=_parse_metadata(args.metadata),
    )
    print(json.dumps({"status": "ok", "package": str(package_root)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":  # pragma: no cover - wejście CLI
    raise SystemExit(main())
