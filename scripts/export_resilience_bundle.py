from __future__ import annotations

import argparse
import base64
import hashlib
import hmac
import json
import logging
import os
import tarfile
from dataclasses import dataclass
from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path
from typing import Iterable, Mapping, MutableMapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
import sys

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from bot_core.resilience.bundle import ResilienceBundleBuilder
from bot_core.security.signing import canonical_json_bytes


@dataclass(frozen=True)
class _Stage6Input:
    source: Path
    target: str


def _timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _parse_metadata(entries: Sequence[str] | None) -> MutableMapping[str, str]:
    metadata: MutableMapping[str, str] = {}
    for item in entries or ():
        if "=" not in item:
            raise ValueError(f"Metadane muszą być w formacie klucz=wartość (otrzymano {item!r})")
        key, value = item.split("=", 1)
        key = key.strip()
        if not key:
            raise ValueError("Klucz metadany nie może być pusty")
        metadata[key] = value.strip()
    return metadata


def _load_signing_key(*, inline: str | None, env: str | None, path: Path | None) -> tuple[bytes | None, str | None]:
    if inline:
        key = inline.encode("utf-8")
        if len(key) < 16:
            raise ValueError("Klucz HMAC musi mieć co najmniej 16 bajtów")
        return key, None
    if env:
        raw = os.environ.get(env)
        if not raw:
            raise ValueError(f"Zmienna środowiskowa {env} jest pusta")
        key = raw.encode("utf-8")
        if len(key) < 16:
            raise ValueError("Klucz HMAC musi mieć co najmniej 16 bajtów")
        return key, env
    if path:
        resolved = path.expanduser()
        if not resolved.exists():
            raise ValueError(f"Plik klucza HMAC nie istnieje: {resolved}")
        data = resolved.read_bytes().strip()
        if len(data) < 16:
            raise ValueError("Klucz HMAC musi mieć co najmniej 16 bajtów")
        return data, str(resolved)
    return None, None


def _stage6_inputs(
    *,
    reports: Sequence[Path],
    signatures: Sequence[Path],
    includes: Sequence[Path],
) -> Iterable[_Stage6Input]:
    for path in reports:
        yield _Stage6Input(source=path, target=f"reports/{path.name}")
    for path in signatures:
        yield _Stage6Input(source=path, target=f"reports/{path.name}")
    for path in includes:
        yield _Stage6Input(source=path, target=f"extras/{path.name}")


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _stage6_manifest(
    *,
    version: str,
    files: Sequence[tuple[str, bytes]],
    metadata: Mapping[str, str],
) -> tuple[bytes, Mapping[str, object]]:
    entries = [
        {
            "path": name,
            "sha256": _sha256(payload),
            "size_bytes": len(payload),
        }
        for name, payload in files
    ]
    manifest: Mapping[str, object] = {
        "schema": "stage6.resilience.bundle.manifest",
        "schema_version": "1.0",
        "generated_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "version": version,
        "file_count": len(entries),
        "files": entries,
        "metadata": dict(metadata),
    }
    manifest_bytes = json.dumps(manifest, ensure_ascii=False, indent=2).encode("utf-8") + b"\n"
    return manifest_bytes, manifest


def _stage6_signature(
    *,
    manifest_bytes: bytes,
    key: bytes | None,
    key_id: str | None,
) -> tuple[bytes | None, Mapping[str, object] | None]:
    if key is None:
        return None, None
    payload = {
        "path": "manifest.json",
        "sha256": _sha256(manifest_bytes),
    }
    mac = hmac.new(key, canonical_json_bytes(payload), hashlib.sha256).digest()
    signature = {
        "payload": payload,
        "signature": {
            "algorithm": "HMAC-SHA256",
            "value": base64.b64encode(mac).decode("ascii"),
        },
    }
    if key_id:
        signature["signature"]["key_id"] = key_id
    signature_bytes = json.dumps(signature, ensure_ascii=False, indent=2).encode("utf-8") + b"\n"
    return signature_bytes, signature


def _write_tar(
    *,
    destination: Path,
    manifest_bytes: bytes,
    signature_bytes: bytes | None,
    payloads: Sequence[tuple[str, bytes]],
) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(destination, "w:gz") as archive:
        def _add(name: str, data: bytes) -> None:
            info = tarfile.TarInfo(name)
            info.size = len(data)
            archive.addfile(info, BytesIO(data))

        _add("./manifest.json", manifest_bytes)
        if signature_bytes is not None:
            _add("./manifest.sig", signature_bytes)
        for name, payload in payloads:
            _add(name, payload)


def _run_stage6(argv: Sequence[str]) -> int:
    parser = argparse.ArgumentParser(description="Eksport paczki odporności Stage6 (tar.gz)")
    parser.add_argument("--version", required=True, help="Wersja paczki umieszczona w manifeście")
    parser.add_argument("--report", action="append", default=[], help="Plik raportu do umieszczenia w katalogu reports/")
    parser.add_argument(
        "--signature",
        action="append",
        default=[],
        help="Podpis raportu do umieszczenia w katalogu reports/",
    )
    parser.add_argument(
        "--include",
        action="append",
        default=[],
        help="Dodatkowe pliki do katalogu extras/",
    )
    parser.add_argument(
        "--metadata",
        action="append",
        default=[],
        help="Metadane manifestu w formacie klucz=wartość (można powtórzyć)",
    )
    parser.add_argument("--output-dir", required=True, type=Path, help="Katalog wynikowy")
    parser.add_argument("--signing-key", help="Klucz HMAC przekazany inline")
    parser.add_argument("--signing-key-env", help="Zmienna środowiskowa z kluczem HMAC")
    parser.add_argument("--signing-key-file", type=Path, help="Plik z kluczem HMAC")
    parser.add_argument("--key-id", help="Identyfikator klucza HMAC")
    parser.add_argument(
        "--bundle-name",
        default="resilience-bundle",
        help="Prefiks nazwy generowanego archiwum (domyślnie resilience-bundle)",
    )
    args = parser.parse_args(argv)

    report_paths = [Path(item).expanduser().resolve() for item in args.report]
    signature_paths = [Path(item).expanduser().resolve() for item in args.signature]
    include_paths = [Path(item).expanduser().resolve() for item in args.include]
    for path in (*report_paths, *signature_paths, *include_paths):
        if not path.exists():
            raise SystemExit(f"Plik {path} nie istnieje")
        if path.is_dir():
            raise SystemExit(f"Nie można dołączyć katalogu: {path}")

    payloads: list[tuple[str, bytes]] = []
    for item in _stage6_inputs(
        reports=report_paths,
        signatures=signature_paths,
        includes=include_paths,
    ):
        payloads.append((item.target, item.source.read_bytes()))

    metadata = _parse_metadata(args.metadata)
    manifest_bytes, manifest = _stage6_manifest(
        version=args.version,
        files=payloads,
        metadata=metadata,
    )

def _write_tar(
    *,
    destination: Path,
    manifest_bytes: bytes,
    signature_bytes: bytes | None,
    payloads: Sequence[tuple[str, bytes]],
) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(destination, "w:gz") as archive:
        def _add(name: str, data: bytes) -> None:
            info = tarfile.TarInfo(name)
            info.size = len(data)
            archive.addfile(info, BytesIO(data))

        _add("./manifest.json", manifest_bytes)
        if signature_bytes is not None:
            _add("./manifest.sig", signature_bytes)
        for name, payload in payloads:
            _add(name, payload)


def _run_stage6(argv: Sequence[str]) -> int:
    parser = argparse.ArgumentParser(description="Eksport paczki odporności Stage6 (tar.gz)")
    parser.add_argument("--version", required=True, help="Wersja paczki umieszczona w manifeście")
    parser.add_argument("--report", action="append", default=[], help="Plik raportu do umieszczenia w katalogu reports/")
    parser.add_argument(
        "--signature",
        action="append",
        default=[],
        help="Podpis raportu do umieszczenia w katalogu reports/",
    )
    parser.add_argument(
        "--include",
        action="append",
        default=[],
        help="Dodatkowe pliki do katalogu extras/",
    )
    parser.add_argument(
        "--metadata",
        action="append",
        default=[],
        help="Metadane manifestu w formacie klucz=wartość (można powtórzyć)",
    )
    parser.add_argument("--output-dir", required=True, type=Path, help="Katalog wynikowy")
    parser.add_argument("--signing-key", help="Klucz HMAC przekazany inline")
    parser.add_argument("--signing-key-env", help="Zmienna środowiskowa z kluczem HMAC")
    parser.add_argument("--signing-key-file", type=Path, help="Plik z kluczem HMAC")
    parser.add_argument("--key-id", help="Identyfikator klucza HMAC")
    parser.add_argument(
        "--bundle-name",
        default="resilience-bundle",
        help="Prefiks nazwy generowanego archiwum (domyślnie resilience-bundle)",
    )
    args = parser.parse_args(argv)

    report_paths = [Path(item).expanduser().resolve() for item in args.report]
    signature_paths = [Path(item).expanduser().resolve() for item in args.signature]
    include_paths = [Path(item).expanduser().resolve() for item in args.include]
    for path in (*report_paths, *signature_paths, *include_paths):
        if not path.exists():
            raise SystemExit(f"Plik {path} nie istnieje")
        if path.is_dir():
            raise SystemExit(f"Nie można dołączyć katalogu: {path}")

    payloads: list[tuple[str, bytes]] = []
    for item in _stage6_inputs(
        reports=report_paths,
        signatures=signature_paths,
        includes=include_paths,
    ):
        payloads.append((item.target, item.source.read_bytes()))

    metadata = _parse_metadata(args.metadata)
    manifest_bytes, manifest = _stage6_manifest(
        version=args.version,
        files=payloads,
        metadata=metadata,
    )

    key_bytes, _ = _load_signing_key(
        inline=args.signing_key,
        env=args.signing_key_env,
        path=args.signing_key_file,
    )
    signature_bytes, _ = _stage6_signature(
        manifest_bytes=manifest_bytes,
        key=key_bytes,
        key_id=args.key_id,
    )

    archive_name = f"{args.bundle_name}-{args.version}.tar.gz"
    archive_path = args.output_dir.expanduser() / archive_name
    _write_tar(
        destination=archive_path,
        manifest_bytes=manifest_bytes,
        signature_bytes=signature_bytes,
        payloads=payloads,
    )

    archive_path.parent.mkdir(parents=True, exist_ok=True)
    print(archive_path)
    return 0


def _build_stage5_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Eksport paczki odporności Stage5 (zip)")
    parser.add_argument("--source", required=True, type=Path, help="Katalog źródłowy z artefaktami")
    parser.add_argument("--output-dir", required=True, type=Path, help="Katalog docelowy na paczkę")
    parser.add_argument("--bundle-name", required=True, help="Prefiks nazwy archiwum ZIP")
    parser.add_argument(
        "--include",
        action="append",
        default=None,
        help="Wzorzec glob plików do uwzględnienia (można powtarzać)",
    )
    parser.add_argument(
        "--exclude",
        action="append",
        default=None,
        help="Wzorzec glob plików do pominięcia (można powtarzać)",
    )
    parser.add_argument(
        "--metadata",
        action="append",
        default=None,
        help="Metadane manifestu w formacie klucz=wartość",
    )
    parser.add_argument("--hmac-key", dest="hmac_key", help="Klucz HMAC podany inline")
    parser.add_argument("--signing-key", dest="signing_key", help="Alias inline dla klucza HMAC")
    parser.add_argument("--hmac-key-file", dest="hmac_key_file", type=Path, help="Plik z kluczem HMAC")
    parser.add_argument(
        "--signing-key-path",
        dest="signing_key_path",
        type=Path,
        help="Alias ścieżki do pliku z kluczem HMAC",
    )
    parser.add_argument(
        "--hmac-key-env",
        dest="hmac_key_env",
        help="Zmienna środowiskowa z kluczem HMAC",
    )
    parser.add_argument(
        "--signing-key-env",
        dest="signing_key_env",
        help="Alias zmiennej środowiskowej z kluczem HMAC",
    )
    parser.add_argument(
        "--hmac-key-id",
        "--key-id",
        dest="hmac_key_id",
        help="Identyfikator klucza HMAC w podpisie",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        help="Poziom logowania (domyślnie INFO)",
    )
    return parser


def _load_stage5_signing_key(args: argparse.Namespace) -> tuple[bytes | None, str | None]:
    inline = args.hmac_key or getattr(args, "signing_key", None)
    env_name = args.hmac_key_env or getattr(args, "signing_key_env", None)
    file_path = args.hmac_key_file or getattr(args, "signing_key_path", None)
    return _load_signing_key(inline=inline, env=env_name, path=file_path)


def _run_stage5(argv: Sequence[str]) -> int:
    parser = _build_stage5_parser()
    args = parser.parse_args(argv)
    logging.basicConfig(level=args.log_level.upper(), format="%(levelname)s %(message)s")

    try:
        signing_key, key_source = _load_stage5_signing_key(args)
    except ValueError as exc:
        logging.error("Błąd odczytu klucza HMAC: %s", exc)
        return 1

    include = tuple(args.include) if args.include else None
    exclude = tuple(args.exclude) if args.exclude else None
    metadata = _parse_metadata(args.metadata)

    try:
        builder = ResilienceBundleBuilder(
            args.source,
            include=include,
            exclude=exclude,
        )
        artifacts = builder.build(
            bundle_name=args.bundle_name,
            output_dir=args.output_dir,
            metadata=metadata,
            signing_key=signing_key,
            signing_key_id=args.hmac_key_id,
        )
    except ValueError as exc:
        logging.error("Budowa paczki nie powiodła się: %s", exc)
        return 2
    except Exception:  # noqa: BLE001 - zachowujemy stack trace w logach
        logging.exception("Budowa paczki nie powiodła się")
        return 2

    if key_source:
        logging.info("Użyto klucza HMAC z %s", key_source)

def _load_stage5_signing_key(args: argparse.Namespace) -> tuple[bytes | None, str | None]:
    inline = args.hmac_key or getattr(args, "signing_key", None)
    env_name = args.hmac_key_env or getattr(args, "signing_key_env", None)
    file_path = args.hmac_key_file or getattr(args, "signing_key_path", None)
    return _load_signing_key(inline=inline, env=env_name, path=file_path)


def _run_stage5(argv: Sequence[str]) -> int:
    parser = _build_stage5_parser()
    args = parser.parse_args(argv)
    logging.basicConfig(level=args.log_level.upper(), format="%(levelname)s %(message)s")

def main(argv: Sequence[str] | None = None) -> int:
    try:
        effective = sys.argv[1:] if argv is None else argv
        return run(effective)
    except SystemExit as exc:  # pragma: no cover - przekazujemy kod wyjścia
        raise
    except Exception as exc:  # noqa: BLE001 - komunikat przyjazny CLI
        print(str(exc))
        return 2

    include = tuple(args.include) if args.include else None
    exclude = tuple(args.exclude) if args.exclude else None
    metadata = _parse_metadata(args.metadata)

    try:
        builder = ResilienceBundleBuilder(
            args.source,
            include=include,
            exclude=exclude,
        )
        artifacts = builder.build(
            bundle_name=args.bundle_name,
            output_dir=args.output_dir,
            metadata=metadata,
            signing_key=signing_key,
            signing_key_id=args.hmac_key_id,
        )
    except ValueError as exc:
        logging.error("Budowa paczki nie powiodła się: %s", exc)
        return 2
    except Exception:  # noqa: BLE001 - zachowujemy stack trace w logach
        logging.exception("Budowa paczki nie powiodła się")
        return 2

    if key_source:
        logging.info("Użyto klucza HMAC z %s", key_source)

    print(artifacts.bundle_path)
    return 0


def run(argv: Sequence[str] | None = None) -> int:
    args = list(argv or ())
    if not args or any(item.startswith("--source") or item == "--source" for item in args):
        return _run_stage5(args)
    return _run_stage6(args)


def main(argv: Sequence[str] | None = None) -> int:
    try:
        effective = sys.argv[1:] if argv is None else argv
        return run(effective)
    except SystemExit as exc:  # pragma: no cover - przekazujemy kod wyjścia
        raise
    except Exception as exc:  # noqa: BLE001 - komunikat przyjazny CLI
        print(str(exc))
        return 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
