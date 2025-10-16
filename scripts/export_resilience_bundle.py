"""Eksportuje podpisane paczki odporności Stage6 na podstawie artefaktów w ``var/``."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from bot_core.resilience.bundle import ResilienceBundleBuilder


def _parse_metadata(values: list[str] | None) -> dict[str, Any]:
    metadata: dict[str, Any] = {}
    if not values:
        return metadata
    for item in values:
        if "=" not in item:
            raise ValueError("Metadane muszą mieć format klucz=wartość")
        key, value = item.split("=", 1)
        key = key.strip()
        if not key:
            raise ValueError("Klucz metadanych nie może być pusty")
        metadata[key] = value.strip()
    return metadata


def _load_hmac_key(args: argparse.Namespace) -> tuple[bytes | None, str | None]:
    inline = args.hmac_key
    file_path = args.hmac_key_file
    env_name = args.hmac_key_env
    key_id = args.hmac_key_id

    if inline and file_path:
        raise ValueError("Podaj klucz HMAC jako wartość lub plik, nie oba jednocześnie")

    if inline:
        key = inline.encode("utf-8")
    elif file_path:
        path = Path(file_path).expanduser()
        if not path.is_file():
            raise ValueError(f"Plik klucza HMAC nie istnieje: {path}")
        if os.name != "nt":
            mode = path.stat().st_mode
            if mode & 0o077:
                raise ValueError("Plik klucza HMAC powinien mieć uprawnienia maks. 600")
        key = path.read_bytes()
    elif env_name:
        value = os.getenv(env_name)
        if not value:
            raise ValueError(f"Zmienna środowiskowa {env_name} nie zawiera klucza HMAC")
        key = value.encode("utf-8")
    else:
        return None, None

    if len(key) < 16:
        raise ValueError("Klucz HMAC powinien mieć co najmniej 16 bajtów")
    return key, key_id


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Buduje paczkę odpornościową Stage6 z podpisanym manifestem.",
    )
    parser.add_argument(
        "--source",
        required=True,
        help="Katalog źródłowy z artefaktami (np. var/audit/resilience)",
    )
    parser.add_argument(
        "--output-dir",
        default=str(REPO_ROOT / "var" / "resilience"),
        help="Katalog docelowy dla paczki i manifestu",
    )
    parser.add_argument(
        "--bundle-name",
        default="stage6-resilience",
        help="Prefiks nazwy paczki (domyślnie stage6-resilience)",
    )
    parser.add_argument(
        "--include",
        action="append",
        help="Wzorce plików do uwzględnienia (glob, można podać wielokrotnie)",
    )
    parser.add_argument(
        "--exclude",
        action="append",
        help="Wzorce plików do pominięcia (glob, można podać wielokrotnie)",
    )
    parser.add_argument(
        "--metadata",
        action="append",
        help="Dodatkowe metadane w formacie klucz=wartość (można podać wielokrotnie)",
    )
    parser.add_argument(
        "--hmac-key",
        help="Wartość klucza HMAC podpisującego manifest",
    )
    parser.add_argument(
        "--hmac-key-file",
        help="Plik zawierający klucz HMAC (UTF-8)",
    )
    parser.add_argument(
        "--hmac-key-env",
        help="Nazwa zmiennej środowiskowej z kluczem HMAC",
    )
    parser.add_argument(
        "--hmac-key-id",
        help="Opcjonalny identyfikator klucza HMAC",
    )
    return parser


def run(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        metadata = _parse_metadata(args.metadata)
        signing_key, key_id = _load_hmac_key(args)
        builder = ResilienceBundleBuilder(
            Path(args.source),
            include=args.include,
            exclude=args.exclude,
        )
        artifacts = builder.build(
            bundle_name=args.bundle_name,
            output_dir=Path(args.output_dir),
            metadata=metadata,
            signing_key=signing_key,
            signing_key_id=key_id,
        )
    except Exception as exc:  # noqa: BLE001 - CLI komunikat dla operatora
        print(f"Błąd: {exc}", file=sys.stderr)
        return 1

    summary = {
        "bundle": artifacts.bundle_path.as_posix(),
        "manifest": artifacts.manifest_path.as_posix(),
        "files": artifacts.manifest["file_count"],
        "size_bytes": artifacts.manifest["total_size_bytes"],
    }
    if artifacts.signature_path:
        summary["signature"] = artifacts.signature_path.as_posix()
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover - wejście CLI
    raise SystemExit(run())

