#!/usr/bin/env python3
"""CLI do pakowania, weryfikacji i instalacji offline'owych aktualizacji."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from core.update.installer import (
    create_release_archive,
    install_release_archive,
    verify_release_archive,
)
from scripts._cli_common import parse_signing_key


def _cmd_prepare(args: argparse.Namespace) -> None:
    key_id, key = parse_signing_key(args.signing_key)
    archive_path = Path(args.output)
    result_path = create_release_archive(
        version=args.version,
        output_path=archive_path,
        models_dir=Path(args.models_dir) if args.models_dir else None,
        strategies_dir=Path(args.strategies_dir) if args.strategies_dir else None,
        signing_key=key,
        signing_key_id=key_id,
    )
    manifest = verify_release_archive(result_path, signing_key=key)
    payload: dict[str, Any] = {
        "archive": str(result_path),
        "manifest": manifest,
    }
    json.dump(payload, fp=sys.stdout, ensure_ascii=False, indent=2, sort_keys=True)
    sys.stdout.write("\n")


def _cmd_verify(args: argparse.Namespace) -> None:
    _, key = parse_signing_key(args.signing_key)
    manifest = verify_release_archive(Path(args.archive), signing_key=key)
    json.dump(manifest, fp=sys.stdout, ensure_ascii=False, indent=2, sort_keys=True)
    sys.stdout.write("\n")


def _cmd_install(args: argparse.Namespace) -> None:
    _, key = parse_signing_key(args.signing_key)
    if args.require_signature and key is None:
        raise SystemExit("--require-signature wymaga podania --signing-key")
    result = install_release_archive(
        Path(args.archive),
        signing_key=key,
        models_target=Path(args.models_dir) if args.models_dir else None,
        strategies_target=Path(args.strategies_dir) if args.strategies_dir else None,
        backup_dir=Path(args.backup_dir) if args.backup_dir else None,
    )
    json.dump(result, fp=sys.stdout, ensure_ascii=False, indent=2, sort_keys=True)
    sys.stdout.write("\n")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare = subparsers.add_parser("prepare-release", help="Zbuduj paczkę offline")
    prepare.add_argument("--version", required=True, help="Wersja release'u")
    prepare.add_argument("--output", required=True, help="Ścieżka docelowa archiwum (tar.gz)")
    prepare.add_argument("--models-dir", default="data/models", help="Źródło modeli")
    prepare.add_argument("--strategies-dir", default="data/strategies", help="Źródło strategii")
    prepare.add_argument("--signing-key", help="Klucz HMAC (KEY_ID=SECRET) do podpisania manifestu")

    verify = subparsers.add_parser("verify-release", help="Zweryfikuj podpis i sumy kontrolne paczki")
    verify.add_argument("--archive", required=True, help="Ścieżka do archiwum tar.gz")
    verify.add_argument("--signing-key", help="Klucz HMAC (KEY_ID=SECRET) do weryfikacji manifestu")

    install = subparsers.add_parser("install-release", help="Zainstaluj paczkę offline")
    install.add_argument("--archive", required=True, help="Ścieżka do archiwum tar.gz")
    install.add_argument("--models-dir", default="data/models", help="Katalog docelowy modeli")
    install.add_argument("--strategies-dir", default="data/strategies", help="Katalog docelowy strategii")
    install.add_argument(
        "--backup-dir",
        default="var/offline_updates/backups",
        help="Katalog na kopie zapasowe aktualnych modeli/strategii",
    )
    install.add_argument("--signing-key", help="Klucz HMAC (KEY_ID=SECRET) do weryfikacji manifestu")
    install.add_argument(
        "--require-signature",
        action="store_true",
        help="Wymagaj poprawnego podpisu manifestu przed instalacją",
    )

    return parser


def main(argv: list[str] | None = None) -> None:
    parser = _build_parser()
    args = parser.parse_args(argv)
    if args.command == "prepare-release":
        _cmd_prepare(args)
    elif args.command == "verify-release":
        _cmd_verify(args)
    elif args.command == "install-release":
        _cmd_install(args)
    else:  # pragma: no cover - argparse gwarantuje poprawność
        parser.error(f"Nieznane polecenie: {args.command}")


if __name__ == "__main__":  # pragma: no cover
    import sys

    main(sys.argv[1:])
