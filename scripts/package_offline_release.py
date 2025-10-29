#!/usr/bin/env python3
"""Buduje paczkę offline z modelami i strategiami."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from core.update.installer import create_release_archive, verify_release_archive
from scripts._cli_common import parse_signing_key


def _write_manifest(path: Path, manifest: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--version", required=True, help="Wersja release'u umieszczona w manifescie")
    parser.add_argument("--output", required=True, help="Ścieżka docelowa archiwum tar.gz")
    parser.add_argument("--models-dir", default="data/models", help="Katalog ze zserializowanymi modelami")
    parser.add_argument("--strategies-dir", default="data/strategies", help="Katalog ze strategiami")
    parser.add_argument("--signing-key", help="Klucz HMAC (KEY_ID=SECRET) do podpisania manifestu")
    parser.add_argument(
        "--manifest-output",
        help="Opcjonalny plik na wygenerowany manifest (domyślnie tylko stdout)",
    )
    parser.add_argument(
        "--suppress-stdout",
        action="store_true",
        help="Nie wypisuj manifestu na stdout (przydatne w skryptach CI)",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = _build_parser()
    args = parser.parse_args(argv)

    key_id, key = parse_signing_key(args.signing_key)
    archive_path = Path(args.output)
    archive = create_release_archive(
        version=args.version,
        output_path=archive_path,
        models_dir=Path(args.models_dir) if args.models_dir else None,
        strategies_dir=Path(args.strategies_dir) if args.strategies_dir else None,
        signing_key=key,
        signing_key_id=key_id,
    )
    manifest = verify_release_archive(archive, signing_key=key)

    if args.manifest_output:
        _write_manifest(Path(args.manifest_output).expanduser(), manifest)  # pragma: no cover - ścieżki usera

    if not args.suppress_stdout:
        payload: dict[str, Any] = {
            "archive": str(archive),
            "manifest": manifest,
        }
        json.dump(payload, fp=sys.stdout, ensure_ascii=False, indent=2, sort_keys=True)
        sys.stdout.write("\n")


if __name__ == "__main__":  # pragma: no cover
    main(sys.argv[1:])
