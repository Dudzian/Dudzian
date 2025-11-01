from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from deploy.packaging.offline_distribution import build_offline_distribution
from scripts._cli_common import parse_signing_key


def _parse_metadata(entries: list[str]) -> dict[str, Any]:
    metadata: dict[str, Any] = {}
    for item in entries:
        if "=" not in item:
            raise SystemExit(f"Metadana musi mieć format klucz=wartość (otrzymano: {item!r})")
        key, value = item.split("=", 1)
        metadata[key.strip()] = value.strip()
    return metadata


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Buduje paczkę .kbot do dystrybucji offline.")
    parser.add_argument("--payload", required=True, help="Katalog z artefaktami aktualizacji")
    parser.add_argument("--output", required=True, help="Ścieżka docelowa paczki .kbot")
    parser.add_argument("--package-id", required=True, help="Identyfikator pakietu aktualizacji")
    parser.add_argument("--version", required=True, help="Wersja pakietu")
    parser.add_argument("--fingerprint", help="Opcjonalny fingerprint przypisany do pakietu")
    parser.add_argument("--metadata", nargs="*", default=[], help="Dodatkowe metadane w formacie klucz=wartość")
    parser.add_argument("--signing-key", help="Klucz HMAC w formacie KEY_ID=SECRET")
    parser.add_argument("--manifest-output", help="Ścieżka do zapisu manifestu JSON")
    parser.add_argument("--rotation-registry", help="Plik rejestru rotacji fingerprintów")
    parser.add_argument(
        "--rotation-purpose",
        default="offline_distribution",
        help="Nazwa wpisu w rejestrze rotacji (domyślnie offline_distribution)",
    )
    parser.add_argument(
        "--suppress-stdout",
        action="store_true",
        help="Nie wypisuj podsumowania na stdout (przydatne w CI)",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    metadata = _parse_metadata(args.metadata) if args.metadata else {}
    key_id, signing_key = parse_signing_key(args.signing_key)

    result = build_offline_distribution(
        package_id=args.package_id,
        version=args.version,
        payload_dir=Path(args.payload),
        output_path=Path(args.output),
        fingerprint=args.fingerprint,
        metadata=metadata or None,
        signing_key=signing_key,
        signing_key_id=key_id,
        rotation_registry_path=Path(args.rotation_registry) if args.rotation_registry else None,
        rotation_purpose=args.rotation_purpose,
        manifest_output=Path(args.manifest_output) if args.manifest_output else None,
    )

    if not args.suppress_stdout:
        json.dump(result.summary, sys.stdout, ensure_ascii=False, indent=2, sort_keys=True)
        sys.stdout.write("\n")

    return 0


if __name__ == "__main__":  # pragma: no cover - punkt wejścia CLI
    raise SystemExit(main())


__all__ = ["main"]
