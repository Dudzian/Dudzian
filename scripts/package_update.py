"""Tworzenie podpisanych pakietów aktualizacji `.dudzianpkg`."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Mapping

from core.update.offline_updater import OFFLINE_PACKAGE_EXTENSION
from deploy.packaging.offline_package import build_offline_package


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


def _load_signing_key(args: argparse.Namespace) -> tuple[bytes | None, str | None]:
    if args.signing_key and args.signing_key_file:
        raise SystemExit(
            "Podaj klucz podpisu poprzez --signing-key lub --signing-key-file, nie oba naraz"
        )
    if args.signing_key:
        return args.signing_key.encode("utf-8"), args.signing_key_id
    if args.signing_key_file:
        key_path = Path(args.signing_key_file).expanduser()
        return key_path.read_bytes().strip(), args.signing_key_id
    return None, None


def parse_arguments(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("payload", type=Path, help="Katalog z plikami aktualizacji")
    parser.add_argument(
        "output",
        type=Path,
        help=f"Ścieżka do wynikowej paczki {OFFLINE_PACKAGE_EXTENSION}",
    )
    parser.add_argument("--package-id", required=True, help="Identyfikator pakietu")
    parser.add_argument("--version", required=True, help="Wersja pakietu")
    parser.add_argument("--fingerprint", help="Opcjonalny fingerprint urządzenia")
    parser.add_argument(
        "--metadata", nargs="*", default=[], help="Dodatkowe metadane w formacie klucz=wartość"
    )
    parser.add_argument("--signing-key", help="Klucz HMAC wprost w wierszu poleceń")
    parser.add_argument("--signing-key-file", help="Plik zawierający klucz HMAC")
    parser.add_argument("--signing-key-id", help="Identyfikator klucza HMAC")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_arguments(argv)
    metadata = _parse_metadata(args.metadata) if args.metadata else {}
    signing_key, signing_key_id = _load_signing_key(args)
    try:
        build_offline_package(
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


__all__ = ["build_offline_package", "main", "parse_arguments"]
