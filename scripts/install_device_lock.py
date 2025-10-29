"""Installs a hardware fingerprint lock for runtime bootstrap."""
from __future__ import annotations

import argparse
import json
from typing import Mapping

from bot_core.security.fingerprint import DeviceFingerprintGenerator, FingerprintError
from bot_core.security.fingerprint_lock import FingerprintLockError, write_fingerprint_lock


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        default="var/security/device_fingerprint.json",
        help="Ścieżka docelowa pliku blokady fingerprintu",
    )
    parser.add_argument(
        "--include-factors",
        action="store_true",
        help="Dołącz czynniki fingerprintu do metadanych pliku blokady (dla audytu).",
    )
    parser.add_argument(
        "--pretty",
        action="store_true",
        help="Wypisz fingerprint i ewentualne metadane na stdout w formacie JSON.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(list(argv) if argv is not None else None)
    generator = DeviceFingerprintGenerator()

    try:
        factors = generator.collect_factors()
        fingerprint = generator.generate_fingerprint(factors=factors)
    except FingerprintError as exc:
        raise SystemExit(f"Nie udało się wygenerować fingerprintu urządzenia: {exc}") from exc

    metadata: Mapping[str, object] | None = None
    if args.include_factors:
        metadata = {"factors": factors}

    try:
        lock = write_fingerprint_lock(fingerprint, path=args.output, metadata=metadata)
    except FingerprintLockError as exc:
        raise SystemExit(f"Nie udało się zapisać blokady fingerprintu: {exc}") from exc

    print(f"Zapisano blokadę fingerprintu w {lock.path}")
    if args.pretty:
        payload = {
            "fingerprint": lock.fingerprint,
            "path": str(lock.path),
            "metadata": metadata,
        }
        print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))

    return 0


if __name__ == "__main__":  # pragma: no cover - wejście CLI
    raise SystemExit(main())
