"""Diagnozuje status dowodu TPM/secure enclave i raportuje wynik w formacie JSON."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from bot_core.security import ui_bridge


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sprawdza ważność dowodu TPM/secure enclave")
    parser.add_argument("evidence", help="Ścieżka do pliku z dowodem TPM")
    parser.add_argument(
        "--expected-fingerprint",
        dest="expected_fingerprint",
        default=None,
        help="Oczekiwany fingerprint sprzętowy (opcjonalnie)",
    )
    parser.add_argument(
        "--license-path",
        dest="license_path",
        default=None,
        help="Ścieżka do licencji – zostanie użyta do odczytu fingerprintu referencyjnego",
    )
    parser.add_argument(
        "--keyring",
        dest="keyring",
        default=None,
        help="Ścieżka do pakietu kluczy OEM do weryfikacji podpisu TPM",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    result = ui_bridge.verify_tpm_evidence(
        evidence_path=str(Path(args.evidence).expanduser()),
        expected_fingerprint=args.expected_fingerprint,
        license_path=args.license_path,
        keyring_path=args.keyring,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    status = str(result.get("status", "")).lower()
    if status in {"invalid", "missing"}:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
