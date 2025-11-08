"""Weryfikuje pakiet aktualizacji offline i raportuje wynik w formacie JSON."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from bot_core.security.fingerprint import decode_secret
from bot_core.security.update import UpdateVerificationError, verify_update_bundle


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sprawdza integralność pakietu aktualizacji")
    parser.add_argument("package", help="Ścieżka do katalogu pakietu aktualizacji")
    parser.add_argument(
        "--manifest",
        dest="manifest",
        default="manifest.json",
        help="Nazwa pliku manifestu w katalogu pakietu (domyślnie manifest.json)",
    )
    parser.add_argument(
        "--signature",
        dest="signature",
        default=None,
        help="Opcjonalna ścieżka do pliku z podpisem manifestu",
    )
    parser.add_argument(
        "--hmac-key",
        dest="hmac_key",
        default=None,
        help="Klucz HMAC w formacie rozpoznawanym przez decode_secret (np. hex:abcd)",
    )
    return parser.parse_args(argv)


def _resolve_key(value: str | None) -> bytes | None:
    if value is None:
        return None
    candidate = Path(value)
    if candidate.exists() and candidate.is_file():
        content = candidate.read_text(encoding="utf-8").strip()
        return decode_secret(content)
    return decode_secret(value)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    package_dir = Path(args.package).expanduser()
    manifest_path = package_dir / args.manifest
    signature_path = Path(args.signature).expanduser() if args.signature else None
    key_bytes = _resolve_key(args.hmac_key)

    try:
        result = verify_update_bundle(
            manifest_path=manifest_path,
            base_dir=package_dir,
            signature_path=signature_path,
            hmac_key=key_bytes,
        )
    except UpdateVerificationError as exc:
        payload = {"status": "error", "message": str(exc)}
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return 2

    output = {
        "status": "ok" if result.is_successful else "failed",
        "signature_valid": result.signature_valid,
        "signature_checked": result.signature_checked,
        "license_ok": result.license_ok,
        "audit": result.artifact_checks,
        "errors": result.errors,
        "warnings": result.warnings,
    }
    print(json.dumps(output, ensure_ascii=False, indent=2))
    return 0 if result.is_successful else 1


if __name__ == "__main__":
    raise SystemExit(main())
