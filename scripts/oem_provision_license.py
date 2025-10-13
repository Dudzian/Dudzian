"""Provisioning licencji OEM offline z podpisami HMAC."""

from __future__ import annotations

import argparse
import base64
import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Sequence

from bot_core.security.fingerprint import DeviceFingerprintGenerator, FingerprintError
from bot_core.security.rotation import RotationRegistry
from bot_core.security.signing import canonical_json_bytes

LICENSE_SIGNATURE_ALGORITHM = "HMAC-SHA384"
DEFAULT_REGISTRY = Path("var/licenses/registry.jsonl")
DEFAULT_ROTATION_LOG = Path("var/licenses/key_rotation.json")
DEFAULT_PURPOSE = "oem-license-signing"


class ProvisioningError(RuntimeError):
    """Błąd krytyczny podczas provisioning licencji."""


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fingerprint", help="Fingerprint urządzenia (domyślnie odczyt z lokalnego generatora)")
    parser.add_argument("--fingerprint-file", help="Ścieżka do pliku z fingerprintem (tekst)"
                        )
    parser.add_argument("--signing-key-path", required=True, help="Plik zawierający klucz HMAC (min 32 bajty)")
    parser.add_argument("--key-id", help="Identyfikator klucza użytego do podpisu")
    parser.add_argument("--issuer", default="OEM-Control", help="Identyfikator wystawcy licencji")
    parser.add_argument("--profile", default="paper", help="Profil pracy bota (paper/live/demo)")
    parser.add_argument("--bundle-version", default="0.0.0", help="Wersja bundla Core OEM dla której generujemy licencję")
    parser.add_argument("--feature", action="append", dest="features", default=[], help="Dodatkowe feature flagi w licencji")
    parser.add_argument("--valid-days", type=int, default=365, help="Liczba dni ważności licencji")
    parser.add_argument("--notes", help="Opcjonalna notatka np. numer zamówienia")
    parser.add_argument("--output", default=str(DEFAULT_REGISTRY), help="Rejestr JSONL do którego dopisujemy licencję")
    parser.add_argument("--emit-qr", action="store_true", help="Wypisz payload licencji jako kod QR (ASCII)")
    parser.add_argument("--usb-target", help="Ścieżka do pliku na nośniku USB, gdzie zapisujemy licencję")
    parser.add_argument("--rotation-log", default=str(DEFAULT_ROTATION_LOG), help="Plik logu rotacji klucza HMAC")
    parser.add_argument("--rotation-interval-days", type=float, default=90.0, help="Interwał rotacji klucza w dniach")
    parser.add_argument("--no-mark-rotation", action="store_true", help="Nie zapisuj informacji o rotacji po podpisaniu")
    return parser.parse_args(argv)


def _load_key(path: Path) -> bytes:
    data = path.read_bytes()
    if len(data) < 32:
        raise ProvisioningError("Klucz podpisu musi mieć co najmniej 32 bajty")
    return data


def _resolve_fingerprint(args: argparse.Namespace) -> str:
    fingerprint_provided = args.fingerprint is not None
    file_provided = args.fingerprint_file is not None

    if fingerprint_provided and file_provided:
        raise ProvisioningError("Podaj fingerprint jako string lub plik, nie oba jednocześnie")

    if file_provided:
        return Path(args.fingerprint_file).read_text(encoding="utf-8").strip()

    if fingerprint_provided:
        return args.fingerprint.strip()

    generator = DeviceFingerprintGenerator()
    return generator.generate_fingerprint()


def _validate_fingerprint(value: str) -> str:
    if not value:
        raise ProvisioningError("Fingerprint nie może być pusty")
    normalized = value.strip().upper()
    for char in normalized:
        if char not in "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-":
            raise ProvisioningError("Fingerprint zawiera niedozwolone znaki")
    return normalized


def _now() -> datetime:
    return datetime.now(timezone.utc).replace(microsecond=0)


def _build_payload(args: argparse.Namespace, fingerprint: str) -> dict[str, object]:
    issued_at = _now()
    expires_at = issued_at + timedelta(days=args.valid_days)
    features = sorted({feature.strip() for feature in args.features if feature})
    payload = {
        "schema": "core.oem.license",  # identyfikator dokumentu
        "schema_version": "1.0",
        "fingerprint": fingerprint,
        "issued_at": issued_at.isoformat().replace("+00:00", "Z"),
        "expires_at": expires_at.isoformat().replace("+00:00", "Z"),
        "profile": args.profile,
        "issuer": args.issuer,
        "bundle_version": args.bundle_version,
        "features": features,
    }
    if args.notes:
        payload["notes"] = args.notes.strip()
    return payload


def _sign_payload(payload: dict[str, object], *, key: bytes, key_id: str | None) -> dict[str, str]:
    digest = canonical_json_bytes(payload)
    signature = base64.b64encode(_hmac_sha384(key, digest)).decode("ascii")
    result = {"algorithm": LICENSE_SIGNATURE_ALGORITHM, "value": signature}
    if key_id:
        result["key_id"] = key_id
    return result


def _hmac_sha384(key: bytes, data: bytes) -> bytes:
    import hashlib
    import hmac

    return hmac.new(key, data, hashlib.sha384).digest()


def _write_registry(document: dict[str, object], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(document, ensure_ascii=False, sort_keys=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(payload + "\n")


def _maybe_emit_qr(document: dict[str, object]) -> None:
    data = json.dumps(document, sort_keys=True, ensure_ascii=False)
    try:
        import qrcode
    except ModuleNotFoundError:
        print("[WARN] Biblioteka qrcode niedostępna – drukuję base64 payloadu")
        encoded = base64.b64encode(data.encode("utf-8")).decode("ascii")
        print(encoded)
        return

    qr = qrcode.QRCode(border=1)
    qr.add_data(data)
    qr.make(fit=True)
    matrix = qr.get_matrix()
    for row in matrix:
        line = "".join("██" if cell else "  " for cell in row)
        print(line)


def _maybe_export_usb(document: dict[str, object], target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(document, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"[INFO] Licencja zapisana na nośniku: {target}")


def main(argv: Sequence[str] | None = None) -> int:
    try:
        args = _parse_args(argv)
        if args.valid_days <= 0:
            raise ProvisioningError("Ważność licencji musi być dodatnia")

        key_path = Path(args.signing_key_path)
        signing_key = _load_key(key_path)

        fingerprint = _validate_fingerprint(_resolve_fingerprint(args))
        payload = _build_payload(args, fingerprint)

        rotation_registry = None
        if args.key_id and args.rotation_log:
            rotation_registry = RotationRegistry(Path(args.rotation_log))
            status = rotation_registry.status(
                args.key_id,
                DEFAULT_PURPOSE,
                interval_days=args.rotation_interval_days,
                now=_now(),
            )
            if status.last_rotated is not None and status.is_overdue:
                raise ProvisioningError(
                    f"Klucz '{args.key_id}' jest przeterminowany – wykonaj rotację przed provisioning licencji."
                )

        signature = _sign_payload(payload, key=signing_key, key_id=args.key_id)
        document = {"payload": payload, "signature": signature}

        if rotation_registry and args.key_id and not args.no_mark_rotation:
            rotation_registry.mark_rotated(args.key_id, DEFAULT_PURPOSE, timestamp=_now())

        registry_path = Path(args.output)
        _write_registry(document, registry_path)
        print(f"[OK] Dodano licencję dla fingerprintu {fingerprint} -> {registry_path}")

        if args.emit_qr:
            _maybe_emit_qr(document)
        if args.usb_target:
            _maybe_export_usb(document, Path(args.usb_target))

        return 0
    except ProvisioningError as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 2
    except FingerprintError as exc:
        print(f"[ERROR] Fingerprint: {exc}", file=sys.stderr)
        return 3


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
