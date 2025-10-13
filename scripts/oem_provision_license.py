"""CLI do provisioning OEM – generacja i walidacja licencji JSONL."""

from __future__ import annotations

import argparse
import base64
import binascii
import json
import sys
import re
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from bot_core.security.fingerprint import decode_secret
from bot_core.security.rotation import RotationRegistry
from bot_core.security.signing import build_hmac_signature

_DEFAULT_LICENSE_ROTATION = "var/licenses/license_rotation.json"
_DEFAULT_LICENSE_REGISTRY = "var/licenses/registry.jsonl"
_DEFAULT_LICENSE_PURPOSE = "oem-license"
_ALLOWED_CPU_PREFIXES = ("intel", "amd", "apple", "arm", "qualcomm", "ibm")


class ProvisioningError(RuntimeError):
    """Błąd domenowy provisioning OEM."""


def _isoformat(timestamp: datetime | None = None) -> str:
    current = timestamp or datetime.now(timezone.utc)
    return current.replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _parse_key_entries(entries: Sequence[str] | None) -> dict[str, bytes]:
    result: dict[str, bytes] = {}
    for raw in entries or ():
        if "=" not in raw:
            raise ProvisioningError("Klucz musi mieć format key_id=sekret.")
        key_id, value = raw.split("=", 1)
        key_id = key_id.strip()
        if not key_id:
            raise ProvisioningError("Identyfikator klucza nie może być pusty.")
        result[key_id] = decode_secret(value)
    return result


def _maybe_base64_decode(raw: str) -> str:
    text = raw.strip()
    if not text:
        return raw
    try:
        decoded = base64.b64decode(text, validate=True)
    except (ValueError, binascii.Error):
        return raw
    try:
        decoded_text = decoded.decode("utf-8")
    except UnicodeDecodeError:
        return raw
    if decoded_text.strip().startswith("{"):
        return decoded_text
    return raw


def _load_json_payload(raw: str) -> dict[str, Any]:
    try:
        return json.loads(raw)
    except json.JSONDecodeError as exc:  # pragma: no cover - defensywny log
        raise ProvisioningError(f"Nie udało się zdekodować fingerprintu JSON: {exc}") from exc


def load_fingerprint(source: str, mode: str) -> dict[str, Any]:
    path = Path(source)
    if mode == "usb" and not path.exists():
        raise ProvisioningError(f"W trybie USB oczekiwano pliku z fingerprintem: {source}")

    if path.exists():
        raw_text = path.read_text(encoding="utf-8")
    else:
        raw_text = source

    decoded = _maybe_base64_decode(raw_text)
    return _load_json_payload(decoded)


def verify_fingerprint_signature(
    fingerprint: Mapping[str, Any],
    keys: Mapping[str, bytes],
) -> None:
    payload = fingerprint.get("payload")
    signature = fingerprint.get("signature")
    if not isinstance(payload, Mapping) or not isinstance(signature, Mapping):
        raise ProvisioningError("Fingerprint nie zawiera struktur payload/signature.")

    key_id = signature.get("key_id")
    if not isinstance(key_id, str) or not key_id:
        raise ProvisioningError("Fingerprint nie zawiera identyfikatora klucza HMAC.")

    try:
        key = keys[key_id]
    except KeyError as exc:
        raise ProvisioningError(
            f"Brak klucza HMAC '{key_id}' potrzebnego do weryfikacji fingerprintu."
        ) from exc

    expected = build_hmac_signature(payload, key=key, key_id=key_id)
    if expected["value"] != signature.get("value"):
        raise ProvisioningError("Sygnatura fingerprintu nie zgadza się z oczekiwanym HMAC.")


@dataclass(slots=True)
class OemPolicy:
    allowed_cpu_prefixes: tuple[str, ...] = _ALLOWED_CPU_PREFIXES
    require_mac: bool = True
    require_tpm_usb: bool = True
    require_tpm_qr: bool = False
    require_dongle_usb: bool = True


def evaluate_policy(payload: Mapping[str, Any], mode: str, policy: OemPolicy) -> dict[str, list[str]]:
    components = payload.get("components")
    if not isinstance(components, Mapping):
        raise ProvisioningError("Fingerprint payload jest niepełny (brak sekcji components).")

    errors: list[str] = []
    warnings: list[str] = []

    cpu_entry = components.get("cpu")
    if isinstance(cpu_entry, Mapping):
        normalized = str(cpu_entry.get("normalized", "")).strip()
        vendor_raw = normalized.split(" ")[0]
        vendor = re.sub(r"[^a-z0-9]", "", vendor_raw)
        if vendor and not any(vendor.startswith(prefix) for prefix in policy.allowed_cpu_prefixes):
            errors.append(
                f"Vendor CPU '{vendor}' nie znajduje się na liście dozwolonych ({', '.join(policy.allowed_cpu_prefixes)})."
            )
    else:
        errors.append("Fingerprint nie zawiera identyfikatora CPU.")

    mac_entry = components.get("mac")
    if policy.require_mac and not isinstance(mac_entry, Mapping):
        errors.append("Fingerprint nie zawiera poprawnego adresu MAC.")

    tpm_entry = components.get("tpm")
    if mode == "usb" and policy.require_tpm_usb and not isinstance(tpm_entry, Mapping):
        errors.append("Tryb USB wymaga aktywnego modułu TPM.")
    if mode == "qr" and policy.require_tpm_qr and not isinstance(tpm_entry, Mapping):
        warnings.append("Fingerprint QR nie zawiera TPM – OEM zweryfikuje manualnie.")

    dongle_entry = components.get("dongle")
    if mode == "usb" and policy.require_dongle_usb and not isinstance(dongle_entry, Mapping):
        errors.append("Tryb USB wymaga obecności klucza sprzętowego (dongle).")

    return {"errors": errors, "warnings": warnings}


def build_license_payload(
    fingerprint: Mapping[str, Any],
    mode: str,
    policy: OemPolicy,
    *,
    issued_at: datetime | None = None,
) -> dict[str, Any]:
    payload = fingerprint.get("payload")
    if not isinstance(payload, Mapping):
        raise ProvisioningError("Fingerprint nie zawiera sekcji payload.")

    evaluation = evaluate_policy(payload, mode, policy)
    if evaluation["errors"]:
        raise ProvisioningError("Fingerprint nie spełnia wymagań OEM: " + "; ".join(evaluation["errors"]))

    issued = issued_at or datetime.now(timezone.utc)
    license_id = uuid.uuid4().hex

    return {
        "license_id": license_id,
        "issued_at": _isoformat(issued),
        "mode": mode,
        "fingerprint": payload.get("fingerprint"),
        "fingerprint_payload": payload,
        "fingerprint_signature": fingerprint.get("signature"),
        "policy": {
            "allowed_cpu_prefixes": policy.allowed_cpu_prefixes,
            "require_mac": policy.require_mac,
            "require_tpm_usb": policy.require_tpm_usb,
            "require_tpm_qr": policy.require_tpm_qr,
            "require_dongle_usb": policy.require_dongle_usb,
        },
        "policy_evaluation": evaluation,
    }


def append_jsonl(path: Path, record: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        json.dump(record, handle, ensure_ascii=False)
        handle.write("\n")


def write_usb_artifact(target: str, record: Mapping[str, Any]) -> Path:
    directory = Path(target)
    directory.mkdir(parents=True, exist_ok=True)
    license_id = record.get("payload", {}).get("license_id", "license")
    path = directory / f"{license_id}.json"
    path.write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return path


def write_qr_payload(target: str | None, record: Mapping[str, Any]) -> str:
    payload = json.dumps(record, ensure_ascii=False, separators=(",", ":"))
    encoded = base64.b64encode(payload.encode("utf-8")).decode("ascii")
    if target:
        Path(target).write_text(encoded + "\n", encoding="utf-8")
    return encoded


def _build_license_provider(
    keys: Mapping[str, bytes],
    *,
    rotation_log: str,
    purpose: str = _DEFAULT_LICENSE_PURPOSE,
    interval_days: float = 180.0,
):
    from bot_core.security.fingerprint import RotatingHmacKeyProvider

    registry = RotationRegistry(rotation_log)
    return RotatingHmacKeyProvider(keys, registry, purpose=purpose, interval_days=interval_days)


def sign_license(
    payload: Mapping[str, Any],
    provider,
    *,
    issued_at: datetime | None = None,
) -> dict[str, Any]:
    timestamp = issued_at or datetime.now(timezone.utc)
    key_id, signature = provider.sign(payload, now=timestamp)
    return {"payload": dict(payload), "signature": signature}


def validate_registry(
    registry_path: Path,
    license_keys: Mapping[str, bytes],
    fingerprint_keys: Mapping[str, bytes] | None = None,
) -> list[str]:
    if not registry_path.exists():
        return []

    errors: list[str] = []
    for index, line in enumerate(registry_path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            errors.append(f"Linia {index}: nieprawidłowy JSON.")
            continue

        payload = record.get("payload")
        signature = record.get("signature")
        if not isinstance(payload, Mapping) or not isinstance(signature, Mapping):
            errors.append(f"Linia {index}: brak struktury payload/signature.")
            continue

        key_id = signature.get("key_id")
        if not isinstance(key_id, str) or key_id not in license_keys:
            errors.append(f"Linia {index}: brak klucza licencji '{key_id}'.")
            continue

        expected = build_hmac_signature(payload, key=license_keys[key_id], key_id=key_id)
        if expected["value"] != signature.get("value"):
            errors.append(f"Linia {index}: podpis licencji niezgodny z HMAC.")

        if fingerprint_keys:
            fp_sig = payload.get("fingerprint_signature")
            fp_payload = payload.get("fingerprint_payload")
            if isinstance(fp_sig, Mapping) and isinstance(fp_payload, Mapping):
                fp_key = fp_sig.get("key_id")
                if isinstance(fp_key, str) and fp_key in fingerprint_keys:
                    expected_fp = build_hmac_signature(
                        fp_payload,
                        key=fingerprint_keys[fp_key],
                        key_id=fp_key,
                    )
                    if expected_fp["value"] != fp_sig.get("value"):
                        errors.append(f"Linia {index}: podpis fingerprintu niepoprawny.")
                else:
                    errors.append(f"Linia {index}: brak klucza fingerprintu '{fp_key}'.")
    return errors


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Provisioning OEM licencji JSONL.")
    parser.add_argument("--fingerprint", help="Fingerprint JSON/BASE64 lub ścieżka do pliku.")
    parser.add_argument(
        "--mode",
        choices=["qr", "usb"],
        default="qr",
        help="Tryb dostarczenia fingerprintu (QR/USB).",
    )
    parser.add_argument(
        "--license-key",
        action="append",
        dest="license_keys",
        help="Klucz HMAC licencji w formacie key_id=sekret (wymagane).",
    )
    parser.add_argument(
        "--license-rotation-log",
        default=_DEFAULT_LICENSE_ROTATION,
        help="Ścieżka do rejestru rotacji kluczy licencyjnych.",
    )
    parser.add_argument(
        "--fingerprint-key",
        action="append",
        dest="fingerprint_keys",
        help="Klucz HMAC fingerprintu w formacie key_id=sekret (do walidacji).",
    )
    parser.add_argument(
        "--registry",
        default=_DEFAULT_LICENSE_REGISTRY,
        help="Rejestr licencji JSONL (output).",
    )
    parser.add_argument(
        "--qr-output",
        help="Opcjonalny plik z payloadem zakodowanym do QR (BASE64).",
    )
    parser.add_argument(
        "--usb-output",
        help="Katalog docelowy na artefakt licencyjny w trybie USB.",
    )
    parser.add_argument(
        "--validate-registry",
        action="store_true",
        help="Tylko waliduje istniejący rejestr licencji i kończy działanie.",
    )
    parser.add_argument(
        "--license-interval-days",
        type=float,
        default=180.0,
        help="Interwał rotacji kluczy licencyjnych w dniach.",
    )
    parser.add_argument(
        "--license-purpose",
        default=_DEFAULT_LICENSE_PURPOSE,
        help="Cel wpisu w rejestrze rotacji kluczy licencyjnych.",
    )
    return parser.parse_args(argv)


def _run_validation(args: argparse.Namespace) -> int:
    license_keys = _parse_key_entries(args.license_keys)
    if not license_keys:
        raise ProvisioningError("Do walidacji rejestru wymagane są klucze licencyjne.")

    fingerprint_keys = _parse_key_entries(args.fingerprint_keys)
    registry_path = Path(args.registry)
    errors = validate_registry(registry_path, license_keys, fingerprint_keys)
    if errors:
        for msg in errors:
            print(msg, file=sys.stderr)
        return 1
    print(f"Rejestr {registry_path} – wszystkie podpisy prawidłowe.")
    return 0


def _run_provision(args: argparse.Namespace) -> int:
    if not args.fingerprint:
        raise ProvisioningError("Brak fingerprintu do przetworzenia.")

    license_keys = _parse_key_entries(args.license_keys)
    if not license_keys:
        raise ProvisioningError("Wymagany jest co najmniej jeden klucz licencyjny (--license-key).")

    fingerprint_keys = _parse_key_entries(args.fingerprint_keys)
    fingerprint = load_fingerprint(args.fingerprint, args.mode)
    if fingerprint_keys:
        verify_fingerprint_signature(fingerprint, fingerprint_keys)

    policy = OemPolicy()
    payload = build_license_payload(fingerprint, args.mode, policy)
    provider = _build_license_provider(
        license_keys,
        rotation_log=args.license_rotation_log,
        purpose=args.license_purpose,
        interval_days=args.license_interval_days,
    )
    record = sign_license(payload, provider)

    registry_path = Path(args.registry)
    append_jsonl(registry_path, record)

    if args.mode == "usb" and args.usb_output:
        target = write_usb_artifact(args.usb_output, record)
        print(f"Zapisano licencję do {target}")

    encoded = write_qr_payload(args.qr_output if args.mode == "qr" else None, record)
    if args.mode == "qr":
        print("Payload QR (BASE64):")
        print(encoded)

    print(f"Dodano licencję {record['payload']['license_id']} do rejestru {registry_path}.")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    try:
        if args.validate_registry:
            return _run_validation(args)
        return _run_provision(args)
    except ProvisioningError as exc:
        print(f"Błąd provisioning: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":  # pragma: no cover - CLI entrypoint
    raise SystemExit(main())

