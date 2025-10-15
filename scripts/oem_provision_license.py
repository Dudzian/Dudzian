"""Provisioning licencji OEM – generacja, podpis HMAC, walidacja rejestru i obsługa fingerprintów JSON/Base32."""
from __future__ import annotations

import argparse
import base64
import binascii
import json
import re
import sys
import uuid
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

# --- zależności bot_core -----------------------------------------------------
from bot_core.security.fingerprint import DeviceFingerprintGenerator, FingerprintError
from bot_core.security.rotation import RotationRegistry
from bot_core.security.signing import build_hmac_signature, canonical_json_bytes

# --- stałe -------------------------------------------------------------------
LICENSE_SIGNATURE_ALGORITHM = "HMAC-SHA384"
DEFAULT_REGISTRY = Path("var/licenses/registry.jsonl")
DEFAULT_ROTATION_LOG = Path("var/licenses/key_rotation.json")
DEFAULT_PURPOSE = "oem-license-signing"
_ALLOWED_CPU_PREFIXES = ("intel", "amd", "apple", "arm", "qualcomm", "ibm")


# --- wyjątki -----------------------------------------------------------------
class ProvisioningError(RuntimeError):
    """Błąd krytyczny podczas provisioning licencji."""


# --- narzędzia ogólne --------------------------------------------------------
def _iso_now() -> datetime:
    return datetime.now(timezone.utc).replace(microsecond=0)


def _isoformat(ts: datetime) -> str:
    return ts.isoformat().replace("+00:00", "Z")


def _decode_secret(value: str) -> bytes:
    """Obsługuje prefixy 'hex:' i 'base64:'; inaczej traktuje jako UTF-8."""
    text = value.strip()
    if text.startswith("hex:"):
        return bytes.fromhex(text[4:])
    if text.startswith("base64:"):
        return base64.b64decode(text[7:])
    return text.encode("utf-8")


def _parse_key_entries(entries: Sequence[str] | None) -> dict[str, bytes]:
    result: dict[str, bytes] = {}
    for raw in entries or ():
        if "=" not in raw:
            raise ProvisioningError("Klucz musi mieć format key_id=sekret.")
        key_id, value = raw.split("=", 1)
        key_id = key_id.strip()
        if not key_id:
            raise ProvisioningError("Identyfikator klucza nie może być pusty.")
        result[key_id] = _decode_secret(value)
    return result


# --- fingerprint: wejście i walidacja ----------------------------------------
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
    return decoded_text if decoded_text.strip().startswith("{") else raw


def _load_text_or_file(value: str) -> str:
    path = Path(value)
    if path.exists():
        return path.read_text(encoding="utf-8")
    return value


def load_fingerprint_maybe_json(source: str) -> dict | str:
    """Zwraca fingerprint jako dict (JSON) lub string (prosty)."""
    raw = _load_text_or_file(source)
    candidate = _maybe_base64_decode(raw)
    try:
        parsed = json.loads(candidate)
        if isinstance(parsed, dict) and "payload" in parsed and "signature" in parsed:
            return parsed
    except json.JSONDecodeError:
        pass
    return raw.strip()


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
        raise ProvisioningError(f"Brak klucza HMAC '{key_id}' do weryfikacji fingerprintu.") from exc

    expected = build_hmac_signature(payload, key=key, key_id=key_id)
    if expected["value"] != signature.get("value"):
        raise ProvisioningError("Sygnatura fingerprintu niezgodna z oczekiwanym HMAC.")


def _validate_simple_fingerprint(text: str) -> str:
    norm = text.strip().upper()
    if not norm:
        raise ProvisioningError("Fingerprint nie może być pusty.")
    allowed = set("ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-")
    if any(ch not in allowed for ch in norm):
        raise ProvisioningError("Fingerprint (tryb prosty) zawiera niedozwolone znaki.")
    return norm


# --- polityka OEM dla fingerprintu JSON --------------------------------------
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
                f"Vendor CPU '{vendor}' nie jest dozwolony ({', '.join(policy.allowed_cpu_prefixes)})."
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
        warnings.append("Fingerprint QR nie zawiera TPM – wymagany będzie przegląd manualny.")

    dongle_entry = components.get("dongle")
    if mode == "usb" and policy.require_dongle_usb and not isinstance(dongle_entry, Mapping):
        errors.append("Tryb USB wymaga obecności klucza sprzętowego (dongle).")

    return {"errors": errors, "warnings": warnings}


# --- budowa payloadu licencji ------------------------------------------------
def build_license_payload_simple(
    *,
    fingerprint_text: str,
    issuer: str,
    profile: str,
    bundle_version: str,
    features: Sequence[str],
    notes: str | None,
    valid_days: int,
) -> dict[str, Any]:
    issued = _iso_now()
    expires = issued + timedelta(days=valid_days)
    features_norm = sorted({f.strip() for f in features if f and f.strip()})
    payload = {
        "schema": "core.oem.license",
        "schema_version": "1.0",
        "fingerprint": fingerprint_text,
        "issued_at": _isoformat(issued),
        "expires_at": _isoformat(expires),
        "profile": profile,
        "issuer": issuer,
        "bundle_version": bundle_version,
        "features": features_norm,
    }
    if notes:
        payload["notes"] = notes.strip()
    return payload


def build_license_payload_rich(
    *,
    fingerprint_doc: Mapping[str, Any],
    mode: str,
    issuer: str,
    profile: str,
    bundle_version: str,
    features: Sequence[str],
    notes: str | None,
    valid_days: int,
    policy: OemPolicy,
) -> dict[str, Any]:
    payload = fingerprint_doc.get("payload")
    signature = fingerprint_doc.get("signature")
    if not isinstance(payload, Mapping) or not isinstance(signature, Mapping):
        raise ProvisioningError("Fingerprint JSON jest niekompletny.")

    evaluation = evaluate_policy(payload, mode, policy)
    if evaluation["errors"]:
        raise ProvisioningError("Fingerprint nie spełnia wymagań OEM: " + "; ".join(evaluation["errors"]))

    issued = _iso_now()
    expires = issued + timedelta(days=valid_days)
    features_norm = sorted({f.strip() for f in features if f and f.strip()})

    return {
        "schema": "core.oem.license",
        "schema_version": "1.0",
        "license_id": uuid.uuid4().hex,
        "issued_at": _isoformat(issued),
        "expires_at": _isoformat(expires),
        "mode": mode,
        "issuer": issuer,
        "profile": profile,
        "bundle_version": bundle_version,
        "features": features_norm,
        "notes": notes.strip() if notes else None,
        "fingerprint": payload.get("fingerprint"),
        "fingerprint_payload": payload,
        "fingerprint_signature": signature,
        "policy": {
            "allowed_cpu_prefixes": policy.allowed_cpu_prefixes,
            "require_mac": policy.require_mac,
            "require_tpm_usb": policy.require_tpm_usb,
            "require_tpm_qr": policy.require_tpm_qr,
            "require_dongle_usb": policy.require_dongle_usb,
        },
        "policy_evaluation": evaluation,
    }


# --- podpis i rejestr --------------------------------------------------------
def sign_with_single_key(payload: Mapping[str, Any], *, key_bytes: bytes, key_id: str | None) -> dict[str, Any]:
    digest = canonical_json_bytes(payload)
    import hashlib, hmac  # lokalnie – dla zgodności z HMAC-SHA384
    value = base64.b64encode(hmac.new(key_bytes, digest, hashlib.sha384).digest()).decode("ascii")
    sig = {"algorithm": LICENSE_SIGNATURE_ALGORITHM, "value": value}
    if key_id:
        sig["key_id"] = key_id
    return {"payload": dict(payload), "signature": sig}


def sign_with_rotating_keys(
    payload: Mapping[str, Any],
    *,
    keys: Mapping[str, bytes],
    rotation_log: Path,
    purpose: str,
    interval_days: float,
    mark_rotation: bool,
) -> dict[str, Any]:
    # Lokalny provider: wybieramy najstarszy / wymagający rotacji
    registry = RotationRegistry(rotation_log)
    now = _iso_now()
    statuses = {kid: registry.status(kid, purpose, interval_days=interval_days, now=now) for kid in keys}

    # priorytety: overdue > due > OK; w każdej grupie najstarszy
    def _prio(st):
        if not st.is_due and not st.is_overdue:
            p = 0
        elif st.is_due and not st.is_overdue:
            p = 1
        else:
            p = 2
        return (2 - p, st.last_rotated.timestamp() if st.last_rotated else 0.0)

    selected = sorted(statuses.items(), key=lambda kv: _prio(kv[1]))[-1][0]
    signature = build_hmac_signature(payload, key=keys[selected], key_id=selected)
    if mark_rotation:
        registry.mark_rotated(selected, purpose, timestamp=now)
    return {"payload": dict(payload), "signature": signature}


def append_jsonl(path: Path, record: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        json.dump(record, handle, ensure_ascii=False, sort_keys=True)
        handle.write("\n")


def write_usb_artifact(target: str, record: Mapping[str, Any]) -> Path:
    p = Path(target)
    p.parent.mkdir(parents=True, exist_ok=True)
    if p.suffix.lower() == ".json":
        path = p
    else:
        license_id = record.get("payload", {}).get("license_id") or uuid.uuid4().hex
        path = p / f"{license_id}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def emit_qr_ascii_or_base64(record: Mapping[str, Any], *, ascii_qr: bool, output_base64_path: str | None) -> None:
    payload = json.dumps(record, ensure_ascii=False, separators=(",", ":"))
    encoded = base64.b64encode(payload.encode("utf-8")).decode("ascii")
    if ascii_qr:
        try:
            import qrcode  # type: ignore
        except ModuleNotFoundError:
            print("[WARN] qrcode niedostępne – wypisuję BASE64:", encoded)
            return
        qr = qrcode.QRCode(border=1)
        qr.add_data(payload)
        qr.make(fit=True)
        for row in qr.get_matrix():
            print("".join("██" if cell else "  " for cell in row))
    if output_base64_path:
        Path(output_base64_path).write_text(encoded + "\n", encoding="utf-8")
    if not ascii_qr and not output_base64_path:
        print("Payload QR (BASE64):")
        print(encoded)


# --- walidacja rejestru ------------------------------------------------------
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
                    expected_fp = build_hmac_signature(fp_payload, key=fingerprint_keys[fp_key], key_id=fp_key)
                    if expected_fp["value"] != fp_sig.get("value"):
                        errors.append(f"Linia {index}: podpis fingerprintu niepoprawny.")
                else:
                    errors.append(f"Linia {index}: brak klucza fingerprintu '{fp_key}'.")
    return errors


# --- CLI ---------------------------------------------------------------------
def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    # fingerprint
    parser.add_argument("--fingerprint", help="Fingerprint: string lub JSON/BASE64 albo ścieżka do pliku.")
    parser.add_argument(
        "--mode", choices=["qr", "usb"], default="qr", help="Tryb dostarczenia fingerprintu (wpływa na politykę)."
    )
    # parametry licencji
    parser.add_argument("--issuer", default="OEM-Control", help="Identyfikator wystawcy licencji")
    parser.add_argument("--profile", default="paper", help="Profil pracy bota (paper/live/demo)")
    parser.add_argument("--bundle-version", default="0.0.0", help="Wersja bundla Core OEM")
    parser.add_argument("--feature", action="append", dest="features", default=[], help="Dodatkowe feature flagi")
    parser.add_argument("--valid-days", type=int, default=365, help="Ważność licencji (dni)")
    parser.add_argument("--notes", help="Opcjonalna notatka")
    # podpis – wariant A: pojedynczy klucz
    parser.add_argument("--signing-key-path", help="Plik zawierający klucz HMAC (min 32 bajty)")
    parser.add_argument("--key-id", help="Identyfikator klucza użytego do podpisu (wariant pojedynczy)")
    # podpis – wariant B: rotujący zestaw kluczy
    parser.add_argument("--license-key", action="append", dest="license_keys", help="Klucz w formacie key_id=sekret")
    parser.add_argument(
        "--rotation-log",
        default=str(DEFAULT_ROTATION_LOG),
        help="Rejestr rotacji kluczy HMAC (dla wariantu zestawu).",
    )
    parser.add_argument(
        "--rotation-interval-days", type=float, default=90.0, help="Interwał rotacji kluczy HMAC (dni)."
    )
    parser.add_argument("--no-mark-rotation", action="store_true", help="Nie zapisuj rotacji po podpisaniu")
    # walidacja fingerprintu JSON
    parser.add_argument(
        "--fingerprint-key",
        action="append",
        dest="fingerprint_keys",
        help="Klucz HMAC fingerprintu (key_id=sekret) do weryfikacji JSON fingerprintu.",
    )
    # rejestr / artefakty
    parser.add_argument("--output", default=str(DEFAULT_REGISTRY), help="Rejestr JSONL")
    parser.add_argument("--emit-qr", action="store_true", help="Wypisz ASCII QR do stdout")
    parser.add_argument("--qr-output", help="Plik z payloadem BASE64 (do QR)")
    parser.add_argument("--usb-target", help="Plik lub katalog docelowy na artefakt licencji (USB)")
    # tryb walidacji rejestru
    parser.add_argument("--validate-registry", action="store_true", help="Waliduj rejestr i zakończ.")
    return parser.parse_args(argv)


# --- main --------------------------------------------------------------------
def main(argv: Sequence[str] | None = None) -> int:
    try:
        args = _parse_args(argv)

        # Walidacja rejestru – krótsza ścieżka
        if args.validate_registry:
            license_keys = _parse_key_entries(args.license_keys)
            if not license_keys:
                raise ProvisioningError("Do walidacji rejestru wymagane są klucze licencyjne (--license-key).")
            fingerprint_keys = _parse_key_entries(args.fingerprint_keys)
            errors = validate_registry(Path(args.output), license_keys, fingerprint_keys)
            if errors:
                for msg in errors:
                    print(msg, file=sys.stderr)
                return 1
            print(f"Rejestr {args.output} – wszystkie podpisy prawidłowe.")
            return 0

        if args.valid_days <= 0:
            raise ProvisioningError("Ważność licencji musi być dodatnia.")

        # Wczytanie fingerprintu
        if args.fingerprint:
            fp_source = load_fingerprint_maybe_json(args.fingerprint)
        else:
            # Brak wejścia – spróbuj wygenerować prosty fingerprint lokalnie
            generator = DeviceFingerprintGenerator()
            fp_source = generator.generate_fingerprint()

        # Budowa payloadu
        if isinstance(fp_source, dict):
            # tryb „bogaty” (JSON)
            if args.fingerprint_keys:
                verify_fingerprint_signature(fp_source, _parse_key_entries(args.fingerprint_keys))
            payload = build_license_payload_rich(
                fingerprint_doc=fp_source,
                mode=args.mode,
                issuer=args.issuer,
                profile=args.profile,
                bundle_version=args.bundle_version,
                features=args.features,
                notes=args.notes,
                valid_days=args.valid_days,
                policy=OemPolicy(),
            )
        else:
            # tryb „prosty” (string)
            fingerprint_text = _validate_simple_fingerprint(fp_source)
            payload = build_license_payload_simple(
                fingerprint_text=fingerprint_text,
                issuer=args.issuer,
                profile=args.profile,
                bundle_version=args.bundle_version,
                features=args.features,
                notes=args.notes,
                valid_days=args.valid_days,
            )

        # Podpis licencji
        record: dict[str, Any]
        license_keys = _parse_key_entries(args.license_keys)
        if license_keys:
            record = sign_with_rotating_keys(
                payload,
                keys=license_keys,
                rotation_log=Path(args.rotation_log),
                purpose=DEFAULT_PURPOSE,
                interval_days=float(args.rotation_interval_days),
                mark_rotation=not args.no_mark_rotation,
            )
        else:
            if not args.signing_key_path:
                raise ProvisioningError(
                    "Podaj albo --license-key (można wiele), albo --signing-key-path (pojedynczy klucz)."
                )
            key_path = Path(args.signing_key_path)
            key_bytes = key_path.read_bytes()
            if len(key_bytes) < 32:
                raise ProvisioningError("Klucz podpisu musi mieć co najmniej 32 bajty.")
            record = sign_with_single_key(payload, key_bytes=key_bytes, key_id=args.key_id)

        # Zapis do rejestru
        registry_path = Path(args.output)
        append_jsonl(registry_path, record)
        print(f"[OK] Dodano licencję do rejestru {registry_path}")

        # Artefakty dodatkowe
        if args.emit_qr or args.qr_output:
            emit_qr_ascii_or_base64(record, ascii_qr=bool(args.emit_qr), output_base64_path=args.qr_output)
        if args.usb_target:
            target = write_usb_artifact(args.usb_target, record)
            print(f"[INFO] Licencja zapisana: {target}")

        return 0

    except ProvisioningError as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 2
    except FingerprintError as exc:
        print(f"[ERROR] Fingerprint: {exc}", file=sys.stderr)
        return 3


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
