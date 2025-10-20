"""Generator licencji offline zgodny z podpisem Ed25519."""
from __future__ import annotations

import argparse
import base64
import json
import hashlib
from pathlib import Path
from typing import Any

from nacl.signing import SigningKey, VerifyKey


def _load_payload(path: Path) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise SystemExit(f"Niepoprawny plik JSON: {exc}") from exc
    if not isinstance(data, dict):
        raise SystemExit("Payload licencji musi być obiektem JSON.")
    return data


def sign_license(payload: dict[str, Any], signing_key: SigningKey) -> dict[str, str]:
    payload_bytes = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    signature = signing_key.sign(payload_bytes).signature
    return {
        "payload_b64": base64.b64encode(payload_bytes).decode("ascii"),
        "signature_b64": base64.b64encode(signature).decode("ascii"),
    }


def verify_license(bundle: dict[str, str], verify_key: VerifyKey) -> dict[str, Any]:
    payload_b64 = bundle.get("payload_b64")
    signature_b64 = bundle.get("signature_b64")
    if not isinstance(payload_b64, str) or not isinstance(signature_b64, str):
        raise SystemExit("Pakiet licencji musi zawierać pola payload_b64 i signature_b64.")
    payload_bytes = base64.b64decode(payload_b64.encode("ascii"))
    signature_bytes = base64.b64decode(signature_b64.encode("ascii"))
    verify_key.verify(payload_bytes, signature_bytes)
    return json.loads(payload_bytes.decode("utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("payload", type=Path, help="Ścieżka do pliku JSON z payloadem licencji")
    parser.add_argument("key", help="Klucz prywatny Ed25519 w formacie hex")
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        default=Path("license_bundle.lic"),
        help="Plik wyjściowy licencji (domyślnie license_bundle.lic)",
    )
    parser.add_argument(
        "--verify",
        help="Opcjonalny publiczny klucz Ed25519 (hex) do weryfikacji wygenerowanego pakietu",
    )
    args = parser.parse_args()

    payload = _load_payload(args.payload)
    signing_key = SigningKey(bytes.fromhex(args.key))
    bundle = sign_license(payload, signing_key)
    args.output.write_text(json.dumps(bundle, indent=2, ensure_ascii=False), encoding="utf-8")

    payload_bytes = base64.b64decode(bundle["payload_b64"].encode("ascii"))
    print(f"Zapisano licencję: {args.output}")
    print(f"Payload SHA-256: {hashlib.sha256(payload_bytes).hexdigest()}")

    if args.verify:
        verify_key = VerifyKey(bytes.fromhex(args.verify))
        verified_payload = verify_license(bundle, verify_key)
        print("Weryfikacja podpisu zakończona powodzeniem.")
        print("Edycja:", verified_payload.get("edition"))


if __name__ == "__main__":  # pragma: no cover - CLI
    main()
