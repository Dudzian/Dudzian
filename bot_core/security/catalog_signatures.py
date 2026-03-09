"""Weryfikacja podpisów katalogu Marketplace (HMAC + Ed25519)."""

from __future__ import annotations

import base64
import binascii
import hashlib
import hmac
import json
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Mapping

try:  # pragma: no cover - fallback dla minimalnych środowisk CI bez cryptography
    from cryptography.exceptions import InvalidSignature
    from cryptography.hazmat.primitives import serialization
    from cryptography.hazmat.primitives.asymmetric import ed25519
except ModuleNotFoundError:  # pragma: no cover
    InvalidSignature = Exception  # type: ignore[assignment]
    serialization = None  # type: ignore[assignment]
    ed25519 = None  # type: ignore[assignment]


def _load_signature(path: Path) -> Mapping[str, Any]:
    document = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(document, Mapping):
        raise ValueError("Dokument podpisu katalogu musi być słownikiem JSON.")
    return document


def _load_ed25519_public_key(data: bytes) -> ed25519.Ed25519PublicKey:
    if ed25519 is None or serialization is None:
        raise ValueError("Brak zależności 'cryptography' do weryfikacji Ed25519")
    if isinstance(data, (bytes, bytearray)):
        if len(data) == 32:
            return ed25519.Ed25519PublicKey.from_public_bytes(bytes(data))
        payload = bytes(data).strip()
    else:
        payload = str(data).strip().encode("utf-8")
    if len(payload) == 32:
        return ed25519.Ed25519PublicKey.from_public_bytes(payload)
    try:
        candidate = base64.b64decode(payload, validate=False)
        if candidate:
            return ed25519.Ed25519PublicKey.from_public_bytes(candidate)
    except Exception:  # noqa: BLE001 - defensywne próby parsowania
        pass
    try:
        return serialization.load_pem_public_key(payload)
    except Exception as exc:  # noqa: BLE001 - defensywne
        raise ValueError("Niepoprawny klucz publiczny Ed25519 dla katalogu Marketplace") from exc


def _ed25519_public_key_to_pem(data: bytes) -> bytes:
    payload = bytes(data).strip()
    if payload.startswith(b"-----BEGIN"):
        return payload
    if len(payload) != 32:
        try:
            payload = base64.b64decode(payload, validate=False)
        except Exception as exc:  # noqa: BLE001 - defensywne
            raise ValueError("Niepoprawny klucz publiczny Ed25519 dla katalogu Marketplace") from exc
    if len(payload) != 32:
        raise ValueError("Niepoprawny klucz publiczny Ed25519 dla katalogu Marketplace")
    der = bytes.fromhex("302a300506032b6570032100") + payload
    body = base64.encodebytes(der).replace(b"\n", b"")
    lines = [body[i : i + 64] for i in range(0, len(body), 64)]
    return b"-----BEGIN PUBLIC KEY-----\n" + b"\n".join(lines) + b"\n-----END PUBLIC KEY-----\n"


def _verify_hmac_block(
    content: bytes, signature: Mapping[str, Any], key: bytes | None
) -> list[str]:
    if not signature:
        return []
    if key is None:
        return ["Brak klucza HMAC do weryfikacji katalogu Marketplace."]
    algorithm = str(signature.get("algorithm") or "").upper()
    if not algorithm.startswith("HMAC-"):
        return [f"Nieobsługiwany algorytm HMAC katalogu: {algorithm or '<brak>'}"]
    expected = base64.b64encode(hmac.new(key, content, hashlib.sha256).digest()).decode("ascii")
    actual = signature.get("value")
    if not isinstance(actual, str):
        return ["Podpis HMAC katalogu nie zawiera wartości 'value'."]
    if not hmac.compare_digest(actual, expected):
        return ["Podpis HMAC katalogu jest niepoprawny (mismatch value)."]
    return []


def _verify_ed25519_with_openssl(
    *,
    content: bytes,
    signature_bytes: bytes,
    public_key_pem: bytes,
) -> list[str]:
    with tempfile.TemporaryDirectory(prefix="catalog-signature-") as tmp_dir:
        root = Path(tmp_dir)
        pub_path = root / "catalog-ed25519.pub"
        sig_path = root / "catalog-ed25519.sig"
        content_path = root / "catalog-content.bin"
        pub_path.write_bytes(public_key_pem)
        sig_path.write_bytes(signature_bytes)
        content_path.write_bytes(content)

        command = [
            "openssl",
            "pkeyutl",
            "-verify",
            "-pubin",
            "-inkey",
            str(pub_path),
            "-sigfile",
            str(sig_path),
            "-rawin",
            "-in",
            str(content_path),
        ]
        try:
            result = subprocess.run(
                command,
                check=False,
                capture_output=True,
                text=True,
            )
        except FileNotFoundError:
            return [
                "Nie można zweryfikować podpisu Ed25519 katalogu: brak narzędzia 'openssl' w PATH."
            ]
        except OSError as exc:
            return [f"Nie można uruchomić 'openssl' do weryfikacji podpisu Ed25519: {exc}"]

    if result.returncode == 0:
        return []

    details = (result.stderr or result.stdout or "").strip()
    if details:
        return [
            "Podpis Ed25519 katalogu jest niepoprawny lub nie można go zweryfikować przez openssl: "
            + details
        ]
    return ["Podpis Ed25519 katalogu jest niepoprawny (signature mismatch)."]


def _verify_ed25519_block(
    content: bytes, signature: Mapping[str, Any], key: bytes | None
) -> list[str]:
    if not signature:
        return []
    if key is None:
        return ["Brak klucza publicznego Ed25519 do weryfikacji katalogu Marketplace."]
    value = signature.get("value")
    if not isinstance(value, str):
        return ["Podpis Ed25519 katalogu nie zawiera wartości 'value'."]
    try:
        signature_bytes = base64.b64decode(value)
    except (ValueError, binascii.Error):
        return ["Niepoprawny format base64 w podpisie Ed25519 katalogu."]
    algorithm = str(signature.get("algorithm") or "").lower() or "ed25519"
    if algorithm != "ed25519":
        return [f"Nieobsługiwany algorytm podpisu katalogu: {algorithm}"]

    if ed25519 is not None and serialization is not None:
        try:
            public_key = _load_ed25519_public_key(key)
        except ValueError as exc:
            return [str(exc)]
        try:
            public_key.verify(signature_bytes, content)
        except InvalidSignature:
            return ["Podpis Ed25519 katalogu jest niepoprawny (signature mismatch)."]
        return []

    try:
        pem_key = _ed25519_public_key_to_pem(key)
    except ValueError as exc:
        return [str(exc)]

    return _verify_ed25519_with_openssl(
        content=content,
        signature_bytes=signature_bytes,
        public_key_pem=pem_key,
    )


def verify_catalog_signature(
    *,
    content: bytes,
    signature: Mapping[str, Any],
    hmac_key: bytes | None,
    ed25519_key: bytes | None,
    target_path: Path | None = None,
) -> list[str]:
    """Waliduje podpis katalogu Marketplace.

    Zwraca listę błędów – pusta lista oznacza poprawny podpis.
    """

    errors: list[str] = []

    sha256_value = signature.get("sha256")
    if not isinstance(sha256_value, str):
        errors.append("Podpis katalogu nie zawiera pola 'sha256'.")
    else:
        digest = hashlib.sha256(content).hexdigest()
        if digest != sha256_value:
            errors.append("Suma SHA256 katalogu różni się od zapisanej w podpisie.")

    if target_path is not None:
        target = signature.get("target")
        if isinstance(target, str) and target.strip():
            try:
                if Path(target).resolve() != target_path.resolve():
                    errors.append("Podpis katalogu wskazuje inny plik docelowy (pole 'target').")
            except OSError:
                errors.append("Nie udało się zinterpretować pola 'target' w podpisie katalogu.")

    hmac_errors = _verify_hmac_block(content, signature.get("hmac") or {}, hmac_key)
    ed_errors = _verify_ed25519_block(content, signature.get("ed25519") or {}, ed25519_key)

    if not signature.get("hmac") and not signature.get("ed25519"):
        errors.append("Podpis katalogu nie zawiera sekcji HMAC ani Ed25519.")

    errors.extend(hmac_errors)
    errors.extend(ed_errors)
    return errors


def verify_catalog_signature_file(
    path: Path,
    *,
    hmac_key: bytes | None,
    ed25519_key: bytes | None,
) -> list[str]:
    """Waliduje podpis dla wskazanego pliku katalogu (JSON/Markdown)."""

    signature_path = path.with_suffix(path.suffix + ".sig")
    if not signature_path.exists():
        return [f"Brak pliku podpisu katalogu: {signature_path}"]
    signature_doc = _load_signature(signature_path)
    content = path.read_bytes()
    return verify_catalog_signature(
        content=content,
        signature=signature_doc,
        hmac_key=hmac_key,
        ed25519_key=ed25519_key,
        target_path=path,
    )


__all__ = [
    "verify_catalog_signature",
    "verify_catalog_signature_file",
]
