"""Narzędzia do audytu certyfikatów TLS i pinningu."""
from __future__ import annotations

import hashlib
import os
import ssl
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping


__all__ = [
    "extract_pem_certificates",
    "certificate_fingerprint",
    "describe_certificate",
    "certificate_reference_metadata",
]


_BEGIN = b"-----BEGIN CERTIFICATE-----"
_END = b"-----END CERTIFICATE-----"


def extract_pem_certificates(pem_data: bytes | str | None) -> list[bytes]:
    """Zwraca listę certyfikatów PEM wyodrębnionych z danych."""

    if pem_data is None:
        return []
    if isinstance(pem_data, str):
        data = pem_data.encode("utf-8")
    else:
        data = pem_data
    certificates: list[bytes] = []
    search_from = 0
    while True:
        start = data.find(_BEGIN, search_from)
        if start == -1:
            break
        end = data.find(_END, start)
        if end == -1:
            break
        end += len(_END)
        block = data[start:end]
        certificates.append(block)
        search_from = end
    return certificates


def certificate_fingerprint(pem_block: bytes | str, *, algorithm: str = "sha256") -> str:
    """Oblicza fingerprint certyfikatu dla wskazanego algorytmu."""

    if isinstance(pem_block, bytes):
        pem_text = pem_block.decode("ascii")
    else:
        pem_text = pem_block
    try:
        der_bytes = ssl.PEM_cert_to_DER_cert(pem_text)
    except ValueError as exc:  # pragma: no cover - dane wejściowe muszą być PEM
        raise ValueError("Nieprawidłowy blok PEM certyfikatu") from exc
    digest = hashlib.new(algorithm)
    digest.update(der_bytes)
    return digest.hexdigest()


def _decode_certificate(pem_block: bytes) -> Mapping[str, Any] | None:
    decoder = getattr(getattr(ssl, "_ssl", None), "_test_decode_cert", None)
    if decoder is None:
        return None
    with tempfile.NamedTemporaryFile("wb", delete=False) as handle:
        handle.write(pem_block)
        tmp_name = handle.name
    try:
        return decoder(tmp_name)
    except Exception:  # pragma: no cover - diagnostyka dekodera
        return None
    finally:
        try:
            os.unlink(tmp_name)
        except OSError:
            pass


def _normalize_name(entries: Iterable[Iterable[tuple[str, str]]] | None) -> dict[str, Any]:
    result: dict[str, list[str]] = {}
    if not entries:
        return {}
    for group in entries:
        for key, value in group:
            result.setdefault(str(key), []).append(str(value))
    collapsed: dict[str, Any] = {}
    for key, values in result.items():
        if len(values) == 1:
            collapsed[key] = values[0]
        else:
            collapsed[key] = tuple(values)
    return collapsed


def _format_name(entries: Mapping[str, Any] | None) -> str | None:
    if not entries:
        return None
    parts: list[str] = []
    for key, value in entries.items():
        if isinstance(value, tuple):
            joined = "/".join(value)
            parts.append(f"{key}={joined}")
        else:
            parts.append(f"{key}={value}")
    return ", ".join(parts) or None


def _parse_time(label: str | None) -> datetime | None:
    if not label:
        return None
    try:
        parsed = datetime.strptime(label, "%b %d %H:%M:%S %Y %Z")
    except ValueError:
        return None
    return parsed.replace(tzinfo=timezone.utc)


def describe_certificate(pem_block: bytes | str) -> dict[str, Any]:
    """Zwraca podstawowe metadane certyfikatu X.509."""

    pem_bytes = pem_block.encode("utf-8") if isinstance(pem_block, str) else pem_block
    decoded = _decode_certificate(pem_bytes)
    if not decoded:
        return {}
    subject = _normalize_name(decoded.get("subject"))
    issuer = _normalize_name(decoded.get("issuer"))
    not_before = _parse_time(decoded.get("notBefore"))
    not_after = _parse_time(decoded.get("notAfter"))
    alt_names_raw = decoded.get("subjectAltName") or []
    alt_names = [
        {"type": str(entry[0]), "value": str(entry[1])}
        for entry in alt_names_raw
        if isinstance(entry, tuple) and len(entry) == 2
    ]
    metadata: dict[str, Any] = {
        "subject": subject,
        "issuer": issuer,
        "serial_number": decoded.get("serialNumber"),
    }
    if alt_names:
        metadata["subject_alt_names"] = alt_names
    if not_before:
        metadata["not_before"] = not_before.isoformat()
    if not_after:
        metadata["not_after"] = not_after.isoformat()
    subject_label = _format_name(subject)
    if subject_label:
        metadata["subject_label"] = subject_label
    issuer_label = _format_name(issuer)
    if issuer_label:
        metadata["issuer_label"] = issuer_label
    if not_after:
        metadata["expires_in_days"] = (not_after - datetime.now(timezone.utc)).total_seconds() / 86_400.0
    return metadata


def certificate_reference_metadata(
    path: str | Path,
    *,
    role: str = "tls_cert",
    warn_expiring_within_days: float = 30.0,
) -> dict[str, Any]:
    """Łączy metadane pliku z informacjami o certyfikatach i ostrzeżeniami."""

    from bot_core.runtime.file_metadata import file_reference_metadata

    metadata = dict(file_reference_metadata(path, role=role))
    warnings = list(metadata.get("security_warnings", ()))
    metadata["security_warnings"] = warnings
    if not metadata.get("exists"):
        return metadata
    try:
        pem_bytes = Path(path).expanduser().read_bytes()
    except OSError as exc:
        warnings.append(f"Nie udało się odczytać certyfikatu TLS ({exc}).")
        return metadata
    blocks = extract_pem_certificates(pem_bytes)
    if not blocks:
        warnings.append("Plik nie zawiera certyfikatów w formacie PEM.")
        return metadata
    certificates: list[dict[str, Any]] = []
    now = datetime.now(timezone.utc)
    for index, block in enumerate(blocks):
        entry: dict[str, Any] = {"index": index}
        try:
            entry["fingerprint_sha256"] = certificate_fingerprint(block, algorithm="sha256")
        except ValueError:
            warnings.append("Nie udało się obliczyć fingerprintu certyfikatu (nieprawidłowy blok PEM).")
            continue
        details = describe_certificate(block)
        entry.update(details)
        certificates.append(entry)
        expiry_iso = details.get("not_after")
        expiry_dt: datetime | None = None
        if isinstance(expiry_iso, str):
            try:
                expiry_dt = datetime.fromisoformat(expiry_iso)
            except ValueError:
                expiry_dt = None
        if expiry_dt is None:
            warnings.append("Nie udało się ustalić daty wygaśnięcia certyfikatu.")
            continue
        remaining_days = (expiry_dt - now).total_seconds() / 86_400.0
        subject = details.get("subject_label") or details.get("subject", {}).get("commonName")
        label = subject or f"#{index}"
        if remaining_days < 0:
            warnings.append(f"Certyfikat {label} wygasł {abs(remaining_days):.1f} dnia temu.")
        elif remaining_days <= warn_expiring_within_days:
            warnings.append(
                f"Certyfikat {label} wygaśnie za {remaining_days:.1f} dnia – zaplanuj rotację."
            )
    metadata["certificates"] = certificates
    if not certificates:
        warnings.append("Nie znaleziono żadnych poprawnych certyfikatów w pliku.")
    return metadata
