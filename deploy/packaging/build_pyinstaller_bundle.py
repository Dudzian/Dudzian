"""Build PyInstaller/Briefcase bundles combining bot_core runtime and Qt UI."""
from __future__ import annotations

import argparse
import base64
import binascii
import configparser
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import ssl
import string
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping
from urllib import error as urlerror
from urllib import parse as urlparse
from urllib import request as urlrequest

import tomllib
import yaml


def _ensure_mutable_mapping(value: object, *, context: str) -> dict[str, object]:
    """Ensure ``value`` is a mutable dictionary for metadata composition."""

    if isinstance(value, dict):
        return value
    if isinstance(value, Mapping):
        return dict(value)
    raise SystemExit(
        f"Pole metadanych {context} musi być obiektem JSON (mapą), otrzymano {type(value).__name__}"
    )


def _apply_metadata_entry(metadata: dict[str, object], key: str, value: object) -> None:
    """Apply dotted metadata entries like ``release.commit`` into nested objects."""

    parts = key.split(".")
    if any(not part for part in parts):
        raise SystemExit(f"Klucz metadanych {key} zawiera pusty segment")

    cursor: dict[str, object] = metadata
    for index, part in enumerate(parts):
        is_leaf = index == len(parts) - 1
        if is_leaf:
            if part in cursor:
                raise SystemExit(f"Klucz {key} został już zdefiniowany w metadanych")
            cursor[part] = value
            return

        existing = cursor.get(part)
        if existing is None:
            nested: dict[str, object] = {}
            cursor[part] = nested
            cursor = nested
            continue

        nested_mapping = _ensure_mutable_mapping(existing, context=part)
        cursor[part] = nested_mapping
        cursor = nested_mapping

from bot_core.security.license_store import LicenseStore, LicenseStoreError
from bot_core.security.signing import build_hmac_signature


@dataclass(slots=True)
class ArtifactSpec:
    """Descriptor for artifacts staged inside the bundle."""

    bundle_path: Path
    source_path: Path


@dataclass(slots=True)
class BundleLayout:
    """Resolved staging layout for the bundle."""

    root: Path
    daemon_dir: Path
    ui_dir: Path
    extras_dir: Path


def _resolve_layout(output_dir: Path, version: str, platform: str) -> BundleLayout:
    bundle_root = output_dir / f"core-runtime-{version}-{platform}"
    daemon_dir = bundle_root / "daemon"
    ui_dir = bundle_root / "ui"
    extras_dir = bundle_root / "resources"
    for path in (daemon_dir, ui_dir, extras_dir):
        path.mkdir(parents=True, exist_ok=True)
    return BundleLayout(root=bundle_root, daemon_dir=daemon_dir, ui_dir=ui_dir, extras_dir=extras_dir)


def _hash_file(path: Path) -> str:
    digest = hashlib.sha384()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _collect_artifact_metadata(artifacts: Iterable[ArtifactSpec]) -> list[Mapping[str, object]]:
    entries: list[Mapping[str, object]] = []
    for artifact in artifacts:
        size = artifact.source_path.stat().st_size
        digest = _hash_file(artifact.source_path)
        entries.append(
            {
                "path": artifact.bundle_path.as_posix(),
                "sha384": digest,
                "size": size,
            }
        )
    return entries


def _write_integrity_manifest(layout: BundleLayout) -> ArtifactSpec:
    manifest_entries: list[Mapping[str, object]] = []
    for path in sorted(layout.root.rglob("*")):
        if not path.is_file():
            continue
        relative = path.relative_to(layout.root).as_posix()
        if relative.endswith(".zip"):
            continue
        manifest_entries.append(
            {
                "path": relative,
                "sha384": _hash_file(path),
                "size": path.stat().st_size,
            }
        )

    manifest_payload = {
        "algorithm": "sha384",
        "generated_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "files": manifest_entries,
    }

    target = layout.extras_dir / "integrity_manifest.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(manifest_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return ArtifactSpec(bundle_path=target.relative_to(layout.root), source_path=target)


def _decode_secret(value: str) -> bytes:
    try:
        return base64.b64decode(value, validate=True)
    except binascii.Error:
        pass
    hexdigits = set(string.hexdigits)
    if len(value) % 2 == 0 and all(ch in hexdigits for ch in value):
        try:
            return bytes.fromhex(value)
        except ValueError:
            pass
    return value.encode("utf-8")


def _parse_license_hmac_key(option: str | None) -> tuple[str | None, bytes] | None:
    if option is None:
        return None
    key_id: str | None = None
    secret = option
    if "=" in option:
        key_id, secret = option.split("=", 1)
        key_id = key_id.strip() or None
    secret_bytes = _decode_secret(secret.strip())
    return key_id, secret_bytes


def _embed_encrypted_license(
    layout: BundleLayout,
    *,
    license_json: str | None,
    fingerprint: str | None,
    output_name: str,
    hmac_key: tuple[str | None, bytes] | None,
) -> list[ArtifactSpec]:
    if not license_json:
        return []
    license_path = Path(license_json).expanduser().resolve()
    if not license_path.exists():
        raise SystemExit(f"Plik licencji {license_path} nie istnieje")
    try:
        payload = json.loads(license_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise SystemExit(f"Licencja {license_path} zawiera niepoprawny JSON: {exc}") from exc
    if not isinstance(payload, Mapping):
        raise SystemExit("Payload licencji musi być obiektem JSON")
    if not fingerprint:
        raise SystemExit("--license-json wymaga wskazania --license-fingerprint")
    license_id = str(payload.get("license_id") or payload.get("licenseId") or "").strip()
    if not license_id:
        raise SystemExit("Licencja musi zawierać pole license_id")

    license_dir = layout.extras_dir / "license"
    license_dir.mkdir(parents=True, exist_ok=True)
    store_path = license_dir / output_name
    store = LicenseStore(path=store_path, fingerprint_override=fingerprint)
    store_payload: dict[str, Any] = {
        "licenses": {
            license_id: {
                "payload": payload,
                "status": "provisioned",
                "issues": [],
                "hardware": {},
                "provisioned_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            }
        }
    }
    try:
        store.save(store_payload)
    except LicenseStoreError as exc:
        raise SystemExit(f"Nie udało się zaszyfrować magazynu licencji: {exc}") from exc

    bundle_path = store_path.relative_to(layout.root)
    digest_entry = {
        "path": bundle_path.as_posix(),
        "sha384": _hash_file(store_path),
        "license_id": license_id,
    }
    report_payload = {
        "generated_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "license_store": digest_entry,
    }
    if hmac_key is not None:
        key_id, secret = hmac_key
        signature = build_hmac_signature(
            {"generated_at": report_payload["generated_at"], "license_store": digest_entry},
            key=secret,
            key_id=key_id,
        )
        report_payload["signature"] = signature

    integrity_path = license_dir / "license_integrity.json"
    with integrity_path.open("w", encoding="utf-8") as handle:
        json.dump(report_payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")

    return [
        ArtifactSpec(bundle_path=bundle_path, source_path=store_path),
        ArtifactSpec(
            bundle_path=integrity_path.relative_to(layout.root),
            source_path=integrity_path,
        ),
    ]


def _parse_key_value_pairs(values: Iterable[str] | None, *, option: str) -> dict[str, object]:
    """Convert ``key=value`` pairs into a dictionary, parsing JSON values when possible."""

    result: dict[str, object] = {}
    if not values:
        return result

    for item in values:
        key, sep, raw_value = item.partition("=")
        if not key or sep != "=":
            raise SystemExit(
                f"Niepoprawny format {option}: {item}. Wymagany zapis <klucz>=<wartość>."
            )
        try:
            value: object = json.loads(raw_value)
        except json.JSONDecodeError:
            value = raw_value
        if key in result:
            raise SystemExit(f"Klucz {key} został zduplikowany w opcji {option}")
        result[key] = value
    return result


def _parse_http_headers(values: Iterable[str] | None, *, option: str) -> dict[str, str]:
    """Parse ``Key=Value`` entries into HTTP headers, rejecting duplicates."""

    headers: dict[str, str] = {}
    if not values:
        return headers

    seen: set[str] = set()
    for item in values:
        key, sep, raw_value = item.partition("=")
        if sep != "=" or not key:
            raise SystemExit(
                f"Niepoprawny format {option}: {item}. Wymagany zapis <nazwa>=<wartość>."
            )
        header_name = key.strip()
        if not header_name:
            raise SystemExit(f"Nagłówek w {option} musi mieć nazwę różną od pustej: {item}")
        lower_name = header_name.lower()
        if lower_name in seen:
            raise SystemExit(f"Nagłówek {header_name} został zduplikowany w opcji {option}")
        seen.add(lower_name)
        headers[header_name] = raw_value.strip()

    return headers


_EKU_ALIAS_MAP: dict[str, str] = {
    "serverauth": "1.3.6.1.5.5.7.3.1",
    "tls web server authentication": "1.3.6.1.5.5.7.3.1",
    "clientauth": "1.3.6.1.5.5.7.3.2",
    "tls web client authentication": "1.3.6.1.5.5.7.3.2",
    "codesigning": "1.3.6.1.5.5.7.3.3",
    "code signing": "1.3.6.1.5.5.7.3.3",
    "emailprotection": "1.3.6.1.5.5.7.3.4",
    "email protection": "1.3.6.1.5.5.7.3.4",
    "e-mail protection": "1.3.6.1.5.5.7.3.4",
    "timestamping": "1.3.6.1.5.5.7.3.8",
    "time stamping": "1.3.6.1.5.5.7.3.8",
    "ocspsigning": "1.3.6.1.5.5.7.3.9",
    "ocsp signing": "1.3.6.1.5.5.7.3.9",
    "anyextendedkeyusage": "2.5.29.37.0",
    "any extended key usage": "2.5.29.37.0",
}

_POLICY_ALIAS_MAP: dict[str, str] = {
    "anypolicy": "2.5.29.32.0",
    "any policy": "2.5.29.32.0",
}

_OID_PATTERN = re.compile(r"^\d+(?:\.\d+)*$")


def _parse_cert_fingerprints(
    values: Iterable[str] | None, *, option: str
) -> dict[str, set[str]]:
    """Validate and normalize certificate fingerprint declarations."""

    fingerprints: dict[str, set[str]] = {}
    if not values:
        return fingerprints

    allowed_algorithms = {"sha256", "sha384", "sha512"}

    for item in values:
        algorithm, sep, fingerprint = item.partition(":")
        if sep != ":" or not algorithm or not fingerprint:
            raise SystemExit(
                f"Niepoprawny format {option}: {item}. Wymagany zapis <algorytm>:<odcisk>."
            )
        normalized_algorithm = algorithm.lower()
        if normalized_algorithm not in allowed_algorithms:
            allowed = ", ".join(sorted(allowed_algorithms))
            raise SystemExit(
                f"Algorytm {algorithm} w {option} nie jest obsługiwany. Dozwolone: {allowed}"
            )

        cleaned = fingerprint.replace(":", "").replace(" ", "").lower()
        if not cleaned or any(char not in string.hexdigits for char in cleaned):
            raise SystemExit(
                f"Odcisk palca w {option} musi być wartością szesnastkową: {item}"
            )

        expected_length = hashlib.new(normalized_algorithm).digest_size * 2
        if len(cleaned) != expected_length:
            raise SystemExit(
                f"Odcisk palca dla {normalized_algorithm} w {option} ma niepoprawną długość"
            )

        fingerprints.setdefault(normalized_algorithm, set()).add(cleaned)

    return fingerprints


def _parse_cert_subject_requirements(
    values: Iterable[str] | None, *, option: str
) -> dict[str, set[str]]:
    """Validate and normalize certificate subject attribute requirements."""

    requirements: dict[str, set[str]] = {}
    if not values:
        return requirements

    for item in values:
        key, sep, value = item.partition("=")
        if sep != "=" or not key or not value:
            raise SystemExit(
                f"Niepoprawny format {option}: {item}. Wymagany zapis <atrybut>=<wartość>."
            )
        normalized_key = key.strip().lower()
        if not normalized_key:
            raise SystemExit(
                f"Atrybut tematu certyfikatu w {option} nie może być pusty: {item}"
            )
        requirements.setdefault(normalized_key, set()).add(value)

    return requirements


def _parse_cert_issuer_requirements(
    values: Iterable[str] | None, *, option: str
) -> dict[str, set[str]]:
    """Validate and normalize certificate issuer attribute requirements."""

    requirements: dict[str, set[str]] = {}
    if not values:
        return requirements

    for item in values:
        key, sep, value = item.partition("=")
        if sep != "=" or not key or not value:
            raise SystemExit(
                f"Niepoprawny format {option}: {item}. Wymagany zapis <atrybut>=<wartość>."
            )
        normalized_key = key.strip().lower()
        if not normalized_key:
            raise SystemExit(
                f"Atrybut wystawcy certyfikatu w {option} nie może być pusty: {item}"
            )
        requirements.setdefault(normalized_key, set()).add(value)

    return requirements


def _parse_cert_san_requirements(
    values: Iterable[str] | None, *, option: str
) -> dict[str, set[str]]:
    """Validate and normalize certificate subjectAltName requirements."""

    requirements: dict[str, set[str]] = {}
    if not values:
        return requirements

    for item in values:
        key, sep, value = item.partition("=")
        if sep != "=" or not key or not value:
            raise SystemExit(
                f"Niepoprawny format {option}: {item}. Wymagany zapis <typ>=<wartość>."
            )
        normalized_key = key.strip().lower()
        if not normalized_key:
            raise SystemExit(
                f"Typ wpisu subjectAltName w {option} nie może być pusty: {item}"
            )
        requirements.setdefault(normalized_key, set()).add(value)

    return requirements


def _parse_cert_extended_key_usage(
    values: Iterable[str] | None, *, option: str
) -> set[str]:
    """Validate required Extended Key Usage entries."""

    requirements: set[str] = set()
    if not values:
        return requirements

    for item in values:
        candidate = item.strip()
        if not candidate:
            raise SystemExit(
                f"Wartość w {option} nie może być pusta. Użyj aliasu (np. serverAuth) lub OID."
            )
        normalized = candidate.lower()
        if normalized in _EKU_ALIAS_MAP:
            requirements.add(_EKU_ALIAS_MAP[normalized])
            continue
        if _OID_PATTERN.fullmatch(candidate):
            requirements.add(candidate)
            continue
        raise SystemExit(
            f"Nieznany identyfikator EKU {item} w {option}. Użyj aliasu (np. serverAuth) lub OID."
        )

    return requirements


def _parse_cert_policy_requirements(
    values: Iterable[str] | None, *, option: str
) -> set[str]:
    """Validate required certificate policy identifiers."""

    requirements: set[str] = set()
    if not values:
        return requirements

    for item in values:
        candidate = item.strip()
        if not candidate:
            raise SystemExit(
                f"Wartość w {option} nie może być pusta. Użyj aliasu (np. anyPolicy) lub OID."
            )
        normalized = candidate.lower()
        alias = _POLICY_ALIAS_MAP.get(normalized)
        if alias:
            requirements.add(alias)
            continue
        if _OID_PATTERN.fullmatch(candidate):
            requirements.add(candidate)
            continue
        raise SystemExit(
            f"Nieznany identyfikator polityki certyfikatu {item} w {option}. Użyj aliasu lub OID."
        )

    return requirements


def _parse_cert_serial_requirements(
    values: Iterable[str] | None, *, option: str
) -> set[str]:
    """Validate and normalize allowed certificate serial numbers."""

    serials: set[str] = set()
    if not values:
        return serials

    for item in values:
        raw_value = item.strip()
        if not raw_value:
            raise SystemExit(
                f"Numer seryjny w {option} nie może być pusty"
            )

        candidate = raw_value
        if candidate.lower().startswith("0x"):
            candidate = candidate[2:]

        cleaned = candidate.replace(":", "").replace(" ", "")
        if not cleaned:
            raise SystemExit(
                f"Numer seryjny w {option} musi zawierać cyfry szesnastkowe lub dziesiętne"
            )

        is_decimal = (
            cleaned.isdigit()
            and not raw_value.lower().startswith("0x")
            and ":" not in raw_value
            and " " not in raw_value
        )

        if is_decimal:
            try:
                normalized_int = int(cleaned, 10)
            except ValueError as exc:  # pragma: no cover - defensive
                raise SystemExit(
                    f"Numer seryjny w {option} musi być liczbą dziesiętną lub szesnastkową"
                ) from exc
            normalized = format(normalized_int, "x")
        else:
            if any(char not in string.hexdigits for char in cleaned):
                raise SystemExit(
                    f"Numer seryjny w {option} musi być liczbą dziesiętną lub szesnastkową"
                )
            normalized = cleaned.lstrip("0") or "0"

        serials.add(normalized.lower())

    return serials


def _decode_certificate_metadata(peer_cert: bytes) -> Mapping[str, object]:
    """Decode DER certificate into mapping provided by ``ssl._test_decode_cert``."""

    try:
        decode_cert = ssl._ssl._test_decode_cert  # type: ignore[attr-defined]
    except AttributeError as exc:  # pragma: no cover - środowiska bez CPythonowego helpera
        raise SystemExit(
            "Środowisko wykonawcze nie obsługuje dekodowania certyfikatów TLS dla metadanych"
        ) from exc

    pem_data = ssl.DER_cert_to_PEM_cert(peer_cert)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", delete=False) as handle:
        handle.write(pem_data)
        temp_path = handle.name
    try:
        decoded = decode_cert(temp_path)
    finally:
        try:
            os.unlink(temp_path)
        except OSError:  # pragma: no cover - defensywne sprzątanie
            pass

    return decoded


def _extract_certificate_subject(peer_cert: bytes) -> dict[str, list[str]]:
    """Decode certificate subject attributes into a mapping."""

    decoded = _decode_certificate_metadata(peer_cert)
    subject_entries = decoded.get("subject", [])
    subject_map: dict[str, list[str]] = {}
    for entry in subject_entries:
        for attr_name, attr_value in entry:
            subject_map.setdefault(attr_name.lower(), []).append(attr_value)
    return subject_map


def _extract_certificate_issuer(peer_cert: bytes) -> dict[str, list[str]]:
    """Decode certificate issuer attributes into a mapping."""

    decoded = _decode_certificate_metadata(peer_cert)
    issuer_entries = decoded.get("issuer", [])
    issuer_map: dict[str, list[str]] = {}
    for entry in issuer_entries:
        for attr_name, attr_value in entry:
            issuer_map.setdefault(attr_name.lower(), []).append(attr_value)
    return issuer_map


def _extract_certificate_subject_alternative_names(peer_cert: bytes) -> dict[str, list[str]]:
    """Decode certificate subjectAltName entries into a mapping."""

    decoded = _decode_certificate_metadata(peer_cert)
    san_entries = decoded.get("subjectAltName", [])
    san_map: dict[str, list[str]] = {}
    for entry_type, entry_value in san_entries:
        san_map.setdefault(entry_type.lower(), []).append(entry_value)
    return san_map


def _extract_certificate_extended_key_usage(peer_cert: bytes) -> tuple[bool, set[str]]:
    """Decode certificate Extended Key Usage entries into normalized OIDs."""

    decoded = _decode_certificate_metadata(peer_cert)
    eku_entries = decoded.get("extendedKeyUsage")
    if not eku_entries:
        return False, set()

    normalized: set[str] = set()
    for entry in eku_entries:
        if not isinstance(entry, str):
            continue
        candidate = entry.strip()
        if not candidate:
            continue
        lowered = candidate.lower()
        alias = _EKU_ALIAS_MAP.get(lowered)
        if alias:
            normalized.add(alias)
            continue
        if _OID_PATTERN.fullmatch(candidate):
            normalized.add(candidate)

    return True, normalized


def _extract_certificate_policies(peer_cert: bytes) -> tuple[bool, set[str]]:
    """Decode certificate policy identifiers into normalized OIDs."""

    decoded = _decode_certificate_metadata(peer_cert)
    policy_entries = decoded.get("certificatePolicies")
    if not policy_entries:
        return False, set()

    normalized: set[str] = set()

    def _store(value: object) -> None:
        if not isinstance(value, str):
            return
        candidate = value.strip()
        if not candidate:
            return
        lowered = candidate.lower()
        alias = _POLICY_ALIAS_MAP.get(lowered)
        if alias:
            normalized.add(alias)
            return
        if _OID_PATTERN.fullmatch(candidate):
            normalized.add(candidate)

    entries: Iterable[object]
    if isinstance(policy_entries, (list, tuple, set)):
        entries = policy_entries
    else:
        entries = [policy_entries]

    for entry in entries:
        if isinstance(entry, Mapping):
            _store(entry.get("policyIdentifier"))
            continue
        if isinstance(entry, (list, tuple)):
            if entry:
                _store(entry[0])
            continue
        _store(entry)

    return True, normalized


def _extract_certificate_serial_number(peer_cert: bytes) -> str:
    """Return the normalized serial number of the TLS certificate."""

    decoded = _decode_certificate_metadata(peer_cert)
    serial = decoded.get("serialNumber")
    if isinstance(serial, int):
        if serial < 0:
            raise SystemExit("Certyfikat TLS źródła metadanych ma ujemny numer seryjny")
        return format(serial, "x")

    if isinstance(serial, str):
        raw_serial = serial.strip()
        cleaned = raw_serial.replace(":", "").replace(" ", "")
        if cleaned.lower().startswith("0x"):
            cleaned = cleaned[2:]
        cleaned = cleaned.strip()
        if not cleaned:
            raise SystemExit("Certyfikat TLS źródła metadanych ma pusty numer seryjny")
        is_decimal = (
            cleaned.isdigit()
            and not raw_serial.lower().startswith("0x")
            and ":" not in raw_serial
            and " " not in raw_serial
        )
        if is_decimal:
            return format(int(cleaned, 10), "x")
        if all(char in string.hexdigits for char in cleaned):
            return (cleaned.lstrip("0") or "0").lower()
        raise SystemExit("Certyfikat TLS źródła metadanych ma nieprawidłowy numer seryjny")

    raise SystemExit("Certyfikat TLS źródła metadanych nie zawiera numeru seryjnego")


def _load_metadata_files(paths: Iterable[str] | None) -> dict[str, object]:
    """Read JSON mappings from files and merge them into a single metadata dictionary."""

    metadata: dict[str, object] = {}
    if not paths:
        return metadata

    for raw_path in paths:
        path = Path(raw_path).expanduser().resolve()
        if not path.exists():
            raise SystemExit(f"Plik metadanych nie istnieje: {path}")
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise SystemExit(f"Plik metadanych {path} zawiera niepoprawny JSON: {exc}") from exc
        if not isinstance(payload, Mapping):
            raise SystemExit(f"Plik metadanych {path} musi zawierać obiekt JSON")
        for key, value in payload.items():
            if key in metadata:
                raise SystemExit(
                    f"Klucz {key} został zduplikowany między plikami metadanych lub wcześniejszymi wartościami"
                )
            if isinstance(value, Mapping) and not isinstance(value, dict):
                metadata[key] = dict(value)
            else:
                metadata[key] = value
    return metadata


def _build_ssl_context(
    *,
    ca_file: str | None,
    ca_path: str | None,
    client_cert: str | None,
    client_key: str | None,
) -> ssl.SSLContext | None:
    """Prepare SSL context for metadata downloads when custom material is supplied."""

    if client_key and not client_cert:
        raise SystemExit(
            "Opcja --metadata-url-client-key wymaga jednoczesnego podania --metadata-url-client-cert"
        )

    if not any([ca_file, ca_path, client_cert]):
        return None

    try:
        context = ssl.create_default_context(cafile=ca_file, capath=ca_path)
    except ssl.SSLError as exc:  # pragma: no cover - zależne od środowiska
        raise SystemExit(f"Nie można zainicjalizować kontekstu SSL dla metadanych: {exc}") from exc

    if client_cert:
        try:
            context.load_cert_chain(certfile=client_cert, keyfile=client_key)
        except ssl.SSLError as exc:  # pragma: no cover - zależne od środowiska
            raise SystemExit(
                f"Nie można załadować certyfikatu klienta {client_cert} dla metadanych URL: {exc}"
            ) from exc

    return context


def _load_metadata_from_urls(
    urls: Iterable[str] | None,
    *,
    headers: Mapping[str, str] | None = None,
    timeout: float | None = None,
    max_size: int | None = None,
    allow_insecure_http: bool = False,
    allowed_hosts: Iterable[str] | None = None,
    ca_file: str | None = None,
    ca_path: str | None = None,
    client_cert: str | None = None,
    client_key: str | None = None,
    cert_fingerprints: Mapping[str, set[str]] | None = None,
    cert_subject_requirements: Mapping[str, set[str]] | None = None,
    cert_issuer_requirements: Mapping[str, set[str]] | None = None,
    cert_san_requirements: Mapping[str, set[str]] | None = None,
    cert_extended_key_usage: set[str] | None = None,
    cert_policy_requirements: set[str] | None = None,
    cert_serial_requirements: set[str] | None = None,
) -> dict[str, object]:
    """Fetch JSON metadata mappings from remote URLs and merge them."""

    metadata: dict[str, object] = {}
    if not urls:
        return metadata

    effective_timeout = timeout if timeout is not None else 10.0
    if max_size is not None and max_size <= 0:
        raise SystemExit("Wielkość limitu --metadata-url-max-size musi być dodatnia")
    request_headers = dict(headers or {})

    normalized_hosts = {host.lower(): host for host in allowed_hosts or []}
    fingerprint_map = cert_fingerprints or {}
    subject_requirements = cert_subject_requirements or {}
    issuer_requirements = cert_issuer_requirements or {}
    san_requirements = cert_san_requirements or {}
    eku_requirements = cert_extended_key_usage or set()
    policy_requirements = cert_policy_requirements or set()
    ssl_context = _build_ssl_context(
        ca_file=ca_file,
        ca_path=ca_path,
        client_cert=client_cert,
        client_key=client_key,
    )

    for url in urls:
        parsed = urlparse.urlparse(url)
        if not parsed.scheme:
            raise SystemExit(f"Adres metadanych {url} nie zawiera schematu (np. https://)")
        scheme = parsed.scheme.lower()
        if scheme not in {"http", "https"}:
            raise SystemExit(
                f"Adres metadanych {url} używa nieobsługiwanego schematu {parsed.scheme}"
            )
        if scheme != "https" and not allow_insecure_http:
            raise SystemExit(
                "Adresy metadanych HTTP wymagają flagi --metadata-url-allow-http"
            )
        hostname = (parsed.hostname or "").lower()
        if normalized_hosts and hostname not in normalized_hosts:
            allowed = ", ".join(normalized_hosts.values())
            raise SystemExit(
                f"Host {hostname or '<brak>'} nie znajduje się na liście dozwolonych hostów: {allowed}"
            )

        try:
            request = urlrequest.Request(url, headers=request_headers)
            open_kwargs: dict[str, object] = {"timeout": effective_timeout}
            if scheme == "https" and ssl_context is not None:
                open_kwargs["context"] = ssl_context
            with urlrequest.urlopen(  # noqa: S310 - świadomie pozwalamy na URL
                request, **open_kwargs
            ) as response:
                status_code = getattr(response, "status", None) or response.getcode()
                if status_code is not None and not (200 <= status_code < 300):
                    raise SystemExit(
                        f"Źródło metadanych {url} zwróciło kod statusu {status_code}, oczekiwano 2xx"
                    )
                peer_cert: bytes | None = None
                if scheme == "https" and (
                    fingerprint_map
                    or subject_requirements
                    or issuer_requirements
                    or san_requirements
                    or eku_requirements
                    or policy_requirements
                    or cert_serial_requirements
                ):
                    try:
                        raw_socket = response.fp.raw._sock  # type: ignore[attr-defined]
                        peer_cert = raw_socket.getpeercert(True)
                    except AttributeError as exc:  # pragma: no cover - zależne od implementacji
                        raise SystemExit(
                            "Nie można uzyskać certyfikatu serwera dla dodatkowej weryfikacji TLS"
                        ) from exc

                    if not peer_cert:
                        raise SystemExit(
                            "Serwer HTTPS nie zwrócił certyfikatu do dodatkowej weryfikacji TLS"
                        )

                if scheme == "https" and fingerprint_map:
                    assert peer_cert is not None  # dla mypy

                    matched = False
                    for algorithm, expected_set in fingerprint_map.items():
                        digest = hashlib.new(algorithm, peer_cert).hexdigest()
                        if digest in expected_set:
                            matched = True
                            break

                    if not matched:
                        raise SystemExit(
                            "Certyfikat HTTPS źródła metadanych {} nie pasuje do żadnego dozwolonego odcisku"
                            .format(url)
                        )

                if scheme == "https" and subject_requirements:
                    assert peer_cert is not None
                    subject_map = _extract_certificate_subject(peer_cert)
                    for attribute, expected_values in subject_requirements.items():
                        actual_values = subject_map.get(attribute)
                        if not actual_values:
                            raise SystemExit(
                                "Certyfikat HTTPS źródła metadanych {} nie zawiera atrybutu tematu {}"
                                .format(url, attribute)
                            )
                        if not any(value in expected_values for value in actual_values):
                            raise SystemExit(
                                "Certyfikat HTTPS źródła metadanych {} ma atrybut {} bez wymaganych wartości"
                                .format(url, attribute)
                            )
                if scheme == "https" and issuer_requirements:
                    assert peer_cert is not None
                    issuer_map = _extract_certificate_issuer(peer_cert)
                    for attribute, expected_values in issuer_requirements.items():
                        actual_values = issuer_map.get(attribute)
                        if not actual_values:
                            raise SystemExit(
                                "Certyfikat HTTPS źródła metadanych {} został wystawiony przez CA bez atrybutu {}".format(
                                    url, attribute
                                )
                            )
                        if not any(value in expected_values for value in actual_values):
                            raise SystemExit(
                                "Certyfikat HTTPS źródła metadanych {} ma atrybut wystawcy {} bez wymaganych wartości"
                                .format(url, attribute)
                            )
                if scheme == "https" and san_requirements:
                    assert peer_cert is not None
                    san_map = _extract_certificate_subject_alternative_names(peer_cert)
                    for entry_type, expected_values in san_requirements.items():
                        actual_values = san_map.get(entry_type)
                        if not actual_values:
                            raise SystemExit(
                                "Certyfikat HTTPS źródła metadanych {} nie zawiera wpisu SAN typu {}"
                                .format(url, entry_type)
                            )
                        if not any(value in expected_values for value in actual_values):
                            raise SystemExit(
                                "Certyfikat HTTPS źródła metadanych {} ma SAN typu {} bez wymaganych wartości"
                                .format(url, entry_type)
                            )
                if scheme == "https" and eku_requirements:
                    assert peer_cert is not None
                    has_extension, eku_values = _extract_certificate_extended_key_usage(peer_cert)
                    if not has_extension:
                        raise SystemExit(
                            "Certyfikat HTTPS źródła metadanych {} nie zawiera rozszerzenia Extended Key Usage"
                            .format(url)
                        )
                    missing_eku = eku_requirements - eku_values
                    if missing_eku:
                        missing_text = ", ".join(sorted(missing_eku))
                        raise SystemExit(
                            "Certyfikat HTTPS źródła metadanych {} nie zawiera wymaganych EKU: {}"
                            .format(url, missing_text)
                        )
                if scheme == "https" and policy_requirements:
                    assert peer_cert is not None
                    has_extension, policy_values = _extract_certificate_policies(peer_cert)
                    if not has_extension:
                        raise SystemExit(
                            "Certyfikat HTTPS źródła metadanych {} nie zawiera rozszerzenia certificatePolicies"
                            .format(url)
                        )
                    missing_policies = policy_requirements - policy_values
                    if missing_policies:
                        missing_text = ", ".join(sorted(missing_policies))
                        raise SystemExit(
                            "Certyfikat HTTPS źródła metadanych {} nie zawiera wymaganych polityk: {}"
                            .format(url, missing_text)
                        )
                if scheme == "https" and cert_serial_requirements:
                    assert peer_cert is not None
                    serial_number = _extract_certificate_serial_number(peer_cert)
                    if serial_number not in cert_serial_requirements:
                        raise SystemExit(
                            "Certyfikat HTTPS źródła metadanych {} ma numer seryjny nieobecny na liście dozwolonych"
                            .format(url)
                        )
                content_type = response.headers.get("Content-Type")
                if content_type and "json" not in content_type.lower():
                    raise SystemExit(
                        f"Źródło metadanych {url} zadeklarowało Content-Type {content_type}, oczekiwano JSON"
                    )
                if max_size is not None:
                    content_length = response.headers.get("Content-Length")
                    if content_length is not None:
                        try:
                            declared_size = int(content_length)
                        except ValueError:
                            raise SystemExit(
                                f"Źródło metadanych {url} podało nieprawidłowy Content-Length: {content_length}"
                            ) from None
                        if declared_size > max_size:
                            raise SystemExit(
                                f"Odpowiedź metadanych z {url} przekracza limit {max_size} bajtów (Content-Length)"
                            )
                    payload_bytes = response.read(max_size + 1)
                    if len(payload_bytes) > max_size:
                        raise SystemExit(
                            f"Odpowiedź metadanych z {url} przekroczyła limit {max_size} bajtów"
                        )
                else:
                    payload_bytes = response.read()
                charset = response.headers.get_content_charset() or "utf-8"
        except urlerror.URLError as exc:
            raise SystemExit(f"Nie można pobrać metadanych z {url}: {exc}") from exc

        try:
            payload_text = payload_bytes.decode(charset)
        except (LookupError, UnicodeDecodeError) as exc:
            raise SystemExit(
                f"Nie można zdekodować odpowiedzi metadanych z {url} przy użyciu {charset}: {exc}"
            ) from exc

        try:
            payload = json.loads(payload_text)
        except json.JSONDecodeError as exc:
            raise SystemExit(f"Odpowiedź metadanych z {url} nie zawiera poprawnego JSON: {exc}") from exc

        if not isinstance(payload, Mapping):
            raise SystemExit(f"Odpowiedź metadanych z {url} musi zawierać obiekt JSON")

        for key, value in payload.items():
            if key in metadata:
                raise SystemExit(
                    f"Klucz {key} został zduplikowany między adresami URL metadanych lub wcześniejszymi wartościami"
                )
            if isinstance(value, Mapping) and not isinstance(value, dict):
                metadata[key] = dict(value)
            else:
                metadata[key] = value

    return metadata


def _load_metadata_from_ini(paths: Iterable[str] | None) -> dict[str, object]:
    """Read INI mappings and merge them into metadata entries."""

    metadata: dict[str, object] = {}
    if not paths:
        return metadata

    for raw_path in paths:
        path = Path(raw_path).expanduser().resolve()
        if not path.exists():
            raise SystemExit(f"Plik INI z metadanymi nie istnieje: {path}")

        parser = configparser.ConfigParser(interpolation=None, strict=True)
        parser.optionxform = str
        raw_text = path.read_text(encoding="utf-8")
        try:
            parser.read_string(raw_text)
        except configparser.MissingSectionHeaderError:
            parser = configparser.ConfigParser(interpolation=None, strict=True)
            parser.optionxform = str
            try:
                parser.read_string("[DEFAULT]\n" + raw_text)
            except configparser.Error as exc:  # pragma: no cover - ochronna gałąź
                raise SystemExit(f"Plik INI {path} zawiera niepoprawną składnię: {exc}") from exc
        except configparser.Error as exc:
            raise SystemExit(f"Plik INI {path} zawiera niepoprawną składnię: {exc}") from exc

        def _store(full_key: str, raw_value: str) -> None:
            normalized_key = full_key.replace("__", ".")
            if normalized_key in metadata:
                raise SystemExit(
                    "Klucz {key} został zduplikowany w plikach INI lub wcześniejszych źródłach metadanych"
                    .format(key=normalized_key)
                )
            candidate = raw_value.strip()
            try:
                value: object = json.loads(candidate)
            except json.JSONDecodeError:
                value = candidate
            metadata[normalized_key] = value

        for key, raw_value in parser.defaults().items():
            _store(key, raw_value)

        sections = getattr(parser, "_sections")  # type: ignore[attr-defined]
        for section, entries in sections.items():
            normalized_section = section.replace("__", ".")
            for key, raw_value in entries.items():
                if key == "__name__":
                    continue
                full_key = f"{normalized_section}.{key}"
                _store(full_key, raw_value)

    return metadata


def _load_metadata_from_toml(paths: Iterable[str] | None) -> dict[str, object]:
    """Read TOML mappings and merge them into metadata entries."""

    metadata: dict[str, object] = {}
    if not paths:
        return metadata

    for raw_path in paths:
        path = Path(raw_path).expanduser().resolve()
        if not path.exists():
            raise SystemExit(f"Plik TOML z metadanymi nie istnieje: {path}")
        try:
            payload = tomllib.loads(path.read_text(encoding="utf-8"))
        except tomllib.TOMLDecodeError as exc:
            raise SystemExit(f"Plik TOML {path} zawiera niepoprawną składnię: {exc}") from exc
        if not isinstance(payload, Mapping):
            raise SystemExit(f"Plik TOML {path} musi zawierać tabelę mapującą na metadane")
        for key, value in payload.items():
            if key in metadata:
                raise SystemExit(
                    f"Klucz {key} został zduplikowany między plikami TOML lub wcześniejszymi wartościami"
                )
            if isinstance(value, Mapping) and not isinstance(value, dict):
                metadata[key] = dict(value)
            else:
                metadata[key] = value

    return metadata


def _load_metadata_from_yaml(paths: Iterable[str] | None) -> dict[str, object]:
    """Read YAML mappings and merge them into metadata entries."""

    metadata: dict[str, object] = {}
    if not paths:
        return metadata

    for raw_path in paths:
        path = Path(raw_path).expanduser().resolve()
        if not path.exists():
            raise SystemExit(f"Plik YAML z metadanymi nie istnieje: {path}")
        try:
            payload = yaml.safe_load(path.read_text(encoding="utf-8"))
        except yaml.YAMLError as exc:  # pragma: no cover - trudne do uzyskania w testach
            raise SystemExit(f"Plik YAML {path} zawiera niepoprawną składnię: {exc}") from exc
        if payload is None:
            continue
        if not isinstance(payload, Mapping):
            raise SystemExit(f"Plik YAML {path} musi zawierać obiekt mapujący na metadane")
        for key, value in payload.items():
            if key in metadata:
                raise SystemExit(
                    f"Klucz {key} został zduplikowany między plikami YAML lub wcześniejszymi wartościami"
                )
            if isinstance(value, Mapping) and not isinstance(value, dict):
                metadata[key] = dict(value)
            else:
                metadata[key] = value

    return metadata


def _load_metadata_from_env(prefixes: Iterable[str] | None) -> dict[str, object]:
    """Collect metadata entries from environment variables using provided prefixes."""

    metadata: dict[str, object] = {}
    if not prefixes:
        return metadata

    for prefix in prefixes:
        if not prefix:
            raise SystemExit("Prefiks --metadata-env-prefix nie może być pusty")
        for name, raw_value in os.environ.items():
            if not name.startswith(prefix):
                continue
            key = name[len(prefix) :]
            if not key:
                raise SystemExit(
                    f"Zmienna środowiskowa {name} nie zawiera klucza po prefiksie {prefix}"
                )
            normalized_key = key.replace("__", ".")
            if normalized_key in metadata:
                raise SystemExit(
                    f"Klucz {normalized_key} został zduplikowany w zmiennych środowiskowych"
                )
            try:
                value: object = json.loads(raw_value)
            except json.JSONDecodeError:
                value = raw_value
            metadata[normalized_key] = value

    return metadata


def _load_metadata_from_dotenv(paths: Iterable[str] | None) -> dict[str, object]:
    """Collect metadata entries from dotenv-style files."""

    metadata: dict[str, object] = {}
    if not paths:
        return metadata

    for raw_path in paths:
        path = Path(raw_path).expanduser().resolve()
        if not path.exists():
            raise SystemExit(f"Plik dotenv z metadanymi nie istnieje: {path}")
        for index, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            if stripped.startswith("export "):
                stripped = stripped[len("export ") :].lstrip()
            key, sep, raw_value = stripped.partition("=")
            if sep != "=":
                raise SystemExit(
                    f"Linia {index} w pliku {path} nie ma formatu KEY=VALUE i nie może zostać użyta"
                )
            key = key.strip()
            if not key:
                raise SystemExit(f"Linia {index} w pliku {path} nie zawiera nazwy klucza")
            normalized_key = key.replace("__", ".")
            value_candidate = raw_value.strip()
            try:
                value: object = json.loads(value_candidate)
            except json.JSONDecodeError:
                value = value_candidate
            if normalized_key in metadata:
                raise SystemExit(
                    f"Klucz {normalized_key} został zduplikowany w pliku dotenv {path} lub poprzednich plikach"
                )
            metadata[normalized_key] = value

    return metadata


def _run_pyinstaller(entrypoint: Path, workdir: Path, binary_name: str, *, hidden_imports: Iterable[str]) -> Path:
    output_dir = workdir / "pyinstaller"
    build_dir = output_dir / "build"
    dist_dir = output_dir / "dist"
    output_dir.mkdir(parents=True, exist_ok=True)

    args = [
        "pyinstaller",
        "--clean",
        "--noconfirm",
        "--name",
        binary_name,
        "--distpath",
        str(dist_dir),
        "--workpath",
        str(build_dir),
    ]
    for hidden_import in hidden_imports:
        args.extend(["--hidden-import", hidden_import])
    args.append(str(entrypoint))

    subprocess.run(args, check=True)

    extension = ".exe" if sys.platform.startswith("win") else ""
    candidate = dist_dir / binary_name
    if candidate.is_dir():
        executable = candidate / f"{binary_name}{extension}"
    else:
        executable = candidate.with_suffix(extension)

    if not executable.exists():
        raise RuntimeError(f"PyInstaller nie wygenerował pliku wykonywalnego pod {executable}")

    return executable


def _run_briefcase(project_path: Path, platform: str, workdir: Path) -> Path:
    """Invoke Briefcase to build the Qt client bundle."""

    build_dir = workdir / "briefcase"
    build_dir.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env.setdefault("BRIEFCASE_ROOT", str(build_dir))

    subprocess.run(["briefcase", "create", platform], cwd=project_path, env=env, check=True)
    subprocess.run(["briefcase", "build", platform], cwd=project_path, env=env, check=True)
    subprocess.run(["briefcase", "package", platform], cwd=project_path, env=env, check=True)

    artifacts_dir = project_path / "build" / platform
    if not artifacts_dir.exists():
        raise RuntimeError(f"Briefcase nie wygenerował artefaktów dla platformy {platform}")
    return artifacts_dir


def _stage_tree(source: Path, destination: Path) -> list[ArtifactSpec]:
    artifacts: list[ArtifactSpec] = []
    if not source.exists():
        return artifacts
    if destination.exists():
        shutil.rmtree(destination)
    shutil.copytree(source, destination)
    for item in destination.rglob("*"):
        if item.is_file():
            relative = item.relative_to(destination.parent)
            artifacts.append(ArtifactSpec(bundle_path=relative, source_path=item))
    return artifacts


def _copy_file(source: Path, destination: Path) -> ArtifactSpec:
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
    bundle_path = destination.relative_to(destination.parents[1])
    return ArtifactSpec(bundle_path=bundle_path, source_path=destination)


def build_bundle(args: argparse.Namespace) -> Path:
    output_dir = Path(args.output_dir).expanduser().resolve()
    workdir = Path(args.workdir).expanduser().resolve()
    entrypoint = Path(args.entrypoint).expanduser().resolve()
    qt_dist = Path(args.qt_dist).expanduser().resolve() if args.qt_dist else None
    briefcase_project = Path(args.briefcase_project).expanduser().resolve() if args.briefcase_project else None

    layout = _resolve_layout(output_dir, args.version, args.platform)
    artifacts: list[ArtifactSpec] = []

    hidden_imports = args.hidden_import or []
    binary_name = args.runtime_name or "bot_core_runtime"
    runtime_executable = _run_pyinstaller(entrypoint, workdir, binary_name, hidden_imports=hidden_imports)
    runtime_dest = layout.daemon_dir / runtime_executable.name
    artifacts.append(_copy_file(runtime_executable, runtime_dest))

    if briefcase_project:
        briefcase_artifacts = _run_briefcase(briefcase_project, args.platform, workdir)
        artifacts.extend(_stage_tree(briefcase_artifacts, layout.ui_dir))
    elif qt_dist:
        artifacts.extend(_stage_tree(qt_dist, layout.ui_dir))

    for include in args.include or []:
        name, _, path = include.partition("=")
        if not name or not path:
            raise SystemExit(f"Niepoprawny format --include: {include}. Wymagany zapis <nazwa>=<ścieżka>.")
        source = Path(path).expanduser().resolve()
        destination = layout.extras_dir / name
        if source.is_dir():
            artifacts.extend(_stage_tree(source, destination))
        else:
            artifacts.append(_copy_file(source, destination))

    license_artifacts = _embed_encrypted_license(
        layout,
        license_json=getattr(args, "license_json", None),
        fingerprint=getattr(args, "license_fingerprint", None),
        output_name=getattr(args, "license_output_name", "license_store.json"),
        hmac_key=_parse_license_hmac_key(getattr(args, "license_hmac_key", None)),
    )
    artifacts.extend(license_artifacts)

    integrity_manifest_spec = _write_integrity_manifest(layout)
    artifacts.append(integrity_manifest_spec)

    metadata = _load_metadata_files(getattr(args, "metadata_file", None))
    url_headers = _parse_http_headers(
        getattr(args, "metadata_url_header", None), option="--metadata-url-header"
    )
    fingerprint_map = _parse_cert_fingerprints(
        getattr(args, "metadata_url_cert_fingerprint", None),
        option="--metadata-url-cert-fingerprint",
    )
    subject_requirements = _parse_cert_subject_requirements(
        getattr(args, "metadata_url_cert_subject", None),
        option="--metadata-url-cert-subject",
    )
    issuer_requirements = _parse_cert_issuer_requirements(
        getattr(args, "metadata_url_cert_issuer", None),
        option="--metadata-url-cert-issuer",
    )
    san_requirements = _parse_cert_san_requirements(
        getattr(args, "metadata_url_cert_san", None),
        option="--metadata-url-cert-san",
    )
    eku_requirements = _parse_cert_extended_key_usage(
        getattr(args, "metadata_url_cert_eku", None),
        option="--metadata-url-cert-eku",
    )
    policy_requirements = _parse_cert_policy_requirements(
        getattr(args, "metadata_url_cert_policy", None),
        option="--metadata-url-cert-policy",
    )
    serial_requirements = _parse_cert_serial_requirements(
        getattr(args, "metadata_url_cert_serial", None),
        option="--metadata-url-cert-serial",
    )
    raw_timeout = getattr(args, "metadata_url_timeout", None)
    timeout_value: float | None
    if raw_timeout is None:
        timeout_value = None
    else:
        try:
            timeout_value = float(raw_timeout)
        except (TypeError, ValueError) as exc:  # pragma: no cover - defensive path
            raise SystemExit("Wartość --metadata-url-timeout musi być liczbą dodatnią") from exc
        if timeout_value <= 0:
            raise SystemExit("Wartość --metadata-url-timeout musi być dodatnia")
    raw_max_size = getattr(args, "metadata_url_max_size", None)
    max_size_value: int | None
    if raw_max_size is None:
        max_size_value = None
    else:
        try:
            max_size_value = int(raw_max_size)
        except (TypeError, ValueError) as exc:  # pragma: no cover - defensive path
            raise SystemExit("Wartość --metadata-url-max-size musi być liczbą dodatnią") from exc
        if max_size_value <= 0:
            raise SystemExit("Wartość --metadata-url-max-size musi być dodatnia")
    ca_file_option = getattr(args, "metadata_url_ca", None)
    ca_file_path: Path | None = None
    if ca_file_option:
        ca_file_path = Path(ca_file_option).expanduser().resolve()
        if not ca_file_path.exists():
            raise SystemExit(f"Plik CA {ca_file_path} podany w --metadata-url-ca nie istnieje")
        if not ca_file_path.is_file():
            raise SystemExit(f"Ścieżka {ca_file_path} z --metadata-url-ca nie jest plikiem")
    ca_path_option = getattr(args, "metadata_url_capath", None)
    ca_path_dir: Path | None = None
    if ca_path_option:
        ca_path_dir = Path(ca_path_option).expanduser().resolve()
        if not ca_path_dir.exists():
            raise SystemExit(f"Katalog CA {ca_path_dir} podany w --metadata-url-capath nie istnieje")
        if not ca_path_dir.is_dir():
            raise SystemExit(f"Ścieżka {ca_path_dir} z --metadata-url-capath nie jest katalogiem")
    client_cert_option = getattr(args, "metadata_url_client_cert", None)
    client_cert_path: Path | None = None
    if client_cert_option:
        client_cert_path = Path(client_cert_option).expanduser().resolve()
        if not client_cert_path.exists():
            raise SystemExit(
                f"Certyfikat klienta {client_cert_path} podany w --metadata-url-client-cert nie istnieje"
            )
        if not client_cert_path.is_file():
            raise SystemExit(
                f"Ścieżka {client_cert_path} z --metadata-url-client-cert nie jest plikiem"
            )
    client_key_option = getattr(args, "metadata_url_client_key", None)
    client_key_path: Path | None = None
    if client_key_option:
        client_key_path = Path(client_key_option).expanduser().resolve()
        if not client_key_path.exists():
            raise SystemExit(
                f"Klucz prywatny {client_key_path} podany w --metadata-url-client-key nie istnieje"
            )
        if not client_key_path.is_file():
            raise SystemExit(
                f"Ścieżka {client_key_path} z --metadata-url-client-key nie jest plikiem"
            )
    url_metadata = _load_metadata_from_urls(
        getattr(args, "metadata_url", None),
        headers=url_headers,
        timeout=timeout_value,
        max_size=max_size_value,
        allow_insecure_http=bool(getattr(args, "metadata_url_allow_http", False)),
        allowed_hosts=getattr(args, "metadata_url_allowed_host", None),
        ca_file=str(ca_file_path) if ca_file_path else None,
        ca_path=str(ca_path_dir) if ca_path_dir else None,
        client_cert=str(client_cert_path) if client_cert_path else None,
        client_key=str(client_key_path) if client_key_path else None,
        cert_fingerprints=fingerprint_map,
        cert_subject_requirements=subject_requirements,
        cert_issuer_requirements=issuer_requirements,
        cert_san_requirements=san_requirements,
        cert_extended_key_usage=eku_requirements,
        cert_policy_requirements=policy_requirements,
        cert_serial_requirements=serial_requirements,
    )
    for key, value in url_metadata.items():
        if "." in key:
            _apply_metadata_entry(metadata, key, value)
            continue
        if key in metadata:
            raise SystemExit(
                f"Klucz {key} z adresów URL metadanych został już zdefiniowany w źródłach metadanych"
            )
        metadata[key] = value
    ini_metadata = _load_metadata_from_ini(getattr(args, "metadata_ini", None))
    for key, value in ini_metadata.items():
        if "." in key:
            _apply_metadata_entry(metadata, key, value)
            continue
        if key in metadata:
            raise SystemExit(
                f"Klucz {key} z plików INI został już zdefiniowany w źródłach metadanych"
            )
        metadata[key] = value

    toml_metadata = _load_metadata_from_toml(getattr(args, "metadata_toml", None))
    for key, value in toml_metadata.items():
        if "." in key:
            _apply_metadata_entry(metadata, key, value)
            continue
        if key in metadata:
            raise SystemExit(
                f"Klucz {key} z plików TOML został już zdefiniowany w źródłach metadanych"
            )
        metadata[key] = value

    yaml_metadata = _load_metadata_from_yaml(getattr(args, "metadata_yaml", None))
    for key, value in yaml_metadata.items():
        if "." in key:
            _apply_metadata_entry(metadata, key, value)
            continue
        if key in metadata:
            raise SystemExit(
                f"Klucz {key} z plików YAML został już zdefiniowany w źródłach metadanych"
            )
        metadata[key] = value

    dotenv_metadata = _load_metadata_from_dotenv(getattr(args, "metadata_dotenv", None))
    for key, value in dotenv_metadata.items():
        if "." in key:
            _apply_metadata_entry(metadata, key, value)
            continue
        if key in metadata:
            raise SystemExit(
                f"Klucz {key} z plików dotenv został już zdefiniowany w plikach metadanych"
            )
        metadata[key] = value

    env_metadata = _load_metadata_from_env(getattr(args, "metadata_env_prefix", None))
    for key, value in env_metadata.items():
        if "." in key:
            _apply_metadata_entry(metadata, key, value)
            continue
        if key in metadata:
            raise SystemExit(
                f"Klucz {key} z zmiennych środowiskowych został już zdefiniowany w plikach metadanych"
            )
        metadata[key] = value
    cli_metadata = _parse_key_value_pairs(getattr(args, "metadata", None), option="--metadata")
    for key, value in cli_metadata.items():
        if "." in key:
            _apply_metadata_entry(metadata, key, value)
            continue
        if key in metadata:
            raise SystemExit(
                f"Klucz {key} z opcji --metadata został już zdefiniowany w źródłach metadanych"
            )
        metadata[key] = value
    manifest: dict[str, object] = {
        "version": args.version,
        "platform": args.platform,
        "runtime": runtime_dest.name,
        "artifacts": _collect_artifact_metadata(artifacts),
        "generator": "build_pyinstaller_bundle",
    }

    if metadata:
        manifest["metadata"] = metadata

    allowed_profiles = getattr(args, "allowed_profile", None) or []
    if allowed_profiles:
        manifest["allowed_profiles"] = list(dict.fromkeys(allowed_profiles))

    manifest_path = layout.root / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    if args.signing_key:
        key_candidate = Path(args.signing_key)
        key_bytes = key_candidate.read_bytes() if key_candidate.exists() else args.signing_key.encode("utf-8")
        signature = build_hmac_signature(payload=manifest, key=key_bytes, key_id=args.signing_key_id)
        signature_path = layout.root / "manifest.sig"
        signature_path.write_text(json.dumps(signature, ensure_ascii=False, indent=2), encoding="utf-8")

    archive_path = layout.root.parent / f"{layout.root.name}.zip"
    if archive_path.exists():
        archive_path.unlink()
    shutil.make_archive(str(layout.root), "zip", layout.root)
    return archive_path


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build PyInstaller bundle with Qt UI")
    parser.add_argument("--entrypoint", default="scripts/run_multi_strategy_scheduler.py", help="Python entrypoint for PyInstaller")
    parser.add_argument("--qt-dist", help="Path to compiled Qt distribution (Release directory)")
    parser.add_argument("--briefcase-project", help="Path to Briefcase project to build the Qt shell")
    parser.add_argument("--platform", required=True, help="Target platform identifier")
    parser.add_argument("--version", required=True, help="Bundle version string")
    parser.add_argument("--output-dir", default="var/dist", help="Root directory for generated bundles")
    parser.add_argument("--workdir", default="var/build", help="Temporary work directory for build tools")
    parser.add_argument("--hidden-import", action="append", help="Additional hidden imports for PyInstaller")
    parser.add_argument("--runtime-name", help="Override runtime binary name")
    parser.add_argument("--include", action="append", help="Additional resources to include <name>=<path>")
    parser.add_argument(
        "--license-json",
        help="Offline license JSON that should be encrypted and bundled",
    )
    parser.add_argument(
        "--license-fingerprint",
        help="Target machine fingerprint used to encrypt the license store",
    )
    parser.add_argument(
        "--license-output-name",
        default="license_store.json",
        help="Output filename of the encrypted license store inside the bundle",
    )
    parser.add_argument(
        "--license-hmac-key",
        help="Optional HMAC key (KEY_ID=SECRET) for signing license integrity metadata",
    )
    parser.add_argument("--signing-key", help="Secret value or path used to sign manifest.json")
    parser.add_argument("--signing-key-id", help="Identifier added to manifest signature")
    parser.add_argument("--metadata", action="append", help="Add metadata entry <key>=<value> (value may be JSON)")
    parser.add_argument("--metadata-file", action="append", help="Load additional metadata from a JSON file")
    parser.add_argument(
        "--metadata-url",
        action="append",
        help="Pobierz dodatkowe metadane z adresu URL zwracającego obiekt JSON",
    )
    parser.add_argument(
        "--metadata-url-header",
        action="append",
        help="Dodaj nagłówek HTTP do zapytań --metadata-url w formacie Nazwa=Wartość",
    )
    parser.add_argument(
        "--metadata-url-timeout",
        type=float,
        help="Limit czasu w sekundach na pobranie metadanych z URL (domyślnie 10)",
    )
    parser.add_argument(
        "--metadata-url-max-size",
        type=int,
        help="Maksymalny rozmiar (w bajtach) odpowiedzi metadanych pobieranej z URL",
    )
    parser.add_argument(
        "--metadata-url-allow-http",
        action="store_true",
        help="Zezwól na pobieranie metadanych przez HTTP (domyślnie wymagane HTTPS)",
    )
    parser.add_argument(
        "--metadata-url-allowed-host",
        action="append",
        help="Ogranicz pobieranie metadanych do wskazanych hostów (można podać wiele)",
    )
    parser.add_argument(
        "--metadata-url-cert-fingerprint",
        action="append",
        help=(
            "Zaufaj tylko serwerom HTTPS z podanym odciskiem certyfikatu w formacie algorytm:HEX "
            "(obsługiwane algorytmy: sha256, sha384, sha512)"
        ),
    )
    parser.add_argument(
        "--metadata-url-cert-subject",
        action="append",
        help=(
            "Zaufaj tylko certyfikatom TLS zawierającym wskazane atrybuty tematu w formacie "
            "Atrybut=Wartość (np. commonName=updates.example.com)"
        ),
    )
    parser.add_argument(
        "--metadata-url-cert-issuer",
        action="append",
        help=(
            "Zaufaj tylko certyfikatom TLS wystawionym przez CA zawierające wskazane atrybuty "
            "wystawcy w formacie Atrybut=Wartość (np. organizationName=Trusted CA)"
        ),
    )
    parser.add_argument(
        "--metadata-url-cert-san",
        action="append",
        help=(
            "Zaufaj tylko certyfikatom TLS zawierającym wpisy subjectAltName w formacie "
            "Typ=Wartość (np. DNS=updates.example.com)"
        ),
    )
    parser.add_argument(
        "--metadata-url-cert-eku",
        action="append",
        help=(
            "Zaufaj tylko certyfikatom TLS zawierającym wymagane rozszerzenia Extended Key Usage "
            "(np. serverAuth lub OID)"
        ),
    )
    parser.add_argument(
        "--metadata-url-cert-policy",
        action="append",
        help=(
            "Zaufaj tylko certyfikatom TLS zawierającym wymagane identyfikatory certificatePolicies "
            "(np. anyPolicy lub OID)"
        ),
    )
    parser.add_argument(
        "--metadata-url-cert-serial",
        action="append",
        help=(
            "Zaufaj tylko certyfikatom TLS z numerami seryjnymi z listy dozwolonych (format dziesiętny," \
            " szesnastkowy lub z separacją dwukropkami)"
        ),
    )
    parser.add_argument(
        "--metadata-url-ca",
        help="Ścieżka do dodatkowego pliku z certyfikatem CA używanym przy pobieraniu metadanych",
    )
    parser.add_argument(
        "--metadata-url-capath",
        help="Katalog z plikami certyfikatów CA używany przy pobieraniu metadanych",
    )
    parser.add_argument(
        "--metadata-url-client-cert",
        help="Certyfikat klienta (PEM) używany do uwierzytelnienia przy pobieraniu metadanych",
    )
    parser.add_argument(
        "--metadata-url-client-key",
        help="Klucz prywatny (PEM) odpowiadający certyfikatowi klienta metadanych",
    )
    parser.add_argument(
        "--metadata-ini",
        action="append",
        help="Wczytaj metadane z pliku INI (obsługa sekcji i kluczy z __ jako separator kropki)",
    )
    parser.add_argument(
        "--metadata-toml",
        action="append",
        help="Wczytaj metadane z pliku TOML (wartości interpretowane jako JSON, wspiera klucze kropkowane)",
    )
    parser.add_argument(
        "--metadata-yaml",
        action="append",
        help="Wczytaj metadane z pliku YAML (wartości interpretowane jako JSON, wspiera klucze kropkowane)",
    )
    parser.add_argument(
        "--metadata-dotenv",
        action="append",
        help="Ścieżka do pliku .env z wpisami metadanych (wspiera klucze z __ jako separator kropki)",
    )
    parser.add_argument(
        "--metadata-env-prefix",
        action="append",
        help=(
            "Zbierz metadane ze zmiennych środowiskowych rozpoczynających się od prefiksu; "
            "po prefiksie użyj __ jako separatora segmentów klucza"
        ),
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    try:
        archive = build_bundle(args)
    except subprocess.CalledProcessError as exc:  # pragma: no cover - CLI failure path
        raise SystemExit(f"Proces budowania zakończył się błędem: {exc}") from exc
    except Exception as exc:  # pragma: no cover - CLI diagnostics
        raise SystemExit(str(exc)) from exc
    print(f"Zbudowano bundla: {archive}")
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry
    raise SystemExit(main())
