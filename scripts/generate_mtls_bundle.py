"""Generuje pakiet certyfikatów mTLS (CA, serwer, klient) z audytem i wsparciem rotacji kluczy."""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from ipaddress import ip_address
from pathlib import Path
from typing import Iterable, Mapping

# --- opcjonalne zależności ---------------------------------------------------
try:  # cryptography – preferowany backend
    from cryptography import x509
    from cryptography.hazmat.backends import default_backend
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import rsa
    from cryptography.x509.oid import ExtendedKeyUsageOID, NameOID

    _HAS_CRYPTO = True
except Exception:  # pragma: no cover
    _HAS_CRYPTO = False

try:  # audyt TLS (opcjonalny)
    from bot_core.security.tls_audit import audit_mtls_bundle as _audit_mtls_bundle  # type: ignore
except Exception:  # pragma: no cover
    _audit_mtls_bundle = None  # type: ignore

try:  # metadane referencyjne certyfikatów (opcjonalne)
    from bot_core.security.certificates import certificate_reference_metadata  # type: ignore
except Exception:  # pragma: no cover
    def certificate_reference_metadata(path: Path, *, role: str) -> Mapping[str, object]:  # type: ignore
        return {"role": role, "path": str(path)}

from bot_core.security.rotation import RotationRegistry

DEFAULT_HOSTNAMES = ("127.0.0.1", "localhost")


def _parse_subject(subject: str) -> dict[str, str]:
    parts: dict[str, str] = {}
    for chunk in subject.split("/"):
        chunk = chunk.strip()
        if not chunk:
            continue
        key, _, value = chunk.partition("=")
        key = key.strip().upper()
        value = value.strip()
        if key and value:
            parts[key] = value
    return parts


def _slugify(value: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", value.lower())
    slug = slug.strip("-")
    return slug or "mtls-bundle"


@dataclass(slots=True)
class BundleConfig:
    output_dir: Path
    bundle_name: str
    common_name: str
    organization: str
    valid_days: int
    key_size: int
    server_hostnames: tuple[str, ...]
    rotation_registry: Path | None
    ca_key_passphrase: bytes | None
    server_key_passphrase: bytes | None
    client_key_passphrase: bytes | None


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------
def _parse_args(argv: Iterable[str] | None = None) -> BundleConfig:
    parser = argparse.ArgumentParser(description="Generuje pakiet certyfikatów mTLS (CA/server/client).")
    parser.add_argument("--output-dir", required=True, help="Katalog docelowy na wygenerowany pakiet")
    parser.add_argument("--bundle-name", default="core-oem", help="Identyfikator pakietu (prefiks plików)")
    parser.add_argument("--common-name", default="Dudzian OEM", help="CN używany dla certyfikatów")
    parser.add_argument("--organization", default="Dudzian", help="Pole O w certyfikatach")
    parser.add_argument("--valid-days", type=int, default=365, help="Okres ważności certyfikatów w dniach")
    parser.add_argument("--key-size", type=int, default=4096, help="Rozmiar klucza RSA w bitach")
    parser.add_argument(
        "--server-hostname",
        action="append",
        dest="server_hostnames",
        help="Hostname/IP do SAN certyfikatu serwerowego (można podać wielokrotnie)",
    )
    parser.add_argument(
        "--rotation-registry",
        help="Ścieżka do rejestru rotacji TLS (JSON). Brak oznacza pominięcie aktualizacji.",
    )
    parser.add_argument(
        "--ca-key-passphrase-env",
        help="Nazwa ENV z passphrase dla klucza CA (PKCS#8).",
    )
    parser.add_argument(
        "--server-key-passphrase-env",
        help="Nazwa ENV z passphrase dla klucza serwera (UWAGA: gRPC zwykle wymaga kluczy bez hasła).",
    )
    parser.add_argument(
        "--client-key-passphrase-env",
        help="Nazwa ENV z passphrase dla klucza klienta.",
    )
    args = parser.parse_args(argv)

    output_dir = Path(args.output_dir).expanduser().resolve()
    hostnames = tuple(args.server_hostnames) if args.server_hostnames else DEFAULT_HOSTNAMES

    return BundleConfig(
        output_dir=output_dir,
        bundle_name=str(args.bundle_name),
        common_name=str(args.common_name),
        organization=str(args.organization),
        valid_days=max(1, int(args.valid_days)),
        key_size=max(2048, int(args.key_size)),
        server_hostnames=hostnames,
        rotation_registry=Path(args.rotation_registry).expanduser().resolve() if args.rotation_registry else None,
        ca_key_passphrase=_env_passphrase(args.ca_key_passphrase_env),
        server_key_passphrase=_env_passphrase(args.server_key_passphrase_env),
        client_key_passphrase=_env_passphrase(args.client_key_passphrase_env),
    )


def _env_passphrase(env_name: str | None) -> bytes | None:
    if not env_name:
        return None
    value = os.environ.get(env_name)
    if value is None:
        raise SystemExit(f"Zmienna środowiskowa {env_name} nie jest ustawiona")
    if not value:
        raise SystemExit(f"Zmienna {env_name} zawiera pusty passphrase")
    return value.encode("utf-8")


# -----------------------------------------------------------------------------
# Backend 1: cryptography (preferowany)
# -----------------------------------------------------------------------------
def _generate_with_cryptography(config: BundleConfig) -> Mapping[str, Path]:
    def _generate_private_key(key_size: int) -> rsa.RSAPrivateKey:
        return rsa.generate_private_key(public_exponent=65537, key_size=key_size, backend=default_backend())

    def _serialize_key(key: rsa.RSAPrivateKey, passphrase: bytes | None) -> bytes:
        if passphrase:
            encryption = serialization.BestAvailableEncryption(passphrase)
        else:
            encryption = serialization.NoEncryption()
        return key.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.PKCS8,
            encryption_algorithm=encryption,
        )

    def _build_name(cn: str, org: str) -> x509.Name:
        return x509.Name(
            [x509.NameAttribute(NameOID.COMMON_NAME, cn), x509.NameAttribute(NameOID.ORGANIZATION_NAME, org)]
        )

    def _issue_certificate(
        *,
        subject: x509.Name,
        issuer: x509.Name,
        public_key,
        issuer_key,
        serial: int,
        valid_from: datetime,
        valid_to: datetime,
        is_ca: bool,
        hostnames: Iterable[str] = (),
        usages: Iterable[x509.ExtendedKeyUsageOID] = (),
    ) -> x509.Certificate:
        builder = (
            x509.CertificateBuilder()
            .subject_name(subject)
            .issuer_name(issuer)
            .public_key(public_key)
            .serial_number(serial)
            .not_valid_before(valid_from)
            .not_valid_after(valid_to)
        )
        if is_ca:
            builder = builder.add_extension(x509.BasicConstraints(ca=True, path_length=1), critical=True)
            builder = builder.add_extension(
                x509.KeyUsage(
                    digital_signature=True,
                    content_commitment=True,
                    key_encipherment=False,
                    data_encipherment=False,
                    key_agreement=False,
                    key_cert_sign=True,
                    crl_sign=True,
                    encipher_only=False,
                    decipher_only=False,
                ),
                critical=True,
            )
        else:
            builder = builder.add_extension(x509.BasicConstraints(ca=False, path_length=None), critical=True)
            builder = builder.add_extension(
                x509.KeyUsage(
                    digital_signature=True,
                    content_commitment=False,
                    key_encipherment=True,
                    data_encipherment=False,
                    key_agreement=False,
                    key_cert_sign=False,
                    crl_sign=False,
                    encipher_only=False,
                    decipher_only=False,
                ),
                critical=True,
            )

        # SAN
        san_entries: list[x509.GeneralName] = []
        for name in hostnames:
            name = name.strip()
            if not name:
                continue
            try:
                ip_obj = ip_address(name)
                san_entries.append(x509.IPAddress(ip_obj))
            except ValueError:
                san_entries.append(x509.DNSName(name))
        if san_entries:
            builder = builder.add_extension(x509.SubjectAlternativeName(san_entries), critical=False)

        if usages:
            builder = builder.add_extension(x509.ExtendedKeyUsage(list(usages)), critical=False)

        return builder.sign(private_key=issuer_key, algorithm=hashes.SHA384(), backend=default_backend())

    def _write_file(path: Path, data: bytes, *, mode: int = 0o600) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        try:
            os.chmod(path, mode)
        except PermissionError:  # pragma: no cover
            pass

    now = datetime.now(timezone.utc)
    valid_to = now + timedelta(days=config.valid_days)

    ca_key = _generate_private_key(config.key_size)
    ca_cert = _issue_certificate(
        subject=_build_name(f"{config.common_name} Root CA", config.organization),
        issuer=_build_name(f"{config.common_name} Root CA", config.organization),
        public_key=ca_key.public_key(),
        issuer_key=ca_key,
        serial=x509.random_serial_number(),
        valid_from=now,
        valid_to=valid_to,
        is_ca=True,
    )

    server_key = _generate_private_key(config.key_size)
    server_cert = _issue_certificate(
        subject=_build_name(f"{config.common_name} Trading Daemon", config.organization),
        issuer=ca_cert.subject,
        public_key=server_key.public_key(),
        issuer_key=ca_key,
        serial=x509.random_serial_number(),
        valid_from=now,
        valid_to=valid_to,
        is_ca=False,
        hostnames=config.server_hostnames,
        usages=(ExtendedKeyUsageOID.SERVER_AUTH,),
    )

    client_key = _generate_private_key(config.key_size)
    client_cert = _issue_certificate(
        subject=_build_name(f"{config.common_name} Desktop Shell", config.organization),
        issuer=ca_cert.subject,
        public_key=client_key.public_key(),
        issuer_key=ca_key,
        serial=x509.random_serial_number(),
        valid_from=now,
        valid_to=valid_to,
        is_ca=False,
        usages=(ExtendedKeyUsageOID.CLIENT_AUTH,),
    )

    file_prefix = _slugify(config.bundle_name)

    files = {
        "ca_certificate": config.output_dir / f"{file_prefix}-ca.pem",
        "ca_key": config.output_dir / f"{file_prefix}-ca-key.pem",
        "server_certificate": config.output_dir / f"{file_prefix}-server.pem",
        "server_key": config.output_dir / f"{file_prefix}-server-key.pem",
        "client_certificate": config.output_dir / f"{file_prefix}-client.pem",
        "client_key": config.output_dir / f"{file_prefix}-client-key.pem",
        "metadata": config.output_dir / f"{file_prefix}-metadata.json",
    }

    _write_file(files["ca_certificate"], ca_cert.public_bytes(serialization.Encoding.PEM))
    _write_file(files["ca_key"], _serialize_key(ca_key, config.ca_key_passphrase))
    _write_file(files["server_certificate"], server_cert.public_bytes(serialization.Encoding.PEM))
    _write_file(files["server_key"], _serialize_key(server_key, config.server_key_passphrase))
    _write_file(files["client_certificate"], client_cert.public_bytes(serialization.Encoding.PEM))
    _write_file(files["client_key"], _serialize_key(client_key, config.client_key_passphrase))

    return files, valid_to


# -----------------------------------------------------------------------------
# Backend 2: OpenSSL (fallback - prostszy, bez passphrase'ów)
# -----------------------------------------------------------------------------
def _run_openssl(args: list[str], *, input_data: bytes | None = None) -> None:
    try:
        subprocess.run(args, input=input_data, check=True, capture_output=True)
    except FileNotFoundError as exc:  # pragma: no cover
        raise RuntimeError("Polecenie 'openssl' nie jest dostępne w PATH") from exc
    except subprocess.CalledProcessError as exc:  # pragma: no cover
        stderr = exc.stderr.decode("utf-8", errors="ignore") if exc.stderr else ""
        raise RuntimeError(f"Polecenie {' '.join(args)} nie powiodło się: {stderr}") from exc


def _generate_with_openssl(config: BundleConfig) -> Mapping[str, Path]:
    cfg = config
    base = cfg.output_dir
    base.mkdir(parents=True, exist_ok=True)

    file_prefix = _slugify(cfg.bundle_name)

    files = {
        "ca_certificate": base / f"{file_prefix}-ca.pem",
        "ca_key": base / f"{file_prefix}-ca-key.pem",
        "server_certificate": base / f"{file_prefix}-server.pem",
        "server_key": base / f"{file_prefix}-server-key.pem",
        "client_certificate": base / f"{file_prefix}-client.pem",
        "client_key": base / f"{file_prefix}-client-key.pem",
        "metadata": base / f"{file_prefix}-metadata.json",
    }

    # CA
    _run_openssl(["openssl", "genrsa", "-out", str(files["ca_key"]), str(cfg.key_size)])
    _run_openssl(
        [
            "openssl",
            "req",
            "-x509",
            "-new",
            "-key",
            str(files["ca_key"]),
            "-sha256",
            "-days",
            str(cfg.valid_days),
            "-subj",
            f"/CN={cfg.common_name} Root CA/O={cfg.organization}",
            "-out",
            str(files["ca_certificate"]),
        ]
    )

    # Server
    _run_openssl(["openssl", "genrsa", "-out", str(files["server_key"]), str(cfg.key_size)])
    server_csr = base / f"{file_prefix}-server.csr"
    _run_openssl(
        [
            "openssl",
            "req",
            "-new",
            "-key",
            str(files["server_key"]),
            "-subj",
            f"/CN={cfg.common_name} Trading Daemon/O={cfg.organization}",
            "-out",
            str(server_csr),
        ]
    )
    # SAN + EKU przez -addext (OpenSSL 1.1.1+)
    san_parts = []
    for name in cfg.server_hostnames:
        try:
            ip_address(name)
            san_parts.append(f"IP:{name}")
        except ValueError:
            san_parts.append(f"DNS:{name}")
    san_value = ",".join(san_parts) if san_parts else "DNS:localhost,IP:127.0.0.1"
    _run_openssl(
        [
            "openssl",
            "x509",
            "-req",
            "-in",
            str(server_csr),
            "-CA",
            str(files["ca_certificate"]),
            "-CAkey",
            str(files["ca_key"]),
            "-CAcreateserial",
            "-out",
            str(files["server_certificate"]),
            "-days",
            str(cfg.valid_days),
            "-sha256",
            "-addext",
            f"subjectAltName={san_value}",
            "-addext",
            "extendedKeyUsage=serverAuth",
            "-addext",
            "basicConstraints=CA:FALSE",
            "-addext",
            "keyUsage = digitalSignature, keyEncipherment",
        ]
    )
    if server_csr.exists():
        server_csr.unlink()

    # Client
    _run_openssl(["openssl", "genrsa", "-out", str(files["client_key"]), str(cfg.key_size)])
    client_csr = base / f"{file_prefix}-client.csr"
    _run_openssl(
        [
            "openssl",
            "req",
            "-new",
            "-key",
            str(files["client_key"]),
            "-subj",
            f"/CN={cfg.common_name} Desktop Shell/O={cfg.organization}",
            "-out",
            str(client_csr),
        ]
    )
    _run_openssl(
        [
            "openssl",
            "x509",
            "-req",
            "-in",
            str(client_csr),
            "-CA",
            str(files["ca_certificate"]),
            "-CAkey",
            str(files["ca_key"]),
            "-out",
            str(files["client_certificate"]),
            "-days",
            str(cfg.valid_days),
            "-sha256",
            "-addext",
            "extendedKeyUsage=clientAuth",
            "-addext",
            "basicConstraints=CA:FALSE",
            "-addext",
            "keyUsage = digitalSignature, keyEncipherment",
        ]
    )
    if client_csr.exists():
        client_csr.unlink()

    return files


# -----------------------------------------------------------------------------
# Wspólne: zapis metadanych + audyt + rotacja
# -----------------------------------------------------------------------------
def _write_metadata(
    files: Mapping[str, Path],
    *,
    bundle_name: str,
    hostnames: Iterable[str],
    rotation_registry: Path | None,
    valid_until: datetime | None,
) -> Mapping[str, object]:
    now = datetime.now(timezone.utc)
    payload: dict[str, object] = {
        "generated_at": now.isoformat(),
        "valid_until": valid_until.isoformat() if valid_until else None,
        "bundle": bundle_name,
        "hostnames": tuple(hostnames),
        "files": {k: str(p) for k, p in files.items() if k != "metadata"},
        "artifacts": {
            "ca": certificate_reference_metadata(files["ca_certificate"], role="tls_ca"),
            "server": certificate_reference_metadata(files["server_certificate"], role="tls_server_cert"),
            "client": certificate_reference_metadata(files["client_certificate"], role="tls_client_cert"),
        },
    }

    # Opcjonalny audyt TLS (nie blokuje generacji)
    if _audit_mtls_bundle is not None:  # pragma: no cover
        try:
            audit = _audit_mtls_bundle(Path(files["ca_certificate"]).parent)
            payload["tls_audit"] = audit
        except Exception as exc:
            payload["tls_audit_error"] = repr(exc)

    # Rotacja
    if rotation_registry is not None:
        reg = RotationRegistry(rotation_registry)
        reg.mark_rotated(bundle_name, "tls_ca", timestamp=now)
        reg.mark_rotated(bundle_name, "tls_server", timestamp=now)
        reg.mark_rotated(bundle_name, "tls_client", timestamp=now)

    meta_path = files["metadata"]
    meta_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return payload


def generate_mtls_bundle(
    bundle_path: Path,
    *,
    ca_subject: str,
    server_subject: str,
    client_subject: str,
    validity_days: int = 365,
    overwrite: bool = False,
    rotation_registry: Path | None = None,
    key_size: int = 4096,
    server_hostnames: Iterable[str] | None = None,
    ca_passphrase: bytes | None = None,
    server_passphrase: bytes | None = None,
    client_passphrase: bytes | None = None,
    bundle_name: str | None = None,
) -> Mapping[str, object]:
    base = Path(bundle_path)
    base.mkdir(parents=True, exist_ok=True)

    ca_parts = _parse_subject(ca_subject)
    server_parts = _parse_subject(server_subject)
    client_parts = _parse_subject(client_subject)

    common_name = ca_parts.get("CN") or server_parts.get("CN") or "Core OEM"
    organization = ca_parts.get("O") or server_parts.get("O") or "Dudzian"

    requested_name = (bundle_name or "").strip() or base.name or common_name
    bundle_slug = _slugify(requested_name)

    host_candidates: list[str] = []
    if server_hostnames:
        host_candidates.extend(server_hostnames)
    server_cn = server_parts.get("CN")
    if server_cn:
        host_candidates.append(server_cn)
    host_candidates.extend(DEFAULT_HOSTNAMES)
    normalized_hostnames = tuple(dict.fromkeys(name.strip() for name in host_candidates if name.strip()))

    destination_files = {
        "ca_certificate": base / "ca" / "ca.pem",
        "ca_key": base / "ca" / "ca.key",
        "server_certificate": base / "server" / "server.crt",
        "server_key": base / "server" / "server.key",
        "client_certificate": base / "client" / "client.crt",
        "client_key": base / "client" / "client.key",
    }

    metadata_destination = base / "bundle.json"

    if not overwrite:
        for path in (*destination_files.values(), metadata_destination):
            if path.exists():
                raise FileExistsError(f"Ścieżka {path} już istnieje – użyj overwrite=True, aby nadpisać")

    config = BundleConfig(
        output_dir=base,
        bundle_name=requested_name,
        common_name=common_name,
        organization=organization,
        valid_days=validity_days,
        key_size=key_size,
        server_hostnames=normalized_hostnames or DEFAULT_HOSTNAMES,
        rotation_registry=rotation_registry,
        ca_key_passphrase=ca_passphrase,
        server_key_passphrase=server_passphrase,
        client_key_passphrase=client_passphrase,
    )

    metadata = generate_bundle(config)

    source_files = {
        "ca_certificate": base / f"{bundle_slug}-ca.pem",
        "ca_key": base / f"{bundle_slug}-ca-key.pem",
        "server_certificate": base / f"{bundle_slug}-server.pem",
        "server_key": base / f"{bundle_slug}-server-key.pem",
        "client_certificate": base / f"{bundle_slug}-client.pem",
        "client_key": base / f"{bundle_slug}-client-key.pem",
    }

    for key, source in source_files.items():
        target = destination_files[key]
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            if not overwrite:
                raise FileExistsError(f"Ścieżka {target} już istnieje – użyj overwrite=True, aby nadpisać")
            target.unlink()
        source.replace(target)

    metadata_path = base / f"{bundle_slug}-metadata.json"
    if metadata_path.exists():
        metadata_path.unlink()

    artifacts = {
        "ca": certificate_reference_metadata(destination_files["ca_certificate"], role="tls_ca"),
        "server": certificate_reference_metadata(destination_files["server_certificate"], role="tls_server_cert"),
        "client": certificate_reference_metadata(destination_files["client_certificate"], role="tls_client_cert"),
    }

    metadata = dict(metadata)
    metadata["bundle_path"] = str(base)
    metadata["subjects"] = {
        "ca": ca_subject,
        "server": server_subject,
        "client": client_subject,
    }
    metadata["files"] = {key: str(path) for key, path in destination_files.items()}
    metadata["files"]["metadata"] = str(metadata_destination)
    metadata["artifacts"] = artifacts

    if _audit_mtls_bundle is not None:  # pragma: no cover - zależność opcjonalna
        try:
            metadata["tls_audit"] = _audit_mtls_bundle(base)
        except Exception as exc:
            metadata["tls_audit_error"] = repr(exc)

    if rotation_registry is not None:
        registry = RotationRegistry(rotation_registry)
        now = datetime.now(timezone.utc)
        normalized_key = requested_name.strip() or "mtls"
        rotation_keys = {normalized_key, "mtls"}
        for key in rotation_keys:
            registry.mark_rotated(key, "ca", timestamp=now)
            registry.mark_rotated(key, "server", timestamp=now)
            registry.mark_rotated(key, "client", timestamp=now)

    metadata_destination.write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    return metadata


def generate_bundle(config: BundleConfig) -> Mapping[str, object]:
    config.output_dir.mkdir(parents=True, exist_ok=True)

    if _HAS_CRYPTO:
        files, valid_to = _generate_with_cryptography(config)
    else:
        # Fallback – bez passphrase’ów (openssl nie jest tu konfigurowany pod szyfrowanie PKCS#8)
        files = _generate_with_openssl(config)
        valid_to = datetime.now(timezone.utc) + timedelta(days=config.valid_days)

    metadata = _write_metadata(
        files,
        bundle_name=config.bundle_name,
        hostnames=config.server_hostnames,
        rotation_registry=config.rotation_registry,
        valid_until=valid_to,
    )
    return metadata


def main(argv: Iterable[str] | None = None) -> int:
    cfg = _parse_args(argv)
    metadata = generate_mtls_bundle(
        cfg.output_dir,
        ca_subject=f"/CN={cfg.common_name} Root CA/O={cfg.organization}",
        server_subject=f"/CN={cfg.common_name} Trading Daemon/O={cfg.organization}",
        client_subject=f"/CN={cfg.common_name} Desktop Shell/O={cfg.organization}",
        validity_days=cfg.valid_days,
        overwrite=True,
        rotation_registry=cfg.rotation_registry,
        key_size=cfg.key_size,
        server_hostnames=cfg.server_hostnames,
        ca_passphrase=cfg.ca_key_passphrase,
        server_passphrase=cfg.server_key_passphrase,
        client_passphrase=cfg.client_key_passphrase,
        bundle_name=cfg.bundle_name,
    )
    print(json.dumps(metadata, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover - punkt wejścia CLI
    raise SystemExit(main())
