"""Generator pakietu mTLS dla demona tradingowego i klienta UI."""
from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Iterable, Mapping

from cryptography import x509
from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.x509.oid import ExtendedKeyUsageOID, NameOID

from bot_core.security.certificates import certificate_reference_metadata
from bot_core.security.rotation import RotationRegistry


DEFAULT_HOSTNAMES = ("127.0.0.1", "localhost")


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
        help="Hostname/IP wpisywane do SAN certyfikatu serwerowego (można podać wielokrotnie)",
    )
    parser.add_argument(
        "--rotation-registry",
        help="Ścieżka do rejestru rotacji TLS (JSON). Brak oznacza pominięcie aktualizacji.",
    )
    parser.add_argument(
        "--ca-key-passphrase-env",
        help="Nazwa zmiennej ENV zawierającej passphrase dla klucza CA (PKCS8)",
    )
    parser.add_argument(
        "--server-key-passphrase-env",
        help="Nazwa zmiennej ENV z passphrase do klucza serwera (UWAGA: gRPC wymaga kluczy bez hasła)",
    )
    parser.add_argument(
        "--client-key-passphrase-env",
        help="Nazwa zmiennej ENV z passphrase do klucza klienta",
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
        rotation_registry=Path(args.rotation_registry).expanduser().resolve()
        if args.rotation_registry
        else None,
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


def _build_name(common_name: str, organization: str) -> x509.Name:
    return x509.Name(
        [
            x509.NameAttribute(NameOID.COMMON_NAME, common_name),
            x509.NameAttribute(NameOID.ORGANIZATION_NAME, organization),
        ]
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

    san_entries = [x509.DNSName(name) for name in hostnames if name]
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
    except PermissionError:
        pass


def _metadata_entry(path: Path, *, role: str) -> Mapping[str, object]:
    return certificate_reference_metadata(path, role=role)


def generate_bundle(config: BundleConfig) -> Mapping[str, object]:
    config.output_dir.mkdir(parents=True, exist_ok=True)
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

    bundle_files = {
        "ca_certificate": config.output_dir / f"{config.bundle_name}-ca.pem",
        "ca_key": config.output_dir / f"{config.bundle_name}-ca-key.pem",
        "server_certificate": config.output_dir / f"{config.bundle_name}-server.pem",
        "server_key": config.output_dir / f"{config.bundle_name}-server-key.pem",
        "client_certificate": config.output_dir / f"{config.bundle_name}-client.pem",
        "client_key": config.output_dir / f"{config.bundle_name}-client-key.pem",
        "metadata": config.output_dir / f"{config.bundle_name}-metadata.json",
    }

    _write_file(bundle_files["ca_certificate"], ca_cert.public_bytes(serialization.Encoding.PEM))
    _write_file(bundle_files["ca_key"], _serialize_key(ca_key, config.ca_key_passphrase))
    _write_file(bundle_files["server_certificate"], server_cert.public_bytes(serialization.Encoding.PEM))
    _write_file(bundle_files["server_key"], _serialize_key(server_key, config.server_key_passphrase))
    _write_file(bundle_files["client_certificate"], client_cert.public_bytes(serialization.Encoding.PEM))
    _write_file(bundle_files["client_key"], _serialize_key(client_key, config.client_key_passphrase))

    metadata = {
        "generated_at": now.isoformat(),
        "valid_until": valid_to.isoformat(),
        "bundle": config.bundle_name,
        "hostnames": config.server_hostnames,
        "files": {
            key: str(path)
            for key, path in bundle_files.items()
            if key != "metadata"
        },
        "artifacts": {
            "ca": _metadata_entry(bundle_files["ca_certificate"], role="tls_ca"),
            "server": _metadata_entry(bundle_files["server_certificate"], role="tls_server_cert"),
            "client": _metadata_entry(bundle_files["client_certificate"], role="tls_client_cert"),
        },
    }

    bundle_files["metadata"].write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    if config.rotation_registry:
        registry = RotationRegistry(config.rotation_registry)
        registry.mark_rotated(config.bundle_name, "tls_ca", timestamp=now)
        registry.mark_rotated(config.bundle_name, "tls_server", timestamp=now)
        registry.mark_rotated(config.bundle_name, "tls_client", timestamp=now)

    return metadata


def main(argv: Iterable[str] | None = None) -> int:
    config = _parse_args(argv)
    metadata = generate_bundle(config)
    print(json.dumps(metadata, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover - punkt wejścia CLI
    raise SystemExit(main())

