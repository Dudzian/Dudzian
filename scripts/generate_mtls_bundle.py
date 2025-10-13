"""Generuje pakiet certyfikatów mTLS (CA, serwer, klient) z audytem."""

from __future__ import annotations

import argparse
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from bot_core.security.rotation import RotationRegistry
from bot_core.security.tls_audit import audit_mtls_bundle


def _run_openssl(args: list[str], *, input_data: bytes | None = None) -> None:
    try:
        subprocess.run(args, input=input_data, check=True, capture_output=True)
    except FileNotFoundError as exc:  # pragma: no cover - brak openssl
        raise RuntimeError("Polecenie 'openssl' nie jest dostępne w PATH") from exc
    except subprocess.CalledProcessError as exc:  # pragma: no cover - diagnostyka CLI
        stderr = exc.stderr.decode("utf-8", errors="ignore") if exc.stderr else ""
        raise RuntimeError(f"Polecenie {' '.join(args)} nie powiodło się: {stderr}") from exc


def generate_mtls_bundle(
    output_dir: Path,
    *,
    ca_subject: str,
    server_subject: str,
    client_subject: str,
    validity_days: int,
    overwrite: bool = False,
    rotation_registry: Path | None = None,
) -> Mapping[str, Any]:
    """Generuje certyfikaty mTLS i zwraca raport audytu."""

    if validity_days <= 0:
        raise ValueError("validity_days musi być dodatnie")

    base = output_dir.expanduser()
    if base.exists() and not overwrite:
        raise FileExistsError(f"Katalog {base} już istnieje – użyj --overwrite aby zastąpić")

    ca_dir = base / "ca"
    server_dir = base / "server"
    client_dir = base / "client"
    for directory in (ca_dir, server_dir, client_dir):
        directory.mkdir(parents=True, exist_ok=True)

    ca_key = ca_dir / "ca.key"
    ca_cert = ca_dir / "ca.pem"
    ca_serial = ca_dir / "ca.srl"
    server_key = server_dir / "server.key"
    server_csr = server_dir / "server.csr"
    server_cert = server_dir / "server.crt"
    client_key = client_dir / "client.key"
    client_csr = client_dir / "client.csr"
    client_cert = client_dir / "client.crt"

    bundle_json = base / "bundle.json"

    # --- CA ------------------------------------------------------------------
    _run_openssl(["openssl", "genrsa", "-out", str(ca_key), "4096"])
    _run_openssl(
        [
            "openssl",
            "req",
            "-x509",
            "-new",
            "-key",
            str(ca_key),
            "-sha256",
            "-days",
            str(validity_days),
            "-subj",
            ca_subject,
            "-out",
            str(ca_cert),
        ]
    )

    # --- Server --------------------------------------------------------------
    _run_openssl(["openssl", "genrsa", "-out", str(server_key), "4096"])
    _run_openssl(
        [
            "openssl",
            "req",
            "-new",
            "-key",
            str(server_key),
            "-subj",
            server_subject,
            "-out",
            str(server_csr),
        ]
    )
    _run_openssl(
        [
            "openssl",
            "x509",
            "-req",
            "-in",
            str(server_csr),
            "-CA",
            str(ca_cert),
            "-CAkey",
            str(ca_key),
            "-CAcreateserial",
            "-CAserial",
            str(ca_serial),
            "-out",
            str(server_cert),
            "-days",
            str(validity_days),
            "-sha256",
        ]
    )

    # --- Client --------------------------------------------------------------
    _run_openssl(["openssl", "genrsa", "-out", str(client_key), "4096"])
    _run_openssl(
        [
            "openssl",
            "req",
            "-new",
            "-key",
            str(client_key),
            "-subj",
            client_subject,
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
            str(ca_cert),
            "-CAkey",
            str(ca_key),
            "-CAserial",
            str(ca_serial),
            "-out",
            str(client_cert),
            "-days",
            str(validity_days),
            "-sha256",
        ]
    )

    # Usuń CSR i plik seriala – nie są wymagane w bundle
    for temporary in (server_csr, client_csr):
        if temporary.exists():
            temporary.unlink()

    metadata = audit_mtls_bundle(base)
    metadata["generated_at"] = datetime.now(timezone.utc).isoformat()
    bundle_json.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    if rotation_registry is not None:
        registry = RotationRegistry(rotation_registry)
        now = datetime.now(timezone.utc)
        registry.mark_rotated("mtls", "ca", timestamp=now)
        registry.mark_rotated("mtls", "server", timestamp=now)
        registry.mark_rotated("mtls", "client", timestamp=now)

    return metadata


def _build_subject(common_name: str, organization: str | None) -> str:
    parts = [f"/CN={common_name}"]
    if organization:
        parts.append(f"/O={organization}")
    return "".join(parts)


def main() -> None:  # pragma: no cover - CLI wrapper
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path, help="Katalog docelowy pakietu mTLS")
    parser.add_argument("--server-cn", required=True, help="CommonName certyfikatu serwera")
    parser.add_argument("--client-cn", required=True, help="CommonName certyfikatu klienta")
    parser.add_argument(
        "--ca-cn", default="BotCore mTLS CA", help="CommonName certyfikatu CA"
    )
    parser.add_argument(
        "--organization",
        default="BotCore",
        help="Pole O (Organization) dodawane do subject wszystkich certyfikatów",
    )
    parser.add_argument(
        "--days",
        type=int,
        default=365,
        help="Ważność certyfikatów w dniach",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Nadpisuje istniejący katalog",
    )
    parser.add_argument(
        "--rotation-registry",
        type=Path,
        help="Opcjonalny rejestr rotacji do aktualizacji",
    )

    args = parser.parse_args()
    organization = args.organization if args.organization else None
    metadata = generate_mtls_bundle(
        args.output,
        ca_subject=_build_subject(args.ca_cn, organization),
        server_subject=_build_subject(args.server_cn, organization),
        client_subject=_build_subject(args.client_cn, organization),
        validity_days=args.days,
        overwrite=args.overwrite,
        rotation_registry=args.rotation_registry,
    )

    print(json.dumps(metadata, ensure_ascii=False, indent=2))


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    main()
