#!/usr/bin/env python3
"""Podpisuje presety Marketplace (JSON + MD) kluczami HMAC i Ed25519."""

from __future__ import annotations

import argparse
import base64
import hashlib
import hmac
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import ed25519

from bot_core.marketplace.signing_key_policy import (
    resolve_hmac_signing_key,
    resolve_signing_private_key,
    validate_signing_identity,
    validate_signing_issuer,
)


def _load_hmac_key(path: Path) -> bytes:
    key = path.read_bytes().strip()
    if not key:
        raise ValueError(f"Plik klucza HMAC {path} jest pusty")
    return key


def _load_ed25519_key(path: Path) -> ed25519.Ed25519PrivateKey:
    data = path.read_bytes()
    try:
        return serialization.load_pem_private_key(data, password=None)
    except ValueError:
        try:
            return ed25519.Ed25519PrivateKey.from_private_bytes(base64.b64decode(data))
        except Exception as exc:  # noqa: BLE001
            raise ValueError(f"Nie udało się wczytać klucza Ed25519 z {path}") from exc


def _signature_payload(
    *,
    path: Path,
    content: bytes,
    hmac_key: bytes,
    hmac_key_id: str,
    ed25519_key: ed25519.Ed25519PrivateKey,
    ed25519_key_id: str,
    issuer: str | None,
) -> dict:
    timestamp = datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")
    sha256 = hashlib.sha256(content).hexdigest()

    hmac_value = hmac.new(hmac_key, content, hashlib.sha256).digest()
    ed_signature = ed25519_key.sign(content)
    ed_public = ed25519_key.public_key().public_bytes(
        encoding=serialization.Encoding.Raw,
        format=serialization.PublicFormat.Raw,
    )

    return {
        "target": str(path),
        "sha256": sha256,
        "ed25519": {
            "algorithm": "ed25519",
            "key_id": ed25519_key_id,
            "issuer": issuer,
            "signed_at": timestamp,
            "value": base64.b64encode(ed_signature).decode("ascii"),
            "public_key": base64.b64encode(ed_public).decode("ascii"),
        },
        "hmac": {
            "algorithm": "HMAC-SHA256",
            "key_id": hmac_key_id,
            "signed_at": timestamp,
            "value": base64.b64encode(hmac_value).decode("ascii"),
        },
    }


def sign_presets(
    *,
    roots: Iterable[Path],
    hmac_key: bytes,
    hmac_key_id: str,
    ed25519_key: ed25519.Ed25519PrivateKey,
    ed25519_key_id: str,
    issuer: str | None,
) -> int:
    signed = 0
    for root in roots:
        for path in sorted(root.rglob("*.json")):
            payload = _signature_payload(
                path=path,
                content=path.read_bytes(),
                hmac_key=hmac_key,
                hmac_key_id=hmac_key_id,
                ed25519_key=ed25519_key,
                ed25519_key_id=ed25519_key_id,
                issuer=issuer,
            )
            path.with_suffix(path.suffix + ".sig").write_text(_serialize(payload), encoding="utf-8")
            signed += 1
        for path in sorted(root.rglob("*.md")):
            payload = _signature_payload(
                path=path,
                content=path.read_bytes(),
                hmac_key=hmac_key,
                hmac_key_id=hmac_key_id,
                ed25519_key=ed25519_key,
                ed25519_key_id=ed25519_key_id,
                issuer=issuer,
            )
            path.with_suffix(path.suffix + ".sig").write_text(_serialize(payload), encoding="utf-8")
            signed += 1
    return signed


def _serialize(payload: dict) -> str:
    import json

    return json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        action="append",
        default=["config/marketplace/presets"],
        help="Katalog z presetami (można podać wielokrotnie)",
    )
    parser.add_argument(
        "--hmac-key",
        help="Jawna ścieżka klucza HMAC; obowiązkowa i zewnętrzna dla production.",
    )
    parser.add_argument(
        "--hmac-key-id",
        help="Identyfikator klucza HMAC",
    )
    parser.add_argument(
        "--ed25519-key",
        "--private-key",
        dest="private_key",
        help="Jawna ścieżka klucza Ed25519; obowiązkowa dla production.",
    )
    parser.add_argument("--environment", choices=("dev", "test", "production"), default="dev")
    parser.add_argument(
        "--ed25519-key-id",
        help="Identyfikator klucza Ed25519",
    )
    parser.add_argument("--issuer", help="Opcjonalny identyfikator wystawcy podpisu")
    args = parser.parse_args(argv)

    hmac_key_id = validate_signing_identity(
        args.hmac_key_id or ("dev-hmac" if args.environment != "production" else None),
        environment=args.environment,
        option_name="--hmac-key-id",
    )
    ed25519_key_id = validate_signing_identity(
        args.ed25519_key_id
        or ("dev-presets-ed25519" if args.environment != "production" else None),
        environment=args.environment,
        option_name="--ed25519-key-id",
    )
    roots = [Path(entry).expanduser().resolve() for entry in args.root]
    hmac_key_path = resolve_hmac_signing_key(args.hmac_key, environment=args.environment)
    hmac_key = _load_hmac_key(hmac_key_path)
    private_key = resolve_signing_private_key(args.private_key, environment=args.environment)
    issuer = validate_signing_issuer(args.issuer, environment=args.environment)
    ed_key = _load_ed25519_key(private_key)

    count = sign_presets(
        roots=roots,
        hmac_key=hmac_key,
        hmac_key_id=hmac_key_id,
        ed25519_key=ed_key,
        ed25519_key_id=ed25519_key_id,
        issuer=issuer,
    )
    print(f"Podpisano {count} plików presetów")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
