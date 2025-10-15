"""Verify HMAC-signed manifest documents used by OEM/Stage4 bundles."""

from __future__ import annotations

import argparse
import base64
import hashlib
import hmac
import json
import os
import stat
import sys
from pathlib import Path
from typing import Optional, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from bot_core.security.signing import canonical_json_bytes  # noqa: E402


def _load_key(path: Path) -> bytes:
    resolved = path.expanduser().resolve()
    if not resolved.exists():
        raise FileNotFoundError(f"Signing key not found: {resolved}")
    if resolved.is_dir():
        raise ValueError(f"Signing key path must be a file: {resolved}")
    if os.name != "nt":
        mode = resolved.stat().st_mode
        if mode & (stat.S_IRWXG | stat.S_IRWXO):
            raise PermissionError(
                "Signing key file permissions must restrict access to the owner"
            )
    return resolved.read_bytes()


def _load_signature(path: Path) -> dict:
    document = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(document, dict):
        raise ValueError("Signature document must be a JSON object")
    if "payload" not in document or "signature" not in document:
        raise ValueError("Signature document missing 'payload' or 'signature'")
    return document


def _validate_digest(
    *,
    manifest_path: Path,
    payload: dict,
    digest_field: str,
) -> None:
    expected = payload.get(digest_field)
    if not isinstance(expected, str):
        raise ValueError(f"Payload missing digest field '{digest_field}'")
    hasher = hashlib.new(digest_field)
    with manifest_path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    computed = hasher.hexdigest()
    if computed != expected:
        raise ValueError(
            "Manifest digest mismatch: payload={}, computed={}".format(
                expected, computed
            )
        )


def _verify_signature(
    *,
    key: bytes,
    payload: dict,
    signature: dict,
    digest_algorithm: str,
) -> None:
    algorithm = signature.get("algorithm")
    expected_algorithm = f"HMAC-{digest_algorithm.upper()}"
    if algorithm and algorithm.upper() != expected_algorithm:
        raise ValueError(
            f"Unsupported signature algorithm {algorithm!r}; expected {expected_algorithm}"
        )
    value = signature.get("value")
    if not isinstance(value, str):
        raise ValueError("Signature missing base64 'value'")
    mac = hmac.new(key, canonical_json_bytes(payload), getattr(hashlib, digest_algorithm))
    computed = base64.b64encode(mac.digest()).decode("ascii")
    if computed != value:
        raise ValueError("Signature verification failed")


def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path, help="Path to manifest.json")
    parser.add_argument(
        "--signature",
        required=True,
        type=Path,
        help="Path to manifest signature document (JSON)",
    )
    parser.add_argument(
        "--signing-key",
        required=True,
        type=Path,
        help="Path to HMAC key used for verification",
    )
    parser.add_argument(
        "--digest",
        default="sha384",
        help="Digest algorithm declared in the signature payload (default: sha384)",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parse_args(argv)
    key = _load_key(args.signing_key)
    manifest_path = args.manifest.expanduser().resolve()
    signature_path = args.signature.expanduser().resolve()

    payload_document = _load_signature(signature_path)
    payload = payload_document["payload"]
    signature = payload_document["signature"]

    _validate_digest(
        manifest_path=manifest_path,
        payload=payload,
        digest_field=args.digest,
    )
    _verify_signature(
        key=key,
        payload=payload,
        signature=signature,
        digest_algorithm=args.digest,
    )
    key_id = signature.get("key_id", "n/a")
    print(
        f"Signature verification succeeded for {manifest_path.name} (key_id={key_id})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
