"""Offline licensing portal for OEM deployments."""
from __future__ import annotations

import argparse
import base64
import json
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from bot_core.security.license_store import (
    LicenseStore,
    LicenseStoreDecryptionError,
    LicenseStoreDocument,
    LicenseStoreError,
)
from bot_core.security.signing import validate_hmac_signature


@dataclass(slots=True)
class PortalContext:
    store_path: Path
    fingerprint: str


def _decode_secret(value: str | None) -> bytes | None:
    if not value:
        return None
    text = value.strip()
    if text.startswith("env:"):
        env_name = text[4:]
        import os

        resolved = os.environ.get(env_name)
        if resolved is None:
            raise SystemExit(f"Environment variable {env_name!r} is not defined")
        text = resolved.strip()
    elif text.startswith("file:"):
        path = Path(text[5:]).expanduser()
        text = path.read_text(encoding="utf-8").strip()
    if text.startswith("hex:"):
        return bytes.fromhex(text[4:])
    if text.startswith("base64:"):
        return base64.b64decode(text[7:])
    try:
        return base64.b64decode(text)
    except Exception:
        return text.encode("utf-8")


def _load_fingerprint(*, fingerprint: str | None, fingerprint_file: str | None, read_local: bool) -> str:
    if fingerprint:
        return fingerprint.strip()
    if fingerprint_file:
        path = Path(fingerprint_file).expanduser()
        return path.read_text(encoding="utf-8").strip()
    if read_local:
        try:
            from bot_core.security.fingerprint import get_local_fingerprint  # type: ignore
        except Exception as exc:  # pragma: no cover - optional dependency
            raise SystemExit(f"Cannot read local fingerprint: {exc}") from exc
        candidate = get_local_fingerprint()
        if not candidate:
            raise SystemExit("Local fingerprint provider returned an empty value")
        return str(candidate).strip()
    raise SystemExit("Fingerprint must be provided via --fingerprint, --fingerprint-file or --read-local")


def _load_store_document(path: Path, fingerprint: str) -> LicenseStoreDocument:
    store = LicenseStore(path=path, fingerprint_override=fingerprint)
    try:
        return store.load()
    except LicenseStoreDecryptionError as exc:
        raise SystemExit(f"Failed to decrypt license store: {exc}") from exc
    except LicenseStoreError as exc:
        raise SystemExit(f"License store error: {exc}") from exc


def _normalise_license_payload(payload: Mapping[str, Any]) -> Mapping[str, Any]:
    if "payload" in payload and isinstance(payload.get("payload"), Mapping):
        return payload["payload"]  # type: ignore[return-value]
    return payload


def _collect_store_summary(document: LicenseStoreDocument) -> dict[str, Any]:
    licenses = document.data.get("licenses") if isinstance(document.data, Mapping) else {}
    summary: dict[str, Any] = {
        "fingerprint_hash": document.fingerprint_hash,
        "migrated": document.migrated,
        "licenses": [],
    }
    if isinstance(licenses, Mapping):
        for license_id, metadata in licenses.items():
            if isinstance(metadata, Mapping):
                summary["licenses"].append(
                    {
                        "license_id": license_id,
                        "status": metadata.get("status"),
                        "fingerprint": metadata.get("fingerprint"),
                        "issues": metadata.get("issues"),
                    }
                )
    return summary


def _command_status(args: argparse.Namespace) -> None:
    fingerprint = _load_fingerprint(
        fingerprint=args.fingerprint,
        fingerprint_file=args.fingerprint_file,
        read_local=args.read_local,
    )
    context = PortalContext(store_path=Path(args.store).expanduser(), fingerprint=fingerprint)
    document = _load_store_document(context.store_path, context.fingerprint)
    summary = _collect_store_summary(document)
    summary.update({"store_path": str(context.store_path), "fingerprint": fingerprint})
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def _command_verify(args: argparse.Namespace) -> None:
    fingerprint = _load_fingerprint(
        fingerprint=args.fingerprint,
        fingerprint_file=args.fingerprint_file,
        read_local=args.read_local,
    )
    context = PortalContext(store_path=Path(args.store).expanduser(), fingerprint=fingerprint)
    document = _load_store_document(context.store_path, context.fingerprint)
    license_path = Path(args.license).expanduser()
    payload_raw = json.loads(license_path.read_text(encoding="utf-8"))
    payload = _normalise_license_payload(payload_raw)

    signature_status: dict[str, Any] | None = None
    key_bytes = _decode_secret(args.hmac_key)
    signature = None
    if isinstance(payload_raw, Mapping):
        signature = payload_raw.get("signature")
    if key_bytes is not None and isinstance(signature, Mapping):
        algorithm = signature.get("algorithm") or "HMAC-SHA256"
        signature_status = {
            "algorithm": algorithm,
            "errors": validate_hmac_signature(payload, {"signature": signature}, key=key_bytes, algorithm=str(algorithm)),
        }
        signature_status["valid"] = not signature_status["errors"]

    license_id = payload.get("license_id") or payload.get("licenseId") or payload.get("id")
    license_id = str(license_id) if license_id is not None else None

    raw_expected = (
        payload.get("fingerprint")
        or payload.get("fingerprints")
        or payload.get("hardware_fingerprint")
    )
    if isinstance(raw_expected, (list, tuple)) and raw_expected:
        raw_expected = raw_expected[0]
    expected_fingerprint_value = str(raw_expected).strip() if raw_expected else None

    issues: list[str] = []
    if expected_fingerprint_value and expected_fingerprint_value != fingerprint:
        issues.append("fingerprint-mismatch")

    store_contains = False
    licenses = document.data.get("licenses") if isinstance(document.data, Mapping) else {}
    if license_id and isinstance(licenses, Mapping):
        store_contains = license_id in licenses
        if not store_contains:
            issues.append("missing-in-store")

    result = {
        "store_path": str(context.store_path),
        "fingerprint": fingerprint,
        "license_path": str(license_path),
        "license_id": license_id,
        "expected_fingerprint": expected_fingerprint_value,
        "signature": signature_status,
        "store_contains": store_contains,
        "issues": issues,
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))


def _command_recover(args: argparse.Namespace) -> None:
    old_fingerprint = _load_fingerprint(
        fingerprint=args.old_fingerprint,
        fingerprint_file=args.old_fingerprint_file,
        read_local=False,
    )
    new_fingerprint = _load_fingerprint(
        fingerprint=args.new_fingerprint,
        fingerprint_file=args.new_fingerprint_file,
        read_local=args.read_local_new,
    )

    store_path = Path(args.store).expanduser()
    output_path = Path(args.output or args.store).expanduser()
    document = _load_store_document(store_path, old_fingerprint)

    if output_path.exists():
        backup_path = output_path.with_suffix(output_path.suffix + ".bak")
        shutil.copy2(output_path, backup_path)
    else:
        output_path.parent.mkdir(parents=True, exist_ok=True)

    new_store = LicenseStore(path=output_path, fingerprint_override=new_fingerprint)
    new_store.save(document.data)

    summary = {
        "store_path": str(store_path),
        "output_path": str(output_path),
        "old_fingerprint": old_fingerprint,
        "new_fingerprint": new_fingerprint,
        "licenses": len(document.data.get("licenses", {})) if isinstance(document.data, Mapping) else 0,
    }

    if args.report:
        report_path = Path(args.report).expanduser()
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        summary["report_path"] = str(report_path)

    print(json.dumps(summary, ensure_ascii=False, indent=2))


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    def _add_fingerprint_options(subparser: argparse.ArgumentParser, *, allow_local: bool = True) -> None:
        subparser.add_argument("--store", required=True, help="Path to license store (JSON)")
        subparser.add_argument("--fingerprint", help="Fingerprint override")
        subparser.add_argument("--fingerprint-file", help="Read fingerprint from file")
        if allow_local:
            subparser.add_argument("--read-local", action="store_true", help="Read fingerprint from local provider")
        else:
            subparser.add_argument("--read-local", action="store_true", help=argparse.SUPPRESS)

    status_parser = subparsers.add_parser("status", help="Show license store status")
    _add_fingerprint_options(status_parser)

    verify_parser = subparsers.add_parser("verify", help="Verify a license payload")
    _add_fingerprint_options(verify_parser)
    verify_parser.add_argument("--license", required=True, help="Path to license JSON payload")
    verify_parser.add_argument(
        "--hmac-key",
        help="HMAC key for signature validation (supports env:/file:/hex:/base64: prefixes)",
    )

    recover_parser = subparsers.add_parser("recover", help="Re-encrypt license store for new fingerprint")
    recover_parser.add_argument("--store", required=True, help="Path to encrypted license store")
    recover_parser.add_argument("--output", help="Destination path for recovered store (defaults to --store)")
    recover_parser.add_argument("--old-fingerprint", help="Original fingerprint")
    recover_parser.add_argument("--old-fingerprint-file", help="File containing original fingerprint")
    recover_parser.add_argument("--new-fingerprint", help="Target fingerprint")
    recover_parser.add_argument("--new-fingerprint-file", help="File containing target fingerprint")
    recover_parser.add_argument("--read-local-new", action="store_true", help="Read new fingerprint from local provider")
    recover_parser.add_argument("--report", help="Write JSON report to path")

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.command == "status":
        _command_status(args)
    elif args.command == "verify":
        _command_verify(args)
    elif args.command == "recover":
        _command_recover(args)
    else:  # pragma: no cover - defensive
        parser.error(f"Unknown command: {args.command}")
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entrypoint
    sys.exit(main())
