"""Validate signed resilience bundles produced for Stage6 drills."""

from __future__ import annotations

import argparse
import base64
import binascii
import hashlib
import hmac
import json
import logging
import os
import sys
import tarfile
from pathlib import Path
from typing import Dict, Iterable, Mapping, Optional, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


from bot_core.security.signing import canonical_json_bytes  # noqa: E402
from deploy.packaging.build_core_bundle import (  # noqa: E402
    _ensure_no_symlinks,
    _ensure_windows_safe_component,
    _ensure_windows_safe_tree,
)


LOGGER = logging.getLogger("verify_resilience_bundle")


def _load_signing_key(path: Optional[Path], env: Optional[str]) -> bytes:
    if path is not None:
        _ensure_no_symlinks(path, label="Signing key path")
        resolved = path.resolve()
        _ensure_windows_safe_tree(resolved, label="Signing key path")
        if not resolved.is_file():
            raise ValueError(f"Signing key path does not reference a file: {resolved}")
        data = resolved.read_bytes()
        if len(data) < 32:
            raise ValueError("Signing key must contain at least 32 bytes")
        return data
    if env is not None:
        value = os.environ.get(env)
        if not value:
            raise ValueError(f"Environment variable {env} is not set or empty")
        data = value.encode("utf-8")
        if len(data) < 32:
            raise ValueError("Signing key from environment must contain at least 32 bytes")
        return data
    raise ValueError("Signing key must be provided via --signing-key-path or --signing-key-env")


def _normalise_name(name: str) -> str:
    if name.startswith("./"):
        name = name[2:]
    return name


def _load_archive_members(archive: tarfile.TarFile) -> Dict[str, tarfile.TarInfo]:
    members: Dict[str, tarfile.TarInfo] = {}
    for member in archive.getmembers():
        normalised = _normalise_name(member.name)
        members[normalised] = member
    return members


def _assert_safe_member(member: tarfile.TarInfo) -> None:
    if member.isdir():
        return
    if member.issym() or member.islnk():
        raise ValueError(f"Bundle contains unsupported link entry: {member.name}")
    parts = Path(_normalise_name(member.name)).parts
    if any(part in ("..", "") for part in parts):
        raise ValueError(f"Bundle contains unsafe path: {member.name}")
    _ensure_windows_safe_component(
        component=parts[-1],
        label="Bundle entry",
        context=member.name,
    )


def _read_json(archive: tarfile.TarFile, member: tarfile.TarInfo) -> Mapping[str, object]:
    extracted = archive.extractfile(member)
    if extracted is None:
        raise ValueError(f"Failed to read {member.name} from archive")
    with extracted:
        data = json.load(extracted)
    if not isinstance(data, Mapping):
        raise ValueError(f"Expected JSON object in {member.name}")
    return data


def _compute_sha256(archive: tarfile.TarFile, member: tarfile.TarInfo) -> str:
    extracted = archive.extractfile(member)
    if extracted is None:
        raise ValueError(f"Failed to read {member.name} from archive")
    hasher = hashlib.sha256()
    with extracted:
        for chunk in iter(lambda: extracted.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def _compute_digest(
    archive: tarfile.TarFile, member: tarfile.TarInfo, algorithm: str
) -> str:
    extracted = archive.extractfile(member)
    if extracted is None:
        raise ValueError(f"Failed to read {member.name} from archive")
    try:
        hasher = hashlib.new(algorithm)
    except ValueError as exc:
        raise ValueError(f"Unsupported digest algorithm declared in signature: {algorithm}") from exc
    with extracted:
        for chunk in iter(lambda: extracted.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def _verify_manifest_signature(
    *,
    archive: tarfile.TarFile,
    manifest_member: tarfile.TarInfo,
    manifest: Mapping[str, object],
    signature_doc: Mapping[str, object],
    signing_key: bytes,
) -> None:
    signature_section = signature_doc.get("signature")
    payload_section = signature_doc.get("payload")
    if not isinstance(signature_section, Mapping) or not isinstance(payload_section, Mapping):
        raise ValueError("Signature document must contain 'signature' and 'payload' objects")

    payload_path = payload_section.get("path")
    if payload_path != "manifest.json":
        raise ValueError("Signature payload references unexpected path")

    digest_items = [key for key in payload_section if key != "path"]
    if len(digest_items) != 1:
        raise ValueError("Signature payload must declare exactly one digest entry")
    digest_key = digest_items[0]
    expected_digest = payload_section[digest_key]
    if not isinstance(expected_digest, str):
        raise ValueError("Digest value must be a hexadecimal string")

    actual_digest = _compute_digest(archive, manifest_member, digest_key)
    if actual_digest != expected_digest:
        raise ValueError("Manifest digest does not match payload entry")

    algorithm = signature_section.get("algorithm")
    if not isinstance(algorithm, str) or not algorithm.upper().startswith("HMAC-"):
        raise ValueError("Unsupported signature algorithm")
    digest_name = algorithm.split("-", 1)[1].lower()
    if not hasattr(hashlib, digest_name):
        raise ValueError(f"Unsupported HMAC digest: {digest_name}")

    signature_value = signature_section.get("value")
    if not isinstance(signature_value, str):
        raise ValueError("Signature value must be a base64 string")
    try:
        decoded_signature = base64.b64decode(signature_value, validate=True)
    except binascii.Error as exc:
        raise ValueError(f"Invalid base64 signature: {exc}") from exc

    expected_mac = hmac.new(
        signing_key,
        canonical_json_bytes(payload_section),
        getattr(hashlib, digest_name),
    ).digest()
    if not hmac.compare_digest(decoded_signature, expected_mac):
        raise ValueError("Manifest signature verification failed")


def verify_resilience_bundle(
    *,
    bundle_path: Path,
    signing_key: bytes,
    logger: Optional[logging.Logger] = None,
) -> None:
    log = logger or LOGGER

    _ensure_no_symlinks(bundle_path, label="Bundle path")
    resolved_bundle = bundle_path.resolve()
    _ensure_windows_safe_tree(resolved_bundle, label="Bundle path")
    if not resolved_bundle.is_file():
        raise FileNotFoundError(f"Bundle does not exist: {resolved_bundle}")

    with tarfile.open(resolved_bundle, "r:gz") as archive:
        members = _load_archive_members(archive)
        manifest_member = members.get("manifest.json") or members.get("manifest")
        signature_member = members.get("manifest.json.sig") or members.get("manifest.sig")
        if manifest_member is None or signature_member is None:
            raise ValueError("Bundle is missing manifest or signature")

        for member in members.values():
            _assert_safe_member(member)

        manifest = _read_json(archive, manifest_member)
        signature_doc = _read_json(archive, signature_member)
        _verify_manifest_signature(
            archive=archive,
            manifest_member=manifest_member,
            manifest=manifest,
            signature_doc=signature_doc,
            signing_key=signing_key,
        )

        files = manifest.get("files")
        if not isinstance(files, Iterable):
            raise ValueError("Manifest does not expose a file list")

        for entry in files:
            if not isinstance(entry, Mapping):
                raise ValueError("Manifest entry must be a mapping")
            path = entry.get("path")
            digest = entry.get("sha256")
            if not isinstance(path, str) or not isinstance(digest, str):
                raise ValueError("Manifest entry requires 'path' and 'sha256'")
            member = members.get(path) or members.get(f"./{path}")
            if member is None:
                raise ValueError(f"Bundle is missing artefact: {path}")
            computed = _compute_sha256(archive, member)
            if computed != digest:
                raise ValueError(
                    f"Digest mismatch for {path}: expected {digest}, computed {computed}"
                )

        log.info("Resilience bundle verification successful: %s", resolved_bundle)


def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", required=True, type=Path, help="Path to the resilience bundle")
    parser.add_argument("--signing-key-path", type=Path, help="File containing the HMAC key")
    parser.add_argument("--signing-key-env", help="Environment variable providing the HMAC key")
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parse_args(argv)
    logging.basicConfig(level=args.log_level, format="%(asctime)s %(levelname)s %(message)s")
    try:
        signing_key = _load_signing_key(args.signing_key_path, args.signing_key_env)
        verify_resilience_bundle(bundle_path=args.bundle, signing_key=signing_key)
    except Exception as exc:  # pragma: no cover - handled in CLI tests
        LOGGER.error("Verification failed: %s", exc)
        return 2
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

