"""Core OEM bundle builder.

This module assembles daemon/UI binaries together with signed configuration
artifacts into a single archive per platform.  The generated bundle is offline
friendly and ships with bootstrap scripts that verify device fingerprint data
before installation.
"""

from __future__ import annotations

import argparse
import base64
import datetime as _dt
import hashlib
import hmac
import json
import logging
import os
import shutil
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional

from bot_core.security.signing import canonical_json_bytes

BUNDLE_NAME = "core-oem"
SUPPORTED_PLATFORMS = {"linux", "macos", "windows"}


def _now_utc() -> str:
    return _dt.datetime.utcnow().replace(microsecond=0).isoformat() + "Z"


@dataclass
class BundleInputs:
    """Container for bundle inputs."""

    daemon_paths: List[Path]
    ui_paths: List[Path]
    config_paths: Dict[str, Path]
    resources: Dict[str, List[Path]] = field(default_factory=dict)
    fingerprint_placeholder: str = "UNPROVISIONED"


class SignatureManager:
    """Computes hashes and HMAC signatures for bundle artifacts."""

    def __init__(
        self,
        key: bytes,
        *,
        digest_algorithm: str = "sha384",
        key_id: Optional[str] = None,
    ) -> None:
        if len(key) < 32:
            raise ValueError("Signing key must be at least 32 bytes")
        if not hasattr(hashlib, digest_algorithm):
            raise ValueError(f"Unsupported digest algorithm: {digest_algorithm}")
        self._key = key
        self._digest_algorithm = digest_algorithm
        self._key_id = key_id
        self._signature_algorithm = f"HMAC-{digest_algorithm.upper()}"

    def digest_file(self, path: Path) -> str:
        hasher = hashlib.new(self._digest_algorithm)
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                hasher.update(chunk)
        return hasher.hexdigest()

    def build_signature_document(self, payload: Mapping[str, Any]) -> Dict[str, Any]:
        digest = hmac.new(
            self._key,
            canonical_json_bytes(payload),
            getattr(hashlib, self._digest_algorithm),
        ).digest()
        signature = {
            "algorithm": self._signature_algorithm,
            "value": base64.b64encode(digest).decode("ascii"),
        }
        if self._key_id:
            signature["key_id"] = self._key_id
        return {
            "payload": dict(payload),
            "signature": signature,
        }

    def write_signature_document(
        self,
        payload: Mapping[str, Any],
        destination: Path,
    ) -> None:
        document = self.build_signature_document(payload)
        destination.write_text(
            json.dumps(document, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )


class CoreBundleBuilder:
    """Assembles the Core OEM bundle."""

    def __init__(
        self,
        *,
        platform: str,
        version: str,
        signing_key: bytes,
        output_dir: Path,
        inputs: BundleInputs,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        if platform not in SUPPORTED_PLATFORMS:
            raise ValueError(f"Unsupported platform: {platform}")
        self.platform = platform
        self.version = version
        self.signatures = SignatureManager(signing_key)
        self.output_dir = output_dir
        self.inputs = inputs
        self.logger = logger or logging.getLogger(__name__)

    def build(self) -> Path:
        with tempfile.TemporaryDirectory(prefix="core_oem_") as temp_dir:
            staging_root = Path(temp_dir) / "core_oem_staging"
            staging_root.mkdir(parents=True)
            self._stage_components(staging_root)
            manifest = self._build_manifest(staging_root)
            manifest_path = staging_root / "manifest.json"
            manifest_path.write_text(
                json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
            )
            manifest_payload = {
                "path": "manifest.json",
                "sha384": self.signatures.digest_file(manifest_path),
            }
            self.signatures.write_signature_document(
                manifest_payload, manifest_path.with_suffix(".sig")
            )
            archive_path = self._archive(staging_root)
            destination = self.output_dir / archive_path.name
            self.logger.info("Bundle created at %s", archive_path)
            shutil.copy2(archive_path, destination)
            return destination

    def _stage_components(self, staging_root: Path) -> None:
        (staging_root / "daemon").mkdir()
        (staging_root / "ui").mkdir()
        (staging_root / "config").mkdir()
        bootstrap_dir = staging_root / "bootstrap"
        bootstrap_dir.mkdir()

        for path in self.inputs.daemon_paths:
            self._copy_path(path, staging_root / "daemon")

        for path in self.inputs.ui_paths:
            self._copy_path(path, staging_root / "ui")

        for relative, source in self.inputs.config_paths.items():
            destination = staging_root / "config" / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            self._copy_single_file(source, destination)
            payload = {
                "path": destination.relative_to(staging_root).as_posix(),
                "sha384": self.signatures.digest_file(destination),
            }
            signature_path = destination.parent / f"{destination.name}.sig"
            self.signatures.write_signature_document(payload, signature_path)

        for directory, paths in self.inputs.resources.items():
            target_dir = staging_root / directory
            target_dir.mkdir(parents=True, exist_ok=True)
            for resource in paths:
                self._copy_path(resource, target_dir)

        self._write_fingerprint_template(staging_root)
        self._write_bootstrap_scripts(staging_root / "bootstrap")

    def _copy_path(self, source: Path, destination_dir: Path) -> None:
        if source.is_dir():
            shutil.copytree(source, destination_dir / source.name)
        else:
            self._copy_single_file(source, destination_dir / source.name)

    def _copy_single_file(self, source: Path, destination: Path) -> None:
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)

    def _write_fingerprint_template(self, staging_root: Path) -> None:
        expected_path = staging_root / "config" / "fingerprint.expected.json"
        payload = {
            "fingerprint": self.inputs.fingerprint_placeholder,
            "generated_at": _now_utc(),
        }
        document = self.signatures.build_signature_document(payload)
        expected_path.write_text(
            json.dumps(document, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )

    def _write_bootstrap_scripts(self, bootstrap_dir: Path) -> None:
        verify_script = bootstrap_dir / "verify_fingerprint.py"
        verify_script.write_text(_VERIFY_FINGERPRINT_TEMPLATE, encoding="utf-8")
        os.chmod(verify_script, 0o755)

        install_script = bootstrap_dir / "install.sh"
        install_script.write_text(_INSTALL_SH_TEMPLATE, encoding="utf-8")
        os.chmod(install_script, 0o755)

        powershell_script = bootstrap_dir / "install.ps1"
        powershell_script.write_text(_INSTALL_PS1_TEMPLATE, encoding="utf-8")

    def _build_manifest(self, staging_root: Path) -> Dict[str, object]:
        files: List[Dict[str, str]] = []
        for path in sorted(staging_root.rglob("*")):
            if path.is_file():
                relative = path.relative_to(staging_root).as_posix()
                digest = self.signatures.digest_file(path)
                files.append({"path": relative, "sha384": digest})
        manifest = {
            "bundle": BUNDLE_NAME,
            "platform": self.platform,
            "version": self.version,
            "created": _now_utc(),
            "files": files,
        }
        return manifest

    def _archive(self, staging_root: Path) -> Path:
        base_name = f"{BUNDLE_NAME}-{self.version}-{self.platform}"
        archive_base = Path(staging_root.parent) / base_name
        format_name = "gztar"
        if self.platform == "windows":
            format_name = "zip"
        archive = shutil.make_archive(
            str(archive_base),
            format_name,
            root_dir=staging_root.parent,
            base_dir=staging_root.name,
        )
        return Path(archive)


_VERIFY_FINGERPRINT_TEMPLATE = '''#!/usr/bin/env python3
"""Fingerprint verification helper for Core OEM bundles."""

import argparse
import base64
import hashlib
import hmac
import importlib
import importlib.util
import json
import os
from pathlib import Path


_CANONICAL_SEPARATORS = (",", ":")


def _canonical_json_bytes(payload):
    return json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=_CANONICAL_SEPARATORS,
    ).encode("utf-8")


def _load_expected(path: Path):
    document = json.loads(path.read_text(encoding="utf-8"))
    if "payload" not in document or "signature" not in document:
        raise SystemExit("fingerprint.expected.json has invalid structure")
    return document["payload"], document["signature"]


def _load_key():
    key_b64 = os.environ.get("OEM_BUNDLE_HMAC_KEY")
    if not key_b64:
        raise SystemExit("OEM_BUNDLE_HMAC_KEY environment variable is required")
    return base64.b64decode(key_b64)


def _read_runtime_fingerprint():
    env_value = os.environ.get("OEM_FINGERPRINT")
    if env_value:
        return env_value.strip()
    spec = importlib.util.find_spec("bot_core.security.fingerprint")
    if spec is None:
        raise SystemExit(
            "Fingerprint module not available; set OEM_FINGERPRINT to continue"
        )
    module = importlib.import_module("bot_core.security.fingerprint")
    if hasattr(module, "get_local_fingerprint"):
        candidate = module.get_local_fingerprint()
        return str(candidate).strip()
    raise SystemExit(
        "bot_core.security.fingerprint missing get_local_fingerprint()"
    )


def _verify_signature(payload, signature, key: bytes):
    algorithm = signature.get("algorithm", "HMAC-SHA384")
    if algorithm.upper() != "HMAC-SHA384":
        raise SystemExit(f"Unsupported signature algorithm: {algorithm}")
    value = signature.get("value")
    if not value:
        raise SystemExit("Signature missing 'value' field")
    digest = hmac.new(key, _canonical_json_bytes(payload), hashlib.sha384)
    computed = base64.b64encode(digest.digest()).decode("ascii")
    if computed != value:
        raise SystemExit("Fingerprint signature verification failed")


def _verify(payload, signature, runtime_value: str, key: bytes):
    expected = payload.get("fingerprint")
    if not expected:
        raise SystemExit("Fingerprint payload missing 'fingerprint'")
    _verify_signature(payload, signature, key)
    if runtime_value != expected:
        raise SystemExit("Fingerprint mismatch between bundle and runtime")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--expected",
        type=Path,
        default=Path(__file__).resolve().parent.parent
        / "config"
        / "fingerprint.expected.json",
    )
    args = parser.parse_args()
    payload, signature = _load_expected(args.expected)
    key = _load_key()
    runtime_value = _read_runtime_fingerprint()
    _verify(payload, signature, runtime_value, key)
    key_id = signature.get("key_id", "n/a")
    print(f"[core-oem] Fingerprint verified for {runtime_value} (key_id={key_id})")


if __name__ == "__main__":
    main()
'''

_INSTALL_SH_TEMPLATE = """#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUNDLE_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

python3 "${SCRIPT_DIR}/verify_fingerprint.py" --expected "${BUNDLE_ROOT}/config/fingerprint.expected.json"

echo "[core-oem] Fingerprint verified. Proceed with installing daemon and UI." \
     "Ensure binaries from 'daemon' and 'ui' directories are placed in target locations."
"""

_INSTALL_PS1_TEMPLATE = """Param()
$ErrorActionPreference = "Stop"

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$BundleRoot = Resolve-Path (Join-Path $ScriptDir "..")

python "${ScriptDir}/verify_fingerprint.py" --expected (Join-Path $BundleRoot "config/fingerprint.expected.json")
Write-Host "[core-oem] Fingerprint verified. Proceed with Windows service installation."
"""


def _parse_config_arguments(values: Iterable[str]) -> Dict[str, Path]:
    resolved: Dict[str, Path] = {}
    for raw in values:
        if "=" not in raw:
            raise ValueError(f"Invalid config specification: {raw}")
        name, path_str = raw.split("=", 1)
        path = Path(path_str).expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(path)
        resolved[name] = path
    return resolved


def _resolve_paths(values: Iterable[str]) -> List[Path]:
    result: List[Path] = []
    for raw in values:
        path = Path(raw).expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(path)
        result.append(path)
    return result


def build_from_cli(argv: Optional[List[str]] = None) -> Path:
    parser = argparse.ArgumentParser(description="Build Core OEM bundle")
    parser.add_argument("--platform", required=True, choices=sorted(SUPPORTED_PLATFORMS))
    parser.add_argument("--version", required=True)
    parser.add_argument("--signing-key-path", required=True)
    parser.add_argument(
        "--daemon",
        dest="daemon",
        action="append",
        required=True,
        help="Path to daemon artifact (file or directory)",
    )
    parser.add_argument(
        "--ui",
        dest="ui",
        action="append",
        required=True,
        help="Path to UI artifact (file or directory)",
    )
    parser.add_argument(
        "--config",
        dest="config",
        action="append",
        required=True,
        help="Configuration file entry in the form name=path",
    )
    parser.add_argument(
        "--resource",
        dest="resource",
        action="append",
        default=[],
        help="Additional resource path in the form directory=path",
    )
    parser.add_argument(
        "--fingerprint-placeholder",
        default="UNPROVISIONED",
        help="Placeholder fingerprint value stored in bundle",
    )
    parser.add_argument(
        "--output-dir",
        default="var/dist",
        help="Destination directory for bundle archives",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    args = parser.parse_args(argv)

    logging.basicConfig(level=getattr(logging, args.log_level))

    key_path = Path(args.signing_key_path).expanduser().resolve()
    signing_key = key_path.read_bytes()

    daemon_paths = _resolve_paths(args.daemon)
    ui_paths = _resolve_paths(args.ui)
    config_paths = _parse_config_arguments(args.config)

    resources: Dict[str, List[Path]] = {}
    for entry in args.resource:
        if "=" not in entry:
            raise ValueError(f"Invalid resource specification: {entry}")
        directory, raw_path = entry.split("=", 1)
        resource_path = Path(raw_path).expanduser().resolve()
        if not resource_path.exists():
            raise FileNotFoundError(resource_path)
        resources.setdefault(directory, []).append(resource_path)

    inputs = BundleInputs(
        daemon_paths=daemon_paths,
        ui_paths=ui_paths,
        config_paths=config_paths,
        resources=resources,
        fingerprint_placeholder=args.fingerprint_placeholder,
    )

    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    builder = CoreBundleBuilder(
        platform=args.platform,
        version=args.version,
        signing_key=signing_key,
        output_dir=output_dir,
        inputs=inputs,
    )
    return builder.build()


def main(argv: Optional[List[str]] = None) -> int:
    build_from_cli(argv)
    return 0


if __name__ == "__main__":
    sys.exit(main())
