"""Core OEM bundle builder.

This module assembles daemon/UI binaries together with signed configuration
artifacts into a single archive per platform. The generated bundle is offline
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
import stat
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import Any, Dict, Iterable, List, Mapping, Optional

try:  # pragma: no cover - optional dependency
    import yaml
except Exception:  # pragma: no cover - optional dependency
    yaml = None

_PACKAGE_ROOT = Path(__file__).resolve().parent
REPO_ROOT = _PACKAGE_ROOT.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from bot_core.security.signing import canonical_json_bytes

if __package__ in (None, ""):
    from deploy.packaging.pipeline import (  # type: ignore
        PackagingContext,
        PackagingPipeline,
        PackagingPipelineReport,
        build_pipeline_from_mapping,
    )
else:
    from .pipeline import (
        PackagingContext,
        PackagingPipeline,
        PackagingPipelineReport,
        build_pipeline_from_mapping,
    )

BUNDLE_NAME = "core-oem"
SUPPORTED_PLATFORMS = {"linux", "macos", "windows"}
_DEFAULT_DRY_RUN_VERSION = "0.0.0-dry-run"
_SAMPLES_ROOT = Path(__file__).with_name("samples")
_DRY_RUN_SAMPLE_DAEMON = _SAMPLES_ROOT / "daemon"
_DRY_RUN_SAMPLE_UI = _SAMPLES_ROOT / "ui"
_DRY_RUN_SAMPLE_CONFIG = _SAMPLES_ROOT / "config" / "core.yaml"
_DRY_RUN_SAMPLE_RESOURCE_DIR = _SAMPLES_ROOT / "resources" / "extras"
_DRY_RUN_SAMPLE_SIGNING_KEY = _SAMPLES_ROOT / "signing.key"


@dataclass
class _DryRunPlaceholderAssets:
    temp_dir: tempfile.TemporaryDirectory
    daemon_dir: Path
    ui_dir: Path
    config_file: Path
    resource_dir: Path
    signing_key: Path


_DRY_RUN_PLACEHOLDER_ASSETS: Optional[_DryRunPlaceholderAssets] = None
_RESERVED_RESOURCE_PREFIXES = {"daemon", "ui", "config", "bootstrap"}
_RESERVED_RESOURCE_PREFIXES_CASEFOLD = {prefix.casefold() for prefix in _RESERVED_RESOURCE_PREFIXES}
_RESERVED_CONFIG_PATHS = {
    "fingerprint.expected.json",
    "fingerprint.expected.json.sig",
}
_WINDOWS_DEVICE_NAMES = {
    "con",
    "prn",
    "aux",
    "nul",
    "clock$",
    *{f"com{i}" for i in range(1, 10)},
    *{f"lpt{i}" for i in range(1, 10)},
}
_WINDOWS_INVALID_CHARS = set("<>:\\|?*")
_ALLOWED_VERSION_CHARS = set("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789._-")
_ALLOWED_FINGERPRINT_PLACEHOLDER_CHARS = set("ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-")


def _load_pipeline_config(path: Path) -> Mapping[str, object]:
    """Load a pipeline configuration from JSON or YAML."""

    path = path.expanduser()
    if not path.exists():
        raise FileNotFoundError(path)
    payload = path.read_text(encoding="utf-8")
    data: Any
    if path.suffix.lower() in {".yaml", ".yml"}:
        if yaml is None:
            raise RuntimeError("PyYAML is required to parse YAML pipeline configuration files")
        data = yaml.safe_load(payload)
    else:
        try:
            data = json.loads(payload)
        except json.JSONDecodeError:
            if yaml is None:
                raise
            data = yaml.safe_load(payload)
    if not isinstance(data, Mapping):
        raise ValueError("Pipeline configuration must be a mapping")
    return data


def _write_pipeline_report(
    builder: "CoreBundleBuilder",
    *,
    report_path: Path,
    archive_path: Path,
) -> None:
    """Zapisz raport pipeline'u do pliku JSON."""

    report = builder.last_pipeline_report
    payload: dict[str, object] = {
        "bundle": BUNDLE_NAME,
        "platform": builder.platform,
        "version": builder.version,
        "archive_path": str(archive_path),
        "pipeline": report.to_mapping() if report is not None else None,
    }

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    builder.logger.info("Pipeline report zapisany do %s", report_path)


def _casefold_path(value: str) -> str:
    """Return a POSIX-style path converted to a case-insensitive key."""
    return value.casefold()


def _is_prefix(candidate: str, prefix: str) -> bool:
    """Return ``True`` if ``prefix`` is a direct prefix of ``candidate``."""
    return candidate == prefix or candidate.startswith(f"{prefix}/")


def _now_utc() -> str:
    return _dt.datetime.utcnow().replace(microsecond=0).isoformat() + "Z"


def _validate_bundle_version(value: str) -> str:
    """Ensure bundle version strings are filename-safe across platforms."""
    if not value:
        raise ValueError("Version string cannot be empty")
    if len(value) > 64:
        raise ValueError("Version string must be at most 64 characters")
    for ch in value:
        codepoint = ord(ch)
        if codepoint < 32 or codepoint == 127:
            raise ValueError(f"Version string contains control character U+{codepoint:04X}")
        if ch not in _ALLOWED_VERSION_CHARS:
            raise ValueError(f"Version string contains unsupported character: {ch!r}")
    if not value[0].isalnum():
        raise ValueError("Version string must start with an alphanumeric character")
    if not value[-1].isalnum():
        raise ValueError("Version string must end with an alphanumeric character")
    return value


def _validate_fingerprint_placeholder(value: str) -> str:
    """Ensure fingerprint placeholders are simple, portable tokens."""
    if not value:
        raise ValueError("Fingerprint placeholder cannot be empty")
    if len(value) > 128:
        raise ValueError("Fingerprint placeholder must be at most 128 characters")
    if value.strip() != value:
        raise ValueError("Fingerprint placeholder must not start or end with whitespace")
    for ch in value:
        codepoint = ord(ch)
        if codepoint < 32 or codepoint == 127:
            raise ValueError("Fingerprint placeholder contains control characters")
        if ch not in _ALLOWED_FINGERPRINT_PLACEHOLDER_CHARS:
            raise ValueError("Fingerprint placeholder contains unsupported character")
    return value


def _ensure_windows_safe_component(*, component: str, label: str, context: str) -> None:
    """Validate a single path component for Windows compatibility."""
    trimmed = component.rstrip(" .")
    if trimmed != component:
        raise ValueError(f"{label} contains entry ending with a space or dot: {context}")
    if not trimmed:
        raise ValueError(f"{label} contains empty path component: {context}")
    control_char = next((ch for ch in trimmed if ord(ch) < 32 or ord(ch) == 127), None)
    if control_char is not None:
        codepoint = ord(control_char)
        raise ValueError(f"{label} contains control character U+{codepoint:04X}: {context}")
    base, _, _ = trimmed.partition(".")
    if base.casefold() in _WINDOWS_DEVICE_NAMES:
        raise ValueError(f"{label} contains Windows reserved device name: {context}")
    invalid_char = next((c for c in trimmed if c in _WINDOWS_INVALID_CHARS), None)
    if invalid_char is not None:
        raise ValueError(f"{label} contains character '{invalid_char}' disallowed on Windows: {context}")


def _ensure_windows_safe_virtual_path(path: PurePosixPath, *, label: str) -> None:
    """Validate that bundle-internal path components are Windows compatible."""
    for component in path.parts:
        _ensure_windows_safe_component(component=component, label=label, context=path.as_posix())


def _ensure_windows_safe_tree(path: Path, *, label: str) -> None:
    """Validate on-disk paths to avoid Windows-incompatible components."""
    _ensure_windows_safe_component(component=path.name, label=label, context=str(path))
    if not path.is_dir():
        return
    for current_root, dirnames, filenames in os.walk(path):
        current_path = Path(current_root)
        for dirname in dirnames:
            candidate = current_path / dirname
            _ensure_windows_safe_component(component=dirname, label=label, context=str(candidate))
        for filename in filenames:
            candidate = current_path / filename
            _ensure_windows_safe_component(component=filename, label=label, context=str(candidate))


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
        digest = hmac.new(self._key, canonical_json_bytes(payload), getattr(hashlib, self._digest_algorithm)).digest()
        signature = {
            "algorithm": self._signature_algorithm,
            "value": base64.b64encode(digest).decode("ascii"),
        }
        if self._key_id:
            signature["key_id"] = self._key_id
        return {"payload": dict(payload), "signature": signature}

    def write_signature_document(self, payload: Mapping[str, Any], destination: Path) -> None:
        document = self.build_signature_document(payload)
        destination.write_text(json.dumps(document, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _iter_symlinks(root: Path) -> Iterable[Path]:
    """Yield every symlink at or below ``root`` without following links."""
    if root.is_symlink():
        yield root
        return
    if not root.is_dir():
        return
    for current_root, dirnames, filenames in os.walk(root, followlinks=False):
        current_path = Path(current_root)
        for dirname in list(dirnames):
            candidate = current_path / dirname
            if candidate.is_symlink():
                yield candidate
        for filename in filenames:
            candidate = current_path / filename
            if candidate.is_symlink():
                yield candidate


def _ensure_no_symlinks(path: Path, *, label: str) -> None:
    """Raise ``ValueError`` if ``path`` or any children are symlinks."""
    if path.is_symlink():
        raise ValueError(f"{label} must not be a symlink: {path}")
    for candidate in _iter_symlinks(path):
        raise ValueError(f"{label} contains forbidden symlink: {candidate}")


def _ensure_casefold_safe_tree(path: Path, *, label: str) -> None:
    """Ensure ``path`` does not contain siblings differing only by case."""
    if not path.is_dir():
        return
    for current_root, dirnames, filenames in os.walk(path):
        seen: Dict[str, Path] = {}
        current_path = Path(current_root)
        for name in list(dirnames) + list(filenames):
            candidate = current_path / name
            key = name.casefold()
            existing = seen.get(key)
            if existing is None:
                seen[key] = candidate
                continue
            if existing == candidate:
                continue
            if existing.name == candidate.name:
                continue
            raise ValueError(
                "{} contains entries that would conflict on a "
                "case-insensitive filesystem: {} vs {}".format(label, existing, candidate)
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
        ensure_output_dir: bool = True,
        pipeline: PackagingPipeline | None = None,
    ) -> None:
        if platform not in SUPPORTED_PLATFORMS:
            raise ValueError(f"Unsupported platform: {platform}")
        self.platform = platform
        self.version = _validate_bundle_version(version)
        self.signatures = SignatureManager(signing_key)

        prepared_output_dir = output_dir.expanduser()
        if prepared_output_dir.exists():
            if prepared_output_dir.is_symlink():
                raise ValueError(f"Output directory must not be a symlink: {prepared_output_dir}")
            if not prepared_output_dir.is_dir():
                raise ValueError(f"Output directory is not a directory: {prepared_output_dir}")
        elif ensure_output_dir:
            prepared_output_dir.mkdir(parents=True, exist_ok=True)

        self.output_dir = prepared_output_dir.resolve()
        self.inputs = BundleInputs(
            daemon_paths=list(inputs.daemon_paths),
            ui_paths=list(inputs.ui_paths),
            config_paths=dict(inputs.config_paths),
            resources={directory: list(paths) for directory, paths in inputs.resources.items()},
            fingerprint_placeholder=_validate_fingerprint_placeholder(inputs.fingerprint_placeholder),
        )
        self.logger = logger or logging.getLogger(__name__)
        self._pipeline = pipeline
        self._last_pipeline_report = None

    def expected_archive_path(self) -> Path:
        """Return the destination path for the bundle archive."""

        return self.output_dir / self._expected_archive_name()

    @property
    def last_pipeline_report(self) -> Optional[PackagingPipelineReport]:
        """Return the last pipeline report produced by :meth:`build`."""

        return self._last_pipeline_report

    def build(self) -> Path:
        destination = self.expected_archive_path()
        expected_archive_name = destination.name
        if destination.exists():
            raise FileExistsError(f"Bundle archive already exists: {destination}")

        self._last_pipeline_report = None
        with tempfile.TemporaryDirectory(prefix="core_oem_") as temp_dir:
            staging_root = Path(temp_dir) / "core_oem_staging"
            staging_root.mkdir(parents=True)
            self._stage_components(staging_root)
            manifest = self._build_manifest(staging_root)
            manifest_path = staging_root / "manifest.json"
            manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
            manifest_payload = {
                "path": "manifest.json",
                "sha384": self.signatures.digest_file(manifest_path),
            }
            self.signatures.write_signature_document(manifest_payload, manifest_path.with_suffix(".sig"))
            archive_path = self._archive(staging_root)
            if archive_path.name != expected_archive_name:
                raise RuntimeError("Archive name mismatch between expected and generated bundle")
            self.logger.info("Bundle created at %s", archive_path)
            shutil.copy2(archive_path, destination)
            if self._pipeline is not None:
                context = PackagingContext(
                    staging_root=staging_root,
                    archive_path=destination,
                    manifest=manifest,
                )
                self._last_pipeline_report = self._pipeline.execute(context)
            return destination

    def _expected_archive_name(self) -> str:
        suffix = ".zip" if self.platform == "windows" else ".tar.gz"
        return f"{BUNDLE_NAME}-{self.version}-{self.platform}{suffix}"

    def _stage_components(self, staging_root: Path) -> None:
        (staging_root / "daemon").mkdir()
        (staging_root / "ui").mkdir()
        (staging_root / "config").mkdir()
        bootstrap_dir = staging_root / "bootstrap"
        bootstrap_dir.mkdir()

        for path in self.inputs.daemon_paths:
            _ensure_no_symlinks(path, label="Daemon artifact")
            self._copy_path(path, staging_root / "daemon")

        for path in self.inputs.ui_paths:
            _ensure_no_symlinks(path, label="UI artifact")
            self._copy_path(path, staging_root / "ui")

        for relative, source in self.inputs.config_paths.items():
            _ensure_no_symlinks(source, label=f"Config entry '{relative}'")
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
                _ensure_no_symlinks(resource, label=f"Resource '{directory}'")
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
        expected_path.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n", encoding="utf-8")

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
import sys
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


def _log_install_event(*, fingerprint, status, key_id=None, expected=None, expected_path=None, message=None):
    try:
        module = importlib.import_module("bot_core.security.fingerprint")
    except Exception:
        return
    writer = getattr(module, "append_fingerprint_audit", None)
    if writer is None:
        return
    metadata = {
        "script": str(Path(__file__).resolve()),
    }
    if expected is not None:
        metadata["expected_fingerprint"] = expected
    if expected_path is not None:
        metadata["expected_file"] = str(expected_path)
    if message:
        metadata["message"] = message
    try:
        writer(
            event="installer_run",
            fingerprint=fingerprint or "",
            status=status,
            key_id=key_id,
            metadata=metadata,
        )
    except Exception as exc:  # pragma: no cover - ostrzeżenie przy problemie z audytem
        print(f"[core-oem] Warning: failed to write fingerprint audit: {exc}", file=sys.stderr)


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
    runtime_value = None
    key_identifier = signature.get("key_id")
    expected_value = payload.get("fingerprint")
    try:
        key = _load_key()
        runtime_value = _read_runtime_fingerprint()
        _verify(payload, signature, runtime_value, key)
    except SystemExit as exc:
        _log_install_event(
            fingerprint=runtime_value,
            status="failed",
            key_id=key_identifier,
            expected=expected_value,
            expected_path=args.expected,
            message=str(exc),
        )
        raise
    except Exception as exc:  # pragma: no cover - obsługa wyjątków instalatora
        _log_install_event(
            fingerprint=runtime_value,
            status="error",
            key_id=key_identifier,
            expected=expected_value,
            expected_path=args.expected,
            message=str(exc),
        )
        raise
    else:
        _log_install_event(
            fingerprint=runtime_value,
            status="verified",
            key_id=key_identifier,
            expected=expected_value,
            expected_path=args.expected,
        )
        key_label = key_identifier if key_identifier is not None else "n/a"
        print(f"[core-oem] Fingerprint verified for {runtime_value} (key_id={key_label})")


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


def _normalize_bundle_relative(value: str) -> str:
    """Return a normalized bundle-internal relative path.

    Values are expressed with POSIX separators regardless of host platform.
    Absolute paths and parent directory traversals are rejected to prevent
    accidental escape from the bundle staging directory.
    """
    normalized = value.replace("\\", "/")
    candidate = PurePosixPath(normalized)
    if candidate.is_absolute():
        raise ValueError(f"Bundle entries must be relative: {value}")
    parts = [part for part in candidate.parts if part not in ("", ".")]
    if not parts:
        raise ValueError(f"Bundle entry cannot be empty: {value}")
    if any(part == ".." for part in parts):
        raise ValueError(f"Parent traversal is not allowed: {value}")
    normalized_path = PurePosixPath(*parts)
    _ensure_windows_safe_virtual_path(normalized_path, label="Bundle entry")
    return normalized_path.as_posix()


def _parse_config_arguments(values: Iterable[str]) -> Dict[str, Path]:
    resolved: Dict[str, Path] = {}
    resolved_casefold: Dict[str, str] = {}
    signature_targets_casefold: set[str] = set()
    for raw in values:
        if "=" not in raw:
            raise ValueError(f"Invalid config specification: {raw}")
        name, path_str = raw.split("=", 1)
        normalized_name = _normalize_bundle_relative(name)
        normalized_lower = _casefold_path(normalized_name)

        for reserved in _RESERVED_CONFIG_PATHS:
            reserved_lower = _casefold_path(reserved)
            if _is_prefix(normalized_lower, reserved_lower):
                raise ValueError("Config entry cannot use reserved name: " f"{normalized_name}")
        if normalized_lower.endswith(".sig"):
            raise ValueError("Config entry must not end with '.sig' because signatures are generated automatically")

        if normalized_lower in resolved_casefold:
            raise ValueError("Config entry would conflict on a case-insensitive filesystem: " f"{normalized_name}")

        for existing_lower, existing_name in resolved_casefold.items():
            if _is_prefix(normalized_lower, existing_lower):
                raise ValueError("Config entry nests within another entry: " f"{normalized_name}")
            if _is_prefix(existing_lower, normalized_lower):
                raise ValueError(
                    "Config entry would become a parent directory of another entry: " f"{normalized_name}"
                )

        if normalized_lower in signature_targets_casefold or any(
            _is_prefix(normalized_lower, signature_lower) for signature_lower in signature_targets_casefold
        ):
            raise ValueError("Config entry conflicts with auto-generated signature file: " f"{normalized_name}")

        source_path = Path(path_str).expanduser()
        if not source_path.exists():
            raise FileNotFoundError(source_path)
        _ensure_no_symlinks(source_path, label=f"Config entry '{normalized_name}'")
        path = source_path.resolve()
        _ensure_windows_safe_tree(path, label=f"Config entry '{normalized_name}'")
        if not path.is_file():
            raise ValueError(f"Config entry '{normalized_name}' must reference a file: {path}")
        signature_name = f"{normalized_name}.sig"
        signature_lower = _casefold_path(signature_name)
        if signature_lower in resolved_casefold or any(
            _is_prefix(existing_lower, signature_lower) for existing_lower in resolved_casefold
        ):
            raise ValueError("Config entry would collide with another entry's name: " f"{signature_name}")

        for reserved in _RESERVED_CONFIG_PATHS:
            reserved_lower = _casefold_path(reserved)
            if signature_lower == reserved_lower or _is_prefix(reserved_lower, signature_lower):
                raise ValueError("Config entry would produce reserved signature path: " f"{signature_name}")

        resolved[normalized_name] = path
        resolved_casefold[normalized_lower] = normalized_name
        signature_targets_casefold.add(signature_lower)
    return resolved


def _resolve_paths(values: Iterable[str], *, label: str) -> List[Path]:
    result: List[Path] = []
    seen: set[Path] = set()
    seen_names: Dict[str, Path] = {}
    for raw in values:
        source_path = Path(raw).expanduser()
        if not source_path.exists():
            raise FileNotFoundError(source_path)
        _ensure_no_symlinks(source_path, label=label)
        path = source_path.resolve()
        _ensure_windows_safe_tree(path, label=label)
        if path in seen:
            raise ValueError(f"Duplicate {label} entry: {path}")
        seen.add(path)
        name_key = path.name.casefold()
        existing = seen_names.get(name_key)
        if existing is not None:
            raise ValueError(f"{label} names would conflict on a case-insensitive filesystem:" f" {existing} vs {path}")
        seen_names[name_key] = path
        if path.is_dir():
            _ensure_casefold_safe_tree(path, label=label)
        result.append(path)
    return result


def _ensure_dry_run_placeholders() -> _DryRunPlaceholderAssets:
    """Create placeholder assets for ``--dry-run`` when samples are unavailable."""

    global _DRY_RUN_PLACEHOLDER_ASSETS
    if _DRY_RUN_PLACEHOLDER_ASSETS is not None:
        return _DRY_RUN_PLACEHOLDER_ASSETS

    temp_dir = tempfile.TemporaryDirectory(prefix="core_bundle_dry_run_")
    base = Path(temp_dir.name)

    daemon_dir = base / "daemon"
    daemon_dir.mkdir(parents=True, exist_ok=True)
    (daemon_dir / "placeholder.txt").write_text(
        "Dry-run daemon placeholder artifact\n",
        encoding="utf-8",
    )

    ui_dir = base / "ui"
    ui_dir.mkdir(parents=True, exist_ok=True)
    (ui_dir / "placeholder.txt").write_text(
        "Dry-run UI placeholder artifact\n",
        encoding="utf-8",
    )

    config_dir = base / "config"
    config_dir.mkdir(parents=True, exist_ok=True)
    config_file = config_dir / "core.yaml"
    config_file.write_text(
        "# Generated dry-run configuration\ntrading:\n  mode: placeholder\n",
        encoding="utf-8",
    )

    resource_dir = base / "extras"
    resource_dir.mkdir(parents=True, exist_ok=True)
    (resource_dir / "placeholder.txt").write_text(
        "Dry-run extras placeholder\n",
        encoding="utf-8",
    )

    signing_key = base / "signing.key"
    signing_key.write_text(
        "dry-run-signing-key-material-placeholder-abcdef0123456789",
        encoding="utf-8",
    )

    _DRY_RUN_PLACEHOLDER_ASSETS = _DryRunPlaceholderAssets(
        temp_dir=temp_dir,
        daemon_dir=daemon_dir,
        ui_dir=ui_dir,
        config_file=config_file,
        resource_dir=resource_dir,
        signing_key=signing_key,
    )
    return _DRY_RUN_PLACEHOLDER_ASSETS


def _apply_dry_run_defaults(args: argparse.Namespace) -> None:
    """Populate CLI arguments with sample artifacts when running ``--dry-run``."""

    placeholders: Optional[_DryRunPlaceholderAssets] = None

    def _placeholders() -> _DryRunPlaceholderAssets:
        nonlocal placeholders
        if placeholders is None:
            placeholders = _ensure_dry_run_placeholders()
        return placeholders

    if not args.daemon:
        target = _DRY_RUN_SAMPLE_DAEMON if _DRY_RUN_SAMPLE_DAEMON.exists() else _placeholders().daemon_dir
        args.daemon = [str(target)]
    if not args.ui:
        target = _DRY_RUN_SAMPLE_UI if _DRY_RUN_SAMPLE_UI.exists() else _placeholders().ui_dir
        args.ui = [str(target)]
    if not args.config:
        if _DRY_RUN_SAMPLE_CONFIG.exists():
            config_path = _DRY_RUN_SAMPLE_CONFIG
        else:
            config_path = _placeholders().config_file
        args.config = [f"core.yaml={config_path}"]

    resource_entries = list(args.resource or [])
    if not resource_entries:
        if _DRY_RUN_SAMPLE_RESOURCE_DIR.exists():
            resource_entries.append(f"extras={_DRY_RUN_SAMPLE_RESOURCE_DIR}")
        else:
            resource_entries.append(f"extras={_placeholders().resource_dir}")
    args.resource = resource_entries

    if args.signing_key_path is None:
        if _DRY_RUN_SAMPLE_SIGNING_KEY.exists():
            key_path = _DRY_RUN_SAMPLE_SIGNING_KEY
        else:
            key_path = _placeholders().signing_key
        args.signing_key_path = str(key_path)


def _is_dry_run_autofilled_signing_key(path: Path) -> bool:
    """Return ``True`` if ``path`` corresponds to CLI-provided dry-run key material."""

    resolved = path.resolve()
    if resolved == _DRY_RUN_SAMPLE_SIGNING_KEY.resolve():
        return True
    assets = _DRY_RUN_PLACEHOLDER_ASSETS
    if assets is None:
        return False
    return resolved == assets.signing_key.resolve()


def build_from_cli(argv: Optional[List[str]] = None) -> Path:
    parser = argparse.ArgumentParser(description="Build Core OEM bundle")
    parser.add_argument("--platform", required=True, choices=sorted(SUPPORTED_PLATFORMS))
    parser.add_argument(
        "--version",
        required=False,
        help="Bundle version string (required unless --dry-run is specified)",
    )
    parser.add_argument(
        "--signing-key-path",
        required=False,
        help="Path to signing key (required unless --dry-run is specified)",
    )
    parser.add_argument(
        "--daemon",
        dest="daemon",
        action="append",
        required=False,
        help="Path to daemon artifact (file or directory). Required unless --dry-run is specified",
    )
    parser.add_argument(
        "--ui",
        dest="ui",
        action="append",
        required=False,
        help="Path to UI artifact (file or directory). Required unless --dry-run is specified",
    )
    parser.add_argument(
        "--config",
        dest="config",
        action="append",
        required=False,
        help="Configuration file entry in the form name=path. Required unless --dry-run is specified",
    )
    parser.add_argument(
        "--resource",
        dest="resource",
        action="append",
        required=False,
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
        "--pipeline-config",
        dest="pipeline_config",
        default=None,
        help="Path to JSON/YAML configuration enabling notarization, delta updates and fingerprint validation",
    )
    parser.add_argument(
        "--pipeline-report",
        dest="pipeline_report",
        default=None,
        help="Optional path to JSON file capturing results of pipeline kroków (notaryzacja, delty, fingerprint)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate inputs without creating bundle archives. When other arguments are omitted, sample artifacts from deploy/packaging/samples/ are used",
    )
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    args = parser.parse_args(argv)

    logging.basicConfig(level=getattr(logging, args.log_level))

    if args.resource is None:
        args.resource = []

    if args.dry_run:
        _apply_dry_run_defaults(args)

    version = args.version
    if version is None:
        if args.dry_run:
            version = _DEFAULT_DRY_RUN_VERSION
        else:
            raise ValueError("--version is required unless --dry-run is specified")
    version = str(version)

    if not args.daemon:
        raise ValueError("--daemon is required unless --dry-run is specified")
    if not args.ui:
        raise ValueError("--ui is required unless --dry-run is specified")
    if not args.config:
        raise ValueError("--config is required unless --dry-run is specified")

    if args.signing_key_path is None:
        if args.dry_run:
            signing_key = b"dry-run-signing-key-material-placeholder"
        else:
            raise ValueError("--signing-key-path is required unless --dry-run is specified")
    else:
        raw_key_path = Path(args.signing_key_path).expanduser()
        if not raw_key_path.exists():
            raise FileNotFoundError(raw_key_path)
        _ensure_no_symlinks(raw_key_path, label="Signing key path")
        key_path = raw_key_path.resolve()
        skip_permission_check = args.dry_run and _is_dry_run_autofilled_signing_key(key_path)
        _ensure_windows_safe_tree(key_path, label="Signing key path")
        if not key_path.is_file():
            raise ValueError(f"Signing key path must reference a file: {key_path}")
        if os.name != "nt" and not skip_permission_check:
            mode = key_path.stat().st_mode
            if mode & (stat.S_IRWXG | stat.S_IRWXO):
                raise ValueError("Signing key file permissions must restrict access to the owner: " f"{key_path}")

        signing_key = key_path.read_bytes()

    daemon_paths = _resolve_paths(args.daemon, label="Daemon artifact")
    ui_paths = _resolve_paths(args.ui, label="UI artifact")
    config_paths = _parse_config_arguments(args.config)

    resources: Dict[str, List[Path]] = {}
    resource_names: Dict[str, set[str]] = {}
    resource_dirs_casefold: Dict[str, str] = {}
    for entry in args.resource:
        if "=" not in entry:
            raise ValueError(f"Invalid resource specification: {entry}")
        directory, raw_path = entry.split("=", 1)
        normalized_directory = _normalize_bundle_relative(directory)
        normalized_directory_lower = _casefold_path(normalized_directory)
        existing_directory = resource_dirs_casefold.get(normalized_directory_lower)
        if existing_directory is None:
            resource_dirs_casefold[normalized_directory_lower] = normalized_directory
        elif existing_directory != normalized_directory:
            raise ValueError(
                "Resource directory would conflict on a case-insensitive filesystem: " f"{normalized_directory}"
            )
        top_level = normalized_directory.split("/", 1)[0]
        if _casefold_path(top_level) in _RESERVED_RESOURCE_PREFIXES_CASEFOLD:
            raise ValueError("Resource directory cannot use reserved prefix: " f"{normalized_directory}")
        source_path = Path(raw_path).expanduser()
        if not source_path.exists():
            raise FileNotFoundError(source_path)
        _ensure_no_symlinks(source_path, label=f"Resource '{normalized_directory}'")
        resource_path = source_path.resolve()
        _ensure_windows_safe_tree(resource_path, label=f"Resource '{normalized_directory}'")
        if resource_path.is_dir():
            _ensure_casefold_safe_tree(resource_path, label=f"Resource '{normalized_directory}'")
        basename = resource_path.name
        names = resource_names.setdefault(normalized_directory, set())
        basename_key = basename.casefold()
        if basename_key in names:
            raise ValueError("Duplicate resource entry for " f"{normalized_directory}/{basename}")
        names.add(basename_key)
        resources.setdefault(normalized_directory, []).append(resource_path)

    fingerprint_placeholder = _validate_fingerprint_placeholder(args.fingerprint_placeholder)

    inputs = BundleInputs(
        daemon_paths=daemon_paths,
        ui_paths=ui_paths,
        config_paths=config_paths,
        resources=resources,
        fingerprint_placeholder=fingerprint_placeholder,
    )

    raw_output_dir = Path(args.output_dir).expanduser()
    if raw_output_dir.exists():
        if raw_output_dir.is_symlink():
            raise ValueError(f"Output directory must not be a symlink: {raw_output_dir}")
        if not raw_output_dir.is_dir():
            raise ValueError(f"Output directory is not a directory: {raw_output_dir}")
    elif not args.dry_run:
        raw_output_dir.mkdir(parents=True, exist_ok=True)

    output_dir = raw_output_dir.resolve()

    pipeline: PackagingPipeline | None = None
    if args.pipeline_config:
        config_path = Path(args.pipeline_config).expanduser()
        pipeline_mapping = _load_pipeline_config(config_path)
        pipeline = build_pipeline_from_mapping(pipeline_mapping, base_dir=config_path.parent)

    builder = CoreBundleBuilder(
        platform=args.platform,
        version=version,
        signing_key=signing_key,
        output_dir=output_dir,
        inputs=inputs,
        ensure_output_dir=not args.dry_run,
        pipeline=pipeline,
    )
    if args.dry_run:
        destination = builder.expected_archive_path()
        if destination.exists():
            raise FileExistsError(f"Bundle archive already exists: {destination}")
        builder.logger.info("Dry run: validations completed. Bundle would be created at %s", destination)
        return destination

    destination = builder.build()
    if args.pipeline_report:
        report_path = Path(args.pipeline_report).expanduser()
        _write_pipeline_report(
            builder,
            report_path=report_path,
            archive_path=destination,
        )
    return destination


def main(argv: Optional[List[str]] = None) -> int:
    build_from_cli(argv)
    return 0


if __name__ == "__main__":
    sys.exit(main())
