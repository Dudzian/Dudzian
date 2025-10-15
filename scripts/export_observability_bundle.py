"""Build signed observability bundles for Grafana dashboards and alert rules.

The Stage4 release pipeline requires shipping a reproducible package that
contains the Grafana dashboards and Prometheus alert definitions used by the
multi-strategy runtime.  This script assembles the artefacts into a tarball,
generates a manifest with file metadata and signs it using the same HMAC
mechanism as the OEM bundle builders.

The resulting archive is suitable for offline deployment – operators can copy
the tarball to the observability host, verify the manifest signature and then
extract the dashboards/alerts into the provisioning directories described in
the Stage4 runbooks.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import json
import logging
import os
import shutil
import stat
import subprocess
import sys
import tarfile
import tempfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Dict, List, Mapping, Optional, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


from deploy.packaging.build_core_bundle import (  # type: ignore  # noqa: E402
    SignatureManager,
    _ensure_casefold_safe_tree,
    _ensure_no_symlinks,
    _ensure_windows_safe_component,
    _ensure_windows_safe_tree,
    _validate_bundle_version,
)


DEFAULT_DASHBOARD_DIRS = [
    REPO_ROOT / "deploy" / "grafana" / "provisioning" / "dashboards"
]
DEFAULT_ALERT_RULES = [
    REPO_ROOT / "deploy" / "prometheus" / "rules" / "multi_strategy_rules.yml",
    REPO_ROOT / "deploy" / "prometheus" / "stage5_alerts.yaml",
    REPO_ROOT / "deploy" / "prometheus" / "stage6_alerts.yaml",
]
ARCHIVE_PREFIX = "observability-bundle"


@dataclass(frozen=True)
class ObservabilityAsset:
    """Represents a dashboard or alert file included in the bundle."""

    virtual_path: PurePosixPath
    source: Path
    kind: str


def _now_utc() -> str:
    return _dt.datetime.utcnow().replace(microsecond=0).isoformat() + "Z"


def _git_revision() -> Optional[str]:
    try:
        output = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT)
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None
    return output.decode("ascii").strip()


def _ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def _load_signing_key(path: Path) -> bytes:
    _ensure_no_symlinks(path, label="Signing key path")
    resolved = path.resolve()
    _ensure_windows_safe_tree(resolved, label="Signing key path")
    if not resolved.is_file():
        raise ValueError(f"Signing key path must reference a file: {resolved}")
    if os.name != "nt":
        mode = resolved.stat().st_mode
        if mode & (stat.S_IRWXG | stat.S_IRWXO):
            raise ValueError(
                "Signing key permissions are too permissive; expected 600-style access"
            )
    return resolved.read_bytes()


def _validate_target_directory(path: Path) -> Path:
    path = path.expanduser()
    if path.exists():
        if path.is_symlink():
            raise ValueError(f"Output directory must not be a symlink: {path}")
        if not path.is_dir():
            raise ValueError(f"Output directory is not a directory: {path}")
    else:
        path.mkdir(parents=True)
    return path


def _collect_dashboard_files(
    directories: Sequence[Path], additional_files: Sequence[Path]
) -> List[ObservabilityAsset]:
    assets: List[ObservabilityAsset] = []
    seen: Dict[str, PurePosixPath] = {}

    def add_asset(source: Path, *, prefix: str) -> None:
        _ensure_no_symlinks(source, label=f"{prefix} source")
        resolved = source.resolve()
        _ensure_windows_safe_tree(resolved, label=f"{prefix} source")
        if not resolved.exists():
            raise FileNotFoundError(f"{prefix} not found: {resolved}")
        if resolved.is_dir():
            _ensure_casefold_safe_tree(resolved, label=f"{prefix} directory")
            raise ValueError(f"{prefix} path must be a file, not a directory: {resolved}")
        name = resolved.name
        _ensure_windows_safe_component(component=name, label=f"{prefix} name", context=name)
        virtual_path = PurePosixPath("dashboards") / name
        key = virtual_path.as_posix().casefold()
        if key in seen:
            raise ValueError(
                f"Duplicate dashboard entry detected: {virtual_path} clashes with {seen[key]}"
            )
        seen[key] = virtual_path
        assets.append(
            ObservabilityAsset(virtual_path=virtual_path, source=resolved, kind="dashboard")
        )

    for directory in directories:
        resolved = directory.resolve()
        if not resolved.exists():
            raise FileNotFoundError(f"Dashboard directory does not exist: {resolved}")
        if not resolved.is_dir():
            raise ValueError(f"Dashboard directory must be a directory: {resolved}")
        _ensure_no_symlinks(resolved, label="Dashboard directory")
        _ensure_windows_safe_tree(resolved, label="Dashboard directory")
        _ensure_casefold_safe_tree(resolved, label="Dashboard directory")
        for candidate in sorted(resolved.glob("*.json")):
            add_asset(candidate, prefix="Dashboard file")

    for file_path in additional_files:
        add_asset(file_path, prefix="Dashboard file")

    if not assets:
        raise ValueError("No dashboards were discovered; specify --dashboard or --dashboards-dir")
    return assets


def _collect_alert_files(files: Sequence[Path]) -> List[ObservabilityAsset]:
    assets: List[ObservabilityAsset] = []
    seen: Dict[str, PurePosixPath] = {}

    for file_path in files:
        _ensure_no_symlinks(file_path, label="Alert rule path")
        resolved = file_path.resolve()
        _ensure_windows_safe_tree(resolved, label="Alert rule path")
        if not resolved.exists():
            raise FileNotFoundError(f"Alert rule does not exist: {resolved}")
        if resolved.is_dir():
            _ensure_casefold_safe_tree(resolved, label="Alert rule directory")
            raise ValueError(f"Alert rule path must be a file: {resolved}")
        name = resolved.name
        _ensure_windows_safe_component(
            component=name, label="Alert rule name", context=name
        )
        virtual_path = PurePosixPath("alert_rules") / name
        key = virtual_path.as_posix().casefold()
        if key in seen:
            raise ValueError(
                f"Duplicate alert rule entry detected: {virtual_path} clashes with {seen[key]}"
            )
        seen[key] = virtual_path
        assets.append(
            ObservabilityAsset(virtual_path=virtual_path, source=resolved, kind="alert_rule")
        )

    if not assets:
        raise ValueError("At least one alert rule file must be provided")
    return assets


def _compute_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _build_manifest(
    *,
    version: str,
    assets: Sequence[ObservabilityAsset],
    staging_root: Path,
    dashboard_entries: Sequence[str],
    alert_entries: Sequence[str],
) -> Mapping[str, object]:
    files: List[Mapping[str, object]] = []
    for asset in assets:
        target = staging_root / asset.virtual_path.as_posix()
        files.append(
            {
                "path": asset.virtual_path.as_posix(),
                "kind": asset.kind,
                "sha256": _compute_sha256(target),
                "size_bytes": target.stat().st_size,
            }
        )

    manifest: Dict[str, object] = {
        "version": version,
        "generated_at": _now_utc(),
        "files": sorted(files, key=lambda entry: entry["path"]),
        "dashboards": list(dashboard_entries),
        "alert_rules": list(alert_entries),
    }

    revision = _git_revision()
    if revision:
        manifest["git_revision"] = revision
    return manifest


def _tar_directory(source: Path, destination: Path) -> None:
    with tarfile.open(destination, "w:gz") as archive:
        archive.add(source, arcname=".")


def build_observability_bundle(
    *,
    version: str,
    dashboards: Sequence[Path],
    alerts: Sequence[Path],
    output_dir: Path,
    signing_key: bytes,
    digest_algorithm: str = "sha384",
    key_id: Optional[str] = None,
    logger: Optional[logging.Logger] = None,
) -> Path:
    log = logger or logging.getLogger("observability_bundle")
    version = _validate_bundle_version(version)

    output_dir = _validate_target_directory(output_dir)
    archive_name = f"{ARCHIVE_PREFIX}-{version}.tar.gz"
    archive_path = output_dir / archive_name
    if archive_path.exists():
        raise FileExistsError(f"Bundle already exists: {archive_path}")

    dashboards = [path.expanduser() for path in dashboards]
    alerts = [path.expanduser() for path in alerts]

    dashboard_dirs: List[Path] = []
    dashboard_files: List[Path] = []
    for candidate in dashboards:
        if candidate.is_dir():
            dashboard_dirs.append(candidate)
        else:
            dashboard_files.append(candidate)

    dashboard_assets = _collect_dashboard_files(dashboard_dirs, dashboard_files)
    alert_assets = _collect_alert_files(alerts)
    assets = dashboard_assets + alert_assets

    staging_root = Path(tempfile.mkdtemp(prefix="observability_bundle_"))
    try:
        for asset in assets:
            destination = staging_root / asset.virtual_path.as_posix()
            _ensure_parent(destination)
            shutil.copy2(asset.source, destination)

        manifest = _build_manifest(
            version=version,
            assets=assets,
            staging_root=staging_root,
            dashboard_entries=[asset.virtual_path.as_posix() for asset in dashboard_assets],
            alert_entries=[asset.virtual_path.as_posix() for asset in alert_assets],
        )

        manifest_path = staging_root / "manifest.json"
        manifest_path.write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )

        signature_manager = SignatureManager(
            signing_key, digest_algorithm=digest_algorithm, key_id=key_id
        )
        manifest_digest = signature_manager.digest_file(manifest_path)
        signature_manager.write_signature_document(
            {"path": "manifest.json", digest_algorithm: manifest_digest},
            manifest_path.with_suffix(".sig"),
        )

        _tar_directory(staging_root, archive_path)
        log.info("Created observability bundle: %s", archive_path)
        return archive_path
    finally:
        shutil.rmtree(staging_root, ignore_errors=True)


def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--version", required=True, help="Bundle version identifier")
    parser.add_argument(
        "--output-dir",
        required=True,
        type=Path,
        help="Directory where the signed bundle will be written",
    )
    parser.add_argument(
        "--signing-key",
        required=True,
        type=Path,
        help="Path to the HMAC key used to sign the manifest",
    )
    parser.add_argument(
        "--digest",
        default="sha384",
        help="Digest algorithm for HMAC signatures (default: sha384)",
    )
    parser.add_argument("--key-id", help="Optional key identifier embedded in the signature")
    parser.add_argument(
        "--dashboard",
        action="append",
        default=[],
        type=Path,
        help="Additional dashboard JSON file to include",
    )
    parser.add_argument(
        "--dashboards-dir",
        action="append",
        default=[],
        type=Path,
        help="Directory containing Grafana dashboard JSON files",
    )
    parser.add_argument(
        "--alert-rule",
        action="append",
        default=[],
        type=Path,
        help="Prometheus alert rule file to include",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parse_args(argv)
    logging.basicConfig(level=getattr(logging, args.log_level))

    dashboards: List[Path] = list(args.dashboard)
    dashboards.extend(args.dashboards_dir)
    if not dashboards:
        dashboards = list(DEFAULT_DASHBOARD_DIRS)

    alerts: List[Path] = list(args.alert_rule)
    if not alerts:
        alerts = list(DEFAULT_ALERT_RULES)

    signing_key = _load_signing_key(args.signing_key)

    try:
        build_observability_bundle(
            version=args.version,
            dashboards=dashboards,
            alerts=alerts,
            output_dir=args.output_dir,
            signing_key=signing_key,
            digest_algorithm=args.digest,
            key_id=args.key_id,
        )
    except Exception as exc:  # pragma: no cover - error reporting
        logging.getLogger("observability_bundle").error("%s", exc)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
