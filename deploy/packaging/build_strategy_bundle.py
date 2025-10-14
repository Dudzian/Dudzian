"""Build signed distribution bundles for Stage4 strategy packages."""

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
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from deploy.packaging.build_core_bundle import (  # type: ignore
    SignatureManager,
    _ensure_casefold_safe_tree,
    _ensure_no_symlinks,
    _ensure_windows_safe_component,
    _ensure_windows_safe_tree,
    _validate_bundle_version,
)


BUNDLE_NAME = "stage4-strategies"
DEFAULT_DIGEST = "sha384"
_ALLOWED_ENTRY_CHARS = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-.")


@dataclass(frozen=True)
class StrategyAsset:
    """Definition of a strategy asset to include in the bundle."""

    name: str
    source: Path
    module: Optional[str] = None


@dataclass(frozen=True)
class DatasetAsset:
    """Definition of a dataset asset to include in the bundle."""

    name: str
    source: Path
    format: Optional[str] = None


def _now_utc() -> str:
    return _dt.datetime.utcnow().replace(microsecond=0).isoformat() + "Z"


def _git_revision() -> Optional[str]:
    try:
        result = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None
    return result.decode("ascii").strip()


def _validate_entry_name(value: str, *, label: str) -> str:
    if not value:
        raise ValueError(f"{label} cannot be empty")
    for ch in value:
        if ch not in _ALLOWED_ENTRY_CHARS:
            raise ValueError(
                f"{label} contains unsupported character {ch!r}; "
                "allowed: letters, digits, '_', '-' and '.'"
            )
    if value[0] == "." or value[-1] == ".":
        raise ValueError(f"{label} cannot start or end with '.'")
    return value


def _ensure_parent_directory(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def _infer_module(path: Path) -> Optional[str]:
    try:
        relative = path.relative_to(REPO_ROOT)
    except ValueError:
        return None
    if relative.suffix != ".py":
        return None
    parts = list(relative.with_suffix("").parts)
    return ".".join(parts)


def _relative_to_repo(path: Path) -> str:
    try:
        return path.relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def _ensure_file(path: Path, *, label: str) -> Path:
    if not path.exists():
        raise FileNotFoundError(f"{label} does not exist: {path}")
    _ensure_no_symlinks(path, label=label)
    resolved = path.resolve()
    _ensure_windows_safe_tree(resolved, label=label)
    if resolved.is_dir():
        _ensure_casefold_safe_tree(resolved, label=label)
        raise ValueError(f"{label} must reference a file: {resolved}")
    return resolved


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
                "Signing key file permissions must restrict access to the owner: "
                f"{resolved}"
            )
    return resolved.read_bytes()


def _copy_and_digest(source: Path, destination: Path) -> Mapping[str, str | int]:
    _ensure_parent_directory(destination)
    shutil.copy2(source, destination)
    digest = hashlib.sha256()
    size = 0
    with destination.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            size += len(chunk)
            digest.update(chunk)
    return {"sha256": digest.hexdigest(), "size_bytes": size}


def _default_stage4_assets() -> tuple[List[StrategyAsset], List[DatasetAsset]]:
    strategies = [
        StrategyAsset(
            name="mean_reversion",
            source=REPO_ROOT / "bot_core/strategies/mean_reversion.py",
            module="bot_core.strategies.mean_reversion",
        ),
        StrategyAsset(
            name="volatility_target",
            source=REPO_ROOT / "bot_core/strategies/volatility_target.py",
            module="bot_core.strategies.volatility_target",
        ),
        StrategyAsset(
            name="cross_exchange_arbitrage",
            source=REPO_ROOT
            / "bot_core/strategies/cross_exchange_arbitrage.py",
            module="bot_core.strategies.cross_exchange_arbitrage",
        ),
    ]
    datasets = [
        DatasetAsset(
            name="manifest",
            source=REPO_ROOT / "data/backtests/normalized/manifest.yaml",
            format="yaml",
        ),
        DatasetAsset(
            name="mean_reversion_dataset",
            source=REPO_ROOT / "data/backtests/normalized/mean_reversion.csv",
            format="csv",
        ),
        DatasetAsset(
            name="volatility_target_dataset",
            source=REPO_ROOT / "data/backtests/normalized/volatility_target.csv",
            format="csv",
        ),
        DatasetAsset(
            name="cross_exchange_arbitrage_dataset",
            source=REPO_ROOT
            / "data/backtests/normalized/cross_exchange_arbitrage.csv",
            format="csv",
        ),
    ]
    return strategies, datasets


def _extend_with_custom_assets(
    existing: List[StrategyAsset],
    additions: Sequence[str],
    *,
    label: str,
) -> List[StrategyAsset]:
    result = list(existing)
    seen = {asset.name.casefold() for asset in existing}
    for entry in additions:
        if "=" not in entry:
            raise ValueError(f"Invalid {label} specification: {entry}")
        raw_name, raw_path = entry.split("=", 1)
        name = _validate_entry_name(raw_name.strip(), label=f"{label} name")
        key = name.casefold()
        if key in seen:
            raise ValueError(f"Duplicate {label} name: {name}")
        seen.add(key)
        source_path = Path(raw_path).expanduser()
        if not source_path.is_absolute():
            source_path = (REPO_ROOT / source_path).resolve()
        resolved = _ensure_file(source_path, label=f"{label} '{name}' source")
        module = _infer_module(resolved)
        result.append(StrategyAsset(name=name, source=resolved, module=module))
    return result


def _extend_datasets(
    existing: List[DatasetAsset],
    additions: Sequence[str],
    *,
    label: str,
) -> List[DatasetAsset]:
    result = list(existing)
    seen = {asset.name.casefold() for asset in existing}
    for entry in additions:
        if "=" not in entry:
            raise ValueError(f"Invalid {label} specification: {entry}")
        raw_name, raw_path = entry.split("=", 1)
        name = _validate_entry_name(raw_name.strip(), label=f"{label} name")
        key = name.casefold()
        if key in seen:
            raise ValueError(f"Duplicate {label} name: {name}")
        seen.add(key)
        source_path = Path(raw_path).expanduser()
        if not source_path.is_absolute():
            source_path = (REPO_ROOT / source_path).resolve()
        resolved = _ensure_file(source_path, label=f"{label} '{name}' source")
        result.append(
            DatasetAsset(
                name=name,
                source=resolved,
                format=resolved.suffix.lstrip("."),
            )
        )
    return result


class StrategyBundleBuilder:
    """Builder responsible for packaging strategy assets and datasets."""

    def __init__(
        self,
        *,
        version: str,
        signing_key: bytes,
        signing_key_id: Optional[str],
        digest_algorithm: str,
        output_dir: Path,
        strategies: Sequence[StrategyAsset],
        datasets: Sequence[DatasetAsset],
    ) -> None:
        self._version = _validate_bundle_version(version)
        self._signing_key = signing_key
        self._signing_key_id = signing_key_id
        self._digest_algorithm = digest_algorithm
        self._output_dir = output_dir
        self._strategies = list(strategies)
        self._datasets = list(datasets)

    def _prepare_output_dir(self) -> Path:
        if self._output_dir.exists():
            if self._output_dir.is_symlink():
                raise ValueError(
                    f"Output directory must not be a symlink: {self._output_dir}"
                )
            if not self._output_dir.is_dir():
                raise ValueError(
                    f"Output directory must be a directory: {self._output_dir}"
                )
        else:
            self._output_dir.mkdir(parents=True, exist_ok=True)
        _ensure_windows_safe_tree(self._output_dir, label="Output directory")
        return self._output_dir.resolve()

    def build(self) -> Path:
        output_dir = self._prepare_output_dir()
        archive_basename = f"{BUNDLE_NAME}-{self._version}"
        archive_path = output_dir / f"{archive_basename}.zip"
        if archive_path.exists():
            raise ValueError(f"Archive already exists: {archive_path}")

        with tempfile.TemporaryDirectory() as tmpdir:
            temp_root = Path(tmpdir)
            bundle_root = temp_root / BUNDLE_NAME
            strategies_dir = bundle_root / "strategies"
            datasets_dir = bundle_root / "datasets"
            strategies_dir.mkdir(parents=True, exist_ok=True)
            datasets_dir.mkdir(parents=True, exist_ok=True)

            manifest: Dict[str, object] = {
                "bundle": BUNDLE_NAME,
                "version": self._version,
                "created_at": _now_utc(),
                "git_revision": _git_revision(),
                "digest_algorithm": self._digest_algorithm,
                "strategies": [],
                "datasets": [],
            }

            for asset in sorted(self._strategies, key=lambda item: item.name):
                target_name = f"{asset.name}{asset.source.suffix}"
                _ensure_windows_safe_component(
                    component=target_name,
                    label="Strategy artifact name",
                    context=target_name,
                )
                destination = strategies_dir / target_name
                digest_info = _copy_and_digest(asset.source, destination)
                strategy_record = {
                    "name": asset.name,
                    "module": asset.module or _infer_module(asset.source),
                    "source_path": _relative_to_repo(asset.source),
                    "bundle_path": f"strategies/{target_name}",
                    **digest_info,
                }
                manifest["strategies"].append(strategy_record)

            for asset in sorted(self._datasets, key=lambda item: item.name):
                target_name = asset.source.name
                _ensure_windows_safe_component(
                    component=target_name,
                    label="Dataset artifact name",
                    context=target_name,
                )
                destination = datasets_dir / target_name
                digest_info = _copy_and_digest(asset.source, destination)
                dataset_record = {
                    "name": asset.name,
                    "format": asset.format or asset.source.suffix.lstrip("."),
                    "source_path": _relative_to_repo(asset.source),
                    "bundle_path": f"datasets/{target_name}",
                    **digest_info,
                }
                manifest["datasets"].append(dataset_record)

            manifest_path = bundle_root / "manifest.json"
            manifest_path.write_text(
                json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True)
                + "\n",
                encoding="utf-8",
            )

            signer = SignatureManager(
                self._signing_key,
                digest_algorithm=self._digest_algorithm,
                key_id=self._signing_key_id,
            )
            signature_path = bundle_root / "manifest.json.sig"
            signer.write_signature_document(manifest, signature_path)

            shutil.make_archive(
                str(output_dir / archive_basename),
                "zip",
                root_dir=temp_root,
                base_dir=bundle_root.name,
            )

            # Copy manifest and signature next to the archive for easier auditing.
            shutil.copy2(manifest_path, output_dir / f"{archive_basename}.manifest.json")
            shutil.copy2(signature_path, output_dir / f"{archive_basename}.manifest.sig")

        logging.info("Strategy bundle created: %s", archive_path)
        return archive_path


def build_from_cli(argv: Optional[Sequence[str]] = None) -> Path:
    parser = argparse.ArgumentParser(
        description="Build signed strategy bundle for Stage4 distribution"
    )
    parser.add_argument("--version", required=True)
    parser.add_argument("--signing-key-path", required=True)
    parser.add_argument("--signing-key-id")
    parser.add_argument(
        "--digest",
        default=DEFAULT_DIGEST,
        help="Digest algorithm for HMAC signatures (default: sha384)",
    )
    parser.add_argument(
        "--output-dir",
        default="var/dist/strategies",
        help="Destination directory for generated bundle",
    )
    parser.add_argument(
        "--preset",
        default="stage4",
        choices=["stage4"],
        help="Preset of strategy and dataset assets to package",
    )
    parser.add_argument(
        "--strategy",
        action="append",
        default=[],
        help="Additional strategy entry in the form name=path",
    )
    parser.add_argument(
        "--dataset",
        action="append",
        default=[],
        help="Additional dataset entry in the form name=path",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )

    args = parser.parse_args(argv)
    logging.basicConfig(level=getattr(logging, args.log_level))

    signing_key_path = Path(args.signing_key_path).expanduser()
    signing_key = _load_signing_key(signing_key_path)

    strategies, datasets = _default_stage4_assets()
    if args.strategy:
        strategies = _extend_with_custom_assets(
            strategies, args.strategy, label="Strategy"
        )
    if args.dataset:
        datasets = _extend_datasets(datasets, args.dataset, label="Dataset")

    output_dir = Path(args.output_dir).expanduser()

    builder = StrategyBundleBuilder(
        version=args.version,
        signing_key=signing_key,
        signing_key_id=args.signing_key_id,
        digest_algorithm=args.digest,
        output_dir=output_dir,
        strategies=strategies,
        datasets=datasets,
    )
    return builder.build()


def main(argv: Optional[Sequence[str]] = None) -> int:
    try:
        build_from_cli(argv)
    except Exception as exc:  # pragma: no cover - CLI guardrail
        logging.error("Failed to build strategy bundle: %s", exc)
        return 1
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entrypoint
    sys.exit(main())
