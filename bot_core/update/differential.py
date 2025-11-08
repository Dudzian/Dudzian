"""Helpers for managing differential OEM updates."""
from __future__ import annotations

import hashlib
import json
import logging
import shutil
import urllib.parse
import urllib.request
from collections import Counter
from dataclasses import dataclass
import re
from pathlib import Path, PurePosixPath
from typing import Callable, Mapping, MutableMapping, TYPE_CHECKING

from bot_core.security.hwid import HwIdProvider, HwIdProviderError
from bot_core.security.signing import validate_hmac_signature
from packaging.version import InvalidVersion, Version

if TYPE_CHECKING:  # pragma: no cover - mypy-only imports
    from bot_core.security.update import LicenseValidationResult, UpdateVerificationResult


LOGGER = logging.getLogger(__name__)


@dataclass(slots=True)
class DeltaManifestValidation:
    """Validation outcome for a delta manifest."""

    manifest_path: Path
    payload: Mapping[str, object]
    signature_valid: bool | None
    fingerprint_ok: bool
    issues: list[str]

    @property
    def base_version(self) -> str | None:
        candidate = self.payload.get("base_version")
        return str(candidate) if isinstance(candidate, str) else None

    @property
    def target_version(self) -> str | None:
        candidate = self.payload.get("target_version")
        return str(candidate) if isinstance(candidate, str) else None


@dataclass(slots=True)
class DownloadedPackage:
    """Metadata describing a downloaded update package."""

    source: str
    path: Path
    sha256: str
    sha384: str
    size: int


class DifferentialUpdateManager:
    """Coordinates verification and rollback of differential update bundles."""

    def __init__(
        self,
        *,
        storage_dir: Path,
        manifest_key: bytes | None = None,
        package_key: bytes | None = None,
        hwid_provider: HwIdProvider | None = None,
        downloader: Callable[[str, Path], Path] | None = None,
    ) -> None:
        self._storage_dir = Path(storage_dir).expanduser()
        self._storage_dir.mkdir(parents=True, exist_ok=True)
        self._manifest_key = manifest_key
        self._package_key = package_key
        self._hwid_provider = hwid_provider
        self._downloader = downloader

    def validate_manifest(
        self,
        manifest_path: Path,
        *,
        signature_path: Path | None = None,
    ) -> DeltaManifestValidation:
        manifest_path = manifest_path.expanduser()
        payload = self._load_manifest(manifest_path)
        issues: list[str] = []
        signature_valid: bool | None = None

        if signature_path is None:
            candidate = manifest_path.with_suffix(manifest_path.suffix + ".sig")
            if candidate.exists():
                signature_path = candidate
        signature_payload = self._load_signature(signature_path) if signature_path else None

        if self._manifest_key is not None:
            if signature_payload is None:
                signature_valid = False
                issues.append("missing-signature")
            else:
                errors = validate_hmac_signature(payload, signature_payload, key=self._manifest_key)
                signature_valid = not errors
                issues.extend(errors)
        elif signature_payload is not None:
            signature_valid = None

        bundle_name = payload.get("bundle")
        if not isinstance(bundle_name, str) or not bundle_name.strip():
            issues.append("missing-bundle")

        base_version_value = payload.get("base_version")
        base_version_parsed: Version | None = None
        if isinstance(base_version_value, str) and base_version_value.strip():
            try:
                base_version_parsed = Version(base_version_value.strip())
            except InvalidVersion:
                issues.append("invalid-base-version")
        else:
            issues.append("missing-base-version")

        target_version_value = payload.get("target_version")
        target_version_parsed: Version | None = None
        if isinstance(target_version_value, str) and target_version_value.strip():
            try:
                target_version_parsed = Version(target_version_value.strip())
            except InvalidVersion:
                issues.append("invalid-target-version")
        else:
            issues.append("missing-target-version")

        platform = payload.get("platform")
        if not isinstance(platform, str) or not platform.strip():
            issues.append("missing-platform")

        if base_version_parsed is not None and target_version_parsed is not None:
            if target_version_parsed <= base_version_parsed:
                issues.append("non-incremental-version")

        changed_files_list: list[str] | None = None
        changed_files = payload.get("changed_files")
        if changed_files is None:
            issues.append("missing-changed-files")
        elif not isinstance(changed_files, list) or any(
            not isinstance(entry, str) or not entry.strip() for entry in changed_files
        ):
            issues.append("invalid-changed-files")
        else:
            changed_files_list = [entry.strip() for entry in changed_files]
            if any(count > 1 for count in Counter(changed_files_list).values()):
                issues.append("duplicate-changed-files")
            elif _has_casefold_duplicates(changed_files_list):
                issues.append("duplicate-changed-files-casefold")
            elif not _all_manifest_paths_safe(changed_files_list):
                issues.append("invalid-changed-file-path")

        removed_files_list: list[str] | None = None
        removed_files = payload.get("removed_files")
        if removed_files is None:
            issues.append("missing-removed-files")
        elif not isinstance(removed_files, list) or any(
            not isinstance(entry, str) or not entry.strip() for entry in removed_files
        ):
            issues.append("invalid-removed-files")
        else:
            removed_files_list = [entry.strip() for entry in removed_files]
            if any(count > 1 for count in Counter(removed_files_list).values()):
                issues.append("duplicate-removed-files")
            elif _has_casefold_duplicates(removed_files_list):
                issues.append("duplicate-removed-files-casefold")
            elif not _all_manifest_paths_safe(removed_files_list):
                issues.append("invalid-removed-file-path")

        if changed_files_list and removed_files_list:
            if set(changed_files_list) & set(removed_files_list):
                issues.append("conflicting-file-lists")
            elif _lists_conflict_casefold(changed_files_list, removed_files_list):
                issues.append("conflicting-file-lists-casefold")

        if (
            changed_files_list is not None
            and removed_files_list is not None
            and not changed_files_list
            and not removed_files_list
        ):
            issues.append("empty-file-lists")

        fingerprint_ok = True
        fingerprint = _extract_fingerprint(payload)
        if fingerprint and self._hwid_provider is not None:
            try:
                local_fingerprint = self._hwid_provider.read()
            except HwIdProviderError as exc:  # pragma: no cover - defensywne logowanie
                fingerprint_ok = False
                issues.append("fingerprint-unavailable")
                LOGGER.warning("Cannot read local HWID: %s", exc)
            else:
                if local_fingerprint != fingerprint:
                    fingerprint_ok = False
                    issues.append("fingerprint-mismatch")

        return DeltaManifestValidation(
            manifest_path=manifest_path,
            payload=payload,
            signature_valid=signature_valid,
            fingerprint_ok=fingerprint_ok,
            issues=issues,
        )

    def download_package(self, source: str, *, filename: str | None = None) -> DownloadedPackage:
        """Fetch an update package from local or remote storage."""

        parsed = urllib.parse.urlparse(source)
        target_name = filename or Path(parsed.path).name or "update-package.bin"
        destination = self._storage_dir / target_name

        if self._downloader is not None:
            resolved = self._downloader(source, destination)
        elif parsed.scheme in {"http", "https"}:
            with urllib.request.urlopen(source) as response, destination.open("wb") as handle:
                shutil.copyfileobj(response, handle)
            resolved = destination
        else:
            if parsed.scheme == "file":
                source_path = Path(urllib.request.url2pathname(parsed.path))
            else:
                source_path = Path(source)
            source_path = source_path.expanduser()
            if not source_path.exists():
                raise FileNotFoundError(source_path)
            shutil.copy2(source_path, destination)
            resolved = destination

        sha256 = _hash_file(resolved, "sha256")
        sha384 = _hash_file(resolved, "sha384")
        size = resolved.stat().st_size
        return DownloadedPackage(source=source, path=resolved, sha256=sha256, sha384=sha384, size=size)

    def verify_package(
        self,
        package_dir: Path,
        *,
        manifest_path: Path | None = None,
        signature_path: Path | None = None,
        license_result: "LicenseValidationResult" | None = None,
    ) -> "UpdateVerificationResult":
        package_dir = Path(package_dir).expanduser()
        if manifest_path is None:
            manifest_path = package_dir / "manifest.json"
        else:
            manifest_path = Path(manifest_path).expanduser()
        if signature_path is None:
            candidate = manifest_path.with_suffix(manifest_path.suffix + ".sig")
            if candidate.exists():
                signature_path = candidate
        else:
            signature_path = Path(signature_path).expanduser()

        from bot_core.security.update import verify_update_bundle  # import here to avoid circular dependency

        return verify_update_bundle(
            manifest_path=manifest_path,
            base_dir=package_dir,
            signature_path=signature_path,
            hmac_key=self._package_key,
            license_result=license_result,
        )

    def rollback(self, backup_dir: Path, target_dir: Path) -> None:
        """Restore resources from ``backup_dir`` back into ``target_dir``."""

        backup_dir = Path(backup_dir).expanduser()
        target_dir = Path(target_dir).expanduser()
        if not backup_dir.exists():
            raise FileNotFoundError(backup_dir)
        target_dir.mkdir(parents=True, exist_ok=True)

        for entry in list(target_dir.iterdir()):
            if entry.is_dir():
                shutil.rmtree(entry)
            else:
                entry.unlink()

        for entry in backup_dir.rglob("*"):
            if entry.is_dir():
                continue
            relative = entry.relative_to(backup_dir)
            destination = target_dir / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(entry, destination)

    @staticmethod
    def _load_manifest(path: Path) -> Mapping[str, object]:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:  # pragma: no cover - defensywne logowanie
            raise ValueError(f"Manifest {path} zawiera niepoprawny JSON: {exc}") from exc
        if not isinstance(payload, Mapping):
            raise ValueError("Manifest delta powinien być obiektem JSON")
        return payload

    @staticmethod
    def _load_signature(path: Path | None) -> MutableMapping[str, object] | None:
        if path is None:
            return None
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:  # pragma: no cover - defensywne logowanie
            raise ValueError(f"Podpis {path} zawiera niepoprawny JSON: {exc}") from exc
        if not isinstance(payload, MutableMapping):
            raise ValueError("Plik podpisu powinien być obiektem JSON")
        return payload


def _extract_fingerprint(payload: Mapping[str, object]) -> str | None:
    fingerprint = payload.get("fingerprint")
    if isinstance(fingerprint, str) and fingerprint.strip():
        return fingerprint.strip()
    metadata = payload.get("metadata")
    if isinstance(metadata, Mapping):
        candidate = metadata.get("fingerprint")
        if isinstance(candidate, str) and candidate.strip():
            return candidate.strip()
    return None


def _hash_file(path: Path, algorithm: str) -> str:
    digest = hashlib.new(algorithm)
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


_WINDOWS_DRIVE_RE = re.compile(r"^[a-zA-Z]:")


def _has_casefold_duplicates(entries: list[str]) -> bool:
    seen: set[str] = set()
    for entry in entries:
        lowered = entry.casefold()
        if lowered in seen:
            return True
        seen.add(lowered)
    return False


def _lists_conflict_casefold(left: list[str], right: list[str]) -> bool:
    left_folded = {entry.casefold() for entry in left}
    right_folded = {entry.casefold() for entry in right}
    return bool(left_folded & right_folded)


def _all_manifest_paths_safe(entries: list[str]) -> bool:
    """Ensure every manifest path is relative and free from traversal attempts."""

    for entry in entries:
        if "\\" in entry:
            return False
        if _WINDOWS_DRIVE_RE.match(entry):
            return False
        candidate = PurePosixPath(entry)
        if candidate.is_absolute():
            return False
        if any(part in {"", ".", ".."} for part in candidate.parts):
            return False
        if candidate.as_posix() != entry:
            return False
    return True


__all__ = [
    "DeltaManifestValidation",
    "DifferentialUpdateManager",
    "DownloadedPackage",
]
