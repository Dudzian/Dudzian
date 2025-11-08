"""High-level packaging pipeline helpers (notarization, delta updates, fingerprint checks)."""
from __future__ import annotations

import base64
import hashlib
import hmac
import json
import logging
import os
import shutil
import subprocess
import tarfile
import tempfile
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Mapping, MutableMapping, Sequence

from bot_core.security.hwid import HwIdProvider, HwIdProviderError
from bot_core.security.signing import build_hmac_signature, canonical_json_bytes, validate_hmac_signature

_LOGGER = logging.getLogger(__name__)


@dataclass(slots=True)
class PackagingContext:
    """Common metadata describing the bundle being processed."""

    staging_root: Path
    archive_path: Path
    manifest: Mapping[str, object]

    @property
    def bundle_name(self) -> str:
        return str(self.manifest.get("bundle") or "")

    @property
    def version(self) -> str:
        return str(self.manifest.get("version") or "")

    @property
    def platform(self) -> str:
        return str(self.manifest.get("platform") or "")


@dataclass(slots=True)
class HardwareFingerprintReport:
    """Result of fingerprint validation."""

    document_path: Path
    fingerprint: str | None
    expected_fingerprint: str | None
    signature_valid: bool | None
    issues: list[str] = field(default_factory=list)

    def to_mapping(self) -> Mapping[str, object]:
        payload: MutableMapping[str, object] = {
            "document_path": str(self.document_path),
            "fingerprint": self.fingerprint,
            "expected": self.expected_fingerprint,
            "signature_valid": self.signature_valid,
            "issues": list(self.issues),
        }
        return payload


class HardwareFingerprintValidator:
    """Validates the fingerprint document embedded in the bundle."""

    def __init__(
        self,
        *,
        expected_fingerprint: str | None = None,
        signature_key: bytes | None = None,
        fail_on_missing: bool = True,
        allow_placeholder: bool = False,
        verify_local: bool = False,
    ) -> None:
        self._expected = expected_fingerprint.strip() if expected_fingerprint else None
        self._signature_key = signature_key
        self._fail_on_missing = fail_on_missing
        self._allow_placeholder = allow_placeholder
        self._verify_local = verify_local

    def validate(self, context: PackagingContext) -> HardwareFingerprintReport:
        document_path = context.staging_root / "config" / "fingerprint.expected.json"
        if not document_path.exists():
            message = "fingerprint.expected.json is missing from bundle"
            if self._fail_on_missing:
                raise FileNotFoundError(message)
            return HardwareFingerprintReport(
                document_path=document_path,
                fingerprint=None,
                expected_fingerprint=self._expected,
                signature_valid=None,
                issues=[message],
            )

        try:
            document = json.loads(document_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:  # pragma: no cover - defensive
            raise ValueError(f"fingerprint.expected.json contains invalid JSON: {exc}") from exc

        payload = document.get("payload")
        signature = document.get("signature")
        if not isinstance(payload, Mapping) or not isinstance(signature, Mapping):
            raise ValueError("fingerprint.expected.json must contain 'payload' and 'signature' objects")

        fingerprint_raw = payload.get("fingerprint")
        fingerprint = str(fingerprint_raw).strip() if isinstance(fingerprint_raw, str) else None
        issues: list[str] = []

        if fingerprint is None:
            issues.append("missing-fingerprint")
        elif not self._allow_placeholder and fingerprint.upper() in {"UNPROVISIONED", "PLACEHOLDER"}:
            issues.append("placeholder-fingerprint")

        signature_valid: bool | None = None
        if self._signature_key is not None:
            signature_mapping = signature if isinstance(signature, Mapping) else None
            if signature_mapping is None:
                issues.append("signature-missing")
            else:
                algorithm = signature_mapping.get("algorithm")
                if not isinstance(algorithm, str):
                    issues.append("signature-missing-algorithm")
                else:
                    errors = _verify_hmac_signature(payload, signature_mapping, key=self._signature_key, algorithm=algorithm)
                    signature_valid = not errors
                    issues.extend(errors)

        if self._expected and fingerprint and fingerprint != self._expected:
            issues.append("fingerprint-mismatch")

        if self._verify_local and fingerprint:
            local = _read_local_fingerprint()
            if local is None:
                issues.append("local-fingerprint-unavailable")
            elif local != fingerprint:
                issues.append("local-fingerprint-mismatch")

        return HardwareFingerprintReport(
            document_path=document_path,
            fingerprint=fingerprint,
            expected_fingerprint=self._expected,
            signature_valid=signature_valid,
            issues=issues,
        )


def _read_local_fingerprint() -> str | None:
    try:
        from bot_core.security.fingerprint import get_local_fingerprint  # type: ignore
    except Exception:  # pragma: no cover - optional dependency
        return None

    try:
        candidate = get_local_fingerprint()
    except Exception:  # pragma: no cover - hardware dependent
        return None
    return str(candidate).strip() if candidate else None


@dataclass(slots=True)
class DeltaUpdateResult:
    """Summary of a generated delta archive."""

    base_version: str
    archive_path: Path
    changed_files: list[str]
    removed_files: list[str]
    metadata: Mapping[str, object]
    manifest_path: Path | None = None
    signature_path: Path | None = None

    def to_mapping(self) -> Mapping[str, object]:
        payload: dict[str, object] = {
            "base_version": self.base_version,
            "archive_path": str(self.archive_path),
            "changed_files": list(self.changed_files),
            "removed_files": list(self.removed_files),
            "metadata": dict(self.metadata),
        }
        if self.manifest_path is not None:
            payload["manifest_path"] = str(self.manifest_path)
        if self.signature_path is not None:
            payload["signature_path"] = str(self.signature_path)
        return payload


@dataclass(slots=True)
class DeltaManifestResult:
    """Result of exporting a delta manifest."""

    base_version: str
    manifest_path: Path
    signature_path: Path | None
    metadata: Mapping[str, object]

    def to_mapping(self) -> Mapping[str, object]:
        payload: dict[str, object] = {
            "base_version": self.base_version,
            "manifest_path": str(self.manifest_path),
            "metadata": dict(self.metadata),
        }
        if self.signature_path is not None:
            payload["signature_path"] = str(self.signature_path)
        return payload


@dataclass(slots=True)
class UpdatePackageResult:
    """Metadata describing an update package produced by the pipeline."""

    package_id: str
    version: str
    package_path: Path
    manifest_path: Path
    signature_path: Path | None
    artifacts: list[str]

    def to_mapping(self) -> Mapping[str, object]:
        payload: dict[str, object] = {
            "package_id": self.package_id,
            "version": self.version,
            "package_path": str(self.package_path),
            "manifest_path": str(self.manifest_path),
            "artifacts": list(self.artifacts),
        }
        if self.signature_path is not None:
            payload["signature_path"] = str(self.signature_path)
        return payload


@dataclass(slots=True)
class CodeSigningResult:
    """Outcome of a code signing attempt for a specific target."""

    target: Path
    command: list[str]
    return_code: int | None
    stdout: str | None
    stderr: str | None
    dry_run: bool

    def to_mapping(self) -> Mapping[str, object]:
        return {
            "target": str(self.target),
            "command": list(self.command),
            "return_code": self.return_code,
            "stdout": self.stdout,
            "stderr": self.stderr,
            "dry_run": self.dry_run,
        }


class DeltaUpdateBuilder:
    """Creates differential update archives between bundle versions."""

    def __init__(
        self,
        *,
        base_manifests: Sequence[Path],
        output_dir: Path,
        compression: str = "zip",
    ) -> None:
        self._base_manifests = [path.expanduser() for path in base_manifests]
        self._output_dir = output_dir.expanduser()
        self._compression = compression

    def build(self, context: PackagingContext) -> list[DeltaUpdateResult]:
        if not self._base_manifests:
            return []
        self._output_dir.mkdir(parents=True, exist_ok=True)
        results: list[DeltaUpdateResult] = []
        for manifest_path in self._base_manifests:
            base_manifest = _load_manifest(manifest_path)
            base_version = str(base_manifest.get("version") or "unknown")
            delta_archive = self._create_delta_archive(base_manifest, context, base_version)
            results.append(delta_archive)
        return results

    def _create_delta_archive(
        self,
        base_manifest: Mapping[str, object],
        context: PackagingContext,
        base_version: str,
    ) -> DeltaUpdateResult:
        base_files = _manifest_entries(base_manifest)
        current_files = _manifest_entries(context.manifest)
        changed: list[str] = []
        removed: list[str] = []

        for path, digest in current_files.items():
            base_digest = base_files.get(path)
            if base_digest != digest:
                changed.append(path)

        for path in base_files:
            if path not in current_files:
                removed.append(path)

        delta_name = _delta_archive_name(context, base_version, self._compression)
        delta_path = self._output_dir / delta_name
        _LOGGER.info("Building delta update: base=%s target=%s -> %s", base_version, context.version, delta_path)
        if delta_path.exists():
            delta_path.unlink()

        with tempfile.TemporaryDirectory(prefix="core_delta_") as temp_dir:
            temp_dir_path = Path(temp_dir)
            metadata = {
                "bundle": context.bundle_name,
                "platform": context.platform,
                "base_version": base_version,
                "target_version": context.version,
                "changed_files": [],
                "removed_files": removed,
            }
            for relative_path in sorted(changed):
                source = context.staging_root / relative_path
                if not source.exists():
                    continue
                archive_path = temp_dir_path / relative_path
                archive_path.parent.mkdir(parents=True, exist_ok=True)
                archive_path.write_bytes(source.read_bytes())
                metadata["changed_files"].append(relative_path)

            metadata_path = temp_dir_path / "delta.json"
            metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")

            if self._compression == "zip":
                with zipfile.ZipFile(delta_path, mode="w", compression=zipfile.ZIP_DEFLATED) as archive:
                    for file_path in sorted(temp_dir_path.rglob("*")):
                        if file_path.is_file():
                            archive.write(file_path, file_path.relative_to(temp_dir_path).as_posix())
            elif self._compression == "tar.gz":
                with tarfile.open(delta_path, mode="w:gz") as archive:
                    archive.add(temp_dir_path, arcname=".")
            else:
                raise ValueError(f"Unsupported delta compression format: {self._compression}")

        return DeltaUpdateResult(
            base_version=base_version,
            archive_path=delta_path,
            changed_files=metadata["changed_files"],
            removed_files=removed,
            metadata=dict(metadata),
        )


class DeltaManifestPublisher:
    """Exports generated delta metadata to standalone JSON manifests."""

    def __init__(
        self,
        *,
        output_dir: Path,
        signing_key: bytes | None = None,
        signing_key_id: str | None = None,
        embed_hwid: bool = False,
    ) -> None:
        self._output_dir = output_dir.expanduser()
        self._signing_key = signing_key
        self._signing_key_id = signing_key_id
        self._embed_hwid = embed_hwid
        self._hwid_provider: HwIdProvider | None = HwIdProvider() if embed_hwid else None

    def publish(
        self,
        context: PackagingContext,
        deltas: Sequence[DeltaUpdateResult],
    ) -> list[DeltaManifestResult]:
        if not deltas:
            return []
        self._output_dir.mkdir(parents=True, exist_ok=True)
        results: list[DeltaManifestResult] = []
        fingerprint: str | None = None
        if self._hwid_provider is not None:
            try:
                fingerprint = self._hwid_provider.read()
            except HwIdProviderError as exc:  # pragma: no cover - defensywne logowanie
                _LOGGER.warning("Unable to read hardware fingerprint for delta manifest: %s", exc)
        for delta in deltas:
            metadata = dict(delta.metadata)
            if fingerprint and "fingerprint" not in metadata:
                metadata["fingerprint"] = fingerprint
            manifest_name = _delta_manifest_name(context, delta.base_version)
            manifest_path = self._output_dir / manifest_name
            manifest_path.write_text(
                json.dumps(metadata, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )

            signature_path: Path | None = None
            if self._signing_key is not None:
                signature = build_hmac_signature(metadata, key=self._signing_key, key_id=self._signing_key_id)
                signature_path = manifest_path.with_suffix(manifest_path.suffix + ".sig")
                signature_payload = {"signature": signature}
                signature_path.write_text(
                    json.dumps(signature_payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
                    encoding="utf-8",
                )

            delta.manifest_path = manifest_path
            delta.signature_path = signature_path

            results.append(
                DeltaManifestResult(
                    base_version=delta.base_version,
                    manifest_path=manifest_path,
                    signature_path=signature_path,
                    metadata=metadata,
                )
            )
        return results


class UpdatePackageBuilder:
    """Creates offline update packages with optional differential patches."""

    def __init__(
        self,
        *,
        output_dir: Path,
        package_id: str,
        runtime: str,
        signing_key: bytes | None = None,
        signing_key_id: str | None = None,
        embed_hwid: bool = False,
        extra_metadata: Mapping[str, object] | None = None,
    ) -> None:
        self._output_dir = output_dir.expanduser()
        self._package_id = package_id
        self._runtime = runtime
        self._signing_key = signing_key
        self._signing_key_id = signing_key_id
        self._embed_hwid = embed_hwid
        self._extra_metadata = dict(extra_metadata or {})
        self._hwid_provider: HwIdProvider | None = HwIdProvider() if embed_hwid else None

    def build(
        self,
        context: PackagingContext,
        deltas: Sequence[DeltaUpdateResult],
        delta_manifests: Sequence[DeltaManifestResult],
    ) -> list[UpdatePackageResult]:
        self._output_dir.mkdir(parents=True, exist_ok=True)
        package_root = self._output_dir / f"{self._package_id}-{context.version}"
        if package_root.exists():
            shutil.rmtree(package_root)
        package_root.mkdir(parents=True, exist_ok=True)

        payload_name = f"{self._package_id}-{context.version}.tar.gz"
        payload_path = package_root / payload_name
        with tarfile.open(payload_path, mode="w:gz") as archive:
            for entry in sorted(context.staging_root.rglob("*")):
                if entry.is_file():
                    archive.add(entry, arcname=entry.relative_to(context.staging_root).as_posix())

        payload_sha384 = _hash_file(payload_path, "sha384")
        payload_sha256 = _hash_file(payload_path, "sha256")

        metadata: MutableMapping[str, object] = {
            "bundle": context.bundle_name,
            "platform": context.platform,
        }
        metadata.update(self._extra_metadata)

        fingerprint: str | None = None
        if self._hwid_provider is not None:
            try:
                fingerprint = self._hwid_provider.read()
            except HwIdProviderError as exc:  # pragma: no cover - defensywne logowanie
                _LOGGER.warning("Unable to read hardware fingerprint for update package: %s", exc)
        if fingerprint:
            metadata.setdefault("fingerprint", fingerprint)

        artifacts: list[MutableMapping[str, object]] = [
            {
                "path": payload_name,
                "sha384": payload_sha384,
                "sha256": payload_sha256,
                "size": payload_path.stat().st_size,
                "type": "full",
            }
        ]

        delta_dir = package_root / "delta"
        delta_dir.mkdir(exist_ok=True)
        manifest_map = {manifest.base_version: manifest for manifest in delta_manifests}
        for delta in deltas:
            target_path = delta_dir / delta.archive_path.name
            shutil.copy2(delta.archive_path, target_path)
            sha384 = _hash_file(target_path, "sha384")
            sha256 = _hash_file(target_path, "sha256")
            artifacts.append(
                {
                    "path": str(Path("delta") / target_path.name),
                    "sha384": sha384,
                    "sha256": sha256,
                    "size": target_path.stat().st_size,
                    "type": "diff",
                    "base_id": delta.base_version,
                }
            )
            manifest_entry = manifest_map.get(delta.base_version)
            if manifest_entry is not None:
                manifest_target = delta_dir / manifest_entry.manifest_path.name
                shutil.copy2(manifest_entry.manifest_path, manifest_target)
                metadata.setdefault("delta_manifests", []).append(
                    {
                        "base_version": delta.base_version,
                        "path": str(Path("delta") / manifest_target.name),
                    }
                )
                if manifest_entry.signature_path is not None:
                    signature_target = delta_dir / manifest_entry.signature_path.name
                    shutil.copy2(manifest_entry.signature_path, signature_target)
                    metadata.setdefault("delta_signatures", []).append(
                        {
                            "base_version": delta.base_version,
                            "path": str(Path("delta") / signature_target.name),
                        }
                    )

        manifest_payload: MutableMapping[str, object] = {
            "id": self._package_id,
            "version": context.version,
            "platform": context.platform,
            "runtime": self._runtime,
            "artifacts": artifacts,
            "metadata": metadata,
        }

        manifest_path = package_root / "manifest.json"
        signature_path: Path | None = None
        manifest_payload_with_signature: MutableMapping[str, object]
        if self._signing_key is not None:
            signature = build_hmac_signature(manifest_payload, key=self._signing_key, key_id=self._signing_key_id)
            manifest_payload_with_signature = dict(manifest_payload)
            manifest_payload_with_signature["signature"] = signature
            signature_path = package_root / "manifest.sig"
            signature_path.write_text(
                json.dumps({"signature": signature}, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
        else:
            manifest_payload_with_signature = manifest_payload

        manifest_path.write_text(
            json.dumps(manifest_payload_with_signature, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

        return [
            UpdatePackageResult(
                package_id=self._package_id,
                version=context.version,
                package_path=payload_path,
                manifest_path=manifest_path,
                signature_path=signature_path,
                artifacts=[artifact["path"] for artifact in artifacts],
            )
        ]


class CodeSigningService:
    """Runs external signing tools (codesign/signtool) for produced artifacts."""

    def __init__(
        self,
        *,
        command: Sequence[str],
        targets: Sequence[str],
        environment: Mapping[str, str] | None = None,
        dry_run: bool = False,
        timeout_seconds: int | None = None,
    ) -> None:
        if not command:
            raise ValueError("Code signing requires a command template")
        self._command_template = list(command)
        self._targets = list(targets)
        self._env = {**os.environ, **dict(environment)} if environment else None
        self._dry_run = dry_run
        self._timeout = timeout_seconds

    def sign(self, context: PackagingContext, report: "PackagingPipelineReport") -> list[CodeSigningResult]:
        targets = self._resolve_targets(context, report)
        results: list[CodeSigningResult] = []
        for target in targets:
            command = [part.format(target=str(target)) for part in self._command_template]
            if self._dry_run:
                results.append(
                    CodeSigningResult(
                        target=target,
                        command=command,
                        return_code=0,
                        stdout=None,
                        stderr=None,
                        dry_run=True,
                    )
                )
                continue

            _LOGGER.info("Signing %s using %s", target, " ".join(command))
            completed = subprocess.run(
                command,
                capture_output=True,
                text=True,
                check=False,
                env=self._env,
                timeout=self._timeout,
            )
            results.append(
                CodeSigningResult(
                    target=target,
                    command=command,
                    return_code=completed.returncode,
                    stdout=completed.stdout,
                    stderr=completed.stderr,
                    dry_run=False,
                )
            )
            if completed.returncode != 0:
                raise RuntimeError(
                    "Code signing command failed for {}: {}".format(target, completed.stderr.strip())
                )
        return results

    def _resolve_targets(self, context: PackagingContext, report: "PackagingPipelineReport") -> list[Path]:
        resolved: list[Path] = []
        for selector in self._targets:
            normalized = selector.lower()
            if normalized == "archive":
                resolved.append(context.archive_path)
            elif normalized == "delta_archives":
                resolved.extend(delta.archive_path for delta in report.delta_updates)
            elif normalized == "delta_manifests":
                resolved.extend(entry.manifest_path for entry in report.delta_manifests)
            elif normalized == "update_packages":
                resolved.extend(package.package_path for package in report.update_packages)
            elif normalized == "update_manifests":
                resolved.extend(package.manifest_path for package in report.update_packages)
            else:
                resolved.append(Path(selector.format(
                    bundle=context.bundle_name,
                    version=context.version,
                    platform=context.platform,
                )))
        unique: list[Path] = []
        seen: set[Path] = set()
        for path in resolved:
            candidate = path.expanduser()
            if candidate not in seen:
                seen.add(candidate)
                unique.append(candidate)
        return unique


def _load_manifest(path: Path) -> Mapping[str, object]:
    path = path.expanduser()
    if not path.exists():
        raise FileNotFoundError(path)
    if path.is_dir():
        manifest_path = path / "manifest.json"
        if not manifest_path.exists():
            raise FileNotFoundError(manifest_path)
        return json.loads(manifest_path.read_text(encoding="utf-8"))
    if path.suffix == ".zip":
        with zipfile.ZipFile(path) as archive:
            with archive.open("manifest.json") as handle:
                return json.loads(handle.read().decode("utf-8"))
    if path.suffixes[-2:] == [".tar", ".gz"]:
        with tarfile.open(path, mode="r:gz") as archive:
            member = archive.getmember("manifest.json")
            with archive.extractfile(member) as handle:  # type: ignore[arg-type]
                if handle is None:
                    raise FileNotFoundError("manifest.json not found in archive")
                return json.loads(handle.read().decode("utf-8"))
    raise ValueError(f"Unsupported manifest container: {path}")


def _manifest_entries(manifest: Mapping[str, object]) -> dict[str, str]:
    files = manifest.get("files")
    result: dict[str, str] = {}
    if isinstance(files, Sequence):
        for entry in files:
            if isinstance(entry, Mapping):
                path = entry.get("path")
                digest = entry.get("sha384") or entry.get("sha256")
                if isinstance(path, str) and isinstance(digest, str):
                    result[path] = digest
    return result


def _delta_archive_name(context: PackagingContext, base_version: str, compression: str) -> str:
    suffix = ".zip" if compression == "zip" else ".tar.gz"
    bundle = context.bundle_name or "bundle"
    platform = context.platform or "any"
    return f"{bundle}-{platform}-{base_version}-to-{context.version}.delta{suffix}"


def _delta_manifest_name(context: PackagingContext, base_version: str) -> str:
    bundle = context.bundle_name or "bundle"
    platform = context.platform or "any"
    return f"{bundle}-{platform}-{base_version}-to-{context.version}.delta.json"


def _hash_file(path: Path, algorithm: str) -> str:
    normalized = algorithm.lower()
    if normalized not in {"sha256", "sha384"}:
        raise ValueError(f"Unsupported hash algorithm: {algorithm}")
    digest = hashlib.new(normalized)
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _verify_hmac_signature(
    payload: Mapping[str, object],
    signature: Mapping[str, object],
    *,
    key: bytes,
    algorithm: str,
) -> list[str]:
    declared_algorithm = signature.get("algorithm")
    if not isinstance(declared_algorithm, str):
        return ["signature-missing-algorithm"]

    if declared_algorithm != algorithm:
        return ["signature-algorithm-mismatch"]

    algorithm_upper = declared_algorithm.upper()
    if not algorithm_upper.startswith("HMAC-"):
        fallback_errors = validate_hmac_signature(payload, {"signature": signature}, key=key, algorithm=algorithm)
        return [f"signature-{error}" for error in fallback_errors] if fallback_errors else []

    digest_name = algorithm_upper.split("-", 1)[-1].lower()
    if not digest_name or not hasattr(hashlib, digest_name):
        return ["signature-unsupported-algorithm"]

    digest_func = getattr(hashlib, digest_name)
    expected_value = base64.b64encode(hmac.new(key, canonical_json_bytes(payload), digest_func).digest()).decode("ascii")
    actual_value = signature.get("value")
    if not isinstance(actual_value, str):
        return ["signature-missing-value"]
    if not hmac.compare_digest(actual_value, expected_value):
        return ["signature-mismatch"]
    return []


@dataclass(slots=True)
class NotarizationResult:
    """Summary of a notarization attempt."""

    command: list[str]
    stapled: bool
    ticket_path: Path | None
    dry_run: bool
    log_path: Path | None = None

    def to_mapping(self) -> Mapping[str, object]:
        return {
            "command": self.command,
            "stapled": self.stapled,
            "ticket_path": str(self.ticket_path) if self.ticket_path else None,
            "dry_run": self.dry_run,
            "log_path": str(self.log_path) if self.log_path else None,
        }


class NotarizationService:
    """Handles submission of archives to external notarization services."""

    def __init__(
        self,
        *,
        bundle_id: str,
        tool: str = "xcrun",
        profile: str | None = None,
        apple_id: str | None = None,
        password: str | None = None,
        team_id: str | None = None,
        staple: bool = True,
        dry_run: bool = False,
        log_path: Path | None = None,
        timeout_seconds: int | None = None,
    ) -> None:
        self._bundle_id = bundle_id
        self._tool = tool
        self._profile = profile
        self._apple_id = apple_id
        self._password = password
        self._team_id = team_id
        self._staple = staple
        self._dry_run = dry_run
        self._log_path = log_path.expanduser() if isinstance(log_path, Path) else (Path(log_path).expanduser() if log_path else None)
        self._timeout = timeout_seconds

    def notarize(self, context: PackagingContext) -> NotarizationResult:
        command = self._build_command(context.archive_path)
        log_path = None
        if self._log_path:
            self._log_path.parent.mkdir(parents=True, exist_ok=True)
            log_path = self._log_path

        if self._dry_run:
            _LOGGER.info("Dry-run notarization: %s", " ".join(command))
            if log_path:
                log_path.write_text(
                    json.dumps(
                        {
                            "bundle": context.bundle_name,
                            "version": context.version,
                            "platform": context.platform,
                            "command": command,
                            "mode": "dry-run",
                        },
                        ensure_ascii=False,
                        indent=2,
                    )
                    + "\n",
                    encoding="utf-8",
                )
            return NotarizationResult(command=command, stapled=False, ticket_path=None, dry_run=True, log_path=log_path)

        _LOGGER.info("Submitting archive for notarization: %s", context.archive_path)
        result = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            timeout=self._timeout,
        )
        if result.returncode != 0:
            raise RuntimeError(
                "Notarization command failed with exit code {}: {}".format(result.returncode, result.stderr.strip())
            )
        if log_path:
            log_path.write_text(result.stdout, encoding="utf-8")

        stapled = False
        ticket_path: Path | None = None
        if self._staple:
            stapled, ticket_path = self._staple_ticket(context.archive_path)
        return NotarizationResult(command=command, stapled=stapled, ticket_path=ticket_path, dry_run=False, log_path=log_path)

    def _build_command(self, archive_path: Path) -> list[str]:
        command = [self._tool, "notarytool", "submit", str(archive_path), "--wait", "--bundle-id", self._bundle_id]
        if self._profile:
            command.extend(["--keychain-profile", self._profile])
        else:
            if not (self._apple_id and self._password and self._team_id):
                raise ValueError("Notarization requires either keychain profile or Apple ID credentials")
            command.extend([
                "--apple-id",
                self._apple_id,
                "--password",
                self._password,
                "--team-id",
                self._team_id,
            ])
        return command

    def _staple_ticket(self, archive_path: Path) -> tuple[bool, Path | None]:
        stapler = [self._tool, "stapler", "staple", str(archive_path)]
        _LOGGER.info("Stapling notarization ticket: %s", " ".join(stapler))
        result = subprocess.run(stapler, capture_output=True, text=True)
        if result.returncode != 0:
            _LOGGER.warning("Stapler command failed: %s", result.stderr.strip())
            return False, None
        ticket_path = archive_path.with_suffix(archive_path.suffix + ".ticket")
        try:
            ticket_path.write_text(result.stdout, encoding="utf-8")
        except OSError:
            ticket_path = None
        return True, ticket_path


@dataclass(slots=True)
class PackagingPipelineReport:
    """Combined report from the packaging pipeline."""

    fingerprint: HardwareFingerprintReport | None = None
    delta_updates: list[DeltaUpdateResult] = field(default_factory=list)
    delta_manifests: list[DeltaManifestResult] = field(default_factory=list)
    update_packages: list[UpdatePackageResult] = field(default_factory=list)
    code_signatures: list[CodeSigningResult] = field(default_factory=list)
    notarization: NotarizationResult | None = None

    def to_mapping(self) -> Mapping[str, object]:
        payload: dict[str, object] = {}
        if self.fingerprint is not None:
            payload["fingerprint"] = self.fingerprint.to_mapping()
        if self.delta_updates:
            payload["delta_updates"] = [entry.to_mapping() for entry in self.delta_updates]
        if self.delta_manifests:
            payload["delta_manifests"] = [entry.to_mapping() for entry in self.delta_manifests]
        if self.update_packages:
            payload["update_packages"] = [entry.to_mapping() for entry in self.update_packages]
        if self.code_signatures:
            payload["code_signatures"] = [entry.to_mapping() for entry in self.code_signatures]
        if self.notarization is not None:
            payload["notarization"] = self.notarization.to_mapping()
        return payload


class PackagingPipeline:
    """Coordinates post-build packaging steps."""

    def __init__(
        self,
        *,
        fingerprint_validator: HardwareFingerprintValidator | None = None,
        delta_builder: DeltaUpdateBuilder | None = None,
        delta_manifest_publisher: DeltaManifestPublisher | None = None,
        update_package_builder: UpdatePackageBuilder | None = None,
        code_signing_service: CodeSigningService | None = None,
        notarization_service: NotarizationService | None = None,
        logger: logging.Logger | None = None,
    ) -> None:
        self._fingerprint_validator = fingerprint_validator
        self._delta_builder = delta_builder
        self._delta_manifest_publisher = delta_manifest_publisher
        self._update_package_builder = update_package_builder
        self._code_signing_service = code_signing_service
        self._notarization_service = notarization_service
        self._logger = logger or _LOGGER

    def execute(self, context: PackagingContext) -> PackagingPipelineReport:
        report = PackagingPipelineReport()

        if self._fingerprint_validator:
            self._logger.debug("Validating hardware fingerprint document")
            report.fingerprint = self._fingerprint_validator.validate(context)
            if report.fingerprint.issues:
                self._logger.warning("Fingerprint validation reported issues: %s", ", ".join(report.fingerprint.issues))
            else:
                self._logger.info("Fingerprint document verified for %s", context.version)

        if self._delta_builder:
            self._logger.debug("Generating delta updates")
            report.delta_updates = self._delta_builder.build(context)
            for delta in report.delta_updates:
                self._logger.info(
                    "Generated delta archive %s (changed=%d removed=%d)",
                    delta.archive_path,
                    len(delta.changed_files),
                    len(delta.removed_files),
                )

        if self._delta_manifest_publisher and report.delta_updates:
            self._logger.debug("Exporting delta manifests")
            report.delta_manifests = self._delta_manifest_publisher.publish(context, report.delta_updates)
            for manifest in report.delta_manifests:
                self._logger.info("Wrote delta manifest %s", manifest.manifest_path)

        if self._update_package_builder:
            self._logger.debug("Building offline update package")
            report.update_packages = self._update_package_builder.build(
                context,
                report.delta_updates,
                report.delta_manifests,
            )
            for package in report.update_packages:
                self._logger.info("Update package generated: %s", package.package_path)

        if self._notarization_service:
            self._logger.debug("Submitting archive for notarization")
            report.notarization = self._notarization_service.notarize(context)
            if report.notarization.dry_run:
                self._logger.info("Notarization dry-run completed for %s", context.archive_path)
            else:
                self._logger.info("Archive notarized successfully: %s", context.archive_path)

        if self._code_signing_service:
            self._logger.debug("Running code signing commands")
            report.code_signatures = self._code_signing_service.sign(context, report)
            for signature in report.code_signatures:
                self._logger.info("Code signing completed for %s", signature.target)

        return report


def build_pipeline_from_mapping(config: Mapping[str, object], *, base_dir: Path | None = None) -> PackagingPipeline:
    """Create a :class:`PackagingPipeline` from a configuration mapping."""

    base_dir = base_dir or Path.cwd()

    fingerprint_validator = None
    fingerprint_cfg = config.get("fingerprint_validation")
    if isinstance(fingerprint_cfg, Mapping):
        fingerprint_validator = HardwareFingerprintValidator(
            expected_fingerprint=_get_str(fingerprint_cfg, "expected"),
            signature_key=_get_secret(fingerprint_cfg.get("hmac_key"), base_dir=base_dir),
            fail_on_missing=bool(fingerprint_cfg.get("fail_on_missing", True)),
            allow_placeholder=bool(fingerprint_cfg.get("allow_placeholder", False)),
            verify_local=bool(fingerprint_cfg.get("verify_local", False)),
        )

    delta_builder = None
    delta_manifest_publisher = None
    delta_cfg = config.get("delta_updates")
    if isinstance(delta_cfg, Mapping):
        base_entries = delta_cfg.get("bases")
        if isinstance(base_entries, Sequence):
            base_paths = [_resolve_path(str(entry), base_dir=base_dir) for entry in base_entries]
        else:
            base_paths = []
        output_dir_entry = delta_cfg.get("output_dir")
        if output_dir_entry:
            output_dir = _resolve_path(str(output_dir_entry), base_dir=base_dir)
            compression = str(delta_cfg.get("compression", "zip"))
            delta_builder = DeltaUpdateBuilder(base_manifests=base_paths, output_dir=output_dir, compression=compression)
        manifest_dir_entry = delta_cfg.get("manifest_dir")
        if manifest_dir_entry:
            manifest_dir = _resolve_path(str(manifest_dir_entry), base_dir=base_dir)
            signing_key = _get_secret(delta_cfg.get("manifest_signing_key"), base_dir=base_dir)
            signing_key_id = _get_str(delta_cfg, "manifest_signing_key_id")
            embed_hwid = _get_bool(delta_cfg, "embed_hwid")
            delta_manifest_publisher = DeltaManifestPublisher(
                output_dir=manifest_dir,
                signing_key=signing_key,
                signing_key_id=signing_key_id,
                embed_hwid=embed_hwid,
            )

    update_package_builder = None
    update_package_cfg = config.get("update_package")
    if isinstance(update_package_cfg, Mapping):
        package_output = update_package_cfg.get("output_dir")
        package_id = _get_str(update_package_cfg, "package_id")
        runtime = _get_str(update_package_cfg, "runtime")
        if package_output and package_id and runtime:
            metadata_cfg = update_package_cfg.get("metadata")
            metadata = metadata_cfg if isinstance(metadata_cfg, Mapping) else {}
            update_package_builder = UpdatePackageBuilder(
                output_dir=_resolve_path(str(package_output), base_dir=base_dir),
                package_id=package_id,
                runtime=runtime,
                signing_key=_get_secret(update_package_cfg.get("signing_key"), base_dir=base_dir),
                signing_key_id=_get_str(update_package_cfg, "signing_key_id"),
                embed_hwid=_get_bool(update_package_cfg, "embed_hwid"),
                extra_metadata=metadata,
            )

    code_signing_service = None
    code_signing_cfg = config.get("code_signing")
    if isinstance(code_signing_cfg, Mapping):
        command_entries = code_signing_cfg.get("command")
        if isinstance(command_entries, Sequence) and command_entries:
            command = [str(part) for part in command_entries]
            targets_raw = code_signing_cfg.get("targets")
            if isinstance(targets_raw, Sequence) and targets_raw:
                targets = [str(target) for target in targets_raw]
            else:
                targets = ["archive"]
            env_cfg = code_signing_cfg.get("env")
            environment = {str(k): str(v) for k, v in env_cfg.items()} if isinstance(env_cfg, Mapping) else None
            code_signing_service = CodeSigningService(
                command=command,
                targets=targets,
                environment=environment,
                dry_run=_get_bool(code_signing_cfg, "dry_run"),
                timeout_seconds=_get_int(code_signing_cfg, "timeout_seconds"),
            )

    notarization_service = None
    notarization_cfg = config.get("notarization")
    if isinstance(notarization_cfg, Mapping):
        bundle_id = _get_str(notarization_cfg, "bundle_id")
        if bundle_id:
            notarization_service = NotarizationService(
                bundle_id=bundle_id,
                tool=str(notarization_cfg.get("tool", "xcrun")),
                profile=_get_str(notarization_cfg, "profile"),
                apple_id=_get_str(notarization_cfg, "apple_id"),
                password=_get_str(notarization_cfg, "password"),
                team_id=_get_str(notarization_cfg, "team_id"),
                staple=bool(notarization_cfg.get("staple", True)),
                dry_run=bool(notarization_cfg.get("dry_run", False)),
                log_path=_resolve_optional_path(notarization_cfg.get("log_path"), base_dir=base_dir),
                timeout_seconds=_get_int(notarization_cfg, "timeout_seconds"),
            )

    return PackagingPipeline(
        fingerprint_validator=fingerprint_validator,
        delta_builder=delta_builder,
        delta_manifest_publisher=delta_manifest_publisher,
        update_package_builder=update_package_builder,
        code_signing_service=code_signing_service,
        notarization_service=notarization_service,
    )


def _resolve_path(value: str, *, base_dir: Path) -> Path:
    candidate = Path(value)
    if not candidate.is_absolute():
        candidate = base_dir / candidate
    return candidate.expanduser().resolve()


def _resolve_optional_path(value: object, *, base_dir: Path) -> Path | None:
    if value in (None, ""):
        return None
    return _resolve_path(str(value), base_dir=base_dir)


def _get_secret(value: object, *, base_dir: Path) -> bytes | None:
    if value in (None, ""):
        return None
    if isinstance(value, (bytes, bytearray)):
        return bytes(value)
    text = str(value).strip()
    if text.startswith("env:"):
        env_name = text[4:]
        resolved = os.environ.get(env_name)
        if resolved is None:
            raise KeyError(f"Environment variable {env_name!r} is not defined")
        text = resolved.strip()
    elif text.startswith("file:"):
        path = _resolve_path(text[5:], base_dir=base_dir)
        text = path.read_text(encoding="utf-8").strip()
    if text.startswith("hex:"):
        return bytes.fromhex(text[4:])
    if text.startswith("base64:"):
        return base64.b64decode(text[7:])
    try:
        return base64.b64decode(text)
    except Exception:
        return text.encode("utf-8")


def _get_str(mapping: Mapping[str, object], key: str) -> str | None:
    value = mapping.get(key)
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _get_int(mapping: Mapping[str, object], key: str) -> int | None:
    value = mapping.get(key)
    if value in (None, ""):
        return None
    return int(value)


def _get_bool(mapping: Mapping[str, object], key: str, default: bool = False) -> bool:
    value = mapping.get(key)
    if value in (None, ""):
        return default
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    return text not in {"0", "false", "no", "off"}


__all__ = [
    "PackagingContext",
    "HardwareFingerprintValidator",
    "HardwareFingerprintReport",
    "DeltaUpdateBuilder",
    "DeltaUpdateResult",
    "DeltaManifestResult",
    "UpdatePackageResult",
    "CodeSigningResult",
    "DeltaManifestPublisher",
    "UpdatePackageBuilder",
    "CodeSigningService",
    "NotarizationService",
    "NotarizationResult",
    "PackagingPipeline",
    "PackagingPipelineReport",
    "build_pipeline_from_mapping",
]
