"""Read-only validation of the immutable Stage-9 MSI handoff; never Stage-10 evidence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path, PurePosixPath, PureWindowsPath
import re
import shutil
import stat
import tempfile
from typing import Any
import zipfile

from deployment.platform_evidence import validate_windows_clean_install_receipt
from deployment.platform_readiness import _validated_results, load_contract
from deployment.windows_installer.build import installed_payload_digest, sha256
from deployment.windows_stage9_production_trust import (
    CEREMONY_ID,
    load_production_trust,
    validate_public_package_layout,
)

ARTIFACT_NAME = "windows-stage9-qualified-msi"
REPOSITORY = "Dudzian/Dudzian"
WORKFLOW = ".github/workflows/platform-deployment.yml"
JSON_MEMBERS = frozenset(
    {
        "installer-manifest.json",
        "windows-clean-install-receipt.json",
        "windows-clean-install-evidence.json",
    }
)
EXECUTABLE_PATHS = {
    "CryptoHunterBackend.exe": "CryptoHunterBackend.exe",
    "CryptoHunterFreshnessVerifier.exe": "CryptoHunterFreshnessVerifier.exe",
    "CryptoHunterPostgreSQL.exe": "PostgreSQL/CryptoHunterPostgreSQL.exe",
    "CryptoHunterProvision.exe": "CryptoHunterProvision.exe",
}
MAX_JSON_BYTES = 8 * 1024 * 1024
MAX_ARCHIVE_BYTES = 4 * 1024 * 1024 * 1024


class ArtifactPreflightError(ValueError):
    """The selected download cannot establish an operator-reviewable MSI handoff."""


def _require(condition: bool, reason: str) -> None:
    if not condition:
        raise ArtifactPreflightError(reason)


def _digest(value: Any, name: str) -> str:
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise ArtifactPreflightError(f"invalid {name}")
    return value


def _identifier(value: Any, name: str) -> str:
    if not isinstance(value, str) or re.fullmatch(r"[1-9][0-9]*", value) is None:
        raise ArtifactPreflightError(f"invalid {name}")
    return value


def _no_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        _require(key not in result, f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _reject_constant(value: str) -> None:
    raise ArtifactPreflightError(f"non-finite JSON value: {value}")


def _document(path: Path) -> dict[str, Any]:
    _require(path.stat().st_size <= MAX_JSON_BYTES, f"JSON exceeds size limit: {path.name}")
    value = json.loads(
        path.read_text(encoding="utf-8"),
        object_pairs_hook=_no_duplicate_keys,
        parse_constant=_reject_constant,
    )
    if not isinstance(value, dict):
        raise ArtifactPreflightError(f"JSON object required: {path.name}")
    return value


def _is_link(path: Path) -> bool:
    return path.is_symlink() or bool(
        getattr(path.lstat(), "st_file_attributes", 0)
        & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)
    )


def authorized_path(
    path: Path, allowed_root: Path, *, directory: bool = False, output: bool = False
) -> Path:
    """Confine input reads/output writes to the operator's explicit staging directory."""
    root = allowed_root.absolute()
    _require(root.is_dir(), "authorized staging root is unavailable")
    for component in (root, *root.parents):
        _require(not _is_link(component), "linked/reparse staging root is forbidden")
    candidate = path.absolute()
    _require(
        candidate != root and candidate.is_relative_to(root),
        "path is outside authorized staging root",
    )
    _require(".." not in candidate.parts, "parent traversal is forbidden")
    for component in (candidate, *candidate.parents):
        if component == root:
            break
        if output and component == candidate and not component.exists():
            continue
        _require(not _is_link(component), "linked/reparse input or output is forbidden")
    if output:
        _require(candidate.parent.is_dir(), "report parent directory is unavailable")
    else:
        _require(
            candidate.is_dir() if directory else candidate.is_file(),
            "required staged input is unavailable",
        )
    _require(
        candidate.resolve().is_relative_to(root.resolve()),
        "resolved path escapes authorized staging root",
    )
    return candidate


def _metadata(
    artifact: dict[str, Any],
    run: dict[str, Any],
    *,
    artifact_id: str,
    archive_digest: str,
    revision: str,
    run_id: str,
) -> None:
    workflow_run = artifact.get("workflow_run")
    if not isinstance(workflow_run, dict):
        raise ArtifactPreflightError("artifact workflow_run binding is missing")
    _require(
        type(artifact.get("id")) is int
        and str(artifact["id"]) == artifact_id
        and artifact.get("name") == ARTIFACT_NAME
        and artifact.get("digest") == "sha256:" + archive_digest
        and artifact.get("expired") is False
        and artifact.get("archive_download_url")
        == f"https://api.github.com/repos/{REPOSITORY}/actions/artifacts/{artifact_id}/zip",
        "artifact identity, download location, expiry or digest differs from independent pins",
    )
    _require(
        type(workflow_run.get("id")) is int
        and str(workflow_run["id"]) == run_id
        and workflow_run.get("head_sha") == revision
        and workflow_run.get("head_branch") == "main",
        "artifact belongs to a different revision, branch or CI run",
    )
    _require(
        type(run.get("id")) is int
        and str(run["id"]) == run_id
        and run.get("head_sha") == revision
        and run.get("head_branch") == "main"
        and run.get("event") == "workflow_dispatch"
        # REST can include the workflow ref. Compare exact main representations:
        # removing an arbitrary suffix would also admit tags, PRs or foreign refs.
        and run.get("path") in (WORKFLOW, f"{WORKFLOW}@main", f"{WORKFLOW}@refs/heads/main")
        and run.get("status") == "completed"
        and run.get("conclusion") == "success"
        and isinstance(run.get("repository"), dict)
        and run["repository"].get("full_name") == REPOSITORY,
        "qualification must be a successful, completed operator-dispatched main workflow in Dudzian/Dudzian",
    )


def _safe_member(name: Any) -> bool:
    if not isinstance(name, str) or not name or "\\" in name or ":" in name:
        return False
    posix, windows = PurePosixPath(name), PureWindowsPath(name)
    return (
        not posix.is_absolute()
        and not windows.is_absolute()
        and all(part not in {"", ".", ".."} for part in name.split("/"))
        and all(not part.endswith((" ", ".")) for part in name.split("/"))
    )


def _extract(archive: Path, target: Path) -> None:
    """Materialize only the four reviewed data files inside a temporary directory."""
    with zipfile.ZipFile(archive) as zipped:
        entries = zipped.infolist()
        names = [item.filename for item in entries]
        _require(
            len(entries) == 4 and len({name.casefold() for name in names}) == 4,
            "qualified artifact must contain exactly four unique files",
        )
        _require(
            JSON_MEMBERS.issubset(names),
            "qualified artifact receipt, evidence or manifest is missing",
        )
        _require(
            sum(item.file_size for item in entries) <= MAX_ARCHIVE_BYTES,
            "uncompressed artifact exceeds size limit",
        )
        for item in entries:
            mode = item.external_attr >> 16
            _require(
                _safe_member(item.filename) and "/" not in item.filename and not item.is_dir(),
                "archive member location is forbidden",
            )
            _require(
                stat.S_IFMT(mode) in {0, stat.S_IFREG} and not item.flag_bits & 1,
                "archive links, special files or encryption are forbidden",
            )
            _require(
                item.filename in JSON_MEMBERS
                or re.fullmatch(
                    r"CryptoHunter-[0-9]+\.[0-9]+\.[0-9]+-windows-x64\.msi", item.filename
                )
                is not None,
                "unexpected qualified artifact member",
            )
            _require(
                item.filename not in JSON_MEMBERS or item.file_size <= MAX_JSON_BYTES,
                "artifact JSON exceeds size limit",
            )
            with zipped.open(item) as source, (target / item.filename).open("xb") as destination:
                shutil.copyfileobj(source, destination, length=1024 * 1024)


def _payload(manifest: dict[str, Any]) -> None:
    installed = manifest.get("installed_files")
    executables = manifest.get("production_executables")
    _require(
        type(manifest.get("schema_version")) is int and manifest["schema_version"] == 1,
        "unsupported installer manifest schema",
    )
    if not isinstance(installed, dict) or not installed:
        raise ArtifactPreflightError("payload file hashes are missing")
    if not isinstance(executables, dict) or set(executables) != set(EXECUTABLE_PATHS):
        raise ArtifactPreflightError("production executable set differs")
    folded: set[str] = set()
    for name, digest in installed.items():
        _require(
            _safe_member(name) and name.casefold() not in folded,
            "payload path is unsafe or collides on Windows",
        )
        _require(
            all(PureWindowsPath(part).is_reserved() is False for part in name.split("/")),
            "reserved Windows payload path",
        )
        folded.add(name.casefold())
        _digest(digest, "payload file SHA-256")
    _require(
        _digest(manifest.get("installed_payload_sha256"), "payload SHA-256")
        == installed_payload_digest(installed),
        "payload digest differs from canonical installed file map",
    )
    for name, relative in EXECUTABLE_PATHS.items():
        _require(
            _digest(executables[name], "production executable SHA-256") == installed.get(relative),
            "production executable and payload hash differ",
        )
    pins = _document(Path(__file__).with_name("windows_installer") / "pins.json")["postgresql"]
    _require(
        manifest.get("postgresql_payload_sha256") == pins["sha256"],
        "PostgreSQL payload digest differs from reviewed pin",
    )
    _require(
        manifest.get("postgresql_source_url") == pins["source_url"],
        "PostgreSQL source differs from reviewed pin",
    )


def validate_qualified_artifact(
    archive: Path,
    artifact_metadata: Path,
    workflow_run_metadata: Path,
    production_trust_package: Path,
    *,
    allowed_root: Path,
    expected_artifact_id: str,
    expected_archive_sha256: str,
    expected_revision: str,
    expected_run_id: str,
    expected_production_trust_run_id: str,
    expected_production_trust_manifest_sha256: str,
) -> dict[str, Any]:
    """Check immutable download and shared Stage-9 validators, with no installed-product access."""
    artifact_id = _identifier(expected_artifact_id, "artifact ID")
    run_id = _identifier(expected_run_id, "qualification CI run ID")
    trust_run_id = _identifier(expected_production_trust_run_id, "Production Trust CI run ID")
    archive_digest = _digest(expected_archive_sha256, "independently pinned artifact digest")
    trust_digest = _digest(
        expected_production_trust_manifest_sha256,
        "independently pinned Production Trust manifest digest",
    )
    _require(
        isinstance(expected_revision, str)
        and re.fullmatch(r"[0-9a-f]{40}", expected_revision) is not None,
        "full independently pinned revision is required",
    )
    archive = authorized_path(archive, allowed_root)
    artifact_metadata = authorized_path(artifact_metadata, allowed_root)
    workflow_run_metadata = authorized_path(workflow_run_metadata, allowed_root)
    production_trust_package = authorized_path(
        production_trust_package, allowed_root, directory=True
    )
    artifact, run = _document(artifact_metadata), _document(workflow_run_metadata)
    _metadata(
        artifact,
        run,
        artifact_id=artifact_id,
        archive_digest=archive_digest,
        revision=expected_revision,
        run_id=run_id,
    )
    _require(archive.stat().st_size <= MAX_ARCHIVE_BYTES, "downloaded archive exceeds size limit")
    _require(
        sha256(archive) == archive_digest,
        "downloaded artifact archive SHA-256 differs from independent pin",
    )
    validate_public_package_layout(production_trust_package)
    for public_file in production_trust_package.iterdir():
        authorized_path(public_file, allowed_root)
    _require(
        sha256(production_trust_package / "package_manifest.json") == trust_digest,
        "Production Trust package manifest differs from independent pin",
    )
    trust = load_production_trust(production_trust_package)
    with tempfile.TemporaryDirectory(
        prefix="stage10-artifact-rehearsal-", dir=allowed_root
    ) as temporary:
        staging = Path(temporary)
        _extract(archive, staging)
        manifest_path = staging / "installer-manifest.json"
        receipt_path = staging / "windows-clean-install-receipt.json"
        manifest = _document(manifest_path)
        receipt = _document(receipt_path)
        evidence = _document(staging / "windows-clean-install-evidence.json")
        _payload(manifest)
        msi_value = manifest.get("msi")
        if not isinstance(msi_value, dict) or set(msi_value) != {"file", "sha256"}:
            raise ArtifactPreflightError("manifest MSI binding is missing or malformed")
        _require(
            isinstance(manifest.get("product_version"), str)
            and msi_value["file"] == f"CryptoHunter-{manifest['product_version']}-windows-x64.msi",
            "manifest MSI filename differs from product version",
        )
        _require(
            _safe_member(msi_value["file"]) and "/" not in msi_value["file"],
            "manifest MSI location is forbidden",
        )
        msi_path = staging / msi_value["file"]
        validate_windows_clean_install_receipt(
            receipt_path,
            msi=msi_path,
            manifest=manifest_path,
            production_trust_package=production_trust_package,
            source_revision=expected_revision,
            ci_provider="https://github.com",
            ci_run_id=run_id,
            runner_arch="X64",
        )
        _require(
            type(evidence.get("schema_version")) is int,
            "evidence schema version must be an integer",
        )
        results = _validated_results(
            load_contract(), [evidence], expected_revision, "https://github.com", run_id
        )
        _require(
            set(results) == {"WINDOWS_CLEAN_INSTALL"}
            and results["WINDOWS_CLEAN_INSTALL"]["status"] == "PASS"
            and evidence["runner_arch"] == "X64",
            "qualified clean-install evidence must be exactly Windows x64 PASS",
        )
        _require(
            results["WINDOWS_CLEAN_INSTALL"]["details"]
            == f"MSI sha256={receipt['msi_sha256']}; version={receipt['product_version']}; architecture=X64",
            "clean-install evidence does not describe the qualified MSI",
        )
        expected_trust = {
            "ceremony_id": CEREMONY_ID,
            "package_manifest_sha256": trust_digest,
            "artifact_run_id": trust_run_id,
            "frozen_production_trust": "PASS",
        }
        _require(
            evidence["production_trust"] == expected_trust
            and receipt.get("production_trust_artifact_run_id") == trust_run_id
            and receipt.get("production_trust_package_manifest_sha256") == trust_digest,
            "receipt/evidence Production Trust binding differs from independent pins",
        )
        details = {
            "artifact_id": artifact_id,
            "artifact_name": ARTIFACT_NAME,
            "archive_sha256": archive_digest,
            "source_revision": expected_revision,
            "ci_run_id": run_id,
            "msi_file": msi_path.name,
            "msi_sha256": sha256(msi_path),
            "manifest_sha256": sha256(manifest_path),
            "installed_payload_sha256": manifest["installed_payload_sha256"],
            "production_trust": expected_trust,
            "release_payload_digest": trust.release_payload_digest,
        }
    return details


def artifact_preflight(*args: Any, **kwargs: Any) -> dict[str, Any]:
    """Return an explicit preparatory result; validation failures never grant Stage-10 PASS."""
    report: dict[str, Any] = {
        "report_type": "windows-stage10-read-only-artifact-preflight",
        "schema_version": 1,
        "overall_status": "FAIL",
        "stage10_qualification": "NOT_RUN",
        "status": "FAIL",
        "stage10_pass": False,
        "scope": "read-only artifact rehearsal; no installation or target runtime proof",
    }
    try:
        report["details"] = validate_qualified_artifact(*args, **kwargs)
    except (
        OSError,
        ValueError,
        RuntimeError,
        KeyError,
        TypeError,
        AttributeError,
        zipfile.BadZipFile,
        EOFError,
    ) as exc:
        report["reason"] = f"{type(exc).__name__}: {exc}"
    else:
        report["status"] = "PASS"
        report["reason"] = (
            "Pinned qualified download, MSI hash, payload hash contract and shared Stage-9/Production Trust validators passed. Target installed payload remains unverified."
        )
    report["overall_status"] = report["status"]
    report["checks"] = [
        {
            "name": "qualified_artifact_provenance",
            "status": report["status"],
            "details": report["reason"],
        }
    ]
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in (
        "archive",
        "artifact-metadata",
        "workflow-run-metadata",
        "production-trust-package",
        "allowed-root",
        "output",
    ):
        parser.add_argument("--" + name, type=Path, required=True)
    for name in (
        "artifact-id",
        "archive-sha256",
        "revision",
        "run-id",
        "production-trust-run-id",
        "production-trust-manifest-sha256",
    ):
        parser.add_argument("--expected-" + name, required=True)
    args = parser.parse_args(argv)
    # Validate output before reading inputs; never overwrite a staged qualification file.
    try:
        output = authorized_path(args.output, args.allowed_root, output=True)
        _require(
            not output.exists(), "report must be a new file, never a staged qualification input"
        )
        _require(
            not output.is_relative_to(args.production_trust_package.absolute()),
            "report cannot modify the public Production Trust package",
        )
    except (OSError, ValueError) as exc:
        print(json.dumps({"status": "FAIL", "stage10_pass": False, "reason": str(exc)}))
        return 1
    report = artifact_preflight(
        args.archive,
        args.artifact_metadata,
        args.workflow_run_metadata,
        args.production_trust_package,
        allowed_root=args.allowed_root,
        expected_artifact_id=args.expected_artifact_id,
        expected_archive_sha256=args.expected_archive_sha256,
        expected_revision=args.expected_revision,
        expected_run_id=args.expected_run_id,
        expected_production_trust_run_id=args.expected_production_trust_run_id,
        expected_production_trust_manifest_sha256=args.expected_production_trust_manifest_sha256,
    )
    try:
        with output.open("x", encoding="utf-8") as stream:
            stream.write(json.dumps(report, indent=2) + "\n")
    except OSError as exc:
        print(json.dumps({"status": "FAIL", "stage10_pass": False, "reason": str(exc)}))
        return 1
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
