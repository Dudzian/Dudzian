"""Fresh authenticated custody checks for a separately approved Stage-9 MSI handoff.

This module reads staged public files and GitHub metadata. It never executes an
MSI, changes a host, authenticates an operator, or authorizes Stage 10.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import zipfile
from collections.abc import Callable
from pathlib import Path
from typing import Any

from deployment.windows_stage10_artifact_preflight import (
    ArtifactProvenanceUnavailable,
    _no_duplicate_keys,
    _reject_constant,
    _require,
    authorized_path,
    validate_qualified_artifact,
)


def _github_get(endpoint: str) -> Any:
    """Use gh's configured authentication for GET only; never retain token/output diagnostics."""
    try:
        result = subprocess.run(
            ["gh", "api", "--method", "GET", "--hostname", "github.com", endpoint],
            check=True,
            capture_output=True,
            text=True,
            timeout=45,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise ArtifactProvenanceUnavailable("authenticated GitHub access is unavailable") from exc
    return json.loads(
        result.stdout.lstrip("\ufeff"),
        object_pairs_hook=_no_duplicate_keys,
        parse_constant=_reject_constant,
    )


def qualified_handoff(
    *args: Any,
    github_get: Callable[[str], Any] | None = None,
    **kwargs: Any,
) -> dict[str, Any]:
    """Revalidate original ZIP, exact attempt, MSI/manifest and trust before every use.

    ``github_get`` is the read-only authenticated transport dependency, injectable
    for hermetic tests. Offline metadata and saved PASS reports cannot replace it.
    Expected pins (including ``expected_run_attempt``) come from the independent
    reviewed selection; this function never infers them from downloaded content.
    """
    report: dict[str, Any] = {
        "schema_version": 1,
        "report_type": "windows-stage9-qualified-msi-handoff",
        "status": "FAIL",
        "overall_status": "FAIL",
        "stage10_qualification": "NOT_RUN",
        "stage10_pass": False,
        "stage10_execution_authorized": False,
        "installation_executed": False,
        "installation_authorized": False,
        "scope": (
            "fresh read-only authenticated artifact custody; "
            "separate host/operator approval required"
        ),
    }
    try:
        details = validate_qualified_artifact(
            *args, github_get=github_get if github_get is not None else _github_get, **kwargs
        )
    except (ArtifactProvenanceUnavailable, OSError) as exc:
        report.update(status="BLOCKED", reason=f"{type(exc).__name__}: {exc}")
    except (
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
        report.update(
            status="PASS",
            details=details,
            reason=(
                "Authenticated canonical producer success and independently pinned "
                "artifact/MSI/manifest/Production Trust checks passed; "
                "no host operation is authorized."
            ),
        )
    report["overall_status"] = report["status"]
    report["checks"] = [
        {"name": "qualified_msi_custody", "status": report["status"], "details": report["reason"]}
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
        "run-attempt",
        "production-trust-run-id",
        "production-trust-manifest-sha256",
    ):
        parser.add_argument("--expected-" + name, required=True)
    args = vars(parser.parse_args(argv))
    try:
        output = authorized_path(args.pop("output"), args["allowed_root"], output=True)
        _require(not output.exists(), "handoff report must be a new file")
        _require(
            not output.is_relative_to(args["production_trust_package"].absolute()),
            "handoff report cannot modify the public Production Trust package",
        )
    except (OSError, ValueError) as exc:
        print(json.dumps({"status": "FAIL", "stage10_pass": False, "reason": str(exc)}))
        return 1
    report = qualified_handoff(**args)
    try:
        with output.open("x", encoding="utf-8") as stream:
            stream.write(json.dumps(report, indent=2) + "\n")
    except OSError as exc:
        print(json.dumps({"status": "FAIL", "stage10_pass": False, "reason": str(exc)}))
        return 1
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
