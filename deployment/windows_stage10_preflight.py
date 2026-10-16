"""Summarize PRE-01A rehearsal findings without granting lifecycle authority."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import stat
import sys
from typing import Any

CURRENT_STATUS = Path(__file__).with_name("stage9_current_status.json")
STATUSES = {"PASS", "FAIL", "BLOCKED", "UNVERIFIED"}


def _load_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("report must be a JSON object")
    return value


def _component(path: Path | None, kind: str) -> dict[str, str]:
    if path is None:
        return {
            "name": kind,
            "status": "UNVERIFIED",
            "details": "No actual host observations or qualified artifact supplied. "
            "Tests and configuration examples do not establish this condition.",
        }
    try:
        value = _load_object(path)
        checks = value.get("checks")
        if (
            value.get("schema_version") != 1
            or value.get("report_type") != f"windows-stage10-read-only-{kind}-preflight"
            or value.get("stage10_qualification") != "NOT_RUN"
            or value.get("overall_status") not in STATUSES
            or not isinstance(checks, list)
            or not checks
        ):
            raise ValueError("unsupported component report")
        for check in checks:
            if (
                not isinstance(check, dict)
                or check.get("status") not in STATUSES
                or not isinstance(check.get("name"), str)
                or not check["name"]
                or not isinstance(check.get("details"), str)
                or not check["details"]
            ):
                raise ValueError("malformed component check")
        statuses = {check["status"] for check in checks}
        if value["overall_status"] == "PASS" and statuses != {"PASS"}:
            raise ValueError("component PASS contradicts individual checks")
        status = "FAIL" if "FAIL" in statuses else value["overall_status"]
        details = "; ".join(
            f"{check['name']}={check['status']}: {check['details']}" for check in checks
        )
        return {
            "name": kind,
            "status": status,
            "details": "Imported rehearsal summary; revalidate original inputs on transfer. "
            + details,
        }
    except (OSError, ValueError, TypeError) as exc:
        return {"name": kind, "status": "FAIL", "details": f"Invalid component report: {exc}"}


def prepare_report(
    *, runner_report: Path | None = None, artifact_report: Path | None = None
) -> dict[str, Any]:
    """This report is intentionally outside the platform execution evidence schema."""
    checks = [_component(runner_report, "runner"), _component(artifact_report, "artifact")]
    try:
        status = _load_object(CURRENT_STATUS)
        checks.append(
            {
                "name": "stage9_closure_and_legal_enrollment",
                "status": "BLOCKED",
                "details": f"Current source: stage_9={status.get('stage_9')}; "
                f"legal_production_enrollment={status.get('legal_production_enrollment')}; "
                f"stage_10_prerequisite={status.get('stage_10_prerequisite')}. "
                "PRE-01A cannot close Stage 9 or authorize enrollment/lifecycle execution.",
            }
        )
    except (OSError, ValueError) as exc:
        checks.append(
            {"name": "stage9_closure_and_legal_enrollment", "status": "FAIL", "details": str(exc)}
        )
    checks.extend(
        [
            {
                "name": "target_host_installation",
                "status": "BLOCKED",
                "details": "The Stage-10 prepare job downloads clean-install evidence only. "
                "It does not deliver or install windows-stage9-qualified-msi on the target. "
                "S9-S10-PRE-01B must implement the separately operator-approved handoff, "
                "exact MSI installation and passive post-install verification.",
            },
            {
                "name": "stage10_execution",
                "status": "BLOCKED",
                "details": "No installation, reboot, TPM/NV mutation, enrollment, private-key "
                "operation or 24-hour qualification is performed or authorized. "
                "Separate operator approval and all existing live gates remain required.",
            },
        ]
    )
    return {
        "schema_version": 1,
        "report_type": "windows-stage10-read-only-preflight-summary",
        "roadmap_task": "S9-S10-PRE-01A",
        "overall_status": "FAIL" if any(c["status"] == "FAIL" for c in checks) else "BLOCKED",
        "verification_scope": "REHEARSAL_REPORT_ONLY",
        "stage10_qualification": "NOT_RUN",
        "stage10_execution_authorized": False,
        "checks": checks,
        "next_task": "S9-S10-PRE-01B",
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runner-report", type=Path)
    parser.add_argument("--artifact-report", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    report = prepare_report(runner_report=args.runner_report, artifact_report=args.artifact_report)
    try:
        output = args.output.absolute()
        for component in (output, *output.parents):
            if component.is_symlink() or (
                component.exists()
                and getattr(component.lstat(), "st_file_attributes", 0)
                & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)
            ):
                raise ValueError("report output cannot use symlink/reparse paths")
        with output.open("x", encoding="utf-8") as stream:
            stream.write(json.dumps(report, indent=2) + "\n")
    except (OSError, ValueError) as exc:
        print(f"Cannot create a new preflight report: {exc}", file=sys.stderr)
        return 1
    print(f"PRE-01A={report['overall_status']}; Stage 10 NOT_RUN")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
