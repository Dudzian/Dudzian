"""Unprivileged diagnostic only; never bypasses Source/CCA preflight."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import NoReturn


_COMMITS = (
    "782816540373d7b31f426dab645ad115320d1234",
    "26e17e7c23ea6cc1b265d57bfda1a1e6589d9e3f",
)
_INSTALLATION_REASONS = (
    "INDEPENDENT_INVENTORY_MISMATCH",
    "INSTALLATION_OBSERVATION_OR_VERIFICATION_FAILED",
    "INDEPENDENT_EVIDENCE_UNAVAILABLE_OR_INVALID",
)
_FAILURE_CODES = (
    "PERMISSION_DENIED",
    "NOT_FOUND",
    "IDENTITY_CHANGED",
    "MALFORMED",
    "IO_ERROR",
    "LIMIT",
    "EMPTY",
)
_PROCESS_REASONS = (
    "NO_ATOMIC_HOST_SNAPSHOT",
    "NO_PROOF_FOR_COPIED_OR_CLOSED_FILES",
    "HOST_NAMESPACE_NOT_EXTERNALLY_ATTESTED",
    "INSTALLATION_METADATA_INCOMPLETE",
    "INSTALLATION_METADATA_INCOMPLETE_OR_CHANGED",
    "PROCFS_INCOMPLETE",
    "PROCFS_VISIBILITY_UNCERTAIN",
    "PROCESS_SWEEP_LIMIT",
    "PROCESS_PID_LIMIT",
    "PROCESS_INODE_LIMIT",
    "PROCESS_OBSERVATION_CHANGED",
    "PROCESS_CONTENT_CHANGED",
    "THREAD_CONTENT_CHANGED",
    "LINK_MALFORMED",
    "MAPS_LINE_LIMIT",
) + tuple(
    source + "_" + failure
    for source in (
        "PROCESS_ENUMERATION",
        "PROCESS",
        "THREAD",
        "STAT",
        "STATUS",
        "CGROUP",
        "MAPS",
        "EXE",
        "CWD",
        "ROOT",
        "FD",
    )
    for failure in _FAILURE_CODES
)


def _closed_code(value: object, choices: tuple[str, ...], fallback: str) -> str:
    # Return a reviewed literal from choices, never a string from observations.
    if type(value) is str:
        for code in choices:
            if value == code:
                return code
    return fallback


def _safe_projection(report: object) -> dict[str, object]:
    """Publish only closed status codes and booleans, never host observations.

    diagnose/resources/processes contain NSS records, paths and procfs-derived
    data. Unknown fields are intentionally not traversed: a future observation
    must receive a separate output-schema review before it can reach stdout.
    """
    result: dict[str, object] = {
        "schema": 1,
        "read_only": True,
        "qualification": "BLOCKED",
        "coexistence_authorized": False,
        "operator_authorization": "NOT_CONSUMED_DIAGNOSTIC_ONLY",
        "physical_test_status": "PHYSICAL_TEST_PENDING",
        "status": "IN_PROGRESS",
        "production_status": "PRODUCTION_NO_GO",
        "preflight_exception": False,
        "evidence_pin_verified": False,
        "blockers": [
            "NO_COMPLETE_INDEPENDENT_HOST_ISOLATION_PROOF",
            "NO_OTHER_INSTALLATIONS_GATE_RETAINED",
        ],
    }
    if type(report) is not dict:
        result["reason"] = "DIAGNOSTIC_REPORT_SCHEMA_UNAVAILABLE"
        return result
    if "reason" in report:
        result["reason"] = _closed_code(
            report["reason"],
            (
                "DIAGNOSTIC_INPUT_OR_OBSERVATION_UNAVAILABLE",
                "PRIVILEGED_DIAGNOSTICS_NOT_IMPLEMENTED_EXTERNAL_VERIFIED_PROTECTED_LOADER_REQUIRED",
            ),
            "DIAGNOSTIC_REPORT_SCHEMA_UNAVAILABLE",
        )
    result["evidence_pin_verified"] = report.get("evidence_pin_verified") is True
    installations = report.get("installations")
    if type(installations) is list and len(installations) == len(_COMMITS):
        projected = []
        for expected_commit, item in zip(_COMMITS, installations, strict=True):
            if (
                type(item) is not dict
                or type(item.get("commit")) is not str
                or item["commit"] != expected_commit
            ):
                continue
            projected.append(
                {
                    "commit": expected_commit,
                    "integrity": _closed_code(item.get("integrity"), ("PASS",), "UNCERTAIN"),
                    "provenance": _closed_code(
                        item.get("provenance"), ("PASS", "BLOCKED"), "UNCERTAIN"
                    ),
                    "reason": _closed_code(
                        item.get("reason"), _INSTALLATION_REASONS, "NO_CLOSED_REASON_REPORTED"
                    ),
                }
            )
        result["installations"] = projected
    resources = report.get("resources")
    if type(resources) is dict:
        result["resources"] = {
            "state": _closed_code(resources.get("state"), ("BLOCKED",), "UNCERTAIN"),
            "scope": "KNOWN_RESOURCES_AND_SOURCE_CCA_PLAN",
            "reason": _closed_code(
                resources.get("reason"),
                ("RESOURCE_OBSERVATION_INCOMPLETE_OR_CHANGED",),
                "NO_CLOSED_REASON_REPORTED",
            ),
            "limitations": [
                "LEGACY_RESOURCE_INVENTORY_NOT_CLOSED",
                "NSS_ENUMERATION_NOT_PROOF_OF_ABSENCE",
                "UNIT_CONTENT_AND_ALIASES_NOT_FULLY_VERIFIED",
            ],
        }
    processes = report.get("process_observation")
    if type(processes) is dict:
        uncertainty = processes.get("uncertainty")
        codes = (
            [
                code
                for code in _PROCESS_REASONS
                if any(type(item) is str and item == code for item in uncertainty)
            ]
            if type(uncertainty) is list
            else []
        )
        result["process_observation"] = {
            "state": _closed_code(processes.get("state"), ("BLOCKED",), "UNCERTAIN"),
            "scope": "VISIBLE_PROCFS_ALL_THREADS_EXE_CWD_ROOT_FD_MAPS",
            "full_confirmation": False,
            "stable_visible_snapshot": processes.get("stable_visible_snapshot") is True,
            "stable_process_identities": processes.get("stable_process_identities") is True,
            "stable_installation_bindings": processes.get("stable_installation_bindings") is True,
            "reason": _closed_code(
                processes.get("reason"),
                ("PROCESS_OBSERVATION_INCOMPLETE_OR_CHANGED",),
                "NO_CLOSED_REASON_REPORTED",
            ),
            "uncertainty": codes,
            "limitations": [
                "NO_ATOMIC_HOST_SNAPSHOT",
                "NO_PROOF_FOR_COPIED_OR_CLOSED_FILES",
                "HOST_NAMESPACE_NOT_EXTERNALLY_ATTESTED",
            ],
        }
    return result


class _ArgumentParser(argparse.ArgumentParser):
    def error(self, message: str) -> NoReturn:
        # argparse's message can include untrusted argument values and paths.
        self.exit(
            2,
            json.dumps(
                {
                    "qualification": "BLOCKED",
                    "reason": "DIAGNOSTIC_ARGUMENTS_INVALID",
                    "physical_test_status": "PHYSICAL_TEST_PENDING",
                    "status": "IN_PROGRESS",
                    "production_status": "PRODUCTION_NO_GO",
                },
                sort_keys=True,
            )
            + "\n",
        )


def main(argv: list[str] | None = None) -> int:
    parser = _ArgumentParser(prog="diagnose_stage9_stage41b_coexistence", description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--uid-base", type=int, default=62000)
    parser.add_argument("--evidence", type=Path)
    parser.add_argument("--evidence-sha256", default="")
    args = parser.parse_args(argv)
    # Refuse before importing control code or reading host/installation data.
    # An acknowledgement flag cannot authenticate already executed Python code.
    if hasattr(os, "geteuid") and os.geteuid() == 0:
        print(
            json.dumps(
                {
                    "qualification": "BLOCKED",
                    "physical_test_status": "PHYSICAL_TEST_PENDING",
                    "status": "IN_PROGRESS",
                    "production_status": "PRODUCTION_NO_GO",
                    "reason": "PRIVILEGED_DIAGNOSTICS_NOT_IMPLEMENTED_EXTERNAL_VERIFIED_PROTECTED_LOADER_REQUIRED",
                }
            )
        )
        return 2
    raw = b""
    try:
        from deployment.stage9_source_cca_test_package import plan
        from deployment.stage9_stage41b_coexistence import LIMIT, _read, diagnose

        if args.evidence is not None and args.evidence.lstat().st_size <= LIMIT:
            raw = _read(args.evidence)
        report = _safe_projection(
            diagnose(plan(args.run_id, args.uid_base), raw, args.evidence_sha256)
        )
    except Exception:
        report = _safe_projection({"reason": "DIAGNOSTIC_INPUT_OR_OBSERVATION_UNAVAILABLE"})
    print(json.dumps(report, sort_keys=True, indent=2))
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
