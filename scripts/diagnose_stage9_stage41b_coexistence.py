"""Unprivileged diagnostic only; never bypasses Source/CCA preflight."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


def _redact_sensitive(value: object) -> object:
    sensitive_keys = {"legacy_identity_aliases", "passwd", "group"}
    if isinstance(value, dict):
        return {
            key: ("<redacted>" if key in sensitive_keys else _redact_sensitive(item))
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [_redact_sensitive(item) for item in value]
    return value


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
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
    from deployment.stage9_source_cca_test_package import plan
    from deployment.stage9_stage41b_coexistence import LIMIT, _read, diagnose

    raw = b""
    try:
        if args.evidence is not None and args.evidence.lstat().st_size <= LIMIT:
            raw = _read(args.evidence)
        report = diagnose(plan(args.run_id, args.uid_base), raw, args.evidence_sha256)
    except Exception:
        report = {
            "qualification": "BLOCKED",
            "reason": "DIAGNOSTIC_INPUT_OR_OBSERVATION_UNAVAILABLE",
            "physical_test_status": "PHYSICAL_TEST_PENDING",
            "status": "IN_PROGRESS",
            "production_status": "PRODUCTION_NO_GO",
        }
    print(json.dumps(_redact_sensitive(report), sort_keys=True, indent=2))
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
