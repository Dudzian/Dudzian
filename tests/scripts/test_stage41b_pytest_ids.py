"""Cross-platform collection checks, including Linux-only Stage 41B cases."""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys


_STAGE41B_MODULES = (
    "test_stage41b_signer_isolation.py",
    "test_stage41b_harness_lifecycle.py",
    "test_stage41b_environment_preparation.py",
)

# Inspect actual collected items without running their fixtures or test bodies.
# The terminal plugin stays disabled so a regressed multi-megabyte auto-ID is
# reported as a numeric length, never expanded into the subprocess's output.
_COLLECTION_PROBE = r"""
import json
from pathlib import Path
import sys

import pytest


class Probe:
    def pytest_collection_finish(self, session):
        summary = {
            "modules": {},
            "max_nodeid_bytes": 0,
            "max_case_id_bytes": 0,
            "oversized_byte_cases": [],
        }
        for item in session.items:
            module = item.path.name
            summary["modules"][module] = summary["modules"].get(module, 0) + 1
            summary["max_nodeid_bytes"] = max(
                summary["max_nodeid_bytes"], len(item.nodeid.encode("utf-8"))
            )
            callspec = getattr(item, "callspec", None)
            if callspec is None:
                continue
            summary["max_case_id_bytes"] = max(
                summary["max_case_id_bytes"], len(callspec.id.encode("utf-8"))
            )
            if item.originalname == "test_json_decoder_accepts_only_bounded_exact_bytes":
                raw = callspec.params["raw"]
                limit = item.module.worker.MAX_REQUEST_BYTES
                if type(raw) is bytes and len(raw) > limit:
                    summary["oversized_byte_cases"].append({
                        "case_id": callspec.id[:64],
                        "size": len(raw),
                        "limit": limit,
                    })
        Path(sys.argv[1]).write_text(json.dumps(summary), encoding="utf-8")


raise SystemExit(int(pytest.main([
    "--collect-only", "-p", "no:terminal", "-p", "no:cacheprovider",
    "-o", "addopts=", *sys.argv[2:],
], plugins=[Probe()])))
"""


def test_stage41b_collection_keeps_ids_bounded_and_exact_oversized_input(tmp_path):
    repository = Path(__file__).resolve().parents[2]
    report_path = tmp_path / "stage41b-collection.json"
    environment = dict(os.environ)
    environment.pop("PYTEST_ADDOPTS", None)
    collected = subprocess.run(
        [
            sys.executable,
            "-c",
            _COLLECTION_PROBE,
            str(report_path),
            *(str(Path("tests/security") / name) for name in _STAGE41B_MODULES),
        ],
        cwd=repository,
        env=environment,
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert collected.returncode == 0, collected.stderr[-2000:]
    summary = json.loads(report_path.read_text(encoding="utf-8"))
    assert set(summary["modules"]) == set(_STAGE41B_MODULES)
    assert all(count > 0 for count in summary["modules"].values())
    assert summary["max_nodeid_bytes"] <= 240
    assert summary["max_case_id_bytes"] <= 96
    oversized = summary["oversized_byte_cases"]
    assert len(oversized) == 1
    assert oversized[0]["case_id"] == "one-byte-over-limit"
    assert oversized[0]["size"] == oversized[0]["limit"] + 1
