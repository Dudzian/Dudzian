from __future__ import annotations

import json
from pathlib import Path
import platform

import pytest

from deployment.platform_evidence import (
    EvidenceProductionError,
    aggregate_core_markers,
    run_windows_scm_probe,
)
from deployment.core_test_plan import load_manifest, marker_document


def test_windows_scm_producer_refuses_non_windows_host(tmp_path: Path) -> None:
    if platform.system() == "Windows":
        pytest.skip("negative host-binding test applies only to non-Windows runners")
    with pytest.raises(EvidenceProductionError, match="actual Windows runner"):
        run_windows_scm_probe("revision", tmp_path / "evidence.json")


def test_core_aggregation_requires_same_revision_and_all_operating_systems(
    tmp_path: Path,
) -> None:
    markers = []
    manifest = load_manifest()
    for runner_os in ("Linux", "Windows", "macOS"):
        marker = tmp_path / f"{runner_os}.json"
        marker.write_text(
            json.dumps(marker_document(
                manifest, "current", runner_os, "run-1", "https://github.com"
            )),
            encoding="utf-8",
        )
        markers.append(marker)
    output = tmp_path / "core.json"
    aggregate_core_markers(
        "current", markers, output,
        ci_run_id="run-1", ci_provider="https://github.com",
    )
    evidence = json.loads(output.read_text(encoding="utf-8"))
    assert evidence["platform"] == "CROSS_PLATFORM_CORE"
    assert evidence["source_revision"] == "current"
    assert evidence["results"][0]["item"] == "CORE_REQUIRED_SUITES"

    stale = tmp_path / "Linux.json"
    stale.write_text(
        json.dumps(marker_document(
            manifest, "previous", "Linux", "run-1", "https://github.com"
        )),
        encoding="utf-8",
    )
    with pytest.raises(EvidenceProductionError, match="stale"):
        aggregate_core_markers(
            "current", markers, output,
            ci_run_id="run-1", ci_provider="https://github.com",
        )


def test_core_aggregation_rejects_stale_plan_duplicate_runner_and_wrong_run(
    tmp_path: Path,
) -> None:
    manifest = load_manifest()

    def write(name: str, runner_os: str, **updates: str) -> Path:
        value = marker_document(
            manifest, "current", runner_os, "run-1", "https://github.com"
        )
        value.update(updates)
        path = tmp_path / name
        path.write_text(json.dumps(value), encoding="utf-8")
        return path

    linux = write("linux.json", "Linux")
    windows = write("windows.json", "Windows")
    macos = write("macos.json", "macOS")
    stale_plan = write("old-plan.json", "Linux", test_plan_digest="0" * 64)
    with pytest.raises(EvidenceProductionError, match="stale"):
        aggregate_core_markers(
            "current", [stale_plan, windows, macos], tmp_path / "out.json",
            ci_run_id="run-1", ci_provider="https://github.com",
        )
    duplicate = write("linux-duplicate.json", "Linux")
    with pytest.raises(EvidenceProductionError, match="exactly three|duplicate"):
        aggregate_core_markers(
            "current", [linux, duplicate, windows, macos], tmp_path / "out.json",
            ci_run_id="run-1", ci_provider="https://github.com",
        )
    wrong_run = write("wrong-run.json", "Linux", ci_run_id="run-old")
    with pytest.raises(EvidenceProductionError, match="stale"):
        aggregate_core_markers(
            "current", [wrong_run, windows, macos], tmp_path / "out.json",
            ci_run_id="run-1", ci_provider="https://github.com",
        )
