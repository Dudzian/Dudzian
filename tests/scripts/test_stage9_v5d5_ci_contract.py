"""V5d5 CI exercises portable contracts without authorizing native operations."""

from __future__ import annotations

import os
from pathlib import Path
import re
import subprocess
import sys
import textwrap
import tomllib

from coverage import CoverageData
import pytest


ROOT = Path(__file__).resolve().parents[2]


def workflow_job(name: str) -> str:
    source = (ROOT / ".github/workflows/platform-deployment.yml").read_text(encoding="utf-8")
    match = re.search(
        rf"^  {re.escape(name)}:\n(?P<job>.*?)(?=^  [\w-]+:|\Z)",
        source,
        flags=re.MULTILINE | re.DOTALL,
    )
    assert match is not None
    return match["job"]


def test_shared_contract_enters_configured_mypy_scope() -> None:
    configuration = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    files = configuration["tool"]["mypy"]["files"]
    assert files.count("scripts/stage9_v5d5_shared.py") == 1
    for module in ("dependencies", "environment", "launch", "abi", "closure"):
        assert f"scripts/stage9_v5d4_{module}.py" in files


def test_shared_contract_matrix_keeps_regressions_and_nonprivileged_scope() -> None:
    job = workflow_job("stage9-v5d5-shared-wheel-members")
    assert "os: [ubuntu-24.04, windows-latest, macos-latest]" in job
    assert 'with: {python-version: "3.12"}' in job
    assert "fail-fast: false" in job
    assert "permissions: {contents: read}" in job
    assert 'ref: "${{ github.sha }}", fetch-depth: 0, persist-credentials: false' in job
    assert "PYTEST_DISABLE_PLUGIN_AUTOLOAD" in job
    assert "python -m ruff check" in job
    assert "python -m ruff format --check" in job
    assert "python -m mypy scripts/stage9_v5d4_*.py scripts/stage9_v5d5_shared.py" in job
    assert "python -m mypy --platform win32" in job
    for inventory in (
        "tests/deployment/test_stage9_v5d5_*.py",
        "tests/deployment/test_stage9_v5d4_*.py",
        "tests/deployment/test_stage9_v5d3_release.py",
        "tests/scripts/test_stage9_operator_v5d2_inventory.py",
        "tests/scripts/test_stage9_v5d4_ci_history.py",
        "tests/scripts/test_stage9_v5d5_ci_contract.py",
    ):
        assert inventory in job
    guard = "git cat-file -e '17e03a9cb42a84302401415f96ac90e756926915^{commit}'"
    assert job.index(guard) < job.index("python -m pytest")
    assert "--cov=scripts.stage9_v5d5_shared" in job
    assert "--cov-branch" in job
    assert "COVERAGE_FILE: stage9-v5d5.coverage" in job
    assert "--fail-under" not in job
    assert "if: always()" in job
    assert "stage9-v5d5-static.xml" in job
    assert "stage9-v5d5-v5d2-regression.xml" in job
    assert "stage9-v5d5-coverage.xml" in job
    assert "if-no-files-found: error" in job
    for privileged in ("sudo ", "--execute", "run_stage10 == true", "run_linux_physical == true"):
        assert privileged not in job


def test_full_platform_matrix_is_required_before_combined_coverage_gate() -> None:
    job = workflow_job("stage9-v5d5-shared-wheel-coverage")
    assert "needs: stage9-v5d5-shared-wheel-members" in job
    assert "if: always() &&" in job
    assert "${{ needs.stage9-v5d5-shared-wheel-members.result }}" in job
    assert "V5D5_COVERAGE_MATRIX_FAILED" in job
    assert "fetch-depth: 0" in job
    assert 'runners = ("ubuntu-24.04", "windows-latest", "macos-latest")' in job
    assert "pattern: stage9-v5d5-static-*-${{ github.sha }}-${{ github.run_attempt }}" in job
    assert "V5D5_COVERAGE_ARTIFACT_SET_REFUSED" in job
    assert "V5D5_COVERAGE_SCOPE_REFUSED" in job
    assert "data.has_arcs()" in job
    assert "--compare-branch=origin/main --fail-under=70" in job
    assert "stage9-v5d5-combined-coverage.xml" in job
    assert "stage9-v5d5-combined.coverage" in job
    assert "if-no-files-found: error" in job


def run_aggregation(
    tmp_path: Path, change: str | None = None, windows_paths: bool = False
) -> subprocess.CompletedProcess[str]:
    job = workflow_job("stage9-v5d5-shared-wheel-coverage")
    match = re.search(
        r"      - name: Połącz dokładnie trzy pełne zakresy V5d5\n"
        r"        shell: python\n        run: \|\n(?P<script>(?:          [^\n]*\n)+)",
        job,
    )
    assert match is not None
    required = [
        *(
            f"scripts/stage9_v5d4_{name}.py"
            for name in ("abi", "closure", "dependencies", "environment", "launch")
        ),
        "scripts/stage9_v5d5_shared.py",
    ]
    revision = "a" * 40
    paths = [
        tmp_path
        / "stage9-v5d5-coverage-data"
        / f"stage9-v5d5-static-{runner}-{revision}-1"
        / "stage9-v5d5.coverage"
        for runner in ("ubuntu-24.04", "windows-latest", "macos-latest")
    ]
    for index, path in enumerate(paths):
        path.parent.mkdir(parents=True)
        data = CoverageData(basename=str(path))
        members = required[:1] if change == "incomplete" and index == 1 else required
        if windows_paths and index == 1:
            members = [member.replace("/", "\\") for member in members]
        if change == "branchless" and index == 1:
            data.add_lines({member: {index + 1} for member in members})
        else:
            data.add_arcs({member: {(-1, index + 1), (index + 1, -1)} for member in members})
        data.write()
    if change == "missing":
        paths[1].unlink()
    elif change == "empty":
        paths[1].write_bytes(b"")
    elif change == "unexpected":
        unexpected = tmp_path / "stage9-v5d5-coverage-data" / "unexpected"
        unexpected.mkdir()
        (unexpected / "stage9-v5d5.coverage").write_bytes(paths[0].read_bytes())
    (tmp_path / ".coveragerc").write_text("[run]\nrelative_files = True\n", encoding="utf-8")
    return subprocess.run(
        [sys.executable, "-c", textwrap.dedent(match["script"])],
        cwd=tmp_path,
        env={
            **{
                name: os.environ[name]
                for name in ("SYSTEMROOT", "WINDIR", "COMSPEC", "TEMP", "TMP")
                if name in os.environ
            },
            "PATH": os.defpath,
            "GITHUB_SHA": revision,
            "GITHUB_RUN_ATTEMPT": "1",
            "STAGE9_V5D5_MATRIX_RESULT": "failure" if change == "failed_matrix" else "success",
            "COVERAGE_FILE": "stage9-v5d5-combined.coverage",
        },
        capture_output=True,
        text=True,
        timeout=20,
        check=False,
    )


@pytest.mark.parametrize("windows_paths", [False, True])
def test_coverage_gate_combines_all_three_real_data_files(
    tmp_path: Path, windows_paths: bool
) -> None:
    result = run_aggregation(tmp_path, windows_paths=windows_paths)
    assert result.returncode == 0, result.stderr
    combined = CoverageData(basename=str(tmp_path / "stage9-v5d5-combined.coverage"))
    combined.read()
    assert combined.has_arcs()
    shared = [
        name
        for name in combined.measured_files()
        if name.replace("\\", "/") == "scripts/stage9_v5d5_shared.py"
    ]
    assert len(shared) == 1
    assert set(combined.lines(shared[0]) or ()) == {1, 2, 3}


@pytest.mark.parametrize(
    "change", ["missing", "empty", "unexpected", "incomplete", "branchless", "failed_matrix"]
)
def test_coverage_gate_refuses_partial_or_failed_matrix(tmp_path: Path, change: str) -> None:
    result = run_aggregation(tmp_path, change)
    assert result.returncode != 0
    assert "V5D5_COVERAGE_" in result.stderr
    assert not (tmp_path / "stage9-v5d5-combined.coverage").exists()
