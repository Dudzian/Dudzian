from __future__ import annotations

import json
import os
import shlex
import shutil
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
import yaml

WORKFLOW = Path(__file__).resolve().parents[2] / ".github/workflows/ci.yml"
REPOSITORY = "Dudzian/Dudzian"
SHA = "842edcc7877e39e91d8d65e0bf97ed826d64d901"
BASH = shutil.which("bash")
JQ = shutil.which("jq")
Resolver = Callable[..., tuple[subprocess.CompletedProcess[str], str]]


@pytest.fixture
def parity_job() -> dict[str, Any]:
    jobs = yaml.safe_load(WORKFLOW.read_text(encoding="utf-8"))["jobs"]
    return jobs["marketing-bundle-parity"]


@pytest.fixture
def resolver_script(parity_job: dict[str, Any]) -> str:
    (step,) = [step for step in parity_job["steps"] if step.get("id") == "stress-lab-run"]
    assert not parity_job.get("continue-on-error", False)
    assert not step.get("continue-on-error", False)
    assert step["shell"] == "bash"
    assert step["env"]["GH_TOKEN"] == "${{ github.token }}"
    script = step["run"]
    assert isinstance(script, str)
    return script.replace("${{ github.repository }}", REPOSITORY)


def _run(
    run_id: int,
    *,
    branch: str = "main",
    sha: str = SHA,
    status: str = "completed",
    conclusion: str | None = "success",
) -> dict[str, str | int | None]:
    return {
        "id": run_id,
        "head_branch": branch,
        "head_sha": sha,
        "status": status,
        "conclusion": conclusion,
        "event": "push",
    }


@pytest.fixture
def resolve(tmp_path: Path, resolver_script: str) -> Resolver:
    if sys.platform != "linux" or BASH is None or JQ is None:
        pytest.skip("Replaying the Ubuntu workflow step requires Bash and jq on Linux")

    fake_gh = tmp_path / "gh"
    fake_gh.write_text(
        r"""#!/bin/sh
printf '%s\n' "$@" > "$GH_ARGUMENTS_FILE"
cat "$GH_RESPONSE_FILE"
if [ "$GH_EXIT_CODE" != 0 ]; then
    printf '%s\n' 'gh: Resource not accessible by integration (HTTP 403)' >&2
fi
exit "$GH_EXIT_CODE"
""",
        encoding="utf-8",
    )
    fake_gh.chmod(0o700)
    gh_arguments = tmp_path / "gh-arguments"
    gh_response = tmp_path / "gh-response"
    output = tmp_path / "github-output"
    output.touch()

    def invoke(
        pages: list[list[dict[str, str | int | None]]],
        *,
        branch: str = "main",
        exit_code: int = 0,
    ) -> tuple[subprocess.CompletedProcess[str], str]:
        total_count = sum(map(len, pages))
        gh_response.write_text(
            "\n".join(
                json.dumps({"total_count": total_count, "workflow_runs": runs}) for runs in pages
            ),
            encoding="utf-8",
        )
        environment = {
            "PATH": os.pathsep.join([str(tmp_path), os.environ.get("PATH", os.defpath)]),
            "GH_ARGUMENTS_FILE": str(gh_arguments),
            "GH_RESPONSE_FILE": str(gh_response),
            "GH_EXIT_CODE": str(exit_code),
            "GH_TOKEN": "fixture-token",
            "GITHUB_REPOSITORY": REPOSITORY,
            "GITHUB_SHA": SHA,
            "GITHUB_OUTPUT": str(output),
            "BRANCH_NAME": branch,
            "DEFAULT_BRANCH": "main",
            "EVENT_NAME": "push",
        }
        result = subprocess.run(
            [BASH, "--noprofile", "--norc", "-e", "-o", "pipefail", "-c", resolver_script],
            env=environment,
            cwd=tmp_path,
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
        arguments = gh_arguments.read_text(encoding="utf-8").splitlines()
        assert arguments[0] == "api"
        assert arguments[arguments.index("--method") + 1] == "GET"
        assert f"/repos/{REPOSITORY}/actions/workflows/stress-lab-report.yml/runs" in arguments
        assert f"branch={branch}" in arguments
        assert f"head_sha={SHA}" in arguments
        assert "--paginate" in arguments
        assert not any(argument.startswith("status=") for argument in arguments)
        return result, output.read_text(encoding="utf-8")

    return invoke


def test_download_places_report_directly_in_validation_directory(
    parity_job: dict[str, Any],
) -> None:
    (download,) = [
        step["with"]
        for step in parity_job["steps"]
        if step.get("uses", "").startswith("actions/download-artifact@")
        and step.get("with", {}).get("name") == "stress-lab-report"
    ]
    (validation,) = [
        step["run"]
        for step in parity_job["steps"]
        if "scripts.audit.marketing_parity_check" in step.get("run", "")
    ]
    args = shlex.split(validation.replace("\\\n", ""))

    assert download["run-id"] == "${{ steps.stress-lab-run.outputs.run_id }}"
    assert download["github-token"] == "${{ github.token }}"
    assert download["path"] == args[args.index("--stress-lab-dir") + 1]
    assert Path(download["path"]).name == "stress_lab"


def test_finds_incident_push_after_newer_ineligible_runs(resolve: Resolver) -> None:
    result, output = resolve(
        [
            [
                _run(50_000_000_005, status="queued", conclusion=None),
                _run(50_000_000_004, status="in_progress"),
                _run(50_000_000_003, conclusion="failure"),
                _run(50_000_000_002, branch="develop"),
                _run(50_000_000_001, sha="a" * 40),
            ],
            [_run(45890546559), _run(45890546558)],
        ]
    )

    assert result.returncode == 0, (result.stdout, result.stderr)
    assert output.splitlines() == ["run_id=45890546559", "missing=false"]


def test_fails_main_if_no_completed_success_matches_sha_and_branch(resolve: Resolver) -> None:
    result, output = resolve(
        [[_run(3, sha="a" * 40), _run(2, branch="develop"), _run(1, conclusion="failure")]]
    )

    assert result.returncode != 0
    assert "::error::No successful stress-lab-report workflow run found" in result.stdout
    assert not output


def test_feature_branch_keeps_missing_report_output(resolve: Resolver) -> None:
    result, output = resolve([[]], branch="feature/marketing")

    assert result.returncode == 0, (result.stdout, result.stderr)
    assert output.splitlines() == ["run_id=", "missing=true"]


def test_api_failure_cannot_accept_partial_results(resolve: Resolver) -> None:
    result, output = resolve([[_run(45890546559)]], exit_code=1)

    assert result.returncode != 0
    assert "HTTP 403" in result.stderr
    assert not output
