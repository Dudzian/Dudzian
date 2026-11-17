"""Pełna historia CI i rzeczywista odmowa Git przy brakującym obiekcie."""

from __future__ import annotations

import os
from pathlib import Path
import re
import shutil
import subprocess

import pytest


ROOT = Path(__file__).resolve().parents[2]
WORKFLOW = ROOT / ".github/workflows/platform-deployment.yml"
HISTORY_SHA = "17e03a9cb42a84302401415f96ac90e756926915"
GUARD = f"git cat-file -e '{HISTORY_SHA}^{{commit}}'"
GIT = shutil.which("git")
BASH = shutil.which("bash")


def job_source() -> str:
    source = WORKFLOW.read_text(encoding="utf-8")
    match = re.search(
        r"^  stage9-v5d4-static-environment:\n(?P<job>.*?)(?=^  [\w-]+:|\Z)",
        source,
        flags=re.MULTILINE | re.DOTALL,
    )
    assert match is not None, "Brak joba statycznego środowiska V5d4"
    return match["job"]


def test_ci_requires_full_history_before_historical_tests() -> None:
    job = job_source()
    checkouts = re.findall(r"^      - uses: actions/checkout@[^\n]+\n([^\n]+)", job, re.MULTILINE)
    assert len(checkouts) == 1
    assert "fetch-depth: 0" in checkouts[0]
    assert 'ref: "${{ github.sha }}"' in checkouts[0]
    assert "persist-credentials: false" in checkouts[0]
    commands = [line.strip() for line in job.splitlines()]
    assert commands.count(GUARD) == 1
    first_pytest = next(index for index, line in enumerate(commands) if "python -m pytest" in line)
    assert commands.index(GUARD) < first_pytest
    assert "tests/deployment/test_stage9_v5d3_release.py" in job
    assert "tests/scripts/test_stage9_v5d4_ci_history.py" in job


@pytest.mark.skipif(GIT is None or BASH is None, reason="Wymagane rzeczywiste Git i Bash")
def test_history_guard_refuses_shallow_clone_then_accepts_full_history(tmp_path: Path) -> None:
    assert GIT is not None and BASH is not None
    environment = {
        "PATH": os.pathsep.join((str(Path(GIT).parent), str(Path(BASH).parent))),
        "LANG": "C",
        "LC_ALL": "C",
        "GIT_CONFIG_NOSYSTEM": "1",
        "GIT_CONFIG_GLOBAL": os.devnull,
        "GIT_TERMINAL_PROMPT": "0",
        "GIT_AUTHOR_NAME": "Fixture V5d4",
        "GIT_AUTHOR_EMAIL": "fixture@example.invalid",
        "GIT_COMMITTER_NAME": "Fixture V5d4",
        "GIT_COMMITTER_EMAIL": "fixture@example.invalid",
    }

    def git(*arguments: str, cwd: Path | None = None) -> str:
        result = subprocess.run(
            [GIT, "-c", "core.hooksPath=/dev/null", "-c", "commit.gpgsign=false", *arguments],
            cwd=cwd,
            env=environment,
            text=True,
            capture_output=True,
            timeout=20,
            check=True,
        )
        return result.stdout.strip()

    origin = tmp_path / "origin"
    clone = tmp_path / "shallow"
    git("init", "--initial-branch=main", str(origin))
    (origin / "fixture.txt").write_text("pierwszy\n", encoding="utf-8")
    git("add", "fixture.txt", cwd=origin)
    git("commit", "-m", "Historyczna baza fixture", cwd=origin)
    base = git("rev-parse", "HEAD", cwd=origin)
    (origin / "fixture.txt").write_text("drugi\n", encoding="utf-8")
    git("commit", "-am", "Nowa rewizja fixture", cwd=origin)
    git("clone", "--depth=1", "--no-tags", origin.as_uri(), str(clone))
    assert git("rev-parse", "--is-shallow-repository", cwd=clone) == "true"

    # Ten sam sprawdzony kontrakt polecenia; fixture ma własny autentyczny commit.
    guard = GUARD.replace(HISTORY_SHA, base)
    marker = clone / "pytest-reached"

    def run_guard() -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [BASH, "-e", "-c", f"{guard}\nprintf reached > pytest-reached\n"],
            cwd=clone,
            env=environment,
            text=True,
            capture_output=True,
            timeout=20,
            check=False,
        )

    refused = run_guard()
    assert refused.returncode != 0
    assert "Not a valid object name" in refused.stderr
    assert not marker.exists(), "Brak historii musi zatrzymać krok przed testami"

    git("fetch", "--unshallow", "origin", cwd=clone)
    assert git("rev-parse", "--is-shallow-repository", cwd=clone) == "false"
    assert git("cat-file", "-t", base, cwd=clone) == "commit"
    accepted = run_guard()
    assert accepted.returncode == 0, accepted.stderr
    assert marker.read_text(encoding="utf-8") == "reached"
