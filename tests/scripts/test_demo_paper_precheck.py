from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "scripts/demo_paper_precheck.py"
DEMO_CONFIG = REPO_ROOT / "config/e2e/demo_paper.yml"


def _run(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(SCRIPT), *args],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )


def test_demo_paper_precheck_passes_for_repo_config() -> None:
    result = _run("--config", str(DEMO_CONFIG))
    assert result.returncode == 0, result.stderr or result.stdout
    assert "OK:" in result.stdout


def test_demo_paper_precheck_fails_for_live_enabled(tmp_path: Path) -> None:
    payload = yaml.safe_load(DEMO_CONFIG.read_text(encoding="utf-8"))
    payload["execution"]["live"]["enabled"] = True
    mutated = tmp_path / "unsafe_demo_paper.yml"
    mutated.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")

    result = _run("--config", str(mutated))
    assert result.returncode != 0
    assert "unsafe_flag:execution.live.enabled" in result.stdout


def test_demo_paper_precheck_json_output() -> None:
    result = _run("--config", str(DEMO_CONFIG), "--json")
    assert result.returncode == 0

    payload = json.loads(result.stdout)
    assert payload["status"] == "ok"
    assert payload["checks"]["execution.default_mode"]["observed"] == "paper"
