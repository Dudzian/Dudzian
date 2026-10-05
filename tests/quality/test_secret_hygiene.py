from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

import scripts.quality.betterleaks_hook as betterleaks_hook


ROOT = Path(__file__).resolve().parents[2]
FORBIDDEN_TRACKED_STORES = {
    ".env",
    "secrets/binance_paper.json",
    "secrets/headless_secrets.json",
    "var/secrets/headless.json",
}


def _tracked_files() -> set[str]:
    completed = subprocess.run(
        ["git", "ls-files", "-z"],
        cwd=ROOT,
        check=True,
        capture_output=True,
    )
    return {item.decode() for item in completed.stdout.split(b"\0") if item}


def test_local_runtime_secret_stores_are_not_tracked() -> None:
    tracked = _tracked_files()

    assert tracked.isdisjoint(FORBIDDEN_TRACKED_STORES)
    assert "env.example" in tracked
    assert "secrets/licensing/offline_portal.py" in tracked


@pytest.mark.parametrize(
    "path",
    [".env", "secrets/operator-runtime.json", "var/secrets/operator-runtime.json"],
)
def test_runtime_secret_locations_are_ignored(path: str) -> None:
    ignored = subprocess.run(
        ["git", "check-ignore", "--quiet", "--no-index", path],
        cwd=ROOT,
        check=False,
    )

    assert ignored.returncode == 0


def test_runtime_secret_examples_are_reviewable() -> None:
    example = subprocess.run(
        ["git", "check-ignore", "--quiet", "--no-index", "secrets/operator.example.json"],
        cwd=ROOT,
        check=False,
    )
    assert example.returncode == 1


def test_baseline_has_no_runtime_secret_store_entries() -> None:
    baseline = json.loads((ROOT / ".betterleaks-baseline.json").read_text(encoding="utf-8"))

    assert {finding["File"] for finding in baseline}.isdisjoint(FORBIDDEN_TRACKED_STORES)


def test_new_synthetic_secret_is_blocked(monkeypatch, tmp_path: Path) -> None:
    candidate = tmp_path / "ordinary.txt"
    candidate.write_text("synthetic-secret-for-gate-regression", encoding="utf-8")
    monkeypatch.setattr(betterleaks_hook, "_find_betterleaks", lambda: "betterleaks")

    def fake_run(command: list[str], *, check: bool) -> subprocess.CompletedProcess[str]:
        assert check is False
        assert command[-1] == "."
        assert "gate-regression" in candidate.read_text(encoding="utf-8")
        return subprocess.CompletedProcess(command, 1)

    monkeypatch.setattr(betterleaks_hook.subprocess, "run", fake_run)

    assert betterleaks_hook.main([str(candidate)]) == 1


def test_reviewed_false_positive_uses_baseline_and_passes(monkeypatch, tmp_path: Path) -> None:
    candidate = tmp_path / "reviewed.txt"
    candidate.write_text("documented-synthetic-fixture", encoding="utf-8")
    monkeypatch.chdir(ROOT)
    monkeypatch.setattr(betterleaks_hook, "_find_betterleaks", lambda: "betterleaks")

    def fake_run(command: list[str], *, check: bool) -> subprocess.CompletedProcess[str]:
        assert check is False
        assert command[command.index("--baseline-path") + 1] == ".betterleaks-baseline.json"
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(betterleaks_hook.subprocess, "run", fake_run)

    assert betterleaks_hook.main([str(candidate)]) == 0
