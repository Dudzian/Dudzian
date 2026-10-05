from __future__ import annotations

import json
import os
import subprocess
import sys
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


def _git(repository: Path, *args: str) -> None:
    subprocess.run(["git", *args], cwd=repository, check=True, capture_output=True)


def _repository(tmp_path: Path) -> Path:
    repository = tmp_path / "repo with spaces"
    repository.mkdir()
    _git(repository, "init", "-q")
    (repository / ".gitignore").write_text(".env\n/secrets/*.json\n", encoding="utf-8")
    _git(repository, "add", ".gitignore")
    _git(
        repository,
        "-c",
        "user.name=Test",
        "-c",
        "user.email=test@example.invalid",
        "commit",
        "-qm",
        "fixture",
    )
    return repository


def _capture_scanner(monkeypatch, returncode: int = 0) -> list[list[str]]:
    calls: list[list[str]] = []
    real_run = subprocess.run
    monkeypatch.setattr(betterleaks_hook, "_find_betterleaks", lambda: "betterleaks")

    def fake_run(command: list[str], **kwargs: object) -> subprocess.CompletedProcess[bytes]:
        if command[0] == "git":
            return real_run(command, **kwargs)
        calls.append(command)
        return subprocess.CompletedProcess(command, returncode)

    monkeypatch.setattr(betterleaks_hook.subprocess, "run", fake_run)
    return calls


def test_ignored_runtime_stores_are_not_scanned(monkeypatch, tmp_path: Path) -> None:
    repository = _repository(tmp_path)
    (repository / ".env").write_text("SYNTHETIC_TOKEN=fixture-only", encoding="utf-8")
    (repository / "secrets").mkdir()
    (repository / "secrets/operator-runtime.json").write_text(
        '{"token":"synthetic-fixture-only"}', encoding="utf-8"
    )
    monkeypatch.chdir(repository)
    calls = _capture_scanner(monkeypatch)

    assert betterleaks_hook.main([]) == 0
    assert calls == []


def test_force_added_runtime_store_is_scanned(monkeypatch, tmp_path: Path) -> None:
    repository = _repository(tmp_path)
    store = repository / "secrets/operator-runtime.json"
    store.parent.mkdir()
    store.write_text('{"token":"synthetic-fixture-only"}', encoding="utf-8")
    _git(repository, "add", "-f", "secrets/operator-runtime.json")
    monkeypatch.chdir(repository)
    calls = _capture_scanner(monkeypatch, returncode=1)

    assert betterleaks_hook.main([]) == 1
    assert len(calls) == 1
    assert calls[0][-1] == "secrets/operator-runtime.json"


def test_new_synthetic_secret_is_blocked(monkeypatch, tmp_path: Path) -> None:
    repository = _repository(tmp_path)
    candidate = repository / "source with ünicode.txt"
    candidate.write_text("synthetic-secret-for-gate-regression", encoding="utf-8")
    _git(repository, "add", candidate.name)
    monkeypatch.chdir(repository)
    calls = _capture_scanner(monkeypatch, returncode=1)

    assert betterleaks_hook.main([]) == 1
    assert len(calls) == 1
    assert calls[0][-1] == candidate.name


def test_reviewed_false_positive_uses_baseline_and_passes(monkeypatch, tmp_path: Path) -> None:
    repository = _repository(tmp_path)
    candidate = repository / "reviewed.txt"
    candidate.write_text("documented-synthetic-fixture", encoding="utf-8")
    (repository / ".betterleaks-baseline.json").write_text("[]", encoding="utf-8")
    monkeypatch.chdir(repository)
    monkeypatch.setattr(betterleaks_hook, "_find_betterleaks", lambda: "betterleaks")

    def fake_run(command: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        assert kwargs["check"] is False
        assert Path(command[command.index("--baseline-path") + 1]) == (
            repository / ".betterleaks-baseline.json"
        )
        assert command[-1] == os.path.relpath(candidate, repository).replace(os.sep, "/")
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(betterleaks_hook.subprocess, "run", fake_run)

    assert betterleaks_hook.main([str(candidate)]) == 0


def test_no_staged_files_does_not_start_scanner(monkeypatch, tmp_path: Path) -> None:
    repository = _repository(tmp_path)
    monkeypatch.chdir(repository)
    calls = _capture_scanner(monkeypatch)

    assert betterleaks_hook.main([]) == 0
    assert calls == []


@pytest.mark.parametrize("explicit", [True, False])
def test_cli_uses_explicit_paths_or_staged_index(tmp_path: Path, explicit: bool) -> None:
    repository = _repository(tmp_path)
    staged = repository / "staged.txt"
    explicit_path = repository / "explicit.txt"
    staged.write_text("staged fixture", encoding="utf-8")
    explicit_path.write_text("explicit fixture", encoding="utf-8")
    _git(repository, "add", staged.name)

    log = repository / "scanner-arguments.json"
    (repository / "dir").write_text(
        "import json, os, sys\n"
        "from pathlib import Path\n"
        "Path(os.environ['SCANNER_ARGUMENT_LOG']).write_text(json.dumps(sys.argv[1:]))\n",
        encoding="utf-8",
    )
    command = [sys.executable, str(ROOT / "scripts/quality/betterleaks_hook.py")]
    if explicit:
        command.append(str(explicit_path))
    completed = subprocess.run(
        command,
        cwd=repository,
        env={
            **os.environ,
            "BETTERLEAKS_BIN": sys.executable,
            "SCANNER_ARGUMENT_LOG": str(log),
        },
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    scanner_arguments = json.loads(log.read_text(encoding="utf-8"))
    separator = scanner_arguments.index("--")
    assert scanner_arguments[separator + 1 :] == [explicit_path.name if explicit else staged.name]
