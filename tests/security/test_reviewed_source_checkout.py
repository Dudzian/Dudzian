"""Rzeczywisty checkout Git zachowuje bajty przejrzanych pinów na obu platformach."""

from __future__ import annotations

import hashlib
from pathlib import Path
import subprocess

import pytest

from bot_core.freshness_deployment_qualification import REVIEWED_CODE_DIGESTS

ROOT = Path(__file__).resolve().parents[2]
# Zakres pięciu źródeł wprowadzonych/przejrzanych w PR #3234. Wcześniejszy
# drift pinu account_genesis pozostaje osobnym blockerem, bez akceptacji hasha.
REVIEWED_FILES = (
    "linux_service_ipc.py",
    "linux_service_postgresql.py",
    "production_signer_ipc.py",
    "freshness_semantic_verifier.py",
    "freshness_verifier_service.py",
)


def _checkout(tmp_path: Path, attributes: bytes, autocrlf: str) -> Path:
    repository = tmp_path / "git-index"
    repository.mkdir()

    def git(*arguments: str) -> None:
        subprocess.run(
            ["git", "-C", str(repository), *arguments],
            check=True,
            capture_output=True,
            timeout=20,
        )

    git("init", "--quiet")
    git("config", "core.autocrlf", "false")
    (repository / ".gitattributes").write_bytes(attributes)
    (repository / "bot_core").mkdir()
    for name in REVIEWED_FILES:
        (repository / "bot_core" / name).write_bytes((ROOT / "bot_core" / name).read_bytes())
    git("add", "--", ".gitattributes", *["bot_core/" + name for name in REVIEWED_FILES])
    git("config", "core.autocrlf", autocrlf)
    exported = tmp_path / "checked-out"
    exported.mkdir()
    git("checkout-index", "--all", "--force", "--prefix=" + exported.as_posix() + "/")
    return exported


@pytest.mark.parametrize("autocrlf", ["true", "false"])
def test_git_checkout_preserves_all_five_reviewed_exact_source_pins(tmp_path, autocrlf):
    """Git wykonuje rzeczywistą konwersję checkoutu; nie jest to model Windows CI."""
    exported = _checkout(tmp_path, (ROOT / ".gitattributes").read_bytes(), autocrlf)
    for name in REVIEWED_FILES:
        actual = (exported / "bot_core" / name).read_bytes()
        assert b"\r" not in actual
        assert actual == (ROOT / "bot_core" / name).read_bytes()
        assert hashlib.sha256(actual).hexdigest() == REVIEWED_CODE_DIGESTS[name]


def test_old_auto_text_policy_reproduces_exact_reported_windows_failure(tmp_path):
    exported = _checkout(tmp_path, b"* text=auto\n", "true")
    raw = (exported / "bot_core/linux_service_ipc.py").read_bytes()
    assert b"\r\n" in raw
    assert hashlib.sha256(raw).hexdigest() == (
        "51aa24224d5a61e9224afd3a6d384e6da79f45a59fe9fdf56cdb4e65988601ee"
    )
    assert hashlib.sha256(raw).hexdigest() != REVIEWED_CODE_DIGESTS["linux_service_ipc.py"]
