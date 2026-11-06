"""Materiał do review: poprawka działa wyłącznie na kopii w katalogu testu.

FRESHNESS_REMEDIATION_SOURCE=baseline uruchamia te same asercje przeciwko
niezmienionemu verifierowi, aby odtworzyć usterki. Domyślny candidate sprawdza
proponowany diff; nie publikuje wydania ani nie podmienia żadnego pinu.
"""

from __future__ import annotations

import hashlib
import importlib.util
import os
import subprocess
import sys
from pathlib import Path
from types import ModuleType

from deployment.freshness_application import REVIEWED_SUBSTRATE_DIGESTS

SOURCE = "bot_core/freshness_semantic_verifier.py"
ROOT = Path(__file__).resolve().parents[2]
PATCH = ROOT / "tests/fixtures/freshness_verifier_remediation_review.patch"


def load_verifier(tmp_path: Path) -> ModuleType:
    """Załaduj dokładny baseline albo proponowany diff w prywatnym katalogu."""
    source = ROOT / SOURCE
    approved = REVIEWED_SUBSTRATE_DIGESTS[SOURCE]
    if hashlib.sha256(source.read_bytes()).hexdigest() != approved:
        raise ValueError("Odmowa: źródło bazowe nie odpowiada zachowanemu zatwierdzeniu")
    selection = os.environ.get("FRESHNESS_REMEDIATION_SOURCE", "candidate")
    if selection not in {"baseline", "candidate"}:
        raise ValueError("Odmowa: nieznany wariant testu remediacji")
    if selection == "baseline":
        from bot_core import freshness_semantic_verifier

        return freshness_semantic_verifier
    target = tmp_path / SOURCE
    target.parent.mkdir(parents=True)
    target.write_bytes(source.read_bytes())
    for options in (("--check",), ()):
        subprocess.run(
            ["git", "apply", "--unidiff-zero", *options, str(PATCH)],
            cwd=tmp_path,
            capture_output=True,
            text=True,
            check=True,
        )
    name = f"bot_core._freshness_remediation_review_{id(tmp_path)}"
    spec = importlib.util.spec_from_file_location(name, target)
    if spec is None or spec.loader is None:
        raise RuntimeError("Nie można załadować tymczasowego materiału do review")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module
