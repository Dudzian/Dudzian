from __future__ import annotations

import subprocess
import sys
import textwrap
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]


def _run_python(script: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-c", script],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
    )


def test_runtime_package_import_does_not_expand_security_chain() -> None:
    script = textwrap.dedent(
        """
        import importlib.abc
        import pathlib
        import sys

        repo_root = pathlib.Path.cwd()
        sys.path.insert(0, str(repo_root))

        security_hits = []

        class _SecurityProbe(importlib.abc.MetaPathFinder):
            def find_spec(self, fullname, path, target=None):
                if fullname == "bot_core.security" or fullname.startswith("bot_core.security."):
                    security_hits.append(fullname)
                    raise ImportError(f"blocked {fullname}")
                return None

        sys.meta_path.insert(0, _SecurityProbe())

        import bot_core
        baseline_hits = len(security_hits)

        import bot_core.runtime as runtime

        assert hasattr(runtime, "__all__")
        runtime_hits = len(security_hits) - baseline_hits
        assert runtime_hits == 0, security_hits
        """
    )

    result = _run_python(script)
    assert result.returncode == 0, result.stderr
