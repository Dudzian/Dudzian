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


def test_runtime_controller_import_without_security_chain() -> None:
    script = textwrap.dedent(
        """
        import importlib.abc
        import pathlib
        import sys

        repo_root = pathlib.Path.cwd()
        sys.path.insert(0, str(repo_root))

        class _SecurityBlocker(importlib.abc.MetaPathFinder):
            def find_spec(self, fullname, path, target=None):
                if fullname == "bot_core.security" or fullname.startswith("bot_core.security."):
                    raise ImportError(f"blocked {fullname}")
                return None

        sys.meta_path.insert(0, _SecurityBlocker())

        import bot_core.runtime.controller as controller

        assert hasattr(controller, "TradingController")
        assert "bot_core.security" not in sys.modules
        """
    )

    result = _run_python(script)
    assert result.returncode == 0, result.stderr
