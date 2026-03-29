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


def test_ai_health_import_does_not_preload_inference() -> None:
    script = textwrap.dedent(
        """
        import importlib.abc
        import pathlib
        import sys

        repo_root = pathlib.Path.cwd()
        sys.path.insert(0, str(repo_root))

        class _InferenceBlocker(importlib.abc.MetaPathFinder):
            def find_spec(self, fullname, path, target=None):
                if fullname == "bot_core.ai.inference":
                    raise ImportError("blocked bot_core.ai.inference")
                return None

        sys.meta_path.insert(0, _InferenceBlocker())

        import bot_core.ai.health as health

        assert health.ModelHealthMonitor.__name__ == "ModelHealthMonitor"
        assert "bot_core.ai.inference" not in sys.modules

        import bot_core.ai as ai

        assert ai.ModelHealthMonitor is health.ModelHealthMonitor
        assert "DecisionModelInference" in ai.__all__

        try:
            _ = ai.DecisionModelInference
        except ImportError as exc:
            assert "blocked bot_core.ai.inference" in str(exc)
        else:
            raise AssertionError("DecisionModelInference import should be blocked in this probe")
        """
    )

    result = _run_python(script)
    assert result.returncode == 0, result.stderr
