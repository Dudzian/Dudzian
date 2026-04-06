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


def test_runtime_controller_import_without_ai_models_signing_module() -> None:
    script = textwrap.dedent(
        """
        import importlib.abc
        import pathlib
        import sys
        import types

        repo_root = pathlib.Path.cwd()
        sys.path.insert(0, str(repo_root))
        security_module = types.ModuleType("bot_core.security")
        security_module.__path__ = []
        guards_module = types.ModuleType("bot_core.security.guards")
        guards_module.get_capability_guard = lambda: None
        sys.modules.setdefault("bot_core.security", security_module)
        sys.modules.setdefault("bot_core.security.guards", guards_module)

        class _SecurityBlocker(importlib.abc.MetaPathFinder):
            def find_spec(self, fullname, path, target=None):
                if fullname == "bot_core.security.signing":
                    raise ModuleNotFoundError(f"blocked {fullname}", name=fullname)
                return None

        sys.meta_path.insert(0, _SecurityBlocker())

        import bot_core.runtime.controller as controller

        assert hasattr(controller, "TradingController")
        assert "bot_core.security.signing" not in sys.modules
        """
    )

    result = _run_python(script)
    assert result.returncode == 0, result.stderr


def test_models_generate_signing_path_requires_security_module_at_runtime() -> None:
    script = textwrap.dedent(
        """
        import importlib.abc
        import pathlib
        import sys
        import tempfile
        from datetime import datetime, timezone

        repo_root = pathlib.Path.cwd()
        sys.path.insert(0, str(repo_root))

        class _SecurityBlocker(importlib.abc.MetaPathFinder):
            def find_spec(self, fullname, path, target=None):
                if fullname == "bot_core.security.signing":
                    raise ModuleNotFoundError(f"blocked {fullname}", name=fullname)
                return None

        sys.meta_path.insert(0, _SecurityBlocker())

        from bot_core.ai.models import ModelArtifact, generate_model_artifact_bundle

        artifact = ModelArtifact(
            feature_names=("f1",),
            model_state={"weights": [0.0]},
            trained_at=datetime.now(timezone.utc),
            metrics={"summary": {"mae": 0.0, "rmse": 0.0}},
            metadata={"model_version": "test-1", "symbol": "BTCUSDT"},
            target_scale=1.0,
            training_rows=1,
            validation_rows=1,
            test_rows=1,
            feature_scalers={"f1": (0.0, 1.0)},
        )

        with tempfile.TemporaryDirectory() as output_dir:
            try:
                generate_model_artifact_bundle(
                    artifact,
                    output_dir,
                    name="model",
                    signing_key=b"blocked-key",
                )
            except RuntimeError as exc:
                assert str(exc) == "HMAC signing path requires module 'bot_core.security.signing'"
            else:
                raise AssertionError("Expected RuntimeError when signing helpers are unavailable")
        """
    )

    result = _run_python(script)
    assert result.returncode == 0, result.stderr


def test_models_load_signing_path_requires_security_module_at_runtime() -> None:
    script = textwrap.dedent(
        """
        import importlib.abc
        import pathlib
        import sys
        import tempfile
        from datetime import datetime, timezone

        repo_root = pathlib.Path.cwd()
        sys.path.insert(0, str(repo_root))

        from bot_core.ai.models import (
            ModelArtifact,
            generate_model_artifact_bundle,
            load_model_artifact_bundle,
        )

        artifact = ModelArtifact(
            feature_names=("f1",),
            model_state={"weights": [0.0]},
            trained_at=datetime.now(timezone.utc),
            metrics={"summary": {"mae": 0.0, "rmse": 0.0}},
            metadata={"model_version": "test-1", "symbol": "BTCUSDT"},
            target_scale=1.0,
            training_rows=1,
            validation_rows=1,
            test_rows=1,
            feature_scalers={"f1": (0.0, 1.0)},
        )

        class _SecurityBlocker(importlib.abc.MetaPathFinder):
            def find_spec(self, fullname, path, target=None):
                if fullname == "bot_core.security.signing":
                    raise ModuleNotFoundError(f"blocked {fullname}", name=fullname)
                return None

        with tempfile.TemporaryDirectory() as output_dir:
            generate_model_artifact_bundle(
                artifact,
                output_dir,
                name="model",
                signing_key=b"bundle-key",
                signing_key_id="primary",
            )
            sys.meta_path.insert(0, _SecurityBlocker())
            sys.modules.pop("bot_core.security.signing", None)
            try:
                load_model_artifact_bundle(
                    output_dir,
                    expected_artifact="model.json",
                    signing_keys={"primary": b"bundle-key"},
                )
            except RuntimeError as exc:
                assert str(exc) == "HMAC signing path requires module 'bot_core.security.signing'"
            else:
                raise AssertionError("Expected RuntimeError when signing helpers are unavailable")
        """
    )

    result = _run_python(script)
    assert result.returncode == 0, result.stderr
