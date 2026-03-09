from __future__ import annotations

import importlib.util
import shutil
import subprocess
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = REPO_ROOT / "bot_core" / "security" / "catalog_signatures.py"


def _load_catalog_signatures_module():
    spec = importlib.util.spec_from_file_location("test_catalog_signatures_module", MODULE_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load module from {MODULE_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_openssl_fallback_reports_missing_binary(monkeypatch: pytest.MonkeyPatch) -> None:
    module = _load_catalog_signatures_module()

    monkeypatch.setattr(module, "ed25519", None)
    monkeypatch.setattr(module, "serialization", None)

    def _missing(*_args, **_kwargs):
        raise FileNotFoundError("openssl not found")

    monkeypatch.setattr(module.subprocess, "run", _missing)

    errors = module.verify_catalog_signature_file(
        REPO_ROOT / "config/marketplace/catalog.json",
        hmac_key=(REPO_ROOT / "config/marketplace/keys/dev-hmac.key").read_bytes().strip(),
        ed25519_key=(REPO_ROOT / "config/marketplace/keys/dev-presets-ed25519.pub")
        .read_bytes()
        .strip(),
    )

    assert any("brak narzędzia 'openssl'" in error for error in errors)


def test_openssl_fallback_uses_readable_temp_paths(monkeypatch: pytest.MonkeyPatch) -> None:
    module = _load_catalog_signatures_module()

    monkeypatch.setattr(module, "ed25519", None)
    monkeypatch.setattr(module, "serialization", None)

    def _fake_run(command, **_kwargs):
        inkey_path = Path(command[command.index("-inkey") + 1])
        sig_path = Path(command[command.index("-sigfile") + 1])
        payload_path = Path(command[command.index("-in") + 1])
        assert inkey_path.exists()
        assert sig_path.exists()
        assert payload_path.exists()
        assert inkey_path.read_bytes().startswith(b"-----BEGIN PUBLIC KEY-----")
        assert len(sig_path.read_bytes()) > 0
        assert len(payload_path.read_bytes()) > 0
        return subprocess.CompletedProcess(
            command, 0, stdout="Signature Verified Successfully", stderr=""
        )

    monkeypatch.setattr(module.subprocess, "run", _fake_run)

    errors = module.verify_catalog_signature_file(
        REPO_ROOT / "config/marketplace/catalog.json",
        hmac_key=(REPO_ROOT / "config/marketplace/keys/dev-hmac.key").read_bytes().strip(),
        ed25519_key=(REPO_ROOT / "config/marketplace/keys/dev-presets-ed25519.pub")
        .read_bytes()
        .strip(),
    )

    assert errors == []


def test_openssl_fallback_reports_process_details(monkeypatch: pytest.MonkeyPatch) -> None:
    module = _load_catalog_signatures_module()

    monkeypatch.setattr(module, "ed25519", None)
    monkeypatch.setattr(module, "serialization", None)

    def _failing_run(command, **_kwargs):
        return subprocess.CompletedProcess(command, 1, stdout="", stderr="unsupported option")

    monkeypatch.setattr(module.subprocess, "run", _failing_run)

    errors = module.verify_catalog_signature_file(
        REPO_ROOT / "config/marketplace/catalog.json",
        hmac_key=(REPO_ROOT / "config/marketplace/keys/dev-hmac.key").read_bytes().strip(),
        ed25519_key=(REPO_ROOT / "config/marketplace/keys/dev-presets-ed25519.pub")
        .read_bytes()
        .strip(),
    )

    assert any("unsupported option" in error for error in errors)


def test_openssl_fallback_end_to_end_with_real_binary(monkeypatch: pytest.MonkeyPatch) -> None:
    module = _load_catalog_signatures_module()

    if shutil.which("openssl") is None:
        pytest.skip("openssl is not available in PATH")

    monkeypatch.setattr(module, "ed25519", None)
    monkeypatch.setattr(module, "serialization", None)

    errors = module.verify_catalog_signature_file(
        REPO_ROOT / "config/marketplace/catalog.json",
        hmac_key=(REPO_ROOT / "config/marketplace/keys/dev-hmac.key").read_bytes().strip(),
        ed25519_key=(REPO_ROOT / "config/marketplace/keys/dev-presets-ed25519.pub")
        .read_bytes()
        .strip(),
    )

    assert errors == []
