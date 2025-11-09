from __future__ import annotations

import json
from pathlib import Path

import pytest

from probe_keyring import HwidValidationError, install_hook_main


def _write_expected_file(tmp_path: Path, fingerprint: str) -> Path:
    path = tmp_path / "expected.json"
    path.write_text(json.dumps({"fingerprint": fingerprint}), encoding="utf-8")
    return path


def test_install_hook_main_accepts_fake_fingerprint(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    fingerprint = "stub-fingerprint"
    expected = _write_expected_file(tmp_path, fingerprint)

    monkeypatch.setenv("KBOT_FAKE_FINGERPRINT", fingerprint)
    try:
        result = install_hook_main(str(expected))
    finally:
        monkeypatch.delenv("KBOT_FAKE_FINGERPRINT", raising=False)

    assert result == fingerprint


def test_install_hook_main_detects_mismatch(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    expected = _write_expected_file(tmp_path, "expected")

    monkeypatch.setenv("KBOT_FAKE_FINGERPRINT", "different")
    try:
        with pytest.raises(HwidValidationError):
            install_hook_main(str(expected))
    finally:
        monkeypatch.delenv("KBOT_FAKE_FINGERPRINT", raising=False)
