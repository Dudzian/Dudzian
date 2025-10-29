from __future__ import annotations

from pathlib import Path

import pytest

from bot_core.runtime import bootstrap
from bot_core.security.fingerprint_lock import write_fingerprint_lock


class _StubGenerator:
    def __init__(self, fingerprint: str) -> None:
        self._fingerprint = fingerprint

    def generate_fingerprint(self) -> str:
        return self._fingerprint


def _patch_generator(monkeypatch: pytest.MonkeyPatch, fingerprint: str) -> None:
    monkeypatch.setattr(
        bootstrap,
        "DeviceFingerprintGenerator",
        lambda: _StubGenerator(fingerprint),
    )


def test_hardware_enforcement_accepts_matching_lock(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    lock_path = tmp_path / "fingerprint.json"
    write_fingerprint_lock("AAAA-BBBB", path=lock_path)
    monkeypatch.setenv("BOT_CORE_FINGERPRINT_LOCK", str(lock_path))
    _patch_generator(monkeypatch, "AAAA-BBBB")

    # Should not raise
    bootstrap._enforce_installation_hardware()  # type: ignore[attr-defined]


def test_hardware_enforcement_rejects_foreign_host(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    lock_path = tmp_path / "fingerprint.json"
    write_fingerprint_lock("AAAA-BBBB", path=lock_path)
    monkeypatch.setenv("BOT_CORE_FINGERPRINT_LOCK", str(lock_path))
    _patch_generator(monkeypatch, "CCCC-DDDD")

    recorded: dict[str, object] = {}

    def _record_alert(message: str, **kwargs: object) -> None:  # pragma: no cover - helper
        recorded["message"] = message
        recorded["kwargs"] = kwargs

    monkeypatch.setattr(bootstrap, "emit_alert", _record_alert)

    with pytest.raises(RuntimeError):
        bootstrap._enforce_installation_hardware()  # type: ignore[attr-defined]

    assert "Fingerprint" in str(recorded.get("message", ""))
