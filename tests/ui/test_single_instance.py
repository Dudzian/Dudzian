from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


def _read_last_event(log_path: Path) -> dict[str, Any]:
    content = log_path.read_text(encoding="utf-8").strip().splitlines()
    assert content, "Oczekiwano wpisu w logu audytu."
    return json.loads(content[-1])


@pytest.mark.timeout(5)
def test_single_instance_event_is_logged(tmp_path: Path) -> None:
    from bot_core.security.fingerprint import report_single_instance_event

    log_path = tmp_path / "audit.log"
    lock_path = tmp_path / "app.lock"

    destination = report_single_instance_event(
        lock_path=lock_path,
        owner_pid=4242,
        owner_host="ci-host",
        owner_application="bot-trading-shell",
        fingerprint="ABC12345",
        log_path=log_path,
    )

    assert destination == log_path
    payload = _read_last_event(log_path)
    assert payload["event"] == "ui_single_instance_conflict"
    assert payload["status"] == "denied"
    assert payload["metadata"]["lock_path"] == str(lock_path)
    assert payload["metadata"]["owner_pid"] == 4242
    assert payload["metadata"]["owner_host"] == "ci-host"
    assert payload["metadata"]["owner_application"] == "bot-trading-shell"
    assert payload["fingerprint"].startswith("ABC12345")


@pytest.mark.timeout(5)
def test_cli_reports_single_instance_event(tmp_path: Path) -> None:
    log_path = tmp_path / "audit.log"
    lock_path = tmp_path / "app.lock"

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "bot_core.security.fingerprint",
            "report-single-instance",
            "--lock-path",
            str(lock_path),
            "--owner-pid",
            "1337",
            "--owner-host",
            "ci-host",
            "--owner-application",
            "bot-trading-shell",
            "--fingerprint",
            "abc123",
            "--log-path",
            str(log_path),
        ],
        capture_output=True,
        text=True,
        check=True,
    )

    assert result.returncode == 0
    assert "Traceback" not in result.stderr

    payload = _read_last_event(log_path)
    assert payload["event"] == "ui_single_instance_conflict"
    assert payload["status"] == "denied"
    assert payload["metadata"]["lock_path"] == str(lock_path)
    assert payload["metadata"]["owner_pid"] == 1337
    assert payload["metadata"]["owner_host"] == "ci-host"
    assert payload["metadata"]["owner_application"] == "bot-trading-shell"
    assert payload["fingerprint"].startswith("ABC123")


@pytest.mark.timeout(5)
def test_report_single_instance_event_normalizes_metadata(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from bot_core.security import fingerprint as fp_module

    monkeypatch.setattr(fp_module, "get_local_fingerprint", lambda: "abc123")
    monkeypatch.setattr(fp_module, "_normalize_binding_fingerprint", lambda value: f"NORM::{value}")
    monkeypatch.chdir(tmp_path)

    log_path = tmp_path / "audit.log"

    destination = fp_module.report_single_instance_event(
        lock_path="relative.lock",
        owner_pid=None,
        owner_host=" host.local ",
        owner_application=" bot-trading-shell ",
        fingerprint=None,
        log_path=log_path,
    )

    assert destination == log_path

    payload = _read_last_event(log_path)
    assert payload["metadata"]["lock_path"] == str((tmp_path / "relative.lock").resolve())
    assert payload["metadata"]["owner_host"] == "host.local"
    assert payload["metadata"]["owner_application"] == "bot-trading-shell"
    assert "owner_pid" not in payload["metadata"]
    assert payload["fingerprint"] == "NORM::abc123"
