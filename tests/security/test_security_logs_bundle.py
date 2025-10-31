import json
from pathlib import Path

import pytest

from bot_core.security.logs import export_security_bundle
from bot_core.security import ui_bridge


@pytest.fixture()
def audit_key(monkeypatch):
    monkeypatch.setenv("BOT_CORE_SECURITY_AUDIT_KEY", "hex:" + ("11" * 32))
    monkeypatch.setenv("BOT_CORE_SECURITY_AUDIT_KEY_ID", "ops-test")


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    payload = "\n".join(json.dumps(item, ensure_ascii=False) for item in rows)
    path.write_text(payload + "\n", encoding="utf-8")


def test_export_security_bundle_creates_signed_payload(tmp_path, audit_key):
    audit_log = tmp_path / "security_admin.log"
    alerts_log = tmp_path / "security_alerts.log"
    extra_log = tmp_path / "controller.log"

    _write_jsonl(
        audit_log,
        [
            {"category": "audit", "message": "user login", "severity": "info"},
            {"category": "audit", "message": "updated profile", "severity": "info"},
        ],
    )
    _write_jsonl(
        alerts_log,
        [
            {"category": "security", "message": "anomaly detected", "severity": 2},
        ],
    )
    extra_log.write_text("first\nsecond\nthird\n", encoding="utf-8")

    destination = tmp_path / "exports"
    result = export_security_bundle(
        audit_path=str(audit_log),
        alerts_path=str(alerts_log),
        destination_dir=str(destination),
        include_logs=[str(extra_log)],
        key_source=None,
        key_id="ops-test",
        metadata={"operator": "qa"},
    )

    assert result.bundle_path.exists()
    bundle_payload = json.loads(result.bundle_path.read_text(encoding="utf-8"))
    assert bundle_payload["bundle"]["audit"]["entries"][-1]["message"] == "updated profile"
    assert bundle_payload["bundle"]["alerts"]["entries"][0]["message"] == "anomaly detected"
    assert bundle_payload["bundle"]["logs"][str(extra_log.resolve())]["line_count"] == 3
    assert bundle_payload["signature"]["key_id"] == "ops-test"


def test_ui_bridge_export_security_bundle(tmp_path, monkeypatch, audit_key):
    audit_log = tmp_path / "admin.jsonl"
    alerts_log = tmp_path / "alerts.jsonl"
    _write_jsonl(audit_log, [{"category": "audit", "message": "ok"}])
    _write_jsonl(alerts_log, [{"category": "alert", "message": "ping"}])

    args = [
        "export-security-bundle",
        "--audit-path",
        str(audit_log),
        "--alerts-path",
        str(alerts_log),
        "--output-dir",
        str(tmp_path / "bundle.json"),
        "--include-log",
        str(audit_log),
    ]

    exit_code = ui_bridge.main(args)
    assert exit_code == 0
    payload = json.loads((tmp_path / "bundle.json").read_text(encoding="utf-8"))
    assert "bundle" in payload
    assert payload["bundle"]["audit"]["entries"][0]["message"] == "ok"
    assert payload["bundle"]["alerts"]["entries"][0]["message"] == "ping"
