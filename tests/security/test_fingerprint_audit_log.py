import json
from pathlib import Path

from bot_core.security.fingerprint import append_fingerprint_audit


def test_append_fingerprint_audit_writes_entry(tmp_path: Path) -> None:
    log_path = tmp_path / "logs" / "security_admin.log"
    append_fingerprint_audit(
        event="installer_run",
        fingerprint="ABC123",
        status="verified",
        key_id="key-1",
        metadata={"bundle": "core-oem", "expected_file": "fingerprint.expected.json"},
        log_path=log_path,
    )

    content = log_path.read_text("utf-8").strip().splitlines()
    assert len(content) == 1
    entry = json.loads(content[0])
    assert entry["event"] == "installer_run"
    assert entry["fingerprint"] == "ABC123"
    assert entry["status"] == "verified"
    assert entry["key_id"] == "key-1"
    assert entry["metadata"]["bundle"] == "core-oem"
    assert entry["timestamp"].endswith("Z")
