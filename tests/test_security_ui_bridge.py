import json
from pathlib import Path

from bot_core.security.fingerprint import decode_secret

from bot_core.security import ui_bridge


def test_dump_state_reads_profiles_and_license(tmp_path):
    license_path = tmp_path / "license.json"
    license_path.write_text(
        json.dumps(
            {
                "fingerprint": "ABC-123",
                "valid": {"from": "2024-01-01T00:00:00Z", "to": "2024-12-31T23:59:59Z"},
            }
        ),
        encoding="utf-8",
    )

    profiles_path = tmp_path / "profiles.json"
    profiles_path.write_text(
        json.dumps(
            [
                {"user_id": "ops", "display_name": "Ops", "roles": ["metrics.read"]},
                {"user_id": "qa", "display_name": "QA", "roles": ["metrics.write"]},
            ]
        ),
        encoding="utf-8",
    )

    state = ui_bridge.dump_state(
        license_path=str(license_path),
        profiles_path=str(profiles_path),
    )

    assert state["license"]["fingerprint"] == "ABC-123"
    assert state["license"]["status"] == "active"
    assert len(state["profiles"]) == 2
    assert {p["user_id"] for p in state["profiles"]} == {"ops", "qa"}


def test_assign_profile_updates_store_and_logs(tmp_path):
    profiles_path = tmp_path / "profiles.json"
    log_path = tmp_path / "logs" / "security.log"

    result = ui_bridge.assign_profile(
        profiles_path=str(profiles_path),
        user_id="alice",
        display_name="Alice",
        roles=["metrics.write", "metrics.read"],
        log_path=str(log_path),
        actor="admin",
    )

    assert result["status"] == "ok"
    assert result["profile"]["user_id"] == "alice"
    assert result["profile"]["roles"] == ["metrics.read", "metrics.write"]
    assert Path(result["log_path"]).exists()

    stored_profiles = json.loads(profiles_path.read_text(encoding="utf-8"))
    assert stored_profiles[0]["user_id"] == "alice"
    assert stored_profiles[0]["roles"] == ["metrics.read", "metrics.write"]

    log_entries = [json.loads(line) for line in log_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert len(log_entries) == 1
    assert "admin" in log_entries[0]["message"]


def test_assign_profile_expands_tilde(tmp_path, monkeypatch):
    fake_home = tmp_path / "home"
    fake_home.mkdir()
    profiles = fake_home / "profiles.json"
    log_file = fake_home / "audit.log"

    monkeypatch.setenv("HOME", str(fake_home))

    ui_bridge.assign_profile(
        profiles_path="~/profiles.json",
        user_id="bob",
        display_name=None,
        roles=["metrics.read"],
        log_path="~/audit.log",
        actor=None,
    )

    data = json.loads(profiles.read_text(encoding="utf-8"))
    assert data[0]["user_id"] == "bob"

    log_records = [json.loads(line) for line in log_file.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert log_records


def test_remove_profile_entry_deletes_and_logs(tmp_path):
    profiles_path = tmp_path / "profiles.json"
    profiles_path.write_text(
        json.dumps(
            [
                {"user_id": "alice", "display_name": "Alice", "roles": ["metrics.read"]},
                {"user_id": "bob", "display_name": "Bob", "roles": ["metrics.write"]},
            ]
        ),
        encoding="utf-8",
    )
    log_path = tmp_path / "admin.log"

    removed = ui_bridge.remove_profile_entry(
        profiles_path=str(profiles_path),
        user_id="bob",
        log_path=str(log_path),
        actor="qa",
    )

    assert removed["status"] == "ok"
    assert removed["removed"]["user_id"] == "bob"
    data = json.loads(profiles_path.read_text(encoding="utf-8"))
    assert len(data) == 1
    assert data[0]["user_id"] == "alice"

    log_records = [json.loads(line) for line in log_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert len(log_records) == 1
    assert "qa removed profile bob" in log_records[0]["message"]

    missing = ui_bridge.remove_profile_entry(
        profiles_path=str(profiles_path),
        user_id="charlie",
        log_path=str(log_path),
        actor="qa",
    )
    assert missing["status"] == "not_found"


def test_main_fingerprint_uses_keys_file(monkeypatch, tmp_path, capsys):
    keys_file = tmp_path / "keys.json"
    keys_file.write_text(json.dumps({"keys": {"fp-key": "hex:0011"}}), encoding="utf-8")

    captured: dict[str, object] = {}

    class FakeRecord:
        def __init__(self, dongle: str | None) -> None:
            self._dongle = dongle

        def as_dict(self) -> dict[str, object]:
            return {"fingerprint": "demo", "dongle": self._dongle}

    class FakeService:
        def __init__(self, provider: object) -> None:
            captured["provider"] = provider

        def build(self, dongle_serial: str | None = None) -> FakeRecord:
            captured["dongle"] = dongle_serial
            return FakeRecord(dongle_serial)

    def fake_provider(keys, rotation_log, purpose, interval_days):
        captured["keys"] = dict(keys)
        captured["rotation_log"] = rotation_log
        captured["purpose"] = purpose
        captured["interval"] = interval_days
        return {"keys": keys}

    monkeypatch.setattr(ui_bridge, "HardwareFingerprintService", FakeService)
    monkeypatch.setattr(ui_bridge, "build_key_provider", fake_provider)

    exit_code = ui_bridge.main(
        [
            "fingerprint",
            "--keys-file",
            str(keys_file),
            "--rotation-log",
            "var/licenses/custom_rotation.json",
            "--purpose",
            "demo",
            "--interval-days",
            "45",
            "--dongle",
            "USB-1",
        ]
    )

    assert exit_code == 0
    std = capsys.readouterr().out.strip()
    payload = json.loads(std)
    assert payload["fingerprint"] == "demo"
    assert payload["dongle"] == "USB-1"
    assert captured["keys"]["fp-key"] == decode_secret("hex:0011")
    assert captured["rotation_log"] == "var/licenses/custom_rotation.json"
    assert captured["purpose"] == "demo"
    assert captured["interval"] == 45.0
