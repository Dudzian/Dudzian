import json
from pathlib import Path

from bot_core.security.fingerprint import decode_secret

from bot_core.security import ui_bridge
from bot_core.security.signing import build_hmac_signature


LICENSE_KEY = bytes.fromhex("11" * 32)
FINGERPRINT_KEY = bytes.fromhex("22" * 32)
REVOCATION_KEY = bytes.fromhex("33" * 32)


def _write_signed_license(base: Path) -> dict[str, Path]:
    fingerprint_payload = {
        "version": 1,
        "collected_at": "2024-03-01T12:00:00Z",
        "components": {},
        "component_digests": {},
        "fingerprint": {"algorithm": "sha256", "value": "SIGNED-ABC"},
    }
    fingerprint_signature = build_hmac_signature(
        fingerprint_payload,
        key=FINGERPRINT_KEY,
        algorithm="HMAC-SHA384",
        key_id="fp-test",
    )
    license_payload = {
        "schema": "core.oem.license",
        "schema_version": "1.0",
        "issued_at": "2024-03-01T00:00:00Z",
        "expires_at": "2040-03-01T00:00:00Z",
        "issuer": "qa",
        "profile": "paper",
        "license_id": "signed-lic",
        "fingerprint": fingerprint_payload["fingerprint"],
        "fingerprint_payload": fingerprint_payload,
        "fingerprint_signature": fingerprint_signature,
    }
    license_signature = build_hmac_signature(
        license_payload,
        key=LICENSE_KEY,
        algorithm="HMAC-SHA384",
        key_id="lic-test",
    )
    license_document = {"payload": license_payload, "signature": license_signature}
    fingerprint_document = {"payload": fingerprint_payload, "signature": fingerprint_signature}

    license_path = base / "license.json"
    fingerprint_path = base / "fingerprint.json"
    license_path.write_text(json.dumps(license_document, ensure_ascii=False), encoding="utf-8")
    fingerprint_path.write_text(
        json.dumps(fingerprint_document, ensure_ascii=False), encoding="utf-8"
    )

    license_keys = base / "license_keys.json"
    fingerprint_keys = base / "fingerprint_keys.json"
    license_keys.write_text(
        json.dumps({"keys": {"lic-test": f"hex:{LICENSE_KEY.hex()}"}}, ensure_ascii=False),
        encoding="utf-8",
    )
    fingerprint_keys.write_text(
        json.dumps({"keys": {"fp-test": f"hex:{FINGERPRINT_KEY.hex()}"}}, ensure_ascii=False),
        encoding="utf-8",
    )

    revocation_payload = {
        "revoked": [{"license_id": "other"}],
        "generated_at": "2024-03-01T12:00:00Z",
    }
    revocation_signature = build_hmac_signature(
        revocation_payload,
        key=REVOCATION_KEY,
        algorithm="HMAC-SHA384",
        key_id="rev-test",
    )
    revocation_document = {"payload": revocation_payload, "signature": revocation_signature}
    revocation_path = base / "revocations.json"
    revocation_path.write_text(json.dumps(revocation_document, ensure_ascii=False), encoding="utf-8")

    revocation_keys = base / "revocation_keys.json"
    revocation_keys.write_text(
        json.dumps({"keys": {"rev-test": f"hex:{REVOCATION_KEY.hex()}"}}, ensure_ascii=False),
        encoding="utf-8",
    )

    return {
        "license": license_path,
        "fingerprint": fingerprint_path,
        "license_keys": license_keys,
        "fingerprint_keys": fingerprint_keys,
        "revocation": revocation_path,
        "revocation_keys": revocation_keys,
    }


def test_dump_state_reads_profiles_and_license(tmp_path):
    license_path = tmp_path / "license.json"
    license_payload = {
        "schema": "core.oem.license",
        "schema_version": "1.0",
        "issued_at": "2024-01-01T00:00:00Z",
        "expires_at": "2040-12-31T23:59:59Z",
        "issuer": "qa",
        "profile": "paper",
        "license_id": "ui-demo",
        "fingerprint": {"algorithm": "sha256", "value": "ABC-123"},
    }
    license_document = {
        "payload": license_payload,
        "signature": {"algorithm": "HMAC-SHA384", "value": "dummy", "key_id": "lic"},
    }
    license_path.write_text(json.dumps(license_document, ensure_ascii=False), encoding="utf-8")

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
    assert state["license"]["profile"] == "paper"
    assert state["license"]["issuer"] == "qa"
    assert state["license"]["schema"] == "core.oem.license"
    assert state["license"]["schema_version"] == "1.0"
    assert state["license"]["license_id"] == "ui-demo"
    assert state["license"]["revocation_status"] == "skipped"
    assert state["license"]["revocation_checked"] is False
    assert state["license"]["revocation_reason"] is None
    assert state["license"]["revocation_revoked_at"] is None
    assert state["license"].get("revocation_actor") is None
    assert len(state["profiles"]) == 2
    assert {p["user_id"] for p in state["profiles"]} == {"ops", "qa"}


def test_dump_state_with_keys_validates_signatures(tmp_path):
    artifacts = _write_signed_license(tmp_path)
    profiles_path = tmp_path / "profiles.json"
    profiles_path.write_text("[]", encoding="utf-8")

    state = ui_bridge.dump_state(
        license_path=str(artifacts["license"]),
        profiles_path=str(profiles_path),
        fingerprint_path=str(artifacts["fingerprint"]),
        license_keys_path=str(artifacts["license_keys"]),
        fingerprint_keys_path=str(artifacts["fingerprint_keys"]),
        revocation_path=str(artifacts["revocation"]),
        revocation_keys_path=str(artifacts["revocation_keys"]),
        revocation_signature_required=True,
    )

    assert state["license"]["status"] == "active"
    assert state["license"]["fingerprint"] == "SIGNED-ABC"
    assert state["license"].get("license_key_id") == "lic-test"
    assert state["license"].get("fingerprint_key_id") == "fp-test"
    assert state["license"].get("revocation_key_id") == "rev-test"
    assert state["license"]["profile"] == "paper"
    assert state["license"]["issuer"] == "qa"
    assert state["license"]["schema"] == "core.oem.license"
    assert state["license"]["schema_version"] == "1.0"
    assert state["license"]["license_id"] == "signed-lic"
    assert state["license"]["revocation_status"] == "clear"
    assert state["license"]["revocation_checked"] is True
    assert state["license"]["revocation_reason"] is None
    assert state["license"]["revocation_revoked_at"] is None
    assert state["license"].get("revocation_actor") is None
    assert state["license"]["errors"] == []


def test_dump_state_reports_invalid_on_corrupted_signature(tmp_path):
    artifacts = _write_signed_license(tmp_path)
    document = json.loads(artifacts["license"].read_text(encoding="utf-8"))
    document["signature"]["value"] = "tampered"
    artifacts["license"].write_text(json.dumps(document, ensure_ascii=False), encoding="utf-8")
    profiles_path = tmp_path / "profiles.json"
    profiles_path.write_text("[]", encoding="utf-8")

    state = ui_bridge.dump_state(
        license_path=str(artifacts["license"]),
        profiles_path=str(profiles_path),
        fingerprint_path=str(artifacts["fingerprint"]),
        license_keys_path=str(artifacts["license_keys"]),
        fingerprint_keys_path=str(artifacts["fingerprint_keys"]),
        revocation_path=str(artifacts["revocation"]),
        revocation_keys_path=str(artifacts["revocation_keys"]),
        revocation_signature_required=True,
    )

    assert state["license"]["status"] == "invalid"
    assert any("HMAC" in msg for msg in state["license"]["errors"])
    assert state["license"]["revocation_reason"] is None
    assert state["license"]["revocation_revoked_at"] is None
    assert state["license"].get("revocation_actor") is None


def test_dump_state_requires_revocation_keys(tmp_path):
    artifacts = _write_signed_license(tmp_path)
    profiles_path = tmp_path / "profiles.json"
    profiles_path.write_text("[]", encoding="utf-8")

    state = ui_bridge.dump_state(
        license_path=str(artifacts["license"]),
        profiles_path=str(profiles_path),
        fingerprint_path=str(artifacts["fingerprint"]),
        license_keys_path=str(artifacts["license_keys"]),
        fingerprint_keys_path=str(artifacts["fingerprint_keys"]),
        revocation_path=str(artifacts["revocation"]),
        revocation_signature_required=True,
    )

    assert state["license"]["status"] == "invalid"
    assert any("kluczy HMAC" in msg for msg in state["license"]["errors"])
    assert state["license"]["revocation_reason"] is None
    assert state["license"]["revocation_revoked_at"] is None
    assert state["license"].get("revocation_actor") is None


def test_dump_state_reports_revocation_reason(tmp_path):
    artifacts = _write_signed_license(tmp_path)
    revocation_doc = json.loads(artifacts["revocation"].read_text(encoding="utf-8"))
    revocation_doc["payload"]["revoked"] = [
        {
            "license_id": "signed-lic",
            "reason": "Wniosek klienta",
            "revoked_at": "2024-07-02T14:15:00Z",
            "actor": "Compliance Ops",
        }
    ]
    revocation_doc["signature"] = build_hmac_signature(
        revocation_doc["payload"],
        key=REVOCATION_KEY,
        algorithm="HMAC-SHA384",
        key_id=revocation_doc["signature"]["key_id"],
    )
    artifacts["revocation"].write_text(
        json.dumps(revocation_doc, ensure_ascii=False),
        encoding="utf-8",
    )
    profiles_path = tmp_path / "profiles.json"
    profiles_path.write_text("[]", encoding="utf-8")

    state = ui_bridge.dump_state(
        license_path=str(artifacts["license"]),
        profiles_path=str(profiles_path),
        fingerprint_path=str(artifacts["fingerprint"]),
        license_keys_path=str(artifacts["license_keys"]),
        fingerprint_keys_path=str(artifacts["fingerprint_keys"]),
        revocation_path=str(artifacts["revocation"]),
        revocation_keys_path=str(artifacts["revocation_keys"]),
        revocation_signature_required=True,
    )

    assert state["license"]["status"] == "invalid"
    assert state["license"]["revocation_status"] == "revoked"
    assert state["license"]["revocation_reason"] == "Wniosek klienta"
    assert state["license"]["revocation_revoked_at"] == "2024-07-02T14:15:00+00:00"
    assert state["license"]["revocation_actor"] == "Compliance Ops"


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
