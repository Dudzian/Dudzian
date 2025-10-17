import json

import pytest

from bot_core.security.profiles import (
    UserProfile,
    load_profiles,
    log_admin_event,
    remove_profile,
    save_profiles,
    upsert_profile,
)


def test_load_profiles_roundtrip(tmp_path):
    storage = tmp_path / "profiles.json"
    payload = [
        {
            "user_id": "ops",
            "display_name": "Operations",
            "roles": ["metrics.read", "metrics.write"],
            "updated_at": "2024-01-01T00:00:00Z",
        }
    ]
    storage.write_text(json.dumps(payload), encoding="utf-8")

    profiles = load_profiles(storage)
    assert len(profiles) == 1
    profile = profiles[0]
    assert profile.user_id == "ops"
    assert profile.display_name == "Operations"
    assert profile.roles == ("metrics.read", "metrics.write")

    # Ensure save rewrites file with canonical ordering and newline.
    output = tmp_path / "out.json"
    save_profiles(profiles, output)
    saved = json.loads(output.read_text(encoding="utf-8"))
    assert saved[0]["roles"] == ["metrics.read", "metrics.write"]
    assert output.read_text(encoding="utf-8").endswith("\n")


def test_upsert_profile_adds_and_updates(tmp_path):
    profiles: list[UserProfile] = []
    created = upsert_profile(profiles, user_id="alice", display_name="Alice", roles=["metrics.write", "metrics.read"])
    assert created.user_id == "alice"
    assert created.roles == ("metrics.read", "metrics.write")
    assert len(profiles) == 1

    updated = upsert_profile(profiles, user_id="alice", display_name="Alice Ops", roles=["metrics.read"])
    assert updated.display_name == "Alice Ops"
    assert updated.roles == ("metrics.read",)
    assert profiles[0] is updated

    with pytest.raises(ValueError):
        upsert_profile(profiles, user_id=" ")


def test_remove_profile_removes_and_returns_entry():
    profiles = [
        UserProfile(user_id="alice", display_name="Alice", roles=("metrics.read",), updated_at="ts"),
        UserProfile(user_id="bob", display_name="Bob", roles=("metrics.write",), updated_at="ts"),
    ]

    removed = remove_profile(profiles, user_id="bob")
    assert removed is not None
    assert removed.user_id == "bob"
    assert len(profiles) == 1
    assert profiles[0].user_id == "alice"

    missing = remove_profile(profiles, user_id="carol")
    assert missing is None

    with pytest.raises(ValueError):
        remove_profile(profiles, user_id=" ")


def test_log_admin_event_appends_jsonl(tmp_path):
    log_path = tmp_path / "logs" / "admin.log"
    log_admin_event("actor updated", log_path=log_path)
    log_admin_event("second entry", log_path=log_path)

    lines = log_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 2
    entry = json.loads(lines[0])
    assert entry["message"] == "actor updated"
    assert "timestamp" in entry


def test_load_profiles_with_tilde(tmp_path, monkeypatch):
    # Simulate home directory expansion when reading profiles.
    fake_home = tmp_path / "home"
    fake_home.mkdir()
    storage = fake_home / "profiles.json"
    storage.write_text("[]", encoding="utf-8")

    monkeypatch.setenv("HOME", str(fake_home))
    profiles = load_profiles("~/profiles.json")
    assert profiles == []
