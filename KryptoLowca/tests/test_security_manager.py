# tests/test_security_manager.py
# -*- coding: utf-8 -*-
"""
Unit tests for security_manager.py.
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

import pytest

from KryptoLowca.managers.security_manager import SecurityError, SecurityManager


@pytest.fixture
def security_manager(tmp_path: Path) -> SecurityManager:
    key_file = tmp_path / "keys.enc"
    return SecurityManager(key_file=str(key_file))


def test_save_and_load_keys(security_manager: SecurityManager) -> None:
    keys = {
        "testnet": {"key": "test_key", "secret": "test_secret"},
        "live": {"key": "live_key", "secret": "live_secret"},
    }
    password = "password123"

    security_manager.save_encrypted_keys(keys, password)
    assert security_manager.key_file.exists()

    loaded_keys = security_manager.load_encrypted_keys(password)
    assert loaded_keys == keys


def test_invalid_password(security_manager: SecurityManager) -> None:
    keys = {"testnet": {"key": "test_key", "secret": "test_secret"}}
    security_manager.save_encrypted_keys(keys, "password123")

    with pytest.raises(SecurityError, match="Invalid password"):
        security_manager.load_encrypted_keys("wrong_password")


def test_missing_key_file(security_manager: SecurityManager) -> None:
    with pytest.raises(SecurityError, match="Key file .* not found"):
        security_manager.load_encrypted_keys("password123")


def test_invalid_keys(security_manager: SecurityManager) -> None:
    with pytest.raises(SecurityError, match="Keys must be a non-empty dictionary"):
        security_manager.save_encrypted_keys({}, "password123")


def test_invalid_password_type(security_manager: SecurityManager) -> None:
    with pytest.raises(SecurityError, match="Password must be a non-empty string"):
        security_manager.save_encrypted_keys({"testnet": {"key": "k", "secret": "s"}}, "")


def test_audit_callback_records_events(security_manager: SecurityManager) -> None:
    events: List[Tuple[str, dict]] = []

    def _callback(action: str, payload: dict) -> None:
        events.append((action, payload))

    security_manager.register_audit_callback(_callback)

    keys = {
        "testnet": {"key": "T" * 32, "secret": "S" * 32},
        "live": {"key": "L" * 32, "secret": "Z" * 32},
    }
    password = "audit-pass"

    security_manager.save_encrypted_keys(keys, password)
    loaded = security_manager.load_encrypted_keys(password)

    assert loaded == keys
    actions = [action for action, _ in events]
    assert "encrypt_keys" in actions
    assert "decrypt_keys" in actions
