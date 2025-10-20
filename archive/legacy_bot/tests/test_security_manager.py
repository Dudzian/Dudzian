# tests/test_security_manager.py
# -*- coding: utf-8 -*-
"""
Unit tests for security_manager.py.
"""
import pytest
import json
from pathlib import Path
from KryptoLowca.security_manager import SecurityManager, SecurityError
from cryptography.fernet import InvalidToken

@pytest.fixture
def security_manager(tmp_path):
    key_file = tmp_path / "keys.enc"
    return SecurityManager(key_file=str(key_file))

def test_save_and_load_keys(security_manager, tmp_path):
    keys = {
        "testnet": {"key": "test_key", "secret": "test_secret"},
        "live": {"key": "live_key", "secret": "live_secret"}
    }
    password = "password123"
    
    security_manager.save_encrypted_keys(keys, password)
    assert security_manager.key_file.exists()
    
    loaded_keys = security_manager.load_encrypted_keys(password)
    assert loaded_keys == keys

def test_invalid_password(security_manager):
    keys = {"testnet": {"key": "test_key", "secret": "test_secret"}}
    security_manager.save_encrypted_keys(keys, "password123")
    
    with pytest.raises(SecurityError, match="Invalid password"):
        security_manager.load_encrypted_keys("wrong_password")

def test_missing_key_file(security_manager):
    with pytest.raises(SecurityError, match="Key file .* not found"):
        security_manager.load_encrypted_keys("password123")

def test_invalid_keys(security_manager):
    with pytest.raises(SecurityError, match="Keys must be a non-empty dictionary"):
        security_manager.save_encrypted_keys({}, "password123")

def test_invalid_password_type(security_manager):
    with pytest.raises(SecurityError, match="Password must be a non-empty string"):
        security_manager.save_encrypted_keys({"testnet": {"key": "k", "secret": "s"}}, "")