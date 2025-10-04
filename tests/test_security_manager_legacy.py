"""Testy odporności SecretManagera na alternatywne formaty sekretów."""
from __future__ import annotations

import json

from bot_core.exchanges.base import Environment
from bot_core.security.base import SecretManager, SecretStorage


class _StubStorage(SecretStorage):
    """Minimalny magazyn zwracający z góry przygotowany sekret."""

    def __init__(self, value: str | None) -> None:
        self._value = value

    def get_secret(self, key: str) -> str | None:  # pragma: no cover - prosta implementacja
        return self._value

    def set_secret(self, key: str, value: str) -> None:  # pragma: no cover - nieużywane
        raise NotImplementedError

    def delete_secret(self, key: str) -> None:  # pragma: no cover - nieużywane
        raise NotImplementedError


def _manager_with(payload: dict[str, object]) -> SecretManager:
    storage = _StubStorage(json.dumps(payload))
    return SecretManager(storage, namespace="tests.namespace")


def test_load_exchange_credentials_accepts_legacy_api_fields() -> None:
    manager = _manager_with(
        {
            "api_key": "API123",
            "api_secret": "SECRET456",
            "permissions": ["READ", "trade"],
            "environment": "paper",
        }
    )

    creds = manager.load_exchange_credentials(
        "binance_paper_trading",
        expected_environment=Environment.PAPER,
    )

    assert creds.key_id == "API123"
    assert creds.secret == "SECRET456"
    assert set(creds.permissions) == {"read", "trade"}


def test_load_exchange_credentials_supports_keyid_alias() -> None:
    manager = _manager_with(
        {
            "keyId": "custom-id",
            "secret": "sekret",
            "environment": "paper",
        }
    )

    creds = manager.load_exchange_credentials(
        "binance_paper_trading",
        expected_environment=Environment.PAPER,
    )

    assert creds.key_id == "custom-id"
    assert creds.secret == "sekret"


def test_load_exchange_credentials_missing_environment_defaults_to_expected() -> None:
    manager = _manager_with(
        {
            "api_key": "legacy-key",
            "api_secret": "legacy-secret",
            "permissions": ["TRADE"],
        }
    )

    creds = manager.load_exchange_credentials(
        "binance_paper_trading",
        expected_environment=Environment.PAPER,
    )

    assert creds.key_id == "legacy-key"
    assert creds.secret == "legacy-secret"
    assert creds.permissions == ("trade",)
    assert creds.environment is Environment.PAPER
