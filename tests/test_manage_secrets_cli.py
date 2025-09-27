"""Testy narzędzia CLI do zarządzania sekretami."""
from __future__ import annotations

import sys
import types
from pathlib import Path

import pytest

sys.path.append(str(Path(__file__).resolve().parents[1]))

from scripts import manage_secrets


class _InMemoryKeyring:
    """Prosty backend keyring używany do testowania CLI."""

    class errors:
        class PasswordDeleteError(Exception):
            """Wyjątek zgodny z interfejsem biblioteki keyring."""

    def __init__(self) -> None:
        self._store: dict[tuple[str, str], str] = {}

    def get_password(self, service: str, username: str) -> str | None:
        return self._store.get((service, username))

    def set_password(self, service: str, username: str, password: str) -> None:
        self._store[(service, username)] = password

    def delete_password(self, service: str, username: str) -> None:
        try:
            del self._store[(service, username)]
        except KeyError as exc:  # pragma: no cover - zachowanie zgodne z biblioteką keyring
            raise self.errors.PasswordDeleteError(str(exc)) from exc


@pytest.fixture(autouse=True)
def fake_keyring() -> types.ModuleType:
    """Podmienia moduł ``keyring`` na implementację pamięciową."""

    module = types.ModuleType("keyring")
    backend = _InMemoryKeyring()
    module.get_password = backend.get_password
    module.set_password = backend.set_password
    module.delete_password = backend.delete_password
    module.errors = backend.errors
    sys.modules["keyring"] = module
    yield module
    sys.modules.pop("keyring", None)


@pytest.fixture(autouse=True)
def fake_desktop_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    """Wymusza wariant desktopowy create_default_secret_storage."""

    monkeypatch.setenv("DISPLAY", ":1")
    monkeypatch.setenv("DBUS_SESSION_BUS_ADDRESS", "session")
    monkeypatch.setattr("platform.system", lambda: "Linux")


def test_store_and_show_exchange_credentials(capsys: pytest.CaptureFixture[str]) -> None:
    exit_code = manage_secrets.main(
        [
            "--namespace",
            "tests.cli",
            "store-exchange",
            "--key",
            "binance_paper",
            "--key-id",
            "pub123",
            "--secret",
            "sec456",
            "--environment",
            "paper",
            "--permission",
            "read",
            "--permission",
            "trade",
        ]
    )
    assert exit_code == 0

    exit_code = manage_secrets.main(
        [
            "--namespace",
            "tests.cli",
            "show-exchange",
            "--key",
            "binance_paper",
            "--environment",
            "paper",
        ]
    )
    assert exit_code == 0
    output = capsys.readouterr().out
    assert "pub123" in output
    assert "paper" in output
    assert "sec456" not in output
    secret_line = next(line for line in output.splitlines() if line.startswith("Sekret:"))
    assert "*" in secret_line


def test_store_show_and_delete_generic_secret(capsys: pytest.CaptureFixture[str]) -> None:
    exit_code = manage_secrets.main(
        [
            "--namespace",
            "tests.cli",
            "store-secret",
            "--key",
            "telegram_bot",
            "--purpose",
            "alerts",
            "--value",
            "token123",
        ]
    )
    assert exit_code == 0

    exit_code = manage_secrets.main(
        [
            "--namespace",
            "tests.cli",
            "show-secret",
            "--key",
            "telegram_bot",
            "--purpose",
            "alerts",
        ]
    )
    assert exit_code == 0
    output = capsys.readouterr().out
    assert "telegram_bot" in output
    assert "alerts" in output
    assert "token123" not in output
    assert "Podgląd" in output

    exit_code = manage_secrets.main(
        [
            "--namespace",
            "tests.cli",
            "show-secret",
            "--key",
            "telegram_bot",
            "--purpose",
            "alerts",
            "--reveal",
        ]
    )
    assert exit_code == 0
    output = capsys.readouterr().out
    assert "token123" in output

    exit_code = manage_secrets.main(
        [
            "--namespace",
            "tests.cli",
            "delete-secret",
            "--key",
            "telegram_bot",
            "--purpose",
            "alerts",
        ]
    )
    assert exit_code == 0

    exit_code = manage_secrets.main(
        [
            "--namespace",
            "tests.cli",
            "show-secret",
            "--key",
            "telegram_bot",
            "--purpose",
            "alerts",
        ]
    )
    assert exit_code == 1
    output = capsys.readouterr().out
    assert output == ""


def test_show_missing_exchange_returns_error() -> None:
    exit_code = manage_secrets.main(
        [
            "--namespace",
            "tests.cli",
            "show-exchange",
            "--key",
            "missing",
            "--environment",
            "paper",
        ]
    )
    assert exit_code == 1
