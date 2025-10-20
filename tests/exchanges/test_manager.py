import logging

from bot_core.exchanges.manager import ExchangeManager


def test_set_credentials_accepts_empty_values_and_logs_zero_lengths(caplog) -> None:
    manager = ExchangeManager()
    caplog.clear()

    with caplog.at_level(logging.INFO):
        manager.set_credentials(None, None)

    messages = [message for message in caplog.messages if "Credentials set" in message]
    assert messages, "Expected credentials log entry"
    assert "api_key=0" in messages[-1]
    assert "secret=0" in messages[-1]


def test_set_credentials_logs_lengths_for_non_empty_values(caplog) -> None:
    manager = ExchangeManager()
    caplog.clear()

    with caplog.at_level(logging.INFO):
        manager.set_credentials("abcdef", "secret123")

    messages = [message for message in caplog.messages if "Credentials set" in message]
    assert messages, "Expected credentials log entry"
    assert "api_key=6" in messages[-1]
    assert "secret=9" in messages[-1]
