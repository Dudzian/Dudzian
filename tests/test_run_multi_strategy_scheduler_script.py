"""Testy CLI skryptu run_multi_strategy_scheduler."""
from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import pytest

sys.path.append(str(Path(__file__).resolve().parents[1]))

from scripts import run_multi_strategy_scheduler  # type: ignore  # noqa: E402


class _DummyScheduler:
    def __init__(self) -> None:
        self.run_once_called = False
        self.run_forever_called = False

    async def run_once(self) -> None:
        self.run_once_called = True

    async def run_forever(self) -> None:
        self.run_forever_called = True

    def stop(self) -> None:  # pragma: no cover - nie wywołujemy w scenariuszach pozytywnych
        pass


def _invoke_main(
    monkeypatch: pytest.MonkeyPatch,
    argv: Sequence[str],
) -> tuple[_DummyScheduler, dict[str, Any]]:
    captured: dict[str, Any] = {}
    scheduler = _DummyScheduler()

    def fake_build_scheduler(*, adapter_factories: Mapping[str, object] | None, **kwargs: object) -> _DummyScheduler:
        captured["adapter_factories"] = adapter_factories
        captured["kwargs"] = kwargs
        return scheduler

    def fake_asyncio_run(coro: Any, /) -> None:
        captured["async_coro_name"] = (
            getattr(coro, "cr_code", None).co_name if hasattr(coro, "cr_code") else None
        )
        loop = asyncio.new_event_loop()
        try:
            loop.run_until_complete(coro)
        finally:
            loop.close()

    monkeypatch.setattr(run_multi_strategy_scheduler, "_build_scheduler", fake_build_scheduler)
    monkeypatch.setattr(run_multi_strategy_scheduler.asyncio, "run", fake_asyncio_run)
    monkeypatch.setattr(sys, "argv", ["run_multi_strategy_scheduler.py", *argv])

    run_multi_strategy_scheduler.main()

    return scheduler, captured


def test_main_invokes_run_once_with_cli_overrides(monkeypatch: pytest.MonkeyPatch) -> None:
    argv = [
        "--environment",
        "binance_paper",
        "--run-once",
        "--adapter-factory",
        "kucoin_spot=bot_core.exchanges.kucoin:KuCoinSpotAdapter",
        "--adapter-factory",
        "bybit_spot=json:{\"path\": \"bot_core.exchanges.bybit:BybitSpotAdapter\", \"override\": true}",
    ]

    scheduler, captured = _invoke_main(monkeypatch, argv)

    assert captured["async_coro_name"] == "run_once"
    adapter_factories = captured["adapter_factories"]
    assert isinstance(adapter_factories, Mapping)
    assert adapter_factories == {
        "kucoin_spot": "bot_core.exchanges.kucoin:KuCoinSpotAdapter",
        "bybit_spot": {"path": "bot_core.exchanges.bybit:BybitSpotAdapter", "override": True},
    }
    assert scheduler.run_once_called is True
    assert scheduler.run_forever_called is False


def test_main_invokes_run_forever_without_overrides(monkeypatch: pytest.MonkeyPatch) -> None:
    argv = [
        "--environment",
        "okx_paper",
    ]

    scheduler, captured = _invoke_main(monkeypatch, argv)

    assert captured["async_coro_name"] == "run_forever"
    assert captured["adapter_factories"] is None
    assert scheduler.run_once_called is False
    assert scheduler.run_forever_called is True


def test_main_supports_remove_spec(monkeypatch: pytest.MonkeyPatch) -> None:
    argv = [
        "--environment",
        "coinbase_paper",
        "--run-once",
        "--adapter-factory",
        "kraken_spot=!remove",
    ]

    scheduler, captured = _invoke_main(monkeypatch, argv)

    assert captured["adapter_factories"] == {"kraken_spot": {"remove": True}}
    assert captured["async_coro_name"] == "run_once"
    assert scheduler.run_once_called is True
    assert scheduler.run_forever_called is False
