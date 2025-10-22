import asyncio
import time

import pytest

from KryptoLowca.exchanges.polling import MarketDataPoller


class RecordingAdapter:
    def __init__(self) -> None:
        self.calls: list[str] = []

    async def fetch_market_data(self, symbol: str) -> dict[str, float]:
        self.calls.append(symbol)
        await asyncio.sleep(0)  # yield control
        return {"symbol": symbol, "last": 100.0}


@pytest.mark.asyncio
async def test_market_data_poller_invokes_callback_for_each_symbol():
    adapter = RecordingAdapter()
    captured: list[tuple[str, dict[str, float]]] = []
    ready = asyncio.Event()

    async def callback(symbol: str, payload: dict[str, float]) -> None:
        captured.append((symbol, payload))
        if len(captured) >= 3:
            ready.set()

    poller = MarketDataPoller(
        adapter,
        symbols=["BTC-PLN", "ETH-PLN", "LTC-PLN"],
        interval=0.01,
        callback=callback,
    )

    async with poller:
        await asyncio.wait_for(ready.wait(), timeout=1.0)

    assert {symbol for symbol, _ in captured} >= {"BTC-PLN", "ETH-PLN", "LTC-PLN"}
    assert adapter.calls[:3] == ["BTC-PLN", "ETH-PLN", "LTC-PLN"]


class FlakyAdapter:
    def __init__(self) -> None:
        self.timestamps: list[float] = []
        self._failures = 0

    async def fetch_market_data(self, symbol: str) -> dict[str, float]:
        self.timestamps.append(time.monotonic())
        if not self._failures:
            self._failures += 1
            raise RuntimeError("boom")
        return {"symbol": symbol, "last": 1.0}


@pytest.mark.asyncio
async def test_market_data_poller_applies_backoff_on_errors():
    adapter = FlakyAdapter()
    errors: list[tuple[str, Exception]] = []
    completed = asyncio.Event()

    async def callback(symbol: str, payload: dict[str, float]) -> None:
        completed.set()

    async def error_callback(symbol: str, exc: Exception) -> None:
        errors.append((symbol, exc))

    poller = MarketDataPoller(
        adapter,
        symbols=["BTC-PLN"],
        interval=0.01,
        callback=callback,
        error_callback=error_callback,
        backoff_initial=0.05,
        backoff_multiplier=1.0,
        backoff_max=0.05,
    )

    async with poller:
        await asyncio.wait_for(completed.wait(), timeout=1.0)

    assert len(adapter.timestamps) >= 2
    assert errors and isinstance(errors[0][1], RuntimeError)
    delay = adapter.timestamps[1] - adapter.timestamps[0]
    assert delay >= 0.04
