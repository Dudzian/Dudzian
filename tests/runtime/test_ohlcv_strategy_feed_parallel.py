from __future__ import annotations

import threading
import time
import types

from bot_core.data.base import OHLCVRequest
from bot_core.runtime.pipeline import OHLCVStrategyFeed, MarketSnapshot


class _StubSource:
    def __init__(self, with_overlap: bool) -> None:
        self._with_overlap = with_overlap
        self.thread_names: set[str] = set()
        self._start_event = threading.Event()
        self._overlap_event = threading.Event()

    def fetch_ohlcv(self, request: OHLCVRequest):
        self.thread_names.add(threading.current_thread().name)
        if self._with_overlap:
            if not self._start_event.is_set():
                self._start_event.set()
                time.sleep(0.05)
            else:
                self._overlap_event.set()
        columns = ("open_time", "open", "high", "low", "close", "volume")
        base = int(request.start or 0)
        rows = [
            (base, 1.0, 2.0, 0.5, 1.5, 100.0),
            (base + 60_000, 1.1, 2.1, 0.6, 1.6, 120.0),
        ]
        return types.SimpleNamespace(columns=columns, rows=rows)

    @property
    def overlap_detected(self) -> bool:
        return self._overlap_event.is_set()


def test_parallel_history_fetch_uses_thread_pool():
    source = _StubSource(with_overlap=True)
    feed = OHLCVStrategyFeed(
        data_source=source,  # type: ignore[arg-type]
        symbols_map={"demo": ("BTC/USDT", "ETH/USDT")},
        interval_map={},
        default_interval="1m",
        max_workers=2,
    )

    snapshots = feed.load_history("demo", bars=2)

    assert len(snapshots) == 4
    assert all(isinstance(snapshot, MarketSnapshot) for snapshot in snapshots)
    assert len(source.thread_names) >= 2
    assert source.overlap_detected


def test_parallel_latest_fetch_falls_back_to_single_thread():
    source = _StubSource(with_overlap=False)
    feed = OHLCVStrategyFeed(
        data_source=source,  # type: ignore[arg-type]
        symbols_map={"demo": ("BTC/USDT",)},
        interval_map={},
        default_interval="1m",
        max_workers=4,
    )

    snapshots = feed.fetch_latest("demo")

    assert len(snapshots) == 1
    assert len(source.thread_names) == 1
