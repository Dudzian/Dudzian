from __future__ import annotations

import json
import threading
from datetime import datetime, timedelta, timezone
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Iterable, Iterator

import pandas as pd
import pytest

from bot_core.backtest.walk_forward import (
    StrategyDefinition,
    WalkForwardBacktester,
    WalkForwardSegment,
)
from bot_core.exchanges.streaming import LocalLongPollStream
from bot_core.runtime.streaming_bridge import (
    history_to_stream_batches,
    stream_batches_to_frame,
)
from bot_core.strategies.catalog import DEFAULT_STRATEGY_CATALOG


def _build_history(rows: int = 24) -> list[dict[str, float | int]]:
    start = datetime(2024, 1, 1, tzinfo=timezone.utc)
    history: list[dict[str, float | int]] = []
    price = 100.0
    for index in range(rows):
        timestamp = start + timedelta(minutes=5 * index)
        open_price = price
        close_price = price + ((-1) ** index) * 0.5 + index * 0.1
        high_price = max(open_price, close_price) + 0.2
        low_price = min(open_price, close_price) - 0.2
        volume = 50.0 + index * 1.5
        history.append(
            {
                "timestamp_ms": int(timestamp.timestamp() * 1000),
                "open": round(open_price, 5),
                "high": round(high_price, 5),
                "low": round(low_price, 5),
                "close": round(close_price, 5),
                "volume": round(volume, 5),
            }
        )
        price = close_price
    return history


def _history_frame(history: Iterable[dict[str, float | int]]) -> pd.DataFrame:
    frame = pd.DataFrame(list(history))
    frame.index = pd.to_datetime(frame["timestamp_ms"], unit="ms", utc=True)
    frame.index.name = "timestamp"
    return frame[["open", "high", "low", "close", "volume"]]


class _StreamRequestHandler(BaseHTTPRequestHandler):
    server: "HTTPServer"

    def do_GET(self) -> None:  # noqa: N802
        payload = self.server.responses.pop(0) if self.server.responses else {"batches": [], "retry_after": 0.0}
        data = json.dumps(payload).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


def _start_stream_server(responses: list[dict[str, object]]) -> tuple[HTTPServer, threading.Thread]:
    server = HTTPServer(("127.0.0.1", 0), _StreamRequestHandler)
    server.responses = [  # type: ignore[attr-defined]
        {"batches": [batch], "retry_after": 0.0}
        for batch in responses
    ]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server, thread


def _stop_stream_server(server: HTTPServer, thread: threading.Thread) -> None:
    server.shutdown()
    thread.join()


def _build_definition() -> StrategyDefinition:
    return StrategyDefinition(
        name="trend-demo",
        engine="daily_trend_momentum",
        license_tier="standard",
        risk_classes=("directional",),
        required_data=("ohlcv",),
        parameters={
            "fast_ma": 3,
            "slow_ma": 7,
            "breakout_lookback": 5,
            "momentum_window": 3,
            "atr_window": 5,
            "atr_multiplier": 1.0,
            "min_trend_strength": 0.0,
            "min_momentum": 0.0,
        },
    )


def test_streaming_matches_walk_forward_history() -> None:
    history = _build_history()
    frame = _history_frame(history)
    dataset = {"BTCUSDT": frame}

    index = frame.index
    segments = [
        WalkForwardSegment(
            train_start=index[0].to_pydatetime(),
            train_end=index[7].to_pydatetime(),
            test_start=index[8].to_pydatetime(),
            test_end=index[15].to_pydatetime(),
        ),
        WalkForwardSegment(
            train_start=index[8].to_pydatetime(),
            train_end=index[15].to_pydatetime(),
            test_start=index[16].to_pydatetime(),
            test_end=index[-1].to_pydatetime(),
        ),
    ]

    backtester = WalkForwardBacktester(DEFAULT_STRATEGY_CATALOG)
    baseline = backtester.run(_build_definition(), dataset, segments)

    batches = history_to_stream_batches(history, channel="ohlcv", batch_size=5)
    server, thread = _start_stream_server(batches)
    try:
        stream = LocalLongPollStream(
            base_url=f"http://127.0.0.1:{server.server_port}",
            path="/stream",
            channels=["ohlcv"],
            adapter="demo",
            scope="public",
            environment="paper",
            poll_interval=0.0,
            timeout=1.0,
            max_retries=1,
            backoff_base=0.0,
            backoff_cap=0.0,
            jitter=(0.0, 0.0),
        )
        collected: list = []
        for batch in stream:
            collected.append(batch)
            total = sum(len(item.events) for item in collected)
            if total >= len(history):
                break
        stream.close()
    finally:
        _stop_stream_server(server, thread)

    stream_frame = stream_batches_to_frame(collected)
    pd.testing.assert_frame_equal(stream_frame, frame)

    streamed_dataset = {"BTCUSDT": stream_frame}
    streamed_report = backtester.run(_build_definition(), streamed_dataset, segments)

    assert streamed_report.total_return_pct == pytest.approx(baseline.total_return_pct, rel=1e-6)
    assert (
        streamed_report.cost_summary.total_notional
        == pytest.approx(baseline.cost_summary.total_notional, rel=1e-6)
    )
    assert streamed_report.cost_summary.total_trades == baseline.cost_summary.total_trades

