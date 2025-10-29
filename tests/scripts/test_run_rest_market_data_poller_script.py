from __future__ import annotations

import json
from pathlib import Path

from scripts import run_rest_market_data_poller as script


class _FakePoller:
    def __init__(self, exchanges, **kwargs):
        self.exchanges = [exchange.upper() for exchange in exchanges]
        self.interval = kwargs.get("interval")
        self.profile = kwargs.get("profile")
        self.config_dir = kwargs.get("config_dir")
        self.refresh_calls = 0
        self.started = False
        self.stopped = False

    def refresh_now(self) -> None:
        self.refresh_calls += 1

    def snapshot(self, exchange: str) -> list[dict[str, object]]:
        return [
            {
                "instrument": {
                    "exchange": exchange.upper(),
                    "symbol": "BTC/USDT",
                    "venue_symbol": "BTCUSDT",
                    "quote_currency": "USDT",
                    "base_currency": "BTC",
                },
                "price_step": 0.1,
            }
        ]

    def start(self) -> None:
        self.started = True

    def stop(self) -> None:
        self.stopped = True


def test_run_rest_market_data_once_writes_snapshots(tmp_path: Path) -> None:
    created: list[_FakePoller] = []

    def _factory(exchanges, **kwargs):
        poller = _FakePoller(exchanges, **kwargs)
        created.append(poller)
        return poller

    args = script.build_parser().parse_args(
        ["--exchange", "binance", "--output", str(tmp_path), "--once", "--interval", "0.1"]
    )

    exit_code = script.run_cli(args, poller_factory=_factory)

    assert exit_code == 0
    assert created and created[0].refresh_calls == 1
    assert created[0].profile == "paper"
    assert created[0].config_dir is None

    snapshot_path = tmp_path / "binance.json"
    aggregate_path = tmp_path / "snapshots.json"

    assert snapshot_path.exists()
    assert aggregate_path.exists()

    snapshot = json.loads(snapshot_path.read_text(encoding="utf-8"))
    assert snapshot[0]["instrument"]["exchange"] == "BINANCE"

    aggregate = json.loads(aggregate_path.read_text(encoding="utf-8"))
    assert "BINANCE" in aggregate
