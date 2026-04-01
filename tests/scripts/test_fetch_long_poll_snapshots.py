from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts import fetch_long_poll_snapshots as fetcher


def _fake_snapshot(adapter: str, environment: str) -> dict[str, object]:
    return {
        "labels": {
            "adapter": adapter,
            "scope": "public",
            "environment": environment,
            "region": "eu",
        },
        "requestLatency": {"count": 2, "p95": 0.5},
        "deliveryLag": {"count": 2, "p95": 0.7},
        "httpErrors": {"total": 0, "byStatusCode": {}},
        "reconnects": {"attempts": 0, "success": 0, "failure": 0},
    }


def test_fetch_snapshots_writes_all_required_profiles(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def _stub_collect_single_snapshot(**kwargs):
        return _fake_snapshot(kwargs["adapter"], kwargs["environment"])

    monkeypatch.setattr(
        fetcher,
        "_load_snapshot_environment_configs",
        lambda _path: {
            "deribit_futures_paper": fetcher.SnapshotEnvironmentConfig(
                adapter_settings={"stream": {"public_params": {"symbol": "BTC-PERPETUAL"}}}
            ),
            "deribit_futures_live": fetcher.SnapshotEnvironmentConfig(adapter_settings={}),
            "bitmex_futures_paper": fetcher.SnapshotEnvironmentConfig(
                adapter_settings={"stream": {"public_params": {"symbol": "XBTUSD"}}}
            ),
            "bitmex_futures_live": fetcher.SnapshotEnvironmentConfig(adapter_settings={}),
        },
    )
    monkeypatch.setattr(fetcher, "_load_health_check_public_symbol", lambda *_args: "XBTUSD")
    monkeypatch.setattr(fetcher, "_collect_single_snapshot", _stub_collect_single_snapshot)

    output = tmp_path / "long_poll_snapshots.json"
    fetcher.fetch_snapshots(
        base_url="http://127.0.0.1:8765",
        config_path=tmp_path / "core.yaml",
        output_path=output,
        channels=("ticker",),
        timeout_seconds=1.0,
    )

    payload = json.loads(output.read_text(encoding="utf-8"))
    assert "collected_at" in payload
    assert len(payload["snapshots"]) == 4


def test_fetch_snapshots_fails_when_required_profile_missing(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def _stub_collect_single_snapshot(**kwargs):
        adapter = kwargs["adapter"]
        environment = kwargs["environment"]
        if adapter == "bitmex_futures" and environment == "live":
            return _fake_snapshot("bitmex_futures", "paper")
        return _fake_snapshot(adapter, environment)

    monkeypatch.setattr(
        fetcher,
        "_load_snapshot_environment_configs",
        lambda _path: {
            "deribit_futures_paper": fetcher.SnapshotEnvironmentConfig(adapter_settings={}),
            "deribit_futures_live": fetcher.SnapshotEnvironmentConfig(adapter_settings={}),
            "bitmex_futures_paper": fetcher.SnapshotEnvironmentConfig(adapter_settings={}),
            "bitmex_futures_live": fetcher.SnapshotEnvironmentConfig(adapter_settings={}),
        },
    )
    monkeypatch.setattr(fetcher, "_load_health_check_public_symbol", lambda *_args: "XBTUSD")
    monkeypatch.setattr(fetcher, "_collect_single_snapshot", _stub_collect_single_snapshot)

    with pytest.raises(RuntimeError, match="Brak wymaganych snapshotów"):
        fetcher.fetch_snapshots(
            base_url="http://127.0.0.1:8765",
            config_path=tmp_path / "core.yaml",
            output_path=tmp_path / "long_poll_snapshots.json",
            channels=("ticker",),
            timeout_seconds=1.0,
        )


def test_collect_single_snapshot_uses_stream_gateway_path_and_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    class _DummyStream:
        def __init__(self, **kwargs):
            captured.update(kwargs)

        def start(self):
            return None

        def export_metrics_snapshot(self):
            return {
                "labels": {"adapter": "deribit_futures", "scope": "public", "environment": "paper"},
                "requestLatency": {"count": 1, "p95": 0.2},
            }

        def close(self):
            return None

    monkeypatch.setattr(fetcher, "LocalLongPollStream", _DummyStream)

    snapshot = fetcher._collect_single_snapshot(
        base_url="http://127.0.0.1:8765",
        adapter="deribit_futures",
        environment="paper",
        scope="public",
        channels=("ticker",),
        params={"symbol": "BTC-PERPETUAL"},
        timeout_seconds=1.0,
    )

    assert captured["path"] == "/stream/deribit_futures/public"
    assert captured["environment"] == "paper"
    assert captured["scope"] == "public"
    assert captured["params"] == {"symbol": "BTC-PERPETUAL"}
    assert snapshot["labels"]["environment"] == "paper"


def test_resolve_public_stream_params_prefers_public_params_for_ticker(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _raise_if_called(*_args):
        raise AssertionError("fallback should not be used")

    monkeypatch.setattr(fetcher, "_load_health_check_public_symbol", _raise_if_called)

    params = fetcher._resolve_public_stream_params(
        config_path=Path("config/core.yaml"),
        environment_name="deribit_futures_paper",
        environment_config=fetcher.SnapshotEnvironmentConfig(
            adapter_settings={"stream": {"public_params": {"symbol": "ETH-PERPETUAL"}}},
        ),
        channels=("ticker",),
    )
    assert params == {"symbol": "ETH-PERPETUAL"}


def test_resolve_public_stream_params_falls_back_to_health_check_symbol(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []

    def _fallback(_config_path: Path, environment_name: str) -> str | None:
        calls.append(environment_name)
        return "XBTUSD"

    monkeypatch.setattr(fetcher, "_load_health_check_public_symbol", _fallback)

    params = fetcher._resolve_public_stream_params(
        config_path=Path("config/core.yaml"),
        environment_name="bitmex_futures_live",
        environment_config=fetcher.SnapshotEnvironmentConfig(adapter_settings={}),
        channels=("ticker",),
    )
    assert params == {"symbol": "XBTUSD"}
    assert calls == ["bitmex_futures_live"]


def test_resolve_public_stream_params_ticker_requires_symbol(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(fetcher, "_load_health_check_public_symbol", lambda *_args: None)
    with pytest.raises(RuntimeError, match="Brak konfiguracji symbolu.*bitmex_futures_paper"):
        fetcher._resolve_public_stream_params(
            config_path=Path("config/core.yaml"),
            environment_name="bitmex_futures_paper",
            environment_config=fetcher.SnapshotEnvironmentConfig(adapter_settings={}),
            channels=("ticker",),
        )


def test_fetch_snapshots_passes_symbol_from_config_to_stream(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    captured_params: dict[tuple[str, str], object] = {}

    def _stub_collect_single_snapshot(**kwargs):
        captured_params[(kwargs["adapter"], kwargs["environment"])] = kwargs["params"]
        return _fake_snapshot(kwargs["adapter"], kwargs["environment"])

    monkeypatch.setattr(
        fetcher,
        "_load_snapshot_environment_configs",
        lambda _path: {
            "deribit_futures_paper": fetcher.SnapshotEnvironmentConfig(
                adapter_settings={"stream": {"public_params": {"symbol": "BTC-PERPETUAL"}}},
            ),
            "deribit_futures_live": fetcher.SnapshotEnvironmentConfig(adapter_settings={}),
            "bitmex_futures_paper": fetcher.SnapshotEnvironmentConfig(
                adapter_settings={"stream": {"public_params": {"symbol": "XBTUSD"}}},
            ),
            "bitmex_futures_live": fetcher.SnapshotEnvironmentConfig(adapter_settings={}),
        },
    )
    monkeypatch.setattr(
        fetcher,
        "_load_health_check_public_symbol",
        lambda _config_path, environment_name: "BTC-PERPETUAL"
        if environment_name.startswith("deribit")
        else "XBTUSD",
    )
    monkeypatch.setattr(fetcher, "_collect_single_snapshot", _stub_collect_single_snapshot)

    fetcher.fetch_snapshots(
        base_url="http://127.0.0.1:8765",
        config_path=tmp_path / "core.yaml",
        output_path=tmp_path / "long_poll_snapshots.json",
        channels=("ticker",),
        timeout_seconds=1.0,
    )

    assert captured_params[("deribit_futures", "paper")] == {"symbol": "BTC-PERPETUAL"}
    assert captured_params[("deribit_futures", "live")] == {"symbol": "BTC-PERPETUAL"}
    assert captured_params[("bitmex_futures", "paper")] == {"symbol": "XBTUSD"}
    assert captured_params[("bitmex_futures", "live")] == {"symbol": "XBTUSD"}


def test_fetch_snapshots_fails_fast_for_ticker_without_symbol(
    tmp_path,
) -> None:
    fetcher_configs = {
        "deribit_futures_paper": fetcher.SnapshotEnvironmentConfig(adapter_settings={}),
        "deribit_futures_live": fetcher.SnapshotEnvironmentConfig(adapter_settings={}),
        "bitmex_futures_paper": fetcher.SnapshotEnvironmentConfig(adapter_settings={}),
        "bitmex_futures_live": fetcher.SnapshotEnvironmentConfig(adapter_settings={}),
    }
    with pytest.MonkeyPatch.context() as monkeypatch:
        monkeypatch.setattr(
            fetcher,
            "_load_snapshot_environment_configs",
            lambda _path: fetcher_configs,
        )
        monkeypatch.setattr(fetcher, "_load_health_check_public_symbol", lambda *_args: None)
        with pytest.raises(RuntimeError, match="Brak konfiguracji symbolu.*deribit_futures_paper"):
            fetcher.fetch_snapshots(
                base_url="http://127.0.0.1:8765",
                config_path=tmp_path / "core.yaml",
                output_path=tmp_path / "long_poll_snapshots.json",
                channels=("ticker",),
                timeout_seconds=1.0,
            )


def test_resolve_public_stream_params_keeps_extra_public_params(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(fetcher, "_load_health_check_public_symbol", lambda *_args: "IGNORED")
    params = fetcher._resolve_public_stream_params(
        config_path=Path("config/core.yaml"),
        environment_name="deribit_futures_paper",
        environment_config=fetcher.SnapshotEnvironmentConfig(
            adapter_settings={
                "stream": {"public_params": {"symbol": "BTC-PERPETUAL", "depth": 100}}
            },
        ),
        channels=("ticker",),
    )
    assert params == {"symbol": "BTC-PERPETUAL", "depth": 100}


def test_resolve_public_stream_params_accepts_symbols_field(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(fetcher, "_load_health_check_public_symbol", lambda *_args: "IGNORED")
    params = fetcher._resolve_public_stream_params(
        config_path=Path("config/core.yaml"),
        environment_name="bitmex_futures_live",
        environment_config=fetcher.SnapshotEnvironmentConfig(
            adapter_settings={"stream": {"public_params": {"symbols": ["XBTUSD", "ETHUSD"]}}},
        ),
        channels=("ticker",),
    )
    assert params == {"symbols": ["XBTUSD", "ETHUSD"]}


def test_resolve_public_stream_params_detects_ticker_case_insensitive(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(fetcher, "_load_health_check_public_symbol", lambda *_args: "XBTUSD")
    params = fetcher._resolve_public_stream_params(
        config_path=Path("config/core.yaml"),
        environment_name="bitmex_futures_live",
        environment_config=fetcher.SnapshotEnvironmentConfig(adapter_settings={}),
        channels=(" Trades ", "TICKER"),
    )
    assert params == {"symbol": "XBTUSD"}


def test_resolve_public_stream_params_detects_tickers_plural(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(fetcher, "_load_health_check_public_symbol", lambda *_args: "XBTUSD")
    params = fetcher._resolve_public_stream_params(
        config_path=Path("config/core.yaml"),
        environment_name="bitmex_futures_live",
        environment_config=fetcher.SnapshotEnvironmentConfig(adapter_settings={}),
        channels=("tickers",),
    )
    assert params == {"symbol": "XBTUSD"}
