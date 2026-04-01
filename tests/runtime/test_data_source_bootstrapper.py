"""Safety-net tests dla DataSourceBootstrapper."""

from __future__ import annotations

import logging
from types import SimpleNamespace

import pytest

from bot_core.observability.metrics import MetricsRegistry
from bot_core.runtime.data_source_bootstrapper import DataSourceBootstrapper


def test_create_cached_source_online_path(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: dict[str, object] = {}

    def _fake_namespace(environment: object) -> str:
        calls["namespace_environment"] = environment
        return "ns-online"

    def _fake_create(adapter: object, **kwargs: object) -> object:
        calls["adapter"] = adapter
        calls.update(kwargs)
        return "cached-source"

    monkeypatch.setattr(
        "bot_core.runtime.data_source_bootstrapper.resolve_cache_namespace",
        _fake_namespace,
    )
    monkeypatch.setattr(
        "bot_core.runtime.data_source_bootstrapper.create_cached_ohlcv_source",
        _fake_create,
    )

    environment = SimpleNamespace(
        data_cache_path="/tmp/cache",
        offline_mode=False,
        data_source=SimpleNamespace(enable_snapshots=True),
    )
    adapter = object()

    result = DataSourceBootstrapper().create_cached_source(
        adapter=adapter,  # type: ignore[arg-type]
        environment=environment,  # type: ignore[arg-type]
    )

    assert result == "cached-source"
    assert calls["adapter"] is adapter
    assert calls["enable_snapshots"] is True
    assert calls["allow_network_upstream"] is True
    assert calls["namespace"] == "ns-online"


def test_create_cached_source_offline_forces_no_network_and_no_snapshots(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: dict[str, object] = {}

    monkeypatch.setattr(
        "bot_core.runtime.data_source_bootstrapper.resolve_cache_namespace",
        lambda _environment: "ns-offline",
    )

    def _fake_create(_adapter: object, **kwargs: object) -> object:
        calls.update(kwargs)
        return "cached-source"

    monkeypatch.setattr(
        "bot_core.runtime.data_source_bootstrapper.create_cached_ohlcv_source",
        _fake_create,
    )

    environment = SimpleNamespace(
        data_cache_path="/tmp/cache",
        offline_mode=True,
        data_source=SimpleNamespace(enable_snapshots=True),
    )

    DataSourceBootstrapper().create_cached_source(
        adapter=object(),  # type: ignore[arg-type]
        environment=environment,  # type: ignore[arg-type]
    )

    assert calls["enable_snapshots"] is False
    assert calls["allow_network_upstream"] is False


def test_create_cached_source_missing_data_modules_raises_runtime_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "bot_core.runtime.data_source_bootstrapper.create_cached_ohlcv_source",
        None,
    )

    with pytest.raises(RuntimeError, match="Brakuje modułów data"):
        DataSourceBootstrapper().create_cached_source(
            adapter=object(),  # type: ignore[arg-type]
            environment=SimpleNamespace(data_cache_path="/tmp/cache"),  # type: ignore[arg-type]
        )


def test_create_cached_source_logs_and_reraises(monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture) -> None:
    monkeypatch.setattr(
        "bot_core.runtime.data_source_bootstrapper.resolve_cache_namespace",
        lambda _environment: "ns",
    )

    def _broken_create(_adapter: object, **_kwargs: object) -> object:
        raise ValueError("boom-create")

    monkeypatch.setattr(
        "bot_core.runtime.data_source_bootstrapper.create_cached_ohlcv_source",
        _broken_create,
    )

    caplog.set_level(logging.ERROR)
    with pytest.raises(ValueError, match="boom-create"):
        DataSourceBootstrapper().create_cached_source(
            adapter=object(),  # type: ignore[arg-type]
            environment=SimpleNamespace(data_cache_path="/tmp/cache", offline_mode=False),  # type: ignore[arg-type]
        )

    assert any(
        "cached source creation" in record.getMessage()
        for record in caplog.records
    )


def test_ensure_local_market_data_availability_builds_scheduler_with_default_columns(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    class _FakeScheduler:
        def __init__(self, data_source: object, **kwargs: object) -> None:
            captured["data_source"] = data_source
            captured.update(kwargs)

        def ensure_ohlcv_availability(self, **kwargs: object) -> None:
            captured["ensure_kwargs"] = kwargs

    monkeypatch.setattr("bot_core.runtime.data_source_bootstrapper.BackfillScheduler", _FakeScheduler)

    bootstrapper = DataSourceBootstrapper()
    environment = SimpleNamespace(name="paper")
    markets = {"BTCUSDT": object()}

    bootstrapper.ensure_local_market_data_availability(
        environment=environment,  # type: ignore[arg-type]
        data_source="source",  # type: ignore[arg-type]
        markets=markets,
        interval="1h",
        backfill_service="backfill",  # type: ignore[arg-type]
        adapter="adapter",  # type: ignore[arg-type]
    )

    assert captured["default_columns"] == (
        "open_time",
        "open",
        "high",
        "low",
        "close",
        "volume",
    )
    assert captured["ensure_kwargs"] == {
        "symbols": markets.keys(),
        "interval": "1h",
        "environment": environment,
    }


def test_ensure_local_market_data_availability_missing_scheduler_raises_runtime_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("bot_core.runtime.data_source_bootstrapper.BackfillScheduler", None)

    with pytest.raises(RuntimeError, match="Brakuje BackfillScheduler"):
        DataSourceBootstrapper().ensure_local_market_data_availability(
            environment=SimpleNamespace(),  # type: ignore[arg-type]
            data_source=object(),  # type: ignore[arg-type]
            markets={},
            interval="1h",
        )


def test_ensure_local_market_data_availability_logs_and_reraises(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    class _BrokenScheduler:
        def __init__(self, *_args: object, **_kwargs: object) -> None:
            pass

        def ensure_ohlcv_availability(self, **_kwargs: object) -> None:
            raise ValueError("boom-ensure")

    monkeypatch.setattr(
        "bot_core.runtime.data_source_bootstrapper.BackfillScheduler",
        _BrokenScheduler,
    )

    caplog.set_level(logging.ERROR)
    with pytest.raises(ValueError, match="boom-ensure"):
        DataSourceBootstrapper().ensure_local_market_data_availability(
            environment=SimpleNamespace(),  # type: ignore[arg-type]
            data_source=object(),  # type: ignore[arg-type]
            markets={"BTCUSDT": object()},
            interval="1h",
        )

    assert any(
        "OHLCV availability check" in record.getMessage()
        for record in caplog.records
    )


@pytest.mark.parametrize(
    ("adapter", "expected"),
    [
        (SimpleNamespace(metrics_registry=MetricsRegistry()), "metrics_registry"),
        (SimpleNamespace(_metrics=MetricsRegistry()), "_metrics"),
        (SimpleNamespace(), None),
        (None, None),
    ],
)
def test_resolve_adapter_metrics_registry_paths(adapter: object, expected: object) -> None:
    result = DataSourceBootstrapper.resolve_adapter_metrics_registry(adapter)
    if expected in {"metrics_registry", "_metrics"}:
        assert isinstance(result, MetricsRegistry)
    else:
        assert result is expected


def test_build_streaming_feed_delegates_successfully() -> None:
    captured: dict[str, object] = {}

    def _factory(**kwargs: object) -> object:
        captured.update(kwargs)
        return "stream-feed"

    bootstrapper = DataSourceBootstrapper()
    result = bootstrapper.build_streaming_feed(
        stream_factory=_factory,
        stream_config=object(),
        stream_settings={"poll_interval": 0.5},
        adapter_metrics=None,
        base_feed=object(),  # type: ignore[arg-type]
        symbols_map={"s1": ["BTCUSDT"]},
        exchange="binance",
        environment_name="paper",
    )

    assert result == "stream-feed"
    assert captured["exchange"] == "binance"
    assert captured["environment_name"] == "paper"


def test_build_streaming_feed_logs_and_reraises(caplog: pytest.LogCaptureFixture) -> None:
    def _broken_factory(**_kwargs: object) -> object:
        raise ValueError("boom-stream")

    caplog.set_level(logging.ERROR)

    with pytest.raises(ValueError, match="boom-stream"):
        DataSourceBootstrapper().build_streaming_feed(
            stream_factory=_broken_factory,
            stream_config=object(),
            stream_settings={},
            adapter_metrics=None,
            base_feed=object(),  # type: ignore[arg-type]
            symbols_map={},
            exchange="binance",
            environment_name="paper",
        )

    assert any(
        "streaming feed wiring" in record.getMessage()
        for record in caplog.records
    )
