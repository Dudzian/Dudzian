import asyncio
import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path


from bot_core.config.loader import load_core_config
from bot_core.config.models import (
    EnvironmentDataQualityConfig,
    InstrumentBackfillWindow,
    InstrumentConfig,
    InstrumentUniverseConfig,
)
from bot_core.data.ohlcv import CoverageStatus, ManifestEntry, SQLiteCacheStorage
from bot_core.exchanges.base import Environment
from bot_core.exchanges.binance.futures import BinanceFuturesAdapter
from bot_core.exchanges.binance.spot import BinanceSpotAdapter
from bot_core.exchanges.kraken.futures import KrakenFuturesAdapter
from bot_core.exchanges.kraken.spot import KrakenSpotAdapter
from bot_core.exchanges.zonda.spot import ZondaSpotAdapter

import scripts.backfill as backfill


def test_build_public_source_supports_all_exchanges_from_universe():
    config = load_core_config("config/core.yaml")
    universe = config.instrument_universes["core_multi_exchange"]

    expected_adapters = {
        "binance_spot": BinanceSpotAdapter,
        "binance_futures": BinanceFuturesAdapter,
        "kraken_spot": KrakenSpotAdapter,
        "kraken_futures": KrakenFuturesAdapter,
        "zonda_spot": ZondaSpotAdapter,
    }

    exchanges = {
        exchange_name
        for instrument in universe.instruments
        for exchange_name in instrument.exchange_symbols.keys()
    }

    for exchange in exchanges:
        source = backfill._build_public_source(exchange, Environment.PAPER)
        assert isinstance(source.exchange_adapter, expected_adapters[exchange])
        assert source.exchange_adapter.credentials.key_id == "public"
        assert source.exchange_adapter.credentials.environment == Environment.PAPER


def test_build_interval_plans_assigns_refresh_seconds_and_lookbacks():
    universe = InstrumentUniverseConfig(
        name="test",
        description="test",
        instruments=(
            InstrumentConfig(
                name="BTC_USDT",
                base_asset="BTC",
                quote_asset="USDT",
                categories=("core",),
                exchange_symbols={"binance_spot": "BTCUSDT"},
                backfill_windows=(
                    InstrumentBackfillWindow(interval="1d", lookback_days=365),
                    InstrumentBackfillWindow(interval="1h", lookback_days=30),
                    InstrumentBackfillWindow(interval="15m", lookback_days=5),
                ),
            ),
        ),
    )

    plans, symbols = backfill._build_interval_plans(
        universe=universe,
        exchange_name="binance_spot",
        incremental_lookback_days=7,
        refresh_overrides={"1h": 120, "15m": 60},
        jitter_overrides={"1d": 900, "15m": 30},
    )

    assert symbols == {"BTCUSDT"}

    assert plans["1d"].refresh_seconds == backfill._DEFAULT_REFRESH_SECONDS["1d"]
    assert plans["1d"].incremental_lookback_ms == 7 * backfill._MILLISECONDS_IN_DAY
    assert plans["1d"].jitter_seconds == backfill._DEFAULT_JITTER_SECONDS["1d"]

    assert plans["1h"].refresh_seconds == 120
    assert plans["1h"].incremental_lookback_ms == 7 * backfill._MILLISECONDS_IN_DAY
    assert plans["1h"].jitter_seconds == backfill._DEFAULT_JITTER_SECONDS["1h"]

    assert plans["15m"].refresh_seconds == 60
    assert plans["15m"].incremental_lookback_ms == 5 * backfill._MILLISECONDS_IN_DAY
    assert plans["15m"].jitter_seconds == 30


def test_format_plan_summary_lists_intervals_and_symbols():
    plans = {
        "1d": backfill._IntervalPlan(
            symbols={"BTCUSDT", "ETHUSDT"},
            backfill_start_ms=1_600_000_000_000,
            incremental_lookback_ms=3 * backfill._MILLISECONDS_IN_DAY,
            refresh_seconds=backfill._DEFAULT_REFRESH_SECONDS["1d"],
            jitter_seconds=backfill._DEFAULT_JITTER_SECONDS["1d"],
        ),
        "1h": backfill._IntervalPlan(
            symbols={"BTCUSDT"},
            backfill_start_ms=1_600_100_000_000,
            incremental_lookback_ms=0,
            refresh_seconds=0,  # dziedziczy z CLI
            jitter_seconds=0,
        ),
    }

    summary = backfill._format_plan_summary(
        plans,
        exchange_name="binance_spot",
        environment_name="binance_paper",
    )

    assert "Plan backfillu" in summary
    assert "symbole=2" in summary
    assert "1d" in summary and "1h" in summary
    assert "dziedziczy (--refresh-seconds)" in summary
    assert "jitter=0s" in summary


class _DummyScheduler:
    def __init__(self) -> None:
        self.jobs: list[dict] = []
        self.stopped = False

    def add_job(self, **kwargs):
        self.jobs.append(kwargs)

    async def run_forever(self):
        return

    def stop(self):
        self.stopped = True


def test_run_scheduler_uses_interval_specific_frequency():
    scheduler = _DummyScheduler()
    plans = {
        "1d": backfill._IntervalPlan(
            symbols={"BTCUSDT"},
            backfill_start_ms=0,
            incremental_lookback_ms=backfill._MILLISECONDS_IN_DAY,
            refresh_seconds=backfill._DEFAULT_REFRESH_SECONDS["1d"],
            jitter_seconds=backfill._DEFAULT_JITTER_SECONDS["1d"],
        ),
        "1h": backfill._IntervalPlan(
            symbols={"ETHUSDT"},
            backfill_start_ms=0,
            incremental_lookback_ms=3 * backfill._MILLISECONDS_IN_DAY,
            refresh_seconds=900,
            jitter_seconds=0,
        ),
    }

    asyncio.run(
        backfill._run_scheduler(
            scheduler=scheduler,
            plans=plans,
            refresh_seconds=600,
        )
    )

    assert scheduler.stopped is True
    assert len(scheduler.jobs) == 2

    job_daily = next(job for job in scheduler.jobs if job["interval"] == "1d")
    job_hourly = next(job for job in scheduler.jobs if job["interval"] == "1h")

    assert job_daily["frequency_seconds"] == backfill._DEFAULT_REFRESH_SECONDS["1d"]
    assert job_daily["lookback_ms"] == backfill._MILLISECONDS_IN_DAY
    assert job_daily["jitter_seconds"] == backfill._DEFAULT_JITTER_SECONDS["1d"]

    assert job_hourly["frequency_seconds"] == 900
    assert job_hourly["lookback_ms"] == 3 * backfill._MILLISECONDS_IN_DAY
    assert job_hourly["jitter_seconds"] == 0


# --------------------- manifest health tests ---------------------

class _CollectingRouter:
    def __init__(self) -> None:
        self.messages = []

    def dispatch(self, message):
        self.messages.append(message)


def _build_universe(symbol: str, interval: str) -> InstrumentUniverseConfig:
    return InstrumentUniverseConfig(
        name="test",
        description="test",
        instruments=(
            InstrumentConfig(
                name=symbol,
                base_asset=symbol.split("_")[0],
                quote_asset=symbol.split("_")[1],
                categories=("core",),
                exchange_symbols={"binance_spot": symbol.replace("_", "")},
                backfill_windows=(
                    InstrumentBackfillWindow(interval=interval, lookback_days=30),
                ),
            ),
        ),
    )


def test_report_manifest_health_does_not_alert_when_everything_ok(tmp_path):
    manifest = tmp_path / "manifest.sqlite"
    storage = SQLiteCacheStorage(manifest, store_rows=False)
    universe = _build_universe("BTC_USDT", "1h")
    router = _CollectingRouter()
    as_of = datetime(2024, 1, 1, 12, 0, tzinfo=timezone.utc)

    metadata = storage.metadata()
    metadata["last_timestamp::BTCUSDT::1h"] = str(int((as_of - timedelta(minutes=30)).timestamp() * 1000))
    metadata["row_count::BTCUSDT::1h"] = "720"

    backfill._report_manifest_health(
        manifest_path=manifest,
        universe=universe,
        exchange_name="binance_spot",
        environment_name="binance_paper",
        alert_router=router,
        data_quality=None,
        as_of=as_of,
    )

    assert router.messages == []


def test_report_manifest_health_emits_warning_for_long_gap(tmp_path):
    manifest = tmp_path / "manifest.sqlite"
    storage = SQLiteCacheStorage(manifest, store_rows=False)
    universe = InstrumentUniverseConfig(
        name="test",
        description="test",
        instruments=(
            InstrumentConfig(
                name="BTC_USDT",
                base_asset="BTC",
                quote_asset="USDT",
                categories=("core",),
                exchange_symbols={"binance_spot": "BTCUSDT"},
                backfill_windows=(
                    InstrumentBackfillWindow(interval="1d", lookback_days=30),
                    InstrumentBackfillWindow(interval="1h", lookback_days=30),
                ),
            ),
        ),
    )
    router = _CollectingRouter()
    as_of = datetime(2024, 1, 1, 12, 0, tzinfo=timezone.utc)

    metadata = storage.metadata()
    metadata["last_timestamp::BTCUSDT::1d"] = str(int((as_of - timedelta(hours=12)).timestamp() * 1000))
    metadata["row_count::BTCUSDT::1d"] = "365"
    metadata["last_timestamp::BTCUSDT::1h"] = str(int((as_of - timedelta(hours=3)).timestamp() * 1000))
    metadata["row_count::BTCUSDT::1h"] = "720"

    backfill._report_manifest_health(
        manifest_path=manifest,
        universe=universe,
        exchange_name="binance_spot",
        environment_name="binance_paper",
        alert_router=router,
        data_quality=None,
        as_of=as_of,
    )

    assert len(router.messages) == 1
    message = router.messages[0]
    assert message.severity == "warning"
    assert "BTCUSDT" in message.body
    assert message.context["environment"] == "binance_paper"


def test_report_manifest_health_alerts_on_threshold_only(tmp_path):
    manifest = tmp_path / "manifest.sqlite"
    storage = SQLiteCacheStorage(manifest, store_rows=False)
    universe = _build_universe("BTC_USDT", "1h")
    router = _CollectingRouter()
    as_of = datetime(2024, 1, 1, 12, 0, tzinfo=timezone.utc)

    metadata = storage.metadata()
    metadata["last_timestamp::BTCUSDT::1h"] = str(int((as_of - timedelta(minutes=60)).timestamp() * 1000))
    metadata["row_count::BTCUSDT::1h"] = "720"

    data_quality = EnvironmentDataQualityConfig(max_gap_minutes=30.0, min_ok_ratio=None)

    backfill._report_manifest_health(
        manifest_path=manifest,
        universe=universe,
        exchange_name="binance_spot",
        environment_name="binance_paper",
        alert_router=router,
        data_quality=data_quality,
        as_of=as_of,
    )

    assert len(router.messages) == 1
    message = router.messages[0]
    assert message.severity == "warning"
    assert "Alert progów jakości danych" in message.body
    assert "max_gap_exceeded" in message.context.get("threshold_issues", "")


def test_report_manifest_health_emits_critical_for_missing_metadata(tmp_path):
    manifest = tmp_path / "manifest.sqlite"
    SQLiteCacheStorage(manifest, store_rows=False)
    universe = _build_universe("ETH_USDT", "1h")
    router = _CollectingRouter()
    as_of = datetime(2024, 1, 1, 12, 0, tzinfo=timezone.utc)

    backfill._report_manifest_health(
        manifest_path=manifest,
        universe=universe,
        exchange_name="binance_spot",
        environment_name="binance_paper",
        alert_router=router,
        data_quality=None,
        as_of=as_of,
    )

    assert len(router.messages) == 1
    message = router.messages[0]
    assert message.severity == "critical"
    assert "ETHUSDT" in message.body


def test_main_plan_only_outputs_summary_and_skips_execution(monkeypatch, capsys):
    def _fail(*_args, **_kwargs):
        raise AssertionError("Nie powinno być wywołane w trybie plan-only")

    # W trybie plan-only nie powinno instancjować źródła/uprawnień
    monkeypatch.setattr(backfill, "_build_public_source", _fail)
    monkeypatch.setattr(backfill, "create_default_secret_storage", _fail)

    exit_code = backfill.main(
        ["--environment", "binance_paper", "--plan-only", "--log-level", "ERROR"]
    )

    captured = capsys.readouterr()

    assert exit_code == 0
    assert "Plan backfillu" in captured.out


def test_report_manifest_health_prints_table(monkeypatch, tmp_path, capsys):
    entries = [
        ManifestEntry(
            symbol="BTCUSDT",
            interval="1d",
            row_count=3650,
            last_timestamp_ms=1_700_000_000_000,
            last_timestamp_iso="2023-11-14T00:00:00+00:00",
            gap_minutes=90.0,
            threshold_minutes=120,
            status="ok",
        ),
        ManifestEntry(
            symbol="ETHUSDT",
            interval="1h",
            row_count=1800,
            last_timestamp_ms=1_700_000_900_000,
            last_timestamp_iso="2023-11-14T00:15:00+00:00",
            gap_minutes=240.0,
            threshold_minutes=120,
            status="warning",
        ),
    ]

    def _status(entry: ManifestEntry) -> CoverageStatus:
        issues = ()
        if entry.status != "ok":
            issues = (f"manifest_status:{entry.status}",)
        return CoverageStatus(
            symbol=entry.symbol,
            interval=entry.interval,
            manifest_entry=entry,
            required_rows=None,
            issues=issues,
        )

    monkeypatch.setattr(
        backfill,
        "evaluate_coverage",
        lambda **_: [_status(entry) for entry in entries],
    )

    manifest = tmp_path / "manifest.sqlite"
    manifest.touch()
    universe = _build_universe("BTC_USDT", "1d")

    backfill._report_manifest_health(
        manifest_path=manifest,
        universe=universe,
        exchange_name="binance_spot",
        environment_name="binance_paper",
        alert_router=None,
        data_quality=None,
        as_of=datetime(2024, 1, 1, tzinfo=timezone.utc),
        output_format="table",
    )

    output = capsys.readouterr().out
    assert "Symbol" in output
    assert "BTCUSDT" in output and "ETHUSDT" in output
    assert "Podsumowanie statusów:" in output


def test_report_manifest_health_prints_json(monkeypatch, tmp_path, capsys):
    entries = [
        ManifestEntry(
            symbol="BTCUSDT",
            interval="1d",
            row_count=3650,
            last_timestamp_ms=1_700_000_000_000,
            last_timestamp_iso="2023-11-14T00:00:00+00:00",
            gap_minutes=30.0,
            threshold_minutes=120,
            status="ok",
        )
    ]

    def _status(entry: ManifestEntry) -> CoverageStatus:
        return CoverageStatus(
            symbol=entry.symbol,
            interval=entry.interval,
            manifest_entry=entry,
            required_rows=None,
            issues=(),
        )

    monkeypatch.setattr(
        backfill,
        "evaluate_coverage",
        lambda **_: [_status(entry) for entry in entries],
    )

    manifest = tmp_path / "manifest.sqlite"
    manifest.touch()
    universe = _build_universe("BTC_USDT", "1d")

    backfill._report_manifest_health(
        manifest_path=manifest,
        universe=universe,
        exchange_name="binance_spot",
        environment_name="binance_paper",
        alert_router=None,
        data_quality=None,
        as_of=datetime(2024, 1, 1, tzinfo=timezone.utc),
        output_format="json",
    )

    output = capsys.readouterr().out
    payload = json.loads(output)
    assert payload["environment"] == "binance_paper"
    assert payload["exchange"] == "binance_spot"
    assert payload["summary"]["ok"] == 1
    assert payload["summary"]["manifest_status_counts"]["ok"] == 1
    assert payload["summary"]["issue_counts"] == {}
    assert payload["entries"][0]["symbol"] == "BTCUSDT"
