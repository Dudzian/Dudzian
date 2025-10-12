from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
import sys
from typing import Mapping

import pytest

sys.path.append(str(Path(__file__).resolve().parents[1]))

from bot_core.config.models import (
    DailyTrendMomentumStrategyConfig,
    EnvironmentConfig,
    InstrumentBackfillWindow,
    InstrumentConfig,
    InstrumentUniverseConfig,
    RiskProfileConfig,
)
from bot_core.data.ohlcv import ManifestEntry, ManifestMetricsExporter, SQLiteCacheStorage
from bot_core.data.ohlcv.manifest_metrics import STATUS_SEVERITY, status_to_severity
from bot_core.exchanges.base import Environment
from bot_core.observability.metrics import MetricsRegistry
from bot_core.security.signing import build_hmac_signature

import scripts.export_manifest_metrics as export_manifest_metrics


def _make_entry(symbol: str, interval: str, *, status: str, gap: float, rows: int, threshold: int) -> ManifestEntry:
    return ManifestEntry(
        symbol=symbol,
        interval=interval,
        row_count=rows,
        last_timestamp_ms=1,
        last_timestamp_iso="2024-01-01T00:00:00+00:00",
        gap_minutes=gap,
        threshold_minutes=threshold,
        status=status,
    )


def test_manifest_metrics_exporter_updates_gauges() -> None:
    registry = MetricsRegistry()
    exporter = ManifestMetricsExporter(
        registry=registry,
        environment="paper_env",
        exchange="binance_spot",
        stage="paper",
        risk_profile="balanced",
    )

    entries = [
        _make_entry("BTCUSDT", "1h", status="ok", gap=10.0, rows=120, threshold=90),
        _make_entry("BTCUSDT", "1d", status="warning", gap=1800.0, rows=40, threshold=720),
    ]

    summary = exporter.observe(entries)

    assert summary["status_counts"] == {"ok": 1, "warning": 1}
    assert summary["total_entries"] == 2
    assert summary["worst_status"] == "warning"

    gap_metric = registry.get("ohlcv_manifest_gap_minutes")
    assert gap_metric.value(
        labels={
            "environment": "paper_env",
            "exchange": "binance_spot",
            "stage": "paper",
            "risk_profile": "balanced",
            "symbol": "BTCUSDT",
            "interval": "1d",
        }
    ) == pytest.approx(1800.0)

    status_metric = registry.get("ohlcv_manifest_status_code")
    assert status_metric.value(
        labels={
            "environment": "paper_env",
            "exchange": "binance_spot",
            "stage": "paper",
            "risk_profile": "balanced",
            "symbol": "BTCUSDT",
            "interval": "1d",
        }
    ) == pytest.approx(float(status_to_severity("warning")))

    totals_metric = registry.get("ohlcv_manifest_entries_total")
    assert totals_metric.value(
        labels={
            "environment": "paper_env",
            "exchange": "binance_spot",
            "stage": "paper",
            "risk_profile": "balanced",
            "status": "warning",
        }
    ) == pytest.approx(1.0)


def test_manifest_metrics_exporter_resets_missing_entries() -> None:
    registry = MetricsRegistry()
    exporter = ManifestMetricsExporter(
        registry=registry,
        environment="paper_env",
        exchange="binance_spot",
        stage="paper",
    )

    entry = _make_entry("ETHUSDT", "1h", status="warning", gap=200.0, rows=50, threshold=90)
    exporter.observe([entry])

    # Druga obserwacja bez wpisów powinna wyzerować poprzednie metryki.
    exporter.observe([])

    gap_metric = registry.get("ohlcv_manifest_gap_minutes")
    assert gap_metric.value(
        labels={
            "environment": "paper_env",
            "exchange": "binance_spot",
            "stage": "paper",
            "symbol": "ETHUSDT",
            "interval": "1h",
        }
    ) == pytest.approx(0.0)

    status_metric = registry.get("ohlcv_manifest_status_code")
    assert status_metric.value(
        labels={
            "environment": "paper_env",
            "exchange": "binance_spot",
            "stage": "paper",
            "symbol": "ETHUSDT",
            "interval": "1h",
        }
    ) == pytest.approx(float(STATUS_SEVERITY["unknown"]))


@pytest.fixture()
def simple_core_config(tmp_path: Path):
    universe = InstrumentUniverseConfig(
        name="test",
        description="",
        instruments=(
            InstrumentConfig(
                name="BTC_USDT",
                base_asset="BTC",
                quote_asset="USDT",
                categories=("core",),
                exchange_symbols={"zonda_spot": "BTC-USDT"},
                backfill_windows=(InstrumentBackfillWindow(interval="1h", lookback_days=30),),
            ),
        ),
    )

    environment = EnvironmentConfig(
        name="paper",
        exchange="zonda_spot",
        environment=Environment.PAPER,
        keychain_key="demo",
        data_cache_path=str(tmp_path),
        risk_profile="balanced",
        alert_channels=(),
        instrument_universe="test",
    )

    strategy = DailyTrendMomentumStrategyConfig(
        name="dummy",
        fast_ma=5,
        slow_ma=10,
        breakout_lookback=5,
        momentum_window=5,
        atr_window=5,
        atr_multiplier=1.5,
        min_trend_strength=0.01,
        min_momentum=0.005,
    )

    profile = RiskProfileConfig(
        name="balanced",
        max_daily_loss_pct=0.01,
        max_position_pct=0.02,
        target_volatility=0.1,
        max_leverage=2.0,
        stop_loss_atr_multiple=1.0,
        max_open_positions=3,
        hard_drawdown_pct=0.05,
    )

    class _StubCoreConfig:
        environments = {"paper": environment}
        risk_profiles = {"balanced": profile}
        instrument_universes = {"test": universe}
        strategies = {"dummy": strategy}

    return _StubCoreConfig()


def test_export_manifest_metrics_script(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, simple_core_config) -> None:
    manifest = tmp_path / "ohlcv_manifest.sqlite"
    storage = SQLiteCacheStorage(manifest, store_rows=False)

    as_of = datetime(2024, 1, 2, 0, 0, tzinfo=timezone.utc)
    last_ts = int((as_of - timedelta(minutes=30)).timestamp() * 1000)
    storage.write(
        "BTC-USDT::1h",
        {
            "rows": [
                [float(last_ts), 1.0, 1.0, 1.0, 1.0, 1.0],
            ]
        },
    )

    monkeypatch.setattr(export_manifest_metrics, "load_core_config", lambda path: simple_core_config)

    metrics_path = tmp_path / "metrics.prom"
    summary_path = tmp_path / "summary.json"

    exit_code = export_manifest_metrics.main(
        [
            "--config",
            "dummy",
            "--environment",
            "paper",
            "--manifest-path",
            str(manifest),
            "--as-of",
            as_of.isoformat(),
            "--output",
            str(metrics_path),
            "--summary-output",
            str(summary_path),
            "--symbol",
            "BTC-USDT",
            "--interval",
            "1h",
        ]
    )

    assert exit_code == 0
    assert "ohlcv_manifest_gap_minutes" in metrics_path.read_text(encoding="utf-8")

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert summary["status_counts"].get("ok", 0) == 1


def test_export_manifest_metrics_script_denies_status(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, simple_core_config) -> None:
    manifest = tmp_path / "ohlcv_manifest.sqlite"
    SQLiteCacheStorage(manifest, store_rows=False)  # inicjalizuje strukturę bez wpisów

    monkeypatch.setattr(export_manifest_metrics, "load_core_config", lambda path: simple_core_config)

    exit_code = export_manifest_metrics.main(
        [
            "--config",
            "dummy",
            "--environment",
            "paper",
            "--manifest-path",
            str(manifest),
            "--deny-status",
            "missing_metadata",
        ]
    )

    assert exit_code == 2


def test_export_manifest_metrics_script_signs_summary(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    simple_core_config,
) -> None:
    manifest = tmp_path / "ohlcv_manifest.sqlite"
    storage = SQLiteCacheStorage(manifest, store_rows=False)
    as_of = datetime(2024, 1, 2, 0, 0, tzinfo=timezone.utc)
    last_ts = int((as_of - timedelta(minutes=5)).timestamp() * 1000)
    storage.write(
        "BTC-USDT::1h",
        {"rows": [[float(last_ts), 1.0, 1.0, 1.0, 1.0, 1.0]]},
    )

    monkeypatch.setattr(export_manifest_metrics, "load_core_config", lambda path: simple_core_config)

    metrics_path = tmp_path / "metrics.prom"
    summary_path = tmp_path / "summary.json"
    key_material = "super_secret_manifest_key"
    exit_code = export_manifest_metrics.main(
        [
            "--config",
            "dummy",
            "--environment",
            "paper",
            "--manifest-path",
            str(manifest),
            "--output",
            str(metrics_path),
            "--summary-output",
            str(summary_path),
            "--summary-hmac-key",
            key_material,
            "--summary-hmac-key-id",
            "ci-manifest",
        ]
    )

    assert exit_code == 0
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    signature = summary.get("summary_signature")
    assert isinstance(signature, Mapping)

    payload_without_signature = {
        key: value for key, value in summary.items() if key != "summary_signature"
    }
    expected_signature = build_hmac_signature(
        payload_without_signature,
        key=key_material.strip().encode("utf-8"),
        key_id="ci-manifest",
    )
    assert signature == expected_signature


def test_export_manifest_metrics_script_requires_signature_when_enforced(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    simple_core_config,
) -> None:
    manifest = tmp_path / "ohlcv_manifest.sqlite"
    SQLiteCacheStorage(manifest, store_rows=False)

    monkeypatch.setattr(export_manifest_metrics, "load_core_config", lambda path: simple_core_config)

    with pytest.raises(SystemExit) as exc:
        export_manifest_metrics.main(
            [
                "--config",
                "dummy",
                "--environment",
                "paper",
                "--manifest-path",
                str(manifest),
                "--require-summary-signature",
            ]
        )

    assert exc.value.code == 2
