"""Seeduje snapshoty signal_quality i checklisty HyperCare dla adapterów futures."""

from __future__ import annotations

import argparse
from pathlib import Path

from bot_core.exchanges.bitmex.futures import BitmexFuturesAdapter
from bot_core.exchanges.deribit.futures import DeribitFuturesAdapter
from bot_core.exchanges.signal_quality import SignalQualityReporter


def _seed_reporter(adapter_cls, signal_dir: Path) -> SignalQualityReporter:
    reporter = SignalQualityReporter(
        exchange_id=adapter_cls.name,
        report_dir=signal_dir,
        enable_csv_export=True,
        csv_dir=signal_dir,
    )
    # Minimal reprezentatywny rekord, aby CSV i snapshot nie były puste.
    reporter.record_success(
        backend="rest",
        symbol="BTC/USDT",
        side="buy",
        order_type="market",
        requested_quantity=1.0,
        requested_price=50_000.0,
        filled_quantity=1.0,
        executed_price=50_005.0,
        latency=0.2,
    )
    reporter.record_failure(
        backend="long_poll",
        symbol="ETH/USDT",
        side="sell",
        order_type="limit",
        requested_quantity=0.5,
        requested_price=2_500.0,
        error=RuntimeError("snapshot seed"),
    )
    return reporter


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--signal-quality-dir",
        default="reports/exchanges/signal_quality",
        help="Katalog z plikami JSON/CSV jakości sygnałów.",
    )
    parser.add_argument(
        "--hypercare-dir",
        default="reports/exchanges/hypercare",
        help="Katalog docelowy checklist HyperCare.",
    )
    parser.add_argument(
        "--daily-csv-dir",
        default="reports/exchanges",
        help="Katalog na zbiorczy CSV checklist HyperCare.",
    )
    args = parser.parse_args(argv)

    signal_dir = Path(args.signal_quality_dir)
    hypercare_dir = Path(args.hypercare_dir)
    daily_dir = Path(args.daily_csv_dir)
    signal_dir.mkdir(parents=True, exist_ok=True)
    hypercare_dir.mkdir(parents=True, exist_ok=True)
    daily_dir.mkdir(parents=True, exist_ok=True)

    for adapter_cls in (DeribitFuturesAdapter, BitmexFuturesAdapter):
        reporter = _seed_reporter(adapter_cls, signal_dir)
        adapter_cls.export_hypercare_assets(
            signal_quality_dir=signal_dir,
            report_dir=hypercare_dir,
            daily_csv_dir=daily_dir,
            reporter=reporter,
            load_existing_snapshot=False,
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
