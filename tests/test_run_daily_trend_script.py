"""CLI do uruchamiania pipeline'u strategii Daily Trend w trybie paper/testnet."""
from __future__ import annotations

import argparse
import json
import os
import logging
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from collections.abc import Iterable, Mapping, Sequence
from typing import Any, cast

from bot_core.alerts import AlertMessage
from bot_core.exchanges.base import (
    AccountSnapshot,
    Environment,
    ExchangeAdapter,
    ExchangeAdapterFactory,
    ExchangeCredentials,
    OrderResult,
)
from bot_core.reporting.upload import SmokeArchiveUploader
from bot_core.data.ohlcv import evaluate_coverage
from bot_core.runtime.bootstrap import parse_adapter_factory_cli_specs
from bot_core.runtime.pipeline import build_daily_trend_pipeline, create_trading_controller
from bot_core.runtime.realtime import DailyTrendRealtimeRunner
from scripts.run_daily_trend import _format_percentage
import scripts.run_daily_trend as daily_trend
from bot_core.security import SecretStorageError
from scripts._cli_common import create_secret_manager

_LOGGER = logging.getLogger(__name__)


_parse_args = daily_trend._parse_args


_create_secret_manager = create_secret_manager


_log_order_results = daily_trend._log_order_results


_parse_iso_date = daily_trend._parse_iso_date


_resolve_date_window = daily_trend._resolve_date_window


_hash_file = daily_trend._hash_file


_as_float = daily_trend._as_float


_as_int = daily_trend._as_int


_format_money = daily_trend._format_money


_normalize_position_entry = daily_trend._normalize_position_entry


_compute_ledger_metrics = daily_trend._compute_ledger_metrics


_export_smoke_report = daily_trend._export_smoke_report


_write_smoke_readme = daily_trend._write_smoke_readme


_archive_smoke_report = daily_trend._archive_smoke_report


_render_smoke_summary = daily_trend._render_smoke_summary


_ensure_smoke_cache = daily_trend._ensure_smoke_cache


_verify_manifest_coverage = daily_trend._verify_manifest_coverage


_offline_adapter_factory = daily_trend._offline_adapter_factory


_run_loop = daily_trend._run_loop


_collect_storage_health = daily_trend._collect_storage_health


_prepare_smoke_report_directory = daily_trend._prepare_smoke_report_directory


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    logging.basicConfig(level=getattr(logging, args.log_level.upper()), stream=sys.stdout)

    try:
        secret_manager = _create_secret_manager(args)
    except SecretStorageError as exc:
        _LOGGER.error("Nie udało się zainicjalizować magazynu sekretów: %s", exc)
        return 2

    config_path = Path(args.config)
    if not config_path.exists():
        _LOGGER.error("Plik konfiguracyjny %s nie istnieje", config_path)
        return 1

    cli_adapter_specs = parse_adapter_factory_cli_specs(
        getattr(args, "adapter_factories", None)
    )
    adapter_factories_payload: dict[str, object] | None = None
    if args.paper_smoke:
        adapter_factories_payload = {
            "binance_spot": _offline_adapter_factory,
            "binance_futures": _offline_adapter_factory,
            "kraken_spot": _offline_adapter_factory,
            "kraken_futures": _offline_adapter_factory,
            "zonda_spot": _offline_adapter_factory,
        }

    if cli_adapter_specs:
        if adapter_factories_payload is None:
            adapter_factories_payload = {}
        adapter_factories_payload.update(cli_adapter_specs)

    adapter_factories: Mapping[str, object] | None = (
        adapter_factories_payload if adapter_factories_payload else None
    )

    try:
        pipeline = build_daily_trend_pipeline(
            environment_name=args.environment,
            strategy_name=args.strategy,
            controller_name=args.controller,
            config_path=config_path,
            secret_manager=secret_manager,
            adapter_factories=cast(
                Mapping[str, ExchangeAdapterFactory] | None, adapter_factories
            ),
            risk_profile_name=args.risk_profile,
        )
    except Exception as exc:  # noqa: BLE001
        _LOGGER.exception("Nie udało się zbudować pipeline'u daily trend: %s", exc)
        return 1

    # Bezpieczne logowanie (np. w testach mock może nie mieć pól)
    strategy_name = getattr(pipeline, "strategy_name", args.strategy)
    controller_name = getattr(pipeline, "controller_name", args.controller)
    _LOGGER.info(
        "Pipeline gotowy: środowisko=%s, strategia=%s, kontroler=%s",
        args.environment,
        strategy_name,
        controller_name,
    )

    environment = pipeline.bootstrap.environment.environment
    if environment is Environment.LIVE and not args.allow_live:
        _LOGGER.error(
            "Środowisko %s to LIVE – dla bezpieczeństwa użyj --allow-live po wcześniejszych testach paper.",
            args.environment,
        )
        return 3

    if args.dry_run:
        _LOGGER.info("Dry-run zakończony sukcesem. Pipeline gotowy do uruchomienia.")
        return 0

    if args.paper_smoke:
        try:
            start_ms, end_ms, window_meta = _resolve_date_window(args.date_window)
        except ValueError as exc:
            _LOGGER.error("Niepoprawny zakres dat: %s", exc)
            return 1

        end_dt = datetime.fromisoformat(window_meta["end"])
        tick_seconds = float(getattr(pipeline.controller, "tick_seconds", 86400.0) or 86400.0)
        tick_ms = max(1, int(tick_seconds * 1000))
        window_duration_ms = max(0, end_ms - start_ms)
        approx_bars = max(1, int(window_duration_ms / tick_ms) + 1)
        history_bars = max(1, min(int(args.history_bars), approx_bars))
        runner_start_ms = max(0, end_ms - history_bars * tick_ms)
        sync_start = min(start_ms, runner_start_ms)

        _LOGGER.info(
            "Startuję smoke test paper trading dla %s w zakresie %s – %s.",
            args.environment,
            window_meta["start"],
            window_meta["end"],
        )

        required_bars = max(history_bars, max(1, int((end_ms - sync_start) / tick_ms) + 1))
        data_checks: Mapping[str, object] | None = None
        try:
            data_checks = _ensure_smoke_cache(
                pipeline=pipeline,
                symbols=pipeline.controller.symbols,
                interval=pipeline.controller.interval,
                start_ms=sync_start,
                end_ms=end_ms,
                required_bars=required_bars,
                tick_ms=tick_ms,
            )
        except RuntimeError as exc:
            _LOGGER.error("%s", exc)
            return 1

        pipeline.backfill_service.synchronize(
            symbols=pipeline.controller.symbols,
            interval=pipeline.controller.interval,
            start=sync_start,
            end=end_ms,
        )

        trading_controller = create_trading_controller(
            pipeline, pipeline.bootstrap.alert_router, health_check_interval=0.0
        )

        runner = DailyTrendRealtimeRunner(
            controller=pipeline.controller,
            trading_controller=trading_controller,
            history_bars=history_bars,
            clock=lambda end=end_dt: end,
        )

        results = runner.run_once()
        if results:
            _log_order_results(results)
        else:
            _LOGGER.info("Smoke test zakończony – brak sygnałów w zadanym oknie.")

        report_dir = _prepare_smoke_report_directory(args.smoke_output)
        storage_info = _collect_storage_health(report_dir, min_free_mb=args.smoke_min_free_mb)
        alert_snapshot = pipeline.bootstrap.alert_router.health_snapshot()

        # Snapshot stanu ryzyka (opcjonalnie)
        risk_snapshot: Mapping[str, object] | None = None
        try:
            risk_engine = getattr(pipeline.bootstrap, "risk_engine", None)
            if risk_engine is not None and hasattr(risk_engine, "snapshot_state"):
                risk_snapshot = risk_engine.snapshot_state(
                    pipeline.risk_profile_name
                    if hasattr(pipeline, "risk_profile_name")
                    else pipeline.bootstrap.environment.risk_profile
                )
        except NotImplementedError:
            _LOGGER.warning("Silnik ryzyka nie udostępnia metody snapshot_state – pomijam stan ryzyka")
        except Exception as exc:  # noqa: BLE001
            _LOGGER.warning("Nie udało się pobrać stanu ryzyka: %s", exc)

        core_config = getattr(pipeline.bootstrap, "core_config", None)
        reporting_source = core_config
        if reporting_source is not None and hasattr(reporting_source, "reporting"):
            reporting_source = getattr(reporting_source, "reporting", None)
        upload_cfg = SmokeArchiveUploader.resolve_config(reporting_source)

        # Wywołanie kompatybilne z testami monkeypatchującymi _export_smoke_report
        try:
            summary_path = _export_smoke_report(
                report_dir=report_dir,
                results=results,
                ledger=pipeline.execution_service.ledger(),
                window=window_meta,
                environment=args.environment,
                alert_snapshot=alert_snapshot,
                risk_state=risk_snapshot,
                data_checks=data_checks,
                storage_info=storage_info,
            )
        except TypeError:
            try:
                summary_path = _export_smoke_report(
                    report_dir=report_dir,
                    results=results,
                    ledger=pipeline.execution_service.ledger(),
                    window=window_meta,
                    environment=args.environment,
                    alert_snapshot=alert_snapshot,
                    risk_state=risk_snapshot,
                    data_checks=data_checks,
                )
            except TypeError:
                summary_path = _export_smoke_report(
                    report_dir=report_dir,
                    results=results,
                    ledger=pipeline.execution_service.ledger(),
                    window=window_meta,
                    environment=args.environment,
                    alert_snapshot=alert_snapshot,
                    risk_state=risk_snapshot,
                )

        summary_hash = _hash_file(summary_path)
        try:
            summary_payload = json.loads(summary_path.read_text(encoding="utf-8"))
        except Exception as exc:  # noqa: BLE001
            _LOGGER.error("Nie udało się odczytać summary.json: %s", exc)
            summary_payload = {
                "environment": args.environment,
                "window": dict(window_meta),
                "orders": [],
                "ledger_entries": 0,
                "alert_snapshot": alert_snapshot,
                "risk_state": risk_snapshot or {},
            }

        summary_text = _render_smoke_summary(summary=summary_payload, summary_sha256=summary_hash)
        summary_txt_path = summary_path.with_suffix(".txt")
        summary_txt_path.write_text(summary_text + "\n", encoding="utf-8")
        readme_path = _write_smoke_readme(report_dir)
        _LOGGER.info("Raport smoke testu zapisany w %s (summary sha256=%s)", report_dir, summary_hash)
        _LOGGER.info("Podsumowanie smoke testu:%s%s", os.linesep, summary_text)

        archive_path: Path | None = None
        archive_required = bool(args.archive_smoke or upload_cfg)
        if archive_required:
            archive_path = _archive_smoke_report(report_dir)
            if args.archive_smoke:
                _LOGGER.info("Utworzono archiwum smoke testu: %s", archive_path)
            else:
                _LOGGER.info("Archiwum smoke testu wygenerowane automatycznie na potrzeby uploadu: %s", archive_path)

        upload_result = None
        if upload_cfg and archive_path:
            try:
                uploader = SmokeArchiveUploader(upload_cfg, secret_manager=secret_manager)
                upload_result = uploader.upload(
                    archive_path,
                    environment=args.environment,
                    summary_sha256=summary_hash,
                    window=window_meta,
                )
                _LOGGER.info("Przesłano archiwum smoke testu (%s) do %s", upload_result.backend, upload_result.location)
            except Exception as exc:  # noqa: BLE001
                _LOGGER.error("Nie udało się przesłać archiwum smoke testu: %s", exc)

        storage_context: dict[str, str] = {}
        storage_status = None
        if isinstance(storage_info, Mapping):
            storage_status = str(storage_info.get("status", ""))
            storage_context = {"storage_status": storage_status}
            free_mb = storage_info.get("free_mb")
            if free_mb is not None:
                storage_context["storage_free_mb"] = f"{float(free_mb):.2f}"
            threshold_mb = storage_info.get("threshold_mb")
            if threshold_mb is not None:
                storage_context["storage_threshold_mb"] = f"{float(threshold_mb):.2f}"

        message = AlertMessage(
            category="paper_smoke",
            title=f"Smoke test paper trading ({args.environment})",
            body=(
                "Zakończono smoke test paper trading."
                f" Zamówienia: {len(results)}, raport: {summary_path},"
                f" sha256: {summary_hash}"
            ),
            severity="info",
            context={
                "environment": args.environment,
                "report_dir": str(report_dir),
                "orders": str(len(results)),
                "summary_sha256": summary_hash,
                "summary_text_path": str(summary_txt_path),
                "readme_path": str(readme_path),
                **({"archive_path": str(archive_path)} if archive_path else {}),
                **(
                    {
                        "archive_upload_backend": upload_result.backend,
                        "archive_upload_location": upload_result.location,
                    }
                    if upload_result
                    else {}
                ),
                **storage_context,
            },
        )
        pipeline.bootstrap.alert_router.dispatch(message)
        return 0

    trading_controller = create_trading_controller(
        pipeline, pipeline.bootstrap.alert_router, health_check_interval=args.health_interval
    )

    runner = DailyTrendRealtimeRunner(
        controller=pipeline.controller,
        trading_controller=trading_controller,
        history_bars=max(1, args.history_bars),
    )

    if args.run_once:
        _LOGGER.info("Uruchamiam pojedynczą iterację strategii dla środowiska %s", args.environment)
        results = runner.run_once()
        if results:
            _log_order_results(results)
        else:
            _LOGGER.info("Brak sygnałów w tej iteracji – nic nie zlecam.")
        return 0

    return _run_loop(runner, args.poll_seconds)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
