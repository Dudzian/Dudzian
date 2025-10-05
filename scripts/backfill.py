"""Automatyczny backfill oraz odświeżanie inkrementalne danych OHLCV."""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Mapping, Sequence

from bot_core.alerts import AlertMessage, DefaultAlertRouter, build_coverage_alert_context
from bot_core.config.loader import load_core_config
from bot_core.config.models import (
    CoreConfig,
    EnvironmentConfig,
    EnvironmentDataQualityConfig,
    InstrumentUniverseConfig,
)
from bot_core.data.ohlcv import (
    BackfillSummary,
    CachedOHLCVSource,
    CoverageStatus,
    DataGapIncidentTracker,
    DualCacheStorage,
    GapAlertPolicy,
    OHLCVBackfillService,
    OHLCVRefreshScheduler,
    ParquetCacheStorage,
    PublicAPIDataSource,
    SQLiteCacheStorage,
    evaluate_coverage,
    summarize_coverage,
)
from bot_core.data.ohlcv.audit import JSONLGapAuditLogger
from bot_core.exchanges.base import Environment, ExchangeCredentials
from bot_core.exchanges.binance.futures import BinanceFuturesAdapter
from bot_core.exchanges.binance.spot import BinanceSpotAdapter
from bot_core.exchanges.kraken.futures import KrakenFuturesAdapter
from bot_core.exchanges.kraken.spot import KrakenSpotAdapter
from bot_core.exchanges.zonda.spot import ZondaSpotAdapter
from bot_core.runtime.bootstrap import build_alert_channels
from bot_core.security import SecretManager, SecretStorageError, create_default_secret_storage

_LOGGER = logging.getLogger(__name__)

_MILLISECONDS_IN_DAY = 86_400_000

# Domyślne częstotliwości odświeżania per interwał (sekundy)
_DEFAULT_REFRESH_SECONDS: Mapping[str, int] = {
    "1d": 24 * 60 * 60,
    "1h": 15 * 60,
    "15m": 5 * 60,
}

# Opcjonalny jitter (sekundy) – aby rozproszyć starty zadań w czasie
_DEFAULT_JITTER_SECONDS: Mapping[str, int] = {
    "1d": 15 * 60,
    "1h": 2 * 60,
    "15m": 45,
}


@dataclass(slots=True)
class _IntervalPlan:
    symbols: set[str]
    backfill_start_ms: int
    incremental_lookback_ms: int
    # 0 => użyj wartości przekazanej z CLI (--refresh-seconds)
    refresh_seconds: int
    jitter_seconds: int = 0


def _build_public_source(exchange: str, environment: Environment) -> PublicAPIDataSource:
    builders: Mapping[str, Callable[[Environment], PublicAPIDataSource]] = {
        "binance_spot": lambda env: PublicAPIDataSource(
            exchange_adapter=BinanceSpotAdapter(ExchangeCredentials(key_id="public", environment=env))
        ),
        "binance_futures": lambda env: PublicAPIDataSource(
            exchange_adapter=BinanceFuturesAdapter(
                ExchangeCredentials(key_id="public", environment=env), environment=env
            )
        ),
        "kraken_spot": lambda env: PublicAPIDataSource(
            exchange_adapter=KrakenSpotAdapter(ExchangeCredentials(key_id="public", environment=env), environment=env)
        ),
        "kraken_futures": lambda env: PublicAPIDataSource(
            exchange_adapter=KrakenFuturesAdapter(
                ExchangeCredentials(key_id="public", environment=env),
                environment=env,
            )
        ),
        "zonda_spot": lambda env: PublicAPIDataSource(
            exchange_adapter=ZondaSpotAdapter(
                ExchangeCredentials(key_id="public", environment=env),
                environment=env,
            )
        ),
    }
    try:
        builder = builders[exchange]
    except KeyError as exc:  # pragma: no cover
        raise ValueError(f"Brak obsługi exchange={exchange} dla backfillu") from exc
    return builder(environment)


def _resolve_universe(core_config: CoreConfig, environment: EnvironmentConfig) -> InstrumentUniverseConfig:
    if not environment.instrument_universe:
        raise SystemExit(
            "Środowisko nie posiada przypisanego uniwersum instrumentów – zdefiniuj instrument_universe w config/core.yaml."
        )
    try:
        return core_config.instrument_universes[environment.instrument_universe]
    except KeyError as exc:
        raise SystemExit(
            f"Środowisko {environment.name} wskazuje nieistniejące uniwersum {environment.instrument_universe}."
        ) from exc


def _utc_now_ms() -> int:
    return int(datetime.now(tz=timezone.utc).timestamp() * 1000)


def _build_interval_plans(
    *,
    universe: InstrumentUniverseConfig,
    exchange_name: str,
    incremental_lookback_days: int,
    refresh_overrides: Mapping[str, int] | None = None,
    jitter_overrides: Mapping[str, int] | None = None,
) -> tuple[dict[str, _IntervalPlan], set[str]]:
    plans: dict[str, _IntervalPlan] = {}
    symbols: set[str] = set()
    now_ms = _utc_now_ms()

    # przefiltruj override’y do dodatnich intów
    refresh_cfg = {
        k: int(v)
        for k, v in (refresh_overrides or {}).items()
        if isinstance(v, (int, float)) and int(v) > 0
    }
    jitter_cfg = {
        k: max(0, int(v))
        for k, v in (jitter_overrides or {}).items()
        if isinstance(v, (int, float)) and int(v) >= 0
    }

    for instrument in universe.instruments:
        symbol = instrument.exchange_symbols.get(exchange_name)
        if not symbol:
            continue
        symbols.add(symbol)

        for window in instrument.backfill_windows:
            start = now_ms - window.lookback_days * _MILLISECONDS_IN_DAY
            plan = plans.get(window.interval)
            if plan is None:
                refresh_seconds = refresh_cfg.get(window.interval, _DEFAULT_REFRESH_SECONDS.get(window.interval, 0))
                jitter_seconds = jitter_cfg.get(window.interval, _DEFAULT_JITTER_SECONDS.get(window.interval, 0))
                plan = _IntervalPlan(
                    symbols=set(),
                    backfill_start_ms=start,
                    incremental_lookback_ms=0,
                    refresh_seconds=refresh_seconds,
                    jitter_seconds=jitter_seconds,
                )
                plans[window.interval] = plan
            plan.symbols.add(symbol)
            plan.backfill_start_ms = min(plan.backfill_start_ms, start)
            effective_days = max(1, min(window.lookback_days, incremental_lookback_days))
            plan.incremental_lookback_ms = max(plan.incremental_lookback_ms, effective_days * _MILLISECONDS_IN_DAY)

    return plans, symbols


def _format_plan_summary(
    plans: Mapping[str, _IntervalPlan],
    *,
    exchange_name: str,
    environment_name: str,
) -> str:
    """Buduje czytelną reprezentację planu backfillu dla trybu plan-only."""
    if not plans:
        return (
            f"Plan backfillu dla środowiska {environment_name} (exchange={exchange_name}):\n"
            "Brak interwałów do synchronizacji."
        )

    all_symbols: list[str] = sorted({symbol for plan in plans.values() for symbol in plan.symbols})
    preview = ", ".join(all_symbols[:10])
    if len(all_symbols) > 10:
        preview += f" … (+{len(all_symbols) - 10})"

    lines = [
        f"Plan backfillu dla środowiska {environment_name} (exchange={exchange_name}):",
        f"- symbole w uniwersum: {len(all_symbols)} -> {preview or '-'}",
    ]

    for interval, plan in sorted(plans.items(), key=lambda item: item[0]):
        start_dt = datetime.fromtimestamp(max(0, plan.backfill_start_ms) / 1000, tz=timezone.utc)
        if plan.incremental_lookback_ms <= 0:
            lookback_desc = "domyślny 1.0d"
        else:
            lookback_days = plan.incremental_lookback_ms / _MILLISECONDS_IN_DAY
            lookback_desc = f"{lookback_days:.1f}d"

        refresh_desc = f"{plan.refresh_seconds}s" if plan.refresh_seconds else "dziedziczy (--refresh-seconds)"
        jitter_desc = f"{plan.jitter_seconds}s" if plan.jitter_seconds else "0s"

        symbols_sorted = sorted(plan.symbols)
        sample = ", ".join(symbols_sorted[:5])
        if len(symbols_sorted) > 5:
            sample += f" … (+{len(symbols_sorted) - 5})"

        lines.append(
            "- "
            + (
                f"{interval}: symbole={len(symbols_sorted)}, backfill_od={start_dt.isoformat()}, "
                f"inkrementalny_lookback={lookback_desc}, refresh={refresh_desc}, jitter={jitter_desc}, "
                f"przykłady=[{sample or '-'}]"
            )
        )

    return "\n".join(lines)


def _extract_gap_policy(environment: EnvironmentConfig) -> GapAlertPolicy:
    settings: Mapping[str, object] = {}
    if isinstance(environment.adapter_settings, Mapping):
        candidate = environment.adapter_settings.get("ohlcv_gap_alerts")
        if isinstance(candidate, Mapping):
            settings = candidate

    warnings_cfg = {}
    raw_warnings = settings.get("warning_gap_minutes") if settings else None
    if isinstance(raw_warnings, Mapping):
        warnings_cfg = {
            str(interval): max(1, int(value))
            for interval, value in raw_warnings.items()
            if value is not None and int(value) > 0
        }

    def _safe_int(key: str, default: int) -> int:
        value = settings.get(key) if settings else None
        try:
            return max(1, int(value))
        except (TypeError, ValueError):
            return default

    return GapAlertPolicy(
        warning_gap_minutes=warnings_cfg,
        incident_threshold_count=_safe_int("incident_threshold_count", 5),
        incident_window_minutes=_safe_int("incident_window_minutes", 10),
        sms_escalation_minutes=_safe_int("sms_escalation_minutes", 15),
        warning_throttle_minutes=_safe_int("warning_throttle_minutes", 5),
    )


def _build_gap_callback(
    gap_tracker: DataGapIncidentTracker | None,
) -> Callable[[str, Sequence[BackfillSummary], int], None] | None:
    if gap_tracker is None:
        return None

    def _callback(interval: str, summaries: Sequence[BackfillSummary], as_of_ms: int) -> None:
        gap_tracker.handle_summaries(interval=interval, summaries=summaries, as_of_ms=as_of_ms)

    return _callback


def _report_manifest_health(
    *,
    manifest_path: Path,
    universe: InstrumentUniverseConfig,
    exchange_name: str,
    environment_name: str,
    alert_router: DefaultAlertRouter | None,
    data_quality: EnvironmentDataQualityConfig | None,
    as_of: datetime | None = None,
    output_format: str | None = None,
) -> None:
    """Loguje stan manifestu i wysyła alerty o wykrytych lukach."""
    report_as_of = as_of or datetime.now(timezone.utc)

    statuses = list(
        evaluate_coverage(
            manifest_path=manifest_path,
            universe=universe,
            exchange_name=exchange_name,
            as_of=report_as_of,
        )
    )

    if not statuses:
        _LOGGER.info(
            "Manifest OHLCV nie zawiera wpisów dla exchange=%s – pomijam raport",
            exchange_name,
        )
        return

    coverage_summary = summarize_coverage(statuses)
    summary_payload = coverage_summary.to_mapping()
    manifest_summary = dict(coverage_summary.manifest_status_counts)
    entries = [status.manifest_entry for status in statuses]

    _LOGGER.info(
        "Manifest OHLCV %s/%s – status=%s total=%s ok=%s warning=%s error=%s stale=%s",
        environment_name,
        exchange_name,
        summary_payload.get("status"),
        summary_payload.get("total"),
        summary_payload.get("ok"),
        summary_payload.get("warning"),
        summary_payload.get("error"),
        summary_payload.get("stale_entries"),
    )
    if coverage_summary.issue_counts:
        _LOGGER.info(
            "Manifest OHLCV %s/%s – issue_counts=%s",
            environment_name,
            exchange_name,
            coverage_summary.issue_counts,
        )

    if output_format:
        if output_format == "table":
            print(_format_manifest_table(entries, manifest_summary))
        elif output_format == "json":
            payload = {
                "environment": environment_name,
                "exchange": exchange_name,
                "generated_at": report_as_of.isoformat(),
                "summary": summary_payload,
                "entries": [asdict(entry) for entry in entries],
            }
            print(json.dumps(payload, ensure_ascii=False, indent=2))
        else:  # pragma: no cover - walidacja w parserze argumentów
            _LOGGER.warning("Nieobsługiwany format raportu manifestu: %s", output_format)

    issue_statuses = [status for status in statuses if status.issues]
    if not issue_statuses:
        return

    for status in issue_statuses:
        entry = status.manifest_entry
        gap_display = "-" if entry.gap_minutes is None else f"{float(entry.gap_minutes):.1f} min"
        _LOGGER.warning(
            "Manifest alert: %s %s status=%s issues=%s gap=%s rows=%s last=%s",
            entry.symbol,
            entry.interval,
            entry.status,
            ", ".join(status.issues) if status.issues else "brak",
            gap_display,
            entry.row_count if entry.row_count is not None else "-",
            entry.last_timestamp_iso or "-",
        )

    if threshold_result.thresholds:
        _LOGGER.info(
            "Progi jakości danych %s/%s – %s",
            environment_name,
            exchange_name,
            threshold_result.thresholds,
        )
    if threshold_issues:
        _LOGGER.warning(
            "Przekroczone progi jakości danych %s/%s – %s",
            environment_name,
            exchange_name,
            threshold_issues,
        )

    if not alert_router:
        return

    critical_statuses = [
        status
        for status in issue_statuses
        if status.manifest_entry.status in {"missing_metadata", "invalid_metadata"}
    ]
    critical_keys = {(status.symbol, status.interval) for status in critical_statuses}
    warning_statuses = [
        status
        for status in issue_statuses
        if (status.symbol, status.interval) not in critical_keys
    ]

    worst_gap_payload = coverage_summary.worst_gap

    def _format_alert_body(statuses_seq: Sequence[CoverageStatus]) -> str:
        lines = ["Problemy w manifeście OHLCV:"]
        for status in list(statuses_seq)[:10]:
            entry = status.manifest_entry
            gap_display = "-" if entry.gap_minutes is None else f"{float(entry.gap_minutes):.1f} min"
            threshold_display = (
                "-" if entry.threshold_minutes is None else str(entry.threshold_minutes)
            )
            row_display = "-" if entry.row_count is None else str(entry.row_count)
            required_display = "-" if status.required_rows is None else str(status.required_rows)
            last_display = entry.last_timestamp_iso or "-"
            issue_display = ", ".join(status.issues) if status.issues else entry.status
            lines.append(
                f"- {entry.symbol} {entry.interval} – {issue_display}; gap={gap_display}, "
                f"próg={threshold_display}, wiersze={row_display}, wymagane={required_display}, "
                f"ostatnia={last_display} UTC"
            )
        if len(statuses_seq) > 10:
            lines.append(f"… oraz {len(statuses_seq) - 10} kolejnych wpisów.")
        if worst_gap_payload:
            lines.append(
                "Największa luka: {symbol}/{interval} {gap} min (próg {threshold})".format(
                    symbol=worst_gap_payload.get("symbol", "?"),
                    interval=worst_gap_payload.get("interval", "?"),
                    gap=worst_gap_payload.get("gap_minutes", "?"),
                    threshold=worst_gap_payload.get("threshold_minutes", "-"),
                )
            )
        lines.append("Szczegóły w logach backfillu oraz pliku manifestu.")
        return "\n".join(lines)

    coverage_context = build_coverage_alert_context(
        summary=coverage_summary,
        threshold_result=threshold_result,
    )
    context = {
        "environment": environment_name,
        "exchange": exchange_name,
        "issues": str(len(issue_statuses)),
        "summary": json.dumps(summary_payload, ensure_ascii=False),
        "stale_entries": str(coverage_summary.stale_entries),
    }
    if coverage_summary.issue_counts:
        context["issue_counts"] = json.dumps(coverage_summary.issue_counts, ensure_ascii=False)
    if coverage_summary.issue_examples:
        context["issue_examples"] = json.dumps(
            coverage_summary.issue_examples, ensure_ascii=False
        )
    if coverage_summary.worst_gap:
        context["worst_gap"] = json.dumps(coverage_summary.worst_gap, ensure_ascii=False)

    if critical_statuses:
        alert_router.dispatch(
            AlertMessage(
                category="data.ohlcv",
                title=f"Krytyczne braki w manifeście OHLCV ({environment_name})",
                body=_format_alert_body(critical_statuses),
                severity="critical",
                context=context,
            )
        )
        alert_sent = True

    if warning_statuses:
        alert_router.dispatch(
            AlertMessage(
                category="data.ohlcv",
                title=f"Ostrzeżenia w manifeście OHLCV ({environment_name})",
                body=_format_alert_body(warning_statuses),
                severity="warning",
                context=context,
            )
        )
        alert_sent = True

    if not alert_sent and threshold_issues:
        severity = "warning"
        if any(issue.startswith("ok_ratio_below_threshold") for issue in threshold_issues):
            severity = "critical"
        alert_router.dispatch(
            AlertMessage(
                category="data.ohlcv",
                title=f"Alert progów jakości danych OHLCV ({environment_name})",
                body=_format_alert_body((), threshold_result),
                severity=severity,
                context=context,
            )
        )


def _initialize_alerting(
    *,
    args: argparse.Namespace,
    config: CoreConfig,
    environment: EnvironmentConfig,
) -> tuple[DefaultAlertRouter | None, GapAlertPolicy | None, str]:
    if not args.enable_alerts:
        return None, None, "Alerty wyłączone flagą CLI"

    try:
        storage = create_default_secret_storage(
            namespace=args.secret_namespace,
            headless_passphrase=args.headless_passphrase,
            headless_path=args.headless_secrets_path,
        )
    except SecretStorageError as exc:
        return None, None, f"Nie udało się przygotować magazynu sekretów: {exc}"

    secret_manager = SecretManager(storage, namespace=args.secret_namespace)

    try:
        _, router, _ = build_alert_channels(
            core_config=config,
            environment=environment,
            secret_manager=secret_manager,
        )
    except SecretStorageError as exc:
        return None, None, f"Nie udało się zbudować kanałów alertów: {exc}"

    policy = _extract_gap_policy(environment)
    return router, policy, "Kanały alertowe zainicjalizowane"


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Backfill danych OHLCV zgodnie z config/core.yaml")
    parser.add_argument("--config", default="config/core.yaml", help="Ścieżka do pliku konfiguracyjnego CoreConfig")
    parser.add_argument("--environment", default="binance_paper", help="Nazwa środowiska do backfillu")
    parser.add_argument(
        "--refresh-seconds",
        type=int,
        default=900,
        help="Częstotliwość odświeżania inkrementalnego (w sekundach)",
    )
    parser.add_argument(
        "--incremental-lookback-days",
        type=int,
        default=3,
        help="Ile dni historii pobierać przy odświeżaniu inkrementalnym",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Poziom logowania",
    )
    parser.add_argument(
        "--run-once",
        action="store_true",
        help="Wykonaj tylko pełny backfill i zakończ (bez harmonogramu)",
    )
    parser.add_argument(
        "--plan-only",
        action="store_true",
        help="Wypisz plan backfillu i zakończ bez pobierania danych",
    )
    parser.add_argument(
        "--enable-alerts",
        action="store_true",
        help="Aktywuj wysyłkę alertów o lukach danych (wymaga skonfigurowanych sekretów)",
    )
    parser.add_argument(
        "--secret-namespace",
        default="dudzian.trading",
        help="Namespace używany przy odczycie sekretów (keychain / plik szyfrowany)",
    )
    parser.add_argument(
        "--headless-passphrase",
        default=None,
        help="Hasło do magazynu sekretów w środowiskach headless (Linux).",
    )
    parser.add_argument(
        "--headless-secrets-path",
        default=None,
        help="Ścieżka do zaszyfrowanego magazynu sekretów w trybie headless.",
    )
    parser.add_argument(
        "--manifest-report-format",
        choices=["none", "table", "json"],
        default="none",
        help="Opcjonalny wydruk raportu manifestu po backfillu (tabela lub JSON).",
    )
    return parser.parse_args(argv)


def _perform_backfill(
    *,
    service: OHLCVBackfillService,
    plans: Mapping[str, _IntervalPlan],
    end_timestamp: int,
    gap_tracker: DataGapIncidentTracker | None = None,
) -> None:
    for interval, plan in plans.items():
        start = max(0, plan.backfill_start_ms)
        _LOGGER.info(
            "Backfill interval=%s, start=%s, end=%s, symbole=%s",
            interval,
            start,
            end_timestamp,
            ",".join(sorted(plan.symbols)),
        )
        summaries = service.synchronize(
            symbols=tuple(sorted(plan.symbols)),
            interval=interval,
            start=start,
            end=end_timestamp,
        )
        if gap_tracker:
            gap_tracker.handle_summaries(interval=interval, summaries=summaries, as_of_ms=end_timestamp)
        total = sum(summary.fetched_candles for summary in summaries)
        _LOGGER.info(
            "Zakończono backfill dla interval=%s – pobrano %s nowych świec",
            interval,
            total,
        )


async def _run_scheduler(
    *,
    scheduler: OHLCVRefreshScheduler,
    plans: Mapping[str, _IntervalPlan],
    refresh_seconds: int,
) -> None:
    for interval, plan in plans.items():
        frequency = plan.refresh_seconds or refresh_seconds
        scheduler.add_job(
            symbols=tuple(sorted(plan.symbols)),
            interval=interval,
            lookback_ms=plan.incremental_lookback_ms or (_MILLISECONDS_IN_DAY * 1),
            frequency_seconds=frequency,
            jitter_seconds=plan.jitter_seconds,
            name=f"{interval}:{len(plan.symbols)}",
        )
        _LOGGER.debug(
            "Zarejestrowano zadanie interval=%s, refresh_seconds=%s, lookback_ms=%s, jitter_seconds=%s",
            interval,
            frequency,
            plan.incremental_lookback_ms,
            plan.jitter_seconds,
        )

    _LOGGER.info(
        "Uruchamiam harmonogram odświeżania (%s zadań, domyślna częstotliwość %s sekund)",
        len(plans),
        refresh_seconds,
    )
    try:
        await scheduler.run_forever()
    finally:
        scheduler.stop()


def _format_manifest_table(
    entries: Sequence[object],
    summary: Mapping[str, int],
) -> str:
    headers = ["Symbol", "Interval", "Status", "Gap[min]", "Threshold[min]", "Rows", "Last UTC"]
    rows: list[list[str]] = []

    for entry in entries:
        gap_display = "-" if entry.gap_minutes is None else f"{entry.gap_minutes:.1f}"
        threshold_display = "-" if entry.threshold_minutes is None else str(entry.threshold_minutes)
        row_display = "-" if entry.row_count is None else str(entry.row_count)
        rows.append(
            [
                entry.symbol,
                entry.interval,
                entry.status,
                gap_display,
                threshold_display,
                row_display,
                entry.last_timestamp_iso or "-",
            ]
        )

    widths = [len(header) for header in headers]
    for row in rows:
        for idx, value in enumerate(row):
            widths[idx] = max(widths[idx], len(value))

    def _format_row(row_vals: Sequence[str]) -> str:
        return "  ".join(val.ljust(widths[idx]) for idx, val in enumerate(row_vals))

    lines = [_format_row(headers)]
    lines.append("  ".join("-" * width for width in widths))
    lines.extend(_format_row(row) for row in rows)

    if summary:
        summary_items = ", ".join(f"{key}={value}" for key, value in sorted(summary.items()))
        lines.append("")
        lines.append(f"Podsumowanie statusów: {summary_items}")

    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    logging.basicConfig(level=getattr(logging, args.log_level.upper()))

    config = load_core_config(args.config)
    try:
        environment = config.environments[args.environment]
    except KeyError as exc:
        raise SystemExit(f"Nie znaleziono środowiska {args.environment} w konfiguracji") from exc

    universe = _resolve_universe(config, environment)

    # odczyt override’ów częstotliwości i jittera z adapter_settings (opcjonalnie)
    refresh_overrides: Mapping[str, int] = {}
    jitter_overrides: Mapping[str, int] = {}
    if isinstance(environment.adapter_settings, Mapping):
        raw_refresh = (
            environment.adapter_settings.get("ohlcv_refresh_seconds")
            or environment.adapter_settings.get("ohlcv_refresh_overrides")
        )
        if isinstance(raw_refresh, Mapping):
            refresh_overrides = raw_refresh
        raw_jitter = environment.adapter_settings.get("ohlcv_refresh_jitter")
        if isinstance(raw_jitter, Mapping):
            jitter_overrides = raw_jitter

    plans, symbols = _build_interval_plans(
        universe=universe,
        exchange_name=environment.exchange,
        incremental_lookback_days=max(1, args.incremental_lookback_days),
        refresh_overrides=refresh_overrides,
        jitter_overrides=jitter_overrides,
    )
    if not plans:
        _LOGGER.warning("Brak instrumentów z zakresem backfill dla giełdy %s", environment.exchange)
        return 0

    # Tryb „plan only” — nie dotyka sekretów ani sieci
    if args.plan_only:
        summary = _format_plan_summary(
            plans,
            exchange_name=environment.exchange,
            environment_name=environment.name,
        )
        print(summary)
        return 0

    cache_root = Path(environment.data_cache_path)
    parquet_storage = ParquetCacheStorage(cache_root / "ohlcv_parquet", namespace=environment.exchange)
    manifest_path = cache_root / "ohlcv_manifest.sqlite"
    manifest_storage = SQLiteCacheStorage(manifest_path, store_rows=False)
    storage = DualCacheStorage(primary=parquet_storage, manifest=manifest_storage)
    audit_logger = JSONLGapAuditLogger(cache_root / "audit" / f"{environment.name}_ohlcv_gaps.jsonl")

    # Alerty + polityka luk
    alert_router, gap_policy, alert_message = _initialize_alerting(
        args=args,
        config=config,
        environment=environment,
    )
    gap_tracker: DataGapIncidentTracker | None = None
    if alert_router and gap_policy:
        gap_tracker = DataGapIncidentTracker(
            router=alert_router,
            metadata_provider=storage.metadata,
            policy=gap_policy,
            environment_name=environment.name,
            exchange=environment.exchange,
            audit_logger=audit_logger,
        )
        if alert_message:
            _LOGGER.info(alert_message)
    elif alert_message:
        level = logging.INFO if not args.enable_alerts else logging.ERROR
        _LOGGER.log(level, alert_message)

    upstream_source = _build_public_source(environment.exchange, environment.environment)
    upstream_source.exchange_adapter.configure_network(ip_allowlist=environment.ip_allowlist)

    cached_source = CachedOHLCVSource(storage=storage, upstream=upstream_source)
    cached_source.warm_cache(symbols, plans.keys())

    backfill_service = OHLCVBackfillService(cached_source)
    now_ts = _utc_now_ms()
    _perform_backfill(
        service=backfill_service,
        plans=plans,
        end_timestamp=now_ts,
        gap_tracker=gap_tracker,
    )

    manifest_format = None if args.manifest_report_format == "none" else args.manifest_report_format
    _report_manifest_health(
        manifest_path=manifest_path,
        universe=universe,
        exchange_name=environment.exchange,
        environment_name=environment.name,
        alert_router=alert_router,
        data_quality=getattr(environment, "data_quality", None),
        as_of=datetime.fromtimestamp(now_ts / 1000, tz=timezone.utc),
        output_format=manifest_format,
    )

    if args.run_once:
        _LOGGER.info("Tryb run-once – kończę po pełnym backfillu")
        return 0

    scheduler = OHLCVRefreshScheduler(
        backfill_service,
        on_job_complete=_build_gap_callback(gap_tracker),
    )
    try:
        asyncio.run(
            _run_scheduler(
                scheduler=scheduler,
                plans=plans,
                refresh_seconds=args.refresh_seconds,
            )
        )
    except KeyboardInterrupt:  # pragma: no cover
        _LOGGER.info("Przerwano przez użytkownika – zamykam harmonogram")

    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
