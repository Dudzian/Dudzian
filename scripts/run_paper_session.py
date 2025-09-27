"""Uruchamia pojedynczą iterację sesji paper trading na podstawie konfiguracji."""
from __future__ import annotations

import argparse
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Mapping, MutableMapping, Sequence

from bot_core.config.models import (
    CoreConfig,
    DailyTrendMomentumStrategyConfig,
    InstrumentUniverseConfig,
    RiskProfileConfig,
)
from bot_core.data.base import OHLCVRequest, OHLCVResponse
from bot_core.data.ohlcv import CachedOHLCVSource, PublicAPIDataSource, SQLiteCacheStorage
from bot_core.execution.paper import MarketMetadata, PaperTradingExecutionService
from bot_core.exchanges.base import AccountSnapshot, ExchangeAdapter, ExchangeCredentials
from bot_core.risk.profiles.manual import ManualProfile
from bot_core.runtime.bootstrap import bootstrap_environment
from bot_core.runtime.session import InstrumentConfig, TradingSession
from bot_core.security import SecretManager, SecretStorageError, create_default_secret_storage
from bot_core.strategies.base import MarketSnapshot
from bot_core.strategies.daily_trend import DailyTrendMomentumSettings, DailyTrendMomentumStrategy

_LOGGER = logging.getLogger("scripts.run_paper_session")


class _PaperAccountAdapter(ExchangeAdapter):
    """Udostępnia wirtualny rachunek paper trading dla modułu runtime."""

    def __init__(
        self,
        credentials: ExchangeCredentials,
        execution: PaperTradingExecutionService,
        *,
        default_quote_asset: str,
    ) -> None:
        super().__init__(credentials)
        self._execution = execution
        self._default_quote = default_quote_asset

    def configure_network(self, *, ip_allowlist: Sequence[str] | None = None) -> None:  # noqa: D401, ARG002
        """Adapter paper trading nie wymaga dodatkowej konfiguracji sieciowej."""

    def fetch_account_snapshot(self) -> AccountSnapshot:
        balances = self._execution.balances()
        total_equity = sum(balances.values())
        available_margin = balances.get(self._default_quote, total_equity)
        return AccountSnapshot(
            balances=balances,
            total_equity=total_equity,
            available_margin=available_margin,
            maintenance_margin=0.0,
        )

    def fetch_symbols(self) -> Iterable[str]:  # pragma: no cover - nieużywane w tym skrypcie
        return ()

    def fetch_ohlcv(
        self,
        symbol: str,
        interval: str,
        start: int | None = None,
        end: int | None = None,
        limit: int | None = None,
    ) -> Sequence[Sequence[float]]:  # pragma: no cover - ochrona przed przypadkowym użyciem
        raise NotImplementedError("Adapter paper trading nie obsługuje pobierania danych OHLCV.")

    def place_order(self, request: "OrderRequest") -> "OrderResult":  # pragma: no cover - bezpieczeństwo
        raise RuntimeError(
            "Zlecenia należy kierować przez PaperTradingExecutionService – adapter nie wykonuje tradingu."
        )

    def cancel_order(self, order_id: str, *, symbol: str | None = None) -> None:  # pragma: no cover - bezpieczeństwo
        raise RuntimeError("Anulacje nie są obsługiwane przez adapter paper trading.")

    def stream_public_data(self, *, channels: Sequence[str]):  # pragma: no cover - bezpieczeństwo
        raise NotImplementedError("Streaming danych nie jest wspierany w trybie paper.")

    def stream_private_data(self, *, channels: Sequence[str]):  # pragma: no cover - bezpieczeństwo
        raise NotImplementedError("Streaming danych nie jest wspierany w trybie paper.")


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Uruchamia pojedynczą iterację paper tradingu.")
    parser.add_argument("--config", default="config/core.yaml", help="Ścieżka do pliku konfiguracji")
    parser.add_argument("--environment", default="binance_paper", help="Nazwa środowiska z konfiguracji")
    parser.add_argument("--strategy", default="core_daily_trend", help="Strategia do uruchomienia")
    parser.add_argument("--symbol", help="Symbol giełdowy – domyślnie pierwszy z uniwersum środowiska")
    parser.add_argument("--interval", default="1d", help="Interwał OHLCV (np. 1d, 1h)")
    parser.add_argument(
        "--history-limit",
        type=int,
        help="Liczba świec do pobrania (domyślnie minimalna liczba wymagana przez strategię)",
    )
    parser.add_argument("--quote-balance", type=float, default=100_000.0, help="Saldo waluty kwotowanej w symulatorze")
    parser.add_argument("--base-balance", type=float, default=0.0, help="Saldo aktywa bazowego w symulatorze")
    parser.add_argument("--maker-fee", type=float, default=0.0004, help="Prowizja maker w symulatorze")
    parser.add_argument("--taker-fee", type=float, default=0.0006, help="Prowizja taker w symulatorze")
    parser.add_argument("--slippage-bps", type=float, default=5.0, help="Poślizg cenowy w punktach bazowych")
    parser.add_argument("--min-quantity", type=float, default=0.0, help="Minimalna ilość kontraktowa")
    parser.add_argument("--min-notional", type=float, default=0.0, help="Minimalny notional zlecenia")
    parser.add_argument("--step-size", type=float, help="Krok ilościowy (lot size)")
    parser.add_argument("--tick-size", type=float, help="Krok cenowy (tick size)")
    parser.add_argument("--portfolio-id", default="paper-demo", help="Identyfikator portfela w logach")
    parser.add_argument(
        "--headless-passphrase",
        help="Hasło do magazynu sekretów w trybie Linux headless (EncryptedFileSecretStorage)",
    )
    parser.add_argument(
        "--headless-secret-path",
        help="Ścieżka do zaszyfrowanego magazynu sekretów dla trybu headless",
    )
    parser.add_argument(
        "--secret-namespace",
        default="dudzian.trading",
        help="Prefiks nazw kluczy w magazynie sekretów",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Poziom logowania",
    )
    return parser.parse_args(argv)


def _resolve_universe(core_config: CoreConfig, universe_key: str | None) -> InstrumentUniverseConfig | None:
    if universe_key is None:
        return None
    universe = core_config.instrument_universes.get(universe_key)
    if universe is None:
        raise SystemExit(
            f"Środowisko wskazuje uniwersum '{universe_key}', które nie istnieje w konfiguracji."
        )
    return universe


def _select_symbol(
    universe: InstrumentUniverseConfig | None,
    exchange_name: str,
    explicit_symbol: str | None,
    *,
    prebuilt: Mapping[str, InstrumentConfig] | None = None,
) -> tuple[str, InstrumentConfig]:
    instruments: MutableMapping[str, InstrumentConfig] = dict(prebuilt or {})
    if universe and not instruments:
        for instrument in universe.instruments:
            symbol = instrument.exchange_symbols.get(exchange_name)
            if not symbol:
                continue
            instruments[symbol] = InstrumentConfig(
                symbol=symbol,
                base_asset=instrument.base_asset,
                quote_asset=instrument.quote_asset,
                min_quantity=0.0,
                min_notional=0.0,
                step_size=None,
            )

    if explicit_symbol:
        try:
            return explicit_symbol, instruments[explicit_symbol]
        except KeyError as exc:
            available = ", ".join(sorted(instruments)) or "brak"
            raise SystemExit(
                f"Symbol '{explicit_symbol}' nie jest dostępny w uniwersum dla giełdy {exchange_name}."
                f" Dostępne symbole: {available}."
            ) from exc

    if not instruments:
        raise SystemExit(
            "Środowisko nie definiuje żadnych instrumentów dla wybranej giełdy. Uzupełnij sekcję "
            "instrument_universes lub przekaż symbol ręcznie poprzez --symbol."
        )

    selected_symbol, metadata = next(iter(instruments.items()))
    return selected_symbol, metadata


def _build_strategy(
    config: DailyTrendMomentumStrategyConfig,
) -> tuple[DailyTrendMomentumStrategy, DailyTrendMomentumSettings]:
    settings = DailyTrendMomentumSettings(
        fast_ma=config.fast_ma,
        slow_ma=config.slow_ma,
        breakout_lookback=config.breakout_lookback,
        momentum_window=config.momentum_window,
        atr_window=config.atr_window,
        atr_multiplier=config.atr_multiplier,
        min_trend_strength=config.min_trend_strength,
        min_momentum=config.min_momentum,
    )
    return DailyTrendMomentumStrategy(settings), settings


def _build_risk_profile(config: RiskProfileConfig) -> ManualProfile:
    return ManualProfile(
        name=config.name,
        max_positions=config.max_open_positions,
        max_leverage=config.max_leverage,
        drawdown_limit=config.hard_drawdown_pct,
        daily_loss_limit=config.max_daily_loss_pct,
        max_position_pct=config.max_position_pct,
        target_volatility=config.target_volatility,
        stop_loss_atr_multiple=config.stop_loss_atr_multiple,
    )


def _build_data_source(
    base_adapter: ExchangeAdapter,
    cache_path: Path,
) -> CachedOHLCVSource:
    storage = SQLiteCacheStorage(cache_path / "ohlcv.sqlite")
    upstream = PublicAPIDataSource(exchange_adapter=base_adapter)
    return CachedOHLCVSource(storage=storage, upstream=upstream)


def _rows_to_snapshots(symbol: str, response: OHLCVResponse) -> list[MarketSnapshot]:
    snapshots: list[MarketSnapshot] = []
    for row in response.rows:
        if len(row) < 6:
            continue
        try:
            timestamp = int(float(row[0]))
            open_price = float(row[1])
            high = float(row[2])
            low = float(row[3])
            close = float(row[4])
            volume = float(row[5])
        except (TypeError, ValueError):
            continue
        snapshots.append(
            MarketSnapshot(
                symbol=symbol,
                timestamp=timestamp,
                open=open_price,
                high=high,
                low=low,
                close=close,
                volume=volume,
            )
        )
    return snapshots


def _log_audit_entries(audit_log: Iterable[Mapping[str, str]]) -> None:
    for entry in audit_log:
        channel = entry.get("channel", "unknown")
        title = entry.get("title", "brak tytułu")
        severity = entry.get("severity", "info")
        body = entry.get("body", "")
        _LOGGER.info("AUDYT [%s][%s] %s — %s", channel, severity.upper(), title, body)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    logging.basicConfig(level=getattr(logging, args.log_level.upper()))

    try:
        storage = create_default_secret_storage(
            namespace=args.secret_namespace,
            headless_passphrase=args.headless_passphrase,
            headless_path=args.headless_secret_path,
        )
    except SecretStorageError as exc:
        _LOGGER.error("Nie udało się zainicjalizować magazynu sekretów: %s", exc)
        return 1

    secret_manager = SecretManager(storage, namespace=args.secret_namespace)

    try:
        context = bootstrap_environment(
            args.environment,
            config_path=args.config,
            secret_manager=secret_manager,
        )
    except SecretStorageError as exc:
        _LOGGER.error("Brak wymaganych sekretów: %s", exc)
        return 1
    except Exception as exc:  # noqa: BLE001
        _LOGGER.exception("Błąd podczas bootstrappingu środowiska: %s", exc)
        return 1

    environment_cfg = context.environment
    core_config = context.core_config

    risk_config = core_config.risk_profiles.get(environment_cfg.risk_profile)
    if risk_config is None:
        _LOGGER.error(
            "Profil ryzyka '%s' nie istnieje w konfiguracji – zaktualizuj config/core.yaml.",
            environment_cfg.risk_profile,
        )
        return 1

    try:
        universe = _resolve_universe(core_config, environment_cfg.instrument_universe)
    except SystemExit as exc:
        _LOGGER.error(str(exc))
        return 1
    symbol, instrument_metadata = _select_symbol(
        universe,
        environment_cfg.exchange,
        args.symbol,
        prebuilt=context.instruments,
    )

    strategy_config = core_config.strategies.get(args.strategy)
    if strategy_config is None:
        _LOGGER.error("Strategia '%s' nie została zdefiniowana w konfiguracji.", args.strategy)
        return 1

    strategy, strategy_settings = _build_strategy(strategy_config)
    risk_profile = _build_risk_profile(risk_config)
    context.risk_engine.register_profile(risk_profile)

    cache_root = Path(environment_cfg.data_cache_path)
    if not cache_root.is_absolute():
        cache_root = (Path(args.config).resolve().parent / cache_root).resolve()

    data_source = _build_data_source(context.adapter, cache_root)

    required_history = strategy_config.slow_ma + 5
    history_limit = args.history_limit or max(required_history, strategy_settings.max_history())
    now_ms = int(datetime.now(tz=timezone.utc).timestamp() * 1000)
    request = OHLCVRequest(symbol=symbol, interval=args.interval, start=0, end=now_ms, limit=history_limit + 1)
    response = data_source.fetch_ohlcv(request)
    snapshots = _rows_to_snapshots(symbol, response)

    if len(snapshots) <= 1:
        _LOGGER.error(
            "Brak danych OHLCV dla %s (%s). Uruchom najpierw scripts/backfill_ohlcv.py, aby zasilić cache.",
            symbol,
            args.interval,
        )
        return 1

    history, latest = snapshots[:-1], snapshots[-1]
    strategy.warm_up(history)

    markets = {
        symbol: MarketMetadata(
            base_asset=instrument_metadata.base_asset,
            quote_asset=instrument_metadata.quote_asset,
            min_quantity=args.min_quantity or instrument_metadata.min_quantity,
            min_notional=args.min_notional or instrument_metadata.min_notional,
            step_size=args.step_size or instrument_metadata.step_size,
            tick_size=args.tick_size,
        )
    }

    execution = PaperTradingExecutionService(
        markets,
        initial_balances={
            instrument_metadata.quote_asset: args.quote_balance,
            instrument_metadata.base_asset: args.base_balance,
        },
        maker_fee=args.maker_fee,
        taker_fee=args.taker_fee,
        slippage_bps=args.slippage_bps,
    )

    adapter = _PaperAccountAdapter(
        context.credentials,
        execution,
        default_quote_asset=instrument_metadata.quote_asset,
    )

    session = TradingSession(
        strategy=strategy,
        strategy_name=args.strategy,
        adapter=adapter,
        risk_engine=context.risk_engine,
        risk_profile=risk_profile,
        execution=execution,
        alert_router=context.alert_router,
        instruments={symbol: instrument_metadata},
        environment=environment_cfg.environment,
        portfolio_id=args.portfolio_id,
        context_metadata={"config_path": str(Path(args.config).resolve())},
    )

    _LOGGER.info(
        "Start sesji paper trading: env=%s, symbol=%s, interval=%s, historia=%s świec",
        args.environment,
        symbol,
        args.interval,
        len(snapshots),
    )

    results = session.process_snapshot(latest)

    if results:
        for result in results:
            avg_price = result.avg_price if result.avg_price is not None else latest.close
            _LOGGER.info(
                "Zrealizowano zlecenie paper trading: id=%s status=%s qty=%.8f price=%.2f",
                result.order_id,
                result.status,
                result.filled_quantity,
                avg_price,
            )
    else:
        _LOGGER.info("Strategia nie wygenerowała zleceń dla bieżącej świecy – brak zmian w portfelu.")

    _log_audit_entries(context.audit_log.export())

    balances = execution.balances()
    for asset, value in sorted(balances.items()):
        _LOGGER.info("Saldo paper trading %s: %.8f", asset, value)

    return 0


if __name__ == "__main__":  # pragma: no cover - punkt wejścia CLI
    raise SystemExit(main())

