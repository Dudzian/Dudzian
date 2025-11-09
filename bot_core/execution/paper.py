"""Silnik paper trading odzwierciedlający koszty i poślizg."""
from __future__ import annotations

import itertools
import json
import logging
import os
import threading
import time
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Mapping, MutableMapping, Optional

from bot_core.execution.base import ExecutionContext, ExecutionService, PriceResolver
from bot_core.exchanges.base import OrderRequest, OrderResult

# --- Observability (wymagana dla zgodności metryk) ---------------------------
try:  # pragma: no cover - w testach import może zostać zamockowany
    from bot_core.observability.metrics import MetricsRegistry, get_global_metrics_registry
except Exception as exc:  # pragma: no cover
    raise RuntimeError(
        "Moduł 'bot_core.observability.metrics' jest wymagany przez PaperTradingExecutionService. "
        "Dołącz komponenty observability do środowiska runtime lub zainstaluj extras 'observability'."
    ) from exc


_LOGGER = logging.getLogger(__name__)


@dataclass(slots=True, frozen=True)
class MarketMetadata:
    """Parametry rynku wykorzystywane przez symulator paper tradingu."""

    base_asset: str
    quote_asset: str
    min_quantity: float = 0.0
    min_notional: float = 0.0
    step_size: Optional[float] = None
    tick_size: Optional[float] = None


@dataclass(slots=True)
class LedgerEntry:
    """Pojedynczy wpis audytowy."""

    timestamp: float
    order_id: str
    symbol: str
    side: str
    quantity: float
    price: float
    fee: float
    fee_asset: str
    status: str
    leverage: float
    position_value: float

    def to_mapping(self) -> Mapping[str, object]:
        return {
            "timestamp": self.timestamp,
            "order_id": self.order_id,
            "symbol": self.symbol,
            "side": self.side,
            "quantity": self.quantity,
            "price": self.price,
            "fee": self.fee,
            "fee_asset": self.fee_asset,
            "status": self.status,
            "leverage": self.leverage,
            "position_value": self.position_value,
        }


class InsufficientBalanceError(RuntimeError):
    """Rzucany, gdy na rachunku brakuje środków na realizację zlecenia."""


@dataclass(slots=True)
class ShortPosition:
    """Reprezentacja pozycji krótkiej wraz z depozytem zabezpieczającym."""

    quantity: float
    entry_price: float
    margin: float
    leverage: float

    def to_mapping(self) -> Mapping[str, float]:
        return {
            "quantity": self.quantity,
            "entry_price": self.entry_price,
            "margin": self.margin,
            "leverage": self.leverage,
        }


class PaperTradingExecutionService(ExecutionService):
    """Symulator giełdy realizujący zlecenia natychmiast z kosztem prowizji i poślizgu."""

    def __init__(
        self,
        markets: Mapping[str, MarketMetadata],
        *,
        initial_balances: Optional[Mapping[str, float]] = None,
        maker_fee: float = 0.0004,
        taker_fee: float = 0.0006,
        slippage_bps: float = 5.0,
        time_source: Optional[Callable[[], float]] = None,
        metrics: MetricsRegistry | None = None,
        ledger_directory: str | Path | None = None,
        ledger_filename_pattern: str = "ledger-%Y%m%d.jsonl",
        ledger_retention_days: int | None = 730,
        ledger_fsync: bool = False,
        ledger_encoding: str = "utf-8",
        price_resolver: PriceResolver | None = None,
    ) -> None:
        if not markets:
            raise ValueError("Wymagana jest co najmniej jedna definicja rynku.")
        self._markets: Dict[str, MarketMetadata] = dict(markets)
        self._balances: MutableMapping[str, float] = {
            asset: float(value) for asset, value in (initial_balances or {}).items()
        }
        self._maker_fee = max(0.0, maker_fee)
        self._taker_fee = max(0.0, taker_fee)
        self._slippage_bps = max(0.0, slippage_bps)
        self._time = time_source or time.time
        self._order_counter = itertools.count(1)
        self._ledger: List[LedgerEntry] = []
        self._short_positions: Dict[str, ShortPosition] = {}
        self._maintenance_profiles: Mapping[str, float] = {
            "conservative": 0.3,
            "balanced": 0.2,
            "aggressive": 0.1,
        }
        self._default_maintenance_margin = 0.25

        self._ledger_directory: Path | None = None
        self._ledger_filename_pattern = ledger_filename_pattern
        self._ledger_retention_days = ledger_retention_days
        self._ledger_fsync = ledger_fsync
        self._ledger_encoding = ledger_encoding
        self._ledger_lock: threading.Lock | None = None

        if ledger_directory:
            self._ledger_directory = Path(ledger_directory)
            self._ledger_directory.mkdir(parents=True, exist_ok=True)
            datetime.now(timezone.utc).strftime(self._ledger_filename_pattern)
            self._ledger_lock = threading.Lock()

        # metrics
        self._metrics = metrics or get_global_metrics_registry()
        self._metric_orders_total = self._metrics.counter(
            "paper_orders_total", "Liczba zleceń zrealizowanych w symulatorze paper tradingu."
        )
        self._metric_orders_rejected = self._metrics.counter(
            "paper_orders_rejected_total", "Liczba zleceń odrzuconych przez symulator paper tradingu."
        )
        self._metric_traded_notional = self._metrics.counter(
            "paper_traded_notional_total", "Skumulowany notional obrotu w symulatorze paper tradingu."
        )
        self._metric_latency = self._metrics.histogram(
            "paper_execution_latency_seconds",
            "Czas realizacji zleceń w symulatorze paper tradingu (sekundy).",
            buckets=(0.001, 0.005, 0.01, 0.05, 0.1, 0.25, 0.5, 1.0),
        )
        self._price_resolver: PriceResolver | None = price_resolver

    # --- API ExecutionService -------------------------------------------------
    def execute(self, request: OrderRequest, context: ExecutionContext) -> OrderResult:
        symbol = request.symbol
        market = self._markets.get(symbol)
        if market is None:
            raise KeyError(f"Brak konfiguracji rynku dla symbolu {symbol}")

        side = request.side.lower()
        if side not in {"buy", "sell"}:
            raise ValueError("Obsługiwane są wyłącznie zlecenia kupna lub sprzedaży.")

        if request.quantity <= 0:
            raise ValueError("Wielkość zlecenia musi być dodatnia.")

        self._validate_lot_size(request.quantity, market)

        start_time = self._time()
        try:
            result = self._execute_internal(request, context, market, side)
        except InsufficientBalanceError:
            self._metric_orders_rejected.inc(labels={"symbol": symbol, "reason": "insufficient_balance"})
            elapsed = max(0.0, self._time() - start_time)
            self._metric_latency.observe(elapsed, labels={"symbol": symbol, "status": "rejected"})
            raise
        except Exception:
            self._metric_orders_rejected.inc(labels={"symbol": symbol, "reason": "error"})
            elapsed = max(0.0, self._time() - start_time)
            self._metric_latency.observe(elapsed, labels={"symbol": symbol, "status": "error"})
            raise

        filled_notional = (result.avg_price or 0.0) * (result.filled_quantity or 0.0)
        elapsed = max(0.0, self._time() - start_time)
        self._metric_orders_total.inc(labels={"symbol": symbol, "side": side})
        self._metric_latency.observe(elapsed, labels={"symbol": symbol, "status": "filled"})
        self._metric_traded_notional.inc(filled_notional, labels={"symbol": symbol, "side": side})
        return result

    def _execute_internal(
        self,
        request: OrderRequest,
        context: ExecutionContext,
        market: MarketMetadata,
        side: str,
    ) -> OrderResult:
        symbol = request.symbol
        reference_price = self._determine_reference_price(request, context)
        notional = reference_price * request.quantity
        if notional < market.min_notional:
            raise ValueError(
                f"Notional {notional:.8f} jest mniejszy niż minimalna wartość {market.min_notional:.8f} dla {symbol}."
            )

        fill_price = self._apply_slippage(reference_price, side)
        fee_rate = self._taker_fee if request.order_type.lower() == "market" else self._maker_fee
        fee = request.quantity * fill_price * fee_rate

        leverage = self._extract_leverage(context)
        if side == "buy":
            self._process_buy(symbol, market, request.quantity, fill_price, fee, context.risk_profile)
        else:
            self._process_sell(symbol, market, request.quantity, fill_price, fee, leverage, context.risk_profile)

        position_value = self._position_value(symbol, market, fill_price)
        ledger_leverage = self._short_positions.get(symbol).leverage if symbol in self._short_positions else 1.0

        order_id = f"paper-{next(self._order_counter)}"
        result = OrderResult(
            order_id=order_id,
            status="filled",
            filled_quantity=request.quantity,
            avg_price=fill_price,
            raw_response={
                "environment": context.environment,
                "risk_profile": context.risk_profile,
                "portfolio_id": context.portfolio_id,
                "fee": fee,
                "fee_asset": market.quote_asset,
            },
        )
        self._ledger.append(
            LedgerEntry(
                timestamp=self._time(),
                order_id=order_id,
                symbol=symbol,
                side=side,
                quantity=request.quantity,
                price=fill_price,
                fee=fee,
                fee_asset=market.quote_asset,
                status=result.status,
                leverage=ledger_leverage,
                position_value=position_value,
            )
        )
        self._persist_ledger_entry(self._ledger[-1])
        _LOGGER.debug(
            "Paper trade %s %s qty=%s price=%s fee=%s (env=%s)",
            side,
            symbol,
            request.quantity,
            fill_price,
            fee,
            context.environment,
        )
        return result

    def cancel(self, order_id: str, context: ExecutionContext) -> None:  # noqa: ARG002
        # Zlecenia w trybie paper trading są realizowane natychmiast – rejestrujemy jedynie próbę anulacji.
        self._ledger.append(
            LedgerEntry(
                timestamp=self._time(),
                order_id=order_id,
                symbol="*",
                side="cancel",
                quantity=0.0,
                price=0.0,
                fee=0.0,
                fee_asset="",
                status="cancelled",
                leverage=1.0,
                position_value=0.0,
            )
        )
        _LOGGER.info("Zarejestrowano anulację %s w symulatorze paper trading (brak otwartych zleceń).", order_id)
        self._persist_ledger_entry(self._ledger[-1])

    def flush(self) -> None:
        # Symulator realizuje zlecenia natychmiast – flush nie musi nic robić.
        _LOGGER.debug("PaperTradingExecutionService.flush() – brak zaległych operacji.")

    # --- Funkcje pomocnicze ---------------------------------------------------
    def _determine_reference_price(self, request: OrderRequest, context: ExecutionContext) -> float:
        price = request.price
        if price is not None and price > 0:
            return price

        symbol = request.symbol

        for resolver in self._iter_price_resolvers(context):
            try:
                resolved = resolver(symbol)
            except Exception:  # pragma: no cover - defensywnie ignorujemy błędne resolvery
                _LOGGER.debug("Resolver ceny rynku %s zgłosił wyjątek.", symbol, exc_info=True)
                continue
            if resolved is not None and resolved > 0:
                return resolved

        metadata_price = self._extract_price_from_metadata(symbol, context.metadata)
        if metadata_price is not None:
            return metadata_price

        raise ValueError(
            "Brak ceny referencyjnej dla symulacji – podaj price lub dostarcz resolver danych rynkowych."
        )

    def _iter_price_resolvers(self, context: ExecutionContext) -> Iterable[PriceResolver]:
        if context.price_resolver is not None:
            yield context.price_resolver
        if self._price_resolver is not None and self._price_resolver is not context.price_resolver:
            yield self._price_resolver

    @staticmethod
    def _extract_price_from_metadata(symbol: str, metadata: Mapping[str, str]) -> float | None:
        candidate_keys = (
            f"last_price:{symbol}",
            f"reference_price:{symbol}",
            f"close_price:{symbol}",
            "last_price",
            "reference_price",
            "close_price",
            "ohlcv_last_close",
        )
        for key in candidate_keys:
            raw_value = metadata.get(key)
            if raw_value is None:
                continue
            try:
                price = float(raw_value)
            except (TypeError, ValueError):
                continue
            if price > 0:
                return price
        return None

    def _apply_slippage(self, price: float, side: str) -> float:
        if self._slippage_bps <= 0:
            return price
        adjustment = price * (self._slippage_bps / 10_000.0)
        if side == "buy":
            return price + adjustment
        return max(0.0, price - adjustment)

    def _validate_lot_size(self, quantity: float, market: MarketMetadata) -> None:
        if quantity < market.min_quantity:
            raise ValueError(
                f"Wielkość zlecenia {quantity:.8f} jest mniejsza niż minimalna {market.min_quantity:.8f}."
            )
        if market.step_size:
            remainder = (quantity / market.step_size) - round(quantity / market.step_size)
            if abs(remainder) > 1e-8:
                raise ValueError(
                    f"Wielkość {quantity:.8f} nie spełnia kroku {market.step_size:.8f} dla rynku {market.base_asset}/{market.quote_asset}."
                )

    def _process_buy(
        self,
        symbol: str,
        market: MarketMetadata,
        quantity: float,
        price: float,
        fee: float,
        risk_profile: str,
    ) -> None:
        cost = quantity * price + fee
        quote_balance = self._balances.get(market.quote_asset, 0.0)

        short_position = self._short_positions.get(symbol)
        cover_quantity = 0.0
        margin_release = 0.0
        if short_position and short_position.quantity > 0:
            prev_quantity = short_position.quantity
            cover_quantity = min(quantity, prev_quantity)
            if cover_quantity > 0:
                proportion = cover_quantity / prev_quantity if prev_quantity else 0.0
                margin_release = short_position.margin * proportion
                short_position.margin -= margin_release
                short_position.quantity = prev_quantity - cover_quantity
                if short_position.quantity <= 1e-12:
                    self._short_positions.pop(symbol, None)
                else:
                    short_position.leverage = self._recalculate_leverage(short_position, price)

        new_quote_balance = quote_balance - cost + margin_release
        if new_quote_balance + 1e-12 < 0.0:
            raise InsufficientBalanceError(
                f"Brak środków {market.quote_asset}. Dostępne {quote_balance:.8f}, wymagane {cost - margin_release:.8f}."
            )
        self._balances[market.quote_asset] = new_quote_balance

        remaining_quantity = quantity - cover_quantity
        if remaining_quantity > 0:
            self._balances[market.base_asset] = self._balances.get(market.base_asset, 0.0) + remaining_quantity

        if symbol in self._short_positions:
            self._enforce_maintenance_margin(symbol, market, price, risk_profile)
        _LOGGER.debug(
            "BUY %s: -%s %s, +%s %s (fee=%s)",
            symbol,
            cost,
            market.quote_asset,
            quantity,
            market.base_asset,
            fee,
        )

    def _process_sell(
        self,
        symbol: str,
        market: MarketMetadata,
        quantity: float,
        price: float,
        fee: float,
        leverage: float,
        risk_profile: str,
    ) -> None:
        base_balance = self._balances.get(market.base_asset, 0.0)
        quote_balance = self._balances.get(market.quote_asset, 0.0)
        fee_per_unit = fee / quantity if quantity > 0 else 0.0

        spot_quantity = min(base_balance, quantity)
        spot_fee = spot_quantity * fee_per_unit
        spot_proceeds = spot_quantity * price - spot_fee
        new_base_balance = base_balance
        new_quote_balance = quote_balance
        if spot_quantity > 0:
            new_base_balance = base_balance - spot_quantity
            new_quote_balance += max(0.0, spot_proceeds)

        remaining = quantity - spot_quantity
        short_proceeds = 0.0
        short_position = None
        required_margin = 0.0
        if remaining > 0:
            leverage = max(1.0, leverage)
            short_fee = remaining * fee_per_unit
            short_proceeds = remaining * price - short_fee
            new_quote_balance += max(0.0, short_proceeds)
            required_margin = (remaining * price) / leverage
            if new_quote_balance + 1e-12 < required_margin:
                raise InsufficientBalanceError(
                    f"Brak środków {market.quote_asset} na depozyt zabezpieczający. Dostępne {new_quote_balance:.8f}, wymagane {required_margin:.8f}."
                )
            new_quote_balance -= required_margin
            short_position = self._short_positions.get(symbol)

        self._balances[market.base_asset] = new_base_balance
        self._balances[market.quote_asset] = new_quote_balance

        if remaining > 0:
            if short_position is None:
                short_position = ShortPosition(quantity=0.0, entry_price=price, margin=0.0, leverage=leverage)
                self._short_positions[symbol] = short_position
            prev_quantity = short_position.quantity
            total_quantity = prev_quantity + remaining
            if total_quantity > 0:
                short_position.entry_price = (
                    (short_position.entry_price * prev_quantity + price * remaining) / total_quantity
                    if prev_quantity > 0
                    else price
                )
            short_position.quantity = total_quantity
            short_position.margin += required_margin
            short_position.leverage = self._recalculate_leverage(short_position, price)

        if symbol in self._short_positions:
            self._enforce_maintenance_margin(symbol, market, price, risk_profile)

        _LOGGER.debug(
            "SELL %s: -%s %s, +%s %s (fee=%s)",
            symbol,
            quantity,
            market.base_asset,
            spot_proceeds + short_proceeds,
            market.quote_asset,
            fee,
        )

    # --- Funkcje obserwowalne -------------------------------------------------
    def balances(self) -> Mapping[str, float]:
        """Zwraca bieżące saldo konta paper trading."""
        return dict(self._balances)

    def ledger(self) -> Iterable[Mapping[str, object]]:
        """Zwraca kopię wpisów audytowych (do raportów compliance)."""
        return [entry.to_mapping() for entry in self._ledger]

    def ledger_files(self) -> Iterable[Path]:
        """Zwraca uporządkowaną listę plików ledger, jeśli trwały zapis jest włączony."""

        if not self._ledger_directory:
            return ()
        return tuple(sorted(self._ledger_directory.glob("*")))

    def short_positions(self) -> Mapping[str, Mapping[str, float]]:
        """Zwraca aktualne pozycje krótkie wraz z depozytem zabezpieczającym."""

        return {symbol: position.to_mapping() for symbol, position in self._short_positions.items()}

    # --- Obsługa dźwigni i margin --------------------------------------------
    def _extract_leverage(self, context: ExecutionContext) -> float:
        raw = context.metadata.get("leverage") if context.metadata else None
        try:
            leverage = float(raw) if raw is not None else 1.0
        except (TypeError, ValueError):
            leverage = 1.0
        return max(1.0, leverage)

    def _position_value(self, symbol: str, market: MarketMetadata, price: float) -> float:
        base_quantity = self._balances.get(market.base_asset, 0.0)
        short_quantity = self._short_positions.get(symbol).quantity if symbol in self._short_positions else 0.0
        return (base_quantity + short_quantity) * price

    def _persist_ledger_entry(self, entry: LedgerEntry) -> None:
        if not self._ledger_directory or not self._ledger_lock:
            return

        timestamp = datetime.fromtimestamp(entry.timestamp, timezone.utc)
        filename = timestamp.strftime(self._ledger_filename_pattern)
        target = self._ledger_directory / filename
        payload = json.dumps(entry.to_mapping(), ensure_ascii=False, separators=(",", ":"))

        with self._ledger_lock:
            target.parent.mkdir(parents=True, exist_ok=True)
            with target.open("a", encoding=self._ledger_encoding) as handle:
                handle.write(payload)
                handle.write("\n")
                handle.flush()
                if self._ledger_fsync:
                    os.fsync(handle.fileno())
            self._purge_ledger_files(current_date=timestamp.date())

    def _purge_ledger_files(self, *, current_date: date) -> None:
        if not self._ledger_directory:
            return
        if not self._ledger_retention_days or self._ledger_retention_days <= 0:
            return

        cutoff = current_date - timedelta(days=self._ledger_retention_days - 1)
        for file_path in self._ledger_directory.glob("*"):
            if not file_path.is_file():
                continue
            try:
                file_date = datetime.strptime(file_path.name, self._ledger_filename_pattern).date()
            except ValueError:
                continue
            if file_date < cutoff:
                try:
                    file_path.unlink()
                except OSError:
                    continue

    def _maintenance_margin_ratio(self, risk_profile: str) -> float:
        return self._maintenance_profiles.get(risk_profile, self._default_maintenance_margin)

    def _recalculate_leverage(self, position: ShortPosition, price: float) -> float:
        if position.margin <= 0:
            return 0.0
        return max(1.0, (position.quantity * price) / position.margin)

    def _enforce_maintenance_margin(
        self,
        symbol: str,
        market: MarketMetadata,
        price: float,
        risk_profile: str,
    ) -> None:
        position = self._short_positions.get(symbol)
        if not position or position.quantity <= 0:
            return
        maintenance_ratio = self._maintenance_margin_ratio(risk_profile)
        position_value = position.quantity * price
        maintenance_requirement = position_value * maintenance_ratio
        unrealized_pnl = (position.entry_price - price) * position.quantity
        equity = position.margin + unrealized_pnl
        if equity + 1e-12 < maintenance_requirement:
            self._liquidate_short(symbol, market, price)

    def _liquidate_short(self, symbol: str, market: MarketMetadata, price: float) -> None:
        position = self._short_positions.pop(symbol, None)
        if position is None:
            return
        quantity = position.quantity
        if quantity <= 0:
            return
        cost = quantity * price
        quote_balance = self._balances.get(market.quote_asset, 0.0)
        available = quote_balance + position.margin
        if available + 1e-12 < cost:
            raise InsufficientBalanceError(
                f"Brak środków {market.quote_asset} do likwidacji pozycji short {symbol}."
            )
        self._balances[market.quote_asset] = available - cost
        _LOGGER.warning(
            "Pozycja short %s została zlikwidowana przy cenie %s (maintenance margin).",
            symbol,
            price,
        )


__all__ = [
    "PaperTradingExecutionService",
    "MarketMetadata",
    "LedgerEntry",
    "InsufficientBalanceError",
    "ShortPosition",
]
