"""Generator raportów podatkowych bazujących na wpisach w dzienniku."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence

from bot_core.execution import LedgerEntry

try:  # pragma: no cover - PyYAML opcjonalny w środowisku runtime
    import yaml
except ModuleNotFoundError:  # pragma: no cover
    yaml = None  # type: ignore[assignment]

from ..tax.calculators import (
    AverageCostBasisCalculator,
    CostBasisCalculator,
    FIFOCostBasisCalculator,
    LIFOCostBasisCalculator,
)
from ..tax.fx import FXRateProvider
from ..tax.models import (
    AssetBreakdown,
    DisposalEvent,
    TaxLot,
    TaxReport,
    TaxReportTotals,
    VenueBreakdown,
    PeriodBreakdown,
)

_QUOTE_CURRENCIES_DEFAULT = ("USDT", "USD", "EUR", "PLN", "BTC", "ETH")


@dataclass(slots=True)
class _TradeRecord:
    timestamp: datetime
    symbol: str
    asset: str
    quote_currency: Optional[str]
    side: str
    quantity: float
    price: float
    fee: float
    fee_currency: Optional[str]
    venue: Optional[str]
    source: str
    order_id: Optional[str]

    @property
    def proceeds(self) -> float:
        return self.quantity * self.price


class TaxReportGenerator:
    """Agreguje wpisy z księgi i lokalnego magazynu zleceń."""

    def __init__(
        self,
        *,
        config_path: Path | str | None = None,
        jurisdiction_overrides: Mapping[str, str] | None = None,
        base_currency: str | None = None,
        fx_rate_provider: FXRateProvider | None = None,
    ) -> None:
        self._trades: List[_TradeRecord] = []
        self._trade_index: Dict[str, _TradeRecord] = {}
        self._trade_keys: Dict[int, List[str]] = {}
        self._config = self._load_config(config_path)
        self._jurisdiction_overrides = dict(jurisdiction_overrides or {})
        self._quote_currencies = tuple(
            self._config.get("quote_currencies", _QUOTE_CURRENCIES_DEFAULT)
        )
        self._default_long_term_days = float(self._config.get("default_long_term_days", 365))
        self._default_tax_rates = self._parse_default_tax_rates(self._config)
        config_base = self._normalise_currency(self._config.get("base_currency"))
        self._base_currency = config_base
        explicit_base = self._normalise_currency(base_currency)
        if explicit_base:
            self._base_currency = explicit_base
        self._fx_rate_provider = fx_rate_provider

    @property
    def base_currency(self) -> str | None:
        return self._base_currency

    def set_base_currency(self, currency: str | None) -> None:
        self._base_currency = self._normalise_currency(currency)

    def set_fx_rate_provider(self, provider: FXRateProvider | None) -> None:
        self._fx_rate_provider = provider

    # --- Ładowanie konfiguracji -------------------------------------------------
    def _load_config(self, config_path: Path | str | None) -> Mapping[str, object]:
        candidates: List[Path] = []
        if config_path:
            candidates.append(Path(config_path))
        base_dir = Path(__file__).resolve().parents[3]
        candidates.extend(
            [
                base_dir / "config" / "compliance" / "tax_methods.yaml",
                base_dir / "config" / "compliance" / "tax_methods.yml",
            ]
        )
        for candidate in candidates:
            if candidate.exists():
                if yaml is None:
                    raise RuntimeError("PyYAML jest wymagany do wczytania konfiguracji podatkowej")
                with candidate.open("r", encoding="utf-8") as handle:
                    data = yaml.safe_load(handle) or {}
                return data
        return {"default_method": "fifo"}

    def _parse_default_tax_rates(self, config: Mapping[str, object]) -> tuple[float, float]:
        default_rates = config.get("default_tax_rates", {})
        if isinstance(default_rates, Mapping):
            short_rate = self._coerce_rate(default_rates.get("short_term"))
            long_rate = self._coerce_rate(default_rates.get("long_term"))
            return (
                short_rate if short_rate is not None else 0.0,
                long_rate if long_rate is not None else 0.0,
            )
        return 0.0, 0.0

    def _coerce_rate(self, value: object) -> float | None:
        try:
            rate = float(value)  # type: ignore[arg-type]
        except (TypeError, ValueError):
            return None
        if rate < 0:
            return None
        return rate

    # --- Ingest -----------------------------------------------------------------
    def ingest_ledger_entries(self, entries: Iterable[LedgerEntry], *, source: str = "ledger") -> None:
        for entry in entries:
            side = str(entry.side).lower()
            if side not in {"buy", "sell"}:
                continue
            quantity = float(entry.quantity)
            if quantity < 0:
                quantity = abs(quantity)
                if side != "sell":
                    side = "sell"
            if quantity <= 0:
                continue
            timestamp = datetime.fromtimestamp(float(entry.timestamp), timezone.utc)
            asset = self._infer_asset(entry.symbol)
            venue = None
            symbol = entry.symbol
            if ":" in symbol:
                venue, symbol = symbol.split(":", 1)
            venue = self._normalise_venue(venue)
            quote_currency = self._infer_quote(entry.symbol)
            fee_currency = self._normalise_currency(entry.fee_asset) or quote_currency
            order_id = (entry.order_id or "").strip() or None
            trade = _TradeRecord(
                timestamp=timestamp,
                symbol=symbol,
                asset=asset,
                quote_currency=quote_currency,
                side=side,
                quantity=quantity,
                price=float(entry.price),
                fee=float(entry.fee),
                fee_currency=fee_currency,
                venue=venue,
                source=source,
                order_id=order_id,
            )
            self._register_trade(trade)

    def ingest_ledger_directory(self, ledger_dir: Path | str) -> None:
        directory = Path(ledger_dir)
        if not directory.exists():
            return
        for path in sorted(directory.glob("*.jsonl")):
            with path.open("r", encoding="utf-8") as handle:
                for line in handle:
                    line = line.strip()
                    if not line:
                        continue
                    payload = json.loads(line)
                    entry = LedgerEntry(
                        timestamp=float(payload["timestamp"]),
                        order_id=str(payload.get("order_id", "")),
                        symbol=str(payload.get("symbol", "")),
                        side=str(payload.get("side", "")),
                        quantity=float(payload.get("quantity", 0.0)),
                        price=float(payload.get("price", 0.0)),
                        fee=float(payload.get("fee", 0.0)),
                        fee_asset=str(payload.get("fee_asset", "")),
                        status=str(payload.get("status", "")),
                        leverage=float(payload.get("leverage", 1.0)),
                        position_value=float(payload.get("position_value", 0.0)),
                    )
                    self.ingest_ledger_entries([entry], source=f"file:{path.name}")

    def ingest_execution_services(self, services: Iterable[object]) -> None:
        for service in services:
            ledger = getattr(service, "_ledger", None)
            if ledger:
                self.ingest_ledger_entries(ledger, source=service.__class__.__name__)
            stream = getattr(service, "iter_ledger_entries", None)
            if callable(stream):
                self.ingest_ledger_entries(stream(), source=service.__class__.__name__)

    def ingest_local_order_store(
        self,
        store: object,
        *,
        symbols: Sequence[str] | None = None,
        since: datetime | None = None,
    ) -> None:
        if store is None:
            return
        sync_accessor = getattr(store, "sync", store)
        fetch_trades = getattr(sync_accessor, "fetch_trades", None)
        if not callable(fetch_trades):
            raise ValueError("Magazyn zleceń nie udostępnia fetch_trades")
        kwargs: MutableMapping[str, object] = {"limit": 10000}
        if since:
            kwargs["since"] = since
        if symbols:
            for symbol in symbols:
                rows = fetch_trades(symbol=symbol, **kwargs)
                self._ingest_trade_rows(rows)
            return
        rows = fetch_trades(**kwargs)
        self._ingest_trade_rows(rows)

    def _ingest_trade_rows(self, rows: Sequence[Mapping[str, object]]) -> None:
        for row in rows:
            symbol = str(row.get("symbol", ""))
            asset = self._infer_asset(symbol)
            side = str(row.get("side", "")).lower()
            if side not in {"buy", "sell"}:
                continue
            timestamp_raw = row.get("ts")
            if isinstance(timestamp_raw, str):
                timestamp = datetime.fromisoformat(timestamp_raw)
            elif isinstance(timestamp_raw, datetime):
                timestamp = timestamp_raw
            elif isinstance(timestamp_raw, (int, float)):
                timestamp = datetime.fromtimestamp(float(timestamp_raw), timezone.utc)
            else:
                timestamp = datetime.fromtimestamp(0, timezone.utc)
            if timestamp.tzinfo is None:
                timestamp = timestamp.replace(tzinfo=timezone.utc)
            quantity_raw = row.get("quantity", 0.0)
            try:
                quantity = float(quantity_raw)
            except (TypeError, ValueError):
                continue
            if quantity < 0:
                quantity = abs(quantity)
                if side != "sell":
                    side = "sell"
            if quantity <= 0:
                continue
            order_id = None
            raw_order_id = row.get("order_id")
            if raw_order_id:
                order_id = str(raw_order_id).strip() or None
            trade = _TradeRecord(
                timestamp=timestamp,
                symbol=symbol,
                asset=asset,
                quote_currency=self._infer_quote(symbol),
                side=side,
                quantity=quantity,
                price=float(row.get("price", 0.0)),
                fee=float(row.get("fee", 0.0)),
                fee_currency=self._normalise_currency(
                    row.get("fee_currency") or row.get("fee_asset")
                ),
                venue=self._normalise_venue(row.get("mode")),
                source="local-store",
                order_id=order_id,
            )
            self._register_trade(trade)

    # --- Raportowanie -----------------------------------------------------------
    def generate_report(
        self,
        jurisdiction: str,
        *,
        symbols: Sequence[str] | None = None,
        start: datetime | None = None,
        end: datetime | None = None,
    ) -> TaxReport:
        method, long_term_days = self._resolve_method_and_threshold(jurisdiction)
        tax_rates = self._resolve_tax_rates(jurisdiction)
        grouped: MutableMapping[str, List[_TradeRecord]] = {}
        for trade in self._trades:
            if symbols and trade.symbol not in symbols and trade.asset not in symbols:
                continue
            if end and trade.timestamp > end:
                continue
            grouped.setdefault(trade.asset, []).append(trade)
        events: List[DisposalEvent] = []
        open_lots: List[TaxLot] = []
        for asset, trades in grouped.items():
            calculator = self._build_calculator(method)
            trades_sorted = sorted(trades, key=lambda item: item.timestamp)
            for trade in trades_sorted:
                lot_id = trade.order_id or f"{trade.source}:{trade.timestamp.timestamp()}"
                if trade.side == "buy":
                    cost = self._convert_to_base(
                        trade.quantity * trade.price,
                        trade.quote_currency,
                        trade.timestamp,
                    )
                    fee = self._convert_to_base(
                        trade.fee,
                        trade.fee_currency or trade.quote_currency,
                        trade.timestamp,
                    )
                    lot = TaxLot(
                        lot_id=lot_id,
                        asset=asset,
                        acquisition_time=trade.timestamp,
                        quantity=trade.quantity,
                        cost_basis=cost,
                        fee=fee,
                        venue=trade.venue,
                        source=trade.source,
                    )
                    calculator.add_lot(lot)
                else:
                    proceeds = self._convert_to_base(
                        trade.proceeds,
                        trade.quote_currency,
                        trade.timestamp,
                    )
                    fee = self._convert_to_base(
                        trade.fee,
                        trade.fee_currency or trade.quote_currency,
                        trade.timestamp,
                    )
                    event = DisposalEvent(
                        event_id=lot_id,
                        asset=asset,
                        disposal_time=trade.timestamp,
                        quantity=trade.quantity,
                        proceeds=proceeds,
                        fee=fee,
                        venue=trade.venue,
                        source=trade.source,
                    )
                    event = calculator.dispose(event)
                    self._annotate_holding_period(event, long_term_days, tax_rates)
                    if start and event.disposal_time < start:
                        continue
                    events.append(event)
            open_lots.extend(self._normalise_remaining(calculator, asset))
        generated_at = datetime.now(timezone.utc)
        reference_time = end or generated_at
        if reference_time.tzinfo is None:
            reference_time = reference_time.replace(tzinfo=timezone.utc)
        self._annotate_open_lots(open_lots, reference_time)
        totals = self._compute_totals(events, open_lots, long_term_days)
        breakdown = self._compute_asset_breakdown(events, open_lots, long_term_days)
        venue_breakdown = self._compute_venue_breakdown(events, open_lots, long_term_days)
        period_breakdown = self._compute_period_breakdown(events)
        return TaxReport(
            jurisdiction=jurisdiction,
            method=method,
            generated_at=generated_at,
            events=sorted(events, key=lambda e: e.disposal_time),
            open_lots=sorted(open_lots, key=lambda lot: lot.acquisition_time),
            totals=totals,
            asset_breakdown=breakdown,
            venue_breakdown=venue_breakdown,
            period_breakdown=period_breakdown,
            base_currency=self._base_currency,
        )

    # --- Pomocnicze -------------------------------------------------------------
    def _register_trade(self, trade: _TradeRecord) -> None:
        candidate_keys = self._candidate_keys(trade)
        existing: _TradeRecord | None = None
        for key in candidate_keys:
            existing = self._trade_index.get(key)
            if existing:
                break
        if existing is None:
            self._trades.append(trade)
            self._trade_keys[id(trade)] = []
            self._register_keys(trade, candidate_keys)
            return
        if existing.source == "local-store" and trade.source != "local-store":
            self._copy_trade(existing, trade)
        else:
            self._merge_trade(existing, trade)
        self._refresh_trade_keys(existing)

    def _copy_trade(self, target: _TradeRecord, source: _TradeRecord) -> None:
        target.timestamp = source.timestamp
        target.symbol = source.symbol
        target.asset = source.asset
        target.quote_currency = source.quote_currency
        target.side = source.side
        target.quantity = source.quantity
        target.price = source.price
        target.fee = source.fee
        target.fee_currency = source.fee_currency
        target.venue = source.venue
        target.source = source.source
        target.order_id = source.order_id

    def _merge_trade(self, target: _TradeRecord, candidate: _TradeRecord) -> None:
        if not target.order_id and candidate.order_id:
            target.order_id = candidate.order_id
        if target.venue is None and candidate.venue is not None:
            target.venue = candidate.venue
        if target.quote_currency is None and candidate.quote_currency is not None:
            target.quote_currency = candidate.quote_currency
        if target.source == "local-store" and candidate.source != "local-store":
            target.source = candidate.source
        if abs(target.fee) <= 1e-12 and abs(candidate.fee) > 0:
            target.fee = candidate.fee
        if target.fee_currency is None and candidate.fee_currency is not None:
            target.fee_currency = candidate.fee_currency
        if abs(target.price) <= 1e-12 and abs(candidate.price) > 0:
            target.price = candidate.price
        if abs(target.quantity) <= 1e-12 and abs(candidate.quantity) > 0:
            target.quantity = candidate.quantity

    def _refresh_trade_keys(self, trade: _TradeRecord) -> None:
        self._unregister_keys(trade)
        self._register_keys(trade, self._candidate_keys(trade))

    def _register_keys(self, trade: _TradeRecord, keys: Sequence[str]) -> None:
        key_list = self._trade_keys.setdefault(id(trade), [])
        for key in keys:
            if key in self._trade_index and self._trade_index[key] is not trade:
                continue
            if key not in key_list:
                key_list.append(key)
            self._trade_index[key] = trade

    def _unregister_keys(self, trade: _TradeRecord) -> None:
        key_list = self._trade_keys.get(id(trade), [])
        for key in key_list:
            current = self._trade_index.get(key)
            if current is trade:
                self._trade_index.pop(key, None)
        self._trade_keys[id(trade)] = []

    def _candidate_keys(self, trade: _TradeRecord) -> List[str]:
        keys: List[str] = []
        if trade.order_id:
            keys.append(f"id:{trade.side}:{trade.order_id.lower()}")
        timestamp_component = f"{trade.timestamp.timestamp():.6f}"
        quantity_component = f"{trade.quantity:.12f}"
        price_component = f"{trade.price:.12f}"
        symbol_component = trade.symbol.upper()
        fallback = ":".join(
            (
                "fallback",
                trade.side,
                symbol_component,
                timestamp_component,
                quantity_component,
                price_component,
            )
        )
        keys.append(fallback)
        return keys

    def _resolve_method_and_threshold(self, jurisdiction: str) -> tuple[str, float]:
        jurisdiction_key = jurisdiction.lower()
        method_override = self._jurisdiction_overrides.get(jurisdiction_key)
        config_methods = self._config.get("jurisdictions", {})
        method = str(self._config.get("default_method", "fifo")).lower()
        long_term_days = float(self._default_long_term_days)
        if isinstance(config_methods, Mapping):
            entry = config_methods.get(jurisdiction_key)
            if isinstance(entry, Mapping):
                if "method" in entry:
                    method = str(entry["method"]).lower()
                if "long_term_days" in entry:
                    try:
                        long_term_days = float(entry["long_term_days"])
                    except (TypeError, ValueError):
                        long_term_days = float(self._default_long_term_days)
        if method_override:
            method = method_override.lower()
        return method, long_term_days

    def _resolve_tax_rates(self, jurisdiction: str) -> tuple[float, float]:
        jurisdiction_key = jurisdiction.lower()
        config_methods = self._config.get("jurisdictions", {})
        short_rate, long_rate = self._default_tax_rates
        if isinstance(config_methods, Mapping):
            entry = config_methods.get(jurisdiction_key)
            if isinstance(entry, Mapping):
                tax_rates = entry.get("tax_rates")
                if isinstance(tax_rates, Mapping):
                    short_override = self._coerce_rate(tax_rates.get("short_term"))
                    long_override = self._coerce_rate(tax_rates.get("long_term"))
                    if short_override is not None:
                        short_rate = short_override
                    if long_override is not None:
                        long_rate = long_override
        return short_rate, long_rate

    def _convert_to_base(
        self, amount: float, currency: Optional[str], timestamp: datetime
    ) -> float:
        if amount == 0.0:
            return 0.0
        currency_normalized = self._normalise_currency(currency)
        base_normalized = self._base_currency
        if not currency_normalized or not base_normalized:
            return amount
        if currency_normalized == base_normalized:
            return amount
        if self._fx_rate_provider is None:
            return amount
        try:
            rate = self._fx_rate_provider.get_rate(currency_normalized, timestamp)
        except KeyError:
            return amount
        return amount * rate

    def _normalise_currency(self, value: object) -> str | None:
        if isinstance(value, str):
            text = value.strip().upper()
            return text or None
        return None

    def _normalise_venue(self, value: object) -> Optional[str]:
        if isinstance(value, str):
            text = value.strip()
            if not text:
                return None
            return text.upper()
        return None

    def _build_calculator(self, method: str) -> CostBasisCalculator:
        method_lower = method.lower()
        if method_lower == "fifo":
            return FIFOCostBasisCalculator()
        if method_lower == "lifo":
            return LIFOCostBasisCalculator()
        if method_lower in {"average", "average_cost", "avg"}:
            return AverageCostBasisCalculator()
        raise ValueError(f"Nieobsługiwana metoda wyceny kosztu: {method}")

    def _compute_totals(
        self,
        events: Sequence[DisposalEvent],
        open_lots: Sequence[TaxLot],
        long_term_days: float,
    ) -> TaxReportTotals:
        proceeds = sum(event.proceeds for event in events)
        cost_basis = sum(event.cost_basis for event in events)
        fees = sum(event.fee for event in events) + sum(
            m.fee for event in events for m in event.matched_lots
        )
        realized_gain = sum(event.realized_gain for event in events)
        short_term_gain = sum(event.short_term_gain for event in events)
        long_term_gain = sum(event.long_term_gain for event in events)
        short_term_tax = sum(event.short_term_tax for event in events)
        long_term_tax = sum(event.long_term_tax for event in events)
        unrealized_cost_basis = sum(lot.cost_basis + lot.fee for lot in open_lots)
        unrealized_quantity = sum(lot.quantity for lot in open_lots)
        open_short_term_cost_basis = 0.0
        open_long_term_cost_basis = 0.0
        open_short_term_quantity = 0.0
        open_long_term_quantity = 0.0
        open_weighted_days = 0.0
        threshold = max(0.0, long_term_days)
        for lot in open_lots:
            open_weighted_days += lot.holding_period_days * lot.quantity
            if lot.holding_period_days >= threshold:
                open_long_term_quantity += lot.quantity
                open_long_term_cost_basis += lot.cost_basis + lot.fee
            else:
                open_short_term_quantity += lot.quantity
                open_short_term_cost_basis += lot.cost_basis + lot.fee
        total_open_quantity = open_short_term_quantity + open_long_term_quantity
        average_open_holding_period_days = (
            open_weighted_days / total_open_quantity if total_open_quantity > 0 else 0.0
        )
        short_term_quantity = sum(event.short_term_quantity for event in events)
        long_term_quantity = sum(event.long_term_quantity for event in events)
        disposed_quantity = short_term_quantity + long_term_quantity
        weighted_holding_days = sum(
            event.average_holding_period_days
            * (event.short_term_quantity + event.long_term_quantity)
            for event in events
        )
        average_holding_period_days = (
            weighted_holding_days / disposed_quantity if disposed_quantity > 0 else 0.0
        )
        total_tax_liability = short_term_tax + long_term_tax
        return TaxReportTotals(
            proceeds=proceeds,
            cost_basis=cost_basis,
            fees=fees,
            realized_gain=realized_gain,
            short_term_gain=short_term_gain,
            long_term_gain=long_term_gain,
            unrealized_cost_basis=unrealized_cost_basis,
            unrealized_quantity=unrealized_quantity,
            short_term_quantity=short_term_quantity,
            long_term_quantity=long_term_quantity,
            average_holding_period_days=average_holding_period_days,
            unrealized_short_term_cost_basis=open_short_term_cost_basis,
            unrealized_long_term_cost_basis=open_long_term_cost_basis,
            unrealized_short_term_quantity=open_short_term_quantity,
            unrealized_long_term_quantity=open_long_term_quantity,
            average_open_holding_period_days=average_open_holding_period_days,
            short_term_tax=short_term_tax,
            long_term_tax=long_term_tax,
            total_tax_liability=total_tax_liability,
        )

    def _compute_asset_breakdown(
        self,
        events: Sequence[DisposalEvent],
        open_lots: Sequence[TaxLot],
        long_term_days: float,
    ) -> List[AssetBreakdown]:
        summary = self._aggregate_breakdown(
            events,
            open_lots,
            long_term_days,
            key_fn_event=lambda event: event.asset,
            key_fn_lot=lambda lot: lot.asset,
        )
        breakdown = [
            AssetBreakdown(
                asset=str(key) if key is not None else "*",
                proceeds=data["proceeds"],
                cost_basis=data["cost_basis"],
                fees=data["fees"],
                realized_gain=data["realized_gain"],
                disposed_quantity=data["disposed_quantity"],
                open_quantity=data["open_quantity"],
                open_cost_basis=data["open_cost_basis"],
                short_term_gain=data["short_term_gain"],
                long_term_gain=data["long_term_gain"],
                short_term_quantity=data["short_term_quantity"],
                long_term_quantity=data["long_term_quantity"],
                average_holding_period_days=(
                    data["holding_days_weighted_sum"] / data["disposed_quantity"]
                    if data["disposed_quantity"] > 0
                    else 0.0
                ),
                open_short_term_quantity=data["open_short_term_quantity"],
                open_long_term_quantity=data["open_long_term_quantity"],
                open_short_term_cost_basis=data["open_short_term_cost_basis"],
                open_long_term_cost_basis=data["open_long_term_cost_basis"],
                open_average_holding_period_days=(
                    data["open_holding_days_weighted_sum"] / data["open_quantity"]
                    if data["open_quantity"] > 0
                    else 0.0
                ),
                short_term_tax=data["short_term_tax"],
                long_term_tax=data["long_term_tax"],
                total_tax_liability=data["short_term_tax"] + data["long_term_tax"],
            )
            for key, data in summary.items()
        ]
        return sorted(breakdown, key=lambda item: item.asset)

    def _compute_venue_breakdown(
        self,
        events: Sequence[DisposalEvent],
        open_lots: Sequence[TaxLot],
        long_term_days: float,
    ) -> List[VenueBreakdown]:
        summary = self._aggregate_breakdown(
            events,
            open_lots,
            long_term_days,
            key_fn_event=lambda event: event.venue,
            key_fn_lot=lambda lot: lot.venue,
        )
        breakdown = [
            VenueBreakdown(
                venue=key,
                proceeds=data["proceeds"],
                cost_basis=data["cost_basis"],
                fees=data["fees"],
                realized_gain=data["realized_gain"],
                disposed_quantity=data["disposed_quantity"],
                open_quantity=data["open_quantity"],
                open_cost_basis=data["open_cost_basis"],
                short_term_gain=data["short_term_gain"],
                long_term_gain=data["long_term_gain"],
                short_term_quantity=data["short_term_quantity"],
                long_term_quantity=data["long_term_quantity"],
                average_holding_period_days=(
                    data["holding_days_weighted_sum"] / data["disposed_quantity"]
                    if data["disposed_quantity"] > 0
                    else 0.0
                ),
                open_short_term_quantity=data["open_short_term_quantity"],
                open_long_term_quantity=data["open_long_term_quantity"],
                open_short_term_cost_basis=data["open_short_term_cost_basis"],
                open_long_term_cost_basis=data["open_long_term_cost_basis"],
                open_average_holding_period_days=(
                    data["open_holding_days_weighted_sum"] / data["open_quantity"]
                    if data["open_quantity"] > 0
                    else 0.0
                ),
                short_term_tax=data["short_term_tax"],
                long_term_tax=data["long_term_tax"],
                total_tax_liability=data["short_term_tax"] + data["long_term_tax"],
            )
            for key, data in summary.items()
        ]
        return sorted(breakdown, key=lambda item: (item.venue or ""))

    def _compute_period_breakdown(
        self,
        events: Sequence[DisposalEvent],
    ) -> List[PeriodBreakdown]:
        summary: Dict[datetime, Dict[str, float]] = {}

        def ensure(key: datetime) -> Dict[str, float]:
            bucket = summary.get(key)
            if bucket is None:
                bucket = {
                    "proceeds": 0.0,
                    "cost_basis": 0.0,
                    "fees": 0.0,
                    "realized_gain": 0.0,
                    "disposed_quantity": 0.0,
                    "short_term_gain": 0.0,
                    "long_term_gain": 0.0,
                    "short_term_quantity": 0.0,
                    "long_term_quantity": 0.0,
                    "holding_days_weighted_sum": 0.0,
                    "short_term_tax": 0.0,
                    "long_term_tax": 0.0,
                }
                summary[key] = bucket
            return bucket

        for event in events:
            disposal_time = event.disposal_time
            if disposal_time.tzinfo is None:
                disposal_time = disposal_time.replace(tzinfo=timezone.utc)
            period_start = self._period_start(disposal_time)
            bucket = ensure(period_start)
            bucket["proceeds"] += event.proceeds
            bucket["cost_basis"] += event.cost_basis
            bucket["fees"] += event.fee
            bucket["fees"] += sum(matched.fee for matched in event.matched_lots)
            bucket["realized_gain"] += event.realized_gain
            bucket["disposed_quantity"] += event.quantity
            bucket["short_term_gain"] += event.short_term_gain
            bucket["long_term_gain"] += event.long_term_gain
            bucket["short_term_quantity"] += event.short_term_quantity
            bucket["long_term_quantity"] += event.long_term_quantity
            bucket["short_term_tax"] += event.short_term_tax
            bucket["long_term_tax"] += event.long_term_tax
            disposed_quantity = event.short_term_quantity + event.long_term_quantity
            bucket["holding_days_weighted_sum"] += (
                event.average_holding_period_days * disposed_quantity
            )

        breakdown: List[PeriodBreakdown] = []
        for period_start, data in summary.items():
            disposed_quantity = data["short_term_quantity"] + data["long_term_quantity"]
            average_days = (
                data["holding_days_weighted_sum"] / disposed_quantity
                if disposed_quantity > 0
                else 0.0
            )
            period_label = f"{period_start.year:04d}-{period_start.month:02d}"
            period_end = self._period_end(period_start)
            breakdown.append(
                PeriodBreakdown(
                    period=period_label,
                    period_start=period_start,
                    period_end=period_end,
                    proceeds=data["proceeds"],
                    cost_basis=data["cost_basis"],
                    fees=data["fees"],
                    realized_gain=data["realized_gain"],
                    disposed_quantity=data["disposed_quantity"],
                    short_term_gain=data["short_term_gain"],
                    long_term_gain=data["long_term_gain"],
                    short_term_quantity=data["short_term_quantity"],
                    long_term_quantity=data["long_term_quantity"],
                    average_holding_period_days=average_days,
                    short_term_tax=data["short_term_tax"],
                    long_term_tax=data["long_term_tax"],
                    total_tax_liability=data["short_term_tax"] + data["long_term_tax"],
                )
            )
        return sorted(breakdown, key=lambda item: item.period_start)

    def _aggregate_breakdown(
        self,
        events: Sequence[DisposalEvent],
        open_lots: Sequence[TaxLot],
        long_term_days: float,
        *,
        key_fn_event: Callable[[DisposalEvent], Optional[str]],
        key_fn_lot: Callable[[TaxLot], Optional[str]],
    ) -> Dict[Optional[str], Dict[str, float]]:
        summary: Dict[Optional[str], Dict[str, float]] = {}

        def ensure(key: Optional[str]) -> Dict[str, float]:
            bucket = summary.get(key)
            if bucket is None:
                bucket = {
                    "proceeds": 0.0,
                    "cost_basis": 0.0,
                    "fees": 0.0,
                    "realized_gain": 0.0,
                    "disposed_quantity": 0.0,
                    "open_quantity": 0.0,
                    "open_cost_basis": 0.0,
                    "short_term_gain": 0.0,
                    "long_term_gain": 0.0,
                    "short_term_quantity": 0.0,
                    "long_term_quantity": 0.0,
                    "short_term_tax": 0.0,
                    "long_term_tax": 0.0,
                    "holding_days_weighted_sum": 0.0,
                    "open_short_term_quantity": 0.0,
                    "open_long_term_quantity": 0.0,
                    "open_short_term_cost_basis": 0.0,
                    "open_long_term_cost_basis": 0.0,
                    "open_holding_days_weighted_sum": 0.0,
                }
                summary[key] = bucket
            return bucket

        for event in events:
            key = key_fn_event(event)
            bucket = ensure(key)
            bucket["proceeds"] += event.proceeds
            bucket["cost_basis"] += event.cost_basis
            bucket["fees"] += event.fee
            bucket["fees"] += sum(matched.fee for matched in event.matched_lots)
            bucket["realized_gain"] += event.realized_gain
            bucket["disposed_quantity"] += event.quantity
            bucket["short_term_gain"] += event.short_term_gain
            bucket["long_term_gain"] += event.long_term_gain
            bucket["short_term_quantity"] += event.short_term_quantity
            bucket["long_term_quantity"] += event.long_term_quantity
            bucket["short_term_tax"] += event.short_term_tax
            bucket["long_term_tax"] += event.long_term_tax
            disposed_quantity = event.short_term_quantity + event.long_term_quantity
            bucket["holding_days_weighted_sum"] += (
                event.average_holding_period_days * disposed_quantity
            )

        threshold = max(0.0, long_term_days)
        for lot in open_lots:
            key = key_fn_lot(lot)
            bucket = ensure(key)
            bucket["open_quantity"] += lot.quantity
            bucket["open_cost_basis"] += lot.cost_basis + lot.fee
            bucket["open_holding_days_weighted_sum"] += (
                lot.holding_period_days * lot.quantity
            )
            if lot.holding_period_days >= threshold:
                bucket["open_long_term_quantity"] += lot.quantity
                bucket["open_long_term_cost_basis"] += lot.cost_basis + lot.fee
            else:
                bucket["open_short_term_quantity"] += lot.quantity
                bucket["open_short_term_cost_basis"] += lot.cost_basis + lot.fee

        return summary

    def _annotate_open_lots(self, lots: Sequence[TaxLot], as_of: datetime) -> None:
        if as_of.tzinfo is None:
            as_of = as_of.replace(tzinfo=timezone.utc)
        for lot in lots:
            acquisition = lot.acquisition_time
            if acquisition.tzinfo is None:
                acquisition = acquisition.replace(tzinfo=timezone.utc)
            delta = as_of - acquisition
            holding_days = max(0.0, delta.total_seconds() / 86400.0)
            lot.holding_period_days = holding_days

    def _annotate_holding_period(
        self,
        event: DisposalEvent,
        long_term_days: float,
        tax_rates: tuple[float, float],
    ) -> None:
        total_quantity = sum(m.quantity for m in event.matched_lots) or event.quantity
        if total_quantity <= 0:
            event.short_term_gain = 0.0
            event.long_term_gain = 0.0
            event.short_term_tax = 0.0
            event.long_term_tax = 0.0
            return
        short_gain = 0.0
        long_gain = 0.0
        short_quantity = 0.0
        long_quantity = 0.0
        weighted_holding_days = 0.0
        for matched in event.matched_lots:
            ratio = matched.quantity / total_quantity if total_quantity else 0.0
            proceeds_share = event.proceeds * ratio
            fee_share = event.fee * ratio
            gain = proceeds_share - fee_share - (matched.cost_basis + matched.fee)
            if abs(gain) < 1e-9:
                gain = 0.0
            if matched.holding_period_days >= max(0.0, long_term_days):
                long_gain += gain
                long_quantity += matched.quantity
            else:
                short_gain += gain
                short_quantity += matched.quantity
            weighted_holding_days += matched.holding_period_days * matched.quantity
        correction = event.realized_gain - (short_gain + long_gain)
        if abs(correction) > 1e-6:
            if abs(long_gain) >= abs(short_gain):
                long_gain += correction
            else:
                short_gain += correction
        elif abs(correction) > 1e-9:
            long_gain += correction
        event.short_term_gain = short_gain
        event.long_term_gain = long_gain
        disposed_quantity = short_quantity + long_quantity
        remainder = total_quantity - disposed_quantity
        if remainder > 1e-9:
            short_quantity += remainder
        disposed_quantity = short_quantity + long_quantity
        event.short_term_quantity = short_quantity
        event.long_term_quantity = long_quantity
        if disposed_quantity > 0:
            event.average_holding_period_days = weighted_holding_days / disposed_quantity
        else:
            event.average_holding_period_days = 0.0
        short_rate, long_rate = tax_rates
        event.short_term_tax = max(0.0, short_gain) * short_rate
        event.long_term_tax = max(0.0, long_gain) * long_rate

    def _normalise_remaining(self, calculator: CostBasisCalculator, asset: str) -> List[TaxLot]:
        leftovers: List[TaxLot] = []
        for lot in calculator.remaining_lots():
            if lot.asset == "*":
                lot = TaxLot(
                    lot_id=lot.lot_id,
                    asset=asset,
                    acquisition_time=lot.acquisition_time,
                    quantity=lot.quantity,
                    cost_basis=lot.cost_basis,
                    fee=lot.fee,
                    venue=lot.venue,
                    source=lot.source,
                )
            leftovers.append(lot)
        return leftovers

    def _infer_asset(self, symbol: str) -> str:
        text = symbol.upper()
        if ":" in text:
            text = text.split(":", 1)[1]
        for separator in ("/", "-", "_"):
            if separator in text:
                return text.split(separator)[0]
        for quote in self._quote_currencies:
            if text.endswith(quote) and len(text) > len(quote):
                return text[: -len(quote)]
        return text

    def _infer_quote(self, symbol: str) -> Optional[str]:
        text = symbol.upper()
        if ":" in text:
            text = text.split(":", 1)[1]
        for separator in ("/", "-", "_"):
            if separator in text:
                return text.split(separator)[1]
        for quote in self._quote_currencies:
            if text.endswith(quote) and len(text) > len(quote):
                return quote
        return None

    def _period_start(self, moment: datetime) -> datetime:
        if moment.tzinfo is None:
            moment = moment.replace(tzinfo=timezone.utc)
        return moment.replace(day=1, hour=0, minute=0, second=0, microsecond=0)

    def _period_end(self, period_start: datetime) -> datetime:
        if period_start.tzinfo is None:
            period_start = period_start.replace(tzinfo=timezone.utc)
        year = period_start.year
        month = period_start.month
        if month == 12:
            next_month = period_start.replace(
                year=year + 1,
                month=1,
                day=1,
                hour=0,
                minute=0,
                second=0,
                microsecond=0,
            )
        else:
            next_month = period_start.replace(
                year=year,
                month=month + 1,
                day=1,
                hour=0,
                minute=0,
                second=0,
                microsecond=0,
            )
        return next_month - timedelta(microseconds=1)
