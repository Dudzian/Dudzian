"""Kontrolery spinające warstwy: dane/strategia/ryzyko/egzekucja oraz alerty."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import (
    Any,
    Callable,
    Mapping,
    MutableMapping,
    Optional,
    Sequence,
    Mapping as TypingMapping,
)

# --- elastyczne importy (różne gałęzie mogą mieć różne ścieżki modułów) -----

# Alerts
from bot_core.alerts import AlertMessage, DefaultAlertRouter  # dostępne w obu gałęziach

# Execution
try:
    from bot_core.execution import ExecutionContext, ExecutionService  # re-eksport
except Exception:  # pragma: no cover
    from bot_core.execution.base import ExecutionContext, ExecutionService  # fallback

# Exchanges commons
from bot_core.exchanges.base import AccountSnapshot, OrderRequest, OrderResult

# Risk
try:
    from bot_core.risk import RiskEngine, RiskCheckResult  # re-eksport
except Exception:  # pragma: no cover
    from bot_core.risk.base import RiskEngine, RiskCheckResult  # fallback

# Strategy types (interfejsy)
try:
    from bot_core.strategies import StrategySignal, MarketSnapshot  # gałąź z interfejsami w pakiecie
except Exception:  # pragma: no cover
    from bot_core.strategies.base import StrategySignal, MarketSnapshot  # alternatywna ścieżka

# Dane OHLCV – w zależności od gałęzi
try:
    # wariant modułowy
    from bot_core.data.base import OHLCVRequest  # type: ignore
    from bot_core.data.ohlcv.backfill import OHLCVBackfillService  # type: ignore
    from bot_core.data.ohlcv.cache import CachedOHLCVSource  # type: ignore
except Exception:  # pragma: no cover
    # wariant monolityczny
    from bot_core.data.ohlcv import (  # type: ignore
        OHLCVBackfillService,
        CachedOHLCVSource,
    )

# Konfiguracja runtime kontrolerów – tylko jeśli istnieje w danej gałęzi
try:
    from bot_core.config.models import CoreConfig, ControllerRuntimeConfig  # type: ignore
except Exception:  # pragma: no cover
    CoreConfig = Any  # type: ignore
    ControllerRuntimeConfig = Any  # type: ignore

_LOGGER = logging.getLogger(__name__)


# =============================================================================
# TradingController – przetwarza sygnały (BUY/SELL), pilnuje ryzyka i wysyła alerty
# =============================================================================

def _as_timedelta(value: timedelta | float | int) -> timedelta:
    if isinstance(value, timedelta):
        return value
    seconds = float(value)
    if seconds < 0:
        raise ValueError("Czas health-check nie może być ujemny")
    return timedelta(seconds=seconds)


def _now() -> datetime:
    return datetime.now(timezone.utc)


@dataclass(slots=True)
class ControllerSignal:
    """Zbiera sygnał strategii wraz ze snapshotem rynku."""

    snapshot: MarketSnapshot
    signal: StrategySignal


@dataclass(slots=True, kw_only=True)
class TradingController:
    """Podstawowy kontroler zarządzający przepływem sygnałów i alertów.

    - wysyła alert o nadejściu sygnału,
    - wykonuje pre-trade checks w RiskEngine,
    - zleca egzekucję (ExecutionService),
    - wysyła alert o odrzuceniu oraz alert trybu awaryjnego (liquidation),
    - okresowo publikuje health-check kanałów alertowych.
    """

    risk_engine: RiskEngine
    execution_service: ExecutionService
    alert_router: DefaultAlertRouter
    account_snapshot_provider: Callable[[], AccountSnapshot]
    portfolio_id: str
    environment: str
    risk_profile: str
    order_metadata_defaults: Mapping[str, str] | None = None
    clock: Callable[[], datetime] = _now
    health_check_interval: timedelta | float | int = timedelta(hours=1)
    execution_metadata: Mapping[str, str] | None = None

    _clock: Callable[[], datetime] = field(init=False, repr=False)
    _health_interval: timedelta = field(init=False, repr=False)
    _execution_context: ExecutionContext = field(init=False, repr=False)
    _order_defaults: dict[str, str] = field(init=False, repr=False)
    _last_health_report: datetime = field(init=False, repr=False)
    _liquidation_alerted: bool = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._clock = self.clock
        self._health_interval = _as_timedelta(self.health_check_interval)
        metadata: MutableMapping[str, str] = {}
        if self.execution_metadata:
            metadata.update({str(k): str(v) for k, v in self.execution_metadata.items()})
        self._execution_context = ExecutionContext(
            portfolio_id=self.portfolio_id,
            risk_profile=self.risk_profile,
            environment=self.environment,
            metadata=metadata,
        )
        self._order_defaults = {str(k): str(v) for k, v in (self.order_metadata_defaults or {}).items()}
        self._last_health_report = self._clock()
        self._liquidation_alerted = False

    # ----------------------------------------------- API -----------------------------------------------
    def process_signals(self, signals: Sequence[StrategySignal]) -> list[OrderResult]:
        """Przetwarza listę sygnałów strategii i zarządza alertami."""
        results: list[OrderResult] = []
        for signal in signals:
            if signal.side.upper() not in {"BUY", "SELL"}:
                _LOGGER.debug("Pomijam sygnał %s o kierunku %s", signal.symbol, signal.side)
                continue
            try:
                result = self._handle_signal(signal)
            except Exception:  # noqa: BLE001
                _LOGGER.exception("Błąd podczas przetwarzania sygnału %s", signal)
                raise
            if result is not None:
                results.append(result)

        self.maybe_report_health()
        return results

    def maybe_report_health(self, *, force: bool = False) -> None:
        """Publikuje raport health-check, gdy minął interwał lub wymusimy wysyłkę."""
        if self._health_interval.total_seconds() == 0 and not force:
            return

        now = self._clock()
        if not force and now - self._last_health_report < self._health_interval:
            return

        snapshot = self.alert_router.health_snapshot()
        body_lines = []
        for channel, data in snapshot.items():
            status = data.get("status", "unknown")
            rest = {k: v for k, v in data.items() if k != "status"}
            details = ", ".join(f"{k}={v}" for k, v in rest.items())
            if details:
                body_lines.append(f"{channel}: {status} ({details})")
            else:
                body_lines.append(f"{channel}: {status}")
        body = "\n".join(body_lines) if body_lines else "Brak kanałów alertowych do zraportowania."

        message = AlertMessage(
            category="health",
            title="Raport health-check kanałów alertowych",
            body=body,
            severity="info",
            context={
                "environment": self.environment,
                "portfolio": self.portfolio_id,
                "risk_profile": self.risk_profile,
                "channel_count": str(len(snapshot)),
                "generated_at": now.isoformat(),
            },
        )
        _LOGGER.info("Publikuję raport health-check (%s kanałów)", len(snapshot))
        self.alert_router.dispatch(message)
        self._last_health_report = now

    # ------------------------------------------- internals ----------------------------------------------
    def _handle_signal(self, signal: StrategySignal) -> OrderResult | None:
        self._emit_signal_alert(signal)
        request = self._build_order_request(signal)
        account = self.account_snapshot_provider()
        risk_result = self.risk_engine.apply_pre_trade_checks(
            request,
            account=account,
            profile_name=self.risk_profile,
        )

        if not risk_result.allowed:
            self._emit_order_rejected_alert(signal, request, risk_result)
            self._handle_liquidation_state(risk_result)
            return None

        result = self.execution_service.execute(request, self._execution_context)
        self._handle_liquidation_state(risk_result)
        return result

    def _build_order_request(self, signal: StrategySignal) -> OrderRequest:
        metadata = dict(self._order_defaults)
        # w StrategySignal.metadata spodziewamy się m.in. quantity/price/order_type/time_in_force/client_order_id
        for k, v in signal.metadata.items():
            metadata[str(k)] = str(v)

        try:
            quantity = float(metadata["quantity"])
        except KeyError as exc:
            raise ValueError("Sygnał nie zawiera wielkości zlecenia (quantity)") from exc
        except ValueError as exc:
            raise ValueError("Wielkość zlecenia musi być liczbą zmiennoprzecinkową") from exc

        if quantity <= 0:
            raise ValueError("Wielkość zlecenia musi być dodatnia")

        price_value = metadata.get("price")
        price = float(price_value) if price_value is not None else None

        order_type = (metadata.get("order_type") or "market").upper()
        time_in_force = metadata.get("time_in_force")
        client_order_id = metadata.get("client_order_id")

        return OrderRequest(
            symbol=signal.symbol,
            side=signal.side.upper(),
            quantity=quantity,
            order_type=order_type,
            price=price,
            time_in_force=time_in_force,
            client_order_id=client_order_id,
        )

    def _emit_signal_alert(self, signal: StrategySignal) -> None:
        context: dict[str, str] = {
            "symbol": signal.symbol,
            "side": signal.side.upper(),
            "confidence": f"{signal.confidence:.2f}",
            "environment": self.environment,
            "risk_profile": self.risk_profile,
        }
        for key, value in signal.metadata.items():
            context[f"meta_{key}"] = str(value)

        message = AlertMessage(
            category="strategy",
            title=f"Sygnał {signal.side.upper()} dla {signal.symbol}",
            body="Strategia wygenerowała sygnał wymagający decyzji egzekucyjnej.",
            severity="info",
            context=context,
        )
        _LOGGER.debug("Publikuję alert sygnału %s %s", signal.side.upper(), signal.symbol)
        self.alert_router.dispatch(message)

    def _emit_order_rejected_alert(
        self,
        signal: StrategySignal,
        request: OrderRequest,
        risk_result: RiskCheckResult,
    ) -> None:
        adjustments = risk_result.adjustments or {}
        context: dict[str, str] = {
            "symbol": request.symbol,
            "side": request.side,
            "order_type": request.order_type,
            "quantity": f"{request.quantity:.8f}",
            "environment": self.environment,
            "risk_profile": self.risk_profile,
        }
        context.update({f"adjust_{k}": str(v) for k, v in adjustments.items()})

        message = AlertMessage(
            category="risk",
            title=f"Zlecenie odrzucone przez risk engine ({request.symbol})",
            body=risk_result.reason or "Brak szczegółów",
            severity="warning",
            context=context,
        )
        _LOGGER.warning(
            "Risk engine odrzucił zlecenie %s %s: %s",
            request.side,
            request.symbol,
            risk_result.reason,
        )
        self.alert_router.dispatch(message)

    def _handle_liquidation_state(self, risk_result: RiskCheckResult) -> None:
        if not self.risk_engine.should_liquidate(profile_name=self.risk_profile):
            if self._liquidation_alerted:
                _LOGGER.info("Profil %s wyszedł z trybu awaryjnego", self.risk_profile)
            self._liquidation_alerted = False
            return

        if self._liquidation_alerted:
            return

        reason = risk_result.reason or "Profil ryzyka w trybie awaryjnym – przekroczono limit straty lub obsunięcia."
        context = {
            "risk_profile": self.risk_profile,
            "environment": self.environment,
            "portfolio": self.portfolio_id,
        }
        message = AlertMessage(
            category="risk",
            title="Profil w trybie awaryjnym",
            body=reason,
            severity="critical",
            context=context,
        )
        _LOGGER.error("Profil %s przekroczył limity – wysyłam alert krytyczny", self.risk_profile)
        self.alert_router.dispatch(message)
        self._liquidation_alerted = True


# =============================================================================
# DailyTrendController – cykl: backfill -> strategia -> ryzyko -> egzekucja
# =============================================================================

@dataclass(slots=True)
class DailyTrendController:
    """Prosty kontroler realizujący cykl: backfill -> strategia -> ryzyko -> egzekucja.

    Ten kontroler jest niezależny od warstwy alertów – nadaje się do wsadowego
    uruchamiania testów walk-forward lub automatycznych backfilli + egzekucji.
    """

    core_config: CoreConfig
    environment_name: str
    controller_name: str
    symbols: Sequence[str]
    backfill_service: OHLCVBackfillService
    data_source: CachedOHLCVSource
    strategy: Any  # StrategyEngine kompatybilny: posiada on_data(snapshot)->Sequence[StrategySignal]
    risk_engine: RiskEngine
    execution_service: ExecutionService
    account_loader: Callable[[], AccountSnapshot]
    execution_context: ExecutionContext
    position_size: float = 1.0

    _environment: Any = field(init=False, repr=False)
    _runtime: ControllerRuntimeConfig = field(init=False, repr=False)
    _risk_profile: str = field(init=False, repr=False)
    _positions: dict[str, float] = field(init=False, repr=False, default_factory=dict)

    def __post_init__(self) -> None:
        if not self.symbols:
            raise ValueError("Wymagany jest przynajmniej jeden symbol do obsługi.")
        try:
            self._environment = self.core_config.environments[self.environment_name]
        except KeyError as exc:
            raise KeyError(f"Brak konfiguracji środowiska '{self.environment_name}' w CoreConfig") from exc
        try:
            self._runtime = self.core_config.runtime_controllers[self.controller_name]
        except Exception as exc:
            raise KeyError(f"Brak sekcji runtime dla kontrolera '{self.controller_name}' w CoreConfig") from exc
        self._risk_profile = self._environment.risk_profile

    @property
    def tick_seconds(self) -> float:
        return float(self._runtime.tick_seconds)

    @property
    def interval(self) -> str:
        return str(self._runtime.interval)

    def collect_signals(self, *, start: int, end: int) -> list[ControllerSignal]:
        """Zwraca sygnały strategii wzbogacone o parametry egzekucyjne."""

        if start > end:
            raise ValueError("Parametr start nie może być większy niż end")

        self.backfill_service.synchronize(
            symbols=self.symbols,
            interval=self.interval,
            start=start,
            end=end,
        )

        collected: list[ControllerSignal] = []
        for symbol in self.symbols:
            try:
                response = self.data_source.fetch_ohlcv(  # type: ignore[attr-defined]
                    OHLCVRequest(symbol=symbol, interval=self.interval, start=start, end=end)
                )
                columns: Sequence[str] = response.columns  # type: ignore[attr-defined]
                rows: Sequence[Sequence[float]] = response.rows  # type: ignore[attr-defined]
            except Exception:
                rows = self.data_source.fetch_ohlcv(symbol, self.interval, start=start, end=end)  # type: ignore[misc]
                columns = ("open_time", "open", "high", "low", "close", "volume")

            snapshots = self._to_snapshots(symbol, columns, rows)
            for snapshot in snapshots:
                strategy_signals: Sequence[StrategySignal] = self.strategy.on_data(snapshot)
                for raw_signal in strategy_signals:
                    enriched = self._enrich_signal(raw_signal, snapshot)
                    if enriched is None:
                        continue
                    collected.append(ControllerSignal(snapshot=snapshot, signal=enriched))
        return collected

    def run_cycle(self, *, start: int, end: int) -> list[OrderResult]:
        """Przeprowadza pojedynczy cykl przetwarzania danych i składania zleceń."""

        collected = self.collect_signals(start=start, end=end)

        executed: list[OrderResult] = []
        for controller_signal in collected:
            executed.extend(
                self._handle_signals(controller_signal.snapshot, (controller_signal.signal,))
            )
        return executed

    # ----------------------------------------- helpers -----------------------------------------
    def _handle_signals(
        self,
        snapshot: MarketSnapshot,
        signals: Sequence[StrategySignal],
    ) -> list[OrderResult]:
        results: list[OrderResult] = []
        for signal in signals:
            base_request = self._build_order_request(snapshot, signal)
            account_snapshot = self.account_loader()
            risk_result = self.risk_engine.apply_pre_trade_checks(
                base_request,
                account=account_snapshot,
                profile_name=self._risk_profile,
            )
            if not risk_result.allowed:
                _LOGGER.info(
                    "Kontroler %s: sygnał %s dla %s odrzucony przez silnik ryzyka (%s)",
                    self.controller_name,
                    signal.side,
                    snapshot.symbol,
                    risk_result.reason,
                )
                continue

            quantity = base_request.quantity
            if risk_result.adjustments and "quantity" in risk_result.adjustments:
                quantity = float(risk_result.adjustments["quantity"])
            if quantity <= 0:
                _LOGGER.debug(
                    "Kontroler %s: dostosowana wielkość <= 0 dla %s – pomijam egzekucję.",
                    self.controller_name,
                    snapshot.symbol,
                )
                continue

            request = OrderRequest(
                symbol=base_request.symbol,
                side=base_request.side,
                quantity=quantity,
                order_type=base_request.order_type,
                price=base_request.price,
                time_in_force=base_request.time_in_force,
                client_order_id=base_request.client_order_id,
            )
            result = self.execution_service.execute(request, self.execution_context)
            self._post_fill(signal.side, snapshot.symbol, request, result)
            results.append(result)
        return results

    def _build_order_request(self, snapshot: MarketSnapshot, signal: StrategySignal) -> OrderRequest:
        side = signal.side.lower()
        metadata = dict(signal.metadata)
        quantity = float(metadata.get("quantity", self.position_size))
        price = float(metadata.get("price", snapshot.close))
        order_type = str(metadata.get("order_type", "market"))
        time_in_force = metadata.get("time_in_force")
        client_order_id = metadata.get("client_order_id")

        tif_str = str(time_in_force) if time_in_force is not None else None
        client_id_str = str(client_order_id) if client_order_id is not None else None

        return OrderRequest(
            symbol=snapshot.symbol,
            side=side,
            quantity=quantity,
            order_type=order_type,
            price=price,
            time_in_force=tif_str,
            client_order_id=client_id_str,
        )

    def _to_snapshots(
        self,
        symbol: str,
        columns: Sequence[str],
        rows: Sequence[Sequence[float]],
    ) -> list[MarketSnapshot]:
        if not rows:
            return []

        index: TypingMapping[str, int] = {column.lower(): idx for idx, column in enumerate(columns)}
        # open_time / timestamp
        try:
            open_time_idx = index["open_time"]
        except KeyError as exc:
            if "timestamp" in index:
                open_time_idx = index["timestamp"]
            else:
                raise ValueError("Brak kolumny open_time w danych OHLCV") from exc
        for key in ("open", "high", "low", "close"):
            if key not in index:
                raise ValueError(f"Brak kolumny '{key}' w danych OHLCV")
        open_idx = index["open"]
        high_idx = index["high"]
        low_idx = index["low"]
        close_idx = index["close"]
        volume_idx = index.get("volume")

        snapshots = [
            MarketSnapshot(
                symbol=symbol,
                timestamp=int(float(row[open_time_idx])),
                open=float(row[open_idx]),
                high=float(row[high_idx]),
                low=float(row[low_idx]),
                close=float(row[close_idx]),
                volume=float(row[volume_idx]) if volume_idx is not None else 0.0,
            )
            for row in rows
            if len(row) > close_idx
        ]
        snapshots.sort(key=lambda item: item.timestamp)
        return snapshots

    def _enrich_signal(
        self, signal: StrategySignal, snapshot: MarketSnapshot
    ) -> StrategySignal | None:
        side = signal.side.upper()
        if side not in {"BUY", "SELL"}:
            return None

        metadata: dict[str, object] = dict(signal.metadata)
        metadata.setdefault("quantity", float(self.position_size))
        metadata.setdefault("price", float(snapshot.close))
        metadata.setdefault("order_type", "market")

        return StrategySignal(
            symbol=snapshot.symbol,
            side=side,
            confidence=signal.confidence,
            metadata=metadata,
        )

    def _post_fill(
        self,
        side: str,
        symbol: str,
        request: OrderRequest,
        result: OrderResult,
    ) -> None:
        avg_price = result.avg_price or request.price or 0.0
        notional = avg_price * request.quantity
        side_lower = side.lower()
        pnl = 0.0
        if side_lower == "buy":
            self._positions[symbol] = avg_price
            position_value = notional
        else:
            entry_price = self._positions.pop(symbol, avg_price)
            pnl = (avg_price - entry_price) * request.quantity
            position_value = 0.0
        self.risk_engine.on_fill(
            profile_name=self._risk_profile,
            symbol=symbol,
            side=side_lower,
            position_value=position_value,
            pnl=pnl,
        )
        _LOGGER.info(
            "Kontroler %s: wykonano %s %s qty=%s avg_price=%s pnl=%s",
            self.controller_name,
            side_lower,
            symbol,
            request.quantity,
            avg_price,
            pnl,
        )


__all__ = ["TradingController", "DailyTrendController", "ControllerSignal"]
