# event_emitter_adapter.py
# -*- coding: utf-8 -*-
from __future__ import annotations

import time
import threading
import random
import logging
import traceback
import queue
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Union, DefaultDict, Tuple
from collections import defaultdict


# ===== Public API: Event & EventType =============================================================

@dataclass
class Event:
    """
    Prosty obiekt zdarzenia przekazywany do subskrybentów.
    Gwarancja: callback NIGDY nie dostanie None — zawsze Event albo (dla debounca) list[Event].
    """
    type: Union[str, Any]
    payload: Optional[dict] = None
    ts: float = field(default_factory=time.time)


class EventType:
    """
    Nazwy zdarzeń w jednym miejscu. Utrzymujemy je jako stringi dla prostoty i kompatybilności.
    """
    MARKET_TICK      = "MARKET_TICK"
    SIGNAL           = "SIGNAL"
    ORDER_REQUEST    = "ORDER_REQUEST"
    ORDER_STATUS     = "ORDER_STATUS"
    TRADE_EXECUTED   = "TRADE_EXECUTED"
    POSITION_UPDATE  = "POSITION_UPDATE"
    PNL_UPDATE       = "PNL_UPDATE"
    METRICS          = "METRICS"
    ATR_UPDATE       = "ATR_UPDATE"
    ATR_SPIKE        = "ATR_SPIKE"
    WFO_TRIGGER      = "WFO_TRIGGER"
    WFO_STATUS       = "WFO_STATUS"
    AUTOTRADE_STATUS = "AUTOTRADE_STATUS"
    RISK_ALERT       = "RISK_ALERT"
    LOG              = "LOG"


# ===== Debounce / Batch rule =====================================================================

class DebounceRule:
    """
    Reguła kolejkowania zdarzeń dla danego subskrybenta:
      - window: ile sekund „zbierać” zdarzenia zanim dostarczymy batch
      - max_batch: maksymalny rozmiar batcha; przekroczony -> natychmiastowy flush

    Kompatybilność:
      - można podać window_sec=..., throttle=..., throttle_sec=... zamiast window
    """
    def __init__(self,
                 window: Optional[float] = None,
                 max_batch: int = 50,
                 **kwargs: Any) -> None:
        if window is None:
            window = kwargs.pop("window_sec", None)
        if window is None:
            window = kwargs.pop("throttle", None)
        if window is None:
            window = kwargs.pop("throttle_sec", None)
        if window is None:
            window = 0.2
        self.window: float = float(window)
        self.max_batch: int = int(max_batch)
        self.deliver_list: bool = bool(kwargs.pop("deliver_list", True))


# ===== EventBus ==================================================================================

class EventBus:
    """
    Lekki event-bus z obsługą:
      - subscribe(event_type, callback, rule=None)
      - unsubscribe(event_type, callback)
      - publish(event_type, payload)      # główna metoda
      - emit(...) / emit_event(...) / post(...)  # aliasy
    Debounce: gdy sub ma DebounceRule, callback dostaje list[Event] (batch).
    Bez DebounceRule: callback dostaje pojedynczy Event.
    """

    _Callback = Callable[[Any], None]  # Any => Event lub list[Event]

    @dataclass
    class _Sub:
        callback: _Callback
        rule: Optional[DebounceRule] = None
        buffer: List[Event] = field(default_factory=list)
        timer: Optional[threading.Timer] = None
        lock: threading.Lock = field(default_factory=threading.Lock)

    def __init__(self) -> None:
        self._subs: DefaultDict[str, List[EventBus._Sub]] = defaultdict(list)
        self._lock = threading.Lock()
        self._closed = False
        self._queue: "queue.Queue[Optional[Event]]" = queue.Queue()
        self._thread: Optional[threading.Thread] = None
        self._async_mode = False

    def start(self) -> None:
        with self._lock:
            if self._thread and self._thread.is_alive():
                return
            self._closed = False
            self._async_mode = True
            self._thread = threading.Thread(target=self._run, name="EventBus", daemon=True)
            self._thread.start()

    def stop(self) -> None:
        with self._lock:
            self._closed = True
            if not self._async_mode:
                return
        self._queue.put(None)
        if self._thread:
            try:
                self._thread.join(timeout=1.0)
            except Exception:
                pass
        with self._lock:
            self._async_mode = False
            self._thread = None

    def subscribe(self,
                  event_type: Union[str, Any],
                  callback: _Callback,
                  rule: Optional[DebounceRule] = None) -> None:
        key = self._key(event_type)
        sub = EventBus._Sub(callback=callback, rule=rule)
        with self._lock:
            self._subs[key].append(sub)

    def unsubscribe(self,
                    event_type: Union[str, Any],
                    callback: _Callback) -> None:
        key = self._key(event_type)
        with self._lock:
            lst = self._subs.get(key, [])
            self._subs[key] = [s for s in lst if s.callback is not callback]

    def publish(self,
                event_type: Union[str, Any],
                payload: Optional[dict] = None) -> None:
        if self._closed:
            return
        evt = Event(type=self._key(event_type), payload=payload)
        if self._async_mode:
            self._queue.put(evt)
        else:
            self._dispatch(evt)

    # Aliasy zgodności
    def emit(self, event_type: Union[str, Any], payload: Optional[dict] = None) -> None:
        self.publish(event_type, payload)

    def emit_event(self, event_type: Union[str, Any], payload: Optional[dict] = None) -> None:
        self.publish(event_type, payload)

    def post(self, event_type: Union[str, Any], payload: Optional[dict] = None) -> None:
        self.publish(event_type, payload)

    def close(self) -> None:
        self.stop()
        with self._lock:
            for subs in self._subs.values():
                for s in subs:
                    with s.lock:
                        if s.timer is not None:
                            try:
                                s.timer.cancel()
                            except Exception:
                                pass
                            s.timer = None

    def _key(self, event_type: Union[str, Any]) -> str:
        if isinstance(event_type, str):
            return event_type
        return f"{event_type}"

    def _dispatch(self, evt: Event) -> None:
        key = self._key(evt.type)
        with self._lock:
            subscribers = list(self._subs.get(key, []))

        for sub in subscribers:
            rule = sub.rule
            if rule is None:
                self._safe_call(sub.callback, evt)
                continue

            with sub.lock:
                sub.buffer.append(evt)
                if len(sub.buffer) >= max(1, rule.max_batch):
                    self._flush_batch(sub)
                    continue
                if sub.timer is None:
                    sub.timer = threading.Timer(rule.window, lambda: self._flush_batch(sub))
                    sub.timer.daemon = True
                    sub.timer.start()

    def _flush_batch(self, sub: "EventBus._Sub") -> None:
        with sub.lock:
            buf = sub.buffer
            sub.buffer = []
            t = sub.timer
            sub.timer = None
        if t is not None:
            try:
                t.cancel()
            except Exception:
                pass
        if not buf:
            return
        rule = sub.rule
        if rule is not None and not rule.deliver_list:
            for evt in buf:
                self._safe_call(sub.callback, evt)
        else:
            self._safe_call(sub.callback, buf)

    @staticmethod
    def _safe_call(cb: _Callback, arg: Any) -> None:
        try:
            cb(arg)
        except Exception as ex:
            try:
                logging.getLogger("event-bus").warning("Subscriber callback error: %s", ex)
            except Exception:
                pass

    def _run(self) -> None:
        while True:
            try:
                evt = self._queue.get(timeout=0.25)
            except queue.Empty:
                if self._closed:
                    break
                continue
            if evt is None:
                break
            self._dispatch(evt)


@dataclass
class EmitterConfig:
    pf_min: float = 1.4
    expectancy_min: float = 0.0
    min_trades_for_pf: int = 20
    atr_period: int = 14
    atr_trigger_growth_pct: float = 30.0
    component: str = "EmitterAdapter"


class EventEmitter:
    """Wygodny wrapper na EventBus z obsługą tagów i logowania."""

    def __init__(self, bus: Optional[EventBus] = None) -> None:
        self.bus = bus or EventBus()
        self._logger = logging.getLogger("event-emitter")
        self._tag_map: DefaultDict[Tuple[str, str], List[Callable[[Any], None]]] = defaultdict(list)
        self._lock = threading.Lock()

    def on(
        self,
        event_type: Union[str, Any],
        callback: Callable[[Any], None],
        *,
        tag: Optional[str] = None,
        debounce: Optional[Union[DebounceRule, float]] = None,
    ) -> None:
        rule: Optional[DebounceRule]
        if debounce is None:
            rule = None
        elif isinstance(debounce, DebounceRule):
            rule = debounce
        else:
            rule = DebounceRule(window=float(debounce))
        self.bus.subscribe(event_type, callback, rule=rule)
        if tag:
            key = (self._key(event_type), tag)
            with self._lock:
                self._tag_map[key].append(callback)

    def off(
        self,
        event_type: Union[str, Any],
        callback: Optional[Callable[[Any], None]] = None,
        *,
        tag: Optional[str] = None,
    ) -> None:
        key = self._key(event_type)
        if callback is not None:
            self.bus.unsubscribe(event_type, callback)
            with self._lock:
                for k in list(self._tag_map.keys()):
                    if k[0] != key:
                        continue
                    if callback in self._tag_map[k]:
                        self._tag_map[k].remove(callback)
                    if not self._tag_map[k]:
                        del self._tag_map[k]
            return

        if tag is not None:
            with self._lock:
                callbacks = self._tag_map.pop((key, tag), [])
            for cb in callbacks:
                self.bus.unsubscribe(event_type, cb)
            return

        with self._lock:
            for k in [k for k in self._tag_map if k[0] == key]:
                for cb in self._tag_map.pop(k, []):
                    self.bus.unsubscribe(event_type, cb)

    def emit(self, event_type: Union[str, Any], **payload: Any) -> None:
        self.bus.publish(event_type, payload)

    def emit_event(self, event_type: Union[str, Any], **payload: Any) -> None:
        self.emit(event_type, **payload)

    def log(self, message: str, level: str = "INFO", *, component: Optional[str] = None, **extra: Any) -> None:
        lvl = level.upper()
        comp = component or "EventEmitter"
        record = {
            "message": message,
            "level": lvl,
            "component": comp,
            "ts": time.time(),
        }
        if extra:
            record.update(extra)
        try:
            self.bus.publish(EventType.LOG, record)
        except Exception:
            self._logger.debug("Failed to emit log event", exc_info=True)
        log_method = getattr(self._logger, lvl.lower(), self._logger.info)
        log_method("[%s] %s", comp, message)

    def close(self) -> None:
        try:
            self.bus.close()
        finally:
            with self._lock:
                self._tag_map.clear()

    @staticmethod
    def _key(event_type: Union[str, Any]) -> str:
        if isinstance(event_type, str):
            return event_type
        return f"{event_type}"
# ===== EmitterAdapter (zgodność z dotychczasowymi importami) ====================================

class EmitterAdapter:
    """Adapter zapewniający metody ułatwiające komunikację z GUI/serwisami."""

    def __init__(
        self,
        bus: Optional[EventBus] = None,
        cfg: Optional[EmitterConfig] = None,
        **kwargs: Any,
    ) -> None:
        if bus is None:
            bus = EventBus()
        self.bus = bus
        self.bus.start()
        cfg_overrides = {k: kwargs.pop(k) for k in list(kwargs.keys()) if hasattr(EmitterConfig, k)}
        self.cfg = cfg or EmitterConfig(**cfg_overrides)
        self.emitter = EventEmitter(self.bus)

    # proxy helpers
    def subscribe(self, *a, **kw) -> None:
        return self.bus.subscribe(*a, **kw)

    def unsubscribe(self, *a, **kw) -> None:
        return self.bus.unsubscribe(*a, **kw)

    def publish(self, *a, **kw) -> None:
        return self.bus.publish(*a, **kw)

    def emit(self, *a, **kw) -> None:
        return self.bus.emit(*a, **kw)

    def emit_event(self, *a, **kw) -> None:
        return self.bus.emit_event(*a, **kw)

    def post(self, *a, **kw) -> None:
        return self.bus.post(*a, **kw)

    def log(self, message: str, level: str = "INFO", **extra: Any) -> None:
        self.emitter.log(message, level=level, component=self.cfg.component, **extra)

    def push_market_tick(self, symbol: str, *, price: float, ts: Optional[float] = None,
                         high: Optional[float] = None, low: Optional[float] = None,
                         close: Optional[float] = None, volume: Optional[float] = None) -> None:
        payload: Dict[str, Any] = {
            "symbol": symbol,
            "price": float(price),
            "ts": float(ts if ts is not None else time.time()),
        }
        if high is not None:
            payload["high"] = float(high)
        if low is not None:
            payload["low"] = float(low)
        if close is not None:
            payload["close"] = float(close)
        if volume is not None:
            payload["volume"] = float(volume)
        self.bus.publish(EventType.MARKET_TICK, payload)

    def push_signal(self, symbol: str, *, side: str, strength: Optional[float] = None,
                    confidence: Optional[float] = None, ts: Optional[float] = None) -> None:
        payload: Dict[str, Any] = {
            "symbol": symbol,
            "side": side,
            "ts": float(ts if ts is not None else time.time()),
        }
        if strength is not None:
            payload["strength"] = float(strength)
        if confidence is not None:
            payload["confidence"] = float(confidence)
        self.bus.publish(EventType.SIGNAL, payload)

    def push_order_status(self, **info: Any) -> None:
        payload = dict(info)
        payload.setdefault("ts", time.time())
        self.bus.publish(EventType.ORDER_STATUS, payload)

    def update_metrics(self, symbol: str, *, pf: Optional[float], expectancy: Optional[float],
                       trades: int, ts: Optional[float] = None, **extra: Any) -> None:
        payload: Dict[str, Any] = {
            "symbol": symbol,
            "pf": float(pf) if pf is not None else None,
            "expectancy": float(expectancy) if expectancy is not None else None,
            "trades": int(trades),
            "ts": float(ts if ts is not None else time.time()),
        }
        if extra:
            payload.update(extra)
        self.bus.publish(EventType.METRICS, payload)

        if trades >= self.cfg.min_trades_for_pf:
            triggers: List[str] = []
            if pf is not None and pf < self.cfg.pf_min:
                triggers.append("pf_drop")
            if expectancy is not None and expectancy < self.cfg.expectancy_min:
                triggers.append("expectancy_drop")
            if triggers:
                self.bus.publish(EventType.WFO_TRIGGER, {
                    "symbol": symbol,
                    "reason": "+".join(triggers),
                    "metrics": payload,
                    "ts": payload["ts"],
                })

    def close(self) -> None:
        self.emitter.close()


# ===== DummyMarketFeed ===========================================================================
@dataclass
class DummyMarketFeedConfig:
    symbol: str = "BTCUSDT"
    interval_sec: float = 1.0          # co ile sekund tick
    start_price: float = 30000.0
    drift_bps: float = 0.0             # dryf w bps na tick (0.01% = 1 bps)
    vol_bps: float = 10.0              # odchylenie standardowe w bps na tick
    max_ticks: Optional[int] = None    # None = nieskończenie
    seed: Optional[int] = None


class DummyMarketFeed:
    """
    Prosty generator ticków (random walk). Emituje EventType.MARKET_TICK na busie.

    Payload ticka:
      {
        "symbol": str,
        "price": float,
        "ts": float
      }

    **Zgodność wywołań**:
      ✓ DummyMarketFeed(bus, cfg=DummyMarketFeedConfig(...))
      ✓ DummyMarketFeed(adapter, symbol="BTCUSDT", start_price=..., tick_interval_s=1.0)
      ✓ DummyMarketFeed(bus, symbol="...", interval=..., drift=..., sigma=...)
    """
    def __init__(self,
                 bus_or_adapter: Optional[Union[EventBus, EmitterAdapter]] = None,
                 cfg: Optional[DummyMarketFeedConfig] = None,
                 **kwargs: Any) -> None:
        # Rozpoznaj bus z pierwszego parametru lub z kwargs
        bus = kwargs.pop("bus", None)
        adapter = kwargs.pop("adapter", None)

        if isinstance(bus_or_adapter, EmitterAdapter):
            bus = bus_or_adapter.bus
        elif isinstance(bus_or_adapter, EventBus):
            bus = bus_or_adapter

        if bus is None and isinstance(adapter, EmitterAdapter):
            bus = adapter.bus

        if bus is None:
            # gdy ktoś wywoła bez busa – tworzymy własny (niezalecane, ale działa)
            bus = EventBus()

        self.bus: EventBus = bus

        # Jeśli dostaliśmy już gotową konfigurację – użyj jej
        if isinstance(cfg, DummyMarketFeedConfig):
            self.cfg = cfg
        else:
            # Zbierz aliasy starych nazw
            symbol = kwargs.pop("symbol", None)
            start_price = kwargs.pop("start_price", None)

            # Interwał: aliasy
            interval_sec = kwargs.pop("interval_sec", None)
            interval_sec = interval_sec if interval_sec is not None else kwargs.pop("tick_interval_s", None)
            interval_sec = interval_sec if interval_sec is not None else kwargs.pop("tick_interval", None)
            interval_sec = interval_sec if interval_sec is not None else kwargs.pop("interval", None)
            interval_sec = interval_sec if interval_sec is not None else kwargs.pop("dt", None)

            # Dryf/zmienność: aliasy
            drift_bps = kwargs.pop("drift_bps", None)
            drift_bps = drift_bps if drift_bps is not None else kwargs.pop("mu_bps", None)
            drift_bps = drift_bps if drift_bps is not None else kwargs.pop("drift", None)  # dopuszczamy bps

            vol_bps = kwargs.pop("vol_bps", None)
            vol_bps = vol_bps if vol_bps is not None else kwargs.pop("sigma_bps", None)
            vol_bps = vol_bps if vol_bps is not None else kwargs.pop("sigma", None)        # dopuszczamy bps

            max_ticks = kwargs.pop("max_ticks", None)
            seed = kwargs.pop("seed", None)

            # Ustal wartości default
            c = DummyMarketFeedConfig()
            if symbol is not None:       c.symbol = str(symbol)
            if start_price is not None:  c.start_price = float(start_price)
            if interval_sec is not None: c.interval_sec = float(interval_sec)
            if drift_bps is not None:    c.drift_bps = float(drift_bps)
            if vol_bps is not None:      c.vol_bps = float(vol_bps)
            if max_ticks is not None:    c.max_ticks = int(max_ticks)
            if seed is not None:         c.seed = int(seed)
            self.cfg = c

        # Stan wewnętrzny
        self._thr: Optional[threading.Thread] = None
        self._stop = threading.Event()
        self._price = float(self.cfg.start_price)
        if self.cfg.seed is not None:
            random.seed(self.cfg.seed)

        # Jeśli ktoś podał dodatkowe śmieciowe kwargs – ignorujemy bez krzyczenia
        # (po to, by „raz a dobrze” wyciąć błąd-po-błędzie z nieznanym parametrem)

    def start(self) -> "DummyMarketFeed":
        if self._thr and self._thr.is_alive():
            return self
        self._stop.clear()
        self._thr = threading.Thread(target=self._run, name=f"DummyFeed-{self.cfg.symbol}", daemon=True)
        self._thr.start()
        return self

    def stop(self) -> None:
        self._stop.set()

    def join(self, timeout: Optional[float] = None) -> None:
        t = self._thr
        if t:
            t.join(timeout=timeout)

    def _run(self) -> None:
        tick_count = 0
        iv = max(0.01, float(self.cfg.interval_sec))
        drift = float(self.cfg.drift_bps) / 10_000.0
        vol = max(0.0, float(self.cfg.vol_bps)) / 10_000.0

        while not self._stop.is_set():
            # losowy krok: price *= (1 + drift + eps), eps ~ N(0, vol)
            eps = random.gauss(mu=0.0, sigma=vol)
            self._price *= (1.0 + drift + eps)
            now = time.time()
            self.bus.publish(EventType.MARKET_TICK, {"symbol": self.cfg.symbol, "price": self._price, "ts": now})
            tick_count += 1

            if self.cfg.max_ticks is not None and tick_count >= self.cfg.max_ticks:
                break

            time.sleep(iv)


# ===== Logi → EventBus (GUI) ====================================================================

class _BusLogHandler(logging.Handler):
    """
    logging.Handler, który pcha każdy log do EventBus jako EventType.LOG.
    """
    def __init__(self, bus: EventBus, level: int = logging.INFO, event_type: Union[str, Any] = EventType.LOG) -> None:
        super().__init__(level=level)
        self.bus = bus
        self.event_type = event_type
        self._attached_loggers: List[logging.Logger] = []

    def emit(self, record: logging.LogRecord) -> None:
        try:
            msg = self.format(record)
            payload = {
                "level": record.levelname,
                "logger": record.name,
                "message": msg,
                "ts": getattr(record, "created", time.time()),
            }
            # Załącz trace jeśli jest
            if record.exc_info:
                payload["trace"] = "".join(traceback.format_exception(*record.exc_info)).strip()
            self.bus.publish(self.event_type, payload)
        except Exception:
            # Nie wywalaj logowania globalnie
            pass


def wire_gui_logs_to_adapter(
    target: Optional[Union[EventBus, EmitterAdapter]] = None,
    *,
    bus: Optional[EventBus] = None,
    adapter: Optional[EmitterAdapter] = None,
    level: int = logging.INFO,
    logger_names: Optional[List[str]] = None,
    event_type: Union[str, Any] = EventType.LOG,
    formatter: Optional[logging.Formatter] = None,
) -> logging.Handler:
    """
    Spina logging do EventBus → GUI.

    Wywołania kompatybilne:
      wire_gui_logs_to_adapter(bus)
      wire_gui_logs_to_adapter(adapter)
      wire_gui_logs_to_adapter(bus=bus)
      wire_gui_logs_to_adapter(adapter=adapter)
      wire_gui_logs_to_adapter(adapter, logger_names=['runner','services.walkforward_service'])

    Zwraca handler; można go potem odpiąć przez `unwire_gui_logs_from_adapter(handler)`.
    """
    # Rozwiąż bus
    if bus is None and adapter is not None:
        bus = adapter.bus
    if bus is None and isinstance(target, EmitterAdapter):
        bus = target.bus
    if bus is None and isinstance(target, EventBus):
        bus = target
    if bus is None:
        raise ValueError("wire_gui_logs_to_adapter: brak EventBus/Adapter.")

    handler = _BusLogHandler(bus=bus, level=level, event_type=event_type)
    if formatter is None:
        formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    handler.setFormatter(formatter)

    # Domyślnie: root logger
    if not logger_names:
        logger = logging.getLogger()
        logger.addHandler(handler)
        handler._attached_loggers.append(logger)
        if logger.level > level or logger.level == logging.NOTSET:
            logger.setLevel(level)
    else:
        for name in logger_names:
            logger = logging.getLogger(name)
            logger.addHandler(handler)
            handler._attached_loggers.append(logger)
            if logger.level > level or logger.level == logging.NOTSET:
                logger.setLevel(level)

    return handler


def unwire_gui_logs_from_adapter(handler: logging.Handler) -> None:
    """
    Usuwa wcześniej podpięty handler z loggerów.
    """
    if isinstance(handler, _BusLogHandler):
        for lg in list(getattr(handler, "_attached_loggers", [])):
            try:
                lg.removeHandler(handler)
            except Exception:
                pass
        handler._attached_loggers = []
    else:
        # Próba best-effort: odpinamy z root
        try:
            logging.getLogger().removeHandler(handler)
        except Exception:
            pass


# ===== Alias kompatybilnościowy dla starszych importów ===========================================

# Niektóre pliki importują starą nazwę:
#   from event_emitter_adapter import EventEmitterAdapter
# Utrzymujemy alias, aby uniknąć błędów po aktualizacjach.
EventEmitterAdapter = EmitterAdapter

# Alias nazwowy na wypadek innych wariantów importów:
wire_logging_to_bus = wire_gui_logs_to_adapter

__all__ = [
    "Event",
    "EventType",
    "DebounceRule",
    "EventBus",
    "EventEmitter",
    "EmitterConfig",
    "EmitterAdapter",
    "EventEmitterAdapter",
    "DummyMarketFeed",
    "DummyMarketFeedConfig",
    "wire_gui_logs_to_adapter",
    "unwire_gui_logs_from_adapter",
    "wire_logging_to_bus",
]
