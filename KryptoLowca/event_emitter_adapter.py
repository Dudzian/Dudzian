# event_emitter_adapter.py
# -*- coding: utf-8 -*-
from __future__ import annotations

import time
import threading
import random
import logging
import traceback
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
        self._dispatch(evt)

    # Aliasy zgodności
    def emit(self, event_type: Union[str, Any], payload: Optional[dict] = None) -> None:
        self.publish(event_type, payload)

    def emit_event(self, event_type: Union[str, Any], payload: Optional[dict] = None) -> None:
        self.publish(event_type, payload)

    def post(self, event_type: Union[str, Any], payload: Optional[dict] = None) -> None:
        self.publish(event_type, payload)

    def close(self) -> None:
        with self._lock:
            self._closed = True
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


# ===== EmitterAdapter (zgodność z dotychczasowymi importami) ====================================

class EmitterAdapter:
    """
    Cienki wrapper zapewniający, że stare importy będą działały. Udostępnia `.bus`
    i te same metody co EventBus.
    """
    def __init__(self, bus: Optional[EventBus] = None) -> None:
        self.bus = bus or EventBus()

    # proxy
    def subscribe(self, *a, **kw) -> None:   return self.bus.subscribe(*a, **kw)
    def unsubscribe(self, *a, **kw) -> None: return self.bus.unsubscribe(*a, **kw)
    def publish(self, *a, **kw) -> None:     return self.bus.publish(*a, **kw)
    def emit(self, *a, **kw) -> None:        return self.bus.emit(*a, **kw)
    def emit_event(self, *a, **kw) -> None:  return self.bus.emit_event(*a, **kw)
    def post(self, *a, **kw) -> None:        return self.bus.post(*a, **kw)
    def close(self) -> None:                  return self.bus.close()


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
    "EmitterAdapter",
    "EventEmitterAdapter",
    "DummyMarketFeed",
    "DummyMarketFeedConfig",
    "wire_gui_logs_to_adapter",
    "unwire_gui_logs_from_adapter",
    "wire_logging_to_bus",
]
