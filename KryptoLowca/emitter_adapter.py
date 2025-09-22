# event_emitter_adapter.py
# -*- coding: utf-8 -*-
"""
Lekki adapter / emitter zdarzeń do integracji z Twoim GUI tradingowym.
- Prosty event-bus (publish/subscribe) z bezpieczną dla wątków dystrybucją.
- Typy zdarzeń: order, trade, position, metrics, log, reopt.
- Niskie zależności: wyłącznie standardowa biblioteka.
- Może też logować zdarzenia do SQLite (opcjonalnie).

Użycie w skrócie:
    from event_emitter_adapter import EventEmitter, Events

    bus = EventEmitter()
    bus.subscribe(Events.METRICS, lambda e: print("metrics:", e))
    bus.emit(Events.METRICS, {"pf": 1.7, "exp": 3.2})

Współpracuje z "run_trading_gui_paper_emitter.py", który uruchamia GUI i
równolegle monitoruje DB w trybie walk-forward + auto-reoptymalizacja.
"""
from __future__ import annotations

import json
import queue
import sqlite3
import threading
import time
from dataclasses import dataclass, asdict
from enum import Enum
from typing import Any, Callable, DefaultDict, Dict, List, Optional, Tuple
from collections import defaultdict


class Events(str, Enum):
    ORDER = "order"
    TRADE = "trade"
    POSITION = "position"
    METRICS = "metrics"
    ATR = "atr"
    LOG = "log"
    REOPT = "reopt"  # powiadomienie o konieczności reoptymalizacji


@dataclass
class Event:
    type: Events
    payload: Dict[str, Any]
    ts: float


class EventEmitter:
    """Bardzo prosty event-bus: bez zależności, thread-safe, asynchroniczny dispatch."""

    def __init__(self) -> None:
        self._subs: DefaultDict[Events, List[Callable[[Event], None]]] = defaultdict(list)
        self._q: "queue.Queue[Event]" = queue.Queue()
        self._stop = threading.Event()
        self._worker = threading.Thread(target=self._run, name="EventEmitter", daemon=True)
        self._worker.start()

    def subscribe(self, event_type: Events, callback: Callable[[Event], None]) -> None:
        self._subs[event_type].append(callback)

    def emit(self, event_type: Events, payload: Dict[str, Any]) -> None:
        self._q.put(Event(type=event_type, payload=payload, ts=time.time()))

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                evt = self._q.get(timeout=0.25)
            except queue.Empty:
                continue
            for cb in list(self._subs.get(evt.type, [])):
                try:
                    cb(evt)
                except Exception as e:  # pragma: no cover
                    # Nie przerywamy całego busa z powodu błędu subskrybenta
                    print(f"[EventEmitter] Błąd callbacka {cb}: {e}")

    def close(self) -> None:
        self._stop.set()
        self._worker.join(timeout=1.0)


# --- Opcjonalny logger do SQLite ------------------------------------------------

class SQLiteEventLogger:
    """Prosty logger zdarzeń do SQLite: przechowuje JSON payload + timestamp.
    Tabela: event_log(type TEXT, ts REAL, payload TEXT)
    """

    def __init__(self, db_path: str) -> None:
        self.db_path = db_path
        self._ensure_schema()

    def _ensure_schema(self) -> None:
        con = sqlite3.connect(self.db_path)
        try:
            cur = con.cursor()
            cur.execute(
                """CREATE TABLE IF NOT EXISTS event_log(
                       type TEXT NOT NULL,
                       ts   REAL NOT NULL,
                       payload TEXT NOT NULL
                   )"""
            )
            cur.execute(
                """CREATE INDEX IF NOT EXISTS ix_event_log_type_ts
                   ON event_log(type, ts)"""
            )
            con.commit()
        finally:
            con.close()

    def attach_to(self, bus: EventEmitter, types: Optional[List[Events]] = None) -> None:
        """Podłącza logger jako subskrybenta do wskazanych typów (domyślnie wszystkie)."""
        watch = types or list(Events)
        for t in watch:
            bus.subscribe(t, self._on_event)

    def _on_event(self, evt: Event) -> None:
        try:
            con = sqlite3.connect(self.db_path, timeout=5.0)
            cur = con.cursor()
            cur.execute(
                "INSERT INTO event_log(type, ts, payload) VALUES(?,?,?)",
                (evt.type.value, evt.ts, json.dumps(evt.payload, ensure_ascii=False)),
            )
            con.commit()
        except Exception as e:  # pragma: no cover
            print(f"[SQLiteEventLogger] Błąd zapisu: {e}")
        finally:
            try:
                con.close()
            except Exception:
                pass


# --- Użyteczne emitery krótkiej ręki -------------------------------------------

def emit_order(bus: EventEmitter, **kw: Any) -> None:
    bus.emit(Events.ORDER, kw)


def emit_trade(bus: EventEmitter, **kw: Any) -> None:
    bus.emit(Events.TRADE, kw)


def emit_position(bus: EventEmitter, **kw: Any) -> None:
    bus.emit(Events.POSITION, kw)


def emit_metrics(bus: EventEmitter, **kw: Any) -> None:
    bus.emit(Events.METRICS, kw)


def emit_atr(bus: EventEmitter, **kw: Any) -> None:
    bus.emit(Events.ATR, kw)


def emit_log(bus: EventEmitter, level: str, message: str, **extra: Any) -> None:
    payload = {"level": level, "message": message}
    payload.update(extra)
    bus.emit(Events.LOG, payload)


def emit_reopt(bus: EventEmitter, reason: str, details: Optional[Dict[str, Any]] = None) -> None:
    bus.emit(Events.REOPT, {"reason": reason, "details": details or {}})
