# services/persistence.py
# -*- coding: utf-8 -*-
from __future__ import annotations

import os
import json
import time
import sqlite3
import logging
from typing import Iterable, Union, Optional

try:
    from KryptoLowca.event_emitter_adapter import Event, EventType, EventBus
except Exception as e:
    raise ImportError(f"persistence: brak event_emitter_adapter ({e})")

log = logging.getLogger("services.persistence")


class PersistenceService:
    """
    Lekka trwała persystencja do SQLite:
      - TRADE_EXECUTED  -> tabela trades
      - ORDER_STATUS    -> order_status
      - POSITION_UPDATE -> positions
      - PNL_UPDATE      -> pnl
      - WFO_STATUS      -> wfo_status
      - AUTOTRADE_STATUS-> autotrade_status
    """
    def __init__(self, bus: EventBus, db_path: str = "data/runtime.db") -> None:
        self.bus = bus
        self.db_path = db_path
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self.conn.execute("PRAGMA journal_mode=WAL;")
        self._migrate()

        self.bus.subscribe(EventType.TRADE_EXECUTED, self._on_trades)
        self.bus.subscribe(EventType.ORDER_STATUS, self._on_order_status)
        self.bus.subscribe(EventType.POSITION_UPDATE, self._on_position)
        self.bus.subscribe(EventType.PNL_UPDATE, self._on_pnl)
        self.bus.subscribe(EventType.WFO_STATUS, self._on_wfo)
        self.bus.subscribe(EventType.AUTOTRADE_STATUS, self._on_autotrade)

        log.info("PersistenceService: SQLite @ %s", os.path.abspath(self.db_path))

    # --- utils -----------------------------------------------------------------------------------

    def _iter_events(self, x: Union[Event, Iterable[Event], None]) -> Iterable[Event]:
        if x is None:
            return []
        if isinstance(x, Event):
            return [x]
        try:
            return list(x)
        except Exception:
            return []

    def _migrate(self) -> None:
        cur = self.conn.cursor()
        cur.execute("""
        CREATE TABLE IF NOT EXISTS trades(
            ts REAL, symbol TEXT, side TEXT, qty REAL, price REAL,
            realized_pnl REAL, order_id TEXT
        )""")
        cur.execute("""
        CREATE TABLE IF NOT EXISTS order_status(
            ts REAL, order_id TEXT, status TEXT, symbol TEXT,
            side TEXT, qty REAL, price REAL
        )""")
        cur.execute("""
        CREATE TABLE IF NOT EXISTS positions(
            ts REAL, symbol TEXT, qty REAL, avg_price REAL
        )""")
        cur.execute("""
        CREATE TABLE IF NOT EXISTS pnl(
            ts REAL, symbol TEXT, cash REAL, position_qty REAL,
            avg_price REAL, last_price REAL, realized_pnl REAL,
            unrealized_pnl REAL, equity REAL
        )""")
        cur.execute("""
        CREATE TABLE IF NOT EXISTS wfo_status(
            ts REAL, symbol TEXT, phase TEXT, payload_json TEXT
        )""")
        cur.execute("""
        CREATE TABLE IF NOT EXISTS autotrade_status(
            ts REAL, component TEXT, action TEXT, symbol TEXT, payload_json TEXT
        )""")
        self.conn.commit()

    # --- handlers --------------------------------------------------------------------------------

    def _on_trades(self, evs):  # TRADE_EXECUTED
        cur = self.conn.cursor()
        rows = []
        for ev in self._iter_events(evs):
            p = ev.payload or {}
            rows.append((
                float(p.get("ts", time.time())),
                str(p.get("symbol", "")),
                str(p.get("side", "")),
                float(p.get("qty", 0.0)),
                float(p.get("price", 0.0)),
                float(p.get("realized_pnl", 0.0)),
                str(p.get("order_id", ""))
            ))
        if rows:
            cur.executemany("INSERT INTO trades VALUES (?,?,?,?,?,?,?)", rows)
            self.conn.commit()

    def _on_order_status(self, evs):  # ORDER_STATUS
        cur = self.conn.cursor()
        rows = []
        for ev in self._iter_events(evs):
            p = ev.payload or {}
            rows.append((
                float(p.get("ts", time.time())),
                str(p.get("order_id", "")),
                str(p.get("status", "")),
                str(p.get("symbol", "")),
                str(p.get("side", "")),
                float(p.get("qty", 0.0)),
                float(p.get("price", 0.0)),
            ))
        if rows:
            cur.executemany("INSERT INTO order_status VALUES (?,?,?,?,?,?,?)", rows)
            self.conn.commit()

    def _on_position(self, evs):  # POSITION_UPDATE
        cur = self.conn.cursor()
        rows = []
        for ev in self._iter_events(evs):
            p = ev.payload or {}
            rows.append((
                float(p.get("ts", time.time())),
                str(p.get("symbol", "")),
                float(p.get("qty", 0.0)),
                float(p.get("avg_price", 0.0)),
            ))
        if rows:
            cur.executemany("INSERT INTO positions VALUES (?,?,?,?)", rows)
            self.conn.commit()

    def _on_pnl(self, evs):  # PNL_UPDATE
        cur = self.conn.cursor()
        rows = []
        for ev in self._iter_events(evs):
            p = ev.payload or {}
            rows.append((
                float(p.get("ts", time.time())),
                str(p.get("symbol", "")),
                float(p.get("cash", 0.0)),
                float(p.get("position_qty", 0.0)),
                float(p.get("avg_price", 0.0)),
                float(p.get("last_price", 0.0)),
                float(p.get("realized_pnl", 0.0)),
                float(p.get("unrealized_pnl", 0.0)),
                float(p.get("equity", 0.0)),
            ))
        if rows:
            cur.executemany("INSERT INTO pnl VALUES (?,?,?,?,?,?,?,?,?)", rows)
            self.conn.commit()

    def _on_wfo(self, evs):  # WFO_STATUS
        cur = self.conn.cursor()
        rows = []
        for ev in self._iter_events(evs):
            p = ev.payload or {}
            payload_json = json.dumps(p, ensure_ascii=False)
            rows.append((
                float(p.get("ts", time.time())),
                str(p.get("symbol", "")),
                str(p.get("phase", "")),
                payload_json,
            ))
        if rows:
            cur.executemany("INSERT INTO wfo_status VALUES (?,?,?,?)", rows)
            self.conn.commit()

    def _on_autotrade(self, evs):  # AUTOTRADE_STATUS
        cur = self.conn.cursor()
        rows = []
        for ev in self._iter_events(evs):
            p = ev.payload or {}
            payload_json = json.dumps(p, ensure_ascii=False)
            rows.append((
                float(p.get("ts", time.time())),
                str(p.get("component", "")),
                str(p.get("action", "")),
                str(p.get("symbol", p.get("sym", ""))),
                payload_json,
            ))
        if rows:
            cur.executemany("INSERT INTO autotrade_status VALUES (?,?,?,?,?)", rows)
            self.conn.commit()
