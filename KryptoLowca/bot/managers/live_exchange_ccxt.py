# managers/live_exchange_ccxt.py
# -*- coding: utf-8 -*-
"""
LiveExchangeCCXT – backend „live” dla ExchangeAdapter, oparty o CCXT (spot + testnet).

Funkcje:
- Połączenie do giełdy CCXT (domyślnie Binance).
- Testnet via set_sandbox_mode(True) – dla Binance Spot CCXT przełącza na testnet.
- Odczyt ceny (fetch_ticker) – działa bez kluczy API.
- Składanie/anulowanie zleceń MARKET/LIMIT (wymagane klucze API).
- Walidacja i ZAOKRĄGLANIE **W GÓRĘ** ilości/ceny do kroku (stepSize/tickSize) oraz egzekwowanie minQty/minNotional.
- Zapis do DB (mode='live'): orders/trades/positions.

Uwaga:
- Backend zoptymalizowany pod Binance (spot). STOP/STOP_LIMIT dołożymy w kolejnej iteracji.
"""
from __future__ import annotations

import math
import datetime as dt
from typing import Dict, List, Optional, Tuple

import ccxt

from managers.database_manager import DatabaseManager


class LiveExchangeCCXT:
    def __init__(
        self,
        db: DatabaseManager,
        *,
        symbol: str = "BTC/USDT",
        exchange_id: str = "binance",
        apiKey: Optional[str] = None,
        secret: Optional[str] = None,
        password: Optional[str] = None,
        options: Optional[Dict] = None,
        enable_rate_limit: bool = True,
        testnet: bool = False,
    ) -> None:
        self.db = db
        self.symbol = symbol
        self.mode = "live"

        exchange_class = getattr(ccxt, exchange_id)
        self.client = exchange_class({
            "apiKey": apiKey or "",
            "secret": secret or "",
            "password": password or "",
            "enableRateLimit": enable_rate_limit,
            "options": options or {},
        })

        # Sandbox (Binance Spot Testnet)
        if hasattr(self.client, "set_sandbox_mode"):
            try:
                self.client.set_sandbox_mode(bool(testnet))
            except Exception:
                pass

        # Wymuś SPOT (ważne na Binance, żeby nie próbował futuresów)
        try:
            self.client.options = self.client.options or {}
            self.client.options["defaultType"] = "spot"
        except Exception:
            pass

        self._has_keys = bool(apiKey and secret)
        self._last_price: Optional[float] = None
        self._last_ts: Optional[dt.datetime] = None

        # Rynki i ograniczenia
        try:
            self.markets = self.client.load_markets()
        except Exception:
            self.markets = {}
        self.market = self.markets.get(self.symbol, {}) if isinstance(self.markets, dict) else {}

        # precyzje i limity
        self.price_precision: Optional[int] = None
        self.amount_precision: Optional[int] = None
        self.min_qty: Optional[float] = None
        self.min_notional: Optional[float] = None
        self.amount_step: Optional[float] = None   # krok ilości (np. 0.000001)
        self.tick_size: Optional[float] = None     # krok ceny

        self._extract_symbol_constraints()

        # Fallback dla Binance: jeśli giełda nie oddała minNotional – przyjmij 10 USDT (spot/testnet)
        if self.client.id == "binance" and (self.min_notional is None or self.min_notional <= 0):
            self.min_notional = 10.0

    # --------- Pomocnicze: precyzje i limity ----------
    def _extract_symbol_constraints(self) -> None:
        m = self.market or {}

        # precyzje
        prec = m.get("precision") or {}
        self.price_precision = prec.get("price")
        self.amount_precision = prec.get("amount")

        # limity
        limits = m.get("limits") or {}
        amount_lims = limits.get("amount") or {}
        cost_lims = limits.get("cost") or {}
        self.min_qty = amount_lims.get("min", None)
        self.min_notional = cost_lims.get("min", None)

        # dla Binance spróbuj z info.filters
        info = m.get("info") or {}
        filters = info.get("filters") or []
        tick_size = None
        step_size = None
        min_notional_f = None
        for f in filters:
            ftype = f.get("filterType")
            if ftype == "PRICE_FILTER":
                tick_size = self._to_float_safe(f.get("tickSize"))
            elif ftype == "LOT_SIZE":
                step_size = self._to_float_safe(f.get("stepSize"))
                if self.min_qty is None:
                    self.min_qty = self._to_float_safe(f.get("minQty"))
            elif ftype in ("MIN_NOTIONAL", "NOTIONAL"):
                v = self._to_float_safe(f.get("minNotional") or f.get("notional"))
                if v is not None:
                    min_notional_f = v

        if self.min_notional is None and min_notional_f is not None:
            self.min_notional = min_notional_f

        # krok ilości/ceny
        if step_size is not None and step_size > 0:
            self.amount_step = step_size
        elif self.amount_precision is not None:
            self.amount_step = 10 ** (-self.amount_precision)

        if tick_size is not None and tick_size > 0:
            self.tick_size = tick_size
        elif self.price_precision is not None:
            self.tick_size = 10 ** (-self.price_precision)

        # jeśli brak precyzji, wylicz z kroku
        if self.price_precision is None and self.tick_size is not None:
            self.price_precision = self._decimals_from_step(self.tick_size)
        if self.amount_precision is None and self.amount_step is not None:
            self.amount_precision = self._decimals_from_step(self.amount_step)

    @staticmethod
    def _to_float_safe(v) -> Optional[float]:
        try:
            return float(v) if v is not None else None
        except Exception:
            return None

    @staticmethod
    def _decimals_from_step(step: float | str | None) -> Optional[int]:
        if step is None:
            return None
        try:
            s = str(step)
            if "1e-" in s:
                return int(s.split("e-")[1])
            if "." in s:
                return len(s.split(".")[1].rstrip("0"))
            return 0
        except Exception:
            return None

    # --- zaokrąglanie do kroku (floor i ceil) ---
    def _floor_to_step(self, value: float, step: Optional[float]) -> float:
        if step is None or step <= 0:
            return value
        return math.floor(value / step + 1e-12) * step

    def _ceil_to_step(self, value: float, step: Optional[float]) -> float:
        if step is None or step <= 0:
            return value
        return math.ceil(value / step - 1e-12) * step

    def _round_amount_floor(self, amount: float) -> float:
        return self._floor_to_step(amount, self.amount_step)

    def _round_amount_ceil(self, amount: float) -> float:
        return self._ceil_to_step(amount, self.amount_step)

    def _round_price_floor(self, price: Optional[float]) -> Optional[float]:
        if price is None:
            return None
        return self._floor_to_step(price, self.tick_size)

    # Egzekwowanie minimum (z zaokrągleniem W GÓRĘ, jeśli trzeba podnieść)
    def _enforce_minimums(self, side: str, amount: float, price: Optional[float]) -> Tuple[float, Optional[float]]:
        # najpierw „porządkujemy” wejście do siatki kroków (floor)
        amt = max(0.0, self._round_amount_floor(amount))
        prc = self._round_price_floor(price)

        # minQty – jeśli za mało, PODNIEŚ do minQty i zrób ceil do kroku
        if self.min_qty is not None and amt < self.min_qty - 1e-12:
            amt = self._round_amount_ceil(self.min_qty)

        # minNotional – jeśli znamy cenę referencyjną
        ref_price = prc if prc is not None else self._last_price
        if self.min_notional is not None and ref_price is not None and ref_price > 0:
            notional = amt * ref_price
            if notional < self.min_notional - 1e-9:
                target_amt = self.min_notional / ref_price
                amt = self._round_amount_ceil(target_amt)
                # jeszcze raz upewnij się, że >= minQty
                if self.min_qty is not None and amt < self.min_qty - 1e-12:
                    amt = self._round_amount_ceil(self.min_qty)

        return amt, prc

    # --------- API zgodne z adapterem ----------
    def process_tick(self, price: Optional[float] = None, ts: Optional[dt.datetime] = None) -> None:
        if price is None:
            try:
                ticker = self.client.fetch_ticker(self.symbol)
                price = float(ticker.get("last") or ticker.get("close") or ticker.get("bid") or ticker.get("ask"))
            except Exception:
                return
        now = ts or dt.datetime.utcnow()
        self._last_price = float(price)
        self._last_ts = now

    def create_order(
        self,
        *,
        side: str,
        type: str,
        quantity: float,
        price: Optional[float] = None,
        client_order_id: Optional[str] = None,
        extra: Optional[Dict] = None,
    ) -> int:
        if not self._has_keys:
            raise RuntimeError("Brak kluczy API – zlecenia live są zablokowane. Ustaw apiKey/secret.")

        side = side.upper()
        type = type.upper()
        if type not in {"MARKET", "LIMIT"}:
            raise NotImplementedError("Na tym etapie obsługujemy tylko MARKET i LIMIT w trybie live.")

        # 1) DB: NEW -> OPEN
        oid = self.db.sync.record_order({
            "symbol": self.symbol,
            "side": side,
            "type": type,
            "quantity": float(quantity),
            "price": float(price) if price is not None else None,
            "status": "NEW",
            "client_order_id": client_order_id,
            "mode": self.mode,
            "extra": extra or {},
        })
        self.db.sync.update_order_status(order_id=oid, status="OPEN")

        # 2) Precyzje/limity z twardym egzekwowaniem (ceil do kroku gdy trzeba podnieść)
        qty, prc = self._enforce_minimums(side, float(quantity), float(price) if price is not None else None)
        if type == "LIMIT" and prc is None:
            raise ValueError("Limit price jest wymagany dla LIMIT.")

        # 3) Parametry giełdy
        params: Dict = {}
        if self.client.id == "binance" and client_order_id:
            params["newClientOrderId"] = client_order_id

        # 4) Wyślij zlecenie
        try:
            if type == "MARKET":
                ccxt_order = self.client.create_order(self.symbol, "market", side.lower(), qty, None, params)
            else:
                ccxt_order = self.client.create_order(self.symbol, "limit", side.lower(), qty, prc, params)
        except Exception as e:
            self.db.sync.update_order_status(order_id=oid, status="CANCELED", extra={"error": str(e)})
            raise

        # 5) Aktualizacja DB wg odpowiedzi giełdy
        exchange_order_id = ccxt_order.get("id")
        status = (ccxt_order.get("status") or "open").upper()
        last_avg_price = ccxt_order.get("average") or ccxt_order.get("price")
        filled = float(ccxt_order.get("filled") or 0.0)

        self.db.sync.update_order_status(
            order_id=oid,
            status="FILLED" if status in ("CLOSED", "FILLED") or (filled >= float(qty) - 1e-12) else status,
            price=float(last_avg_price) if last_avg_price else None,
            exchange_order_id=str(exchange_order_id) if exchange_order_id else None,
        )

        # 6) Zapis trade (jeśli cokolwiek się wypełniło)
        if filled > 0.0 and last_avg_price:
            fee_cost = 0.0
            if "fees" in ccxt_order and isinstance(ccxt_order["fees"], list) and ccxt_order["fees"]:
                fee_cost = sum(float(f.get("cost") or 0.0) for f in ccxt_order["fees"])
            elif "fee" in ccxt_order and ccxt_order["fee"]:
                fee_cost = float(ccxt_order["fee"].get("cost") or 0.0)

            self.db.sync.record_trade({
                "symbol": self.symbol,
                "side": side,
                "quantity": filled,
                "price": float(last_avg_price),
                "fee": float(fee_cost),
                "order_id": oid,
                "mode": self.mode
            })

            # Minimalny update pozycji po naszej stronie (jak w paper)
            pos = self.db.sync.get_open_positions(mode="live")
            current = next((p for p in pos if p["symbol"] == self.symbol), None)
            if side == "BUY":
                if current and current["side"] == "LONG":
                    new_qty = current["quantity"] + filled
                    new_avg = (current["avg_price"] * current["quantity"] + filled * float(last_avg_price)) / new_qty
                    self.db.sync.upsert_position({
                        "symbol": self.symbol, "side": "LONG", "quantity": new_qty, "avg_price": new_avg,
                        "unrealized_pnl": 0.0, "mode": self.mode
                    })
                elif current and current["side"] == "SHORT":
                    remain = current["quantity"] - filled
                    if remain > 1e-12:
                        self.db.sync.upsert_position({
                            "symbol": self.symbol, "side": "SHORT", "quantity": remain, "avg_price": current["avg_price"],
                            "unrealized_pnl": 0.0, "mode": self.mode
                        })
                    else:
                        self.db.sync.close_position(self.symbol)
                else:
                    self.db.sync.upsert_position({
                        "symbol": self.symbol, "side": "LONG", "quantity": filled, "avg_price": float(last_avg_price),
                        "unrealized_pnl": 0.0, "mode": self.mode
                    })
            else:  # SELL
                if current and current["side"] == "LONG":
                    remain = current["quantity"] - filled
                    if remain > 1e-12:
                        self.db.sync.upsert_position({
                            "symbol": self.symbol, "side": "LONG", "quantity": remain, "avg_price": current["avg_price"],
                            "unrealized_pnl": 0.0, "mode": self.mode
                        })
                    else:
                        self.db.sync.close_position(self.symbol)
                elif current and current["side"] == "SHORT":
                    new_qty = current["quantity"] + filled
                    new_avg = (current["avg_price"] * current["quantity"] + filled * float(last_avg_price)) / new_qty
                    self.db.sync.upsert_position({
                        "symbol": self.symbol, "side": "SHORT", "quantity": new_qty, "avg_price": new_avg,
                        "unrealized_pnl": 0.0, "mode": self.mode
                    })
                else:
                    self.db.sync.upsert_position({
                        "symbol": self.symbol, "side": "SHORT", "quantity": filled, "avg_price": float(last_avg_price),
                        "unrealized_pnl": 0.0, "mode": self.mode
                    })

        return oid

    def cancel_order(self, *, order_id: Optional[int] = None, client_order_id: Optional[str] = None) -> bool:
        if not self._has_keys:
            raise RuntimeError("Brak kluczy API – anulowanie live zleceń zablokowane.")
        try:
            open_orders = self.client.fetch_open_orders(self.symbol)
        except Exception:
            open_orders = []
        target = None
        if client_order_id:
            for o in open_orders:
                client_id = o.get("clientOrderId") or o.get("clientOrderID") or (o.get("info", {}) or {}).get("clientOrderId")
                if client_id == client_order_id:
                    target = o
                    break
        if target is None and open_orders:
            target = open_orders[0]
        if target is None:
            return False
        ex_id = target.get("id")
        try:
            self.client.cancel_order(ex_id, self.symbol)
            if client_order_id:
                self.db.sync.update_order_status(client_order_id=client_order_id, status="CANCELED")
            return True
        except Exception:
            return False

    def get_open_orders(self) -> List[Dict[str, object]]:
        try:
            data = self.client.fetch_open_orders(self.symbol)
        except Exception:
            data = []
        out: List[Dict[str, object]] = []
        for o in data:
            out.append({
                "id": o.get("id"),
                "client_order_id": o.get("clientOrderId") or o.get("clientOrderID") or (o.get("info", {}) or {}).get("clientOrderId"),
                "side": (o.get("side") or "").upper(),
                "type": (o.get("type") or "").upper(),
                "quantity": float(o.get("amount") or 0.0),
                "remaining": float(o.get("remaining") or 0.0),
                "price": float(o.get("price") or 0.0),
                "status": (o.get("status") or "open").upper(),
                "created_at": o.get("datetime") or "",
            })
        return out

    def get_position(self) -> Dict[str, object]:
        pos = self.db.sync.get_open_positions(mode="live")
        current = next((p for p in pos if p["symbol"] == self.symbol), None)
        if not current:
            return {"symbol": self.symbol, "side": "FLAT", "quantity": 0.0, "avg_price": 0.0, "unrealized_pnl": 0.0}
        last_price = self._last_price
        if last_price is None:
            upnl = 0.0
        else:
            if current["side"] == "LONG":
                upnl = (last_price - current["avg_price"]) * current["quantity"]
            elif current["side"] == "SHORT":
                upnl = (current["avg_price"] - last_price) * current["quantity"]
            else:
                upnl = 0.0
        return {
            "symbol": self.symbol,
            "side": current["side"],
            "quantity": current["quantity"],
            "avg_price": current["avg_price"],
            "unrealized_pnl": upnl,
        }
