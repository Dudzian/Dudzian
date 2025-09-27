# trading/paper_broker.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from KryptoLowca.event_emitter_adapter import EmitterAdapter, EventType


@dataclass
class Position:
    side: Optional[str] = None   # "long" / "short" / None
    qty: float = 0.0
    entry_price: float = 0.0


class PaperBroker:
    """
    Bardzo prosty broker paper trading:
    - submit_market("buy"/"sell", qty)
    - update_mark(last_price)  -> mark-to-market + emituje ORDER_STATUS partial/filled i TRADE_EXECUTED przy zamknięciu.
    Strategia: pojedyncza pozycja odwracalna (flip).
    """
    def __init__(self, adapter: EmitterAdapter, symbol: str) -> None:
        self.adapter = adapter
        self.symbol = symbol
        self.pos = Position()
        self.last_price: float = 0.0

    # --- API ---

    def submit_market(self, side: str, qty: float) -> None:
        side = side.lower()
        # BUY -> long, SELL -> short (odwracamy jeśli trzeba)
        if side == "buy":
            self._open_or_flip("long", qty)
        elif side == "sell":
            self._open_or_flip("short", qty)
        else:
            self._push_order("rejected", reason=f"unknown side: {side}")

    def update_mark(self, px: float) -> None:
        self.last_price = float(px)
        # mark-to-market: tu można rozbudować o unrealized PnL publish
        self._push_order("mark", last=px)

    # --- Core ---

    def _open_or_flip(self, target_side: str, qty: float) -> None:
        qty = float(qty)
        # jeśli mamy pozycję przeciwną — najpierw zamknij
        closed_pnl = None
        if self.pos.side and self.pos.side != target_side:
            closed_pnl = self._close_position(self.last_price)

        # jeśli nie mamy pozycji albo była przeciwna — otwórz nową
        if self.pos.side != target_side:
            self.pos.side = target_side
            self.pos.qty = qty
            self.pos.entry_price = self.last_price

        self._push_order("filled", qty=qty, side=target_side, pnl=closed_pnl)

    def _close_position(self, px: float) -> float:
        if not self.pos.side or self.pos.qty <= 0:
            return 0.0
        pnl = 0.0
        if self.pos.side == "long":
            pnl = (px - self.pos.entry_price) * self.pos.qty
        elif self.pos.side == "short":
            pnl = (self.pos.entry_price - px) * self.pos.qty
        # Emit TRADE_EXECUTED
        self.adapter.publish(EventType.TRADE_EXECUTED, {
            "symbol": self.symbol,
            "side": self.pos.side,
            "qty": self.pos.qty,
            "entry": self.pos.entry_price,
            "exit": px,
            "pnl": pnl,
        })
        # reset
        self.pos = Position()
        return pnl

    # --- Events ---

    def _push_order(self, status: str, **extra) -> None:
        payload = {"symbol": self.symbol, "status": status, **extra}
        self.adapter.publish(EventType.ORDER_STATUS, payload)
