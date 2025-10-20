# run_trading_gui_paper_emitter.py
from __future__ import annotations

from pathlib import Path
import sys


def _ensure_repo_root() -> None:
    current_dir = Path(__file__).resolve().parent
    for candidate in (current_dir, *current_dir.parents):
        package_init = candidate / "KryptoLowca" / "__init__.py"
        if package_init.exists():
            candidate_str = str(candidate)
            if candidate_str not in sys.path:
                sys.path.insert(0, candidate_str)
            break


if __package__ in (None, ""):
    _ensure_repo_root()


import logging
import os
import threading
import time
from typing import Any, List, Optional

# Logging – czytelne, po PL
LOG_LEVEL = os.getenv("APP_LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("runner")

# Event adapter
from KryptoLowca.event_emitter_adapter import (
    EventBus,
    DebounceRule,
    EmitterAdapter,
    EmitterConfig,
    Event,
    EventType,
)

# Import GUI (pakiet modułowy udostępnia klasę TradingGUI)
try:
    from KryptoLowca.ui.trading import TradingGUI
    from KryptoLowca.ui.trading.risk_helpers import apply_runtime_risk_context
except Exception:
    logger.exception(
        "Nie udało się zaimportować TradingGUI z KryptoLowca.ui.trading"
    )
    raise


DEFAULT_PAPER_ORDER_NOTIONAL = 25.0


# ==========================================
# Opcjonalna „demonstracja” feedu (bez danych)
# Możesz wyłączyć ustawiając EMITTER_DEMO=0
# ==========================================

class DemoFeeder(threading.Thread):
    def __init__(self, adapter: EmitterAdapter, symbol: str = "BTCUSDT", delay: float = 0.5) -> None:
        super().__init__(name="DemoFeeder", daemon=True)
        self.adapter = adapter
        self.symbol = symbol
        self.delay = delay
        self._stop = threading.Event()
        self._price = 50000.0

    def run(self) -> None:
        high = self._price
        low = self._price
        close = self._price
        k = 0
        while not self._stop.is_set():
            k += 1
            # proste „wygibasy” żeby ATR zadziałał
            step = (1 + (k % 7) / 100.0)
            self._price = max(10.0, self._price * step / (1.0 + (k % 13) / 100.0))
            jitter = ((k % 5) - 2) * 5.0
            high = self._price + abs(jitter)
            low = max(1.0, self._price - abs(jitter) * 1.1)
            close = (high + low) / 2.0

            self.adapter.push_market_tick(self.symbol, price=self._price, high=high, low=low, close=close)
            if k % 11 == 0:
                # pseudo-metryki do triggera PF/Expectancy
                pf = 1.2 + (k % 9) * 0.05
                exp = -0.0005 + (k % 7) * 0.0002
                trades = 25 + (k % 10)
                self.adapter.update_metrics(self.symbol, pf=pf, expectancy=exp, trades=trades)

            if k % 17 == 0:
                self.adapter.push_order_status(oid=f"D-{k}", status="filled", symbol=self.symbol, filled_qty=0.01)

            if k % 23 == 0:
                self.adapter.push_signal(self.symbol, side="buy" if (k % 2 == 0) else "sell", strength=0.8)

            time.sleep(self.delay)

    def stop(self) -> None:
        self._stop.set()


# ==========================================
# Helper: integracja z GUI (bez twardych zależności)
# ==========================================


def _configure_runtime_risk(
    gui: TradingGUI,
    *,
    entrypoint: str = "trading_gui",
    config_path: Optional[str] = None,
) -> None:
    """Uzupełnia GUI o dane profilu ryzyka oraz loguje domyślne wartości."""

    apply_runtime_risk_context(
        gui,
        entrypoint=entrypoint,
        config_path=config_path,
        default_notional=DEFAULT_PAPER_ORDER_NOTIONAL,
        logger=logger,
    )

def _wire_gui_with_bus(root, gui: Any, bus: EventBus) -> None:
    """
    Delikatne podpięcie EventBus do GUI.
    Jeśli GUI ma metodę 'handle_events(events: List[Event])' – użyjemy jej w trybie batch.
    Jeśli ma 'handle_event(event: Event)' – wywołujemy pojedynczo.
    Jeśli nie ma żadnej – ignorujemy (GUI i tak się uruchomi).
    """
    has_batch = hasattr(gui, "handle_events") and callable(getattr(gui, "handle_events"))
    has_single = hasattr(gui, "handle_event") and callable(getattr(gui, "handle_event"))

    if not has_batch and not has_single:
        logger.info("GUI nie udostępnia handle_event(s) – EventBus działa, ale GUI nie reaguje (to OK na tym etapie).")
        return

    def on_batch(ev_list: List[Event]) -> None:
        try:
            if has_batch:
                gui.handle_events(ev_list)  # preferuj batch
            elif has_single:
                for ev in ev_list:
                    gui.handle_event(ev)
        except Exception:
            logger.exception("Wyjątek w handlerze GUI dla paczki zdarzeń.")

        # tickle Tk, by odświeżyć UI z wątku busa
        try:
            root.event_generate("<<EventBusTick>>", when="tail")
        except Exception:
            pass

    # subskrybuj kluczowe kanały – paczkami (debounce-window 150ms)
    rule = DebounceRule(window_sec=0.15, max_batch=200, deliver_list=True)
    for et in (EventType.MARKET_TICK, EventType.ORDER_STATUS, EventType.SIGNAL,
               EventType.WFO_TRIGGER, EventType.WFO_STATUS, EventType.AUTOTRADE_STATUS, EventType.RISK_ALERT, EventType.LOG):
        bus.subscribe(et, on_batch, rule=rule)

    logger.info("GUI podłączone do EventBus (batch=%.2fs).", rule.window_sec)


# ==========================================
# Main
# ==========================================

def main() -> None:
    # 1) Tk
    try:
        import tkinter as tk
    except Exception:
        logger.exception("Brak tkinter – to środowisko nie obsługuje GUI.")
        raise

    root = tk.Tk()
    root.title("KryptoŁowca — Paper (Event Emitter)")
    # 2) EventBus + Adapter
    bus = EventBus()
    bus.start()
    adapter = EmitterAdapter(
        bus,
        cfg=EmitterConfig(
            atr_period=14,
            atr_trigger_growth_pct=30.0,  # demo: dość czułe
            pf_min=1.4,
            expectancy_min=0.0,
            min_trades_for_pf=20,
        ),
    )

    # 3) Utworzenie GUI (zgodnie z tym, co wcześniej wyskoczyło – GUI wymaga parametru 'root')
    try:
        gui = TradingGUI(root)
    except TypeError:
        # Spróbuj wariantu z event_bus (jeśli GUI ma taki konstruktor)
        try:
            gui = TradingGUI(root, event_bus=bus)
        except TypeError:
            # Ostatecznie – znów klasyczny
            gui = TradingGUI(root)

    _configure_runtime_risk(gui)

    # 4) Miękkie spięcie EventBus -> GUI (jeśli GUI ma odpowiednie metody)
    _wire_gui_with_bus(root, gui, bus)

    # 5) (Opcjonalnie) odpal demo feeder
    demo_on = os.getenv("EMITTER_DEMO", "1") not in ("0", "false", "False", "no", "NO")
    feeder = None
    if demo_on:
        feeder = DemoFeeder(adapter, symbol=os.getenv("EMITTER_DEMO_SYMBOL", "BTCUSDT"), delay=0.4)
        feeder.start()
        logger.info("Demo feeder: ON (EMITTER_DEMO=1).")

    # 6) Sprzątanie przy zamknięciu
    def on_close():
        try:
            if feeder is not None:
                feeder.stop()
        except Exception:
            pass
        try:
            bus.stop()
        except Exception:
            pass
        try:
            root.destroy()
        except Exception:
            pass

    root.protocol("WM_DELETE_WINDOW", on_close)

    # 7) Run
    try:
        root.mainloop()
    finally:
        on_close()


if __name__ == "__main__":
    try:
        main()
    except Exception:
        logger.exception("Błąd podczas uruchamiania aplikacji.")
        sys.exit(1)
