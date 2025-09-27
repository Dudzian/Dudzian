# run_autotrade_paper.py
# -*- coding: utf-8 -*-
from __future__ import annotations

import logging
import signal
import sys
import threading
import time
from pathlib import Path

# — wyszukiwanie katalogu projektu, aby importy KryptoLowca działały przy uruchamianiu jako skrypt —
if __package__ in {None, ""}:
    _current_file = Path(__file__).resolve()
    for _parent in _current_file.parents:
        candidate = _parent / "KryptoLowca" / "__init__.py"
        if candidate.exists():
            sys.path.insert(0, str(_parent))
            break
    else:  # pragma: no cover
        raise ModuleNotFoundError(
            "Nie można zlokalizować pakietu 'KryptoLowca'. Uruchom skrypt z katalogu projektu lub "
            "zainstaluj pakiet w środowisku (pip install -e .)."
        )

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
log = logging.getLogger("runner")

try:
    from KryptoLowca.event_emitter_adapter import (
        Event, EventType,
        EventBus, EmitterAdapter, EventEmitterAdapter,
        DummyMarketFeed, DummyMarketFeedConfig,
        wire_gui_logs_to_adapter,
    )
except Exception as e:
    log.error("Nie udało się zaimportować event_emitter_adapter: %s", e, exc_info=True)
    raise

from KryptoLowca.services.marketdata import MarketDataConfig, MarketDataService
from KryptoLowca.services.order_router import PaperBroker, PaperBrokerConfig
from KryptoLowca.services.performance_monitor import PerfMonitorConfig, PerformanceMonitor
from KryptoLowca.services.persistence import PersistenceService
from KryptoLowca.services.position_sizer import PositionSizer, PositionSizerConfig
from KryptoLowca.services.risk_guard import RiskGuard, RiskGuardConfig
from KryptoLowca.services.risk_manager import RiskConfig, RiskManager
from KryptoLowca.services.stop_tp import StopTPConfig, StopTPService
from KryptoLowca.services.strategy_engine import StrategyConfig, StrategyEngine
from KryptoLowca.services.walkforward_service import (
    ObjectiveWeights,
    WFOServiceConfig,
    WalkForwardService,
)

SYMBOL = "BTCUSDT"


def _start_gui_in_main_thread(adapter: EmitterAdapter, enable_gui: bool = True):
    if not enable_gui:
        log.info("GUI disabled by flag.")
        return
    try:
        import tkinter as tk  # noqa
        try:
            # poprawiony import tak, aby moduł był dostępny jako 'trading_gui'
            from KryptoLowca import trading_gui as trading_gui  # noqa: F401
        except Exception as e:
            log.info("GUI: nie udało się załadować trading_gui (%s). Uruchamiam bez GUI.", e)
            return
        root = tk.Tk()
        try:
            gui = trading_gui.TradingGUI(root)
            log.info("Załadowano GUI: trading_gui.TradingGUI(root)")
        except TypeError:
            gui = trading_gui.TradingGUI()
            log.info("Załadowano GUI: trading_gui.TradingGUI")
        except Exception as e:
            log.exception("GUI mainloop error podczas konstrukcji: %s", e)
            return
        wire_gui_logs_to_adapter(adapter)
        log.info("GUI start (main thread).")
        root.mainloop()
        log.info("GUI zamknięte.")
    except Exception as e:
        log.info("GUI niedostępne lub błąd uruchomienia: %s", e)


def main(use_dummy_feed: bool = True, enable_gui: bool = True) -> None:
    adapter = EmitterAdapter()
    bus = adapter.bus
    wire_gui_logs_to_adapter(adapter)

    # --- persistence -----------------------------------------------------------------------------
    persistence = PersistenceService(bus, db_path="data/runtime.db")

    # --- services: market data / ATR -------------------------------------------------------------
    md = MarketDataService(bus, MarketDataConfig(
        symbol=SYMBOL, timeframe_sec=60, atr_len=14, publish_intermediate_bars=False
    ))

    # --- WFO & ryzyko ----------------------------------------------------------------------------
    wf = WalkForwardService(bus, WFOServiceConfig(
        symbol=SYMBOL, cooldown_sec=180.0, auto_apply=True, obj_weights=ObjectiveWeights(),
        min_is_bars=800, min_oos_bars=300, step_bars=100,
        fast_grid=(10, 15, 20, 25, 30), slow_grid=(40, 60, 80, 100, 120), qty_grid=(0.01, 0.02)
    ))

    risk = RiskManager(bus, RiskConfig(
        symbol=SYMBOL, atr_lookback=100, spike_threshold_pct=50.0, publish_every_n=10
    ))

    risk_guard = RiskGuard(bus, RiskGuardConfig(
        symbol=SYMBOL, max_daily_loss_pct=5.0, max_drawdown_pct=20.0, auto_resume_cooldown_sec=300, publish_every_n=5
    ))

    # --- broker / strategia ----------------------------------------------------------------------
    broker = PaperBroker(bus, PaperBrokerConfig(
        symbol=SYMBOL, initial_cash=10_000.0, fee_bps=2.0, slippage_bps=1.0, allow_short=True
    ))

    strat = StrategyEngine(bus, StrategyConfig(
        symbol=SYMBOL, enabled=True, qty=0.01, max_abs_position=0.05, fast_len=20, slow_len=60, order_cooldown_sec=10.0
    ))

    # dynamiczny sizing + SL/TP na bazie ATR
    sizer = PositionSizer(bus, PositionSizerConfig(
        symbol=SYMBOL, risk_per_trade_pct=0.5, min_qty=0.005, max_qty=0.05, sl_atr_mult=2.0, tp_atr_mult=3.0, atr_tf="60s"
    ))
    stop_tp = StopTPService(bus, StopTPConfig(
        symbol=SYMBOL, default_sl_atr_mult=2.0, default_tp_atr_mult=3.0, cooldown_after_exit_sec=5.0
    ))

    # monitor wyników (opcjonalny)
    try:
        perf = PerformanceMonitor(bus, PerfMonitorConfig(
            symbol=SYMBOL, window_trades=100, min_trades_to_eval=20, pf_min=1.1, exp_min=0.0, consecutive_breaches=3
        ))
    except Exception as e:
        log.info("PerformanceMonitor unavailable; skipping. (%s)", e)

    # --- feed ------------------------------------------------------------------------------------
    if use_dummy_feed:
        feed = DummyMarketFeed(adapter, symbol=SYMBOL, start_price=30_000.0, tick_interval_s=1.0)
    else:
        raise NotImplementedError("Realny feed nie jest jeszcze skonfigurowany.")

    stop_event = threading.Event()

    def _feed_worker():
        log.info("Dummy feed started.")
        try:
            feed.start().join()
        except Exception as e:
            log.exception("Feed worker error: %s", e)
        finally:
            log.info("Dummy feed stopped.")
            stop_event.set()

    t_feed = threading.Thread(target=_feed_worker, name="feed-worker", daemon=True)
    t_feed.start()

    # --- GUI w main thread -----------------------------------------------------------------------
    _start_gui_in_main_thread(adapter, enable_gui=enable_gui)

    # --- graceful shutdown -----------------------------------------------------------------------
    def _sigint(sig, frame):
        log.info("Ctrl+C received. Shutting down...")
        stop_event.set()
    signal.signal(signal.SIGINT, _sigint)

    while not stop_event.is_set():
        time.sleep(0.25)

    try:
        feed.stop()
    except Exception:
        pass
    try:
        t_feed.join(timeout=2.0)
    except Exception:
        pass

    log.info("Zamykanie zakończone.")


if __name__ == "__main__":
    use_dummy = True
    enable_gui = True
    # Flagi: python run_autotrade_paper.py nogui | real
    for arg in sys.argv[1:]:
        a = arg.lower()
        if a == "nogui":
            enable_gui = False
        if a.startswith("real"):
            use_dummy = False
    main(use_dummy_feed=use_dummy, enable_gui=enable_gui)
