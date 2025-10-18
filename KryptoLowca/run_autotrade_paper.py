# run_autotrade_paper.py
# -*- coding: utf-8 -*-
from __future__ import annotations

from pathlib import Path
import sys
import logging
import signal
import threading
import time
from typing import Optional


def _ensure_repo_root() -> None:
    """Dopisuje katalog repo do sys.path tak, by import 'KryptoLowca' działał przy uruchamianiu skryptu bez instalacji."""
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
from KryptoLowca.paper_auto_trade_app import PaperAutoTradeApp
from KryptoLowca.risk_settings_loader import DEFAULT_CORE_CONFIG_PATH

SYMBOL = "BTCUSDT"


def _start_gui_in_main_thread(
    adapter: EmitterAdapter,
    enable_gui: bool = True,
    *,
    paper_app: Optional[PaperAutoTradeApp] = None,
) -> None:
    if not enable_gui:
        log.info("GUI disabled by flag.")
        return
    try:
        import tkinter as tk  # noqa: F401
        try:
            import KryptoLowca.trading_gui as trading_gui
        except Exception as e:
            log.info("GUI: nie udało się załadować trading_gui (%s). Uruchamiam bez GUI.", e)
            return
        root = tk.Tk()
        try:
            gui = trading_gui.TradingGUI(
                root,
                event_bus=adapter.bus,
                core_config_path=getattr(paper_app, "core_config_path", None),
                core_environment=getattr(paper_app, "core_environment", None),
            )
            log.info("Załadowano GUI: trading_gui.TradingGUI(root)")
        except TypeError:
            gui = trading_gui.TradingGUI()
            log.info("Załadowano GUI: trading_gui.TradingGUI")
        except Exception as e:
            log.exception("GUI mainloop error podczas konstrukcji: %s", e)
            return
        if paper_app is not None:
            paper_app.gui = gui
            try:
                paper_app.reload_risk_settings()
            except Exception as exc:
                log.warning("Nie udało się przeładować limitów ryzyka w GUI: %s", exc)
        wire_gui_logs_to_adapter(adapter)
        log.info("GUI start (main thread).")
        root.mainloop()
        log.info("GUI zamknięte.")
    except Exception as e:
        log.info("GUI niedostępne lub błąd uruchomienia: %s", e)


def main(
    use_dummy_feed: bool = True,
    enable_gui: bool = True,
    *,
    core_config_path: str | None = None,
    core_environment: str | None = None,
) -> None:
    adapter = EmitterAdapter()
    bus = adapter.bus
    wire_gui_logs_to_adapter(adapter)

    resolved_core_path = (
        Path(core_config_path).expanduser().resolve()
        if core_config_path
        else Path(DEFAULT_CORE_CONFIG_PATH)
    )
    paper_app = PaperAutoTradeApp(
        core_config_path=resolved_core_path,
        core_environment=core_environment,
    )

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
    position_sizer = PositionSizer(bus, PositionSizerConfig(
        symbol=SYMBOL, risk_per_trade_pct=0.5, min_qty=0.005, max_qty=0.05, sl_atr_mult=2.0, tp_atr_mult=3.0, atr_tf="60s"
    ))
    stop_tp = StopTPService(bus, StopTPConfig(
        symbol=SYMBOL, default_sl_atr_mult=2.0, default_tp_atr_mult=3.0, cooldown_after_exit_sec=5.0
    ))

    def _apply_limits(settings, profile_name, _profile_cfg) -> None:
        if not settings:
            log.warning("Brak ustawień ryzyka do zastosowania (profil=%s)", profile_name)
            return

        def _as_pct(value: float | int | None) -> Optional[float]:
            if value is None:
                return None
            pct = float(value)
            return pct * 100.0 if pct <= 1.0 else pct

        daily_loss = _as_pct(settings.get("max_daily_loss_pct"))
        drawdown = _as_pct(settings.get("max_drawdown_pct"))
        if daily_loss is not None:
            risk_guard.cfg.max_daily_loss_pct = daily_loss
        if drawdown is not None:
            risk_guard.cfg.max_drawdown_pct = drawdown

        risk_per_trade = _as_pct(settings.get("max_risk_per_trade"))
        if risk_per_trade is not None:
            position_sizer.cfg.risk_per_trade_pct = risk_per_trade

        sl_mult = settings.get("stop_loss_atr_multiple")
        if sl_mult is not None:
            value = float(sl_mult)
            position_sizer.cfg.sl_atr_mult = value
            stop_tp.cfg.default_sl_atr_mult = value
            stop_tp._sl_mult = value

        portfolio_risk = settings.get("max_portfolio_risk")
        if portfolio_risk is not None:
            strat.cfg.max_abs_position = float(portfolio_risk)

        try:
            position_sizer._maybe_publish_update()
        except Exception as exc:
            log.debug("PositionSizer update skipped: %s", exc)

        log.info(
            "Zastosowano limity ryzyka: profil=%s, dzienny=%.4f%%, DD=%.4f%%, risk/trade=%.4f%%",
            profile_name,
            daily_loss if daily_loss is not None else -1.0,
            drawdown if drawdown is not None else -1.0,
            risk_per_trade if risk_per_trade is not None else -1.0,
        )

    paper_app.add_listener(_apply_limits)

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

    def _cli_worker():
        while not stop_event.is_set():
            try:
                line = input()
            except EOFError:
                break
            if not line:
                continue
            handled = paper_app.handle_cli_command(line)
            if handled:
                log.info("Przeładowano limity ryzyka komendą CLI: %s", line.strip())
            else:
                log.info("Nieznana komenda CLI: %s", line.strip())

    t_cli = threading.Thread(target=_cli_worker, name="cli-listener", daemon=True)
    t_cli.start()

    # --- GUI w main thread -----------------------------------------------------------------------
    try:
        paper_app.reload_risk_settings()
    except Exception as exc:
        log.warning("Wstępne wczytanie limitów ryzyka nie powiodło się: %s", exc)

    try:
        paper_app.start_auto_reload()
    except Exception as exc:
        log.warning("Auto-reload core.yaml nie zostanie uruchomiony: %s", exc)

    _start_gui_in_main_thread(adapter, enable_gui=enable_gui, paper_app=paper_app)

    # --- graceful shutdown -----------------------------------------------------------------------
    def _sigint(sig, frame):
        log.info("Ctrl+C received. Shutting down...")
        stop_event.set()
    signal.signal(signal.SIGINT, _sigint)

    try:
        def _sighup_handler(sig, frame):
            log.info("Odebrano SIGHUP — przeładowuję core.yaml")
            try:
                paper_app.reload_risk_settings()
            except Exception as exc:
                log.warning("SIGHUP reload failed: %s", exc)

        signal.signal(signal.SIGHUP, _sighup_handler)
    except AttributeError:
        pass

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
    try:
        t_cli.join(timeout=1.0)
    except Exception:
        pass

    try:
        paper_app.stop_auto_reload()
    except Exception:
        pass

    log.info("Zamykanie zakończone.")


if __name__ == "__main__":
    use_dummy = True
    enable_gui = True
    core_arg: str | None = None
    env_arg: str | None = None
    # Flagi: python run_autotrade_paper.py nogui | real | core=... | env=...
    for arg in sys.argv[1:]:
        a = arg.lower()
        if a == "nogui":
            enable_gui = False
        if a.startswith("real"):
            use_dummy = False
        if a.startswith("core="):
            core_arg = arg.split("=", 1)[1] or None
        if a.startswith("env="):
            env_arg = arg.split("=", 1)[1] or None
    main(
        use_dummy_feed=use_dummy,
        enable_gui=enable_gui,
        core_config_path=core_arg,
        core_environment=env_arg,
    )
