# wfa_daemon.py
# -*- coding: utf-8 -*-
"""
Lekki daemon walk-forward optimization (WFO).

Aktualna wersja integruje się z odświeżonym modułem `WalkForwardService`, który
wymaga EventBus-a, dlatego całość jest bardziej zbliżona do architektury bota
niż poprzednia implementacja uruchamiana w izolacji.
"""

from __future__ import annotations

from pathlib import Path
import sys
import argparse
import csv
import json
import logging
import time
from dataclasses import asdict
from datetime import datetime
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

# Repo bootstrap
if __package__ in {None, ""}:
    _current_file = Path(__file__).resolve()
    for _parent in _current_file.parents:
        candidate = _parent / "KryptoLowca" / "__init__.py"
        if candidate.exists():
            sys.path.insert(0, str(_parent))
            __package__ = "KryptoLowca"
            break
    else:  # pragma: no cover - diagnostyka uruchomienia
        raise ModuleNotFoundError(
            "Nie można zlokalizować pakietu 'KryptoLowca'. Uruchom daemon z katalogu projektu lub"
            " zainstaluj pakiet w środowisku (pip install -e .)."
        )

from KryptoLowca.event_emitter_adapter import Event, EventBus, EventType
from KryptoLowca.services.walkforward_service import (
    ObjectiveWeights,
    WFOServiceConfig,
    WalkForwardService,
)

log = logging.getLogger("KryptoLowca.wfa_daemon")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")


def emit_stdout(event: str, payload: Dict[str, Any]) -> None:
    rec = {"ts": datetime.utcnow().isoformat(timespec="seconds"), "event": event, "payload": payload}
    sys.stdout.write(json.dumps(rec, ensure_ascii=False) + "\n")
    sys.stdout.flush()


def _iter_events(events: Optional[Iterable[Event] | Event]) -> List[Event]:
    if not events:
        return []
    if isinstance(events, list):
        return events
    return [events] if isinstance(events, Event) else list(events) if isinstance(events, Event) else list(events) if isinstance(events, Event) else list(events) if isinstance(events, Event) else list(events) if isinstance(events, Event) else list(events)


def load_config(path: Path) -> Dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    lines: List[str] = []
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("//"):
            continue
        lines.append(line)
    return json.loads("\n".join(lines))


def _as_tuple(seq: Any, caster) -> Sequence[Any]:
    if seq is None:
        return ()
    if isinstance(seq, (list, tuple, set)):
        return tuple(caster(x) for x in seq)
    return (caster(seq),)


def build_service_config(cfg_dict: Dict[str, Any]) -> Tuple[WFOServiceConfig, Dict[str, Any]]:
    defaults = WFOServiceConfig()
    obj_defaults = ObjectiveWeights()

    symbol = cfg_dict.get("symbol")
    if not symbol:
        symbols = cfg_dict.get("symbols")
        if isinstance(symbols, (list, tuple)) and symbols:
            symbol = symbols[0]
        else:
            symbol = defaults.symbol

    obj_cfg = cfg_dict.get("objective_weights") or cfg_dict.get("obj_weights") or {}
    obj_weights = ObjectiveWeights(
        w_pf=float(obj_cfg.get("w_pf", obj_defaults.w_pf)),
        w_expectancy=float(obj_cfg.get("w_expectancy", obj_defaults.w_expectancy)),
        w_sharpe=float(obj_cfg.get("w_sharpe", obj_defaults.w_sharpe)),
        w_maxdd=float(obj_cfg.get("w_maxdd", obj_defaults.w_maxdd)),
        pf_cap=float(obj_cfg.get("pf_cap", obj_defaults.pf_cap)),
    )

    fast_grid = _as_tuple(
        cfg_dict.get("fast_grid") or (cfg_dict.get("grids") or {}).get("fast") or defaults.fast_grid,
        int,
    )
    slow_grid = _as_tuple(
        cfg_dict.get("slow_grid") or (cfg_dict.get("grids") or {}).get("slow") or defaults.slow_grid,
        int,
    )
    qty_grid = _as_tuple(
        cfg_dict.get("qty_grid") or (cfg_dict.get("grids") or {}).get("qty") or defaults.qty_grid,
        float,
    )

    cfg = WFOServiceConfig(
        symbol=str(symbol),
        cooldown_sec=float(cfg_dict.get("cooldown_sec", defaults.cooldown_sec)),
        auto_apply=bool(cfg_dict.get("auto_apply", defaults.auto_apply)),
        obj_weights=obj_weights,
        min_is_bars=int(cfg_dict.get("min_is_bars", cfg_dict.get("wf_train_bars", defaults.min_is_bars))),
        min_oos_bars=int(cfg_dict.get("min_oos_bars", cfg_dict.get("wf_test_bars", defaults.min_oos_bars))),
        step_bars=int(cfg_dict.get("step_bars", cfg_dict.get("wf_step_bars", defaults.step_bars))),
        price_buffer=int(cfg_dict.get("price_buffer", defaults.price_buffer)),
        fast_grid=tuple(int(x) for x in fast_grid) or defaults.fast_grid,
        slow_grid=tuple(int(x) for x in slow_grid) or defaults.slow_grid,
        qty_grid=tuple(float(x) for x in qty_grid) or defaults.qty_grid,
    )

    runtime = {
        "loop_forever": bool(cfg_dict.get("loop_forever", False)),
        "loop_sleep_sec": float(cfg_dict.get("loop_sleep_sec", 30.0)),
        "timeframe": cfg_dict.get("timeframe", "1h"),
        "data": cfg_dict.get("data") or {},
    }
    return cfg, runtime


def _load_ohlc_from_csv(path: Path) -> Optional[List[Dict[str, float]]]:
    if not path.exists():
        log.warning("Plik CSV z danymi OHLC nie istnieje: %s", path)
        return None
    candles: List[Dict[str, float]] = []
    with path.open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            def _get(keys: Sequence[str]) -> Optional[float]:
                for key in keys:
                    value = row.get(key)
                    if value in (None, ""):
                        continue
                    try:
                        return float(value)
                    except ValueError:
                        continue
                return None

            close = _get(["close", "c"])
            if close is None:
                continue
            candle = {
                "close": close,
                "high": _get(["high", "h"]) or close,
                "low": _get(["low", "l"]) or close,
            }
            candles.append(candle)
    return candles or None


def fetch_ohlc(runtime_cfg: Dict[str, Any], symbol: str, timeframe: str) -> Optional[List[Dict[str, float]]]:
    data_cfg = runtime_cfg.get("data") or {}
    csv_path = data_cfg.get("ohlc_csv") or data_cfg.get("csv")
    if csv_path:
        return _load_ohlc_from_csv(Path(csv_path))
    # brak podłączonego providera -> odsyłamy None, żeby daemon wypisał instrukcję
    return None


def _forward(event_name: str):
    def _callback(events: Optional[Iterable[Event] | Event]) -> None:
        for evt in _iter_events(events):
            emit_stdout(event_name, evt.payload or {})
    return _callback


def _pump_prices(bus: EventBus, symbol: str, candles: List[Dict[str, float]]) -> None:
    for candle in candles:
        price = candle.get("close")
        if price is None:
            continue
        bus.publish(EventType.MARKET_TICK, {"symbol": symbol, "price": float(price)})


def main() -> None:
    parser = argparse.ArgumentParser(description="WalkForward daemon (paper mode)")
    parser.add_argument("--config", required=True, help="Ścieżka do pliku konfiguracyjnego JSON/JSON5")
    args = parser.parse_args()

    cfg_path = Path(args.config)
    if not cfg_path.exists():
        raise FileNotFoundError(f"Nie znaleziono pliku konfiguracyjnego: {cfg_path}")

    cfg_dict = load_config(cfg_path)
    service_cfg, runtime_cfg = build_service_config(cfg_dict)

    bus = EventBus()
    bus.subscribe(EventType.WFO_STATUS, _forward("WFO_STATUS"))
    bus.subscribe(EventType.AUTOTRADE_STATUS, _forward("AUTOTRADE_STATUS"))

    service = WalkForwardService(bus=bus, cfg=service_cfg)
    emit_stdout("DAEMON_STARTED", {"config": asdict(service_cfg)})

    try:
        while True:
            candles = fetch_ohlc(runtime_cfg, service_cfg.symbol, runtime_cfg["timeframe"])
            if not candles:
                emit_stdout("NO_DATA", {
                    "symbol": service_cfg.symbol,
                    "hint": "Dodaj sekcję data.ohlc_csv z drogą do pliku OHLC lub podłącz realny provider.",
                })
            else:
                _pump_prices(bus, service_cfg.symbol, candles)
                bus.publish(EventType.WFO_TRIGGER, {"symbol": service_cfg.symbol, "source": "wfa_daemon"})
                emit_stdout("WFO_TRIGGERED", {"bars": len(candles)})

            if not runtime_cfg["loop_forever"]:
                break
            time.sleep(max(0.1, runtime_cfg["loop_sleep_sec"]))
    except KeyboardInterrupt:
        emit_stdout("DAEMON_STOP", {"reason": "keyboard_interrupt"})
    finally:
        bus.stop()


if __name__ == "__main__":  # pragma: no cover - manualne uruchomienie
    main()
