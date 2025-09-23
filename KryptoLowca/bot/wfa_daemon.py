# wfa_daemon.py
# -*- coding: utf-8 -*-
"""
Prosty daemon WF:
- Ładuje config JSON,
- Buduje WalkForwardService,
- Emisja eventów jako NDJSON (1 linia = {"event": "...", "payload": {...}}) na stdout.
GUI może to czytać przez pipe/WebSocket bridge.

Użycie:
  python wfa_daemon.py --config wfa_config.json

Jeśli chcesz tryb ciągły (live), ustaw w configu: "loop_forever": true.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

# import serwisu
from services.walkforward_service import (
    WalkForwardService,
    WFConfig,
    ReoptRules,
    SubprocessCmds,
)

# ===== Emiter eventów -> stdout (NDJSON) =====

def emit_stdout(event: str, payload: Dict[str, Any]) -> None:
    rec = {"ts": datetime.utcnow().isoformat(timespec="seconds"), "event": event, "payload": payload}
    sys.stdout.write(json.dumps(rec, ensure_ascii=False) + "\n")
    sys.stdout.flush()

# ===== Opcjonalny provider OHLC =====
# Podmień to na swój loader danych (DB, pliki, API).
# Powinien zwrócić listę słowników: {"high": float, "low": float, "close": float} w kolejności po czasie.
def ohlc_provider_stub(symbol: str, timeframe: str, dt_from, dt_to) -> Optional[List[Dict[str, float]]]:
    # Brak realnego providera – zwracamy None (ATR wyłączony)
    return None

# ===== Parsowanie configu =====

def load_config(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()
    # prosty parser JSON (usuń //-komentarze jeśli używasz czystego JSON)
    lines = []
    for line in text.splitlines():
        line = line.strip()
        if line.startswith("//"):
            continue
        lines.append(line)
    return json.loads("\n".join(lines))

def build_service_from_config(cfg_dict: Dict[str, Any]) -> WalkForwardService:
    rules = cfg_dict.get("rules") or {}
    sp = cfg_dict.get("subprocess_cmds") or {}

    wf_cfg = WFConfig(
        symbols=cfg_dict["symbols"],
        timeframe=cfg_dict["timeframe"],
        max_bars=int(cfg_dict.get("max_bars", 12000)),
        wf_train_bars=int(cfg_dict.get("wf_train_bars", 6000)),
        wf_test_bars=int(cfg_dict.get("wf_test_bars", 1500)),
        wf_step_bars=int(cfg_dict.get("wf_step_bars", 0)) or None,
        min_trades=int(cfg_dict.get("min_trades", 10)),
        ema_slow=int(cfg_dict.get("ema_slow", 150)),
        capital=float(cfg_dict.get("capital", 10000.0)),
        risk_pct=float(cfg_dict.get("risk_pct", 0.5)),
        grids=cfg_dict.get("grids") or {},
        workdir=Path(cfg_dict.get("workdir", "walkforwards")),
        rules=ReoptRules(
            reopt_pf_below=float(rules.get("reopt_pf_below", 1.20)),
            reopt_pf_drop_pct=float(rules.get("reopt_pf_drop_pct", 35.0)),
            reopt_exp_below=float(rules.get("reopt_exp_below", 0.0)),
            reopt_exp_drop_pct=float(rules.get("reopt_exp_drop_pct", 40.0)),
            atr_lookback=int(rules.get("atr_lookback", 14)),
            atr_rise_pct=float(rules.get("atr_rise_pct", 25.0)),
        ),
        poll_interval_sec=float(cfg_dict.get("poll_interval_sec", 0.5)),
        subprocess_cmds=SubprocessCmds(
            optimize_cmd=sp.get("optimize_cmd"),
            run_cmd=sp.get("run_cmd"),
        ),
        loop_forever=bool(cfg_dict.get("loop_forever", False)),
        loop_sleep_sec=float(cfg_dict.get("loop_sleep_sec", 30.0)),
    )

    # ATR działa tylko, jeśli podasz realnego providera
    service = WalkForwardService(cfg=wf_cfg, emit=emit_stdout, ohlc_provider=None)
    return service

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="Ścieżka do pliku konfiguracyjnego JSON/JSON5")
    args = ap.parse_args()

    cfg_path = Path(args.config)
    cfg = load_config(cfg_path)
    service = build_service_from_config(cfg)

    service.start()

    # Prostolinijne oczekiwanie na zakończenie.
    # W GUI zwykle nie używasz tej pętli – serwis żyje w tle, a GUI obsługuje interakcję.
    try:
        while service.is_running():
            time.sleep(0.3)
    except KeyboardInterrupt:
        service.stop()
        # daj mu ładnie się zamknąć
        for _ in range(50):
            if not service.is_running():
                break
            time.sleep(0.1)

if __name__ == "__main__":
    main()
