# ---- 08_more_fixes.py ----
from __future__ import annotations
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PKG = ROOT / "KryptoLowca"

def read(p: Path) -> str:
    return p.read_text(encoding="utf-8")

def write(p: Path, txt: str):
    b = p.with_suffix(p.suffix + ".bak")
    if not b.exists():
        b.write_text(read(p), encoding="utf-8")
    p.write_text(txt, encoding="utf-8")
    print(f"[patch] {p}")

def ensure_import(p: Path, line: str):
    txt = read(p)
    if line not in txt:
        # dodaj po pierwszym imporcie
        parts = txt.splitlines()
        for i, ln in enumerate(parts):
            if ln.startswith("from ") or ln.startswith("import "):
                parts.insert(i+1, line)
                write(p, "\n".join(parts) + "\n")
                return
        write(p, line + "\n" + txt)

# --- 1) protobuf: _FACTORY.GetPrototype -> dajemy _FACTORY: Any
def fix_telemetry():
    f = PKG / "telemetry_pb.py"
    if not f.exists():
        return
    txt = read(f)
    if "MessageFactory" in txt:
        ensure_import(f, "from typing import Any  # added by patch")
        txt2 = re.sub(
            r"(_FACTORY\s*=\s*message_factory\.MessageFactory\(\))",
            r"_FACTORY: Any = message_factory.MessageFactory()",
            txt,
            count=1,
        )
        if txt2 != txt:
            write(f, txt2)

# --- 2) on_events / on_batch: Event | list[Event] i normalizacja
def widen_event_handlers():
    # walkforward_service.on_events
    f = PKG / "backtest" / "walkforward_service.py"
    if f.exists():
        t = read(f)
        t2 = re.sub(
            r"(def\s+on_events\s*\(\s*events:\s*)(?:list|List)\s*\[\s*Event\s*\](\s*\)\s*->\s*None\s*:)",
            r"\1Event | list[Event]\2",
            t,
            count=1,
        )
        if "def on_events" in t2 and "isinstance(events, list)" not in t2:
            t2 = re.sub(
                r"(def\s+on_events\s*\([^\)]*\)\s*->\s*None\s*:\s*\n)",
                r"\1    if not isinstance(events, list):\n        events = [events]\n",
                t2,
                count=1,
            )
        if t2 != t:
            write(f, t2)

    # run_trading_gui_paper_emitter.on_batch
    f = PKG / "run_trading_gui_paper_emitter.py"
    if f.exists():
        t = read(f)
        t2 = re.sub(
            r"(def\s+on_batch\s*\(\s*events:\s*)(?:list|List)\s*\[\s*Event\s*\](\s*\)\s*->\s*None\s*:)",
            r"\1Event | list[Event]\2",
            t,
            count=1,
        )
        if "def on_batch" in t2 and "isinstance(events, list)" not in t2:
            t2 = re.sub(
                r"(def\s+on_batch\s*\([^\)]*\)\s*->\s*None\s*:\s*\n)",
                r"\1    if not isinstance(events, list):\n        events = [events]\n",
                t2,
                count=1,
            )
        if t2 != t:
            write(f, t2)

# --- 3) bezpieczny float w paru punktach + casty json
def safe_float_in(file_rel: str, patterns: list[tuple[str, str]]):
    f = PKG / file_rel
    if not f.exists():
        return
    t = read(f)
    new = t
    imported = False
    for rx, rp in patterns:
        n2 = re.sub(rx, rp, new)
        if n2 != new:
            new = n2
            imported = True
    if imported:
        ensure_import(f, "from KryptoLowca.types_common import to_float  # added by patch")
        write(f, new)

def json_casts(file_rel: str):
    f = PKG / file_rel
    if not f.exists():
        return
    t = read(f)
    ensure_import(f, "from typing import Any, Dict, cast  # added by patch")
    # return json.loads(....) -> return cast(Dict[str, Any], json.loads(...))
    new = re.sub(
        r"return\s+json\.loads\((.+?)\)",
        r"return cast(Dict[str, Any], json.loads(\1))",
        t,
        flags=re.S,
    )
    if new != t:
        write(f, new)

def fix_config_manager():
    # float(object) -> to_float(...)
    safe_float_in(
        "config_manager.py",
        [
            (r"float\(\s*bucket\.get\(\"window_seconds\",\s*0\.0\)\s*\)", r"to_float(bucket.get(\"window_seconds\", 0.0))"),
        ],
    )
    # preset.items() -> cast(...) .items()
    f = PKG / "config_manager.py"
    if f.exists():
        t = read(f)
        ensure_import(f, "from typing import Any, Dict, cast  # added by patch")
        n = re.sub(r"for\s+section,\s+payload\s+in\s+preset\.items\(\):",
                   r"for section, payload in cast(Dict[str, Any], preset).items():", t)
        if n != t:
            write(f, n)
    # json.loads(...) zwroty
    json_casts("config_manager.py")

def fix_secret_store():
    f = PKG / "security" / "secret_store.py"
    if not f.exists(): return
    t = read(f)
    ensure_import(f, "from typing import Optional, cast  # added by patch")
    # „return v” w funkcji deklarowanej jako -> str | None
    n = re.sub(r"return\s+v\s*$", r"return cast(Optional[str], v)", t, flags=re.M)
    if n != t:
        write(f, n)

def fix_paper_adapter():
    safe_float_in(
        "core/services/paper_adapter.py",
        [
            (r"float\(\s*close\s*\)", r"to_float(close)"),
        ],
    )

def fix_simulation_hotspots():
    # najczęstsze float(object)
    safe_float_in(
        "backtest/simulation.py",
        [
            (r"float\(\s*open_trade\.get\(\"fees\",\s*0\.0\)\s*\)", r"to_float(open_trade.get(\"fees\", 0.0))"),
            (r"float\(\s*open_trade\.get\(\"slippage\",\s*0\.0\)\s*\)", r"to_float(open_trade.get(\"slippage\", 0.0))"),
            (r"float\(\s*open_trade\.get\(\"volume\",\s*0\.0\)\s*\)", r"to_float(open_trade.get(\"volume\", 0.0))"),
            (r"float\(\s*open_trade\[\s*\"entry_equity\"\s*\]\s*\)", r"to_float(open_trade[\"entry_equity\"])"),
            (r"float\(\s*open_trade\[\s*\"entry_price\"\s*\]\s*\)", r"to_float(open_trade[\"entry_price\"])"),
            (r"float\(\s*open_trade\[\s*\"fees\"\s*\]\s*\)", r"to_float(open_trade[\"fees\"])"),
            (r"float\(\s*open_trade\[\s*\"slippage\"\s*\]\s*\)", r"to_float(open_trade[\"slippage\"])"),
            (r"float\(\s*open_trade\[\s*\"volume\"\s*\]\s*\)", r"to_float(open_trade[\"volume\"])"),
            (r"float\(\s*market_payload\.get\(\"price\"\)\s*or\s*0\.0\)", r"to_float(market_payload.get(\"price\") or 0.0)"),
            (r"float\(\s*config\.get\(\"max_position_notional_pct\",\s*0\.02\)\s*\)", r"to_float(config.get(\"max_position_notional_pct\", 0.02))"),
            (r"float\(\s*config\.get\(\"max_leverage\",\s*1\.0\)\s*\)", r"to_float(config.get(\"max_leverage\", 1.0))"),
        ],
    )

def relax_dashboard_desktop():
    f = PKG / "dashboard" / "desktop.py"
    if not f.exists(): return
    t = read(f)
    ensure_import(f, "from typing import Any  # added by patch")
    n = t
    n = re.sub(r"(self\.root\.title\([^\)]*\))", r"\1  # type: ignore[attr-defined]", n)
    n = re.sub(r"(self\.root\s*=\s*None)", r"\1  # type: ignore[assignment]", n)
    if n != t:
        write(f, n)

def main():
    fix_telemetry()
    widen_event_handlers()
    fix_config_manager()
    fix_secret_store()
    fix_paper_adapter()
    fix_simulation_hotspots()
    relax_dashboard_desktop()
    print("Dodatkowe poprawki zastosowane.")

if __name__ == "__main__":
    main()
