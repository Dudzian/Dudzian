from __future__ import annotations
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent
P = ROOT / "KryptoLowca"

def patch_file(path: Path, replacers: list[tuple[re.Pattern,str]]) -> bool:
    if not path.exists():
        return False
    txt = path.read_text(encoding="utf-8")
    orig = txt
    for rx, repl in replacers:
        txt = rx.sub(repl, txt)
    if txt != orig:
        path.with_suffix(path.suffix + ".bak").write_text(orig, encoding="utf-8")
        path.write_text(txt, encoding="utf-8")
        print(f"[patch] {path}")
        return True
    return False

def ensure_import(path: Path, import_line: str) -> None:
    txt = path.read_text(encoding="utf-8")
    if import_line not in txt:
        # dodaj po pierwszym imporcie lub na początku pliku
        lines = txt.splitlines()
        for i,l in enumerate(lines):
            if l.startswith("from ") or l.startswith("import "):
                lines.insert(i+1, import_line)
                break
        else:
            lines.insert(0, import_line)
        new = "\n".join(lines) + "\n"
        path.with_suffix(path.suffix + ".bak2").write_text(txt, encoding="utf-8")
        path.write_text(new, encoding="utf-8")
        print(f"[import+] {path}: {import_line}")

def patch_walkforward_service():
    f = P / "backtest" / "walkforward_service.py"
    if not f.exists(): return
    txt = f.read_text(encoding="utf-8")

    # def on_events(events: list[Event]) -> None  -> Event | list[Event]
    rx_sig = re.compile(r"(def\s+on_events\s*\(\s*events:\s*)list\[\s*Event\s*\](\s*\)\s*->\s*None\s*:\s*)")
    txt2 = rx_sig.sub(r"\1Event | list[Event]\2", txt)

    # wstaw normalizację na początku funkcji, jeśli jej nie ma
    if "def on_events" in txt2 and "isinstance(events, list)" not in txt2:
        txt2 = re.sub(
            r"(def\s+on_events\s*\([^\)]*\)\s*->\s*None\s*:\s*\n)",
            r"\1    if not isinstance(events, list):\n        events = [events]\n",
            txt2
        )

    if txt2 != txt:
        f.with_suffix(".py.bak").write_text(txt, encoding="utf-8")
        f.write_text(txt2, encoding="utf-8")
        print(f"[patch] {f}")

def patch_run_trading_gui_paper_emitter():
    f = P / "run_trading_gui_paper_emitter.py"
    if not f.exists(): return
    # def on_batch(events: list[Event]) -> None  -> Event | list[Event]
    rx_sig = re.compile(r"(def\s+on_batch\s*\(\s*events:\s*)list\[\s*Event\s*\](\s*\)\s*->\s*None\s*:\s*)")
    txt = f.read_text(encoding="utf-8")
    txt2 = rx_sig.sub(r"\1Event | list[Event]\2", txt)
    if "def on_batch" in txt2 and "isinstance(events, list)" not in txt2:
        txt2 = re.sub(
            r"(def\s+on_batch\s*\([^\)]*\)\s*->\s*None\s*:\s*\n)",
            r"\1    if not isinstance(events, list):\n        events = [events]\n",
            txt2
        )
    if txt2 != txt:
        f.with_suffix(".py.bak").write_text(txt, encoding="utf-8")
        f.write_text(txt2, encoding="utf-8")
        print(f"[patch] {f}")

def patch_multi_account_manager():
    f = P / "managers" / "multi_account_manager.py"
    if not f.exists(): return
    txt = f.read_text(encoding="utf-8")
    # tasks.append(managed.adapter.stream_market_data(...)) -> asyncio.create_task(...)
    rx = re.compile(r"\.append\(\s*managed\.adapter\.stream_market_data\(", re.M)
    txt2 = rx.sub(".append(asyncio.create_task(managed.adapter.stream_market_data(", txt)
    if txt2 != txt:
        f.with_suffix(".py.bak").write_text(txt, encoding="utf-8")
        f.write_text(txt2, encoding="utf-8")
        print(f"[patch] {f}")
        ensure_import(f, "import asyncio")

def patch_database_manager():
    f = P / "managers" / "database_manager.py"
    if not f.exists(): return
    txt = f.read_text(encoding="utf-8")
    # Uprość wywołanie sessionmaker -> async_sessionmaker(...)
    rx = re.compile(r"sessionmaker\s*\([\s\S]*?\)", re.M)
    newcall = "async_sessionmaker(bind=self._engine, autoflush=True, expire_on_commit=False)"
    txt2 = rx.sub(newcall, txt, count=1)
    if txt2 != txt:
        f.with_suffix(".py.bak").write_text(txt, encoding="utf-8")
        f.write_text(txt2, encoding="utf-8")
        print(f"[patch] {f}")
        ensure_import(f, "from sqlalchemy.ext.asyncio import async_sessionmaker")

def patch_preset_store():
    f = P / "backtest" / "preset_store.py"
    if not f.exists(): return
    txt = f.read_text(encoding="utf-8")
    rx = re.compile(r"items\s*=\s*\[p\s+for\s+p\s+in\s+list\(items\)\s+if\s+metric\s+in\s+p\.metrics\]")
    repl = "items_list = [p for p in list(items) if metric in p.metrics]\n            items = items_list"
    txt2 = rx.sub(repl, txt)
    if txt2 != txt:
        f.with_suffix(".py.bak").write_text(txt, encoding="utf-8")
        f.write_text(txt2, encoding="utf-8")
        print(f"[patch] {f}")

def patch_wfa_daemon():
    f = P / "wfa_daemon.py"
    if not f.exists(): return
    txt = f.read_text(encoding="utf-8")
    rx = re.compile(r"return\s+\[events\]")
    txt2 = rx.sub("return [events] if isinstance(events, Event) else list(events)", txt)
    if txt2 != txt:
        f.with_suffix(".py.bak").write_text(txt, encoding="utf-8")
        f.write_text(txt2, encoding="utf-8")
        print(f"[patch] {f}")

def patch_zonda():
    f = P / "exchanges" / "zonda.py"
    if not f.exists(): return
    txt = f.read_text(encoding="utf-8")
    txt2 = txt
    # targets = symbols or [None] -> symbols or []
    txt2 = re.sub(r"targets\s*=\s*symbols\s*or\s*\[None\]", "targets = symbols or []", txt2)
    # dict(template["params"]) -> dict(cast(Mapping[str, Any], template["params"]))
    txt2 = re.sub(r'dict\(\s*template\["params"\]\s*\)', 'dict(cast(Mapping[str, Any], template["params"]))', txt2)
    if txt2 != txt:
        f.with_suffix(".py.bak").write_text(txt, encoding="utf-8")
        f.write_text(txt2, encoding="utf-8")
        print(f"[patch] {f}")
        ensure_import(f, "from typing import Any, Mapping, cast")

def patch_trading_gui():
    f = P / "trading_gui.py"
    if not f.exists(): return
    txt = f.read_text(encoding="utf-8")
    # Zamień 'proj = []' na 'proj: list[float] = []' tylko jeśli w pliku jest 'proj.append('
    if "proj.append(" in txt:
        txt2 = re.sub(r"\bproj\s*=\s*\[\s*\]", "proj: list[float] = []", txt, count=1)
        if txt2 != txt:
            f.with_suffix(".py.bak").write_text(txt, encoding="utf-8")
            f.write_text(txt2, encoding="utf-8")
            print(f"[patch] {f}")

def patch_run_autotrade_paper():
    f = P / "run_autotrade_paper.py"
    if not f.exists(): return
    txt = f.read_text(encoding="utf-8")
    if "TradingGUI()" in txt:
        txt2 = txt
        # wstrzyknij root = tk.Tk() i przekazuj root
        txt2 = txt2.replace(
            "gui = trading_gui.TradingGUI()",
            "import tkinter as tk\n        root = tk.Tk()\n        gui = trading_gui.TradingGUI(root)"
        )
        if txt2 != txt:
            f.with_suffix(".py.bak").write_text(txt, encoding="utf-8")
            f.write_text(txt2, encoding="utf-8")
            print(f"[patch] {f}")

def main():
    patch_walkforward_service()
    patch_run_trading_gui_paper_emitter()
    patch_multi_account_manager()
    patch_database_manager()
    patch_preset_store()
    patch_wfa_daemon()
    patch_zonda()
    patch_trading_gui()
    patch_run_autotrade_paper()
    print("Hotfixy zastosowane.")

if __name__ == "__main__":
    main()
