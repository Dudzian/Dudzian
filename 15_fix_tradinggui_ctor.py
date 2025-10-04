from pathlib import Path, PurePath
import re

p = Path("KryptoLowca/run_autotrade_paper.py")
txt = p.read_text(encoding="utf-8")
changed = False

# Podmień dokładnie "TradingGUI()" -> "TradingGUI(tk.Tk())"
if "TradingGUI()" in txt:
    if "import tkinter as tk" not in txt:
        # wstaw import po ostatnim imporcie
        lines = txt.splitlines()
        insert_at = 0
        for i, L in enumerate(lines):
            if L.startswith(("import ", "from ")):
                insert_at = i + 1
        lines.insert(insert_at, "import tkinter as tk")
        txt = "\n".join(lines)
    txt = txt.replace("TradingGUI()", "TradingGUI(tk.Tk())")
    changed = True

if changed:
    p.with_suffix(p.suffix + ".bak_gui").write_text(Path(p).read_text(encoding="utf-8"), encoding="utf-8")
    p.write_text(txt, encoding="utf-8")
    print(f"[gui-ctor-fix] {p}")
else:
    print("[gui-ctor-fix] no change")