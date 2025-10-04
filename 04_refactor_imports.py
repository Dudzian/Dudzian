import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PKG = ROOT / "KryptoLowca"

patterns = [
    # backtest.* -> KryptoLowca.backtest.*
    (re.compile(r'(^\s*from\s+)backtest(\s+import\s+)', re.M), r'\1KryptoLowca.backtest\2'),
    (re.compile(r'(^\s*from\s+)backtest\.', re.M), r'\1KryptoLowca.backtest.'),
    # engine / metrics używane "goło" w podpakiecie backtest
    (re.compile(r'(^\s*from\s+)engine(\s+import\s+)', re.M), r'\1KryptoLowca.backtest.engine\2'),
    (re.compile(r'(^\s*from\s+)metrics(\s+import\s+)', re.M), r'\1KryptoLowca.backtest.metrics\2'),
    # reporting top-level
    (re.compile(r'(^\s*from\s+)reporting(\s+import\s+)', re.M), r'\1KryptoLowca.reporting\2'),
    # run_trading_gui_paper
    (re.compile(r'(^\s*from\s+)run_trading_gui_paper(\s+import\s+)', re.M), r'\1KryptoLowca.run_trading_gui_paper\2'),
    # trading_strategies
    (re.compile(r'(^\s*from\s+)trading_strategies(\s+import\s+)', re.M), r'\1KryptoLowca.trading_strategies\2'),
    # bridges
    (re.compile(r'(^\s*from\s+)bridges(\.ai_trading_bridge\s+import\s+)', re.M), r'\1KryptoLowca.bridges\2'),
    # "import backtest" -> import KryptoLowca.backtest as backtest
    (re.compile(r'(^\s*)import\s+backtest\s*$', re.M), r'\1import KryptoLowca.backtest as backtest'),
]

def rewrite_text(text: str) -> str:
    out = text
    for rx, repl in patterns:
        out = rx.sub(repl, out)
    return out

def main():
    changed = 0
    for py in PKG.rglob("*.py"):
        if ".venv" in py.parts:
            continue
        src = py.read_text(encoding="utf-8")
        dst = rewrite_text(src)
        if dst != src:
            backup = py.with_suffix(py.suffix + ".bak")
            backup.write_text(src, encoding="utf-8")
            py.write_text(dst, encoding="utf-8")
            print(f"[imports] {py}")
            changed += 1
    print(f"Zmieniono plików: {changed}")

if __name__ == "__main__":
    main()
