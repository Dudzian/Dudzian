import re
from pathlib import Path

ROOT = Path("KryptoLowca")
if not ROOT.exists():
    print("[skip] no KryptoLowca dir")
    raise SystemExit(0)

patts = [
    (re.compile(r"\(\\\""), '("'),   # .get(\"foo\", ...) -> .get("foo", ...)
    (re.compile(r"\(\\\'"), "('"),   # .get(\'foo\', ...) -> .get('foo', ...)
    (re.compile(r"\[\\\""), '["'),   # ["foo\"] -> ["foo"]
    (re.compile(r"\\\"\]"), '"]'),   # ["foo\"] -> ["foo"]
]

changed_total = 0
for py in ROOT.rglob("*.py"):
    if ".venv" in py.parts: 
        continue
    txt = py.read_text(encoding="utf-8")
    new = txt
    for rx, repl in patts:
        new = rx.sub(repl, new)
    if new != txt:
        bak = py.with_suffix(py.suffix + ".bak")
        bak.write_text(txt, encoding="utf-8")
        py.write_text(new, encoding="utf-8")
        print(f"[fix] {py}")
        changed_total += 1

print(f"[fix] files changed: {changed_total}")