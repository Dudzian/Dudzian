from pathlib import Path
import re

path = Path("KryptoLowca/backtest/walkforward_service.py")
src = path.read_text(encoding="utf-8")
lines = src.splitlines()

pat = re.compile(r'^(\s*)if\s+not\s+isinstance\(\s*events\s*,\s*list\s*\)\s*:\s*$')
changed = False

for i, L in enumerate(lines):
    m = pat.match(L)
    if not m:
        continue
    if_indent = len(m.group(1))
    # znajdź pierwszą następną niepustą i niekomentowaną linię
    j = i + 1
    while j < len(lines) and (not lines[j].strip() or lines[j].lstrip().startswith("#")):
        j += 1
    if j >= len(lines):
        break
    lead_j = len(lines[j]) - len(lines[j].lstrip(' '))
    if lead_j <= if_indent:
        # do-wcięcie tej linii (zachowaj treść bez wiodących spacji)
        lines[j] = ' ' * (if_indent + 4) + lines[j].lstrip(' ')
        changed = True
    break  # naprawiamy pierwsze wystąpienie

if changed:
    backup = path.with_suffix(path.suffix + ".bak_ifindent")
    backup.write_text(src, encoding="utf-8")
    path.write_text("\n".join(lines) + ("\n" if src.endswith("\n") else ""), encoding="utf-8")
    print(f"[if-indent-fix] {path}")
else:
    print("[if-indent-fix] brak zmian")