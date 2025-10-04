import re
from pathlib import Path

path = Path("KryptoLowca/backtest/walkforward_service.py")
if not path.exists():
    raise SystemExit("Nie znaleziono pliku: " + str(path))

src = path.read_text(encoding="utf-8")
lines = src.splitlines()
changed = False

for i, line in enumerate(lines):
    m = re.match(r'^(\s*)def\s+on_events\s*\(', line)
    if not m:
        continue

    def_indent = len(m.group(1))
    # znajdź koniec ciała: pierwsza linia na poziomie <= def_indent,
    # która zaczyna def/class lub wywołanie subscribe
    end = None
    for k in range(i + 1, len(lines)):
        L = lines[k]
        if not L.strip():
            continue
        lead = len(L) - len(L.lstrip(' '))
        head = L.lstrip()
        if lead <= def_indent and re.match(r'(def\s+|class\s+|self\.bus\.subscribe\(|bus\.subscribe\()', head):
            end = k
            break
    if end is None:
        end = len(lines)

    # upewnij się, że po def jest przynajmniej jedna wcięta linia
    j = i + 1
    while j < len(lines) and not lines[j].strip():
        j += 1
    if j >= len(lines) or (len(lines[j]) - len(lines[j].lstrip(' '))) <= def_indent:
        lines.insert(i + 1, ' ' * (def_indent + 4) + 'pass')
        end += 1
        changed = True

    # do-wcięcie linii należących do ciała
    for k in range(i + 1, end):
        L = lines[k]
        if not L.strip():
            continue
        lead = len(L) - len(L.lstrip(' '))
        if lead <= def_indent:
            # podnieś na poziom ciała funkcji
            lines[k] = ' ' * (def_indent + 4) + L.lstrip(' ')
            changed = True
        elif lead < def_indent + 4:
            # wyrównaj minimalny poziom ciała
            pad = (def_indent + 4) - lead
            lines[k] = ' ' * pad + L
            changed = True
    break  # naprawiamy pierwsze wystąpienie

if changed:
    backup = path.with_suffix(path.suffix + ".bak_indent")
    backup.write_text(src, encoding="utf-8")
    path.write_text("\n".join(lines) + ("\n" if src.endswith("\n") else ""), encoding="utf-8")
    print(f"[indent-fix] {path}")
else:
    print("[indent-fix] brak zmian")