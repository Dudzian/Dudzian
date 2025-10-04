from pathlib import Path

p = Path("KryptoLowca/run_autotrade_paper.py")
orig = p.read_text(encoding="utf-8")
lines = orig.splitlines()
changed = False

# Znajdź linie z logiem dot. TradingGUI i wyrównaj wcięcie do poprzedniej linii
targets = [i for i, L in enumerate(lines) if "log.info" in L and "TradingGUI" in L]
for i in targets:
    j = i - 1
    while j >= 0 and lines[j].strip() == "":
        j -= 1
    if j >= 0:
        prev = lines[j]
        prev_indent = len(prev) - len(prev.lstrip())
        this_indent = len(lines[i]) - len(lines[i].lstrip())
        # Jeśli poprzednia linia NIE otwiera bloku (nie kończy się dwukropkiem),
        # a bieżąca ma większe wcięcie – zredukuj je do wcięcia poprzedniej.
        if not prev.rstrip().endswith(":") and this_indent > prev_indent:
            lines[i] = " " * prev_indent + lines[i].lstrip()
            changed = True

if changed:
    p.with_suffix(p.suffix + ".bak_indent").write_text(orig, encoding="utf-8")
    p.write_text("\n".join(lines) + ("\n" if orig.endswith("\n") else ""), encoding="utf-8")
    print(f"[indent-fix] {p}")
else:
    print("[indent-fix] no change")