from pathlib import Path

p = Path("KryptoLowca/managers/multi_account_manager.py")
orig = p.read_text(encoding="utf-8")
lines = orig.splitlines()
changed = False

# napraw brakujące ')'
for i, L in enumerate(lines):
    if "tasks.append(" in L and "asyncio.create_task" in L and "stream_market_data(" in L:
        opens = L.count("(")
        closes = L.count(")")
        if opens > closes:
            lines[i] = L + (")" * (opens - closes))
            changed = True
        break

# dodaj import asyncio jeśli używamy create_task, a importu brak
txt = "\n".join(lines)
if "asyncio.create_task" in txt and "import asyncio" not in txt:
    # wstaw po bloku importów
    insert_at = 0
    for j, LJ in enumerate(lines):
        if LJ.startswith(("import ", "from ")):
            insert_at = j + 1
    lines.insert(insert_at, "import asyncio")
    changed = True

if changed:
    p.with_suffix(p.suffix + ".bak_paren").write_text(orig, encoding="utf-8")
    p.write_text("\n".join(lines) + ("\n" if orig.endswith("\n") else ""), encoding="utf-8")
    print(f"[paren-fix] {p}")
else:
    print("[paren-fix] brak zmian")