import re
from pathlib import Path

ROOT = Path("KryptoLowca")
changed = 0

def fix_text(txt: str) -> str:
    out = txt

    # --- .get("...") / .get('...') ---
    # otwierający cudzysłów/apostrof po .get(
    out = re.sub(r'(\.get\()\\"', r'\1"', out)
    out = re.sub(r"(\.get\()\\'", r"\1'", out)
    # zamykający cudzysłów/apostrof tuż przed , ) ]
    out = re.sub(r'(\.get\([^)]{0,300})\\(["\'])(\s*[,)\]])', r'\1\2\3', out)

    # --- indeksowanie ["..."] / ['...'] ---
    # otwierający cudzysłów/apostrof po [
    out = re.sub(r'(\[)\\"', r'\1"', out)
    out = re.sub(r"(\[)\\'", r"\1'", out)
    # zamykający cudzysłów/apostrof tuż przed ]
    out = re.sub(r'\\(["\'])(\])', r'\1\2', out)

    return out

for p in ROOT.rglob("*.py"):
    if ".venv" in p.parts:
        continue
    txt = p.read_text(encoding="utf-8")
    out = fix_text(txt)
    if out != txt:
        p.with_suffix(p.suffix + ".bak_quotes").write_text(txt, encoding="utf-8")
        p.write_text(out, encoding="utf-8")
        print(f"[fix-edge] {p}")
        changed += 1

print(f"[fix-edge] files changed: {changed}")