from pathlib import Path

ROOT = Path(__file__).resolve().parent
target = ROOT / "KryptoLowca" / "types_common.py"

content = """from __future__ import annotations
from typing import TypedDict, NotRequired, Any

class OpenTradeTD(TypedDict, total=False):
    entry_time: Any  # datetime | str | float
    entry_price: float
    volume: float
    fees: NotRequired[float]
    slippage: NotRequired[float]
    position: NotRequired[float]
    entry_equity: NotRequired[float]

def to_float(v: object, default: float = 0.0) -> float:
    try:
        return float(v)  # type: ignore[arg-type]
    except Exception:
        return default
"""

if not target.exists():
    target.write_text(content, encoding="utf-8")
    print(f"[create] {target}")
else:
    print(f"[skip] {target} już istnieje")
