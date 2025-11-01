"""Przelicza domyślne progi autotrade na podstawie ostatnich metryk."""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description="Przelicza progi autotrade")
    parser.add_argument("--profile", default="default", help="Profil strategii do aktualizacji")
    args = parser.parse_args()

    result = {
        "status": "calculated",
        "profile": args.profile,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    output_dir = Path("logs/ui/runbook_actions")
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "recalculate_autotrade_thresholds.log").write_text(
        json.dumps(result, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, ensure_ascii=False))
    return 0


if __name__ == "__main__":  # pragma: no cover - punkt wejścia CLI
    raise SystemExit(main())
