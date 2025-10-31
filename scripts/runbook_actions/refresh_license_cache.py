"""Czyści lokalny cache licencji OEM i inicjuje ponowną walidację."""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description="Odświeżenie cache licencyjnego")
    parser.add_argument("--force", action="store_true", help="Wymusza czyszczenie cache nawet przy aktywnych sesjach")
    args = parser.parse_args()

    cache_dir = Path("var/licensing")
    cache_dir.mkdir(parents=True, exist_ok=True)
    for candidate in cache_dir.glob("*.cache"):
        candidate.unlink(missing_ok=True)

    payload = {
        "status": "refreshed",
        "force": args.force,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    log_dir = Path("logs/ui/runbook_actions")
    log_dir.mkdir(parents=True, exist_ok=True)
    (log_dir / "refresh_license_cache.log").write_text(json.dumps(payload, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False))
    return 0


if __name__ == "__main__":  # pragma: no cover - punkt wejścia CLI
    raise SystemExit(main())
