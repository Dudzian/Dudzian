"""Restartuje lokalną kolejkę I/O runtime."""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path


def _log_action(result: dict[str, object]) -> None:
    log_dir = Path("logs/ui/runbook_actions")
    log_dir.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(result, ensure_ascii=False)
    (log_dir / "restart_io_queue.log").write_text(payload + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Restartuje kolejkę I/O runtime")
    parser.add_argument("--queue", default="*", help="Nazwa kolejki do restartu")
    parser.add_argument("--dry-run", action="store_true", help="Symuluje restart bez wykonania")
    args = parser.parse_args()

    timestamp = datetime.now(timezone.utc).isoformat()
    if args.dry_run:
        _log_action({"status": "dry-run", "queue": args.queue, "timestamp": timestamp})
        print(json.dumps({"status": "dry-run", "queue": args.queue}, ensure_ascii=False))
        return 0

    # W docelowej implementacji tutaj nastąpi restart kolejki.
    _log_action({"status": "restarted", "queue": args.queue, "timestamp": timestamp})
    print(json.dumps({"status": "restarted", "queue": args.queue}, ensure_ascii=False))
    return 0


if __name__ == "__main__":  # pragma: no cover - punkt wejścia CLI
    raise SystemExit(main())
