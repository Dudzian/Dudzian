#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path

CHUNK_SIZE = 25
BATCH_DIR = Path("tmp/ruff_batches")
STATE_PATH = Path("tmp/ruff_micro_state.json")


def main() -> None:
    batch_files = sorted(BATCH_DIR.glob("batch_*.txt"))
    if not batch_files:
        raise SystemExit("Brak tmp/ruff_batches/batch_*.txt")

    total_files = 0
    for batch_file in batch_files:
        lines = batch_file.read_text(encoding="utf-8").splitlines()
        total_files += sum(1 for line in lines if line.strip())

    state = {
        "chunk_size": CHUNK_SIZE,
        "batch_index": 0,
        "line_offset": 0,
        "total_batches": len(batch_files),
        "total_files": total_files,
    }

    STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    STATE_PATH.write_text(json.dumps(state, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(state, indent=2))


if __name__ == "__main__":
    main()
