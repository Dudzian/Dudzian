from __future__ import annotations

import json
from pathlib import Path

from scripts import load_test_scheduler


def test_load_test_scheduler_cli(tmp_path: Path) -> None:
    output = tmp_path / "result.json"
    exit_code = load_test_scheduler.main(
        [
            "--iterations",
            "3",
            "--schedules",
            "2",
            "--signals",
            "2",
            "--latency-ms",
            "0.5",
            "--output",
            str(output),
        ]
    )
    assert exit_code == 0
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["signals_emitted"] == 3 * 2 * 2
    assert payload["settings"]["iterations"] == 3
