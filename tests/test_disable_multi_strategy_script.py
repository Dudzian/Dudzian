from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.disable_multi_strategy import DISABLE_FILENAME, run as disable_scheduler


def test_disable_scheduler_creates_override(tmp_path: Path) -> None:
    output_dir = tmp_path / "overrides"
    exit_code = disable_scheduler(
        [
            "--output-dir",
            str(output_dir),
            "--reason",
            "Awaria feedu danych",
            "--requested-by",
            "noc",
            "--ticket",
            "INC-2042",
            "--duration-minutes",
            "45",
        ]
    )

    assert exit_code == 0

    override_path = output_dir / DISABLE_FILENAME
    assert override_path.exists()
    payload = json.loads(override_path.read_text(encoding="utf-8"))
    assert payload["action"] == "disable_multi_strategy"
    assert payload["reason"] == "Awaria feedu danych"
    assert payload["ticket"] == "INC-2042"
    assert "expires_at" in payload

    if os.name != "nt":
        assert (override_path.stat().st_mode & 0o077) == 0

    with pytest.raises(FileExistsError):
        disable_scheduler([
            "--output-dir",
            str(output_dir),
            "--reason",
            "Ponowna próba",
        ])
