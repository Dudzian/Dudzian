from __future__ import annotations

import base64
import json

import pytest

from scripts import log_stage5_training


def test_log_stage5_training_writes_signed_file(tmp_path):
    key = base64.b64encode(b"trening_klucz").decode("ascii")
    output_path = tmp_path / "training.json"

    exit_code = log_stage5_training.run(
        [
            "S05-TRAIN-01",
            "Stage5 Compliance",
            "Anna Trainer",
            "--summary",
            "Przegląd procedur Stage5",
            "--participants",
            "Anna,Bob",
            "--topics",
            "Stage5,Compliance",
            "--signing-key",
            key,
            "--output",
            str(output_path),
        ]
    )

    assert exit_code == 0
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload["participants"] == ["Anna", "Bob"]
    assert payload["signature"]["algorithm"] == "HMAC-SHA256"


def test_log_stage5_training_requires_summary(tmp_path):
    with pytest.raises(SystemExit):
        log_stage5_training.run([
            "S05-TRAIN-02",
            "Stage5 Compliance",
            "Trainer",
        ])
