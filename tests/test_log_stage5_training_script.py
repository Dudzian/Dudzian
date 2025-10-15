from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bot_core.security.signing import build_hmac_signature
from scripts.log_stage5_training import run as log_training


def _write_key(path: Path) -> bytes:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = os.urandom(48)
    path.write_bytes(data)
    if os.name != "nt":
        path.chmod(0o600)
    return data


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def test_log_stage5_training_end_to_end(tmp_path: Path) -> None:
    training_log = tmp_path / "var" / "audit" / "training" / "stage5.jsonl"
    decision_log = tmp_path / "var" / "audit" / "decisions" / "stage5.jsonl"
    log_key_path = tmp_path / "keys" / "training.key"
    decision_key_path = tmp_path / "keys" / "decision.key"
    artifact = tmp_path / "materials" / "slides.pdf"
    artifact.parent.mkdir(parents=True, exist_ok=True)
    artifact.write_bytes(b"dummy")

    log_key = _write_key(log_key_path)
    decision_key = _write_key(decision_key_path)

    exit_code = log_training(
        [
            "--log-path",
            str(training_log),
            "--training-date",
            "2024-05-18",
            "--start-time",
            "09:00",
            "--duration-minutes",
            "210",
            "--facilitator",
            "Anna Kowalska",
            "--location",
            "Sala 3A",
            "--participant",
            "Jan Nowak",
            "--participant",
            "Ewa Wiśniewska",
            "--topic",
            "TCO workflows",
            "--material",
            "Prezentacja PDF",
            "--artifact",
            str(artifact),
            "--notes",
            "Warsztat Stage5 – edycja pilotażowa",
            "--log-hmac-key-file",
            str(log_key_path),
            "--decision-log-path",
            str(decision_log),
            "--decision-log-hmac-key-file",
            str(decision_key_path),
            "--decision-log-key-id",
            "stage5-dl",
            "--decision-log-category",
            "stage5.training",
            "--decision-log-notes",
            "Warsztat pilotażowy",
        ]
    )

    assert exit_code == 0
    assert training_log.is_file()
    entries = _read_jsonl(training_log)
    assert len(entries) == 1
    entry = entries[0]
    signature = entry.pop("signature")
    expected_signature = build_hmac_signature(entry, key=log_key, algorithm="HMAC-SHA256")
    assert signature == expected_signature
    assert entry["schema"] == "stage5.training_log"
    assert entry["training_date"] == "2024-05-18"
    assert entry["facilitator"] == "Anna Kowalska"
    assert entry["participants"] == ["Jan Nowak", "Ewa Wiśniewska"]
    artifacts = entry["artifacts"]
    assert len(artifacts) == 1
    artifact_record = next(iter(artifacts.values()))
    assert artifact_record["size_bytes"] == artifact.stat().st_size

    decision_entries = _read_jsonl(decision_log)
    assert len(decision_entries) == 1
    decision_entry = decision_entries[0]
    decision_signature = decision_entry.pop("signature")
    expected_decision_signature = build_hmac_signature(
        decision_entry,
        key=decision_key,
        algorithm="HMAC-SHA256",
        key_id="stage5-dl",
    )
    assert decision_signature == expected_decision_signature
    assert decision_entry["schema"] == "stage5.training_session"
    assert decision_entry["category"] == "stage5.training"
    assert decision_entry["participants"] == ["Jan Nowak", "Ewa Wiśniewska"]
    assert decision_entry["notes"] == "Warsztat pilotażowy"


def test_log_stage5_training_rejects_duplicate_participants(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        log_training(
            [
                "--facilitator",
                "Anna",
                "--location",
                "Sala",
                "--participant",
                "Jan",
                "--participant",
                "jan",
            ]
        )
