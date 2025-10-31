from __future__ import annotations

from pathlib import Path

import pytest


from bot_core.security.signing import build_hmac_signature  # noqa: E402
from scripts.log_stage5_training import run as log_training  # noqa: E402
from tests._json_helpers import read_jsonl
from tests._signing_helpers import write_random_hmac_key


ROOT = Path(__file__).resolve().parents[1]
def test_log_stage5_training_writes_signed_file(tmp_path: Path) -> None:
    """
    Zastępuje starszy test z HEAD: weryfikuje, że narzędzie zapisuje
    podpisany wpis szkoleniowy (JSONL) – sprawdzamy strukturę i algorytm.
    """
    training_log = tmp_path / "var" / "audit" / "training" / "stage5.jsonl"
    log_key_path = tmp_path / "keys" / "training.key"
    write_random_hmac_key(log_key_path)

    exit_code = log_training(
        [
            "--log-path",
            str(training_log),
            "--training-date",
            "2024-05-18",
            "--start-time",
            "10:00",
            "--duration-minutes",
            "90",
            "--facilitator",
            "Anna Trainer",
            "--location",
            "Sala 101",
            "--participant",
            "Anna",
            "--participant",
            "Bob",
            "--topic",
            "Stage5",
            "--topic",
            "Compliance",
            "--log-hmac-key-file",
            str(log_key_path),
        ]
    )
    assert exit_code == 0
    assert training_log.is_file()

    entries = read_jsonl(training_log)
    assert len(entries) == 1
    entry = entries[0]
    assert entry["participants"] == ["Anna", "Bob"]
    assert "signature" in entry and isinstance(entry["signature"], dict)
    assert entry["signature"]["algorithm"] == "HMAC-SHA256"


def test_log_stage5_training_end_to_end(tmp_path: Path) -> None:
    """
    Pełny scenariusz z 'main': log szkoleniowy + wpis decision logu, oba z podpisami.
    """
    training_log = tmp_path / "var" / "audit" / "training" / "stage5.jsonl"
    decision_log = tmp_path / "var" / "audit" / "decisions" / "stage5.jsonl"
    log_key_path = tmp_path / "keys" / "training.key"
    decision_key_path = tmp_path / "keys" / "decision.key"
    artifact = tmp_path / "materials" / "slides.pdf"
    artifact.parent.mkdir(parents=True, exist_ok=True)
    artifact.write_bytes(b"dummy")

    log_key = write_random_hmac_key(log_key_path)
    decision_key = write_random_hmac_key(decision_key_path)

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
    entries = read_jsonl(training_log)
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

    decision_entries = read_jsonl(decision_log)
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
