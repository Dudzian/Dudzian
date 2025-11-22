from __future__ import annotations

from datetime import datetime, timezone
import json
import base64

from bot_core.compliance.training import (
    TrainingSession,
    build_training_log_entry,
    write_training_log,
)


def test_training_session_payload_normalization(tmp_path):
    session = TrainingSession(
        session_id="2024-05-01",
        title="Szkolenie AML",
        trainer="Jan Kowalski",
        participants=[" Alice ", "Bob"],
        topics=["Procedury", " raportowanie "],
        occurred_at=datetime(2024, 5, 1, 9, 0, tzinfo=timezone.utc),
        duration_minutes=120,
        summary="Omówiono aktualizacje AML.",
        actions={"risk_team": "Przygotować aktualizację procedur."},
        materials=["slides.pdf", " checklist .md"],
        compliance_tags=["Stage5", "AML"],
        metadata={"room": "C.1"},
    )

    payload = session.to_payload()
    assert payload["participants"] == ["Alice", "Bob"]
    assert payload["topics"] == ["Procedury", "raportowanie"]
    assert payload["materials"] == ["slides.pdf", "checklist .md".strip()]
    assert payload["compliance_tags"] == ["aml", "stage5"]

    key = base64.b64encode(b"sekretny_klucz").decode("ascii")
    entry = build_training_log_entry(session, signing_key=base64.b64decode(key))
    assert "signature" in entry
    assert entry["signature"]["algorithm"] == "HMAC-SHA256"

    output = write_training_log(session, output=tmp_path / "training.json")
    content = json.loads(output.read_text(encoding="utf-8"))
    assert content["session_id"] == "2024-05-01"


def test_training_log_entry_logged_at_timezone():
    session = TrainingSession(
        session_id="S1",
        title="Compliance",
        trainer="Trainer",
        participants=[],
        topics=[],
        occurred_at=datetime(2024, 5, 1, 9, 0),  # naive
        duration_minutes=60,
        summary="Test",
    )
    logged = datetime(2024, 5, 2, 12, 0)
    entry = build_training_log_entry(session, logged_at=logged)
    assert entry["logged_at"].endswith("+00:00")
    assert entry["occurred_at"].endswith("+00:00")
