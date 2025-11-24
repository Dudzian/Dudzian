import json
from datetime import datetime, timezone
from pathlib import Path

from bot_core.security.fingerprint_lock import load_fingerprint_lock


def test_load_fingerprint_lock_reads_json_metadata(tmp_path: Path) -> None:
    lock_path = tmp_path / "fingerprint.json"
    payload = {
        "fingerprint": "abcd-1234",
        "created_at": "2024-03-01T02:03:04Z",
        "metadata": {"region": "eu"},
    }
    lock_path.write_text(json.dumps(payload), encoding="utf-8")

    lock = load_fingerprint_lock(lock_path)

    assert lock is not None
    assert lock.path == lock_path
    assert lock.fingerprint == "ABCD-1234"
    assert lock.created_at == datetime(2024, 3, 1, 2, 3, 4, tzinfo=timezone.utc)
    assert lock.metadata == {"region": "eu"}

