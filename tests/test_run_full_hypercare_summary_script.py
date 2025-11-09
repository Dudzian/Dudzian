from __future__ import annotations

import json
from pathlib import Path

from bot_core.security.signing import build_hmac_signature
from scripts import run_full_hypercare_summary


def _write_summary(path: Path, payload: dict[str, object], key: bytes, key_id: str) -> Path:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    signature_path = path.with_suffix(path.suffix + ".sig")
    signature_path.write_text(
        json.dumps(build_hmac_signature(payload, key=key, key_id=key_id), ensure_ascii=False, separators=(",", ":")),
        encoding="utf-8",
    )
    return signature_path


def test_cli_builds_summary(tmp_path: Path) -> None:
    stage5_payload = {
        "type": "stage5_hypercare_summary",
        "overall_status": "ok",
        "issues": [],
        "warnings": [],
        "artifacts": {},
    }
    stage6_payload = {
        "type": "stage6_hypercare_summary",
        "overall_status": "ok",
        "issues": [],
        "warnings": [],
        "components": {},
    }

    stage5_path = tmp_path / "stage5.json"
    stage5_signature = _write_summary(stage5_path, stage5_payload, key=b"stage5", key_id="s5")
    stage6_path = tmp_path / "stage6.json"
    stage6_signature = _write_summary(stage6_path, stage6_payload, key=b"stage6", key_id="s6")

    output_path = tmp_path / "full.json"
    signature_path = tmp_path / "full.sig"

    archive_dir = tmp_path / "archive"

    exit_code = run_full_hypercare_summary.run(
        [
            "--stage5-summary",
            stage5_path.as_posix(),
            "--stage5-signature",
            stage5_signature.as_posix(),
            "--stage5-signing-key",
            "stage5",
            "--stage6-summary",
            stage6_path.as_posix(),
            "--stage6-signature",
            stage6_signature.as_posix(),
            "--stage6-signing-key",
            "stage6",
            "--output",
            output_path.as_posix(),
            "--signature",
            signature_path.as_posix(),
            "--signing-key",
            "full",
            "--require-stage5-signature",
            "--require-stage6-signature",
            "--archive-dir",
            archive_dir.as_posix(),
        ]
    )

    assert exit_code == 0
    assert output_path.exists()
    data = json.loads(output_path.read_text(encoding="utf-8"))
    assert data["type"] == "full_hypercare_summary"
    assert "archive_path" in data
    assert signature_path.exists()
    assert archive_dir.exists()
    archived = list(archive_dir.iterdir())
    assert archived, "Brak wygenerowanych artefaktów archiwum"


def test_cli_reports_missing_key(tmp_path: Path) -> None:
    stage5_path = tmp_path / "stage5.json"
    stage5_path.write_text("{}", encoding="utf-8")
    stage6_path = tmp_path / "stage6.json"
    stage6_path.write_text("{}", encoding="utf-8")

    exit_code = run_full_hypercare_summary.run(
        [
            "--stage5-summary",
            stage5_path.as_posix(),
            "--stage6-summary",
            stage6_path.as_posix(),
            "--output",
            (tmp_path / "full.json").as_posix(),
            "--signing-key",
            "full",
            "--signing-key-file",
            (tmp_path / "key.bin").as_posix(),
        ]
    )

    assert exit_code == 2

