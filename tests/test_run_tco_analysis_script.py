from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bot_core.security.signing import build_hmac_signature  # noqa: E402
from scripts.run_tco_analysis import main as run_tco  # noqa: E402


def test_run_tco_analysis_cli(tmp_path: Path) -> None:
    input_payload = {
        "currency": "USD",
        "items": [
            {"name": "Serwer", "category": "infrastructure", "monthly_cost": 150.0},
            {"name": "Szkolenia operatorów", "category": "operations", "monthly_cost": 60.0},
        ],
    }
    input_path = tmp_path / "tco.json"
    input_path.write_text(json.dumps(input_payload, ensure_ascii=False, indent=2), encoding="utf-8")

    key_path = tmp_path / "key.bin"
    key_value = b"stage5-hypercare"
    key_path.write_bytes(key_value)

    artifact_root = tmp_path / "audit"
    timestamp = "20240501T120000Z"

    exit_code = run_tco(
        [
            "--input",
            str(input_path),
            "--artifact-root",
            str(artifact_root),
            "--timestamp",
            timestamp,
            "--monthly-trades",
            "180",
            "--monthly-volume",
            "420000",
            "--signing-key-file",
            str(key_path),
            "--signing-key-id",
            "stage5",
            "--tag",
            "weekly-cycle",
            "--print-summary",
        ]
    )

    assert exit_code == 0

    run_dir = artifact_root / timestamp
    json_path = run_dir / "tco_summary.json"
    csv_path = run_dir / "tco_breakdown.csv"
    signature_path = run_dir / "tco_summary.signature.json"

    assert json_path.exists()
    assert csv_path.exists()
    assert signature_path.exists()

    payload = json.loads(json_path.read_text(encoding="utf-8"))
    assert payload["monthly_total"] == 210.0
    assert payload["usage"]["monthly_trades"] == 180.0
    assert payload["tag"] == "weekly-cycle"
    assert payload["items_count"] == 2

    expected_signature = build_hmac_signature(payload, key=key_value, key_id="stage5")
    signature_contents = json.loads(signature_path.read_text(encoding="utf-8"))
    assert signature_contents == expected_signature

    csv_contents = csv_path.read_text(encoding="utf-8").splitlines()
    assert csv_contents[0].startswith("category,item")
    assert any("Serwer" in line for line in csv_contents[1:])
