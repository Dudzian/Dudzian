from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bot_core.reporting.tco import (
    TcoUsageMetrics,
    aggregate_costs,
    load_cost_items,
    write_summary_csv,
    write_summary_json,
    write_summary_signature,
)
from bot_core.security.signing import build_hmac_signature


def test_tco_summary_and_exports(tmp_path: Path) -> None:
    input_payload = {
        "currency": "USD",
        "items": [
            {"name": "Serwer OEM", "category": "infrastructure", "monthly_cost": 120.5},
            {"name": "Backup", "category": "infrastructure", "monthly_cost": 35.0},
            {"name": "Szkolenia", "category": "operations", "monthly_cost": 42.75, "notes": "Warsztaty Stage5"},
        ],
    }
    cost_file = tmp_path / "costs.json"
    cost_file.write_text(json.dumps(input_payload, ensure_ascii=False, indent=2), encoding="utf-8")

    items = load_cost_items(cost_file)
    assert len(items) == 3

    summary = aggregate_costs(items)
    assert summary.currency == "USD"
    assert summary.monthly_total == pytest.approx(198.25)
    assert summary.annual_total == pytest.approx(2379.0)
    categories = {entry.category: entry.monthly_total for entry in summary.categories}
    assert categories == {"infrastructure": pytest.approx(155.5), "operations": pytest.approx(42.75)}

    usage = TcoUsageMetrics(monthly_trades=250.0, monthly_volume=480_000.0)
    generated_at = datetime(2024, 5, 1, 12, 0, tzinfo=timezone.utc)

    summary_path = tmp_path / "tco.json"
    payload = write_summary_json(summary, summary_path, generated_at=generated_at, usage=usage, metadata={"tag": "stage5"})
    assert summary_path.exists()
    assert payload["monthly_total"] == pytest.approx(198.25)
    assert payload["usage"]["cost_per_trade"] == pytest.approx(0.793)
    assert payload["usage"]["cost_per_volume_unit"] == pytest.approx(0.00041298, rel=1e-3)
    assert payload["tag"] == "stage5"

    csv_path = tmp_path / "tco.csv"
    write_summary_csv(summary, csv_path)
    csv_contents = csv_path.read_text(encoding="utf-8").splitlines()
    assert csv_contents[0].split(",")[:4] == ["category", "item", "monthly_cost", "annual_cost"]
    assert any("Serwer OEM" in line for line in csv_contents[1:])

    key = b"stage5-secret"
    signature_path = tmp_path / "tco.signature.json"
    signature = write_summary_signature(payload, signature_path, key=key, key_id="stage5")
    expected_signature = build_hmac_signature(payload, key=key, key_id="stage5")
    assert signature == expected_signature
    assert signature_path.exists()
