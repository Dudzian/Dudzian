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
from scripts.run_tco_analysis import run as run_tco


def _write_key(path: Path) -> bytes:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = os.urandom(48)
    path.write_bytes(data)
    if os.name != "nt":
        path.chmod(0o600)
    return data


def test_run_tco_analysis_generates_signed_reports(tmp_path: Path) -> None:
    fills_path = tmp_path / "fills.jsonl"
    fills_path.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "timestamp": "2024-03-01T12:00:00Z",
                        "strategy": "mean_reversion",
                        "risk_profile": "balanced",
                        "instrument": "BTC/USDT",
                        "exchange": "binance",
                        "side": "buy",
                        "quantity": 0.25,
                        "price": 20000,
                        "commission": 2.5,
                        "slippage": 1.0,
                        "funding": 0.25,
                    }
                ),
                json.dumps(
                    {
                        "timestamp": "2024-03-01T12:05:00Z",
                        "strategy": "volatility_target",
                        "risk_profile": "aggressive",
                        "instrument": "ETH/USDT",
                        "exchange": "kraken",
                        "side": "sell",
                        "quantity": 1.5,
                        "price": 3000,
                        "commission": 3.0,
                        "slippage": 0.5,
                        "funding": 0.1,
                        "other_costs": 0.05,
                    }
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    signing_key_path = tmp_path / "keys" / "hmac.key"
    signing_key = _write_key(signing_key_path)

    output_dir = tmp_path / "reports"
    exit_code = run_tco(
        [
            "--fills",
            str(fills_path),
            "--output-dir",
            str(output_dir),
            "--basename",
            "daily",
            "--signing-key-path",
            str(signing_key_path),
            "--signing-key-id",
            "stage5-tco",
            "--cost-limit-bps",
            "4.5",
            "--metadata",
            "environment=paper",
        ]
    )
    assert exit_code == 0

    csv_path = output_dir / "daily.csv"
    pdf_path = output_dir / "daily.pdf"
    json_path = output_dir / "daily.json"

    for artifact in (csv_path, pdf_path, json_path):
        assert artifact.exists()
        signature = artifact.with_suffix(artifact.suffix + ".sig")
        assert signature.exists()
        document = json.loads(signature.read_text(encoding="utf-8"))
        expected_signature = build_hmac_signature(
            document["payload"],
            key=signing_key,
            algorithm="HMAC-SHA256",
            key_id="stage5-tco",
        )
        assert document["signature"] == expected_signature

    payload = json.loads(json_path.read_text(encoding="utf-8"))
    assert payload["metadata"]["environment"] == "paper"
    assert payload["metadata"]["strategy_count"] == 2
    assert payload["total"]["trade_count"] == 2

    csv_contents = csv_path.read_text(encoding="utf-8").strip().splitlines()
    assert csv_contents[0].startswith("strategy,profile,trade_count")
    assert any("mean_reversion" in line for line in csv_contents)
    assert pdf_path.read_bytes().startswith(b"%PDF-1.4")


def test_run_tco_analysis_requires_key_length(tmp_path: Path) -> None:
    fills_path = tmp_path / "fills.jsonl"
    fills_path.write_text(
        json.dumps(
            {
                "timestamp": "2024-03-01T12:00:00Z",
                "strategy": "mean_reversion",
                "risk_profile": "balanced",
                "instrument": "BTC/USDT",
                "exchange": "binance",
                "side": "buy",
                "quantity": 0.25,
                "price": 20000,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    key_path = tmp_path / "bad.key"
    key_path.write_bytes(b"too-short-key")

    with pytest.raises(ValueError):
        run_tco(
            [
                "--fills",
                str(fills_path),
                "--output-dir",
                str(tmp_path / "reports"),
                "--signing-key-path",
                str(key_path),
            ]
        )
