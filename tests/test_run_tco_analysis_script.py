from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# HMAC helper wspólny dla obu wariantów
from bot_core.security.signing import build_hmac_signature  # noqa: E402

# Obsłuż oba typy eksportów funkcji CLI (run | main)
try:
    from scripts.run_tco_analysis import run as run_tco  # noqa: E402
except Exception:  # pragma: no cover
    from scripts.run_tco_analysis import main as run_tco  # type: ignore[no-redef]  # noqa: E402

# Potrzebujemy parsera, żeby wykryć kształt CLI (HEAD vs main)
import scripts.run_tco_analysis as run_tco_mod  # noqa: E402


# ----------------- detekcja wariantu CLI -----------------
def _parser():
    build = getattr(run_tco_mod, "_build_parser", None)
    if build is None:
        raise RuntimeError("scripts.run_tco_analysis._build_parser is not available")
    return build()


def _parser_supports(*flags: str) -> bool:
    parser = _parser()
    actions = getattr(parser, "_actions", ())
    option_set = set()
    for act in actions:
        option_set.update(getattr(act, "option_strings", []) or [])
    return all(flag in option_set for flag in flags)


def _supports_head_cli() -> bool:
    # HEAD: --input / --artifact-root / --signing-key-file / --json-name / --csv-name / --signature-name
    return _parser_supports("--input", "--artifact-root")


def _supports_main_cli() -> bool:
    # main: --fills / --output-dir / --basename / --signing-key-path
    return _parser_supports("--fills", "--output-dir")


# ----------------- pomocnicze -----------------
def _write_key(path: Path) -> bytes:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = os.urandom(48)
    path.write_bytes(data)
    if os.name != "nt":
        path.chmod(0o600)
    return data


# ----------------- TESTY -----------------
def test_run_tco_analysis_generates_signed_reports_or_summary(tmp_path: Path) -> None:
    if _supports_head_cli():
        # Wariant HEAD: wejście to skonsolidowany JSON kosztów + podpis jednego podsumowania
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
        # wartości z sumy pozycji
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
        return

    if _supports_main_cli():
        # Wariant main: wejście to JSONL "fills" + CSV/PDF/JSON + podpisy dla każdego artefaktu
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
        # W zależności od implementacji może to być liczba strategii lub sumaryczna metryka
        assert payload["total"]["trade_count"] == 2

        csv_contents = csv_path.read_text(encoding="utf-8").strip().splitlines()
        assert csv_contents[0].startswith("strategy,profile,trade_count")
        assert any("mean_reversion" in line for line in csv_contents)
        # PDF nagłówek
        assert pdf_path.read_bytes().startswith(b"%PDF-")
        return

    pytest.skip("run_tco_analysis CLI shape not recognized (neither HEAD nor main)")


def test_run_tco_analysis_requires_key_length_if_applicable(tmp_path: Path) -> None:
    if not _supports_main_cli():
        pytest.skip("Key length validation applies to 'main' CLI variant only")

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
