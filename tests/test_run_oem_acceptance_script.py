from __future__ import annotations

import base64
import hashlib
import hmac
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_oem_acceptance import main as run_acceptance


def _write_signing_key(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(os.urandom(48))
    if os.name != "nt":
        path.chmod(0o600)


def _write_core_config(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rotation_registry = path.parent / "rotation_registry.json"
    rotation_audit = path.parent / "rotation_audit"
    config = {
        "risk_profiles": {
            "conservative": {
                "max_daily_loss_pct": 0.02,
                "max_position_pct": 0.05,
                "target_volatility": 0.1,
                "max_leverage": 3.0,
                "stop_loss_atr_multiple": 1.0,
                "max_open_positions": 3,
                "hard_drawdown_pct": 0.1,
                "data_quality": {"max_gap_minutes": 1440, "min_ok_ratio": 0.9},
                "strategy_allocations": {},
                "instrument_buckets": ["core"],
            }
        },
        "environments": {
            "paper": {
                "exchange": "binance",
                "environment": "paper",
                "keychain_key": "paper",
                "data_cache_path": str(path.parent / "data"),
                "risk_profile": "conservative",
                "alert_channels": [],
            }
        },
        "decision_engine": {
            "orchestrator": {
                "max_cost_bps": 12.0,
                "min_net_edge_bps": 2.0,
                "max_daily_loss_pct": 0.05,
                "max_drawdown_pct": 0.12,
                "max_position_ratio": 0.4,
                "max_open_positions": 5,
                "max_latency_ms": 250.0,
            },
            "min_probability": 0.0,
            "require_cost_data": False,
            "penalty_cost_bps": 0.0,
        },
        "observability": {
            "slo": {
                "latency": {
                    "name": "latency",
                    "metric": "scheduler_latency_ms",
                    "objective": 150.0,
                    "comparator": "<=",
                    "window_minutes": 1440.0,
                    "aggregation": "average",
                    "label_filters": {"profile": "conservative"},
                    "min_samples": 1,
                }
            },
            "key_rotation": {
                "registry_path": str(rotation_registry),
                "audit_directory": str(rotation_audit),
                "default_interval_days": 30.0,
                "default_warn_within_days": 5.0,
                "entries": [
                    {
                        "key": "oem_bundle",
                        "purpose": "signing",
                        "interval_days": 30.0,
                        "warn_within_days": 5.0,
                    }
                ],
            },
        },
    }
    path.write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")


def test_run_oem_acceptance_end_to_end(tmp_path: Path) -> None:
    daemon_artifact = tmp_path / "daemon" / "botd"
    daemon_artifact.parent.mkdir(parents=True, exist_ok=True)
    daemon_artifact.write_text("binary", encoding="utf-8")

    ui_artifact = tmp_path / "ui" / "app"
    ui_artifact.parent.mkdir(parents=True, exist_ok=True)
    ui_artifact.write_text("qt", encoding="utf-8")

    core_config = tmp_path / "config" / "core.yaml"
    _write_core_config(core_config)

    bundle_key = tmp_path / "keys" / "bundle.key"
    license_key = tmp_path / "keys" / "license.key"
    decision_key = tmp_path / "keys" / "decision.key"
    tco_key = tmp_path / "keys" / "tco.key"
    slo_key = tmp_path / "keys" / "slo.key"
    observability_key = tmp_path / "keys" / "observability.key"
    orchestrator_key = tmp_path / "keys" / "orchestrator.key"

    _write_signing_key(bundle_key)
    _write_signing_key(license_key)
    _write_signing_key(decision_key)
    _write_signing_key(tco_key)
    _write_signing_key(slo_key)
    _write_signing_key(observability_key)
    _write_signing_key(orchestrator_key)

    summary_path = tmp_path / "summary" / "acceptance.json"
    artifact_root = tmp_path / "artifacts"
    bundle_output = tmp_path / "dist"
    license_registry = tmp_path / "licenses" / "registry.jsonl"
    risk_output = tmp_path / "reports"
    mtls_output = tmp_path / "mtls"
    tco_output = tmp_path / "reports" / "tco"
    slo_output = tmp_path / "reports" / "slo"
    observability_output = tmp_path / "observability"
    decision_output = tmp_path / "decision" / "decision_report.json"

    decision_log_path = tmp_path / "audit" / "decision_log.jsonl"

    tco_fills_path = tmp_path / "data" / "fills.jsonl"
    tco_fills_path.parent.mkdir(parents=True, exist_ok=True)
    tco_fills_path.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "timestamp": "2024-03-01T12:00:00Z",
                        "strategy": "mean_reversion",
                        "risk_profile": "conservative",
                        "instrument": "BTC/USDT",
                        "exchange": "binance",
                        "side": "buy",
                        "quantity": 0.25,
                        "price": 20000,
                        "commission": 2.5,
                        "slippage": 1.0,
                    }
                ),
                json.dumps(
                    {
                        "timestamp": "2024-03-01T12:05:00Z",
                        "strategy": "volatility_target",
                        "risk_profile": "conservative",
                        "instrument": "ETH/USDT",
                        "exchange": "kraken",
                        "side": "sell",
                        "quantity": 1.5,
                        "price": 3000,
                        "commission": 3.0,
                        "slippage": 0.5,
                        "funding": 0.1,
                    }
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    risk_snapshot_path = tmp_path / "data" / "risk_snapshot.json"
    risk_snapshot_path.write_text(
        json.dumps(
            {
                "conservative": {
                    "start_of_day_equity": 100_000.0,
                    "last_equity": 101_000.0,
                    "peak_equity": 101_000.0,
                    "daily_realized_pnl": 500.0,
                    "positions": {
                        "BTC/USDT": {"notional": 10_000.0},
                    },
                }
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    decision_candidates_path = tmp_path / "data" / "decision_candidates.json"
    decision_candidates_path.write_text(
        json.dumps(
            {
                "candidates": [
                    {
                        "strategy": "mean_reversion",
                        "action": "enter",
                        "risk_profile": "conservative",
                        "symbol": "BTC/USDT",
                        "notional": 5_000.0,
                        "expected_return_bps": 8.0,
                        "expected_probability": 0.9,
                        "latency_ms": 120.0,
                    }
                ]
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    slo_metrics_path = tmp_path / "data" / "slo_metrics.jsonl"
    slo_metrics_path.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "metric": "scheduler_latency_ms",
                        "value": 120.0,
                        "timestamp": "2024-03-01T12:00:00Z",
                        "labels": {"profile": "conservative"},
                    }
                ),
                json.dumps(
                    {
                        "metric": "scheduler_latency_ms",
                        "value": 100.0,
                        "timestamp": "2024-03-01T13:00:00Z",
                        "labels": {"profile": "conservative"},
                    }
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    decision_output.parent.mkdir(parents=True, exist_ok=True)
    observability_output.mkdir(parents=True, exist_ok=True)
    rotation_registry = core_config.parent / "rotation_registry.json"
    rotation_registry.write_text("{}\n", encoding="utf-8")

    exit_code = run_acceptance(
        [
            "--bundle-platform",
            "linux",
            "--bundle-version",
            "1.2.3",
            "--bundle-signing-key",
            str(bundle_key),
            "--bundle-daemon",
            str(daemon_artifact),
            "--bundle-ui",
            str(ui_artifact),
            "--bundle-config",
            f"core.yaml={core_config}",
            "--bundle-output-dir",
            str(bundle_output),
            "--bundle-fingerprint-placeholder",
            "PLACEHOLDER-FP",
            "--license-signing-key",
            str(license_key),
            "--license-fingerprint",
            "ABCDEF123456",
            "--license-registry",
            str(license_registry),
            "--license-bundle-version",
            "1.2.3",
            "--license-valid-days",
            "30",
            "--license-feature",
            "paper",
            "--risk-config",
            str(core_config),
            "--risk-environment",
            "paper",
            "--risk-output-dir",
            str(risk_output),
            "--risk-json-name",
            "report.json",
            "--risk-pdf-name",
            "report.pdf",
            "--mtls-output-dir",
            str(mtls_output),
            "--mtls-bundle-name",
            "core-oem",
            "--tco-fill",
            str(tco_fills_path),
            "--tco-output-dir",
            str(tco_output),
            "--tco-basename",
            "oem_tco",
            "--tco-signing-key",
            str(tco_key),
            "--tco-signing-key-id",
            "tco-test",
            "--decision-config",
            str(core_config),
            "--decision-risk-snapshot",
            str(risk_snapshot_path),
            "--decision-candidates",
            str(decision_candidates_path),
            "--decision-output",
            str(decision_output),
            "--decision-tco-report",
            str(tco_output / "oem_tco.json"),
            "--decision-allow-empty",
            "--decision-signing-key-file",
            str(orchestrator_key),
            "--decision-signing-key-id",
            "decision-test",
            "--slo-config",
            str(core_config),
            "--slo-metric",
            str(slo_metrics_path),
            "--slo-output-dir",
            str(slo_output),
            "--slo-basename",
            "slo_oem",
            "--slo-signing-key",
            str(slo_key),
            "--slo-signing-key-id",
            "slo-test",
            "--rotation-config",
            str(core_config),
            "--rotation-output-dir",
            str(core_config.parent / "rotation_audit"),
            "--rotation-basename",
            "rotation_plan_oem",
            "--observability-version",
            "1.2.3",
            "--observability-output-dir",
            str(observability_output),
            "--observability-signing-key",
            str(observability_key),
            "--observability-key-id",
            "obs-test",
            "--summary-path",
            str(summary_path),
            "--artifact-root",
            str(artifact_root),
            "--decision-log-path",
            str(decision_log_path),
            "--decision-log-hmac-key-file",
            str(decision_key),
            "--decision-log-key-id",
            "oem-1",
            "--decision-log-category",
            "release.oem.acceptance",
            "--decision-log-notes",
            "Dry-run release 2024-Phase2",
        ]
    )

    assert exit_code == 0

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    expected_steps = {"bundle", "license", "risk", "mtls", "tco", "decision", "slo", "rotation", "observability"}
    assert {entry["step"] for entry in summary} == expected_steps
    assert all(entry["status"] == "ok" for entry in summary)

    bundle_details = next(item for item in summary if item["step"] == "bundle")
    assert Path(bundle_details["details"]["archive"]).exists()

    license_details = next(item for item in summary if item["step"] == "license")
    registry_path = Path(license_details["details"]["registry"])
    assert registry_path.exists()
    registry_line = registry_path.read_text(encoding="utf-8").strip()
    assert "ABCDEF123456" in registry_line

    risk_details = next(item for item in summary if item["step"] == "risk")
    risk_json = Path(risk_details["details"]["json_report"])
    risk_pdf = Path(risk_details["details"]["pdf_report"])
    assert risk_json.exists()
    assert risk_pdf.exists()

    mtls_details = next(item for item in summary if item["step"] == "mtls")
    metadata_path = Path(mtls_details["details"]["metadata"])
    assert metadata_path.exists()
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    assert metadata["bundle"] == "core-oem"
    assert Path(mtls_details["details"]["ca_certificate"]).exists()
    assert Path(mtls_details["details"]["server_certificate"]).exists()
    assert Path(mtls_details["details"]["client_certificate"]).exists()

    tco_details = next(item for item in summary if item["step"] == "tco")
    tco_csv_path = Path(tco_details["details"]["csv"])
    tco_pdf_path = Path(tco_details["details"]["pdf"])
    tco_json_path = Path(tco_details["details"]["json"])
    assert tco_csv_path.exists()
    assert tco_pdf_path.exists()
    assert tco_json_path.exists()

    decision_details = next(item for item in summary if item["step"] == "decision")
    decision_report_path = Path(decision_details["details"]["report"])
    assert decision_report_path.exists()

    slo_details = next(item for item in summary if item["step"] == "slo")
    slo_report_path = Path(slo_details["details"]["report"])
    assert slo_report_path.exists()

    rotation_details = next(item for item in summary if item["step"] == "rotation")
    rotation_plan_path = Path(rotation_details["details"]["plan"])
    assert rotation_plan_path.exists()

    observability_details = next(item for item in summary if item["step"] == "observability")
    observability_archive_path = Path(observability_details["details"]["archive"])
    assert observability_archive_path.exists()

    decision_log_lines = decision_log_path.read_text(encoding="utf-8").splitlines()
    assert len(decision_log_lines) == 1
    decision_entry = json.loads(decision_log_lines[0])
    assert decision_entry["status"] == "ok"
    assert decision_entry["category"] == "release.oem.acceptance"
    assert decision_entry["context"]["bundle_version"] == "1.2.3"
    assert decision_entry["notes"] == "Dry-run release 2024-Phase2"

    signature = decision_entry.get("signature")
    assert signature is not None
    assert signature["algorithm"] == "HMAC-SHA256"
    assert signature["key_id"] == "oem-1"

    key_bytes = decision_key.read_bytes().strip()
    entry_copy = dict(decision_entry)
    entry_copy.pop("signature", None)
    canonical = json.dumps(entry_copy, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    expected = base64.b64encode(hmac.new(key_bytes, canonical, hashlib.sha256).digest()).decode("ascii")
    assert signature["value"] == expected

    artifact_runs = list(artifact_root.iterdir())
    assert len(artifact_runs) == 1
    acceptance_dir = artifact_runs[0]
    metadata_path = acceptance_dir / "metadata.json"
    summary_artifact = acceptance_dir / "summary.json"
    assert metadata_path.exists()
    assert summary_artifact.exists()

    copied_summary = json.loads(summary_artifact.read_text(encoding="utf-8"))
    assert {entry["step"] for entry in copied_summary} == expected_steps

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    assert metadata["exit_code"] == 0
    assert metadata["bundle"]["archive"].endswith(Path(bundle_details["details"]["archive"]).name)

    bundle_dir = acceptance_dir / "bundle"
    assert (bundle_dir / Path(bundle_details["details"]["archive"]).name).exists()
    manifest_copy = bundle_dir / "manifest.json"
    assert manifest_copy.exists()
    signatures = list((bundle_dir / "signatures").rglob("*.sig"))
    assert signatures

    license_dir = acceptance_dir / "license"
    assert (license_dir / Path(license_details["details"]["registry"]).name).exists()

    risk_dir = acceptance_dir / "paper_labs"
    assert (risk_dir / Path(risk_details["details"]["json_report"]).name).exists()
    assert (risk_dir / Path(risk_details["details"]["pdf_report"]).name).exists()

    mtls_dir = acceptance_dir / "mtls"
    for key, value in mtls_details["details"].items():
        assert (mtls_dir / Path(value).name).exists()

    tco_dir = acceptance_dir / "tco"
    for key, value in tco_details["details"].items():
        assert (tco_dir / Path(value).name).exists()

    decision_dir = acceptance_dir / "decision"
    for key, value in decision_details["details"].items():
        assert (decision_dir / Path(value).name).exists()

    slo_dir = acceptance_dir / "slo"
    for key, value in slo_details["details"].items():
        assert (slo_dir / Path(value).name).exists()

    rotation_dir = acceptance_dir / "rotation"
    assert (rotation_dir / rotation_plan_path.name).exists()

    observability_dir = acceptance_dir / "observability"
    assert (observability_dir / observability_archive_path.name).exists()

    decision_dir = acceptance_dir / "decision_log"
    assert (decision_dir / "entry.json").exists()
    assert (decision_dir / decision_log_path.name).exists()
    tco_details = next(item for item in summary if item["step"] == "tco")
    tco_csv = Path(tco_details["details"]["csv"])
    tco_pdf = Path(tco_details["details"]["pdf"])
    tco_json = Path(tco_details["details"]["json"])
    assert tco_csv.exists()
    assert tco_pdf.exists()
    assert tco_json.exists()

    decision_details = next(item for item in summary if item["step"] == "decision")
    decision_report = Path(decision_details["details"]["report"])
    assert decision_report.exists()

    slo_details = next(item for item in summary if item["step"] == "slo")
    slo_report = Path(slo_details["details"]["report"])
    assert slo_report.exists()

    rotation_details = next(item for item in summary if item["step"] == "rotation")
    rotation_plan = Path(rotation_details["details"]["plan"])
    assert rotation_plan.exists()

    observability_details = next(item for item in summary if item["step"] == "observability")
    observability_archive = Path(observability_details["details"]["archive"])
    assert observability_archive.exists()
