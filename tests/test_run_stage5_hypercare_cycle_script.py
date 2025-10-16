from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from bot_core.compliance.training import TrainingSession, write_training_log
from bot_core.reporting.tco import (
    TcoCostItem,
    aggregate_costs,
    write_summary_json,
    write_summary_signature,
)
from bot_core.security.rotation import RotationStatus
from bot_core.security.rotation_report import RotationRecord, RotationSummary, write_rotation_summary
from bot_core.security.signing import build_hmac_signature
from scripts import run_stage5_hypercare_cycle


def _prepare_inputs(tmp_path: Path) -> dict[str, Path | str]:
    items = [
        TcoCostItem(name="Exchange Fees", category="operations", monthly_cost=100.0),
        TcoCostItem(name="Ops", category="operations", monthly_cost=20.0),
    ]
    summary = aggregate_costs(items)
    tco_path = tmp_path / "tco.json"
    tco_signature_path = tmp_path / "tco.signature.json"
    payload = write_summary_json(
        summary,
        tco_path,
        generated_at=datetime(2024, 1, 1, tzinfo=timezone.utc),
    )
    write_summary_signature(payload, tco_signature_path, key=b"tco-secret", key_id="tco")

    executed_at = datetime(2024, 2, 1, tzinfo=timezone.utc)
    status = RotationStatus(
        key="stage5-key",
        purpose="trading",
        interval_days=90.0,
        last_rotated=datetime(2023, 11, 1, tzinfo=timezone.utc),
        days_since_rotation=92.0,
        due_in_days=-2.0,
        is_due=True,
        is_overdue=True,
    )
    rotation_record = RotationRecord(
        environment="paper",
        key="stage5-key",
        purpose="trading",
        registry_path=tmp_path / "rotation_log.json",
        status_before=status,
        rotated_at=executed_at,
        interval_days=90.0,
        metadata={"exchange": "binance"},
    )
    rotation_summary = RotationSummary(operator="Ops", executed_at=executed_at, records=[rotation_record])
    rotation_path = tmp_path / "rotation_summary.json"
    write_rotation_summary(rotation_summary, output=rotation_path, signing_key=b"rotation-secret", signing_key_id="rot")

    now = datetime(2024, 3, 1, tzinfo=timezone.utc).isoformat()
    compliance = {
        "report_id": "stage5-001",
        "report_type": "stage5_compliance",
        "generated_at": now,
        "controls": [
            {"control_id": "stage5.oem.dry_run", "status": "pass"},
            {"control_id": "stage5.rotations", "status": "pass"},
        ],
    }
    compliance_key = b"compliance-secret"
    compliance["signature"] = build_hmac_signature(compliance, key=compliance_key, key_id="comp")
    compliance_path = tmp_path / "compliance.json"
    compliance_path.write_text(json.dumps(compliance, ensure_ascii=False, indent=2), encoding="utf-8")

    session = TrainingSession(
        session_id="TRAIN-01",
        title="Procedury Stage5",
        trainer="Alice",
        participants=["Alice", "Bob"],
        topics=["TCO"],
        occurred_at=datetime(2024, 2, 10, tzinfo=timezone.utc),
        duration_minutes=60,
        summary="Omówienie raportów hypercare",
    )
    training_path = tmp_path / "training.json"
    write_training_log(session, output=training_path, signing_key=b"training-secret", signing_key_id="train")

    slo_payload = {
        "results": {"availability": {"status": "ok"}},
        "summary": {
            "slo": {"status_counts": {"ok": 1, "breach": 0, "fail": 0, "warning": 0}},
            "composites": {"status_counts": {"ok": 1}},
        },
    }
    slo_report = tmp_path / "slo.json"
    slo_report.write_text(json.dumps(slo_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    slo_signature = tmp_path / "slo.signature.json"
    slo_signature.write_text(
        json.dumps(build_hmac_signature(slo_payload, key=b"slo-secret", key_id="slo"), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    oem_summary = [
        {"step": "bundle", "status": "ok", "details": {"archive": "bundle.tar.gz"}},
        {"step": "license", "status": "ok", "details": {"registry": "registry.jsonl"}},
        {"step": "risk", "status": "ok", "details": {"json_report": "risk.json"}},
        {"step": "mtls", "status": "ok", "details": {"metadata": "metadata.json"}},
    ]
    oem_path = tmp_path / "oem_summary.json"
    oem_path.write_text(json.dumps(oem_summary, ensure_ascii=False, indent=2), encoding="utf-8")
    oem_signature_path = tmp_path / "oem_summary.signature.json"
    oem_signature_path.write_text(
        json.dumps(build_hmac_signature(oem_summary, key=b"oem-secret", key_id="oem"), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    return {
        "tco": tco_path,
        "tco_signature": tco_signature_path,
        "rotation": rotation_path,
        "compliance": compliance_path,
        "training": training_path,
        "slo": slo_report,
        "slo_signature": slo_signature,
        "oem": oem_path,
        "oem_signature": oem_signature_path,
    }


def test_run_stage5_hypercare_cycle_creates_summary(tmp_path: Path, capsys: "CaptureFixture[str]") -> None:
    paths = _prepare_inputs(tmp_path)
    output_path = tmp_path / "summary.json"
    signature_path = tmp_path / "summary.signature.json"

    exit_code = run_stage5_hypercare_cycle.main(
        [
            "--tco-summary",
            str(paths["tco"]),
            "--tco-signature",
            str(paths["tco_signature"]),
            "--tco-signing-key",
            "tco-secret",
            "--tco-require-signature",
            "--rotation-summary",
            str(paths["rotation"]),
            "--rotation-signing-key",
            "rotation-secret",
            "--rotation-require-signature",
            "--compliance-report",
            str(paths["compliance"]),
            "--compliance-signing-key",
            "compliance-secret",
            "--compliance-require-signature",
            "--training-log",
            str(paths["training"]),
            "--training-signing-key",
            "training-secret",
            "--training-require-signature",
            "--slo-report",
            str(paths["slo"]),
            "--slo-signature",
            str(paths["slo_signature"]),
            "--slo-signing-key",
            "slo-secret",
            "--slo-require-signature",
            "--oem-summary",
            str(paths["oem"]),
            "--oem-signature",
            str(paths["oem_signature"]),
            "--oem-signing-key",
            "oem-secret",
            "--oem-require-signature",
            "--output",
            str(output_path),
            "--signature",
            str(signature_path),
            "--signing-key",
            "hypercare-secret",
            "--signing-key-id",
            "stage5",
        ]
    )

    assert exit_code == 0
    captured = capsys.readouterr()
    summary = json.loads(captured.out)
    assert summary["status"] == "ok"
    assert Path(summary["output"]).exists()
    assert Path(summary["signature"]).exists()

    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload["overall_status"] == "ok"
    assert payload["artifacts"]["tco"]["details"]["signature"]["verified"] is True

    signature = json.loads(signature_path.read_text(encoding="utf-8"))
    assert build_hmac_signature(payload, key=b"hypercare-secret", key_id="stage5")["value"] == signature["value"]
