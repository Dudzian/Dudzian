from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pytest

from bot_core.compliance.training import TrainingSession, write_training_log
from bot_core.reporting.tco import (
    TcoCostItem,
    aggregate_costs,
    write_summary_json,
    write_summary_signature,
)
from bot_core.runtime.stage5_hypercare import (
    Stage5ComplianceConfig,
    Stage5DecisionEngineConfig,
    Stage5HypercareConfig,
    Stage5HypercareCycle,
    Stage5OemAcceptanceConfig,
    Stage5RotationConfig,
    Stage5SloConfig,
    Stage5TcoConfig,
    Stage5TrainingConfig,
)
from bot_core.security.rotation import RotationStatus
from bot_core.security.rotation_report import RotationRecord, RotationSummary, write_rotation_summary
from bot_core.security.signing import build_hmac_signature, verify_hmac_signature


def _write_tco(tmp_path: Path) -> tuple[Path, Path, bytes]:
    items = [
        TcoCostItem(name="Exchange Fees", category="operations", monthly_cost=120.0),
        TcoCostItem(name="Power", category="infrastructure", monthly_cost=45.0),
    ]
    summary = aggregate_costs(items)
    summary_path = tmp_path / "tco.json"
    signature_path = tmp_path / "tco.signature.json"
    payload = write_summary_json(
        summary,
        summary_path,
        generated_at=datetime(2024, 1, 1, tzinfo=timezone.utc),
        metadata={"tag": "unit"},
    )
    key = b"tco-secret"
    write_summary_signature(payload, signature_path, key=key, key_id="tco-key")
    return summary_path, signature_path, key


def _write_rotation(tmp_path: Path) -> tuple[Path, bytes]:
    executed_at = datetime(2024, 2, 1, tzinfo=timezone.utc)
    status = RotationStatus(
        key="stage5-key",
        purpose="trading",
        interval_days=90.0,
        last_rotated=datetime(2023, 12, 1, tzinfo=timezone.utc),
        days_since_rotation=62.0,
        due_in_days=28.0,
        is_due=False,
        is_overdue=False,
    )
    record = RotationRecord(
        environment="paper",
        key="stage5-key",
        purpose="trading",
        registry_path=tmp_path / "rotation_log.json",
        status_before=status,
        rotated_at=executed_at,
        interval_days=90.0,
        metadata={"exchange": "binance"},
    )
    summary = RotationSummary(operator="Ops", executed_at=executed_at, records=[record])
    summary_path = tmp_path / "rotation_summary.json"
    key = b"rotation-secret"
    write_rotation_summary(summary, output=summary_path, signing_key=key, signing_key_id="rot-key")
    return summary_path, key


def _write_compliance(tmp_path: Path) -> tuple[Path, bytes]:
    now = datetime(2024, 3, 1, tzinfo=timezone.utc).isoformat()
    report = {
        "report_id": "stage5-001",
        "report_type": "stage5_compliance",
        "generated_at": now,
        "controls": [
            {"control_id": "stage5.oem.dry_run", "status": "pass"},
            {"control_id": "stage5.rotations", "status": "warn", "description": "Monitor"},
        ],
    }
    key = b"compliance-secret"
    report["signature"] = build_hmac_signature(report, key=key, key_id="comp")
    path = tmp_path / "compliance.json"
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return path, key


def _write_training(tmp_path: Path) -> tuple[Path, bytes]:
    session = TrainingSession(
        session_id="S05-TRAIN-01",
        title="Stage5 onboarding",
        trainer="Alice Trainer",
        participants=["Alice", "Bob"],
        topics=["Hypercare", "Compliance"],
        occurred_at=datetime(2024, 2, 15, 10, 0, tzinfo=timezone.utc),
        duration_minutes=90,
        summary="Omówienie procedur hypercare Stage5",
    )
    output = tmp_path / "training.json"
    key = b"training-secret"
    write_training_log(session, output=output, signing_key=key, signing_key_id="train-key")
    return output, key


def _write_slo(tmp_path: Path) -> tuple[Path, Path, bytes]:
    payload = {
        "results": {"availability": {"status": "ok", "indicator": "availability"}},
        "summary": {
            "slo": {"status_counts": {"ok": 1, "breach": 0, "fail": 0, "warning": 0}},
            "composites": {"status_counts": {"ok": 1, "breach": 0}},
        },
    }
    report_path = tmp_path / "slo.json"
    report_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    signature_path = tmp_path / "slo.signature.json"
    key = b"slo-secret"
    signature_path.write_text(
        json.dumps(build_hmac_signature(payload, key=key, key_id="slo-key"), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return report_path, signature_path, key


def _write_oem(tmp_path: Path) -> tuple[Path, Path, bytes]:
    payload = [
        {"step": "bundle", "status": "ok", "details": {"archive": "bundle.tar.gz"}},
        {"step": "license", "status": "ok", "details": {"registry": "registry.jsonl"}},
        {"step": "risk", "status": "ok", "details": {"json_report": "risk.json"}},
        {"step": "mtls", "status": "ok", "details": {"metadata": "metadata.json"}},
    ]
    path = tmp_path / "oem_summary.json"
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    signature_path = tmp_path / "oem_summary.signature.json"
    key = b"oem-secret"
    signature_path.write_text(
        json.dumps(build_hmac_signature(payload, key=key, key_id="oem-key"), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return path, signature_path, key


def _write_decision_summary(tmp_path: Path) -> Path:
    payload = {
        "type": "decision_engine_summary",
        "generated_at": datetime(2024, 5, 1, tzinfo=timezone.utc).isoformat().replace("+00:00", "Z"),
        "total": 12,
        "accepted": 9,
        "rejected": 3,
        "acceptance_rate": 0.75,
        "history_limit": 512,
        "history_window": 12,
        "latest_model": "gbm_v5",
        "latest_thresholds": {"min_probability": 0.62, "max_cost_bps": 15.0},
        "latest_generated_at": "2024-05-01T12:00:00Z",
        "rejection_reasons": {"too_costly": 2, "risk": 1},
        "filters": {"environment": "paper"},
    }
    path = tmp_path / "decision_summary.json"
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def test_stage5_hypercare_cycle_builds_signed_summary(tmp_path: Path) -> None:
    tco_summary, tco_signature, tco_key = _write_tco(tmp_path)
    rotation_summary, rotation_key = _write_rotation(tmp_path)
    compliance_report, compliance_key = _write_compliance(tmp_path)
    training_log, training_key = _write_training(tmp_path)
    slo_report, slo_signature, slo_key = _write_slo(tmp_path)
    oem_summary, oem_signature, oem_key = _write_oem(tmp_path)
    decision_summary = _write_decision_summary(tmp_path)

    output_path = tmp_path / "hypercare_summary.json"
    signature_path = tmp_path / "hypercare_summary.signature.json"
    summary_key = b"hypercare-secret"

    config = Stage5HypercareConfig(
        output_path=output_path,
        signature_path=signature_path,
        signing_key=summary_key,
        signing_key_id="stage5-summary",
        tco=Stage5TcoConfig(
            summary_path=tco_summary,
            signature_path=tco_signature,
            signing_key=tco_key,
            require_signature=True,
        ),
        rotation=Stage5RotationConfig(
            summary_path=rotation_summary,
            signing_key=rotation_key,
            require_signature=True,
        ),
        compliance=Stage5ComplianceConfig(
            reports=[compliance_report],
            signing_key=compliance_key,
            require_signature=True,
        ),
        training=Stage5TrainingConfig(
            logs=[training_log],
            signing_key=training_key,
            require_signature=True,
        ),
        slo=Stage5SloConfig(
            report_path=slo_report,
            signature_path=slo_signature,
            signing_key=slo_key,
            require_signature=True,
        ),
        oem=Stage5OemAcceptanceConfig(
            summary_path=oem_summary,
            signature_path=oem_signature,
            signing_key=oem_key,
            require_signature=True,
        ),
        decision_engine=Stage5DecisionEngineConfig(
            summary_path=decision_summary,
            min_acceptance_rate=0.7,
            warn_acceptance_rate=0.72,
            min_observations=10,
        ),
    )

    cycle = Stage5HypercareCycle(config)
    result = cycle.run()

    assert result.output_path == output_path
    assert result.signature_path == signature_path

    payload = result.payload
    assert payload["overall_status"] == "ok"
    assert payload["issues"] == []

    artifacts = payload["artifacts"]
    assert artifacts["tco"]["status"] == "ok"
    assert artifacts["tco"]["details"]["signature"]["verified"] is True

    assert artifacts["key_rotation"]["status"] == "ok"
    assert artifacts["key_rotation"]["details"]["signature"]["verified"] is True

    assert artifacts["compliance"]["status"] == "ok"
    assert artifacts["training"]["status"] == "ok"
    assert artifacts["slo_monitor"]["status"] == "ok"
    assert artifacts["oem_acceptance"]["status"] == "ok"
    assert artifacts["decision_engine"]["status"] == "ok"
    assert artifacts["decision_engine"]["details"]["acceptance_rate"] == pytest.approx(0.75)
    assert artifacts["oem_acceptance"]["details"]["signature"]["verified"] is True

    signature = json.loads(signature_path.read_text(encoding="utf-8"))
    assert verify_hmac_signature(payload, signature, key=summary_key)


def test_oem_signature_required_but_missing(tmp_path: Path) -> None:
    oem_summary, oem_signature, oem_key = _write_oem(tmp_path)
    oem_signature.unlink()

    config = Stage5HypercareConfig(
        output_path=tmp_path / "out.json",
        oem=Stage5OemAcceptanceConfig(
            summary_path=oem_summary,
            signature_path=oem_signature,
            signing_key=oem_key,
            require_signature=True,
        ),
    )

    result = Stage5HypercareCycle(config).run()
    oem_artifact = result.payload["artifacts"]["oem_acceptance"]
    assert oem_artifact["status"] == "fail"
    assert any("Brak pliku z podpisem" in issue for issue in oem_artifact["issues"])


def test_oem_signature_invalid(tmp_path: Path) -> None:
    oem_summary, oem_signature, oem_key = _write_oem(tmp_path)
    oem_signature.write_text(
        json.dumps({"algorithm": "HMAC-SHA256", "value": "invalid"}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    config = Stage5HypercareConfig(
        output_path=tmp_path / "out.json",
        oem=Stage5OemAcceptanceConfig(
            summary_path=oem_summary,
            signature_path=oem_signature,
            signing_key=oem_key,
            require_signature=True,
        ),
    )

    result = Stage5HypercareCycle(config).run()
    oem_artifact = result.payload["artifacts"]["oem_acceptance"]
    assert oem_artifact["status"] == "fail"
    assert any("Podpis HMAC podsumowania OEM jest niepoprawny" in issue for issue in oem_artifact["issues"])


def test_stage5_decision_engine_thresholds(tmp_path: Path) -> None:
    payload = {
        "type": "decision_engine_summary",
        "generated_at": datetime(2024, 5, 2, tzinfo=timezone.utc).isoformat(),
        "total": 3,
        "accepted": 1,
        "rejected": 2,
        "acceptance_rate": 0.4,
        "history_limit": 128,
        "history_window": 3,
        "latest_model": None,
        "rejection_reasons": {"too_costly": 2},
    }
    decision_path = tmp_path / "decision_summary_bad.json"
    decision_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    output_path = tmp_path / "hypercare_summary.json"
    config = Stage5HypercareConfig(
        output_path=output_path,
        decision_engine=Stage5DecisionEngineConfig(
            summary_path=decision_path,
            min_acceptance_rate=0.6,
            warn_acceptance_rate=0.5,
            min_observations=5,
            require_latest_model=True,
            require_thresholds=True,
        ),
    )

    result = Stage5HypercareCycle(config).run()
    artifact = result.payload["artifacts"]["decision_engine"]
    assert artifact["status"] == "fail"
    assert any("Zbyt mało ewaluacji" in issue for issue in artifact["issues"])
    assert any("Akceptacja Decision Engine" in issue for issue in artifact["issues"])
    assert any("migawki progów" in warning for warning in artifact["warnings"])
