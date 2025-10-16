from __future__ import annotations

import json
from pathlib import Path

import pytest

from bot_core.resilience.audit import BundleAuditResult
from bot_core.resilience.drill import (
    FailoverDrillPlan,
    evaluate_failover_drill,
    load_failover_plan,
    write_summary_csv,
    write_summary_json,
    write_summary_signature,
)


@pytest.fixture()
def sample_manifest() -> dict[str, object]:
    return {
        "files": [
            {"path": "runbooks/scheduler.md"},
            {"path": "configs/fallback.env"},
            {"path": "sql/backups/latest.sql"},
        ]
    }


def _build_plan(**overrides: object) -> FailoverDrillPlan:
    base = {
        "drill_name": "stage6-drill",
        "executed_at": "2024-01-01T00:00:00Z",
        "metadata": {"owner": "resilience"},
        "services": [
            {
                "name": "scheduler",
                "max_rto_minutes": 15,
                "max_rpo_minutes": 5,
                "observed_rto_minutes": 10,
                "observed_rpo_minutes": 3,
                "required_artifacts": ["runbooks/*.md"],
                "metadata": {"tier": "critical"},
            }
        ],
    }
    base.update(overrides)
    return FailoverDrillPlan.from_mapping(base)


def test_load_failover_plan(tmp_path: Path) -> None:
    path = tmp_path / "plan.json"
    path.write_text(
        json.dumps(
            {
                "drill_name": "demo",
                "services": [
                    {
                        "name": "scheduler",
                        "max_rto_minutes": 5,
                        "max_rpo_minutes": 2,
                        "observed_rto_minutes": 4,
                        "observed_rpo_minutes": 1,
                        "required_artifacts": ["runbooks/*.md"],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    plan = load_failover_plan(path)
    assert plan.drill_name == "demo"
    assert len(plan.services) == 1
    assert plan.services[0].required_artifacts == ("runbooks/*.md",)


def test_evaluate_failover_success(sample_manifest: dict[str, object]) -> None:
    plan = _build_plan()
    summary = evaluate_failover_drill(plan, sample_manifest)
    assert summary.status == "ok"
    assert summary.counts == {"total": 1, "ok": 1, "warning": 0, "failed": 0}
    service = summary.services[0]
    assert service.status == "ok"
    assert not service.missing_artifacts
    assert "runbooks/scheduler.md" in service.matched_artifacts


def test_evaluate_failover_detects_gaps(sample_manifest: dict[str, object]) -> None:
    plan = _build_plan(
        services=[
            {
                "name": "scheduler",
                "max_rto_minutes": 8,
                "max_rpo_minutes": 5,
                "observed_rto_minutes": 12,
                "observed_rpo_minutes": None,
                "required_artifacts": ["runbooks/*.md", "docs/drill.pdf"],
            }
        ]
    )
    summary = evaluate_failover_drill(plan, sample_manifest)
    service = summary.services[0]
    assert service.status == "failed"
    assert "docs/drill.pdf" in service.missing_artifacts
    assert any("RTO 12" in issue for issue in service.issues)
    assert any("Brak pomiaru RPO" in issue for issue in service.issues)
    assert summary.status == "failed"


def test_bundle_audit_influences_summary(sample_manifest: dict[str, object]) -> None:
    plan = _build_plan()
    bundle_audit = BundleAuditResult(
        bundle_path=Path("/tmp/bundle.zip"),
        manifest_path=Path("/tmp/bundle.manifest.json"),
        signature_path=None,
        manifest=sample_manifest,
        errors=(),
        warnings=("policy warning",),
        verified_at="2024-01-01T00:00:00Z",
    )
    summary = evaluate_failover_drill(plan, sample_manifest, bundle_audit=bundle_audit)
    assert summary.status == "warning"
    assert summary.bundle_audit is not None
    assert summary.bundle_audit["warnings"] == ["policy warning"]


def test_summary_serialisation(tmp_path: Path, sample_manifest: dict[str, object]) -> None:
    plan = _build_plan()
    summary = evaluate_failover_drill(plan, sample_manifest)
    json_payload = write_summary_json(summary, tmp_path / "summary.json")
    assert json_payload["schema"] == "stage6.resilience.failover_drill.summary"
    write_summary_csv(summary, tmp_path / "summary.csv")
    signature = write_summary_signature(
        json_payload,
        tmp_path / "summary.sig",
        key=b"secret",
        key_id="stage6",
        target="summary.json",
    )
    assert signature["schema"] == "stage6.resilience.failover_drill.summary.signature"
    assert signature["signature"]["key_id"] == "stage6"
