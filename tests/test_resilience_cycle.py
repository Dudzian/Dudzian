from pathlib import Path

import json

from bot_core.resilience.hypercare import (
    AuditConfig,
    BundleConfig,
    FailoverConfig,
    ResilienceCycleConfig,
    ResilienceHypercareCycle,
    SelfHealingConfig,
)
from bot_core.resilience.policy import MetadataRequirement, PatternRequirement, ResiliencePolicy


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_resilience_cycle_end_to_end(tmp_path: Path) -> None:
    source_dir = tmp_path / "source"
    (source_dir / "artifacts").mkdir(parents=True)
    artifact = source_dir / "artifacts" / "snapshot.txt"
    artifact.write_text("stage6", encoding="utf-8")

    plan_payload = {
        "drill_name": "stage6-drill",
        "executed_at": "2024-01-01T00:00:00Z",
        "services": [
            {
                "name": "portfolio-governor",
                "max_rto_minutes": 5,
                "max_rpo_minutes": 10,
                "observed_rto_minutes": 3,
                "observed_rpo_minutes": 4,
                "required_artifacts": ["artifacts/snapshot.txt"],
                "metadata": {"owner": "noc"},
            }
        ],
        "metadata": {"cycle": "nightly"},
    }
    plan_path = tmp_path / "plan.json"
    _write_json(plan_path, plan_payload)

    self_heal_payload = {
        "rules": [
            {
                "name": "restart-governor",
                "service_pattern": "portfolio*",
                "statuses": ["ok", "warning", "failed"],
                "severity": "critical",
                "actions": [
                    {
                        "module": "bot_core.runtime", 
                        "command": ["echo", "restart"],
                        "delay_seconds": 0,
                        "tags": ["stage6"],
                    }
                ],
            }
        ]
    }
    self_heal_config_path = tmp_path / "self_heal.json"
    _write_json(self_heal_config_path, self_heal_payload)

    audit_dir = tmp_path / "audit"
    bundle_dir = tmp_path / "bundles"

    policy = ResiliencePolicy(
        pattern_requirements=(
            PatternRequirement(pattern="artifacts/*.txt", description="wymagane snapshoty"),
        ),
        metadata_requirements=(
            MetadataRequirement(key="audit_report", description="raport audytu", required=True),
        ),
    )

    cycle = ResilienceHypercareCycle(
        ResilienceCycleConfig(
            bundle=BundleConfig(
                source=source_dir,
                output_dir=bundle_dir,
                include=("**",),
                metadata={"stage": "6"},
            ),
            audit=AuditConfig(
                json_path=audit_dir / "audit_summary.json",
                csv_path=audit_dir / "audit_summary.csv",
                require_signature=False,
                policy=policy,
            ),
            failover=FailoverConfig(
                plan_path=plan_path,
                json_path=audit_dir / "failover_summary.json",
                csv_path=audit_dir / "failover_summary.csv",
            ),
            signing_key=b"supersecretkey12345",
            signing_key_id="stage6",
            self_healing=SelfHealingConfig(
                rules_path=self_heal_config_path,
                output_path=audit_dir / "self_heal.json",
            ),
        )
    )

    result = cycle.run()

    assert result.audit_result.is_successful()
    assert result.failover_summary.status == "ok"
    assert result.self_healing_payload is not None
    assert result.self_healing_payload.get("mode") == "plan"

    assert result.audit_summary_path.exists()
    assert result.failover_summary_path.exists()
    assert result.verification["signature_verified"] is True

    manifest_data = json.loads(result.bundle_artifacts.manifest_path.read_text(encoding="utf-8"))
    audit_metadata = manifest_data["metadata"]["audit_report"]
    assert audit_metadata["json"] == result.audit_summary_path.as_posix()
    assert audit_metadata["policy_enforced"] is True

    failover_metadata = manifest_data["metadata"]["failover_summary"]
    assert failover_metadata["plan"] == plan_path.as_posix()

    if result.self_healing_report_path:
        self_heal_metadata = manifest_data["metadata"]["self_healing"]
        assert self_heal_metadata["mode"] == "plan"
        assert self_heal_metadata["output"] == result.self_healing_report_path.as_posix()

    audit_report = json.loads(result.audit_summary_path.read_text(encoding="utf-8"))
    assert audit_report["ok"] == 1

    failover_report = json.loads(result.failover_summary_path.read_text(encoding="utf-8"))
    assert failover_report["status"] == "ok"
