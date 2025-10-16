from pathlib import Path

import json

from scripts.run_stage6_resilience_cycle import run as run_cycle


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_run_stage6_resilience_cycle_script(tmp_path: Path) -> None:
    source_dir = tmp_path / "source"
    artifacts_dir = source_dir / "artifacts"
    artifacts_dir.mkdir(parents=True)
    (artifacts_dir / "snapshot.txt").write_text("ok", encoding="utf-8")

    plan_payload = {
        "drill_name": "stage6-cycle",
        "services": [
            {
                "name": "portfolio-governor",
                "max_rto_minutes": 10,
                "max_rpo_minutes": 15,
                "observed_rto_minutes": 5,
                "observed_rpo_minutes": 6,
                "required_artifacts": ["artifacts/snapshot.txt"],
            }
        ],
    }
    plan_path = tmp_path / "plan.json"
    _write_json(plan_path, plan_payload)

    policy_payload = {
        "required_patterns": [
            {"pattern": "artifacts/*.txt", "description": "snapshots"}
        ],
        "metadata": [
            {"key": "audit_report", "description": "audit metadata"}
        ],
    }
    policy_path = tmp_path / "policy.json"
    _write_json(policy_path, policy_payload)

    self_heal_payload = {
        "rules": [
            {
                "service_pattern": "portfolio*",
                "statuses": ["ok"],
                "actions": [
                    {
                        "module": "bot_core.runtime",
                        "command": ["echo", "restart"],
                    }
                ],
            }
        ]
    }
    self_heal_config = tmp_path / "self_heal.json"
    _write_json(self_heal_config, self_heal_payload)

    audit_json = tmp_path / "audit" / "audit.json"
    audit_csv = tmp_path / "audit" / "audit.csv"
    audit_sig = tmp_path / "audit" / "audit.json.sig"
    failover_json = tmp_path / "audit" / "failover.json"
    failover_csv = tmp_path / "audit" / "failover.csv"
    failover_sig = tmp_path / "audit" / "failover.json.sig"
    self_heal_output = tmp_path / "audit" / "self_heal_report.json"
    self_heal_sig = tmp_path / "audit" / "self_heal_report.json.sig"
    bundle_dir = tmp_path / "bundles"

    exit_code = run_cycle(
        [
            "--source",
            str(source_dir),
            "--plan",
            str(plan_path),
            "--bundle-output-dir",
            str(bundle_dir),
            "--audit-json",
            str(audit_json),
            "--audit-csv",
            str(audit_csv),
            "--audit-signature",
            str(audit_sig),
            "--audit-policy",
            str(policy_path),
            "--failover-json",
            str(failover_json),
            "--failover-csv",
            str(failover_csv),
            "--failover-signature",
            str(failover_sig),
            "--self-heal-config",
            str(self_heal_config),
            "--self-heal-output",
            str(self_heal_output),
            "--self-heal-signature",
            str(self_heal_sig),
            "--signing-key",
            "supersecretkey12345",
            "--signing-key-id",
            "stage6",
        ]
    )

    assert exit_code == 0
    assert audit_json.exists()
    assert failover_json.exists()
    assert self_heal_output.exists()

    manifest_files = list(bundle_dir.glob("*.manifest.json"))
    assert manifest_files
    manifest = json.loads(manifest_files[0].read_text(encoding="utf-8"))
    assert manifest["metadata"]["audit_report"]["json"] == audit_json.as_posix()
    assert manifest["metadata"]["failover_summary"]["json"] == failover_json.as_posix()
    assert manifest["metadata"]["self_healing"]["output"] == self_heal_output.as_posix()
