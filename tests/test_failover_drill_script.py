from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

from bot_core.resilience.bundle import ResilienceBundleBuilder


def test_failover_drill_cli(tmp_path: Path) -> None:
    source = tmp_path / "source"
    (source / "runbooks").mkdir(parents=True)
    (source / "runbooks" / "scheduler.md").write_text("instrukcje", encoding="utf-8")
    (source / "sql").mkdir()
    (source / "sql" / "backup.sql").write_text("SELECT 1;", encoding="utf-8")

    builder = ResilienceBundleBuilder(source, include=("**",))
    artifacts = builder.build(bundle_name="stage6", output_dir=tmp_path / "bundles")

    plan_path = tmp_path / "plan.json"
    plan_path.write_text(
        json.dumps(
            {
                "drill_name": "integration-drill",
                "executed_at": "2024-05-01T12:00:00Z",
                "services": [
                    {
                        "name": "scheduler",
                        "max_rto_minutes": 20,
                        "max_rpo_minutes": 10,
                        "observed_rto_minutes": 12,
                        "observed_rpo_minutes": 4,
                        "required_artifacts": ["runbooks/*.md", "sql/*.sql"],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    key_path = tmp_path / "hmac.key"
    key_path.write_bytes(b"super-secret-key")

    summary_path = tmp_path / "summary.json"
    csv_path = tmp_path / "summary.csv"
    signature_path = tmp_path / "summary.sig"

    env = os.environ.copy()
    env["PYTHONPATH"] = "."

    result = subprocess.run(
        [
            sys.executable,
            "scripts/failover_drill.py",
            "--bundle",
            str(artifacts.bundle_path),
            "--plan",
            str(plan_path),
            "--output-json",
            str(summary_path),
            "--output-csv",
            str(csv_path),
            "--signing-key",
            str(key_path),
            "--signature-path",
            str(signature_path),
        ],
        check=True,
        env=env,
    )
    assert result.returncode == 0

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert summary["status"] == "ok"
    assert summary["bundle_audit"]["status"] == "ok"

    signature_doc = json.loads(signature_path.read_text(encoding="utf-8"))
    assert signature_doc["schema"] == "stage6.resilience.failover_drill.summary.signature"
    assert signature_doc["signature"]["algorithm"] == "HMAC-SHA256"

    csv_content = csv_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(csv_content) == 2
    assert csv_content[0].startswith("service,status")


def test_failover_drill_cli_with_self_heal(tmp_path: Path) -> None:
    source = tmp_path / "source"
    (source / "runbooks").mkdir(parents=True)
    (source / "runbooks" / "scheduler.md").write_text("instrukcje", encoding="utf-8")
    (source / "sql").mkdir()
    (source / "sql" / "backup.sql").write_text("SELECT 1;", encoding="utf-8")

    builder = ResilienceBundleBuilder(source, include=("**",))
    artifacts = builder.build(bundle_name="stage6", output_dir=tmp_path / "bundles")

    plan_path = tmp_path / "plan.json"
    plan_path.write_text(
        json.dumps(
            {
                "drill_name": "integration-drill",
                "executed_at": "2024-05-01T12:00:00Z",
                "services": [
                    {
                        "name": "scheduler",
                        "max_rto_minutes": 10,
                        "max_rpo_minutes": 5,
                        "observed_rto_minutes": 18,
                        "observed_rpo_minutes": 4,
                        "required_artifacts": ["runbooks/*.md", "sql/*.sql"],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    self_heal_config = tmp_path / "self_heal.json"
    self_heal_config.write_text(
        json.dumps(
            {
                "rules": [
                    {
                        "service_pattern": "scheduler",
                        "actions": [
                            {
                                "module": "runtime.scheduler",
                                "command": [
                                    sys.executable,
                                    "-c",
                                    "print('restart scheduler')",
                                ],
                            }
                        ],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    key_path = tmp_path / "hmac.key"
    key_path.write_bytes(b"super-secret-key")
    self_heal_key = tmp_path / "self_heal.key"
    self_heal_key.write_bytes(b"another-secret-key")

    summary_path = tmp_path / "summary.json"
    report_path = tmp_path / "self_heal_report.json"
    self_heal_signature = tmp_path / "self_heal_report.sig"

    env = os.environ.copy()
    env["PYTHONPATH"] = "."

    result = subprocess.run(
        [
            sys.executable,
            "scripts/failover_drill.py",
            "--bundle",
            str(artifacts.bundle_path),
            "--plan",
            str(plan_path),
            "--output-json",
            str(summary_path),
            "--signing-key",
            str(key_path),
            "--self-heal-config",
            str(self_heal_config),
            "--self-heal-mode",
            "execute",
            "--self-heal-output",
            str(report_path),
            "--self-heal-signing-key",
            str(self_heal_key),
            "--self-heal-signature-path",
            str(self_heal_signature),
        ],
        check=True,
        env=env,
    )
    assert result.returncode == 0

    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["schema"] == "stage6.resilience.self_healing.report"
    assert report["mode"] == "execute"
    assert report["status"] == "success"
    assert report["actions"][0]["status"] == "success"
    assert "restart scheduler" in (report["actions"][0]["output"] or "")

    signature_doc = json.loads(self_heal_signature.read_text(encoding="utf-8"))
    assert signature_doc["schema"] == "stage6.resilience.self_healing.report.signature"
