from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

# Upewnij się, że projekt jest na ścieżce importów, aby importy skryptów i modułów działały zarówno
# w bezpośrednich wywołaniach, jak i w podprocesach z ustawionym PYTHONPATH=".".
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bot_core.resilience.bundle import ResilienceBundleBuilder  # noqa: E402
from scripts.failover_drill import main as run_failover_drill  # noqa: E402


def test_failover_drill_cli(tmp_path: Path) -> None:
    # Przygotuj źródło artefaktów do paczki resilience
    source = tmp_path / "source"
    (source / "runbooks").mkdir(parents=True)
    (source / "runbooks" / "scheduler.md").write_text("instrukcje", encoding="utf-8")
    (source / "sql").mkdir()
    (source / "sql" / "backup.sql").write_text("SELECT 1;", encoding="utf-8")

    # Zbuduj paczkę z artefaktami
    builder = ResilienceBundleBuilder(source, include=("**",))
    artifacts = builder.build(bundle_name="stage6", output_dir=tmp_path / "bundles")

    # Zbuduj plan ćwiczenia
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

    # Klucz HMAC do podpisu raportu
    key_path = tmp_path / "hmac.key"
    key_path.write_bytes(b"super-secret-key")

    summary_path = tmp_path / "summary.json"
    csv_path = tmp_path / "summary.csv"
    signature_path = tmp_path / "summary.sig"

    # Uruchom skrypt w podprocesie (symulacja rzeczywistego wywołania CLI)
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

    # Weryfikuj podsumowanie
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert summary["status"] == "ok"
    assert summary["bundle_audit"]["status"] == "ok"

    # Weryfikuj podpis JSON
    signature_doc = json.loads(signature_path.read_text(encoding="utf-8"))
    assert signature_doc["schema"] == "stage6.resilience.failover_drill.summary.signature"
    assert signature_doc["signature"]["algorithm"] == "HMAC-SHA256"

    # CSV z wynikami usług
    csv_content = csv_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(csv_content) == 2
    assert csv_content[0].startswith("service,status")


def test_failover_drill_cli_with_self_heal(tmp_path: Path) -> None:
    # Przygotuj źródło artefaktów do paczki resilience
    source = tmp_path / "source"
    (source / "runbooks").mkdir(parents=True)
    (source / "runbooks" / "scheduler.md").write_text("instrukcje", encoding="utf-8")
    (source / "sql").mkdir()
    (source / "sql" / "backup.sql").write_text("SELECT 1;", encoding="utf-8")

    # Zbuduj paczkę z artefaktami
    builder = ResilienceBundleBuilder(source, include=("**",))
    artifacts = builder.build(bundle_name="stage6", output_dir=tmp_path / "bundles")

    # Plan z naruszeniem RTO/RPO aby wyzwolić self-heal
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

    # Konfiguracja self-healing
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

    # Klucze HMAC do podpisów
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

    # Weryfikuj raport self-healing
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["schema"] == "stage6.resilience.self_healing.report"
    assert report["mode"] == "execute"
    assert report["status"] == "success"
    assert report["actions"][0]["status"] == "success"
    assert "restart scheduler" in (report["actions"][0]["output"] or "")

    signature_doc = json.loads(self_heal_signature.read_text(encoding="utf-8"))
    assert signature_doc["schema"] == "stage6.resilience.self_healing.report.signature"


@pytest.mark.parametrize("fail_on_breach", [False, True])
def test_failover_drill_cli_config_mode(
    tmp_path: Path, fail_on_breach: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    """
    Wariant testu z wersji 'main' — uruchamia funkcję main skryptu failover_drill
    korzystając z konfiguracji core.yaml i mechanizmu require_success/fail_on_breach.
    """
    output_path = tmp_path / ("resilience_fail.json" if fail_on_breach else "resilience_ok.json")
    # Klucz HMAC przez ENV (zgodnie z opcjami skryptu)
    monkeypatch.setenv("STAGE6_RESILIENCE_SIGNING_KEY", "unit-test-secret")

    argv = [
        "--config",
        "config/core.yaml",
        "--output",
        str(output_path),
    ]
    if fail_on_breach:
        argv.append("--fail-on-breach")

    exit_code = run_failover_drill(argv)
    assert exit_code == 0
    assert output_path.exists()
