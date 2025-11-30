from datetime import datetime, timezone

import json

import scripts.aggregate_licensing_drift_reports as agg


def test_generates_empty_outputs_when_matrix_missing(tmp_path):
    input_dir = tmp_path / "reports"
    dashboard_dir = input_dir / "dashboard"
    input_dir.mkdir()

    pytest_log = input_dir / "pytest.log"
    pytest_log.write_text("=== 1 passed, 0 failed in 0.10 seconds ===\n", encoding="utf-8")

    exit_code = agg.main(
        [
            "--input-dir",
            str(input_dir),
            "--dashboard-dir",
            str(dashboard_dir),
            "--run-id",
            "testrun",
        ]
    )

    assert exit_code == 0

    summary_path = input_dir / "licensing_drift_summary.json"
    assert summary_path.exists()

    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    assert payload["totals"] == {"match": 0, "degraded": 0, "rebind_required": 0, "unknown": 0}
    assert payload["records"] == []
    assert payload["pytest"]["status"] == "passed"
    assert "Brak pliku" in payload["pytest"].get("compatibility", "")
    assert any("Brak pliku" in diag for diag in payload.get("diagnostics", []))

    prom_path = dashboard_dir / "licensing_drift.prom"
    assert prom_path.exists()
    prom = prom_path.read_text(encoding="utf-8")
    assert "licensing_drift_rejections_total 0" in prom
    assert "licensing_drift_degraded_total 0" in prom


def test_collects_records_and_metrics(tmp_path):
    input_dir = tmp_path / "reports"
    dashboard_dir = input_dir / "dashboard"
    input_dir.mkdir()

    generated_at = datetime(2024, 5, 4, 12, 0, tzinfo=timezone.utc)
    matrix = {
        "generated_at": generated_at.isoformat(),
        "baseline": {"status": "match"},
        "scenarios": [
            {"name": "disk", "status": "degraded", "changed_components": ["disk"], "tolerated": ["disk"]},
            {"name": "cpu", "status": "rebind_required", "blocked": ["cpu"]},
        ],
    }
    compatibility = input_dir / "compatibility.json"
    compatibility.write_text(json.dumps(matrix), encoding="utf-8")

    pytest_log = input_dir / "pytest.log"
    pytest_log.write_text("=== 2 passed, 0 failed in 0.10 seconds ===\n", encoding="utf-8")

    exit_code = agg.main(
        [
            "--input-dir",
            str(input_dir),
            "--dashboard-dir",
            str(dashboard_dir),
            "--run-id",
            "run-42",
        ]
    )

    assert exit_code == 0

    summary = json.loads((input_dir / "licensing_drift_summary.json").read_text(encoding="utf-8"))
    assert summary["totals"] == {"match": 1, "degraded": 1, "rebind_required": 1, "unknown": 0}
    assert summary["records"][0]["scenario"] == "baseline"
    assert summary["records"][1]["scenario"] == "disk"
    assert summary["records"][2]["scenario"] == "cpu"
    assert summary["records"][2]["status"] == "rebind_required"
    assert summary.get("diagnostics") == []

    csv_content = (input_dir / "licensing_drift_summary.csv").read_text(encoding="utf-8")
    assert "run-42" in csv_content
    assert "disk,degraded" in csv_content.replace(",,", ",")

    prom = (dashboard_dir / "licensing_drift.prom").read_text(encoding="utf-8")
    assert 'licensing_drift_status{scenario="cpu",status="rebind_required"} 1' in prom
    assert 'licensing_drift_rejections_total 1' in prom
    assert f'licensing_drift_run_timestamp{{run_id="run-42"}} {int(generated_at.timestamp())}' in prom

    # Kopie dla dashboardu powinny istnieć
    assert (dashboard_dir / "licensing_drift_summary.json").exists()
    assert (dashboard_dir / "licensing_drift_summary.csv").exists()


def test_handles_invalid_json_matrix(tmp_path):
    input_dir = tmp_path / "reports"
    dashboard_dir = input_dir / "dashboard"
    input_dir.mkdir()

    # Zapisujemy uszkodzony plik, aby zweryfikować ścieżkę awaryjną
    compatibility = input_dir / "compatibility.json"
    compatibility.write_text("{not-json", encoding="utf-8")

    pytest_log = input_dir / "pytest.log"
    pytest_log.write_text("=== 1 passed, 0 failed in 0.10 seconds ===\n", encoding="utf-8")

    exit_code = agg.main(
        [
            "--input-dir",
            str(input_dir),
            "--dashboard-dir",
            str(dashboard_dir),
            "--run-id",
            "broken-json",
        ]
    )

    assert exit_code == 0

    summary_path = input_dir / "licensing_drift_summary.json"
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    assert payload["totals"] == {"match": 0, "degraded": 0, "rebind_required": 0, "unknown": 0}
    assert payload["records"] == []
    assert payload["pytest"]["status"] == "passed"
    assert "Niepoprawny JSON" in payload["pytest"].get("compatibility", "")
    assert any("Niepoprawny JSON" in diag for diag in payload.get("diagnostics", []))

    prom = (dashboard_dir / "licensing_drift.prom").read_text(encoding="utf-8")
    assert "licensing_drift_rejections_total 0" in prom
    assert "licensing_drift_degraded_total 0" in prom


def test_collects_diagnostics_when_pytest_missing(tmp_path):
    input_dir = tmp_path / "reports"
    dashboard_dir = input_dir / "dashboard"
    input_dir.mkdir()

    # Brak logu pytest i brak compatibility.json
    exit_code = agg.main(
        [
            "--input-dir",
            str(input_dir),
            "--dashboard-dir",
            str(dashboard_dir),
            "--run-id",
            "missing-pytest",
        ]
    )

    assert exit_code == 0

    payload = json.loads((input_dir / "licensing_drift_summary.json").read_text(encoding="utf-8"))
    diagnostics = payload.get("diagnostics", [])
    assert any("Brak logu pytest" in diag for diag in diagnostics)
    assert any("compatibility" in diag for diag in diagnostics)
