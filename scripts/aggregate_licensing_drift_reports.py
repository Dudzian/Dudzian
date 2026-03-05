"""Konsoliduje wyniki testów dryfu licencyjnego w spójne artefakty.

Skrypt jest wywoływany w workflow "Licensing drift consolidation" i musi
obsługiwać brakujące lub uszkodzone artefakty (np. gdy etap kompatybilności
zakończył się błędem). Dzięki temu kolejne kroki CI nie flakują, a diagnostyka
jest jednoznaczna.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import pandas as pd


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Konsoliduje artefakty dryfu licencji na potrzeby dashboardów."
    )
    parser.add_argument(
        "--input-dir",
        default="reports/ci/licensing_drift",
        help="Katalog z artefaktami licencyjnymi (domyślnie reports/ci/licensing_drift)",
    )
    parser.add_argument(
        "--compatibility",
        default=None,
        help="Ścieżka do compatibility.json (domyślnie <input-dir>/compatibility.json)",
    )
    parser.add_argument(
        "--pytest-log",
        default=None,
        help="Ścieżka logu pytest (domyślnie <input-dir>/pytest.log)",
    )
    parser.add_argument(
        "--json-output",
        default=None,
        help="Ścieżka pliku JSON z podsumowaniem (domyślnie <input-dir>/licensing_drift_summary.json)",
    )
    parser.add_argument(
        "--csv-output",
        default=None,
        help="Ścieżka pliku CSV z podsumowaniem (domyślnie <input-dir>/licensing_drift_summary.csv)",
    )
    parser.add_argument(
        "--parquet-output",
        default=None,
        help="Ścieżka pliku Parquet z podsumowaniem (domyślnie <input-dir>/licensing_drift_summary.parquet)",
    )
    parser.add_argument(
        "--dashboard-dir",
        default="reports/ci/licensing_drift/dashboard",
        help="Docelowy katalog dla dashboardów (domyślnie reports/ci/licensing_drift/dashboard)",
    )
    parser.add_argument(
        "--prom-output",
        default=None,
        help="Ścieżka pliku z metrykami Prometheus (domyślnie <dashboard-dir>/licensing_drift.prom)",
    )
    parser.add_argument(
        "--run-id",
        default=os.environ.get("GITHUB_RUN_ID", "manual"),
        help="Identyfikator uruchomienia CI doklejany do podsumowania (domyślnie zmienna GITHUB_RUN_ID lub 'manual')",
    )
    return parser.parse_args(argv)


def _load_matrix(path: Path) -> tuple[dict[str, Any] | None, str | None]:
    if not path.exists():
        return None, f"Brak pliku {path}"
    try:
        return json.loads(path.read_text()), None
    except json.JSONDecodeError as exc:
        return None, f"Niepoprawny JSON w {path}: {exc}"


def _parse_generated_at(matrix: Mapping[str, Any]) -> datetime:
    raw = matrix.get("generated_at")
    if isinstance(raw, str):
        try:
            return datetime.fromisoformat(raw.replace("Z", "+00:00"))
        except ValueError:
            pass
    return datetime.now(timezone.utc)


def _collect_records(
    matrix: Mapping[str, Any], run_id: str, generated_at: datetime
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []

    def _append(name: str, payload: Mapping[str, Any]) -> None:
        status = str(payload.get("status", "unknown"))
        records.append(
            {
                "run_id": run_id,
                "scenario": name,
                "status": status,
                "changed_components": list(payload.get("changed_components", []) or []),
                "tolerated": list(payload.get("tolerated", []) or []),
                "blocked": list(payload.get("blocked", []) or []),
                "generated_at": generated_at.isoformat(),
            }
        )

    baseline = matrix.get("baseline", {}) if isinstance(matrix, Mapping) else {}
    _append("baseline", baseline)

    scenarios = matrix.get("scenarios") if isinstance(matrix, Mapping) else None
    if isinstance(scenarios, list):
        for scenario in scenarios:
            name = (
                str(scenario.get("name", "scenario"))
                if isinstance(scenario, Mapping)
                else "scenario"
            )
            payload = scenario if isinstance(scenario, Mapping) else {}
            _append(name, payload)

    return records


def _parse_pytest_status(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"status": "missing", "summary": "Brak logu pytest"}

    summary = ""
    status = "unknown"
    for line in reversed(path.read_text().splitlines()):
        if "failed" in line and "passed" in line:
            summary = line.strip()
            failed_match = re.search(r"(\d+)\s+failed", line)
            failed = int(failed_match.group(1)) if failed_match else 0
            status = "failed" if failed else "passed"
            break
        if line.startswith("===") and "seconds" in line:
            summary = line.strip()
            status = "passed" if "failed" not in line else "failed"
            break
    if not summary:
        summary = "Nie znaleziono sekcji podsumowania pytest"
    return {"status": status, "summary": summary}


def _write_json_summary(
    path: Path,
    *,
    run_id: str,
    generated_at: datetime,
    records: list[dict[str, Any]],
    pytest_status: dict[str, Any],
    diagnostics: list[str],
) -> None:
    totals: dict[str, int] = {"match": 0, "degraded": 0, "rebind_required": 0, "unknown": 0}
    for record in records:
        totals[record.get("status", "unknown")] = totals.get(record.get("status", "unknown"), 0) + 1
    payload = {
        "run_id": run_id,
        "generated_at": generated_at.isoformat(),
        "totals": totals,
        "records": records,
        "pytest": pytest_status,
        "diagnostics": diagnostics,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _write_csv(path: Path, records: list[dict[str, Any]]) -> None:
    fieldnames = [
        "run_id",
        "generated_at",
        "scenario",
        "status",
        "changed_components",
        "tolerated",
        "blocked",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            row = {
                key: ",".join(record.get(key, []))
                if isinstance(record.get(key), list)
                else record.get(key, "")
                for key in fieldnames
            }
            writer.writerow(row)


def _write_parquet(path: Path, records: list[dict[str, Any]]) -> None:
    fieldnames = [
        "run_id",
        "generated_at",
        "scenario",
        "status",
        "changed_components",
        "tolerated",
        "blocked",
    ]

    if records:
        df = pd.DataFrame.from_records(records, columns=fieldnames)
    else:
        df = pd.DataFrame(columns=fieldnames, dtype=object)

    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)


def _write_prometheus_metrics(
    path: Path, *, records: list[dict[str, Any]], generated_at: datetime, run_id: str
) -> None:
    statuses = ("match", "degraded", "rebind_required")
    totals = {status: 0 for status in statuses}
    lines = [
        "# HELP licensing_drift_status Status per scenariusz dryfu (1=aktywny, 0=nieaktywny)",
        "# TYPE licensing_drift_status gauge",
    ]

    for record in records:
        scenario = record.get("scenario", "scenario")
        status = record.get("status", "unknown")
        for candidate in statuses:
            value = 1 if status == candidate else 0
            if status == candidate:
                totals[candidate] = totals.get(candidate, 0) + 1
            lines.append(
                f'licensing_drift_status{{scenario="{scenario}",status="{candidate}"}} {value}'
            )

    lines.extend(
        [
            "# HELP licensing_drift_rejections_total Liczba scenariuszy wymagających rebind w danym uruchomieniu",
            "# TYPE licensing_drift_rejections_total gauge",
            f"licensing_drift_rejections_total {totals.get('rebind_required', 0)}",
            "# HELP licensing_drift_degraded_total Liczba scenariuszy w stanie degraded",
            "# TYPE licensing_drift_degraded_total gauge",
            f"licensing_drift_degraded_total {totals.get('degraded', 0)}",
            "# HELP licensing_drift_run_timestamp Znacznik czasu generacji podsumowania (Unix epoch)",
            "# TYPE licensing_drift_run_timestamp gauge",
            f'licensing_drift_run_timestamp{{run_id="{run_id}"}} {int(generated_at.timestamp())}',
        ]
    )

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    input_dir = Path(args.input_dir)
    compatibility_path = (
        Path(args.compatibility) if args.compatibility else input_dir / "compatibility.json"
    )
    pytest_log_path = Path(args.pytest_log) if args.pytest_log else input_dir / "pytest.log"
    json_output = (
        Path(args.json_output) if args.json_output else input_dir / "licensing_drift_summary.json"
    )
    csv_output = (
        Path(args.csv_output) if args.csv_output else input_dir / "licensing_drift_summary.csv"
    )
    parquet_output = (
        Path(args.parquet_output)
        if args.parquet_output
        else input_dir / "licensing_drift_summary.parquet"
    )
    dashboard_dir = Path(args.dashboard_dir)
    prom_output = (
        Path(args.prom_output) if args.prom_output else dashboard_dir / "licensing_drift.prom"
    )

    diagnostics: list[str] = []
    matrix, matrix_error = _load_matrix(compatibility_path)
    if matrix is None:
        generated_at = datetime.now(timezone.utc)
        records: list[dict[str, Any]] = []
        if matrix_error:
            diagnostics.append(matrix_error)
        else:
            diagnostics.append(f"Brak pliku {compatibility_path} – wygenerowano puste podsumowanie")
    else:
        generated_at = _parse_generated_at(matrix)
        records = _collect_records(matrix, args.run_id, generated_at)
    pytest_status = _parse_pytest_status(pytest_log_path)

    if pytest_status.get("status") == "missing":
        diagnostics.append(pytest_status.get("summary", "Brak logu pytest"))
    elif pytest_status.get("status") == "failed":
        diagnostics.append(
            f"Testy kompatybilności HWID zakończone niepowodzeniem: {pytest_status.get('summary', '')}"
        )

    if matrix is None:
        pytest_status = {
            "status": pytest_status.get("status", "unknown"),
            "summary": pytest_status.get("summary", ""),
        }
        compatibility_note = (
            matrix_error or f"Brak pliku {compatibility_path} – wygenerowano puste podsumowanie"
        )
        pytest_status["compatibility"] = compatibility_note
        if compatibility_note not in diagnostics:
            diagnostics.append(compatibility_note)

    _write_json_summary(
        json_output,
        run_id=args.run_id,
        generated_at=generated_at,
        records=records,
        pytest_status=pytest_status,
        diagnostics=diagnostics,
    )
    _write_csv(csv_output, records)
    _write_parquet(parquet_output, records)
    _write_prometheus_metrics(
        prom_output, records=records, generated_at=generated_at, run_id=args.run_id
    )

    dashboard_dir.mkdir(parents=True, exist_ok=True)
    (dashboard_dir / json_output.name).write_text(
        json_output.read_text(encoding="utf-8"), encoding="utf-8"
    )
    (dashboard_dir / csv_output.name).write_text(
        csv_output.read_text(encoding="utf-8"), encoding="utf-8"
    )
    (dashboard_dir / parquet_output.name).write_bytes(parquet_output.read_bytes())

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
