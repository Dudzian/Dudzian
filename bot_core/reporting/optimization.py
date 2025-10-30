"""Generowanie raportów z przebiegów optymalizacji strategii."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Mapping, Sequence

from bot_core.optimization import StrategyOptimizationReport

_HTML_HEADER = """<!DOCTYPE html><html lang=\"pl\"><head><meta charset=\"utf-8\"><title>Raport optymalizacji</title><style>body{font-family:Arial,sans-serif;margin:24px;}table{border-collapse:collapse;width:100%;}th,td{border:1px solid #ccc;padding:6px 10px;text-align:left;}th{background:#f0f0f0;}</style></head><body>"""
_HTML_FOOTER = "</body></html>"


def render_html_report(report: StrategyOptimizationReport) -> str:
    """Buduje raport HTML z wynikami optymalizacji."""

    rows: list[str] = []
    for index, trial in enumerate(report.trials, start=1):
        params = ", ".join(f"{key}={value}" for key, value in trial.parameters.items())
        metadata = ", ".join(f"{key}={value}" for key, value in trial.metadata.items())
        rows.append(
            "<tr><td>{idx}</td><td>{score:.6f}</td><td>{params}</td><td>{meta}</td></tr>".format(
                idx=index,
                score=trial.score,
                params=params or "-",
                meta=metadata or "-",
            )
        )
    body = "".join(rows)
    started = report.started_at.strftime("%Y-%m-%d %H:%M:%S")
    completed = report.completed_at.strftime("%Y-%m-%d %H:%M:%S")
    tags = ", ".join(report.tags)
    header = (
        f"<h1>Raport optymalizacji – {report.strategy}</h1>"
        f"<p>Silnik: <strong>{report.engine}</strong><br>"
        f"Cel: <strong>{report.objective}</strong> ({report.goal})<br>"
        f"Algorytm: <strong>{report.algorithm}</strong><br>"
        f"Dataset: <strong>{report.dataset or 'brak'}</strong><br>"
        f"Znaczniki: <strong>{tags or 'brak'}</strong><br>"
        f"Start: {started}, zakończenie: {completed}, czas [s]: {report.duration_seconds:.2f}</p>"
    )
    table = (
        "<table><thead><tr><th>#</th><th>Wynik</th><th>Parametry</th><th>Metadane</th></tr></thead><tbody>"
        f"{body}</tbody></table>"
    )
    summary = (
        "<h2>Najlepszy wynik</h2>"
        f"<p>Score: <strong>{report.best.score:.6f}</strong><br>"
        f"Parametry: {', '.join(f'{k}={v}' for k, v in report.best.parameters.items()) or '-'}<br>"
        f"Metadane: {', '.join(f'{k}={v}' for k, v in report.best.metadata.items()) or '-'}" + "</p>"
    )
    return _HTML_HEADER + header + table + summary + _HTML_FOOTER


def render_json_report(report: StrategyOptimizationReport) -> str:
    """Zwraca raport w formacie JSON."""

    payload: Mapping[str, object] = {
        "strategy": report.strategy,
        "engine": report.engine,
        "algorithm": report.algorithm,
        "objective": report.objective,
        "goal": report.goal,
        "dataset": report.dataset,
        "tags": list(report.tags),
        "started_at": report.started_at.isoformat(),
        "completed_at": report.completed_at.isoformat(),
        "duration_seconds": report.duration_seconds,
        "best": {
            "parameters": dict(report.best.parameters),
            "score": report.best.score,
            "metadata": dict(report.best.metadata),
        },
        "trials": [
            {
                "parameters": dict(trial.parameters),
                "score": trial.score,
                "metadata": dict(trial.metadata),
            }
            for trial in report.trials
        ],
    }
    return json.dumps(payload, ensure_ascii=False, indent=2)


def _render_plain_text(report: StrategyOptimizationReport) -> str:
    lines = [
        f"Raport optymalizacji dla strategii {report.strategy}",
        f"Silnik: {report.engine}",
        f"Algorytm: {report.algorithm}, cel: {report.objective} ({report.goal})",
        f"Okres: {report.started_at.isoformat()} -> {report.completed_at.isoformat()} ({report.duration_seconds:.2f}s)",
        f"Najlepszy score: {report.best.score:.6f}",
        "Parametry najlepszego wyniku:",
    ]
    for key, value in report.best.parameters.items():
        lines.append(f"  - {key}: {value}")
    lines.append("\nPróby:")
    for idx, trial in enumerate(report.trials, start=1):
        params = ", ".join(f"{k}={v}" for k, v in trial.parameters.items())
        lines.append(f"  {idx}. score={trial.score:.6f} [{params}]")
    return "\n".join(lines)


def _render_pdf(report: StrategyOptimizationReport) -> bytes:
    text = _render_plain_text(report)
    escaped = text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")
    lines = escaped.splitlines() or [""]
    content_lines = ["BT /F1 11 Tf 40 780 Td"]
    for line in lines:
        content_lines.append(f"({line}) Tj")
        content_lines.append("T*")
    content_lines.append("ET")
    content_stream = "\n".join(content_lines)
    content_bytes = content_stream.encode("utf-8")
    length = len(content_bytes)
    objects = [
        "1 0 obj <</Type /Catalog /Pages 2 0 R>> endobj",
        "2 0 obj <</Type /Pages /Kids [3 0 R] /Count 1>> endobj",
        "3 0 obj <</Type /Page /Parent 2 0 R /MediaBox [0 0 595 842] /Contents 4 0 R /Resources <</Font <</F1 5 0 R>>>>>> endobj",
        f"4 0 obj <</Length {length}>> stream\n{content_stream}\nendstream endobj",
        "5 0 obj <</Type /Font /Subtype /Type1 /BaseFont /Helvetica>> endobj",
    ]
    xref_positions = []
    pdf_parts = [b"%PDF-1.4\n"]
    offset = len(pdf_parts[0])
    for obj in objects:
        xref_positions.append(offset)
        encoded = (obj + "\n").encode("utf-8")
        pdf_parts.append(encoded)
        offset += len(encoded)
    xref_start = offset
    xref_lines = ["xref", "0 6", "0000000000 65535 f "]
    for position in xref_positions:
        xref_lines.append(f"{position:010} 00000 n ")
    trailer = "trailer <</Size 6 /Root 1 0 R>>"
    pdf_parts.append("\n".join(xref_lines).encode("utf-8"))
    pdf_parts.append(("\n" + trailer + f"\nstartxref\n{xref_start}\n%%EOF").encode("utf-8"))
    return b"".join(pdf_parts)


def export_report(
    report: StrategyOptimizationReport,
    directory: str | Path,
    *,
    formats: Sequence[str] = ("html", "json"),
    prefix: str | None = None,
) -> Mapping[str, Path]:
    """Zapisuje raport w wybranych formatach i zwraca mapę format->ścieżka."""

    target_dir = Path(directory)
    target_dir.mkdir(parents=True, exist_ok=True)
    timestamp = report.completed_at.strftime("%Y%m%d-%H%M%S")
    base_name = prefix or f"{report.strategy}_{timestamp}"
    exported: dict[str, Path] = {}
    normalized_formats = tuple(format_.lower() for format_ in formats)
    if "html" in normalized_formats:
        html_path = target_dir / f"{base_name}.html"
        html_path.write_text(render_html_report(report), encoding="utf-8")
        exported["html"] = html_path
    if "json" in normalized_formats:
        json_path = target_dir / f"{base_name}.json"
        json_path.write_text(render_json_report(report), encoding="utf-8")
        exported["json"] = json_path
    if "pdf" in normalized_formats:
        pdf_path = target_dir / f"{base_name}.pdf"
        pdf_path.write_bytes(_render_pdf(report))
        exported["pdf"] = pdf_path
    return exported


__all__ = [
    "render_html_report",
    "render_json_report",
    "export_report",
]
