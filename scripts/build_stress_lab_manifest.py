#!/usr/bin/env python3
"""Buduje manifest i streszczenie dla raportu Stage6 Stress Lab."""
from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping


def _sha256(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(131072), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def _load_report(path: Path) -> Mapping[str, Any]:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise SystemExit(f"Nie znaleziono raportu Stress Lab: {path}") from exc
    except json.JSONDecodeError as exc:  # pragma: no cover - błędny raport
        raise SystemExit(f"Raport Stress Lab ma niepoprawny format JSON ({path}): {exc}") from exc
    if not isinstance(raw, Mapping):  # pragma: no cover - zła struktura
        raise SystemExit(f"Raport Stress Lab powinien być obiektem JSON, otrzymano: {type(raw)!r}")
    return raw


def _build_manifest(
    *,
    report_path: Path,
    signature_path: Path,
    output_path: Path,
    report_payload: Mapping[str, Any],
) -> Path:
    scenarios = report_payload.get("scenarios")
    failure_count = int(report_payload.get("failure_count", 0))
    manifest = {
        "version": "1.0",
        "generated_at": _dt.datetime.now(_dt.timezone.utc).isoformat(),
        "report": {
            "path": report_path.as_posix(),
            "size": report_path.stat().st_size,
            "sha256": _sha256(report_path),
            "failure_count": failure_count,
            "scenario_count": len(scenarios) if isinstance(scenarios, list) else None,
        },
        "signature": {
            "path": signature_path.as_posix(),
            "size": signature_path.stat().st_size if signature_path.exists() else None,
            "sha256": _sha256(signature_path) if signature_path.exists() else None,
        },
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return output_path


def _build_summary(
    *,
    summary_path: Path,
    report_payload: Mapping[str, Any],
    manifest_path: Path,
) -> Path:
    scenarios = report_payload.get("scenarios")
    lines = [
        "# Raport Stress Lab – podsumowanie CI",
        "",
        f"- Manifest: `{manifest_path.as_posix()}`",
    ]
    generated_at = report_payload.get("generated_at")
    if isinstance(generated_at, str):
        lines.append(f"- Raport wygenerowano: `{generated_at}`")
    failure_count = int(report_payload.get("failure_count", 0))
    lines.append(f"- Liczba scenariuszy z naruszeniami: **{failure_count}**")
    if isinstance(scenarios, list):
        lines.append("\n## Status scenariuszy")
        for scenario in scenarios:
            if not isinstance(scenario, Mapping):  # pragma: no cover - dane niezgodne z kontraktem
                continue
            name = scenario.get("name", "(brak nazwy)")
            status = scenario.get("status", "unknown")
            failures = scenario.get("failures", [])
            metrics = scenario.get("metrics", {})
            lines.append(f"- **{name}** – status: `{status}`")
            if failures:
                lines.append("  - Naruszenia:")
                for item in failures:
                    lines.append(f"    - {item}")
            if isinstance(metrics, Mapping) and metrics:
                metrics_line = ", ".join(
                    f"{key}={value}" for key, value in sorted(metrics.items())
                )
                lines.append(f"  - Metryki: {metrics_line}")
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return summary_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Buduje manifest i podsumowanie raportu Stress Lab")
    parser.add_argument("--report", required=True, type=Path, help="Ścieżka do raportu JSON")
    parser.add_argument("--signature", required=True, type=Path, help="Ścieżka do podpisu HMAC")
    parser.add_argument("--output", required=True, type=Path, help="Ścieżka zapisu manifestu JSON")
    parser.add_argument("--summary", type=Path, help="Opcjonalny plik podsumowania Markdown")
    args = parser.parse_args(argv)

    report_path = args.report.expanduser().resolve()
    signature_path = args.signature.expanduser().resolve()
    output_path = args.output.expanduser().resolve()
    summary_path = args.summary.expanduser().resolve() if args.summary else None

    payload = _load_report(report_path)
    manifest_path = _build_manifest(
        report_path=report_path,
        signature_path=signature_path,
        output_path=output_path,
        report_payload=payload,
    )

    if summary_path is not None:
        _build_summary(summary_path=summary_path, report_payload=payload, manifest_path=manifest_path)

    print(json.dumps({"status": "ok", "manifest": manifest_path.as_posix()}))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
