"""Generowanie podsumowania smoke testu paper trading w formacie Markdown."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Mapping, Sequence


DEFAULT_MAX_JSON_CHARS = 2000
__all__ = ["DEFAULT_MAX_JSON_CHARS", "render_summary_markdown"]


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Wczytuje plik summary.json wygenerowany przez run_daily_trend i tworzy "
            "skondensowane podsumowanie w formacie Markdown (np. dla GitHub Actions)."
        )
    )
    parser.add_argument(
        "--summary-json",
        required=True,
        help="Ścieżka do pliku paper_smoke_summary.json utworzonego przez run_daily_trend.py",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Opcjonalna ścieżka pliku wynikowego. Jeśli brak – wynik trafia na stdout.",
    )
    parser.add_argument(
        "--title",
        default=None,
        help="Nagłówek raportu (domyślnie: 'Podsumowanie smoke paper trading — <environment>').",
    )
    parser.add_argument(
        "--max-json-chars",
        type=int,
        default=DEFAULT_MAX_JSON_CHARS,
        help="Maksymalna liczba znaków prezentowanych w blokach JSON (domyślnie 2000).",
    )
    return parser.parse_args(argv)


def _load_summary(path: Path) -> Mapping[str, object]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, Mapping):
        raise TypeError("Plik podsumowania musi zawierać obiekt JSON (dict)")
    return data


def _stringify(value: object) -> str:
    if value is None:
        return "—"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return str(value)
    if isinstance(value, str):
        text = value.strip()
        return text or "—"
    try:
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    except TypeError:
        return str(value)


def _escape_table(value: str) -> str:
    return value.replace("|", "\\|").replace("\n", "<br>")


def _build_table(rows: Sequence[tuple[str, object | None]]) -> str:
    filtered = [(label, item) for label, item in rows if item not in (None, "")]
    if not filtered:
        return "_brak danych_\n"
    lines = ["| Pole | Wartość |", "| --- | --- |"]
    for label, item in filtered:
        text = _escape_table(_stringify(item))
        lines.append(f"| {label} | {text} |")
    lines.append("")
    return "\n".join(lines)


def _truncate_json(data: Mapping[str, object] | Sequence[object] | object, limit: int) -> str:
    try:
        raw = json.dumps(data, ensure_ascii=False, indent=2, sort_keys=True)
    except TypeError:
        raw = _stringify(data)
    if limit > 0 and len(raw) > limit:
        return raw[: max(limit - 1, 1)] + "…"
    return raw


def _append_json_block(lines: list[str], title: str, payload: Mapping[str, object] | object, *, limit: int) -> None:
    if payload is None:
        return
    rendered = _truncate_json(payload, limit)
    lines.append(f"<details><summary>{title}</summary>")
    lines.append("")
    lines.append("```json")
    lines.append(rendered)
    lines.append("```")
    lines.append("</details>")
    lines.append("")


def render_summary_markdown(
    summary: Mapping[str, object], *, title_override: str | None = None, max_json_chars: int = DEFAULT_MAX_JSON_CHARS
) -> str:
    environment = _stringify(summary.get("environment"))
    title = title_override or f"Podsumowanie smoke paper trading — {environment}"
    timestamp = _stringify(summary.get("timestamp"))
    operator = _stringify(summary.get("operator"))
    severity = _stringify(summary.get("severity"))
    window = summary.get("window") if isinstance(summary.get("window"), Mapping) else {}
    start = _stringify(window.get("start") if isinstance(window, Mapping) else None)
    end = _stringify(window.get("end") if isinstance(window, Mapping) else None)

    lines: list[str] = [f"# {title}", ""]
    lines.append(f"*Data (UTC):* `{timestamp}`  ")
    lines.append(f"*Operator:* `{operator}`  ")
    lines.append(f"*Krytyczność:* `{severity}`  ")
    lines.append(f"*Okno danych:* `{start}` → `{end}`")
    lines.append("")

    report = summary.get("report") if isinstance(summary.get("report"), Mapping) else None
    if report:
        lines.append("## Artefakty raportu")
        lines.append(_build_table([
            ("Katalog", report.get("directory")),
            ("Plik summary", report.get("summary_path")),
            ("Hash SHA-256", report.get("summary_sha256")),
        ]))

    storage = summary.get("storage") if isinstance(summary.get("storage"), Mapping) else None
    if storage:
        lines.append("## Stan przestrzeni dyskowej")
        storage_rows = [(key.replace("_", " ").title(), value) for key, value in sorted(storage.items())]
        lines.append(_build_table(storage_rows))

    precheck = summary.get("precheck") if isinstance(summary.get("precheck"), Mapping) else {}
    lines.append("## Paper pre-check")
    lines.append(
        _build_table(
            [
                ("Status", precheck.get("status")),
                ("Pokrycie", precheck.get("coverage_status")),
                ("Ryzyko", precheck.get("risk_status")),
            ]
        )
    )
    payload = precheck.get("payload") if isinstance(precheck, Mapping) else None
    if isinstance(payload, Mapping):
        _append_json_block(lines, "Szczegóły pre-check", payload, limit=max_json_chars)

    telemetry = summary.get("telemetry") if isinstance(summary.get("telemetry"), Mapping) else None
    if telemetry:
        lines.append("## Telemetria runtime")
        telemetry_rows: list[tuple[str, object | None]] = [
            ("Plik summary", telemetry.get("summary_path")),
            ("Decision log", telemetry.get("decision_log_path")),
            ("Źródło metryk", telemetry.get("metrics_source_path")),
        ]

        risk_profile = telemetry.get("risk_profile")
        if isinstance(risk_profile, Mapping):
            telemetry_rows.extend(
                [
                    ("Profil ryzyka", risk_profile.get("name")),
                    ("Źródło profilu", risk_profile.get("source")),
                    ("Fallback środowiska", risk_profile.get("environment_fallback")),
                    ("Plik presetów", risk_profile.get("profiles_file")),
                ]
            )

        snippets = telemetry.get("snippets")
        if isinstance(snippets, Mapping):
            telemetry_rows.extend(
                [
                    ("Snippet ENV", snippets.get("env_path")),
                    ("Snippet YAML", snippets.get("yaml_path")),
                ]
            )

        decision_log_report = telemetry.get("decision_log_report")
        if isinstance(decision_log_report, Mapping):
            telemetry_rows.extend(
                [
                    ("Raport decision log", decision_log_report.get("status")),
                    ("Kod wyjścia verify", decision_log_report.get("exit_code")),
                    ("Plik raportu", decision_log_report.get("path")),
                ]
            )

        required_scopes = telemetry.get("required_auth_scopes")
        if isinstance(required_scopes, (list, tuple, set)) and required_scopes:
            telemetry_rows.append(
                ("Wymagane scope'y", ", ".join(sorted(str(scope) for scope in required_scopes)))
            )

        bundle_info = telemetry.get("bundle")
        if isinstance(bundle_info, Mapping):
            telemetry_rows.extend(
                [
                    ("Katalog bundla", bundle_info.get("output_dir")),
                    ("Manifest bundla", bundle_info.get("manifest_path")),
                ]
            )

        lines.append(_build_table(telemetry_rows))

        if isinstance(decision_log_report, Mapping):
            verify_command = decision_log_report.get("command")
            if verify_command:
                lines.append("### Komenda verify_decision_log")
                lines.append("```")
                if isinstance(verify_command, (list, tuple)):
                    lines.append(" ".join(str(part) for part in verify_command))
                else:
                    lines.append(str(verify_command))
                lines.append("```")
                lines.append("")
            verify_payload = decision_log_report.get("payload")
            if isinstance(verify_payload, Mapping):
                _append_json_block(
                    lines,
                    "Szczegóły raportu verify_decision_log",
                    verify_payload,
                    limit=max_json_chars,
                )

        auth_scope_requirements = telemetry.get("auth_scope_requirements")
        if isinstance(auth_scope_requirements, Mapping) and auth_scope_requirements:
            lines.append("### Scope'y RBAC usług runtime")
            scope_rows: list[tuple[str, object | None]] = []
            for service, info in sorted(auth_scope_requirements.items()):
                if not isinstance(info, Mapping):
                    continue
                scopes = info.get("required_scopes")
                rendered_scopes: object | None = None
                if isinstance(scopes, (list, tuple, set)):
                    rendered_scopes = ", ".join(str(scope) for scope in scopes)
                elif isinstance(scopes, str):
                    rendered_scopes = scopes
                scope_rows.append((service, rendered_scopes))
            lines.append(_build_table(scope_rows))

            for service, info in sorted(auth_scope_requirements.items()):
                if not isinstance(info, Mapping):
                    continue
                sources = info.get("sources")
                if isinstance(sources, (list, tuple, set)) and sources:
                    for index, source_entry in enumerate(sources, start=1):
                        if isinstance(source_entry, Mapping):
                            _append_json_block(
                                lines,
                                f"{service} — źródło #{index}",
                                source_entry,
                                limit=max_json_chars,
                            )

        risk_requirements = telemetry.get("risk_service_requirements")
        if isinstance(risk_requirements, Mapping) and risk_requirements:
            lines.append("### Wymagania RiskService")
            risk_details = risk_requirements.get("details")
            detail_rows: list[tuple[str, object | None]] = []
            if isinstance(risk_details, Mapping):
                detail_rows.extend(
                    [
                        ("TLS wymagany", risk_details.get("require_tls")),
                        ("Materiały TLS", risk_details.get("tls_materials")),
                        ("Pinning SHA-256", risk_details.get("expected_server_sha256")),
                        ("Wymagane scope'y", risk_details.get("required_scopes")),
                        ("Wymagany token", risk_details.get("require_auth_token")),
                    ]
                )
            cli_args = risk_requirements.get("cli_args")
            if cli_args:
                detail_rows.append(("Flagi verify_decision_log", " ".join(str(arg) for arg in cli_args)))
            lines.append(_build_table(detail_rows))

            combined_meta = risk_requirements.get("combined_metadata")
            if isinstance(combined_meta, Mapping):
                _append_json_block(
                    lines,
                    "RiskService – metadane scalone",
                    combined_meta,
                    limit=max_json_chars,
                )

            metadata_entries = risk_requirements.get("metadata")
            if isinstance(metadata_entries, (list, tuple, set)):
                for index, entry in enumerate(metadata_entries, start=1):
                    if isinstance(entry, Mapping):
                        _append_json_block(
                            lines,
                            f"RiskService – źródło #{index}",
                            entry,
                            limit=max_json_chars,
                        )

    json_log = summary.get("json_log") if isinstance(summary.get("json_log"), Mapping) else None
    if json_log:
        lines.append("## Dziennik JSONL")
        lines.append(
            _build_table(
                [
                    ("Ścieżka", json_log.get("path")),
                    ("Record ID", json_log.get("record_id")),
                    ("Backend synchronizacji", (json_log.get("sync") or {}).get("backend") if isinstance(json_log.get("sync"), Mapping) else None),
                    ("Lokalizacja", (json_log.get("sync") or {}).get("location") if isinstance(json_log.get("sync"), Mapping) else None),
                ]
            )
        )
        record_payload = json_log.get("record")
        if isinstance(record_payload, Mapping):
            _append_json_block(lines, "Rekord JSONL", record_payload, limit=max_json_chars)
        sync_meta = None
        sync_info = json_log.get("sync") if isinstance(json_log.get("sync"), Mapping) else None
        if sync_info and isinstance(sync_info.get("metadata"), Mapping):
            sync_meta = sync_info["metadata"]
        if isinstance(sync_meta, Mapping):
            _append_json_block(lines, "Metadane synchronizacji JSON", sync_meta, limit=max_json_chars)

    archive = summary.get("archive") if isinstance(summary.get("archive"), Mapping) else None
    if archive:
        lines.append("## Archiwum smoke")
        lines.append(
            _build_table(
                [
                    ("Ścieżka", archive.get("path")),
                    ("Backend uploadu", (archive.get("upload") or {}).get("backend") if isinstance(archive.get("upload"), Mapping) else None),
                    ("Lokalizacja", (archive.get("upload") or {}).get("location") if isinstance(archive.get("upload"), Mapping) else None),
                ]
            )
        )
        upload_info = archive.get("upload") if isinstance(archive.get("upload"), Mapping) else None
        if upload_info and isinstance(upload_info.get("metadata"), Mapping):
            _append_json_block(lines, "Metadane uploadu archiwum", upload_info["metadata"], limit=max_json_chars)

    manifest = summary.get("manifest") if isinstance(summary.get("manifest"), Mapping) else None
    if manifest:
        lines.append("## Manifest danych OHLCV")
        deny_status_value = manifest.get("deny_status")
        if isinstance(deny_status_value, (list, tuple, set)):
            deny_status_text = ", ".join(str(item) for item in deny_status_value)
        else:
            deny_status_text = _stringify(deny_status_value)
        manifest_rows = [
            ("Plik manifestu", manifest.get("manifest_path")),
            ("Plik metryk", manifest.get("metrics_path")),
            ("Plik podsumowania", manifest.get("summary_path")),
            ("Status", manifest.get("worst_status")),
            ("Kod wyjścia", manifest.get("exit_code")),
            ("Etap", manifest.get("stage")),
            ("Profil ryzyka", manifest.get("risk_profile")),
            ("Łączna liczba wpisów", manifest.get("total_entries")),
            ("Blokowane statusy", deny_status_text if deny_status_value else None),
        ]
        lines.append(_build_table(manifest_rows))

        status_counts = manifest.get("status_counts")
        if isinstance(status_counts, Mapping) and status_counts:
            lines.append("### Liczba wpisów manifestu wg statusu")
            status_rows = [(status, status_counts.get(status)) for status in sorted(status_counts)]
            lines.append(_build_table(status_rows))

        signature_payload = manifest.get("summary_signature")
        if isinstance(signature_payload, Mapping):
            signature_rows = [
                ("Podpis – algorytm", signature_payload.get("algorithm")),
                ("Podpis – wartość", signature_payload.get("value")),
                ("Podpis – key_id", signature_payload.get("key_id")),
            ]
            lines.append(_build_table(signature_rows))
            _append_json_block(
                lines,
                "Szczegóły podpisu manifestu",
                signature_payload,
                limit=max_json_chars,
            )

        manifest_summary = manifest.get("summary")
        if isinstance(manifest_summary, Mapping):
            _append_json_block(lines, "Szczegóły manifestu", manifest_summary, limit=max_json_chars)

    security_baseline = summary.get("security_baseline") if isinstance(summary.get("security_baseline"), Mapping) else None
    if security_baseline:
        lines.append("## Audyt bezpieczeństwa (TLS + RBAC)")
        warnings_payload = security_baseline.get("warnings") if isinstance(security_baseline, Mapping) else None
        errors_payload = security_baseline.get("errors") if isinstance(security_baseline, Mapping) else None
        warning_count = len(warnings_payload) if isinstance(warnings_payload, (list, tuple, set)) else (1 if warnings_payload else 0)
        error_count = len(errors_payload) if isinstance(errors_payload, (list, tuple, set)) else (1 if errors_payload else 0)
        lines.append(
            _build_table(
                [
                    ("Raport JSON", security_baseline.get("report_path")),
                    ("Status", security_baseline.get("status")),
                    ("Status raportu", security_baseline.get("baseline_status")),
                    ("Kod wyjścia", security_baseline.get("exit_code")),
                    ("Liczba ostrzeżeń", warning_count if warning_count else None),
                    ("Liczba błędów", error_count if error_count else None),
                ]
            )
        )
        signature_payload = security_baseline.get("summary_signature")
        if isinstance(signature_payload, Mapping):
            signature_rows = [
                ("Podpis – algorytm", signature_payload.get("algorithm")),
                ("Podpis – wartość", signature_payload.get("value")),
                ("Podpis – key_id", signature_payload.get("key_id")),
            ]
            lines.append(_build_table(signature_rows))
            _append_json_block(
                lines,
                "Szczegóły podpisu audytu bezpieczeństwa",
                signature_payload,
                limit=max_json_chars,
            )
        if isinstance(warnings_payload, (list, tuple, set)) and warnings_payload:
            lines.append("### Ostrzeżenia bezpieczeństwa")
            for entry in warnings_payload:
                lines.append(f"- {entry}")
            lines.append("")
        elif isinstance(warnings_payload, str) and warnings_payload:
            lines.append("### Ostrzeżenia bezpieczeństwa")
            lines.append(f"- {warnings_payload}")
            lines.append("")
        if isinstance(errors_payload, (list, tuple, set)) and errors_payload:
            lines.append("### Błędy bezpieczeństwa")
            for entry in errors_payload:
                lines.append(f"- {entry}")
            lines.append("")
        elif isinstance(errors_payload, str) and errors_payload:
            lines.append("### Błędy bezpieczeństwa")
            lines.append(f"- {errors_payload}")
            lines.append("")
        baseline_report_payload = security_baseline.get("report") if isinstance(security_baseline, Mapping) else None
        if isinstance(baseline_report_payload, Mapping):
            _append_json_block(
                lines,
                "Raport audytu bezpieczeństwa",
                baseline_report_payload,
                limit=max_json_chars,
            )

    token_audit = summary.get("token_audit") if isinstance(summary.get("token_audit"), Mapping) else None
    if token_audit:
        lines.append("## Audyt tokenów RBAC")
        token_warnings = token_audit.get("warnings") if isinstance(token_audit, Mapping) else None
        token_errors = token_audit.get("errors") if isinstance(token_audit, Mapping) else None
        warning_count = len(token_warnings) if isinstance(token_warnings, (list, tuple, set)) else (1 if token_warnings else 0)
        error_count = len(token_errors) if isinstance(token_errors, (list, tuple, set)) else (1 if token_errors else 0)
        lines.append(
            _build_table(
                [
                    ("Raport JSON", token_audit.get("report_path")),
                    ("Status", token_audit.get("status")),
                    ("Kod wyjścia", token_audit.get("exit_code")),
                    ("Liczba ostrzeżeń", warning_count if warning_count else None),
                    ("Liczba błędów", error_count if error_count else None),
                ]
            )
        )
        if isinstance(token_warnings, (list, tuple, set)) and token_warnings:
            lines.append("### Ostrzeżenia tokenów")
            for entry in token_warnings:
                lines.append(f"- {entry}")
            lines.append("")
        elif isinstance(token_warnings, str) and token_warnings:
            lines.append("### Ostrzeżenia tokenów")
            lines.append(f"- {token_warnings}")
            lines.append("")
        if isinstance(token_errors, (list, tuple, set)) and token_errors:
            lines.append("### Błędy tokenów")
            for entry in token_errors:
                lines.append(f"- {entry}")
            lines.append("")
        elif isinstance(token_errors, str) and token_errors:
            lines.append("### Błędy tokenów")
            lines.append(f"- {token_errors}")
            lines.append("")
        token_report_payload = token_audit.get("report") if isinstance(token_audit, Mapping) else None
        if isinstance(token_report_payload, Mapping):
            _append_json_block(lines, "Raport audytu tokenów", token_report_payload, limit=max_json_chars)

    tls_audit = summary.get("tls_audit") if isinstance(summary.get("tls_audit"), Mapping) else None
    if tls_audit:
        lines.append("## Audyt TLS usług runtime")
        warnings_payload = tls_audit.get("warnings") if isinstance(tls_audit, Mapping) else None
        errors_payload = tls_audit.get("errors") if isinstance(tls_audit, Mapping) else None
        warning_count = len(warnings_payload) if isinstance(warnings_payload, (list, tuple, set)) else (1 if warnings_payload else 0)
        error_count = len(errors_payload) if isinstance(errors_payload, (list, tuple, set)) else (1 if errors_payload else 0)
        lines.append(
            _build_table(
                [
                    ("Raport JSON", tls_audit.get("report_path")),
                    ("Status", tls_audit.get("status")),
                    ("Kod wyjścia", tls_audit.get("exit_code")),
                    ("Liczba ostrzeżeń", warning_count if warning_count else None),
                    ("Liczba błędów", error_count if error_count else None),
                ]
            )
        )
        if isinstance(warnings_payload, (list, tuple, set)) and warnings_payload:
            lines.append("### Ostrzeżenia TLS")
            for entry in warnings_payload:
                lines.append(f"- {entry}")
            lines.append("")
        elif isinstance(warnings_payload, str) and warnings_payload:
            lines.append("### Ostrzeżenia TLS")
            lines.append(f"- {warnings_payload}")
            lines.append("")
        if isinstance(errors_payload, (list, tuple, set)) and errors_payload:
            lines.append("### Błędy TLS")
            for entry in errors_payload:
                lines.append(f"- {entry}")
            lines.append("")
        elif isinstance(errors_payload, str) and errors_payload:
            lines.append("### Błędy TLS")
            lines.append(f"- {errors_payload}")
            lines.append("")
        report_payload = tls_audit.get("report") if isinstance(tls_audit, Mapping) else None
        if isinstance(report_payload, Mapping):
            _append_json_block(lines, "Raport audytu TLS", report_payload, limit=max_json_chars)

    publish = summary.get("publish") if isinstance(summary.get("publish"), Mapping) else None
    if publish:
        lines.append("## Auto-publikacja artefaktów")
        lines.append(
            _build_table(
                [
                    ("Status", publish.get("status")),
                    ("Wymagana", publish.get("required")),
                    ("Kod wyjścia", publish.get("exit_code")),
                    ("Powód", publish.get("reason")),
                ]
            )
        )
        if publish.get("raw_stdout"):
            _append_json_block(lines, "Wyjście publish_paper_smoke_artifacts (stdout)", publish["raw_stdout"], limit=max_json_chars)
        if publish.get("raw_stderr"):
            _append_json_block(lines, "Wyjście publish_paper_smoke_artifacts (stderr)", publish["raw_stderr"], limit=max_json_chars)

    return "\n".join(lines).rstrip() + "\n"


def _write_output(text: str, output_path: Path | None) -> None:
    if output_path is None:
        print(text, end="")
        return
    output_path = output_path.expanduser()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(text, encoding="utf-8")


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    summary_path = Path(args.summary_json)
    if not summary_path.exists():
        raise FileNotFoundError(summary_path)

    summary = _load_summary(summary_path)
    report = render_summary_markdown(summary, title_override=args.title, max_json_chars=max(args.max_json_chars, 0))
    _write_output(report, Path(args.output) if args.output else None)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
