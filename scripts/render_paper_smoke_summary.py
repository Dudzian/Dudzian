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
