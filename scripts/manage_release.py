"""Narzędzia wspierające przygotowanie wydań OEM."""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Mapping

_REPO_ROOT = Path(__file__).resolve().parents[1]
_TEMPLATE_DIR = _REPO_ROOT / "docs" / "deploy" / "templates"

_TEMPLATE_MAP: Dict[str, str] = {
    "checklist": "oem_checklist_template.md",
    "license-report": "license_report_template.md",
    "compliance-report": "compliance_report_template.md",
    "test-report": "test_report_template.md",
}


class _DefaultDict(dict):
    """Słownik z domyślnym tekstem dla brakujących kluczy."""

    def __missing__(self, key: str) -> str:  # pragma: no cover - proste domyślne zachowanie
        return "N/D"


@dataclass
class DocumentResult:
    path: Path
    template: str
    context: Mapping[str, str]


def _render_template(template_key: str, context: Mapping[str, str]) -> str:
    if template_key not in _TEMPLATE_MAP:
        raise KeyError(f"Nieznany klucz szablonu: {template_key}")
    template_path = _TEMPLATE_DIR / _TEMPLATE_MAP[template_key]
    if not template_path.exists():
        raise FileNotFoundError(f"Brak szablonu: {template_path}")
    raw = template_path.read_text(encoding="utf-8")
    safe_context = _DefaultDict(context)
    return raw.format_map(safe_context)


def generate_document(template_key: str, output_path: Path, context: Mapping[str, str]) -> DocumentResult:
    """Generuje dokument na bazie szablonu i zwraca informacje o wyniku."""

    enriched_context = dict(context)
    enriched_context.setdefault("timestamp", datetime.utcnow().strftime("%Y-%m-%d %H:%M:%SZ"))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    content = _render_template(template_key, enriched_context)
    output_path.write_text(content, encoding="utf-8")
    return DocumentResult(path=output_path, template=_TEMPLATE_MAP[template_key], context=enriched_context)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Zarządzanie materiałami wydania OEM")
    subparsers = parser.add_subparsers(dest="command", required=True)

    checklist = subparsers.add_parser("generate-checklist", help="Generuje checklistę wydania OEM")
    checklist.add_argument("--version", required=True, help="Numer wersji wydania")
    checklist.add_argument("--output", required=True, help="Ścieżka docelowa pliku Markdown")
    checklist.add_argument("--owner", default="N/D", help="Osoba lub zespół odpowiedzialny za wydanie")
    checklist.add_argument("--release-tag", default="N/D", help="Tag Git lub identyfikator wydania")
    checklist.add_argument("--license-report", default="N/D", help="Ścieżka do raportu licencyjnego")
    checklist.add_argument("--compliance-report", default="N/D", help="Ścieżka do raportu zgodności")
    checklist.add_argument("--test-report", default="N/D", help="Ścieżka do raportu testów")
    checklist.add_argument("--notes", default="Brak dodatkowych uwag.", help="Uwagi końcowe")

    license_report = subparsers.add_parser("generate-license-report", help="Generuje raport licencyjny")
    license_report.add_argument("--version", required=True)
    license_report.add_argument("--output", required=True)
    license_report.add_argument("--license-scope", default="OEM")
    license_report.add_argument("--summary", default="Brak podsumowania.")
    license_report.add_argument("--activation-table", default="Brak szczegółów aktywacji.")
    license_report.add_argument("--notes", default="Brak uwag.")

    compliance_report = subparsers.add_parser("generate-compliance-report", help="Generuje raport zgodności")
    compliance_report.add_argument("--version", required=True)
    compliance_report.add_argument("--output", required=True)
    compliance_report.add_argument("--owner", default="Zespół compliance")
    compliance_report.add_argument("--summary", default="Brak naruszeń.")
    compliance_report.add_argument("--violations", default="Brak naruszeń do zgłoszenia.")
    compliance_report.add_argument("--recommendations", default="Brak rekomendacji.")

    test_report = subparsers.add_parser("generate-test-report", help="Generuje raport testów")
    test_report.add_argument("--version", required=True)
    test_report.add_argument("--output", required=True)
    test_report.add_argument("--environment", default="Lokalne lab")
    test_report.add_argument("--summary", default="Brak podsumowania.")
    test_report.add_argument("--scenarios", default="Brak scenariuszy.")
    test_report.add_argument("--defects", default="Brak defektów.")

    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> DocumentResult:
    args = _parse_args(argv)
    output_path = Path(args.output)

    if args.command == "generate-checklist":
        context = {
            "version": args.version,
            "owner": args.owner,
            "release_tag": args.release_tag,
            "license_report": args.license_report,
            "compliance_report": args.compliance_report,
            "test_report": args.test_report,
            "notes": args.notes,
        }
        return generate_document("checklist", output_path, context)

    if args.command == "generate-license-report":
        context = {
            "version": args.version,
            "license_scope": args.license_scope,
            "summary": args.summary,
            "activation_table": args.activation_table,
            "notes": args.notes,
        }
        return generate_document("license-report", output_path, context)

    if args.command == "generate-compliance-report":
        context = {
            "version": args.version,
            "owner": args.owner,
            "summary": args.summary,
            "violations": args.violations,
            "recommendations": args.recommendations,
        }
        return generate_document("compliance-report", output_path, context)

    if args.command == "generate-test-report":
        context = {
            "version": args.version,
            "environment": args.environment,
            "summary": args.summary,
            "scenarios": args.scenarios,
            "defects": args.defects,
        }
        return generate_document("test-report", output_path, context)

    raise ValueError(f"Nieobsługiwane polecenie: {args.command}")


if __name__ == "__main__":  # pragma: no cover - wejście CLI
    main()
