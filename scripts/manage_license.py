"""CLI do analizy i eksportu raportów licencyjnych."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Sequence

from core.security import LicenseAuditError, generate_license_audit_report

LOGGER = logging.getLogger("manage_license")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Zarządza raportami licencyjnymi OEM")
    subparsers = parser.add_subparsers(dest="command", required=True)

    audit = subparsers.add_parser("audit", help="Generuje raport i wypisuje go na stdout")
    _add_shared_arguments(audit)
    audit.add_argument(
        "--format",
        choices=("json", "markdown"),
        default="json",
        help="Format wyjścia (domyślnie json)",
    )

    export = subparsers.add_parser("export", help="Eksportuje raport do plików JSON/Markdown")
    _add_shared_arguments(export)
    export.add_argument(
        "--output-dir",
        type=Path,
        default=Path("reports/security"),
        help="Katalog docelowy eksportu (domyślnie reports/security)",
    )
    export.add_argument(
        "--basename",
        default="license_audit",
        help="Bazowa nazwa plików eksportu (domyślnie license_audit)",
    )

    parser.add_argument(
        "--log-level",
        default="INFO",
        help="Poziom logowania (domyślnie INFO)",
    )
    return parser


def _add_shared_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--status-path",
        type=Path,
        default=Path("var/security/license_status.json"),
        help="Ścieżka do statusu licencji",
    )
    parser.add_argument(
        "--audit-log",
        type=Path,
        default=Path("logs/security_admin.log"),
        help="Ścieżka do dziennika audytu licencji",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=50,
        help="Limit wpisów aktywacji w raporcie (domyślnie 50)",
    )


def _command_audit(args: argparse.Namespace) -> int:
    report = generate_license_audit_report(
        status_path=args.status_path,
        audit_log_path=args.audit_log,
        activation_limit=args.limit,
    )
    if args.format == "markdown":
        print(report.to_markdown(), end="")
    else:
        print(json.dumps(report.to_dict(), ensure_ascii=False, indent=2))

    if report.warnings:
        for warning in report.warnings:
            LOGGER.warning("%s", warning)
    return 0


def _command_export(args: argparse.Namespace) -> int:
    report = generate_license_audit_report(
        status_path=args.status_path,
        audit_log_path=args.audit_log,
        activation_limit=args.limit,
    )

    output_dir = args.output_dir.expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)
    basename = args.basename.strip() or "license_audit"

    json_path = output_dir / f"{basename}.json"
    json_path.write_text(json.dumps(report.to_dict(), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    markdown_path = output_dir / f"{basename}.md"
    markdown_path.write_text(report.to_markdown(), encoding="utf-8")

    print(json_path)
    print(markdown_path)

    if report.warnings:
        for warning in report.warnings:
            LOGGER.warning("%s", warning)
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=getattr(logging, str(args.log_level).upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s :: %(message)s",
    )

    try:
        if args.command == "audit":
            return _command_audit(args)
        if args.command == "export":
            return _command_export(args)
    except LicenseAuditError as exc:
        LOGGER.error("Błąd audytu licencji: %s", exc)
        return 2
    except OSError as exc:
        LOGGER.error("Błąd IO: %s", exc)
        return 3

    return 1


if __name__ == "__main__":  # pragma: no cover - wejście z CLI
    raise SystemExit(main())

