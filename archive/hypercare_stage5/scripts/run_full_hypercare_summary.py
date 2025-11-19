"""Łączy raporty Stage5 i Stage6 w podpisany przegląd hypercare."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import shlex
import sys
from pathlib import Path
from typing import Mapping, Sequence

from bot_core.runtime.full_hypercare import (
    FullHypercareSummaryBuilder,
    FullHypercareSummaryConfig,
)


def _load_key(value: str | None, path: str | None) -> bytes | None:
    if value and path:
        raise ValueError("Podaj klucz HMAC w formie tekstu lub pliku, nie obu naraz.")
    if value:
        return value.encode("utf-8")
    if path:
        return Path(path).expanduser().read_bytes().strip()
    return None


def _load_metadata(path: str | None) -> Mapping[str, object] | None:
    if not path:
        return None
    payload = json.loads(Path(path).expanduser().read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError("Plik metadanych powinien zawierać obiekt JSON")
    return payload


def _build_command_parts(args: argparse.Namespace) -> list[str]:
    script_path = Path(__file__).resolve()
    parts: list[str] = ["python", script_path.as_posix(), "--stage5-summary", args.stage5_summary, "--stage6-summary", args.stage6_summary, "--output", args.output]

    if args.stage5_signature:
        parts.extend(["--stage5-signature", args.stage5_signature])
    if args.stage6_signature:
        parts.extend(["--stage6-signature", args.stage6_signature])
    if args.stage5_signing_key:
        parts.extend(["--stage5-signing-key", args.stage5_signing_key])
    if args.stage5_signing_key_file:
        parts.extend(["--stage5-signing-key-file", args.stage5_signing_key_file])
    if args.stage6_signing_key:
        parts.extend(["--stage6-signing-key", args.stage6_signing_key])
    if args.stage6_signing_key_file:
        parts.extend(["--stage6-signing-key-file", args.stage6_signing_key_file])
    if args.require_stage5_signature:
        parts.append("--require-stage5-signature")
    if args.require_stage6_signature:
        parts.append("--require-stage6-signature")
    if args.signing_key:
        parts.extend(["--signing-key", args.signing_key])
    if args.signing_key_file:
        parts.extend(["--signing-key-file", args.signing_key_file])
    if args.signing_key_id:
        parts.extend(["--signing-key-id", args.signing_key_id])
    if args.metadata:
        parts.extend(["--metadata", args.metadata])
    if args.archive_dir and not args.no_archive:
        parts.extend(["--archive-dir", args.archive_dir])
    if args.no_archive:
        parts.append("--no-archive")
    if args.archive_extra:
        for extra in args.archive_extra:
            parts.extend(["--archive-extra", extra])
    if args.archive_timestamp:
        parts.extend(["--archive-timestamp", args.archive_timestamp])
    return parts


def _write_cron_template(path: Path, parts: Sequence[str]) -> None:
    path = path.expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    command = " ".join(shlex.quote(item) for item in parts)
    content = (
        "# Automatyczny wpis cron generujący raport full hypercare\n"
        "# Uruchomienie co poniedziałek o 02:30 UTC\n"
        f"30 2 * * MON {command}\n"
    )
    path.write_text(content, encoding="utf-8")


def _write_windows_task_template(path: Path, parts: Sequence[str]) -> None:
    path = path.expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    if parts:
        executable = parts[0]
        arguments = " ".join(shlex.quote(item) for item in parts[1:])
    else:  # pragma: no cover - defensywne
        executable = "python"
        arguments = ""
    xml = f"""<?xml version=\"1.0\" encoding=\"UTF-16\"?>
<Task version=\"1.4\" xmlns=\"http://schemas.microsoft.com/windows/2004/02/mit/task\">
  <Triggers>
    <CalendarTrigger>
      <ScheduleByWeek>
        <WeeksInterval>1</WeeksInterval>
        <DaysOfWeek>
          <Monday />
        </DaysOfWeek>
      </ScheduleByWeek>
      <StartBoundary>2024-01-01T02:30:00</StartBoundary>
    </CalendarTrigger>
  </Triggers>
  <Actions Context=\"Author\">
    <Exec>
      <Command>{executable}</Command>
      <Arguments>{arguments}</Arguments>
    </Exec>
  </Actions>
</Task>
"""
    path.write_text(xml, encoding="utf-8")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Agreguje raporty Stage5 i Stage6 do podpisanego przeglądu hypercare.",
    )

    parser.add_argument("--stage5-summary", required=True, help="Raport Stage5 hypercare JSON")
    parser.add_argument("--stage5-signature", help="Podpis HMAC raportu Stage5")
    parser.add_argument("--stage5-signing-key", help="Klucz HMAC do weryfikacji Stage5")
    parser.add_argument("--stage5-signing-key-file", help="Plik z kluczem HMAC Stage5")
    parser.add_argument(
        "--require-stage5-signature",
        action="store_true",
        help="Wymagaj obecności podpisu HMAC w raporcie Stage5",
    )

    parser.add_argument("--stage6-summary", required=True, help="Raport Stage6 hypercare JSON")
    parser.add_argument("--stage6-signature", help="Podpis HMAC raportu Stage6")
    parser.add_argument("--stage6-signing-key", help="Klucz HMAC do weryfikacji Stage6")
    parser.add_argument("--stage6-signing-key-file", help="Plik z kluczem HMAC Stage6")
    parser.add_argument(
        "--require-stage6-signature",
        action="store_true",
        help="Wymagaj obecności podpisu HMAC w raporcie Stage6",
    )

    parser.add_argument(
        "--output",
        default="var/audit/hypercare/full_hypercare_summary.json",
        help="Ścieżka docelowa raportu zbiorczego",
    )
    parser.add_argument("--signature", help="Opcjonalny podpis raportu zbiorczego")
    parser.add_argument("--signing-key", help="Klucz HMAC do podpisu raportu zbiorczego")
    parser.add_argument("--signing-key-file", help="Plik z kluczem HMAC do podpisu raportu")
    parser.add_argument("--signing-key-id", help="Identyfikator klucza podpisującego raport")
    parser.add_argument(
        "--metadata",
        help="Plik JSON z dodatkowymi metadanymi dołączanymi do raportu zbiorczego",
    )
    parser.add_argument(
        "--archive-dir",
        default="var/audit/hypercare/archive",
        help="Katalog, w którym archiwizowane będą raporty hypercare",
    )
    parser.add_argument("--no-archive", action="store_true", help="Wyłącz automatyczną archiwizację raportów")
    parser.add_argument("--archive-extra", action="append", help="Dodatkowe pliki dołączne do archiwum")
    parser.add_argument("--archive-timestamp", help="Wymuszony znacznik czasu archiwizacji (ISO8601)")
    parser.add_argument("--cron-template", help="Ścieżka pliku z wygenerowanym wpisem cron")
    parser.add_argument(
        "--windows-task-template",
        help="Ścieżka pliku XML z harmonogramem dla Windows Scheduler",
    )

    return parser


def run(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    try:
        stage5_key = _load_key(args.stage5_signing_key, args.stage5_signing_key_file)
        stage6_key = _load_key(args.stage6_signing_key, args.stage6_signing_key_file)
        signing_key = _load_key(args.signing_key, args.signing_key_file)
        metadata = _load_metadata(args.metadata)

        archive_dir = None if args.no_archive else Path(args.archive_dir)
        archive_extra = tuple(Path(item) for item in args.archive_extra or ())
        archive_timestamp = None
        if args.archive_timestamp:
            try:
                archive_timestamp = datetime.fromisoformat(args.archive_timestamp)
            except ValueError as exc:
                raise ValueError("Parametr archive-timestamp wymaga formatu ISO8601") from exc
            if archive_timestamp.tzinfo is None:
                archive_timestamp = archive_timestamp.replace(tzinfo=timezone.utc)

        config = FullHypercareSummaryConfig(
            stage5_summary_path=Path(args.stage5_summary),
            stage6_summary_path=Path(args.stage6_summary),
            stage5_signature_path=Path(args.stage5_signature) if args.stage5_signature else None,
            stage6_signature_path=Path(args.stage6_signature) if args.stage6_signature else None,
            stage5_signing_key=stage5_key,
            stage6_signing_key=stage6_key,
            stage5_require_signature=bool(args.require_stage5_signature),
            stage6_require_signature=bool(args.require_stage6_signature),
            output_path=Path(args.output),
            signature_path=Path(args.signature) if args.signature else None,
            signing_key=signing_key,
            signing_key_id=args.signing_key_id,
            metadata=metadata,
            archive_dir=archive_dir,
            archive_timestamp=archive_timestamp,
            archive_extra_files=archive_extra,
        )
    except (ValueError, FileNotFoundError) as exc:
        print(json.dumps({"error": str(exc)}), file=sys.stderr)
        return 2

    result = FullHypercareSummaryBuilder(config).run()

    command_parts = _build_command_parts(args)
    cron_template_path: Path | None = None
    windows_template_path: Path | None = None
    if args.cron_template:
        cron_template_path = Path(args.cron_template)
        _write_cron_template(cron_template_path, command_parts)
    if args.windows_task_template:
        windows_template_path = Path(args.windows_task_template)
        _write_windows_task_template(windows_template_path, command_parts)

    output = {
        "summary_path": result.output_path.as_posix(),
        "signature_path": result.signature_path.as_posix() if result.signature_path else None,
        "overall_status": result.payload.get("overall_status"),
        "issues": result.payload.get("issues", []),
        "warnings": result.payload.get("warnings", []),
    }
    if result.archive_path:
        output["archive_path"] = result.archive_path.as_posix()
    if cron_template_path:
        output["cron_template"] = cron_template_path.expanduser().as_posix()
    if windows_template_path:
        output["windows_task_template"] = windows_template_path.expanduser().as_posix()
    print(json.dumps(output, ensure_ascii=False, indent=2))
    return 0


def main() -> None:  # pragma: no cover - uruchomienie jako skrypt
    sys.exit(run())


if __name__ == "__main__":  # pragma: no cover
    main()

