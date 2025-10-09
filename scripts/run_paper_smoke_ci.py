"""Orkiestracja smoke testu paper trading na potrzeby pipeline'u CI."""
from __future__ import annotations

import argparse
import json
import logging
import os
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any, Sequence

from scripts.render_paper_smoke_summary import DEFAULT_MAX_JSON_CHARS, render_summary_markdown


_LOGGER = logging.getLogger(__name__)
_RAW_OUTPUT_LIMIT = 2000


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Uruchamia smoke test strategii Daily Trend w trybie paper trading "
            "z pełną automatyzacją publikacji artefaktów."
        )
    )
    parser.add_argument("--config", default="config/core.yaml", help="Ścieżka do CoreConfig")
    parser.add_argument(
        "--environment",
        default="binance_paper",
        help="Nazwa środowiska papierowego z pliku konfiguracyjnego",
    )
    parser.add_argument(
        "--output-dir",
        default="var/paper_smoke_ci",
        help="Katalog roboczy na raporty, logi audytowe i podsumowanie smoke",
    )
    parser.add_argument(
        "--operator",
        default=None,
        help="Nazwa operatora zapisywana w logach audytowych (domyślnie PAPER_SMOKE_OPERATOR)",
    )
    parser.add_argument(
        "--allow-auto-publish-failure",
        action="store_true",
        help="Nie wymagaj sukcesu auto-publikacji artefaktów",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Wyświetl polecenie bez jego wykonywania",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Poziom logowania skryptu",
    )
    parser.add_argument(
        "--run-daily-trend-arg",
        action="append",
        default=[],
        help=(
            "Dodatkowe argumenty przekazywane do run_daily_trend.py. "
            "Wartość może zawierać wiele parametrów rozdzielonych spacjami; "
            "w razie potrzeby użyj cudzysłowów zgodnych z powłoką. "
            "Opcję można powtarzać."
        ),
    )
    parser.add_argument(
        "--env-file",
        default=None,
        help=(
            "Opcjonalna ścieżka pliku środowiskowego (KEY=VALUE), do którego zostaną dopisane "
            "kluczowe informacje o wyniku smoke testu (ścieżki raportów, statusy)."
        ),
    )
    parser.add_argument(
        "--render-summary-markdown",
        default=None,
        help=(
            "Jeśli podano, zapisze raport Markdown wygenerowany na bazie summary.json w tej ścieżce."
        ),
    )
    parser.add_argument(
        "--render-summary-title",
        default=None,
        help="Opcjonalny tytuł raportu Markdown (przekazywany do renderera).",
    )
    parser.add_argument(
        "--render-summary-max-json-chars",
        type=int,
        default=None,
        help=(
            "Maksymalna liczba znaków JSON w sekcjach 'details' raportu Markdown (domyślnie 2000)."
        ),
    )
    parser.add_argument(
        "--skip-summary-validation",
        action="store_true",
        help="Pomiń automatyczną walidację summary.json po zakończeniu smoke testu.",
    )
    parser.add_argument(
        "--summary-validator-arg",
        action="append",
        default=[],
        help=(
            "Dodatkowe argumenty przekazywane do validate_paper_smoke_summary.py. "
            "Wartość może zawierać kilka parametrów rozdzielonych spacjami i może być powtarzana."
        ),
    )
    return parser.parse_args(argv)


def _build_command(
    *,
    config_path: Path,
    environment: str,
    output_dir: Path,
    operator: str,
    auto_publish_required: bool,
    extra_run_daily_trend_args: Sequence[str],
) -> tuple[list[str], dict[str, Path]]:
    script_path = Path(__file__).with_name("run_daily_trend.py")
    if not script_path.exists():  # pragma: no cover - brak pliku to błąd środowiska
        raise FileNotFoundError("run_daily_trend.py not found next to run_paper_smoke_ci.py")

    output_dir.mkdir(parents=True, exist_ok=True)

    summary_path = output_dir / "paper_smoke_summary.json"
    json_log_path = output_dir / "paper_trading_log.jsonl"
    audit_log_path = output_dir / "paper_trading_log.md"
    precheck_dir = output_dir / "paper_precheck_reports"
    precheck_dir.mkdir(parents=True, exist_ok=True)
    smoke_runs_dir = output_dir / "runs"
    smoke_runs_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable,
        str(script_path),
        "--config",
        str(config_path),
        "--environment",
        environment,
        "--paper-smoke",
        "--paper-smoke-json-log",
        str(json_log_path),
        "--paper-smoke-audit-log",
        str(audit_log_path),
        "--paper-precheck-audit-dir",
        str(precheck_dir),
        "--paper-smoke-summary-json",
        str(summary_path),
        "--paper-smoke-operator",
        operator,
        "--smoke-output",
        str(smoke_runs_dir),
        "--archive-smoke",
    ]

    cmd.append("--paper-smoke-auto-publish")
    if auto_publish_required:
        cmd.append("--paper-smoke-auto-publish-required")

    for raw in extra_run_daily_trend_args:
        if not raw:
            continue
        cmd.extend(shlex.split(raw))

    return cmd, {
        "summary": summary_path,
        "json_log": json_log_path,
        "audit_log": audit_log_path,
    }


def _load_summary(summary_path: Path) -> dict:
    if not summary_path.exists():
        raise RuntimeError(
            "Brak pliku podsumowania smoke. Upewnij się, że run_daily_trend zakończył się poprawnie."
        )
    return json.loads(summary_path.read_text(encoding="utf-8"))


def _truncate_text(text: str, *, limit: int = _RAW_OUTPUT_LIMIT) -> str:
    if len(text) <= limit:
        return text
    half = max(limit - 3, 0)
    return text[:half] + "..."


def _run_summary_validation(
    *,
    summary_path: Path,
    environment: str,
    operator: str,
    publish_required: bool,
    extra_args: Sequence[str],
) -> dict[str, Any]:
    script_path = Path(__file__).with_name("validate_paper_smoke_summary.py")
    if not script_path.exists():
        return {
            "status": "failed",
            "reason": "validator_missing",
            "exit_code": 1,
            "stdout": "",
            "stderr": "",
            "required_publish_success": publish_required,
        }

    cmd: list[str] = [
        sys.executable,
        str(script_path),
        "--summary",
        str(summary_path),
        "--require-environment",
        environment,
    ]
    if operator:
        cmd.extend(["--require-operator", operator])
    if publish_required:
        cmd.extend(
            [
                "--require-publish-success",
                "--require-publish-required",
                "--require-publish-exit-zero",
            ]
        )
    for raw in extra_args:
        if not raw:
            continue
        cmd.extend(shlex.split(raw))

    _LOGGER.info("Waliduję podsumowanie smoke przy pomocy validate_paper_smoke_summary.py")

    try:
        completed = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=False,
        )
    except OSError:  # pragma: no cover - uruchomienie interpretatora może zawieść
        return {
            "status": "failed",
            "reason": "validator_exec_failed",
            "exit_code": 1,
            "stdout": "",
            "stderr": "",
            "required_publish_success": publish_required,
        }

    stdout = (completed.stdout or "").strip()
    stderr = (completed.stderr or "").strip()
    trimmed_stdout = _truncate_text(stdout) if stdout else ""
    trimmed_stderr = _truncate_text(stderr) if stderr else ""

    parsed: Any | None = None
    if stdout:
        try:
            parsed = json.loads(stdout)
        except json.JSONDecodeError:
            parsed = None

    status = "ok" if completed.returncode == 0 else "failed"

    return {
        "status": status,
        "exit_code": completed.returncode,
        "stdout": trimmed_stdout,
        "stderr": trimmed_stderr,
        "raw_stdout_present": bool(stdout),
        "raw_stderr_present": bool(stderr),
        "result": parsed,
        "required_publish_success": publish_required,
        "command": cmd,
    }


def _write_env_file(
    env_file: Path,
    *,
    operator: str,
    paths: dict[str, Path],
    summary: dict,
    markdown_path: Path | None,
    validation: dict[str, Any],
) -> None:
    env_file = env_file.expanduser()
    env_file.parent.mkdir(parents=True, exist_ok=True)

    def _normalise(value: str) -> str:
        # GitHub Actions obsługuje format KEY=VALUE, dlatego zabezpieczamy znaki nowej linii.
        return value.replace("\n", "\\n")

    publish_status = summary.get("publish", {}).get("status", "unknown")
    validation_status = validation.get("status", "unknown")

    data = {
        "PAPER_SMOKE_OPERATOR": operator,
        "PAPER_SMOKE_SUMMARY_PATH": str(paths["summary"].resolve()),
        "PAPER_SMOKE_JSON_LOG_PATH": str(paths["json_log"].resolve()),
        "PAPER_SMOKE_AUDIT_LOG_PATH": str(paths["audit_log"].resolve()),
        "PAPER_SMOKE_STATUS": summary.get("status", "unknown"),
        "PAPER_SMOKE_PUBLISH_STATUS": publish_status,
        "PAPER_SMOKE_VALIDATION_STATUS": validation_status,
    }

    if markdown_path is not None:
        data["PAPER_SMOKE_MARKDOWN_PATH"] = str(markdown_path.resolve())

    with env_file.open("a", encoding="utf-8") as fp:
        for key, value in data.items():
            fp.write(f"{key}={_normalise(value)}\n")


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    logging.basicConfig(level=getattr(logging, args.log_level.upper()))

    operator = (
        args.operator
        or os.environ.get("PAPER_SMOKE_OPERATOR")
        or os.environ.get("CI_OPERATOR")
        or "CI Agent"
    )

    config_path = Path(args.config)
    if not config_path.exists():
        raise FileNotFoundError(config_path)

    output_dir = Path(args.output_dir)
    command, paths = _build_command(
        config_path=config_path,
        environment=args.environment,
        output_dir=output_dir,
        operator=operator,
        auto_publish_required=not args.allow_auto_publish_failure,
        extra_run_daily_trend_args=args.run_daily_trend_arg,
    )

    if args.dry_run:
        print(json.dumps({"status": "dry_run", "command": command}, indent=2))
        return 0

    _LOGGER.info(
        "Uruchamiam smoke test paper trading w trybie CI: %s",
        " ".join(map(shlex.quote, command)),
    )

    completed = subprocess.run(command, text=True, check=False)

    if completed.returncode != 0:
        _LOGGER.error("Smoke test zakończył się kodem %s", completed.returncode)
        return completed.returncode

    summary = _load_summary(paths["summary"])

    markdown_path: Path | None = None
    if args.render_summary_markdown:
        markdown_path = Path(args.render_summary_markdown).expanduser()
        markdown_path.parent.mkdir(parents=True, exist_ok=True)
        limit = (
            args.render_summary_max_json_chars
            if args.render_summary_max_json_chars is not None
            else DEFAULT_MAX_JSON_CHARS
        )
        markdown = render_summary_markdown(
            summary,
            title_override=args.render_summary_title,
            max_json_chars=max(limit, 0),
        )
        markdown_path.write_text(markdown, encoding="utf-8")

    if args.skip_summary_validation:
        validation_result = {
            "status": "skipped",
            "reason": "validation_disabled",
            "exit_code": 0,
            "required_publish_success": not args.allow_auto_publish_failure,
        }
    else:
        validation_result = _run_summary_validation(
            summary_path=paths["summary"],
            environment=args.environment,
            operator=operator,
            publish_required=not args.allow_auto_publish_failure,
            extra_args=args.summary_validator_arg,
        )

    summary["validation"] = validation_result
    paths["summary"].write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    if args.env_file:
        _write_env_file(
            Path(args.env_file),
            operator=operator,
            paths=paths,
            summary=summary,
            markdown_path=markdown_path,
            validation=validation_result,
        )

    payload = {
        "summary_path": str(paths["summary"]),
        "summary": summary,
        "json_log_path": str(paths["json_log"]),
        "audit_log_path": str(paths["audit_log"]),
        "validation": validation_result,
    }
    if markdown_path is not None:
        payload["markdown_report_path"] = str(markdown_path)

    exit_code = int(validation_result.get("exit_code", 0))
    payload["status"] = "ok" if exit_code == 0 else "validation_failed"
    print(json.dumps(payload, indent=2))
    if exit_code != 0:
        _LOGGER.error("Walidacja summary.json zakończyła się kodem %s", exit_code)
    return exit_code


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
