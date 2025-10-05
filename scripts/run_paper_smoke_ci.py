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
from typing import Sequence


_LOGGER = logging.getLogger(__name__)


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


def _write_env_file(env_file: Path, *, operator: str, paths: dict[str, Path], summary: dict) -> None:
    env_file = env_file.expanduser()
    env_file.parent.mkdir(parents=True, exist_ok=True)

    def _normalise(value: str) -> str:
        # GitHub Actions obsługuje format KEY=VALUE, dlatego zabezpieczamy znaki nowej linii.
        return value.replace("\n", "\\n")

    publish_status = summary.get("publish", {}).get("status", "unknown")

    data = {
        "PAPER_SMOKE_OPERATOR": operator,
        "PAPER_SMOKE_SUMMARY_PATH": str(paths["summary"].resolve()),
        "PAPER_SMOKE_JSON_LOG_PATH": str(paths["json_log"].resolve()),
        "PAPER_SMOKE_AUDIT_LOG_PATH": str(paths["audit_log"].resolve()),
        "PAPER_SMOKE_STATUS": summary.get("status", "unknown"),
        "PAPER_SMOKE_PUBLISH_STATUS": publish_status,
    }

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

    _LOGGER.info("Uruchamiam smoke test paper trading w trybie CI: %s", " ".join(map(shlex.quote, command)))

    completed = subprocess.run(command, text=True, check=False)

    if completed.returncode != 0:
        _LOGGER.error("Smoke test zakończył się kodem %s", completed.returncode)
        return completed.returncode

    summary = _load_summary(paths["summary"])

    if args.env_file:
        _write_env_file(Path(args.env_file), operator=operator, paths=paths, summary=summary)

    payload = {
        "status": "ok",
        "summary_path": str(paths["summary"]),
        "summary": summary,
        "json_log_path": str(paths["json_log"]),
        "audit_log_path": str(paths["audit_log"]),
    }
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
