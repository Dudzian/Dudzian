"""Weryfikacja progów Stage6 w `config/core.yaml`.

Skrypt odczytuje konfigurację core i porównuje kluczowe wartości Market Intel,
Portfolio Governora oraz Stress Lab z ustaleniami warsztatowymi z 2024-06-07.

Zwracany kod wyjścia:
* 0 – wszystkie progi zgodne,
* 1 – wykryto rozbieżności lub brak sekcji.
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
import sys
from pathlib import Path
from typing import Iterable

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from bot_core.config.loader import load_core_config  # noqa: E402  # isort:skip
from bot_core.config.stage6_thresholds import (  # noqa: E402  # isort:skip
    collect_stage6_threshold_differences,
)


def _format_path(path: Path) -> str:
    try:
        return str(path.relative_to(Path.cwd()))
    except ValueError:
        return str(path)


def _collect_differences(config_path: Path) -> list[str]:
    config = load_core_config(str(config_path))
    return collect_stage6_threshold_differences(config)


def _build_report(differences: list[str], config_path: Path) -> dict[str, object]:
    status = "ok" if not differences else "mismatch"
    timestamp = datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace(
        "+00:00", "Z"
    )
    return {
        "timestamp": timestamp,
        "config_path": str(config_path),
        "status": status,
        "differences": differences,
    }


def _write_json_report(report_path: Path, payload: dict[str, object]) -> None:
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )


def _print_result(differences: Iterable[str], config_path: Path) -> int:
    diffs = list(differences)
    if not diffs:
        print(
            f"✅ Konfiguracja Stage6 w {_format_path(config_path)} jest zgodna z ustaleniami warsztatowymi."
        )
        return 0

    print(
        f"❌ Wykryto {len(diffs)} rozbieżności w {_format_path(config_path)} względem warsztatowych progów Stage6:"
    )
    for diff in diffs:
        print(f"  - {diff}")
    return 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "config",
        nargs="?",
        help="(przestarzałe) Ścieżka do pliku konfiguracyjnego core",
    )
    parser.add_argument(
        "--config",
        dest="config_flag",
        default=None,
        help="Ścieżka do pliku konfiguracyjnego core (domyślnie config/core.yaml)",
    )
    parser.add_argument(
        "--json-report",
        dest="json_report",
        default=None,
        help=(
            "Ścieżka do opcjonalnego raportu JSON. Katalogi zostaną utworzone, a plik "
            "będzie zawierał status audytu oraz listę rozbieżności."
        ),
    )

    args = parser.parse_args(argv)
    config_value = args.config_flag or args.config or "config/core.yaml"
    config_path = Path(config_value).expanduser().resolve()

    if not config_path.exists():
        print(f"❌ Nie znaleziono pliku konfiguracyjnego: {_format_path(config_path)}", file=sys.stderr)
        return 1

    differences = _collect_differences(config_path)
    exit_code = _print_result(differences, config_path)

    if args.json_report:
        report_path = Path(args.json_report).expanduser()
        if not report_path.is_absolute():
            report_path = (Path.cwd() / report_path).resolve()
        payload = _build_report(differences, config_path)
        _write_json_report(report_path, payload)
        print(f"ℹ️ Raport JSON zapisany w {_format_path(report_path)}")

    return exit_code


if __name__ == "__main__":  # pragma: no cover - CLI entrypoint
    sys.exit(main())
