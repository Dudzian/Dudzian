"""Uruchamia zestaw testów smoke i zapisuje raporty do `reports/smoke`."""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pytest


@dataclass
class SmokeSummary:
    exit_status: int
    tests_collected: int
    passed: int
    failed: int
    skipped: int
    duration_seconds: float
    tag: str | None

    def to_json(self) -> dict[str, Any]:
        return {
            "exit_status": self.exit_status,
            "tests_collected": self.tests_collected,
            "passed": self.passed,
            "failed": self.failed,
            "skipped": self.skipped,
            "duration_seconds": round(self.duration_seconds, 3),
            "tag": self.tag,
        }


class SmokeReportPlugin:
    def __init__(self) -> None:
        self._start: float = 0.0
        self.summary: SmokeSummary | None = None

    def pytest_sessionstart(self, session: pytest.Session) -> None:  # pragma: no cover - hook
        self._start = time.perf_counter()

    def pytest_sessionfinish(self, session: pytest.Session, exitstatus: int) -> None:  # pragma: no cover - hook
        reporter = session.config.pluginmanager.get_plugin("terminalreporter")
        stats = getattr(reporter, "stats", {}) if reporter else {}
        def _count(name: str) -> int:
            items = stats.get(name, [])
            return len(items) if isinstance(items, list) else 0

        duration = time.perf_counter() - self._start
        self.summary = SmokeSummary(
            exit_status=exitstatus,
            tests_collected=session.testscollected,
            passed=_count("passed"),
            failed=_count("failed"),
            skipped=_count("skipped"),
            duration_seconds=duration,
            tag=os.environ.get("RC_TAG"),
        )


def _determine_rc_tag() -> str | None:
    tag = os.environ.get("RC_TAG")
    if tag:
        return tag
    try:
        import tomllib
    except ModuleNotFoundError:  # pragma: no cover - Python <3.11 fallback
        import tomli as tomllib  # type: ignore

    config = tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))
    version = config.get("project", {}).get("version", "0.0.0")
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    tag = f"rc-{version}-{timestamp}"
    os.environ.setdefault("RC_TAG", tag)
    return tag


def _write_reports(summary: SmokeSummary, output_dir: Path) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    base_name = f"smoke_report_{timestamp}"
    json_path = output_dir / f"{base_name}.json"
    md_path = output_dir / f"{base_name}.md"

    json_path.write_text(json.dumps(summary.to_json(), indent=2, ensure_ascii=False), encoding="utf-8")

    md_content = [
        "# Raport testów smoke",
        "",
        f"- Tag RC: `{summary.tag or 'n/a'}`",
        f"- Kod zakończenia: `{summary.exit_status}`",
        "",
        "| Metryka | Wartość |",
        "| --- | --- |",
        f"| Liczba testów | {summary.tests_collected} |",
        f"| Sukcesy | {summary.passed} |",
        f"| Porażki | {summary.failed} |",
        f"| Pominięte | {summary.skipped} |",
        f"| Czas [s] | {round(summary.duration_seconds, 3)} |",
        "",
    ]
    md_path.write_text("\n".join(md_content), encoding="utf-8")
    return json_path, md_path


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Uruchom smoke testy i zapisz raporty")
    parser.add_argument(
        "--report-dir",
        default="reports/smoke",
        help="Katalog wyjściowy raportów (domyślnie reports/smoke)",
    )
    parser.add_argument(
        "--pytest-args",
        nargs=argparse.REMAINDER,
        help="Dodatkowe argumenty przekazywane do pytest po znaczniku --",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    tag = _determine_rc_tag()
    plugin = SmokeReportPlugin()
    pytest_args = ["-m", "smoke", "--maxfail=1", "--disable-warnings", "tests/smoke"]
    if args.pytest_args:
        pytest_args.extend(args.pytest_args)

    exit_code = pytest.main(pytest_args, plugins=[plugin])
    summary = plugin.summary or SmokeSummary(
        exit_status=exit_code,
        tests_collected=0,
        passed=0,
        failed=0,
        skipped=0,
        duration_seconds=0.0,
        tag=tag,
    )
    # Upewnij się, że tag jest zawsze zapisany w raporcie.
    summary.tag = summary.tag or tag
    output_dir = Path(args.report_dir)
    _write_reports(summary, output_dir)
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
