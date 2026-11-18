"""Sprawdź wykonanie i kompletność raportu, niezależnie od liczby znalezisk."""

from __future__ import annotations

import argparse
from collections import Counter
import json
import subprocess
import sys
from pathlib import Path

SEMGREP_VERSION = "1.179.0"
MAX_REPORT_BYTES = 64 * 1024 * 1024

# To są filtry wyboru celu/reguły, a nie błędy analizy. Nie dodawać tutaj
# timeoutów, problemów parsera, uprawnień ani limitów zasobów.
TARGET_SELECTION_REASONS = frozenset(
    {
        "always_skipped",
        "semgrepignore_patterns_match",
        "cli_include_flags_do_not_match",
        "cli_exclude_flags_match",
        "excluded_by_config",
        "wrong_language",
        "irrelevant_rule",
        "Gitignore_patterns_match",
        "Dotfile",
    }
)
ISSUE_CATEGORIES = frozenset(
    {
        "scanner_error",
        "incomplete_target",
        "skipped_rule",
        "blocking_finding",
        "report_invalid",
        "report_unavailable",
        "scanner_exit",
        "scanner_unavailable",
    }
)


def _unique_json_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    value: dict[str, object] = {}
    for key, item in pairs:
        if key in value:
            raise ValueError("powtórzony klucz JSON")
        value[key] = item
    return value


def _load_report(report: Path) -> object:
    with report.open("rb") as stream:
        raw = stream.read(MAX_REPORT_BYTES + 1)
    if len(raw) > MAX_REPORT_BYTES:
        raise ValueError("raport przekracza limit odczytu")
    return json.loads(raw, object_pairs_hook=_unique_json_object)


def validate_report(report: Path, *, blocking: bool) -> list[str]:
    issues: list[str] = []
    try:
        data = _load_report(report)
        if not isinstance(data, dict) or data.get("version") != SEMGREP_VERSION:
            raise ValueError("brak rozpoznanej wersji raportu")
        for name in ("results", "errors", "skipped_rules"):
            if not isinstance(data.get(name), list):
                raise ValueError(f"brak poprawnego pola {name}")
        paths = data.get("paths")
        if not isinstance(paths, dict):
            raise ValueError("brak informacji o pokryciu")
        scanned, skipped = paths.get("scanned"), paths.get("skipped")
        if (
            not isinstance(scanned, list)
            or not scanned
            or any(not isinstance(path, str) or not path for path in scanned)
            or not isinstance(skipped, list)
        ):
            raise ValueError("brak pełnego wykazu scanned/skipped lub pusty skan")
        for error in data["errors"]:
            if not isinstance(error, dict):
                raise ValueError("niepoprawny wpis błędu skanera")
            issues.append("scanner_error")
        for item in skipped:
            if not isinstance(item, dict) or not isinstance(item.get("path"), str):
                raise ValueError("niepoprawny wpis pominiętego celu")
            reason = item.get("reason")
            if not isinstance(reason, str) or reason not in TARGET_SELECTION_REASONS:
                issues.append("incomplete_target")
        if data["skipped_rules"]:
            issues.extend(["skipped_rule"] * len(data["skipped_rules"]))
        if blocking and data["results"]:
            issues.extend(["blocking_finding"] * len(data["results"]))
    except OSError:
        issues.append("report_unavailable")
    except (ValueError, TypeError):
        issues.append("report_invalid")
    return issues


def write_summary(
    report: Path, issues: list[str], scanner_exit_code: int | None
) -> dict[str, object]:
    """Publikuj tylko liczniki i stałe kategorie, nigdy treść danych skanera."""
    summary: dict[str, object] = {
        "status": "REFUSED" if issues else "COMPLETE",
        "scanner_exit_code": scanner_exit_code,
        "issue_counts": dict(
            Counter(issue if issue in ISSUE_CATEGORIES else "report_invalid" for issue in issues)
        ),
    }
    try:
        data = _load_report(report)
        if isinstance(data, dict):
            for name in ("results", "errors", "skipped_rules"):
                value = data.get(name)
                summary[name + "_count"] = len(value) if isinstance(value, list) else None
            paths = data.get("paths")
            if isinstance(paths, dict):
                for name in ("scanned", "skipped"):
                    value = paths.get(name)
                    summary[name + "_count"] = len(value) if isinstance(value, list) else None
                skipped = paths.get("skipped")
                if isinstance(skipped, list):
                    categories: Counter[str] = Counter()
                    for item in skipped:
                        reason = item.get("reason") if isinstance(item, dict) else None
                        category = (
                            reason
                            if isinstance(reason, str) and reason in TARGET_SELECTION_REASONS
                            else "incomplete_or_unknown"
                        )
                        categories[category] += 1
                    summary["target_selection_counts"] = dict(categories)
    except (OSError, ValueError, TypeError):
        # Przyczyna jest już w issues; nie zastępuj awarii pustym wynikiem.
        pass
    report.with_suffix(".summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--config", action="append", required=True)
    parser.add_argument("--exclude", action="append", default=[])
    parser.add_argument("--blocking", action="store_true")
    parser.add_argument("targets", nargs="+")
    args = parser.parse_args(argv)
    command = [
        "semgrep",
        "scan",
        "--strict",
        "--verbose",
        "--metrics=off",
        "--disable-version-check",
        "--max-target-bytes=0",
        "--json",
        "--output",
        str(args.report),
    ]
    for config in args.config:
        command.extend(("--config", config))
    for excluded in args.exclude:
        command.extend(("--exclude", excluded))
    if args.blocking:
        command.append("--error")
    command.extend(args.targets)
    scanner_exit_code = None
    try:
        # Stary raport nie może zastąpić wyniku awarii nowego procesu.
        args.report.unlink(missing_ok=True)
        args.report.with_suffix(".summary.json").unlink(missing_ok=True)
        # Semgrep potrafi wypisać kod także przy --json --output. Surowe logi
        # pozostają obok raportu, poza publicznym logiem i uploadem Actions.
        with (
            args.report.with_suffix(".stdout.log").open("wb") as stdout,
            args.report.with_suffix(".stderr.log").open("wb") as stderr,
        ):
            completed = subprocess.run(command, check=False, stdout=stdout, stderr=stderr)
        scanner_exit_code = completed.returncode
        issues = validate_report(args.report, blocking=args.blocking)
        if scanner_exit_code:
            issues.insert(0, "scanner_exit")
    except OSError:
        issues = ["scanner_unavailable"]
    try:
        summary = write_summary(args.report, issues, scanner_exit_code)
    except OSError:
        print("SEMGREP_INCOMPLETE: nie zapisano podsumowania", file=sys.stderr)
        return 2
    # Jedyny publiczny log to ten sam zredagowany zestaw liczników co artefakt.
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))
    if issues:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
