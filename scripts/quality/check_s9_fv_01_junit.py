"""Bramka TEST_ONLY: każdy obowiązkowy przypadek S9-FV-01 musi mieć wynik PASS.

JUnit nie dowodzi kwalifikacji produkcji, PostgreSQL ani dwóch UID. Wykaz jest
statycznym materiałem review; nigdy nie powstaje z ocenianego raportu.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from xml.etree import ElementTree

DEFAULT_REQUIRED_CASES = (
    Path(__file__).resolve().parents[2] / "tests/fixtures/s9_fv_01_required_cases.json"
)
_MAX_REPORT_BYTES = 16 * 1024 * 1024
_LEGACY_MODULES = frozenset(
    {
        "tests.security.test_freshness_runtime_ipc",
        "tests.security.test_freshness_runtime_ipc_identity",
        "tests.security.test_freshness_verifier_transport_remediation",
        "tests.security.test_freshness_verifier_pg_remediation",
        "tests.security.test_freshness_verifier_remediation_scope",
    }
)
CaseID = tuple[str, str]


@dataclass(frozen=True)
class GateResult:
    issues: tuple[str, ...]
    required_cases: int
    passed_cases: int = 0
    legacy_skipped_cases: int = 0

    @property
    def blocked(self) -> bool:
        return bool(self.issues)


def load_required_cases(path: Path = DEFAULT_REQUIRED_CASES) -> frozenset[CaseID]:
    """Odczytaj niepusty wykaz dokładnych tożsamości, również parametrów pytest."""
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ValueError("Nie można odczytać wykazu obowiązkowych przypadków") from exc
    if (
        type(value) is not dict
        or set(value) != {"schema_version", "cases"}
        or type(value["schema_version"]) is not int
        or value["schema_version"] != 1
        or type(value["cases"]) is not list
        or not value["cases"]
    ):
        raise ValueError("Niepoprawny lub pusty wykaz obowiązkowych przypadków")
    cases: set[CaseID] = set()
    for item in value["cases"]:
        if (
            type(item) is not dict
            or set(item) != {"classname", "name"}
            or any(type(item[key]) is not str or not item[key].strip() for key in item)
        ):
            raise ValueError("Niekompletna tożsamość obowiązkowego przypadku")
        identity = (item["classname"], item["name"])
        if identity in cases:
            raise ValueError("Powtórzona tożsamość obowiązkowego przypadku")
        cases.add(identity)
    return frozenset(cases)


def _counter_issues(root: ElementTree.Element) -> list[str]:
    issues = []
    for suite in root.iter():
        if suite.tag not in {"testsuites", "testsuite"}:
            continue
        cases = list(suite.iter("testcase"))
        actual = {
            "tests": len(cases),
            "skipped": sum(case.find("skipped") is not None for case in cases),
            "failures": sum(case.find("failure") is not None for case in cases),
            "errors": sum(case.find("error") is not None for case in cases),
            "disabled": 0,
        }
        for name, count in actual.items():
            declared = suite.get(name)
            if declared is not None and (
                not declared.isascii()
                or not declared.isdigit()
                or len(declared) > 10
                or int(declared) != count
            ):
                issues.append(f"Niespójny licznik JUnit {name}={declared!r}, oczekiwano {count}")
    return issues


def validate_report(
    report: Path, required: frozenset[CaseID], *, pytest_exit_code: int = 0
) -> GateResult:
    """Odmów przy braku wykonania; starsze skipped nie zastępują wymaganych PASS."""
    issues = []
    if not required:
        issues.append("Pusty wykaz obowiązkowych przypadków")
    if pytest_exit_code != 0:
        issues.append(f"pytest nie zakończył się sukcesem: exit {pytest_exit_code}")
    try:
        with report.open("rb") as stream:
            raw = stream.read(_MAX_REPORT_BYTES + 1)
        if len(raw) > _MAX_REPORT_BYTES:
            raise ValueError("Raport JUnit przekracza limit rozmiaru")
        xml = raw.decode("utf-8")
        if "<!DOCTYPE" in xml.upper() or "<!ENTITY" in xml.upper():
            raise ValueError("Deklaracje DTD/ENTITY nie należą do raportu JUnit")
        root = ElementTree.fromstring(xml)
        if root.tag not in {"testsuite", "testsuites"}:
            raise ValueError("Niepoprawny element główny JUnit")
    except (OSError, ValueError, ElementTree.ParseError) as exc:
        issues.append(f"Nie można ocenić raportu JUnit: {exc}")
        return GateResult(tuple(issues), len(required))

    issues.extend(_counter_issues(root))
    for suites in root.iter("testsuites"):
        if any(child.tag not in {"testsuite", "testsuites"} for child in suites):
            issues.append("Niepoprawna struktura testsuites")
    for suite in root.iter("testsuite"):
        if any(
            child.tag not in {"testcase", "testsuite", "properties", "system-out", "system-err"}
            for child in suite
        ):
            issues.append("Niepoprawna struktura testsuite")
    parents = {child: parent for parent in root.iter() for child in parent}
    seen: set[CaseID] = set()
    passed: set[CaseID] = set()
    legacy_skipped = 0
    required_modules = {classname for classname, _ in required}
    for case in root.iter("testcase"):
        classname, name = case.get("classname", ""), case.get("name", "")
        if not classname.strip() or not name.strip():
            issues.append("Niekompletna tożsamość testcase w JUnit")
            continue
        identity = (classname, name)
        label = f"{classname}::{name}"
        if identity in seen:
            issues.append(f"Powtórzony przypadek: {label}")
        seen.add(identity)
        if parents[case].tag != "testsuite" or any(
            child.tag
            not in {"skipped", "failure", "error", "properties", "system-out", "system-err"}
            for child in case
        ):
            issues.append(f"Niekompletna lub niepoprawna struktura przypadku: {label}")
        skipped = case.find("skipped") is not None
        failed = case.find("failure") is not None or case.find("error") is not None
        incomplete = (
            case.get("status", "run") not in {"run", "passed"}
            or case.get("result", "passed") not in {"passed", "completed", "success"}
            or case.get("disabled", "false") != "false"
        )
        if failed:
            issues.append(f"Nieudany przypadek: {label}")
        if incomplete:
            issues.append(f"Brak zakończonego wykonania: {label}")
        if identity in required:
            if skipped:
                issues.append(f"Pominięty obowiązkowy przypadek: {label}")
            elif not failed and not incomplete:
                passed.add(identity)
        elif classname in required_modules:
            issues.append(f"Przypadek spoza zatwierdzonego wykazu: {label}")
        elif skipped:
            if classname in _LEGACY_MODULES:
                legacy_skipped += 1
            else:
                issues.append(f"Pominięcie poza starszym zakresem regresji: {label}")
    for classname, name in sorted(required - seen):
        issues.append(f"Brak obowiązkowego przypadku: {classname}::{name}")
    return GateResult(tuple(issues), len(required), len(passed), legacy_skipped)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", type=Path)
    parser.add_argument("--pytest-exit-code", type=int, default=0)
    args = parser.parse_args(argv)
    try:
        required = load_required_cases()
    except ValueError as exc:
        print(f"S9-FV-01: BLOCKED — {exc}")
        return 1
    result = validate_report(args.report, required, pytest_exit_code=args.pytest_exit_code)
    state = "BLOCKED" if result.blocked else "PASS"
    print(
        f"S9-FV-01: {state} — wymagane={result.required_cases}, "
        f"PASS={result.passed_cases}, starsze_SKIPPED={result.legacy_skipped_cases}"
    )
    for issue in result.issues:
        print(f"- {issue}")
    return int(result.blocked)


if __name__ == "__main__":
    raise SystemExit(main())
