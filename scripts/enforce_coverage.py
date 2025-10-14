"""Weryfikacja progów pokrycia testami dla krytycznych modułów Stage 4."""
from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping
import xml.etree.ElementTree as ET


class CoverageValidationError(RuntimeError):
    """Błąd walidacji pliku pokrycia."""


@dataclass(slots=True)
class _PackageRequirement:
    name: str
    minimum: float | None = None


@dataclass(slots=True)
class CoverageSummary:
    overall: float
    packages: Mapping[str, float]


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--coverage-file",
        default="coverage.xml",
        help="Ścieżka do raportu coverage XML (domyślnie coverage.xml)",
    )
    parser.add_argument(
        "--minimum",
        type=float,
        default=82.5,
        help="Minimalny procent pokrycia dla całości raportu (domyślnie 82.5)",
    )
    parser.add_argument(
        "--package",
        action="append",
        default=[],
        metavar="NAME[=THRESHOLD]",
        help="Pakiet do zweryfikowania wraz z opcjonalnym własnym progiem",
    )
    return parser.parse_args(argv)


def _parse_package_requirements(raw: Iterable[str]) -> list[_PackageRequirement]:
    requirements: list[_PackageRequirement] = []
    for entry in raw:
        if "=" in entry:
            name, threshold = entry.split("=", 1)
            try:
                minimum = float(threshold)
            except ValueError as exc:  # pragma: no cover - defensywne
                raise CoverageValidationError(
                    f"Nieprawidłowa wartość progu dla pakietu '{entry}'"
                ) from exc
            requirements.append(_PackageRequirement(name=name.strip(), minimum=minimum))
        else:
            requirements.append(_PackageRequirement(name=entry.strip()))
    return requirements


def _load_summary(path: Path) -> CoverageSummary:
    if not path.exists():
        raise CoverageValidationError(f"Plik coverage '{path}' nie istnieje")

    try:
        tree = ET.parse(path)
    except ET.ParseError as exc:  # pragma: no cover - niepoprawny XML
        raise CoverageValidationError(f"Nie udało się odczytać XML z {path}") from exc

    root = tree.getroot()
    line_rate = root.get("line-rate") or root.get("line_rate")
    if line_rate is None:
        raise CoverageValidationError("Raport coverage nie zawiera pola line-rate")

    try:
        overall = float(line_rate) * 100.0
    except ValueError as exc:  # pragma: no cover - błędny format
        raise CoverageValidationError("Nieprawidłowa wartość line-rate w raporcie") from exc

    packages_section = root.find("packages")
    package_rates: dict[str, float] = {}
    if packages_section is not None:
        for package in packages_section.findall("package"):
            name = package.get("name")
            rate = package.get("line-rate") or package.get("line_rate")
            if not name or not rate:
                continue
            try:
                package_rates[name] = float(rate) * 100.0
            except ValueError:
                continue

    return CoverageSummary(overall=overall, packages=package_rates)


def _validate(
    summary: CoverageSummary,
    *,
    minimum: float,
    requirements: Iterable[_PackageRequirement],
) -> tuple[bool, list[str]]:
    messages: list[str] = []
    passed = True

    if summary.overall < minimum:
        passed = False
        messages.append(
            f"Pokrycie całkowite {summary.overall:.2f}% poniżej progu {minimum:.2f}%"
        )
    else:
        messages.append(
            f"Pokrycie całkowite {summary.overall:.2f}% (próg {minimum:.2f}%)"
        )

    for requirement in requirements:
        package_rate = summary.packages.get(requirement.name)
        if package_rate is None:
            passed = False
            messages.append(
                f"Brak pakietu '{requirement.name}' w raporcie coverage"
            )
            continue
        threshold = requirement.minimum if requirement.minimum is not None else minimum
        if package_rate < threshold:
            passed = False
            messages.append(
                f"Pakiet {requirement.name}={package_rate:.2f}% poniżej progu {threshold:.2f}%"
            )
        else:
            messages.append(
                f"Pakiet {requirement.name} spełnia próg ({package_rate:.2f}% >= {threshold:.2f}%)"
            )
    return passed, messages


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    requirements = _parse_package_requirements(args.package)
    coverage_path = Path(args.coverage_file)

    try:
        summary = _load_summary(coverage_path)
    except CoverageValidationError as exc:
        print(f"[coverage] {exc}", file=sys.stderr)
        return 2

    passed, messages = _validate(summary, minimum=args.minimum, requirements=requirements)
    for message in messages:
        print(f"[coverage] {message}")
    return 0 if passed else 1


if __name__ == "__main__":  # pragma: no cover - wejście CLI
    raise SystemExit(main())
