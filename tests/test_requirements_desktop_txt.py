"""Walidacja spójności `requirements-desktop.txt` z `pyproject.toml`."""

from __future__ import annotations

from pathlib import Path

import tomllib
from packaging.requirements import Requirement
from packaging.utils import canonicalize_name
from packaging.version import Version


REQUIREMENTS_PATH = Path("deploy/packaging/requirements-desktop.txt")
PYPROJECT_PATH = Path("pyproject.toml")


def _iter_pinned_requirements() -> list[tuple[str, str]]:
    entries: list[tuple[str, str]] = []
    for raw_line in REQUIREMENTS_PATH.read_text().splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue

        requirement = Requirement(line)
        pins = [item.version for item in requirement.specifier if item.operator == "=="]
        if len(pins) != 1:
            raise AssertionError(
                "Każdy wpis w requirements-desktop.txt powinien mieć przypiętą wersję "
                f"(znaleziono: '{raw_line}')"
            )
        entries.append((requirement.name, pins[0]))
    return entries


def _load_pyproject() -> dict:
    return tomllib.loads(PYPROJECT_PATH.read_text())


def _collect_requirements(section: list[str]) -> dict[str, Requirement]:
    collected: dict[str, Requirement] = {}
    for entry in section:
        requirement = Requirement(entry)
        normalized = canonicalize_name(requirement.name)
        collected[normalized] = requirement
    return collected


def test_pinned_versions_cover_pyproject_dependencies() -> None:
    pyproject = _load_pyproject()["project"]
    base_requirements = _collect_requirements(pyproject.get("dependencies", []))
    extras = pyproject.get("optional-dependencies", {})
    desktop_requirements = _collect_requirements(extras.get("desktop", []))

    pinned_entries = _iter_pinned_requirements()

    seen: set[str] = set()
    for name, version in pinned_entries:
        normalized = canonicalize_name(name)
        assert normalized not in seen, f"Duplikat wpisu w requirements-desktop.txt: {name}"
        seen.add(normalized)

        requirement = base_requirements.get(normalized)
        source = "project.dependencies"
        if requirement is None:
            requirement = desktop_requirements.get(normalized)
            source = "project.optional-dependencies.desktop"

        assert requirement is not None, (
            "Pakiet {pkg} nie jest zadeklarowany w pyproject.toml (sekcja dependencies ani "
            "extras.desktop)"
        ).format(pkg=name)

        pinned_version = Version(version)
        assert requirement.specifier.contains(pinned_version, prereleases=True), (
            "Wersja {ver} pakietu {pkg} nie spełnia ograniczeń z {section}: {spec}"
        ).format(
            ver=version,
            pkg=name,
            section=source,
            spec=requirement.specifier or "(brak specyfikatora)",
        )


def test_macos_closure_pins_are_darwin_only() -> None:
    requirements = {
        canonicalize_name(requirement.name): requirement
        for raw in REQUIREMENTS_PATH.read_text().splitlines()
        if (line := raw.strip())
        and not line.startswith("#")
        and canonicalize_name((requirement := Requirement(line)).name)
        in {"dmgbuild", "ds-store", "mac-alias"}
    }
    assert {name: str(requirement.specifier) for name, requirement in requirements.items()} == {
        "dmgbuild": "==1.6.7",
        "ds-store": "==1.3.3",
        "mac-alias": "==2.2.3",
    }
    for requirement in requirements.values():
        assert requirement.marker is not None
        assert requirement.marker.evaluate({"platform_system": "Darwin"})
        assert not requirement.marker.evaluate({"platform_system": "Windows"})
        assert not requirement.marker.evaluate({"platform_system": "Linux"})
