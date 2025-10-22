from __future__ import annotations

from pathlib import Path

import re

LOCK_PATH = Path("deploy/packaging/requirements-desktop.lock")
REQUIREMENTS_PATH = Path("deploy/packaging/requirements-desktop.txt")


def _iter_requirements(path: Path) -> list[tuple[str, str]]:
    pattern = re.compile(r"^(?P<name>[A-Za-z0-9_.-]+)==(?P<version>.+)$")
    parsed: list[tuple[str, str]] = []
    for raw_line in path.read_text().splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        match = pattern.match(line)
        if not match:
            raise AssertionError(f"Niepoprawny format wpisu: '{raw_line}'")
        parsed.append((match.group("name"), match.group("version")))
    return parsed


def _parse_requirements(path: Path) -> dict[str, str]:
    packages: dict[str, str] = {}
    for name, version in _iter_requirements(path):
        normalized = name.lower()
        if normalized in packages:
            raise AssertionError(f"Duplikat wpisu dla pakietu: {normalized}")
        packages[normalized] = version
    return packages


def test_lockfile_contains_all_pinned_versions():
    txt_packages = _parse_requirements(REQUIREMENTS_PATH)
    lock_packages = _parse_requirements(LOCK_PATH)

    missing = sorted(name for name in txt_packages if name not in lock_packages)
    assert not missing, (
        "Lockfile nie zawiera wszystkich pakietów z requirements-desktop.txt: "
        + ", ".join(missing)
    )

    mismatched = {
        name: (txt_packages[name], lock_packages[name])
        for name in txt_packages
        if txt_packages[name] != lock_packages[name]
    }
    assert not mismatched, (
        "Wersje w lockfile nie zgadzają się z requirements-desktop.txt: "
        + ", ".join(
            f"{pkg} ({expected} != {actual})"
            for pkg, (expected, actual) in mismatched.items()
        )
    )


def test_lockfile_has_only_pinned_versions():
    parsed = _iter_requirements(LOCK_PATH)
    assert parsed, "Lockfile nie zawiera żadnych pakietów"

    for name, version in parsed:
        assert name and version, "Pusty wpis w lockfile"
        assert "==" not in name, "Nazwa pakietu nie powinna zawierać separatora wersji"
        assert version and " " not in version, "Wersja powinna być pojedynczym tokenem"


def test_lockfile_is_sorted():
    names = [name.lower() for name, _ in _iter_requirements(LOCK_PATH)]
    assert names == sorted(names), "Lockfile powinien być posortowany alfabetycznie"
