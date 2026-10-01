"""Small, stdlib-only helpers used before release dependencies are installed."""

from __future__ import annotations

import re
import tomllib
from pathlib import Path


_PACKAGE_NAME = re.compile(r"^[A-Za-z0-9](?:[A-Za-z0-9._-]*[A-Za-z0-9])?$")
_EXACT_VERSION = re.compile(r"^[^\s,;!=<>~]+$")


def canonicalize_package_name(name: str) -> str:
    """Return the PEP 503 normalized form of a validated project name."""
    if not _PACKAGE_NAME.fullmatch(name):
        raise ValueError(f"invalid package name: {name!r}")
    return re.sub(r"[-_.]+", "-", name).lower()


def locked_versions(path: Path) -> dict[str, str]:
    """Read the deliberately narrow ``name==version [; marker]`` lock format."""
    result: dict[str, str] = {}
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue

        requirement, separator, marker = line.partition(";")
        if separator and not marker.strip():
            raise ValueError(f"empty environment marker in lock entry: {line}")
        if requirement.count("==") != 1:
            raise ValueError(f"non-exact lock entry: {line}")

        name, version = (part.strip() for part in requirement.split("==", 1))
        if not _EXACT_VERSION.fullmatch(version):
            raise ValueError(f"non-exact lock entry: {line}")
        normalized_name = canonicalize_package_name(name)
        if normalized_name in result:
            raise ValueError(f"duplicate lock entry: {normalized_name}")
        result[normalized_name] = version
    return result


def project_requirements(path: Path, extras: tuple[str, ...] = ()) -> list[str]:
    """Read static base and selected optional dependencies without invoking a build backend."""
    document = tomllib.loads(path.read_text(encoding="utf-8"))
    project = document.get("project")
    if not isinstance(project, dict):
        raise ValueError(f"missing [project] table: {path}")
    dynamic = project.get("dynamic", [])
    if not isinstance(dynamic, list) or not all(isinstance(item, str) for item in dynamic):
        raise ValueError(f"invalid project.dynamic: {path}")
    if {"dependencies", "optional-dependencies"}.intersection(dynamic):
        raise ValueError(f"project dependencies must be statically declared: {path}")

    dependencies = project.get("dependencies")
    optional = project.get("optional-dependencies")
    if not isinstance(dependencies, list) or not all(
        isinstance(requirement, str) and requirement.strip() for requirement in dependencies
    ):
        raise ValueError(f"invalid project.dependencies: {path}")
    if not isinstance(optional, dict):
        raise ValueError(f"invalid project.optional-dependencies: {path}")

    result = list(dependencies)
    for extra in extras:
        requirements = optional.get(extra)
        if not isinstance(requirements, list) or not all(
            isinstance(requirement, str) and requirement.strip() for requirement in requirements
        ):
            raise ValueError(f"invalid or missing project extra {extra!r}: {path}")
        result.extend(requirements)
    return result
