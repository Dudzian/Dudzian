"""Syntetyczne dystrybucje; żaden wheel ani kod natywny nie jest ładowany."""

from __future__ import annotations

import base64
import csv
import hashlib
import io
import json
from pathlib import Path
import subprocess
import sys
import zipfile

import pytest

from scripts import stage9_v5d4_dependencies as deps

pytestmark = [pytest.mark.security, pytest.mark.unit]


def wheel(
    directory: Path,
    name: str,
    selected: str = "1.0",
    requires: tuple[str, ...] = (),
    extras: tuple[str, ...] = (),
    tag: str = "py3-none-any",
    files: dict[str, bytes] | None = None,
    requires_python: str = ">=3.8",
) -> tuple[Path, dict[str, str]]:
    prefix = f"{name}-{selected}.dist-info"
    entries = {
        f"{name}/__init__.py": b"raise RuntimeError('wheel code must never run')\n",
        prefix + "/METADATA": (
            f"Metadata-Version: 2.1\nName: {name}\nVersion: {selected}\nRequires-Python: {requires_python}\n"
            + "".join(f"Requires-Dist: {item}\n" for item in requires)
            + "".join(f"Provides-Extra: {item}\n" for item in extras)
            + "\n"
        ).encode(),
        prefix + "/WHEEL": f"Wheel-Version: 1.0\nRoot-Is-Purelib: true\nTag: {tag}\n".encode(),
        **(files or {}),
    }
    record = io.StringIO()
    writer = csv.writer(record)
    for path, data in entries.items():
        sha = base64.urlsafe_b64encode(hashlib.sha256(data).digest()).rstrip(b"=").decode()
        writer.writerow((path, "sha256=" + sha, len(data)))
    writer.writerow((prefix + "/RECORD", "", ""))
    entries[prefix + "/RECORD"] = record.getvalue().encode()
    path = directory / f"{name}-{selected}-{tag}.whl"
    with zipfile.ZipFile(path, "w") as archive:
        for member, data in entries.items():
            archive.writestr(member, data)
    return path, {
        "url": f"https://example.invalid/{path.name}",
        "source": "TEST ONLY",
        "sha256": deps.digest(path.read_bytes()),
    }


@pytest.fixture
def material(tmp_path: Path) -> tuple[Path, Path, dict[str, dict[str, str]]]:
    project = tmp_path / "pyproject.toml"
    project.write_text(
        '[project]\nrequires-python=">=3.11"\ndependencies=["application[async]>=1,<2"]\n'
        '[project.optional-dependencies]\nobserver-kvm-preflight=["defusedxml==0.7.1"]\n',
        encoding="utf-8",
    )
    directory = tmp_path / "wheels"
    directory.mkdir()
    definitions = [
        (
            "application",
            "1.0",
            (
                "transitive>=2; python_version >= '3.12'",
                "native==1.0; extra == 'async'",
                "missing; sys_platform == 'win32'",
            ),
            ("async",),
            "py3-none-any",
            None,
        ),
        ("transitive", "2.1", (), (), "py3-none-any", None),
        (
            "native",
            "1.0",
            (),
            (),
            "cp39-abi3-manylinux_2_28_x86_64",
            {"native/_binding.abi3.so": b"\x7fELF synthetic not executable"},
        ),
        ("defusedxml", "0.7.1", (), (), "py3-none-any", None),
    ]
    origins = {}
    for name, selected, requires, extras, tag, files in definitions:
        path, origin = wheel(directory, name, selected, requires, extras, tag, files)
        origins[path.name] = origin
    return project, directory, origins


def native_inventory(proposal: dict) -> dict[str, str]:
    return {
        package["filename"] + ":" + member: sha
        for package in proposal["packages"]
        for member, sha in package["native_members"].items()
    }


def test_complete_closure_keeps_contract_and_extras(material) -> None:
    project, directory, origins = material
    proposal = deps.propose(project, directory, origins)
    assert {item["name"] for item in proposal["packages"]} == {
        "application",
        "transitive",
        "native",
        "defusedxml",
    }
    assert proposal["root_requirements"] == ["application[async]>=1,<2", "defusedxml==0.7.1"]
    assert proposal["target"]["soabi"] == "cpython-312-x86_64-linux-gnu"
    result = deps.verify(
        project,
        directory,
        origins,
        proposal,
        deps.digest(deps.canonical(proposal)),
        native_inventory(proposal),
    )
    assert result["status"] == "LOCAL_CONSISTENCY_VERIFIED"
    assert result["authorization"] == "NOT_AUTHORIZED"
    assert result["native_members"] == 1


@pytest.mark.parametrize("name", ["native", "transitive", "defusedxml"])
def test_missing_transitive_or_required_dependency_refuses(material, name: str) -> None:
    project, directory, origins = material
    path = next(directory.glob(name + "-*.whl"))
    path.unlink()
    origins.pop(path.name)
    with pytest.raises(deps.Refusal, match="MISSING_DEPENDENCY:" + name):
        deps.propose(project, directory, origins)


def test_wrong_selected_version_refuses(material) -> None:
    project, directory, origins = material
    project.write_text(
        project.read_text().replace("application[async]>=1,<2", "application[async]>=2,<3")
    )
    with pytest.raises(deps.Refusal, match="DEPENDENCY_VERSION:application"):
        deps.propose(project, directory, origins)


def test_distribution_sha256_refuses(material) -> None:
    project, directory, origins = material
    next(iter(origins.values()))["sha256"] = "0" * 64
    with pytest.raises(deps.Refusal, match="ORIGIN_SHA256"):
        deps.propose(project, directory, origins)


def test_proposal_not_its_own_trust_root(material) -> None:
    project, directory, origins = material
    proposal = deps.propose(project, directory, origins)
    with pytest.raises(deps.Refusal, match="INDEPENDENT_EXPECTATION_MISSING"):
        deps.verify(project, directory, origins, proposal, "", native_inventory(proposal))
    with pytest.raises(deps.Refusal, match="PROPOSAL_SHA256_MISMATCH"):
        deps.verify(project, directory, origins, proposal, "0" * 64, native_inventory(proposal))
    with pytest.raises(deps.Refusal, match="UNAPPROVED_NATIVE_LIBRARY"):
        deps.verify(
            project, directory, origins, proposal, deps.digest(deps.canonical(proposal)), {}
        )


def test_resealed_proposal_cannot_rewrite_package_metadata(material) -> None:
    project, directory, origins = material
    proposal = deps.propose(project, directory, origins)
    proposal["packages"][0]["version"] = "1000"
    with pytest.raises(deps.Refusal, match="PROPOSAL_CONTENT_MISMATCH"):
        deps.verify(
            project,
            directory,
            origins,
            proposal,
            deps.digest(deps.canonical(proposal)),
            native_inventory(proposal),
        )


@pytest.mark.parametrize(
    "tag",
    [
        "cp311-cp311-manylinux_2_28_x86_64",
        "cp312-cp312-manylinux_2_40_x86_64",
        "cp312-cp312-win_amd64",
        "cp313-abi3-manylinux_2_28_x86_64",
        "cp312-cp312-manylinux_0_0_x86_64",
        "cp312-cp312-manylinux_1_99_x86_64",
        "cp312-cp312-manylinux_2_4_x86_64",
    ],
)
def test_wrong_abi_refuses(tmp_path: Path, tag: str) -> None:
    path, origin = wheel(tmp_path, "example", tag=tag)
    with pytest.raises(deps.Refusal, match="WHEEL_ABI_MISMATCH"):
        deps.inspect_wheel(path, origin)


def test_native_member_soabi_overrides_misleading_wheel_tag(tmp_path: Path) -> None:
    path, origin = wheel(
        tmp_path,
        "example",
        tag="cp312-cp312-manylinux_2_28_x86_64",
        files={"example/_native.cpython-311-x86_64-linux-gnu.so": b"\x7fELF"},
    )
    with pytest.raises(deps.Refusal, match="NATIVE_MEMBER_SOABI_MISMATCH"):
        deps.inspect_wheel(path, origin)


def test_package_requires_python_is_enforced(tmp_path: Path) -> None:
    path, origin = wheel(tmp_path, "example", requires_python=">=3.13")
    with pytest.raises(deps.Refusal, match="PYTHON_VERSION_MISMATCH"):
        deps.inspect_wheel(path, origin)


def test_ambiguous_selected_versions_refuse(material) -> None:
    project, directory, origins = material
    path, origin = wheel(directory, "transitive", "2.2")
    origins[path.name] = origin
    with pytest.raises(deps.Refusal, match="AMBIGUOUS_PACKAGE_VERSION"):
        deps.propose(project, directory, origins)


def test_unknown_requested_extra_refuses(material) -> None:
    project, directory, origins = material
    project.write_text(project.read_text().replace("application[async]", "application[unknown]"))
    with pytest.raises(deps.Refusal, match="MISSING_PACKAGE_EXTRA"):
        deps.propose(project, directory, origins)


def test_full_wheelhouse_cannot_include_unreachable_artifacts(material) -> None:
    project, directory, origins = material
    path, origin = wheel(directory, "unused")
    origins[path.name] = origin
    with pytest.raises(deps.Refusal, match="UNREACHABLE_WHEEL"):
        deps.propose(project, directory, origins)


@pytest.mark.parametrize(
    "expression, expected",
    [
        ("python_version >= '3.12' and platform_machine == 'x86_64'", True),
        ("extra == 'binary' or (sys_platform == 'win32' and python_version < '3.0')", False),
        ("python_version not in '2.7,3.8'", True),
        ("extra != 'binary'", True),
    ],
)
def test_marker_parser(expression: str, expected: bool) -> None:
    assert deps.marker(expression, {**deps.TARGET, "extra": ""}) is expected


@pytest.mark.parametrize(
    "expression",
    [
        "platform_release == 'test'",
        "python_version >= frobnicate()",
        "python_version == '3.12' trailing",
        "unknown == 'x'",
        "sys_platform > 'linux'",
    ],
)
def test_unsupported_marker_refuses(expression: str) -> None:
    with pytest.raises(deps.Refusal):
        deps.marker(expression, {**deps.TARGET, "extra": ""})


@pytest.mark.parametrize(
    "actual, specifier, expected",
    [
        ("1.2", "~=1.2", True),
        ("2.0", "~=1.2", False),
        ("1.2.3", "==1.2.*", True),
        ("1.2.4", "!=1.2.*", False),
        ("1.2.post0", ">1.2", False),
        ("1.2.post1", ">1.2.post0", True),
        ("1.2", "==1.2.0", True),
    ],
)
def test_supported_version_profile(actual: str, specifier: str, expected: bool) -> None:
    assert deps.satisfies(actual, specifier) is expected


@pytest.mark.parametrize(
    "member",
    [
        "../escape.py",
        "package/../../escape.py",
        "/absolute.py",
        "package\\escape.py",
        "startup.pth",
    ],
)
def test_unsafe_wheel_members_refuse(tmp_path: Path, member: str) -> None:
    path, origin = wheel(tmp_path, "example", files={member: b"payload"})
    with pytest.raises(deps.Refusal):
        deps.inspect_wheel(path, origin)


def test_filename_metadata_binding_refuses(tmp_path: Path) -> None:
    path, origin = wheel(tmp_path, "example")
    replacement = path.with_name(path.name.replace("example", "imposter"))
    path.rename(replacement)
    with pytest.raises(deps.Refusal, match="WHEEL_FILENAME_METADATA_MISMATCH"):
        deps.inspect_wheel(replacement, origin)


def test_record_integrity_refuses_even_with_updated_distribution_hash(tmp_path: Path) -> None:
    path, origin = wheel(tmp_path, "example")
    with zipfile.ZipFile(path) as archive:
        entries = {name: archive.read(name) for name in archive.namelist()}
    entries["example/__init__.py"] = b"changed source"
    with zipfile.ZipFile(path, "w") as archive:
        for member, value in entries.items():
            archive.writestr(member, value)
    origin["sha256"] = deps.digest(path.read_bytes())
    with pytest.raises(deps.Refusal, match="WHEEL_RECORD_INTEGRITY"):
        deps.inspect_wheel(path, origin)


def test_cli_with_no_site_and_without_importing_wheels(material, tmp_path: Path) -> None:
    project, directory, origins = material
    origin_path = tmp_path / "origins.json"
    origin_path.write_text(json.dumps(origins))
    completed = subprocess.run(
        [
            sys.executable,
            "-I",
            "-S",
            "-B",
            deps.__file__,
            "propose",
            "--pyproject",
            str(project),
            "--wheelhouse",
            str(directory),
            "--origins",
            str(origin_path),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    assert json.loads(completed.stdout)["status"] == "PROPOSAL_REQUIRES_INDEPENDENT_APPROVAL"


def test_real_project_dependencies_are_preserved() -> None:
    import tomllib

    root = Path(__file__).resolve().parents[2]
    project = tomllib.loads((root / "pyproject.toml").read_text())["project"]
    for raw in project["dependencies"] + project["optional-dependencies"]["observer-kvm-preflight"]:
        name, _, specifier, expression = deps.requirement(raw)
        assert name
        deps.satisfies("50.0.1", specifier)
        deps.marker(expression, {**deps.TARGET, "extra": ""})
