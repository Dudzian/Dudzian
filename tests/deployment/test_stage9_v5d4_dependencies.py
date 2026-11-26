"""Syntetyczne dystrybucje; żaden wheel ani kod natywny nie jest ładowany."""

from __future__ import annotations

import base64
from collections.abc import Callable
import copy
import csv
import hashlib
import io
import json
import os
import stat
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
    *,
    include_package_init: bool = True,
) -> tuple[Path, dict[str, str]]:
    prefix = f"{name}-{selected}.dist-info"
    package_source = (
        {f"{name}/__init__.py": b"raise RuntimeError('wheel code must never run')\n"}
        if include_package_init
        else {}
    )
    entries = {
        **package_source,
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
            # ZipInfo(member) normalizuje ukośniki na Windows i ucina NUL.
            # Fixture musi zachować rzeczywiste niepoprawne bajty członka.
            info = zipfile.ZipInfo()
            info.filename = member
            info.orig_filename = member
            info.create_system = 3
            info.external_attr = (stat.S_IFREG | 0o644) << 16
            archive.writestr(info, data)
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


def test_narrow_profile_cannot_replace_full_native_closure(material) -> None:
    project, directory, origins = material
    full = deps.propose(project, directory, origins)
    narrow = {
        **full,
        "profile": "stage9-v5d10-observer-preflight-proposal-v1",
        "packages": [package for package in full["packages"] if package["name"] == "defusedxml"],
    }
    for profile in (narrow["profile"], full["profile"]):
        proposal = {**narrow, "profile": profile}
        with pytest.raises(ValueError, match="PROPOSAL_CONTENT_MISMATCH"):
            deps.verify(
                project, directory, origins, proposal, deps.digest(deps.canonical(proposal)), {}
            )
    with pytest.raises(ValueError, match="UNAPPROVED_NATIVE_LIBRARY"):
        deps.verify(project, directory, origins, full, deps.digest(deps.canonical(full)), {})


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


@pytest.mark.parametrize("member", ["package\\escape.py", "package/\x00escape.py"])
def test_wheel_fixture_preserves_unsafe_raw_member_names(tmp_path: Path, member: str) -> None:
    path, origin = wheel(tmp_path, "example", files={member: b"inert"})
    with zipfile.ZipFile(path) as archive:
        assert member in {item.orig_filename for item in archive.infolist()}
    with pytest.raises(deps.Refusal, match="^WHEEL_MEMBER_PATH$"):
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


CACHE_MEMBERS = (
    "PySide6/support/__pycache__/__init__.cpython-311.pyc",
    "PySide6/support/__pycache__/deprecated.cpython-311.pyc",
)


def cache_material() -> dict[str, bytes]:
    """Nieprzezroczyste dane testowe; brak kompilacji, marshal i wykonania."""
    files: dict[str, bytes] = {}
    for member, size in zip(CACHE_MEMBERS, (223, 605), strict=True):
        source = member.replace("/__pycache__/", "/").replace(".cpython-311.pyc", ".py")
        files[source] = b"#" + b"x" * (size - 2) + b"\n"
        header = (
            bytes.fromhex("a70d0d0a")
            + (0).to_bytes(4, "little")
            + (1774939357).to_bytes(4, "little")
            + size.to_bytes(4, "little")
        )
        files[member] = header + b"TEST ONLY: opaque non-marshal payload; never execute"
    return files


def pin_test_cache(monkeypatch: pytest.MonkeyPatch, path: Path, files: dict[str, bytes]) -> None:
    """Podstawienie polityki dotyczy wyłącznie syntetycznego wheel w tym teście."""
    policy = copy.deepcopy(deps.BYTECODE_POLICY)
    policy.update(filename=path.name, sha256=deps.digest(path.read_bytes()))
    policy["members"] = {
        member: {
            "sha256": deps.digest(files[member]),
            "source": member.replace("/__pycache__/", "/").replace(".cpython-311.pyc", ".py"),
            "source_sha256": deps.digest(
                files[member.replace("/__pycache__/", "/").replace(".cpython-311.pyc", ".py")]
            ),
            "header": files[member][:16].hex(),
        }
        for member in CACHE_MEMBERS
    }
    monkeypatch.setattr(deps, "BYTECODE_POLICY", policy)


@pytest.fixture
def foreign_cache(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    files = cache_material()
    path, origin = wheel(
        tmp_path,
        "pyside6_essentials",
        "6.10.3",
        tag="cp39-abi3-manylinux_2_34_x86_64",
        files=files,
    )
    pin_test_cache(monkeypatch, path, files)
    return path, origin, files


def rewrite_wheel(path: Path, updates: dict[str, bytes]) -> None:
    """Zmiana ZIP zachowuje pierwotny RECORD, aby badać odmowy integralności."""
    with zipfile.ZipFile(path) as archive:
        entries = {name: archive.read(name) for name in archive.namelist()}
    entries.update(updates)
    with zipfile.ZipFile(path, "w") as archive:
        for name, content in entries.items():
            archive.writestr(name, content)


def test_exact_foreign_cache_is_reported_and_excluded_from_materialization(foreign_cache) -> None:
    path, origin, files = foreign_cache
    before = path.read_bytes()
    package = deps.inspect_wheel(path, origin)
    omitted = {item["member"]: item for item in package["omitted_members"]}
    assert set(omitted) == set(CACHE_MEMBERS)
    for member in CACHE_MEMBERS:
        assert omitted[member]["distribution"] == "pyside6-essentials"
        assert omitted[member]["version"] == "6.10.3"
        assert omitted[member]["sha256"] == deps.digest(files[member])
        assert omitted[member]["source"] in package["materialized_members"]
        assert omitted[member]["source_sha256"] == deps.digest(files[omitted[member]["source"]])
        assert member not in package["materialized_members"]
    assert path.read_bytes() == before
    with zipfile.ZipFile(path) as archive:
        assert set(CACHE_MEMBERS) <= set(archive.namelist())


def test_synthetic_cache_does_not_match_production_pin(tmp_path: Path) -> None:
    path, origin = wheel(
        tmp_path,
        "pyside6_essentials",
        "6.10.3",
        tag="cp39-abi3-manylinux_2_34_x86_64",
        files=cache_material(),
    )
    with pytest.raises(deps.Refusal):
        deps.inspect_wheel(path, origin)


@pytest.mark.parametrize("member", CACHE_MEMBERS)
def test_record_is_checked_for_each_cache_before_omission(
    foreign_cache, monkeypatch: pytest.MonkeyPatch, member: str
) -> None:
    path, origin, _ = foreign_cache
    rewrite_wheel(path, {member: b"changed bytes with stale RECORD"})
    origin["sha256"] = deps.digest(path.read_bytes())
    monkeypatch.setitem(deps.BYTECODE_POLICY, "sha256", origin["sha256"])
    with pytest.raises(deps.Refusal, match="WHEEL_RECORD_INTEGRITY"):
        deps.inspect_wheel(path, origin)


@pytest.mark.parametrize("member", ["startup.pth", "unapproved.pyc"])
def test_record_integrity_precedes_startup_policy(tmp_path: Path, member: str) -> None:
    path, origin = wheel(tmp_path, "example", files={member: b"original"})
    rewrite_wheel(path, {member: b"changed"})
    origin["sha256"] = deps.digest(path.read_bytes())
    with pytest.raises(deps.Refusal, match="WHEEL_RECORD_INTEGRITY"):
        deps.inspect_wheel(path, origin)


def test_full_metadata_is_checked_before_startup_policy(tmp_path: Path) -> None:
    path, origin = wheel(
        tmp_path,
        "example",
        files={
            "unapproved.pyc": b"opaque",
            "example-1.0.dist-info/METADATA": b"Metadata-Version: 2.1\nVersion: 1.0\n\n",
        },
    )
    with pytest.raises(deps.Refusal, match="WHEEL_METADATA_DUPLICATE_OR_MISSING"):
        deps.inspect_wheel(path, origin)


@pytest.mark.parametrize("member", CACHE_MEMBERS)
def test_foreign_cache_requires_corresponding_source(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, member: str
) -> None:
    complete = cache_material()
    source = member.replace("/__pycache__/", "/").replace(".cpython-311.pyc", ".py")
    files = {name: content for name, content in complete.items() if name != source}
    path, origin = wheel(
        tmp_path,
        "pyside6_essentials",
        "6.10.3",
        tag="cp39-abi3-manylinux_2_34_x86_64",
        files=files,
    )
    pin_test_cache(monkeypatch, path, complete)
    with pytest.raises(deps.Refusal):
        deps.inspect_wheel(path, origin)


@pytest.mark.parametrize(
    "extra",
    [
        "PySide6/support/__pycache__/unknown.cpython-311.pyc",
        "PySide6/support/legacy.pyc",
        "PySide6/support/legacy.pyo",
        "PySide6/support/__pycache__/payload.txt",
        "startup.pth",
        "sitecustomize.py",
        "usercustomize.py",
        "pyside6_essentials-6.10.3.data/scripts/run",
    ],
)
def test_foreign_cache_exception_does_not_admit_extra_startup_members(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, extra: str
) -> None:
    files = {**cache_material(), extra: b"raise RuntimeError('must never run')\n"}
    path, origin = wheel(
        tmp_path,
        "pyside6_essentials",
        "6.10.3",
        tag="cp39-abi3-manylinux_2_34_x86_64",
        files=files,
    )
    pin_test_cache(monkeypatch, path, files)
    with pytest.raises(deps.Refusal):
        deps.inspect_wheel(path, origin)


@pytest.mark.parametrize("change", ["magic", "cache_tag", "flags", "source_size", "source_bytes"])
def test_foreign_cache_requires_exact_version_header_and_source(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, change: str
) -> None:
    approved = cache_material()
    files = approved.copy()
    member = CACHE_MEMBERS[0]
    source = "PySide6/support/__init__.py"
    if change == "magic":
        files[member] = bytes.fromhex("cb0d0d0a") + files[member][4:]
    elif change == "cache_tag":
        files[member.replace("cpython-311", "cpython-312")] = files.pop(member)
    elif change == "flags":
        files[member] = files[member][:4] + (1).to_bytes(4, "little") + files[member][8:]
    elif change == "source_size":
        files[member] = files[member][:12] + (224).to_bytes(4, "little") + files[member][16:]
    else:
        files[source] = b"!" + files[source][1:]
    path, origin = wheel(
        tmp_path,
        "pyside6_essentials",
        "6.10.3",
        tag="cp39-abi3-manylinux_2_34_x86_64",
        files=files,
    )
    pin_test_cache(monkeypatch, path, approved)
    with pytest.raises(deps.Refusal):
        deps.inspect_wheel(path, origin)


@pytest.mark.parametrize("change", ["cpython312_magic", "hash_based_flags", "source_size"])
def test_foreign_cache_header_checks_remain_even_with_test_pins(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, change: str
) -> None:
    files = cache_material()
    member = CACHE_MEMBERS[0]
    if change == "cpython312_magic":
        files[member] = bytes.fromhex("cb0d0d0a") + files[member][4:]
    elif change == "hash_based_flags":
        files[member] = files[member][:4] + (1).to_bytes(4, "little") + files[member][8:]
    else:
        files["PySide6/support/__init__.py"] += b"# changed source size\n"
    path, origin = wheel(
        tmp_path,
        "pyside6_essentials",
        "6.10.3",
        tag="cp39-abi3-manylinux_2_34_x86_64",
        files=files,
    )
    # Zgodne hashe testowe nie zatwierdzają odmiennego formatu bytecode.
    pin_test_cache(monkeypatch, path, files)
    with pytest.raises(deps.Refusal, match="WHEEL_BYTECODE_PIN_MISMATCH"):
        deps.inspect_wheel(path, origin)


@pytest.mark.parametrize("missing", CACHE_MEMBERS)
def test_foreign_cache_policy_requires_complete_exact_pair(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, missing: str
) -> None:
    approved = cache_material()
    files = {name: content for name, content in approved.items() if name != missing}
    path, origin = wheel(
        tmp_path,
        "pyside6_essentials",
        "6.10.3",
        tag="cp39-abi3-manylinux_2_34_x86_64",
        files=files,
    )
    pin_test_cache(monkeypatch, path, approved)
    with pytest.raises(deps.Refusal):
        deps.inspect_wheel(path, origin)


@pytest.mark.parametrize("name", ["other", "pyside6_addons"])
def test_foreign_cache_policy_is_distribution_specific(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, name: str
) -> None:
    files = cache_material()
    path, origin = wheel(
        tmp_path, name, "6.10.3", tag="cp39-abi3-manylinux_2_34_x86_64", files=files
    )
    pin_test_cache(monkeypatch, path, files)
    with pytest.raises(deps.Refusal):
        deps.inspect_wheel(path, origin)


def test_omission_expectation_is_separate_and_complete(foreign_cache, tmp_path: Path) -> None:
    path, origin, _ = foreign_cache
    project = tmp_path / "pyproject.toml"
    project.write_text(
        '[project]\nrequires-python=">=3.11"\ndependencies=["pyside6-essentials==6.10.3"]\n'
        "[project.optional-dependencies]\nobserver-kvm-preflight=[]\n",
        encoding="utf-8",
    )
    directory = tmp_path / "wheelhouse"
    directory.mkdir()
    moved = directory / path.name
    path.rename(moved)
    origins = {moved.name: origin}
    proposal = deps.propose(project, directory, origins)
    expected_sha = deps.digest(deps.canonical(proposal))
    approved = deps.omission_expectation(proposal)
    assert approved["installation_contract_exception"] == "OMIT_PINNED_FOREIGN_BYTECODE"
    assert {item["member"] for item in approved["members"]} == set(CACHE_MEMBERS)
    with pytest.raises(deps.Refusal):
        deps.verify(project, directory, origins, proposal, expected_sha, {})
    for altered in (
        {**approved, "members": approved["members"][:1]},
        {**approved, "members": approved["members"] + [approved["members"][0]]},
        {**approved, "installation_contract_exception": "AUTOMATICALLY_APPROVED"},
        {
            **approved,
            "members": [
                {**approved["members"][0], "sha256": "0" * 64},
                approved["members"][1],
            ],
        },
    ):
        with pytest.raises(deps.Refusal):
            deps.verify(project, directory, origins, proposal, expected_sha, {}, altered)
    result = deps.verify(project, directory, origins, proposal, expected_sha, {}, approved)
    assert result["status"] == "LOCAL_CONSISTENCY_VERIFIED"
    assert result["authorization"] == "NOT_AUTHORIZED"


@pytest.mark.parametrize("member", ["package//x.py", "package/./x.py", "C:/x.py", "package/\x00x"])
def test_noncanonical_or_ambiguous_member_paths_refuse(tmp_path: Path, member: str) -> None:
    path, origin = wheel(tmp_path, "example", files={member: b"inert"})
    with pytest.raises((deps.Refusal, zipfile.BadZipFile)):
        deps.inspect_wheel(path, origin)


def test_zip_symlink_refuses_without_following_target(tmp_path: Path) -> None:
    path, origin = wheel(tmp_path, "example", files={"example/link.py": b"../../outside"})
    with zipfile.ZipFile(path) as archive:
        entries = [(item, archive.read(item)) for item in archive.infolist()]
    with zipfile.ZipFile(path, "w") as archive:
        for item, data in entries:
            if item.filename == "example/link.py":
                item.create_system = 3
                item.external_attr = (stat.S_IFLNK | 0o777) << 16
            archive.writestr(item, data)
    origin["sha256"] = deps.digest(path.read_bytes())
    with pytest.raises(deps.Refusal):
        deps.inspect_wheel(path, origin)


def test_zip_local_filename_disagreement_refuses(tmp_path: Path) -> None:
    path, origin = wheel(tmp_path, "example", files={"example/safe.py": b"inert"})
    data = path.read_bytes()
    original = b"example/safe.py"
    offset = data.index(original)
    changed = data[:offset] + b"example/evil.py" + data[offset + len(original) :]
    path.write_bytes(changed)
    origin["sha256"] = deps.digest(changed)
    with pytest.raises((deps.Refusal, zipfile.BadZipFile)):
        deps.inspect_wheel(path, origin)


def test_zip_duplicate_member_refuses(tmp_path: Path) -> None:
    path, origin = wheel(tmp_path, "example")
    with pytest.warns(UserWarning, match="Duplicate name"):
        with zipfile.ZipFile(path, "a") as archive:
            archive.writestr("example/__init__.py", b"different bytes")
    origin["sha256"] = deps.digest(path.read_bytes())
    with pytest.raises(deps.Refusal):
        deps.inspect_wheel(path, origin)


@pytest.mark.parametrize("second", [b"same", b"different"])
def test_cross_wheel_collision_requires_separate_review(tmp_path: Path, second: bytes) -> None:
    directory = tmp_path / "wheels"
    directory.mkdir()
    origins = {}
    for name, data in (("essentials", b"same"), ("addons", second)):
        path, origin = wheel(directory, name, files={"shared/module.py": data})
        origins[path.name] = origin
    project = tmp_path / "pyproject.toml"
    project.write_text(
        '[project]\nrequires-python=">=3.11"\ndependencies=["essentials", "addons"]\n'
        "[project.optional-dependencies]\nobserver-kvm-preflight=[]\n",
        encoding="utf-8",
    )
    with pytest.raises(deps.Refusal):
        deps.propose(project, directory, origins)


@pytest.mark.parametrize("other", ["shared.py", "shared.py/child.py"])
def test_relocated_or_ancestor_materialization_collision_refuses(
    tmp_path: Path, other: str
) -> None:
    path, origin = wheel(
        tmp_path,
        "example",
        files={"example-1.0.data/platlib/shared.py": b"first", other: b"second"},
    )
    with pytest.raises(deps.Refusal):
        deps.inspect_wheel(path, origin)


@pytest.mark.parametrize(
    "member",
    [
        "sitecustomize/__init__.py",
        "usercustomize/__init__.py",
        "namespace/sitecustomize/helper.py",
        "namespace/usercustomize/helper.py",
        "sitecustomize.abi3.so",
        "usercustomize.cpython-312-x86_64-linux-gnu.so",
        "namespace/sitecustomize.py/data.txt",
        "namespace/usercustomize.pyi/data.txt",
        "example-1.0.data/platlib/sitecustomize/__init__.py",
    ],
)
def test_customization_modules_and_package_components_are_startup_hooks(
    tmp_path: Path, member: str
) -> None:
    path, origin = wheel(tmp_path, "example", files={member: b"inert source"})
    with pytest.raises(deps.Refusal, match="WHEEL_STARTUP_CODE"):
        deps.inspect_wheel(path, origin)


@pytest.mark.parametrize(
    "header",
    [
        "",
        "Metadata-Version: 2.1\nMetadata-Version: 2.1\n",
        "Metadata-Version: 1.0\n",
        "Metadata-Version: 1.1\n",
        "Metadata-Version: 1.2\n",
        "Metadata-Version: 2.0\n",
        "Metadata-Version: 2.5\n",
        "Metadata-Version: unknown\n",
    ],
)
def test_metadata_version_must_be_single_and_supported(tmp_path: Path, header: str) -> None:
    path, origin = wheel(
        tmp_path,
        "example",
        files={
            "example-1.0.dist-info/METADATA": (
                header + "Name: example\nVersion: 1.0\nRequires-Python: >=3.8\n\n"
            ).encode()
        },
    )
    with pytest.raises(deps.Refusal):
        deps.inspect_wheel(path, origin)


@pytest.mark.parametrize("selected", ["2.1", "2.2", "2.3", "2.4"])
def test_supported_metadata_versions_remain_accepted(tmp_path: Path, selected: str) -> None:
    path, origin = wheel(
        tmp_path,
        "example",
        files={
            "example-1.0.dist-info/METADATA": (
                f"Metadata-Version: {selected}\nName: example\nVersion: 1.0\n\n"
            ).encode()
        },
    )
    assert deps.inspect_wheel(path, origin)["name"] == "example"


@pytest.mark.parametrize(
    "magic",
    [
        b"MZ\x90\x00",
        bytes.fromhex("feedface"),
        bytes.fromhex("cefaedfe"),
        bytes.fromhex("feedfacf"),
        bytes.fromhex("cffaedfe"),
        bytes.fromhex("cafebabe"),
        bytes.fromhex("bebafeca"),
        bytes.fromhex("cafebabf"),
        bytes.fromhex("bfbafeca"),
    ],
)
def test_foreign_executable_magic_is_not_accepted_as_data(tmp_path: Path, magic: bytes) -> None:
    path, origin = wheel(tmp_path, "example", files={"example/payload.bin": magic + b"inert"})
    with pytest.raises(deps.Refusal):
        deps.inspect_wheel(path, origin)


@pytest.mark.parametrize("suffix", [".exe", ".COM", ".bat", ".CMD", ".ps1", ".SH"])
def test_script_or_foreign_executable_suffix_refuses(tmp_path: Path, suffix: str) -> None:
    path, origin = wheel(tmp_path, "example", files={"example/payload" + suffix: b"inert"})
    with pytest.raises(deps.Refusal):
        deps.inspect_wheel(path, origin)


@pytest.mark.parametrize("member", ["example/helper", "example/helper.txt"])
def test_non_python_shebang_refuses(tmp_path: Path, member: str) -> None:
    path, origin = wheel(tmp_path, "example", files={member: b"#!/bin/sh\nexit 99\n"})
    with pytest.raises(deps.Refusal):
        deps.inspect_wheel(path, origin)


def test_python_source_shebang_is_inventoried_without_execution(tmp_path: Path) -> None:
    member = "example/helper.py"
    source = b"#!/usr/bin/python3\nraise RuntimeError('must never run')\n"
    path, origin = wheel(tmp_path, "example", files={member: source})
    package = deps.inspect_wheel(path, origin)
    assert package["materialized_members"][member] == {
        "path": member,
        "sha256": deps.digest(source),
    }
    assert package["native_members"] == {}


@pytest.mark.parametrize(
    "member, content",
    [
        ("example/payload", b"MZ\x90\x00 inert"),
        ("example/payload.bin", bytes.fromhex("feedfacf") + b"inert"),
        ("example/payload.SH", b"inert"),
        ("example/helper.txt", b"#!/bin/sh\nexit 99\n"),
        ("sitecustomize/__init__.py", b"inert source"),
    ],
)
def test_record_integrity_precedes_executable_and_hook_rejection(
    tmp_path: Path, member: str, content: bytes
) -> None:
    path, origin = wheel(tmp_path, "example", files={member: content})
    rewrite_wheel(path, {member: content + b"changed"})
    origin["sha256"] = deps.digest(path.read_bytes())
    with pytest.raises(deps.Refusal, match="WHEEL_RECORD_INTEGRITY"):
        deps.inspect_wheel(path, origin)


def test_native_data_relocation_is_refused_by_inspector_and_materializer(tmp_path: Path) -> None:
    from scripts import stage9_v5d4_environment as environment

    path, origin = wheel(
        tmp_path,
        "example",
        files={"example-1.0.data/platlib/example/native.abi3.so": b"\x7fELF inert"},
    )
    with pytest.raises(deps.Refusal, match="NATIVE_DATA_RELOCATION_REQUIRES_REVIEW"):
        deps.inspect_wheel(path, origin)
    destination = tmp_path / "packages"
    destination.mkdir()
    with pytest.raises(ValueError, match="NATIVE_DATA_RELOCATION_REQUIRES_REVIEW"):
        environment.extract_wheel(
            path.read_bytes(), destination, {"filename": path.name, "origin": origin}
        )
    assert list(destination.iterdir()) == []


def test_archive_executable_bit_does_not_create_unreviewed_native_or_executable_file(
    tmp_path: Path,
) -> None:
    from scripts import stage9_v5d4_environment as environment

    member = "example/data.txt"
    path, origin = wheel(tmp_path, "example", files={member: b"plain inert data"})
    with zipfile.ZipFile(path) as archive:
        entries = [(item, archive.read(item)) for item in archive.infolist()]
    with zipfile.ZipFile(path, "w") as archive:
        for item, content in entries:
            if item.filename == member:
                item.create_system = 3
                item.external_attr = (stat.S_IFREG | 0o755) << 16
            archive.writestr(item, content)
    archive_bytes = path.read_bytes()
    origin["sha256"] = deps.digest(archive_bytes)
    package = deps.inspect_wheel(path, origin)
    assert package["native_members"] == {}
    approved_members = copy.deepcopy(package["materialized_members"])
    destination = tmp_path / "packages"
    destination.mkdir()
    environment.extract_wheel(archive_bytes, destination, package)
    materialized = destination / member
    assert materialized.read_bytes() == b"plain inert data"
    assert deps.digest(materialized.read_bytes()) == approved_members[member]["sha256"]
    mode = stat.S_IMODE(materialized.stat().st_mode)
    assert mode & 0o111 == 0
    if os.name == "posix":
        assert mode == 0o600
    environment.verify_materialized(
        destination,
        {entry["path"]: entry["sha256"] for entry in approved_members.values()},
    )
    assert package["materialized_members"] == approved_members
    assert path.read_bytes() == archive_bytes


@pytest.fixture(params=("dependencies", "environment", "abi"))
def evidence_reader(request: pytest.FixtureRequest) -> Callable[[Path], bytes]:
    if request.param == "environment":
        from scripts import stage9_v5d4_environment

        return stage9_v5d4_environment.read
    if request.param == "abi":
        from scripts import stage9_v5d4_abi

        return stage9_v5d4_abi.read_bytes
    return deps.read_regular


@pytest.mark.skipif(os.name != "posix", reason="Wymaga rzeczywistych flag odczytu POSIX.")
@pytest.mark.parametrize("missing", ["O_NOFOLLOW", "O_NONBLOCK"])
def test_missing_posix_reader_protection_refuses_before_open(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    evidence_reader: Callable[[Path], bytes],
    missing: str,
) -> None:
    def forbidden_open(*args, **kwargs):
        raise AssertionError("Czytnik bez ochrony nie może otworzyć wejścia.")

    monkeypatch.setattr(os, missing, None)
    monkeypatch.setattr(os, "open", forbidden_open)
    with pytest.raises(ValueError, match="^SECURE_FILE_READER_UNAVAILABLE$"):
        evidence_reader(tmp_path / "unopened.bin")


@pytest.mark.skipif(os.name != "nt", reason="Wymaga natywnego czytnika Windows i uchwytów Win32.")
def test_windows_native_reader_preserves_binary_bytes_and_unicode_path(
    tmp_path: Path, evidence_reader: Callable[[Path], bytes]
) -> None:
    path = tmp_path / "zażółć.bin"
    content = b"\x00\r\n\x1a\xff\x80opaque data; never execute\r\n"
    path.write_bytes(content)
    assert evidence_reader(path) == content


@pytest.mark.skipif(os.name != "nt", reason="Wymaga rzeczywistego wyboru backendu Windows.")
def test_windows_reader_without_native_backend_refuses_without_fallback(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    evidence_reader: Callable[[Path], bytes],
) -> None:
    import ctypes

    path = tmp_path / "evidence.bin"
    path.write_bytes(b"readable evidence cannot bypass missing protection")
    monkeypatch.setattr(ctypes, "WinDLL", None)
    with pytest.raises(ValueError, match="^SECURE_FILE_READER_UNAVAILABLE$"):
        evidence_reader(path)


@pytest.mark.skipif(os.name != "nt", reason="Wymaga rzeczywistego hardlinku NTFS.")
def test_windows_native_reader_rejects_hardlinked_file(
    tmp_path: Path, evidence_reader: Callable[[Path], bytes]
) -> None:
    from scripts import stage9_v5d4_abi

    path = tmp_path / "source.bin"
    path.write_bytes(b"required evidence")
    linked = tmp_path / "hardlink.bin"
    os.link(path, linked)
    refusal = (
        "FILE_REJECTED" if evidence_reader is stage9_v5d4_abi.read_bytes else "NONREGULAR_INPUT"
    )
    with pytest.raises(ValueError, match="^" + refusal + "$"):
        evidence_reader(path)
    with pytest.raises(ValueError, match="^" + refusal + "$"):
        evidence_reader(linked)


@pytest.mark.skipif(os.name != "nt", reason="Wymaga rzeczywistych reparse points Windows.")
@pytest.mark.parametrize("kind", ["file", "ancestor_directory"])
def test_windows_native_reader_rejects_symlink_and_symlink_ancestor(
    tmp_path: Path, kind: str, evidence_reader: Callable[[Path], bytes]
) -> None:
    from scripts import stage9_v5d4_abi

    target = tmp_path / "target"
    target.mkdir()
    (target / "evidence.bin").write_bytes(b"required evidence")
    link = tmp_path / "alias"
    try:
        if kind == "file":
            link.symlink_to(target / "evidence.bin")
            attempted = link
        else:
            link.symlink_to(target, target_is_directory=True)
            attempted = link / "evidence.bin"
    except OSError as exc:
        if getattr(exc, "winerror", None) == 1314:
            pytest.skip("Windows ERROR_PRIVILEGE_NOT_HELD przy tworzeniu symlinku fixture.")
        raise
    refusal = "NONREGULAR_INPUT" if kind == "file" else "WINDOWS_REPARSE_OR_SPECIAL_INPUT"
    if evidence_reader is stage9_v5d4_abi.read_bytes:
        refusal = "SYMLINK_REJECTED"
    with pytest.raises(ValueError, match="^" + refusal + "$"):
        evidence_reader(attempted)


@pytest.mark.skipif(os.name != "nt", reason="Wymaga rzeczywistej kontroli współdzielenia Win32.")
def test_windows_native_reader_refuses_evidence_open_for_writing(
    tmp_path: Path, evidence_reader: Callable[[Path], bytes]
) -> None:
    path = tmp_path / "evidence.bin"
    path.write_bytes(b"required evidence")
    with path.open("r+b"):
        with pytest.raises(ValueError, match="^WINDOWS_PROTECTED_OPEN_FAILED$"):
            evidence_reader(path)
    assert evidence_reader(path) == b"required evidence"


@pytest.mark.skipif(os.name != "nt", reason="Wymaga natywnego ograniczonego odczytu Windows.")
def test_windows_native_reader_enforces_byte_limit(tmp_path: Path) -> None:
    path = tmp_path / "evidence.bin"
    path.write_bytes(b"bounded evidence")
    with pytest.raises(deps.Refusal, match="^INPUT_TOO_LARGE$"):
        deps.read_regular(path, limit=4)
