"""Core Metadata 2.5 is checked entirely from synthetic wheel data."""

from pathlib import Path

import pytest

from scripts import stage9_v5d4_dependencies as deps
from tests.deployment.test_stage9_v5d4_dependencies import wheel

pytestmark = [pytest.mark.security, pytest.mark.unit]


def metadata_wheel(
    directory: Path,
    name: str,
    fields: str,
    *,
    metadata_version: str = "2.5",
    body: str = "",
) -> tuple[Path, dict[str, str]]:
    metadata = (
        f"Metadata-Version: {metadata_version}\nName: {name}\nVersion: 1.0\n"
        f"Requires-Python: >=3.11\n{fields}\n{body}"
    )
    return wheel(directory, name, files={f"{name}-1.0.dist-info/METADATA": metadata.encode()})


def test_v5d7_accepts_valid_import_declarations_and_preserves_dependencies(tmp_path: Path) -> None:
    path, origin = metadata_wheel(
        tmp_path,
        "example",
        "Import-Name: space.plugin ; private\n"
        "Import-Namespace: space\n"
        "Dynamic: Requires-Dist\n"
        "Requires-Dist: wheel-dependency>=2; extra == 'opt'\n"
        "Provides-Extra: opt\n",
    )
    package = deps.inspect_wheel(path, origin, v5d7=True)
    assert package["metadata_version"] == "2.5"
    assert package["import_names"] == {"space.plugin": True}
    assert package["import_namespaces"] == {"space": False}
    assert package["requires_dist"] == ["wheel-dependency>=2; extra == 'opt'"]
    assert package["provides_extra"] == ["opt"]
    with pytest.raises(deps.Refusal, match="WHEEL_METADATA_VERSION_UNSUPPORTED"):
        deps.inspect_wheel(path, origin)


def test_legacy_wheel_keeps_package_shape(tmp_path: Path) -> None:
    path, origin = metadata_wheel(tmp_path, "example", "", metadata_version="2.4")
    for v5d7 in (False, True):
        package = deps.inspect_wheel(path, origin, v5d7=v5d7)
        assert not {"metadata_version", "import_names", "import_namespaces"} & package.keys()
        assert package["requires_dist"] == []
        assert package["provides_extra"] == []


@pytest.mark.parametrize("version", ["2.6", "3.0"])
def test_unimplemented_metadata_versions_refuse(tmp_path: Path, version: str) -> None:
    path, origin = metadata_wheel(tmp_path, "example", "", metadata_version=version)
    with pytest.raises(deps.Refusal, match="WHEEL_METADATA_VERSION_UNSUPPORTED"):
        deps.inspect_wheel(path, origin, v5d7=True)


def test_mislabeled_import_fields_refuse_in_new_profile(tmp_path: Path) -> None:
    path, origin = metadata_wheel(
        tmp_path, "example", "Import-Name: example\n", metadata_version="2.4"
    )
    with pytest.raises(deps.Refusal, match="WHEEL_IMPORT_DECLARATION_VERSION"):
        deps.inspect_wheel(path, origin, v5d7=True)


def test_empty_import_name_represents_an_explicit_declaration(tmp_path: Path) -> None:
    path, origin = metadata_wheel(tmp_path, "example", "Import-Name:\n")
    package = deps.inspect_wheel(path, origin, v5d7=True)
    assert package["import_names"] == {"": False}
    assert package["import_namespaces"] == {}


@pytest.mark.parametrize(
    ("fields", "body", "reason"),
    [
        ("nAmE: other\n", "", "WHEEL_METADATA_DUPLICATE_OR_MISSING"),
        ("Requires-Python: >=3.12\n", "", "WHEEL_METADATA_DUPLICATE_OR_MISSING"),
        ("Summary: first\nsUmMaRy: second\n", "", "WHEEL_METADATA_DUPLICATE_OR_MISSING"),
        ("Description: in header\n", "in body", "WHEEL_METADATA_DUPLICATE_OR_MISSING"),
        ("Future-Field: value\n", "", "WHEEL_METADATA_FIELD_UNSUPPORTED"),
        ("Dynamic: Name\n", "", "WHEEL_METADATA_INVALID_DYNAMIC"),
        ("Import-Name: space..plugin\n", "", "WHEEL_IMPORT_DECLARATION_INVALID"),
        ("Import-Name: class\n", "", "WHEEL_IMPORT_DECLARATION_INVALID"),
        ("Import-Name: K\n", "", "WHEEL_IMPORT_DECLARATION_INVALID"),
        ("Import-Name: space; public\n", "", "WHEEL_IMPORT_DECLARATION_INVALID"),
        ("Import-Namespace:\n", "", "WHEEL_IMPORT_DECLARATION_INVALID"),
        ("Import-Name:\nImport-Name: another\n", "", "WHEEL_IMPORT_DECLARATION_INVALID"),
        (
            "Import-Name: space\nImport-Name: space; private\n",
            "",
            "WHEEL_IMPORT_DECLARATION_DUPLICATE",
        ),
        ("Import-Name: space\nImport-Namespace: space\n", "", "WHEEL_IMPORT_DECLARATION_AMBIGUOUS"),
        ("Import-Name:\nImport-Namespace: space\n", "", "WHEEL_IMPORT_DECLARATION_AMBIGUOUS"),
        ("Provides-Extra: EXTRA\n", "", "WHEEL_METADATA_INVALID_EXTRA"),
        ("Provides-Extra: extra\nProvides-Extra: extra\n", "", "WHEEL_METADATA_INVALID_EXTRA"),
        ("Requires-Dist: other;\n", "", "UNSUPPORTED_REQUIREMENT"),
        ("Requires-Dist: other>=5,bad; extra == 'unused'\n", "", "UNSUPPORTED_SPECIFIER"),
    ],
)
def test_malformed_metadata_refuses(tmp_path: Path, fields: str, body: str, reason: str) -> None:
    path, origin = metadata_wheel(tmp_path, "example", fields, body=body)
    with pytest.raises(deps.Refusal, match=reason):
        deps.inspect_wheel(path, origin, v5d7=True)


def test_bad_utf8_core_metadata_refuses(tmp_path: Path) -> None:
    path, origin = wheel(
        tmp_path,
        "example",
        files={
            "example-1.0.dist-info/METADATA": (
                b"Metadata-Version: 2.5\nName: example\nVersion: 1.0\nImport-Name: \xff\n"
            )
        },
    )
    with pytest.raises(deps.Refusal, match="WHEEL_METADATA_INVALID_UTF8"):
        deps.inspect_wheel(path, origin, v5d7=True)


@pytest.mark.parametrize(
    ("left_fields", "right_fields"),
    [
        ("Import-Name: space\n", "Import-Name: space\n"),
        ("Import-Name: space; private\n", "Import-Namespace: space\n"),
        ("Import-Name: space\n", "Import-Namespace: space.child\n"),
        ("Import-Name: space\n", "Import-Name: space.child\n"),
    ],
)
def test_closure_refuses_declared_cross_project_shadowing(
    tmp_path: Path, left_fields: str, right_fields: str
) -> None:
    candidates = {}
    for name, fields in (("left", left_fields), ("right", right_fields)):
        path, origin = metadata_wheel(tmp_path, name, fields)
        candidates[name] = deps.inspect_wheel(path, origin, v5d7=True)
    with pytest.raises(deps.Refusal, match="WHEEL_IMPORT_NAME_OVERLAP:space"):
        deps.resolve_dependency_closure(["left", "right"], candidates)


def test_matching_namespace_claims_do_not_define_dependencies(tmp_path: Path) -> None:
    candidates = {}
    for name in ("left", "right"):
        fields = f"Import-Namespace: space\nImport-Name: space.{name}\n"
        path, origin = metadata_wheel(tmp_path, name, fields)
        candidates[name] = deps.inspect_wheel(path, origin, v5d7=True)
    packages, edges = deps.resolve_dependency_closure(["left", "right"], candidates)
    assert [package["name"] for package in packages] == ["left", "right"]
    assert edges == [["project", "left", "left"], ["project", "right", "right"]]
    with pytest.raises(deps.Refusal, match="MISSING_DEPENDENCY:space"):
        deps.resolve_dependency_closure(["space"], candidates)
