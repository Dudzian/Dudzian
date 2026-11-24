"""Synthetic header data and TEST ONLY reviewer declarations; never execute wheels."""

from __future__ import annotations

import base64
import copy
import csv
import io
from pathlib import Path
import stat
import zipfile

import pytest

from scripts import stage9_v5d4_dependencies as deps
from scripts import stage9_v5d6_headers as headers
from tests.deployment.test_stage9_v5d4_dependencies import wheel

pytestmark = [pytest.mark.security, pytest.mark.unit]
HEADER_CONTENT = b"/* TEST ONLY inert synthetic header, never compiled. */\n"
TAG = "cp312-cp312-manylinux_2_24_x86_64.manylinux_2_28_x86_64"


def header_wheel(
    directory: Path, name: str, selected: str, files: dict[str, bytes]
) -> tuple[Path, dict[str, str]]:
    """Compressed filename tags have separate expanded WHEEL metadata rows."""
    path, origin = wheel(directory, name, selected, tag=TAG, files=files)
    rewrite(
        path,
        updates={
            f"{name}-{selected}.dist-info/WHEEL": (
                "Wheel-Version: 1.0\nRoot-Is-Purelib: false\n"
                "Tag: cp312-cp312-manylinux_2_24_x86_64\n"
                "Tag: cp312-cp312-manylinux_2_28_x86_64\n"
            ).encode()
        },
        reseal_record=True,
    )
    origin["sha256"] = deps.digest(path.read_bytes())
    return path, origin


def pin_test_header(
    monkeypatch: pytest.MonkeyPatch,
    path: Path,
    content: bytes,
    mode: int = 0o100644,
) -> None:
    """Substitute pins solely within explicitly synthetic tests."""
    monkeypatch.setattr(
        headers,
        "WHEEL",
        {
            **headers.WHEEL,
            "wheel_filename": path.name,
            "wheel_sha256": deps.digest(path.read_bytes()),
        },
    )
    monkeypatch.setattr(
        headers,
        "HEADER",
        {**headers.HEADER, "sha256": deps.digest(content), "size": len(content), "mode": mode},
    )
    monkeypatch.setattr(deps, "header_tools", lambda: vars(headers))


@pytest.fixture
def header_material(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple:
    project = tmp_path / "pyproject.toml"
    project.write_text(
        '[project]\nrequires-python=">=3.11"\ndependencies=["greenlet==3.5.6"]\n'
        "[project.optional-dependencies]\nobserver-kvm-preflight=[]\n",
        encoding="utf-8",
    )
    directory = tmp_path / "wheels"
    directory.mkdir()
    path, origin = header_wheel(
        directory,
        "greenlet",
        "3.5.6",
        files={headers.HEADER["member"]: HEADER_CONTENT},
    )
    pin_test_header(monkeypatch, path, HEADER_CONTENT)
    origins = {path.name: origin}
    proposal = deps.propose(project, directory, origins)
    # This declaration exists only in the synthetic test, never in a tool result.
    expectation = {
        "policy": headers.POLICY_ID,
        "installation_contract_exception": headers.INSTALLATION_EXCEPTION,
        "members": copy.deepcopy(proposal["header_omissions"]["members"]),
        "decision": headers.REVIEW_DECISION,
        "review_reference": "TEST ONLY independent header reviewer fixture",
    }
    return project, directory, origins, proposal, expectation


def rewrite(
    path: Path,
    *,
    updates: dict[str, bytes] | None = None,
    removes: set[str] | None = None,
    modes: dict[str, int] | None = None,
    omit_record_row: str | None = None,
    bad_record_sha: str | None = None,
    reseal_record: bool = False,
) -> None:
    """Edit inert ZIP data; preserve metadata and control RECORD corruption."""
    with zipfile.ZipFile(path) as archive:
        entries = {info.filename: (info, archive.read(info)) for info in archive.infolist()}
    for name in removes or set():
        entries.pop(name)
    for name, content in (updates or {}).items():
        info = entries[name][0] if name in entries else zipfile.ZipInfo(name)
        if name not in entries:
            info.create_system = 3
            info.external_attr = (stat.S_IFREG | 0o644) << 16
        entries[name] = info, content
    record_name = next(name for name in entries if name.endswith(".dist-info/RECORD"))
    if reseal_record:
        record = io.StringIO()
        writer = csv.writer(record)
        for name, (_, content) in entries.items():
            if name != record_name:
                sha = base64.urlsafe_b64encode(bytes.fromhex(deps.digest(content))).rstrip(b"=")
                writer.writerow((name, "sha256=" + sha.decode(), len(content)))
        writer.writerow((record_name, "", ""))
        entries[record_name] = entries[record_name][0], record.getvalue().encode()
    if omit_record_row or bad_record_sha:
        rows = list(csv.reader(io.StringIO(entries[record_name][1].decode())))
        rows = [row for row in rows if row[0] != omit_record_row]
        for row in rows:
            if row[0] == bad_record_sha:
                row[1] = "sha256=" + "A" * 43
        record = io.StringIO()
        csv.writer(record).writerows(rows)
        entries[record_name] = entries[record_name][0], record.getvalue().encode()
    with zipfile.ZipFile(path, "w") as archive:
        for name, (info, content) in entries.items():
            if name in (modes or {}):
                info.external_attr = (modes or {})[name] << 16
            archive.writestr(info, content)


def package_path(material: tuple) -> Path:
    return next(material[1].glob("*.whl"))


def verify_material(material: tuple, approved_headers: dict | None = None) -> dict:
    project, directory, origins, proposal, _ = material
    return deps.verify(
        project,
        directory,
        origins,
        proposal,
        deps.digest(deps.canonical(proposal)),
        {},
        approved_headers=approved_headers,
    )


def test_pinned_header_candidate_has_separate_exact_approval(header_material) -> None:
    project, directory, origins, proposal, expected = header_material
    path = package_path(header_material)
    before = path.read_bytes()
    package = proposal["packages"][0]
    assert package["header_omissions"] == [headers.pinned_row()]
    assert package["omitted_members"] == []
    assert headers.HEADER["member"] not in package["materialized_members"]
    assert proposal["header_omissions"]["status"] == headers.APPROVAL_REQUIRED
    assert "review_reference" not in proposal["header_omissions"]
    with pytest.raises(deps.Refusal, match=headers.APPROVAL_REQUIRED):
        verify_material(header_material)
    verified = verify_material(header_material, expected)
    assert verified["header_expectations"] == expected
    assert verified["header_omissions"] == proposal["header_omissions"]
    assert verified["omitted_members"] == []
    assert verified["authorization"] == "NOT_AUTHORIZED"
    assert deps.propose(project, directory, origins) == proposal
    assert path.read_bytes() == before


def test_synthetic_header_cannot_match_original_production_pin(tmp_path: Path) -> None:
    path, origin = header_wheel(
        tmp_path, "greenlet", "3.5.6", files={headers.HEADER["member"]: HEADER_CONTENT}
    )
    with pytest.raises(deps.Refusal, match="UNSUPPORTED_WHEEL_DATA"):
        deps.inspect_wheel(path, origin)


@pytest.mark.parametrize("change", ["header", "wheel_comment"])
def test_changed_bytes_refuse_even_with_updated_origin(header_material, change: str) -> None:
    path = package_path(header_material)
    if change == "header":
        rewrite(path, updates={headers.HEADER["member"]: b"changed"}, reseal_record=True)
    else:
        with zipfile.ZipFile(path, "a") as archive:
            archive.comment = b"changed original archive identity"
    origin = header_material[2][path.name]
    origin["sha256"] = deps.digest(path.read_bytes())
    with pytest.raises(deps.Refusal, match="UNSUPPORTED_WHEEL_DATA"):
        deps.inspect_wheel(path, origin)


def test_unchanged_wheel_with_changed_origin_sha_refuses(header_material) -> None:
    path = package_path(header_material)
    origin = copy.deepcopy(header_material[2][path.name])
    origin["sha256"] = "0" * 64
    with pytest.raises(deps.Refusal, match="ORIGIN_SHA256"):
        deps.inspect_wheel(path, origin)


@pytest.mark.parametrize(
    "change", ["missing_record_row", "invalid_record_sha", "unrecorded_header"]
)
def test_complete_record_precedes_any_header_exception(header_material, change: str) -> None:
    path = package_path(header_material)
    if change == "missing_record_row":
        rewrite(path, omit_record_row=headers.HEADER["member"])
    elif change == "invalid_record_sha":
        rewrite(path, bad_record_sha=headers.HEADER["member"])
    else:
        rewrite(path, updates={"greenlet-3.5.6.data/headers/extra.h": b"extra"})
    origin = header_material[2][path.name]
    origin["sha256"] = deps.digest(path.read_bytes())
    expected = (
        "WHEEL_RECORD_INTEGRITY" if change == "invalid_record_sha" else "WHEEL_RECORD_INCOMPLETE"
    )
    with pytest.raises(deps.Refusal, match=expected):
        deps.inspect_wheel(path, origin)


def test_modified_header_with_resealed_record_and_wheel_still_must_match_header_pin(
    header_material, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = package_path(header_material)
    rewrite(path, updates={headers.HEADER["member"]: b"changed"}, reseal_record=True)
    origin = header_material[2][path.name]
    origin["sha256"] = deps.digest(path.read_bytes())
    monkeypatch.setattr(headers, "WHEEL", {**headers.WHEEL, "wheel_sha256": origin["sha256"]})
    with pytest.raises(deps.Refusal, match="HEADER_PIN_MISMATCH"):
        deps.inspect_wheel(path, origin)


def test_recorded_extra_header_is_not_admitted_by_generic_header_category(
    header_material, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = package_path(header_material)
    rewrite(path, updates={"greenlet-3.5.6.data/headers/z-extra.h": b"extra"}, reseal_record=True)
    origin = header_material[2][path.name]
    origin["sha256"] = deps.digest(path.read_bytes())
    monkeypatch.setattr(headers, "WHEEL", {**headers.WHEEL, "wheel_sha256": origin["sha256"]})
    with pytest.raises(deps.Refusal, match="UNSUPPORTED_WHEEL_DATA"):
        deps.inspect_wheel(path, origin)


@pytest.mark.parametrize(
    "name,selected,member",
    [
        ("unknown", "3.5.6", "unknown-3.5.6.data/headers/greenlet.h"),
        ("greenlet", "3.5.5", "greenlet-3.5.5.data/headers/greenlet.h"),
        ("greenlet", "3.5.6", "wrong-3.5.6.data/headers/greenlet.h"),
        ("greenlet", "3.5.6", "greenlet-3.5.6.data/scripts/greenlet.h"),
        ("greenlet", "3.5.6", "greenlet-3.5.6.data/data/greenlet.h"),
        ("greenlet", "3.5.6", "greenlet-3.5.6.data/other/greenlet.h"),
    ],
)
def test_unknown_distributions_and_data_paths_refuse(tmp_path, name, selected, member) -> None:
    path, origin = header_wheel(tmp_path, name, selected, files={member: HEADER_CONTENT})
    with pytest.raises(deps.Refusal, match="UNSUPPORTED_WHEEL_DATA"):
        deps.inspect_wheel(path, origin)


@pytest.mark.parametrize("prefix", [b"\x7fELF", b"MZ", b"#!", b"\xfe\xed\xfa\xcf"])
def test_header_policy_never_omits_executable_or_native_bytes(
    header_material, monkeypatch, prefix
) -> None:
    path = package_path(header_material)
    content = prefix + HEADER_CONTENT
    rewrite(path, updates={headers.HEADER["member"]: content}, reseal_record=True)
    origin = header_material[2][path.name]
    origin["sha256"] = deps.digest(path.read_bytes())
    pin_test_header(monkeypatch, path, content)
    with pytest.raises(deps.Refusal, match="HEADER_NATIVE_OR_EXECUTABLE_FORBIDDEN"):
        deps.inspect_wheel(path, origin)


@pytest.mark.parametrize("mode", [0o100755, 0o120644])
def test_executable_mode_and_zip_symlink_refuse(header_material, monkeypatch, mode) -> None:
    path = package_path(header_material)
    rewrite(path, modes={headers.HEADER["member"]: mode})
    origin = header_material[2][path.name]
    origin["sha256"] = deps.digest(path.read_bytes())
    pin_test_header(monkeypatch, path, HEADER_CONTENT, mode)
    reason = (
        "WHEEL_MEMBER_SPECIAL" if stat.S_ISLNK(mode) else "HEADER_NATIVE_OR_EXECUTABLE_FORBIDDEN"
    )
    with pytest.raises(deps.Refusal, match=reason):
        deps.inspect_wheel(path, origin)


@pytest.mark.parametrize(
    "change", ["member", "policy", "decision", "review_reference", "field", "int_alias"]
)
def test_inconsistent_independent_header_expectations_refuse(header_material, change) -> None:
    expectation = copy.deepcopy(header_material[4])
    if change == "member":
        expectation["members"].clear()
    elif change == "int_alias":
        expectation["members"][0]["size"] = float(expectation["members"][0]["size"])
    elif change == "field":
        expectation["automatic_approval"] = True
    else:
        expectation[change] = "   " if change == "review_reference" else "other"
    with pytest.raises(deps.Refusal, match="HEADER_EXPECTATION_MISMATCH"):
        verify_material(header_material, expectation)


def test_bytecode_approval_cannot_substitute_for_header_approval(header_material) -> None:
    project, directory, origins, proposal, expected = header_material
    with pytest.raises(deps.Refusal, match="UNAPPROVED_WHEEL_OMISSIONS"):
        deps.verify(
            project,
            directory,
            origins,
            proposal,
            deps.digest(deps.canonical(proposal)),
            {},
            approved_omissions=expected,
        )


@pytest.mark.parametrize("change", ["missing", "metadata", "member", "target", "alias"])
def test_omission_declarations_cannot_include_header_in_runtime(header_material, change) -> None:
    packages = copy.deepcopy(header_material[3]["packages"])
    package = packages[0]
    member = headers.HEADER["member"]
    if change == "missing":
        package.pop("header_omissions")
    elif change == "metadata":
        package["member_metadata"][member]["mode"] = 0o100755
    else:
        key = member if change == "member" else "alias"
        target = member if change == "target" else "greenlet.h"
        package["materialized_members"][key] = {"path": target, "sha256": headers.HEADER["sha256"]}
    with pytest.raises(headers.Refusal):
        headers.verify_headers(packages, header_material[4])


def test_member_changed_after_inspection_refuses(header_material) -> None:
    path = package_path(header_material)
    rewrite(path, updates={headers.HEADER["member"]: b"changed"}, reseal_record=True)
    with pytest.raises(deps.Refusal, match="ORIGIN_SHA256"):
        verify_material(header_material, header_material[4])


def test_removing_header_cannot_turn_modified_pinned_wheel_into_ordinary_package(
    header_material,
) -> None:
    path = package_path(header_material)
    rewrite(path, removes={headers.HEADER["member"]}, reseal_record=True)
    origin = header_material[2][path.name]
    origin["sha256"] = deps.digest(path.read_bytes())
    with pytest.raises(deps.Refusal, match="UNSUPPORTED_WHEEL_DATA"):
        deps.inspect_wheel(path, origin)


def test_pinned_header_does_not_bypass_subsequent_embedded_cmd_refusal(
    header_material, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The real greenlet also carries this path; no execution-policy exception."""
    path = package_path(header_material)
    rewrite(
        path,
        updates={"greenlet/platform/setup_switch_x64_masm.cmd": b"@echo TEST ONLY inert data\n"},
        reseal_record=True,
    )
    origin = header_material[2][path.name]
    origin["sha256"] = deps.digest(path.read_bytes())
    pin_test_header(monkeypatch, path, HEADER_CONTENT)
    with pytest.raises(deps.Refusal, match="WHEEL_EXECUTABLE_FORMAT_UNSUPPORTED"):
        deps.inspect_wheel(path, origin)


def test_changed_wheel_identity_cannot_hide_missing_omission_declaration(header_material) -> None:
    packages = copy.deepcopy(header_material[3]["packages"])
    packages[0]["sha256"] = "0" * 64
    packages[0].pop("header_omissions")
    with pytest.raises(headers.Refusal, match="HEADER_WHEEL_IDENTITY_MISMATCH"):
        headers.verify_headers(packages, None)


def test_wheel_symlink_does_not_bypass_archive_identity(header_material, tmp_path) -> None:
    path = package_path(header_material)
    link = tmp_path / path.name
    try:
        link.symlink_to(path)
    except (OSError, NotImplementedError):
        pytest.skip("Host does not support unprivileged symlinks")
    with pytest.raises(deps.Refusal):
        deps.inspect_wheel(link, header_material[2][path.name])


def test_old_non_header_packages_need_no_new_approval() -> None:
    package = {
        "name": "unrelated",
        "version": "1.0",
        "filename": "unrelated-1.0-py3-none-any.whl",
        "sha256": "0" * 64,
        "materialized_members": {},
    }
    inventory = headers.verify_headers([package], None)
    assert inventory["members"] == []
    assert inventory["status"] == "NO_HEADER_OMISSIONS"
