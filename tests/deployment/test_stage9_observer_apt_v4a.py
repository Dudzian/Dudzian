"""Public V4a chain comparisons with independently assembled, inert packages."""

from __future__ import annotations

import gzip
import hashlib
import json
import os
import socket
import subprocess
import sys
import tarfile
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import pytest

from deployment import (
    stage9_observer_apt_v4a as apt,
    stage9_observer_offline as offline,
    stage9_observer_package_v3 as v3,
)
from deployment.stage9_observer_offline import Limits
from tests.deployment.test_stage9_observer_package_v3 import PackageCase, build_package_case

pytestmark = [pytest.mark.security, pytest.mark.unit]


def _sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


@dataclass
class AptCase:
    package_case: PackageCase
    index_format: str = "Packages"

    @property
    def directory(self) -> Path:
        return self.package_case.directory

    @property
    def index_path(self) -> Path:
        return self.directory / self.index_format

    @property
    def release_path(self) -> Path:
        return self.directory / "Release"

    @property
    def inventory_raw(self) -> bytes:
        return self.package_case.raw_inventory

    @property
    def declaration(self) -> Any:
        release = self.release_path.read_bytes()
        index = self.index_path.read_bytes()
        return apt.AptDeclarationV4a(
            release_path="Release",
            release_sha256=_sha(release),
            release_size=len(release),
            packages_path=self.index_format,
            packages_sha256=_sha(index),
            packages_size=len(index),
            packages_format=self.index_format,
        )

    def package_stanzas(self) -> bytes:
        return b"\n".join(
            (
                f"Package: {package['package']}\nVersion: {package['version']}\n"
                f"Architecture: {package['architecture']}\nFilename: {package['filename']}\n"
                f"SHA256: {package['sha256']}\nSize: {package['size']}\n"
            ).encode()
            for package in self.package_case.declaration["packages"]
        )

    def set_release(self, raw: bytes) -> None:
        self.release_path.write_bytes(raw)

    def set_index(self, raw: bytes, *, compress: bool = True) -> None:
        if compress and self.index_format == "Packages.gz":
            raw = gzip.compress(raw, mtime=0)
        self.index_path.write_bytes(raw)
        self.set_release(f"SHA256:\n {_sha(raw)} {len(raw)} {self.index_format}\n".encode())

    def refresh(self) -> None:
        self.set_index(self.package_stanzas())

    def verify(self, *, declaration: Any = None, **kwargs: Any) -> Any:
        return apt.verify_apt_provenance_v4a(
            self.directory,
            declaration=self.declaration if declaration is None else declaration,
            inventory_raw=kwargs.pop("inventory_raw", self.inventory_raw),
            **kwargs,
        )


def _build(directory: Path, *, index_format: str = "Packages", **kwargs: Any) -> AptCase:
    case = AptCase(build_package_case(directory, **kwargs), index_format)
    case.refresh()
    return case


@pytest.fixture
def case(tmp_path: Path) -> AptCase:
    if sys.platform != "linux":
        pytest.skip("Public descriptor verifier supports Linux only")
    return _build(tmp_path / "apt")


def _blocked(result: Any) -> None:
    assert result.signature_verification == "NOT_PERFORMED"
    assert result.source_authenticity == "UNVERIFIED"
    assert result.acceptance == "UNVERIFIED"
    assert result.stage_9 == "IN_PROGRESS"
    assert result.qualification == "BLOCKED"
    assert result.STAGE_3A_COMPLETE == "NO"
    assert result.PRODUCTION_QUALIFIED == "NO"
    assert result.production_status == "PRODUCTION_NO_GO"


def _refused(result: Any, code: str | None = None) -> None:
    assert result.consistency in {"MISMATCH", "INCOMPLETE"}
    assert result.reason_codes
    assert result.packages == ()
    if code is not None:
        assert code in result.reason_codes
    _blocked(result)


@pytest.mark.parametrize("index_format", ["Packages", "Packages.gz"])
@pytest.mark.parametrize("variant", ["raw", "gz", "zst"])
@pytest.mark.parametrize("gnu", [True, False], ids=["gnu", "ustar"])
def test_complete_chain_matches_only_local_consistency(
    tmp_path: Path, index_format: str, variant: str, gnu: bool
) -> None:
    if sys.platform != "linux":
        pytest.skip("Public descriptor verifier supports Linux only")
    fixture = _build(tmp_path / "apt", index_format=index_format, variant=variant, gnu=gnu)
    result = fixture.verify()
    assert result.consistency == "MATCH", result.reason_codes
    assert not result.reason_codes
    assert len(result.packages) == 1
    assert result.packages[0].archive_sha256 == fixture.package_case.package["sha256"]
    assert result.packages[0].data.symlinks == 2
    _blocked(result)


def test_two_package_chain_is_atomic(case: AptCase) -> None:
    second = build_package_case(case.directory, package="libpython3.12-minimal")
    case.package_case.declaration["packages"].append(second.package)
    case.refresh()
    result = case.verify()
    assert result.consistency == "MATCH", result.reason_codes
    assert len(result.packages) == 2
    second.path.write_bytes(second.path.read_bytes()[:-1] + b"x")
    _refused(case.verify(), "ARTIFACT_MISMATCH")


@pytest.mark.parametrize("target", ["release", "index", "deb"])
@pytest.mark.parametrize("mutation", ["hash", "size", "content"])
def test_every_chain_link_checks_hash_size_and_content(
    case: AptCase, target: str, mutation: str
) -> None:
    declaration = case.declaration
    if mutation in {"hash", "size"}:
        if target == "deb":
            key = "sha256" if mutation == "hash" else "size"
            package = case.package_case.package
            package[key] = "1" * 64 if mutation == "hash" else package[key] + 1
        else:
            prefix = "release" if target == "release" else "packages"
            key = prefix + ("_sha256" if mutation == "hash" else "_size")
            declaration = replace(
                declaration,
                **{key: "1" * 64 if mutation == "hash" else getattr(declaration, key) + 1},
            )
    else:
        path = {
            "release": case.release_path,
            "index": case.index_path,
            "deb": case.package_case.path,
        }[target]
        raw = path.read_bytes()
        path.write_bytes(raw[:-1] + bytes([raw[-1] ^ 1]))
    _refused(case.verify(declaration=declaration), "ARTIFACT_MISMATCH")


@pytest.mark.parametrize(
    "field", ["Package", "Version", "Architecture", "Filename", "SHA256", "Size"]
)
def test_rehashed_release_cannot_hide_wrong_package_identity_or_archive_binding(
    case: AptCase, field: str
) -> None:
    replacements = {
        "Package": "libpython3.12-minimal",
        "Version": "3.12.3-1ubuntu0.18",
        "Architecture": "arm64",
        "Filename": "wrong.deb",
        "SHA256": "2" * 64,
        "Size": str(case.package_case.package["size"] + 1),
    }
    lines = case.package_stanzas().decode().splitlines()
    changed = [
        f"{field}: {replacements[field]}" if line.startswith(field + ":") else line
        for line in lines
    ]
    case.set_index(("\n".join(changed) + "\n").encode())
    _refused(case.verify())


def test_rehashed_deb_and_index_still_compare_frozen_v3_member_payloads(case: AptCase) -> None:
    raw = case.package_case.path.read_bytes()
    assert b"inert fixture bytes\n" in raw
    changed = raw.replace(b"inert fixture bytes\n", b"wrong fixture bytes\n")
    case.package_case.rebind_archive(changed)
    case.refresh()
    _refused(case.verify(), "ARTIFACT_MISMATCH")


def test_rehashed_all_outer_layers_still_validate_inner_control_identity(tmp_path: Path) -> None:
    if sys.platform != "linux":
        pytest.skip("Public descriptor verifier supports Linux only")
    fixture = _build(tmp_path / "apt", control_package="libpython3.12-minimal")
    _refused(fixture.verify(), "ARTIFACT_MISMATCH")


@pytest.mark.parametrize(
    "mutation", ["hash", "size", "path", "extra-path", "duplicate-path", "algorithm"]
)
def test_release_contents_are_validated_after_frozen_hash_matches(
    case: AptCase, mutation: str
) -> None:
    raw = case.release_path.read_bytes()
    if mutation == "hash":
        raw = raw.replace(_sha(case.index_path.read_bytes()).encode(), b"1" * 64)
    elif mutation == "size":
        raw = raw.replace(f" {case.index_path.stat().st_size} ".encode(), b" 1 ")
    elif mutation == "path":
        raw = raw.replace(b" Packages\n", b" missing/Packages\n")
    elif mutation in {"extra-path", "duplicate-path"}:
        row = raw.splitlines()[1]
        raw += row + (b"\n" if mutation == "duplicate-path" else b".extra\n")
    else:
        raw = raw.replace(b"SHA256:", b"SHA512:")
    case.set_release(raw)
    _refused(case.verify())


@pytest.mark.parametrize("target", ["release", "packages"])
@pytest.mark.parametrize(
    "mutation",
    [
        "duplicate-field",
        "case-duplicate",
        "duplicate-stanza",
        "empty",
        "invalid-utf8",
        "crlf",
        "nul",
        "missing-colon",
    ],
)
def test_metadata_duplicates_and_malformed_records_refuse(
    case: AptCase, target: str, mutation: str
) -> None:
    raw = case.release_path.read_bytes() if target == "release" else case.package_stanzas()
    if mutation == "duplicate-field":
        raw += raw if target == "release" else raw.splitlines()[0] + b"\n"
    elif mutation == "case-duplicate":
        raw += b"sha256:\n" if target == "release" else b"package: python3.12-minimal\n"
    elif mutation == "duplicate-stanza":
        raw += b"\n" + raw
    elif mutation == "empty":
        raw = b""
    elif mutation == "invalid-utf8":
        raw += b"\xff"
    elif mutation == "crlf":
        raw = raw.replace(b"\n", b"\r\n")
    elif mutation == "nul":
        raw += b"\0"
    else:
        raw = raw.replace(b":", b"", 1)
    if target == "release":
        case.set_release(raw)
    else:
        case.set_index(raw)
    _refused(case.verify())


@pytest.mark.parametrize(
    "field", ["Package", "Version", "Architecture", "Filename", "SHA256", "Size"]
)
def test_missing_required_package_field_refuses(case: AptCase, field: str) -> None:
    raw = (
        b"\n".join(
            line
            for line in case.package_stanzas().splitlines()
            if not line.startswith(field.encode() + b":")
        )
        + b"\n"
    )
    case.set_index(raw)
    _refused(case.verify())


@pytest.mark.parametrize("field", ["Filename", "SHA256", "Size", "Package"])
def test_identity_fields_cannot_be_continued(case: AptCase, field: str) -> None:
    raw = case.package_stanzas().replace(field.encode() + b":", field.encode() + b":\n ", 1)
    case.set_index(raw)
    _refused(case.verify())


@pytest.mark.parametrize(
    "value",
    [
        "../Packages",
        "/Packages",
        "./Packages",
        "a//Packages",
        "a/../Packages",
        "a\\Packages",
        "%50ackages",
        "Packagé",
        "Packages\ud800",
    ],
)
def test_declared_paths_reject_traversal_aliases_and_invalid_encoding(
    case: AptCase, value: str
) -> None:
    _refused(case.verify(declaration=replace(case.declaration, packages_path=value)))


@pytest.mark.parametrize(
    "value",
    [
        "../bad.deb",
        "/bad.deb",
        "./bad.deb",
        "a//bad.deb",
        "a/../bad.deb",
        "a\\bad.deb",
        "%62ad.deb",
        "bad\t.deb",
    ],
)
def test_package_filename_paths_do_not_gain_apt_alias_semantics(case: AptCase, value: str) -> None:
    raw = case.package_stanzas().replace(
        case.package_case.package["filename"].encode(), value.encode()
    )
    case.set_index(raw)
    _refused(case.verify())


@pytest.mark.parametrize("target", ["Release", "Packages", "deb"])
@pytest.mark.parametrize("kind", ["symlink", "hardlink"])
def test_local_inputs_cannot_be_links(case: AptCase, target: str, kind: str) -> None:
    path = case.package_case.path if target == "deb" else case.directory / target
    declaration = case.declaration
    outside = case.directory.parent / "outside"
    path.rename(outside)
    if kind == "symlink":
        path.symlink_to(outside)
    else:
        os.link(outside, path)
    _refused(case.verify(declaration=declaration))


@pytest.mark.parametrize("missing", ["Release", "Packages", "deb"])
def test_missing_chain_element_refuses(case: AptCase, missing: str) -> None:
    declaration = case.declaration
    path = case.package_case.path if missing == "deb" else case.directory / missing
    path.unlink()
    _refused(case.verify(declaration=declaration))


@pytest.mark.parametrize("extra", ["file", "directory", "symlink", "package-record"])
def test_undeclared_local_or_index_elements_refuse(case: AptCase, extra: str) -> None:
    if extra == "file":
        (case.directory / "unapproved").write_bytes(b"extra")
    elif extra == "directory":
        (case.directory / "unapproved").mkdir()
    elif extra == "symlink":
        (case.directory / "unapproved").symlink_to("/outside")
    else:
        second = build_package_case(case.directory, package="libpython3.12-minimal")
        second.path.unlink()
        second_case = AptCase(second)
        case.set_index(case.package_stanzas() + b"\n" + second_case.package_stanzas())
    _refused(case.verify())


@pytest.mark.parametrize(
    "mutation", ["wrong-magic", "truncated", "payload", "concatenated", "trailing", "padding"]
)
def test_gzip_is_exactly_one_complete_member(tmp_path: Path, mutation: str) -> None:
    if sys.platform != "linux":
        pytest.skip("Public descriptor verifier supports Linux only")
    fixture = _build(tmp_path / "apt", index_format="Packages.gz")
    raw = fixture.index_path.read_bytes()
    if mutation == "wrong-magic":
        raw = b"invalid gzip"
    elif mutation == "truncated":
        raw = raw[:-1]
    elif mutation == "payload":
        changed = bytearray(raw)
        changed[len(changed) // 2] ^= 0xFF
        raw = bytes(changed)
    elif mutation == "concatenated":
        raw += gzip.compress(b"", mtime=0)
    elif mutation == "padding":
        raw += b"\0" * 8
    else:
        raw += b"trailing attacker data"
    fixture.set_index(raw, compress=False)
    _refused(fixture.verify())


@pytest.mark.parametrize(
    "path,format_name",
    [
        ("InRelease", "Packages"),
        ("Release", "Packages.xz"),
        ("Release", "Packages.zst"),
        ("Release", "auto"),
        ("Release.gpg", "Packages"),
    ],
)
def test_unsupported_profiles_do_not_guess_or_strip_signatures(
    case: AptCase, path: str, format_name: str
) -> None:
    declaration = replace(case.declaration, release_path=path, packages_format=format_name)
    _refused(case.verify(declaration=declaration))


def test_inline_pgp_payload_is_not_accepted_as_release(case: AptCase) -> None:
    case.set_release(
        b"-----BEGIN PGP SIGNED MESSAGE-----\nHash: SHA256\n\n"
        + case.release_path.read_bytes()
        + b"-----BEGIN PGP SIGNATURE-----\nsynthetic\n-----END PGP SIGNATURE-----\n"
    )
    _refused(case.verify())


@pytest.mark.parametrize(
    "limits",
    [
        replace(Limits(), max_file_bytes=1024),
        replace(Limits(), max_total_bytes=1024),
        replace(Limits(), max_unpacked_bytes=100),
        replace(Limits(), max_archive_members=1),
        replace(Limits(), max_files=2),
        replace(Limits(), working_memory_bytes=4096),
        replace(Limits(), timeout_s=float("nan")),
        replace(Limits(), timeout_s=float("inf")),
        replace(Limits(), max_files=True),
        replace(Limits(), chunk_bytes=511),
    ],
)
def test_limits_apply_to_complete_chain(case: AptCase, limits: Limits) -> None:
    _refused(case.verify(limits=limits), "LIMIT_EXCEEDED")


def test_gzip_expansion_is_limited_even_with_correct_outer_hashes(tmp_path: Path) -> None:
    if sys.platform != "linux":
        pytest.skip("Public descriptor verifier supports Linux only")
    fixture = _build(tmp_path / "apt", index_format="Packages.gz")
    fixture.set_index(b"Package: " + b"a" * 200_000 + b"\n")
    _refused(fixture.verify(limits=replace(Limits(), max_unpacked_bytes=65536)), "LIMIT_EXCEEDED")


def test_global_read_budget_includes_release_index_and_deb(case: AptCase) -> None:
    total = (
        case.release_path.stat().st_size
        + case.index_path.stat().st_size
        + case.package_case.path.stat().st_size
    )
    _refused(case.verify(limits=replace(Limits(), max_total_bytes=total - 1)), "LIMIT_EXCEEDED")


def test_no_network_execution_installation_or_link_resolution(
    case: AptCase, monkeypatch: pytest.MonkeyPatch
) -> None:
    def forbidden(*args: Any, **kwargs: Any) -> Any:
        pytest.fail("V4a attempted network, execution, extraction or symlink creation")

    monkeypatch.setattr(socket, "socket", forbidden)
    monkeypatch.setattr(subprocess, "run", forbidden)
    monkeypatch.setattr(subprocess, "Popen", forbidden)
    monkeypatch.setattr(os, "symlink", forbidden)
    monkeypatch.setattr(tarfile.TarFile, "extract", forbidden)
    monkeypatch.setattr(tarfile.TarFile, "extractall", forbidden)
    assert case.verify().consistency == "MATCH"


def test_privileged_or_unsupported_runtime_refuses(
    case: AptCase, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(os, "geteuid", lambda: 0)
    _refused(case.verify(), "FILE_POLICY_REJECTED")
    monkeypatch.setattr(os, "geteuid", lambda: 1000)
    monkeypatch.setattr(sys, "platform", "win32")
    _refused(case.verify(), "UNSUPPORTED_SOURCE")


def test_close_failure_cannot_publish_match(case: AptCase, monkeypatch: pytest.MonkeyPatch) -> None:
    original = offline._InputReader.close

    def close_then_fail(reader: Any) -> None:
        original(reader)
        raise OSError("sensitive cleanup path")

    monkeypatch.setattr(offline._InputReader, "close", close_then_fail)
    result = case.verify()
    _refused(result, "PROVENANCE_INCOMPLETE")
    assert "sensitive" not in repr(result)


def test_premature_empty_read_cannot_publish_match(
    case: AptCase, monkeypatch: pytest.MonkeyPatch
) -> None:
    declaration = case.declaration
    monkeypatch.setattr(os, "pread", lambda *args: b"")
    _refused(case.verify(declaration=declaration))


def test_file_disappearance_after_initial_inventory_refuses(
    case: AptCase, monkeypatch: pytest.MonkeyPatch
) -> None:
    original = offline._InputReader.scan

    def scan_then_remove(reader: Any, expected: Any) -> Any:
        baseline = original(reader, expected)
        case.index_path.unlink(missing_ok=True)
        return baseline

    monkeypatch.setattr(offline._InputReader, "scan", scan_then_remove)
    _refused(case.verify())


def test_descriptor_identity_change_refuses(case: AptCase, monkeypatch: pytest.MonkeyPatch) -> None:
    original = os.fstat
    observed: dict[int, int] = {}

    def changed_fstat(fd: int) -> Any:
        info = original(fd)
        if info.st_size == case.package_case.path.stat().st_size:
            observed[fd] = observed.get(fd, 0) + 1
            if observed[fd] >= 2:
                return os.stat_result(
                    (
                        info.st_mode,
                        info.st_ino + 1,
                        info.st_dev,
                        info.st_nlink,
                        info.st_uid,
                        info.st_gid,
                        info.st_size,
                        info.st_atime,
                        info.st_mtime,
                        info.st_ctime,
                    )
                )
        return info

    monkeypatch.setattr(os, "fstat", changed_fstat)
    _refused(case.verify())


def test_deadline_expiry_during_reads_refuses(
    case: AptCase, monkeypatch: pytest.MonkeyPatch
) -> None:
    original = offline._Budget.read

    def expire(budget: Any, amount: int) -> None:
        budget.started -= 61
        original(budget, amount)

    monkeypatch.setattr(offline._Budget, "read", expire)
    _refused(case.verify(), "LIMIT_EXCEEDED")


def test_inventory_conflicting_identity_refuses_before_partial_result(case: AptCase) -> None:
    changed = json.loads(case.inventory_raw)
    changed["packages"].append(dict(changed["packages"][0]))
    _refused(case.verify(inventory_raw=json.dumps(changed).encode()))


def test_v4a_result_and_declaration_are_separate_from_historical_models(case: AptCase) -> None:
    declaration = case.declaration
    with pytest.raises((AttributeError, TypeError)):
        declaration.packages_path = "Packages.gz"
    result = case.verify()
    assert type(result) is apt.AptResultV4a
    assert type(result) is not v3.PackageInventoryResultV3
    assert not hasattr(result, "report")
    _blocked(result)


@pytest.mark.parametrize("index_format", ["Packages", "Packages.gz"])
def test_release_paths_are_relative_to_its_directory(tmp_path: Path, index_format: str) -> None:
    if sys.platform != "linux":
        pytest.skip("Public descriptor verifier supports Linux only")
    fixture = _build(tmp_path / "apt", index_format=index_format)
    nested = fixture.directory / "dists" / "synthetic"
    index_directory = nested / "main" / "binary-amd64"
    index_directory.mkdir(parents=True)
    fixture.index_path.rename(index_directory / index_format)
    index = (index_directory / index_format).read_bytes()
    release = f"SHA256:\n {_sha(index)} {len(index)} main/binary-amd64/{index_format}\n".encode()
    (nested / "Release").write_bytes(release)
    fixture.release_path.unlink()
    declaration = apt.AptDeclarationV4a(
        release_path="dists/synthetic/Release",
        release_sha256=_sha(release),
        release_size=len(release),
        packages_path=f"dists/synthetic/main/binary-amd64/{index_format}",
        packages_sha256=_sha(index),
        packages_size=len(index),
        packages_format=index_format,
    )
    result = fixture.verify(declaration=declaration)
    assert result.consistency == "MATCH", result.reason_codes
    _blocked(result)


def test_index_selection_cannot_escape_release_directory(case: AptCase) -> None:
    nested = case.directory / "dists"
    nested.mkdir()
    raw = case.release_path.read_bytes()
    case.release_path.rename(nested / "Release")
    declaration = apt.AptDeclarationV4a(
        release_path="dists/Release",
        release_sha256=_sha(raw),
        release_size=len(raw),
        packages_path="Packages",
        packages_sha256=_sha(case.index_path.read_bytes()),
        packages_size=case.index_path.stat().st_size,
        packages_format="Packages",
    )
    _refused(case.verify(declaration=declaration))


@pytest.mark.parametrize("field", ["MD5sum", "SHA1", "SHA512", "Description", "Unknown"])
def test_packages_profile_rejects_extra_fields(case: AptCase, field: str) -> None:
    case.set_index(case.package_stanzas() + f"{field}: supplied data\n".encode())
    _refused(case.verify(), "UNSUPPORTED_SOURCE")


@pytest.mark.parametrize("target", ["release", "packages"])
@pytest.mark.parametrize("marker", [b"\xef\xbb\xbf", b"\x01", b"\x7f", b"\xc2\x85"])
def test_utf8_bom_and_control_characters_are_not_silently_normalized(
    case: AptCase, target: str, marker: bytes
) -> None:
    if target == "release":
        case.set_release(marker + case.release_path.read_bytes())
    else:
        case.set_index(marker + case.package_stanzas())
    _refused(case.verify())


@pytest.mark.parametrize(
    "field,value",
    [
        ("Size", "-1"),
        ("Size", "1.0"),
        ("Size", "1e4"),
        ("Size", "99999999999999999999999999999999999999999999"),
        ("SHA256", "A" * 64),
        ("SHA256", "abc"),
        ("SHA256", "0" * 64),
    ],
)
def test_index_numeric_and_digest_profile_is_strict(case: AptCase, field: str, value: str) -> None:
    raw = (
        b"\n".join(
            f"{field}: {value}".encode() if line.startswith(field.encode() + b":") else line
            for line in case.package_stanzas().splitlines()
        )
        + b"\n"
    )
    case.set_index(raw)
    _refused(case.verify())


@pytest.mark.parametrize("field", ["release_sha256", "packages_sha256"])
@pytest.mark.parametrize("value", ["", "short", "A" * 64, "0" * 64, None])
def test_frozen_digest_declarations_are_validated(case: AptCase, field: str, value: Any) -> None:
    _refused(case.verify(declaration=replace(case.declaration, **{field: value})))


@pytest.mark.parametrize("field", ["release_size", "packages_size"])
@pytest.mark.parametrize("value", [0, -1, True, 1.5, None])
def test_frozen_size_declarations_are_validated(case: AptCase, field: str, value: Any) -> None:
    _refused(case.verify(declaration=replace(case.declaration, **{field: value})))


def test_final_inventory_failure_discards_staged_package_results(
    case: AptCase, monkeypatch: pytest.MonkeyPatch
) -> None:
    original = offline._InputReader.finish

    def add_extra_before_finish(reader: Any, expected: Any) -> None:
        (case.directory / "late-unapproved").write_bytes(b"late")
        original(reader, expected)

    monkeypatch.setattr(offline._InputReader, "finish", add_extra_before_finish)
    _refused(case.verify())


def test_cleanup_failure_preserves_original_refusal(
    case: AptCase, monkeypatch: pytest.MonkeyPatch
) -> None:
    original = offline._InputReader.close

    def close_then_fail(reader: Any) -> None:
        original(reader)
        raise OSError("sensitive cleanup path")

    monkeypatch.setattr(offline._InputReader, "close", close_then_fail)
    case.set_index(case.package_stanzas().replace(b"Architecture: amd64", b"Architecture: arm64"))
    result = case.verify()
    _refused(result, "ARTIFACT_MISMATCH")


def test_root_directory_alias_is_not_followed(case: AptCase) -> None:
    declaration = case.declaration
    alias = case.directory.parent / "alias"
    alias.symlink_to(case.directory, target_is_directory=True)
    result = apt.verify_apt_provenance_v4a(
        alias, declaration=declaration, inventory_raw=case.inventory_raw
    )
    _refused(result)


@pytest.mark.parametrize("index_format", ["Packages", "Packages.gz"])
@pytest.mark.parametrize("ceiling", ["max_unpacked_bytes", "max_archive_members"])
def test_index_and_deb_share_aggregate_decoded_and_record_budgets(
    tmp_path: Path, index_format: str, ceiling: str
) -> None:
    if sys.platform != "linux":
        pytest.skip("Public descriptor verifier supports Linux only")
    fixture = _build(tmp_path / "apt", index_format=index_format)
    package = fixture.package_case.package
    archive_bytes = package["control"]["unpacked_bytes"] + package["data"]["unpacked_bytes"]
    archive_records = len(package["control"]["records"]) + len(package["data"]["records"])
    minimum = {
        "max_unpacked_bytes": archive_bytes + len(fixture.package_stanzas()),
        "max_archive_members": archive_records + 2,
    }[ceiling]
    result = fixture.verify(limits=replace(Limits(), **{ceiling: minimum}))
    assert result.consistency == "MATCH", result.reason_codes
    _refused(fixture.verify(limits=replace(Limits(), **{ceiling: minimum - 1})), "LIMIT_EXCEEDED")


def test_short_outer_preads_are_accumulated_without_skipping_bytes(
    case: AptCase, monkeypatch: pytest.MonkeyPatch
) -> None:
    original = apt._pread

    def short_read(fd: int, size: int, offset: int) -> bytes:
        return original(fd, min(size, 17), offset)

    monkeypatch.setattr(apt, "_pread", short_read)
    result = case.verify()
    assert result.consistency == "MATCH", result.reason_codes


@pytest.mark.parametrize("failure", [OSError, MemoryError, ValueError, OverflowError])
def test_io_failures_have_controlled_sanitized_diagnostics(
    case: AptCase, monkeypatch: pytest.MonkeyPatch, failure: type[Exception]
) -> None:
    declaration = case.declaration

    def fail(*args: Any, **kwargs: Any) -> Any:
        raise failure("/sensitive/path attacker-controlled diagnostic")

    monkeypatch.setattr(os, "pread", fail)
    result = case.verify(declaration=declaration)
    _refused(result)
    assert "sensitive" not in repr(result)
    assert "attacker" not in repr(result)


@pytest.mark.parametrize("target", ["release", "packages"])
def test_metadata_file_caps_apply_before_reading(
    case: AptCase, monkeypatch: pytest.MonkeyPatch, target: str
) -> None:
    declaration = replace(
        case.declaration,
        **{target + "_size": (1024 * 1024 if target == "release" else 2 * 1024 * 1024) + 1},
    )

    def forbidden(*args: Any, **kwargs: Any) -> Any:
        pytest.fail("oversized frozen metadata reached descriptor read")

    monkeypatch.setattr(os, "pread", forbidden)
    _refused(case.verify(declaration=declaration), "LIMIT_EXCEEDED")


@pytest.mark.parametrize("flags", [1, 2, 4, 8, 16, 32, 64, 128])
def test_gzip_optional_and_reserved_flags_are_outside_profile(tmp_path: Path, flags: int) -> None:
    if sys.platform != "linux":
        pytest.skip("Public descriptor verifier supports Linux only")
    fixture = _build(tmp_path / "apt", index_format="Packages.gz")
    raw = bytearray(fixture.index_path.read_bytes())
    raw[3] = flags
    fixture.set_index(bytes(raw), compress=False)
    _refused(fixture.verify(), "UNSUPPORTED_SOURCE")


def test_gzip_streaming_continues_over_multiple_output_blocks(tmp_path: Path) -> None:
    if sys.platform != "linux":
        pytest.skip("Public descriptor verifier supports Linux only")
    fixture = _build(tmp_path / "apt", index_format="Packages.gz")
    second = build_package_case(fixture.directory, package="libpython3.12-minimal")
    fixture.package_case.declaration["packages"].append(second.package)
    third = build_package_case(fixture.directory, package="libpython3.12-stdlib")
    fixture.package_case.declaration["packages"].append(third.package)
    fixture.refresh()
    assert len(fixture.package_stanzas()) > 512
    result = fixture.verify(limits=replace(Limits(), chunk_bytes=512))
    assert result.consistency == "MATCH", result.reason_codes
    assert len(result.packages) == 3


def test_empty_stanza_separators_do_not_create_phantom_packages(case: AptCase) -> None:
    case.set_index(b"\n" + case.package_stanzas() + b"\n\n")
    result = case.verify()
    assert result.consistency == "MATCH", result.reason_codes
    assert len(result.packages) == 1


def test_deadline_expiry_during_reader_close_discards_staged_success(
    case: AptCase, monkeypatch: pytest.MonkeyPatch
) -> None:
    original_close = offline._InputReader.close
    original_monotonic = offline.time.monotonic
    elapsed = [0.0]
    monkeypatch.setattr(offline.time, "monotonic", lambda: original_monotonic() + elapsed[0])

    def close_then_expire(reader: Any) -> None:
        original_close(reader)
        elapsed[0] += 61

    monkeypatch.setattr(offline._InputReader, "close", close_then_expire)
    _refused(case.verify(), "LIMIT_EXCEEDED")
