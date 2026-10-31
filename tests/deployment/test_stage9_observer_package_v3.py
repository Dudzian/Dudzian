"""Whole-package V3 boundaries using inert, independently assembled DEB fixtures."""

from __future__ import annotations

import copy
import gzip
import hashlib
import json
import os
import sys
import tarfile
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import pytest

from deployment import stage9_observer_package_v3 as package_v3
from deployment.stage9_observer_contract import ContractRefusal
from deployment.stage9_observer_offline import Limits

pytestmark = [pytest.mark.security, pytest.mark.unit]
_VERSION = "3.12.3-1ubuntu0.17"


def _sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _tar_fixture(
    entries: list[tuple[str, str, bytes | str]], *, gnu: bool = True
) -> tuple[bytes, dict[str, Any]]:
    """Construct expected semantics from fixture inputs, never the V3 parser."""
    output = bytearray()
    records: list[dict[str, Any]] = []
    for name, kind, payload in entries:
        member = tarfile.TarInfo(name)
        member.mode = 0o755 if kind == "5" else 0o777 if kind == "2" else 0o644
        member.uid = member.gid = 0
        member.uname = member.gname = "root"
        member.mtime = 123456789
        member.type = kind.encode("ascii")
        body = payload if kind == "0" else b""
        assert isinstance(body, bytes)
        member.size = len(body)
        member.linkname = str(payload) if kind == "2" else ""
        header = member.tobuf(format=tarfile.GNU_FORMAT if gnu else tarfile.USTAR_FORMAT)
        assert len(header) == 512
        output.extend(header)
        output.extend(body)
        output.extend(bytes((-len(body)) % 512))
        records.append(
            {
                "name": name,
                "typeflag": kind,
                "size": len(body),
                "mode": member.mode,
                "uid": 0,
                "gid": 0,
                "mtime": member.mtime,
                "linkname": member.linkname,
                "magic_version_hex": header[257:265].hex(),
                "uname": "root",
                "gname": "root",
                "devmajor": header[329:337].hex(),
                "devminor": header[337:345].hex(),
                "sha256": _sha(body) if kind == "0" else None,
                "tail_hex": header[345:512].hex(),
            }
        )
    output.extend(bytes(1024))
    raw = bytes(output)
    return raw, {
        "profile": "gnu-inert-v3" if gnu else "ustar-inert-v3",
        "unpacked_bytes": len(raw),
        "unpacked_sha256": _sha(raw),
        "records": records,
    }


def _ar_fixture(entries: list[tuple[str, bytes]]) -> bytes:
    output = bytearray(b"!<arch>\n")
    for name, body in entries:
        header = (f"{name + '/':<16}{0:<12}{0:<6}{0:<6}{'100644':<8}{len(body):<10}`\n").encode()
        assert len(header) == 60
        output.extend(header)
        output.extend(body)
        if len(body) % 2:
            output.extend(b"\n")
    return bytes(output)


@dataclass
class PackageCase:
    directory: Path
    declaration: dict[str, Any]
    entries: list[tuple[str, bytes]]

    @property
    def package(self) -> dict[str, Any]:
        return self.declaration["packages"][0]

    @property
    def path(self) -> Path:
        return self.directory / self.package["filename"]

    @property
    def raw_inventory(self) -> bytes:
        return json.dumps(self.declaration).encode()

    def rebind_archive(self, raw: bytes) -> None:
        self.path.write_bytes(raw)
        self.package.update(size=len(raw), sha256=_sha(raw))

    def reassemble(self) -> None:
        self.rebind_archive(_ar_fixture(self.entries))

    def verify(self, *, limits: Limits = Limits()) -> Any:
        return package_v3.verify_ubuntu_package_inventory_v3(
            self.directory, inventory_raw=self.raw_inventory, limits=limits
        )


def build_package_case(
    directory: Path,
    *,
    variant: str = "raw",
    gnu: bool = True,
    package: str = "python3.12-minimal",
    control_package: str | None = None,
) -> PackageCase:
    directory.mkdir(exist_ok=True)
    control = (
        f"Package: {control_package or package}\nVersion: {_VERSION}\n"
        "Architecture: amd64\nDescription: synthetic inert V3 fixture\n"
    ).encode()
    control_raw, control_declaration = _tar_fixture(
        [("./", "5", b""), ("./control", "0", control), ("./postinst", "0", b"exit 99\n")],
        gnu=gnu,
    )
    data_raw, data_declaration = _tar_fixture(
        [
            ("./", "5", b""),
            ("./usr/", "5", b""),
            ("./usr/value.txt", "0", b"inert fixture bytes\n"),
            ("./absolute-link", "2", "/etc/python3.12/sitecustomize.py"),
            ("./relative-link", "2", "../outside"),
        ],
        gnu=gnu,
    )
    if variant == "gz":
        suffix = ".gz"
        control_body, data_body = (gzip.compress(raw, mtime=0) for raw in (control_raw, data_raw))
    elif variant == "zst":
        import zstandard

        suffix = ".zst"
        compressor = zstandard.ZstdCompressor(write_checksum=True)
        control_body, data_body = (compressor.compress(raw) for raw in (control_raw, data_raw))
    else:
        suffix = ""
        control_body, data_body = control_raw, data_raw
    entries = [
        ("debian-binary", b"2.0\n"),
        ("control.tar" + suffix, control_body),
        ("data.tar" + suffix, data_body),
    ]
    declaration = {
        "format": "stage9-ubuntu-package-inventory-v3",
        "classification": "SAME_ORIGIN_OBSERVED_COMPARISON_ONLY",
        "source_authenticity": "UNVERIFIED",
        "packages": [
            {
                "package": package,
                "version": _VERSION,
                "architecture": "amd64",
                "filename": f"{package}_{_VERSION}_amd64.deb",
                "size": 0,
                "sha256": "",
                "control": control_declaration,
                "data": data_declaration,
            }
        ],
    }
    case = PackageCase(directory, declaration, entries)
    case.reassemble()
    return case


@pytest.fixture
def package_case(tmp_path: Path) -> PackageCase:
    if sys.platform != "linux":
        pytest.skip("Public descriptor verifier supports Linux only")
    return build_package_case(tmp_path / "debs")


def assert_blocked(result: Any) -> None:
    assert result.scope == "INERT_PACKAGE_INVENTORY_ONLY"
    assert result.source_authenticity == "UNVERIFIED"
    assert result.qualification == "BLOCKED"
    assert result.CONTRACT_APPROVED == "NO"
    assert result.STAGE_3A_COMPLETE == "NO"
    assert result.PRODUCTION_QUALIFIED == "NO"
    assert result.production_status == "PRODUCTION_NO_GO"
    assert result.runtime_attestation == "MISSING"
    assert result.host_pid_namespace == "UNVERIFIED"
    assert result.refused is True


def assert_refused(result: Any, code: str | None = None) -> None:
    assert result.consistency != "MATCH"
    assert result.reason_codes
    assert result.packages == ()
    if code is not None:
        assert code in result.reason_codes
    assert_blocked(result)


@pytest.mark.parametrize("gnu", [True, False], ids=["gnu", "ustar"])
@pytest.mark.parametrize("variant", ["raw", "gz", "zst"])
def test_complete_synthetic_package_matches_only_package_scope(
    tmp_path: Path, gnu: bool, variant: str
) -> None:
    if sys.platform != "linux":
        pytest.skip("Public descriptor verifier supports Linux only")
    case = build_package_case(tmp_path / "debs", gnu=gnu, variant=variant)
    result = case.verify()
    assert result.consistency == "MATCH", result.reason_codes
    assert len(result.packages) == 1
    assert result.packages[0].archive_sha256 == case.package["sha256"]
    assert result.packages[0].archive_bytes == case.package["size"]
    assert result.packages[0].control.members == 3
    assert result.packages[0].data.members == 5
    assert result.packages[0].data.symlinks == 2
    assert_blocked(result)


@pytest.mark.parametrize(
    "mutation",
    [
        "format",
        "classification",
        "authenticity",
        "extra-root",
        "missing-root",
        "empty-packages",
        "package-type",
        "unknown-package",
        "version",
        "architecture",
        "filename-traversal",
        "filename-absolute",
        "filename-wrong",
        "size-bool",
        "size-negative",
        "digest",
        "extra-package",
        "missing-package",
        "duplicate-package",
        "extra-tar",
        "member-bool",
    ],
)
def test_package_declaration_malformed_schema_refuses_before_archive_io(
    package_case: PackageCase, monkeypatch: pytest.MonkeyPatch, mutation: str
) -> None:
    doc = package_case.declaration
    package = package_case.package
    if mutation == "format":
        doc["format"] = "stage9-ubuntu-package-inventory-v4"
    elif mutation == "classification":
        doc["classification"] = "INDEPENDENTLY_ACCEPTED"
    elif mutation == "authenticity":
        doc["source_authenticity"] = "AUTHENTICATED"
    elif mutation == "extra-root":
        doc["authorization"] = True
    elif mutation == "missing-root":
        del doc["format"]
    elif mutation == "empty-packages":
        doc["packages"] = []
    elif mutation == "package-type":
        doc["packages"] = [None]
    elif mutation == "unknown-package":
        package["package"] = "unapproved"
    elif mutation in {"version", "architecture"}:
        package[mutation] = "other"
    elif mutation == "filename-traversal":
        package["filename"] = "../outside.deb"
    elif mutation == "filename-absolute":
        package["filename"] = "/tmp/outside.deb"
    elif mutation == "filename-wrong":
        package["filename"] = "unbound.deb"
    elif mutation == "size-bool":
        package["size"] = True
    elif mutation == "size-negative":
        package["size"] = -1
    elif mutation == "digest":
        package["sha256"] = "0" * 64
    elif mutation == "extra-package":
        package["file_id"] = "runtime-link"
    elif mutation == "missing-package":
        del package["control"]
    elif mutation == "duplicate-package":
        doc["packages"].append(copy.deepcopy(package))
    elif mutation == "extra-tar":
        package["data"]["arbitrary"] = "ignored"
    elif mutation == "member-bool":
        package["data"]["records"][0]["uid"] = False

    def forbidden(*args: Any, **kwargs: Any) -> None:
        pytest.fail("archive IO reached with malformed inventory")

    monkeypatch.setattr(package_v3, "_InputReader", forbidden)
    assert_refused(package_case.verify())


@pytest.mark.parametrize(
    "raw",
    [
        b"null",
        b"[]",
        b"{",
        b'{"format":"x","format":"y"}',
        b'{"x":NaN}',
        b'{"x":Infinity}',
        b'{"x":1.5}',
        b"\xff",
        b'{"x":' + b"[" * 1100 + b"0" + b"]" * 1100 + b"}",
    ],
)
def test_strict_json_refuses_duplicate_nonfinite_floats_and_recursion(raw: bytes) -> None:
    with pytest.raises(ContractRefusal):
        package_v3.parse_package_inventory_v3(raw)


@pytest.mark.parametrize("mutation", ["size", "digest"])
def test_whole_archive_hash_and_size_checked_before_tar_parser(
    package_case: PackageCase, monkeypatch: pytest.MonkeyPatch, mutation: str
) -> None:
    if mutation == "size":
        package_case.path.write_bytes(b"not a DEB")
    else:
        raw = bytearray(package_case.path.read_bytes())
        raw[-1] = 1
        package_case.path.write_bytes(raw)

    def forbidden(*args: Any, **kwargs: Any) -> None:
        pytest.fail("TAR parser reached before whole archive integrity check")

    monkeypatch.setattr(package_v3, "verify_tar_v3", forbidden)
    assert_refused(package_case.verify())


@pytest.mark.parametrize(
    "mutation",
    [
        "magic",
        "version",
        "extra-member",
        "wrong-order",
        "extension",
        "header-end",
        "truncated",
        "trailing",
        "odd-padding",
        "size-numeric",
        "unsupported-xz",
    ],
)
def test_malformed_ar_refuses_after_rebinding_whole_archive_hash(
    package_case: PackageCase, mutation: str
) -> None:
    entries = package_case.entries
    raw = bytearray(package_case.path.read_bytes())
    if mutation == "magic":
        raw[0] = 0
    elif mutation == "version":
        entries[0] = ("debian-binary", b"3.0\n")
        raw = bytearray(_ar_fixture(entries))
    elif mutation == "extra-member":
        raw = bytearray(_ar_fixture(entries + [("extra", b"ignored")]))
    elif mutation == "wrong-order":
        raw = bytearray(_ar_fixture([entries[0], entries[2], entries[1]]))
    elif mutation == "extension":
        entries[1] = ("//", entries[1][1])
        raw = bytearray(_ar_fixture(entries))
    elif mutation == "header-end":
        raw[8 + 58] = 0
    elif mutation == "truncated":
        raw = raw[:-1]
    elif mutation == "trailing":
        raw.extend(b"\n")
    elif mutation == "odd-padding":
        entries[0] = ("debian-binary", b"2.0\nX")
        raw = bytearray(_ar_fixture(entries))
        raw[8 + 60 + 5] = 0
    elif mutation == "size-numeric":
        raw[8 + 48 : 8 + 58] = b"-4        "
    elif mutation == "unsupported-xz":
        entries[1] = ("control.tar.xz", entries[1][1])
        raw = bytearray(_ar_fixture(entries))
    package_case.rebind_archive(bytes(raw))
    assert_refused(package_case.verify())


def test_control_identity_must_match_package_declaration(tmp_path: Path) -> None:
    if sys.platform != "linux":
        pytest.skip("Public descriptor verifier supports Linux only")
    case = build_package_case(tmp_path / "debs", control_package="libpython3.12-dev")
    assert_refused(case.verify(), "ARTIFACT_MISMATCH")


@pytest.mark.parametrize(
    "mutation",
    [
        "duplicate-case",
        "folded-identity",
        "missing-identity",
        "nul",
        "cr",
        "utf8",
        "invalid-key",
        "orphan-continuation",
        "too-many-fields",
    ],
)
def test_malformed_control_identity_is_not_accepted_as_package_metadata(
    package_case: PackageCase, mutation: str
) -> None:
    raw = (f"Package: python3.12-minimal\nVersion: {_VERSION}\nArchitecture: amd64\n").encode()
    if mutation == "duplicate-case":
        raw += b"package: python3.12-minimal\n"
    elif mutation == "folded-identity":
        raw = raw.replace(b"Version:", b" unexpected continuation\nVersion:")
    elif mutation == "missing-identity":
        raw = raw.replace(b"Architecture: amd64\n", b"")
    elif mutation == "nul":
        raw += b"Description: bad\0field\n"
    elif mutation == "cr":
        raw = raw.replace(b"\n", b"\r\n")
    elif mutation == "utf8":
        raw += b"Description: \xff\n"
    elif mutation == "invalid-key":
        raw += b"123: invalid key\n"
    elif mutation == "orphan-continuation":
        raw = b" continuation before any field\n" + raw
    else:
        raw += b"".join(f"Extra-{index}: value\n".encode() for index in range(129))
    body, declaration = _tar_fixture([("./", "5", b""), ("./control", "0", raw)])
    package_case.entries[1] = ("control.tar", body)
    package_case.package["control"] = declaration
    package_case.reassemble()
    assert_refused(package_case.verify())


def test_non_identity_control_fields_can_have_bounded_continuations(
    package_case: PackageCase,
) -> None:
    raw = (
        f"Package: python3.12-minimal\nVersion: {_VERSION}\nArchitecture: amd64\n"
        "Description: inert package\n continuation remains unexecuted data\n"
    ).encode()
    body, declaration = _tar_fixture([("./", "5", b""), ("./control", "0", raw)])
    package_case.entries[1] = ("control.tar", body)
    package_case.package["control"] = declaration
    package_case.reassemble()
    assert package_case.verify().consistency == "MATCH"


@pytest.mark.parametrize("mutation", ["extra", "missing", "symlink", "hardlink", "directory"])
def test_closed_input_tree_refuses_unsafe_or_undeclared_archives(
    package_case: PackageCase, mutation: str
) -> None:
    if mutation == "extra":
        (package_case.directory / "undeclared").write_bytes(b"extra")
    elif mutation == "missing":
        package_case.path.unlink()
    elif mutation == "symlink":
        original = package_case.path
        outside = package_case.directory.parent / "outside.deb"
        original.rename(outside)
        original.symlink_to(outside)
    elif mutation == "hardlink":
        os.link(package_case.path, package_case.directory.parent / "alias.deb")
    elif mutation == "directory":
        package_case.path.unlink()
        package_case.path.mkdir()
    assert_refused(package_case.verify())


def test_symlinked_directory_ancestor_is_not_followed(package_case: PackageCase) -> None:
    alias = package_case.directory.parent / "alias"
    alias.symlink_to(package_case.directory, target_is_directory=True)
    result = package_v3.verify_ubuntu_package_inventory_v3(
        alias, inventory_raw=package_case.raw_inventory
    )
    assert_refused(result)


def test_root_and_unsupported_platform_refuse_before_file_io(
    package_case: PackageCase, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(os, "geteuid", lambda: 0)
    assert_refused(package_case.verify(), "FILE_POLICY_REJECTED")
    monkeypatch.setattr(os, "geteuid", lambda: 1000)
    monkeypatch.setattr(sys, "platform", "win32")
    assert_refused(package_case.verify(), "UNSUPPORTED_SOURCE")


@pytest.mark.parametrize(
    "limits",
    [
        replace(Limits(), max_file_bytes=1024),
        replace(Limits(), max_total_bytes=1024),
        replace(Limits(), max_unpacked_bytes=1024),
        replace(Limits(), max_archive_members=4),
        replace(Limits(), working_memory_bytes=4096),
        replace(Limits(), chunk_bytes=511),
        replace(Limits(), timeout_s=float("nan")),
        replace(Limits(), max_files=True),
    ],
)
def test_limits_refuse_without_partial_success(package_case: PackageCase, limits: Limits) -> None:
    assert_refused(package_case.verify(limits=limits), "LIMIT_EXCEEDED")


@pytest.mark.parametrize(
    "ceiling", ["max_total_bytes", "max_unpacked_bytes", "max_archive_members", "max_files"]
)
def test_two_packages_share_all_aggregate_budgets(tmp_path: Path, ceiling: str) -> None:
    if sys.platform != "linux":
        pytest.skip("Public descriptor verifier supports Linux only")
    first = build_package_case(tmp_path / "debs")
    second = build_package_case(tmp_path / "debs", package="libpython3.12-minimal")
    first.declaration["packages"].append(second.package)
    assert first.verify().consistency == "MATCH"
    amount = {
        "max_total_bytes": max(first.package["size"], second.package["size"]),
        "max_unpacked_bytes": first.package["control"]["unpacked_bytes"]
        + first.package["data"]["unpacked_bytes"],
        "max_archive_members": 8,
        "max_files": 1,
    }[ceiling]
    assert_refused(first.verify(limits=replace(Limits(), **{ceiling: amount})), "LIMIT_EXCEEDED")


def test_declared_archive_bytes_do_not_execute_maintainer_or_link_target(
    package_case: PackageCase, monkeypatch: pytest.MonkeyPatch
) -> None:
    import subprocess

    def forbidden(*args: Any, **kwargs: Any) -> None:
        pytest.fail("inert package verifier tried to execute, extract, or create symlinks")

    monkeypatch.setattr(subprocess, "run", forbidden)
    monkeypatch.setattr(subprocess, "Popen", forbidden)
    monkeypatch.setattr(os, "symlink", forbidden)
    monkeypatch.setattr(tarfile.TarFile, "extract", forbidden)
    monkeypatch.setattr(tarfile.TarFile, "extractall", forbidden)
    assert package_case.verify().consistency == "MATCH"


@pytest.mark.parametrize(
    "mutation", ["mode", "link", "replacement", "directory", "directory-rebinding"]
)
def test_changes_during_verification_cannot_return_match(
    package_case: PackageCase, monkeypatch: pytest.MonkeyPatch, mutation: str
) -> None:
    original = package_v3._verify_archive

    def verify_then_mutate(*args: Any, **kwargs: Any) -> Any:
        result = original(*args, **kwargs)
        if mutation == "mode":
            package_case.path.chmod(0o600)
        elif mutation == "link":
            os.link(package_case.path, package_case.directory.parent / "concurrent-alias")
        elif mutation == "replacement":
            replacement = package_case.directory.parent / "replacement.deb"
            replacement.write_bytes(package_case.path.read_bytes())
            replacement.replace(package_case.path)
        elif mutation == "directory":
            package_case.directory.chmod(0o700)
        else:
            renamed = package_case.directory.parent / "moved-debs"
            package_case.directory.rename(renamed)
            package_case.directory.mkdir()
        return result

    monkeypatch.setattr(package_v3, "_verify_archive", verify_then_mutate)
    assert_refused(package_case.verify(), "FILE_CHANGED")


def test_deadline_expiring_between_full_hash_and_parser_refuses(
    package_case: PackageCase, monkeypatch: pytest.MonkeyPatch
) -> None:
    original = package_v3._verify_archive

    def expire(fd: int, package: Any, limits: Limits, budget: Any) -> Any:
        budget.started -= 61
        return original(fd, package, limits, budget)

    monkeypatch.setattr(package_v3, "_verify_archive", expire)
    assert_refused(package_case.verify(), "LIMIT_EXCEEDED")


def test_incomplete_whole_hash_read_never_calls_parser(
    package_case: PackageCase, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(package_v3, "_pread", lambda *args: b"")

    def forbidden(*args: Any, **kwargs: Any) -> None:
        pytest.fail("TAR parser reached after an incomplete whole-DEB hash")

    monkeypatch.setattr(package_v3, "verify_tar_v3", forbidden)
    assert_refused(package_case.verify(), "PROVENANCE_INCOMPLETE")


def test_failed_reader_cleanup_cannot_publish_a_success(
    package_case: PackageCase, monkeypatch: pytest.MonkeyPatch
) -> None:
    original = package_v3._InputReader.close

    def fail_after_release(reader: Any) -> None:
        original(reader)
        raise OSError("/untrusted/cleanup detail")

    monkeypatch.setattr(package_v3._InputReader, "close", fail_after_release)
    assert_refused(package_case.verify(), "PROVENANCE_INCOMPLETE")


@pytest.mark.parametrize("failure", [OSError, MemoryError, ValueError, OverflowError])
def test_internal_failure_diagnostics_are_controlled_and_sanitized(
    package_case: PackageCase, monkeypatch: pytest.MonkeyPatch, failure: type[Exception]
) -> None:
    def fail(*args: Any, **kwargs: Any) -> Any:
        raise failure("/sensitive/target attacker-controlled diagnostic")

    monkeypatch.setattr(package_v3, "_verify_archive", fail)
    result = package_case.verify()
    assert_refused(result)
    assert "sensitive" not in repr(result)
    assert "attacker" not in repr(result)


def test_parser_allocation_failure_is_a_controlled_limit_refusal(
    package_case: PackageCase, monkeypatch: pytest.MonkeyPatch
) -> None:
    def out_of_memory(*args: Any, **kwargs: Any) -> Any:
        raise MemoryError("untrusted allocator diagnostic")

    monkeypatch.setattr(package_v3.json, "loads", out_of_memory)
    with pytest.raises(ContractRefusal, match="LIMIT_EXCEEDED"):
        package_v3.parse_package_inventory_v3(package_case.raw_inventory)
    assert_refused(package_case.verify(), "LIMIT_EXCEEDED")


@pytest.mark.parametrize(
    "first_error", [False, True], ids=["cleanup-only", "original-refusal-preserved"]
)
def test_native_cleanup_failure_never_matches_or_replaces_the_first_refusal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, first_error: bool
) -> None:
    if sys.platform != "linux":
        pytest.skip("Public descriptor verifier supports Linux only")
    case = build_package_case(tmp_path / "debs", variant="zst")
    native_reader = package_v3._ZstdReader

    class ClosingFailure(native_reader):
        def close(self) -> None:
            super().close()
            raise ContractRefusal("PROVENANCE_INCOMPLETE")

    if first_error:
        case.package["control"]["records"][1]["sha256"] = "1" * 64
    monkeypatch.setattr(package_v3, "_ZstdReader", ClosingFailure)
    result = case.verify()
    assert_refused(result, "ARTIFACT_MISMATCH" if first_error else "PROVENANCE_INCOMPLETE")
    if first_error:
        assert "PROVENANCE_INCOMPLETE" not in result.reason_codes


@pytest.mark.parametrize("variant", ["gz", "zst"])
@pytest.mark.parametrize("mutation", ["payload", "truncated", "concatenated", "trailing"])
def test_compressed_members_refuse_corruption_or_extra_streams(
    tmp_path: Path, variant: str, mutation: str
) -> None:
    if sys.platform != "linux":
        pytest.skip("Public descriptor verifier supports Linux only")
    case = build_package_case(tmp_path / "debs", variant=variant)
    name, body = case.entries[2]
    if mutation == "payload":
        corrupt = bytearray(body)
        corrupt[len(corrupt) // 2] ^= 0xFF
        body = bytes(corrupt)
    elif mutation == "truncated":
        body = body[:-1]
    elif mutation == "concatenated":
        body += body
    else:
        body += b"unparsed extra bytes"
    case.entries[2] = (name, body)
    case.reassemble()
    assert_refused(case.verify())


def test_legacy_models_and_manifest_parser_still_refuse_package_only_data(
    package_case: PackageCase,
) -> None:
    from deployment import stage9_observer_contract as contract
    from deployment.stage9_observer_ubuntu import UbuntuMember

    assert set(UbuntuMember.__dataclass_fields__) == {"member", "file_id", "sha256", "size"}
    with pytest.raises(ContractRefusal):
        contract.parse_manifest(package_case.raw_inventory)
    assert not hasattr(package_case.verify(), "artifact_consistency")
