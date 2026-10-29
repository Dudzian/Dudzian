"""Rootless real byte reads against synthetic expectations; no target code runs."""

from __future__ import annotations

import ast
import copy
import gzip
import hashlib
import io
import json
import os
import subprocess
import sys
import tarfile
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import pytest

from deployment import stage9_observer_contract as contract
from deployment import stage9_observer_offline as offline
from deployment.canonical_json import canonical_json_bytes
from deployment.stage9_observer_ubuntu import UbuntuMember, UbuntuPackage
from tests.test_stage9_observer_contract import proposed_manifest_launch_fixture

pytestmark = [pytest.mark.security, pytest.mark.unit]
_FIXTURES = Path(__file__).parents[1] / "fixtures" / "stage9_observer"
_VERSION = "3.12.3-1ubuntu0.17"


def _sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _tar(entries: list[tuple[str, bytes]], *, variant: str = "raw") -> bytes:
    output = io.BytesIO()
    with tarfile.open(fileobj=output, mode="w", format=tarfile.USTAR_FORMAT) as archive:
        for name, body in entries:
            member = tarfile.TarInfo(name)
            member.size = len(body)
            member.mode = 0o644
            if name.endswith("/"):
                member.type = tarfile.DIRTYPE
            archive.addfile(member, io.BytesIO(body))
    raw = output.getvalue()
    return gzip.compress(raw, mtime=0) if variant == "gz" else raw


def _ar(entries: list[tuple[str, bytes]]) -> bytes:
    output = bytearray(b"!<arch>\n")
    for name, body in entries:
        header = (f"{name + '/':<16}{0:<12}{0:<6}{0:<6}{'100644':<8}{len(body):<10}`\n").encode(
            "ascii"
        )
        assert len(header) == 60
        output.extend(header)
        output.extend(body)
        if len(body) % 2:
            output.extend(b"\n")
    return bytes(output)


@dataclass
class _Case:
    root: Path
    manifest: dict[str, Any]
    launch: dict[str, Any]
    package: UbuntuPackage
    payloads: dict[str, bytes]

    def file(self, file_id: str) -> dict[str, Any]:
        return next(item for item in self.manifest["files"] if item["file_id"] == file_id)

    def path(self, file_id: str) -> Path:
        return self.root / self.file(file_id)["path"].lstrip("/")

    def archive(
        self,
        *,
        version: str = _VERSION,
        variant: str = "raw",
        data_entries: list[tuple[str, bytes]] | None = None,
        suffix: str | None = None,
        maintainer: bytes = b"#!/bin/sh\nexit 99\n",
    ) -> None:
        control = (
            "Package: python3.12-minimal\n"
            f"Version: {version}\nArchitecture: amd64\nDescription: synthetic fixture\n"
        ).encode()
        if data_entries is None:
            data_entries = self.data_entries()
        ending = suffix if suffix is not None else (".gz" if variant == "gz" else "")
        raw = _ar(
            [
                ("debian-binary", b"2.0\n"),
                (
                    "control.tar" + ending,
                    _tar([("control", control), ("postinst", maintainer)], variant=variant),
                ),
                ("data.tar" + ending, _tar(data_entries, variant=variant)),
            ]
        )
        self.set_archive(raw)

    def data_entries(self) -> list[tuple[str, bytes]]:
        directories = [
            (item["provenance"]["member"] + "/", b"")
            for item in self.manifest["files"]
            if item["type"] == "directory"
        ]
        return directories + [
            (member.member, self.payloads[member.file_id]) for member in self.package.members
        ]

    def set_archive(self, raw: bytes) -> None:
        (self.root / self.package.archive_path).write_bytes(raw)
        source = self.manifest["sources"][0]
        source["artifact_size"], source["artifact_sha256"] = len(raw), _sha(raw)

    def replace_archive_member(self, name: str, raw: bytes) -> None:
        archive = (self.root / self.package.archive_path).read_bytes()
        entries = []
        offset = 8
        while offset < len(archive):
            header = archive[offset : offset + 60]
            member_name = header[:16].strip().removesuffix(b"/").decode()
            size = int(header[48:58])
            body = archive[offset + 60 : offset + 60 + size]
            entries.append((member_name, raw if member_name == name else body))
            offset += 60 + size + size % 2
        self.set_archive(_ar(entries))

    def verify(self, **kwargs: Any) -> offline.OfflineResult:
        return offline.verify_offline(
            self.root,
            manifest_raw=canonical_json_bytes(self.manifest),
            ubuntu_packages=(self.package,),
            **kwargs,
        )

    def rebind_launch(self) -> None:
        files = {item["file_id"]: item for item in self.manifest["files"]}
        self.launch["interpreter"]["version"] = "3.12.3"
        self.launch["interpreter"]["package_version"] = _VERSION
        self.launch["imports"]["modules_sha256"] = _sha(
            canonical_json_bytes(self.manifest["imports"])
        )
        for profile in ("seccomp", "lsm"):
            self.launch["sandbox"][profile]["profile_sha256"] = files[profile]["sha256"]
        self.launch["file_policy"]["integrity_profile_sha256"] = files["integrity"]["sha256"]
        self.launch["file_policy"]["files_sha256"] = _sha(
            canonical_json_bytes(self.manifest["files"])
        )
        for section in ("files", "runtime", "imports"):
            self.launch["runtime_dependencies"][section + "_sha256"] = _sha(
                canonical_json_bytes(self.manifest[section])
            )
        self.manifest["launch_policy_sha256"] = _sha(canonical_json_bytes(self.launch))

    def policy(self) -> dict[str, Any]:
        result = json.loads((_FIXTURES / "refused_policy.json").read_bytes())
        result.update(
            target=copy.deepcopy(self.manifest["target"]),
            allowed_sources=copy.deepcopy(self.manifest["sources"]),
            limits=dict(contract._PROPOSED_V1_LIMITS),
            launch_policy_sha256=self.manifest["launch_policy_sha256"],
        )
        return result


@pytest.fixture
def case(tmp_path: Path) -> _Case:
    if sys.platform != "linux":
        pytest.skip("The bounded descriptor reader supports Linux only in V1")
    template = json.loads((_FIXTURES / "launch_policy.json").read_bytes())
    pair = proposed_manifest_launch_fixture(template)
    manifest, launch = pair["manifest"], pair["launch"]
    root = tmp_path / "input"
    root.mkdir()
    payloads: dict[str, bytes] = {}
    members = []
    for item in manifest["files"]:
        item.update(
            uid=os.geteuid(), gid=os.getegid(), mode=0o755 if item["type"] == "directory" else 0o644
        )
        item["provenance"] = {
            "kind": "source_member",
            "source_id": "apt-source",
            "member": item["path"].lstrip("/"),
        }
        target = root / item["path"].lstrip("/")
        target.parent.mkdir(parents=True, exist_ok=True)
        if item["type"] == "directory":
            target.mkdir(exist_ok=True)
        else:
            raw = ("synthetic inert fixture " + item["file_id"] + "\n").encode()
            payloads[item["file_id"]] = raw
            item.update(size=len(raw), sha256=_sha(raw))
            target.write_bytes(raw)
            members.append(
                UbuntuMember(
                    item["provenance"]["member"], item["file_id"], item["sha256"], item["size"]
                )
            )
        target.chmod(item["mode"])
    for item in manifest["imports"]:
        if "file_id" in item:
            item["sha256"] = next(
                file["sha256"] for file in manifest["files"] if file["file_id"] == item["file_id"]
            )
    manifest["transforms"] = []
    proof = b"opaque synthetic source proof; no signature or accepted root\n"
    manifest["sources"] = [
        {
            "source_id": "apt-source",
            "kind": "ubuntu_apt_snapshot",
            "artifact_size": 1,
            "artifact_sha256": _sha(b"pending"),
            "proof_record_sha256": _sha(proof),
        }
    ]
    manifest["target"]["interpreter"].update(version="3.12.3", package_version=_VERSION)
    (root / "sources").mkdir()
    (root / "sources/proof.bin").write_bytes(proof)
    package = UbuntuPackage(
        "apt-source",
        "sources/python.deb",
        "sources/proof.bin",
        "python3.12-minimal",
        _VERSION,
        "amd64",
        tuple(members),
    )
    result = _Case(root, manifest, launch, package, payloads)
    result.archive()
    result.rebind_launch()
    contract.parse_proposed_manifest(canonical_json_bytes(manifest))
    return result


def _blocked(result: offline.OfflineResult) -> None:
    assert result.refused is True
    assert result.acceptance == result.source_authenticity == "UNVERIFIED"
    assert result.execution == result.runtime_attestation == "MISSING"
    assert result.stage_9 == "IN_PROGRESS"
    assert result.qualification == "BLOCKED"
    assert (
        result.CONTRACT_APPROVED == result.STAGE_3A_COMPLETE == result.PRODUCTION_QUALIFIED == "NO"
    )
    assert result.production_status == "PRODUCTION_NO_GO"
    assert result.host_pid_namespace == "UNVERIFIED"
    assert {"TRUST_CONTEXT_MISSING", "RUNTIME_LAUNCH_UNVERIFIED", "HOST_SCOPE_UNVERIFIED"} <= set(
        result.reason_codes
    )


def _refused(result: offline.OfflineResult, reason: str | None = None) -> None:
    _blocked(result)
    assert result.artifact_consistency in {"MISMATCH", "INCOMPLETE"}
    if reason is not None:
        assert reason in result.reason_codes


def test_matching_synthetic_manifest_measures_real_files_and_whole_deb(case: _Case) -> None:
    result = case.verify()
    _blocked(result)
    assert result.structure == "VALID"
    assert result.artifact_consistency == "MATCH"
    assert result.measured_files == len(case.payloads) + 2
    assert result.measured_bytes > sum(map(len, case.payloads.values()))
    assert result.report is None  # Missing documents never acquire fabricated digests.


def test_advisory_report_uses_existing_contract_and_real_document_pins(case: _Case) -> None:
    result = case.verify(
        policy_raw=canonical_json_bytes(case.policy()),
        launch_policy_raw=canonical_json_bytes(case.launch),
    )
    _blocked(result)
    assert result.artifact_consistency == "MATCH"
    assert result.report is not None
    record = result.report.as_dict()
    assert record["synthetic"] is True
    assert record["manifest_sha256"] == _sha(canonical_json_bytes(case.manifest))
    assert record["policy_sha256"] == _sha(canonical_json_bytes(case.policy()))
    assert record["launch_policy_sha256"] == _sha(canonical_json_bytes(case.launch))
    assert contract.parse_offline_result(result.report.raw).authorization == "UNVERIFIED"


def test_changed_single_byte_refuses(case: _Case) -> None:
    raw = case.path("python").read_bytes()
    case.path("python").write_bytes(bytes([raw[0] ^ 1]) + raw[1:])
    _refused(case.verify(), "ARTIFACT_MISMATCH")


@pytest.mark.parametrize("digest", ["invalid-sha256", "0" * 64, "A" * 64, "f" * 64])
def test_invalid_or_wrong_sha256_refuses(case: _Case, digest: str) -> None:
    case.file("python")["sha256"] = digest
    _refused(case.verify())


def test_wrong_size_refuses(case: _Case) -> None:
    case.file("python")["size"] += 1
    _refused(case.verify(), "ARTIFACT_MISMATCH")


def test_missing_file_refuses(case: _Case) -> None:
    case.path("python").unlink()
    _refused(case.verify(), "PROVENANCE_INCOMPLETE")


def test_unexpected_file_refuses(case: _Case) -> None:
    (case.root / "unlisted.py").write_bytes(b"raise SystemExit(99)\n")
    _refused(case.verify(), "ARTIFACT_MISMATCH")


def test_symlink_instead_of_regular_file_refuses(case: _Case, tmp_path: Path) -> None:
    outside = tmp_path / "outside"
    outside.write_bytes(case.payloads["python"])
    case.path("python").unlink()
    case.path("python").symlink_to(outside)
    _refused(case.verify(), "FILE_POLICY_REJECTED")


def test_symlinked_input_directory_refuses(case: _Case, tmp_path: Path) -> None:
    alias = tmp_path / "alias"
    alias.symlink_to(case.root, target_is_directory=True)
    result = offline.verify_offline(
        alias, manifest_raw=canonical_json_bytes(case.manifest), ubuntu_packages=(case.package,)
    )
    _refused(result)


def test_symlinked_ancestor_of_input_directory_refuses(case: _Case, tmp_path: Path) -> None:
    alias = tmp_path / "ancestor-alias"
    alias.symlink_to(case.root.parent, target_is_directory=True)
    result = offline.verify_offline(
        alias / case.root.name,
        manifest_raw=canonical_json_bytes(case.manifest),
        ubuntu_packages=(case.package,),
    )
    _refused(result)


@pytest.mark.parametrize("kind", ["hardlink", "fifo"])
def test_nonregular_or_aliased_file_refuses_without_hanging(
    case: _Case, tmp_path: Path, kind: str
) -> None:
    target = case.path("python")
    if kind == "hardlink":
        os.link(target, tmp_path / "external-hardlink")
    else:
        target.unlink()
        os.mkfifo(target)
    _refused(case.verify(), "FILE_POLICY_REJECTED")


def test_file_changed_during_read_refuses(case: _Case, monkeypatch: pytest.MonkeyPatch) -> None:
    target = case.path("python")
    identity = target.stat().st_ino
    original = os.read
    changed = False

    def changing_read(fd: int, amount: int) -> bytes:
        nonlocal changed
        raw = original(fd, amount)
        if raw and not changed and os.fstat(fd).st_ino == identity:
            changed = True
            target.write_bytes(case.payloads["python"])
            target.chmod(0o600)
        return raw

    monkeypatch.setattr(os, "read", changing_read)
    _refused(case.verify(), "FILE_CHANGED")
    assert changed


def test_path_replacement_during_read_refuses_even_with_identical_bytes(
    case: _Case, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    target = case.path("python")
    identity = target.stat().st_ino
    original = os.read
    replaced = False

    def replacing_read(fd: int, amount: int) -> bytes:
        nonlocal replaced
        raw = original(fd, amount)
        if raw and not replaced and os.fstat(fd).st_ino == identity:
            replaced = True
            target.rename(tmp_path / "retained-original")
            target.write_bytes(case.payloads["python"])
            target.chmod(0o644)
        return raw

    monkeypatch.setattr(os, "read", replacing_read)
    _refused(case.verify(), "FILE_CHANGED")
    assert replaced


def test_changed_source_identifier_refuses(case: _Case) -> None:
    case.package = replace(case.package, source_id="substituted-source")
    _refused(case.verify(), "ARTIFACT_MISMATCH")


def test_dangling_manifest_source_identifier_refuses(case: _Case) -> None:
    case.file("python")["provenance"]["source_id"] = "substituted-source"
    _refused(case.verify(), "CLOSURE_INCOMPLETE")


def test_wrong_package_version_in_control_refuses(case: _Case) -> None:
    case.archive(version="3.12.3-1ubuntu0.16")
    _refused(case.verify(), "ARTIFACT_MISMATCH")


def test_unsupported_expected_package_version_refuses(case: _Case) -> None:
    case.package = replace(case.package, version="3.12.3-1ubuntu0.16")
    _refused(case.verify(), "ARTIFACT_MISMATCH")


def test_manifest_interpreter_version_is_bound_to_its_package(case: _Case) -> None:
    case.manifest["target"]["interpreter"]["package_version"] = "3.12.3-1ubuntu0.16"
    _refused(case.verify(), "ARTIFACT_MISMATCH")


def test_incomplete_declared_archive_member_inventory_refuses(case: _Case) -> None:
    case.package = replace(case.package, members=case.package.members[:-1])
    _refused(case.verify(), "CLOSURE_INCOMPLETE")


def test_missing_actual_archive_member_refuses(case: _Case) -> None:
    entries = case.data_entries()[:-1]
    case.archive(data_entries=entries)
    _refused(case.verify(), "CLOSURE_INCOMPLETE")


def test_unexpected_actual_archive_member_refuses(case: _Case) -> None:
    entries = case.data_entries()
    case.archive(data_entries=entries + [("extra.py", b"print('must remain inert')\n")])
    _refused(case.verify(), "CLOSURE_INCOMPLETE")


@pytest.mark.parametrize("variant", [".xz", ".zst", ".bad"])
def test_unsupported_archive_codec_refuses_in_full(case: _Case, variant: str) -> None:
    case.archive(suffix=variant)
    _refused(case.verify(), "UNSUPPORTED_SOURCE")


def test_invalid_deb_magic_refuses(case: _Case) -> None:
    target = case.root / case.package.archive_path
    raw = b"BAD-DEB!" + target.read_bytes()[8:]
    target.write_bytes(raw)
    case.manifest["sources"][0]["artifact_sha256"] = _sha(raw)
    _refused(case.verify(), "UNSUPPORTED_SOURCE")


def test_gzip_ustar_variant_matches_without_extracting(case: _Case) -> None:
    case.archive(variant="gz")
    result = case.verify()
    assert result.artifact_consistency == "MATCH"
    _blocked(result)


def test_truncated_read_refuses_even_when_file_metadata_is_unchanged(
    case: _Case, monkeypatch: pytest.MonkeyPatch
) -> None:
    inode = case.path("python").stat().st_ino
    original = os.read

    def incomplete_read(fd: int, size: int) -> bytes:
        return b"" if os.fstat(fd).st_ino == inode else original(fd, size)

    monkeypatch.setattr(os, "read", incomplete_read)
    _refused(case.verify(), "FILE_CHANGED")


def test_unreadable_input_refuses_instead_of_counting_a_partial_measurement(
    case: _Case, monkeypatch: pytest.MonkeyPatch
) -> None:
    inode = case.path("python").stat().st_ino
    original = os.read

    def denied_read(fd: int, size: int) -> bytes:
        if os.fstat(fd).st_ino == inode:
            raise PermissionError("synthetic unreadable input")
        return original(fd, size)

    monkeypatch.setattr(os, "read", denied_read)
    _refused(case.verify(), "PROVENANCE_INCOMPLETE")


def test_duplicate_manifest_file_identifier_refuses(case: _Case) -> None:
    case.manifest["files"].append(copy.deepcopy(case.file("python")))
    _refused(case.verify(), "CLOSURE_INCOMPLETE")


def test_duplicate_archive_member_refuses(case: _Case) -> None:
    entries = case.data_entries()
    case.archive(data_entries=entries + [entries[0]])
    _refused(case.verify(), "CLOSURE_INCOMPLETE")


def test_archive_member_payload_mismatch_refuses(case: _Case) -> None:
    entries = case.data_entries()
    index = next(index for index, (_, body) in enumerate(entries) if body)
    name, body = entries[index]
    entries[index] = (name, bytes([body[0] ^ 1]) + body[1:])
    case.archive(data_entries=entries)
    _refused(case.verify(), "ARTIFACT_MISMATCH")


def test_archive_symlink_refuses_without_materializing(case: _Case) -> None:
    output = io.BytesIO()
    with tarfile.open(fileobj=output, mode="w", format=tarfile.USTAR_FORMAT) as archive:
        member = tarfile.TarInfo("outside-link")
        member.type, member.linkname = tarfile.SYMTYPE, "../../outside"
        archive.addfile(member)
    case.replace_archive_member("data.tar", output.getvalue())
    _refused(case.verify(), "UNSUPPORTED_SOURCE")


def test_malformed_gzip_refuses(case: _Case) -> None:
    case.archive(variant="gz")
    case.replace_archive_member("data.tar.gz", b"\x1f\x8b\x08\x00" + bytes(64))
    _refused(case.verify())


def test_missing_control_record_refuses(case: _Case) -> None:
    case.replace_archive_member("control.tar", _tar([("postinst", b"must remain inert")]))
    _refused(case.verify(), "PROVENANCE_INCOMPLETE")


@pytest.mark.parametrize(
    "control",
    [
        f"Package: python3.12-minimal\nVersion: {_VERSION}\nArchitecture: arm64\n",
        f"Package: wrong-package\nVersion: {_VERSION}\nArchitecture: amd64\n",
        f"Package: python3.12-minimal\nVersion: {_VERSION}\nVersion: {_VERSION}\nArchitecture: amd64\n",
    ],
)
def test_control_package_architecture_and_duplicate_fields_refuse(
    case: _Case, control: str
) -> None:
    case.replace_archive_member("control.tar", _tar([("control", control.encode())]))
    _refused(case.verify())


def test_archive_trailing_material_refuses(case: _Case) -> None:
    raw = (case.root / case.package.archive_path).read_bytes()
    case.set_archive(raw + b"unexpected archive member")
    _refused(case.verify(), "CLOSURE_INCOMPLETE")


def test_changed_file_identity_without_changed_bytes_refuses(case: _Case) -> None:
    case.file("python")["uid"] += 1
    _refused(case.verify(), "ARTIFACT_MISMATCH")


def test_nonexistent_directory_provenance_member_refuses(case: _Case) -> None:
    entries = case.data_entries()
    missing_name = case.file("stdlib-root")["provenance"]["member"] + "/"
    case.archive(data_entries=[entry for entry in entries if entry[0] != missing_name])
    _refused(case.verify(), "CLOSURE_INCOMPLETE")


def test_malformed_package_declaration_refuses(case: _Case) -> None:
    case.package = replace(case.package, package=[])
    _refused(case.verify())


def test_undersized_decoder_buffer_refuses_instead_of_partial_gzip_reads(case: _Case) -> None:
    case.archive(variant="gz")
    _refused(case.verify(limits=offline.Limits(chunk_bytes=1)), "LIMIT_EXCEEDED")


@pytest.mark.parametrize("limit", ["max_unpacked_bytes", "max_archive_members"])
def test_archive_limits_apply_to_whole_bundle_not_each_package(case: _Case, limit: str) -> None:
    original = (case.root / case.package.archive_path).read_bytes()
    if limit == "max_unpacked_bytes":
        # Both supported archives are raw USTAR here, so their ar sizes equal
        # their decoded sizes, including TAR headers and terminal padding.
        unpacked = 0
        offset = 8
        while offset < len(original):
            header = original[offset : offset + 60]
            size = int(header[48:58])
            if header[:16].startswith((b"control.tar", b"data.tar")):
                unpacked += size
            offset += 60 + size + size % 2
        limits = offline.Limits(max_unpacked_bytes=unpacked + 1)
    else:
        # The control archive has two records; data has the explicit image
        # directories plus every regular image file.
        members = len(case.manifest["files"]) + 2
        limits = offline.Limits(max_archive_members=members + 1)

    # Establish that an individual complete package is below this ceiling.
    assert case.verify(limits=limits).artifact_consistency == "MATCH"

    source = copy.deepcopy(case.manifest["sources"][0])
    source["source_id"] = "second-source"
    case.manifest["sources"].append(source)
    case.file("seccomp")["provenance"]["source_id"] = "second-source"
    original_members = case.package.members
    case.package = replace(
        case.package,
        members=tuple(
            replace(member, file_id="") if member.file_id == "seccomp" else member
            for member in original_members
        ),
    )
    second = replace(
        case.package,
        source_id="second-source",
        archive_path="sources/second.deb",
        proof_path="sources/second-proof.bin",
        members=tuple(
            member if member.file_id == "seccomp" else replace(member, file_id="")
            for member in original_members
        ),
    )
    (case.root / second.archive_path).write_bytes(original)
    (case.root / second.proof_path).write_bytes((case.root / case.package.proof_path).read_bytes())
    manifest_raw = canonical_json_bytes(case.manifest)
    contract.parse_proposed_manifest(manifest_raw)
    result = offline.verify_offline(
        case.root,
        manifest_raw=manifest_raw,
        ubuntu_packages=(case.package, second),
        limits=limits,
    )
    _refused(result, "LIMIT_EXCEEDED")
    assert result.artifact_consistency == "INCOMPLETE"


@pytest.mark.parametrize(
    "field,value",
    [
        ("member", None),
        ("member", ["not-a-path"]),
        ("file_id", ["python"]),
        ("sha256", None),
        ("sha256", ["not-a-digest"]),
        ("size", "12"),
        ("size", True),
        ("size", -1),
        ("size", float("nan")),
    ],
)
def test_malformed_member_declarations_refuse_without_native_exception(
    case: _Case, field: str, value: Any
) -> None:
    bad = replace(case.package.members[0], **{field: value})
    case.package = replace(case.package, members=(bad,) + case.package.members[1:])
    _refused(case.verify(), "FORMAT_REJECTED")


@pytest.mark.parametrize(
    "limits",
    [
        offline.Limits(max_file_bytes=8),
        offline.Limits(max_files=2),
        offline.Limits(max_total_bytes=32),
        offline.Limits(max_archive_members=2),
        offline.Limits(max_unpacked_bytes=1024),
        offline.Limits(working_memory_bytes=262144),
        offline.Limits(max_path_depth=2),
    ],
)
def test_size_count_memory_unpack_and_depth_limits_refuse(
    case: _Case, limits: offline.Limits
) -> None:
    _refused(case.verify(limits=limits), "LIMIT_EXCEEDED")


@pytest.mark.parametrize("timeout", [0, -1, float("nan"), float("inf"), True, 61])
def test_invalid_deadline_limits_refuse(case: _Case, timeout: float) -> None:
    _refused(case.verify(limits=offline.Limits(timeout_s=timeout)), "LIMIT_EXCEEDED")


def test_elapsed_deadline_refuses_without_partial_success(
    case: _Case, monkeypatch: pytest.MonkeyPatch
) -> None:
    timestamps = iter([0.0, 0.0, 61.0])
    monkeypatch.setattr(offline.time, "monotonic", lambda: next(timestamps, 61.0))
    _refused(case.verify(), "LIMIT_EXCEEDED")


def test_missing_independent_trust_context_never_accepts_matching_bytes(case: _Case) -> None:
    result = case.verify()
    assert result.artifact_consistency == "MATCH"
    _blocked(result)


@pytest.mark.parametrize(
    "context",
    [{"accepted": True}, {"expected_sha256": "local"}, {"authorization": "ACCEPTED"}, b"test-root"],
)
def test_local_measurement_or_self_asserted_trust_context_refuses(
    case: _Case, context: Any
) -> None:
    _refused(case.verify(trust_context=context), "ACCEPTANCE_REJECTED")


def test_forged_structurally_valid_acceptance_refuses(case: _Case) -> None:
    acceptance = json.loads((_FIXTURES / "acceptance.json").read_bytes())
    acceptance["payload"].update(
        manifest_sha256=_sha(canonical_json_bytes(case.manifest)),
        policy_sha256=_sha(canonical_json_bytes(case.policy())),
    )
    acceptance["payload_digest"] = _sha(canonical_json_bytes(acceptance["payload"]))
    raw = canonical_json_bytes(acceptance)
    assert contract.parse_acceptance(raw).authorization == "UNVERIFIED"
    _refused(case.verify(acceptance_raw=raw), "ACCEPTANCE_REJECTED")


def test_incomplete_acceptance_refuses(case: _Case) -> None:
    acceptance = json.loads((_FIXTURES / "acceptance.json").read_bytes())
    acceptance["signatures"] = acceptance["signatures"][:1]
    _refused(case.verify(acceptance_raw=canonical_json_bytes(acceptance)))


@pytest.mark.parametrize(
    "path",
    [
        "../outside",
        "/etc/passwd",
        "sources/../../outside",
        "sources//python.deb",
        "sources\\python.deb",
    ],
)
def test_archive_paths_outside_explicit_input_directory_refuse(case: _Case, path: str) -> None:
    case.package = replace(case.package, archive_path=path)
    _refused(case.verify(), "FILE_POLICY_REJECTED")


def test_manifest_path_traversal_refuses_before_io(case: _Case) -> None:
    case.file("python")["path"] = "/observer/../../outside"
    _refused(case.verify())


def test_archive_member_traversal_refuses(case: _Case) -> None:
    entries = case.data_entries()
    case.archive(data_entries=entries + [("../../outside", b"must not be extracted")])
    _refused(case.verify(), "FILE_POLICY_REJECTED")


def test_absent_source_adapter_is_incomplete(case: _Case) -> None:
    # Remove the unrelated staged source files too, so this exercises missing
    # evidence rather than the separate unexpected-file boundary.
    (case.root / case.package.archive_path).unlink()
    (case.root / case.package.proof_path).unlink()
    (case.root / "sources").rmdir()
    result = offline.verify_offline(case.root, manifest_raw=canonical_json_bytes(case.manifest))
    _refused(result, "UNSUPPORTED_SOURCE")


def test_corrupted_proof_bytes_refuse(case: _Case) -> None:
    (case.root / case.package.proof_path).write_bytes(b"substituted opaque proof")
    _refused(case.verify(), "ARTIFACT_MISMATCH")


def test_target_and_maintainer_code_remain_inert(
    case: _Case, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    sentinel = tmp_path / "executed"
    payload = f"from pathlib import Path\nPath({str(sentinel)!r}).write_text('executed')\n".encode()
    target = case.file("entrypoint")
    target.update(size=len(payload), sha256=_sha(payload))
    case.payloads["entrypoint"] = payload
    case.path("entrypoint").write_bytes(payload)
    for item in case.manifest["imports"]:
        if item.get("file_id") == "entrypoint":
            item["sha256"] = target["sha256"]
    case.package = replace(
        case.package,
        members=tuple(
            replace(member, sha256=target["sha256"], size=len(payload))
            if member.file_id == "entrypoint"
            else member
            for member in case.package.members
        ),
    )
    case.archive(maintainer=payload)

    def forbidden(*args: Any, **kwargs: Any) -> None:
        pytest.fail("offline verifier attempted process execution")

    monkeypatch.setattr(subprocess, "Popen", forbidden)
    monkeypatch.setattr(os, "system", forbidden)
    result = case.verify()
    assert result.artifact_consistency == "MATCH"
    _blocked(result)
    assert not sentinel.exists()


def test_verifier_source_has_no_execution_or_target_import_adapter() -> None:
    root = Path(__file__).parents[2]
    for name in ("stage9_observer_offline.py", "stage9_observer_ubuntu.py"):
        tree = ast.parse((root / "deployment" / name).read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                if isinstance(node.func, ast.Name):
                    assert node.func.id not in {"eval", "exec", "compile", "__import__"}
                if isinstance(node.func, ast.Attribute):
                    assert node.func.attr not in {
                        "extract",
                        "extractall",
                        "system",
                        "Popen",
                        "run",
                        "execve",
                        "execv",
                        "spawnv",
                        "load_module",
                        "exec_module",
                    }
