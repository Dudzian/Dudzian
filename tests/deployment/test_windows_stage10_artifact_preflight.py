"""Hermetic immutable-download tests; TEST_ONLY bytes never qualify a Windows host."""

from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import shutil
import stat
from types import SimpleNamespace
import zipfile

import pytest

from deployment.platform_evidence import evidence_document
from deployment.windows_installer.build import installed_payload_digest
from deployment.windows_stage9_evidence_contract import (
    CLEAN_INSTALL_PRECEREMONY_PROOFS,
    POST_ENROLLMENT_QUALIFICATION_STATE,
)
from deployment.windows_stage9_production_trust import (
    CEREMONY_ID,
    PUBLIC_PACKAGE_FILENAMES,
    RELEASE_PAYLOAD_DIGEST,
    ProductionTrustUnavailable,
)
import deployment.windows_stage10_artifact_preflight as preflight

REVISION = "a" * 40


def digest(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def encode(value: object) -> bytes:
    return (json.dumps(value) + "\n").encode()


@pytest.fixture
def staged(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    root = tmp_path / "staging"
    root.mkdir()
    package = root / CEREMONY_ID
    package.mkdir()
    for name in PUBLIC_PACKAGE_FILENAMES:
        (package / name).write_bytes(b"{}\n")
    trust_digest = digest(b"{}\n")
    msi_name = "CryptoHunter-0.1.0-windows-x64.msi"
    msi = b"TEST_ONLY_MSI_NOT_INSTALLABLE"
    files = {path: digest(name.encode()) for name, path in preflight.EXECUTABLE_PATHS.items()}
    pins = json.loads(
        (Path(preflight.__file__).parent / "windows_installer/pins.json").read_text()
    )["postgresql"]
    manifest = {
        "schema_version": 1,
        "product_version": "0.1.0",
        "architecture": "x64",
        "wix_version": "7.0.0",
        "postgresql_version": "17.11",
        "postgresql_packaging_revision": "4",
        "postgresql_source_url": pins["source_url"],
        "postgresql_payload_sha256": pins["sha256"],
        "msi": {"file": msi_name, "sha256": digest(msi)},
        "production_executables": {
            name: files[path] for name, path in preflight.EXECUTABLE_PATHS.items()
        },
        "installed_files": files,
        "installed_payload_sha256": installed_payload_digest(files),
    }
    receipt = {
        "schema_version": 1,
        "source_revision": REVISION,
        "ci_provider": "https://github.com",
        "ci_run_id": "42",
        "runner_os": "Windows",
        "runner_arch": "X64",
        "probe_id": "cryptohunter.windows.clean-install.v1",
        "msi_sha256": digest(msi),
        "manifest_sha256": digest(encode(manifest)),
        "product_version": "0.1.0",
        "install_exit_code": 0,
        "uninstall_exit_code": 0,
        "proofs": {name: "PASS" for name in CLEAN_INSTALL_PRECEREMONY_PROOFS},
        "post_enrollment_live_qualification": POST_ENROLLMENT_QUALIFICATION_STATE,
        "production_trust_ceremony_id": CEREMONY_ID,
        "production_trust_package_manifest_sha256": trust_digest,
        "production_trust_artifact_run_id": "41",
    }
    evidence = evidence_document(
        platform_name="WINDOWS",
        source_revision=REVISION,
        ci_provider="https://github.com",
        ci_run_id="42",
        runner_os="Windows",
        runner_arch="X64",
        results=[
            {
                "item": "WINDOWS_CLEAN_INSTALL",
                "status": "PASS",
                "evidence_class": "LIVE_WINDOWS_INTEGRATION",
                "test_or_probe": "canonical Stage-9 MSI clean-install proof",
                "details": f"MSI sha256={digest(msi)}; version=0.1.0; architecture=X64",
            }
        ],
    )
    evidence["production_trust"] = {
        "ceremony_id": CEREMONY_ID,
        "package_manifest_sha256": trust_digest,
        "artifact_run_id": "41",
        "frozen_production_trust": "PASS",
    }
    artifact = {
        "id": 9,
        "name": preflight.ARTIFACT_NAME,
        "digest": "",
        "expired": False,
        "archive_download_url": "https://api.github.com/repos/Dudzian/Dudzian/actions/artifacts/9/zip",
        "workflow_run": {"id": 42, "head_sha": REVISION, "head_branch": "main"},
    }
    run = {
        "id": 42,
        "head_sha": REVISION,
        "head_branch": "main",
        "event": "workflow_dispatch",
        "path": preflight.WORKFLOW,
        "status": "completed",
        "conclusion": "success",
        "repository": {"full_name": "Dudzian/Dudzian"},
    }
    args = {
        "archive": root / "qualified.zip",
        "artifact_metadata": root / "artifact.json",
        "workflow_run_metadata": root / "run.json",
        "production_trust_package": package,
        "allowed_root": root,
        "expected_artifact_id": "9",
        "expected_archive_sha256": "",
        "expected_revision": REVISION,
        "expected_run_id": "42",
        "expected_production_trust_run_id": "41",
        "expected_production_trust_manifest_sha256": trust_digest,
    }
    state = SimpleNamespace(
        args=args,
        manifest=manifest,
        receipt=receipt,
        evidence=evidence,
        artifact=artifact,
        run=run,
        members={msi_name: msi},
        root=root,
    )
    state.trust_calls = []

    def verified_public_package(path: Path):
        state.trust_calls.append(path)
        return SimpleNamespace(release_payload_digest=RELEASE_PAYLOAD_DIGEST)

    # Trust semantics are covered by the existing trust suite; this boundary
    # isolates download/proof binding. Separate tests below use the real verifier.
    monkeypatch.setattr(preflight, "load_production_trust", verified_public_package)
    rebuild(state)
    return state


def rebuild(state, *, rebind: bool = True):
    members = dict(state.members)
    members.setdefault("installer-manifest.json", encode(state.manifest))
    members.setdefault("windows-clean-install-receipt.json", encode(state.receipt))
    members.setdefault("windows-clean-install-evidence.json", encode(state.evidence))
    with zipfile.ZipFile(state.args["archive"], "w") as archive:
        for name, data in members.items():
            archive.writestr(name, data)
    archive_digest = digest(state.args["archive"].read_bytes())
    if rebind:
        state.args["expected_archive_sha256"] = archive_digest
        state.artifact["digest"] = "sha256:" + archive_digest
    state.args["artifact_metadata"].write_bytes(encode(state.artifact))
    state.args["workflow_run_metadata"].write_bytes(encode(state.run))


def rebind_manifest(state):
    state.receipt["manifest_sha256"] = digest(encode(state.manifest))


def test_exact_download_rehearsal_is_read_only_and_never_stage10_pass(staged):
    before = {
        path.relative_to(staged.root): path.read_bytes()
        for path in staged.root.rglob("*")
        if path.is_file()
    }
    report = preflight.artifact_preflight(**staged.args)
    assert report["overall_status"] == "PASS", report
    assert report["stage10_qualification"] == "NOT_RUN"
    assert report["stage10_pass"] is False
    assert report["details"]["source_revision"] == REVISION
    assert staged.trust_calls == [staged.args["production_trust_package"]]
    after = {
        path.relative_to(staged.root): path.read_bytes()
        for path in staged.root.rglob("*")
        if path.is_file()
    }
    assert after == before
    assert not list(staged.root.glob("stage10-artifact-rehearsal-*"))


@pytest.mark.parametrize(
    "which,field,value",
    [
        ("receipt", "ci_run_id", "43"),
        ("evidence", "ci_run_id", "43"),
        ("receipt", "source_revision", "b" * 40),
        ("evidence", "source_revision", "b" * 40),
        ("receipt", "msi_sha256", "0" * 64),
        ("receipt", "manifest_sha256", "0" * 64),
        ("receipt", "production_trust_artifact_run_id", "40"),
        ("receipt", "production_trust_package_manifest_sha256", "0" * 64),
        ("receipt", "runner_arch", "ARM64"),
        ("evidence", "runner_arch", "ARM64"),
        ("receipt", "install_exit_code", 1),
        ("receipt", "uninstall_exit_code", 1),
        ("receipt", "post_enrollment_live_qualification", "PASS"),
        ("receipt", "production_trust_package_manifest_sha256", None),
    ],
)
def test_mixed_run_revision_hashes_and_missing_values_fail_closed(staged, which, field, value):
    getattr(staged, which)[field] = value
    rebuild(staged)
    report = preflight.artifact_preflight(**staged.args)
    assert report["overall_status"] == "FAIL", report
    assert report["stage10_pass"] is False
    assert report["reason"]


@pytest.mark.parametrize("which", ["receipt", "evidence"])
def test_missing_trust_binding_fails(staged, which):
    if which == "receipt":
        staged.receipt.pop("production_trust_ceremony_id")
    else:
        staged.evidence.pop("production_trust")
    rebuild(staged)
    assert preflight.artifact_preflight(**staged.args)["status"] == "FAIL"


def test_substituted_msi_rejected_even_with_rebound_archive(staged):
    staged.members[next(iter(staged.members))] = b"CHANGED_TEST_ONLY_MSI"
    rebuild(staged)
    assert preflight.artifact_preflight(**staged.args)["status"] == "FAIL"


def test_archive_digest_cannot_be_self_declared_in_download(staged):
    staged.members[next(iter(staged.members))] = b"CHANGED"
    rebuild(staged, rebind=False)
    report = preflight.artifact_preflight(**staged.args)
    assert report["status"] == "FAIL"
    assert "independent pin" in report["reason"]


@pytest.mark.parametrize(
    "mutation",
    [
        lambda s: s.artifact.update(name="windows-clean-install-evidence"),
        lambda s: s.artifact.update(id=10),
        lambda s: s.artifact.update(expired=True),
        lambda s: s.artifact.update(digest="sha256:" + "0" * 64),
        lambda s: s.artifact["workflow_run"].update(id=43),
        lambda s: s.artifact["workflow_run"].update(head_sha="b" * 40),
        lambda s: s.artifact.update(archive_download_url="https://evil.invalid/qualified.zip"),
        lambda s: s.run.update(event="pull_request"),
        lambda s: s.run.update(head_branch="feature"),
        lambda s: s.run.update(conclusion="failure"),
        lambda s: s.run.update(path="other.yml"),
        lambda s: s.run.update(repository={"full_name": "attacker/Dudzian"}),
    ],
)
def test_foreign_stale_or_untrusted_metadata_rejected(staged, mutation):
    mutation(staged)
    rebuild(staged, rebind=False)
    assert preflight.artifact_preflight(**staged.args)["status"] == "FAIL"


@pytest.mark.parametrize(
    "member,data",
    [
        ("installer-manifest.json", b"{"),
        ("windows-clean-install-receipt.json", b"[]"),
        ("windows-clean-install-evidence.json", b'{"results":NaN}'),
        ("installer-manifest.json", b'{"schema_version":1,"schema_version":1}'),
        ("installer-manifest.json", b"\xff"),
    ],
)
def test_corrupt_and_ambiguous_json_fails(staged, member, data):
    staged.members[member] = data
    rebuild(staged)
    report = preflight.artifact_preflight(**staged.args)
    assert report["status"] == "FAIL", report


@pytest.mark.parametrize(
    "mutation",
    [
        lambda m: m.update(installed_payload_sha256="0" * 64),
        lambda m: m.update(installed_files={}),
        lambda m: m.update(production_executables={}),
        lambda m: m["installed_files"].update({"CryptoHunterBackend.exe": None}),
        lambda m: m["installed_files"].update({"../outside.exe": "a" * 64}),
        lambda m: m["installed_files"].update({"C:/outside.exe": "a" * 64}),
        lambda m: m["installed_files"].update({"CON": "a" * 64}),
        lambda m: m["installed_files"].update({"cryptohunterbackend.exe": "a" * 64}),
        lambda m: m.update(postgresql_payload_sha256="0" * 64),
        lambda m: m["msi"].update(file="../../outside.msi"),
        lambda m: m.update(msi=[]),
    ],
)
def test_payload_and_manifest_location_contracts_fail_closed(staged, mutation):
    mutation(staged.manifest)
    rebind_manifest(staged)
    rebuild(staged)
    assert preflight.artifact_preflight(**staged.args)["status"] == "FAIL"


@pytest.mark.parametrize(
    "name",
    [
        "../outside.msi",
        "/absolute.msi",
        "C:\\outside.msi",
        "nested/product.msi",
        "CryptoHunter-0.1.0-windows-x64.msi:evil",
    ],
)
def test_archive_traversal_and_unauthorized_members_fail(staged, name):
    staged.members = {name: b"TEST_ONLY"}
    rebuild(staged)
    assert preflight.artifact_preflight(**staged.args)["status"] == "FAIL"
    assert not (staged.root.parent / "outside.msi").exists()


def test_archive_symbolic_link_rejected(staged):
    rebuild(staged)
    with zipfile.ZipFile(staged.args["archive"], "r") as archive:
        data = {name: archive.read(name) for name in archive.namelist()}
    with zipfile.ZipFile(staged.args["archive"], "w") as archive:
        for name, content in data.items():
            info = zipfile.ZipInfo(name)
            info.create_system = 3
            info.external_attr = (stat.S_IFLNK | 0o777) << 16
            archive.writestr(info, content)
    staged.args["expected_archive_sha256"] = digest(staged.args["archive"].read_bytes())
    staged.artifact["digest"] = "sha256:" + staged.args["expected_archive_sha256"]
    staged.args["artifact_metadata"].write_bytes(encode(staged.artifact))
    assert preflight.artifact_preflight(**staged.args)["status"] == "FAIL"


def test_input_outside_authorized_staging_is_rejected(staged):
    outside = staged.root.parent / "outside.zip"
    shutil.copyfile(staged.args["archive"], outside)
    staged.args["archive"] = outside
    assert preflight.artifact_preflight(**staged.args)["status"] == "FAIL"


def test_linked_input_is_rejected(staged):
    link = staged.root / "linked.zip"
    link.symlink_to(staged.args["archive"])
    staged.args["archive"] = link
    assert preflight.artifact_preflight(**staged.args)["status"] == "FAIL"


def test_existing_receipt_validator_requires_no_current_runner_environment(staged, monkeypatch):
    for name in ("GITHUB_RUN_ID", "GITHUB_SHA", "GITHUB_SERVER_URL", "RUNNER_OS", "RUNNER_ARCH"):
        monkeypatch.delenv(name, raising=False)
    assert preflight.artifact_preflight(**staged.args)["status"] == "PASS"


def test_real_trust_validator_rejects_synthetic_package(staged, monkeypatch):
    from deployment.windows_stage9_production_trust import load_production_trust

    monkeypatch.setattr(preflight, "load_production_trust", load_production_trust)
    report = preflight.artifact_preflight(**staged.args)
    assert report["status"] == "FAIL"
    assert "ProductionTrustUnavailable" in report["reason"]


def test_unavailable_current_trust_is_not_ignored(staged, monkeypatch):
    def unavailable(path):
        raise ProductionTrustUnavailable("TEST_ONLY_EXPIRED_PACKAGE")

    monkeypatch.setattr(preflight, "load_production_trust", unavailable)
    report = preflight.artifact_preflight(**staged.args)
    assert report["status"] == "FAIL"
    assert "TEST_ONLY_EXPIRED_PACKAGE" in report["reason"]


def cli_args(staged, output: Path):
    args = []
    for name, value in staged.args.items():
        args.extend(["--" + name.replace("_", "-"), str(value)])
    return args + ["--output", str(output)]


def test_cli_writes_preparatory_report_and_refuses_overwrite(staged):
    report = staged.root / "report.json"
    assert preflight.main(cli_args(staged, report)) == 0
    assert json.loads(report.read_text())["stage10_pass"] is False
    assert preflight.main(cli_args(staged, report)) == 1


def test_cli_emits_failure_json_for_missing_archive(staged):
    staged.args["archive"].unlink()
    report = staged.root / "report.json"
    assert preflight.main(cli_args(staged, report)) == 1
    assert json.loads(report.read_text())["overall_status"] == "FAIL"


def test_cli_cannot_write_outside_authorized_staging(staged):
    output = staged.root.parent / "unauthorized.json"
    assert preflight.main(cli_args(staged, output)) == 1
    assert not output.exists()


def test_operator_supplied_real_public_package_uses_full_verifier(staged, monkeypatch):
    source = os.environ.get("CRYPTOHUNTER_STAGE9_FINAL_PACKAGE")
    if not source:
        pytest.skip("genuine public Production Trust package must be operator supplied")
    from deployment.windows_stage9_production_trust import load_production_trust

    monkeypatch.setattr(preflight, "load_production_trust", load_production_trust)
    shutil.rmtree(staged.args["production_trust_package"])
    shutil.copytree(source, staged.args["production_trust_package"])
    public_digest = digest(
        (staged.args["production_trust_package"] / "package_manifest.json").read_bytes()
    )
    staged.args["expected_production_trust_manifest_sha256"] = public_digest
    staged.receipt["production_trust_package_manifest_sha256"] = public_digest
    staged.evidence["production_trust"]["package_manifest_sha256"] = public_digest
    rebuild(staged)
    assert preflight.artifact_preflight(**staged.args)["status"] == "PASS"


@pytest.mark.parametrize(
    "which,field,value",
    [
        ("manifest", "schema_version", True),
        ("receipt", "schema_version", True),
        ("evidence", "schema_version", True),
        ("receipt", "install_exit_code", False),
        ("receipt", "uninstall_exit_code", False),
    ],
)
def test_boolean_json_cannot_impersonate_integer_contract(staged, which, field, value):
    getattr(staged, which)[field] = value
    if which == "manifest":
        rebind_manifest(staged)
    rebuild(staged)
    assert preflight.artifact_preflight(**staged.args)["status"] == "FAIL"


def test_public_package_reparse_child_is_rejected_before_verifier(staged, monkeypatch):
    original = preflight._is_link
    monkeypatch.setattr(
        preflight, "_is_link", lambda path: path.name == "package_manifest.json" or original(path)
    )
    assert preflight.artifact_preflight(**staged.args)["status"] == "FAIL"
    assert staged.trust_calls == []


def test_windows_reparse_attribute_is_detected(monkeypatch):
    monkeypatch.setattr(Path, "is_symlink", lambda path: False)
    monkeypatch.setattr(Path, "lstat", lambda path: SimpleNamespace(st_file_attributes=0x400))
    assert preflight._is_link(Path("TEST_ONLY_REPARSE")) is True


def test_evidence_cannot_describe_another_msi_in_same_run(staged):
    staged.evidence["results"][0]["details"] = "MSI sha256=" + "0" * 64
    rebuild(staged)
    assert preflight.artifact_preflight(**staged.args)["status"] == "FAIL"


def test_duplicate_archive_member_is_rejected(staged):
    with zipfile.ZipFile(staged.args["archive"], "a") as archive:
        archive.writestr("INSTALLER-MANIFEST.JSON", encode(staged.manifest))
    staged.args["expected_archive_sha256"] = digest(staged.args["archive"].read_bytes())
    staged.artifact["digest"] = "sha256:" + staged.args["expected_archive_sha256"]
    staged.args["artifact_metadata"].write_bytes(encode(staged.artifact))
    assert preflight.artifact_preflight(**staged.args)["status"] == "FAIL"


@pytest.mark.parametrize(
    "pin,value",
    [
        ("expected_artifact_id", ""),
        ("expected_run_id", None),
        ("expected_archive_sha256", "sha256:" + "0" * 64),
        ("expected_revision", "a" * 7),
        ("expected_production_trust_run_id", "0"),
        ("expected_production_trust_manifest_sha256", ""),
    ],
)
def test_required_pins_cannot_be_omitted_or_inferred(staged, pin, value):
    staged.args[pin] = value
    assert preflight.artifact_preflight(**staged.args)["status"] == "FAIL"


def test_cli_never_modifies_public_trust_package_with_report(staged):
    report = staged.args["production_trust_package"] / "report.json"
    assert preflight.main(cli_args(staged, report)) == 1
    assert not report.exists()


def test_public_manifest_reparse_is_rejected_before_any_hash_read(staged, monkeypatch):
    original_link = preflight._is_link
    monkeypatch.setattr(
        preflight,
        "_is_link",
        lambda path: path.name == "package_manifest.json" or original_link(path),
    )
    original_hash = preflight.sha256
    reads = []

    def hash_spy(path):
        reads.append(path)
        return original_hash(path)

    monkeypatch.setattr(preflight, "sha256", hash_spy)
    assert preflight.artifact_preflight(**staged.args)["status"] == "FAIL"
    assert staged.args["production_trust_package"] / "package_manifest.json" not in reads
