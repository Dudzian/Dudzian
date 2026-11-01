"""Przekazanie bajtów nie omija niezależnego zatwierdzenia ani bramki Linux."""

from __future__ import annotations

import copy
import base64
import gzip
import io
import json
import os
from pathlib import Path
import stat
import subprocess
import sys
import tomllib
from typing import Any
from urllib.error import HTTPError
from urllib.parse import urlunsplit
import zipfile

import pytest
import yaml

from deployment.linux_evidence_handoff import (
    GitHubClient,
    HandoffError,
    MAX_BUNDLE_BYTES,
    MAX_TRANSPORT_BYTES,
    fetch_bundle,
    ingress,
    pack_bundle,
    require_environment,
    unpack_transport,
)
import deployment.linux_evidence_handoff as handoff

ROOT = Path(__file__).resolve().parents[2]
REPOSITORY = "Dudzian/Dudzian"
NAME = "stage9-linux-bundle-123-2"
POLICY_COMMIT = "b" * 40


def _zip(names: list[str] | None = None, raw: bytes = b'{"untrusted":"bytes"}') -> bytes:
    output = io.BytesIO()
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name in names or ["bundle.json"]:
            archive.writestr(name, raw)
    return output.getvalue()


class Transport:
    def __init__(self) -> None:
        self.artifact: dict[str, Any] = {
            "name": NAME,
            "id": 45,
            "expired": False,
            "size_in_bytes": 150,
            "workflow_run": {"id": 67},
        }
        self.run: dict[str, Any] = {
            "path": ".github/workflows/linux-evidence-ingress.yml",
            "event": "workflow_dispatch",
            "head_branch": "main",
            "status": "completed",
            "conclusion": "success",
            "repository": {"full_name": REPOSITORY},
            "head_repository": {"full_name": REPOSITORY},
            "head_sha": "a" * 40,
        }
        self.environment: dict[str, Any] = {
            "name": "stage9-linux-evidence",
            "can_admins_bypass": False,
            "protection_rules": [
                {
                    "type": "required_reviewers",
                    "prevent_self_review": True,
                    "reviewers": [
                        {
                            "type": "User",
                            "reviewer": {"id": 1, "login": "independent-reviewer", "type": "User"},
                        }
                    ],
                }
            ],
            "deployment_branch_policy": {
                "protected_branches": True,
                "custom_branch_policies": False,
            },
        }
        self.branches = {
            name: {"name": name, "protected": True, "commit": {"sha": revision}}
            for name, revision in (("main", "a" * 40), (handoff.POLICY_BRANCH, POLICY_COMMIT))
        }
        self.protection = {
            "enforce_admins": {"enabled": True},
            "allow_force_pushes": {"enabled": False},
            "allow_deletions": {"enabled": False},
            "required_conversation_resolution": {"enabled": True},
            "required_pull_request_reviews": {
                "dismiss_stale_reviews": True,
                "require_code_owner_reviews": True,
                "require_last_push_approval": True,
                "required_approving_review_count": 2,
                "bypass_pull_request_allowances": {"users": [], "teams": [], "apps": []},
            },
            "required_status_checks": {
                "strict": True,
                "checks": [
                    {"context": context, "app_id": 15368}
                    for context in sorted(handoff.REQUIRED_PROTECTION_CHECKS)
                ],
            },
        }
        self.owners = b"# Zatwierdzony profil testowy\n* @reviewer-one @reviewer-two\n"
        self.permission: dict[str, Any] = {"permission": "write"}
        self.matches = [self.artifact]
        self.payload = _zip()
        self.paths: list[str] = []

    def json(self, path: str) -> dict[str, Any]:
        self.paths.append(path)
        if path.startswith("/actions/artifacts?"):
            return {"total_count": len(self.matches), "artifacts": self.matches}
        if path.startswith("/environments/"):
            return self.environment
        if path.startswith("/branches/"):
            branch = path.split("/")[2]
            return self.protection if path.endswith("/protection") else self.branches[branch]
        if path.startswith("/contents/.github/CODEOWNERS?"):
            return {
                "type": "file",
                "encoding": "base64",
                "content": base64.b64encode(self.owners).decode(),
            }
        if path.startswith("/collaborators/"):
            login = path.split("/")[2]
            return {
                **self.permission,
                "user": {"login": login, "id": sum(ord(char) for char in login), "type": "User"},
            }
        assert path == "/actions/runs/67"
        return self.run

    def archive(self, artifact_id: int) -> bytes:
        assert artifact_id == 45
        return self.payload


def test_signed_data_transport_preserves_exact_bytes_without_claiming_pass(tmp_path: Path) -> None:
    transport = Transport()
    target = tmp_path / "bundle.json"
    fetch_bundle(transport, REPOSITORY, "123", "2", target)
    assert target.read_bytes() == b'{"untrusted":"bytes"}'
    assert json.loads(target.read_bytes()) == {"untrusted": "bytes"}
    assert any("name=stage9-linux-bundle-123-2" in path for path in transport.paths)


@pytest.mark.parametrize(
    "mutation",
    [
        lambda transport: transport.matches.clear(),
        lambda transport: transport.matches.append(copy.deepcopy(transport.artifact)),
        lambda transport: transport.artifact.update(expired=True),
        lambda transport: transport.artifact.update(size_in_bytes=10**9),
        lambda transport: transport.artifact.update(id=True),
        lambda transport: transport.artifact.update(workflow_run={}),
        lambda transport: transport.artifact.update(name="stage9-linux-bundle-123-1"),
        lambda transport: transport.run.update(path=".github/workflows/unreviewed.yml"),
        lambda transport: transport.run.update(event="pull_request"),
        lambda transport: transport.run.update(head_branch="feature"),
        lambda transport: transport.run.update(conclusion="failure"),
        lambda transport: transport.run.update(status="in_progress"),
        lambda transport: transport.run.update(head_repository={"full_name": "attacker/repo"}),
        lambda transport: transport.run.update(head_sha="claimed-revision"),
    ],
)
def test_transport_rejects_missing_duplicate_stale_and_unapproved_artifacts(
    tmp_path: Path, mutation
) -> None:
    transport = Transport()
    mutation(transport)
    target = tmp_path / "bundle.json"
    with pytest.raises(HandoffError):
        fetch_bundle(transport, REPOSITORY, "123", "2", target)
    assert not target.exists()


@pytest.mark.parametrize(
    "payload",
    [
        b"not a ZIP",
        _zip(["../bundle.json"]),
        _zip(["bundle.json", "extra.txt"]),
        _zip(["folder/bundle.json"]),
        _zip(raw=b"x" * (MAX_BUNDLE_BYTES + 1)),
        _zip(raw=b'{"status":"PASS","status":"FAIL"}'),
        _zip(raw=b'{"status":NaN}'),
        _zip(raw=b"[]"),
    ],
)
def test_transport_refuses_unbounded_or_ambiguous_archives(tmp_path: Path, payload: bytes) -> None:
    transport = Transport()
    transport.payload = payload
    with pytest.raises(HandoffError):
        fetch_bundle(transport, REPOSITORY, "123", "2", tmp_path / "bundle.json")


def test_symlink_archive_is_not_materialized(tmp_path: Path) -> None:
    output = io.BytesIO()
    with zipfile.ZipFile(output, "w") as archive:
        member = zipfile.ZipInfo("bundle.json")
        member.create_system = 3
        member.external_attr = (stat.S_IFLNK | 0o777) << 16
        archive.writestr(member, "some-other-file")
    transport = Transport()
    transport.payload = output.getvalue()
    with pytest.raises(HandoffError):
        fetch_bundle(transport, REPOSITORY, "123", "2", tmp_path / "bundle.json")


def test_all_catalog_pages_are_checked_before_accepting_unique_artifact(tmp_path: Path) -> None:
    class PagedTransport(Transport):
        def json(self, path: str) -> dict[str, Any]:
            if path.endswith("&page=1"):
                return {
                    "total_count": 101,
                    "artifacts": [self.artifact] + [{"name": "other"}] * 99,
                }
            if path.endswith("&page=2"):
                return {"total_count": 101, "artifacts": [self.artifact]}
            return super().json(path)

    with pytest.raises(HandoffError, match="dokładnie jeden"):
        fetch_bundle(PagedTransport(), REPOSITORY, "123", "2", tmp_path / "bundle.json")


def test_ingress_treats_shell_metacharacters_as_literal_data(tmp_path: Path) -> None:
    target = tmp_path / "bundle.json"
    sentinel = tmp_path / "should-not-exist"
    raw = json.dumps({"value": f"$(touch {sentinel}); `touch {sentinel}`"}).encode()
    ingress(raw, target)
    assert target.read_bytes() == raw
    assert not sentinel.exists()
    with pytest.raises(FileExistsError):
        ingress(b"{}", target)


def test_ingress_cli_never_echoes_untrusted_input(tmp_path: Path) -> None:
    raw = '{"private-looking-input":"do-not-print"}'
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "deployment.linux_evidence_handoff",
            "ingress",
            "--output",
            str(tmp_path / "bundle.json"),
        ],
        cwd=ROOT,
        env={**os.environ, "LINUX_EVIDENCE_BUNDLE": raw},
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0
    assert completed.stdout == completed.stderr == ""


def test_large_signed_bundle_can_cross_dispatch_limit_without_changing_bytes(
    tmp_path: Path,
) -> None:
    raw = json.dumps({"observations": "typowa powtarzalna obserwacja" * 3000}).encode()
    assert MAX_TRANSPORT_BYTES < len(raw) < MAX_BUNDLE_BYTES
    original, packed = tmp_path / "bundle.json", tmp_path / "transport.txt"
    original.write_bytes(raw)
    pack_bundle(original, packed)
    assert packed.stat().st_size < MAX_TRANSPORT_BYTES
    assert unpack_transport(packed.read_bytes()) == raw
    target = tmp_path / "received.json"
    ingress(unpack_transport(packed.read_bytes()), target)
    assert target.read_bytes() == raw


@pytest.mark.parametrize(
    "compressed",
    [
        gzip.compress(b"x" * (MAX_BUNDLE_BYTES + 1), mtime=0),
        gzip.compress(b"{}", mtime=0)[:-1],
        gzip.compress(b"{}", mtime=0) + gzip.compress(b"{}", mtime=0),
        gzip.compress(b"{}", mtime=0) + b"trailing bytes",
        b"not a gzip stream",
    ],
)
def test_transport_rejects_decompression_bombs_truncation_members_and_trailing_bytes(
    compressed,
) -> None:
    with pytest.raises(HandoffError):
        unpack_transport(b"gzb64:" + base64.b64encode(compressed))


@pytest.mark.parametrize(
    "encoded", [b"gzb64:!!!", b"gzb64:AA==\n", b"x" * (MAX_TRANSPORT_BYTES + 1)]
)
def test_transport_refuses_noncanonical_or_oversized_encoding(encoded) -> None:
    with pytest.raises(HandoffError):
        unpack_transport(encoded)


def test_environment_must_have_independent_approval_and_protected_branches() -> None:
    require_environment(Transport(), "stage9-linux-evidence", POLICY_COMMIT)


@pytest.mark.parametrize(
    "mutation",
    [
        lambda env: env.update(protection_rules=[]),
        lambda env: env["protection_rules"][0].update(reviewers=[]),
        lambda env: env["protection_rules"][0].update(prevent_self_review=False),
        lambda env: env.update(deployment_branch_policy={"protected_branches": False}),
        lambda env: env.update(name="arbitrary"),
    ],
)
def test_missing_environment_protection_fails_closed(mutation) -> None:
    transport = Transport()
    mutation(transport.environment)
    with pytest.raises(HandoffError):
        require_environment(transport, "stage9-linux-evidence", POLICY_COMMIT)


@pytest.mark.parametrize("branch", ["main", handoff.POLICY_BRANCH])
def test_environment_flag_cannot_replace_actual_branch_protection(branch) -> None:
    transport = Transport()
    transport.branches[branch]["protected"] = False
    with pytest.raises(HandoffError, match="Gałąź nie jest chroniona"):
        require_environment(transport, "stage9-linux-evidence", POLICY_COMMIT)
    assert not any(path.startswith("/environments/") for path in transport.paths)


@pytest.mark.parametrize("revision", [None, "main", "", "A" * 40, "a" * 39])
def test_policy_requires_an_immutable_commit_before_any_remote_read(revision) -> None:
    transport = Transport()
    with pytest.raises(HandoffError):
        require_environment(transport, "stage9-linux-evidence", revision)
    assert transport.paths == []


def test_policy_rotation_revokes_the_previously_pinned_commit() -> None:
    transport = Transport()
    transport.branches[handoff.POLICY_BRANCH]["commit"]["sha"] = "c" * 40
    with pytest.raises(HandoffError, match="utraciła aktualność"):
        require_environment(transport, "stage9-linux-evidence", POLICY_COMMIT)


@pytest.mark.parametrize(
    "mutation",
    [
        lambda p: p.update(enforce_admins={"enabled": False}),
        lambda p: p.update(allow_force_pushes={"enabled": True}),
        lambda p: p.update(allow_deletions={"enabled": True}),
        lambda p: p.update(required_conversation_resolution={"enabled": False}),
        lambda p: p.update(required_pull_request_reviews=None),
        lambda p: p["required_pull_request_reviews"].update(required_approving_review_count=1),
        lambda p: p["required_pull_request_reviews"].update(required_approving_review_count=True),
        lambda p: p["required_pull_request_reviews"].update(dismiss_stale_reviews=False),
        lambda p: p["required_pull_request_reviews"].update(require_last_push_approval=False),
        lambda p: p["required_pull_request_reviews"].update(require_code_owner_reviews=False),
        lambda p: p["required_pull_request_reviews"].update(
            bypass_pull_request_allowances={"users": ["owner"], "teams": [], "apps": []}
        ),
        lambda p: p["required_status_checks"].update(strict=False),
        lambda p: p["required_status_checks"].update(checks=[]),
        lambda p: p["required_status_checks"]["checks"][0].update(app_id=-1),
        lambda p: p["required_status_checks"]["checks"][0].update(app_id=True),
        lambda p: p["required_status_checks"]["checks"][0].update(app_id=999999),
    ],
)
def test_weakened_branch_policy_or_unattributed_ci_check_refuses(mutation) -> None:
    transport = Transport()
    mutation(transport.protection)
    with pytest.raises(HandoffError):
        require_environment(transport, "stage9-linux-evidence", POLICY_COMMIT)


@pytest.mark.parametrize(
    "owners",
    [
        b"",
        b"* @only-one\n",
        b"* @same @same\n",
        b"* @one @two\n/deployment/\n",
        b"/deployment/ @one @two\n",
        b"* @org/team @person\n",
        b"* user@example.invalid @person\n",
    ],
)
def test_policy_code_ownership_must_cover_the_complete_reviewed_tree(owners) -> None:
    transport = Transport()
    transport.owners = owners
    with pytest.raises(HandoffError, match="CODEOWNERS|właściciele"):
        require_environment(transport, "stage9-linux-evidence", POLICY_COMMIT)


def test_code_owner_without_actual_write_permission_cannot_protect_policy() -> None:
    transport = Transport()
    transport.permission["permission"] = "read"
    with pytest.raises(HandoffError, match="dostępu do zapisu"):
        require_environment(transport, "stage9-linux-evidence", POLICY_COMMIT)


def test_two_owner_names_cannot_represent_the_same_github_account() -> None:
    class AliasedOwners(Transport):
        def json(self, path):
            value = super().json(path)
            if path.startswith("/collaborators/"):
                value["user"]["id"] = 7
            return value

    with pytest.raises(HandoffError, match="dostępu do zapisu"):
        require_environment(AliasedOwners(), "stage9-linux-evidence", POLICY_COMMIT)


@pytest.mark.parametrize(
    "mutation",
    [
        lambda e: e.update(can_admins_bypass=True),
        lambda e: e.pop("can_admins_bypass"),
        lambda e: e["deployment_branch_policy"].update(custom_branch_policies=True),
        lambda e: e["protection_rules"][0]["reviewers"][0].update(type="Team"),
        lambda e: e["protection_rules"][0]["reviewers"][0]["reviewer"].update(id=True),
        lambda e: e["protection_rules"][0]["reviewers"][0]["reviewer"].update(login=""),
    ],
)
def test_admin_bypass_unknown_configuration_or_ambiguous_reviewer_refuses(mutation) -> None:
    transport = Transport()
    mutation(transport.environment)
    with pytest.raises(HandoffError):
        require_environment(transport, "stage9-linux-evidence", POLICY_COMMIT)


def test_unreadable_github_controls_are_not_treated_as_safe_defaults() -> None:
    class InaccessibleProtection(Transport):
        def json(self, path):
            if path.endswith("/protection"):
                raise HandoffError("GitHub API 403")
            return super().json(path)

    with pytest.raises(HandoffError, match="403"):
        require_environment(InaccessibleProtection(), "stage9-linux-evidence", POLICY_COMMIT)


def test_archive_storage_redirect_never_receives_github_credential(monkeypatch) -> None:
    calls = []

    def read(url, token, limit):
        calls.append((url, token, limit))
        if len(calls) == 1:
            return b"", "https://approved.blob.core.windows.net/signed-archive"
        return b"archive bytes", None

    monkeypatch.setattr(handoff, "_bounded_response", read)
    client = GitHubClient("test-credential", REPOSITORY)
    assert client.archive(45) == b"archive bytes"
    assert calls[0][1] == "test-credential"
    assert calls[1][1] is None


@pytest.mark.parametrize(
    "location",
    [
        "http://approved.blob.core.windows.net/archive",
        "https://attacker.invalid/archive",
        "https://approved.blob.core.windows.net.attacker.invalid/archive",
        urlunsplit(("https", "user:password@approved.blob.core.windows.net", "/archive", "", "")),
        "https://approved.blob.core.windows.net:8443/archive",
    ],
)
def test_archive_refuses_unapproved_redirect_before_following(monkeypatch, location: str) -> None:
    calls = []

    def read(url, token, limit):
        calls.append(url)
        return b"", location

    monkeypatch.setattr(handoff, "_bounded_response", read)
    with pytest.raises(HandoffError):
        GitHubClient("test-credential", REPOSITORY).archive(45)
    assert len(calls) == 1


def test_http_boundary_bounds_bytes_and_hides_native_errors(monkeypatch) -> None:
    calls = []

    class Opener:
        def open(self, request, timeout):
            calls.append((request, timeout))
            return io.BytesIO(b"bounded response")

    monkeypatch.setattr(handoff, "build_opener", lambda *_handlers: Opener())
    raw, redirect = handoff._bounded_response("https://api.github.com/approved", "credential", 30)
    assert raw == b"bounded response" and redirect is None
    assert calls[0][0].get_header("Authorization") == "Bearer credential"
    assert calls[0][1] == 30
    with pytest.raises(HandoffError):
        handoff._bounded_response("https://api.github.com/approved", "credential", 2)


@pytest.mark.parametrize(
    "error",
    [
        OSError("native sensitive detail"),
        TimeoutError("timeout"),
        HTTPError("https://api.github.com/approved", 403, "denied", {}, None),
    ],
)
def test_http_errors_refuse_without_echoing_sensitive_detail(monkeypatch, error) -> None:
    class Opener:
        def open(self, request, timeout):
            raise error

    monkeypatch.setattr(handoff, "build_opener", lambda *_handlers: Opener())
    with pytest.raises(HandoffError) as failure:
        handoff._bounded_response("https://api.github.com/approved", "credential", 30)
    assert "native sensitive detail" not in str(failure.value)
    assert "credential" not in str(failure.value)


def test_redirect_is_returned_for_explicit_validation_without_automatic_follow(monkeypatch) -> None:
    class Opener:
        def open(self, request, timeout):
            raise HTTPError(
                request.full_url,
                302,
                "redirect",
                {"Location": "https://approved.blob.core.windows.net/asset"},
                None,
            )

    monkeypatch.setattr(handoff, "build_opener", lambda *_handlers: Opener())
    assert handoff._bounded_response("https://api.github.com/approved", "credential", 30) == (
        b"",
        "https://approved.blob.core.windows.net/asset",
    )
    assert (
        handoff._NoRedirect().redirect_request(None, None, 302, "redirect", {}, "https://elsewhere")
        is None
    )


@pytest.mark.parametrize(
    "payload", [b"[]", b"{broken", b'{"duplicate":1,"duplicate":2}', b'{"invalid":Infinity}']
)
def test_github_metadata_must_be_an_unambiguous_json_object(monkeypatch, payload) -> None:
    monkeypatch.setattr(handoff, "_bounded_response", lambda *_args: (payload, None))
    with pytest.raises(HandoffError):
        GitHubClient("credential", REPOSITORY).json("/actions/artifacts")


def test_github_metadata_rejects_redirects_and_accepts_bounded_object(monkeypatch) -> None:
    monkeypatch.setattr(handoff, "_bounded_response", lambda *_args: (b'{"value":1}', None))
    assert GitHubClient("credential", REPOSITORY).json("/actions/artifacts") == {"value": 1}
    monkeypatch.setattr(handoff, "_bounded_response", lambda *_args: (b"", "https://elsewhere"))
    with pytest.raises(HandoffError):
        GitHubClient("credential", REPOSITORY).json("/actions/artifacts")


def test_transport_cli_pack_fetch_and_environment_keep_authority_separate(
    tmp_path, monkeypatch, capsys
) -> None:
    original, packed = tmp_path / "original.json", tmp_path / "transport.txt"
    original.write_bytes(b'{"untrusted":"bytes"}')
    assert handoff.main(["pack", "--bundle", str(original), "--output", str(packed)]) == 0
    received = tmp_path / "received.json"
    monkeypatch.setenv("LINUX_EVIDENCE_BUNDLE", packed.read_text())
    assert handoff.main(["ingress", "--output", str(received)]) == 0
    assert received.read_bytes() == original.read_bytes()
    monkeypatch.setattr(handoff, "GitHubClient", lambda *_args: Transport())
    assert (
        handoff.main(
            [
                "require-environment",
                "--repository",
                REPOSITORY,
                "--name",
                "stage9-linux-evidence",
                "--policy-commit",
                POLICY_COMMIT,
            ]
        )
        == 0
    )
    fetched = tmp_path / "fetched.json"
    assert (
        handoff.main(
            [
                "fetch",
                "--repository",
                REPOSITORY,
                "--target-run-id",
                "123",
                "--target-run-attempt",
                "2",
                "--output",
                str(fetched),
            ]
        )
        == 0
    )
    assert fetched.read_bytes() == original.read_bytes()
    assert capsys.readouterr().out == ""


def test_transport_cli_missing_bytes_or_api_identity_refuses(tmp_path, monkeypatch, capsys) -> None:
    monkeypatch.delenv("LINUX_EVIDENCE_BUNDLE", raising=False)
    assert handoff.main(["ingress", "--output", str(tmp_path / "bundle.json")]) == 1
    monkeypatch.delenv("GH_TOKEN", raising=False)
    assert (
        handoff.main(
            [
                "require-environment",
                "--repository",
                REPOSITORY,
                "--name",
                "stage9-linux-evidence",
                "--policy-commit",
                POLICY_COMMIT,
            ]
        )
        == 1
    )
    assert "ODMOWA_TRANSPORTU" in capsys.readouterr().out
    assert not (tmp_path / "bundle.json").exists()


def test_physical_jobs_are_separate_from_pr_and_stage10_and_pin_main() -> None:
    jobs = yaml.safe_load((ROOT / ".github/workflows/platform-deployment.yml").read_text())["jobs"]
    for name in ("linux-physical-challenge", "linux-physical-validation", "linux-production-gate"):
        job = jobs[name]
        condition = job["if"]
        assert "github.event_name == 'workflow_dispatch'" in condition
        assert "inputs.run_linux_physical == true" in condition
        assert "github.ref == 'refs/heads/main'" in condition
        assert "github.repository == 'Dudzian/Dudzian'" in condition
        assert job["runs-on"] == "ubuntu-24.04"
        assert job["permissions"] == {"contents": "read", "actions": "read"}
        assert "continue-on-error" not in job


def test_independent_verification_and_gate_receive_raw_bundle_and_trusted_context() -> None:
    jobs = yaml.safe_load((ROOT / ".github/workflows/platform-deployment.yml").read_text())["jobs"]
    challenge = jobs["linux-physical-challenge"]
    validation = jobs["linux-physical-validation"]
    gate = jobs["linux-production-gate"]
    assert challenge["environment"] == "stage9-linux-challenge"
    assert validation["environment"] == "stage9-linux-evidence"
    assert set(validation["needs"]) == {"linux-physical-challenge"}
    assert set(gate["needs"]) == {
        "linux-physical-validation",
        "core-linux",
        "core-windows",
        "core-macos",
    }
    verification_source = "\n".join(step.get("run", "") for step in validation["steps"])
    assert "linux_physical_evidence verify" in verification_source
    assert "--expected-policy-sha256" in verification_source
    assert '--ci-run-attempt "$GITHUB_RUN_ATTEMPT"' in verification_source
    assert "require-environment" in verification_source
    gate_source = "\n".join(step.get("run", "") for step in gate["steps"])
    for argument in (
        "--linux-bundle",
        "--linux-challenge",
        "--linux-policy",
        "--linux-policy-sha256",
        "--expected-ci-run-attempt",
        "--linux-replay-ledger",
        "--evidence evidence/core-matrix.json",
    ):
        assert argument in gate_source
    assert "aggregate-core" in gate_source
    assert all(
        not step.get("continue-on-error", False) for step in (*validation["steps"], *gate["steps"])
    )


def test_policy_bytes_come_from_a_separately_reviewed_immutable_commit() -> None:
    jobs = yaml.safe_load((ROOT / ".github/workflows/platform-deployment.yml").read_text())["jobs"]
    for name in ("linux-physical-challenge", "linux-physical-validation", "linux-production-gate"):
        job = jobs[name]
        source = "\n".join(step.get("run", "") for step in job["steps"])
        assert "STAGE9_LINUX_EVIDENCE_POLICY_JSON" not in source
        assert "STAGE9_LINUX_EVIDENCE_POLICY_JSON" not in job["env"]
        assert (
            job["env"]["LINUX_POLICY_COMMIT"] == "${{ vars.STAGE9_LINUX_EVIDENCE_POLICY_COMMIT }}"
        )
        controls = next(
            step for step in job["steps"] if "require-environment" in step.get("run", "")
        )
        assert controls["env"]["GH_TOKEN"] == "${{ secrets.STAGE9_GITHUB_CONTROLS_READ_TOKEN }}"
        assert '--policy-commit "$LINUX_POLICY_COMMIT"' in controls["run"]
        policy = next(
            step
            for step in job["steps"]
            if step.get("with", {}).get("path") == "stage9-approved-policy"
        )
        assert policy["with"]["ref"] == "${{ vars.STAGE9_LINUX_EVIDENCE_POLICY_COMMIT }}"
        assert policy["with"]["persist-credentials"] is False
        assert job["steps"].index(controls) < job["steps"].index(policy)
        assert "stage9-approved-policy/policy.json" in source


def test_ingress_workflow_never_interpolates_untrusted_json_into_shell() -> None:
    workflow = yaml.safe_load((ROOT / ".github/workflows/linux-evidence-ingress.yml").read_text())
    job = workflow["jobs"]["save"]
    assert "github.ref == 'refs/heads/main'" in job["if"]
    assert "github.repository == 'Dudzian/Dudzian'" in job["if"]
    assert job["permissions"] == {"contents": "read"}
    incoming = next(step for step in job["steps"] if "ingress --output" in step.get("run", ""))
    assert incoming["env"]["LINUX_EVIDENCE_BUNDLE"] == "${{ inputs.bundle }}"
    assert "${{ inputs.bundle }}" not in incoming["run"]
    assert "reboot" not in incoming["run"]
    upload = next(
        step for step in job["steps"] if step.get("uses", "").startswith("actions/upload-artifact@")
    )
    assert upload["with"]["path"] == "bundle.json"
    assert upload["with"]["overwrite"] is False
    assert upload["with"]["if-no-files-found"] == "error"


def test_new_security_modules_remain_in_type_format_test_and_coverage_gates() -> None:
    modules = {
        "linux_physical_evidence",
        "linux_physical_observations",
        "linux_physical_native",
        "linux_physical_collector",
        "linux_evidence_handoff",
    }
    configuration = tomllib.loads((ROOT / "pyproject.toml").read_text())
    assert {f"deployment/{name}.py" for name in modules} <= set(
        configuration["tool"]["mypy"]["files"]
    )
    job = yaml.safe_load((ROOT / ".github/workflows/quality-security.yml").read_text())["jobs"][
        "quality-ratchet"
    ]
    format_source = "\n".join(
        step.get("run", "") for step in job["steps"] if "ruff format --check" in step.get("run", "")
    )
    lint_source = "\n".join(
        step.get("run", "") for step in job["steps"] if "ruff check" in step.get("run", "")
    )
    coverage_source = next(
        step["run"]
        for step in job["steps"]
        if step.get("name") == "Property and critical pure-logic tests with branch coverage"
    )
    for name in modules:
        assert f"deployment/{name}.py" in format_source
        assert f"deployment/{name}.py" in lint_source
        assert f"--cov=deployment.{name}" in coverage_source
    for path in (
        "tests/deployment/test_linux_physical_evidence.py",
        "tests/deployment/test_linux_physical_collector.py",
        "tests/deployment/test_linux_physical_native.py",
        "tests/scripts/test_linux_physical_evidence_workflow.py",
    ):
        assert path in format_source
        assert path in coverage_source
    assert job["timeout-minutes"] == 120
    coverage_gate = yaml.safe_load((ROOT / ".github/workflows/quality-security.yml").read_text())[
        "jobs"
    ]["quality-ratchet-coverage"]
    assert any(
        "diff-cover coverage.xml" in step.get("run", "") and "--fail-under=70" in step["run"]
        for step in coverage_gate["steps"]
    )
