"""Hermetic handoff provenance tests; TEST_ONLY fixtures never authorize installation."""

from __future__ import annotations

import json
import subprocess
from copy import deepcopy
from types import SimpleNamespace

import pytest

from deployment import (
    windows_stage9_handoff as handoff,
    windows_stage10_artifact_preflight as preflight,
)
from tests.deployment import test_windows_stage10_artifact_preflight as artifact_fixtures
from tests.deployment.test_windows_stage10_artifact_preflight import (
    REVISION,
    cli_args,
    rebuild,
)

staged = artifact_fixtures.staged


@pytest.fixture
def authenticated(staged):
    staged.args["expected_run_attempt"] = 1
    staged.run.update(status="in_progress", conclusion=None)
    rebuild(staged)
    jobs = []
    for identifier, name in enumerate(sorted(preflight.PRODUCER_JOBS), 100):
        jobs.append(
            {
                "id": identifier,
                "name": name,
                "run_id": 42,
                "run_attempt": 1,
                "run_url": "https://api.github.com/repos/Dudzian/Dudzian/actions/runs/42",
                "head_sha": REVISION,
                "head_branch": "main",
                "status": "completed",
                "conclusion": "success",
                "started_at": "2026-10-16T08:00:00Z",
                "completed_at": "2026-10-16T09:01:00Z",
            }
        )
    state = SimpleNamespace(
        staged=staged,
        artifact=deepcopy(staged.artifact),
        run=deepcopy(staged.run),
        jobs=jobs,
        calls=[],
        pages=None,
    )

    def github_get(endpoint):
        state.calls.append(endpoint)
        if endpoint == "repos/Dudzian/Dudzian/actions/artifacts/9":
            return deepcopy(state.artifact)
        if endpoint == "repos/Dudzian/Dudzian/actions/runs/42":
            return deepcopy(state.run)
        prefix = "repos/Dudzian/Dudzian/actions/runs/42/attempts/1/jobs?per_page=100&page="
        assert endpoint.startswith(prefix), endpoint
        page = int(endpoint.removeprefix(prefix))
        if state.pages is not None:
            return deepcopy(state.pages[page - 1])
        return {"total_count": len(state.jobs), "jobs": deepcopy(state.jobs)}

    state.get = github_get
    return state


def run(state):
    return handoff.qualified_handoff(**state.staged.args, github_get=state.get)


def test_active_same_attempt_producer_success_reuses_shared_validator(authenticated):
    before = {
        path: path.read_bytes() for path in authenticated.staged.root.rglob("*") if path.is_file()
    }
    report = run(authenticated)
    assert report["status"] == "PASS", report
    assert report["details"]["repository"] == preflight.REPOSITORY
    assert report["details"]["ci_run_attempt"] == 1
    assert report["details"]["product_version"] == "0.1.0"
    assert report["details"]["receipt_sha256"] == artifact_fixtures.digest(
        artifact_fixtures.encode(authenticated.staged.receipt)
    )
    assert report["details"]["evidence_sha256"] == artifact_fixtures.digest(
        artifact_fixtures.encode(authenticated.staged.evidence)
    )
    assert report["details"]["installed_files"] == authenticated.staged.manifest["installed_files"]
    assert report["details"]["authenticated_provenance"]["source"] == "authenticated-github-rest"
    assert report["stage10_qualification"] == "NOT_RUN"
    assert report["stage10_pass"] is False
    assert report["installation_authorized"] is False
    assert report["installation_executed"] is False
    assert len(authenticated.calls) == 6  # Initial and final artifact/run/jobs reads.
    assert before == {
        path: path.read_bytes() for path in authenticated.staged.root.rglob("*") if path.is_file()
    }


def test_completed_success_still_supported_with_independent_attempt_provenance(authenticated):
    authenticated.run.update(status="completed", conclusion="success")
    assert run(authenticated)["status"] == "PASS"


def test_offline_metadata_never_qualifies_active_workflow(authenticated):
    report = preflight.artifact_preflight(**authenticated.staged.args)
    assert report["status"] == "FAIL"
    assert "operator-dispatched main workflow" in report["reason"]


@pytest.mark.parametrize("value", [None, 0, True, "", "01", "2"])
def test_independent_attempt_is_mandatory_and_exact(authenticated, value):
    authenticated.staged.args["expected_run_attempt"] = value
    assert run(authenticated)["status"] == "FAIL"


def test_staged_boolean_attempt_cannot_impersonate_integer(authenticated):
    authenticated.staged.run["run_attempt"] = True
    rebuild(authenticated.staged)
    assert run(authenticated)["status"] == "FAIL"


@pytest.mark.parametrize(
    "status,conclusion",
    [
        ("queued", None),
        ("completed", "failure"),
        ("completed", "cancelled"),
        ("in_progress", "success"),
    ],
)
def test_failed_cancelled_or_ambiguous_run_cannot_handoff(authenticated, status, conclusion):
    authenticated.run.update(status=status, conclusion=conclusion)
    assert run(authenticated)["status"] == "FAIL"


@pytest.mark.parametrize(
    "status,conclusion",
    [
        ("in_progress", None),
        ("queued", None),
        ("completed", "failure"),
        ("completed", "skipped"),
        ("completed", "cancelled"),
    ],
)
@pytest.mark.parametrize("producer", sorted(preflight.PRODUCER_JOBS))
def test_every_producer_must_succeed_not_just_artifact_exist(
    authenticated, producer, status, conclusion
):
    job = next(job for job in authenticated.jobs if job["name"] == producer)
    job.update(status=status, conclusion=conclusion)
    report = run(authenticated)
    assert report["status"] == "FAIL", report
    assert "every canonical Stage-9 producer" in report["reason"]


@pytest.mark.parametrize(
    "field,value",
    [
        ("run_id", 43),
        ("run_attempt", 2),
        ("head_sha", "b" * 40),
        ("head_branch", "feature"),
        ("run_url", "https://api.github.com/repos/evil/Dudzian/actions/runs/42"),
    ],
)
def test_mixed_job_run_attempt_revision_branch_or_repository_fails(authenticated, field, value):
    authenticated.jobs[0][field] = value
    assert run(authenticated)["status"] == "FAIL"


@pytest.mark.parametrize(
    "mutation",
    [
        lambda s: s.run.update(event="pull_request"),
        lambda s: s.run.update(head_branch="feature"),
        lambda s: s.run.update(head_sha="b" * 40),
        lambda s: s.run.update(path=preflight.WORKFLOW + "@refs/pull/1/merge"),
        lambda s: s.run.update(pull_requests=[{"number": 1}]),
        lambda s: s.run["head_repository"].update(id=456, full_name="evil/Dudzian", fork=True),
        lambda s: s.artifact["workflow_run"].update(head_repository_id=456),
        lambda s: s.artifact["workflow_run"].update(id=43),
        lambda s: s.artifact.update(expired=True),
    ],
)
def test_remote_identity_rejects_pr_foreign_or_mixed_provenance(authenticated, mutation):
    mutation(authenticated)
    # Copy the tampered metadata too: offline agreement is never sufficient.
    authenticated.staged.run = deepcopy(authenticated.run)
    authenticated.staged.artifact = deepcopy(authenticated.artifact)
    rebuild(authenticated.staged, rebind=False)
    assert run(authenticated)["status"] == "FAIL"


@pytest.mark.parametrize(
    "timestamp", ["2026-10-16T07:59:59Z", "2026-10-16T09:02:00Z", None, "badZ"]
)
def test_artifact_from_previous_attempt_cannot_be_reused(authenticated, timestamp):
    authenticated.artifact["created_at"] = timestamp
    authenticated.staged.artifact = deepcopy(authenticated.artifact)
    rebuild(authenticated.staged, rebind=False)
    assert run(authenticated)["status"] == "FAIL"


def test_jobs_pages_are_all_read_and_bound_to_same_attempt(authenticated):
    extra = deepcopy(authenticated.jobs[0])
    extra.update(id=999, name="unrelated-current-run-job", status="in_progress", conclusion=None)
    authenticated.pages = [
        {"total_count": 4, "jobs": authenticated.jobs[:1]},
        {"total_count": 4, "jobs": authenticated.jobs[1:] + [extra]},
    ]
    assert run(authenticated)["status"] == "PASS"
    assert sum("page=2" in endpoint for endpoint in authenticated.calls) == 2


@pytest.mark.parametrize(
    "pages",
    [
        [{"total_count": 4, "jobs": []}],
        [{"total_count": 2, "jobs": []}],
        [{"total_count": True, "jobs": []}],
        [{"total_count": 10001, "jobs": []}],
        [{"total_count": 3, "jobs": [{"id": 1}]}, {"total_count": 4, "jobs": [{"id": 2}]}],
    ],
)
def test_incomplete_or_inconsistent_jobs_pagination_fails(authenticated, pages):
    authenticated.pages = pages
    assert run(authenticated)["status"] == "FAIL"


def test_duplicate_or_missing_producer_is_rejected(authenticated):
    authenticated.jobs[0] = deepcopy(authenticated.jobs[1])
    assert run(authenticated)["status"] == "FAIL"


def test_local_metadata_cannot_replace_authenticated_artifact_identity(authenticated):
    authenticated.artifact["digest"] = "sha256:" + "0" * 64
    assert run(authenticated)["status"] == "FAIL"


def test_github_access_unavailable_is_blocked_and_does_not_trust_cached_success(authenticated):
    def denied(endpoint):
        raise PermissionError("TEST_ONLY_ACCESS_DENIED")

    report = handoff.qualified_handoff(**authenticated.staged.args, github_get=denied)
    assert report["status"] == "BLOCKED"
    assert report["stage10_pass"] is False
    assert "details" not in report


def test_missing_staged_archive_blocks_handoff(authenticated):
    authenticated.staged.args["archive"].unlink()
    assert run(authenticated)["status"] == "BLOCKED"


def test_substituted_msi_fails_even_after_zip_digest_is_rebound(authenticated):
    authenticated.staged.members[next(iter(authenticated.staged.members))] = (
        b"TEST_ONLY_SUBSTITUTED"
    )
    rebuild(authenticated.staged)
    authenticated.artifact = deepcopy(authenticated.staged.artifact)
    assert run(authenticated)["status"] == "FAIL"


def test_saved_pass_does_not_allow_tampered_archive_at_next_use(authenticated):
    assert run(authenticated)["status"] == "PASS"
    with authenticated.staged.args["archive"].open("ab") as stream:
        stream.write(b"TEST_ONLY_TAMPERING")
    assert run(authenticated)["status"] == "FAIL"


def test_trust_mutation_during_validation_fails(authenticated, monkeypatch):
    original = preflight.load_production_trust

    def mutated(path):
        result = original(path)
        (path / "package_manifest.json").write_text('{"TEST_ONLY":"changed"}')
        return result

    monkeypatch.setattr(preflight, "load_production_trust", mutated)
    assert run(authenticated)["status"] == "FAIL"


def test_attempt_change_during_validation_is_rejected(authenticated):
    original = authenticated.get
    calls = 0

    def raced(endpoint):
        nonlocal calls
        if endpoint.endswith("/runs/42"):
            calls += 1
            if calls == 2:
                authenticated.run["run_attempt"] = 2
        return original(endpoint)

    authenticated.get = raced
    assert run(authenticated)["status"] == "FAIL"


def test_gh_transport_uses_only_authenticated_get_and_explicit_timeout(monkeypatch):
    def command(argv, **kwargs):
        assert argv == [
            "gh",
            "api",
            "--method",
            "GET",
            "--hostname",
            "github.com",
            "repos/Dudzian/Dudzian",
        ]
        assert kwargs == {"check": True, "capture_output": True, "text": True, "timeout": 45}
        return SimpleNamespace(stdout='{"id":123}')

    monkeypatch.setattr(subprocess, "run", command)
    assert handoff._github_get("repos/Dudzian/Dudzian") == {"id": 123}


@pytest.mark.parametrize(
    "error",
    [
        FileNotFoundError("gh missing"),
        subprocess.TimeoutExpired("gh", 45),
        subprocess.CalledProcessError(403, "gh", stderr="TEST_ONLY_TOKEN_MUST_NOT_LEAK"),
    ],
)
def test_gh_transport_failure_has_no_process_diagnostics(monkeypatch, error):
    def command(*args, **kwargs):
        raise error

    monkeypatch.setattr(subprocess, "run", command)
    with pytest.raises(preflight.ArtifactProvenanceUnavailable) as caught:
        handoff._github_get("repos/Dudzian/Dudzian")
    assert "TEST_ONLY_TOKEN_MUST_NOT_LEAK" not in str(caught.value)


def test_handoff_cli_writes_report_and_refuses_overwrite(authenticated, monkeypatch):
    monkeypatch.setattr(handoff, "_github_get", authenticated.get)
    output = authenticated.staged.root / "handoff-report.json"
    args = cli_args(authenticated.staged, output)
    assert handoff.main(args) == 0
    assert json.loads(output.read_text())["installation_executed"] is False
    assert handoff.main(args) == 1


def test_handoff_cli_cannot_modify_trust_package(authenticated):
    output = authenticated.staged.args["production_trust_package"] / "report.json"
    assert handoff.main(cli_args(authenticated.staged, output)) == 1
    assert not output.exists()


def test_handoff_cli_outside_staging_is_rejected(authenticated):
    output = authenticated.staged.root.parent / "unauthorized.json"
    assert handoff.main(cli_args(authenticated.staged, output)) == 1
    assert not output.exists()
