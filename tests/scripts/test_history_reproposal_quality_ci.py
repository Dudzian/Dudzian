"""Fail-closed contracts for the independent history reproposal qualification."""

from __future__ import annotations

import ast
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import tomllib
import xml.etree.ElementTree as ET

from coverage import CoverageData
import pytest
import yaml

from tests.scripts.test_general_ci_platform_contract import (
    _contract_bash,
    _pytest_command as _general_pytest_command,
)


ROOT = Path(__file__).resolve().parents[2]
WORKFLOW = ROOT / ".github/workflows/quality-security.yml"
HISTORY_TESTS = {
    "tests/security/test_postgresql_history_successor_reproposal.py",
    "tests/security/test_history_successor_reproposal_evidence.py",
    "tests/security/test_history_successor_reproposal_legacy.py",
    "tests/architecture/test_stage9_history_successor_reproposal.py",
}
HISTORY_MODULES = {
    "bot_core.postgresql_history_successor_reproposal",
    "bot_core.postgresql_history_successor_reproposal_schema",
    "bot_core.postgresql_history_successor_reproposal_qualification",
    "bot_core.postgresql_history_successor_reproposal_evidence",
}
ORIGINAL_ATOMIC_TESTS = {
    "tests/security/test_postgresql_atomic_root_proof_commit.py",
    "tests/security/test_postgresql_root_proof_preparation_v2.py",
    "tests/security/test_postgresql_root_proof_issuance_authority_v2.py",
    "tests/architecture/test_stage9_postgresql_v2_root_proof_preparation.py",
    "tests/security/test_atomic_root_proof_commit_contract.py",
    "tests/architecture/test_stage9_atomic_root_proof_commit_boundary.py",
    "tests/security/test_local_signing_lifecycle_evidence.py",
    "tests/security/test_postgresql_custody_lifecycle.py",
    "tests/security/test_postgresql_custody_signer_enrollment.py",
    "tests/security/test_custody_lifecycle_admin_contract.py",
    "tests/security/test_local_custody_admission_contract.py",
    "tests/security/test_legacy_local_custody_isolation.py",
    "tests/architecture/test_stage9_custody_lifecycle_v2.py",
    "tests/architecture/test_stage9_custody_lifecycle_import_isolation.py",
}
ORIGINAL_ATOMIC_MODULES = {
    "bot_core.postgresql_atomic_root_proof_schema",
    "bot_core.postgresql_atomic_root_proof_preaccount_sql",
    "bot_core.postgresql_atomic_root_proof_qualification",
    "bot_core.postgresql_atomic_root_proof_migration",
    "bot_core.postgresql_atomic_root_proof_commit",
    "bot_core.postgresql_root_proof_issuance_authority_v2",
    "bot_core.postgresql_root_proof_preparation_qualification",
    "bot_core.postgresql_root_proof_preparation_readonly",
    "bot_core.postgresql_root_proof_preparation_v2",
    "bot_core.postgresql_authenticated_issuer_history",
    "bot_core.root_proof_atomic_attempt_consistency",
    "bot_core.root_proof_issuer_runtime",
    "bot_core.licensing.cha_root_proof_attempt_reservation",
    "bot_core.licensing.cha_root_proof_signed_attempt",
    "bot_core.local_signing_custody",
    "bot_core.local_signing_lifecycle_evidence",
    "bot_core.postgresql_custody_lifecycle",
    "bot_core.postgresql_custody_lifecycle_schema",
    "bot_core.postgresql_custody_lifecycle_qualification",
}
# Stage 9 production lifecycle and signer qualification extended the atomic job
# after the history split. Keep the original inventory intact and require the
# additional tests/coverage explicitly, so neither group can silently disappear.
STAGE9_PRODUCTION_ATOMIC_TESTS = {
    "tests/security/test_independent_custody_completion_qualification.py",
    "tests/security/test_authenticated_custody_mutation_source.py",
    "tests/security/test_authenticated_custody_source_processes.py",
    "tests/security/test_independent_cca_decision.py",
    "tests/security/test_cca_durable_decision_log.py",
    "tests/architecture/test_stage9_independent_cca_boundary.py",
    "tests/security/test_postgresql_atomic_root_proof_transaction.py",
    "tests/architecture/test_stage9_root_proof_transaction_scope_addendum_v1.py",
    "tests/architecture/test_stage9_independent_committed_close_reader_contract.py",
    "tests/security/test_production_root_lifecycle_schema.py",
    "tests/security/test_postgresql_production_root_lifecycle_qualification.py",
    "tests/security/test_postgresql_independent_cca_reader_smoke.py",
    "tests/security/test_production_root_lifecycle_identity.py",
    "tests/security/test_production_root_lifecycle_races.py",
    "tests/security/test_production_signer_ipc.py",
    "tests/security/test_production_signer_custody.py",
    "tests/security/test_production_signer_deployment.py",
    "tests/security/test_production_signer_lifecycle.py",
    "tests/security/test_production_signer_linux_processes.py",
    "tests/security/test_production_signer_boundary.py",
}
STAGE9_PRODUCTION_ATOMIC_MODULES = {
    "bot_core.independent_custody_completion_qualification",
    "bot_core.authenticated_custody_mutation_source",
    "bot_core.independent_cca_decision",
    "bot_core.test_only_cca_decision_log",
    "bot_core.postgresql_atomic_root_proof_transaction",
    "bot_core.postgresql_production_root_lifecycle_schema",
    "bot_core.postgresql_production_root_lifecycle_qualification",
    "bot_core.production_root_lifecycle",
    "deployment.production_root_proof_issuer",
    "bot_core.production_signer_ipc",
    "bot_core.production_signer_custody",
    "deployment.production_signer_service",
}
RAW_FILES = (
    "coverage-atomic-root-proof.data",
    "coverage-history-successor-reproposal.data",
)


def _jobs() -> dict:
    return yaml.safe_load(WORKFLOW.read_text(encoding="utf-8"))["jobs"]


def _step(job: dict, name: str) -> dict:
    matches = [step for step in job["steps"] if step.get("name") == name]
    assert len(matches) == 1
    return matches[0]


def _pytest_command(job: dict) -> str:
    matches = [step["run"] for step in job["steps"] if "python -m pytest " in step.get("run", "")]
    assert len(matches) == 1
    return matches[0]


def test_history_qualification_partitions_tests_and_preserves_original_atomic_scope():
    jobs = _jobs()
    original = jobs["atomic-root-proof-commit"]
    history = jobs["history-successor-reproposal"]
    assert original["timeout-minutes"] == 120
    assert history["timeout-minutes"] == 120
    for job, expected_tests, expected_modules in (
        (
            original,
            ORIGINAL_ATOMIC_TESTS | STAGE9_PRODUCTION_ATOMIC_TESTS,
            ORIGINAL_ATOMIC_MODULES | STAGE9_PRODUCTION_ATOMIC_MODULES,
        ),
        (
            history,
            HISTORY_TESTS,
            HISTORY_MODULES | ORIGINAL_ATOMIC_MODULES | STAGE9_PRODUCTION_ATOMIC_MODULES,
        ),
    ):
        command = _pytest_command(job)
        tokens = shlex.split(command)
        selected = [token for token in tokens if token.startswith("tests/")]
        assert set(selected) == expected_tests
        assert len(selected) == len(expected_tests)
        coverage_modules = [
            token.removeprefix("--cov=") for token in tokens if token.startswith("--cov=")
        ]
        assert set(coverage_modules) == expected_modules
        assert len(coverage_modules) == len(expected_modules)
        assert "--cov-branch" in tokens
        assert "--cov-config=.coveragerc" in tokens
        assert "-n" not in tokens and "--numprocesses" not in command
        assert "continue-on-error" not in job
    history_command = _pytest_command(history)
    history_tokens = shlex.split(history_command)[3:]
    for deselection in ("--fast", "--ignore", "--deselect", "-k", "-m"):
        assert not any(token.startswith(deselection) for token in history_tokens)
    assert "-o xfail_strict=true" in history_command
    assert "_history_reproposal_negative_cases.py" not in history_command
    assert "history-successor-reproposal" not in original.get("needs", [])
    assert "Changed-code coverage ratchet" not in {step.get("name") for step in original["steps"]}


def test_independent_cca_qualification_has_configured_types_and_explicit_format_scope():
    quality = _jobs()["quality-ratchet"]
    assert _step(quality, "Mypy configured scope")["run"] == "python -m mypy"
    assert (
        _step(quality, "Mypy Windows configured scope")["run"] == "python -m mypy --platform win32"
    )
    config = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    typed_files = config["tool"]["mypy"]["files"]
    modules = {
        "bot_core/independent_cca_decision.py",
        "bot_core/test_only_cca_decision_log.py",
    }
    for path in modules:
        assert typed_files.count(path) == 1
    format_tokens = shlex.split(
        _step(quality, "Format Stage 9 signing and credential substrate")["run"]
    )
    assert format_tokens[:5] == ["python", "-m", "ruff", "format", "--check"]
    expected = modules | {
        "tests/security/_cca_decision_fixtures.py",
        "tests/security/_cca_log_process.py",
        "tests/security/test_independent_cca_decision.py",
        "tests/security/test_cca_durable_decision_log.py",
        "tests/architecture/test_stage9_independent_cca_boundary.py",
    }
    for path in expected:
        assert format_tokens.count(path) == 1


def test_root_proof_transaction_scope_addendum_has_explicit_format_scope():
    relative_path = "tests/architecture/test_stage9_root_proof_transaction_scope_addendum_v1.py"
    quality = _jobs()["quality-ratchet"]
    format_tokens = shlex.split(
        _step(quality, "Format Stage 9 signing and credential substrate")["run"]
    )
    assert format_tokens[:5] == ["python", "-m", "ruff", "format", "--check"]
    assert format_tokens.count(relative_path) == 1


def test_independent_close_reader_contract_has_explicit_format_scope():
    relative_path = "tests/architecture/test_stage9_independent_committed_close_reader_contract.py"
    quality = _jobs()["quality-ratchet"]
    format_tokens = shlex.split(
        _step(quality, "Format Stage 9 signing and credential substrate")["run"]
    )
    assert format_tokens[:5] == ["python", "-m", "ruff", "format", "--check"]
    assert format_tokens.count(relative_path) == 1


def test_independent_cca_postgresql_smoke_has_explicit_format_scope():
    relative_path = "tests/security/test_postgresql_independent_cca_reader_smoke.py"
    quality = _jobs()["quality-ratchet"]
    format_tokens = shlex.split(
        _step(quality, "Format Stage 9 signing and credential substrate")["run"]
    )
    assert format_tokens[:5] == ["python", "-m", "ruff", "format", "--check"]
    assert format_tokens.count(relative_path) == 1


def test_linux_cca_log_routes_to_integration_and_atomic_without_non_linux_fast_execution():
    relative_path = "tests/security/test_cca_durable_decision_log.py"
    tree = ast.parse((ROOT / relative_path).read_text(encoding="utf-8"))
    marks = [
        node.value
        for node in tree.body
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id == "pytestmark" for target in node.targets
        )
    ]
    assert len(marks) == 1
    assert ast.unparse(marks[0]) == "pytest.mark.integration"
    assert not any(
        isinstance(node, ast.Attribute) and node.attr in {"skip", "skipif", "xfail"}
        for node in ast.walk(tree)
    )
    general = yaml.safe_load((ROOT / ".github/workflows/ci.yml").read_text(encoding="utf-8"))[
        "jobs"
    ]
    integration = general["integration-linux"]
    assert integration["runs-on"] == "ubuntu-latest"
    assert integration["timeout-minutes"] == 60
    integration_tokens = shlex.split(_step(integration, "Pytest integration")["run"])
    assert integration_tokens[integration_tokens.index("-m") + 1] == "integration"
    assert not any(
        token.startswith(("--ignore", "--deselect")) or token in {"-k", "--fast"}
        for token in integration_tokens
    )
    for job_name in ("py-tests-windows", "py-tests-macos"):
        command = _general_pytest_command(general[job_name])
        assert (
            '-m "not integration and not external_postgresql and not e2e_demo_paper '
            'and not retraining and not soak and not performance and not qml"'
        ) in command
    atomic = _pytest_command(_jobs()["atomic-root-proof-commit"])
    assert shlex.split(atomic).count(relative_path) == 1
    assert '-m "not hardware and not destructive and not network"' in atomic


def test_history_qualification_has_independent_durable_pg16_and_real_restart_controls():
    history = _jobs()["history-successor-reproposal"]
    assert history["runs-on"] == "ubuntu-latest"
    assert set(history["services"]) == {"postgres", "postgres-other"}
    for service, port in (("postgres", "55432:5432"), ("postgres-other", "55434:5432")):
        assert history["services"][service]["image"] == "postgres:16"
        assert port in [str(mapping) for mapping in history["services"][service]["ports"]]
    verification = _step(history, "Verify independent durable PostgreSQL 16 clusters")["run"]
    for setting in ("server_version_num", "fsync", "synchronous_commit", "full_page_writes"):
        assert f"current_setting('{setting}')" in verification
    assert "16|on|on|on" in verification
    assert 'test "$primary_id" != "$other_id"' in verification
    assert "pg_control_system()" in verification
    test_step = next(
        step for step in history["steps"] if "python -m pytest " in step.get("run", "")
    )
    env = test_step["env"]
    assert env["PYTEST_DISABLE_PLUGIN_AUTOLOAD"] == "1"
    assert "127.0.0.1 port=55432" in env["DUDZIAN_TEST_POSTGRES_DSN"]
    assert "127.0.0.1 port=55434" in env["CRYPTOHUNTER_TEST_POSTGRES_OTHER_DSN"]
    assert "DUDZIAN_TEST_POSTGRES_RESTART_CONTAINER_ID" not in env
    assert json.loads(env["DUDZIAN_TEST_POSTGRES_RESTART_COMMAND"]) == [
        "docker",
        "restart",
        "${{ job.services.postgres.id }}",
    ]
    assert json.loads(env["DUDZIAN_TEST_POSTGRES_CRASH_COMMAND"]) == [
        "docker",
        "exec",
        "--user",
        "postgres",
        "${{ job.services.postgres.id }}",
        "sh",
        "-c",
        'kill -KILL "$1"',
        "sh",
    ]


def test_combined_gate_requires_both_successful_jobs_and_mandatory_raw_artifacts():
    jobs = _jobs()
    combined = jobs["atomic-root-proof-coverage"]
    assert combined["timeout-minutes"] == 25
    assert combined["needs"] == ["atomic-root-proof-commit", "history-successor-reproposal"]
    assert "if" not in combined and "continue-on-error" not in combined
    downloads = [step for step in combined["steps"] if "download-artifact@" in step.get("uses", "")]
    assert {step["with"]["name"] for step in downloads} == {
        "atomic-root-proof-commit-data",
        "history-successor-reproposal-data",
    }
    assert all(step["with"]["path"] == "atomic-coverage-parts" for step in downloads)
    for job_name, artifact_name in (
        ("atomic-root-proof-commit", "atomic-root-proof-commit-data"),
        ("history-successor-reproposal", "history-successor-reproposal-data"),
    ):
        upload = next(
            step
            for step in jobs[job_name]["steps"]
            if step.get("with", {}).get("name") == artifact_name
        )
        assert upload["with"]["if-no-files-found"] == "error"
        assert "if" not in upload and "continue-on-error" not in upload
    ratchet = _step(combined, "Changed-code coverage ratchet")
    assert ratchet["if"] == "github.event_name == 'pull_request'"
    assert "diff-cover atomic-root-proof-combined-coverage.xml" in ratchet["run"]
    assert '--compare-branch "origin/$BASE_REF" --fail-under=70' in ratchet["run"]
    # The preexisting baseline/PostgreSQL raw-data ratchet remains independent.
    baseline = jobs["quality-ratchet-coverage"]
    assert baseline["needs"] == ["quality-ratchet", "quality-ratchet-postgresql"]
    assert "--fail-under=70" in _step(baseline, "Changed-code coverage ratchet")["run"]
    assert (
        "python -m coverage combine --keep coverage-parts"
        in _step(baseline, "Merge original branch coverage without losing either shard")["run"]
    )


def _run_step(command: str, tmp_path: Path) -> subprocess.CompletedProcess:
    env = dict(os.environ)
    env["PATH"] = os.pathsep.join((str(Path(sys.executable).parent), env["PATH"]))
    env["COVERAGE_FILE"] = str(tmp_path / ".coverage")
    return subprocess.run(
        [_contract_bash(), "-c", command], cwd=tmp_path, env=env, capture_output=True, text=True
    )


def _seed_real_coverage(tmp_path: Path, *, line_only: str | None = None) -> None:
    source = tmp_path / "branch_sample.py"
    source.write_text(
        "import sys\nif sys.argv[1] == 'first':\n    answer = 1\nelse:\n    answer = 2\n",
        encoding="utf-8",
    )
    parts = tmp_path / "atomic-coverage-parts"
    parts.mkdir()
    for filename, branch in zip(RAW_FILES, ("first", "second"), strict=True):
        subprocess.run(
            [
                sys.executable,
                "-m",
                "coverage",
                "run",
                *([] if line_only == filename else ["--branch"]),
                f"--data-file={parts / filename}",
                str(source),
                branch,
            ],
            cwd=tmp_path,
            check=True,
            capture_output=True,
            text=True,
        )


def _merge_command() -> str:
    return _step(
        _jobs()["atomic-root-proof-coverage"],
        "Merge atomic and history branch coverage without losing either qualification",
    )["run"]


@pytest.mark.parametrize("filename", RAW_FILES)
@pytest.mark.parametrize("damage", ("missing", "empty", "corrupt", "line-only"))
def test_aggregate_fails_closed_when_either_required_branch_database_is_invalid(
    tmp_path, filename, damage
):
    _seed_real_coverage(tmp_path, line_only=filename if damage == "line-only" else None)
    path = tmp_path / "atomic-coverage-parts" / filename
    if damage == "missing":
        path.unlink()
    elif damage == "empty":
        path.write_bytes(b"")
    elif damage == "corrupt":
        path.write_bytes(b"not a coverage database")
    outcome = _run_step(_merge_command(), tmp_path)
    assert outcome.returncode != 0, outcome.stdout + outcome.stderr
    assert not (tmp_path / "atomic-root-proof-combined-coverage.xml").exists()


def test_aggregate_combines_real_branch_observations_from_both_required_jobs(tmp_path):
    _seed_real_coverage(tmp_path)
    outcome = _run_step(_merge_command(), tmp_path)
    assert outcome.returncode == 0, outcome.stdout + outcome.stderr
    data = CoverageData(basename=str(tmp_path / ".coverage"))
    data.read()
    assert data.has_arcs()
    assert len(data.measured_files()) == 1
    root = ET.parse(tmp_path / "atomic-root-proof-combined-coverage.xml").getroot()
    assert root.attrib["branches-valid"] == root.attrib["branches-covered"] == "2"
    assert root.attrib["branch-rate"] == "1"


@pytest.mark.parametrize("outcome", ("failures", "errors", "skipped"))
def test_history_completion_gate_rejects_nonpassing_or_skipped_test_evidence(tmp_path, outcome):
    suite = ET.Element("testsuite", tests="1", failures="0", errors="0", skipped="0")
    suite.set(outcome, "1")
    ET.ElementTree(suite).write(tmp_path / "history-successor-reproposal-results.xml")
    command = _step(
        _jobs()["history-successor-reproposal"], "Require complete history reproposal execution"
    )["run"]
    result = _run_step(command, tmp_path)
    assert result.returncode != 0
