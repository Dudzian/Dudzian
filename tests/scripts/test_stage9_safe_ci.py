"""Scope and result integrity of the separate non-signing Stage 9 CI path."""

from __future__ import annotations

import ast
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest
import yaml

from scripts.quality import run_stage9_safe_validation as safe

ROOT = Path(__file__).resolve().parents[2]
HEAD = "a" * 40


def workflow():
    return yaml.safe_load((ROOT / ".github/workflows/platform-deployment.yml").read_text())


def test_dispatch_routes_only_explicit_safe_scope_and_keeps_old_jobs():
    current = workflow()
    old = yaml.safe_load(
        subprocess.check_output(
            [
                "git",
                "show",
                "5e0646a533d071b3f4a1c62d8d9836ece9081c95:.github/workflows/platform-deployment.yml",
            ],
            cwd=ROOT,
            text=True,
        )
    )
    inputs = current.get("on", current.get(True))["workflow_dispatch"]["inputs"]
    assert inputs["validation_scope"]["default"] == "full"
    assert inputs["validation_scope"]["options"] == ["full", "stage9-safe"]
    for name, original in old["jobs"].items():
        selected = current["jobs"][name]
        assert {key: value for key, value in selected.items() if key != "if"} == {
            key: value for key, value in original.items() if key != "if"
        }
        assert "inputs.validation_scope != 'stage9-safe'" in selected["if"]
        if "if" in original:
            assert " ".join(original["if"].split()) in " ".join(selected["if"].split())
    assert set(current["jobs"]) == set(old["jobs"]) | {"stage9-safe-validation"}


def test_safe_matrix_is_unprivileged_and_preserves_exact_commit_and_evidence():
    job = workflow()["jobs"]["stage9-safe-validation"]
    assert (
        job["if"]
        == "github.event_name == 'workflow_dispatch' && inputs.validation_scope == 'stage9-safe'"
    )
    assert job["strategy"]["matrix"]["python"] == ["3.11", "3.12"]
    assert job["strategy"]["fail-fast"] is False
    assert job["permissions"] == {"contents": "read"}
    assert job["runs-on"] == "ubuntu-24.04"
    assert not {"environment", "services", "needs", "continue-on-error"} & set(job)
    checkout = next(
        step for step in job["steps"] if step.get("uses", "").startswith("actions/checkout@")
    )
    assert checkout["with"] == {
        "ref": "${{ github.sha }}",
        "fetch-depth": 0,
        "persist-credentials": False,
    }
    source = "\n".join(step.get("run", "") for step in job["steps"])
    assert "run_stage9_safe_validation.py" in source
    assert "build_freshness_runtime" not in source
    assert "issue-challenge" not in source
    assert "secrets." not in json.dumps(job)
    upload = next(
        step for step in job["steps"] if step.get("uses", "").startswith("actions/upload-artifact@")
    )
    assert upload["if"] == "always()"
    assert upload["with"]["if-no-files-found"] == "error"
    assert ".coverage" in upload["with"]["path"]
    assert "*.xml" in upload["with"]["path"]
    assert "*.json" in upload["with"]["path"]


@pytest.mark.parametrize("change", ["head", "malformed", "stage10", "physical"])
def test_actual_dispatch_guard_refuses_wrong_commit_or_privileged_inputs(change):
    step = workflow()["jobs"]["stage9-safe-validation"]["steps"][0]
    environment = dict(
        os.environ,
        STAGE9_EXPECTED_COMMIT=HEAD,
        GITHUB_SHA=HEAD,
        STAGE9_RUN_STAGE10="false",
        STAGE9_RUN_PHYSICAL="false",
    )
    if change == "head":
        environment["GITHUB_SHA"] = "b" * 40
    elif change == "malformed":
        environment["STAGE9_EXPECTED_COMMIT"] = "main"
    elif change == "stage10":
        environment["STAGE9_RUN_STAGE10"] = "true"
    else:
        environment["STAGE9_RUN_PHYSICAL"] = "true"
    result = subprocess.run(
        [sys.executable, "-I", "-B", "-c", step["run"]],
        env=environment,
        capture_output=True,
        check=False,
    )
    assert result.returncode != 0
    assert b"STAGE9_SAFE_DISPATCH_REFUSED" in result.stderr


def test_safe_selection_never_expands_signed_or_approval_fixtures():
    nodes = safe.safe_test_nodes()
    assert len(nodes) == len(set(nodes))
    for forbidden in (
        "tests/deployment/test_freshness_runtime.py",
        "tests/deployment/test_stage9_vm_environment.py",
        "tests/deployment/test_stage9_vm_trial.py",
        "tests/deployment/test_build_freshness_runtime.py",
        "tests/deployment/test_freshness_runtime_bytecode.py",
    ):
        assert forbidden not in nodes
    for node in nodes:
        if "::" not in node:
            continue
        filename, name = node.split("::", 1)
        functions = {
            n.name: n
            for n in ast.parse((ROOT / filename).read_text()).body
            if isinstance(n, ast.FunctionDef)
        }
        selected = functions[name]
        assert not {"candidate", "approved"} & {arg.arg for arg in selected.args.args}
        assert not any(
            isinstance(n, ast.Name) and n.id in {"_approval", "_facts"} for n in ast.walk(selected)
        )
    assert "test_bootstrap_real_ed25519_invalid_inputs" not in json.dumps(nodes)
    assert "test_successful_phase_never_marks_other_phases_or_native_matrix_pass" not in json.dumps(
        nodes
    )
    assert "test_confirmed_security_violation_is_fail_not_blocked" not in json.dumps(nodes)
    assert safe.NOT_EXECUTED


def test_changed_code_gate_keeps_seventy_and_all_new_production_files(tmp_path, monkeypatch):
    monkeypatch.setattr(
        safe,
        "_git",
        lambda *args: "deployment/new.py\nscripts/quality/new.py\ntests/test_new.py\ndocs/new.md",
    )
    assert safe.coverage_sources(ROOT, safe.AUDITED_BASE, HEAD) == (
        "deployment.new",
        "scripts.quality.new",
    )
    plan = dict(safe.build_steps(sys.executable, ROOT, tmp_path, safe.AUDITED_BASE, HEAD))
    assert "--source=deployment.new,scripts.quality.new" in plan["regression"]
    assert "--noconftest" in plan["regression"]
    assert "-k" not in plan["regression"]
    assert "--fail-under=70" in plan["changed-code-coverage"]
    assert "--compare-branch=" + safe.AUDITED_BASE in plan["changed-code-coverage"]
    assert plan["format-preflight"][1:] == ["-m", "pre_commit", "run", "ruff-format", "--all-files"]
    assert "docs/new.md" in plan["secret-scan"]
    existing = (ROOT / ".github/workflows/quality-security.yml").read_text()
    assert "--fail-under=70" in existing


@pytest.mark.parametrize("paths", ["docs/only.md", "deployment/not-valid.py"])
def test_coverage_never_silently_ignores_unsupported_changed_source(monkeypatch, paths):
    monkeypatch.setattr(safe, "_git", lambda *args: paths)
    with pytest.raises(
        ValueError, match="CHANGED_PYTHON_SOURCE_REQUIRED|COVERAGE_SOURCE_PATH_REFUSED"
    ):
        safe.coverage_sources(ROOT, safe.AUDITED_BASE, HEAD)


@pytest.fixture
def context(tmp_path, monkeypatch):
    checkout = tmp_path / "checkout"
    checkout.mkdir()
    output = tmp_path / "evidence"
    for node in safe.safe_test_nodes():
        source = checkout / node.partition("::")[0]
        source.parent.mkdir(parents=True, exist_ok=True)
        source.touch()  # Inventory data only; no source is imported or executed.
    monkeypatch.setattr(safe, "_git", lambda root, *args: HEAD if args[0] == "rev-parse" else "")
    monkeypatch.setattr(safe.subprocess, "run", lambda *args, **kwargs: None)
    return checkout, output


@pytest.mark.parametrize(
    "change", ["root", "platform", "version", "head", "dirty", "base", "output", "missing-test"]
)
def test_context_refuses_unsupported_or_ambiguous_execution(context, monkeypatch, change):
    checkout, output = context
    expected, base = HEAD, safe.AUDITED_BASE
    if change == "root":
        monkeypatch.setattr(safe.os, "geteuid", lambda: 0)
    elif change == "platform":
        monkeypatch.setattr(safe.sys, "platform", "win32")
    elif change == "version":
        monkeypatch.setattr(safe.sys, "version_info", (3, 13, 0))
    elif change == "head":
        expected = "b" * 40
    elif change == "dirty":
        monkeypatch.setattr(
            safe, "_git", lambda root, *args: HEAD if args[0] == "rev-parse" else " M changed.py"
        )
    elif change == "base":
        base = "b" * 40
    elif change == "output":
        output = checkout / "new-output"
    else:
        (checkout / safe.SAFE_FILES[0]).unlink()
    with pytest.raises(ValueError):
        safe.validate_context(checkout, expected, base, output)
    assert not output.exists()


@pytest.mark.parametrize(
    "executable,code,status",
    [
        ("/usr/bin/true", 0, "PASS"),
        ("/usr/bin/false", 1, "FAIL"),
        ("/nonexistent-stage9-tool", None, "NOT_EXECUTED"),
    ],
)
def test_actual_tool_result_retains_original_exit_or_not_executed(
    tmp_path, executable, code, status
):
    result = safe.run_step("actual", [executable], ROOT, tmp_path, dict(os.environ))
    assert result["exit_code"] == code
    assert result["status"] == status
    assert json.loads((tmp_path / "actual.result.json").read_text()) == result
    assert result["argv"] == [executable]
    assert result["started_at"] <= result["finished_at"]


def test_timeout_is_fail_with_unknown_exit_and_explicit_error(tmp_path, monkeypatch):
    def timed_out(*args, **kwargs):
        raise subprocess.TimeoutExpired("known-tool", 1200)

    monkeypatch.setattr(safe.subprocess, "run", timed_out)
    result = safe.run_step("timeout", ["known-tool"], ROOT, tmp_path, {})
    assert result["status"] == "FAIL" and result["exit_code"] is None
    assert (tmp_path / "timeout.error").read_text() == "TimeoutExpired\n"


@pytest.mark.parametrize(
    "raw,expected,skipped",
    [
        (None, "NOT_EXECUTED", 0),
        ("broken", "FAIL", 0),
        ("<testsuites/>", "FAIL", 0),
        ("<testsuite><testcase><skipped/></testcase></testsuite>", "FAIL", 1),
        ("<testsuite><testcase><failure/></testcase></testsuite>", "FAIL", 0),
        ("<testsuite><testcase><error/></testcase></testsuite>", "FAIL", 0),
        ("<testsuite><testcase/></testsuite>", "PASS", 0),
    ],
)
def test_junit_never_promotes_missing_empty_skipped_or_failed_tests(
    tmp_path, raw, expected, skipped
):
    path = tmp_path / "results.xml"
    if raw is not None:
        path.write_text(raw)
    result = safe.junit_result(path)
    assert result["status"] == expected
    assert result.get("not_executed", 0) == skipped


def test_runner_refuses_output_inside_checkout_without_creating_it():
    output = ROOT / "work-never-created-by-safe-runner"
    assert not output.exists()
    with pytest.raises(SystemExit) as error:
        safe.main(["--expected-commit", HEAD, "--output", str(output)])
    assert error.value.code == 2
    assert not output.exists()


def test_existing_output_is_preserved_and_refused(tmp_path):
    marker = tmp_path / "existing.txt"
    marker.write_text("preserve")
    assert safe.main(["--expected-commit", HEAD, "--output", str(tmp_path)]) == 2
    assert marker.read_text() == "preserve"


def test_blocked_context_keeps_receipt_and_no_steps_execute(tmp_path, monkeypatch):
    output = tmp_path / "blocked"

    def refused(*args):
        raise ValueError("EXPECTED_HEAD_MISMATCH")

    monkeypatch.setattr(safe, "validate_context", refused)
    monkeypatch.setattr(
        safe, "run_step", lambda *args: pytest.fail("blocked execution reached tool")
    )
    assert safe.main(["--expected-commit", HEAD, "--output", str(output)]) == 2
    receipt = json.loads((output / "validation.json").read_text())
    assert receipt["status"] == "BLOCKED"
    assert receipt["checks"] == []
    assert receipt["reason"] == "EXPECTED_HEAD_MISMATCH"
    assert all(record["status"] == "NOT_EXECUTED" for record in receipt["not_executed"])
    assert receipt["s3"] == "BLOCKED" and receipt["production"] == "NO_GO"


def test_job_environment_uses_only_contexts_available_before_runner_assignment():
    # GitHub rejected the old dispatch before creating any run because runner
    # is unavailable in jobs.<id>.env. Step inputs may resolve runner.temp.
    job = workflow()["jobs"]["stage9-safe-validation"]
    assert "runner." not in json.dumps(job["env"])
    invocation = next(
        step for step in job["steps"] if "run_stage9_safe_validation.py" in step.get("run", "")
    )
    assert '--output "$RUNNER_TEMP/$STAGE9_SAFE_OUTPUT"' in invocation["run"]
    upload = next(
        step for step in job["steps"] if step.get("uses", "").startswith("actions/upload-artifact@")
    )
    assert "${{ runner.temp }}/${{ env.STAGE9_SAFE_OUTPUT }}/" in upload["with"]["path"]
