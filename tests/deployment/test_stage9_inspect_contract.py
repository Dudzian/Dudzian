"""Preinstallation INSPECT needs a trusted tool, never an installed product.

The positive orchestration model executes the real reviewed environment helper
source. Only authentication/import-state and read-only host observations are
modeled; separate harness tests cover actual inventory and metadata rejection.
No replacement package, native object or unverified bytecode executes here.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest

from deployment import stage9_vm_trial as trial

pytestmark = pytest.mark.skipif(sys.platform != "linux", reason="Linux diagnostic contract")
REPOSITORY = Path(__file__).resolve().parents[2]


@pytest.fixture
def trusted_inspection(monkeypatch):
    helper = REPOSITORY / "deployment/stage9_vm_environment.py"
    source = helper.read_bytes()
    digest = hashlib.sha256(source).hexdigest()
    events = []

    def authenticated_sources(revision=None):
        assert revision is None
        assert hashlib.sha256(source).hexdigest() == digest
        events.append("authenticate-tool")
        return {"deployment/stage9_vm_environment.py": source}

    def forbidden(*args, **kwargs):
        pytest.fail("INSPECT attempted a product, VM-approval or execution boundary")

    load = trial._environment_helpers

    def authenticated_environment(execute):
        assert execute is False
        namespace = load(execute)
        events.append("authenticated-helper")
        namespace["_query"] = lambda arguments: None
        namespace["_public_text"] = lambda *args: None
        namespace["require_approved_vm"] = forbidden
        return namespace

    monkeypatch.setattr(trial, "_harness_sources", authenticated_sources)
    monkeypatch.setattr(trial, "_import_state", lambda: events.append("isolated-import-state"))
    monkeypatch.setattr(trial, "_environment_helpers", authenticated_environment)
    for name in (
        "load_inputs",
        "_installed_executable",
        "_native_trial",
        "execute_phase",
        "_command",
    ):
        monkeypatch.setattr(trial, name, forbidden)
    return events


def test_inspect_precedes_product_installation_and_vm_approval(trusted_inspection, capsys):
    assert trial.main(["--mode", "INSPECT"]) == 0
    report = json.loads(capsys.readouterr().out)
    assert trusted_inspection[0] == "authenticate-tool"
    assert trusted_inspection.index("isolated-import-state") < trusted_inspection.index(
        "authenticated-helper"
    )
    assert report["environment"]["observation_complete"] is False
    assert report["result"] == "NOT_EXECUTED"
    assert all(check["status"] == "NOT_EXECUTED" for check in report["checks"].values())
    assert "vm_approval" not in report
    assert "manifest_sha256" not in report
    assert report["production_qualification"] == "BLOCKED"
    assert not report["stage9_production_ready"]


@pytest.mark.parametrize(
    "arguments", [["--phase", "install"], ["--inputs", "/never/read"], ["--native-worker"]]
)
def test_inspect_refuses_execution_arguments(trusted_inspection, capsys, arguments):
    assert trial.main(["--mode", "INSPECT", *arguments]) == 2
    report = json.loads(capsys.readouterr().out)
    assert report["reason"] == "INSPECT_EXECUTION_ARGUMENTS_REFUSED"
    assert all(check["status"] in {"NOT_EXECUTED", "FAIL"} for check in report["checks"].values())


def test_authentication_refusal_precedes_diagnostic_helper(monkeypatch, capsys):
    def unauthenticated(revision=None):
        raise trial.TrialError("HARNESS_INTEGRITY_CHANGED")

    monkeypatch.setattr(trial, "_harness_sources", unauthenticated)
    monkeypatch.setattr(
        trial, "_environment_helpers", lambda execute: pytest.fail("unverified helper executed")
    )
    assert trial.main(["--mode", "INSPECT"]) == 2
    report = json.loads(capsys.readouterr().out)
    assert report["reason"] == "HARNESS_INTEGRITY_CHANGED"
    assert "environment" not in report


def test_real_isolated_cli_refuses_unprotected_checkout_before_helper(tmp_path):
    script = """
import runpy, sys
namespace = runpy.run_path(sys.argv[1])
called = []
def forbidden(execute):
    called.append('unverified-helper')
    raise AssertionError('unverified helper executed')
namespace['main'].__globals__['_environment_helpers'] = forbidden
code = namespace['main'](['--mode', 'INSPECT'])
assert not called, 'diagnostics ran from an unprotected checkout'
assert code == 2, 'mutable checkout was accepted as executable installation'
raise SystemExit(code)
"""
    result = subprocess.run(
        [
            sys.executable,
            "-I",
            "-s",
            "-B",
            "-c",
            script,
            str(REPOSITORY / "deployment/stage9_vm_trial.py"),
        ],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
        env={"PATH": "/usr/bin:/bin"},
    )
    assert result.returncode == 2, result.stderr
    report = json.loads(result.stdout)
    assert report["result"] in {"FAIL", "BLOCKED"}
    assert "environment" not in report
    assert all(check["status"] == "NOT_EXECUTED" for check in report["checks"].values())


def test_cli_help_distinguishes_authenticated_tool_and_runtime_installation():
    result = subprocess.run(
        [
            sys.executable,
            "-I",
            "-s",
            "-B",
            str(REPOSITORY / "deployment/stage9_vm_trial.py"),
            "--help",
        ],
        capture_output=True,
        text=True,
        check=False,
        env={"PATH": "/usr/bin:/bin"},
    )
    assert result.returncode == 0
    assert "INSPECT" in result.stdout
    assert "chronion" in result.stdout.lower()
    assert "runtime" in result.stdout.lower()
