"""Regression contracts for the cross-platform General CI security slice."""

from __future__ import annotations

import ast
from pathlib import Path
import subprocess
import sys
import tomllib

import pytest
import yaml


ROOT = Path(__file__).resolve().parents[2]
WORKFLOW = ROOT / ".github/workflows/ci.yml"
EXTERNAL_POSTGRESQL_TESTS = (
    "tests/security/test_postgresql_entitlement_registry.py",
    "tests/security/test_postgresql_security_definer_principal_semantics.py",
    "tests/security/test_postgresql_freshness_authority_core.py",
)


def _general_ci_jobs() -> dict:
    document = yaml.safe_load(WORKFLOW.read_text(encoding="utf-8"))
    return document["jobs"]


def _pytest_command(job: dict) -> str:
    return next(step["run"] for step in job["steps"] if step["name"] == "Run fast pytest suite")


def test_external_postgresql_marker_is_registered_and_applied():
    config = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    markers = config["tool"]["pytest"]["ini_options"]["markers"]
    assert any(marker.startswith("external_postgresql:") for marker in markers)
    for relative_path in EXTERNAL_POSTGRESQL_TESTS:
        source = (ROOT / relative_path).read_text(encoding="utf-8")
        assert "pytestmark = pytest.mark.external_postgresql" in source


def test_ubuntu_runs_external_postgresql_against_postgresql_16_service():
    jobs = _general_ci_jobs()
    ubuntu = jobs["py-tests-ubuntu"]
    postgres = ubuntu["services"]["postgres"]
    assert postgres["image"] == "postgres:16"
    assert postgres["env"]["POSTGRES_HOST_AUTH_METHOD"] == "trust"
    assert "55432:5432" in [str(mapping) for mapping in postgres["ports"]]
    command = _pytest_command(ubuntu)
    assert "not external_postgresql" not in command
    test_step = next(step for step in ubuntu["steps"] if step["name"] == "Run fast pytest suite")
    for variable in ("DUDZIAN_TEST_POSTGRES_DSN", "ENTITLEMENT_REGISTRY_POSTGRES_ADMIN_DSN"):
        assert "127.0.0.1 port=55432" in test_step["env"][variable]


def test_non_linux_general_ci_excludes_only_the_explicit_database_marker():
    jobs = _general_ci_jobs()
    for job_name in ("py-tests-windows", "py-tests-macos"):
        assert "not external_postgresql" in _pytest_command(jobs[job_name])


def test_ubuntu_has_authoritative_elevated_native_peer_auth_execution():
    jobs = _general_ci_jobs()
    step_name = "Run authoritative native Linux peer-auth proof"
    jobs_with_step = [
        job_name
        for job_name, job in jobs.items()
        if any(step.get("name") == step_name for step in job.get("steps", []))
    ]
    assert jobs_with_step == ["py-tests-ubuntu"]
    ubuntu_steps = jobs["py-tests-ubuntu"]["steps"]
    matching_steps = [step for step in ubuntu_steps if step.get("name") == step_name]
    assert len(matching_steps) == 1

    command = matching_steps[0]["run"]
    assert "sudo " in command
    assert "python -m pytest -q" in command
    assert "tests/security/test_freshness_semantic_verifier_postgresql.py" in command
    assert "--collect-only" not in command
    assert '-m "' not in command
    assert "-m '" not in command
    assert "os.geteuid() == 0" in command
    for tool in ("runuser", "useradd", "pg_config", "initdb", "pg_ctl"):
        assert f'"{tool}"' in command

    for job_name in ("py-tests-windows", "py-tests-macos"):
        commands = "\n".join(str(step.get("run", "")) for step in jobs[job_name]["steps"])
        assert step_name not in [step.get("name") for step in jobs[job_name]["steps"]]
        assert "test_freshness_semantic_verifier_postgresql.py" not in commands


def test_peer_auth_platform_guard_precedes_pwd_and_preserves_linux_execution():
    path = ROOT / "tests/security/test_freshness_semantic_verifier_postgresql.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    pwd_import_line = next(
        node.lineno
        for node in ast.walk(tree)
        if isinstance(node, ast.Import) and any(alias.name == "pwd" for alias in node.names)
    )
    platform_guard = next(
        node
        for node in tree.body
        if isinstance(node, ast.If)
        and "sys.platform.startswith" in ast.unparse(node.test)
        and "linux" in ast.unparse(node.test)
    )
    assert platform_guard.lineno < pwd_import_line
    assert isinstance(platform_guard.test, ast.UnaryOp)  # non-Linux only; Linux executes the proof


def test_peer_auth_module_stops_before_pwd_on_simulated_windows():
    script = """
import importlib.abc
import runpy
import sys
import psycopg
import pytest

class PwdBlocker(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path, target=None):
        if fullname == "pwd":
            raise ModuleNotFoundError("blocked pwd", name="pwd")
        return None

sys.modules.pop("pwd", None)
sys.meta_path.insert(0, PwdBlocker())
sys.platform = "win32"
try:
    runpy.run_path(sys.argv[1])
except pytest.skip.Exception:
    pass
else:
    raise AssertionError("non-Linux peer-auth module was not explicitly skipped")
"""
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            script,
            str(ROOT / "tests/security/test_freshness_semantic_verifier_postgresql.py"),
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
    )
    assert result.returncode == 0, result.stderr


def test_semantic_verifier_keeps_crypto_portable_and_guards_only_linux_ipc_proofs():
    path = ROOT / "tests/security/test_freshness_semantic_verifier.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    top_level_calls = [node for node in tree.body if isinstance(node, ast.Expr)]
    assert not any("pytest.skip(" in ast.unparse(node) for node in top_level_calls)

    functions = {
        node.name: node
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    portable = functions["test_three_real_ed25519_signatures_produce_one_derived_preparation"]
    assert "requires_linux_unix_ipc" not in [ast.unparse(item) for item in portable.decorator_list]

    linux_ipc_tests = {
        name
        for name, function in functions.items()
        if name.startswith("test_")
        and any(
            isinstance(call, ast.Call)
            and isinstance(call.func, ast.Name)
            and call.func.id in {"_ipc_server", "_ipc_exchange"}
            for call in ast.walk(function)
        )
    }
    assert linux_ipc_tests == {
        "test_unix_ipc_process_accepts_only_closed_valid_request",
        "test_unix_ipc_has_closed_non_crypto_failure_outcomes",
        "test_unix_ipc_process_rejects_injection_and_malformed_without_oracle",
        "test_unix_ipc_process_rejects_bad_frame_lengths",
        "test_unix_ipc_wrong_peer_cannot_invoke_and_there_is_no_tcp_listener",
    }
    for name in linux_ipc_tests:
        decorators = [ast.unparse(item) for item in functions[name].decorator_list]
        assert "requires_linux_unix_ipc" in decorators

    guard = next(
        node
        for node in tree.body
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id == "requires_linux_unix_ipc"
            for target in node.targets
        )
    )
    guard_source = ast.unparse(guard.value)
    assert "not sys.platform.startswith('linux')" in guard_source
    assert all(token in guard_source for token in ("AF_UNIX", "fork", "SO_PEERCRED", "UID"))


@pytest.mark.parametrize("platform", ["win32", "darwin"])
def test_non_linux_semantic_verifier_marks_native_ipc_before_runtime(platform):
    script = """
import runpy
import sys
import tests.security.test_freshness_semantic_verifier

sys.platform = sys.argv[2]
namespace = runpy.run_path(sys.argv[1])
native = namespace["test_unix_ipc_wrong_peer_cannot_invoke_and_there_is_no_tcp_listener"]
marks = getattr(native, "pytestmark", ())
assert any(mark.name == "skipif" and mark.args == (True,) for mark in marks)
portable = namespace["test_three_real_ed25519_signatures_produce_one_derived_preparation"]
assert not any(mark.name == "skipif" for mark in getattr(portable, "pytestmark", ()))
"""
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            script,
            str(ROOT / "tests/security/test_freshness_semantic_verifier.py"),
            platform,
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
    )
    assert result.returncode == 0, result.stderr
