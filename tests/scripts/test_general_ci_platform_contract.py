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
    assert test_step["env"]["DUDZIAN_TEST_POSTGRES_RESTART_CONTAINER_ID"] == (
        "${{ job.services.postgres.id }}"
    )


def test_postgresql_restart_proof_controls_exact_service_and_waits_on_test_dsn():
    path = ROOT / "tests/security/test_postgresql_freshness_authority_core.py"
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source)
    helper = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "_restart_controlled_postgresql"
    )
    helper_source = ast.get_source_segment(source, helper)
    assert helper_source is not None
    assert "DUDZIAN_TEST_POSTGRES_RESTART_CONTAINER_ID" in helper_source
    assert '["docker", "restart", container_id]' in helper_source
    assert "psycopg.connect(DSN" in helper_source
    assert "time.monotonic() + 30.0" in helper_source
    assert "shell=True" not in helper_source
    assert '["pg_ctlcluster", "16", "main", "restart"]' not in source


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
    assert '"${PYTHON_BIN}" -m pytest -q' in command
    assert "tests/security/test_freshness_semantic_verifier_postgresql.py" in command
    assert "--collect-only" not in command
    assert '-m "' not in command
    assert "-m '" not in command
    assert "os.geteuid() == 0" in command
    assert "mktemp -d /tmp/dz-peer-source-" in command
    assert 'rm -rf "${SOURCE_STAGE}/checkout/.git"' in command
    assert 'chown -R root:root "${SOURCE_STAGE}"' in command
    assert 'chmod -R a-w,u+rwX,go+rX "${SOURCE_STAGE}"' in command
    assert 'runuser -u "${identity}" -- test -r' in command
    assert 'runuser -u "${identity}" -- test -w' in command
    assert 'PYTHON_BIN="$(command -v python)"' in command
    assert '"${PYTHON_BIN}" != /* || ! -x "${PYTHON_BIN}"' in command
    preflight = (
        '"${PYTHON_BIN}" -c '
        "'import bot_core; import bot_core.freshness_semantic_verifier; import psycopg'"
    )
    assert preflight in command
    assert "\n            python -c " not in command
    assert command.index(preflight) < command.index('"${PYTHON_BIN}" -m pytest -q')
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


def test_direct_peer_auth_fixture_stops_before_pwd_on_simulated_windows():
    path = ROOT / "tests/security/test_postgresql_freshness_production_local_authentication.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    pwd_line = next(
        node.lineno
        for node in ast.walk(tree)
        if isinstance(node, ast.Import) and any(alias.name == "pwd" for alias in node.names)
    )
    guard = next(
        node
        for node in tree.body
        if isinstance(node, ast.If) and "sys.platform.startswith" in ast.unparse(node.test)
    )
    assert guard.lineno < pwd_line
    assert isinstance(guard.test, ast.UnaryOp)

    script = """
import importlib.abc
import runpy
import sys
import pytest

class PwdBlocker(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path, target=None):
        if fullname == "pwd":
            raise AssertionError("pwd import was attempted before the platform boundary")
        return None

sys.modules.pop("pwd", None)
sys.meta_path.insert(0, PwdBlocker())
sys.platform = "win32"
try:
    runpy.run_path(sys.argv[1])
except pytest.skip.Exception:
    pass
else:
    raise AssertionError("direct non-Linux peer-auth collection was not skipped")
"""
    result = subprocess.run(
        [sys.executable, "-c", script, str(path)], cwd=ROOT, text=True, capture_output=True
    )
    assert result.returncode == 0, result.stderr


def test_stale_socket_proof_remains_portable_and_uses_short_tmp_root():
    path = ROOT / "tests/security/test_freshness_production_local_deployment.py"
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source)
    functions = {node.name: node for node in tree.body if isinstance(node, ast.FunctionDef)}
    stale = functions["test_only_exact_owned_stale_socket_is_recreated"]
    decorators = "\n".join(ast.unparse(item) for item in stale.decorator_list)
    assert "sys.platform.startswith('linux')" not in decorators
    assert "short_unix_socket_root" in [argument.arg for argument in stale.args.args]
    helper = functions["short_unix_socket_root"]
    helper_source = ast.get_source_segment(source, helper)
    assert 'dir="/tmp"' in helper_source
    assert "< 100" in helper_source


def test_clean_tls_fixture_is_dynamic_and_production_expiry_policy_is_unchanged():
    fixture = (ROOT / "tests/test_audit_tls_assets_script.py").read_text(encoding="utf-8")
    baseline = (ROOT / "tests/test_audit_security_baseline_script.py").read_text(encoding="utf-8")
    policy = (ROOT / "bot_core/security/certificates.py").read_text(encoding="utf-8")
    assert "_healthy_certificate_pair" in baseline
    assert "timedelta(days=400)" in fixture
    assert "_CERT" not in fixture
    assert "warn_expiring_within_days: float = 30.0" in policy
    assert "remaining_days <= warn_expiring_within_days" in policy


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


def test_cha_attempt_store_keeps_portable_semantics_separate_from_posix_mode_proof():
    path = ROOT / "tests/security/test_cha_attempt_store_production_local.py"
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source)
    functions = {node.name: node for node in tree.body if isinstance(node, ast.FunctionDef)}

    portable = functions["test_creates_dedicated_store_with_identity_and_effective_pragmas"]
    assert portable.decorator_list == []
    portable_source = ast.get_source_segment(source, portable)
    assert portable_source is not None
    for proof in (
        "path.is_file()",
        "ProviderRole.CHA_ATTEMPT_STORE",
        "SecurityProfile.PRODUCTION_LOCAL",
        '("wal", 2, 1)',
        "store.credential_identities() == ()",
    ):
        assert proof in portable_source
    assert "st_mode" not in portable_source

    posix = functions["test_posix_store_file_is_owner_read_write_only"]
    decorators = [ast.unparse(item) for item in posix.decorator_list]
    assert any("os.name != 'posix'" in decorator for decorator in decorators)
    assert any(
        "exact 0600 is a POSIX-native filesystem permission proof" in decorator
        for decorator in decorators
    )
    mode_assertions = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Assert) and "st_mode" in ast.unparse(node.test)
    ]
    assert len(mode_assertions) == 1
    assert mode_assertions[0] in ast.walk(posix)
    mode_source = ast.get_source_segment(source, mode_assertions[0].test)
    assert mode_source == "path.stat().st_mode & 0o777 == 0o600"


def test_cha_attempt_store_production_keeps_posix_chmod_0600_guard():
    path = ROOT / "bot_core/cha_attempt_store.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    guarded_chmods = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.If)
        and ast.unparse(node.test) == "os.name == 'posix'"
        and any(
            isinstance(child, ast.Call)
            and ast.unparse(child.func) == "os.chmod"
            and [ast.unparse(argument) for argument in child.args] == ["path", "384"]
            for child in ast.walk(node)
        )
    ]
    assert len(guarded_chmods) == 1


def test_local_signing_custody_keeps_fail_closed_non_posix_lock_contract():
    path = ROOT / "bot_core/local_signing_custody.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    custody_lock = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "_custody_lock"
    )
    guards = [
        node
        for node in custody_lock.body
        if isinstance(node, ast.If) and ast.unparse(node.test) == "os.name != 'posix'"
    ]
    assert len(guards) == 1
    assert any(
        isinstance(node, ast.Raise)
        and "production-local custody locking is unavailable" in ast.unparse(node)
        for node in ast.walk(guards[0])
    )


def test_local_signing_custody_posix_boundary_is_granular_and_complete():
    expected_portable = {
        "tests/security/test_freshness_signing_custody.py": {
            "test_test_profile_cannot_be_provisioned_as_production",
        },
        "tests/security/test_local_signing_custody.py": {
            "test_arbitrary_or_plaintext_secret_backend_is_not_production_custody",
            "test_malicious_path_subclass_is_rejected_before_semantic_methods",
            "test_exact_security_role_and_lifecycle_boundaries",
        },
    }
    for relative_path, portable_names in expected_portable.items():
        tree = ast.parse((ROOT / relative_path).read_text(encoding="utf-8"))
        tests = {
            node.name: {ast.unparse(decorator) for decorator in node.decorator_list}
            for node in tree.body
            if isinstance(node, ast.FunctionDef) and node.name.startswith("test_")
        }
        assert portable_names < tests.keys()
        assert all("requires_posix_custody_locking" not in tests[name] for name in portable_names)
        assert all(
            "requires_posix_custody_locking" in decorators
            for name, decorators in tests.items()
            if name not in portable_names
        )
        assert not any(
            isinstance(node, ast.Assign)
            and any(
                isinstance(target, ast.Name) and target.id == "pytestmark"
                for target in node.targets
            )
            for node in tree.body
        )


def test_simulated_windows_classifies_lock_dependent_and_portable_custody_tests():
    script = """
import importlib
import runpy
import sys
import os

# Load the module and all platform-sensitive stdlib dependencies before the
# simulation, then rebuild only the custody marker under the non-POSIX name.
importlib.import_module(sys.argv[2])
import tests.security._local_signing_platform as boundary

os.name = "nt"
importlib.reload(boundary)
namespace = runpy.run_path(sys.argv[1])
locked = namespace[sys.argv[3]]
portable = namespace[sys.argv[4]]
assert any(mark.name == "skipif" and mark.args == (True,) for mark in locked.pytestmark)
assert not any(mark.name == "skipif" for mark in getattr(portable, "pytestmark", ()))
"""
    cases = (
        (
            "tests/security/test_freshness_signing_custody.py",
            "tests.security.test_freshness_signing_custody",
            "test_distinct_provisioning_restart_snapshot_and_active_signing",
            "test_test_profile_cannot_be_provisioned_as_production",
        ),
        (
            "tests/security/test_local_signing_custody.py",
            "tests.security.test_local_signing_custody",
            "test_offline_provisioning_is_distinct_and_restart_stable",
            "test_arbitrary_or_plaintext_secret_backend_is_not_production_custody",
        ),
    )
    for relative_path, module_name, locked, portable in cases:
        result = subprocess.run(
            [
                sys.executable,
                "-c",
                script,
                str(ROOT / relative_path),
                module_name,
                locked,
                portable,
            ],
            cwd=ROOT,
            text=True,
            capture_output=True,
        )
        assert result.returncode == 0, result.stderr
