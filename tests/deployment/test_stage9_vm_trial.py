"""Modele orkiestracji VM; testy nie wykonują instalacji ani startu systemd."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from deployment import stage9_vm_trial as trial

pytestmark = pytest.mark.skipif(
    sys.platform != "linux", reason="Orkiestracja natywnej próby VM Linux"
)

REVISION = "a" * 40
ENVIRONMENT = "test-only-noble-001"


def test_snapshot_is_readonly_and_detects_content_mode_link_and_absence(tmp_path):
    root = tmp_path / "host"
    root.mkdir()
    file = root / "data"
    file.write_text("reviewed")
    file.chmod(0o644)
    missing = root / "missing"
    paths = (root, missing)
    original = trial.snapshot(paths)
    assert trial.snapshot(paths) == original
    assert list(root.iterdir()) == [file]
    file.write_text("replaced")
    assert trial.snapshot(paths) != original
    file.write_text("reviewed")
    assert trial.snapshot(paths) == original
    file.chmod(0o600)
    assert trial.snapshot(paths) != original
    if os.name == "posix":
        file.unlink()
        file.symlink_to(tmp_path / "outside")
        assert trial.snapshot(paths) != original


@pytest.mark.skipif(os.name != "posix", reason="FIFO jest granicą Linux")
def test_snapshot_refuses_special_file_without_blocking(tmp_path):
    os.mkfifo(tmp_path / "fifo")
    with pytest.raises(trial.TrialError, match="SNAPSHOT_SPECIAL_FILE"):
        trial.snapshot((tmp_path,))


def test_preflight_detects_new_sibling_installer_staging(tmp_path, monkeypatch):
    library = tmp_path / "usr-lib"
    library.mkdir()
    installed = library / "cryptohunter"
    installed.mkdir()
    original_snapshot = trial.snapshot
    monkeypatch.setattr(trial, "STAGING_PARENT", library)
    monkeypatch.setattr(trial, "snapshot", lambda: original_snapshot((installed,)))

    def preflight(*args, **kwargs):
        (library / (".cryptohunter-stage-" + "a" * 64)).mkdir()
        return {"status": "PASS", "state": "fresh"}

    monkeypatch.setattr(trial, "_installer", preflight)
    with pytest.raises(trial.TrialError, match="PREFLIGHT_MUTATED_HOST"):
        trial.checked_preflight({}, REVISION)


def test_staging_inventory_detects_changes_without_hashing_unrelated_data(tmp_path, monkeypatch):
    library = tmp_path / "usr-lib"
    library.mkdir()
    monkeypatch.setattr(trial, "STAGING_PARENT", library)
    before = trial.snapshot(())
    unrelated = library / "unrelated-pgdata"
    unrelated.mkdir()
    payload = unrelated / "active-data"
    payload.write_text("first")
    assert trial.snapshot(()) == before
    staging = library / (trial.STAGING_PREFIX + "a" * 64)
    staging.mkdir()
    staged_file = staging / "payload.py"
    staged_file.write_text("reviewed")
    created = trial.snapshot(())
    assert created != before
    staged_file.write_text("changed")
    assert trial.snapshot(()) != created
    staged_file.unlink()
    staging.rmdir()
    assert trial.snapshot(()) == before
    staging.symlink_to(unrelated, target_is_directory=True)
    linked = trial.snapshot(())
    assert linked != before
    payload.write_text("changed database pages")
    assert trial.snapshot(()) == linked


def test_staging_inventory_refuses_symlink_instead_of_parent(tmp_path, monkeypatch):
    linked = tmp_path / "usr-lib"
    linked.symlink_to(tmp_path, target_is_directory=True)
    monkeypatch.setattr(trial, "STAGING_PARENT", linked)
    with pytest.raises(trial.TrialError, match="SNAPSHOT_STAGING_PARENT_INVALID"):
        trial.snapshot(())


@pytest.mark.parametrize("status", ["BLOCKED", "FAIL", "NOT_EXECUTED"])
def test_summary_never_promotes_incomplete_or_blocked(status):
    assert trial.summarize({"first": {"status": "PASS"}, "last": {"status": status}}) == status


def test_summary_rejects_unknown_or_empty_statuses():
    with pytest.raises(trial.TrialError, match="REPORT_STATUS_INVALID"):
        trial.summarize({"test": {"status": "PENDING"}})
    assert trial.summarize({}) == "NOT_EXECUTED"
    assert trial.summarize({"test": {"status": "PASS"}}) == "PASS"


def test_new_report_has_no_native_or_security_success():
    report = trial.new_report("INSPECT", REVISION, ENVIRONMENT)
    assert report["native_trial"] == "PENDING"
    assert report["production_qualification"] == "BLOCKED"
    assert not report["stage9_production_ready"]
    assert not report["linux_physical_production_attestation"]
    assert trial.summarize(report["negative_checks"]) == "NOT_EXECUTED"
    assert len(report["negative_checks"]) >= 19


def test_inspect_does_not_load_inputs_or_run_product(monkeypatch, capsys):
    def forbidden(*args, **kwargs):
        pytest.fail("INSPECT przekroczył granicę odczytu")

    monkeypatch.setattr(trial, "load_inputs", forbidden)
    monkeypatch.setattr(trial, "execute_phase", forbidden)
    monkeypatch.setattr(trial, "_command", forbidden)
    monkeypatch.setattr(
        trial,
        "_environment_helpers",
        lambda execute: {"inspect_environment": lambda: {"pid1": "tail"}},
    )
    assert trial.main(["--mode", "INSPECT"]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["result"] == "NOT_EXECUTED"
    assert report["environment"]["pid1"] == "tail"


def test_inspect_refuses_execution_arguments(monkeypatch, capsys):
    monkeypatch.setattr(trial, "_environment_helpers", lambda execute: {})
    assert trial.main(["--mode", "INSPECT", "--phase", "install"]) == 2
    assert json.loads(capsys.readouterr().out)["reason"] == "INSPECT_EXECUTION_ARGUMENTS_REFUSED"


def test_guard_runs_before_inputs_or_any_execution_and_sanitizes(monkeypatch, capsys):
    events = []

    def guard(*args, **kwargs):
        events.append("guard")
        raise RuntimeError("PRIVATE_DSN_PASSWORD")

    def forbidden(*args, **kwargs):
        pytest.fail("Odmowa guardu musi poprzedzać odczyt wejść i polecenia produktu")

    monkeypatch.setattr(
        trial, "_environment_helpers", lambda execute: {"require_approved_vm": guard}
    )
    monkeypatch.setattr(trial, "load_inputs", forbidden)
    monkeypatch.setattr(trial, "execute_phase", forbidden)
    assert (
        trial.main(
            [
                "--mode",
                "EXECUTE-VM-TRIAL",
                "--phase",
                "install",
                "--inputs",
                "/not-read",
                "--revision",
                REVISION,
                "--environment-id",
                ENVIRONMENT,
            ]
        )
        == 2
    )
    raw = capsys.readouterr().out
    assert events == ["guard"]
    assert "PRIVATE_DSN_PASSWORD" not in raw
    assert json.loads(raw)["checks"]["install"]["status"] == "BLOCKED"


def test_preflight_compares_host_even_when_installer_refuses(monkeypatch):
    observations = iter(["before", "after"])
    monkeypatch.setattr(trial, "snapshot", lambda: next(observations))
    monkeypatch.setattr(trial, "_installer", lambda *args, **kwargs: {"status": "FAIL"})
    with pytest.raises(trial.TrialError, match="PREFLIGHT_MUTATED_HOST"):
        trial.checked_preflight({}, REVISION)


def test_preflight_preserves_exact_refusal_without_promoting(monkeypatch):
    monkeypatch.setattr(trial, "snapshot", lambda: "same")
    monkeypatch.setattr(
        trial, "_installer", lambda *args, **kwargs: {"status": "FAIL", "exit_code": 24}
    )
    result = trial.checked_preflight({}, REVISION)
    assert result == {
        "status": "FAIL",
        "exit_code": 24,
        "before_sha256": "same",
        "after_sha256": "same",
        "readonly": True,
    }


def test_install_cannot_follow_failed_preflight(monkeypatch):
    monkeypatch.setattr(trial, "checked_preflight", lambda *args: {"status": "FAIL"})
    monkeypatch.setattr(
        trial, "_installer", lambda *args, **kwargs: pytest.fail("install forbidden")
    )
    assert trial.execute_phase("install", {}, REVISION, ENVIRONMENT)["status"] == "BLOCKED"


def test_install_verifies_published_state_and_identical_reinstall(monkeypatch):
    events = []

    def preflight(*args):
        events.append("preflight")
        return {"status": "PASS", "state": "fresh" if len(events) == 1 else "installed"}

    def installer(*args, install):
        assert install
        events.append("install")
        return {"status": "PASS", "state": "installed"}

    monkeypatch.setattr(trial, "checked_preflight", preflight)
    monkeypatch.setattr(trial, "_installer", installer)
    monkeypatch.setattr(trial, "snapshot", lambda: "same")
    result = trial.execute_phase("install", {}, REVISION, ENVIRONMENT)
    assert result["identical_reinstall"] == "PASS"
    assert events == ["preflight", "install", "preflight", "install"]


def test_identical_reinstall_detects_writes(monkeypatch):
    monkeypatch.setattr(
        trial, "checked_preflight", lambda *args: {"status": "PASS", "state": "installed"}
    )
    monkeypatch.setattr(
        trial, "_installer", lambda *args, **kwargs: {"status": "PASS", "state": "installed"}
    )
    observations = iter(["before", "after"])
    monkeypatch.setattr(trial, "snapshot", lambda: next(observations))
    with pytest.raises(trial.TrialError, match="IDENTICAL_REINSTALL_MUTATED_HOST"):
        trial.execute_phase("install", {}, REVISION, ENVIRONMENT)


@pytest.mark.parametrize("phase", ["systemd", "postgresql-ipc"])
def test_native_phases_require_verified_installed_release(monkeypatch, phase):
    monkeypatch.setattr(
        trial, "checked_preflight", lambda *args: {"status": "PASS", "state": "fresh"}
    )
    monkeypatch.setattr(trial, "_native_module", lambda: pytest.fail("native forbidden"))
    assert trial.execute_phase(phase, {}, REVISION, ENVIRONMENT)["status"] == "BLOCKED"


def test_runtime_verify_remains_readonly_and_uses_current_binding(monkeypatch):
    calls = []
    monkeypatch.setattr(trial, "snapshot", lambda: "same")

    def runtime(inputs, revision, environment_id, *, publish):
        calls.append((revision, environment_id, publish))
        return {"status": "PASS"}

    monkeypatch.setattr(trial, "_runtime", runtime)
    assert trial.execute_phase("runtime-verify", {}, REVISION, ENVIRONMENT)["readonly"]
    assert calls == [(REVISION, ENVIRONMENT, False)]


def test_command_has_no_shell_ambient_credentials_or_host_error_output(monkeypatch):
    def run(arguments, **options):
        assert arguments == ["/usr/bin/fixed", "literal argument"]
        assert "shell" not in options
        assert options["env"] == trial.COMMAND_ENV
        assert options["stderr"] == subprocess.DEVNULL
        assert options["stdin"] == subprocess.DEVNULL
        assert options["cwd"] == "/"
        return SimpleNamespace(returncode=0, stdout=b"{}")

    monkeypatch.setattr(trial.subprocess, "run", run)
    assert trial._command(["/usr/bin/fixed", "literal argument"]) == (0, b"{}")


def test_command_timeout_has_fixed_error(monkeypatch):
    def run(*args, **kwargs):
        raise subprocess.TimeoutExpired("SECRET", 1, output=b"SECRET")

    monkeypatch.setattr(trial.subprocess, "run", run)
    with pytest.raises(trial.TrialError, match="^COMMAND_UNAVAILABLE_OR_TIMEOUT$"):
        trial._command(["/usr/bin/fixed"])


@pytest.mark.parametrize("raw", [b'{"a":1,"a":2}', b"[]", b"null", b'"text"', b"{"])
def test_strict_json_refuses_ambiguous_or_non_object(raw):
    with pytest.raises(trial.TrialError):
        trial._json(raw)


def test_inputs_separate_public_authorities_and_refuse_unknown_fields(monkeypatch, tmp_path):
    value = {
        "runtime": {
            "runtime": "/runtime",
            "manifest": "/runtime.json",
            "signature": "/runtime.sig",
            "trusted_key": "/runtime.pub",
        },
        "application": {
            "bundle": "/bundle",
            "manifest": "/app.json",
            "signature": "/app.sig",
            "trusted_key": "/app.pub",
        },
    }
    monkeypatch.setattr(trial, "_protected", lambda *args, **kwargs: None)

    def read(path, limit):
        return json.dumps(value).encode() if limit == 65536 else b"x" * 32

    monkeypatch.setattr(trial, "_read", read)
    with pytest.raises(trial.TrialError, match="AUTHORITIES_NOT_SEPARATE"):
        trial.load_inputs(tmp_path / "inputs")
    value["unexpected"] = {}
    with pytest.raises(trial.TrialError, match="INPUT_SCHEMA_INVALID"):
        trial.load_inputs(tmp_path / "inputs")


def test_installer_legacy_success_is_not_v2_success(monkeypatch):
    monkeypatch.setattr(trial, "_helper", lambda name: Path("/trusted") / name)
    monkeypatch.setattr(
        trial,
        "_command",
        lambda args: (
            0,
            b'{"result":"READY","scope":"cryptohunter.freshness.substrate.v1","state":"fresh","production_qualification":"WITHHELD"}',
        ),
    )
    inputs = {
        "application": {
            name: "/input" for name in ("bundle", "manifest", "signature", "trusted_key")
        }
    }
    with pytest.raises(trial.TrialError, match="INSTALLER_REPLY_INVALID"):
        trial._installer(inputs, REVISION, install=False)


def test_preflight_detects_mutation_even_after_timeout(monkeypatch):
    observations = iter(["before", "after"])
    monkeypatch.setattr(trial, "snapshot", lambda: next(observations))

    def fail(*args, **kwargs):
        raise trial.TrialError("COMMAND_UNAVAILABLE_OR_TIMEOUT", status="BLOCKED")

    monkeypatch.setattr(trial, "_installer", fail)
    with pytest.raises(trial.TrialError, match="PREFLIGHT_MUTATED_HOST") as error:
        trial.checked_preflight({}, REVISION)
    assert error.value.status == "FAIL"


def test_verify_detects_mutation_after_invalid_reply(monkeypatch):
    observations = iter(["before", "after"])
    monkeypatch.setattr(trial, "snapshot", lambda: next(observations))

    def fail(*args, **kwargs):
        raise trial.TrialError("JSON_INVALID")

    monkeypatch.setattr(trial, "_runtime", fail)
    with pytest.raises(trial.TrialError, match="RUNTIME_VERIFY_MUTATED_HOST"):
        trial.execute_phase("runtime-verify", {}, REVISION, ENVIRONMENT)


def test_runtime_cli_uses_verified_helper_and_literal_inputs(monkeypatch):
    monkeypatch.setattr(trial, "_helper", lambda name: Path("/trusted") / name)
    inputs = {
        "runtime": {
            name: "/" + name for name in ("runtime", "manifest", "signature", "trusted_key")
        }
    }

    def command(arguments, *, timeout):
        assert timeout == 600
        assert arguments[:6] == [
            "/usr/bin/python3",
            "-I",
            "-s",
            "-B",
            "/trusted/freshness_runtime.py",
            "verify",
        ]
        assert arguments[-4:] == ["--environment-id", ENVIRONMENT, "--revision", REVISION]
        return 0, json.dumps({"environment_id": ENVIRONMENT, "manifest_sha256": "b" * 64}).encode()

    monkeypatch.setattr(trial, "_command", command)
    assert trial._runtime(inputs, REVISION, ENVIRONMENT, publish=False)["status"] == "PASS"
    monkeypatch.setattr(trial, "_command", lambda *args, **kwargs: (1, b"private error"))
    assert trial._runtime(inputs, REVISION, ENVIRONMENT, publish=False)["status"] == "FAIL"


def test_installer_records_only_validated_fields_and_exact_refusal(monkeypatch):
    monkeypatch.setattr(trial, "_helper", lambda name: Path("/trusted") / name)
    inputs = {
        "application": {
            name: "/input" for name in ("bundle", "manifest", "signature", "trusted_key")
        }
    }
    reply = {
        "result": "READY",
        "scope": "cryptohunter.freshness.substrate.v2",
        "state": "fresh",
        "production_qualification": "WITHHELD",
        "manifest_sha256": "c" * 64,
        "untrusted_secret": "DO_NOT_LOG",
    }
    monkeypatch.setattr(trial, "_command", lambda args: (0, json.dumps(reply).encode()))
    result = trial._installer(inputs, REVISION, install=False)
    assert result["status"] == "PASS" and "DO_NOT_LOG" not in str(result)
    monkeypatch.setattr(
        trial,
        "_command",
        lambda args: (24, b'{"result":"REFUSED","code":24,"reason":"DO_NOT_LOG"}'),
    )
    assert trial._installer(inputs, REVISION, install=False) == {
        "status": "FAIL",
        "reason": "INSTALLER_REFUSED",
        "exit_code": 24,
    }
    monkeypatch.setattr(trial, "_command", lambda args: (24, b'{"result":"REFUSED","code":11}'))
    with pytest.raises(trial.TrialError, match="INSTALLER_REPLY_INVALID"):
        trial._installer(inputs, REVISION, install=False)


def test_inputs_require_external_distinct_authorities(monkeypatch):
    value = {
        "runtime": {
            "runtime": "/runtime",
            "manifest": "/runtime.json",
            "signature": "/runtime.sig",
            "trusted_key": "/runtime.pub",
        },
        "application": {
            "bundle": "/bundle",
            "manifest": "/app.json",
            "signature": "/app.sig",
            "trusted_key": "/app.pub",
        },
    }
    monkeypatch.setattr(trial, "_protected", lambda *args, **kwargs: None)

    def read(path, limit):
        if limit == 65536:
            return json.dumps(value).encode()
        return (b"r" if path.name.startswith("runtime") else b"a") * 32

    monkeypatch.setattr(trial, "_read", read)
    assert trial.load_inputs(Path("/inputs")) == value
    value["application"]["trusted_key"] = "/bundle/app.pub"
    with pytest.raises(trial.TrialError, match="AUTHORITY_INSIDE_PAYLOAD"):
        trial.load_inputs(Path("/inputs"))


def test_protected_inputs_reject_user_writable_and_relative_paths(tmp_path):
    with pytest.raises(trial.TrialError, match="INPUT_PATH_INVALID"):
        trial._protected(Path("relative"))
    file = tmp_path / "not-root-owned"
    file.write_bytes(b"untrusted")
    with pytest.raises(trial.TrialError, match="INPUT_NOT_PROTECTED"):
        trial._protected(file)


def test_read_size_limit_and_stability(tmp_path, monkeypatch):
    file = tmp_path / "public"
    file.write_bytes(b"data")
    monkeypatch.setattr(trial, "_protected", lambda *args, **kwargs: None)
    assert trial._read(file, 4) == b"data"
    with pytest.raises(trial.TrialError, match="INPUT_TOO_LARGE"):
        trial._read(file, 3)


def test_installed_interpreter_is_not_returned_before_system_bootstrap(monkeypatch):
    monkeypatch.setattr(trial, "_helper", lambda name: Path("/trusted") / name)
    monkeypatch.setattr(trial, "_command", lambda *args, **kwargs: (1, b"refused"))
    monkeypatch.setattr(
        trial, "_read", lambda *args: pytest.fail("binding read before verification")
    )
    with pytest.raises(trial.TrialError, match="INSTALLED_RUNTIME_NOT_VERIFIED"):
        trial._installed_executable(REVISION, ENVIRONMENT)


def test_installed_interpreter_binds_revision_and_environment(monkeypatch):
    monkeypatch.setattr(trial, "_helper", lambda name: Path("/trusted") / name)
    monkeypatch.setattr(trial, "_command", lambda *args, **kwargs: (0, b"verified"))
    executable = f"/opt/cryptohunter/freshness-runtime/releases/{ENVIRONMENT}/bin/python3.12"
    binding = {"source_revision": REVISION, "environment_id": ENVIRONMENT, "executable": executable}
    monkeypatch.setattr(trial, "_read", lambda *args: json.dumps(binding).encode())
    assert trial._installed_executable(REVISION, ENVIRONMENT) == executable
    with pytest.raises(trial.TrialError, match="INSTALLED_RUNTIME_BINDING_MISMATCH"):
        trial._installed_executable("b" * 40, ENVIRONMENT)


def test_native_worker_cannot_run_in_unverified_interpreter(monkeypatch):
    args = SimpleNamespace(
        revision=REVISION, environment_id=ENVIRONMENT, phase="postgresql-ipc", native_worker=True
    )
    monkeypatch.setattr(trial, "_installed_executable", lambda *args: "/verified/bin/python3.12")
    monkeypatch.setattr(trial, "execute_phase", lambda *args: pytest.fail("wrong interpreter"))
    with pytest.raises(trial.TrialError, match="NATIVE_WORKER_INTERPRETER_NOT_VERIFIED"):
        trial._native_trial(args, {})


def test_native_worker_uses_signed_interpreter_and_preserves_blocked(monkeypatch):
    args = SimpleNamespace(
        revision=REVISION,
        environment_id=ENVIRONMENT,
        phase="postgresql-ipc",
        native_worker=False,
        approval=Path("/approval"),
        inputs=Path("/inputs"),
    )
    executable = "/verified/bin/python3.12"
    monkeypatch.setattr(trial, "_installed_executable", lambda *args: executable)

    def command(arguments, *, timeout):
        assert arguments[0] == executable
        assert "--native-worker" in arguments
        report = trial.new_report("EXECUTE-VM-TRIAL", REVISION, ENVIRONMENT)
        report["checks"]["postgresql-ipc"] = {"status": "BLOCKED", "diagnostic_exchange": "PASS"}
        return 1, json.dumps(report).encode()

    monkeypatch.setattr(trial, "_command", command)
    assert trial._native_trial(args, {})["status"] == "BLOCKED"


def test_successful_phase_never_marks_other_phases_or_native_matrix_pass(monkeypatch, capsys):
    monkeypatch.setattr(
        trial,
        "_environment_helpers",
        lambda execute: {"require_approved_vm": lambda *args, **kwargs: {"approval": "tested"}},
    )
    monkeypatch.setattr(trial, "load_inputs", lambda *args: {})
    monkeypatch.setattr(trial, "execute_phase", lambda *args: {"status": "PASS"})
    assert (
        trial.main(
            [
                "--mode",
                "EXECUTE-VM-TRIAL",
                "--phase",
                "preflight",
                "--inputs",
                "/inputs",
                "--revision",
                REVISION,
                "--environment-id",
                ENVIRONMENT,
            ]
        )
        == 0
    )
    report = json.loads(capsys.readouterr().out)
    assert report["checks"]["preflight"]["status"] == "PASS"
    assert report["checks"]["systemd"]["status"] == "NOT_EXECUTED"
    assert report["native_trial"] == "PENDING" and report["result"] == "NOT_EXECUTED"


def test_confirmed_security_violation_is_fail_not_blocked(monkeypatch, capsys):
    monkeypatch.setattr(
        trial,
        "_environment_helpers",
        lambda execute: {"require_approved_vm": lambda *args, **kwargs: {}},
    )
    monkeypatch.setattr(trial, "load_inputs", lambda *args: {})

    def mutated(*args):
        raise trial.TrialError("PREFLIGHT_MUTATED_HOST")

    monkeypatch.setattr(trial, "execute_phase", mutated)
    assert (
        trial.main(
            [
                "--mode",
                "EXECUTE-VM-TRIAL",
                "--phase",
                "preflight",
                "--inputs",
                "/inputs",
                "--revision",
                REVISION,
                "--environment-id",
                ENVIRONMENT,
            ]
        )
        == 2
    )
    report = json.loads(capsys.readouterr().out)
    assert report["result"] == "FAIL"
    assert report["checks"]["preflight"]["status"] == "FAIL"
