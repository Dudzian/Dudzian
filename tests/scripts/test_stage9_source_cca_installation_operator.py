"""Unprivileged orchestration tests; never physical installation evidence."""

from __future__ import annotations

import argparse
from contextlib import nullcontext
import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

from scripts import qualify_stage9_source_cca_installation as operator
from scripts import qualify_stage41b_signer_isolation as stage41b

RUN_ID = "012345abcdef"
COMMIT = "a" * 40


@pytest.fixture
def posix_report_filesystem():
    if sys.platform == "win32":
        pytest.skip("POSIX descriptor-relative report reservation and no-follow checks")


@pytest.fixture
def mock_host_signals(monkeypatch):
    """Orchestration tests must not depend on native signal APIs or privileges."""
    monkeypatch.setattr(operator, "sys", SimpleNamespace(platform="linux"))
    monkeypatch.setattr(
        operator,
        "signal",
        SimpleNamespace(
            SIGINT=2,
            SIGTERM=15,
            SIGHUP=1,
            getsignal=lambda number: None,
            signal=lambda number, handler: None,
        ),
    )
    monkeypatch.setattr(stage41b, "_defer_interrupts", nullcontext)


@pytest.fixture
def safe_operator(monkeypatch):
    monkeypatch.setattr(operator, "_commit", lambda **kwargs: COMMIT)

    def forbidden(*args, **kwargs):
        pytest.fail("a read-only mode attempted physical execution")

    monkeypatch.setattr(operator, "PhysicalRun", forbidden)
    return operator


def test_check_is_read_only_and_preserves_no_go(safe_operator, monkeypatch, capsys):
    resources = operator.plan(RUN_ID)
    observed = []
    monkeypatch.setattr(
        operator,
        "preflight",
        lambda value, **kwargs: observed.append((value, kwargs)) or {"ready": True},
    )
    assert safe_operator.main(["--check", "--run-id", RUN_ID]) == 0
    report = json.loads(capsys.readouterr().out)
    assert observed == [(resources, {"administrative": False})]
    assert report["physical_result"] == "NOT_EXECUTED"
    assert report["physical_test_status"] == "PHYSICAL_TEST_PENDING"
    assert report["qualification"] == "NOT_QUALIFIED"
    assert report["production_emission"] == report["status"] == "BLOCKED"
    assert report["commit"] == COMMIT and report["test_only"] == "TEST_ONLY"


def test_preflight_refusal_has_no_physical_side_effect(safe_operator, monkeypatch, capsys):
    monkeypatch.setattr(
        operator,
        "preflight",
        lambda value, **kwargs: {"ready": False, "blockers": ["SYSTEMD"]},
    )
    assert safe_operator.main(["--check", "--run-id", RUN_ID]) == 2
    assert json.loads(capsys.readouterr().out)["physical_result"] == "NOT_EXECUTED"


def test_inspector_reads_exact_plan_and_never_installs(safe_operator, monkeypatch, capsys):
    observed = []
    monkeypatch.setattr(
        operator,
        "inspect_cleanup",
        lambda value, **kwargs: observed.append((value, kwargs))
        or {"scope_clear": True, "clean": False, "full_confirmation": False},
    )
    assert safe_operator.main(["--inspect-cleanup", "--run-id", RUN_ID, "--uid-base", "61000"]) == 0
    assert observed == [(operator.plan(RUN_ID, 61000), {"administrative": False})]
    report = json.loads(capsys.readouterr().out)
    assert report["cleanup"]["scope_clear"] is True
    assert report["cleanup"]["clean"] is report["cleanup"]["full_confirmation"] is False


def test_rootless_inspection_does_not_upgrade_when_invoker_is_root(
    safe_operator, monkeypatch, capsys
):
    monkeypatch.setattr(operator, "os", SimpleNamespace(geteuid=lambda: 0))
    calls = []

    def inspect(value, *, administrative):
        calls.append(administrative)
        return {"scope_clear": True, "clean": False, "full_confirmation": False}

    monkeypatch.setattr(operator, "inspect_cleanup", inspect)
    assert safe_operator.main(["--inspect-cleanup", "--run-id", RUN_ID]) == 0
    report = json.loads(capsys.readouterr().out)
    assert calls == [False]
    assert report["cleanup"]["clean"] is False
    assert report["physical_test_status"] == "PHYSICAL_TEST_PENDING"


def test_administrative_inspection_needs_explicit_root_without_escalation(
    safe_operator, monkeypatch, capsys
):
    monkeypatch.setattr(operator, "sys", SimpleNamespace(platform="linux"))
    monkeypatch.setattr(operator, "os", SimpleNamespace(geteuid=lambda: 1000))
    monkeypatch.setattr(
        operator,
        "inspect_cleanup",
        lambda *args, **kwargs: pytest.fail("non-root administrative scan must not run"),
    )
    assert safe_operator.main(["--inspect-cleanup-admin", "--run-id", RUN_ID]) == 2
    report = json.loads(capsys.readouterr().out)
    assert report["refusals"] == ["ADMINISTRATIVE_INSPECTION_REQUIRES_ROOT"]
    assert report["cleanup"]["clean"] is False
    assert report["status"] == report["production_emission"] == "BLOCKED"


@pytest.mark.parametrize(
    "complete,clean,expected",
    [(True, True, 0), (False, True, 2), (True, False, 2), (False, False, 2)],
)
def test_explicit_administrative_inspection_requires_full_evidence_and_stays_read_only(
    safe_operator, monkeypatch, capsys, complete, clean, expected
):
    monkeypatch.setattr(operator, "sys", SimpleNamespace(platform="linux"))
    monkeypatch.setattr(operator, "os", SimpleNamespace(geteuid=lambda: 0))
    monkeypatch.setattr(
        operator,
        "_authorize",
        lambda *args: pytest.fail("read-only inspection must not authorize execution"),
    )
    observed = []

    def inspect(resources, *, administrative):
        observed.append((resources, administrative))
        return {"scope_clear": clean, "clean": clean, "full_confirmation": complete}

    monkeypatch.setattr(operator, "inspect_cleanup", inspect)
    assert safe_operator.main(["--inspect-cleanup-admin", "--run-id", RUN_ID]) == expected
    report = json.loads(capsys.readouterr().out)
    assert observed == [(operator.plan(RUN_ID), True)]
    assert report["mode"] == "inspect_cleanup_admin"
    assert report["physical_result"] == "NOT_EXECUTED"
    assert report["physical_test_status"] == "PHYSICAL_TEST_PENDING"
    assert report["status"] == report["production_emission"] == "BLOCKED"
    assert report["qualification"] == "NOT_QUALIFIED"


@pytest.mark.parametrize("scope_clear", [False, None])
def test_rootless_uncertain_inspection_never_returns_success(
    safe_operator, monkeypatch, capsys, scope_clear
):
    monkeypatch.setattr(
        operator,
        "inspect_cleanup",
        lambda *args, **kwargs: {
            "scope_clear": scope_clear,
            "clean": False,
            "full_confirmation": False,
            "state": "UNCERTAIN",
        },
    )
    assert safe_operator.main(["--inspect-cleanup", "--run-id", RUN_ID]) == 2
    assert json.loads(capsys.readouterr().out)["cleanup"]["clean"] is False


def test_prepare_prints_only_digest_inventory(safe_operator, monkeypatch, capsys, tmp_path):
    package = {"package_sha256": "b" * 64, "files": {"root_source": b"ELF-not-report-data"}}
    observed = []
    monkeypatch.setattr(operator, "build_package", lambda *args: observed.append(args) or package)
    assert (
        safe_operator.main(["--prepare", "--run-id", RUN_ID, "--package", str(tmp_path / "new")])
        == 0
    )
    output = capsys.readouterr().out
    report = json.loads(output)
    assert "ELF-not-report-data" not in output
    assert report["package"]["package_sha256"] == "b" * 64
    assert observed[0] == (tmp_path / "new", RUN_ID, 62000, COMMIT)


@pytest.mark.parametrize("mode", ["--prepare", "--execute"])
def test_package_required(mode, safe_operator, capsys):
    assert safe_operator.main([mode, "--run-id", RUN_ID]) == 2
    assert "REQUIRED" in json.loads(capsys.readouterr().out)["refusals"][0]


def test_invalid_namespace_fails_before_preflight(safe_operator, monkeypatch, capsys):
    monkeypatch.setattr(
        operator, "preflight", lambda value, **kwargs: pytest.fail("invalid plan reached host")
    )
    assert safe_operator.main(["--check", "--run-id", "../../other"]) == 2
    assert json.loads(capsys.readouterr().out)["refusals"] == ["RUN_ID_INVALID"]


def test_native_exception_is_redacted(safe_operator, monkeypatch, capsys):
    def fail(value, **kwargs):
        raise OSError("private host environment must never leave boundary")

    monkeypatch.setattr(operator, "preflight", fail)
    assert safe_operator.main(["--check", "--run-id", RUN_ID]) == 2
    output = capsys.readouterr().out
    assert "private host" not in output
    assert json.loads(output)["refusals"] == ["OPERATOR_OPERATION_BLOCKED"]


def test_report_and_section_hashes_match(
    safe_operator, monkeypatch, capsys, tmp_path, posix_report_filesystem
):
    monkeypatch.setattr(
        operator,
        "inspect_cleanup",
        lambda value, **kwargs: {"clean": False, "blockers": ["REMAINS"]},
    )
    destination = tmp_path / "report.json"
    assert (
        safe_operator.main(["--inspect-cleanup", "--run-id", RUN_ID, "--report", str(destination)])
        == 2
    )
    raw = destination.read_bytes()
    assert (
        destination.with_suffix(".json.sha256").read_text().strip()
        == hashlib.sha256(raw).hexdigest()
    )
    report = json.loads(raw)
    assert report == json.loads(capsys.readouterr().out)
    for section, digest in report["evidence_sha256"].items():
        assert digest == hashlib.sha256(operator._bytes(report[section])).hexdigest()
    assert destination.stat().st_mode & 0o777 == 0o600


def test_report_never_overwrites_existing_file(tmp_path, posix_report_filesystem):
    path = tmp_path / "report.json"
    path.write_text("owned by someone else")
    with pytest.raises(operator.HarnessRefusal, match="REPORT_WRITE_BLOCKED"):
        operator._write_report(path, {})
    assert path.read_text() == "owned by someone else"


def test_report_never_overwrites_existing_checksum(tmp_path, posix_report_filesystem):
    path = tmp_path / "report.json"
    checksum = path.with_name(path.name + ".sha256")
    checksum.write_text("another report checksum")
    with pytest.raises(operator.HarnessRefusal, match="REPORT_WRITE_BLOCKED"):
        operator._write_report(path, {"status": "BLOCKED"})
    assert checksum.read_text() == "another report checksum"
    assert not path.exists() or path.read_bytes() == b""


def test_pinned_report_parent_replacement_refuses_without_writing_replacement(
    tmp_path, posix_report_filesystem
):
    parent = tmp_path / "reports"
    parent.mkdir()
    path = parent / "report.json"
    destination = operator._ReportDestination(path)
    original = tmp_path / "original-reports"
    try:
        parent.rename(original)
        parent.mkdir()
        path.write_text("unowned replacement report")
        checksum = path.with_name(path.name + ".sha256")
        checksum.write_text("unowned replacement checksum")
        with pytest.raises(operator.HarnessRefusal, match="REPORT_PATH_UNSAFE"):
            destination.write({"status": "BLOCKED"})
        assert path.read_text() == "unowned replacement report"
        assert checksum.read_text() == "unowned replacement checksum"
        assert (original / "report.json").read_bytes() == b""
        assert (original / "report.json.sha256").read_bytes() == b""
    finally:
        destination.close()


def test_report_rejects_symlink_ancestor(tmp_path, posix_report_filesystem):
    real = tmp_path / "real"
    real.mkdir()
    alias = tmp_path / "alias"
    alias.symlink_to(real, target_is_directory=True)
    with pytest.raises(operator.HarnessRefusal, match="REPORT_PATH_UNSAFE"):
        operator._write_report(alias / "report.json", {})
    assert not (real / "report.json").exists()


def _args(**updates):
    values = dict(
        run_id=RUN_ID,
        authorize_test_only=RUN_ID,
        physical_host="Dell-Latitude-5490",
        package_sha256="b" * 64,
        package=Path("package"),
        uid_base=62000,
    )
    return argparse.Namespace(**(values | updates))


@pytest.mark.parametrize(
    "updates,code",
    [
        ({"authorize_test_only": None}, "EXPLICIT_PHYSICAL_TEST_AUTHORIZATION_REQUIRED"),
        ({"authorize_test_only": "other"}, "EXPLICIT_PHYSICAL_TEST_AUTHORIZATION_REQUIRED"),
        ({"physical_host": None}, "EXPLICIT_PHYSICAL_TEST_AUTHORIZATION_REQUIRED"),
        ({"package_sha256": None}, "REVIEWED_PACKAGE_SHA256_REQUIRED"),
    ],
)
def test_execute_needs_explicit_authorization(updates, code):
    with pytest.raises(operator.HarnessRefusal, match=code):
        operator._authorize(_args(**updates))


@pytest.mark.parametrize(
    "name", ["CI", "GITHUB_ACTIONS", "CODEX_THREAD_ID", "CODEX_SANDBOX", "CODEX_CLOUD"]
)
def test_execute_forbidden_in_automated_environments(monkeypatch, name):
    monkeypatch.setenv(name, "1")
    with pytest.raises(operator.HarnessRefusal, match="CI_OR_CODEX_EXECUTION_FORBIDDEN"):
        operator._authorize(_args())


def test_execute_rejects_different_package_commit_before_host_mutation(monkeypatch):
    monkeypatch.setattr(operator, "_authorize", lambda args: None)
    monkeypatch.setattr(
        operator,
        "validate_package",
        lambda *args: {"plan": operator.plan(RUN_ID), "commit": "c" * 40},
    )
    monkeypatch.setattr(
        operator, "preflight", lambda value, **kwargs: pytest.fail("wrong commit reached host")
    )
    with pytest.raises(operator.HarnessRefusal, match="PACKAGE_COMMIT_MISMATCH"):
        operator._execute(_args(), {"commit": COMMIT})


def test_execute_failure_always_cleans_and_independently_inspects(monkeypatch, mock_host_signals):
    calls = []
    resources = operator.plan(RUN_ID)
    monkeypatch.setattr(operator, "_authorize", lambda args: None)
    monkeypatch.setattr(
        operator,
        "validate_package",
        lambda *args: {"plan": resources, "commit": COMMIT, "package_sha256": "b" * 64},
    )
    monkeypatch.setattr(
        operator,
        "preflight",
        lambda value, **kwargs: calls.append(("preflight", kwargs)) or {"ready": True},
    )
    monkeypatch.setattr(
        operator,
        "inspect_cleanup",
        lambda value, **kwargs: calls.append(("inspect", kwargs))
        or {"clean": True, "full_confirmation": True},
    )

    class Run:
        def __init__(self, package):
            pass

        def __enter__(self):
            calls.append("enter")
            return self

        def start(self):
            raise operator.HarnessRefusal("SIMULATED_START_FAILURE")

        def __exit__(self, *args):
            calls.append("exit")

        def cleanup(self):
            calls.append("cleanup")
            return {"clean": True}

    monkeypatch.setattr(operator, "PhysicalRun", Run)
    report = {"commit": COMMIT}
    with pytest.raises(operator.HarnessRefusal, match="SIMULATED_START_FAILURE"):
        operator._execute(_args(), report)
    assert calls == [
        ("preflight", {"administrative": True}),
        "enter",
        "exit",
        "cleanup",
        ("inspect", {"administrative": True}),
    ]
    assert report["cleanup"]["clean"] is True


@pytest.mark.parametrize("cleanup_failure", [None, "refused", "raised", "limited"])
def test_full_execute_keeps_exit_two_and_reserves_report_before_host_changes(
    monkeypatch, mock_host_signals, capsys, tmp_path, cleanup_failure, posix_report_filesystem
):
    """A synthetic full pass exercises CLI orchestration, never native execution."""
    from deployment import stage9_source_cca_test_controls as controls

    calls = []
    resources = operator.plan(RUN_ID)
    report_path = tmp_path / "report.json"
    checksum_path = report_path.with_name(report_path.name + ".sha256")
    matched = {
        "installation_checks": "MATCHED",
        "role_checks": {
            role: {"checks": "MATCHED"} for role in ("root_source", "history_source", "cca")
        },
        "status": "BLOCKED",
        "qualification": "NOT_QUALIFIED",
        "production_emission": "BLOCKED",
        "observed_failures": [],
    }
    monkeypatch.setattr(operator, "_commit", lambda **kwargs: COMMIT)
    monkeypatch.setattr(operator, "_authorize", lambda args: None)
    monkeypatch.setattr(
        operator,
        "validate_package",
        lambda *args: {"plan": resources, "commit": COMMIT, "package_sha256": "b" * 64},
    )
    monkeypatch.setattr(
        operator,
        "preflight",
        lambda value, **kwargs: calls.append(("preflight", kwargs)) or {"ready": True},
    )

    def inspect_gate(manifest, *, approval_pin_path):
        assert manifest == "/synthetic/manifest.json"
        assert approval_pin_path == "/synthetic/TEST_ONLY-pin.json"
        calls.append("gate")
        return matched

    monkeypatch.setattr(operator, "inspect_installation", inspect_gate)
    monkeypatch.setattr(
        operator,
        "inspect_cleanup",
        lambda value, **kwargs: calls.append(("independent-inspection", kwargs))
        or {"clean": True, "full_confirmation": cleanup_failure != "limited"},
    )
    results = [{"name": "synthetic-denial", "result": "REFUSED"}]
    monkeypatch.setattr(controls, "run_controls", lambda run: calls.append("controls") or results)

    class Run:
        manifest_path = Path("/synthetic/manifest.json")
        pin_path = Path("/synthetic/TEST_ONLY-pin.json")

        def __init__(self, package):
            assert report_path.read_bytes() == checksum_path.read_bytes() == b""
            calls.append("reserved-before-resource-creation")

        def __enter__(self):
            calls.append("enter")
            return self

        def start(self):
            calls.append("start")

        def __exit__(self, *args):
            calls.append("exit")

        def cleanup(self):
            calls.append("cleanup")
            if cleanup_failure == "raised":
                raise operator.HarnessRefusal("SIMULATED_CLEANUP_FAILURE")
            return {"clean": cleanup_failure in {None, "limited"}}

    monkeypatch.setattr(operator, "PhysicalRun", Run)
    assert (
        operator.main(
            [
                "--execute",
                "--run-id",
                RUN_ID,
                "--package",
                str(tmp_path / "synthetic-package"),
                "--package-sha256",
                "b" * 64,
                "--authorize-test-only",
                RUN_ID,
                "--physical-host",
                "Dell-Latitude-5490",
                "--report",
                str(report_path),
            ]
        )
        == 2
    )
    report = json.loads(capsys.readouterr().out)
    assert report == json.loads(report_path.read_bytes())
    assert checksum_path.read_text().strip() == hashlib.sha256(report_path.read_bytes()).hexdigest()
    assert report["status"] == report["production_emission"] == "BLOCKED"
    assert report["qualification"] == "NOT_QUALIFIED"
    assert report["stage9"] == "IN_PROGRESS / PRODUCTION_NO_GO"
    assert report["test_only"] == "TEST_ONLY"
    assert report["gate"] == matched and report["checks"] == results
    assert calls == [
        ("preflight", {"administrative": True}),
        "reserved-before-resource-creation",
        "enter",
        "start",
        "gate",
        "controls",
        "exit",
        "cleanup",
        ("independent-inspection", {"administrative": True}),
    ]
    assert report["cleanup"]["independent"]["clean"] is True
    assert report["cleanup"]["clean"] is (cleanup_failure is None)
    if cleanup_failure is None:
        assert report["physical_result"] == "TEST_INSTALLATION_MATCHED"
        assert report["refusals"] == []
    else:
        assert report["physical_result"] == "BLOCKED"
        assert "CLEANUP_BLOCKED" in report["refusals"]


def test_match_requires_all_roles_and_no_production_go():
    matched = {
        "installation_checks": "MATCHED",
        "role_checks": {
            role: {"checks": "MATCHED"} for role in ("root_source", "history_source", "cca")
        },
        "status": "BLOCKED",
        "qualification": "NOT_QUALIFIED",
        "production_emission": "BLOCKED",
        "observed_failures": [],
    }
    assert operator._matched(matched)
    assert not operator._matched(matched | {"production_emission": "GO"})
    assert not operator._matched(matched | {"role_checks": {}})


def test_privileged_operator_checks_loaded_sources_and_interpreter(monkeypatch):
    from scripts import qualify_stage41b_signer_isolation as stage41b

    observed = []
    monkeypatch.setattr(
        stage41b, "_check_protected", lambda path, *, directory: observed.append((path, directory))
    )
    operator._trusted_operator()
    assert (Path(operator.__file__), False) in observed
    assert (Path(operator.sys.executable).resolve(), False) in observed
    assert (Path(operator.__file__).resolve().parents[1] / ".git", True) in observed


def test_unprotected_operator_code_is_refused_without_native_error_text(monkeypatch):
    from scripts import qualify_stage41b_signer_isolation as stage41b

    def unprotected(*args, **kwargs):
        raise PermissionError("private code location")

    monkeypatch.setattr(stage41b, "_check_protected", unprotected)
    with pytest.raises(operator.HarnessRefusal, match="^PROTECTED_REVIEWED_OPERATOR_REQUIRED$"):
        operator._trusted_operator()
