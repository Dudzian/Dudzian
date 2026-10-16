from __future__ import annotations

import base64
import json
import subprocess
from copy import deepcopy
from pathlib import Path, PureWindowsPath
from types import SimpleNamespace

import pytest

from deployment import windows_stage10_runner_preflight as preflight


def approved_config() -> dict:
    return {
        "schema_version": 1,
        "repository": "Dudzian/Dudzian",
        "hostname": "QUALIFIED-WIN",
        "machine_guid": "11111111-1111-4111-8111-111111111111",
        "hardware_uuid": "22222222-2222-4222-8222-222222222222",
        "runner_id": 123,
        "runner_name": "stage10-physical",
        "runner_service_name": "actions.runner.Dudzian.stage10-physical",
        "runner_root": r"C:\actions-runner",
        "runner_service_account": r"QUALIFIED-WIN\runner",
        "runner_group_id": 3,
        "trusted_workflows": [
            "Dudzian/Dudzian/.github/workflows/windows-stage10-manual.yml@refs/heads/main"
        ],
    }


def _complete_runner_tree() -> dict:
    root = PureWindowsPath(approved_config()["runner_root"])
    entries = []
    for index, path in enumerate(
        [*reversed(root.parents), root]
        + [
            root / child
            for child in ("bin", "_work", "_update", ".runner", r"bin\RunnerService.exe")
        ]
    ):
        directory = path.name not in {".runner", "RunnerService.exe"}
        entries.append(
            {
                "path": str(path),
                "final_path": str(path),
                "kind": "directory" if directory else "file",
                "attributes": 16 if directory else 32,
                "file_id": f"01234567:{index:016x}",
                "link_count": 1,
                "length": 0,
                "last_write_ticks": 123,
                "change_ticks": 123,
                "owner_sid": "S-1-5-32-544",
                "inheritance_disabled": False,
                "dacl_present": True,
                "dacl_canonical": True,
                "aces": [
                    {
                        "sid": "S-1-5-32-544",
                        "rights": 2032127,
                        "access_type": "Allow",
                        "inherited": False,
                        "inheritance_flags": 0,
                        "propagation_flags": 0,
                    }
                ],
            }
        )
    return {
        "schema_version": 1,
        "root": str(root),
        "complete": True,
        "errors": [],
        "before": entries,
        "after": deepcopy(entries),
    }


def observation() -> dict:
    config = approved_config()
    sid = "S-1-5-21-111-222-333-1001"
    executable = r"C:\actions-runner\bin\RunnerService.exe"
    workflow = config["trusted_workflows"][0]
    return {
        "host": {
            "os": "Windows",
            "architecture": "64-bit",
            "native_processor_architectures": [9],
            "is_64_bit_os": True,
            "is_64_bit_process": True,
            "process_architecture": "AMD64",
            "hostname": config["hostname"],
            "machine_guid": config["machine_guid"],
            "hardware_uuid": config["hardware_uuid"],
            "manufacturer": "Approved hardware vendor",
            "model": "Physical workstation",
            "runner_registration": {
                "id": config["runner_id"],
                "name": config["runner_name"],
                "github_url": "https://github.com/Dudzian",
            },
            "service": {
                "name": config["runner_service_name"],
                "start_mode": "Auto",
                "state": "Running",
                "process_id": 321,
                "path": f'"{executable}"',
                "process_path": executable,
                "account": config["runner_service_account"],
                "account_sid": sid,
            },
            "identity": {"sid": sid, "groups": ["S-1-5-32-545"], "elevated": False},
            "runner_root_acl": [
                {"sid": "S-1-5-32-544", "rights": 2032127, "access_type": "Allow"},
                {"sid": "S-1-5-32-545", "rights": 131209, "access_type": "Allow"},
            ],
        },
        "github": {
            # Hypothetical organization deployment, not current infrastructure:
            # Dudzian/Dudzian currently has owner.type=User and must be BLOCKED.
            "repository": {
                "full_name": config["repository"],
                "owner": {"login": "Dudzian", "type": "Organization"},
            },
            "runner": {
                "id": config["runner_id"],
                "name": config["runner_name"],
                "os": "windows",
                "status": "online",
                "labels": [{"name": label} for label in preflight.REQUIRED_LABELS],
            },
            "group": {
                "id": config["runner_group_id"],
                "visibility": "selected",
                "restricted_to_workflows": True,
                "selected_workflows": [workflow],
            },
            "group_runners": {"total_count": 1, "runners": [{"id": config["runner_id"]}]},
            "group_repositories": {
                "total_count": 1,
                "repositories": [{"full_name": config["repository"]}],
            },
            "workflows": {workflow: "on:\n  workflow_dispatch:\njobs: {}\n"},
        },
    }


def _check(report: dict, name: str) -> dict:
    return next(item for item in report["checks"] if item["name"] == name)


def _mock_live(monkeypatch: pytest.MonkeyPatch, observed: dict) -> None:
    # Synthetic unit test facts only. The public snapshot interface cannot use
    # these mocks to make any live/host-verification claim.
    monkeypatch.setattr(preflight, "_collect_host", lambda _config: observed["host"])
    monkeypatch.setattr(preflight, "_collect_github", lambda _config: observed["github"])


def test_consistent_live_reads_prepare_runner_without_stage10_pass(monkeypatch) -> None:
    _mock_live(monkeypatch, observation())
    report = preflight.run_runner_preflight(approved_config(), live=True)
    assert report["overall_status"] == "UNVERIFIED"
    assert _check(report, "runner_directory_acl")["status"] == "PASS"
    assert _check(report, "untrusted_pr_isolation")["status"] == "PASS"
    assert _check(report, "runner_directory_tree_acl")["status"] == "UNVERIFIED"
    assert report["stage10_qualification"] == "NOT_RUN"
    assert report["stage10_execution_authorized"] is False


def test_consistent_saved_snapshot_is_never_live_runner_proof() -> None:
    saved = observation()
    saved.update(live=True, verified=True)
    report = preflight.run_runner_preflight(approved_config(), snapshot=saved)
    assert report["overall_status"] == "UNVERIFIED"
    assert _check(report, "approved_configuration")["status"] == "PASS"
    assert all(
        item["status"] == "UNVERIFIED"
        for item in report["checks"]
        if item["name"] != "approved_configuration"
    )


def test_no_actual_runner_blocks_readiness_even_with_approval_booleans() -> None:
    report = preflight.run_runner_preflight(
        approved_config(),
        snapshot={"runner_exists": True, "labels_verified": True, "isolated": True},
    )
    assert report["overall_status"] == "BLOCKED"
    assert _check(report, "actual_runner")["status"] == "BLOCKED"
    assert _check(report, "github_registration_and_isolation")["status"] == "UNVERIFIED"


def test_live_mode_cannot_consume_snapshot(monkeypatch) -> None:
    def unexpected(_config):
        pytest.fail("snapshot/live combination must not collect anything")

    monkeypatch.setattr(preflight, "_collect_host", unexpected)
    report = preflight.run_runner_preflight(approved_config(), live=True, snapshot=observation())
    assert report["overall_status"] == "FAIL"
    assert _check(report, "live_collection")["status"] == "FAIL"


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("schema_version", True),
        ("schema_version", 2),
        ("repository", "Dudzian/Dudzian/../evil"),
        ("hostname", ""),
        ("runner_name", "unapproved; Write-Host bad"),
        ("runner_id", True),
        ("runner_group_id", 0),
        ("machine_guid", "broken"),
        ("hardware_uuid", "00000000-0000-0000-0000-000000000000"),
        ("runner_root", r"\\server\runner"),
        ("runner_root", r"C:\runner\..\other"),
        ("runner_root", r"C:relative"),
        ("runner_root", "relative"),
        ("runner_service_account", None),
        ("trusted_workflows", []),
        ("trusted_workflows", ["Dudzian/Dudzian/.github/workflows/x.yml@refs/pull/123/head"]),
        ("trusted_workflows", ["other/repo/.github/workflows/x.yml@refs/heads/main"]),
    ],
)
def test_invalid_operator_approval_blocks_without_host_reads(key, value) -> None:
    config = approved_config()
    config[key] = value
    report = preflight.run_runner_preflight(config)
    assert report["overall_status"] == "BLOCKED"
    assert len(report["checks"]) == 1


@pytest.mark.parametrize("config", [None, [], {}, {"untrusted_approval": True}])
def test_absent_or_unknown_config_fields_cannot_approve_host(config) -> None:
    assert preflight.run_runner_preflight(config)["overall_status"] == "BLOCKED"


@pytest.mark.parametrize(
    ("section", "key", "value", "check_name"),
    [
        ("host", "os", "Linux", "windows_x64"),
        ("host", "is_64_bit_os", False, "windows_x64"),
        ("host", "is_64_bit_process", False, "windows_x64"),
        ("host", "process_architecture", "x86", "windows_x64"),
        ("host", "process_architecture", "ARM64", "windows_x64"),
        ("host", "native_processor_architectures", [12], "windows_x64"),
        ("host", "native_processor_architectures", [], "windows_x64"),
        ("host", "native_processor_architectures", None, "windows_x64"),
        ("host", "hostname", "OTHER-HOST", "approved_physical_host"),
        ("host", "machine_guid", "other", "approved_physical_host"),
        ("host", "hardware_uuid", "other", "approved_physical_host"),
        ("host", "model", "Virtual Machine", "approved_physical_host"),
        ("host", "manufacturer", "", "approved_physical_host"),
        ("runner_registration", "id", 999, "local_runner_identity"),
        ("runner_registration", "github_url", "https://github.com/other", "local_runner_identity"),
        ("service", "start_mode", "Manual", "automatic_runner_service"),
        ("service", "state", "Stopped", "automatic_runner_service"),
        ("service", "process_id", 0, "automatic_runner_service"),
        ("service", "path", r'"C:\evil.exe"', "automatic_runner_service"),
        ("service", "process_path", r"C:\other\RunnerService.exe", "automatic_runner_service"),
        ("service", "account", r"OTHER\admin", "automatic_runner_service"),
        ("service", "account_sid", "S-1-5-18", "least_privilege_runner"),
        ("identity", "sid", "S-1-5-18", "least_privilege_runner"),
        ("identity", "elevated", True, "least_privilege_runner"),
        ("identity", "groups", ["S-1-5-32-544"], "least_privilege_runner"),
    ],
)
def test_host_mismatches_fail_closed(monkeypatch, section, key, value, check_name) -> None:
    observed = observation()
    target = observed["host"] if section == "host" else observed["host"][section]
    target[key] = value
    _mock_live(monkeypatch, observed)
    report = preflight.run_runner_preflight(approved_config(), live=True)
    assert report["overall_status"] == "FAIL"
    assert _check(report, check_name)["status"] == "FAIL"


@pytest.mark.parametrize(
    "acl", [None, [], [{"sid": "S-1-1-0", "rights": 2, "access_type": "Allow"}], ["bad"]]
)
def test_broad_runner_directory_write_or_missing_acl_is_rejected(monkeypatch, acl) -> None:
    observed = observation()
    observed["host"]["runner_root_acl"] = acl
    _mock_live(monkeypatch, observed)
    report = preflight.run_runner_preflight(approved_config(), live=True)
    assert _check(report, "runner_directory_acl")["status"] == "FAIL"


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("id", 999),
        ("name", "other-runner"),
        ("status", "offline"),
        ("os", "linux"),
        ("labels", [{"name": "self-hosted"}]),
        (
            "labels",
            [{"name": item} for item in preflight.REQUIRED_LABELS] + [{"name": "general-pr"}],
        ),
        ("labels", ["self-hosted", "Windows", "X64", "cryptohunter-stage10"]),
    ],
)
def test_independent_github_registration_mismatches_fail(monkeypatch, key, value) -> None:
    observed = observation()
    observed["github"]["runner"][key] = value
    _mock_live(monkeypatch, observed)
    report = preflight.run_runner_preflight(approved_config(), live=True)
    assert _check(report, "github_runner_registration")["status"] == "FAIL"


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("group", {"id": 3, "visibility": "all"}),
        ("group_runners", {"total_count": 2, "runners": [{"id": 123}]}),
        ("group_runners", {"total_count": 1, "runners": [{"id": 456}]}),
        ("group_repositories", {"total_count": 1, "repositories": [{"full_name": "other/repo"}]}),
        ("group_repositories", None),
        ("workflows", {}),
    ],
)
def test_untrusted_pr_isolation_requires_complete_independent_group_reads(
    monkeypatch, key, value
) -> None:
    observed = observation()
    observed["github"][key] = value
    _mock_live(monkeypatch, observed)
    report = preflight.run_runner_preflight(approved_config(), live=True)
    assert _check(report, "untrusted_pr_isolation")["status"] == "FAIL"


@pytest.mark.parametrize(
    "source",
    [
        "on:\n  pull_request:\n  workflow_dispatch:\n",
        "on:\n  workflow_call:\n",
        "on: [workflow_dispatch, push]\n",
        "on: [broken",
        "null",
        True,
    ],
)
def test_pr_or_malformed_workflow_sources_cannot_prove_isolation(monkeypatch, source) -> None:
    observed = observation()
    workflow = approved_config()["trusted_workflows"][0]
    observed["github"]["workflows"][workflow] = source
    _mock_live(monkeypatch, observed)
    report = preflight.run_runner_preflight(approved_config(), live=True)
    assert _check(report, "untrusted_pr_isolation")["status"] == "FAIL"


def test_malformed_mixed_type_workflow_allowlist_is_fail_closed(monkeypatch) -> None:
    observed = observation()
    observed["github"]["group"]["selected_workflows"] = [1, "workflow"]
    _mock_live(monkeypatch, observed)
    assert preflight.run_runner_preflight(approved_config(), live=True)["overall_status"] == "FAIL"


def test_live_collection_failures_are_blocked_and_do_not_expose_stderr(monkeypatch) -> None:
    def unavailable(_config):
        raise subprocess.CalledProcessError(1, "gh", stderr="sensitive-credential")

    monkeypatch.setattr(preflight, "_collect_host", unavailable)
    monkeypatch.setattr(preflight, "_collect_github", unavailable)
    report = preflight.run_runner_preflight(approved_config(), live=True)
    assert report["overall_status"] == "BLOCKED"
    assert "sensitive-credential" not in json.dumps(report)


def test_non_windows_live_host_collector_never_claims_windows(monkeypatch) -> None:
    monkeypatch.setattr(preflight, "os", SimpleNamespace(name="posix"))
    with pytest.raises(preflight.RunnerPreflightError, match="unavailable"):
        preflight._collect_host(approved_config())


def test_windows_collector_passes_values_as_environment_not_powershell_code(monkeypatch) -> None:
    config = approved_config()
    seen = {}

    def read(command, *, env):
        seen.update(command=command, env=env)
        return observation()["host"]

    monkeypatch.setattr(preflight, "os", SimpleNamespace(name="nt", environ={}))
    native_executable = r"C:\Windows\System32\WindowsPowerShell\v1.0\powershell.exe"
    monkeypatch.setattr(preflight, "native_powershell_path", lambda: native_executable)
    monkeypatch.setattr(preflight, "_read_json_command", read)
    assert preflight._collect_host(config) == observation()["host"]
    assert seen["command"][:3] == [native_executable, "-NoProfile", "-NonInteractive"]
    script = base64.b64decode(seen["command"][-1]).decode("utf-16-le")
    assert config["runner_root"] not in script
    assert seen["env"]["CRYPTOHUNTER_PREFLIGHT_ROOT"] == config["runner_root"]
    assert "Get-CimInstance Win32_Service" in script
    assert "Get-Acl" in script


def test_unavailable_trusted_powershell_never_executes_ambient_host_collector(monkeypatch) -> None:
    monkeypatch.setattr(preflight, "os", SimpleNamespace(name="nt", environ={}))

    def unavailable():
        raise OSError("Native PowerShell unavailable")

    monkeypatch.setattr(preflight, "native_powershell_path", unavailable)
    monkeypatch.setattr(preflight, "_read_json_command", lambda *_args, **_kwargs: pytest.fail())
    with pytest.raises(OSError, match="unavailable"):
        preflight._collect_host(approved_config())


def test_github_collector_only_gets_registration_policy_and_workflow_sources(monkeypatch) -> None:
    observed = observation()["github"]
    responses = [
        observed["repository"],
        observed["runner"],
        observed["group"],
        observed["group_runners"],
        observed["group_repositories"],
        {"encoding": "base64", "content": base64.b64encode(b"on: workflow_dispatch\n").decode()},
    ]
    calls = []

    def read(command):
        calls.append(command)
        return responses[len(calls) - 1]

    monkeypatch.setattr(preflight, "_read_json_command", read)
    result = preflight._collect_github(approved_config())
    assert len(calls) == 6
    assert calls[0][-1] == "repos/Dudzian/Dudzian"
    assert all(
        command[:6] == ["gh", "api", "--method", "GET", "--hostname", "github.com"]
        for command in calls
    )
    assert (
        result["workflows"][approved_config()["trusted_workflows"][0]] == "on: workflow_dispatch\n"
    )


def test_json_reader_limits_execution_and_strips_powershell_bom(monkeypatch) -> None:
    def run(command, **kwargs):
        assert command == ["readonly"]
        assert kwargs == {
            "check": True,
            "capture_output": True,
            "text": True,
            "timeout": 45,
            "env": None,
        }
        return SimpleNamespace(stdout='\ufeff{"ok": true}')

    monkeypatch.setattr(preflight.subprocess, "run", run)
    assert preflight._read_json_command(["readonly"]) == {"ok": True}


def test_cli_saves_missing_runner_and_malformed_json_as_nonpassing_report(tmp_path: Path) -> None:
    config = tmp_path / "approved.json"
    output = tmp_path / "report.json"
    config.write_text(json.dumps(approved_config()))
    assert preflight.main(["--config", str(config), "--output", str(output)]) == 1
    assert json.loads(output.read_text())["overall_status"] == "BLOCKED"
    output.unlink()
    config.write_text("{broken")
    assert preflight.main(["--config", str(config), "--output", str(output)]) == 1
    assert json.loads(output.read_text())["checks"][0]["status"] == "BLOCKED"


def test_cli_rehearsal_snapshot_cannot_issue_stage10_pass(tmp_path: Path) -> None:
    config = tmp_path / "approved.json"
    snapshot = tmp_path / "snapshot.json"
    output = tmp_path / "report.json"
    config.write_text(json.dumps(approved_config()))
    snapshot.write_text(json.dumps(observation()))
    assert (
        preflight.main(
            ["--config", str(config), "--snapshot", str(snapshot), "--output", str(output)]
        )
        == 1
    )
    report = json.loads(output.read_text())
    assert report["overall_status"] == "UNVERIFIED"
    assert report["stage10_qualification"] == "NOT_RUN"


@pytest.mark.parametrize("access_type", [[], {}, None])
def test_malformed_acl_access_type_cannot_crash_or_pass(monkeypatch, access_type) -> None:
    observed = observation()
    observed["host"]["runner_root_acl"][0]["access_type"] = access_type
    _mock_live(monkeypatch, observed)
    report = preflight.run_runner_preflight(approved_config(), live=True)
    assert _check(report, "runner_directory_acl")["status"] == "FAIL"


def test_unknown_domain_group_write_grant_fails_closed(monkeypatch) -> None:
    observed = observation()
    observed["host"]["runner_root_acl"].append(
        {"sid": "S-1-5-21-111-222-333-513", "rights": 2, "access_type": "Allow"}
    )
    _mock_live(monkeypatch, observed)
    assert (
        _check(
            preflight.run_runner_preflight(approved_config(), live=True), "runner_directory_acl"
        )["status"]
        == "FAIL"
    )


def test_cli_never_overwrites_approval_input_or_existing_reports(tmp_path: Path) -> None:
    config = tmp_path / "approved.json"
    original = json.dumps(approved_config())
    config.write_text(original)
    assert preflight.main(["--config", str(config), "--output", str(config)]) == 1
    assert config.read_text() == original
    output = tmp_path / "report.json"
    output.write_text("existing report")
    assert preflight.main(["--config", str(config), "--output", str(output)]) == 1
    assert output.read_text() == "existing report"


def test_report_cannot_write_via_symlink_or_into_runner_directory(tmp_path: Path) -> None:
    actual = tmp_path / "actual"
    actual.mkdir()
    link = tmp_path / "linked"
    link.symlink_to(actual, target_is_directory=True)
    with pytest.raises(preflight.RunnerPreflightError, match="symlinks"):
        preflight._write_new_report(link / "report.json", {}, approved_config())
    assert not (actual / "report.json").exists()
    with pytest.raises(preflight.RunnerPreflightError, match="runner or product"):
        preflight._write_new_report(Path(r"C:\actions-runner\report.json"), {}, approved_config())


@pytest.mark.parametrize("name", ["report.pem", "../report.json"])
def test_report_rejects_unapproved_extension_or_traversal(tmp_path: Path, name) -> None:
    with pytest.raises(preflight.RunnerPreflightError):
        preflight._write_new_report(tmp_path / name, {}, approved_config())


def test_product_directory_is_not_a_report_destination(monkeypatch) -> None:
    monkeypatch.setenv("ProgramData", r"C:\ProgramData")
    with pytest.raises(preflight.RunnerPreflightError, match="runner or product"):
        preflight._write_new_report(
            Path(r"C:\ProgramData\CryptoHunter\report.json"), {}, approved_config()
        )


@pytest.mark.parametrize("value", [[], {}])
def test_malformed_service_sid_cannot_crash_acl_evaluation(monkeypatch, value) -> None:
    observed = observation()
    observed["host"]["service"]["account_sid"] = value
    _mock_live(monkeypatch, observed)
    report = preflight.run_runner_preflight(approved_config(), live=True)
    assert _check(report, "least_privilege_runner")["status"] == "FAIL"


def test_untrusted_sid_with_only_delete_children_right_is_rejected(monkeypatch) -> None:
    observed = observation()
    observed["host"]["runner_root_acl"].append(
        {"sid": "S-1-5-21-111-222-333-513", "rights": 64, "access_type": "Allow"}
    )
    _mock_live(monkeypatch, observed)
    report = preflight.run_runner_preflight(approved_config(), live=True)
    assert report["overall_status"] == "FAIL"
    assert _check(report, "runner_directory_acl")["status"] == "FAIL"
    assert "DeleteSubdirectoriesAndFiles" in _check(report, "runner_directory_acl")["details"]


@pytest.mark.parametrize(
    "metadata",
    [
        {"full_name": "Dudzian/Dudzian", "owner": {"login": "Dudzian", "type": "User"}},
        {"full_name": "Dudzian/Dudzian", "owner": {"login": "Dudzian"}},
        {"full_name": "other/repo", "owner": {"login": "Dudzian", "type": "Organization"}},
        {"full_name": "Dudzian/Dudzian", "owner": {"login": "other", "type": "Organization"}},
        {"full_name": "Dudzian/Dudzian", "owner": {"login": "Dudzian", "type": []}},
        {"full_name": "Dudzian/Dudzian", "owner": None},
        [],
        None,
    ],
)
def test_non_org_or_malformed_owner_never_queries_org_endpoints(monkeypatch, metadata) -> None:
    calls = []

    def get(endpoint):
        calls.append(endpoint)
        assert endpoint == "repos/Dudzian/Dudzian"
        return metadata

    monkeypatch.setattr(preflight, "_github_get", get)
    result = preflight._collect_github(approved_config())
    assert calls == ["repos/Dudzian/Dudzian"]
    assert result == {"repository": metadata, "workflows": {}}
    observed = observation()
    # Even forged, otherwise perfect labels/group data cannot bypass ownership.
    observed["github"]["repository"] = metadata
    metadata_owner = metadata.get("owner") if isinstance(metadata, dict) else None
    if isinstance(metadata_owner, dict) and metadata_owner.get("type") == "User":
        observed["host"]["runner_registration"]["github_url"] = "https://github.com/Dudzian/Dudzian"
    original = deepcopy(observed)
    config = approved_config()
    original_config = deepcopy(config)
    _mock_live(monkeypatch, observed)
    report = preflight.run_runner_preflight(config, live=True)
    assert report["overall_status"] == "BLOCKED"
    assert _check(report, "github_repository_owner")["status"] == "BLOCKED"
    assert _check(report, "untrusted_pr_isolation")["status"] == "BLOCKED"
    assert _check(report, "github_runner_registration")["status"] == "UNVERIFIED"
    assert config == original_config and observed == original


def test_user_owner_has_actionable_infrastructure_diagnostic(monkeypatch) -> None:
    observed = observation()
    observed["github"]["repository"]["owner"]["type"] = "User"
    observed["host"]["runner_registration"]["github_url"] = "https://github.com/Dudzian/Dudzian"
    _mock_live(monkeypatch, observed)
    report = preflight.run_runner_preflight(approved_config(), live=True)
    reason = _check(report, "github_repository_owner")["details"]
    assert "User account" in reason
    assert "operator-approved infrastructure decision" in reason
    assert "Labels cannot replace isolation" in reason
    assert _check(report, "local_runner_identity")["status"] == "UNVERIFIED"
    assert report["overall_status"] == "BLOCKED"
    assert report["stage10_execution_authorized"] is False


def test_root_acl_cannot_establish_descendant_acl_safety(monkeypatch) -> None:
    observed = observation()
    observed["host"]["descendant_acls"] = [
        {"path": r"C:\actions-runner\bin", "untrusted_write": True, "inheritance_disabled": True}
    ]
    observed["host"]["tree_verified"] = True
    _mock_live(monkeypatch, observed)
    report = preflight.run_runner_preflight(approved_config(), live=True)
    assert _check(report, "runner_directory_acl")["status"] == "PASS"
    tree = _check(report, "runner_directory_tree_acl")
    assert tree["status"] == "UNVERIFIED"
    assert all(name in tree["details"] for name in ("bin", "_work", "_update", "PRE-01B"))
    assert report["overall_status"] == "UNVERIFIED"


def test_complete_independently_collected_live_tree_can_prepare_runner(monkeypatch) -> None:
    observed = observation()
    observed["host"]["runner_tree_acl"] = _complete_runner_tree()
    _mock_live(monkeypatch, observed)
    report = preflight.run_runner_preflight(approved_config(), live=True)
    assert _check(report, "runner_directory_tree_acl")["status"] == "PASS"
    assert report["overall_status"] == "PASS"
    assert report["stage10_qualification"] == "NOT_RUN"
    assert report["stage10_execution_authorized"] is False


def test_full_tree_snapshot_cannot_prove_live_readiness() -> None:
    observed = observation()
    observed["host"]["runner_tree_acl"] = _complete_runner_tree()
    observed.update(live=True, independently_collected=True)
    report = preflight.run_runner_preflight(approved_config(), snapshot=observed)
    assert _check(report, "runner_directory_tree_acl")["status"] == "UNVERIFIED"
    assert report["overall_status"] == "UNVERIFIED"


@pytest.mark.parametrize("target", ["bin", "_work", "_update", ".runner", r"bin\RunnerService.exe"])
def test_complete_tree_detects_unsafe_child_despite_safe_root(monkeypatch, target) -> None:
    observed = observation()
    tree = _complete_runner_tree()
    for phase in ("before", "after"):
        entry = next(
            item
            for item in tree[phase]
            if item["path"] == str(PureWindowsPath(tree["root"]) / target)
        )
        entry["aces"][0].update(sid="S-1-5-32-545", rights=2)
        entry["inheritance_disabled"] = True
    observed["host"]["runner_tree_acl"] = tree
    _mock_live(monkeypatch, observed)
    report = preflight.run_runner_preflight(approved_config(), live=True)
    assert _check(report, "runner_directory_acl")["status"] == "PASS"
    assert _check(report, "runner_directory_tree_acl")["status"] == "FAIL"
    assert report["overall_status"] == "FAIL"


def test_absent_optional_update_tree_is_safe_when_complete_inventory_agrees(monkeypatch) -> None:
    observed = observation()
    tree = _complete_runner_tree()
    for phase in ("before", "after"):
        tree[phase] = [
            item for item in tree[phase] if PureWindowsPath(item["path"]).name != "_update"
        ]
    observed["host"]["runner_tree_acl"] = tree
    _mock_live(monkeypatch, observed)
    assert (
        _check(
            preflight.run_runner_preflight(approved_config(), live=True),
            "runner_directory_tree_acl",
        )["status"]
        == "PASS"
    )


@pytest.mark.parametrize(
    "registration",
    [
        {"id": 999, "name": "stage10-physical", "github_url": "https://github.com/Dudzian/Dudzian"},
        {"id": 123, "name": "other-runner", "github_url": "https://github.com/Dudzian/Dudzian"},
        {
            "id": True,
            "name": "stage10-physical",
            "github_url": "https://github.com/Dudzian/Dudzian",
        },
        {"id": 123, "name": "stage10-physical", "github_url": "https://github.com/other/repo"},
        {},
        None,
    ],
)
def test_user_infrastructure_block_does_not_mask_bad_local_identity(
    monkeypatch, registration
) -> None:
    observed = observation()
    observed["github"]["repository"]["owner"]["type"] = "User"
    observed["host"]["runner_registration"] = registration
    _mock_live(monkeypatch, observed)
    report = preflight.run_runner_preflight(approved_config(), live=True)
    assert _check(report, "github_repository_owner")["status"] == "BLOCKED"
    assert _check(report, "local_runner_identity")["status"] == "FAIL"
    assert report["overall_status"] == "FAIL"
