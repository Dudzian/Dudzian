"""Syntetyczne testy diagnostyki odmowy; nie dowodzą fizycznego lifecycle Windows."""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any

import pytest

from deployment import windows_stage10_proof_validation as validation


EXPECTED_PROOFS = {
    "WINDOWS_REBOOT_RECOVERY": (
        "boot_changed",
        "current_revision_runtime_bound",
        "service_autostarted",
        "readiness_contract_after_reboot",
    ),
    "WINDOWS_LONG_RUNNING_LIFECYCLE": (
        "duration_at_least_24h",
        "current_revision_runtime_bound",
        "same_service_instance",
        "readiness_contract_before_after",
    ),
}
PROOF_CASES = [(item, proof) for item, proofs in EXPECTED_PROOFS.items() for proof in proofs]


def assert_complete_refusal(report: dict[str, Any]) -> None:
    assert set(report) == {
        "schema_version",
        "report_kind",
        "status",
        "reason_code",
        "generated_at_utc",
        "native_qualification_performed",
        "results",
    }
    assert type(report["schema_version"]) is int
    assert report["schema_version"] == 1
    assert report["report_kind"] == "windows-stage10-proof-blockers"
    assert report["status"] == "BLOCKED"
    assert report["reason_code"] == validation.UNAVAILABLE
    assert report["native_qualification_performed"] is False
    assert report["generated_at_utc"].endswith("Z")
    timestamp = datetime.fromisoformat(report["generated_at_utc"].replace("Z", "+00:00"))
    assert timestamp.tzinfo == timezone.utc
    assert set(report["results"]) == set(EXPECTED_PROOFS)
    for item, proofs in EXPECTED_PROOFS.items():
        result = report["results"][item]
        assert set(result) == {"status", "proofs"}
        assert result["status"] == "BLOCKED"
        assert tuple(result["proofs"]) == proofs
        for proof in proofs:
            observed = result["proofs"][proof]
            assert set(observed) == {"status", "reason"}
            assert observed["status"] == "BLOCKED"
            assert observed["reason"] == validation.PROOF_REQUIREMENTS[item][proof]
            assert isinstance(observed["reason"], str) and observed["reason"].strip()


def test_report_and_requirements_match_the_exact_lifecycle_contract() -> None:
    contract_path = Path(validation.__file__).with_name("windows_stage10_lifecycle_contract.json")
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    assert contract["results"] == {item: list(proofs) for item, proofs in EXPECTED_PROOFS.items()}
    assert {
        item: tuple(proofs) for item, proofs in validation.PROOF_REQUIREMENTS.items()
    } == EXPECTED_PROOFS
    assert_complete_refusal(validation.blocked_report())


@pytest.mark.parametrize("item,proof", PROOF_CASES)
def test_each_required_proof_has_an_explicit_individual_refusal(item: str, proof: str) -> None:
    report = validation.blocked_report()
    observed = report["results"][item]["proofs"][proof]
    assert observed["status"] == "BLOCKED"
    assert observed["reason"] == validation.PROOF_REQUIREMENTS[item][proof]
    assert observed["reason"].strip()
    with pytest.raises(validation.Stage10ProofUnavailable, match=validation.UNAVAILABLE):
        validation.require_native_proof_authority()


@pytest.mark.parametrize("with_environment", [False, True])
def test_ci_and_operator_environment_cannot_authorize_any_proof(
    monkeypatch: pytest.MonkeyPatch, with_environment: bool
) -> None:
    environment = {
        "GITHUB_SHA": "a" * 40,
        "GITHUB_RUN_ID": "42",
        "GITHUB_RUN_ATTEMPT": "1",
        "GITHUB_SERVER_URL": "https://github.com",
        "RUNNER_OS": "Windows",
        "RUNNER_ARCH": "X64",
        "RUNNER_NAME": "synthetic-host",
        "WINDOWS_STAGE10_PROOF_VALIDATION_READY": "YES",
        "STAGE10_PRODUCTION_READY": "YES",
    }
    for name, value in environment.items():
        if with_environment:
            monkeypatch.setenv(name, value)
        else:
            monkeypatch.delenv(name, raising=False)
    assert_complete_refusal(validation.blocked_report())
    with pytest.raises(validation.Stage10ProofUnavailable, match=validation.UNAVAILABLE):
        validation.require_native_proof_authority()


def test_main_writes_complete_blocked_diagnostics_and_returns_refusal(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    output = tmp_path / "diagnostics.json"
    assert validation.main(["--output", str(output)]) == 1
    assert_complete_refusal(json.loads(output.read_text(encoding="utf-8")))
    assert validation.UNAVAILABLE in capsys.readouterr().err
    assert tuple(tmp_path.iterdir()) == (output,)


def test_write_removes_stale_pass_and_publishes_only_the_complete_report(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = tmp_path / "diagnostics.json"
    output.write_text('{"status":"PASS"}', encoding="utf-8")
    replace = validation.os.replace
    calls: list[Path] = []

    def inspect_then_replace(source: Path, destination: Path) -> None:
        assert destination == output
        assert not output.exists()
        assert source.parent == output.parent
        assert_complete_refusal(json.loads(source.read_text(encoding="utf-8")))
        calls.append(source)
        replace(source, destination)

    monkeypatch.setattr(validation.os, "replace", inspect_then_replace)
    validation.write_blocked_report(output)
    assert len(calls) == 1
    assert not calls[0].exists()
    assert_complete_refusal(json.loads(output.read_text(encoding="utf-8")))
    assert tuple(tmp_path.iterdir()) == (output,)


@pytest.mark.parametrize("failure_at", ["open", "serialize", "fsync", "replace"])
def test_write_failure_removes_stale_pass_and_leaves_no_partial_or_temporary_report(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failure_at: str
) -> None:
    output = tmp_path / "diagnostics.json"
    output.write_text('{"status":"PASS"}', encoding="utf-8")

    def fail(*_args: Any, **_kwargs: Any) -> None:
        assert not output.exists()
        raise OSError("syntetyczna odmowa zapisu")

    target, name = {
        "open": (validation.tempfile, "NamedTemporaryFile"),
        "serialize": (validation.json, "dump"),
        "fsync": (validation.os, "fsync"),
        "replace": (validation.os, "replace"),
    }[failure_at]
    monkeypatch.setattr(target, name, fail)
    assert validation.main(["--output", str(output)]) == 1
    assert not output.exists()
    assert not tuple(tmp_path.iterdir())


@pytest.mark.parametrize("invalid_value", [float("nan"), float("inf"), float("-inf")])
def test_nonfinite_serialization_never_publishes_a_report(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, invalid_value: float
) -> None:
    output = tmp_path / "diagnostics.json"
    output.write_text('{"status":"PASS"}', encoding="utf-8")
    report = validation.blocked_report()
    report["invalid_synthetic_observation"] = invalid_value
    monkeypatch.setattr(validation, "blocked_report", lambda: report)
    assert validation.main(["--output", str(output)]) == 1
    assert not output.exists()
    assert not tuple(tmp_path.iterdir())


@pytest.mark.parametrize("failure_at", ["serialize", "fsync", "replace"])
def test_interrupted_qualifier_does_not_keep_stale_pass_or_publish_partial_report(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failure_at: str
) -> None:
    output = tmp_path / "diagnostics.json"
    output.write_text('{"status":"PASS"}', encoding="utf-8")

    def interrupt(*_args: Any, **_kwargs: Any) -> None:
        assert not output.exists()
        raise KeyboardInterrupt("syntetyczne przerwanie kwalifikatora")

    target, name = {
        "serialize": (validation.json, "dump"),
        "fsync": (validation.os, "fsync"),
        "replace": (validation.os, "replace"),
    }[failure_at]
    monkeypatch.setattr(target, name, interrupt)
    with pytest.raises(KeyboardInterrupt):
        validation.write_blocked_report(output)
    assert not output.exists()
    assert not tuple(tmp_path.iterdir())


def test_absent_output_directory_is_a_refusal(tmp_path: Path) -> None:
    output = tmp_path / "missing-directory" / "diagnostics.json"
    assert validation.main(["--output", str(output)]) == 1
    assert not output.exists()


def test_mutating_one_diagnostic_report_cannot_change_the_next_report() -> None:
    first = validation.blocked_report()
    first["results"]["WINDOWS_REBOOT_RECOVERY"]["proofs"]["boot_changed"]["status"] = "PASS"
    first["results"]["WINDOWS_LONG_RUNNING_LIFECYCLE"]["proofs"].clear()
    assert_complete_refusal(validation.blocked_report())
