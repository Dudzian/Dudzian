"""Syntetyczne testy diagnostyki odmowy; nie dowodzą fizycznego lifecycle Windows."""

from __future__ import annotations

from datetime import datetime, timezone
import json
import re
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
        "attempt_id",
        "report_kind",
        "status",
        "reason_code",
        "generated_at_utc",
        "native_qualification_performed",
        "results",
    }
    assert re.fullmatch(r"[0-9a-f]{32}", report["attempt_id"]) is not None
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


def test_main_defaults_to_stdout_without_selecting_an_output_path(
    capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    def unexpected_write(*_args: Any, **_kwargs: Any) -> None:
        raise AssertionError("Domyślna diagnostyka nie może uruchomić zapisu")

    monkeypatch.setattr(validation, "write_blocked_report", unexpected_write)
    assert validation.main([]) == 1
    captured = capsys.readouterr()
    assert_complete_refusal(json.loads(captured.out))
    assert validation.UNAVAILABLE in captured.err


def test_output_argument_is_not_independent_directory_authorization(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    output = tmp_path / "diagnostics.json"
    assert validation.main(["--output", str(output)]) == 1
    assert not output.exists()
    assert not tuple(tmp_path.iterdir())
    assert validation.UNAVAILABLE in capsys.readouterr().err


def test_existing_output_is_unchanged_when_directory_authorization_is_absent(
    tmp_path: Path,
) -> None:
    output = tmp_path / "diagnostics.json"
    previous = b'{"status":"PASS","attempt_id":"historical"}'
    output.write_bytes(previous)
    assert validation.main(["--output", str(output)]) == 1
    assert output.read_bytes() == previous
    assert tuple(tmp_path.iterdir()) == (output,)


def test_report_requires_separate_explicit_directory_before_any_mutation(tmp_path: Path) -> None:
    output = tmp_path / "diagnostics.json"
    old = b"niepowiazane dane"
    output.write_bytes(old)
    with pytest.raises(ValueError):
        validation.write_blocked_report(output)
    assert output.read_bytes() == old
    assert tuple(tmp_path.iterdir()) == (output,)


@pytest.mark.parametrize("artifact", ["receipt.json", "contract.json", "installer-manifest.json"])
def test_protected_input_alias_refuses_without_changing_input(
    tmp_path: Path, artifact: str
) -> None:
    output = tmp_path / artifact
    original = b"chroniony artefakt wejscia"
    output.write_bytes(original)
    with pytest.raises((ValueError, FileExistsError)):
        validation.write_blocked_report(
            output, approved_directory=tmp_path, protected_inputs=(output,)
        )
    assert output.read_bytes() == original
    assert tuple(tmp_path.iterdir()) == (output,)


def test_each_attempt_has_a_new_identity_without_reusing_a_previous_report() -> None:
    first = validation.blocked_report()
    second = validation.blocked_report()
    assert first["attempt_id"] != second["attempt_id"]
    assert_complete_refusal(first)
    assert_complete_refusal(second)


@pytest.mark.parametrize("artifact", ["receipt.json", "contract.json", "manifest.json"])
def test_cli_protected_input_cannot_be_created_as_a_diagnostic(
    tmp_path: Path, artifact: str
) -> None:
    protected = tmp_path / artifact
    assert (
        validation.main(
            [
                "--approved-output-directory",
                str(tmp_path),
                "--output",
                str(protected),
                "--protected-input",
                str(protected),
            ]
        )
        == 1
    )
    assert not tuple(tmp_path.iterdir())


@pytest.mark.parametrize("invalid_value", [float("nan"), float("inf"), float("-inf")])
def test_nonfinite_serialization_is_refused_before_creating_a_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, invalid_value: float
) -> None:
    report = validation.blocked_report()
    report["invalid_synthetic_observation"] = invalid_value
    monkeypatch.setattr(validation, "blocked_report", lambda: report)
    assert (
        validation.main(
            ["--approved-output-directory", str(tmp_path), "--output", str(tmp_path / "new.json")]
        )
        == 1
    )
    assert not tuple(tmp_path.iterdir())


def test_report_serialization_interruption_preserves_unrelated_existing_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    old = tmp_path / "previous.json"
    original = b'{"status":"PASS","attempt_id":"old"}'
    old.write_bytes(original)

    def interrupt(*_args: Any, **_kwargs: Any) -> dict[str, Any]:
        raise KeyboardInterrupt("syntetyczne przerwanie przed publikacja")

    monkeypatch.setattr(validation, "blocked_report", interrupt)
    with pytest.raises(KeyboardInterrupt):
        validation.write_blocked_report(tmp_path / "new.json", approved_directory=tmp_path)
    assert old.read_bytes() == original
    assert tuple(tmp_path.iterdir()) == (old,)


def test_mutating_one_diagnostic_report_cannot_change_the_next_report() -> None:
    first = validation.blocked_report()
    first["results"]["WINDOWS_REBOOT_RECOVERY"]["proofs"]["boot_changed"]["status"] = "PASS"
    first["results"]["WINDOWS_LONG_RUNNING_LIFECYCLE"]["proofs"].clear()
    assert_complete_refusal(validation.blocked_report())


def test_preserved_historical_pass_is_not_adopted_as_current_evidence(tmp_path: Path) -> None:
    from deployment.platform_evidence import evidence_document
    from deployment.platform_readiness import (
        EvidenceValidationError,
        _validated_results,
        load_contract,
    )

    previous = evidence_document(
        platform_name="WINDOWS",
        source_revision="current-revision",
        ci_provider="https://github.com",
        ci_run_id="42",
        runner_os="Windows",
        runner_arch="X64",
        results=[
            {
                "item": item,
                "status": "PASS",
                "evidence_class": "LIVE_WINDOWS_INTEGRATION",
                "test_or_probe": "historyczny plik",
                "details": "Sama deklaracja nie ma autorytetu natywnego",
            }
            for item in EXPECTED_PROOFS
        ],
    )
    output = tmp_path / "old.json"
    original = json.dumps(previous).encode("utf-8")
    output.write_bytes(original)
    assert validation.main(["--output", str(output)]) == 1
    assert output.read_bytes() == original
    with pytest.raises(EvidenceValidationError, match=validation.UNAVAILABLE):
        _validated_results(
            load_contract(), [previous], "current-revision", "https://github.com", "42"
        )
