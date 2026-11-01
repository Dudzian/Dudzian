"""Regresje granicy produktu: poprawny podpis nie kwalifikuje atrap usług."""

from __future__ import annotations

import copy
import base64
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pytest

from deployment import linux_physical_evidence as evidence
from deployment import linux_release_identity as release
from deployment.platform_readiness import LinuxEvidenceContext, load_contract, production_ready
from tests.deployment.test_linux_physical_evidence import (
    _bind_state_to_challenge,
    _complete_transport,
    _core_document,
    _current_files,
    _gate_cli,
    _payloads,
    _signed_bundle,
    _write,
)
from tests.deployment.test_linux_release_identity import release_context


def _signed_imposter(tmp_path: Path, name: str) -> dict[str, Any]:
    """Zmień produkt i odtwórz wszystkie legalne podpisy oraz przypięcia."""
    context = _complete_transport(tmp_path, current=True)
    installation = context["host"]["installation"]
    installation["package_name"] = name
    materials = context["host"]["materials"]
    materials["installation_contract_sha256"] = evidence.digest(installation)
    payloads = _payloads(context)
    created = datetime.now(timezone.utc) - timedelta(minutes=1)
    context["challenge"] = evidence.issue_challenge(
        context["policy"],
        source_revision=materials["source_revision"],
        ci_run_id="12345",
        ci_run_attempt="2",
        host_id=context["host"]["host_id"],
        now=created,
    )

    def retarget(value: object) -> None:
        if isinstance(value, dict):
            if isinstance(value.get("package"), dict):
                value["package"]["name"] = name
            for child in value.values():
                retarget(child)
        elif isinstance(value, list):
            for child in value:
                retarget(child)

    retarget(payloads)
    for index, payload in enumerate(payloads):
        payload["challenge_id"] = context["challenge"]["challenge_id"]
        payload["materials"] = copy.deepcopy(materials)
        payload["observed_at_utc"] = (
            (created + timedelta(seconds=index + 1)).isoformat().replace("+00:00", "Z")
        )
    _bind_state_to_challenge(payloads, context["challenge"]["challenge_id"], installation)
    context["bundle"] = _signed_bundle(context, payloads)
    return context


def _verify(context: dict[str, Any]) -> dict[str, Any]:
    return evidence.verify_bundle(
        context["bundle"],
        context["challenge"],
        context["policy"],
        source_revision=context["challenge"]["source_revision"],
        ci_run_id="12345",
        ci_run_attempt="2",
    )


def _valid_signed_transport(context: dict[str, Any]) -> None:
    assert len(
        evidence.validate_record_chain(
            context["bundle"], context["challenge"], context["host"], complete=True
        )
    ) == len(evidence.PHASES)


@pytest.mark.parametrize("name", ["arbitrary-placeholder-package", "cryptohunter-pretender"])
def test_approved_signed_imposter_cannot_qualify_as_the_product(tmp_path: Path, name: str) -> None:
    context = _signed_imposter(tmp_path, name)
    _valid_signed_transport(context)
    assert context["host"]["installation"]["package_name"] == name
    with pytest.raises(evidence.LinuxEvidenceError):
        _verify(context)


def test_protected_socket_metadata_without_authenticated_exchange_is_blocked(
    tmp_path: Path,
) -> None:
    context = _complete_transport(tmp_path, current=True)
    _valid_signed_transport(context)
    snapshot = context["bundle"]["records"][1]["payload"]["observations"]["snapshot"]
    assert snapshot["ipc"][0]["mode"] == 0o660
    assert snapshot["ipc"][0]["kind"] == "socket"
    assert snapshot["ipc_qualification"]["status"] == "BLOCKED"
    with pytest.raises(evidence.LinuxEvidenceError):
        _verify(context)


def test_administrative_peer_is_not_an_independent_service_connection_witness(
    tmp_path: Path,
) -> None:
    context = _complete_transport(tmp_path, current=True)
    _valid_signed_transport(context)
    postgres = context["bundle"]["records"][1]["payload"]["observations"]["snapshot"]["postgres"]
    assert postgres["administrative_observation"]["system_user"] == "peer:root"
    assert postgres["service_qualification"]["status"] == "BLOCKED"
    with pytest.raises(evidence.LinuxEvidenceError):
        _verify(context)


@pytest.mark.parametrize("proof", ["ipc", "postgresql"])
def test_signed_pass_flag_cannot_replace_actual_functional_security_evidence(
    tmp_path: Path, proof: str
) -> None:
    context = _complete_transport(tmp_path, current=True)
    payloads = _payloads(context)
    snapshot = payloads[1]["observations"]["snapshot"]
    qualification = (
        snapshot["ipc_qualification"]
        if proof == "ipc"
        else snapshot["postgres"]["service_qualification"]
    )
    qualification["status"] = "PASS"
    qualification["blockers"] = []
    context["bundle"] = _signed_bundle(context, payloads)
    _valid_signed_transport(context)
    with pytest.raises(evidence.LinuxEvidenceError):
        _verify(context)


def test_incomplete_physical_chain_is_not_promoted_despite_known_signer(tmp_path: Path) -> None:
    context = _complete_transport(tmp_path, current=True)
    context["bundle"] = _signed_bundle(context, _payloads(context)[:-1])
    assert (
        len(
            evidence.validate_record_chain(
                context["bundle"], context["challenge"], context["host"], complete=False
            )
        )
        == len(evidence.PHASES) - 1
    )
    with pytest.raises(evidence.LinuxEvidenceError):
        _verify(context)


def test_legacy_saved_pass_and_real_raw_context_do_not_open_the_gate(tmp_path: Path) -> None:
    context, paths, gate_context, saved_claim = _current_files(tmp_path)
    _valid_signed_transport(context)
    assert not production_ready(
        "LINUX",
        load_contract(),
        [_core_document(saved_claim), saved_claim],
        context["challenge"]["source_revision"],
        "https://github.com",
        "12345",
        gate_context,
    )
    with pytest.raises(ValueError):
        gate_context.verify(context["challenge"]["source_revision"], "https://github.com", "12345")


def test_cli_never_writes_success_or_consumes_an_unqualified_product_challenge(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    from deployment.platform_readiness import main

    context, paths, _, saved_claim = _current_files(tmp_path)
    core = _write(tmp_path / "core.json", _core_document(saved_claim))
    ledger = tmp_path / "consumed.sqlite"
    assert main(_gate_cli(context, paths, core, ledger)) == 1
    assert "LINUX_PRODUCTION_READY=PASS" not in capsys.readouterr().out
    assert not ledger.exists()


def test_identical_service_principals_are_not_authorized_by_host_policy(tmp_path: Path) -> None:
    context = _complete_transport(tmp_path, current=True)
    installation = context["host"]["installation"]
    installation["identities"]["verifier"] = copy.deepcopy(installation["identities"]["runtime"])
    context["host"]["materials"]["installation_contract_sha256"] = evidence.digest(installation)
    with pytest.raises(evidence.LinuxEvidenceError):
        evidence.validate_policy(context["policy"])


def test_independently_signed_real_release_profile_cannot_replace_service_security_proofs(
    tmp_path: Path,
) -> None:
    context = _complete_transport(tmp_path, current=True)
    installation = context["host"]["installation"]
    release_fixture = release_context()
    certificate = copy.deepcopy(release_fixture["installation"]["release_identity"])
    statement = certificate["payload"]
    statement["source_revision"] = context["challenge"]["source_revision"]
    statement["installation_profile_sha256"] = evidence.digest(installation)
    statement["artifacts"] = {
        phase: {"sha256": spec["sha256"], "version": spec["version"]}
        for phase, spec in installation["packages"].items()
    }
    statement["components"] = [
        {
            "role": unit["role"],
            "unit": unit["name"],
            "executable_sha256": unit["executable_sha256"],
            "update_executable_sha256": unit["update_executable_sha256"],
            "contracts": list(release.REQUIRED_COMPONENT_CONTRACTS[unit["role"]]),
        }
        for unit in installation["units"]
    ]
    certificate["signature"] = base64.b64encode(
        release_fixture["release_key"].sign(evidence.canonical_bytes(statement))
    ).decode("ascii")
    installation["release_identity"] = certificate
    materials = context["host"]["materials"]
    materials["installation_contract_sha256"] = evidence.digest(installation)
    context["policy"]["release_authorities"] = release_fixture["policy"]["release_authorities"]
    payloads = _payloads(context)
    context["challenge"] = evidence.issue_challenge(
        context["policy"],
        source_revision=materials["source_revision"],
        ci_run_id="12345",
        ci_run_attempt="2",
        host_id=context["host"]["host_id"],
        now=datetime.now(timezone.utc) - timedelta(minutes=1),
    )
    for payload in payloads:
        payload["challenge_id"] = context["challenge"]["challenge_id"]
        payload["materials"] = copy.deepcopy(materials)
    _bind_state_to_challenge(payloads, context["challenge"]["challenge_id"], installation)
    context["bundle"] = _signed_bundle(context, payloads)
    assert (
        release.verify_release_identity(
            installation,
            context["policy"],
            source_revision=materials["source_revision"],
            materials=materials,
        )
        == statement
    )
    _valid_signed_transport(context)
    with pytest.raises(evidence.LinuxEvidenceError) as refused:
        _verify(context)
    from deployment.linux_physical_ipc import LinuxIPCQualificationBlocked

    assert isinstance(refused.value.__cause__, LinuxIPCQualificationBlocked)


def test_native_collector_refuses_approved_mutation_before_creating_driver(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from deployment import linux_physical_collector as collector
    from deployment.linux_physical_ipc import LinuxIPCQualificationBlocked

    context = _complete_transport(tmp_path, current=True)
    prefix = {**context["bundle"], "records": context["bundle"]["records"][:1]}
    identity = prefix["records"][0]["payload"]["host"]
    monkeypatch.setattr(collector, "host_identity", lambda: identity)
    monkeypatch.setattr(collector, "collect_materials", lambda _: context["challenge"]["materials"])
    monkeypatch.setattr(
        collector, "NativeDriver", lambda *_: pytest.fail("Nie wolno rozpoczynać mutacji")
    )
    with pytest.raises(LinuxIPCQualificationBlocked):
        collector.collect_phase(
            challenge=context["challenge"],
            installation=context["host"]["installation"],
            bundle=prefix,
            key_bytes=context["key"],
            phase="clean_install",
            approval=context["challenge"]["challenge_id"] + ":clean_install",
        )
