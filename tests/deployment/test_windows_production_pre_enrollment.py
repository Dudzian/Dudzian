"""Installed production composition rejects declarations and unsafe inputs."""

from __future__ import annotations

import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from bot_core.licensing.production_pre_enrollment import ProductionPreEnrollmentError
from deployment import windows_production_pre_enrollment as pre


@pytest.mark.parametrize("context", [None, object(), SimpleNamespace(environment="PRODUCTION")])
def test_caller_selected_trust_fails_before_authentication(context: object) -> None:
    with pytest.raises(ProductionPreEnrollmentError, match="AUTHENTICATED_PRODUCTION"):
        pre.require_production_request_authentication(context)


def test_missing_authentication_never_becomes_success() -> None:
    declared = SimpleNamespace(
        environment="PRODUCTION", challenge_verified=True, custody_verified=True
    )
    with pytest.raises(ProductionPreEnrollmentError):
        pre.require_production_request_authentication(declared)


@pytest.mark.skipif(os.name == "nt", reason="non-Windows fail-closed boundary")
def test_non_windows_production_composition_cannot_create_key() -> None:
    from deployment.platforms.windows import WindowsDeploymentNotQualified

    with pytest.raises(WindowsDeploymentNotQualified):
        pre.qualify_installed_production_key()


def test_operator_failure_does_not_print_exception_or_secret(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    def test_only_failure() -> dict[str, object]:
        raise RuntimeError("TEST_ONLY private_blob=must-never-appear")

    monkeypatch.setattr(pre, "qualify_installed_production_key", test_only_failure)
    assert pre.main(["--qualify-key"]) == 1
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == "PRODUCTION_PRE_ENROLLMENT_QUALIFICATION_FAILED\n"


@pytest.fixture
def local_orchestration(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    """TEST_ONLY orchestration boundary; cryptographic verifiers have separate tests."""
    from deployment import windows_cng_custody_bridge as bridge
    from deployment.windows_cng_pre_enrollment import WindowsCNGPreEnrollmentKey

    calls = []
    context = SimpleNamespace(release_payload_digest="aa" * 32, release_version=1)
    request = SimpleNamespace(canonical_bytes=b"TEST_ONLY_public_request")
    projection = object()

    class TestOnlyKey:
        public_evidence = {"qualification_scope": "TEST_ONLY_ORCHESTRATION"}

        def __enter__(self):
            calls.append("enter")
            return self

        def __exit__(self, *_):
            calls.append("close")

        def sign_request(self, candidate, *, production_trust_context):
            assert candidate is request
            assert production_trust_context is context
            calls.append("sign")
            return b"TEST_ONLY_signature"

    key = TestOnlyKey()

    def open_key(directory):
        assert directory == tmp_path / "PreEnrollment"
        calls.append("open_fixed_key")
        return key

    def build_request(**arguments):
        assert arguments["context"] is context
        assert arguments["key"] is key
        assert arguments["challenge_raw"] == b"TEST_ONLY_challenge"
        calls.append("build")
        return request

    def collect(candidate, payload, *, ak_key_name, target_tpm_projection):
        assert candidate is key and payload is request
        assert ak_key_name == "TEST_ONLY_retained_ak"
        assert target_tpm_projection is projection
        calls.append("creation")
        return SimpleNamespace(canonical_bytes=b"TEST_ONLY_custody")

    monkeypatch.setattr(pre, "resolve_paths", lambda: SimpleNamespace(state=tmp_path))
    monkeypatch.setattr(pre, "production_trust_package_path", lambda ceremony: tmp_path / ceremony)
    monkeypatch.setattr(pre, "load_production_trust", lambda path: context)
    monkeypatch.setattr(pre, "require_current_production_trust_context", lambda value: value)
    monkeypatch.setattr(WindowsCNGPreEnrollmentKey, "open_or_create", open_key)
    monkeypatch.setattr(pre, "build_local_production_pre_enrollment_request", build_request)
    monkeypatch.setattr(bridge, "certify_pre_enrollment_key_creation", collect)
    monkeypatch.setattr(
        pre.TPMEnrollmentRequestV1,
        "from_canonical_bytes",
        lambda raw: SimpleNamespace(document={"public_projection": {}}),
    )
    monkeypatch.setattr(pre.TPMPublicProjectionV1, "verify", lambda raw: projection)
    return SimpleNamespace(
        calls=calls,
        context=context,
        bridge=bridge,
        arguments={
            "challenge_raw": b"TEST_ONLY_challenge",
            "activation_request_raw": b"activation",
            "tpm_request_raw": b"request",
            "tpm_challenge_raw": b"challenge",
            "tpm_response_raw": b"response",
            "ak_key_name": "TEST_ONLY_retained_ak",
        },
    )


def test_local_preparation_enforces_fixed_factory_creation_then_sign(local_orchestration):
    result = pre.prepare_installed_production_request(**local_orchestration.arguments)
    assert result.request_raw == b"TEST_ONLY_public_request"
    assert result.signature == b"TEST_ONLY_signature"
    assert result.custody_evidence_raw == b"TEST_ONLY_custody"
    assert local_orchestration.calls == [
        "open_fixed_key",
        "enter",
        "build",
        "creation",
        "sign",
        "close",
    ]


def test_creation_failure_closes_key_before_any_request_signature(
    local_orchestration, monkeypatch: pytest.MonkeyPatch
):
    def unsupported(*args, **kwargs):
        raise RuntimeError("PCP_CREATION_EVIDENCE_UNAVAILABLE")

    monkeypatch.setattr(
        local_orchestration.bridge, "certify_pre_enrollment_key_creation", unsupported
    )
    with pytest.raises(RuntimeError, match="PCP_CREATION_EVIDENCE_UNAVAILABLE"):
        pre.prepare_installed_production_request(**local_orchestration.arguments)
    assert local_orchestration.calls == ["open_fixed_key", "enter", "build", "close"]


def test_current_trust_failure_prevents_any_local_key_operation(
    local_orchestration, monkeypatch: pytest.MonkeyPatch
):
    def unavailable(value):
        raise RuntimeError("CURRENT_PRODUCTION_TRUST_UNAVAILABLE")

    monkeypatch.setattr(pre, "require_current_production_trust_context", unavailable)
    with pytest.raises(RuntimeError, match="CURRENT_PRODUCTION_TRUST_UNAVAILABLE"):
        pre.prepare_installed_production_request(**local_orchestration.arguments)
    assert local_orchestration.calls == []


def test_key_qualification_reports_required_artifacts_without_enrollment(local_orchestration):
    result = pre.qualify_installed_production_key()
    assert result["legal_enrollment"] == "NOT_PERFORMED"
    assert result["request_authentication"] == "REQUIRES_PRODUCTION_ARTIFACTS"
    assert result["required_authentication_artifacts"] == list(
        pre.REQUIRED_AUTHENTICATION_ARTIFACTS
    )
    assert local_orchestration.calls == ["open_fixed_key", "enter", "close"]


def test_cli_outputs_only_canonical_public_qualification(monkeypatch, capsysbinary):
    monkeypatch.setattr(pre, "qualify_installed_production_key", lambda: {"public": True})
    assert pre.main(["--qualify-key"]) == 0
    captured = capsysbinary.readouterr()
    assert captured.out == b'{"public":true}\n'
    assert captured.err == b""
