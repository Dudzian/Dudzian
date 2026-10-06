"""Production composition must stop at missing frozen authentication boundaries."""

from __future__ import annotations

import os
from types import SimpleNamespace

import pytest

from deployment import windows_production_pre_enrollment as pre
from deployment.windows_stage9_production_trust import ProductionTrustUnavailable


@pytest.mark.parametrize("context", [None, object(), SimpleNamespace(environment="PRODUCTION")])
def test_caller_selected_trust_fails_before_authentication(context: object) -> None:
    with pytest.raises(ProductionTrustUnavailable):
        pre.require_production_request_authentication(context)


def test_missing_authentication_never_becomes_success(monkeypatch: pytest.MonkeyPatch) -> None:
    # TEST_ONLY harness to reach the unavailable boundary, not production trust.
    test_only_context = object()
    monkeypatch.setattr(pre, "require_verified_production_trust_context", lambda value: value)
    with pytest.raises(pre.ProductionPreEnrollmentUnavailable) as raised:
        pre.require_production_request_authentication(test_only_context)
    for blocker in pre.AUTHENTICATION_BLOCKERS:
        assert blocker in str(raised.value)


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
