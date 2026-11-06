"""Proponowane bajty nie uzyskują zatwierdzenia przez wynik testów remediacji."""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from bot_core import freshness_deployment_qualification as qualification
from deployment import freshness_application as application
from tests.security import test_freshness_qualification_profiles as profiles
from tests.security.freshness_verifier_remediation_support import ROOT, SOURCE, load_verifier

protected_sources = profiles.full_sources

_APPROVED = "64dd105d54a2eabd9be9164846fe642ca0a274b30ca5d0a83735a4436d9f39e1"


def test_production_source_and_both_approval_pins_remain_unchanged():
    assert hashlib.sha256((ROOT / SOURCE).read_bytes()).hexdigest() == _APPROVED
    assert application.REVIEWED_SUBSTRATE_DIGESTS[SOURCE] == _APPROVED
    assert qualification.REVIEWED_CODE_DIGESTS[Path(SOURCE).name] == _APPROVED
    assert SOURCE in application.REQUIRED_FILES


def test_proposed_source_is_rejected_by_existing_source_gate(
    protected_sources, tmp_path, monkeypatch
):
    """Model wyłącznie metadanych fixture; prawdziwe bajty i piny pozostają ścisłe."""
    monkeypatch.setenv("FRESHNESS_REMEDIATION_SOURCE", "candidate")
    module = load_verifier(tmp_path / "candidate")
    candidate = Path(module.__file__).read_bytes()
    assert hashlib.sha256(candidate).hexdigest() != _APPROVED
    target = protected_sources / Path(SOURCE).name
    target.write_bytes(candidate)
    with pytest.raises(qualification.DeploymentQualificationError) as refusal:
        qualification._qualify_reviewed_sources(protected_sources)
    assert str(refusal.value) == f"reviewed source digest mismatch: {target}"
