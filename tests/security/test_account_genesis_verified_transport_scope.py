"""TEST_ONLY: prawdziwe bajty/bramka; wyłącznie metadane instalacji są modelem."""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from bot_core import freshness_deployment_qualification as qualification
from deployment import freshness_application as application
from tests.security import test_freshness_qualification_profiles as profiles
from tests.security.account_genesis_transport_review_support import (
    BASELINE_SOURCE_SHA256,
    ROOT,
    SOURCE,
    load_account_genesis,
)
from tests.security.freshness_verifier_remediation_support import load_verifier

protected_sources = profiles.full_sources

_APPROVED_ACCOUNT_GENESIS = "cd508809f4b4f0663610f592b77695784d74eb4c7e2fba50c42d7a4dbf252f7b"
_APPROVED_VERIFIER = "64dd105d54a2eabd9be9164846fe642ca0a274b30ca5d0a83735a4436d9f39e1"


def test_baseline_drift_does_not_refresh_frozen_product_pin():
    """Niezmienione źródło main nadal nie jest zatwierdzonym AccountGenesis."""
    assert hashlib.sha256((ROOT / SOURCE).read_bytes()).hexdigest() == BASELINE_SOURCE_SHA256
    assert BASELINE_SOURCE_SHA256 != _APPROVED_ACCOUNT_GENESIS
    assert qualification.REVIEWED_CODE_DIGESTS[Path(SOURCE).name] == _APPROVED_ACCOUNT_GENESIS
    assert SOURCE not in application.REQUIRED_FILES
    assert SOURCE not in application.REVIEWED_SUBSTRATE_DIGESTS
    assert application.REVIEWED_SUBSTRATE_DIGESTS == {
        "bot_core/" + name: digest
        for name, digest in qualification.REVIEWED_CODE_DIGESTS.items()
        if name != Path(SOURCE).name
    }


@pytest.mark.parametrize("candidate", ["account_genesis", "verifier", "both"])
def test_existing_release_gate_rejects_each_proposed_composition(
    protected_sources, tmp_path, monkeypatch, candidate
):
    """Bez rehashu: rzeczywisty gate sprawdza obie prywatne propozycje."""
    monkeypatch.setenv("FRESHNESS_REMEDIATION_SOURCE", "candidate")
    if candidate in {"account_genesis", "both"}:
        module = load_account_genesis(tmp_path / "account-candidate")
        source = Path(module.__file__).read_bytes()
        assert hashlib.sha256(source).hexdigest() not in {
            BASELINE_SOURCE_SHA256,
            _APPROVED_ACCOUNT_GENESIS,
        }
        (protected_sources / Path(SOURCE).name).write_bytes(source)
    if candidate in {"verifier", "both"}:
        module = load_verifier(tmp_path / "verifier-candidate")
        source = Path(module.__file__).read_bytes()
        assert hashlib.sha256(source).hexdigest() != _APPROVED_VERIFIER
        (protected_sources / "freshness_semantic_verifier.py").write_bytes(source)
    with pytest.raises(qualification.DeploymentQualificationError) as refusal:
        qualification._qualify_reviewed_sources(protected_sources)
    rejected = (
        "account_genesis_freshness_authority.py"
        if candidate == "account_genesis"
        else "freshness_semantic_verifier.py"
    )
    assert str(refusal.value) == (
        f"reviewed source digest mismatch: {protected_sources / rejected}"
    )
    assert qualification.REVIEWED_CODE_DIGESTS[Path(SOURCE).name] == _APPROVED_ACCOUNT_GENESIS
    assert (
        application.REVIEWED_SUBSTRATE_DIGESTS["bot_core/freshness_semantic_verifier.py"]
        == _APPROVED_VERIFIER
    )
