"""Rozdzielenie podpisanych bajtów i pełnego dowodu; bez kwalifikacji hosta."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
import stat
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from bot_core import freshness_deployment_qualification as qualification
from deployment import freshness_application as application

ROOT = Path(__file__).resolve().parents[2]
ACCOUNT_GENESIS = "account_genesis_freshness_authority.py"


def test_minimal_policy_is_exact_subset_of_frozen_full_product_pins():
    expected = {
        "bot_core/" + name: digest
        for name, digest in qualification.REVIEWED_CODE_DIGESTS.items()
        if name != ACCOUNT_GENESIS
    }
    assert application.REVIEWED_SUBSTRATE_DIGESTS == expected
    assert "bot_core/" + ACCOUNT_GENESIS not in application.REQUIRED_FILES
    assert ACCOUNT_GENESIS in qualification.REVIEWED_CODE_DIGESTS


def test_full_pin_is_preserved_and_checkout_drift_is_explicit_review_material():
    expected = qualification.REVIEWED_CODE_DIGESTS[ACCOUNT_GENESIS]
    assert expected == "cd508809f4b4f0663610f592b77695784d74eb4c7e2fba50c42d7a4dbf252f7b"
    assert (
        hashlib.sha256((ROOT / "bot_core" / ACCOUNT_GENESIS).read_bytes()).hexdigest() != expected
    )


@pytest.fixture
def full_sources(tmp_path, monkeypatch):
    for name in qualification.REVIEWED_CODE_DIGESTS:
        (tmp_path / name).write_bytes((ROOT / "bot_core" / name).read_bytes())

    # Model protected Linux metadata, not Windows' synthetic 0666 permission
    # bits. Preserve real file types and missing-path errors, all source bytes
    # and frozen pins. This fixture never establishes a production PASS.
    read_source_mode = qualification._mode

    def protected_source_mode(path):
        metadata = list(read_source_mode(path))
        metadata[stat.ST_UID] = 0
        metadata[stat.ST_MODE] = stat.S_IFMT(metadata[stat.ST_MODE]) | 0o644
        return os.stat_result(metadata)

    monkeypatch.setattr(qualification, "_mode", protected_source_mode)
    return tmp_path


@pytest.mark.parametrize("host_permissions", [0o644, 0o666], ids=["posix-mode", "windows-mode"])
@pytest.mark.parametrize("change", ["current-unreviewed", "modified", "missing"])
def test_full_source_gate_refuses_account_genesis_without_opening_product(
    full_sources, change, host_permissions
):
    for path in full_sources.iterdir():
        path.chmod(host_permissions)
    source = full_sources / ACCOUNT_GENESIS
    if change == "modified":
        source.write_bytes(source.read_bytes() + b"\n# unauthorized source\n")
    elif change == "missing":
        source.unlink()
    with pytest.raises(qualification.DeploymentQualificationError) as refusal:
        qualification._qualify_reviewed_sources(full_sources)
    reason = (
        "missing or inaccessible path" if change == "missing" else "reviewed source digest mismatch"
    )
    assert str(refusal.value) == f"{reason}: {source}"


@pytest.mark.parametrize(
    "result",
    [
        {},
        {"scope": "cryptohunter.freshness.substrate.v2", "production_qualification": "PASS"},
        {
            "qualification_profile": "PRODUCTION_LOCAL",
            "scope": "cryptohunter.freshness.substrate.v2",
            "production_qualification": "WITHHELD",
        },
        {
            "qualification_profile": "SIGNED_FRESHNESS_SUBSTRATE",
            "scope": "cryptohunter.production.local",
            "production_qualification": "WITHHELD",
        },
    ],
)
def test_ambiguous_or_upgraded_substrate_scope_is_refused(monkeypatch, result):
    monkeypatch.setattr(qualification, "sys", SimpleNamespace(platform="linux"))
    monkeypatch.setattr(qualification, "_load_application", lambda: result)
    with pytest.raises(qualification.DeploymentQualificationError, match="scope"):
        qualification.qualify_signed_freshness_substrate()


@pytest.mark.parametrize("platform", ["win32", "darwin"])
def test_minimal_profile_does_not_claim_linux_proof_on_other_platforms(monkeypatch, platform):
    monkeypatch.setattr(qualification, "sys", SimpleNamespace(platform=platform))
    load = Mock(side_effect=AssertionError("No installed Linux access"))
    monkeypatch.setattr(qualification, "_load_application", load)
    with pytest.raises(qualification.DeploymentQualificationError, match="native Linux"):
        qualification.qualify_signed_freshness_substrate()
    load.assert_not_called()


def test_default_cli_still_calls_full_qualification_and_refuses(monkeypatch, capsys):
    full = Mock(side_effect=qualification.DeploymentQualificationError("blocked"))
    minimal = Mock(side_effect=AssertionError("Full consumer must not downgrade"))
    monkeypatch.setattr(qualification, "qualify_production_local_deployment", full)
    monkeypatch.setattr(qualification, "qualify_signed_freshness_substrate", minimal)
    assert qualification.main([]) == 1
    assert capsys.readouterr().out == "FAIL CLOSED\n"
    full.assert_called_once_with()
    minimal.assert_not_called()


def test_linux_product_consumer_cannot_substitute_minimal_result(monkeypatch):
    from deployment.platforms import linux

    full = Mock(side_effect=qualification.DeploymentQualificationError("full proof withheld"))
    minimal = Mock(side_effect=AssertionError("Minimal bytes are not a live product proof"))
    monkeypatch.setattr(qualification, "qualify_production_local_deployment", full)
    monkeypatch.setattr(qualification, "qualify_signed_freshness_substrate", minimal)
    with pytest.raises(qualification.DeploymentQualificationError, match="withheld"):
        linux.qualify_live()
    full.assert_called_once_with()
    minimal.assert_not_called()
