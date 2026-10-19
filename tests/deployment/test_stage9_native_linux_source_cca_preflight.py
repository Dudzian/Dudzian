"""S9-RPI-21: pre-hardware read-only Linux inventory, never a GO gate."""

from __future__ import annotations

import json
import os
import sys

import pytest

from deployment import stage9_native_linux_source_cca_preflight as preflight


def test_missing_native_accounts_cannot_claim_production_readiness(tmp_path):
    result = preflight.inspect_host(root=tmp_path, accounts={})
    assert result["status"] == "BLOCKED"
    assert result["qualification"] == "NOT_QUALIFIED"
    assert "PRODUCTION_ISSUER_AND_LEGAL_ENROLLMENT_BLOCKED" in result["blockers"]
    if sys.platform == "linux":
        assert len(result["observed_failures"]) == 3
        assert result["principal_observations"] == {}


@pytest.mark.skipif(sys.platform != "linux", reason="Linux NSS and POSIX filesystem")
def test_unique_proposed_accounts_still_lack_live_installation_qualification(tmp_path):
    accounts = {
        "ch_root_custody_source": (62001, 62001, (62001,)),
        "ch_history_custody_source": (62002, 62002, (62002,)),
        "ch_custody_completion": (62003, 62003, (62003,)),
    }
    result = preflight.inspect_host(root=tmp_path, accounts=accounts)
    assert result["topology"] == "PROPOSED_NOT_APPROVED"
    assert result["status"] == result["production_emission"] == "BLOCKED"
    assert len(result["paths"]) == 6
    assert all(
        row["observation"] == "NOT_QUALIFIED" for row in result["paths"].values()
    )
    assert result["principal_observations"]["cca"]["uid"] == 62003
    assert "RUNTIME_SERVICE_IDENTITY_AND_SYSTEMD_NOT_OBSERVED" in result["blockers"]


@pytest.mark.parametrize(
    "right",
    [(62002, 62001, (62001,)), (62001, 62002, (62002,)), (62002, 62002, (62002, 62001))],
)
def test_shared_uid_primary_or_supplementary_group_blocks_separation(right):
    accounts = {
        "ch_root_custody_source": (62001, 62001, (62001,)),
        "ch_history_custody_source": right,
    }
    _identities, problems = preflight._inspect_accounts(accounts)
    assert any("root_source/history_source: shared" in item for item in problems)


@pytest.mark.skipif(sys.platform != "linux", reason="Linux lstat semantics")
def test_missing_or_symlink_directory_fails_without_following_link(tmp_path):
    location = "/candidate"
    status, explanation = preflight._inspect_directory(
        tmp_path, location, uid=os.geteuid(), gid=os.getegid(), mode=0o700
    )
    assert status == "NOT_QUALIFIED"
    assert explanation == "path missing or inaccessible"
    candidate = tmp_path / "candidate"
    candidate.symlink_to(tmp_path)
    status, explanation = preflight._inspect_directory(
        tmp_path, location, uid=os.geteuid(), gid=os.getegid(), mode=0o700
    )
    assert status == "NOT_QUALIFIED"
    assert explanation == "non-directory or symlink in path"


@pytest.mark.skipif(sys.platform != "linux", reason="Linux root/config filesystem rules")
def test_unsafe_world_writable_configuration_ancestor_blocks(tmp_path):
    (tmp_path / "etc").mkdir(mode=0o777)
    (tmp_path / "etc").chmod(0o777)
    (tmp_path / "etc" / "cryptohunter").mkdir(mode=0o755)
    (tmp_path / "etc" / "cryptohunter" / "root-custody-source").mkdir(mode=0o750)
    status, why = preflight._inspect_directory(
        tmp_path,
        "/etc/cryptohunter/root-custody-source",
        uid=os.geteuid(),
        gid=os.getegid(),
        mode=0o750,
    )
    assert status == "NOT_QUALIFIED"
    assert why == "writable ancestor or target"


def test_cli_has_no_success_exit_and_no_installer_or_mutation_flag(monkeypatch, capsys):
    blocked = {
        "schema": "CryptoHunter.Stage9NativeLinuxSourceCCAPreflightV1",
        "status": "BLOCKED",
        "qualification": "NOT_QUALIFIED",
        "blockers": ["INDEPENDENTLY_APPROVED_INSTALLATION_PIN_MISSING"],
    }
    monkeypatch.setattr(preflight, "inspect_host", lambda: blocked)
    assert preflight.main() == 2
    assert json.loads(capsys.readouterr().out) == blocked
    assert not hasattr(preflight, "install") and not hasattr(preflight, "provision")
