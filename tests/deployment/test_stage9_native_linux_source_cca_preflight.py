"""S9-RPI-21: pre-hardware read-only Linux inventory, never a GO gate."""

from __future__ import annotations

import json
import os
from pathlib import Path
import stat
import sys
from types import SimpleNamespace

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
    assert all(row["observation"] == "NOT_QUALIFIED" for row in result["paths"].values())
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


@pytest.mark.parametrize(
    ("role", "name"),
    [
        ("root_source", "ch_root_custody_source"),
        ("history_source", "ch_history_custody_source"),
        ("cca", "ch_custody_completion"),
    ],
)
@pytest.mark.parametrize("extra_gid", [0, 62011, 62999], ids=["root", "signer", "single"])
def test_any_unapproved_supplementary_group_blocks_isolation(role, name, extra_gid):
    accounts = {
        "ch_root_custody_source": (62001, 62001, (62001,)),
        "ch_history_custody_source": (62002, 62002, (62002,)),
        "ch_custody_completion": (62003, 62003, (62003,)),
        "cryptohunter-root-proof-signer": (62011, 62011, (62011,)),
    }
    uid, gid, _groups = accounts[name]
    accounts[name] = (uid, gid, (gid, extra_gid))
    identities, problems = preflight._inspect_accounts(accounts)
    assert identities[role]["groups"] == (gid, extra_gid)
    assert problems == [f"{role}: unapproved supplementary GIDs [{extra_gid}]"]


@pytest.mark.parametrize(
    ("left", "right"),
    [
        ("root_source", "history_source"),
        ("root_source", "cca"),
        ("history_source", "cca"),
    ],
)
def test_shared_foreign_supplementary_group_blocks_each_role_pair(left, right):
    accounts = {
        "ch_root_custody_source": (62001, 62001, (62001,)),
        "ch_history_custody_source": (62002, 62002, (62002,)),
        "ch_custody_completion": (62003, 62003, (62003,)),
    }
    for role, name, _subdir in preflight._PROPOSED:
        if role in (left, right):
            uid, gid, _groups = accounts[name]
            accounts[name] = (uid, gid, (gid, 62999))
    _identities, problems = preflight._inspect_accounts(accounts)
    assert f"{left}: unapproved supplementary GIDs [62999]" in problems
    assert f"{right}: unapproved supplementary GIDs [62999]" in problems
    assert f"{left}/{right}: shared effective group" in problems


def test_exclusive_distinct_primary_groups_have_no_account_failures():
    accounts = {
        "ch_root_custody_source": (62001, 62001, (62001,)),
        "ch_history_custody_source": (62002, 62002, (62002,)),
        "ch_custody_completion": (62003, 62003, (62003,)),
    }
    identities, problems = preflight._inspect_accounts(accounts)
    assert set(identities) == {"root_source", "history_source", "cca"}
    assert problems == []


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


@pytest.mark.skipif(sys.platform != "linux", reason="Linux root filesystem metadata")
@pytest.mark.parametrize("root_path", ["/", "//"])
@pytest.mark.parametrize(
    ("owner", "file_mode", "attributes", "reason"),
    [
        (63000, stat.S_IFDIR | 0o755, [], "ancestor owner mismatch"),
        (0, stat.S_IFDIR | 0o775, [], "writable ancestor or target"),
        (0, stat.S_IFDIR | 0o757, [], "writable ancestor or target"),
        (0, stat.S_IFLNK | 0o777, [], "non-directory or symlink in path"),
        (0, stat.S_IFREG | 0o644, [], "non-directory or symlink in path"),
        (0, stat.S_IFDIR | 0o755, ["system.posix_acl_access"], "extended POSIX ACL present"),
        (0, stat.S_IFDIR | 0o755, ["system.posix_acl_default"], "extended POSIX ACL present"),
    ],
)
def test_native_base_root_metadata_must_be_safe(
    monkeypatch, root_path, owner, file_mode, attributes, reason
):
    root = Path(root_path)
    metadata = {root: SimpleNamespace(st_mode=file_mode, st_uid=owner, st_gid=0)}
    with monkeypatch.context() as patch:
        patch.setattr(os, "geteuid", lambda: 63000)
        patch.setattr(Path, "lstat", lambda node: metadata[node])
        patch.setattr(os, "listxattr", lambda node, **kwargs: attributes)
        status, why = preflight._inspect_directory(
            root, "/candidate", uid=62001, gid=62001, mode=0o700
        )
    assert status == "NOT_QUALIFIED"
    assert why == reason


@pytest.mark.skipif(sys.platform != "linux", reason="Linux POSIX ACL inventory")
def test_unobservable_base_root_acl_fails_closed(monkeypatch):
    def unavailable(node, **kwargs):
        raise OSError("ACL inventory unavailable")

    with monkeypatch.context() as patch:
        patch.setattr(
            Path, "lstat", lambda node: SimpleNamespace(st_mode=stat.S_IFDIR | 0o755, st_uid=0)
        )
        patch.setattr(os, "listxattr", unavailable)
        status, why = preflight._inspect_directory(
            Path("/"), "/candidate", uid=1, gid=1, mode=0o700
        )
    assert status == "NOT_QUALIFIED"
    assert why == "POSIX ACL inventory unavailable"


@pytest.mark.skipif(sys.platform != "linux", reason="Linux lstat semantics")
def test_symlink_base_root_is_rejected_without_following_link(tmp_path):
    root = tmp_path / "linked-root"
    root.symlink_to(tmp_path, target_is_directory=True)
    status, why = preflight._inspect_directory(root, "/candidate", uid=1, gid=1, mode=0o700)
    assert status == "NOT_QUALIFIED"
    assert why == "non-directory or symlink in path"


@pytest.mark.skipif(sys.platform != "linux", reason="Linux filesystem permissions")
@pytest.mark.parametrize("mode", [0o775, 0o757])
def test_writable_temporary_base_root_is_rejected(tmp_path, mode):
    root = tmp_path / "test-root"
    root.mkdir()
    root.chmod(mode)
    status, why = preflight._inspect_directory(root, "/candidate", uid=1, gid=1, mode=0o700)
    assert status == "NOT_QUALIFIED"
    assert why == "writable ancestor or target"


@pytest.mark.skipif(sys.platform != "linux", reason="Linux proposed filesystem metadata")
@pytest.mark.parametrize("native_root", [False, True], ids=["hermetic", "native"])
def test_all_proposed_metadata_matches_still_cannot_qualify_host(
    tmp_path, monkeypatch, capsys, native_root
):
    root = Path("/") if native_root else tmp_path
    accounts = {
        "ch_root_custody_source": (62001, 62001, (62001,)),
        "ch_history_custody_source": (62002, 62002, (62002,)),
        "ch_custody_completion": (62003, 62003, (62003,)),
    }

    def directory(uid, gid, mode):
        return SimpleNamespace(st_mode=stat.S_IFDIR | mode, st_uid=uid, st_gid=gid)

    metadata = {root: directory(0 if native_root else 63000, 0, 0o755)}
    for ancestor in ("etc", "etc/cryptohunter", "var", "var/lib", "var/lib/cryptohunter"):
        metadata[root / ancestor] = directory(0, 0, 0o755)
    for _role, name, subdir in preflight._PROPOSED:
        uid, gid, _groups = accounts[name]
        metadata[root / "etc/cryptohunter" / subdir] = directory(0, gid, 0o750)
        metadata[root / "var/lib/cryptohunter" / subdir] = directory(uid, gid, 0o700)
    acl_observations = []

    def listxattr(node, *, follow_symlinks):
        assert follow_symlinks is False
        acl_observations.append(node)
        return []

    with monkeypatch.context() as patch:
        patch.setattr(os, "geteuid", lambda: 63000)
        patch.setattr(Path, "lstat", lambda node: metadata[node])
        patch.setattr(os, "listxattr", listxattr)
        result = preflight.inspect_host(root=root, accounts=accounts)
    assert set(acl_observations) == set(metadata)
    assert acl_observations.count(root) == 6
    assert result["observed_failures"] == []
    assert len(result["paths"]) == 6
    assert all(
        row["observation"] == "MATCHED_PROPOSED_METADATA_ONLY" for row in result["paths"].values()
    )
    assert result["status"] == result["production_emission"] == "BLOCKED"
    assert result["qualification"] == "NOT_QUALIFIED"
    assert result["topology"] == "PROPOSED_NOT_APPROVED"
    assert result["blockers"] == list(preflight._BLOCKERS)
    monkeypatch.setattr(preflight, "inspect_host", lambda: result)
    assert preflight.main() == 2
    assert json.loads(capsys.readouterr().out) == result


@pytest.mark.parametrize("platform", ["win32", "darwin"])
def test_non_linux_inventory_stops_before_native_account_lookup(monkeypatch, platform):
    def unexpected_native_lookup():
        pytest.fail("non-Linux inventory must not query native accounts")

    with monkeypatch.context() as patch:
        patch.setattr(sys, "platform", platform)
        patch.setattr(preflight, "_native_accounts", unexpected_native_lookup)
        result = preflight.inspect_host()
    assert result["status"] == result["production_emission"] == "BLOCKED"
    assert result["qualification"] == "NOT_QUALIFIED"
    assert result["topology"] == "PROPOSED_NOT_APPROVED"
    assert result["observed_failures"] == ["NATIVE_LINUX_REQUIRED"]
    assert result["paths"] == result["principal_observations"] == {}


_PROPOSED_ACCOUNTS = {
    "ch_root_custody_source": (62001, 62001, (62001,)),
    "ch_history_custody_source": (62002, 62002, (62002,)),
    "ch_custody_completion": (62003, 62003, (62003,)),
}
_REFERENCE_ACCOUNTS = {
    "cryptohunter-root-proof-signer": (62011, 62011, (62011,)),
    "cryptohunter-history-attestation-signer": (62012, 62012, (62012,)),
    "cryptohunter-root-proof-issuer": (62013, 62013, (62013,)),
}


def test_missing_references_are_explicitly_unobserved_not_trusted(tmp_path):
    result = preflight.inspect_host(root=tmp_path, accounts=_PROPOSED_ACCOUNTS)
    if sys.platform != "linux":
        assert result["status"] == "BLOCKED"
        return
    assert result["unobserved_reference_principals"] == [
        "root_signer", "history_signer", "issuer"
    ]
    assert result["reference_principal_observations"] == {}
    assert result["status"] == "BLOCKED"
    assert result["qualification"] == "NOT_QUALIFIED"


@pytest.mark.skipif(sys.platform != "linux", reason="Linux host NSS inventory")
def test_distinct_observed_references_do_not_grant_qualification(tmp_path):
    accounts = {**_PROPOSED_ACCOUNTS, **_REFERENCE_ACCOUNTS}
    result = preflight.inspect_host(root=tmp_path, accounts=accounts)
    assert result["unobserved_reference_principals"] == []
    assert set(result["reference_principal_observations"]) == {
        "root_signer", "history_signer", "issuer"
    }
    assert result["reference_principal_observations"]["root_signer"]["uid"] == 62011
    assert result["status"] == result["production_emission"] == "BLOCKED"
    assert result["qualification"] == "NOT_QUALIFIED"
    assert "INDEPENDENTLY_APPROVED_INSTALLATION_PIN_MISSING" in result["blockers"]


@pytest.mark.skipif(sys.platform != "linux", reason="Linux host NSS inventory")
@pytest.mark.parametrize("role,name,_directory", preflight._PROPOSED)
@pytest.mark.parametrize("reference_role,reference_name", preflight._KNOWN_REFERENCE_AUTHORITIES)
@pytest.mark.parametrize("collision", ["uid", "gid"], ids=["same-uid", "same-group"])
def test_cross_authority_uid_and_group_reuse_is_observed(
    tmp_path, role, name, _directory, reference_role, reference_name, collision
):
    accounts = {**_PROPOSED_ACCOUNTS, **_REFERENCE_ACCOUNTS}
    uid, gid, _groups = accounts[name]
    if collision == "uid":
        accounts[reference_name] = (uid, 63000, (63000,))
    else:
        accounts[reference_name] = (63000, gid, (gid,))
    result = preflight.inspect_host(root=tmp_path, accounts=accounts)
    reason = "shared UID" if collision == "uid" else "shared effective group"
    assert f"{role}/{reference_role}: {reason}" in result["observed_failures"]
    assert result["status"] == "BLOCKED"
    assert result["qualification"] == "NOT_QUALIFIED"


@pytest.mark.skipif(sys.platform != "linux", reason="Linux host NSS inventory")
def test_signer_to_signer_shared_gid_is_observed(tmp_path):
    accounts = {**_PROPOSED_ACCOUNTS, **_REFERENCE_ACCOUNTS}
    accounts["cryptohunter-history-attestation-signer"] = (62012, 62011, (62011,))
    result = preflight.inspect_host(root=tmp_path, accounts=accounts)
    assert "root_signer/history_signer: shared effective group" in result["observed_failures"]
    assert result["status"] == "BLOCKED"


@pytest.mark.skipif(sys.platform != "linux", reason="Linux host NSS inventory")
@pytest.mark.parametrize(
    "invalid",
    [
        (0, 62011, (62011,)),
        (62011, 0, (0,)),
        (62011, 62011, ("not-a-gid",)),
        (62011, 62011, (62011, 0)),
    ],
)
def test_reference_principal_invalid_identity_never_passes(tmp_path, invalid):
    accounts = {**_PROPOSED_ACCOUNTS, **_REFERENCE_ACCOUNTS}
    accounts["cryptohunter-root-proof-signer"] = invalid
    result = preflight.inspect_host(root=tmp_path, accounts=accounts)
    assert any(
        "root_signer: malformed or privileged reference principal" == issue
        for issue in result["observed_failures"]
    )
    assert result["status"] == "BLOCKED"


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
