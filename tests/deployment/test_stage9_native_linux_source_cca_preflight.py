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

    def unexpected_systemctl(*args, **kwargs):
        pytest.fail("non-Linux inventory must not query systemctl")

    import subprocess

    with monkeypatch.context() as patch:
        patch.setattr(sys, "platform", platform)
        patch.setattr(preflight, "_native_accounts", unexpected_native_lookup)
        patch.setattr(subprocess, "run", unexpected_systemctl)
        result = preflight.inspect_host()
    assert result["status"] == result["production_emission"] == "BLOCKED"
    assert result["qualification"] == "NOT_QUALIFIED"
    assert result["topology"] == "PROPOSED_NOT_APPROVED"
    assert result["observed_failures"] == ["NATIVE_LINUX_REQUIRED"]
    assert result["paths"] == result["principal_observations"] == {}
    assert {value["observation"] for value in result["systemd_observations"].values()} == {
        "NOT_OBSERVED_NON_LINUX"
    }


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
    assert result["unobserved_reference_principals"] == ["root_signer", "history_signer", "issuer"]
    assert result["reference_principal_observations"] == {}
    assert result["status"] == "BLOCKED"
    assert result["qualification"] == "NOT_QUALIFIED"


@pytest.mark.skipif(sys.platform != "linux", reason="Linux host NSS inventory")
def test_distinct_observed_references_do_not_grant_qualification(tmp_path):
    accounts = {**_PROPOSED_ACCOUNTS, **_REFERENCE_ACCOUNTS}
    result = preflight.inspect_host(root=tmp_path, accounts=accounts)
    assert result["unobserved_reference_principals"] == []
    assert set(result["reference_principal_observations"]) == {
        "root_signer",
        "history_signer",
        "issuer",
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


def _systemd_show_fixture(
    unit="cryptohunter-root-custody-source.service",
    account="ch_root_custody_source",
):
    properties = {field: "" for field in preflight._SYSTEMD_PROPERTIES}
    properties.update(
        {
            "Id": unit,
            "LoadState": "loaded",
            "ActiveState": "active",
            "User": account,
            "Group": account,
            "DynamicUser": "no",
            "MainPID": "1234",
            "FragmentPath": f"/etc/systemd/system/{unit}",
            "NoNewPrivileges": "yes",
            "PrivateNetwork": "yes",
            "PrivateIPC": "yes",
            "PrivateMounts": "yes",
            "PrivateTmp": "yes",
            "ProtectSystem": "strict",
            "ProtectHome": "yes",
            "RestrictAddressFamilies": "AF_UNIX",
        }
    )
    return properties


def test_rpi23_declared_systemd_settings_are_not_kernel_proof():
    unit = "cryptohunter-root-custody-source.service"
    record = _systemd_show_fixture(unit)
    output = "".join(f"{field}={record[field]}\n" for field in preflight._SYSTEMD_PROPERTIES)
    result = preflight._evaluate_systemd_show(output, unit=unit, account="ch_root_custody_source")
    assert result == {"observation": "PROPOSED_DECLARATIONS_MATCHED_ONLY", "issues": []}
    assert "PASS" not in str(result)


@pytest.mark.parametrize(
    ("key", "value", "reason"),
    [
        ("User", "cryptohunter-root-proof-signer", "User_NOT_MATCHED"),
        ("Group", "cryptohunter-history-attestation-signer", "Group_NOT_MATCHED"),
        ("ActiveState", "inactive", "ActiveState_NOT_MATCHED"),
        ("LoadState", "not-found", "LoadState_NOT_MATCHED"),
        ("DynamicUser", "yes", "DynamicUser_NOT_MATCHED"),
        ("MainPID", "0", "MAIN_PID_NOT_OBSERVED"),
        ("MainPID", "unknown", "MAIN_PID_NOT_OBSERVED"),
        ("PrivateNetwork", "no", "PrivateNetwork_NOT_MATCHED"),
        ("PrivateIPC", "no", "PrivateIPC_NOT_MATCHED"),
        ("PrivateMounts", "no", "PrivateMounts_NOT_MATCHED"),
        ("NoNewPrivileges", "no", "NoNewPrivileges_NOT_MATCHED"),
        ("RestrictAddressFamilies", "AF_INET AF_UNIX", "RestrictAddressFamilies_NOT_MATCHED"),
        ("DropInPaths", "/etc/systemd/system/unit.d/override.conf", "DropInPaths_NOT_MATCHED"),
        ("SupplementaryGroups", "wheel", "SupplementaryGroups_NOT_MATCHED"),
        ("FragmentPath", "relative/unit.service", "UNIT_FRAGMENT_NOT_ABSOLUTE"),
    ],
)
def test_rpi23_refuses_mismatched_systemd_observations(key, value, reason):
    unit = "cryptohunter-root-custody-source.service"
    record = _systemd_show_fixture(unit)
    record[key] = value
    raw = "".join(f"{field}={record[field]}\n" for field in preflight._SYSTEMD_PROPERTIES)
    observation = preflight._evaluate_systemd_show(raw, unit=unit, account="ch_root_custody_source")
    assert observation["observation"] == "NOT_QUALIFIED"
    assert reason in observation["issues"]


@pytest.mark.parametrize(
    ("mutation", "expected"),
    [
        ("", "INCOMPLETE_SYSTEMD_RESPONSE"),
        ("Bogus=1\n", "UNEXPECTED_OR_DUPLICATE_PROPERTY"),
        ("Id=other\n", "UNEXPECTED_OR_DUPLICATE_PROPERTY"),
        ("NoNewPrivileges\n", "MALFORMED_SYSTEMD_RESPONSE"),
        ("\x00", "MALFORMED_SYSTEMD_RESPONSE"),
    ],
)
def test_rpi23_rejects_malformed_systemd_show(mutation, expected):
    unit = "cryptohunter-root-custody-source.service"
    original = _systemd_show_fixture(unit)
    raw = "".join(f"{field}={original[field]}\n" for field in preflight._SYSTEMD_PROPERTIES)
    if not mutation:
        raw = raw.replace("User=ch_root_custody_source\n", "")
    else:
        raw += mutation
    result = preflight._evaluate_systemd_show(raw, unit=unit, account="ch_root_custody_source")
    assert result["observation"] == "NOT_QUALIFIED"
    assert expected in result["issues"]


@pytest.mark.skipif(sys.platform != "linux", reason="Fixed Linux systemctl probe")
def test_rpi23_uses_only_fixed_readonly_systemctl_queries():
    import subprocess

    calls = []

    # One invocation covers all three roles.
    def fixed_runner(command, **kwargs):
        unit = command[6]
        account = next(
            name
            for _role, known_unit, name in preflight._PROPOSED_SYSTEMD_UNITS
            if known_unit == unit
        )
        calls.append((command, kwargs))
        assert command[0] == "/usr/bin/systemctl"
        assert command[1:5] == ("--system", "show", "--all", "--no-pager")
        assert kwargs["timeout"] == 3
        data = _systemd_show_fixture(unit, account)
        return SimpleNamespace(
            returncode=0,
            stdout="".join(f"{field}={data[field]}\n" for field in preflight._SYSTEMD_PROPERTIES),
        )

    observations = preflight._observe_proposed_systemd_units(root=Path("/"), runner=fixed_runner)
    assert len(calls) == 3
    assert set(observations) == {"root_source", "history_source", "cca"}
    assert all(
        value["observation"] == "PROPOSED_DECLARATIONS_MATCHED_ONLY"
        for value in observations.values()
    )


@pytest.mark.skipif(sys.platform != "linux", reason="Fixed Linux systemctl probe")
def test_rpi23_test_fixtures_do_not_call_host_systemctl(tmp_path):
    def forbidden(*args, **kwargs):
        pytest.fail("fixture unexpectedly called systemctl")

    for root in (tmp_path, Path("/")):
        result = preflight._observe_proposed_systemd_units(
            root=root, runner=forbidden if root != Path("/") else None
        )
        assert {value["observation"] for value in result.values()} == {"NOT_OBSERVED_TEST_FIXTURE"}


@pytest.mark.skipif(sys.platform != "linux", reason="Linux systemctl absence")
def test_rpi23_systemctl_failure_is_not_approved():
    import subprocess

    def timeout(*args, **kwargs):
        raise subprocess.TimeoutExpired(cmd="systemctl", timeout=3)

    observed = preflight._observe_proposed_systemd_units(root=Path("/"), runner=timeout)
    assert all(v["observation"] == "NOT_QUALIFIED" for v in observed.values())
    assert all(v["issues"] == ["SYSTEMD_UNAVAILABLE"] for v in observed.values())


@pytest.fixture
def rpi23_host_metadata(monkeypatch):
    accounts = {**_PROPOSED_ACCOUNTS, **_REFERENCE_ACCOUNTS}
    monkeypatch.setattr(preflight, "_native_accounts", lambda: accounts)
    monkeypatch.setattr(
        preflight,
        "_inspect_directory",
        lambda *args, **kwargs: ("MATCHED_PROPOSED_METADATA_ONLY", "no private contents inspected"),
    )
    return accounts


@pytest.mark.skipif(sys.platform != "linux", reason="Linux host systemd observation")
@pytest.mark.parametrize("role,unit,account", preflight._PROPOSED_SYSTEMD_UNITS)
@pytest.mark.parametrize(
    ("failure", "reason"),
    [
        ("wrong-user", "User_NOT_MATCHED"),
        ("public-network", "PrivateNetwork_NOT_MATCHED"),
        ("missing-private-network", "INCOMPLETE_SYSTEMD_RESPONSE"),
        ("unavailable", "SYSTEMD_UNAVAILABLE"),
    ],
)
def test_rpi23_host_systemd_failures_reach_main_report(
    monkeypatch, rpi23_host_metadata, role, unit, account, failure, reason
):
    import subprocess

    calls = []

    def observe(command, **kwargs):
        observed_unit = command[6]
        calls.append(observed_unit)
        observed_account = next(
            name
            for _role, known_unit, name in preflight._PROPOSED_SYSTEMD_UNITS
            if known_unit == observed_unit
        )
        record = _systemd_show_fixture(observed_unit, observed_account)
        returncode = 0
        if observed_unit == unit:
            assert observed_account == account
            if failure == "wrong-user":
                record["User"] = "cryptohunter-root-proof-signer"
            elif failure == "public-network":
                record["PrivateNetwork"] = "no"
            elif failure == "missing-private-network":
                del record["PrivateNetwork"]
            else:
                returncode = 1
        return SimpleNamespace(
            returncode=returncode,
            stdout="".join(f"{field}={value}\n" for field, value in record.items()),
        )

    monkeypatch.setattr(subprocess, "run", observe)
    result = preflight.inspect_host()
    assert calls == [
        known_unit for _role, known_unit, _account in preflight._PROPOSED_SYSTEMD_UNITS
    ]
    assert result["observed_failures"] == [f"{role}.systemd: {reason}"]
    assert result["systemd_observations"][role] == {
        "unit": unit,
        "observation": "NOT_QUALIFIED",
        "issues": [reason],
    }
    assert all(
        observation["observation"] == "PROPOSED_DECLARATIONS_MATCHED_ONLY"
        for observed_role, observation in result["systemd_observations"].items()
        if observed_role != role
    )
    assert result["status"] == result["production_emission"] == "BLOCKED"
    assert result["qualification"] == "NOT_QUALIFIED"
    assert result["topology"] == "PROPOSED_NOT_APPROVED"
    assert result["blockers"] == list(preflight._BLOCKERS)


@pytest.mark.skipif(sys.platform != "linux", reason="Linux fixture systemd isolation")
@pytest.mark.parametrize(
    ("native_root", "fixture_accounts"),
    [(False, True), (True, True), (False, False)],
    ids=[
        "fixture-root-and-accounts",
        "native-root-fixture-accounts",
        "fixture-root-native-accounts",
    ],
)
def test_rpi23_inspect_host_fixtures_do_not_observe_or_report_host_systemd(
    tmp_path, monkeypatch, rpi23_host_metadata, native_root, fixture_accounts
):
    import subprocess

    def forbidden(*args, **kwargs):
        pytest.fail("fixture unexpectedly called host systemctl")

    monkeypatch.setattr(subprocess, "run", forbidden)
    result = preflight.inspect_host(
        root=Path("/") if native_root else tmp_path,
        accounts=rpi23_host_metadata if fixture_accounts else None,
    )
    assert result["observed_failures"] == []
    assert {value["observation"] for value in result["systemd_observations"].values()} == {
        "NOT_OBSERVED_TEST_FIXTURE"
    }
    assert result["status"] == result["production_emission"] == "BLOCKED"
    assert result["qualification"] == "NOT_QUALIFIED"


@pytest.mark.skipif(sys.platform != "linux", reason="Linux systemctl decoding failure")
def test_rpi23_malformed_systemctl_encoding_fails_closed_without_traceback(
    monkeypatch, rpi23_host_metadata, capsys
):
    import subprocess

    calls = []

    def invalid_encoding(command, **kwargs):
        calls.append(command[6])
        raise UnicodeDecodeError("utf-8", b"private-response-\xff", 17, 18, "invalid host response")

    monkeypatch.setattr(subprocess, "run", invalid_encoding)
    assert preflight.main() == 2
    captured = capsys.readouterr()
    result = json.loads(captured.out)
    assert captured.err == ""
    assert "private-response" not in captured.out
    assert "invalid host response" not in captured.out
    assert "UnicodeDecodeError" not in captured.out
    assert calls == [unit for _role, unit, _account in preflight._PROPOSED_SYSTEMD_UNITS]
    assert result["observed_failures"] == [
        f"{role}.systemd: MALFORMED_SYSTEMD_RESPONSE"
        for role, _unit, _account in preflight._PROPOSED_SYSTEMD_UNITS
    ]
    assert all(
        value["observation"] == "NOT_QUALIFIED"
        and value["issues"] == ["MALFORMED_SYSTEMD_RESPONSE"]
        for value in result["systemd_observations"].values()
    )
    assert result["status"] == result["production_emission"] == "BLOCKED"
    assert result["qualification"] == "NOT_QUALIFIED"
    assert result["topology"] == "PROPOSED_NOT_APPROVED"
    assert result["blockers"] == list(preflight._BLOCKERS)
