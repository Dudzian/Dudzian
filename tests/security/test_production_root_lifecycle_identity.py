"""Installation/OS negative paths; mocks never qualify a real deployment."""

from __future__ import annotations

import json
import os
import stat
from dataclasses import asdict
from types import SimpleNamespace

import pytest

from bot_core import production_root_lifecycle as boundary
from bot_core.postgresql_atomic_root_proof_qualification import (
    PostgreSQLAtomicRootProofPhysicalIdentity,
)
from bot_core.postgresql_production_root_lifecycle_schema import (
    PostgreSQLProductionRootLifecycleProvisioning,
)


@pytest.fixture
def installed(monkeypatch):
    roles = sorted(boundary._PRINCIPALS)
    config = PostgreSQLProductionRootLifecycleProvisioning(
        "prod_lifecycle",
        "prod_owner",
        "prod_issuer",
        "prod_administrator",
        "installation-1",
        "production.example",
    )
    physical = asdict(
        PostgreSQLAtomicRootProofPhysicalIdentity(
            12345,
            123,
            "production",
            "prod-cluster",
            "/run/postgresql",
            5432,
            None,
            None,
            "installation-1",
        )
    )
    physical["system_identifier"] = str(physical["system_identifier"])
    principals = {
        role: {
            "unix_user": "ch_" + role,
            "uid": 1000 + i,
            "gid": 2000 + i,
            "postgresql_role": "prod_" + role,
            "postgresql_oid": 3000 + i,
        }
        for i, role in enumerate(roles)
    }
    value = {
        "schema_version": 1,
        "profile": "PRODUCTION_LOCAL",
        "provisioning": asdict(config),
        "identity": physical,
        "catalog_sha256": "a" * 64,
        "role_oids": [
            ["prod_owner", 9000],
            ["prod_issuer", principals["issuer"]["postgresql_oid"]],
            ["prod_administrator", principals["administrator"]["postgresql_oid"]],
        ],
        "principals": principals,
        "signer_sockets": {
            "root_signer": "/run/cryptohunter/root.sock",
            "history_signer": "/run/cryptohunter/history.sock",
        },
    }
    import pwd

    def account(username):
        selected = next(p for p in principals.values() if p["unix_user"] == username)
        return SimpleNamespace(pw_uid=selected["uid"], pw_gid=selected["gid"])

    monkeypatch.setattr(pwd, "getpwnam", account)
    monkeypatch.setattr(os, "getgrouplist", lambda name, gid: [gid])
    monkeypatch.setattr(boundary, "_protected_bytes", lambda path: json.dumps(value).encode())
    return value


def test_protected_installation_rejects_actual_untrusted_owner(tmp_path):
    path = tmp_path / "installation.json"
    path.write_text("{}")
    with pytest.raises(
        boundary.ProductionRootLifecycleError, match="PROTECTED_INSTALLATION_REQUIRED"
    ):
        boundary._protected_bytes(path)


@pytest.mark.parametrize("changed_field", ["st_atime_ns", "st_mtime_ns"])
def test_protected_read_distinguishes_its_own_access_time_from_file_changes(
    tmp_path, monkeypatch, changed_field
):
    path = tmp_path / "installation.json"
    path.write_bytes(b"bounded-installation")
    actual_fstat = os.fstat
    observations = []

    def installed_metadata(descriptor):
        actual = actual_fstat(descriptor)
        values = {name: getattr(actual, name) for name in dir(actual) if name.startswith("st_")}
        # Only installation ownership is simulated here. File I/O, descriptors,
        # inode, size and timestamps use the real isolated filesystem.
        values["st_uid"] = 0
        values["st_mode"] &= ~0o022
        if stat.S_ISREG(actual.st_mode):
            observations.append(descriptor)
            if len(observations) > 1:
                values[changed_field] += 1
        return SimpleNamespace(**values)

    monkeypatch.setattr(os, "fstat", installed_metadata)
    if changed_field == "st_atime_ns":
        assert boundary._protected_bytes(path) == b"bounded-installation"
    else:
        with pytest.raises(boundary.ProductionRootLifecycleError, match="CHANGED_DURING_READ"):
            boundary._protected_bytes(path)


@pytest.mark.parametrize(
    "change",
    [
        "test_profile",
        "shared_uid",
        "shared_material_role",
        "boolean_uid",
        "extra_field",
        "relative_socket",
        "same_socket",
    ],
)
def test_installed_inputs_fail_closed(installed, change):
    if change == "test_profile":
        installed["profile"] = "TEST_ONLY"
    elif change == "shared_uid":
        installed["principals"]["root_signer"]["uid"] = installed["principals"]["history_signer"][
            "uid"
        ]
    elif change == "shared_material_role":
        installed["principals"]["root_signer"]["postgresql_role"] = installed["principals"][
            "issuer"
        ]["postgresql_role"]
    elif change == "boolean_uid":
        installed["principals"]["issuer"]["uid"] = True
    elif change == "extra_field":
        installed["requester_override"] = "caller-controlled"
    elif change == "relative_socket":
        installed["signer_sockets"]["root_signer"] = "relative.sock"
    else:
        installed["signer_sockets"]["root_signer"] = installed["signer_sockets"]["history_signer"]
    with pytest.raises(boundary.ProductionRootLifecycleError):
        boundary._installed_boundary()


def test_duplicate_installation_fields_rejected(monkeypatch):
    monkeypatch.setattr(
        boundary,
        "_protected_bytes",
        lambda path: b'{"profile":"PRODUCTION_LOCAL","profile":"TEST_ONLY"}',
    )
    with pytest.raises(boundary.ProductionRootLifecycleError):
        boundary._installed_boundary()


def test_inbound_supplementary_service_group_rejected(installed, monkeypatch):
    issuer_gid = installed["principals"]["issuer"]["gid"]
    monkeypatch.setattr(os, "getgrouplist", lambda name, gid: [gid, issuer_gid])
    with pytest.raises(boundary.ProductionRootLifecycleError):
        boundary._installed_boundary()


def test_wrong_real_uid_rejected(installed, monkeypatch):
    observed = boundary._installed_boundary()
    issuer = observed.principal("issuer")
    monkeypatch.setattr(os, "getuid", lambda: issuer[1] + 100)
    monkeypatch.setattr(os, "geteuid", lambda: issuer[1])
    monkeypatch.setattr(os, "getgid", lambda: issuer[2])
    monkeypatch.setattr(os, "getegid", lambda: issuer[2])
    with pytest.raises(boundary.ProductionRootLifecycleError, match="PROCESS_IDENTITY"):
        boundary._require_process(observed, "issuer")


def test_installed_config_cannot_change_after_context_creation(installed, monkeypatch):
    observed = boundary._installed_boundary()
    authority = boundary.ProductionRootLifecycleBoundary(observed, "issuer")
    installed["catalog_sha256"] = "b" * 64
    with pytest.raises(boundary.ProductionRootLifecycleError, match="BOUNDARY_CHANGED"):
        authority._current()


def test_production_bind_has_no_test_credential_bypass(installed):
    authority = boundary.ProductionRootLifecycleBoundary(boundary._installed_boundary(), "issuer")
    with pytest.raises(boundary.ProductionRootLifecycleError, match="PRODUCTION_BIND_BLOCKED"):
        authority.commit(profile="TEST_ONLY", attempt="caller-selected")


@pytest.mark.parametrize("role", ["issuer", "administrator"])
@pytest.mark.parametrize("receipt", [b"{}", b"a" * 64, b"", None])
def test_completion_has_no_python_receipt_or_role_bypass(installed, monkeypatch, role, receipt):
    authority = boundary.ProductionRootLifecycleBoundary(boundary._installed_boundary(), role)
    calls = []
    monkeypatch.setattr(
        boundary.ProductionRootLifecycleBoundary,
        "_administrative_operation",
        lambda *args: calls.append(args),
    )
    with pytest.raises(
        boundary.ProductionRootLifecycleError,
        match="INDEPENDENT_CUSTODY_COMPLETION_AUTHORITY_REQUIRED",
    ):
        authority.publish_administrator_completion(None, "close-1", "a" * 64, receipt)
    assert calls == []
    assert boundary.PRODUCTION_CUSTODY_COMPLETION == "BLOCKED"
    assert boundary.PRODUCTION_EMISSION == "BLOCKED"


def test_factory_accepts_no_installation_override():
    with pytest.raises(TypeError):
        boundary.open_installed_production_root_lifecycle(path="/tmp/caller.json")


@pytest.mark.parametrize(
    "reported", [None, "trust:ch_issuer", "peer:ch_claimant", "scram-sha-256:prod_issuer"]
)
def test_socket_connection_alone_cannot_prove_installed_peer(installed, reported):
    observed = boundary._installed_boundary()
    authority = boundary.ProductionRootLifecycleBoundary(observed, "issuer")
    connection = SimpleNamespace(
        execute=lambda query: SimpleNamespace(fetchone=lambda: (reported,))
    )
    with pytest.raises(boundary.ProductionRootLifecycleError, match="PEER_AUTHENTICATION"):
        authority._require_peer_authentication(connection, observed)


def test_installed_peer_must_match_original_system_user(installed):
    observed = boundary._installed_boundary()
    authority = boundary.ProductionRootLifecycleBoundary(observed, "issuer")
    connection = SimpleNamespace(
        execute=lambda query: SimpleNamespace(fetchone=lambda: ("peer:ch_issuer",))
    )
    authority._require_peer_authentication(connection, observed)


@pytest.mark.parametrize("arguments", [[], ["--check", "--installation", "/tmp/override.json"]])
def test_service_cli_rejects_missing_check_or_caller_configuration(arguments, monkeypatch):
    from deployment import production_root_proof_issuer as service

    monkeypatch.setattr("sys.argv", ["issuer-check", *arguments])
    monkeypatch.setattr(
        service,
        "open_installed_production_root_lifecycle",
        lambda: pytest.fail("invalid CLI must not open installed authority"),
    )
    with pytest.raises(SystemExit) as result:
        service.main()
    assert result.value.code == 2


def test_service_cli_qualifies_without_exposing_emission(monkeypatch, capsys):
    from deployment import production_root_proof_issuer as service

    calls = []
    monkeypatch.setattr("sys.argv", ["issuer-check", "--check"])
    monkeypatch.setattr(
        service,
        "open_installed_production_root_lifecycle",
        lambda: SimpleNamespace(qualify_installed_service=lambda: calls.append("qualified")),
    )
    assert service.main() == 0
    assert calls == ["qualified"]
    assert "PRODUCTION_ROOT_PROOF_EMISSION=BLOCKED" in capsys.readouterr().out
