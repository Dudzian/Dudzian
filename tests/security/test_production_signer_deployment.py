"""Deployment policy and CLI tests; these do not attest physical OS isolation."""

from __future__ import annotations

from configparser import ConfigParser
import json
from pathlib import Path
import shlex
import signal
from types import SimpleNamespace

import pytest

from deployment import production_signer_service as cli


_ROOT = Path(__file__).resolve().parents[2]
_SERVICES = {
    "root_signer": "root-proof-signer",
    "history_signer": "history-attestation-signer",
}


@pytest.mark.parametrize("role", _SERVICES)
def test_units_enforce_distinct_diagnostic_principals_and_namespaces(role):
    service = _SERVICES[role]
    other = next(name for name in _SERVICES.values() if name != service)
    # systemd permits repeated Unit conditions (all must pass).
    unit = ConfigParser(interpolation=None, strict=False)
    unit.read(_ROOT / "deployment/systemd" / f"cryptohunter-{service}.service")
    assert set(unit.sections()) == {"Unit", "Service"}
    settings = unit["Service"]
    assert settings["User"] == settings["Group"] == f"cryptohunter-{service}"
    assert shlex.split(settings["ExecStart"])[-2:] == [role, "--serve"]
    assert settings["Restart"] == "no"
    for flag in (
        "NoNewPrivileges",
        "PrivateMounts",
        "PrivateIPC",
        "PrivateNetwork",
        "PrivateTmp",
        "RestrictNamespaces",
        "RestrictSUIDSGID",
    ):
        assert settings[flag] == "yes"
    assert settings["ProtectSystem"] == "strict"
    assert settings["ProtectHome"] == "yes"
    assert settings["CapabilityBoundingSet"] == settings["AmbientCapabilities"] == ""
    assert settings["RestrictAddressFamilies"] == "AF_UNIX"
    assert settings["ReadOnlyPaths"] == f"/var/lib/cryptohunter/{service}"
    assert settings["ReadWritePaths"] == f"/run/cryptohunter/{service}"
    assert set(settings["InaccessiblePaths"].split()) == {
        f"/var/lib/cryptohunter/{other}",
        f"/etc/cryptohunter/{other}",
    }
    assert settings["UMask"] == "0117"
    assert "SupplementaryGroups" not in settings
    assert not (_ROOT / "deployment/systemd" / f"cryptohunter-{service}.socket").exists()


@pytest.mark.parametrize("service", _SERVICES.values())
def test_units_load_mandatory_hardware_identity_from_fixed_kernel_source(service):
    path = _ROOT / "deployment/systemd" / f"cryptohunter-{service}.service"
    credential_settings = [
        line.strip()
        for line in path.read_text().splitlines()
        if line.lstrip().startswith(
            (
                "LoadCredential=",
                "LoadCredentialEncrypted=",
                "SetCredential=",
                "SetCredentialEncrypted=",
                "ImportCredential=",
            )
        )
    ]
    assert credential_settings == [
        "LoadCredential=custody-product-uuid:/sys/devices/virtual/dmi/id/product_uuid"
    ]


def test_runtime_directories_inherit_socket_group_without_shared_process_group():
    records = {}
    path = _ROOT / "deployment/tmpfiles.d/cryptohunter-production-signers.conf"
    for line in path.read_text().splitlines():
        fields = shlex.split(line, comments=True)
        if fields:
            records[fields[1]] = fields
    for service in _SERVICES.values():
        runtime = records[f"/run/cryptohunter/{service}"]
        assert runtime[:5] == [
            "d",
            f"/run/cryptohunter/{service}",
            "2710",
            f"cryptohunter-{service}",
            "cryptohunter-root-proof-issuer",
        ]
        custody = records[f"/var/lib/cryptohunter/{service}"]
        assert custody[2:5] == ["0700", f"cryptohunter-{service}", f"cryptohunter-{service}"]
        config = records[f"/etc/cryptohunter/{service}"]
        assert config[2:5] == ["0750", "root", f"cryptohunter-{service}"]
    users = [
        shlex.split(line, comments=True)
        for line in (_ROOT / "deployment/sysusers.d/cryptohunter-production-signers.conf")
        .read_text()
        .splitlines()
        if line and not line.startswith("#")
    ]
    assert {record[0] for record in users} == {"u"}
    assert {record[1] for record in users} == {
        "cryptohunter-root-proof-signer",
        "cryptohunter-history-attestation-signer",
    }
    assert all(record[-1] == "/usr/sbin/nologin" for record in users)


@pytest.mark.parametrize("role,service", _SERVICES.items())
def test_custody_examples_require_review_and_select_separate_credentials(role, service):
    config = json.loads((_ROOT / "deployment/examples" / f"{service}-custody.json").read_text())
    assert set(config) == {
        "schema_version",
        "profile",
        "trust_domain",
        "service_role",
        "custody_directory",
        "keyring_index_path",
    }
    assert config["schema_version"] == 1
    assert config["profile"] == "PRODUCTION_LOCAL"
    assert config["service_role"] == role
    assert config["trust_domain"].startswith("REPLACE_WITH_")
    own = Path(f"/var/lib/cryptohunter/{service}")
    assert Path(config["custody_directory"]) == own
    assert Path(config["keyring_index_path"]).parent == own


@pytest.mark.parametrize(
    "arguments",
    [
        [],
        ["requester", "--check"],
        ["root_signer"],
        ["root_signer", "--check", "--serve"],
        ["root_signer", "--check", "--socket", "/tmp/requester.sock"],
        ["root_signer", "--check", "--config", "/tmp/requester.json"],
        ["root_signer", "--check", "--sign", "payload"],
        ["root_signer", "--check", "--generate-key"],
    ],
)
def test_cli_rejects_callers_selecting_capabilities(arguments, monkeypatch):
    def forbidden(_role):
        pytest.fail("invalid arguments reached installed service loader")

    monkeypatch.setattr(cli, "open_installed_production_signer_policy", forbidden)
    with pytest.raises(SystemExit) as raised:
        cli.main(arguments)
    assert raised.value.code == 2


@pytest.mark.parametrize("role", _SERVICES)
def test_check_reports_blocked_and_never_starts_listener(role, monkeypatch, capsys):
    loaded = []
    qualified = []
    monkeypatch.setattr(cli, "open_installed_production_signer_policy", loaded.append)

    def qualify(selected):
        qualified.append(selected)
        return SimpleNamespace(
            public_facts=lambda: {"service_role": selected, "private_control_status": "BLOCKED"},
            blockers=("INDEPENDENT_PRIVATE_CONTROL_ATTESTATION_REQUIRED",),
        )

    monkeypatch.setattr(cli, "qualify_installed_signer_custody", qualify)
    monkeypatch.setattr(
        cli, "run_diagnostic_server", lambda *_: pytest.fail("check started server")
    )
    assert cli.main([role, "--check"]) == 2
    assert loaded == qualified == [role]
    result = json.loads(capsys.readouterr().out)
    assert result["status"] == "BLOCKED"
    assert result["facts"]["service_role"] == role
    assert result["blockers"] == ["INDEPENDENT_PRIVATE_CONTROL_ATTESTATION_REQUIRED"]


def test_server_callback_keeps_installed_role_and_reads_fresh_custody(monkeypatch):
    policy = object()
    operations = []
    monkeypatch.setattr(cli, "open_installed_production_signer_policy", lambda role: policy)

    def qualify(role):
        operations.append(role)
        return SimpleNamespace(public_facts=lambda: {"observation": len(operations)}, blockers=())

    monkeypatch.setattr(cli, "qualify_installed_signer_custody", qualify)

    prior_handlers = {
        selected: signal.getsignal(selected) for selected in (signal.SIGTERM, signal.SIGINT)
    }

    def server(selected, callback, *, stop_event):
        assert selected is policy
        assert callback("DIAGNOSTIC")["facts"] == {"observation": 1}
        assert callback("QUALIFY")["facts"] == {"observation": 2}
        with pytest.raises(cli.ProductionSignerCustodyError, match="DIAGNOSTIC_OPERATION"):
            callback("SIGN")
        assert not stop_event.is_set()
        signal.getsignal(signal.SIGTERM)(signal.SIGTERM, None)
        assert stop_event.is_set()

    monkeypatch.setattr(cli, "run_diagnostic_server", server)
    assert cli.main(["root_signer", "--serve"]) == 0
    assert operations == ["root_signer", "root_signer"]
    assert {selected: signal.getsignal(selected) for selected in prior_handlers} == prior_handlers


@pytest.mark.parametrize("stage", ["installation", "custody", "server"])
def test_unavailable_dependencies_fail_closed_without_exposing_native_errors(
    stage, monkeypatch, capsys
):
    def unavailable(*_args, **_kwargs):
        raise OSError("sensitive native backend error")

    monkeypatch.setattr(cli, "open_installed_production_signer_policy", lambda role: object())
    monkeypatch.setattr(cli, "qualify_installed_signer_custody", unavailable)
    monkeypatch.setattr(cli, "run_diagnostic_server", unavailable)
    if stage == "installation":
        monkeypatch.setattr(cli, "open_installed_production_signer_policy", unavailable)
    assert cli.main(["root_signer", "--serve" if stage == "server" else "--check"]) == 2
    output = capsys.readouterr().out
    assert json.loads(output)["status"] == "BLOCKED"
    assert "sensitive" not in output
