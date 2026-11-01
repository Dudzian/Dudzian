"""Hermetyczne fakty testowe; żaden test nie kwalifikuje fizycznego hosta."""

from __future__ import annotations

import copy
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any
from types import SimpleNamespace

import pytest

from deployment import linux_physical_collector as collector
from deployment import linux_physical_native as native
from deployment.linux_physical_evidence import PHASES, digest
from deployment.linux_physical_observations import ObservationError, validate_observations

OLD_BOOT = "00000000-0000-4000-8000-000000000001"
NEW_BOOT = "00000000-0000-4000-8000-000000000002"


def installation_fixture(tmp_path: Path) -> dict[str, Any]:
    """Zamknięty kontrakt syntetyczny do testów producenta i weryfikatora."""
    return {
        "schema_version": 1,
        "purpose": "STAGE9_DISPOSABLE_PHYSICAL_HOST",
        "source_root": str(tmp_path),
        "package_name": "cryptohunter-qualification",
        "packages": {
            kind: {
                "path": str(tmp_path / (kind + ".deb")),
                "sha256": str(index + 1) * 64,
                "version": str(index + 1) + ".0",
            }
            for index, kind in enumerate(("baseline", "update", "failed_update"))
        },
        "identities": {
            "runtime": {"principal": "ch_runtime", "uid": 61001, "gid": 61001},
            "verifier": {"principal": "ch_verifier", "uid": 61002, "gid": 61002},
            "administrator": {"principal": "root", "uid": 0, "gid": 0},
        },
        "units": [
            {
                "name": "ch-qual-" + role + ".service",
                "role": role,
                "executable": "/opt/ch-qualification/" + role,
                "executable_sha256": "4" * 64,
                "update_executable_sha256": "5" * 64,
                "unit_file": {
                    "path": "/etc/systemd/system/ch-qual-" + role + ".service",
                    "sha256": "6" * 64,
                },
            }
            for role in ("runtime", "verifier")
        ],
        "configuration": [
            {
                "path": "/etc/ch-qualification/config.json",
                "sha256": "7" * 64,
                "uid": 0,
                "gid": 0,
                "mode": 0o600,
            }
        ],
        "ipc": [
            {
                "path": "/run/ch-qualification/verifier.sock",
                "uid": 61002,
                "gid": 61002,
                "mode": 0o660,
            }
        ],
        "state_directory": "/var/lib/cryptohunter-stage9-qualification/fixture",
        "clean_paths": [
            "/var/lib/cryptohunter-stage9-qualification/fixture",
            "/etc/systemd/system/ch-qual-runtime.service",
            "/etc/systemd/system/ch-qual-verifier.service",
        ],
        "postgres": {
            "socket": "/run/ch-qualification/postgres",
            "port": 5432,
            "database": "ch_qualification",
            "user": "postgres",
            "expected_system_user": "peer:root",
            "expected_hba_sha256": "8" * 64,
            "expected_ident_sha256": "9" * 64,
        },
    }


def _file(
    path: str, digest_value: str, *, uid: int = 0, gid: int = 0, mode: int = 0o644
) -> dict[str, Any]:
    return {
        "path": path,
        "kind": "regular",
        "sha256": digest_value,
        "uid": uid,
        "gid": gid,
        "mode": mode,
    }


def _snapshot(
    installation: dict[str, Any], *, step: int, updated: bool = False, boot: str = OLD_BOOT
) -> dict[str, Any]:
    units = []
    for index, spec in enumerate(installation["units"]):
        identity = installation["identities"][spec["role"]]
        pid = 4000 + step * 10 + index
        executable_hash = spec["update_executable_sha256" if updated else "executable_sha256"]
        props = {
            "Id": spec["name"],
            "LoadState": "loaded",
            "ActiveState": "active",
            "SubState": "running",
            "UnitFileState": "enabled",
            "MainPID": str(pid),
            "User": identity["principal"],
            "Group": identity["principal"],
            "FragmentPath": spec["unit_file"]["path"],
            "ControlGroup": "/system.slice/" + spec["name"],
            "Restart": "on-failure",
            "KillMode": "control-group",
            "NoNewPrivileges": "yes",
            "ProtectSystem": "strict",
            "ProtectHome": "yes",
            "PrivateTmp": "yes",
            "CapabilityBoundingSet": "",
            "AmbientCapabilities": "",
            "RestrictAddressFamilies": "AF_UNIX",
            "NRestarts": "1" if step >= 4 else "0",
        }
        units.append(
            {
                "name": spec["name"],
                "properties": props,
                "processes": [
                    {
                        "pid": pid,
                        "start_time_ticks": 10000 + step,
                        "uid": identity["uid"],
                        "gid": identity["gid"],
                        "groups": [],
                        "no_new_privs": 1,
                        "exe_sha256": executable_hash,
                        "capabilities": {
                            key: "0000000000000000"
                            for key in ("CapInh", "CapPrm", "CapEff", "CapBnd", "CapAmb")
                        },
                    }
                ],
                "unit_file": _file(spec["unit_file"]["path"], spec["unit_file"]["sha256"]),
                "executable": _file(spec["executable"], executable_hash, mode=0o755),
                "journal": {
                    "boot_id": boot.replace("-", ""),
                    "unit": spec["name"],
                    "realtime_usec": 1000000 + step,
                },
            }
        )
    pg = installation["postgres"]
    state_root = installation["state_directory"]
    return {
        "package": {
            "name": installation["package_name"],
            "version": installation["packages"]["update" if updated else "baseline"]["version"],
            "status": "install ok installed",
        },
        "units": units,
        "configuration": [{**item, "kind": "regular"} for item in installation["configuration"]],
        "ipc": [{**item, "kind": "socket"} for item in installation["ipc"]],
        "postgres": {
            "listen_addresses": "",
            "fsync": "on",
            "synchronous_commit": "on",
            "system_user": pg["expected_system_user"],
            "current_user": pg["user"],
            "session_user": pg["user"],
            "client_addr": None,
            "server_addr": None,
            "ssl": False,
            "hba_sha256": pg["expected_hba_sha256"],
            "ident_sha256": pg["expected_ident_sha256"],
            "system_identifier": "123456789",
        },
        "state": _file(
            state_root + "/.stage9-evidence-" + "a" * 64, "b" * 64, uid=61001, gid=61001, mode=0o600
        ),
        "persistent_journal": {"kind": "directory", "uid": 0},
    }


def observation_fixture(installation: dict[str, Any]) -> list[dict[str, Any]]:
    """Komplet surowych obserwacji; nadal wyłącznie fikstura TEST_ONLY."""
    units = installation["units"]
    return [
        {
            "package": None,
            "paths": [{"path": p, "kind": "absent"} for p in installation["clean_paths"]],
            "units": [{"name": u["name"], "load_state": "not-found"} for u in units],
        },
        {"install_exit_code": 0, "snapshot": _snapshot(installation, step=0)},
        {
            "before": _snapshot(installation, step=1),
            "stopped": [
                {
                    "name": u["name"],
                    "active_state": "inactive",
                    "main_pid": 0,
                    "cgroup_pids": [],
                    "old_processes_alive": [],
                }
                for u in units
            ],
            "started": _snapshot(installation, step=2),
            "restarted": _snapshot(installation, step=3),
            "crashed": _snapshot(installation, step=4),
            "commands": [
                {"action": action, "unit": u["name"], "exit_code": 0}
                for action in ("stop", "start", "restart", "crash")
                for u in units
            ],
        },
        {"snapshot": _snapshot(installation, step=5)},
        {
            "snapshot": _snapshot(installation, step=6, boot=NEW_BOOT),
            "previous_boot_journal": [
                {"boot_id": OLD_BOOT.replace("-", ""), "unit": u["name"], "realtime_usec": 1000005}
                for u in units
            ],
        },
        {
            "before": _snapshot(installation, step=7, boot=NEW_BOOT),
            "install_exit_code": 0,
            "after": _snapshot(installation, step=8, updated=True, boot=NEW_BOOT),
        },
        {
            "before": _snapshot(installation, step=8, updated=True, boot=NEW_BOOT),
            "install_exit_code": 1,
            "after": _snapshot(installation, step=8, updated=True, boot=NEW_BOOT),
        },
    ]


def payload_fixture(installation: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        {
            "phase": phase,
            "challenge_id": "a" * 64,
            "host": {"boot_id": OLD_BOOT if index < 4 else NEW_BOOT},
            "observations": observations,
        }
        for index, (phase, observations) in enumerate(
            zip(PHASES, observation_fixture(installation), strict=True)
        )
    ]


def test_complete_synthetic_observations_are_semantically_consistent(tmp_path: Path) -> None:
    installation = installation_fixture(tmp_path)
    validate_observations(payload_fixture(installation), installation)


@pytest.mark.parametrize(
    "mutation",
    [
        "missing",
        "pass",
        "reboot",
        "state",
        "root",
        "ipc",
        "postgres",
        "orphan",
        "crash",
        "failure",
        "version",
        "configuration",
    ],
)
def test_semantic_failure_is_closed(tmp_path: Path, mutation: str) -> None:
    installation = installation_fixture(tmp_path)
    payloads = payload_fixture(installation)
    if mutation == "missing":
        payloads.pop()
    elif mutation == "pass":
        payloads[1]["observations"] = {"status": "PASS"}
    elif mutation == "reboot":
        payloads[4]["host"]["boot_id"] = OLD_BOOT
    elif mutation == "state":
        payloads[4]["observations"]["snapshot"]["state"]["sha256"] = "c" * 64
    elif mutation == "root":
        payloads[1]["observations"]["snapshot"]["units"][0]["processes"][0]["uid"] = 0
    elif mutation == "ipc":
        payloads[1]["observations"]["snapshot"]["ipc"][0]["mode"] = 0o666
    elif mutation == "postgres":
        payloads[1]["observations"]["snapshot"]["postgres"]["listen_addresses"] = "*"
    elif mutation == "orphan":
        payloads[2]["observations"]["stopped"][0]["cgroup_pids"] = [100]
    elif mutation == "crash":
        payloads[2]["observations"]["crashed"]["units"][0]["properties"]["NRestarts"] = "0"
    elif mutation == "failure":
        payloads[6]["observations"]["install_exit_code"] = 0
    elif mutation == "version":
        payloads[5]["observations"]["after"]["package"]["version"] = "1.0"
    elif mutation == "configuration":
        payloads[1]["observations"]["snapshot"]["configuration"][0]["sha256"] = "c" * 64
    with pytest.raises(ObservationError):
        validate_observations(payloads, installation)


def _challenge(installation: dict[str, Any], identity: dict[str, Any]) -> dict[str, Any]:
    now = datetime.now(timezone.utc)
    value = {
        "schema_version": 1,
        "purpose": "LIVE_PHYSICAL_LINUX",
        "repository": "Dudzian/Dudzian",
        "source_revision": "c" * 40,
        "ci_run_id": "1",
        "ci_run_attempt": "1",
        "host_id": identity["host_id"],
        "policy_sha256": "d" * 64,
        "materials": {
            "source_revision": "c" * 40,
            "collector_sha256": "e" * 64,
            "artifact_sha256": "1" * 64,
            "update_artifact_sha256": "2" * 64,
            "failed_update_artifact_sha256": "3" * 64,
            "configuration_sha256": "7" * 64,
            "installation_contract_sha256": digest(installation),
        },
        "nonce": "f" * 64,
        "created_at_utc": (now - timedelta(minutes=1)).isoformat().replace("+00:00", "Z"),
        "expires_at_utc": (now + timedelta(hours=1)).isoformat().replace("+00:00", "Z"),
    }
    value["challenge_id"] = digest(value)
    return value


def test_producer_refuses_mutation_without_phase_consent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    installation = installation_fixture(tmp_path)

    class ForbiddenDriver:
        def collect(self, *_args):
            pytest.fail("Nie wolno wykonać działania bez zgody")

    with pytest.raises(ObservationError, match="zgody"):
        collector.collect_phase(
            challenge=_challenge(installation, {"host_id": "a" * 64}),
            installation=installation,
            bundle={},
            key_bytes=b"k" * 32,
            phase="clean_install",
            approval=None,
            driver=ForbiddenDriver(),
        )


def test_native_commands_use_no_shell_and_bounded_timeout(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = []

    def run(args, **kwargs):
        calls.append((args, kwargs))
        return object()

    monkeypatch.setattr(native.subprocess, "run", run)
    native.command(["/usr/bin/systemctl", "--system", "show", "safe.service"])
    assert calls[0][1]["timeout"] == 30
    assert calls[0][1].get("shell", False) is False
    assert calls[0][1]["env"] == native.SAFE_ENV
    with pytest.raises(ObservationError):
        native.command(["/bin/sh", "-c", "true"])


def test_symlink_input_is_rejected(tmp_path: Path) -> None:
    target = tmp_path / "raw"
    target.write_bytes(b"safe")
    link = tmp_path / "link"
    link.symlink_to(target)
    with pytest.raises(OSError):
        native.read_regular(link)


def test_bundle_write_is_atomic_and_contains_no_key(tmp_path: Path) -> None:
    destination = tmp_path / "bundle.json"
    value = {"schema_version": 1, "challenge_id": "a" * 64, "records": []}
    collector._atomic_bundle(destination, value)
    assert destination.read_bytes().endswith(b"\n")
    assert not destination.with_name("bundle.json.pending").exists()
    assert "private_key" not in destination.read_text()
    assert destination.stat().st_mode & 0o777 == 0o600


def _identity_fixture(boot: str = OLD_BOOT) -> dict[str, str]:
    return {
        "host_id": "a" * 64,
        "machine_id_sha256": "b" * 64,
        "dmi_product_uuid_sha256": "c" * 64,
        "boot_id": boot,
        "os_id": "ubuntu",
        "os_version": "24.04",
        "kernel": "6.8.0",
        "architecture": "x86_64",
        "virtualization": "none",
        "python_version": "3.11.9",
        "python_executable_sha256": "d" * 64,
    }


def _retarget_state(
    observations: list[dict[str, Any]], installation: dict[str, Any], challenge_id: str
) -> None:
    for observation in observations:
        for key in ("snapshot", "before", "after", "started", "restarted", "crashed"):
            if key in observation:
                observation[key]["state"]["path"] = (
                    installation["state_directory"] + "/.stage9-evidence-" + challenge_id
                )


def test_producer_seven_phases_signature_chain_and_final_semantics(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    installation = installation_fixture(tmp_path)
    current_host = _identity_fixture()
    challenge = _challenge(installation, current_host)
    observations = observation_fixture(installation)
    _retarget_state(observations, installation, challenge["challenge_id"])

    class Driver:
        def collect(self, phase, previous):
            assert phase == PHASES[len(previous)]
            return copy.deepcopy(observations[len(previous)])

    monkeypatch.setattr(collector, "host_identity", lambda: copy.deepcopy(current_host))
    monkeypatch.setattr(collector, "collect_materials", lambda _inst: challenge["materials"])
    bundle = {"schema_version": 1, "challenge_id": challenge["challenge_id"], "records": []}
    for index, phase in enumerate(PHASES):
        if index == 4:
            current_host["boot_id"] = NEW_BOOT
        bundle = collector.collect_phase(
            challenge=challenge,
            installation=installation,
            bundle=bundle,
            key_bytes=b"k" * 32,
            phase=phase,
            approval=challenge["challenge_id"] + ":" + phase,
            driver=Driver(),
        )
    assert len(bundle["records"]) == 7
    assert all(len(record["signature"]) == 88 for record in bundle["records"])
    assert all("PASS" not in str(record["payload"]["observations"]) for record in bundle["records"])


@pytest.mark.parametrize(
    "fault", ["host", "materials", "phase", "baseline", "reboot", "host_drift", "material_drift"]
)
def test_producer_refuses_fault_before_or_after_native_boundary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fault: str
) -> None:
    installation = installation_fixture(tmp_path)
    identity = _identity_fixture()
    challenge = _challenge(installation, identity)
    observation = observation_fixture(installation)[0]
    host_calls = 0
    material_calls = 0

    def host():
        nonlocal host_calls
        host_calls += 1
        result = copy.deepcopy(identity)
        if fault == "host" or (fault == "host_drift" and host_calls > 1):
            result["host_id"] = "e" * 64
        return result

    def materials(_inst):
        nonlocal material_calls
        material_calls += 1
        result = copy.deepcopy(challenge["materials"])
        if fault == "materials" or (fault == "material_drift" and material_calls > 1):
            result["artifact_sha256"] = "f" * 64
        return result

    class Driver:
        def collect(self, *_args):
            result = copy.deepcopy(observation)
            if fault == "baseline":
                result["package"] = {"name": "collision"}
            return result

    monkeypatch.setattr(collector, "host_identity", host)
    monkeypatch.setattr(collector, "collect_materials", materials)
    bundle = {"schema_version": 1, "challenge_id": challenge["challenge_id"], "records": []}
    phase = "clean_install" if fault == "phase" else "baseline"
    if fault == "reboot":
        bundle = collector.collect_phase(
            challenge=challenge,
            installation=installation,
            bundle=bundle,
            key_bytes=b"k" * 32,
            phase="baseline",
            approval=None,
            driver=Driver(),
        )
        identity["boot_id"] = NEW_BOOT
        phase = "clean_install"
    with pytest.raises(ValueError):
        collector.collect_phase(
            challenge=challenge,
            installation=installation,
            bundle=bundle,
            key_bytes=b"k" * 32,
            phase=phase,
            approval=challenge["challenge_id"] + ":" + phase,
            driver=Driver(),
        )


def test_cli_default_is_readonly_diagnostics(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(collector, "host_identity", _identity_fixture)
    monkeypatch.setattr(
        collector, "collect_phase", lambda **_kwargs: pytest.fail("Diagnostyka nie wykonuje faz")
    )
    assert collector.main([]) == 0
    assert "TYLKO_ODCZYT" in capsys.readouterr().out


def test_cli_missing_inputs_refuses_without_leaking_error(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    assert collector.main(["--phase", "clean_install"]) == 1
    assert "ODMOWA" in capsys.readouterr().out


def test_cli_phase_writes_actual_signed_bundle(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import json

    installation = installation_fixture(tmp_path)
    identity = _identity_fixture()
    challenge = _challenge(installation, identity)
    manifest_path, challenge_path = tmp_path / "installation.json", tmp_path / "challenge.json"
    manifest_path.write_text(json.dumps(installation))
    challenge_path.write_text(json.dumps(challenge))
    private_key = tmp_path / "producer.ed25519"
    private_key.write_bytes(b"k" * 32)
    private_key.chmod(0o600)
    destination = tmp_path / "bundle.json"

    class Driver:
        def __init__(self, *_args):
            pass

        def collect(self, *_args):
            return observation_fixture(installation)[0]

    monkeypatch.setattr(collector, "NativeDriver", Driver)
    monkeypatch.setattr(collector, "host_identity", lambda: identity)
    monkeypatch.setattr(collector, "collect_materials", lambda _inst: challenge["materials"])
    monkeypatch.setattr(collector, "read_regular", lambda path, **_kwargs: path.read_bytes())
    original_lstat = Path.lstat

    def protected_key_stat(path, *args, **kwargs):
        result = original_lstat(path, *args, **kwargs)
        return SimpleNamespace(st_uid=0, st_mode=result.st_mode) if path == private_key else result

    monkeypatch.setattr(Path, "lstat", protected_key_stat)
    assert (
        collector.main(
            [
                "--phase",
                "baseline",
                "--challenge",
                str(challenge_path),
                "--manifest",
                str(manifest_path),
                "--bundle",
                str(destination),
                "--private-key",
                str(private_key),
            ]
        )
        == 0
    )
    assert len(json.loads(destination.read_text())["records"]) == 1


def test_collect_materials_hashes_actual_files_and_checkout(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    installation = installation_fixture(tmp_path)
    for relative in collector.CLOSURE:
        target = tmp_path / relative
        target.parent.mkdir(exist_ok=True)
        target.write_bytes(relative.encode())
    for kind, item in installation["packages"].items():
        target = Path(item["path"])
        target.write_bytes((kind + " DEB TEST ONLY").encode())
        item["sha256"] = native.sha256(target.read_bytes())
    configuration = tmp_path / "configuration.json"
    configuration.write_bytes(b'{"mode":"TEST_ONLY"}')
    installation["configuration"][0]["path"] = str(configuration)
    installation["configuration"][0]["sha256"] = native.sha256(configuration.read_bytes())
    monkeypatch.setattr(collector, "__file__", str(tmp_path / collector.CLOSURE[0]))
    monkeypatch.setattr(
        collector, "read_regular", lambda path, **_kwargs: native.read_regular(path)
    )
    monkeypatch.setattr(
        collector, "file_observation", lambda path: {"sha256": native.sha256(path.read_bytes())}
    )
    monkeypatch.setattr(
        collector,
        "command",
        lambda args: SimpleNamespace(
            returncode=0, stdout="c" * 40 + "\n" if args[-1] == "HEAD" else ""
        ),
    )
    result = collector.collect_materials(installation)
    assert result["artifact_sha256"] == native.sha256(
        Path(installation["packages"]["baseline"]["path"]).read_bytes()
    )
    assert result["collector_sha256"] == digest(
        {name: native.sha256((tmp_path / name).read_bytes()) for name in collector.CLOSURE}
    )
    assert result["configuration_sha256"] == digest(
        {str(configuration): native.sha256(configuration.read_bytes())}
    )
    Path(installation["packages"]["baseline"]["path"]).write_bytes(b"PODMIANA")
    with pytest.raises(ObservationError):
        collector.collect_materials(installation)


def test_collect_materials_refuses_dirty_revision(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    installation = installation_fixture(tmp_path)
    monkeypatch.setattr(
        collector, "command", lambda _args: SimpleNamespace(returncode=0, stdout="dirty")
    )
    with pytest.raises(ObservationError, match="czyste"):
        collector.collect_materials(installation)


@pytest.mark.parametrize(
    "location", ["file_uid", "no_new_privs", "journal_uid", "exit_code", "stopped_pid"]
)
def test_boolean_is_not_numeric_observation(tmp_path: Path, location: str) -> None:
    installation = installation_fixture(tmp_path)
    payloads = payload_fixture(installation)
    if location == "file_uid":
        payloads[1]["observations"]["snapshot"]["configuration"][0]["uid"] = False
    elif location == "no_new_privs":
        payloads[1]["observations"]["snapshot"]["units"][0]["processes"][0]["no_new_privs"] = True
    elif location == "journal_uid":
        payloads[1]["observations"]["snapshot"]["persistent_journal"]["uid"] = False
    elif location == "exit_code":
        payloads[2]["observations"]["commands"][0]["exit_code"] = False
    else:
        payloads[2]["observations"]["stopped"][0]["main_pid"] = False
    with pytest.raises(ObservationError):
        validate_observations(payloads, installation)


@pytest.mark.parametrize("fault", ["old_sentinel", "journal_boot", "unchanged_update_process"])
def test_observations_reject_cross_challenge_and_boot_replay(tmp_path: Path, fault: str) -> None:
    installation = installation_fixture(tmp_path)
    payloads = payload_fixture(installation)
    if fault == "old_sentinel":
        for payload in payloads:
            payload["challenge_id"] = "f" * 64
    elif fault == "journal_boot":
        payloads[4]["observations"]["snapshot"]["units"][0]["journal"]["boot_id"] = (
            OLD_BOOT.replace("-", "")
        )
    else:
        payloads[5]["observations"]["after"]["units"][0]["processes"][0]["pid"] = payloads[5][
            "observations"
        ]["before"]["units"][0]["processes"][0]["pid"]
        payloads[5]["observations"]["after"]["units"][0]["properties"]["MainPID"] = str(
            payloads[5]["observations"]["after"]["units"][0]["processes"][0]["pid"]
        )
        payloads[5]["observations"]["after"]["units"][0]["processes"][0]["start_time_ticks"] = (
            payloads[5]["observations"]["before"]["units"][0]["processes"][0]["start_time_ticks"]
        )
    with pytest.raises(ObservationError):
        validate_observations(payloads, installation)
