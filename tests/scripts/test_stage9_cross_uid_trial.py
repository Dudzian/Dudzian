"""Modele bramek kontraktu próby; nie są wynikiem natywnej kwalifikacji.

Modelowane są wyłącznie granice odczytu FS, inspekcji środowiska i istniejącej
zgody administratora. Testy nie zmieniają UID, PID ani sys.executable.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import sys
from typing import Any

import pytest

if sys.platform != "linux":
    pytest.skip("Bramka kontraktu wymaga natywnego Linux.", allow_module_level=True)

from deployment.stage9_vm_environment import VMGuardError
from scripts.diagnostics import stage9_cross_uid_trial as trial


@pytest.fixture
def modeled_trial(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    source = Path(trial.__file__).absolute().parents[2]
    files = {
        source / trial._HARNESS: b"model-zatwierdzonego-harnessu",
        source / trial._CONTRACT: b"model-zatwierdzonego-kontraktu",
    }
    approval = {
        "schema": 1,
        "purpose": "TEST_ONLY",
        "approved": True,
        "revision": "a" * 40,
        "environment_id": "model-noble-001",
        "machine_id": "b" * 32,
        "boot_id": "01234567-89ab-cdef-0123-456789abcdef",
        "vm_image_sha256": "c" * 64,
        "vm_snapshot": "model-S0",
        "packages": {name: "1.0-1" for name in trial.REQUIRED_PACKAGES},
    }
    approval_raw = json.dumps(approval, sort_keys=True).encode()
    manifest = {
        "schema": 1,
        "scope": trial.SCOPE,
        "trial_id": "fedcba98-7654-3210-fedc-ba9876543210",
        **{
            key: approval[key]
            for key in (
                "revision",
                "environment_id",
                "machine_id",
                "boot_id",
                "vm_image_sha256",
                "vm_snapshot",
                "packages",
            )
        },
        "approval_sha256": hashlib.sha256(approval_raw).hexdigest(),
        "source_root": str(source),
        "python_executable": sys.executable,
        "harness_sha256": hashlib.sha256(files[source / trial._HARNESS]).hexdigest(),
        "contract_sha256": hashlib.sha256(files[source / trial._CONTRACT]).hexdigest(),
    }
    facts = {
        "observation_complete": True,
        "os_id": "ubuntu",
        "os_version": "24.04",
        "architecture": "x86_64",
        "pid1": "systemd",
        "python_executable": "/usr/bin/python3.12",
        "python_version": "Python 3.12.3",
        "virtualization": "kvm",
        "container": "none",
        "container_markers": [],
        "shared_mounts": [],
        "network_disconnected": True,
        **{key: copy.deepcopy(manifest[key]) for key in ("machine_id", "boot_id", "packages")},
    }
    calls: list[tuple[str, object]] = []

    def protected_read(path: Path) -> bytes:
        calls.append(("read", path))
        if path == trial.MANIFEST:
            return json.dumps(manifest, sort_keys=True).encode()
        if path == trial.DEFAULT_APPROVAL:
            return approval_raw
        return files[path]

    def source_directory(path: Path) -> None:
        calls.append(("source_directory", path))
        assert path == source

    def interpreter(path: Path) -> None:
        calls.append(("interpreter", path))
        assert str(path) == sys.executable

    def forbidden_rootguard(**_: Any) -> dict[str, Any]:
        raise AssertionError("Odczytowa bramka nie może wywoływać zgody administratora")

    monkeypatch.setattr(trial, "read_protected", protected_read)
    monkeypatch.setattr(trial, "inspect_environment", lambda: copy.deepcopy(facts))
    monkeypatch.setattr(trial, "_protected_directory", source_directory)
    monkeypatch.setattr(trial, "_protected_interpreter", interpreter)
    monkeypatch.setattr(trial, "require_approved_vm", forbidden_rootguard)
    args = argparse.Namespace(
        revision=approval["revision"], environment_id=approval["environment_id"]
    )
    return {
        "manifest": manifest,
        "facts": facts,
        "files": files,
        "approval": approval,
        "approval_raw": approval_raw,
        "calls": calls,
        "args": args,
        "source": source,
    }


def test_readonly_contract_uses_protected_manifest_without_rootguard(
    modeled_trial: dict[str, Any],
) -> None:
    result = trial.require_trial(modeled_trial["args"])
    raw = json.dumps(modeled_trial["manifest"], sort_keys=True).encode()
    assert result["_manifest_sha256"] == hashlib.sha256(raw).hexdigest()
    assert trial.binding(result) == {
        **{
            key: result[key]
            for key in (
                "scope",
                "trial_id",
                "revision",
                "environment_id",
                "machine_id",
                "boot_id",
                "vm_image_sha256",
            )
        },
        "manifest_sha256": hashlib.sha256(raw).hexdigest(),
    }
    assert ("source_directory", modeled_trial["source"]) in modeled_trial["calls"]
    assert ("interpreter", Path(sys.executable)) in modeled_trial["calls"]
    assert ("read", trial.DEFAULT_APPROVAL) not in modeled_trial["calls"]


@pytest.mark.parametrize(
    "raw",
    [
        b'{"a":1,"a":2}',
        b'{"x":{"a":1,"a":2}}',
        b'{"a":NaN}',
        b'{"a":Infinity}',
        b"[]",
        b"{}" * 40000,
        b"\xff",
    ],
)
def test_strict_json_refuses_ambiguous_or_non_object_input(raw: bytes) -> None:
    with pytest.raises(trial.TrialError):
        trial.strict_json(raw)


@pytest.mark.parametrize(
    ("field", "value", "reason"),
    [
        ("schema", True, "Inna wersja"),
        ("scope", "PRODUCTION", "Inny zakres"),
        ("trial_id", "FEDCBA98-7654-3210-FEDC-BA9876543210", "Nieprawidłowe pole"),
        ("revision", "a" * 39, "Nieprawidłowe pole"),
        ("environment_id", "inny env", "Nieprawidłowe pole"),
        ("source_root", "relative", "ścieżka bezwzględna"),
        ("python_executable", "/usr/bin/../bin/python3", "ścieżka bezwzględna"),
        ("packages", {"inne": "1"}, "baza pakietów"),
    ],
)
def test_manifest_refuses_wrong_types_and_scope(
    modeled_trial: dict[str, Any], field: str, value: object, reason: str
) -> None:
    modeled_trial["manifest"][field] = value
    with pytest.raises(trial.TrialError, match=reason):
        trial.require_trial(modeled_trial["args"])


def test_manifest_refuses_missing_field(modeled_trial: dict[str, Any]) -> None:
    del modeled_trial["manifest"]["trial_id"]
    with pytest.raises(trial.TrialError, match="Niepełny schemat"):
        trial.require_trial(modeled_trial["args"])


@pytest.mark.parametrize(
    ("field", "value", "reason"),
    [
        ("observation_complete", False, "niebezpieczne lub niekompletne"),
        ("os_version", "24.10", "niebezpieczne lub niekompletne"),
        ("architecture", "aarch64", "niebezpieczne lub niekompletne"),
        ("virtualization", "none", "niebezpieczne lub niekompletne"),
        ("container", "docker", "niebezpieczne lub niekompletne"),
        ("shared_mounts", ["virtiofs"], "niebezpieczne lub niekompletne"),
        ("network_disconnected", False, "niebezpieczne lub niekompletne"),
        ("machine_id", "d" * 32, "Niezgodna maszyna"),
        ("boot_id", "fedcba98-7654-3210-fedc-ba9876543210", "Niezgodna maszyna"),
        ("packages", {}, "Niezgodna maszyna"),
    ],
)
def test_contract_refuses_environment_drift(
    modeled_trial: dict[str, Any], field: str, value: object, reason: str
) -> None:
    modeled_trial["facts"][field] = value
    with pytest.raises(trial.TrialError, match=reason):
        trial.require_trial(modeled_trial["args"])
    assert not any(name == "interpreter" for name, _ in modeled_trial["calls"])


@pytest.mark.parametrize("attribute", ["revision", "environment_id"])
def test_contract_refuses_cli_binding_drift(modeled_trial: dict[str, Any], attribute: str) -> None:
    setattr(modeled_trial["args"], attribute, "inny")
    with pytest.raises(trial.TrialError, match="Inna rewizja lub środowisko"):
        trial.require_trial(modeled_trial["args"])


@pytest.mark.parametrize("relative", [trial._HARNESS, trial._CONTRACT])
def test_contract_refuses_changed_script_bytes(
    modeled_trial: dict[str, Any], relative: Path
) -> None:
    modeled_trial["files"][modeled_trial["source"] / relative] += b"-zmiana"
    with pytest.raises(trial.TrialError, match="Zmienione chronione źródła"):
        trial.require_trial(modeled_trial["args"])


def test_contract_refuses_other_source_root(modeled_trial: dict[str, Any]) -> None:
    modeled_trial["manifest"]["source_root"] = "/srv/inne-zrodla"
    with pytest.raises(trial.TrialError, match="Inny korzeń"):
        trial.require_trial(modeled_trial["args"])


def test_contract_refuses_other_interpreter_path_without_changing_executable(
    modeled_trial: dict[str, Any],
) -> None:
    modeled_trial["manifest"]["python_executable"] = "/usr/bin/inny-python"
    with pytest.raises(trial.TrialError, match="Inny uruchomiony interpreter"):
        trial.require_trial(modeled_trial["args"])


@pytest.mark.parametrize("change", ["none", "hash", "snapshot"])
def test_administrator_rechecks_original_approval(
    modeled_trial: dict[str, Any], monkeypatch: pytest.MonkeyPatch, change: str
) -> None:
    calls: list[dict[str, str]] = []

    def approved(**kwargs: str) -> dict[str, Any]:
        calls.append(kwargs)
        return {
            "approval": copy.deepcopy(modeled_trial["approval"]),
            "environment": copy.deepcopy(modeled_trial["facts"]),
        }

    monkeypatch.setattr(trial, "require_approved_vm", approved)
    if change == "hash":
        modeled_trial["manifest"]["approval_sha256"] = "e" * 64
    elif change == "snapshot":
        modeled_trial["manifest"]["vm_snapshot"] = "inny-snapshot"
    if change == "none":
        result = trial.require_trial(modeled_trial["args"], administrator=True)
        assert (
            result["approval_sha256"] == hashlib.sha256(modeled_trial["approval_raw"]).hexdigest()
        )
    else:
        reason = "Inny hash" if change == "hash" else "nie odpowiada manifestowi"
        with pytest.raises(trial.TrialError, match=reason):
            trial.require_trial(modeled_trial["args"], administrator=True)
    assert calls == [
        {
            "revision": modeled_trial["args"].revision,
            "environment_id": modeled_trial["args"].environment_id,
        }
    ]


def test_read_protected_refuses_real_unprotected_tmp_file(tmp_path: Path) -> None:
    path = tmp_path / "kontrakt.json"
    path.write_text("{}")
    with pytest.raises(VMGuardError):
        trial.read_protected(path)


@pytest.mark.skipif(
    os.geteuid() == 0, reason="Ta odmowa wymaga rzeczywistego nieuprzywilejowanego UID."
)
def test_unprivileged_process_cannot_create_contract_or_custody_file(tmp_path: Path) -> None:
    path = tmp_path / "nowy.json"
    with pytest.raises(trial.TrialError, match="administratora VM"):
        trial.write_new(path, b"{}", 0o600)
    with pytest.raises(trial.TrialError, match="administratora VM"):
        trial.create_trial(argparse.Namespace())
    assert not path.exists()
