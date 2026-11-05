"""Modele odmów bramki; bez tworzenia kont, socketów i bez natywnego wyniku PASS."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

import pytest

if sys.platform != "linux":
    pytest.skip("Harness dotyczy wyłącznie Linux.", allow_module_level=True)

from scripts.diagnostics import stage9_cross_uid_identity as harness


def arguments() -> argparse.Namespace:
    return argparse.Namespace(
        manifest=Path("/etc/cryptohunter/stage9-cross-uid-trial.json"),
        revision="a" * 40,
        environment_id="noble-cross-uid-001",
        snapshot=Path("/var/lib/cryptohunter-cross-uid/before.json"),
    )


@pytest.mark.parametrize(
    ("entrypoint", "administrator"),
    [
        ("listener", False),
        ("probe", False),
        ("snapshot", True),
        ("run_probe", True),
        ("compare", True),
    ],
)
def test_every_native_entrypoint_refuses_before_host_or_socket_access(
    monkeypatch: pytest.MonkeyPatch, entrypoint: str, administrator: bool
) -> None:
    """Wstrzyknięta odmowa modeluje tylko granicę autoryzacji, nie UID ani kernel."""
    calls: list[tuple[argparse.Namespace, bool]] = []
    args = arguments()

    def refuse(value: argparse.Namespace, *, administrator: bool = False) -> dict[str, Any]:
        calls.append((value, administrator))
        raise ValueError("BRAK_ODRĘBNEJ_ZGODY_VM")

    def forbidden(*_args: Any, **_kwargs: Any) -> Any:
        pytest.fail("Brak zgody musi zatrzymać wejście przed obserwacją albo mutacją hosta")

    monkeypatch.setattr(harness, "require_trial", refuse)
    monkeypatch.setattr(harness.socket, "socket", forbidden)
    monkeypatch.setattr(harness.pwd, "getpwnam", forbidden)
    monkeypatch.setattr(harness.grp, "getgrnam", forbidden)
    monkeypatch.setattr(harness, "systemd", forbidden)
    with pytest.raises(ValueError, match="BRAK_ODRĘBNEJ_ZGODY_VM"):
        getattr(harness, entrypoint)(args)
    assert calls == [(args, administrator)]


def test_native_compare_requires_execution_receipt_before_live_observation(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Brak receipt nie może awansować nawet modelowo zatwierdzonego katalogu próby."""
    trial = {"trial_id": "12345678-1234-1234-1234-123456789abc"}
    requested: list[Path] = []

    def accepted(_args: argparse.Namespace, *, administrator: bool = False) -> dict[str, Any]:
        assert administrator is True
        return trial

    def missing_receipt(path: Path) -> bytes:
        requested.append(path)
        raise FileNotFoundError("Brak zapisu wykonania próby")

    def no_live_snapshot(_trial: dict[str, Any]) -> dict[str, Any]:
        raise AssertionError(
            "Brak receipt musi zatrzymać porównanie przed obserwacją żywego procesu"
        )

    monkeypatch.setattr(harness, "ROOT", tmp_path)
    monkeypatch.setattr(harness, "require_trial", accepted)
    monkeypatch.setattr(harness, "read_protected", missing_receipt)
    monkeypatch.setattr(harness, "capture_snapshot", no_live_snapshot)
    with pytest.raises(FileNotFoundError, match="Brak zapisu wykonania próby"):
        harness.compare(arguments())
    assert requested == [tmp_path / trial["trial_id"] / "execution.json"]
    assert capsys.readouterr().out == ""


@pytest.mark.parametrize("changed", ["client.json", "listener.json", "before.json", "after.json"])
def test_native_compare_rejects_changed_bytes_after_root_custody(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    changed: str,
) -> None:
    """Model integralności plików: także poprawny JSON z innymi bajtami musi odmówić."""
    args = arguments()
    trial = {
        "scope": "TEST_ONLY_STAGE9_CROSS_UID_DENIAL",
        "trial_id": "12345678-1234-1234-1234-123456789abc",
        "revision": args.revision,
        "environment_id": args.environment_id,
        "machine_id": "b" * 32,
        "boot_id": "12345678-1234-1234-1234-123456789def",
        "vm_image_sha256": "c" * 64,
        "_manifest_sha256": "d" * 64,
        "python_executable": "/zatwierdzony-testowy-runtime/bin/python3.12",
    }
    directory = tmp_path / trial["trial_id"]
    originals = {
        name: b"{}\n" for name in ("client.json", "listener.json", "before.json", "after.json")
    }
    receipt = {
        "scope": "TEST_ONLY_ROOT_EXECUTION",
        "trial": harness.binding(trial),
        "collector": "ROOT_RUNUSER_PIPE_V1",
        "argv": harness.probe_command(trial, args.manifest, directory / "before.json"),
        "returncode": 0,
        "hashes": {name: hashlib.sha256(raw).hexdigest() for name, raw in originals.items()},
    }
    files = {**originals, "execution.json": json.dumps(receipt).encode()}
    files[changed] += b" "
    assert json.loads(files[changed]) == json.loads(originals[changed])
    requested: list[Path] = []

    def accepted(_args: argparse.Namespace, *, administrator: bool = False) -> dict[str, Any]:
        assert administrator is True
        return trial

    def modeled_read(path: Path) -> bytes:
        assert path.parent == directory
        requested.append(path)
        return files[path.name]

    def no_live_snapshot(_trial: dict[str, Any]) -> dict[str, Any]:
        raise AssertionError("Podmiana bajtów musi zatrzymać odczyt żywego procesu")

    monkeypatch.setattr(harness, "ROOT", tmp_path)
    monkeypatch.setattr(harness, "require_trial", accepted)
    monkeypatch.setattr(harness, "read_protected", modeled_read)
    monkeypatch.setattr(harness, "capture_snapshot", no_live_snapshot)
    with pytest.raises(ValueError, match="Podmieniony raport lub snapshot po przejęciu dowodów"):
        harness.compare(args)
    assert {path.name for path in requested} == set(files)
    assert capsys.readouterr().out == ""
