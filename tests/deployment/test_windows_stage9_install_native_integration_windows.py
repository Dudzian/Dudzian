"""Real Win32 custody composition on isolated runner files; never execute MSI.

The inventory harness deliberately covers only its four temporary trust domains.
It does not claim acceptance of the runner's shared ancestors or host ACL policy.
No DLL, handle, identity, hash, ACL collector or integrity validator is mocked.
"""

from __future__ import annotations

import ctypes
import hashlib
import json
import os
import zipfile
from pathlib import Path

import pytest

from deployment import windows_stage9_install_native as native

pytestmark = [
    pytest.mark.security,
    pytest.mark.qualification,
    pytest.mark.skipif(os.name != "nt", reason="requires real Windows x64 Win32 APIs"),
]


class SandboxInputs(native.LockedInputs):
    """Keep inventory observations and mutations inside the owned sandbox."""

    def _inventory(self) -> set[str]:
        found: set[str] = set()
        for root in self._trees:
            found.update(self._walk(root))
        return found | self._native_paths


@pytest.fixture
def sandbox(tmp_path):
    api = native._Native()
    roots = {name: tmp_path / name for name in ("artifacts", "policy", "checkout", "staging")}
    for root in roots.values():
        root.mkdir()
    policy = roots["policy"] / "host.json"
    policy.write_bytes(b'{"test_only":true}\n')
    (roots["checkout"] / "executor.py").write_bytes(b"# inert protected source\n")
    payload = b"inert qualification payload; this is not an MSI database\n"
    archive = roots["artifacts"] / "release.zip"
    with zipfile.ZipFile(archive, "w") as package:
        package.writestr("release/exact.msi", payload)
    inputs = SandboxInputs(
        api,
        artifact_root=roots["artifacts"],
        policy_path=policy,
        executor_root=roots["checkout"],
        staging_root=roots["staging"],
    )
    try:
        yield inputs, roots, archive, payload
    finally:
        inputs.close()


@pytest.fixture
def elevated():
    shell = getattr(ctypes, "WinDLL")("shell32.dll", use_last_error=True, winmode=0x00000800)
    shell.IsUserAnAdmin.argtypes = []
    shell.IsUserAnAdmin.restype = ctypes.c_int
    if not shell.IsUserAnAdmin():
        pytest.skip("protected SYSTEM/Administrators creation requires an elevated token")


def source_hashes(roots):
    return {
        path: hashlib.sha256(path.read_bytes()).hexdigest()
        for name in ("artifacts", "policy", "checkout")
        for path in roots[name].rglob("*")
        if path.is_file()
    }


def test_native_inventory_rejects_new_entry_without_changing_sources(sandbox):
    inputs, roots, archive, payload = sandbox
    before = source_hashes(roots)
    inputs._inventory()
    inputs.verify_pins()
    # A directory read handle does not prohibit creating children. Inventory
    # revalidation must reject this real filesystem change before authorization.
    (roots["artifacts"] / "unqualified.txt").write_bytes(b"new unqualified entry")
    with pytest.raises(native.InstallationIntegrityError, match="inventory changed"):
        inputs.verify_pins()
    assert all(
        hashlib.sha256(path.read_bytes()).hexdigest() == digest for path, digest in before.items()
    )
    assert (
        native._handle_sha256(inputs._native, inputs._pins[str(archive).casefold()].handle)
        == before[archive]
    )


def test_native_metadata_mutation_fails_closed(sandbox):
    inputs, roots, archive, payload = sandbox
    inputs._inventory()
    before = source_hashes(roots)
    api = getattr(ctypes, "WinDLL")("kernel32.dll", use_last_error=True, winmode=0x00000800)
    api.CreateFileW.argtypes = [
        ctypes.c_wchar_p,
        ctypes.c_uint32,
        ctypes.c_uint32,
        ctypes.c_void_p,
        ctypes.c_uint32,
        ctypes.c_uint32,
        ctypes.c_void_p,
    ]
    api.CreateFileW.restype = ctypes.c_void_p
    api.SetFileTime.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p]
    api.SetFileTime.restype = ctypes.c_int
    api.CloseHandle.argtypes = [ctypes.c_void_p]
    api.CloseHandle.restype = ctypes.c_int
    # FILE_WRITE_ATTRIBUTES is not denied by the read pin's data-sharing mode.
    # A fixed FILETIME avoids sleeps and ensures an observable write-time change.
    handle = api.CreateFileW(str(archive), 0x100, 7, None, 3, 0, None)
    assert handle not in {None, ctypes.c_void_p(-1).value}, getattr(ctypes, "get_last_error")()
    try:
        changed = ctypes.c_uint64(132537600000000000)
        assert api.SetFileTime(handle, None, None, ctypes.byref(changed))
    finally:
        assert api.CloseHandle(handle)
    with pytest.raises(native.InstallationIntegrityError, match="pinned protected object changed"):
        inputs.verify_pins()
    assert source_hashes(roots) == before


@pytest.mark.parametrize("failure", ["digest", "zip"])
def test_native_materialization_rejects_bad_input_and_releases_every_pin(
    sandbox, elevated, failure
):
    inputs, roots, archive, payload = sandbox
    if failure == "zip":
        archive.write_bytes(b"invalid ZIP, never executable")
    before = source_hashes(roots)
    inputs._inventory()
    with pytest.raises(native.InstallationIntegrityError, match="SHA-256|ZIP could not be read"):
        inputs.materialize_msi(
            archive=archive,
            msi_member="release/exact.msi",
            expected_msi_sha256="0" * 64,
        )
    report = inputs._attempt_report
    assert report is not None
    record = json.loads(report.path.read_text())
    assert record["status"] == "MATERIALIZATION_FAILED"
    assert record["installer_invoked"] is False
    assert record["run_stage10"] is False
    assert source_hashes(roots) == before
    handles = [pin.handle for pin in inputs._pins.values()]
    inputs.close()
    assert inputs._pins == {}
    for handle in handles:
        with pytest.raises(OSError):
            inputs._native.identity(handle)
    # Positive release control covers the untracked extraction write handle too.
    for path in roots["staging"].rglob("*"):
        if path.is_file():
            renamed = path.with_suffix(path.suffix + ".released")
            path.rename(renamed)
            renamed.unlink()


def test_native_protected_creation_has_real_restricted_dacl(tmp_path, elevated):
    api = native._Native()
    directory = tmp_path / "owned-protected"
    api.mkdir(directory)
    path = directory / "inert-payload.bin"
    handle = api.open(path, create=True, writable=True)
    try:
        api.write(handle, iter((b"inert qualification bytes",)))
    finally:
        api.close(handle)
    # Collect the actual DACL without changing any existing ACL. Ancestor
    # acceptance is deliberately left to the independent host qualification.
    observation = native._collect_acl_tree(directory)
    assert observation["complete"] is True, observation["errors"]
    for target in (directory, path):
        entry = next(
            item
            for item in observation["before"]
            if item["path"].casefold() == str(target).casefold()
        )
        assert entry["owner_sid"] == "S-1-5-32-544"
        assert entry["inheritance_disabled"] is True
        assert entry["dacl_present"] is True
        assert entry["dacl_canonical"] is True
        assert {ace["sid"] for ace in entry["aces"]} == {"S-1-5-18", "S-1-5-32-544"}
        assert all(ace["access_type"] == "Allow" and not ace["inherited"] for ace in entry["aces"])
