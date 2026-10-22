"""Synthetic subprocess fixture; never connects to an installed Secret Service."""

from __future__ import annotations

import ctypes
import base64
import errno
import hashlib
import json
import os
from pathlib import Path
import socket
import sys
import types

_ROOT_UNIT = "cryptohunter-root-proof-signer.service"
_HISTORY_UNIT = "cryptohunter-history-attestation-signer.service"
_SYNTHETIC_HARDWARE_UUID = b"00112233-4455-6677-8899-aabbccddeeff\n"


def _restrict_address_families() -> None:
    library = ctypes.CDLL("libseccomp.so.2", use_errno=True)

    class ArgumentComparison(ctypes.Structure):
        _fields_ = [
            ("arg", ctypes.c_uint),
            ("op", ctypes.c_int),
            ("datum_a", ctypes.c_uint64),
            ("datum_b", ctypes.c_uint64),
        ]

    library.seccomp_init.argtypes = [ctypes.c_uint32]
    library.seccomp_init.restype = ctypes.c_void_p
    library.seccomp_syscall_resolve_name.argtypes = [ctypes.c_char_p]
    library.seccomp_syscall_resolve_name.restype = ctypes.c_int
    library.seccomp_rule_add_array.argtypes = [
        ctypes.c_void_p,
        ctypes.c_uint32,
        ctypes.c_int,
        ctypes.c_uint,
        ctypes.POINTER(ArgumentComparison),
    ]
    library.seccomp_rule_add_array.restype = ctypes.c_int
    library.seccomp_load.argtypes = [ctypes.c_void_p]
    library.seccomp_load.restype = ctypes.c_int
    library.seccomp_release.argtypes = [ctypes.c_void_p]
    context = library.seccomp_init(0x7FFF0000)  # SCMP_ACT_ALLOW
    if not context:
        raise RuntimeError("seccomp context unavailable")
    try:
        for syscall_name in (b"socket", b"socketpair"):
            syscall = library.seccomp_syscall_resolve_name(syscall_name)
            if syscall < 0:
                raise RuntimeError("socket syscall unavailable")
            # SCMP_CMP_NE: argument 0 must be AF_UNIX.
            comparison = ArgumentComparison(0, 1, socket.AF_UNIX, 0)
            result = library.seccomp_rule_add_array(
                context, 0x00050000 | errno.EAFNOSUPPORT, syscall, 1, ctypes.byref(comparison)
            )
            if result != 0:
                raise RuntimeError("seccomp rule installation failed")
        if library.seccomp_load(context) != 0:
            raise RuntimeError("seccomp filter installation failed")
    finally:
        library.seccomp_release(context)
    with socket.socket(socket.AF_UNIX):
        pass
    for family in (socket.AF_INET, socket.AF_INET6, socket.AF_NETLINK):
        try:
            socket.socket(family)
        except OSError as exc:
            if exc.errno != errno.EAFNOSUPPORT:
                raise
        else:
            raise AssertionError("non-AF_UNIX socket unexpectedly accepted")


def _install_fixture_keyring(path: Path, *, writable: bool) -> dict[str, str]:
    # An exact class match is tested against this synthetic module. No real
    # keyring module is imported and no native credentials can be accessed.
    records = json.loads(path.read_text()) if path.exists() else {}

    class Keyring:
        def get_password(self, service: str, key: str) -> str | None:
            if service != "synthetic-process-custody":
                raise AssertionError("unexpected fixture service")
            return records.get(key)

        def set_password(self, service: str, key: str, value: str) -> None:
            if not writable or service != "synthetic-process-custody":
                raise AssertionError("read-only process attempted credential mutation")
            records[key] = value
            path.write_text(json.dumps(records, sort_keys=True))

    Keyring.__module__ = "keyring.backends.SecretService"
    backend = Keyring()
    module = types.ModuleType("keyring")
    module.get_keyring = lambda: backend
    module.get_password = backend.get_password
    module.set_password = backend.set_password
    backends = types.ModuleType("keyring.backends")
    native = types.ModuleType("keyring.backends.SecretService")
    native.Keyring = Keyring
    sys.modules.update(
        {"keyring": module, "keyring.backends": backends, "keyring.backends.SecretService": native}
    )
    return records


def _model_installed_root_ownership() -> None:
    # The disposable test executor can remap every host-root inode to nobody.
    # Model installed root ownership ONLY for the provider's fixed source;
    # retain real descriptor types, permissions, data and before/after metadata.
    # Unit tests separately exercise every untrusted-ownership denial.
    actual_fstat = os.fstat

    def fixture_fstat(descriptor: int):
        metadata = actual_fstat(descriptor)
        source = os.readlink(f"/proc/self/fd/{descriptor}")
        if source not in ("/", "/etc", "/etc/machine-id"):
            return metadata
        fields = {
            field: getattr(metadata, field)
            for field in (
                "st_dev",
                "st_ino",
                "st_mode",
                "st_uid",
                "st_gid",
                "st_size",
                "st_mtime_ns",
                "st_ctime_ns",
            )
        }
        fields["st_uid"] = 0
        return types.SimpleNamespace(**fields)

    os.fstat = fixture_fstat


def main() -> None:
    # Keep repository imports independent from the script's tests/security path.
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from bot_core.security import custody_hwid, fingerprint
    from bot_core.security.base import SecretStorageError
    from bot_core.security.keyring_storage import KeyringSecretStorage

    action, fixture_path = sys.argv[1:]
    path = Path(fixture_path)
    records = _install_fixture_keyring(path, writable=action == "encrypt")
    _model_installed_root_ownership()

    # Hardware is a synthetic protected-source fixture, never real DMI data.
    # Exercise the actual UUID parser and the distinct admin/sysfs and runtime/
    # systemd-credential port choices; provider unit tests check OS protections.
    actual_source = custody_hwid._read_protected_source

    def fixture_source(parts, *, credential=False, exact_size=True):
        if parts == ("sys", "devices", "virtual", "dmi", "id", "product_uuid"):
            assert not credential and not exact_size
        elif parts in (
            ("run", "credentials", _ROOT_UNIT, "custody-product-uuid"),
            ("run", "credentials", _HISTORY_UNIT, "custody-product-uuid"),
        ):
            assert credential and exact_size
        else:
            return actual_source(parts, credential=credential, exact_size=exact_size)
        if action == "changed-hardware":
            return b"10213243-5465-7687-98a9-bacbdcedfe0f\n"
        return _SYNTHETIC_HARDWARE_UUID

    custody_hwid._read_protected_source = fixture_source

    def forbidden_uuid_node() -> int:
        raise AssertionError("UUID random fallback must never be consulted")

    fingerprint.uuid.getnode = forbidden_uuid_node
    fingerprint._collect_mac_addresses = lambda env: []
    if action == "changed-installation":
        custody_hwid._read_machine_id = lambda: b"123456789abcdef0123456789abcdef0"
    if action == "private-network":
        addresses = [item.read_text().strip() for item in Path("/sys/class/net").glob("*/address")]
        assert addresses and all(address == "00:00:00:00:00:00" for address in addresses)
    _restrict_address_families()
    unit = (
        None
        if action == "encrypt"
        else (_HISTORY_UNIT if action == "decrypt-history" else _ROOT_UNIT)
    )
    storage = KeyringSecretStorage(
        service_name="synthetic-process-custody",
        index_path=path.with_suffix(".index.json"),
        custody_hwid=True,
        custody_systemd_unit=unit,
    )
    identity = custody_hwid.CustodyHwIdProvider(systemd_unit=unit).read()
    if action == "encrypt":
        # An explicitly prepared synthetic master is a fixture prerequisite.
        # Ordinary custody writes must never initialize or regenerate it.
        assert storage.MASTER_KEY_SLOT not in records
        records[storage.MASTER_KEY_SLOT] = json.dumps(
            {
                "version": storage.SECRET_VERSION,
                "key_b64": base64.b64encode(b"m" * storage.MASTER_KEY_BYTES).decode(),
                "hwid_digest": hashlib.sha256(identity.encode()).hexdigest(),
                "hwid_scheme": custody_hwid.CUSTODY_HWID_SCHEME,
            }
        )
        path.write_text(json.dumps(records, sort_keys=True))
        actual_urandom = os.urandom

        def nonce_only_random(size):
            assert size == 12, "Ordinary custody write attempted master generation"
            return actual_urandom(size)

        os.urandom = nonce_only_random
        storage.set_secret("fixture", "synthetic process restart secret")
    elif action in ("changed-installation", "changed-hardware"):
        try:
            storage.get_existing_native_secret("fixture")
        except SecretStorageError:
            print(json.dumps({"changed_identity_denied": True}))
            return
        raise AssertionError("changed installation or hardware identity accepted")
    elif action in ("decrypt", "decrypt-history"):
        # Also forbid random generation: an existing-key read must never create
        # a master or re-encrypt an envelope.
        os.urandom = lambda size: (_ for _ in ()).throw(AssertionError("random generation"))
        assert storage.get_existing_native_secret("fixture") == "synthetic process restart secret"
    print(
        json.dumps(
            {
                "identity_digest": hashlib.sha256(identity.encode()).hexdigest(),
                "versioned": identity.startswith(custody_hwid.CUSTODY_HWID_PREFIX),
                "af_unix_only": True,
            }
        )
    )


if __name__ == "__main__":
    main()
