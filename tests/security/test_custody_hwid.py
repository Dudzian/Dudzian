"""Protected-source and compatibility boundaries for custody installation HWID."""

from __future__ import annotations

import hashlib
import os
import stat
import sys
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from bot_core.security import custody_hwid
from bot_core.security.custody_hwid import CUSTODY_HWID_PREFIX, CustodyHwIdProvider
from bot_core.security.hwid import HwIdProvider, HwIdProviderError

_PRODUCT_UUID_READER = custody_hwid._read_product_uuid
_HARDWARE = b"12345678-1234-5678-9abc-123456789abc"
_UNIT = "cryptohunter-root-proof-signer.service"
pytestmark = pytest.mark.skipif(sys.platform != "linux", reason="Linux protected-source flags")


def _metadata(*, mode=stat.S_IFREG | 0o444, uid=0, size=33, **changes):
    fields = dict(
        st_mode=mode,
        st_uid=uid,
        st_gid=0,
        st_size=size,
        st_dev=1,
        st_ino=10,
        st_mtime_ns=1,
        st_ctime_ns=1,
    )
    fields.update(changes)
    return SimpleNamespace(**fields)


@pytest.fixture
def protected_source(monkeypatch):
    identity = b"123456789abcdef0123456789abcdef0\n"
    directory = _metadata(mode=stat.S_IFDIR | 0o755)
    file_stat = _metadata()
    fake_os = SimpleNamespace(
        **{
            name: getattr(os, name, 0)
            for name in ("O_RDONLY", "O_CLOEXEC", "O_DIRECTORY", "O_NOFOLLOW", "O_NONBLOCK")
        },
        open=Mock(side_effect=[10, 11, 12]),
        close=Mock(),
        fstat=Mock(side_effect=[directory, directory, file_stat, file_stat]),
        read=Mock(return_value=identity),
        geteuid=lambda: 1000,
        ST_RDONLY=getattr(os, "ST_RDONLY", 1),
        fstatvfs=Mock(return_value=SimpleNamespace(f_flag=getattr(os, "ST_RDONLY", 1))),
    )
    monkeypatch.setattr(custody_hwid, "os", fake_os)
    monkeypatch.setattr(custody_hwid.sys, "platform", "linux")
    monkeypatch.setattr(custody_hwid, "_read_product_uuid", lambda unit: _HARDWARE)
    return fake_os, identity, directory, file_stat


def test_fixed_protected_source_and_versioned_known_vector(protected_source):
    source, identity, _, _ = protected_source
    result = CustodyHwIdProvider().read()
    assert (
        result
        == CUSTODY_HWID_PREFIX
        + hashlib.sha384(
            b"cryptohunter.production-signing-custody\x00linux-host-v1\x00"
            + identity.rstrip(b"\n")
            + b"\x00"
            + _HARDWARE
        ).hexdigest()
    )
    calls = source.open.call_args_list
    assert [call.args[0] for call in calls] == ["/", "etc", "machine-id"]
    assert all(call.args[1] & os.O_NOFOLLOW for call in calls)
    assert all(call.args[1] & os.O_CLOEXEC for call in calls)
    assert calls[1].kwargs == {"dir_fd": 10}
    assert calls[2].kwargs == {"dir_fd": 11}
    assert source.read.call_args.args == (12, 64)
    assert [call.args[0] for call in source.close.call_args_list] == [12, 11, 10]


@pytest.mark.parametrize("component", [0, 1, 2])
@pytest.mark.parametrize(
    "change",
    [
        {"st_uid": 1000},
        {"st_mode": stat.S_IFDIR | 0o775},
        {"st_mode": stat.S_IFDIR | 0o757},
        {"st_mode": stat.S_IFLNK | 0o777},
    ],
)
def test_unprotected_ancestry_or_file_refuses(protected_source, component, change):
    source, _, directory, file_stat = protected_source
    original = directory if component < 2 else file_stat
    changed = SimpleNamespace(**(vars(original) | change))
    source.fstat.side_effect = [directory, directory, file_stat, file_stat][:component] + [changed]
    with pytest.raises(HwIdProviderError, match="Protected root-owned"):
        CustodyHwIdProvider().read()
    source.read.assert_not_called()


@pytest.mark.parametrize(
    "value",
    [
        b"",
        b"uninitialized\n",
        b"0" * 32,
        b"f" * 32,
        b"A" * 32,
        b"1" * 31,
        b"1" * 33,
        b"1" * 32 + b"\r\n",
        b"1" * 32 + b"\nextra",
        b"1" * 32 + b"\x00",
        b"\xff" * 32,
    ],
)
def test_invalid_or_placeholder_identity_has_no_fallback(protected_source, value):
    source, _, directory, _ = protected_source
    file_stat = _metadata(size=len(value))
    source.fstat.side_effect = [directory, directory, file_stat, file_stat]
    source.read.return_value = value
    with pytest.raises(HwIdProviderError):
        CustodyHwIdProvider().read()


@pytest.mark.parametrize(
    "change",
    [
        {"st_mtime_ns": 2},
        {"st_ctime_ns": 2},
        {"st_ino": 11},
        {"st_uid": 1000},
        {"st_mode": stat.S_IFREG | 0o666},
        {"st_size": 34},
    ],
)
def test_change_during_read_refuses(protected_source, change):
    source, _, directory, file_stat = protected_source
    source.fstat.side_effect = [
        directory,
        directory,
        file_stat,
        SimpleNamespace(**(vars(file_stat) | change)),
    ]
    with pytest.raises(HwIdProviderError, match="changed during read"):
        CustodyHwIdProvider().read()
    assert source.close.call_count == 3


@pytest.mark.parametrize("error", [FileNotFoundError(), PermissionError(), OSError("symlink")])
def test_unavailable_source_never_falls_back(protected_source, error):
    source, _, _, _ = protected_source
    source.open.side_effect = error
    with pytest.raises(HwIdProviderError, match="unavailable"):
        CustodyHwIdProvider().read()


def test_oem_environment_cannot_select_custody_identity(protected_source, monkeypatch):
    monkeypatch.setenv("OEM_CPU_ID", "untrusted")
    monkeypatch.setenv("OEM_MAC_ADDRESSES", "01:23:45:67:89:ab")
    monkeypatch.setenv("OEM_FINGERPRINT_SALT", "untrusted")
    monkeypatch.setenv("HWID", "untrusted")
    monkeypatch.setenv("CUSTODY_HWID", "untrusted")
    monkeypatch.setenv("CREDENTIALS_DIRECTORY", "/tmp/untrusted-credentials")
    assert CustodyHwIdProvider().read().startswith(CUSTODY_HWID_PREFIX)
    with pytest.raises(TypeError):
        CustodyHwIdProvider(fingerprint_reader=lambda: "untrusted")


def test_unsupported_platform_refuses_before_source_read(monkeypatch):
    monkeypatch.setattr(custody_hwid.sys, "platform", "win32")
    read_source = Mock()
    monkeypatch.setattr(custody_hwid, "_read_machine_id", read_source)
    with pytest.raises(HwIdProviderError, match="Linux"):
        CustodyHwIdProvider().read()
    read_source.assert_not_called()


def test_linux_source_helpers_refuse_unsupported_platform(monkeypatch):
    monkeypatch.setattr(custody_hwid.sys, "platform", "win32")
    with pytest.raises(HwIdProviderError, match="Linux"):
        custody_hwid._require_protected(_metadata(), directory=False)
    with pytest.raises(HwIdProviderError, match="Linux"):
        custody_hwid._read_protected_source(("etc", "machine-id"))


def test_legacy_provider_still_uses_its_original_reader():
    assert HwIdProvider(fingerprint_reader=lambda: " legacy-license-hwid ").read() == (
        "legacy-license-hwid"
    )


def _hardware_source(protected_source, *, credential, value=_HARDWARE + b"\n"):
    source, _, directory, _ = protected_source
    count = 5 if credential else 7
    source.open.side_effect = list(range(10, 10 + count))
    file_stat = _metadata(
        mode=stat.S_IFREG | 0o400,
        uid=1000 if credential else 0,
        size=len(value) if credential else 4096,
    )
    directories = [directory] * (count - 1)
    if credential:
        directories[-1] = _metadata(mode=stat.S_IFDIR | 0o500, uid=1000)
    source.fstat.side_effect = [*directories, file_stat, file_stat]
    source.read.return_value = value
    return source, directories, file_stat


@pytest.mark.parametrize("credential", [False, True])
def test_hardware_uuid_uses_exact_fixed_port(protected_source, credential):
    source, _, _ = _hardware_source(protected_source, credential=credential)
    assert _PRODUCT_UUID_READER(_UNIT if credential else None) == _HARDWARE
    expected = (
        ["/", "run", "credentials", _UNIT, "custody-product-uuid"]
        if credential
        else ["/", "sys", "devices", "virtual", "dmi", "id", "product_uuid"]
    )
    assert [call.args[0] for call in source.open.call_args_list] == expected
    assert source.fstatvfs.call_count == (2 if credential else 0)


@pytest.mark.parametrize(
    "value",
    [
        b"",
        b"unavailable\n",
        b"00000000-0000-0000-0000-000000000000\n",
        b"ffffffff-ffff-ffff-ffff-ffffffffffff",
        b"1" * 36,
        _HARDWARE + b"\x00",
        _HARDWARE + b"\nextra",
        _HARDWARE + b"\r\n",
    ],
)
def test_invalid_hardware_uuid_refuses_without_fallback(protected_source, value):
    _hardware_source(protected_source, credential=False, value=value)
    with pytest.raises(HwIdProviderError, match="hardware UUID"):
        _PRODUCT_UUID_READER(None)


def test_sysfs_reported_size_is_not_content_length(protected_source):
    source, _, file_stat = _hardware_source(protected_source, credential=False)
    assert file_stat.st_size == 4096
    assert _PRODUCT_UUID_READER(None) == _HARDWARE
    assert source.read.call_count == 1


@pytest.mark.parametrize("flags", [(0,), (1, 0)])
def test_writable_credential_mount_refuses(protected_source, flags):
    source, _, _ = _hardware_source(protected_source, credential=True)
    source.fstatvfs.side_effect = [SimpleNamespace(f_flag=flag) for flag in flags]
    with pytest.raises(HwIdProviderError, match="Read-only systemd credential mount"):
        _PRODUCT_UUID_READER(_UNIT)
    source.read.assert_not_called()


@pytest.mark.parametrize(
    "component,change",
    [
        (0, {"st_uid": 1000}),
        (1, {"st_mode": stat.S_IFDIR | 0o777}),
        (2, {"st_uid": 1000}),
        (3, {"st_uid": 2000}),
        (3, {"st_mode": stat.S_IFDIR | 0o700}),
        (4, {"st_uid": 2000}),
        (4, {"st_mode": stat.S_IFREG | 0o600}),
        (4, {"st_mode": stat.S_IFLNK | 0o400}),
    ],
)
def test_untrusted_credential_ancestry_and_source_refuse(protected_source, component, change):
    source, directories, file_stat = _hardware_source(protected_source, credential=True)
    snapshots = [*directories, file_stat, file_stat]
    snapshots[component] = SimpleNamespace(**(vars(snapshots[component]) | change))
    source.fstat.side_effect = snapshots
    with pytest.raises(HwIdProviderError, match="Protected root-owned"):
        _PRODUCT_UUID_READER(_UNIT)
    source.read.assert_not_called()


def test_installed_and_offline_ports_derive_same_identity(monkeypatch):
    monkeypatch.setattr(custody_hwid.sys, "platform", "linux")
    reads = Mock(
        side_effect=lambda parts, **kwargs: (
            b"123456789abcdef0123456789abcdef0\n" if parts[0] == "etc" else _HARDWARE + b"\n"
        )
    )
    monkeypatch.setattr(custody_hwid, "_read_protected_source", reads)
    assert CustodyHwIdProvider().read() == CustodyHwIdProvider(systemd_unit=_UNIT).read()
    paths = [call.args[0] for call in reads.call_args_list]
    assert ("sys", "devices", "virtual", "dmi", "id", "product_uuid") in paths
    assert ("run", "credentials", _UNIT, "custody-product-uuid") in paths


def test_missing_runtime_credential_never_uses_direct_hardware(monkeypatch):
    monkeypatch.setattr(custody_hwid.sys, "platform", "linux")
    reads = Mock(side_effect=[b"123456789abcdef0123456789abcdef0", PermissionError()])
    monkeypatch.setattr(custody_hwid, "_read_protected_source", reads)
    with pytest.raises(HwIdProviderError, match="unavailable"):
        CustodyHwIdProvider(systemd_unit=_UNIT).read()
    assert [call.args[0][0] for call in reads.call_args_list] == ["etc", "run"]


@pytest.mark.parametrize("unit", ["", "untrusted.service", "../../etc", 1, False, []])
def test_caller_cannot_supply_arbitrary_source_context(unit):
    with pytest.raises(HwIdProviderError, match="Fixed installed"):
        CustodyHwIdProvider(systemd_unit=unit)
