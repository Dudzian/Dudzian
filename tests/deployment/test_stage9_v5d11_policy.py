"""Policy regressions using synthetic data and a disposable filter probe."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
from typing import Any

import pytest

from scripts import stage9_v5d10_contract as contract
from scripts import stage9_v5d11_policy as policy

pytestmark = [pytest.mark.security, pytest.mark.unit]

PIN = "a" * 64
A = "/opt/synthetic-stage9/profile-a"
LIB = "/usr/lib/x86_64-linux-gnu/libexample.so.1.2"
LOADER = "/usr/lib/x86_64-linux-gnu/ld-linux-x86-64.so.2"
MODULE = f"{A}/packages/libvirtmod.example.so"
PYTHON = f"{A}/venv/bin/python3.12"
HEADER = "/opt/synthetic-stage9/base/patchlevel.h"
SOCKET = "/run/libvirt/libvirt-sock-ro"
IMAGE = "/var/lib/libvirt/images/synthetic.img"
DATA = "/etc/cryptohunter/observer-v5d1/disposable.json"


@pytest.fixture
def inputs() -> tuple[dict[str, Any], dict[str, Any]]:
    request = {
        "profile": "stage9-v5d4-static-abi-v1",
        "interpreter": PYTHON,
        "additional_extensions": [policy.QEMU],
        "libvirt_extension": MODULE,
        "library_map": {"libexample.so.1": LIB, "/lib64/ld-linux-x86-64.so.2": LOADER},
    }
    plan = {
        "files": {
            "venv/bin/python3.12": {"sha256": PIN, "mode": 0o755},
            "packages/libvirtmod.example.so": {"sha256": PIN, "mode": 0o600},
            "launch.py": {"sha256": PIN, "mode": 0o600},
        },
        "directories": dict.fromkeys((".", "packages", "venv", "venv/bin"), 0o700),
    }
    receipt = {
        **dict.fromkeys(contract.APPROVAL_DIGESTS | {"approval_sha256"}, PIN),
        **plan,
        "profile": contract.RECEIPT,
        "root": A,
        "entrypoint": contract.ENTRYPOINT,
        "host": {**contract.HOST, "interpreter_sha256": PIN},
        "base_runtime": {
            "profile": "stage9-v5d4-base-runtime-inventory-v1",
            "roots": ["/usr/lib/python3.12"],
            "files": {"/usr/lib/python3.12/pathlib.py": PIN},
            "native": [],
            "symlinks": {},
            "excluded": [],
            "scope": "STATIC_BYTES_SUPERSET_NOT_DYNAMIC_COMPLETENESS",
            "installation_approved": False,
        },
        "external_files": dict.fromkeys((policy.QEMU, LIB, LOADER, HEADER), PIN),
        "authorization": "NOT_AUTHORIZED",
        "native_load": "NOT_EXECUTED",
        "file_plan_sha256": contract.plan_sha256(plan),
        "abi_request_sha256": contract.digest(contract.canonical(request)),
    }
    return receipt, request


def test_exact_rendered_policy(inputs) -> None:
    receipt, request = inputs
    first = policy.render(receipt, request, SOCKET, [IMAGE, DATA])
    assert first == policy.render(receipt, request, SOCKET, [DATA, IMAGE])
    observer, qemu = first.decode().split(f"profile {policy.LABELS[1]}")
    assert f'"{PYTHON}" mrix,' in observer
    assert f'"{MODULE}" mr,' in observer
    assert f'"{HEADER}" r,' in observer
    assert f'"{DATA}" r,' in observer
    assert f'"{SOCKET}" rw,' in observer
    assert "virtqemud-sock-ro" not in observer
    assert "libvirt-admin-sock" not in observer
    assert f"rmPx -> &{policy.LABELS[1]}" in observer
    assert f'"{IMAGE}" rk,' in qemu
    assert f'"{LIB}" mr,' in qemu
    assert f'"{LOADER}" mr,' in qemu
    assert '"/lib64/ld-linux-x86-64.so.2" mr,' in qemu
    assert "/usr/lib/x86_64-linux-gnu/*.so" not in qemu
    assert PYTHON not in qemu
    assert SOCKET not in qemu
    assert "deny network unix," in qemu


@pytest.mark.parametrize(
    "mutation, reason",
    [
        ("extra_provider", "APPARMOR_NATIVE_PIN_REJECTED"),
        ("native_outside_lib", "APPARMOR_LIBRARY_SCOPE_REJECTED"),
        ("unexpected_socket", "APPARMOR_SCOPE_REJECTED"),
        ("wrong_data", "APPARMOR_SCOPE_REJECTED"),
        ("duplicate_data", "APPARMOR_SCOPE_REJECTED"),
        ("pattern_data", "APPARMOR_LITERAL_PATH_REJECTED"),
        ("alias_conflict", "APPARMOR_ALIAS_CONFLICT"),
        ("wrong_interpreter", "APPARMOR_ABI_SCHEMA_REJECTED"),
    ],
)
def test_render_refuses_scope_change(inputs, mutation: str, reason: str) -> None:
    receipt, request = inputs
    socket, data_paths = SOCKET, [IMAGE]
    if mutation == "extra_provider":
        request["library_map"]["extra.so"] = "/usr/lib/x86_64-linux-gnu/extra.so"
    elif mutation == "native_outside_lib":
        request["library_map"]["extra.so"] = HEADER
    elif mutation == "unexpected_socket":
        socket = "/run/libvirt/libvirt-sock"
    elif mutation == "wrong_data":
        data_paths = ["/etc/synthetic.json"]
    elif mutation == "duplicate_data":
        data_paths *= 2
    elif mutation == "pattern_data":
        data_paths = ["/var/lib/libvirt/images/*.img"]
    elif mutation == "alias_conflict":
        receipt["base_runtime"]["symlinks"]["/usr/lib/x86_64-linux-gnu/libexample.so.1"] = {
            "resolved": LOADER
        }
    elif mutation == "wrong_interpreter":
        request["interpreter"] = f"{A}/venv/bin/python3"
    with pytest.raises(policy.Refusal, match=reason):
        policy.render(receipt, request, socket, data_paths)


def test_literal_paths_are_canonical() -> None:
    assert policy.literal("/opt/synthetic.stage/a@b+1") == "/opt/synthetic.stage/a@b+1"
    for path in (
        "foo",
        "/",
        "/foo/../bar",
        "/foo//bar",
        "/foo/{bar,baz}",
        "/foo/\nbar",
        "/a b",
        "/foo/**",
        '/foo/"bar',
    ):
        with pytest.raises(policy.Refusal, match="APPARMOR_LITERAL_PATH_REJECTED"):
            policy.literal(path)


@pytest.fixture
def exported(tmp_path: Path) -> tuple[Path, bytes]:
    if os.name != "posix":
        pytest.skip("Kernel policy export fixture uses POSIX symlinks")
    policy_data = b"synthetic compiled policy\0\x10"
    (tmp_path / "revision").write_bytes(b"42\n")
    raw = tmp_path / "policy/raw_data/42"
    raw.mkdir(parents=True)
    (raw / "raw_data").write_bytes(policy_data)
    for index, name in enumerate(policy.LABELS):
        path = tmp_path / "policy/profiles" / f"{name}.{index}"
        path.mkdir(parents=True)
        (path / "mode").write_text("enforce\n", encoding="ascii")
        (path / "name").write_text(f"{name}\n", encoding="ascii")
        (path / "raw_data").symlink_to("../../raw_data/42/raw_data")
    return tmp_path, policy_data


def test_policy_export_must_match_real_bytes(exported) -> None:
    root, blob = exported
    assert policy.active_profiles(blob, root) == ("42", dict.fromkeys(policy.LABELS, "enforce"))
    with pytest.raises(policy.Refusal, match="APPARMOR_KERNEL_POLICY_MISMATCH"):
        policy.active_profiles(blob + b"different", root)


def test_enforce_and_revision_must_match(exported, monkeypatch: pytest.MonkeyPatch) -> None:
    root, blob = exported
    profile = root / "policy/profiles" / f"{policy.LABELS[1]}.1"
    (profile / "mode").write_bytes(b"complain\n")
    with pytest.raises(policy.Refusal, match="APPARMOR_PROFILE_MODE_REJECTED"):
        policy.active_profiles(blob, root)
    (profile / "mode").write_bytes(b"enforce\n")
    values = iter(("42", "43"))
    monkeypatch.setattr(policy, "kernel_revision", lambda _: next(values))
    with pytest.raises(policy.Refusal, match="APPARMOR_POLICY_CHANGED_DURING_READ"):
        policy.active_profiles(blob, root)


@pytest.mark.skipif(sys.platform != "linux", reason="V5d10 pins use Linux secure_read")
def test_load_checks_reviewed_pins_before_add_and_verifies_kernel_bytes(
    inputs, capsys: pytest.CaptureFixture[str], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    receipt, request = inputs
    receipt_path = tmp_path / "receipt.json"
    request_path = tmp_path / "abi-request.json"
    receipt_path.write_bytes(contract.canonical(receipt))
    request_path.write_bytes(contract.canonical(request))
    args = [
        "--receipt",
        str(receipt_path),
        "--receipt-sha256",
        contract.digest(receipt_path.read_bytes()),
        "--abi-request",
        str(request_path),
        "--socket",
        SOCKET,
    ]
    apparmor = policy.render(receipt, request, SOCKET, [])
    blob = b"reviewed test policy"
    commands = []

    def fake_parser(flags: list[str], raw: bytes) -> bytes:
        commands.append(flags)
        if flags == ["--binary", "--add"]:
            assert raw == blob
            return b""
        assert raw == apparmor and "--skip-kernel-load" in flags
        return blob

    enabled = tmp_path / "export_binary"
    enabled.write_bytes(b"Y\n")
    atomic = tmp_path / "set_load"
    atomic.write_bytes(b"yes\n")
    monkeypatch.setattr(policy, "EXPORT_BINARY", enabled)
    monkeypatch.setattr(policy, "ATOMIC_SET_LOAD", atomic)
    monkeypatch.setattr(policy, "parser_run", fake_parser)
    monkeypatch.setattr(
        policy,
        "active_profiles",
        lambda raw: ("42", dict.fromkeys(policy.LABELS, "enforce")) if raw == blob else None,
    )
    pins = [
        "--source-sha256",
        contract.digest(apparmor),
        "--compiled-sha256",
        contract.digest(blob),
    ]
    assert policy.main(["load", *args, *pins[:-1], PIN]) == 2
    assert "APPARMOR_COMPILED_PIN_MISMATCH" in capsys.readouterr().err
    assert ["--binary", "--add"] not in commands
    assert policy.main(["load", *args, *pins]) == 0
    assert commands[-1] == ["--binary", "--add"]
    assert json.loads(capsys.readouterr().out)["exported_sha256"] == contract.digest(blob)
    request_path.write_bytes(b"{}")
    assert policy.main(["load", *args, *pins]) == 2
    assert "APPARMOR_INPUT_PIN_MISMATCH" in capsys.readouterr().err


@pytest.mark.skipif(
    sys.platform != "linux" or platform.machine() != "x86_64" or shutil.which("cc") is None,
    reason="Classic BPF fixture requires Linux x86-64 and cc",
)
def test_seccomp_filter_has_exact_bytes_and_real_denials(tmp_path: Path) -> None:
    root = Path(policy.__file__).resolve().parents[1]
    probe = tmp_path / "probe.c"
    binary = tmp_path / "probe"
    probe.write_text(
        """#define _GNU_SOURCE
#include "stage9_v5d11_policy.h"
#include <errno.h>
#include <sys/mman.h>
#include <sys/prctl.h>
#include <sys/syscall.h>
#include <unistd.h>
int main(int argc, char **argv) {
    (void)argv;
    if (argc == 2) return stage9_v5d11_dump_seccomp(STDOUT_FILENO) != 0;
    int nnp = prctl(PR_GET_NO_NEW_PRIVS, 0L, 0L, 0L, 0L);
    if (nnp < 0 || (nnp == 0 && (stage9_v5d11_install_seccomp() == 0 || errno != EPERM))) return 1;
    if (prctl(PR_SET_NO_NEW_PRIVS, 1L, 0L, 0L, 0L) != 0) return 2;
    if (stage9_v5d11_install_seccomp() != 0 || stage9_v5d11_probe_seccomp() != 0) return 3;
    if (syscall(SYS_getpid) <= 0) return 4;
    errno = 0;
    if (syscall(SYS_ioctl, -1L, 0x80086601UL, 0L) != -1 || errno != EBADF) return 5;
    errno = 0;
    if (syscall(SYS_mmap, 0L, 0L, PROT_EXEC, MAP_PRIVATE | MAP_ANONYMOUS, -1L, 0L) != -1 || errno != EUCLEAN) return 6;
    return 0;
}
""",
        encoding="ascii",
    )
    subprocess.run(
        [
            "cc",
            "-std=c17",
            "-Wall",
            "-Wextra",
            "-Werror",
            "-I",
            str(root / "scripts"),
            "-o",
            str(binary),
            str(probe),
            str(root / "scripts/stage9_v5d11_policy.c"),
        ],
        check=True,
        timeout=15,
    )
    bpf = subprocess.run([str(binary), "dump"], check=True, capture_output=True, timeout=10).stdout
    pin = (root / "deployment/stage9_v5d11/seccomp.bpf.sha256").read_text(encoding="ascii").strip()
    assert hashlib.sha256(bpf).hexdigest() == pin
    subprocess.run([str(binary)], check=True, timeout=10)
