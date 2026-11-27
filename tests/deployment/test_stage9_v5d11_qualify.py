"""Synthetic ABI and libvirt evidence for V5d11, without native execution."""

from __future__ import annotations

import os
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from scripts import stage9_v5d10_contract as contract
from scripts import stage9_v5d11_diagnostic as diagnostic
from scripts import stage9_v5d11_qualify as qualify

pytestmark = [
    pytest.mark.security,
    pytest.mark.unit,
    pytest.mark.skipif(os.name != "posix", reason="Linux Observer qualification evidence"),
]

ROOT = "/profile-a"
PYTHON = f"{ROOT}/venv/bin/python3.12"
LIBVIRT = f"{ROOT}/packages/libvirtmod.so"
CFFI = f"{ROOT}/packages/_cffi_backend.so"
RUST = f"{ROOT}/packages/cryptography/hazmat/bindings/_rust.abi3.so"
QEMU = contract.QEMU_IMG
LIBC = "/usr/lib/x86_64-linux-gnu/libc.so.6"
SOCKET = "/run/libvirt/libvirt-sock-ro"
DAEMON = "/usr/sbin/libvirtd"
PIN = "a" * 64
MOUNTINFO = (
    b"21 0 00:12 / / ro,nosuid,nodev - tmpfs tmpfs ro\n"
    b"22 21 00:13 / /proc ro,nosuid,nodev,noexec - proc proc ro\n"
    b"23 21 00:14 /null /dev/null ro,nosuid,noexec - tmpfs tmpfs ro\n"
    b"24 21 00:15 /sock /run/libvirt/libvirt-sock-ro ro,nosuid,nodev,noexec - tmpfs tmpfs ro\n"
)
KERNEL_STATUS = (
    b"Uid:\t1000\t1000\t1000\t1000\nGid:\t1000\t1000\t1000\t1000\n"
    b"Groups:\t\nNSpid:\t10001\t123\nNoNewPrivs:\t1\nSeccomp:\t2\n"
    b"CapInh:\t0000000000000000\nCapPrm:\t0000000000000000\n"
    b"CapEff:\t0000000000000000\nCapBnd:\t0000000000000000\nCapAmb:\t0000000000000000\n"
)


def artifact(path: Path, data: bytes) -> dict[str, str]:
    path.write_bytes(data)
    return {"path": path.name, "sha256": contract.digest(data)}


def daemon() -> dict[str, Any]:
    config = {"/etc/libvirt/libvirtd.conf": PIN}
    units = {f"/lib/systemd/system/libvirtd{suffix}": PIN for suffix in ("-ro.socket", ".service")}
    return {
        "socket": SOCKET,
        "socket_identity": [11, 22, 0, 0, 0o777],
        "host_socket_identity": [11, 22, 0, 0, 0o777],
        "client_config_sha256": contract.digest(qualify.CLIENT_BYTES),
        "peer": [0, 0, 0],
        "host_peer": [1080, 0, 0],
        "daemon": {
            "pid": 1080,
            "start_ticks": 100,
            "listener_inode": 23,
            "executable": DAEMON,
            "executable_sha256": PIN,
            "cmdline_sha256": PIN,
            "config": config,
            "unit_files": units,
            "socket_unit": "libvirtd-ro.socket",
            "service_unit": "libvirtd.service",
            "reviewed_readonly_implementation_sha256": PIN,
        },
        "uri": "qemu:///system",
        "connect_paths": [SOCKET],
        "read_results": dict.fromkeys(qualify.READS, "PASS"),
        "fixture_denial": {
            "scope": "DISPOSABLE_FIXTURE",
            "daemon_sha256": PIN,
            "config": config,
            "unit_files": units,
            "client_connect_flags": 0,
            "wire_request": ["REMOTE_PROC_DOMAIN_GET_XML_DESC", 1],
            "wire_reply": "VIR_ERR_OPERATION_DENIED",
        },
    }


def loader_case(path: Path, name: str) -> dict[str, Any]:
    qemu = name.startswith("qemu-")
    executable = QEMU if qemu else PYTHON
    loaded = [executable, LIBC]
    if name == "python-libvirt":
        loaded.append(LIBVIRT)
    elif name == "python-imports":
        loaded.extend([CFFI, RUST])
    argv0 = "/proc/self/fd/4" if qemu else executable
    events = [
        f"file=libc.so.6 [0]; needed by {argv0} [0]",
        "file=libc.so.6 [0]; generating link map",
        f"checking for version `DEMO_1' in file {LIBC} [0] required by file {argv0} [0]",
        f"binding file {argv0} [0] to {LIBC} [0]: normal symbol `demo' [DEMO_1]",
        f"calling init: {LIBC}",
    ]
    for item in loaded[2:]:
        events.extend(
            (f"file={item} [0]; dynamically loaded by {argv0} [0]", f"calling init: {item}")
        )
    trace = "\n".join(f"123:\t{row}" for row in events)
    maps = "\n".join(
        f"{index + 1}000-{index + 2}000 r-xp 00000000 00:23 {index + 50} {item}"
        for index, item in enumerate(loaded)
    )
    maps += "\n9000-a000 r-xp 00000000 00:00 0 [vdso]\n"
    export = "   1: 0000000000000010 12 FUNC GLOBAL DEFAULT 3 demo@@DEMO_1\n"
    weak = (
        "   1: 0000000000000000 0 FUNC WEAK DEFAULT UND demo@DEMO_1 (2)\n"
        "   2: 0000000000000000 0 FUNC WEAK DEFAULT UND missing\n"
    )
    symbols = {
        item: artifact(
            path / f"{name}-{index}.symbols",
            (
                "Symbol table '.dynsym' contains 3 entries:\n" + (export if item == LIBC else weak)
            ).encode(),
        )
        for index, item in enumerate(loaded)
    }
    outcome: dict[str, Any] | str = "PASS"
    if qemu:
        fmt = name.removeprefix("qemu-")
        outcome = {
            "argv": [argv0, "info", "--output=json", "-f", fmt, "/proc/self/fd/5"],
            "fixture_sha256": PIN,
            "exit": 0,
            "json_format": fmt,
        }
    return {
        "pid": 123,
        "argv0": argv0,
        "trace": artifact(path / f"{name}.trace", trace.encode()),
        "maps": artifact(path / f"{name}.maps", maps.encode()),
        "status": artifact(path / f"{name}.status", KERNEL_STATUS),
        "label": artifact(
            path / f"{name}.label", qualify.QEMU_LABEL if qemu else qualify.PARENT_LABEL
        ),
        "objects": {
            item: {"sha256": PIN, "identity": [0, 0x23, index + 50]}
            for index, item in enumerate(loaded)
        },
        "symbols": symbols,
        "outcome": outcome,
    }


@pytest.fixture
def inputs(tmp_path: Path) -> tuple[dict[str, Any], ...]:
    receipt = {
        "profile": contract.RECEIPT,
        "root": ROOT,
        "entrypoint": contract.ENTRYPOINT,
        "host": {**contract.HOST, "interpreter_sha256": PIN},
        "external_files": {QEMU: PIN, LIBC: PIN},
        "files": {
            path.removeprefix(f"{ROOT}/"): {"sha256": PIN, "mode": 0o600}
            for path in (PYTHON, LIBVIRT, CFFI, RUST)
        },
        "directories": {
            item: 0o700
            for item in (
                ".",
                "venv",
                "venv/bin",
                "packages",
                "packages/cryptography",
                "packages/cryptography/hazmat",
                "packages/cryptography/hazmat/bindings",
            )
        },
        "base_runtime": {
            "profile": "stage9-v5d4-base-runtime-inventory-v1",
            "files": {LIBC: PIN},
            "roots": ["/usr/lib/python3.12"],
            "native": [LIBC],
            "excluded": [],
            "scope": "STATIC_BYTES_SUPERSET_NOT_DYNAMIC_COMPLETENESS",
            "installation_approved": False,
            "symlinks": {},
        },
        "authorization": "NOT_AUTHORIZED",
        "native_load": "NOT_EXECUTED",
        **dict.fromkeys(contract.APPROVAL_DIGESTS | {"approval_sha256"}, PIN),
    }
    receipt["file_plan_sha256"] = contract.plan_sha256(
        {"directories": receipt["directories"], "files": receipt["files"]}
    )
    request = {
        "interpreter": PYTHON,
        "libvirt_extension": LIBVIRT,
        "required_imports": {"_cffi_backend": CFFI},
        "library_map": {"libc.so.6": LIBC},
    }
    receipt["abi_request_sha256"] = contract.digest(contract.canonical(request))
    elf = (PYTHON, LIBVIRT, CFFI, RUST, QEMU, LIBC)
    report = {
        "profile": qualify.STATIC_REPORT,
        "status": "STATIC_ABI_MATCH",
        "elf_measurements": dict.fromkeys(elf, PIN),
        "elf": {
            path: {
                "needed": ["libc.so.6"] if path != LIBC else [],
                "interpreter": None,
                "version_needs": {"libc.so.6": ["DEMO_1"]} if path != LIBC else {},
            }
            for path in elf
        },
        "elf_edges": [[path, "libc.so.6", LIBC] for path in elf if path != LIBC],
    }
    evidence = {
        "profile": qualify.PROFILE,
        "receipt_sha256": contract.digest(contract.canonical(receipt)),
        "abi_request_sha256": receipt["abi_request_sha256"],
        "static_report_sha256": contract.digest(contract.canonical(report)),
        "view": {
            "mountinfo": artifact(tmp_path / "mountinfo", MOUNTINFO),
            "apparmor_sha256": PIN,
            "seccomp_sha256": PIN,
            "namespaces": {"mnt": 1, "net": 2, "pid": 3},
        },
        "libvirt": daemon(),
        "loader": {name: loader_case(tmp_path, name) for name in qualify.PROBES},
    }
    return receipt, request, report, evidence


def assess(inputs, artifacts: Path) -> dict[str, Any]:
    return qualify.assess(
        *inputs,
        receipt_sha256=inputs[3]["receipt_sha256"],
        static_report_sha256=inputs[3]["static_report_sha256"],
        artifacts=artifacts,
    )


def test_qualification_reports_observed_cases_with_weak_binding(inputs, tmp_path: Path) -> None:
    result = assess(inputs, tmp_path)
    assert result["status"] == "EVIDENCE_CONSISTENT_REVIEW_REQUIRED"
    assert result["authorization"] == "NOT_GRANTED"
    assert result["weak_symbol_absence"] == "NOT_INFERRED_FROM_TRACE"
    assert result["cases"]["python-imports"] == {
        "observed_elf_objects": 4,
        "observed_dlopen_objects": 2,
        "observed_symbol_bindings": 1,
        "observed_version_checks": 1,
        "weak_references_observed_bound": 1,
        "weak_references_without_observed_binding": 5,
    }


@pytest.mark.parametrize(
    ("field", "value", "error"),
    [
        ("socket", "/run/libvirt/libvirt-sock", "LIBVIRT_SOCKET_SCOPE"),
        ("peer", [123, 0, 0], "LIBVIRT_PEER_IDENTITY"),
        ("host_socket_identity", [11, 24, 0, 0, 0o777], "LIBVIRT_PEER_IDENTITY"),
        ("host_peer", [201, 0, 0], "LIBVIRT_DAEMON_IDENTITY"),
        ("connect_paths", [SOCKET, "/run/libvirt/libvirt-admin-sock"], "LIBVIRT_CLIENT_BOUNDARY"),
        ("client_config_sha256", PIN, "LIBVIRT_CLIENT_BOUNDARY"),
        ("fixture_denial", {}, "LIBVIRT_DENIAL_SCHEMA"),
        ("read_results", {"getURI": "PASS"}, "LIBVIRT_READ_SCOPE"),
    ],
)
def test_refuses_libvirt_boundary_drift(inputs, tmp_path: Path, field, value, error) -> None:
    inputs[3]["libvirt"][field] = value
    with pytest.raises(contract.Refusal, match=error):
        assess(inputs, tmp_path)


def test_readonly_client_flag_does_not_prove_server_denial(inputs, tmp_path: Path) -> None:
    inputs[3]["libvirt"]["fixture_denial"]["client_connect_flags"] = 1
    with pytest.raises(contract.Refusal, match="LIBVIRT_SERVER_READONLY_UNPROVEN"):
        assess(inputs, tmp_path)


@pytest.mark.parametrize(
    ("field", "before", "after", "error"),
    [
        ("trace", LIBC, "/tmp/rogue.so", "UNAPPROVED_RUNTIME_OBJECT"),
        ("trace", "[0]", "[2]", "RUNTIME_LINK_NAMESPACE"),
        ("trace", "123:", "124:", "RUNTIME_TRACE_PROCESS_MISMATCH"),
        ("trace", "symbol `demo'", "symbol `unknown'", "RUNTIME_SYMBOL_PROVIDER_MISMATCH"),
        ("trace", "`DEMO_1' in file", "`OTHER' in file", "RUNTIME_VERSION_PROVIDER"),
        (
            "trace",
            "file=libc.so.6 [0]; needed by",
            "file=other.so [0]; needed by",
            "RUNTIME_NEEDED_OR_DLOPEN_MISMATCH",
        ),
        ("maps", f"50 {PYTHON}", f"99 {PYTHON}", "RUNTIME_OBJECT_MEASUREMENT_MISMATCH"),
        ("maps", f"50 {PYTHON}", f"50 {PYTHON} (deleted)", "RUNTIME_DELETED_MAPPING"),
        ("maps", f"50 {PYTHON}", "50 /tmp/rogue.so", "UNAPPROVED_RUNTIME_OBJECT"),
        ("maps", "r-xp", "rwxp", "RUNTIME_WRITABLE_EXECUTABLE"),
    ],
)
def test_refuses_loader_drift(inputs, tmp_path: Path, field, before, after, error) -> None:
    case = inputs[3]["loader"]["python-startup"]
    path = tmp_path / case[field]["path"]
    case[field] = artifact(path, path.read_text().replace(before, after, 1).encode())
    with pytest.raises(contract.Refusal, match=error):
        assess(inputs, tmp_path)


def test_refuses_static_provider_mismatch(inputs, tmp_path: Path) -> None:
    inputs[2]["elf_edges"][0][-1] = CFFI
    with pytest.raises(contract.Refusal, match="STATIC_ABI_EDGE_MISMATCH"):
        assess(inputs, tmp_path)


def test_refuses_unmeasured_dlopen(inputs, tmp_path: Path) -> None:
    del inputs[3]["loader"]["python-imports"]["objects"][RUST]
    with pytest.raises(contract.Refusal, match="RUNTIME_OBJECT_MEASUREMENTS_INCOMPLETE"):
        assess(inputs, tmp_path)


@pytest.mark.parametrize(
    ("raw", "error"),
    [
        (MOUNTINFO.replace(b"/ / ro", b"/ / rw"), "WRITABLE_SUBMOUNT"),
        (MOUNTINFO + b"25 21 00:15 / /extended ro - tmpfs tmpfs ro\n", "UNAPPROVED_SUBMOUNT"),
        (MOUNTINFO.rsplit(b"\n", 2)[0], "NATIVE_MOUNT_SET_REJECTED"),
    ],
)
def test_refuses_wrong_native_mounts(inputs, tmp_path: Path, raw: bytes, error: str) -> None:
    inputs[3]["view"]["mountinfo"] = artifact(tmp_path / "mountinfo", raw)
    with pytest.raises(ValueError, match=error):
        assess(inputs, tmp_path)


def test_refuses_qemu_with_unapproved_arguments(inputs, tmp_path: Path) -> None:
    inputs[3]["loader"]["qemu-raw"]["outcome"]["argv"][1] = "convert"
    with pytest.raises(contract.Refusal, match="QEMU_PROBE_REJECTED"):
        assess(inputs, tmp_path)


@pytest.mark.parametrize(
    ("field", "replacement", "error"),
    [
        ("label", qualify.PARENT_LABEL, "RUNTIME_APPARMOR_LABEL_REJECTED"),
        (
            "status",
            KERNEL_STATUS.replace(b"Seccomp:\t2", b"Seccomp:\t0"),
            "RUNTIME_KERNEL_STATUS_REJECTED",
        ),
        (
            "status",
            KERNEL_STATUS.replace(b"NSpid:\t10001\t123", b"NSpid:\t124"),
            "RUNTIME_KERNEL_STATUS_REJECTED",
        ),
    ],
)
def test_refuses_nonconforming_qemu_child(
    inputs, tmp_path: Path, field, replacement, error
) -> None:
    inputs[3]["loader"]["qemu-qcow2"][field] = artifact(tmp_path / "qemu-override", replacement)
    with pytest.raises(contract.Refusal, match=error):
        assess(inputs, tmp_path)


def test_cli_requires_independent_pins_and_never_grants_launch(
    inputs, tmp_path: Path, capsys
) -> None:
    receipt, request, report, evidence = inputs
    paths = [tmp_path / f"input-{index}.json" for index in range(4)]
    for path, value in zip(paths, inputs, strict=True):
        artifact(path, contract.canonical(value))
    argv = [
        "--receipt",
        str(paths[0]),
        "--receipt-sha256",
        evidence["receipt_sha256"],
        "--abi-request",
        str(paths[1]),
        "--static-report",
        str(paths[2]),
        "--static-report-sha256",
        evidence["static_report_sha256"],
    ]
    assert qualify.main(argv) == 0
    assert contract.loads(capsys.readouterr().out.encode())["status"] == "NOT_TESTED"
    partial = ["--evidence", str(paths[3])]
    assert qualify.main(argv + partial) == 2
    assert "NATIVE_EVIDENCE_INPUTS_REQUIRED" in capsys.readouterr().out
    full = partial + [
        "--evidence-sha256",
        contract.digest(contract.canonical(evidence)),
        "--artifacts",
        str(tmp_path),
    ]
    assert qualify.main(argv + full) == 0
    assert "EVIDENCE_CONSISTENT_REVIEW_REQUIRED" in capsys.readouterr().out
    paths[3].write_bytes(b"{}")
    assert qualify.main(argv + full) == 2
    assert "QUALIFICATION_ARTIFACT_PIN_MISMATCH" in capsys.readouterr().out


def test_emit_static_abi_report_only_reuses_v5d10(
    inputs, tmp_path: Path, monkeypatch, capsys
) -> None:
    receipt, request, report, _ = inputs
    expected = {"review_reference": "synthetic-reviewed-expectations"}
    receipt["abi_expectations_sha256"] = contract.digest(contract.canonical(expected))
    paths = [tmp_path / f"static-{index}.json" for index in range(3)]
    for path, value in zip(paths, (receipt, request, expected), strict=True):
        artifact(path, contract.canonical(value))

    def replay(request_arg, expected_arg):
        assert request_arg == request
        assert expected_arg == expected
        return report

    monkeypatch.setattr(qualify.abi, "diagnose", replay)
    argv = [
        "--receipt",
        str(paths[0]),
        "--receipt-sha256",
        contract.digest(paths[0].read_bytes()),
        "--abi-request",
        str(paths[1]),
        "--abi-expectations",
        str(paths[2]),
        "--emit-static-report",
    ]
    assert qualify.main(argv) == 0
    assert contract.loads(capsys.readouterr().out.encode()) == report
    paths[2].write_bytes(b"{}")
    assert qualify.main(argv) == 2
    assert "QUALIFICATION_ARTIFACT_PIN_MISMATCH" in capsys.readouterr().out


FIXTURE_UUID = "5b7e52eb-f591-45f4-8c18-b451658b70a1"


@pytest.fixture
def libvirt_double():
    log = []

    def call(name, output):
        def record(*args):
            log.append((name, args))
            return output

        return record

    domain = SimpleNamespace(
        UUIDString=call("UUIDString", FIXTURE_UUID),
        XMLDesc=call(
            "XMLDesc",
            '<domain><devices><interface><source network="fixture"/></interface></devices></domain>',
        ),
        state=call("state", [5, 0]),
        info=call("info", [5, 1024, 512, 1, 0]),
        schedulerParameters=call("schedulerParameters", {}),
        ID=call("ID", -1),
        isActive=call("isActive", 0),
        isPersistent=call("isPersistent", 1),
        hasManagedSaveImage=call("hasManagedSaveImage", 0),
        snapshotListNames=call("snapshotListNames", []),
    )
    network = SimpleNamespace(
        UUIDString=call("network.UUIDString", FIXTURE_UUID),
        isActive=call("network.isActive", 0),
        XMLDesc=call("network.XMLDesc", "<network><name>fixture</name></network>"),
    )
    connection = SimpleNamespace(
        getURI=call("getURI", diagnostic.URI),
        getType=call("getType", "QEMU"),
        getLibVersion=call("getLibVersion", 1),
        lookupByUUIDString=call("lookupByUUIDString", domain),
        networkLookupByName=call("networkLookupByName", network),
        close=call("close", 0),
    )
    module = SimpleNamespace(openReadOnly=call("openReadOnly", connection))
    return module, connection, domain, network, log


def test_diagnostic_reads_only_expected_fixture_rpcs(libvirt_double) -> None:
    module, _, _, _, log = libvirt_double
    assert diagnostic.exercise_libvirt(module, FIXTURE_UUID, "fixture") == dict.fromkeys(
        qualify.READS, "PASS"
    )
    assert log[0] == ("openReadOnly", (diagnostic.URI,))
    assert log[-1] == ("close", ())
    assert [args for name, args in log if name == "XMLDesc"] == [(0,), (2,)]
    assert [args for name, args in log if name == "state"] == [(0,), (0,)]


@pytest.mark.parametrize(
    ("object_index", "method", "result", "error"),
    [
        (1, "getURI", "qemu+ssh://wrong/system", "DIAGNOSTIC_LIBVIRT_CONNECTION_REJECTED"),
        (1, "getLibVersion", 0, "DIAGNOSTIC_LIBVIRT_VERSION_REJECTED"),
        (2, "XMLDesc", None, "DIAGNOSTIC_XML_REJECTED"),
        (2, "XMLDesc", "<domain/>", "DIAGNOSTIC_FIXTURE_NETWORK_REJECTED"),
        (2, "state", [], "DIAGNOSTIC_FIXTURE_STATE_CHANGED"),
        (2, "schedulerParameters", None, "DIAGNOSTIC_FIXTURE_SCHEDULER_REJECTED"),
        (3, "XMLDesc", "<network/>", "DIAGNOSTIC_FIXTURE_NETWORK_REJECTED"),
    ],
)
def test_diagnostic_refuses_incomplete_observation_and_closes_connection(
    libvirt_double, monkeypatch, object_index, method, result, error
) -> None:
    module, _, _, _, log = libvirt_double
    monkeypatch.setattr(libvirt_double[object_index], method, lambda *_: result)
    with pytest.raises(diagnostic.Refusal, match=error):
        diagnostic.exercise_libvirt(module, FIXTURE_UUID, "fixture")
    assert log[-1] == ("close", ())


def test_diagnostic_import_probe_never_imports_observer(monkeypatch) -> None:
    imported = []
    monkeypatch.setattr(diagnostic, "import_module", imported.append)
    assert not imported
    assert diagnostic.exercise_imports() == tuple(imported)
    assert imported == [
        "_cffi_backend",
        "cryptography.hazmat.bindings._rust",
        "defusedxml.ElementTree",
        "pycparser",
    ]
