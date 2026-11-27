"""Synthetic carrier/collector checks without running any profile-A executable."""

from __future__ import annotations

import copy
import io
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

from scripts import stage9_v5d11_capture as capture
from scripts import stage9_v5d11_diagnostic as diagnostic

pytestmark = [pytest.mark.unit, pytest.mark.skipif(sys.platform != "linux", reason="Linux capture")]
PIN = "a" * 64
EXE = "/usr/bin/qemu-img"
ROOT = "/opt/profile-a"
PYEXE = f"{ROOT}/venv/bin/python3.12"
NAMES = {"mnt": 501, "pid": 502, "net": 503}


def valid_config():
    return {
        "root": ROOT,
        "socket": "/run/libvirt/libvirt-sock-ro",
        "domain_uuid": "90ae2678-fd31-4ba7-8fe8-d4ff0f004814",
        "network_name": "isolated-fixture",
        "images": {
            kind: {"path": f"/var/lib/libvirt/images/fixture.{kind}", "sha256": PIN}
            for kind in ("raw", "qcow2")
        },
    }


@pytest.mark.parametrize(
    "change", ["", "root", "socket", "network_name", "image", "collision", "extra"]
)
def test_diagnostic_config(change):
    data = copy.deepcopy(valid_config())
    if change in {"root", "socket", "network_name"}:
        data[change] = "/else/../"
    elif change == "image":
        data["images"]["raw"]["sha256"] = "x"
    elif change == "collision":
        data["images"]["raw"] = data["images"]["qcow2"]
    elif change == "extra":
        data["command"] = "/bin/sh"
    raw = json.dumps(data).encode()
    if change:
        with pytest.raises(ValueError):
            diagnostic.validate_inputs(raw)
    else:
        assert diagnostic.validate_inputs(raw) == data


def test_diagnostic_inputs_bounded(tmp_path, monkeypatch):
    config = tmp_path / "config"
    monkeypatch.setattr(diagnostic, "INPUTS", str(config))
    config.write_bytes(json.dumps(valid_config()).encode())
    assert diagnostic.inputs() == valid_config()
    config.write_bytes(b"x" * 65537)
    with pytest.raises(diagnostic.Refusal, match="INPUT_SIZE"):
        diagnostic.inputs()


@pytest.mark.parametrize("case", diagnostic.PYTHON_CASES)
def test_fixed_python_case_reports_only_closed_events(case, monkeypatch):
    events = []
    acknowledgements = []
    invoked = []
    monkeypatch.setattr(sys, "path", list(sys.path))
    monkeypatch.setattr(
        diagnostic, "emit", lambda kind, name, **kwargs: events.append((kind, name, kwargs))
    )
    monkeypatch.setattr(diagnostic, "control", acknowledgements.append)
    monkeypatch.setattr(diagnostic, "exercise_imports", lambda: invoked.append("imports"))
    monkeypatch.setattr(diagnostic, "view_socket", lambda path: {"socket": path})
    monkeypatch.setattr(diagnostic, "import_module", lambda name: name)

    def read_libvirt(module, domain, network, checkpoint):
        invoked.append(module)
        checkpoint({})
        return {"close": "PASS"}

    monkeypatch.setattr(diagnostic, "exercise_libvirt", read_libvirt)
    diagnostic.python_probe(case, valid_config())
    assert [item[:2] for item in events] == [("ready", case), ("result", case)]
    assert all(event[2]["pid"] == os.getpid() for event in events)
    assert acknowledgements == [b"ACK\n"]
    assert (
        invoked
        == {"python-startup": [], "python-imports": ["imports"], "python-libvirt": ["libvirt"]}[
            case
        ]
    )


def test_fixed_invocation_and_env(monkeypatch):
    flags = SimpleNamespace(isolated=1, no_site=1)
    run = []
    monkeypatch.setattr(sys, "flags", flags)
    monkeypatch.setattr(sys, "dont_write_bytecode", True)
    monkeypatch.setattr(sys, "argv", [diagnostic.PAYLOAD, "all"])
    monkeypatch.setattr(os, "geteuid", lambda: 20)
    monkeypatch.setattr(os, "environ", diagnostic.COMMAND_ENV)
    monkeypatch.setattr(diagnostic, "inputs", valid_config)
    monkeypatch.setattr(diagnostic, "carrier", lambda config: run.append(config["root"]))
    assert diagnostic.main() == 0 and run == [ROOT]
    monkeypatch.setattr(sys, "argv", [diagnostic.PAYLOAD, "/bin/sh"])
    with pytest.raises(diagnostic.Refusal, match="CASE_REFUSED"):
        diagnostic.main()
    monkeypatch.setattr(sys, "argv", ["/tmp/extra", "all"])
    with pytest.raises(diagnostic.Refusal, match="INVOCATION"):
        diagnostic.main()


@pytest.mark.parametrize("consume", [b"ACK\n", b"bad", b""])
def test_control_matches_exact_bytes(monkeypatch, consume):
    stream = io.BytesIO(consume)
    monkeypatch.setattr(os, "read", lambda fd, count: stream.read(count))
    if consume == b"ACK\n":
        diagnostic.control(b"ACK\n")
    else:
        with pytest.raises(diagnostic.Refusal, match="CONTROL_REFUSED"):
            diagnostic.control(b"ACK\n")


def test_capture_event_trace_barrier():
    events_r, events_w = os.pipe()
    trace_r, trace_w = os.pipe()
    protocol = io.BytesIO()
    with os.fdopen(events_r, "rb") as events, os.fdopen(trace_r, "rb") as trace:
        try:
            fake = SimpleNamespace(stdin=protocol, stdout=events, stderr=trace)
            channel = capture.Channel(fake, timeout=1)
            os.write(trace_w, b"trace\n")
            os.write(events_w, b'{"event":"ready","case":"python-startup","pid":2}\n')
            assert channel.event() == {"event": "ready", "case": "python-startup", "pid": 2}
            os.write(trace_w, b"fini\n")
            assert channel.finish_case() == b"trace\nfini\n"
            channel.command(b"ACK\n")
            assert protocol.getvalue() == b"ACK\n"
            with pytest.raises(ValueError, match="LOADER_TRACE_MISSING"):
                channel.finish_case()
        finally:
            os.close(events_w)
            os.close(trace_w)


def test_capture_event_refuses_ambiguity_and_artifact_overwrite(tmp_path):
    capture.expect(
        {"event": "ready", "case": "python-startup", "pid": 2}, "python-startup", "ready", set()
    )
    with pytest.raises(ValueError, match="CASE_PROTOCOL"):
        capture.expect(
            {"event": "ready", "case": "python-startup", "pid": 2, "argv": []},
            "python-startup",
            "ready",
            set(),
        )
    assert capture.artifact(tmp_path, "trace", b"abc")["path"] == "trace"
    with pytest.raises(FileExistsError):
        capture.artifact(tmp_path, "trace", b"changed")
    with pytest.raises(ValueError, match="ARTIFACT_PATH"):
        capture.artifact(tmp_path, "../outside", b"abc")
    with pytest.raises(ValueError, match="ARTIFACT_SIZE"):
        capture.artifact(tmp_path, "empty", b"")
    assert (tmp_path / "trace").read_bytes() == b"abc"


def test_maps_normalization_is_anchored_and_facts_are_kept(tmp_path):
    current = str(tmp_path).encode()
    raw = (
        b"1-2 r-xp 0000 08:01 41 " + current + b"/usr/bin/qemu-img\n2-3 r-xp 0000 00:00 0 [vdso]\n"
    )
    expected = b"1-2 r-xp 0000 08:01 41 /usr/bin/qemu-img\n2-3 r-xp 0000 00:00 0 [vdso]\n"
    assert capture.normalize_maps(raw, tmp_path) == expected
    assert capture.normalize_maps(raw, tmp_path / "different") == raw
    assert capture.field(b"Name:\ta\nNSpid:\t20 1\n", b"NSpid") == [b"20", b"1"]
    with pytest.raises(ValueError, match="PROC_STATUS"):
        capture.field(b"Pid:\t20\nPid:\t21", b"Pid")


def test_real_proc_lifetime_and_namespace_probe():
    current = Path(f"/proc/{os.getpid()}")
    assert capture.ticks(current) > 0
    assert set(capture.namespaces(current)) == set(NAMES)


@pytest.mark.parametrize("drift", [False, True])
def test_complete_capture_authorized_cases_and_drift(tmp_path, monkeypatch, drift):
    observed = []
    commands = []
    events = []
    for i, case in enumerate(capture.CASES, start=2):
        status = {"event": "ready", "case": case, "pid": i}
        result = {**status, "event": "result"}
        if case == "python-libvirt":
            status["view"] = {"socket_identity": [3, 4, 0, 0, 0o666]}
            result["read_results"] = {"close": "PASS"}
        if case.startswith("qemu-"):
            argv = ["/proc/self/fd/1", "info", "--output=json", "-f", case[5:], "/proc/self/fd/2"]
            status.update(event="begin", argv=argv)
            result["outcome"] = {"argv": argv if not drift else [EXE]}
        events.extend((status, result))
        if not case.startswith("qemu-"):
            events.append({"event": "end", "case": case, "pid": i})

    class Channel:
        def __init__(self, process):
            assert process.pid == 100

        def event(self, sample=None):
            if sample:
                sample()
                sample()
            return events.pop(0)

        def command(self, data):
            commands.append(data)

        def finish_case(self):
            return b"glibc loader trace fixture"

    monkeypatch.setattr(capture, "Channel", Channel)
    monkeypatch.setattr(capture, "bounded", lambda *_: b"PPid: 100\nNSpid: 101 1\n")
    monkeypatch.setattr(capture, "ticks", lambda *_: 1)
    monkeypatch.setattr(capture, "namespaces", lambda *_: NAMES)
    monkeypatch.setattr(
        capture, "target", lambda carrier, pid, names: capture.Target(100 + pid, pid, 2)
    )
    monkeypatch.setattr(capture.qualify, "_objects", lambda *_: ({}, {}, {}))
    samples = iter(
        [
            (b"first map", {"/usr/lib/transient.so": (1, 2, 3)}),
            (b"later map", {"/usr/lib/later.so": (1, 2, 4)}),
        ]
        * 2
    )

    def snapshot(*args, qemu):
        raw, identities = next(samples) if qemu else (b"maps", {})
        return capture.Snapshot(raw, raw, b"status", b"label", identities)

    monkeypatch.setattr(capture, "snapshot", snapshot)
    monkeypatch.setattr(capture, "object_evidence", lambda *a, **kw: ({}, {}))
    processo = SimpleNamespace(pid=100, wait=lambda timeout: 0)

    def libvirt(host_pid, start):
        observed.append((host_pid, start))
        return ["/run/libvirt/libvirt-sock-ro"]

    def collect():
        return capture.collect(
            processo,
            101,
            {"namespaces": NAMES},
            {},
            {"interpreter": PYEXE},
            {},
            tmp_path,
            staging_root=tmp_path,
            on_libvirt=libvirt,
        )

    if drift:
        with pytest.raises(ValueError, match="QEMU_ARGV_DRIFT"):
            collect()
    else:
        cases, virt = collect()
        assert list(cases) == list(capture.CASES)
        assert commands[0] == b"DIAGNOSTIC\n" and not events
        assert commands.count(b"RUN\n") == 2
        assert observed == [(104, 2)]
        assert virt["read_results"] == {"close": "PASS"}
        assert (tmp_path / "qemu-raw.trace").read_bytes() == b"glibc loader trace fixture"
        assert [
            (tmp_path / item["raw"]["path"]).read_bytes() for item in cases["qemu-raw"]["maps"]
        ] == [b"first map", b"later map"]


def test_target_checks_real_ancestry_namespace_and_lifetime(monkeypatch):
    carrier = capture.Target(101, 1, 7)
    monkeypatch.setattr(capture, "ticks", lambda proc: 7 if proc.name == "101" else 8)
    monkeypatch.setattr(capture, "namespaces", lambda _: NAMES)

    def proc_file(path, _limit):
        if path.name == "children":
            return b"102 "
        return b"PPid:\t101\nNSpid:\t102 2\n"

    monkeypatch.setattr(capture, "bounded", proc_file)
    assert capture.target(carrier, 2, NAMES) == capture.Target(102, 2, 8)
    with pytest.raises(ValueError, match="CHILD_IDENTITY"):
        capture.target(carrier, 3, NAMES)
    with pytest.raises(ValueError, match="CHILD_IDENTITY"):
        capture.target(carrier, 2, {**NAMES, "pid": 0})
    with pytest.raises(ValueError, match="CAPTURE_TARGET"):
        capture.target(carrier, 1, NAMES)
    monkeypatch.setattr(capture, "bounded", lambda *_: b"102 104")
    with pytest.raises(ValueError, match="CHILD_AMBIGUOUS"):
        capture.target(carrier, 2, NAMES)


@pytest.mark.parametrize(
    ("change", "reason"),
    [
        ("", ""),
        ("namespace", "CHILD_IDENTITY"),
        ("environ", "ENVIRONMENT_REJECTED"),
        ("exe", "EXECUTABLE_IDENTITY"),
        ("maps_inode", "CHILD_IDENTITY"),
        ("reused", "CHILD_REUSED"),
        ("parked", ""),
    ],
)
def test_snapshot_verified_live_inputs(tmp_path, monkeypatch, change, reason):
    tree = tmp_path / "isolated"
    program = tree / EXE[1:]
    program.parent.mkdir(parents=True)
    program.touch()
    proc = tmp_path / "proc"
    (proc / "attr/apparmor").mkdir(parents=True)
    (proc / "exe").symlink_to(program if change != "exe" else __file__)
    (proc / "attr/apparmor/current").write_bytes(
        capture.PARENT_LABEL if change == "parked" else capture.QEMU_LABEL
    )
    (proc / "status").write_bytes(b"synthetic")
    trace_env = {**diagnostic.TRACE_ENV}
    if change == "environ":
        trace_env["LD_PRELOAD"] = "/not-approved.so"
    (proc / "environ").write_bytes(
        b"\0".join(f"{key}={value}".encode() for key, value in trace_env.items()) + b"\0"
    )
    major, minor, inode = capture.file_identity(program.stat())
    mapped_inode = inode + (change == "maps_inode")
    raw_maps = f"1-2 r-xp 0000 {major:x}:{minor:x} {mapped_inode} {program}\n2-3 r-xp 0000 00:00 0 [vdso]\n".encode()
    (proc / "maps").write_bytes(raw_maps)
    monkeypatch.setattr(capture.Target, "proc", property(lambda _: proc))
    monkeypatch.setattr(capture, "ticks", lambda _: 8 if change != "reused" else 9)
    monkeypatch.setattr(capture, "namespaces", lambda _: NAMES if change != "namespace" else {})
    monkeypatch.setattr(capture.qualify, "_kernel_status", lambda *_: None)
    subject = capture.Target(101, 2, 8)

    def run():
        return capture.snapshot(subject, NAMES, tree, EXE, lambda name: name, qemu=True)

    if reason:
        with pytest.raises(ValueError, match=reason):
            run()
    elif change == "parked":
        assert run() is None
    else:
        observed = run()
        assert observed.raw_maps == raw_maps
        assert observed.mapped[EXE] == (major, minor, inode)


def test_object_evidence_pins_symbols_and_inode(tmp_path, monkeypatch):
    tree = tmp_path / "isolated"
    original = tree / EXE[1:]
    original.parent.mkdir(parents=True)
    original.write_bytes(b"approved diagnostic fixture ELF bytes")
    true_lstat = Path.lstat

    def synthetic_ownership(path):
        original_result = true_lstat(path)
        return (
            SimpleNamespace(
                st_mode=original_result.st_mode,
                st_uid=0,
                st_nlink=original_result.st_nlink,
                st_dev=original_result.st_dev,
                st_ino=original_result.st_ino,
            )
            if path == original
            else original_result
        )

    digest = capture.contract.digest(original.read_bytes())
    identity = capture.file_identity(original.stat())
    monkeypatch.setattr(Path, "lstat", synthetic_ownership)
    monkeypatch.setattr(capture.qualify, "_trace", lambda *_: ({EXE}, None, None, None, None))
    monkeypatch.setattr(capture.contract, "secure_read", lambda path, **_: path.read_bytes())
    fake_symbols = b"Symbol table '.dynsym' contains 0 entries:\n"
    monkeypatch.setattr(
        capture.subprocess,
        "run",
        lambda *a, **kw: SimpleNamespace(returncode=0, stdout=fake_symbols),
    )
    output = tmp_path / "bundle"
    output.mkdir()
    subject = capture.Target(101, 2, 7)

    def run(pin, mapped):
        return capture.object_evidence(
            output,
            "qemu-raw",
            tree,
            b"trace",
            mapped,
            subject,
            lambda x: x,
            {EXE: pin},
            {"library_map": {}},
        )

    with pytest.raises(ValueError, match="OBJECT_PIN_MISMATCH"):
        run("0" * 64, {EXE: identity})
    with pytest.raises(ValueError, match="OBJECT_PIN_MISMATCH"):
        run(digest, {EXE: (identity[0], identity[1], identity[2] + 1)})
    with pytest.raises(ValueError, match="OBJECT_MAPPING_UNOBSERVED"):
        run(digest, {})
    obj, symbols = run(digest, {EXE: identity})
    assert obj == {EXE: {"sha256": digest, "identity": list(identity)}}
    assert (output / symbols[EXE]["path"]).read_bytes() == fake_symbols


def test_catalogue_carrier_forks_only_fixed_cases_and_never_dispatches_observer(monkeypatch):
    stages = []
    events = []
    controlled = []
    monkeypatch.setattr(os, "fork", lambda: 30)
    monkeypatch.setattr(os, "waitpid", lambda pid, _: (pid, 0))
    monkeypatch.setattr(
        diagnostic, "emit", lambda event, case, **kw: events.append((event, case, kw))
    )
    monkeypatch.setattr(diagnostic, "control", controlled.append)
    monkeypatch.setattr(diagnostic, "qemu_probe", lambda fmt, _: stages.append(fmt))
    diagnostic.carrier(valid_config())
    assert events == [("end", case, {"pid": 30}) for case in diagnostic.PYTHON_CASES]
    assert stages == ["raw", "qcow2"] and controlled == [b"ACK\n"] * 3


@pytest.mark.parametrize("flags", [0, 0x10])
def test_qemu_refuses_image_without_kernel_immutable_bit_in_fixture(tmp_path, monkeypatch, flags):
    import fcntl
    import struct

    binary = tmp_path / "fake-qemu"
    image = tmp_path / "image"
    binary.touch(mode=0o500)
    image.touch(mode=0o400)
    real_fstat = os.fstat

    def synthetic_ownership(fd):
        value = list(real_fstat(fd))
        value[3], value[4] = 1, 0
        return os.stat_result(value)

    monkeypatch.setattr(diagnostic, "QEMU", str(binary))
    monkeypatch.setattr(os, "fstat", synthetic_ownership)
    monkeypatch.setattr(fcntl, "ioctl", lambda fd, request, raw: struct.pack("=Q", flags))
    if flags:
        first, second = diagnostic.qemu_handles(str(image))
        os.close(first)
        os.close(second)
    else:
        with pytest.raises(ValueError, match="QEMU_NOT_IMMUTABLE"):
            diagnostic.qemu_handles(str(image))


@pytest.mark.parametrize("fmt", ["raw", "qcow2", "different"])
def test_qemu_probe_waits_for_run_and_returns_only_fixed_info(tmp_path, monkeypatch, fmt):
    executable = tmp_path / "binary"
    fixture = tmp_path / "image"
    executable.touch()
    fixture.touch()
    binary = os.open(executable, os.O_RDONLY)
    image = os.open(fixture, os.O_RDONLY)
    input_channel, send = os.pipe()
    # Keep the mock child read end alive as a separate owner, which makes
    # the carrier's RUN write observe the same pipe lifetime as in Linux.
    held_channel = os.dup(input_channel)
    receive, output_channel = os.pipe()
    os.write(output_channel, json.dumps({"format": fmt}).encode())
    channels = iter([(input_channel, send), (receive, output_channel)])
    events = []
    controls = []
    monkeypatch.setattr(diagnostic, "qemu_handles", lambda _: (binary, image))
    monkeypatch.setattr(os, "pipe", lambda: next(channels))
    monkeypatch.setattr(os, "fork", lambda: 37)
    monkeypatch.setattr(os, "waitpid", lambda pid, _: (pid, 0))
    monkeypatch.setattr(
        diagnostic, "emit", lambda kind, case, **kw: events.append((kind, case, kw))
    )
    monkeypatch.setattr(diagnostic, "control", controls.append)
    try:
        if fmt == "different":
            with pytest.raises(ValueError, match="QEMU_OUTPUT"):
                diagnostic.qemu_probe("raw", valid_config())
            assert controls == [b"RUN\n"]
        else:
            diagnostic.qemu_probe(fmt, valid_config())
            assert [event[0] for event in events] == ["begin", "result"]
            outcome = events[1][2]["outcome"]
            assert outcome["fixture_sha256"] == PIN and outcome["json_format"] == fmt
            assert outcome["argv"][1:5] == ["info", "--output=json", "-f", fmt]
            assert controls == [b"RUN\n", b"ACK\n"]
        assert os.read(held_channel, 1) == b"1"
    finally:
        os.close(held_channel)
