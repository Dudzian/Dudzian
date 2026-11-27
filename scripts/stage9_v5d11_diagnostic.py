"""Fixed diagnostic cases for the isolated V5d11 carrier."""

from __future__ import annotations

from collections.abc import Callable
from importlib import import_module
import re
from types import ModuleType
from typing import Any
import uuid

MAX_XML_BYTES = 2 * 1024 * 1024
URI = "qemu:///system"


class Refusal(ValueError):
    """The diagnostic fixture or read observations were incompatible."""


def valid_network_name(value: Any) -> bool:
    return type(value) is str and re.fullmatch(r"[A-Za-z0-9_.-]{1,128}", value) is not None


def exercise_imports() -> tuple[str, ...]:
    """Force the two approved wheel ELF extensions to load, without Observer."""
    names = (
        "_cffi_backend",
        "cryptography.hazmat.bindings._rust",
        "defusedxml.ElementTree",
        "pycparser",
    )
    for name in names:
        import_module(name)
    return names


def _uuid(value: str) -> str:
    if str(uuid.UUID(value)) != value:
        raise Refusal("DIAGNOSTIC_FIXTURE_UUID_REJECTED")
    return value


def _bounded_xml(raw: Any) -> str:
    if type(raw) is not str or not (0 < len(raw.encode("utf-8")) <= MAX_XML_BYTES):
        raise Refusal("DIAGNOSTIC_XML_REJECTED")
    return raw


def exercise_libvirt(
    module: ModuleType,
    domain_uuid: str,
    network_name: str,
    checkpoint: Callable[[dict[str, str]], None] | None = None,
) -> dict[str, str]:
    """Exercise V5d1 read RPCs on a preprovisioned disposable domain."""
    expected_uuid = _uuid(domain_uuid)
    if not valid_network_name(network_name):
        raise Refusal("DIAGNOSTIC_FIXTURE_NETWORK_REJECTED")
    connection = module.openReadOnly(URI)
    if connection is None:
        raise Refusal("DIAGNOSTIC_LIBVIRT_CONNECTION_REJECTED")
    steps: dict[str, str] = {}
    try:
        if connection.getURI() != URI or connection.getType() != "QEMU":
            raise Refusal("DIAGNOSTIC_LIBVIRT_CONNECTION_REJECTED")
        version = connection.getLibVersion()
        if type(version) is not int or version <= 0:
            raise Refusal("DIAGNOSTIC_LIBVIRT_VERSION_REJECTED")
        steps.update(dict.fromkeys(("getURI", "getType", "getLibVersion"), "PASS"))
        domain = connection.lookupByUUIDString(expected_uuid)
        if domain.UUIDString() != expected_uuid:
            raise Refusal("DIAGNOSTIC_FIXTURE_UUID_REJECTED")
        steps.update(dict.fromkeys(("lookupByUUIDString", "UUIDString"), "PASS"))
        etree = import_module("defusedxml.ElementTree")
        active_xml = etree.fromstring(_bounded_xml(domain.XMLDesc(0)))
        inactive_xml = etree.fromstring(_bounded_xml(domain.XMLDesc(2)))
        if active_xml.tag != "domain" or inactive_xml.tag != "domain":
            raise Refusal("DIAGNOSTIC_XML_REJECTED")
        if network_name not in {
            item.get("network") for item in active_xml.findall("./devices/interface/source")
        }:
            raise Refusal("DIAGNOSTIC_FIXTURE_NETWORK_REJECTED")
        steps.update(dict.fromkeys(("XMLDesc(0)", "XMLDesc(2)"), "PASS"))
        state = tuple(domain.state(0))
        info = domain.info()
        if len(state) != 2 or len(info) != 5 or int(info[0]) != int(state[0]):
            raise Refusal("DIAGNOSTIC_FIXTURE_STATE_CHANGED")
        if type(domain.schedulerParameters()) is not dict:
            raise Refusal("DIAGNOSTIC_FIXTURE_SCHEDULER_REJECTED")
        steps.update(dict.fromkeys(("state(0)", "info", "schedulerParameters"), "PASS"))
        if type(domain.ID()) is not int or domain.isActive() not in {0, 1}:
            raise Refusal("DIAGNOSTIC_FIXTURE_STATE_CHANGED")
        if domain.isPersistent() not in {0, 1} or domain.hasManagedSaveImage(0) not in {0, 1}:
            raise Refusal("DIAGNOSTIC_FIXTURE_STATE_CHANGED")
        if type(domain.snapshotListNames(0)) is not list:
            raise Refusal("DIAGNOSTIC_FIXTURE_STATE_CHANGED")
        steps.update(
            dict.fromkeys(
                (
                    "ID",
                    "isActive",
                    "isPersistent",
                    "hasManagedSaveImage(0)",
                    "snapshotListNames(0)",
                ),
                "PASS",
            )
        )
        network = connection.networkLookupByName(network_name)
        _uuid(network.UUIDString())
        if network.isActive() not in {0, 1}:
            raise Refusal("DIAGNOSTIC_FIXTURE_NETWORK_REJECTED")
        network_xml = etree.fromstring(_bounded_xml(network.XMLDesc(0)))
        if network_xml.tag != "network" or network_xml.findtext("name") != network_name:
            raise Refusal("DIAGNOSTIC_FIXTURE_NETWORK_REJECTED")
        steps.update(
            dict.fromkeys(
                (
                    "networkLookupByName",
                    "network.UUIDString",
                    "network.isActive",
                    "network.XMLDesc(0)",
                ),
                "PASS",
            )
        )
        if domain.UUIDString() != expected_uuid or tuple(domain.state(0)) != state:
            raise Refusal("DIAGNOSTIC_FIXTURE_STATE_CHANGED")
        if checkpoint is not None:
            checkpoint(dict(steps))
    finally:
        connection.close()
    steps["close"] = "PASS"
    return steps


COMMAND_ENV = {"LANG": "C", "LC_ALL": "C", "PATH": "/usr/bin:/bin"}
TRACE_ENV = {**COMMAND_ENV, "LD_DEBUG": "files,bindings,versions"}
PAYLOAD = "/run/stage9-v5d11/diagnostic.py"
INPUTS = "/run/stage9-v5d11/diagnostic-inputs.json"
CONFIG = "/etc/libvirt/libvirt.conf"
QEMU = "/usr/bin/qemu-img"
PYTHON_CASES = ("python-startup", "python-imports", "python-libvirt")
IMAGE_FORMATS = ("raw", "qcow2")


def inputs() -> dict[str, Any]:
    with open(INPUTS, "rb") as stream:
        raw = stream.read(65537)
    if not 0 < len(raw) <= 65536:
        raise Refusal("DIAGNOSTIC_INPUT_SIZE")
    return validate_inputs(raw)


def validate_inputs(raw: bytes) -> dict[str, Any]:
    import json
    from pathlib import PurePosixPath

    def path(value: Any) -> bool:
        return (
            type(value) is str
            and re.fullmatch(r"(?:/[A-Za-z0-9_.+@-]+)+", value) is not None
            and all(part not in {".", ".."} for part in value.split("/")[1:])
            and PurePosixPath(value).as_posix() == value
        )

    doc: dict[str, Any] = json.loads(raw)
    if (
        type(doc) is not dict
        or set(doc) != {"root", "socket", "domain_uuid", "network_name", "images"}
        or not path(doc["root"])
        or doc["socket"] not in {"/run/libvirt/libvirt-sock-ro", "/run/libvirt/virtqemud-sock-ro"}
        or _uuid(doc["domain_uuid"]) != doc["domain_uuid"]
        or not valid_network_name(doc["network_name"])
        or type(doc["images"]) is not dict
        or set(doc["images"]) != set(IMAGE_FORMATS)
    ):
        raise Refusal("DIAGNOSTIC_INPUT_SCHEMA")
    for row in doc["images"].values():
        if (
            type(row) is not dict
            or set(row) != {"path", "sha256"}
            or not path(row["path"])
            or not row["path"].startswith("/var/lib/libvirt/images/")
            or type(row["sha256"]) is not str
            or not re.fullmatch(r"[0-9a-f]{64}", row["sha256"])
        ):
            raise Refusal("DIAGNOSTIC_IMAGE_SCHEMA")
    if doc["images"]["raw"]["path"] == doc["images"]["qcow2"]["path"]:
        raise Refusal("DIAGNOSTIC_IMAGE_COLLISION")
    return doc


def emit(event: str, case: str, **data: Any) -> None:
    import json
    import os

    raw = json.dumps({"event": event, "case": case, **data}, separators=(",", ":")).encode() + b"\n"
    if len(raw) > 4096 or os.write(1, raw) != len(raw):
        raise Refusal("DIAGNOSTIC_EVENT_WRITE")


def control(expected: bytes) -> None:
    import os

    for char in expected:
        if os.read(0, 1) != bytes([char]):
            raise Refusal("DIAGNOSTIC_CONTROL_REFUSED")


def view_socket(path: str) -> dict[str, Any]:
    import hashlib
    import os
    import socket
    import stat
    import struct

    meta = os.lstat(path)
    if not stat.S_ISSOCK(meta.st_mode) or meta.st_uid != 0:
        raise Refusal("DIAGNOSTIC_SOCKET_REJECTED")
    with socket.socket(getattr(socket, "AF_UNIX"), socket.SOCK_STREAM) as connection:
        connection.connect(path)
        creds = connection.getsockopt(socket.SOL_SOCKET, getattr(socket, "SO_PEERCRED"), 12)
    with open(CONFIG, "rb") as stream:
        config = stream.read(4096)
    return {
        "socket_identity": [
            meta.st_dev,
            meta.st_ino,
            meta.st_uid,
            meta.st_gid,
            stat.S_IMODE(meta.st_mode),
        ],
        "peer": list(struct.unpack("3i", creds)),
        "client_config_sha256": hashlib.sha256(config).hexdigest(),
    }


def python_probe(case: str, configuration: dict[str, Any]) -> None:
    import os
    import sys

    pid = os.getpid()
    if case != "python-startup":
        # -I -S supplies only CPython's stdlib. The derived private root contains
        # the byte-pinned dependencies and AppArmor refuses Observer sources.
        sys.path.append(configuration["root"] + "/packages")
    if case == "python-imports":
        exercise_imports()
    if case == "python-libvirt":
        view = view_socket(configuration["socket"])

        def checkpoint(_: dict[str, str]) -> None:
            emit("ready", case, pid=pid, view=view)
            control(b"ACK\n")

        reads = exercise_libvirt(
            import_module("libvirt"),
            configuration["domain_uuid"],
            configuration["network_name"],
            checkpoint,
        )
        emit("result", case, pid=pid, read_results=reads)
    else:
        emit("ready", case, pid=pid)
        control(b"ACK\n")
        emit("result", case, pid=pid)


def qemu_handles(path: str) -> tuple[int, int]:
    import fcntl
    import os
    import stat
    import struct

    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW") | os.O_NONBLOCK
    binary = os.open(QEMU, flags)
    try:
        image = os.open(path, flags)
        try:
            for fd in (binary, image):
                meta = os.fstat(fd)
                if (
                    not stat.S_ISREG(meta.st_mode)
                    or meta.st_uid != 0
                    or meta.st_nlink != 1
                    or meta.st_mode & 0o022
                ):
                    raise Refusal("DIAGNOSTIC_QEMU_FILE")
            # The approved source image was copied to an ext4 inode and made
            # immutable by the privileged supervisor before it sealed the root.
            inode_flags = getattr(fcntl, "ioctl")(image, 0x80086601, bytes(8))
            if not struct.unpack("=Q", inode_flags)[0] & 0x10:
                raise Refusal("DIAGNOSTIC_QEMU_NOT_IMMUTABLE")
        except BaseException:
            os.close(image)
            raise
    except BaseException:
        os.close(binary)
        raise
    return binary, image


def qemu_probe(fmt: str, configuration: dict[str, Any]) -> None:
    import json
    import os

    case = "qemu-" + fmt
    fixture = configuration["images"][fmt]
    binary, image = qemu_handles(fixture["path"])
    argv = [
        f"/proc/self/fd/{binary}",
        "info",
        "--output=json",
        "-f",
        fmt,
        f"/proc/self/fd/{image}",
    ]
    fork = getattr(os, "fork")
    input_channel, send = os.pipe()
    receive, output_channel = os.pipe()
    try:
        pid = fork()
        if pid == 0:
            try:
                os.close(send)
                os.close(receive)
                if os.read(input_channel, 1) != b"1":
                    os._exit(125)
                os.close(input_channel)
                os.dup2(output_channel, 1)
                os.close(output_channel)
                os.set_inheritable(binary, True)
                os.set_inheritable(image, True)
                os.execve(argv[0], argv, TRACE_ENV)
            finally:
                os._exit(125)
        os.close(input_channel)
        input_channel = -1
        os.close(output_channel)
        output_channel = -1
        emit("begin", case, pid=pid, argv=argv)
        control(b"RUN\n")
        if os.write(send, b"1") != 1:
            raise Refusal("DIAGNOSTIC_QEMU_CONTROL")
        with os.fdopen(os.dup(receive), "rb") as stream:
            raw = stream.read(65537)
        if not 0 < len(raw) <= 65536:
            raise Refusal("DIAGNOSTIC_QEMU_RESULT")
        child, status = os.waitpid(pid, 0)
        if child != pid or status != 0:
            raise Refusal("DIAGNOSTIC_QEMU_RESULT")
        doc = json.loads(raw)
        if type(doc) is not dict or doc.get("format") != fmt or doc.get("backing-filename"):
            raise Refusal("DIAGNOSTIC_QEMU_OUTPUT")
        emit(
            "result",
            case,
            pid=pid,
            outcome={
                "argv": argv,
                "fixture_sha256": fixture["sha256"],
                "exit": 0,
                "json_format": fmt,
            },
        )
        control(b"ACK\n")
    finally:
        for fd in (binary, image, input_channel, send, receive, output_channel):
            if fd >= 0:
                os.close(fd)


def carrier(configuration: dict[str, Any]) -> None:
    import os
    import sys

    fork = getattr(os, "fork")
    for case in PYTHON_CASES:
        pid = fork()
        if pid == 0:
            try:
                os.execve(
                    sys.executable, [sys.executable, "-I", "-S", "-B", PAYLOAD, case], TRACE_ENV
                )
            finally:
                os._exit(125)
        child, status = os.waitpid(pid, 0)
        if child != pid or status != 0:
            raise Refusal("DIAGNOSTIC_PYTHON_CASE")
        # The child's dynamic loader may still write fini events after its
        # result line. The case barrier follows waitpid, hence closes its trace.
        emit("end", case, pid=pid)
        control(b"ACK\n")
    for fmt in IMAGE_FORMATS:
        qemu_probe(fmt, configuration)


def main() -> int:
    import os
    import sys

    if (
        len(sys.argv) != 2
        or sys.argv[0] != PAYLOAD
        or not sys.flags.isolated
        or not sys.flags.no_site
        or not sys.dont_write_bytecode
        or getattr(os, "geteuid", lambda: 0)() == 0
    ):
        raise Refusal("DIAGNOSTIC_INVOCATION")
    configuration = inputs()
    if sys.argv[1] == "all" and os.environ == COMMAND_ENV:
        carrier(configuration)
    elif sys.argv[1] in PYTHON_CASES and os.environ == TRACE_ENV:
        python_probe(sys.argv[1], configuration)
    else:
        raise Refusal("DIAGNOSTIC_CASE_REFUSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
