"""Read-only local libvirt observations for the nonproduction V5d1 profile.

The deliberately small supported profile is a persistent, stopped KVM domain
with local file disks and no guest/host communication devices. XML conformance
is a configuration observation, never proof of effective host isolation.
"""

from __future__ import annotations

import hashlib
import importlib
import re
from collections.abc import Iterator, Mapping
from dataclasses import dataclass, replace
from typing import Any, Protocol, cast, overload
from uuid import UUID

from defusedxml import ElementTree
from defusedxml.common import DefusedXmlException

URI = "qemu:///system"
MAX_XML_BYTES = 1024 * 1024
MAX_XML_NODES = 4096
SHUTOFF = 5
MISSING_ENFORCEMENT = (
    "QEMU_EFFECTIVE_STATE_UNVERIFIED",
    "HOST_SANDBOX_ENFORCEMENT_UNVERIFIED",
    "PROCESS_LIMIT_ENFORCEMENT_UNVERIFIED",
    "CPU_LIMIT_ENFORCEMENT_UNVERIFIED",
    "MEMORY_LIMIT_ENFORCEMENT_UNVERIFIED",
    "MEASUREMENT_CHANNEL_UNVERIFIED",
    "WHOLE_VM_STOP_AUTHORITY_UNVERIFIED",
)


class _Element(Protocol):
    """Read interface consumed from the hardened parser, independent of XML APIs."""

    @property
    def tag(self) -> str: ...

    @property
    def text(self) -> str | None: ...

    @property
    def tail(self) -> str | None: ...

    @property
    def attrib(self) -> Mapping[str, str]: ...

    def __iter__(self) -> Iterator[_Element]: ...

    def find(self, path: str) -> _Element | None: ...

    def findall(self, path: str) -> list[_Element]: ...

    @overload
    def findtext(self, path: str) -> str | None: ...

    @overload
    def findtext(self, path: str, default: str) -> str: ...

    def iter(self) -> Iterator[_Element]: ...


class DomainReadErrorV5d1(ValueError):
    """Stable diagnostics without leaking host exception details."""


@dataclass(frozen=True, slots=True)
class DomainRequirementsV5d1:
    domain_uuid: str
    disk_paths: tuple[str, ...]
    memory_bytes: int = 512 * 1024 * 1024
    vcpu_count: int = 1
    cpu_period_us: int = 100000
    cpu_quota_us: int = 100000
    disk_formats: tuple[str, ...] = ()
    # Each full ordered chain includes its top disk path as item zero.
    disk_backing_chains: tuple[tuple[str, ...], ...] = ()
    disk_backing_formats: tuple[tuple[str, ...], ...] = ()


@dataclass(frozen=True, slots=True)
class NetworkObservationV5d1:
    name: str
    uuid: str
    active: bool
    xml: str


@dataclass(frozen=True, slots=True)
class DomainSnapshotV5d1:
    domain_uuid: str
    domain_id: int
    state: int
    state_reason: int
    active: bool
    persistent: bool
    xml: str
    inactive_xml: str
    max_memory_kib: int
    memory_kib: int
    vcpu_count: int
    scheduler_parameters: tuple[tuple[str, int | str | bool], ...]
    networks: tuple[NetworkObservationV5d1, ...] = ()
    snapshot_names: tuple[str, ...] = ()
    managed_save: bool = False


@dataclass(frozen=True, slots=True)
class DomainInspectionV5d1:
    config_matches: bool
    reason_codes: tuple[str, ...]
    missing_evidence: tuple[str, ...]
    disk_paths: tuple[str, ...]
    backing_paths: tuple[str, ...]
    channel_paths: tuple[str, ...]
    memory_bytes: int | None
    vcpu_count: int | None
    xml_sha256: str
    inactive_xml_sha256: str


def _uuid(value: str) -> str:
    try:
        if type(value) is not str or str(UUID(value)) != value:
            raise ValueError
    except (ValueError, AttributeError):
        raise DomainReadErrorV5d1("DOMAIN_UUID_REJECTED") from None
    return value


def _path(value: str) -> bool:
    return (
        type(value) is str
        and value.startswith("/")
        and len(value.encode("utf-8")) <= 4096
        and all(part not in {"", ".", ".."} for part in value.split("/")[1:])
        and not any(ord(char) < 32 or ord(char) == 127 for char in value)
    )


def _xml(raw: str) -> _Element:
    if type(raw) is not str or len(raw.encode("utf-8")) > MAX_XML_BYTES or "<!" in raw:
        raise DomainReadErrorV5d1("DOMAIN_XML_REJECTED")
    try:
        root = cast(
            _Element,
            ElementTree.fromstring(
                raw, forbid_dtd=True, forbid_entities=True, forbid_external=True
            ),
        )
    except (ElementTree.ParseError, DefusedXmlException, ValueError):
        raise DomainReadErrorV5d1("DOMAIN_XML_REJECTED") from None
    stack = [(root, 0)]
    count = 0
    while stack:
        element, depth = stack.pop()
        count += 1
        if count > MAX_XML_NODES or depth > 32:
            raise DomainReadErrorV5d1("DOMAIN_XML_REJECTED")
        stack.extend((child, depth + 1) for child in element)
    if root.tag != "domain":
        raise DomainReadErrorV5d1("DOMAIN_XML_REJECTED")
    return root


class LibvirtDomainReaderV5d1:
    """Only opens the local system connection read-only; never opens a console.

    libvirt-python is an operator-host dependency and is imported lazily. No
    mutation API or guest-agent/QMP command is called by this component.
    """

    def __init__(self) -> None:
        try:
            module = importlib.import_module("libvirt")
            self._connection: Any = module.openReadOnly(URI)
            if self._connection is None or self._connection.getURI() != URI:
                if self._connection is not None:
                    self._connection.close()
                raise DomainReadErrorV5d1("LIBVIRT_CONNECTION_REJECTED")
        except DomainReadErrorV5d1:
            raise
        except Exception:
            raise DomainReadErrorV5d1("LIBVIRT_UNAVAILABLE") from None
        self._closed = False

    def read(self, domain_uuid: str) -> DomainSnapshotV5d1:
        _uuid(domain_uuid)
        if self._closed:
            raise DomainReadErrorV5d1("LIBVIRT_CONNECTION_CLOSED")
        try:
            domain = self._connection.lookupByUUIDString(domain_uuid)
            observed_uuid = str(domain.UUIDString())
            if observed_uuid != domain_uuid:
                raise DomainReadErrorV5d1("DOMAIN_UUID_MISMATCH")
            xml = domain.XMLDesc(0)
            _xml(xml)
            # VIR_DOMAIN_XML_INACTIVE=2 is stable libvirt public API. Neither
            # SECURE XML nor guest-agent interfaces are requested.
            inactive_xml = domain.XMLDesc(2)
            _xml(inactive_xml)
            state, reason = domain.state(0)
            info = domain.info()
            if len(info) != 5 or int(info[0]) != int(state):
                raise DomainReadErrorV5d1("DOMAIN_CHANGED_DURING_READ")
            scheduler = domain.schedulerParameters()
            if type(scheduler) is not dict or any(
                type(key) is not str or type(value) not in (int, str, bool)
                for key, value in scheduler.items()
            ):
                raise DomainReadErrorV5d1("DOMAIN_OBSERVATION_INCOMPLETE")
            networks: list[NetworkObservationV5d1] = []
            names = sorted(
                {
                    item.attrib["network"]
                    for item in _xml(xml).findall("./devices/interface/source")
                    if "network" in item.attrib
                }
            )
            for name in names:
                network = self._connection.networkLookupByName(name)
                network_xml = network.XMLDesc(0)
                if type(network_xml) is not str or len(network_xml.encode()) > MAX_XML_BYTES:
                    raise DomainReadErrorV5d1("NETWORK_OBSERVATION_INCOMPLETE")
                networks.append(
                    NetworkObservationV5d1(
                        name, str(network.UUIDString()), bool(network.isActive()), network_xml
                    )
                )
            snapshot = DomainSnapshotV5d1(
                observed_uuid,
                int(domain.ID()),
                int(state),
                int(reason),
                bool(domain.isActive()),
                bool(domain.isPersistent()),
                xml,
                inactive_xml,
                int(info[1]),
                int(info[2]),
                int(info[3]),
                tuple(sorted(scheduler.items())),
                tuple(networks),
                tuple(sorted(domain.snapshotListNames(0))),
                bool(domain.hasManagedSaveImage(0)),
            )
            # A second state read catches changes even within a single sample.
            if domain.UUIDString() != observed_uuid or tuple(domain.state(0)) != (state, reason):
                raise DomainReadErrorV5d1("DOMAIN_CHANGED_DURING_READ")
            return snapshot
        except DomainReadErrorV5d1:
            raise
        except Exception:
            raise DomainReadErrorV5d1("DOMAIN_OBSERVATION_INCOMPLETE") from None

    def close(self) -> None:
        if not self._closed:
            self._closed = True
            try:
                self._connection.close()
            except Exception:
                raise DomainReadErrorV5d1("LIBVIRT_CLOSE_FAILED") from None


def _positive(value: Any) -> bool:
    return type(value) is int and value > 0


def _requirements(requirements: DomainRequirementsV5d1) -> None:
    _uuid(requirements.domain_uuid)
    if (
        type(requirements.disk_paths) is not tuple
        or not requirements.disk_paths
        or len(requirements.disk_paths) > 16
        or len(set(requirements.disk_paths)) != len(requirements.disk_paths)
        or not all(_path(path) for path in requirements.disk_paths)
        or not all(
            _positive(value)
            for value in (
                requirements.memory_bytes,
                requirements.vcpu_count,
                requirements.cpu_period_us,
                requirements.cpu_quota_us,
            )
        )
        or requirements.memory_bytes % 1024
        or requirements.cpu_quota_us > requirements.cpu_period_us * requirements.vcpu_count
    ):
        raise DomainReadErrorV5d1("DOMAIN_REQUIREMENTS_REJECTED")
    if (
        type(requirements.disk_formats) is not tuple
        or (
            requirements.disk_formats
            and (
                len(requirements.disk_formats) != len(requirements.disk_paths)
                or any(
                    type(value) is not str or value not in {"raw", "qcow2"}
                    for value in requirements.disk_formats
                )
            )
        )
        or type(requirements.disk_backing_chains) is not tuple
        or (
            requirements.disk_backing_chains
            and (
                len(requirements.disk_backing_chains) != len(requirements.disk_paths)
                or any(
                    type(chain) is not tuple
                    or not chain
                    or len(chain) > 32
                    or chain[0] != top
                    or len(set(chain)) != len(chain)
                    or not all(_path(path) for path in chain)
                    for top, chain in zip(
                        requirements.disk_paths, requirements.disk_backing_chains, strict=True
                    )
                )
            )
        )
    ):
        raise DomainReadErrorV5d1("DOMAIN_REQUIREMENTS_REJECTED")
    if type(requirements.disk_backing_formats) is not tuple or (
        requirements.disk_backing_formats
        and (
            not requirements.disk_backing_chains
            or len(requirements.disk_backing_formats) != len(requirements.disk_paths)
            or any(
                type(formats) is not tuple
                or len(formats) != len(chain)
                or any(type(value) is not str or value not in {"raw", "qcow2"} for value in formats)
                for formats, chain in zip(
                    requirements.disk_backing_formats, requirements.disk_backing_chains, strict=True
                )
            )
        )
    ):
        raise DomainReadErrorV5d1("DOMAIN_REQUIREMENTS_REJECTED")


def _quantity(element: _Element | None) -> int | None:
    if (
        element is None
        or element.text is None
        or len(element.text) > 20
        or not re.fullmatch(r"[0-9]+", element.text)
    ):
        return None
    multiplier = {"bytes": 1, "KiB": 1024, "MiB": 1024 * 1024, "GiB": 1024**3}.get(
        element.attrib.get("unit", "KiB")
    )
    return None if multiplier is None else int(element.text) * multiplier


# All other paths are rejected, including namespaced qemu commandline/capability
# extensions, firmware/NVRAM, launch security devices and guest-access channels.
_ALLOWED_PATHS = frozenset(
    {
        "domain",
        "domain/name",
        "domain/uuid",
        "domain/title",
        "domain/description",
        "domain/memory",
        "domain/currentMemory",
        "domain/vcpu",
        "domain/cputune",
        "domain/cputune/period",
        "domain/cputune/quota",
        "domain/os",
        "domain/os/type",
        "domain/os/boot",
        "domain/features",
        "domain/features/acpi",
        "domain/features/apic",
        "domain/cpu",
        "domain/cpu/topology",
        "domain/clock",
        "domain/on_poweroff",
        "domain/on_reboot",
        "domain/on_crash",
        "domain/devices",
        "domain/devices/emulator",
        "domain/devices/disk",
        "domain/devices/disk/driver",
        "domain/devices/disk/source",
        "domain/devices/disk/target",
        "domain/devices/disk/readonly",
        "domain/devices/disk/address",
        "domain/devices/disk/alias",
        "domain/devices/disk/backingStore",
        "domain/devices/disk/backingStore/format",
        "domain/devices/disk/backingStore/source",
        "domain/devices/disk/backingStore/backingStore",
        "domain/devices/controller",
        "domain/devices/controller/address",
        "domain/devices/controller/alias",
        "domain/devices/memballoon",
        "domain/seclabel",
        "domain/seclabel/label",
        "domain/seclabel/imagelabel",
        "domain/resource",
        "domain/resource/partition",
    }
)


# An allowlist of element names alone is insufficient: e.g. source startupPolicy,
# driver options and backing/socket attributes can change device semantics.
_ATTRIBUTES: dict[str, frozenset[str]] = {
    "domain": frozenset({"type", "id"}),
    "memory": frozenset({"unit"}),
    "currentMemory": frozenset({"unit"}),
    "vcpu": frozenset({"placement", "current"}),
    "type": frozenset({"arch", "machine"}),
    "boot": frozenset({"dev"}),
    "cpu": frozenset({"mode", "match", "check", "migratable"}),
    "topology": frozenset({"sockets", "dies", "clusters", "cores", "threads"}),
    "clock": frozenset({"offset"}),
    "disk": frozenset({"type", "device", "snapshot"}),
    "driver": frozenset({"name", "type", "cache", "io", "discard", "detect_zeroes"}),
    "source": frozenset({"file"}),
    "target": frozenset({"dev", "bus"}),
    "address": frozenset(
        {
            "type",
            "domain",
            "bus",
            "slot",
            "function",
            "multifunction",
            "controller",
            "target",
            "unit",
        }
    ),
    "alias": frozenset({"name"}),
    "backingStore": frozenset({"type", "index"}),
    "format": frozenset({"type"}),
    "controller": frozenset({"type", "index", "model"}),
    "memballoon": frozenset({"model"}),
    "seclabel": frozenset({"type", "model", "relabel"}),
}


# Values, cardinality and text are checked separately from the name allowlists.
# This is one deliberately narrow profile, not a libvirt schema validator. In
# particular, host label/partition assignment and explicit device addresses need
# independent host evidence that a static XML observation cannot supply.
_LIFECYCLE = ("on_poweroff", "on_reboot", "on_crash")
_DRIVER_PROFILE = {"cache": "none", "io": "native", "discard": "ignore", "detect_zeroes": "off"}
_TEXT_PATHS = frozenset(
    {
        "domain/name",
        "domain/uuid",
        "domain/title",
        "domain/description",
        "domain/memory",
        "domain/currentMemory",
        "domain/vcpu",
        "domain/cputune/period",
        "domain/cputune/quota",
        "domain/os/type",
        "domain/devices/emulator",
        *("domain/" + tag for tag in _LIFECYCLE),
    }
)
_EMPTY_PATHS = frozenset(
    {
        "domain/os/boot",
        "domain/features/acpi",
        "domain/features/apic",
        "domain/cpu/topology",
        "domain/clock",
        "domain/devices/disk/driver",
        "domain/devices/disk/source",
        "domain/devices/disk/target",
        "domain/devices/disk/readonly",
        "domain/devices/disk/alias",
        "domain/devices/disk/backingStore/format",
        "domain/devices/disk/backingStore/source",
        "domain/devices/controller/alias",
        "domain/devices/memballoon",
    }
)


def _reviewed_path(path: str) -> str:
    while "/backingStore/backingStore" in path:
        path = path.replace("/backingStore/backingStore", "/backingStore")
    return path


def _semantic_profile(root: _Element, requirements: DomainRequirementsV5d1) -> list[str]:
    reasons: list[str] = []
    for tag in _LIFECYCLE:
        elements = root.findall(tag)
        if not elements:
            reasons.append("DOMAIN_LIFECYCLE_POLICY_MISSING")
        elif len(elements) != 1 or elements[0].text != "destroy":
            reasons.append("DOMAIN_LIFECYCLE_POLICY_REJECTED")
    for path, code in (
        ("./os/boot", "BOOT_POLICY_UNREVIEWED"),
        ("cpu", "CPU_PROFILE_UNREVIEWED"),
        ("./cpu/topology", "CPU_TOPOLOGY_UNREVIEWED"),
        ("clock", "CLOCK_POLICY_UNREVIEWED"),
        ("./devices/controller", "CONTROLLER_PROFILE_UNVERIFIED"),
        ("./devices/memballoon", "MEMORY_BALLOON_REJECTED"),
    ):
        if len(root.findall(path)) != 1:
            reasons.append(code)
    stack = [(root, "domain")]
    aliases: set[str] = set()
    while stack:
        element, path = stack.pop()
        path = _reviewed_path(path)
        children = list(element)
        tags = [child.tag for child in children]
        if any(tags.count(tag) > 1 and (path, tag) != ("domain/devices", "disk") for tag in tags):
            reasons.append("DUPLICATE_DOMAIN_CONFIGURATION")
        if (
            (path not in _TEXT_PATHS and element.text is not None and element.text.strip())
            or (element.tail is not None and element.tail.strip())
            or (path in _EMPTY_PATHS and children)
        ):
            reasons.append("DOMAIN_VALUE_UNREVIEWED")
        if path.startswith("domain/seclabel"):
            reasons.append("HOST_CONFINEMENT_PROFILE_UNVERIFIED")
        if path.startswith("domain/resource"):
            reasons.append("RESOURCE_PARTITION_UNVERIFIED")
        if path.endswith("/address"):
            reasons.append("DEVICE_ADDRESS_PROFILE_UNVERIFIED")
        if path == "domain" and "id" in element.attrib and element.attrib["id"] != "-1":
            reasons.append("DOMAIN_VALUE_UNREVIEWED")
        if path == "domain/name" and (
            element.text is None
            or re.fullmatch(r"[A-Za-z0-9_][A-Za-z0-9_.-]{0,62}", element.text) is None
        ):
            reasons.append("DOMAIN_VALUE_UNREVIEWED")
        if path in {"domain/title", "domain/description"} and element.text is not None:
            if any(ord(char) < 32 or ord(char) == 127 for char in element.text):
                reasons.append("DOMAIN_VALUE_UNREVIEWED")
        if path == "domain/os/type" and element.attrib.get("machine") != "pc-q35-8.2":
            reasons.append("GUEST_MACHINE_UNREVIEWED")
        if path == "domain/os/boot" and element.attrib != {"dev": "hd"}:
            reasons.append("BOOT_POLICY_UNREVIEWED")
        if path == "domain/cpu":
            if element.attrib.get("mode") != "host-model":
                reasons.append("CPU_MODE_UNREVIEWED")
            # libvirt forbids match for host-model; migratable is defined for
            # host-passthrough/maximum, which this profile does not support.
            if (
                element.attrib.get("check") != "full"
                or "match" in element.attrib
                or "migratable" in element.attrib
            ):
                reasons.append("CPU_PROFILE_UNREVIEWED")
        if path == "domain/cpu/topology" and (
            element.attrib.get("sockets") != "1"
            or element.attrib.get("cores") != str(requirements.vcpu_count)
            or element.attrib.get("threads") != "1"
            or element.attrib.get("dies") != "1"
            or element.attrib.get("clusters") != "1"
        ):
            reasons.append("CPU_TOPOLOGY_UNREVIEWED")
        if path == "domain/clock" and element.attrib != {"offset": "utc"}:
            reasons.append("CLOCK_POLICY_UNREVIEWED")
        if path == "domain/devices/disk" and element.attrib.get("snapshot") != "no":
            reasons.append("DISK_LIFECYCLE_UNVERIFIED")
        if path == "domain/devices/disk/driver" and any(
            element.attrib.get(key) != expected for key, expected in _DRIVER_PROFILE.items()
        ):
            reasons.append("DISK_DRIVER_REJECTED")
        if path.endswith("/alias"):
            alias = element.attrib.get("name", "")
            if not re.fullmatch(r"[A-Za-z0-9_][A-Za-z0-9_.-]{0,62}", alias) or alias in aliases:
                reasons.append("DEVICE_ALIAS_UNREVIEWED")
            aliases.add(alias)
        if path == "domain/devices/controller" and (
            element.attrib.get("type") != "usb"
            or element.attrib.get("model") != "none"
            or element.attrib.get("index", "0") != "0"
        ):
            reasons.append("CONTROLLER_PROFILE_UNVERIFIED")
        stack.extend((child, path + "/" + child.tag) for child in children)
    for disk in root.findall("./devices/disk"):
        source = disk.find("source")
        path = "" if source is None else source.attrib.get("file", "")
        index = requirements.disk_paths.index(path) if path in requirements.disk_paths else None
        target = disk.find("target")
        if (
            index is None
            or target is None
            or target.attrib != {"dev": "vd" + chr(ord("a") + index), "bus": "virtio"}
        ):
            reasons.append("DISK_TARGET_UNREVIEWED")
        if len(disk.findall("driver")) != 1 or len(disk.findall("source")) != 1:
            reasons.append("DISK_SOURCE_REJECTED")
        if len(disk.findall("readonly")) != 1:
            reasons.append("DISK_LIFECYCLE_UNVERIFIED")
        for depth, store in enumerate(disk.findall(".//backingStore"), start=1):
            source = store.find("source")
            format_element = store.find("format")
            if source is None:
                if store.attrib or list(store):
                    reasons.append("BACKING_CONFIGURATION_REJECTED")
            elif (
                store.attrib.get("index", str(depth)) != str(depth)
                or len(store.findall("format")) != 1
                or format_element is None
                or format_element.attrib.get("type") not in {"raw", "qcow2"}
            ):
                reasons.append("BACKING_CONFIGURATION_REJECTED")
    return reasons


def _semantic_tree(element: _Element) -> tuple[Any, ...]:
    """Compare every reviewed setting, ignoring only XML indentation whitespace."""
    return (
        element.tag,
        tuple(sorted(element.attrib.items())),
        element.text if element.text is not None and element.text.strip() else "",
        tuple(_semantic_tree(child) for child in element),
    )


def _configuration(
    root: _Element, requirements: DomainRequirementsV5d1
) -> tuple[list[str], list[str], list[str], list[str], int | None, int | None]:
    reasons = _semantic_profile(root, requirements)
    disks: list[str] = []
    backing: list[str] = []
    channels: list[str] = []
    if root.attrib.get("type") != "kvm" or set(root.attrib) - {"type", "id"}:
        reasons.append("DOMAIN_TYPE_REJECTED")
    if root.findtext("uuid") != requirements.domain_uuid:
        reasons.append("DOMAIN_UUID_MISMATCH")
    stack = [(root, "domain")]
    while stack:
        element, path = stack.pop()
        reviewed_path = _reviewed_path(path)
        if reviewed_path not in _ALLOWED_PATHS:
            reasons.append("UNREVIEWED_DOMAIN_CONFIGURATION")
        permitted = _ATTRIBUTES.get(element.tag, frozenset())
        if set(element.attrib) - permitted:
            reasons.append("UNREVIEWED_DOMAIN_ATTRIBUTES")
        stack.extend((child, path + "/" + child.tag) for child in element)
    for item in root.findall("./devices/interface"):
        reasons.append("NETWORK_DEVICE_PRESENT")
        if item.attrib.get("type") in {"hostdev", "direct"}:
            reasons.append("NETWORK_PASSTHROUGH_PRESENT")
    for tag, code in (
        ("hostdev", "HOST_DEVICE_PRESENT"),
        ("filesystem", "SHARED_DIRECTORY_PRESENT"),
        ("channel", "COMMUNICATION_CHANNEL_PRESENT"),
        ("console", "CONSOLE_DEVICE_PRESENT"),
        ("serial", "CONSOLE_DEVICE_PRESENT"),
        ("parallel", "HOST_DEVICE_PRESENT"),
        ("graphics", "CONSOLE_DEVICE_PRESENT"),
        ("redirdev", "REMOVABLE_DEVICE_PRESENT"),
        ("smartcard", "HOST_DEVICE_PRESENT"),
        ("tpm", "HOST_DEVICE_PRESENT"),
        ("vsock", "COMMUNICATION_CHANNEL_PRESENT"),
        ("shmem", "SHARED_MEMORY_PRESENT"),
        ("rng", "HOST_DEVICE_PRESENT"),
        ("crypto", "HOST_DEVICE_PRESENT"),
        ("audio", "HOST_DEVICE_PRESENT"),
    ):
        if root.find("./devices/" + tag) is not None:
            reasons.append(code)
    for source in root.findall("./devices/channel/source") + root.findall(
        "./devices/console/source"
    ):
        channels.extend(
            source.attrib[key] for key in ("path", "master", "slave") if key in source.attrib
        )
    for element in root.iter():
        if any(key in element.attrib for key in ("socket", "sock", "fd", "fdgroup")):
            reasons.append("HOST_SOCKET_ACCESS_PRESENT")
        if element.tag.startswith("{"):
            reasons.append("QEMU_EXTENSION_REJECTED")
    for disk in root.findall("./devices/disk"):
        if disk.attrib.get("type") != "file" or disk.attrib.get("device") != "disk":
            reasons.append("DISK_DEVICE_REJECTED")
        if disk.attrib.get("device") in {"cdrom", "floppy"}:
            reasons.append("REMOVABLE_DEVICE_PRESENT")
        disk_source = disk.find("source")
        path = "" if disk_source is None else disk_source.attrib.get("file", "")
        if (
            disk_source is None
            or set(disk_source.attrib) != {"file"}
            or list(disk_source)
            or not _path(path)
        ):
            reasons.append("DISK_SOURCE_REJECTED")
        else:
            disks.append(path)
        expected_index = (
            requirements.disk_paths.index(path) if path in requirements.disk_paths else None
        )
        driver = disk.find("driver")
        if (
            driver is None
            or driver.attrib.get("name") != "qemu"
            or driver.attrib.get("type") not in {"raw", "qcow2"}
        ):
            reasons.append("DISK_FORMAT_REJECTED")
        if (
            expected_index is not None
            and requirements.disk_formats
            and (
                driver is None
                or driver.attrib.get("type") != requirements.disk_formats[expected_index]
            )
        ):
            reasons.append("DISK_FORMAT_EXPECTATION_MISMATCH")
        if driver is not None and set(driver.attrib) - {
            "name",
            "type",
            "cache",
            "io",
            "discard",
            "detect_zeroes",
        }:
            reasons.append("DISK_DRIVER_REJECTED")
        if disk.find("shareable") is not None or disk.find("transient") is not None:
            reasons.append("DISK_LIFECYCLE_UNVERIFIED")
        declared_backing: list[str] = []
        declared_formats: list[str | None] = []
        for store in disk.findall(".//backingStore"):
            if (
                len(store.findall("source")) > 1
                or len(store.findall("format")) > 1
                or len(store.findall("backingStore")) > 1
                or store.attrib.get("type", "file") != "file"
            ):
                reasons.append("BACKING_CONFIGURATION_REJECTED")
            if store.find("source") is not None:
                format_element = store.find("format")
                declared_formats.append(
                    None if format_element is None else format_element.attrib.get("type")
                )
        if len(disk.findall("backingStore")) > 1:
            reasons.append("BACKING_CONFIGURATION_REJECTED")
        for item in disk.findall(".//backingStore/source"):
            value = item.attrib.get("file", "")
            if set(item.attrib) != {"file"} or not _path(value):
                reasons.append("BACKING_SOURCE_REJECTED")
            else:
                backing.append(value)
                declared_backing.append(value)
        if (
            expected_index is not None
            and requirements.disk_backing_chains
            and disk.find("backingStore") is not None
            and tuple(declared_backing) != requirements.disk_backing_chains[expected_index][1:]
        ):
            reasons.append("BACKING_CHAIN_EXPECTATION_MISMATCH")
        if (
            expected_index is not None
            and requirements.disk_backing_formats
            and disk.find("backingStore") is not None
            and tuple(declared_formats) != requirements.disk_backing_formats[expected_index][1:]
        ):
            reasons.append("BACKING_FORMAT_EXPECTATION_MISMATCH")
    if sorted(disks) != sorted(requirements.disk_paths):
        reasons.append("DISK_INVENTORY_MISMATCH")
    memory = _quantity(root.find("memory"))
    if memory != requirements.memory_bytes or _quantity(root.find("currentMemory")) != memory:
        reasons.append("MEMORY_CONFIGURATION_MISMATCH")
    vcpu_raw = root.findtext("vcpu", "")
    vcpus = int(vcpu_raw) if len(vcpu_raw) <= 4 and re.fullmatch(r"[0-9]+", vcpu_raw) else None
    if vcpus != requirements.vcpu_count:
        reasons.append("CPU_CONFIGURATION_MISMATCH")
    vcpu_element = root.find("vcpu")
    if vcpu_element is not None and vcpu_element.attrib.get("current", str(vcpus)) != str(vcpus):
        reasons.append("CPU_HOTPLUG_REJECTED")
    for tag, expected in (
        ("period", requirements.cpu_period_us),
        ("quota", requirements.cpu_quota_us),
    ):
        if root.findtext("./cputune/" + tag) != str(expected):
            reasons.append("CPU_LIMIT_CONFIGURATION_MISMATCH")
    if vcpu_element is not None and vcpu_element.attrib.get("placement", "static") != "static":
        reasons.append("CPU_PLACEMENT_UNREVIEWED")
    emulator = root.findtext("./devices/emulator")
    if emulator != "/usr/bin/qemu-system-x86_64":
        reasons.append("EMULATOR_PATH_UNREVIEWED")
    for label in root.findall("seclabel"):
        if label.attrib.get("type") == "none" or label.attrib.get("model") == "none":
            reasons.append("HOST_CONFINEMENT_DISABLED")
    cpu = root.find("cpu")
    if cpu is not None and cpu.attrib.get("mode") not in {"host-model", "custom"}:
        reasons.append("CPU_MODE_UNREVIEWED")
    os_type = root.find("./os/type")
    if os_type is None or os_type.text != "hvm" or os_type.attrib.get("arch") != "x86_64":
        reasons.append("GUEST_MACHINE_UNREVIEWED")
    for controller in root.findall("./devices/controller"):
        if controller.attrib.get("type") not in {"pci", "scsi", "virtio-serial", "usb"}:
            reasons.append("CONTROLLER_DEVICE_UNREVIEWED")
        if controller.attrib.get("type") == "usb" and controller.attrib.get("model") != "none":
            reasons.append("REMOVABLE_CONTROLLER_PRESENT")
    balloon = root.find("./devices/memballoon")
    if balloon is not None and balloon.attrib.get("model") != "none":
        reasons.append("MEMORY_BALLOON_REJECTED")
    if any(
        len(root.findall(tag)) > 1
        for tag in ("uuid", "memory", "currentMemory", "vcpu", "os", "devices", "cputune")
    ):
        reasons.append("DUPLICATE_DOMAIN_CONFIGURATION")
    return reasons, disks, backing, channels, memory, vcpus


def inspect_domain_v5d1(
    snapshot: DomainSnapshotV5d1, requirements: DomainRequirementsV5d1
) -> DomainInspectionV5d1:
    _requirements(requirements)
    root = _xml(snapshot.xml)
    inactive = _xml(snapshot.inactive_xml)
    live = _configuration(root, requirements)
    saved = _configuration(inactive, requirements)
    reasons = live[0] + saved[0]
    if snapshot.domain_uuid != requirements.domain_uuid:
        reasons.append("DOMAIN_UUID_MISMATCH")
    if snapshot.active or snapshot.state != SHUTOFF or snapshot.domain_id != -1:
        reasons.append("DOMAIN_NOT_SHUTOFF")
    if not snapshot.persistent:
        reasons.append("DOMAIN_PERSISTENCE_UNVERIFIED")
    if snapshot.snapshot_names:
        reasons.append("DOMAIN_SNAPSHOTS_PRESENT")
    if snapshot.managed_save:
        reasons.append("DOMAIN_MANAGED_SAVE_PRESENT")
    if snapshot.networks:
        reasons.append("NETWORK_DEVICE_PRESENT")
    if (
        snapshot.max_memory_kib * 1024 != requirements.memory_bytes
        or snapshot.memory_kib * 1024 != requirements.memory_bytes
        or snapshot.vcpu_count != requirements.vcpu_count
    ):
        reasons.append("DOMAIN_EFFECTIVE_RESOURCES_MISMATCH")
    scheduler = dict(snapshot.scheduler_parameters)
    for key, expected in (
        ("vcpu_period", requirements.cpu_period_us),
        ("vcpu_quota", requirements.cpu_quota_us),
    ):
        if key not in scheduler or type(scheduler[key]) is not int:
            reasons.append("CPU_LIMIT_OBSERVATION_INCOMPLETE")
        elif scheduler[key] != expected:
            reasons.append("CPU_LIMIT_OBSERVATION_MISMATCH")
    if live[1:] != saved[1:] or _semantic_tree(root) != _semantic_tree(inactive):
        reasons.append("DOMAIN_LIVE_PERSISTENT_MISMATCH")
    return DomainInspectionV5d1(
        not reasons,
        tuple(sorted(set(reasons))),
        MISSING_ENFORCEMENT,
        tuple(live[1]),
        tuple(live[2]),
        tuple(live[3]),
        live[4],
        live[5],
        hashlib.sha256(snapshot.xml.encode()).hexdigest(),
        hashlib.sha256(snapshot.inactive_xml.encode()).hexdigest(),
    )


def inspect_domain_pair_v5d1(
    first: DomainSnapshotV5d1,
    second: DomainSnapshotV5d1,
    requirements: DomainRequirementsV5d1,
) -> DomainInspectionV5d1:
    """Require stable observations bracketing the external image measurements."""
    initial = inspect_domain_v5d1(first, requirements)
    final = inspect_domain_v5d1(second, requirements)
    reasons = set(initial.reason_codes + final.reason_codes)
    if first != second:
        reasons.add("DOMAIN_CHANGED_DURING_PREFLIGHT")
    return replace(final, config_matches=not reasons, reason_codes=tuple(sorted(reasons)))
