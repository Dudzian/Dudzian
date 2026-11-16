"""SYNTHETIC libvirt fixtures; none establish physical KVM qualification."""

from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace
from typing import Any

import pytest

from deployment import stage9_observer_kvm_domain_v5d1 as domain

pytestmark = [pytest.mark.security, pytest.mark.unit]

UUID = "ee38896c-af63-46b9-b08c-c67fa4292670"
DISK = "/srv/stage9/disposable.qcow2"
XML = f"""<domain type='kvm'>
<name>synthetic-only</name><uuid>{UUID}</uuid>
<memory unit='KiB'>524288</memory><currentMemory unit='KiB'>524288</currentMemory>
<vcpu placement='static'>1</vcpu><cputune><period>100000</period><quota>100000</quota></cputune>
<os><type arch='x86_64' machine='pc-q35-8.2'>hvm</type><boot dev='hd'/></os>
<features><acpi/><apic/></features><cpu mode='host-model' check='full'>
<topology sockets='1' dies='1' clusters='1' cores='1' threads='1'/></cpu><clock offset='utc'/>
<on_poweroff>destroy</on_poweroff><on_reboot>destroy</on_reboot><on_crash>destroy</on_crash>
<devices><emulator>/usr/bin/qemu-system-x86_64</emulator>
<disk type='file' device='disk' snapshot='no'><driver name='qemu' type='qcow2'
cache='none' io='native' discard='ignore' detect_zeroes='off'/><readonly/>
<source file='{DISK}'/><target dev='vda' bus='virtio'/><backingStore/></disk>
<controller type='usb' model='none'/><memballoon model='none'/></devices></domain>"""


def _requirements() -> domain.DomainRequirementsV5d1:
    return domain.DomainRequirementsV5d1(UUID, (DISK,))


def _snapshot(xml: str = XML, **changes: Any) -> domain.DomainSnapshotV5d1:
    value = domain.DomainSnapshotV5d1(
        domain_uuid=UUID,
        domain_id=-1,
        state=5,
        state_reason=1,
        active=False,
        persistent=True,
        xml=xml,
        inactive_xml=xml,
        max_memory_kib=524288,
        memory_kib=524288,
        vcpu_count=1,
        scheduler_parameters=(("vcpu_period", 100000), ("vcpu_quota", 100000)),
    )
    return replace(value, **changes)


def _device(raw: str) -> str:
    return XML.replace("</devices>", raw + "</devices>")


def test_synthetic_minimal_config_is_not_isolation_evidence() -> None:
    inspection = domain.inspect_domain_v5d1(_snapshot(), _requirements())
    assert inspection.config_matches
    assert inspection.reason_codes == ()
    assert inspection.disk_paths == (DISK,)
    assert inspection.memory_bytes == 512 * 1024 * 1024
    assert inspection.vcpu_count == 1
    assert inspection.missing_evidence == domain.MISSING_ENFORCEMENT
    assert len(inspection.xml_sha256) == 64
    assert "PROCESS_LIMIT_ENFORCEMENT_UNVERIFIED" in inspection.missing_evidence
    assert not hasattr(inspection, "execution_authorized")


@pytest.mark.parametrize(
    ("raw", "reason"),
    [
        (
            "<interface type='network'><source network='default'/></interface>",
            "NETWORK_DEVICE_PRESENT",
        ),
        (
            "<interface type='direct'><source dev='eno1'/></interface>",
            "NETWORK_PASSTHROUGH_PRESENT",
        ),
        ("<interface type='hostdev'/>", "NETWORK_PASSTHROUGH_PRESENT"),
        ("<hostdev mode='subsystem' type='pci'/>", "HOST_DEVICE_PRESENT"),
        ("<filesystem><source dir='/home'/></filesystem>", "SHARED_DIRECTORY_PRESENT"),
        (
            "<channel type='unix'><source path='/run/libvirt/agent.sock'/></channel>",
            "COMMUNICATION_CHANNEL_PRESENT",
        ),
        ("<console type='pty'><source path='/dev/pts/1'/></console>", "CONSOLE_DEVICE_PRESENT"),
        (
            "<serial type='unix'><source path='/run/serial.sock'/></serial>",
            "CONSOLE_DEVICE_PRESENT",
        ),
        ("<parallel type='dev'/>", "HOST_DEVICE_PRESENT"),
        ("<graphics type='vnc'/>", "CONSOLE_DEVICE_PRESENT"),
        ("<redirdev bus='usb'/>", "REMOVABLE_DEVICE_PRESENT"),
        ("<smartcard mode='passthrough'/>", "HOST_DEVICE_PRESENT"),
        ("<tpm model='tpm-tis'/>", "HOST_DEVICE_PRESENT"),
        ("<vsock model='virtio'/>", "COMMUNICATION_CHANNEL_PRESENT"),
        ("<shmem name='host'/>", "SHARED_MEMORY_PRESENT"),
        ("<rng model='virtio'/>", "HOST_DEVICE_PRESENT"),
        ("<crypto model='virtio'/>", "HOST_DEVICE_PRESENT"),
        ("<audio type='pipewire'/>", "HOST_DEVICE_PRESENT"),
        ("<unreviewed/>", "UNREVIEWED_DOMAIN_CONFIGURATION"),
        ("<disk socket='/run/host.sock'/>", "HOST_SOCKET_ACCESS_PRESENT"),
        (
            "<qemu:commandline xmlns:qemu='http://libvirt.org/schemas/domain/qemu/1.0'/>",
            "QEMU_EXTENSION_REJECTED",
        ),
        ("<controller type='usb' model='qemu-xhci'/>", "REMOVABLE_CONTROLLER_PRESENT"),
        ("<controller type='unreviewed'/>", "CONTROLLER_DEVICE_UNREVIEWED"),
    ],
)
def test_synthetic_unsafe_devices_are_refused(raw: str, reason: str) -> None:
    inspection = domain.inspect_domain_v5d1(_snapshot(_device(raw)), _requirements())
    assert not inspection.config_matches
    assert reason in inspection.reason_codes


@pytest.mark.parametrize(
    ("old", "new", "reason"),
    [
        ("type='kvm'", "type='qemu'", "DOMAIN_TYPE_REJECTED"),
        (UUID, "00000000-0000-0000-0000-000000000000", "DOMAIN_UUID_MISMATCH"),
        ("unit='KiB'>524288", "unit='MiB'>524288", "MEMORY_CONFIGURATION_MISMATCH"),
        ("unit='KiB'>524288", "unit='unknown'>524288", "MEMORY_CONFIGURATION_MISMATCH"),
        ("unit='KiB'>524288", "unit='KiB'>bad", "MEMORY_CONFIGURATION_MISMATCH"),
        ("placement='static'>1", "placement='static'>2", "CPU_CONFIGURATION_MISMATCH"),
        ("placement='static'>1", "placement='static'>bad", "CPU_CONFIGURATION_MISMATCH"),
        ("placement='static'>1", "current='2'>1", "CPU_HOTPLUG_REJECTED"),
        ("<quota>100000</quota>", "<quota>-1</quota>", "CPU_LIMIT_CONFIGURATION_MISMATCH"),
        ("mode='host-model'", "mode='host-passthrough'", "CPU_MODE_UNREVIEWED"),
        ("arch='x86_64'", "arch='aarch64'", "GUEST_MACHINE_UNREVIEWED"),
        ("type='file' device='disk'", "type='network' device='disk'", "DISK_DEVICE_REJECTED"),
        ("device='disk'", "device='cdrom'", "REMOVABLE_DEVICE_PRESENT"),
        ("name='qemu' type='qcow2'", "name='qemu' type='vmdk'", "DISK_FORMAT_REJECTED"),
        (
            "name='qemu' type='qcow2'",
            "name='qemu' type='qcow2' queues='16'",
            "DISK_DRIVER_REJECTED",
        ),
        (DISK, "/srv/stage9/replaced.qcow2", "DISK_INVENTORY_MISMATCH"),
        (DISK, "../outside.qcow2", "DISK_SOURCE_REJECTED"),
        ("<backingStore/>", "<shareable/>", "DISK_LIFECYCLE_UNVERIFIED"),
        ("<backingStore/>", "<transient/>", "DISK_LIFECYCLE_UNVERIFIED"),
        ("<memballoon model='none'/>", "<memballoon model='virtio'/>", "MEMORY_BALLOON_REJECTED"),
        ("<source file=", "<source startupPolicy='optional' file=", "DISK_SOURCE_REJECTED"),
        (
            "<backingStore/>",
            "<backingStore><source file='../outside'/></backingStore>",
            "BACKING_SOURCE_REJECTED",
        ),
    ],
)
def test_synthetic_unsafe_configuration_is_refused(old: str, new: str, reason: str) -> None:
    inspection = domain.inspect_domain_v5d1(_snapshot(XML.replace(old, new)), _requirements())
    assert not inspection.config_matches
    assert reason in inspection.reason_codes


def test_backing_declarations_are_inventory_only_not_measurements() -> None:
    xml = XML.replace(
        "<backingStore/>",
        "<backingStore><format type='raw'/><source file='/srv/base.raw'/>"
        "<backingStore/></backingStore>",
    )
    inspection = domain.inspect_domain_v5d1(_snapshot(xml), _requirements())
    assert inspection.config_matches
    assert inspection.backing_paths == ("/srv/base.raw",)
    assert inspection.missing_evidence


@pytest.mark.parametrize(
    ("changes", "reason"),
    [
        ({"state": 1, "active": True, "domain_id": 9}, "DOMAIN_NOT_SHUTOFF"),
        ({"domain_id": 1}, "DOMAIN_NOT_SHUTOFF"),
        ({"persistent": False}, "DOMAIN_PERSISTENCE_UNVERIFIED"),
        ({"snapshot_names": ("old",)}, "DOMAIN_SNAPSHOTS_PRESENT"),
        ({"managed_save": True}, "DOMAIN_MANAGED_SAVE_PRESENT"),
        ({"domain_uuid": "wrong"}, "DOMAIN_UUID_MISMATCH"),
        ({"memory_kib": 1}, "DOMAIN_EFFECTIVE_RESOURCES_MISMATCH"),
        ({"max_memory_kib": 1}, "DOMAIN_EFFECTIVE_RESOURCES_MISMATCH"),
        ({"vcpu_count": 2}, "DOMAIN_EFFECTIVE_RESOURCES_MISMATCH"),
        (
            {"networks": (domain.NetworkObservationV5d1("default", UUID, True, "<network/>"),)},
            "NETWORK_DEVICE_PRESENT",
        ),
        (
            {"inactive_xml": XML.replace(DISK, "/srv/changed.raw")},
            "DOMAIN_LIVE_PERSISTENT_MISMATCH",
        ),
    ],
)
def test_synthetic_observed_state_is_checked(changes: dict[str, Any], reason: str) -> None:
    inspection = domain.inspect_domain_v5d1(_snapshot(**changes), _requirements())
    assert reason in inspection.reason_codes


@pytest.mark.parametrize(
    "changes",
    [
        {"xml": XML + " "},
        {"state_reason": 4},
        {"scheduler_parameters": (("vcpu_quota", -1),)},
        {"snapshot_names": ("created",)},
    ],
)
def test_synthetic_drift_between_reads_is_refused(changes: dict[str, Any]) -> None:
    first = _snapshot()
    result = domain.inspect_domain_pair_v5d1(first, replace(first, **changes), _requirements())
    assert not result.config_matches
    assert "DOMAIN_CHANGED_DURING_PREFLIGHT" in result.reason_codes


def test_duplicate_configuration_is_refused() -> None:
    xml = XML.replace("</vcpu>", "</vcpu><vcpu>1</vcpu>")
    result = domain.inspect_domain_v5d1(_snapshot(xml), _requirements())
    assert "DUPLICATE_DOMAIN_CONFIGURATION" in result.reason_codes


@pytest.mark.parametrize(
    "raw",
    [
        "<domain>",
        "<network/>",
        "<!DOCTYPE domain><domain/>",
        "<!ENTITY x 'danger'><domain/>",
        "x" * (domain.MAX_XML_BYTES + 1),
        "<domain>" + "<x>" * 33 + "</x>" * 33 + "</domain>",
        "<domain>" + "<x/>" * domain.MAX_XML_NODES + "</domain>",
    ],
)
def test_malformed_or_unbounded_xml_refused(raw: str) -> None:
    with pytest.raises(domain.DomainReadErrorV5d1, match="DOMAIN_XML_REJECTED"):
        domain.inspect_domain_v5d1(_snapshot(raw), _requirements())


@pytest.mark.parametrize("uuid", ["some-vm-name", UUID.upper(), "000", "", None])
def test_name_and_noncanonical_uuid_refused(uuid: Any) -> None:
    with pytest.raises(domain.DomainReadErrorV5d1, match="DOMAIN_UUID_REJECTED"):
        domain.inspect_domain_v5d1(_snapshot(), replace(_requirements(), domain_uuid=uuid))


@pytest.mark.parametrize(
    "changes",
    [
        {"disk_paths": ()},
        {"disk_paths": (DISK, DISK)},
        {"disk_paths": ("/srv/../bad",)},
        {"disk_paths": ("/srv//bad",)},
        {"disk_paths": ("/srv/bad\n",)},
        {"memory_bytes": True},
        {"memory_bytes": 1},
        {"vcpu_count": 0},
        {"cpu_quota_us": 100001},
    ],
)
def test_bad_requirements_refused(changes: dict[str, Any]) -> None:
    with pytest.raises(domain.DomainReadErrorV5d1, match="DOMAIN_REQUIREMENTS_REJECTED"):
        domain.inspect_domain_v5d1(_snapshot(), replace(_requirements(), **changes))


class _FakeDomain:
    def __init__(self, *, xml: str = XML) -> None:
        self.xml = xml
        self.calls: list[str] = []

    def UUIDString(self) -> str:
        return UUID

    def ID(self) -> int:
        return -1

    def XMLDesc(self, flags: int) -> str:
        self.calls.append("XMLDesc:" + str(flags))
        return self.xml

    def state(self, flags: int) -> tuple[int, int]:
        return (5, 1)

    def info(self) -> tuple[int, int, int, int, int]:
        return (5, 524288, 524288, 1, 0)

    def isActive(self) -> bool:
        return False

    def isPersistent(self) -> bool:
        return True

    def schedulerParameters(self) -> dict[str, int]:
        return {"vcpu_period": 100000, "vcpu_quota": 100000}

    def snapshotListNames(self, flags: int) -> list[str]:
        return []

    def hasManagedSaveImage(self, flags: int) -> bool:
        return False


class _FakeConnection:
    def __init__(self, fixture: _FakeDomain) -> None:
        self.fixture = fixture
        self.closed = False
        self.lookup: list[str] = []

    def getURI(self) -> str:
        return domain.URI

    def lookupByUUIDString(self, uuid: str) -> _FakeDomain:
        self.lookup.append(uuid)
        return self.fixture

    def networkLookupByName(self, name: str) -> Any:
        assert name == "default"
        return SimpleNamespace(
            XMLDesc=lambda flags: "<network/>", UUIDString=lambda: UUID, isActive=lambda: True
        )

    def close(self) -> None:
        self.closed = True


def _reader(
    monkeypatch: pytest.MonkeyPatch, fixture: _FakeDomain | None = None
) -> tuple[domain.LibvirtDomainReaderV5d1, _FakeConnection]:
    connection = _FakeConnection(fixture or _FakeDomain())
    uris: list[str] = []

    def open_read_only(uri: str) -> _FakeConnection:
        uris.append(uri)
        return connection

    monkeypatch.setattr(
        domain.importlib, "import_module", lambda name: SimpleNamespace(openReadOnly=open_read_only)
    )
    reader = domain.LibvirtDomainReaderV5d1()
    assert uris == ["qemu:///system"]
    return reader, connection


def test_native_adapter_uses_only_read_only_fixed_uri_and_uuid(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    reader, connection = _reader(monkeypatch)
    first = reader.read(UUID)
    second = reader.read(UUID)
    result = domain.inspect_domain_pair_v5d1(first, second, _requirements())
    assert result.config_matches and result.missing_evidence
    assert connection.lookup == [UUID, UUID]
    assert connection.fixture.calls == ["XMLDesc:0", "XMLDesc:2"] * 2
    reader.close()
    reader.close()
    assert connection.closed
    with pytest.raises(domain.DomainReadErrorV5d1, match="LIBVIRT_CONNECTION_CLOSED"):
        reader.read(UUID)


def test_network_metadata_is_observed_and_refused(monkeypatch: pytest.MonkeyPatch) -> None:
    reader, _ = _reader(
        monkeypatch,
        _FakeDomain(
            xml=_device("<interface type='network'><source network='default'/></interface>")
        ),
    )
    snapshot = reader.read(UUID)
    assert snapshot.networks == (
        domain.NetworkObservationV5d1("default", UUID, True, "<network/>"),
    )
    assert not domain.inspect_domain_v5d1(snapshot, _requirements()).config_matches
    reader.close()


def test_missing_libvirt_refused_without_host_details(monkeypatch: pytest.MonkeyPatch) -> None:
    def unavailable(name: str) -> None:
        raise ImportError("secret host details")

    monkeypatch.setattr(domain.importlib, "import_module", unavailable)
    with pytest.raises(domain.DomainReadErrorV5d1) as refusal:
        domain.LibvirtDomainReaderV5d1()
    assert str(refusal.value) == "LIBVIRT_UNAVAILABLE"


@pytest.mark.parametrize("mode", ["missing", "remote"])
def test_connection_scope_refused(monkeypatch: pytest.MonkeyPatch, mode: str) -> None:
    connection = SimpleNamespace(getURI=lambda: "qemu+ssh://remote/system", close=lambda: None)
    monkeypatch.setattr(
        domain.importlib,
        "import_module",
        lambda name: SimpleNamespace(
            openReadOnly=lambda uri: None if mode == "missing" else connection
        ),
    )
    with pytest.raises(domain.DomainReadErrorV5d1, match="LIBVIRT_CONNECTION_REJECTED"):
        domain.LibvirtDomainReaderV5d1()


def test_changes_within_native_read_are_refused(monkeypatch: pytest.MonkeyPatch) -> None:
    fixture = _FakeDomain()
    states = iter(((5, 1), (1, 1)))
    monkeypatch.setattr(fixture, "state", lambda flags: next(states))
    reader, _ = _reader(monkeypatch, fixture)
    with pytest.raises(domain.DomainReadErrorV5d1, match="DOMAIN_CHANGED_DURING_READ"):
        reader.read(UUID)
    reader.close()


def test_incomplete_native_observation_refused(monkeypatch: pytest.MonkeyPatch) -> None:
    fixture = _FakeDomain()
    monkeypatch.setattr(fixture, "info", lambda: (5,))
    reader, _ = _reader(monkeypatch, fixture)
    with pytest.raises(domain.DomainReadErrorV5d1, match="DOMAIN_CHANGED_DURING_READ"):
        reader.read(UUID)
    reader.close()


@pytest.mark.parametrize(
    ("old", "new", "reason"),
    [
        ("placement='static'", "placement='auto'", "CPU_PLACEMENT_UNREVIEWED"),
        ("/usr/bin/qemu-system-x86_64", "/tmp/qemu", "EMULATOR_PATH_UNREVIEWED"),
        ("</devices>", "</devices><seclabel type='none'/>", "HOST_CONFINEMENT_DISABLED"),
        (
            "<memory unit='KiB'>",
            "<memory unreviewed='yes' unit='KiB'>",
            "UNREVIEWED_DOMAIN_ATTRIBUTES",
        ),
    ],
)
def test_unreviewed_xml_semantics_are_refused(old: str, new: str, reason: str) -> None:
    result = domain.inspect_domain_v5d1(_snapshot(XML.replace(old, new)), _requirements())
    assert not result.config_matches
    assert reason in result.reason_codes


@pytest.mark.parametrize(
    ("scheduler", "reason"),
    [
        ((), "CPU_LIMIT_OBSERVATION_INCOMPLETE"),
        ((("vcpu_period", 100000), ("vcpu_quota", True)), "CPU_LIMIT_OBSERVATION_INCOMPLETE"),
        ((("vcpu_period", 100000), ("vcpu_quota", -1)), "CPU_LIMIT_OBSERVATION_MISMATCH"),
    ],
)
def test_scheduler_observation_must_match_independent_expectations(
    scheduler: Any, reason: str
) -> None:
    result = domain.inspect_domain_v5d1(_snapshot(scheduler_parameters=scheduler), _requirements())
    assert not result.config_matches
    assert reason in result.reason_codes


def test_native_missing_methods_fail_closed(monkeypatch: pytest.MonkeyPatch) -> None:
    fixture = _FakeDomain()
    monkeypatch.setattr(
        fixture,
        "hasManagedSaveImage",
        lambda flags: (_ for _ in ()).throw(RuntimeError("host detail")),
    )
    reader, _ = _reader(monkeypatch, fixture)
    with pytest.raises(domain.DomainReadErrorV5d1, match="DOMAIN_OBSERVATION_INCOMPLETE"):
        reader.read(UUID)
    reader.close()


def test_driver_must_match_independently_expected_top_format() -> None:
    requirements = replace(_requirements(), disk_formats=("raw",))
    result = domain.inspect_domain_v5d1(_snapshot(), requirements)
    assert not result.config_matches
    assert "DISK_FORMAT_EXPECTATION_MISMATCH" in result.reason_codes


def test_backing_override_must_match_full_independent_chain() -> None:
    requirements = replace(
        _requirements(), disk_formats=("qcow2",), disk_backing_chains=((DISK, "/srv/base.raw"),)
    )
    xml = XML.replace(
        "<backingStore/>",
        "<backingStore><format type='raw'/><source file='/srv/other.raw'/>"
        "<backingStore/></backingStore>",
    )
    result = domain.inspect_domain_v5d1(_snapshot(xml), requirements)
    assert not result.config_matches
    assert "BACKING_CHAIN_EXPECTATION_MISMATCH" in result.reason_codes
    empty_override = domain.inspect_domain_v5d1(_snapshot(), requirements)
    assert "BACKING_CHAIN_EXPECTATION_MISMATCH" in empty_override.reason_codes


def test_multiple_nested_backing_levels_match_full_expected_chain() -> None:
    requirements = replace(
        _requirements(),
        disk_formats=("qcow2",),
        disk_backing_chains=((DISK, "/srv/intermediate.qcow2", "/srv/base.raw"),),
    )
    xml = XML.replace(
        "<backingStore/>",
        "<backingStore type='file'><format type='qcow2'/>"
        "<source file='/srv/intermediate.qcow2'/><backingStore type='file'>"
        "<format type='raw'/><source file='/srv/base.raw'/>"
        "<backingStore/></backingStore></backingStore>",
    )
    result = domain.inspect_domain_v5d1(_snapshot(xml), requirements)
    assert result.config_matches
    assert result.backing_paths == ("/srv/intermediate.qcow2", "/srv/base.raw")


@pytest.mark.parametrize(
    "changes",
    [
        {"disk_formats": ("vmdk",)},
        {"disk_formats": ("raw", "qcow2")},
        {"disk_backing_chains": (("/srv/other.raw",),)},
        {"disk_backing_chains": ((DISK, DISK),)},
        {"disk_backing_chains": ((DISK,), (DISK,))},
        {"disk_backing_chains": ((DISK, "../bad"),)},
    ],
)
def test_invalid_independent_disk_expectations_fail_closed(changes: dict[str, Any]) -> None:
    with pytest.raises(domain.DomainReadErrorV5d1, match="DOMAIN_REQUIREMENTS_REJECTED"):
        domain.inspect_domain_v5d1(_snapshot(), replace(_requirements(), **changes))


def test_backing_format_override_must_match_independent_image_format() -> None:
    requirements = replace(
        _requirements(),
        disk_formats=("qcow2",),
        disk_backing_chains=((DISK, "/srv/base.qcow2"),),
        disk_backing_formats=(("qcow2", "qcow2"),),
    )
    xml = XML.replace(
        "<backingStore/>",
        "<backingStore type='file'><format type='raw'/><source file='/srv/base.qcow2'/>"
        "<backingStore/></backingStore>",
    )
    result = domain.inspect_domain_v5d1(_snapshot(xml), requirements)
    assert "BACKING_FORMAT_EXPECTATION_MISMATCH" in result.reason_codes
    matching = xml.replace("format type='raw'", "format type='qcow2'")
    assert domain.inspect_domain_v5d1(_snapshot(matching), requirements).config_matches


def test_invalid_backing_format_expectations_are_refused() -> None:
    requirements = replace(
        _requirements(), disk_backing_chains=((DISK,),), disk_backing_formats=(("unreviewed",),)
    )
    with pytest.raises(domain.DomainReadErrorV5d1, match="DOMAIN_REQUIREMENTS_REJECTED"):
        domain.inspect_domain_v5d1(_snapshot(), requirements)


def test_backing_device_type_override_is_refused() -> None:
    xml = XML.replace(
        "<backingStore/>",
        "<backingStore type='block'><format type='raw'/><source file='/srv/base.raw'/>"
        "<backingStore/></backingStore>",
    )
    result = domain.inspect_domain_v5d1(_snapshot(xml), _requirements())
    assert "BACKING_CONFIGURATION_REJECTED" in result.reason_codes


def test_xml_parser_explicitly_disables_dtd_entities_and_external_access(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    parse = domain.ElementTree.fromstring
    observed: list[dict[str, bool]] = []

    def guarded_parse(raw: str, **flags: bool) -> Any:
        observed.append(flags)
        return parse(raw, **flags)

    monkeypatch.setattr(domain.ElementTree, "fromstring", guarded_parse)
    assert domain.inspect_domain_v5d1(_snapshot(), _requirements()).config_matches
    assert observed == [{"forbid_dtd": True, "forbid_entities": True, "forbid_external": True}] * 2


def test_defused_xml_exceptions_become_stable_refusals(monkeypatch: pytest.MonkeyPatch) -> None:
    from defusedxml.common import DTDForbidden

    def forbidden(raw: str, **flags: bool) -> Any:
        raise DTDForbidden("domain", None, None)

    monkeypatch.setattr(domain.ElementTree, "fromstring", forbidden)
    with pytest.raises(domain.DomainReadErrorV5d1, match="DOMAIN_XML_REJECTED"):
        domain.inspect_domain_v5d1(_snapshot(), _requirements())


@pytest.mark.parametrize("scope", ["live", "persistent", "both"])
@pytest.mark.parametrize("tag", ["on_poweroff", "on_reboot", "on_crash"])
@pytest.mark.parametrize(
    "policy",
    ["restart", "preserve", "rename-restart", "coredump-restart", "unknown", "", " destroy "],
)
def test_p2_lifecycle_values_fail_closed(scope: str, tag: str, policy: str) -> None:
    changed = XML.replace(f"<{tag}>destroy</{tag}>", f"<{tag}>{policy}</{tag}>")
    snapshot = _snapshot(
        changed if scope in {"live", "both"} else XML,
        inactive_xml=changed if scope in {"persistent", "both"} else XML,
    )
    result = domain.inspect_domain_v5d1(snapshot, _requirements())
    assert result.config_matches is False
    assert "DOMAIN_LIFECYCLE_POLICY_REJECTED" in result.reason_codes
    if scope != "both":
        assert "DOMAIN_LIVE_PERSISTENT_MISMATCH" in result.reason_codes


@pytest.mark.parametrize("tag", ["on_poweroff", "on_reboot", "on_crash"])
@pytest.mark.parametrize("scope", ["live", "persistent", "both"])
def test_p2_missing_lifecycle_never_inherits_libvirt_defaults(tag: str, scope: str) -> None:
    changed = XML.replace(f"<{tag}>destroy</{tag}>", "")
    result = domain.inspect_domain_v5d1(
        _snapshot(
            changed if scope in {"live", "both"} else XML,
            inactive_xml=changed if scope in {"persistent", "both"} else XML,
        ),
        _requirements(),
    )
    assert result.config_matches is False
    assert "DOMAIN_LIFECYCLE_POLICY_MISSING" in result.reason_codes


@pytest.mark.parametrize("tag", ["on_poweroff", "on_reboot", "on_crash"])
@pytest.mark.parametrize("policy", ["destroy", "restart"])
def test_p2_duplicate_lifecycle_is_refused(tag: str, policy: str) -> None:
    changed = XML.replace(f"</{tag}>", f"</{tag}><{tag}>{policy}</{tag}>")
    result = domain.inspect_domain_v5d1(_snapshot(changed), _requirements())
    assert result.config_matches is False
    assert "DUPLICATE_DOMAIN_CONFIGURATION" in result.reason_codes
    assert "DOMAIN_LIFECYCLE_POLICY_REJECTED" in result.reason_codes


@pytest.mark.parametrize(
    ("attribute", "approved", "rejected"),
    [
        ("cache", "none", "unsafe"),
        ("cache", "none", "writeback"),
        ("cache", "none", "writethrough"),
        ("cache", "none", "directsync"),
        ("cache", "none", "unknown"),
        ("io", "native", "threads"),
        ("io", "native", "io_uring"),
        ("io", "native", "unknown"),
        ("discard", "ignore", "unmap"),
        ("discard", "ignore", "unknown"),
        ("detect_zeroes", "off", "on"),
        ("detect_zeroes", "off", "unmap"),
        ("detect_zeroes", "off", "unknown"),
    ],
)
def test_p2_allowlisted_driver_attributes_have_explicit_semantics(
    attribute: str, approved: str, rejected: str
) -> None:
    changed = XML.replace(f"{attribute}='{approved}'", f"{attribute}='{rejected}'")
    result = domain.inspect_domain_v5d1(_snapshot(changed), _requirements())
    assert result.config_matches is False
    assert "DISK_DRIVER_REJECTED" in result.reason_codes


@pytest.mark.parametrize(
    ("attribute", "value"),
    [("cache", "none"), ("io", "native"), ("discard", "ignore"), ("detect_zeroes", "off")],
)
def test_p2_driver_defaults_are_not_assumed_safe(attribute: str, value: str) -> None:
    changed = XML.replace(f"{attribute}='{value}'", "")
    result = domain.inspect_domain_v5d1(_snapshot(changed), _requirements())
    assert result.config_matches is False
    assert "DISK_DRIVER_REJECTED" in result.reason_codes


@pytest.mark.parametrize(
    ("old", "new", "reason"),
    [
        ("type='kvm'", "type='kvm' id='1'", "DOMAIN_VALUE_UNREVIEWED"),
        ("<name>synthetic-only</name>", "<name>../unsafe</name>", "DOMAIN_VALUE_UNREVIEWED"),
        ("machine='pc-q35-8.2'", "machine='unknown'", "GUEST_MACHINE_UNREVIEWED"),
        (" machine='pc-q35-8.2'", "", "GUEST_MACHINE_UNREVIEWED"),
        ("dev='hd'", "dev='network'", "BOOT_POLICY_UNREVIEWED"),
        ("<boot dev='hd'/>", "", "BOOT_POLICY_UNREVIEWED"),
        ("mode='host-model'", "mode='custom'", "CPU_MODE_UNREVIEWED"),
        ("mode='host-model'", "mode='host-model' match='exact'", "CPU_PROFILE_UNREVIEWED"),
        ("mode='host-model'", "mode='host-model' match='minimum'", "CPU_PROFILE_UNREVIEWED"),
        ("mode='host-model'", "mode='host-model' migratable='off'", "CPU_PROFILE_UNREVIEWED"),
        ("mode='host-model'", "mode='host-model' migratable='on'", "CPU_PROFILE_UNREVIEWED"),
        ("check='full'", "check='none'", "CPU_PROFILE_UNREVIEWED"),
        ("check='full'", "check='partial'", "CPU_PROFILE_UNREVIEWED"),
        ("check='full'", "check='unknown'", "CPU_PROFILE_UNREVIEWED"),
        (" check='full'", "", "CPU_PROFILE_UNREVIEWED"),
        ("sockets='1'", "sockets='2'", "CPU_TOPOLOGY_UNREVIEWED"),
        ("dies='1'", "dies='0'", "CPU_TOPOLOGY_UNREVIEWED"),
        (" clusters='1'", "", "CPU_TOPOLOGY_UNREVIEWED"),
        ("cores='1'", "cores='2'", "CPU_TOPOLOGY_UNREVIEWED"),
        (" cores='1'", "", "CPU_TOPOLOGY_UNREVIEWED"),
        ("threads='1'", "threads='2'", "CPU_TOPOLOGY_UNREVIEWED"),
        ("offset='utc'", "offset='localtime'", "CLOCK_POLICY_UNREVIEWED"),
        ("offset='utc'", "offset='unknown'", "CLOCK_POLICY_UNREVIEWED"),
        ("<clock offset='utc'/>", "", "CLOCK_POLICY_UNREVIEWED"),
        ("snapshot='no'", "snapshot='external'", "DISK_LIFECYCLE_UNVERIFIED"),
        ("snapshot='no'", "snapshot='unknown'", "DISK_LIFECYCLE_UNVERIFIED"),
        (" snapshot='no'", "", "DISK_LIFECYCLE_UNVERIFIED"),
        ("<readonly/>", "", "DISK_LIFECYCLE_UNVERIFIED"),
        ("dev='vda'", "dev='vdb'", "DISK_TARGET_UNREVIEWED"),
        ("bus='virtio'", "bus='scsi'", "DISK_TARGET_UNREVIEWED"),
        ("<target dev='vda' bus='virtio'/>", "", "DISK_TARGET_UNREVIEWED"),
        ("<controller type='usb' model='none'/>", "", "CONTROLLER_PROFILE_UNVERIFIED"),
        (
            "type='usb' model='none'",
            "type='pci' model='pcie-root'",
            "CONTROLLER_PROFILE_UNVERIFIED",
        ),
        (
            "type='usb' model='none'",
            "type='scsi' model='virtio-scsi'",
            "CONTROLLER_PROFILE_UNVERIFIED",
        ),
        ("type='usb' model='none'", "type='virtio-serial'", "CONTROLLER_PROFILE_UNVERIFIED"),
        (
            "type='usb' model='none'",
            "type='usb' model='none' index='-1'",
            "CONTROLLER_PROFILE_UNVERIFIED",
        ),
        (
            "type='usb' model='none'",
            "type='usb' model='none' index='1'",
            "CONTROLLER_PROFILE_UNVERIFIED",
        ),
        ("<memballoon model='none'/>", "", "MEMORY_BALLOON_REJECTED"),
        ("<readonly/>", "<readonly>enabled</readonly>", "DOMAIN_VALUE_UNREVIEWED"),
        ("<features>", "<features>unknown", "DOMAIN_VALUE_UNREVIEWED"),
        ("<acpi/>", "<acpi>off</acpi>", "DOMAIN_VALUE_UNREVIEWED"),
        ("<apic/>", "<apic>enabled</apic>", "DOMAIN_VALUE_UNREVIEWED"),
        ("</uuid>", "</uuid>unexpected-tail", "DOMAIN_VALUE_UNREVIEWED"),
    ],
)
def test_p2_allowlisted_values_and_missing_controls_fail_closed(
    old: str, new: str, reason: str
) -> None:
    assert old in XML
    result = domain.inspect_domain_v5d1(_snapshot(XML.replace(old, new)), _requirements())
    assert result.config_matches is False
    assert reason in result.reason_codes


@pytest.mark.parametrize(
    ("extra", "reason"),
    [
        (
            "<seclabel type='dynamic' model='apparmor' relabel='yes'/>",
            "HOST_CONFINEMENT_PROFILE_UNVERIFIED",
        ),
        (
            "<seclabel type='static' model='selinux' relabel='no'><label>unsafe</label><imagelabel>unsafe</imagelabel></seclabel>",
            "HOST_CONFINEMENT_PROFILE_UNVERIFIED",
        ),
        (
            "<resource><partition>/unverified</partition></resource>",
            "RESOURCE_PARTITION_UNVERIFIED",
        ),
        ("<title>bad&#10;title</title>", "DOMAIN_VALUE_UNREVIEWED"),
        ("<description>bad&#127;text</description>", "DOMAIN_VALUE_UNREVIEWED"),
    ],
)
def test_p2_unevaluable_host_profiles_and_invalid_metadata_refused(extra: str, reason: str) -> None:
    result = domain.inspect_domain_v5d1(
        _snapshot(XML.replace("</domain>", extra + "</domain>")), _requirements()
    )
    assert result.config_matches is False
    assert reason in result.reason_codes


@pytest.mark.parametrize(
    "attributes",
    [
        "type='pci' domain='0x0000' bus='0x00' slot='0x01' function='0x0' multifunction='off'",
        "type='drive' controller='0' bus='0' target='0' unit='0'",
        "type='unknown'",
    ],
)
def test_p2_allowlisted_addresses_need_missing_host_profile_evidence(attributes: str) -> None:
    xml = XML.replace("<readonly/>", f"<readonly/><address {attributes}/>")
    result = domain.inspect_domain_v5d1(_snapshot(xml), _requirements())
    assert result.config_matches is False
    assert "DEVICE_ADDRESS_PROFILE_UNVERIFIED" in result.reason_codes


@pytest.mark.parametrize(
    "node",
    [
        "<name>synthetic-only</name>",
        f"<uuid>{UUID}</uuid>",
        "<period>100000</period>",
        "<quota>100000</quota>",
        "<type arch='x86_64' machine='pc-q35-8.2'>hvm</type>",
        "<boot dev='hd'/>",
        "<acpi/>",
        "<apic/>",
        "<topology sockets='1' dies='1' clusters='1' cores='1' threads='1'/>",
        "<clock offset='utc'/>",
        "<emulator>/usr/bin/qemu-system-x86_64</emulator>",
        "<readonly/>",
        f"<source file='{DISK}'/>",
        "<target dev='vda' bus='virtio'/>",
        "<controller type='usb' model='none'/>",
        "<memballoon model='none'/>",
    ],
)
def test_p2_nested_duplicate_configuration_is_refused(node: str) -> None:
    assert node in XML
    result = domain.inspect_domain_v5d1(_snapshot(XML.replace(node, node * 2)), _requirements())
    assert result.config_matches is False
    assert "DUPLICATE_DOMAIN_CONFIGURATION" in result.reason_codes


def test_p2_duplicate_disk_driver_cannot_hide_an_unsafe_value() -> None:
    changed = XML.replace(
        "<readonly/>", "<driver name='qemu' type='qcow2' cache='unsafe'/><readonly/>"
    )
    result = domain.inspect_domain_v5d1(_snapshot(changed), _requirements())
    assert result.config_matches is False
    assert "DUPLICATE_DOMAIN_CONFIGURATION" in result.reason_codes
    assert "DISK_DRIVER_REJECTED" in result.reason_codes


@pytest.mark.parametrize(
    "backing",
    [
        "<backingStore index='1'/>",
        "<backingStore type='file'/>",
        "<backingStore><format type='raw'/></backingStore>",
        "<backingStore><backingStore/></backingStore>",
        "<backingStore index='-1'><format type='raw'/><source file='/srv/base.raw'/></backingStore>",
        "<backingStore index='2'><format type='raw'/><source file='/srv/base.raw'/></backingStore>",
        "<backingStore><format type='unknown'/><source file='/srv/base.raw'/></backingStore>",
        "<backingStore><source file='/srv/base.raw'/></backingStore>",
    ],
)
def test_p2_backing_metadata_has_explicit_semantics(backing: str) -> None:
    result = domain.inspect_domain_v5d1(
        _snapshot(XML.replace("<backingStore/>", backing)), _requirements()
    )
    assert result.config_matches is False
    assert "BACKING_CONFIGURATION_REJECTED" in result.reason_codes


@pytest.mark.parametrize("alias", ["bad/name", "bad name", "", "a" * 64])
def test_p2_alias_metadata_is_bounded_and_canonical(alias: str) -> None:
    result = domain.inspect_domain_v5d1(
        _snapshot(XML.replace("<readonly/>", f"<readonly/><alias name='{alias}'/>")),
        _requirements(),
    )
    assert result.config_matches is False
    assert "DEVICE_ALIAS_UNREVIEWED" in result.reason_codes


def test_p2_duplicate_aliases_across_devices_are_refused() -> None:
    xml = XML.replace("<readonly/>", "<readonly/><alias name='ua-test'/>").replace(
        "<controller type='usb' model='none'/>",
        "<controller type='usb' model='none'><alias name='ua-test'/></controller>",
    )
    result = domain.inspect_domain_v5d1(_snapshot(xml), _requirements())
    assert result.config_matches is False
    assert "DEVICE_ALIAS_UNREVIEWED" in result.reason_codes


def test_p2_individually_approved_live_and_persistent_trees_must_agree() -> None:
    alternate = XML.replace("<readonly/>", "<readonly/><alias name='ua-independent'/>")
    assert domain.inspect_domain_v5d1(_snapshot(alternate), _requirements()).config_matches
    result = domain.inspect_domain_v5d1(_snapshot(inactive_xml=alternate), _requirements())
    assert result.config_matches is False
    assert result.reason_codes == ("DOMAIN_LIVE_PERSISTENT_MISMATCH",)


def test_p2_reviewed_profile_accepts_known_defaults_and_inert_metadata() -> None:
    xml = (
        XML.replace("type='kvm'", "type='kvm' id='-1'")
        .replace("unit='KiB'", "unit='MiB'")
        .replace(">524288</", ">512</")
        .replace("placement='static'", "current='1'")
        .replace(
            "<controller type='usb' model='none'/>",
            "<controller type='usb' model='none' index='0'/>",
        )
        .replace("<readonly/>", "<readonly/><alias name='ua-test'/>")
        .replace(
            "</domain>",
            "<title>Test VM</title><description>Synthetic evidence only</description></domain>",
        )
    )
    result = domain.inspect_domain_v5d1(_snapshot(xml), _requirements())
    assert result.config_matches is True
    assert result.reason_codes == ()
    assert result.missing_evidence == domain.MISSING_ENFORCEMENT


def test_p2_xml_indentation_and_attribute_order_are_not_semantic_drift() -> None:
    alternate = (
        XML.replace("type='kvm'", 'type="kvm"')
        .replace("<target dev='vda' bus='virtio'/>", "<target bus='virtio' dev='vda'/>")
        .replace("\n", "\n  ")
    )
    result = domain.inspect_domain_v5d1(_snapshot(inactive_xml=alternate), _requirements())
    assert result.config_matches is True
    assert result.reason_codes == ()
