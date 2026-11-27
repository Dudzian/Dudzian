"""Payload for a separately authorized native diagnostic carrier.

The carrier installs the reviewed import finder and kernel policy before these
functions run. This module has no standalone entry point and imports no Observer
module. Its results need external process and protocol measurements.
"""

from __future__ import annotations

from importlib import import_module
from types import ModuleType
from typing import Any
import uuid

MAX_XML_BYTES = 2 * 1024 * 1024
URI = "qemu:///system"


class Refusal(ValueError):
    """The diagnostic fixture or read observations were incompatible."""


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


def exercise_libvirt(module: ModuleType, domain_uuid: str, network_name: str) -> dict[str, str]:
    """Exercise V5d1 read RPCs on a preprovisioned disposable domain."""
    expected_uuid = _uuid(domain_uuid)
    if not network_name or len(network_name) > 128:
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
    finally:
        connection.close()
    steps["close"] = "PASS"
    return steps
