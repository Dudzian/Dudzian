
"""Generowanie/pozyskiwanie i podpisywanie fingerprintów sprzętowych (OEM + host)."""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import logging
import os
import platform
import re
import socket
import subprocess
import sys
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from types import MappingProxyType
from typing import Any, Callable, Iterable, Mapping, MutableMapping, Optional, Sequence, cast

from bot_core.security.fingerprint_crypto import (
    current_hwid_digest as crypto_current_hwid_digest,
    decrypt_license_secret as crypto_decrypt_license_secret,
    derive_encryption_key as crypto_derive_encryption_key,
    encrypt_license_secret as crypto_encrypt_license_secret,
)
from bot_core.security.rotation import RotationRegistry, RotationStatus
from bot_core.security.signing import build_hmac_signature, canonical_json_bytes

# ---------------------------------------------------------------------------
# Wspólne stałe/typy
# ---------------------------------------------------------------------------

# OEM (starsze API) – format skrótu i algorytm podpisu
FINGERPRINT_HASH = "SHA384"
SIGNATURE_ALGORITHM = "HMAC-SHA384"
DEFAULT_AUDIT_LOG_PATH = Path("logs/security_admin.log")

# Nowsze API – domyślny "cel" w rejestrze rotacji
_DEFAULT_PURPOSE = "hardware-fingerprint"
# Starsze API – zachowanie zgodności w helperach OEM
_OEM_DEFAULT_PURPOSE = "oem-fingerprint-signing"

_HEX_RE = re.compile(r"[^0-9a-f]")

# Lokalny sekret do podpisywania snapshotów licencji.
LICENSE_SECRET_PATH = Path("var/security/license_secret.key")
LICENSE_SIGNATURE_ALGORITHM = "HMAC-SHA384"

LICENSE_SECRET_KEYRING_SERVICE = "dudzian.license"
LICENSE_SECRET_KEYRING_ENTRY = "binding-secret.v1"
LICENSE_SECRET_FILE_VERSION = 2


LOGGER = logging.getLogger(__name__)


class FingerprintError(RuntimeError):
    """Wyjątek zgłaszany przy problemach z generowaniem/podpisywaniem fingerprintu."""


# ---------------------------------------------------------------------------
# Heurystyki bezpieczeństwa środowiska uruchomieniowego
# ---------------------------------------------------------------------------

_MISSING = object()

_VM_CPU_PATTERNS: tuple[tuple[str, str], ...] = (
    ("kvm", "KVM"),
    ("qemu", "QEMU"),
    ("virtualbox", "VirtualBox"),
    ("vmware", "VMware"),
    ("hyper-v", "Hyper-V"),
    ("microsoft corporation hyper-v", "Hyper-V"),
    ("parallels", "Parallels"),
    ("xen", "Xen"),
    ("bhyve", "Bhyve"),
)

_VM_DMI_PATTERNS: tuple[tuple[str, str], ...] = (
    ("vmware", "VMware"),
    ("virtualbox", "VirtualBox"),
    ("innotek", "VirtualBox"),
    ("qemu", "QEMU"),
    ("kvm", "KVM"),
    ("microsoft corporation", "Hyper-V"),
    ("hyper-v", "Hyper-V"),
    ("parallels", "Parallels"),
    ("xen", "Xen"),
    ("bhyve", "Bhyve"),
    ("amazon", "Amazon EC2"),
    ("google", "Google Cloud"),
    ("digitalocean", "DigitalOcean"),
    ("linode", "Linode"),
    ("ovh", "OVH"),
)

_DMIDECODE_RELEVANT_FIELDS: tuple[str, ...] = (
    "manufacturer",
    "product_name",
    "product_version",
    "family",
    "version",
)

_VM_MAC_PREFIXES: Mapping[str, str] = MappingProxyType(
    {
        "000569": "VMware",
        "000C29": "VMware",
        "005056": "VMware",
        "080027": "VirtualBox",
        "00155D": "Microsoft Hyper-V",
        "001C14": "Microsoft Hyper-V",
        "001C42": "Parallels",
        "525400": "QEMU",  # często używany przez libvirt
        "1AB4D5": "Virtualizor",
    }
)

_VM_PROCESS_PATTERNS: tuple[tuple[str, str], ...] = (
    ("vboxservice", "VirtualBox Tools"),
    ("vboxtray", "VirtualBox Tools"),
    ("vmtoolsd", "VMware Tools"),
    ("vmware", "VMware"),
    ("qemu-ga", "QEMU guest agent"),
    ("qemu-guest-agent", "QEMU guest agent"),
    ("hyperv", "Hyper-V"),
    ("hyper-v", "Hyper-V"),
    ("xenstore", "Xen tools"),
    ("xenstored", "Xen tools"),
    ("prl_tools", "Parallels Tools"),
    ("prlcc", "Parallels Tools"),
)

_VM_FILESYSTEM_MARKERS: tuple[tuple[Path, str], ...] = (
    (Path("/dev/vboxguest"), "Wykryto urządzenie VirtualBox Guest Additions."),
    (Path("/dev/vmmemctl"), "Wykryto urządzenie VMware Memory Control."),
    (Path("/dev/vmci"), "Wykryto urządzenie VMware Communication Interface."),
    (Path("/dev/prl_tg"), "Wykryto urządzenie Parallels Tools."),
    (Path("/proc/xen"), "Wykryto interfejs /proc/xen (Xen)."),
    (Path("/sys/hypervisor/type"), "Wykryto interfejs sysfs hypervisora."),
    (Path("/sys/bus/vmbus/devices"), "Wykryto magistralę Hyper-V (VMBus)."),
)

_VM_KERNEL_MODULE_PATTERNS: tuple[tuple[str, str], ...] = (
    ("vboxguest", "Wykryto moduł VirtualBox Guest Additions."),
    ("vboxsf", "Wykryto moduł VirtualBox Shared Folders."),
    ("vmw_balloon", "Wykryto moduł VMware Balloon."),
    ("vmhgfs", "Wykryto moduł VMware Shared Folders."),
    ("vmmemctl", "Wykryto moduł VMware Memory Control."),
    ("hv_vmbus", "Wykryto moduł Hyper-V VMBus."),
    ("hv_utils", "Wykryto moduł narzędzi Hyper-V."),
    ("xenfs", "Wykryto moduł XenFS."),
    ("xen_platform_pci", "Wykryto moduł Xen platform PCI."),
    ("prl_fs", "Wykryto moduł Parallels Shared Folders."),
)

_CONTAINER_FILESYSTEM_MARKERS: tuple[tuple[Path, str], ...] = (
    (Path("/.dockerenv"), "Wykryto marker kontenera Docker (/.dockerenv)."),
    (Path("/run/.containerenv"), "Wykryto marker środowiska kontenera (.containerenv)."),
)

_CONTAINER_ENV_KEYS: Mapping[str, str] = MappingProxyType(
    {
        "container": "systemd",  # ustawiane przez systemd-detect-virt
        "DOCKER_CONTAINER": "Docker",
        "KUBERNETES_SERVICE_HOST": "Kubernetes",
        "OPENSHIFT_BUILD_NAME": "OpenShift",
        "PODMAN_MACHINE": "Podman",
    }
)

_CONTAINER_CGROUP_PATTERNS: tuple[tuple[str, str], ...] = (
    ("/docker/", "Cgroup PID 1 wskazuje na kontener Docker."),
    ("docker-", "Cgroup PID 1 wskazuje na kontener Docker."),
    ("kubepods", "Cgroup PID 1 wskazuje na Kubernetes."),
    ("containerd", "Cgroup PID 1 wskazuje na runtime containerd."),
    ("libpod", "Cgroup PID 1 wskazuje na Podman."),
    ("podman", "Cgroup PID 1 wskazuje na Podman."),
    ("lxc", "Cgroup PID 1 wskazuje na kontener LXC."),
)

_DEBUGGER_ENV_KEYS: Mapping[str, str] = MappingProxyType(
    {
        "PYCHARM_HOSTED": "PyCharm debugger",
        "PYDEVD_LOAD_VALUES_ASYNC": "pydevd",
        "PYDEVD_USE_FRAME_EVAL": "pydevd",
        "PYDEV_DEBUGGER": "pydevd",
        "DEBUGPY_LAUNCHER_PORT": "debugpy",
        "VSCODE_PID": "VSCode debugger",
    }
)

_SYSTEMD_VM_LABELS: Mapping[str, str] = MappingProxyType(
    {
        "kvm": "KVM",
        "qemu": "QEMU",
        "vmware": "VMware",
        "microsoft": "Hyper-V",
        "hyperv": "Hyper-V",
        "oracle": "VirtualBox",
        "xen": "Xen",
        "bochs": "Bochs",
        "uml": "User Mode Linux",
        "zvm": "IBM z/VM",
        "bhyve": "Bhyve",
        "parallels": "Parallels",
        "apple": "Apple Hypervisor",
        "wsl": "Windows Subsystem for Linux",
    }
)

_SYSTEMD_CONTAINER_LABELS: Mapping[str, str] = MappingProxyType(
    {
        "docker": "Docker",
        "podman": "Podman",
        "lxc": "LXC",
        "lxc-libvirt": "LXC (libvirt)",
        "systemd-nspawn": "systemd-nspawn",
        "container-other": "inne środowisko kontenerowe",
        "chroot": "chroot",
        "openvz": "OpenVZ",
        "proot": "proot",
        "jail": "FreeBSD Jail",
    }
)

_HOSTNAMECTL_VM_LABELS: Mapping[str, str] = MappingProxyType(dict(_SYSTEMD_VM_LABELS))
_HOSTNAMECTL_CONTAINER_LABELS: Mapping[str, str] = MappingProxyType(
    dict(_SYSTEMD_CONTAINER_LABELS)
)

_LSCPU_VENDOR_LABELS: Mapping[str, str] = MappingProxyType(
    {
        "kvm": "KVM",
        "qemu": "QEMU",
        "vmware": "VMware",
        "microsoft": "Hyper-V",
        "hyper-v": "Hyper-V",
        "hyperv": "Hyper-V",
        "oracle": "VirtualBox",
        "xen": "Xen",
        "bhyve": "Bhyve",
        "parallels": "Parallels",
        "apple": "Apple Hypervisor",
        "bochs": "Bochs",
        "uml": "User Mode Linux",
        "wsl": "Windows Subsystem for Linux",
    }
)

_VIRTWHAT_VM_LABELS: Mapping[str, str] = MappingProxyType(
    {
        "kvm": "KVM",
        "qemu": "QEMU",
        "xen": "Xen",
        "xen-hvm": "Xen",
        "xen-dom0": "Xen",
        "xen-domu": "Xen",
        "vmware": "VMware",
        "microsoft": "Hyper-V",
        "hyperv": "Hyper-V",
        "hyper-v": "Hyper-V",
        "oracle": "VirtualBox",
        "virtualbox": "VirtualBox",
        "ovirt": "oVirt",
        "rhev": "Red Hat Virtualization",
        "wsl": "Windows Subsystem for Linux",
        "parallels": "Parallels",
        "bhyve": "Bhyve",
        "ibm_systemz": "IBM System z",
        "ldom": "Oracle LDom",
        "powervm": "IBM PowerVM",
        "zvm": "IBM z/VM",
        "uml": "User Mode Linux",
    }
)

_VIRTWHAT_CONTAINER_LABELS: Mapping[str, str] = MappingProxyType(
    {
        "docker": "Docker",
        "podman": "Podman",
        "lxc": "LXC",
        "lxc-libvirt": "LXC (libvirt)",
        "openvz": "OpenVZ",
        "systemd-nspawn": "systemd-nspawn",
        "vserver": "Linux-VServer",
        "linux_vserver": "Linux-VServer",
        "uml": "User Mode Linux",
        "proot": "proot",
        "flatpak": "Flatpak",
        "snap": "Snap",
        "jail": "FreeBSD Jail",
        "container-other": "inne środowisko kontenerowe",
    }
)


@dataclass(frozen=True, slots=True)
class SecuritySignals:
    """Zebrane heurystyki bezpieczeństwa uruchomienia."""

    vm_indicators: Mapping[str, tuple[str, ...]]
    debugger_indicators: tuple[str, ...]

    @property
    def vm_reasons(self) -> tuple[str, ...]:
        reasons: list[str] = []
        for bucket in self.vm_indicators.values():
            reasons.extend(bucket)
        return tuple(reasons)

    @property
    def debugger_reasons(self) -> tuple[str, ...]:
        return self.debugger_indicators

    @property
    def vm_categories(self) -> tuple[str, ...]:
        return tuple(key for key, values in self.vm_indicators.items() if values)

    @property
    def should_block_vm(self) -> bool:
        if not self.vm_reasons:
            return False
        if (
            self.vm_indicators.get("process")
            or self.vm_indicators.get("mac_vendor")
            or self.vm_indicators.get("dmi")
            or self.vm_indicators.get("dmidecode")
            or self.vm_indicators.get("wmic")
            or self.vm_indicators.get("filesystem")
            or self.vm_indicators.get("kernel_module")
            or self.vm_indicators.get("lscpu")
            or self.vm_indicators.get("container")
            or self.vm_indicators.get("hostnamectl")
            or self.vm_indicators.get("systemd")
            or self.vm_indicators.get("virt_what")
        ):
            return True
        return len(self.vm_categories) >= 2

    @property
    def should_block(self) -> bool:
        return self.should_block_vm or bool(self.debugger_indicators)

    def summary(self) -> tuple[str, ...]:
        return self.vm_reasons + self.debugger_reasons


def _read_cpu_flags() -> tuple[str, ...]:
    cpuinfo_path = Path("/proc/cpuinfo")
    try:
        with cpuinfo_path.open("r", encoding="utf-8", errors="ignore") as handle:
            for line in handle:
                lowered = line.lower()
                if lowered.startswith("flags") or lowered.startswith("features"):
                    _, _, value = line.partition(":")
                    tokens = [token.strip() for token in value.split() if token.strip()]
                    if tokens:
                        # usuń duplikaty przy zachowaniu kolejności
                        unique = tuple(dict.fromkeys(tokens))
                        return unique
    except (FileNotFoundError, OSError):
        return ()
    return ()


def _detect_vm_cpu_signals(cpu_info: str | None, cpu_flags: Sequence[str]) -> list[str]:
    signals: list[str] = []
    if cpu_info:
        normalized = cpu_info.lower()
        for pattern, label in _VM_CPU_PATTERNS:
            if pattern in normalized:
                signals.append(
                    f"Identyfikator CPU wskazuje na środowisko wirtualne ({label})."
                )
    if any(flag == "hypervisor" for flag in cpu_flags):
        signals.append("Flagi CPU zawierają znacznik 'hypervisor'.")
    return signals


def _detect_vm_mac_signals(mac_addresses: Sequence[str]) -> list[str]:
    signals: list[str] = []
    for entry in mac_addresses:
        normalized = _HEX_RE.sub("", entry.lower())
        if len(normalized) < 12:
            continue
        prefix = normalized[:6].upper()
        vendor = _VM_MAC_PREFIXES.get(prefix)
        if vendor:
            formatted = ":".join(normalized[i : i + 2] for i in range(0, 12, 2))
            signals.append(f"Adres MAC {formatted} należy do środowiska {vendor}.")
    return signals


def _list_process_names() -> tuple[str, ...]:
    commands = (
        ["ps", "-eo", "comm="],
        ["ps", "-A", "-o", "comm="],
    )
    for command in commands:
        output = _run_command(command)
        if not output:
            continue
        names = [line.strip() for line in output.splitlines() if line.strip()]
        if names:
            return tuple(dict.fromkeys(names))
    system = platform.system().lower()
    if system == "windows":
        try:
            completed = subprocess.run(
                ["tasklist", "/fo", "csv", "/nh"],
                check=False,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                text=True,
                encoding="utf-8",
            )
        except OSError:
            return ()
        names: list[str] = []
        for line in completed.stdout.splitlines():
            parts = [part.strip('"') for part in line.split(",")]
            if parts and parts[0]:
                names.append(parts[0])
        if names:
            return tuple(dict.fromkeys(names))
    return ()


def _detect_vm_process_signals(processes: Sequence[str]) -> list[str]:
    signals: list[str] = []
    for process in processes:
        lowered = process.strip().lower()
        if not lowered:
            continue
        for pattern, description in _VM_PROCESS_PATTERNS:
            if pattern in lowered:
                signals.append(f"Wykryto proces narzędzi hypervisora ({process.strip()}).")
                break
    return signals


def _detect_vm_filesystem_signals(*, path_exists: Callable[[Path], bool]) -> list[str]:
    signals: list[str] = []
    for marker, description in _VM_FILESYSTEM_MARKERS:
        try:
            if path_exists(marker):
                signals.append(description)
        except OSError:
            continue
    if signals:
        signals = list(dict.fromkeys(signals))
    return signals


def _detect_vm_kernel_module_signals(modules: Sequence[str]) -> list[str]:
    signals: list[str] = []
    for module in modules:
        lowered = module.strip().lower()
        if not lowered:
            continue
        for pattern, description in _VM_KERNEL_MODULE_PATTERNS:
            if pattern in lowered:
                signals.append(description)
                break
    if signals:
        signals = list(dict.fromkeys(signals))
    return signals


def _detect_vm_dmi_signals(entries: Sequence[str]) -> list[str]:
    signals: list[str] = []
    for entry in entries:
        lowered = entry.strip().lower()
        if not lowered:
            continue
        for pattern, label in _VM_DMI_PATTERNS:
            if pattern in lowered:
                signals.append(
                    "Identyfikatory sprzętu (DMI) wskazują na środowisko "
                    f"wirtualne ({label})."
                )
                break
    if signals:
        # usuń duplikaty zachowując kolejność i zapewniając krótszą listę powodów
        signals = list(dict.fromkeys(signals))
    return signals


def _probe_dmidecode_info() -> Mapping[str, str]:
    output = _run_command(["dmidecode", "-t", "system"])
    if not output:
        return {}

    info: dict[str, str] = {}
    for raw_line in output.splitlines():
        line = raw_line.strip()
        if not line or ":" not in line:
            continue
        key, value = line.split(":", 1)
        normalized_key = key.strip().lower().replace(" ", "_")
        sanitized_value = value.strip()
        if not sanitized_value or normalized_key not in _DMIDECODE_RELEVANT_FIELDS:
            continue
        info.setdefault(normalized_key, sanitized_value)
    return info


def _detect_dmidecode_signals(info: Mapping[str, str]) -> list[str]:
    if not info:
        return []

    signals: list[str] = []
    for field in _DMIDECODE_RELEVANT_FIELDS:
        value = info.get(field)
        if not value:
            continue
        lowered = value.strip().lower()
        if not lowered:
            continue
        for pattern, label in _VM_DMI_PATTERNS:
            if pattern in lowered:
                signals.append(
                    f"dmidecode ({field}) wskazuje na środowisko wirtualne ({label})."
                )
                break
    if signals:
        signals = list(dict.fromkeys(signals))
    return signals


def _probe_wmic_system_info() -> Mapping[str, str]:
    system = platform.system().lower()
    if system != "windows":
        return {}

    # WMI często pozwala na format list, co ułatwia parsowanie.
    output = _run_command(
        [
            "wmic",
            "computersystem",
            "get",
            "manufacturer,model,systemtype",
            "/format:list",
        ]
    )
    if not output:
        return {}

    info: dict[str, str] = {}
    for raw_line in output.splitlines():
        if "=" not in raw_line:
            continue
        key, value = raw_line.split("=", 1)
        normalized_key = key.strip().lower().replace(" ", "_")
        sanitized_value = value.strip()
        if not normalized_key or not sanitized_value:
            continue
        info.setdefault(normalized_key, sanitized_value)
    return info


def _detect_wmic_signals(info: Mapping[str, str]) -> list[str]:
    if not info:
        return []

    signals: list[str] = []
    for key, value in info.items():
        if not value:
            continue
        lowered = value.strip().lower()
        if not lowered:
            continue
        for pattern, label in _VM_DMI_PATTERNS:
            if pattern in lowered:
                signals.append(
                    f"WMIC ({key}) wskazuje na środowisko wirtualne ({label})."
                )
                break
    if signals:
        signals = list(dict.fromkeys(signals))
    return signals


def _detect_container_filesystem_signals(*, path_exists: Callable[[Path], bool]) -> list[str]:
    signals: list[str] = []
    for marker, description in _CONTAINER_FILESYSTEM_MARKERS:
        try:
            if path_exists(marker):
                signals.append(description)
        except OSError:
            continue
    if signals:
        signals = list(dict.fromkeys(signals))
    return signals


def _detect_container_env_signals(env: Mapping[str, str]) -> list[str]:
    signals: list[str] = []
    for key, label in _CONTAINER_ENV_KEYS.items():
        value = env.get(key)
        if value:
            signals.append(
                f"Zmienna środowiskowa {key} wskazuje na uruchomienie w kontenerze ({label})."
            )
    return signals


def _detect_container_cgroup_signals(entries: Sequence[str]) -> list[str]:
    signals: list[str] = []
    for entry in entries:
        lowered = entry.strip().lower()
        if not lowered:
            continue
        for pattern, description in _CONTAINER_CGROUP_PATTERNS:
            if pattern in lowered:
                signals.append(description)
                break
    if signals:
        signals = list(dict.fromkeys(signals))
    return signals


def _probe_systemd_virtualization() -> str | None:
    output = _run_command(["systemd-detect-virt"])
    if not output:
        return None
    normalized = output.strip().lower()
    if not normalized or normalized == "none":
        return None
    return normalized


def _probe_hostnamectl_virtualization() -> str | None:
    output = _run_command(["hostnamectl"])
    if not output:
        return None

    for line in output.splitlines():
        if not line:
            continue
        if "virtualization" not in line.lower():
            continue
        _, _, value = line.partition(":")
        candidate = value.strip().lower()
        if not candidate or candidate in {"n/a", "na", "none"}:
            return None
        return candidate

    return None


def _detect_systemd_virtualization_signals(value: str | None) -> tuple[list[str], list[str]]:
    if not value:
        return ([], [])

    vm_signals: list[str] = []
    container_signals: list[str] = []

    vm_label = _SYSTEMD_VM_LABELS.get(value)
    container_label = _SYSTEMD_CONTAINER_LABELS.get(value)

    if vm_label:
        vm_signals.append(
            "systemd-detect-virt zgłasza uruchomienie w środowisku wirtualnym "
            f"({vm_label})."
        )
    elif container_label:
        container_signals.append(
            "systemd-detect-virt zgłasza uruchomienie w kontenerze "
            f"({container_label})."
        )
    else:
        vm_signals.append(
            "systemd-detect-virt zwrócił nierozpoznany typ środowiska: "
            f"{value}."
        )

    return (vm_signals, container_signals)


def _detect_hostnamectl_virtualization_signals(
    value: str | None,
) -> tuple[list[str], list[str]]:
    if not value:
        return ([], [])

    vm_signals: list[str] = []
    container_signals: list[str] = []

    vm_label = _HOSTNAMECTL_VM_LABELS.get(value)
    container_label = _HOSTNAMECTL_CONTAINER_LABELS.get(value)

    if vm_label:
        vm_signals.append(
            "hostnamectl raportuje uruchomienie w środowisku wirtualnym "
            f"({vm_label})."
        )
    elif container_label:
        container_signals.append(
            "hostnamectl raportuje uruchomienie w kontenerze "
            f"({container_label})."
        )
    else:
        vm_signals.append(
            "hostnamectl zwrócił nierozpoznany typ wirtualizacji: "
            f"{value}."
        )

    return (vm_signals, container_signals)


def _probe_lscpu_info() -> Mapping[str, str]:
    output = _run_command(["lscpu"])
    if not output:
        return {}

    info: dict[str, str] = {}
    for raw_line in output.splitlines():
        if not raw_line or "::" in raw_line:
            continue
        if ':' not in raw_line:
            continue
        key, value = raw_line.split(':', 1)
        normalized_key = key.strip().lower()
        if not normalized_key:
            continue
        sanitized_value = value.strip()
        if not sanitized_value:
            continue
        if normalized_key == "hypervisor vendor":
            info["hypervisor_vendor"] = sanitized_value
        elif normalized_key == "virtualization type":
            info["virtualization_type"] = sanitized_value
    return info


def _detect_lscpu_signals(info: Mapping[str, str]) -> tuple[list[str], list[str]]:
    if not info:
        return ([], [])

    vm_signals: list[str] = []
    container_signals: list[str] = []

    vendor = info.get("hypervisor_vendor")
    if vendor:
        label = _LSCPU_VENDOR_LABELS.get(vendor.strip().lower(), vendor.strip())
        vm_signals.append(
            "lscpu raportuje dostawcę hypervisora: " f"{label}."
        )

    virtualization = info.get("virtualization_type")
    if virtualization:
        normalized = virtualization.strip().lower()
        if normalized in {"container", "lxc", "podman", "docker"}:
            container_signals.append(
                "lscpu wskazuje na uruchomienie w kontenerze "
                f"(virtualization type={virtualization})."
            )
        elif normalized not in {"none", "host"}:
            vm_signals.append(
                "lscpu wskazuje typ wirtualizacji: "
                f"{virtualization}."
            )

    if vm_signals:
        vm_signals = list(dict.fromkeys(vm_signals))
    if container_signals:
        container_signals = list(dict.fromkeys(container_signals))

    return (vm_signals, container_signals)


def _probe_virt_what() -> tuple[str, ...]:
    output = _run_command(["virt-what"])
    if not output:
        return ()
    entries = [line.strip().lower() for line in output.splitlines() if line.strip()]
    if not entries:
        return ()
    return tuple(dict.fromkeys(entries))


def _detect_virt_what_signals(entries: Sequence[str]) -> tuple[list[str], list[str]]:
    if not entries:
        return ([], [])

    vm_signals: list[str] = []
    container_signals: list[str] = []

    for entry in entries:
        lowered = entry.strip().lower()
        if not lowered:
            continue
        vm_label = _VIRTWHAT_VM_LABELS.get(lowered)
        container_label = _VIRTWHAT_CONTAINER_LABELS.get(lowered)
        if vm_label:
            vm_signals.append(
                "virt-what zgłasza uruchomienie w środowisku wirtualnym "
                f"({vm_label})."
            )
        elif container_label:
            container_signals.append(
                "virt-what zgłasza uruchomienie w kontenerze "
                f"({container_label})."
            )
        else:
            vm_signals.append(
                f"virt-what zwrócił nierozpoznany sygnał wirtualizacji: {lowered}."
            )

    if vm_signals:
        vm_signals = list(dict.fromkeys(vm_signals))
    if container_signals:
        container_signals = list(dict.fromkeys(container_signals))

    return (vm_signals, container_signals)


def _read_tracer_pid() -> int | None:
    status_path = Path("/proc/self/status")
    try:
        with status_path.open("r", encoding="utf-8", errors="ignore") as handle:
            for line in handle:
                if line.startswith("TracerPid"):
                    _, _, value = line.partition(":")
                    text = value.strip()
                    if not text:
                        return None
                    try:
                        number = int(text)
                    except ValueError:
                        return None
                    return number if number > 0 else None
    except (FileNotFoundError, OSError):
        return None
    return None


def _collect_kernel_modules() -> tuple[str, ...]:
    modules_path = Path("/proc/modules")
    try:
        with modules_path.open("r", encoding="utf-8", errors="ignore") as handle:
            names: list[str] = []
            for line in handle:
                parts = line.split()
                if parts and parts[0]:
                    names.append(parts[0])
    except (FileNotFoundError, OSError):
        return ()
    if not names:
        return ()
    return tuple(dict.fromkeys(names))


def _read_cgroup_lines() -> tuple[str, ...]:
    cgroup_path = Path("/proc/1/cgroup")
    try:
        with cgroup_path.open("r", encoding="utf-8", errors="ignore") as handle:
            lines = [line.strip() for line in handle if line.strip()]
    except (FileNotFoundError, OSError):
        return ()
    return tuple(lines)


def _detect_debugger_signals(
    *,
    trace: object | None,
    tracer_pid: int | None,
    env: Mapping[str, str],
) -> list[str]:
    signals: list[str] = []
    if trace is not None:
        signals.append("sys.gettrace() wskazuje na aktywny debugger.")
    if tracer_pid:
        signals.append(f"Proces jest śledzony (TracerPid={tracer_pid}).")
    for key, description in _DEBUGGER_ENV_KEYS.items():
        if env.get(key):
            signals.append(f"Zmienna środowiskowa {key} sugeruje debugger ({description}).")
    return signals


def collect_security_signals(
    *,
    env: Mapping[str, str] | None = None,
    cpu_info: str | None = None,
    cpu_flags: Sequence[str] | None = None,
    mac_addresses: Sequence[str] | None = None,
    dmi_strings: Sequence[str] | None = None,
    processes: Sequence[str] | None = None,
    kernel_modules: Sequence[str] | None = None,
    filesystem_entries: Iterable[str | Path] | None = None,
    path_exists: Callable[[Path], bool] | None = None,
    lscpu_info: object = _MISSING,
    dmidecode_info: object = _MISSING,
    wmic_info: object = _MISSING,
    cgroup_lines: Sequence[str] | None = None,
    tracer_pid: object = _MISSING,
    trace: object = _MISSING,
    hostname_virtualization: object = _MISSING,
    systemd_virtualization: object = _MISSING,
    virt_what: object = _MISSING,
) -> SecuritySignals:
    """Zbiera heurystyki bezpieczeństwa środowiska."""

    env_mapping = env if env is not None else os.environ
    info = cpu_info if cpu_info is not None else _probe_cpu_info()
    flags = tuple(cpu_flags) if cpu_flags is not None else _read_cpu_flags()
    macs = (
        tuple(mac_addresses)
        if mac_addresses is not None
        else tuple(_collect_mac_addresses(env_mapping))
    )
    dmi_values = (
        tuple(dmi_strings)
        if dmi_strings is not None
        else tuple(_collect_dmi_strings())
    )
    proc_snapshot = tuple(processes) if processes is not None else _list_process_names()
    module_snapshot = (
        tuple(kernel_modules) if kernel_modules is not None else _collect_kernel_modules()
    )
    if cgroup_lines is not None:
        cgroup_snapshot = tuple(cgroup_lines)
    else:
        cgroup_snapshot = _read_cgroup_lines()

    if filesystem_entries is not None:
        entries_set = {Path(entry) for entry in filesystem_entries}

        def _fake_exists(path: Path) -> bool:
            return path in entries_set

        path_exists_fn: Callable[[Path], bool] = _fake_exists
    elif path_exists is not None:
        path_exists_fn = path_exists
    else:
        def _default_exists(path: Path) -> bool:
            try:
                return path.exists()
            except OSError:
                return False

        path_exists_fn = _default_exists

    if lscpu_info is _MISSING:
        lscpu_mapping: Mapping[str, str] = _probe_lscpu_info()
    else:
        extracted: dict[str, str] = {}
        if lscpu_info is None:
            pass
        elif isinstance(lscpu_info, Mapping):
            for key, value in lscpu_info.items():
                if isinstance(key, str) and isinstance(value, str) and value.strip():
                    extracted[key] = value
        elif isinstance(lscpu_info, Iterable) and not isinstance(lscpu_info, (str, bytes)):
            for entry in lscpu_info:
                if (
                    isinstance(entry, tuple)
                    and len(entry) == 2
                    and isinstance(entry[0], str)
                    and isinstance(entry[1], str)
                    and entry[1].strip()
                ):
                    extracted[entry[0]] = entry[1]
        lscpu_mapping = extracted

    if dmidecode_info is _MISSING:
        dmidecode_mapping: Mapping[str, str] = _probe_dmidecode_info()
    else:
        dmidecode_extracted: dict[str, str] = {}
        if dmidecode_info is None:
            pass
        elif isinstance(dmidecode_info, Mapping):
            for key, value in dmidecode_info.items():
                if isinstance(key, str) and isinstance(value, str) and value.strip():
                    dmidecode_extracted[key] = value
        elif isinstance(dmidecode_info, Iterable) and not isinstance(dmidecode_info, (str, bytes)):
            for entry in dmidecode_info:
                if (
                    isinstance(entry, tuple)
                    and len(entry) == 2
                    and isinstance(entry[0], str)
                    and isinstance(entry[1], str)
                    and entry[1].strip()
                ):
                    dmidecode_extracted[entry[0]] = entry[1]
        dmidecode_mapping = dmidecode_extracted

    if wmic_info is _MISSING:
        wmic_mapping: Mapping[str, str] = _probe_wmic_system_info()
    else:
        wmic_extracted: dict[str, str] = {}
        if wmic_info is None:
            pass
        elif isinstance(wmic_info, Mapping):
            for key, value in wmic_info.items():
                if isinstance(key, str) and isinstance(value, str) and value.strip():
                    wmic_extracted[key] = value
        elif isinstance(wmic_info, Iterable) and not isinstance(wmic_info, (str, bytes)):
            for entry in wmic_info:
                if (
                    isinstance(entry, tuple)
                    and len(entry) == 2
                    and isinstance(entry[0], str)
                    and isinstance(entry[1], str)
                    and entry[1].strip()
                ):
                    wmic_extracted[entry[0]] = entry[1]
        else:
            LOGGER.debug("Nieobsługiwany typ wejścia wmic_info: %r", type(wmic_info))
        wmic_mapping = wmic_extracted

    if tracer_pid is _MISSING:
        tracer_pid_value: int | None = _read_tracer_pid()
    else:
        tracer_pid_value = cast(Optional[int], tracer_pid)

    if trace is _MISSING:
        trace_value: object | None = sys.gettrace()
    else:
        trace_value = cast(Optional[object], trace)

    if hostname_virtualization is _MISSING:
        hostname_value = _probe_hostnamectl_virtualization()
    else:
        hostname_value = cast(Optional[str], hostname_virtualization)

    if systemd_virtualization is _MISSING:
        systemd_value = _probe_systemd_virtualization()
    else:
        systemd_value = cast(Optional[str], systemd_virtualization)

    if virt_what is _MISSING:
        virt_what_entries = _probe_virt_what()
    else:
        if virt_what is None:
            virt_what_entries = ()
        elif isinstance(virt_what, str):
            virt_what_entries = (virt_what,)
        else:
            virt_what_entries = tuple(virt_what)

    cpu_signals = _detect_vm_cpu_signals(info, flags)
    mac_signals = _detect_vm_mac_signals(macs)
    process_signals = _detect_vm_process_signals(proc_snapshot)
    dmi_signals = _detect_vm_dmi_signals(dmi_values)
    dmidecode_signals = _detect_dmidecode_signals(dmidecode_mapping)
    wmic_signals = _detect_wmic_signals(wmic_mapping)
    filesystem_signals = _detect_vm_filesystem_signals(path_exists=path_exists_fn)
    module_signals = _detect_vm_kernel_module_signals(module_snapshot)
    container_env_signals = _detect_container_env_signals(env_mapping)
    container_filesystem_signals = _detect_container_filesystem_signals(
        path_exists=path_exists_fn
    )
    container_cgroup_signals = _detect_container_cgroup_signals(cgroup_snapshot)
    hostname_vm_signals, hostname_container_signals = (
        _detect_hostnamectl_virtualization_signals(hostname_value)
    )
    systemd_vm_signals, systemd_container_signals = _detect_systemd_virtualization_signals(
        systemd_value
    )
    lscpu_vm_signals, lscpu_container_signals = _detect_lscpu_signals(lscpu_mapping)
    virt_what_vm_signals, virt_what_container_signals = _detect_virt_what_signals(
        virt_what_entries
    )
    debugger_signals = _detect_debugger_signals(
        trace=trace_value,
        tracer_pid=tracer_pid_value,
        env=env_mapping,
    )

    container_signals = list(
        dict.fromkeys(
            (
                *container_env_signals,
                *container_filesystem_signals,
                *container_cgroup_signals,
                *hostname_container_signals,
                *systemd_container_signals,
                *lscpu_container_signals,
                *virt_what_container_signals,
            )
        )
    )

    hostname_bucket = tuple(
        dict.fromkeys((*hostname_vm_signals, *hostname_container_signals))
    )

    systemd_bucket = tuple(
        dict.fromkeys((*systemd_vm_signals, *systemd_container_signals))
    )

    virt_what_bucket = tuple(
        dict.fromkeys((*virt_what_vm_signals, *virt_what_container_signals))
    )

    lscpu_bucket = tuple(
        dict.fromkeys((*lscpu_vm_signals, *lscpu_container_signals))
    )

    vm_map = {
        "cpu": tuple(cpu_signals),
        "mac_vendor": tuple(mac_signals),
        "process": tuple(process_signals),
        "dmi": tuple(dmi_signals),
        "dmidecode": tuple(dmidecode_signals),
        "wmic": tuple(wmic_signals),
        "filesystem": tuple(filesystem_signals),
        "kernel_module": tuple(module_signals),
        "lscpu": lscpu_bucket,
        "hostnamectl": hostname_bucket,
        "systemd": systemd_bucket,
        "container": tuple(container_signals),
        "virt_what": virt_what_bucket,
    }

    signals = SecuritySignals(
        vm_indicators=MappingProxyType(dict(vm_map)),
        debugger_indicators=tuple(debugger_signals),
    )
    return signals


# ---------------------------------------------------------------------------
# Lokalne podpisy licencji (ochrona przed rollbackiem)
# ---------------------------------------------------------------------------


def _normalize_binding_fingerprint(value: str | None) -> str:
    if value is None:
        raise FingerprintError("Fingerprint urządzenia jest wymagany do podpisania licencji.")
    normalized = value.strip().upper()
    if not normalized:
        raise FingerprintError("Fingerprint urządzenia jest pusty – nie można podpisać licencji.")
    return normalized


def _license_secret_path(path: str | os.PathLike[str] | None) -> Path:
    return Path(path).expanduser() if path is not None else LICENSE_SECRET_PATH


def _current_hwid_digest(fingerprint: str) -> str:
    normalized = _normalize_binding_fingerprint(fingerprint)
    return crypto_current_hwid_digest(normalized)


def _derive_encryption_key(fingerprint: str, salt: bytes) -> bytes:
    normalized = _normalize_binding_fingerprint(fingerprint)
    return crypto_derive_encryption_key(normalized, salt)


def _encrypt_license_secret(secret: bytes, fingerprint: str) -> Mapping[str, object]:
    normalized = _normalize_binding_fingerprint(fingerprint)
    return crypto_encrypt_license_secret(
        secret,
        normalized,
        file_version=LICENSE_SECRET_FILE_VERSION,
    )


def _decrypt_license_secret(document: Mapping[str, object]) -> bytes:
    try:
        fingerprint = get_local_fingerprint()
    except Exception as exc:  # pragma: no cover - propagujemy w postaci FingerprintError
        raise FingerprintError("Nie udało się pobrać lokalnego fingerprintu do odszyfrowania sekretu.") from exc

    normalized = _normalize_binding_fingerprint(fingerprint)
    try:
        secret = crypto_decrypt_license_secret(
            document,
            normalized,
            file_version=LICENSE_SECRET_FILE_VERSION,
        )
    except ValueError as exc:
        raise FingerprintError(str(exc)) from exc
    except Exception as exc:  # pragma: no cover - błędne dane szyfru
        raise FingerprintError("Nie udało się odszyfrować sekretu licencji.") from exc

    if len(secret) < 32:
        raise FingerprintError("Sekret licencji ma niepoprawną długość po odszyfrowaniu.")

    return secret


def _load_secret_from_keyring() -> bytes | None:
    try:  # pragma: no cover - zależne od środowiska testowego
        import keyring  # type: ignore[import-not-found]
    except ImportError:
        return None

    try:
        record = keyring.get_password(LICENSE_SECRET_KEYRING_SERVICE, LICENSE_SECRET_KEYRING_ENTRY)
    except Exception as exc:  # pragma: no cover - logujemy i wracamy do pliku
        LOGGER.warning("Nie udało się pobrać sekretu licencji z keychaina: %s", exc)
        return None

    if record is None:
        return None

    document: Mapping[str, object]
    try:
        document = json.loads(record)
    except json.JSONDecodeError:
        try:
            decoded = base64.b64decode(record.encode("ascii"))
        except Exception as exc:  # pragma: no cover - odziedziczone formaty
            raise FingerprintError("Sekret licencji w keychainie ma niepoprawny format.") from exc
        if len(decoded) < 32:
            raise FingerprintError("Sekret licencji w keychainie ma niepoprawną długość.")
        return decoded

    secret_b64 = document.get("secret_b64")
    if not isinstance(secret_b64, str) or not secret_b64:
        raise FingerprintError("Rekord keychaina nie zawiera poprawnych danych sekretu.")
    try:
        secret = base64.b64decode(secret_b64.encode("ascii"))
    except Exception as exc:  # pragma: no cover - dane mogą być uszkodzone
        raise FingerprintError("Sekret licencji w keychainie jest uszkodzony (base64).") from exc
    if len(secret) < 32:
        raise FingerprintError("Sekret licencji w keychainie ma niepoprawną długość.")
    return secret


def _store_secret_in_keyring(secret: bytes) -> bool:
    try:  # pragma: no cover - zależne od środowiska testowego
        import keyring  # type: ignore[import-not-found]
    except ImportError:
        return False

    payload = {
        "version": 1,
        "secret_b64": base64.b64encode(secret).decode("ascii"),
        "stored_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
    }
    try:
        result = keyring.set_password(
            LICENSE_SECRET_KEYRING_SERVICE,
            LICENSE_SECRET_KEYRING_ENTRY,
            json.dumps(payload, ensure_ascii=False),
        )
    except Exception as exc:  # pragma: no cover - logujemy ostrzeżenie i kontynuujemy
        LOGGER.warning("Nie udało się zapisać sekretu licencji w keychainie: %s", exc)
        return False

    if result is not None:  # pragma: no cover - backend powinien zwrócić None
        LOGGER.warning("Backend keychain zwrócił nieoczekiwany rezultat przy zapisie sekretu licencji: %r", result)
        return False
    return True


def _load_secret_from_disk(path: Path) -> tuple[bytes, str] | None:
    if not path.exists():
        return None
    try:
        raw = path.read_text(encoding="utf-8").strip()
    except OSError as exc:
        raise FingerprintError(f"Nie udało się odczytać pliku sekretu licencji ({path}): {exc}") from exc

    if not raw:
        path.unlink(missing_ok=True)
        return None

    try:
        document = json.loads(raw)
    except json.JSONDecodeError:
        try:
            secret = base64.b64decode(raw.encode("ascii"))
        except Exception as exc:
            raise FingerprintError("Sekret licencji ma niepoprawny format (legacy).") from exc
        if len(secret) < 32:
            raise FingerprintError("Sekret licencji ma niepoprawną długość (legacy).")
        return secret, "legacy"

    if not isinstance(document, Mapping):
        raise FingerprintError("Zaszyfrowany sekret licencji ma niepoprawną strukturę.")
    secret = _decrypt_license_secret(document)
    return secret, "encrypted"


def _write_encrypted_secret(secret: bytes, path: Path) -> None:
    try:
        fingerprint = get_local_fingerprint()
    except Exception as exc:  # pragma: no cover - propagujemy w postaci FingerprintError
        raise FingerprintError("Nie udało się pobrać fingerprintu urządzenia do zapisania sekretu licencji.") from exc

    payload = _encrypt_license_secret(secret, fingerprint)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(tmp_path, path)
    try:
        os.chmod(path, 0o600)
    except PermissionError:  # pragma: no cover - brak uprawnień na niektórych systemach
        pass


def load_license_secret(path: str | os.PathLike[str] | None = None, *, create: bool = True) -> bytes:
    """Zwraca sekret do podpisywania snapshotów licencji."""

    target = _license_secret_path(path)
    secret = _load_secret_from_keyring()
    if secret is not None:
        _write_encrypted_secret(secret, target)
        return secret

    disk_result = _load_secret_from_disk(target)
    if disk_result is not None:
        secret, source = disk_result
        _store_secret_in_keyring(secret)
        if source != "encrypted":
            _write_encrypted_secret(secret, target)
        return secret

    if not create:
        raise FingerprintError("Sekret licencji nie istnieje.")

    secret = os.urandom(48)
    _store_secret_in_keyring(secret)
    _write_encrypted_secret(secret, target)
    return secret


def _derive_license_key(fingerprint: str, secret: bytes) -> bytes:
    normalized = _normalize_binding_fingerprint(fingerprint)
    return hmac.new(secret, normalized.encode("utf-8"), hashlib.sha384).digest()


def sign_license_payload(
    payload: Mapping[str, object],
    *,
    fingerprint: str,
    secret: bytes | None = None,
    secret_path: str | os.PathLike[str] | None = None,
    key_id: str | None = None,
) -> Mapping[str, str]:
    """Podpisuje ładunek licencji wykorzystując lokalny fingerprint."""

    secret_bytes = secret if secret is not None else load_license_secret(secret_path)
    key = _derive_license_key(fingerprint, secret_bytes)
    identifier = key_id or "local-hwid"
    return build_hmac_signature(
        payload,
        key=key,
        algorithm=LICENSE_SIGNATURE_ALGORITHM,
        key_id=identifier,
    )


def verify_license_payload_signature(
    payload: Mapping[str, object],
    signature: Mapping[str, object] | None,
    *,
    fingerprint: str,
    secret: bytes | None = None,
    secret_path: str | os.PathLike[str] | None = None,
) -> bool:
    """Weryfikuje podpis ładunku licencji opartego o fingerprint."""

    if signature is None:
        return False
    algorithm = signature.get("algorithm")
    if algorithm != LICENSE_SIGNATURE_ALGORITHM:
        return False
    try:
        secret_bytes = secret if secret is not None else load_license_secret(secret_path, create=False)
    except FingerprintError:
        return False
    key = _derive_license_key(fingerprint, secret_bytes)
    expected = build_hmac_signature(
        payload,
        key=key,
        algorithm=LICENSE_SIGNATURE_ALGORITHM,
        key_id=signature.get("key_id"),
    )
    return expected.get("value") == signature.get("value")


# ---------------------------------------------------------------------------
# Audyt fingerprintu / instalacji
# ---------------------------------------------------------------------------

def append_fingerprint_audit(
    *,
    event: str,
    fingerprint: str | None,
    status: str,
    key_id: str | None = None,
    metadata: Mapping[str, Any] | None = None,
    log_path: str | os.PathLike[str] | Path | None = None,
) -> Path:
    """Dodaje wpis JSONL do dziennika bezpieczeństwa instalatora.

    Parametry
    ---------
    event:
        Nazwa zdarzenia (np. ``"installer_run"``).
    fingerprint:
        Fingerprint urządzenia – jeżeli nieznany, można przekazać pusty string.
    status:
        Status zdarzenia (np. ``"verified"``, ``"failed"``).
    key_id:
        Identyfikator klucza HMAC użytego do weryfikacji podpisu.
    metadata:
        Dodatkowe informacje diagnostyczne (np. ścieżka do oczekiwanego pliku,
        komunikat błędu).
    log_path:
        Niestandardowa ścieżka logu – domyślnie ``logs/security_admin.log``.
    """

    destination = Path(log_path).expanduser() if log_path else DEFAULT_AUDIT_LOG_PATH
    entry: dict[str, Any] = {
        "timestamp": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "event": event,
        "fingerprint": (fingerprint or "").strip(),
        "status": status,
    }
    if key_id:
        entry["key_id"] = key_id
    if metadata:
        entry["metadata"] = dict(metadata)

    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(entry, ensure_ascii=False) + "\n")
    return destination


# ---------------------------------------------------------------------------
# Raportowanie incydentów UI (blokada instancji)
# ---------------------------------------------------------------------------


def report_single_instance_event(
    *,
    lock_path: str | os.PathLike[str],
    owner_pid: int | None = None,
    owner_host: str | None = None,
    owner_application: str | None = None,
    fingerprint: str | None = None,
    log_path: str | os.PathLike[str] | Path | None = None,
) -> Path:
    """Raportuje konflikt wielu instancji UI do logu licencjonowania."""

    lock_candidate = Path(lock_path).expanduser()
    try:
        canonical_lock = lock_candidate.resolve()
    except OSError:
        canonical_lock = lock_candidate

    metadata: dict[str, Any] = {"lock_path": str(canonical_lock)}
    if owner_pid:
        metadata["owner_pid"] = int(owner_pid)
    host = (owner_host or "").strip()
    if host:
        metadata["owner_host"] = host
    application = (owner_application or "").strip()
    if application:
        metadata["owner_application"] = application

    normalized_fp: str | None = None
    candidate = (fingerprint or "").strip()
    if candidate:
        try:
            normalized_fp = _normalize_binding_fingerprint(candidate)
        except Exception:  # pragma: no cover - defensywne
            normalized_fp = candidate.strip().upper()
    else:
        try:
            normalized_fp = _normalize_binding_fingerprint(get_local_fingerprint())
        except FingerprintError as exc:
            LOGGER.warning(
                "Nie udało się pobrać lokalnego fingerprintu dla raportu konfliktu instancji.",
                exc_info=exc,
            )
            normalized_fp = None
        except Exception:  # pragma: no cover - defensywne
            LOGGER.exception(
                "Niespodziewany błąd podczas normalizacji fingerprintu konfliktu instancji.",
            )
            normalized_fp = None

    return append_fingerprint_audit(
        event="ui_single_instance_conflict",
        fingerprint=normalized_fp,
        status="denied",
        metadata=metadata,
        log_path=log_path,
    )


# ---------------------------------------------------------------------------
# Starsze API – OEM Device Fingerprint
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class FingerprintDocument:
    """Dokument OEM fingerprintu z podpisem HMAC."""

    payload: Mapping[str, object]
    signature: Mapping[str, object]

    def to_json(self) -> str:
        return json.dumps(
            {"payload": self.payload, "signature": self.signature},
            indent=2,
            ensure_ascii=False,
            sort_keys=True,
        )


def _now_iso(timestamp: Optional[datetime] = None) -> str:
    value = timestamp or datetime.now(timezone.utc)
    value = value.astimezone(timezone.utc).replace(microsecond=0)
    return value.isoformat().replace("+00:00", "Z")


def _read_first_line(path: Path) -> Optional[str]:
    try:
        with path.open("r", encoding="utf-8", errors="ignore") as handle:
            line = handle.readline().strip()
    except (FileNotFoundError, OSError):
        return None
    return line or None


def _run_command(command: list[str]) -> Optional[str]:
    try:
        completed = subprocess.run(
            command,
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            encoding="utf-8",
        )
    except (FileNotFoundError, OSError):
        return None
    output = completed.stdout.strip()
    return output or None


def _sanitize(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    candidate = value.strip()
    return candidate or None


def _probe_cpu_identifier(env: Mapping[str, str]) -> Optional[str]:
    override = _sanitize(env.get("OEM_CPU_ID"))
    if override:
        return override

    serial = _read_first_line(Path("/sys/devices/virtual/dmi/id/product_uuid"))
    if serial:
        return serial

    cpuinfo_path = Path("/proc/cpuinfo")
    try:
        for line in cpuinfo_path.read_text(encoding="utf-8", errors="ignore").splitlines():
            if ":" not in line:
                continue
            key, raw = line.split(":", 1)
            if key.strip().lower() in {"serial", "hardware", "processor"}:
                candidate = _sanitize(raw)
                if candidate:
                    return candidate
    except FileNotFoundError:
        pass

    mac = _sanitize(f"{uuid.getnode():012x}")
    if mac:
        return mac

    uname = platform.uname()
    fallback = "-".join(filter(None, [uname.system, uname.node, uname.machine]))
    return fallback or None


def _probe_tpm_identifier(env: Mapping[str, str]) -> Optional[str]:
    override = _sanitize(env.get("OEM_TPM_ID"))
    if override:
        return override

    candidate = _read_first_line(Path("/sys/class/tpm/tpm0/device/unique_id"))
    if candidate:
        return candidate

    candidate = _read_first_line(Path("/sys/class/tpm/tpm0/unique_id"))
    if candidate:
        return candidate

    candidate = _run_command(["tpm2_getcap", "-c", "properties-fixed"])
    if candidate:
        return hashlib.sha256(candidate.encode("utf-8")).hexdigest()

    return None


def _probe_dongle_identifier(env: Mapping[str, str]) -> Optional[str]:
    override = _sanitize(env.get("OEM_DONGLE_ID"))
    if override:
        return override

    default_path = Path(env.get("OEM_DONGLE_PATH", "var/oem/dongle_id.txt"))
    return _read_first_line(default_path)


def _collect_mac_addresses(env: Mapping[str, str]) -> list[str]:
    override = _sanitize(env.get("OEM_MAC_ADDRESSES"))
    if override:
        addresses = {
            candidate.replace(":", "").replace("-", "").lower()
            for candidate in (_sanitize(entry) for entry in override.split(","))
            if candidate
        }
        return sorted(addresses)

    addresses: set[str] = set()
    try:
        node = uuid.getnode()
    except ValueError:
        node = None
    if node is not None:
        addresses.add(f"{node:012x}")

    sys_class = Path("/sys/class/net")
    if sys_class.exists():
        for interface in sys_class.iterdir():
            address_path = interface / "address"
            candidate = _read_first_line(address_path)
            if candidate and candidate != "00:00:00:00:00:00":
                addresses.add(candidate.replace(":", "").lower())

    if not addresses:
        try:
            hostname = socket.gethostname().encode("utf-8")
        except OSError:
            hostname = b""
        if hostname:
            addresses.add(hashlib.sha256(hostname).hexdigest()[:12])

    return sorted(addresses)


def _collect_dmi_strings() -> tuple[str, ...]:
    base = Path("/sys/devices/virtual/dmi/id")
    if not base.exists():
        return ()

    candidates = (
        base / "product_name",
        base / "sys_vendor",
        base / "board_vendor",
        base / "bios_vendor",
        base / "bios_version",
        base / "product_version",
    )

    values: list[str] = []
    for path in candidates:
        candidate = _read_first_line(path)
        if candidate:
            values.append(candidate)

    if not values:
        return ()

    # usuń duplikaty przy zachowaniu kolejności oraz pomiń bardzo krótkie wpisy
    unique = [value for value in dict.fromkeys(values) if len(value.strip()) > 2]
    return tuple(unique)


def _collect_hostname(env: Mapping[str, str]) -> Optional[str]:
    override = _sanitize(env.get("OEM_HOSTNAME"))
    if override:
        return override
    return _sanitize(platform.node())


def _collect_os_info() -> dict[str, str]:
    return {
        "system": platform.system(),
        "release": platform.release(),
        "version": platform.version(),
        "machine": platform.machine(),
    }


def _normalize_factors(factors: Mapping[str, object]) -> dict[str, object]:
    normalized: dict[str, object] = {}
    for key, value in sorted(factors.items()):
        if value is None:
            continue
        if isinstance(value, Mapping):
            normalized[key] = _normalize_factors(value)
        elif isinstance(value, (list, tuple, set)):
            normalized[key] = [str(item) for item in sorted(value)]
        else:
            normalized[key] = str(value)
    return normalized


def _ensure_key_strength(key: bytes) -> None:
    if len(key) < 32:
        raise FingerprintError("Klucz podpisu fingerprintu musi mieć co najmniej 32 bajty")


def _sign_payload(payload: Mapping[str, object], key: bytes, key_id: Optional[str]) -> dict[str, str]:
    digest = hmac.new(key, canonical_json_bytes(payload), hashlib.sha384).digest()
    signature = {
        "algorithm": SIGNATURE_ALGORITHM,
        "value": base64.b64encode(digest).decode("ascii"),
    }
    if key_id:
        signature["key_id"] = str(key_id)
    return signature


class DeviceFingerprintGenerator:
    """Generator fingerprintu bazujący na metadanych sprzętowych (OEM API)."""

    def __init__(
        self,
        *,
        env: Optional[Mapping[str, str]] = None,
        extra_probes: Optional[Iterable[Callable[[MutableMapping[str, object]], None]]] = None,
    ) -> None:
        self._env = env or os.environ
        self._extra_probes = tuple(extra_probes or ())

    def collect_factors(self) -> dict[str, object]:
        factors: MutableMapping[str, object] = {}

        cpu_id = _probe_cpu_identifier(self._env)
        if cpu_id:
            factors["cpu_id"] = cpu_id

        mac_addresses = _collect_mac_addresses(self._env)
        if mac_addresses:
            factors["mac_addresses"] = mac_addresses

        hostname = _collect_hostname(self._env)
        if hostname:
            factors["hostname"] = hostname

        os_info = _collect_os_info()
        factors["os"] = os_info

        tpm = _probe_tpm_identifier(self._env)
        if tpm:
            factors["tpm"] = tpm

        dongle = _probe_dongle_identifier(self._env)
        if dongle:
            factors["dongle"] = dongle

        salt = _sanitize(self._env.get("OEM_FINGERPRINT_SALT"))
        if salt:
            factors["salt"] = salt

        for probe in self._extra_probes:
            probe(factors)

        return dict(factors)

    def generate_fingerprint(self, *, factors: Optional[Mapping[str, object]] = None) -> str:
        factors = factors or self.collect_factors()
        canonical = canonical_json_bytes(_normalize_factors(factors))
        digest = hashlib.sha384(canonical).digest()
        token = base64.b32encode(digest).decode("ascii").rstrip("=").upper()
        grouped = [token[i : i + 8] for i in range(0, len(token), 8)]
        return "-".join(grouped)

    def build_document(
        self,
        *,
        signing_key: bytes,
        key_id: Optional[str] = None,
        factors: Optional[Mapping[str, object]] = None,
        registry: Optional[RotationRegistry] = None,
        purpose: str = _OEM_DEFAULT_PURPOSE,
        rotation_interval_days: float = 90.0,
        mark_rotation: bool = False,
        created_at: Optional[datetime] = None,
    ) -> FingerprintDocument:
        _ensure_key_strength(signing_key)
        collected = _normalize_factors(factors or self.collect_factors())
        fingerprint = self.generate_fingerprint(factors=collected)
        timestamp = datetime.now(timezone.utc) if created_at is None else created_at.astimezone(timezone.utc)

        if registry and key_id:
            status = registry.status(key_id, purpose, interval_days=rotation_interval_days, now=timestamp)
            if status.last_rotated is not None and status.is_overdue:
                raise FingerprintError(
                    f"Klucz '{key_id}' dla celu '{purpose}' jest przeterminowany (ostatnia rotacja {status.last_rotated})."
                )

        payload = {
            "fingerprint": fingerprint,
            "algorithm": FINGERPRINT_HASH,
            "created_at": _now_iso(timestamp),
            "factors": collected,
        }
        signature = _sign_payload(payload, signing_key, key_id)

        if registry and key_id and mark_rotation:
            registry.mark_rotated(key_id, purpose, timestamp=timestamp)

        return FingerprintDocument(payload=payload, signature=signature)


def verify_document(document: Mapping[str, object], *, key: bytes) -> bool:
    payload = document.get("payload")
    signature = document.get("signature")
    if not isinstance(payload, Mapping) or not isinstance(signature, Mapping):
        raise FingerprintError("Nieprawidłowa struktura dokumentu fingerprintu")
    key_id_val = signature.get("key_id")
    key_id_str = key_id_val if isinstance(key_id_val, str) else None
    expected = _sign_payload(payload, key, key_id_str)
    return expected["value"] == signature.get("value")


def get_local_fingerprint() -> str:
    generator = DeviceFingerprintGenerator()
    return generator.generate_fingerprint()


def build_fingerprint_document(
    *,
    signing_key: bytes,
    key_id: Optional[str] = None,
    env: Optional[Mapping[str, str]] = None,
    registry: Optional[RotationRegistry] = None,
    purpose: str = _OEM_DEFAULT_PURPOSE,
    rotation_interval_days: float = 90.0,
    mark_rotation: bool = False,
    created_at: Optional[datetime] = None,
) -> FingerprintDocument:
    """Funkcja pomocnicza budująca podpisany dokument fingerprintu OEM."""
    generator = DeviceFingerprintGenerator(env=env)
    return generator.build_document(
        signing_key=signing_key,
        key_id=key_id,
        registry=registry,
        purpose=purpose,
        rotation_interval_days=rotation_interval_days,
        mark_rotation=mark_rotation,
        created_at=created_at,
    )


# ---------------------------------------------------------------------------
# Nowsze API – Hardware Fingerprint Service (z rotacją kluczy)
# ---------------------------------------------------------------------------

def _ensure_utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _normalize_text(value: str) -> str:
    collapsed = re.sub(r"\s+", " ", value.strip().lower())
    return collapsed


def _component_entry(value: str | None) -> dict[str, str] | None:
    if value is None:
        return None
    raw = value.strip()
    if not raw:
        return None
    normalized = _normalize_text(raw)
    digest = hashlib.sha256(normalized.encode("utf-8")).hexdigest()
    return {"raw": raw, "normalized": normalized, "digest": digest}


def _read_first_existing(paths: Sequence[Path]) -> str | None:
    for candidate in paths:
        try:
            data = candidate.read_text(encoding="utf-8")
        except (FileNotFoundError, OSError):
            continue
        content = data.strip()
        if content:
            return content
    return None


def _probe_cpu_info() -> str | None:
    entries: list[str] = []
    uname = platform.uname()
    entries.extend(filter(None, {uname.system, uname.machine, uname.processor}))

    processor = platform.processor()
    if processor and processor not in entries:
        entries.append(processor)

    try:
        with Path("/proc/cpuinfo").open("r", encoding="utf-8") as handle:
            for line in handle:
                if line.lower().startswith("model name"):
                    _, _, value = line.partition(":")
                    entries.append(value.strip())
                    break
    except (FileNotFoundError, OSError):
        pass

    if not entries:
        return None
    # usuń duplikaty z zachowaniem kolejności
    return " | ".join(dict.fromkeys(entries))


def _probe_tpm_info() -> str | None:
    system = platform.system().lower()
    if system == "windows":
        try:
            completed = subprocess.run(
                ["wmic", "tpm", "get", "ManufacturerId", ",", "ManufacturerVersion"],
                capture_output=True,
                text=True,
                check=False,
            )
        except OSError:
            return None
        output = (completed.stdout or "") + (completed.stderr or "")
        tokens = [token.strip() for token in output.splitlines() if token.strip()]
        if tokens:
            return " | ".join(tokens)
        return None

    if system == "darwin":
        return None

    base = Path("/sys/class/tpm")
    if not base.exists():
        return None

    candidates: list[Path] = []
    for entry in sorted(base.iterdir()):
        if not entry.name.startswith("tpm"):
            continue
        candidates.extend(
            [
                entry / "device" / "description",
                entry / "device" / "manufacturer",
                entry / "device" / "modalias",
            ]
        )

    if not candidates:
        return None

    content = _read_first_existing(candidates)
    if content:
        return content
    return None


def _format_mac_address(value: int) -> str:
    hex_value = f"{value:012x}"
    parts = [hex_value[i : i + 2] for i in range(0, len(hex_value), 2)]
    return ":".join(parts)


def _probe_mac_address() -> str | None:
    node = uuid.getnode()
    if node is None:
        return None
    return _format_mac_address(node)


def _probe_dongle() -> str | None:
    raw = os.environ.get("DUDZIAN_DONGLE_ID")
    if raw is None:
        return None
    raw = raw.strip()
    return raw or None


@dataclass(slots=True)
class FingerprintRecord:
    """Wynik podpisanego fingerprintu (nowe API)."""

    payload: Mapping[str, Any]
    signature: Mapping[str, str]
    key_id: str

    def as_dict(self) -> dict[str, Any]:
        return {"payload": dict(self.payload), "signature": dict(self.signature)}


class RotatingHmacKeyProvider:
    """Zarządza kluczami HMAC w oparciu o ``RotationRegistry``."""

    def __init__(
        self,
        keys: Mapping[str, bytes | str],
        registry: RotationRegistry,
        *,
        purpose: str = _DEFAULT_PURPOSE,
        interval_days: float = 90.0,
    ) -> None:
        if not keys:
            raise ValueError("Wymagany jest co najmniej jeden klucz HMAC.")
        normalized: dict[str, bytes] = {}
        for key_id, raw_value in keys.items():
            if not key_id:
                raise ValueError("Identyfikator klucza HMAC nie może być pusty.")
            if isinstance(raw_value, str):
                value = raw_value.encode("utf-8")
            else:
                value = bytes(raw_value)
            normalized[str(key_id)] = value

        self._keys = normalized
        self._registry = registry
        self._purpose = purpose
        self._interval = interval_days

    @property
    def purpose(self) -> str:
        return self._purpose

    def key_ids(self) -> tuple[str, ...]:
        return tuple(self._keys)

    def status_for(self, key_id: str, *, now: datetime | None = None) -> RotationStatus:
        return self._registry.status(
            key_id,
            self._purpose,
            interval_days=self._interval,
            now=now,
        )

    def _priority_tuple(self, status: RotationStatus) -> tuple[int, float]:
        if not status.is_due and not status.is_overdue:
            priority = 0
        elif status.is_due and not status.is_overdue:
            priority = 1
        else:
            priority = 2
        if status.last_rotated is None:
            timestamp = float("inf")
        else:
            timestamp = -status.last_rotated.timestamp()
        return priority, timestamp

    def select_active_key(self, *, now: datetime | None = None) -> tuple[str, bytes, RotationStatus]:
        statuses: dict[str, RotationStatus] = {kid: self.status_for(kid, now=now) for kid in self._keys}
        sorted_ids = sorted(statuses.items(), key=lambda item: (self._priority_tuple(item[1]), item[0]))
        active_id, status = sorted_ids[0]
        return active_id, self._keys[active_id], status

    def sign(self, payload: Mapping[str, Any], *, now: datetime | None = None) -> tuple[str, dict[str, str]]:
        key_id, key, _status = self.select_active_key(now=now)
        signature = build_hmac_signature(payload, key=key, key_id=key_id, algorithm=SIGNATURE_ALGORITHM)
        return key_id, signature


class HardwareFingerprintService:
    """Buduje deterministyczny fingerprint sprzętowy podpisany HMAC (nowe API)."""

    def __init__(
        self,
        key_provider: RotatingHmacKeyProvider,
        *,
        cpu_probe: Callable[[], str | None] | None = None,
        tpm_probe: Callable[[], str | None] | None = None,
        mac_probe: Callable[[], str | None] | None = None,
        dongle_probe: Callable[[], str | None] | None = None,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        self._provider = key_provider
        self._cpu_probe = cpu_probe or _probe_cpu_info
        self._tpm_probe = tpm_probe or _probe_tpm_info
        self._mac_probe = mac_probe or _probe_mac_address
        self._dongle_probe = dongle_probe or _probe_dongle
        self._clock = clock or _ensure_utc_now

    def _collect_components(self, dongle_serial: str | None) -> dict[str, Any]:
        cpu_component = _component_entry(self._cpu_probe())
        tpm_component = _component_entry(self._tpm_probe())
        mac_raw = self._mac_probe()
        if mac_raw:
            mac_clean = _HEX_RE.sub("", mac_raw.lower())
            mac_component = _component_entry(mac_clean)
            if mac_component:
                mac_component["raw"] = mac_raw.strip()
        else:
            mac_component = None

        if dongle_serial is None:
            dongle_serial = self._dongle_probe()
        dongle_component = _component_entry(dongle_serial)

        return {
            "cpu": cpu_component,
            "tpm": tpm_component,
            "mac": mac_component,
            "dongle": dongle_component,
        }

    def build(self, *, dongle_serial: str | None = None) -> FingerprintRecord:
        components = self._collect_components(dongle_serial)

        digest_material: MutableMapping[str, str] = {}
        for key, entry in components.items():
            if entry is None:
                continue
            digest_material[key] = entry["digest"]

        payload_base = {
            "version": 1,
            "components": digest_material,
        }
        fingerprint_digest = hashlib.sha256(canonical_json_bytes(payload_base)).hexdigest()

        current_time = self._clock().replace(microsecond=0)
        collected_at = current_time.isoformat().replace("+00:00", "Z")

        payload = {
            "version": 1,
            "collected_at": collected_at,
            "components": components,
            "component_digests": digest_material,
            "fingerprint": {
                "algorithm": "sha256",
                "value": fingerprint_digest,
            },
        }

        key_id, signature = self._provider.sign(payload, now=current_time)
        return FingerprintRecord(payload=payload, signature=signature, key_id=key_id)


# ---------------------------------------------------------------------------
# Helpery/CLI (nowe API)
# ---------------------------------------------------------------------------

def decode_secret(value: str) -> bytes:
    text = value.strip()
    if text.startswith("hex:"):
        return bytes.fromhex(text[4:])
    if text.startswith("base64:"):
        return base64.b64decode(text[7:])
    return text.encode("utf-8")


def build_key_provider(
    keys: Mapping[str, str | bytes],
    rotation_log: str | Path,
    *,
    purpose: str = _DEFAULT_PURPOSE,
    interval_days: float = 90.0,
) -> RotatingHmacKeyProvider:
    registry = RotationRegistry(rotation_log)
    return RotatingHmacKeyProvider(keys, registry, purpose=purpose, interval_days=interval_days)


def _parse_key_argument(text: str) -> tuple[str, bytes]:
    if "=" not in text:
        raise ValueError("Parametr klucza musi mieć format key_id=wartość")
    key_id, value = text.split("=", 1)
    return key_id.strip(), decode_secret(value)


def _load_keys_from_args(args: list[str]) -> dict[str, bytes]:
    result: dict[str, bytes] = {}
    for entry in args:
        key_id, secret = _parse_key_argument(entry)
        result[key_id] = secret
    return result


def _cli_generate(argv: list[str]) -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Generuje odcisk sprzętowy hosta i podpisuje go HMAC.")
    parser.add_argument("--dongle", help="Wymuszony identyfikator klucza sprzętowego USB.")
    parser.add_argument(
        "--rotation-log",
        default="var/licenses/fingerprint_rotation.json",
        help="Ścieżka do rejestru rotacji kluczy HMAC.",
    )
    parser.add_argument(
        "--key",
        action="append",
        dest="keys",
        help="Klucz HMAC w formacie key_id=sekret (można podać wielokrotnie).",
    )
    parser.add_argument(
        "--output",
        choices=["json"],
        default="json",
        help="Format wyjścia (domyślnie JSON).",
    )
    parser.add_argument(
        "--purpose",
        default=_DEFAULT_PURPOSE,
        help="Cel wpisów w rejestrze rotacji kluczy.",
    )
    parser.add_argument(
        "--interval-days",
        type=float,
        default=90.0,
        help="Oczekiwany interwał rotacji kluczy HMAC.",
    )

    parsed = parser.parse_args(argv)

    keys = _load_keys_from_args(parsed.keys or [])
    if not keys:
        parser.error("Wymagany jest co najmniej jeden klucz --key do podpisania fingerprintu.")

    provider = build_key_provider(
        keys,
        parsed.rotation_log,
        purpose=parsed.purpose,
        interval_days=parsed.interval_days,
    )
    service = HardwareFingerprintService(provider)
    record = service.build(dongle_serial=parsed.dongle)

    if parsed.output == "json":
        json.dump(record.as_dict(), sys.stdout, ensure_ascii=False)
        sys.stdout.write("\n")
    return 0


def _cli_report_single_instance(argv: list[str]) -> int:
    import argparse

    parser = argparse.ArgumentParser(
        description="Raportuje konflikt wielu instancji UI do logu licencjonowania."
    )
    parser.add_argument("--lock-path", required=True, help="Ścieżka pliku blokady instancji.")
    parser.add_argument("--owner-pid", type=int, default=0, help="PID aktywnej instancji.")
    parser.add_argument("--owner-host", help="Nazwa hosta, na którym działa aktywna instancja.")
    parser.add_argument(
        "--owner-application",
        help="Identyfikator aplikacji zapisany w pliku blokady.",
    )
    parser.add_argument("--fingerprint", help="Odcisk urządzenia do wymuszenia w raporcie.")
    parser.add_argument(
        "--log-path",
        help="Niestandardowa ścieżka logu bezpieczeństwa (domyślnie logs/security_admin.log).",
    )

    parsed = parser.parse_args(argv)

    report_single_instance_event(
        lock_path=parsed.lock_path,
        owner_pid=parsed.owner_pid or None,
        owner_host=parsed.owner_host,
        owner_application=parsed.owner_application,
        fingerprint=parsed.fingerprint,
        log_path=parsed.log_path,
    )
    return 0


def cli(argv: list[str] | None = None) -> int:
    args = sys.argv[1:] if argv is None else argv
    if args and args[0] == "report-single-instance":
        return _cli_report_single_instance(args[1:])
    return _cli_generate(args)


if __name__ == "__main__":  # pragma: no cover - CLI
    raise SystemExit(cli())


__all__ = [
    # OEM API
    "DeviceFingerprintGenerator",
    "FingerprintDocument",
    "FingerprintError",
    "FINGERPRINT_HASH",
    "SIGNATURE_ALGORITHM",
    "LICENSE_SIGNATURE_ALGORITHM",
    "LICENSE_SECRET_PATH",
    "load_license_secret",
    "sign_license_payload",
    "verify_license_payload_signature",
    "append_fingerprint_audit",
    "report_single_instance_event",
    "build_fingerprint_document",
    "get_local_fingerprint",
    "verify_document",
    # Nowe API
    "FingerprintRecord",
    "HardwareFingerprintService",
    "RotatingHmacKeyProvider",
    "build_key_provider",
    "decode_secret",
    "SecuritySignals",
    "collect_security_signals",
]
