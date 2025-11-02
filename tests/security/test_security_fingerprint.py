from datetime import date

import pytest

from bot_core.security.capabilities import build_capabilities_from_payload
from bot_core.security.fingerprint import FingerprintError, collect_security_signals
from bot_core.security.guards import build_capability_guard


_BASE_SIGNAL_ARGS = {
    "cpu_info": "Intel",
    "cpu_flags": (),
    "mac_addresses": ("001122334455",),
    "dmi_strings": (),
    "processes": ("python",),
    "kernel_modules": (),
    "filesystem_entries": (),
    "cgroup_lines": (),
    "env": {},
    "tracer_pid": None,
    "trace": None,
    "hostname_virtualization": None,
    "systemd_virtualization": None,
    "virt_what": (),
    "lscpu_info": None,
    "dmidecode_info": None,
    "wmic_info": None,
}


def _collect_signals(**overrides):
    params = dict(_BASE_SIGNAL_ARGS)
    params.update(overrides)
    return collect_security_signals(**params)


def _sample_capabilities():
    payload = {
        "edition": "pro",
        "environments": ["demo", "paper"],
        "exchanges": {"binance_spot": True},
        "strategies": {"trend_m15": True},
        "runtime": {"paper_trading": True},
        "modules": {"core": True},
    }
    return build_capabilities_from_payload(payload, effective_date=date(2025, 1, 1))


def test_collect_security_signals_detects_vm_markers():
    signals = _collect_signals(
        cpu_info="QEMU Virtual CPU version 2.5+",
        cpu_flags=("hypervisor",),
        mac_addresses=("080027ABCDEF",),
        processes=("VBoxService", "python"),
    )

    assert signals.should_block_vm is True
    assert signals.vm_indicators["cpu"]
    assert signals.vm_indicators["mac_vendor"]
    assert signals.vm_indicators["process"]
    assert signals.should_block is True


def test_collect_security_signals_detects_dmi_markers():
    signals = _collect_signals(
        dmi_strings=("VMware Virtual Platform", "Dell Inc."),
    )

    assert signals.vm_indicators["dmi"]
    assert signals.should_block_vm is True
    assert signals.should_block is True


def test_collect_security_signals_detects_debugger_markers():
    signals = _collect_signals(
        env={"PYCHARM_HOSTED": "1"},
        tracer_pid=321,
        trace=object(),
    )

    assert signals.debugger_reasons
    assert signals.should_block is True


def test_collect_security_signals_detects_dmidecode_markers():
    signals = _collect_signals(
        dmidecode_info={"manufacturer": "Microsoft Corporation", "product_name": "Virtual Machine"}
    )

    assert signals.vm_indicators["dmidecode"]
    assert signals.should_block_vm is True
    assert any("dmidecode" in reason for reason in signals.vm_reasons)


def test_collect_security_signals_detects_wmic_markers():
    signals = _collect_signals(
        wmic_info={"manufacturer": "Microsoft Corporation", "model": "Virtual Machine"}
    )

    assert signals.vm_indicators["wmic"]
    assert signals.should_block_vm is True
    assert any("WMIC" in reason for reason in signals.vm_reasons)


def test_build_capability_guard_blocks_vm_signals():
    capabilities = _sample_capabilities()
    signals = _collect_signals(
        cpu_info="QEMU Virtual CPU",
        cpu_flags=("hypervisor",),
        mac_addresses=("080027ABCDEF",),
        processes=("VBoxService",),
    )

    with pytest.raises(FingerprintError) as excinfo:
        build_capability_guard(capabilities, signals=signals)

    assert "VirtualBox" in str(excinfo.value)


def test_build_capability_guard_blocks_debugger_signals():
    capabilities = _sample_capabilities()
    signals = _collect_signals(
        env={"DEBUGPY_LAUNCHER_PORT": "9000"},
        tracer_pid=42,
        trace=object(),
    )

    with pytest.raises(FingerprintError) as excinfo:
        build_capability_guard(capabilities, signals=signals)

    assert "debugger" in str(excinfo.value).lower()


def test_build_capability_guard_blocks_dmidecode_signals():
    capabilities = _sample_capabilities()
    signals = _collect_signals(
        dmidecode_info={"manufacturer": "Microsoft Corporation", "product_version": "Hyper-V"}
    )

    with pytest.raises(FingerprintError) as excinfo:
        build_capability_guard(capabilities, signals=signals)

    assert "dmidecode" in str(excinfo.value)


def test_build_capability_guard_blocks_wmic_signals():
    capabilities = _sample_capabilities()
    signals = _collect_signals(
        wmic_info={"manufacturer": "Microsoft Corporation", "model": "Virtual Machine"}
    )

    with pytest.raises(FingerprintError) as excinfo:
        build_capability_guard(capabilities, signals=signals)

    assert "WMIC" in str(excinfo.value)


def test_build_capability_guard_allows_clean_signals():
    capabilities = _sample_capabilities()
    safe_signals = _collect_signals(
        tracer_pid=0,
    )

    guard = build_capability_guard(capabilities, signals=safe_signals)
    assert guard.capabilities is capabilities


def test_collect_security_signals_detects_kernel_module_markers():
    signals = _collect_signals(
        kernel_modules=("vboxguest", "snd_hda_intel"),
    )

    assert signals.vm_indicators["kernel_module"]
    assert signals.should_block_vm is True
    assert signals.should_block is True


def test_collect_security_signals_detects_filesystem_markers():
    signals = _collect_signals(
        filesystem_entries=("/dev/vboxguest",),
    )

    assert signals.vm_indicators["filesystem"]
    assert signals.should_block_vm is True
    assert signals.should_block is True


def test_collect_security_signals_detects_container_markers():
    signals = _collect_signals(
        filesystem_entries=("/.dockerenv",),
        cgroup_lines=("0::/docker/123",),
        env={"container": "docker"},
    )

    assert signals.vm_indicators["container"]
    assert signals.should_block_vm is True
    assert signals.should_block is True


def test_build_capability_guard_blocks_container_signals():
    capabilities = _sample_capabilities()
    signals = _collect_signals(
        filesystem_entries=("/.dockerenv",),
        cgroup_lines=("0::/docker/123",),
        env={"container": "docker"},
    )

    with pytest.raises(FingerprintError) as excinfo:
        build_capability_guard(capabilities, signals=signals)

    assert "kontener" in str(excinfo.value).lower()
    assert signals.vm_indicators["container"]


def test_build_capability_guard_blocks_virt_what_signals():
    capabilities = _sample_capabilities()
    signals = _collect_signals(virt_what=("kvm",))

    with pytest.raises(FingerprintError) as excinfo:
        build_capability_guard(capabilities, signals=signals)

    assert "virt-what" in str(excinfo.value)


def test_collect_security_signals_detects_systemd_vm_marker():
    signals = _collect_signals(systemd_virtualization="kvm")

    assert signals.vm_indicators["systemd"]
    assert signals.should_block_vm
    assert any("systemd-detect-virt" in reason for reason in signals.vm_reasons)


def test_collect_security_signals_detects_systemd_container_marker():
    signals = _collect_signals(systemd_virtualization="docker")

    assert signals.vm_indicators["container"]
    assert signals.should_block_vm
    assert any("systemd-detect-virt" in reason for reason in signals.vm_reasons)


def test_collect_security_signals_detects_hostnamectl_vm_marker():
    signals = _collect_signals(hostname_virtualization="kvm")

    assert signals.vm_indicators["hostnamectl"]
    assert signals.should_block_vm
    assert any("hostnamectl" in reason for reason in signals.vm_reasons)


def test_collect_security_signals_detects_hostnamectl_container_marker():
    signals = _collect_signals(hostname_virtualization="docker")

    assert signals.vm_indicators["container"]
    assert signals.should_block_vm
    assert any("hostnamectl" in reason for reason in signals.vm_reasons)


def test_collect_security_signals_detects_lscpu_vm_marker():
    signals = _collect_signals(
        lscpu_info={"hypervisor_vendor": "KVM", "virtualization_type": "full"}
    )

    assert signals.vm_indicators["lscpu"]
    assert signals.should_block_vm
    assert any("lscpu" in reason for reason in signals.vm_reasons)


def test_collect_security_signals_detects_virt_what_vm_marker():
    signals = _collect_signals(virt_what=("kvm",))

    assert signals.vm_indicators["virt_what"]
    assert signals.should_block_vm
    assert any("virt-what" in reason for reason in signals.vm_reasons)


def test_collect_security_signals_detects_lscpu_container_marker():
    signals = _collect_signals(
        lscpu_info={"virtualization_type": "container"}
    )

    assert signals.vm_indicators["lscpu"]
    assert signals.vm_indicators["container"]
    assert signals.should_block_vm
    assert any("lscpu" in reason for reason in signals.vm_reasons)


def test_collect_security_signals_detects_virt_what_container_marker():
    signals = _collect_signals(virt_what=("docker",))

    assert signals.vm_indicators["container"]
    assert signals.should_block_vm
    assert any("virt-what" in reason for reason in signals.vm_reasons)


def test_build_capability_guard_blocks_hostnamectl_signals():
    capabilities = _sample_capabilities()
    signals = _collect_signals(hostname_virtualization="kvm")

    with pytest.raises(FingerprintError) as excinfo:
        build_capability_guard(capabilities, signals=signals)

    assert "hostnamectl" in str(excinfo.value)


def test_build_capability_guard_blocks_lscpu_signals():
    capabilities = _sample_capabilities()
    signals = _collect_signals(lscpu_info={"hypervisor_vendor": "VMware"})

    with pytest.raises(FingerprintError) as excinfo:
        build_capability_guard(capabilities, signals=signals)

    assert "lscpu" in str(excinfo.value)
