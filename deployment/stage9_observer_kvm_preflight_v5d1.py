"""Read-only operator-host preflight; never an execution authorization.

The only native entry point obtains expectations from fixed, protected host
files bound to the operator's separately signed controller policy. Neither a
serialized report nor a Python adapter can promote this result into V5c3
execution evidence. No guest code, VM lifecycle operation or mount is invoked.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal, cast

from deployment.stage9_observer_collector_identity_v5d1 import (
    IdentityReportV5d1,
    read_protected_file_v5d1,
    verify_controller_identity_v5d1,
)
from deployment.stage9_observer_kvm_domain_v5d1 import (
    DomainInspectionV5d1,
    DomainRequirementsV5d1,
    LibvirtDomainReaderV5d1,
    inspect_domain_pair_v5d1,
)
from deployment.stage9_observer_kvm_image_v5d1 import (
    ExpectedImageV5d1,
    ImageMeasurementV5d1,
    measure_image_v5d1,
)
from deployment.stage9_observer_pyc_execution_v5c3 import (
    ControllerObservationV5c3,
    PreparedVMCollectorV5c3,
    ProcessObservationV5c3,
)
from deployment.stage9_observer_runtime_identity_v5c3 import (
    RuntimeFilePinV5c3,
    RuntimeInventoryV5c3,
    validate_runtime_inventory_structure_v5c3,
)

REQUIREMENTS_PATH = Path("/etc/cryptohunter/observer-v5d1/domain-requirements.json")
_MAX_REQUIREMENTS_BYTES = 262144
_DIAGNOSTIC_IDENTITY_GAPS = frozenset(
    {
        "INDEPENDENT_LAUNCH_ATTESTATION_MISSING",
        "LOADED_EXECUTION_CLOSURE_UNVERIFIED",
        "SYSTEM_SITE_BOOTSTRAP_UNVERIFIED",
        "IDENTITY_STARTUP_NOT_ISOLATED",
        "LIBVIRT_EXECUTION_CODE_UNOBSERVED",
    }
)


class PreflightRefusalV5d1(ValueError):
    """A V5d1 refusal is not a V5c3 observation or a qualification receipt."""


@dataclass(frozen=True, slots=True)
class PreflightRequirementsV5d1:
    domain: DomainRequirementsV5d1
    disk_chains: tuple[tuple[ExpectedImageV5d1, ...], ...]
    runtime_inventory: RuntimeInventoryV5c3
    sha256: str


@dataclass(frozen=True, slots=True)
class PreflightReportV5d1:
    identity: IdentityReportV5d1
    reason_codes: tuple[str, ...]
    requirements_sha256: str | None = None
    domain: DomainInspectionV5d1 | None = None
    images: tuple[ImageMeasurementV5d1, ...] = ()
    observation_sha256: tuple[str, ...] = ()
    profile: Literal["stage9-observer-kvm-preflight-v5d1"] = field(
        default="stage9-observer-kvm-preflight-v5d1", init=False
    )
    implementation: Literal["IMPLEMENTED"] = field(default="IMPLEMENTED", init=False)
    host_preflight: Literal["NOT_TESTED"] = field(default="NOT_TESTED", init=False)
    execution_authorization: Literal["EXECUTION_NOT_AUTHORIZED"] = field(
        default="EXECUTION_NOT_AUTHORIZED", init=False
    )
    execution: Literal["EXECUTION_BLOCKED"] = field(default="EXECUTION_BLOCKED", init=False)
    runtime_observation: Literal["NOT_TESTED"] = field(default="NOT_TESTED", init=False)
    qualification: Literal["BLOCKED"] = field(default="BLOCKED", init=False)
    stage9: Literal["IN_PROGRESS"] = field(default="IN_PROGRESS", init=False)
    observer3a: Literal["BLOCKED"] = field(default="BLOCKED", init=False)
    stage3b: Literal["NOT AUTHORIZED"] = field(default="NOT AUTHORIZED", init=False)
    production: Literal["NO-GO"] = field(default="NO-GO", init=False)
    pyc_reconstruction: Literal["UNVERIFIED"] = field(default="UNVERIFIED", init=False)


def _require(condition: bool, reason: str) -> None:
    if not condition:
        raise PreflightRefusalV5d1(reason)


def _object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        _require(key not in result, "REQUIREMENTS_DUPLICATE_KEY")
        result[key] = value
    return result


def _keys(value: Any, expected: set[str]) -> dict[str, Any]:
    _require(type(value) is dict and set(value) == expected, "REQUIREMENTS_SCHEMA_REJECTED")
    return cast(dict[str, Any], value)


def _positive_int(value: Any, maximum: int) -> int:
    _require(type(value) is int and 0 < value <= maximum, "REQUIREMENTS_LIMIT_REJECTED")
    return cast(int, value)


def _path(value: Any) -> str:
    _require(type(value) is str and 1 < len(value) <= 4096, "REQUIREMENTS_PATH_REJECTED")
    _require(
        value.startswith("/")
        and all(part not in {"", ".", ".."} for part in value[1:].split("/"))
        and all(32 <= ord(c) < 127 and c != "\\" for c in value),
        "REQUIREMENTS_PATH_REJECTED",
    )
    return cast(str, value)


def _strings(value: Any, maximum: int) -> tuple[str, ...]:
    _require(type(value) is list and len(value) <= maximum, "REQUIREMENTS_SCHEMA_REJECTED")
    _require(all(type(item) is str for item in value), "REQUIREMENTS_SCHEMA_REJECTED")
    return tuple(value)


def _inventory(raw: Any) -> RuntimeInventoryV5c3:
    obj = _keys(
        raw,
        {
            "files",
            "needed_resolutions",
            "loader_configuration_paths",
            "import_paths",
            "provider_configuration_paths",
        },
    )
    _require(
        type(obj["files"]) is list and 5 <= len(obj["files"]) <= 512,
        "REQUIREMENTS_RUNTIME_REJECTED",
    )
    files: list[RuntimeFilePinV5c3] = []
    for value in obj["files"]:
        pin = _keys(value, {"path", "sha256", "build_id", "role", "aliases"})
        files.append(
            RuntimeFilePinV5c3(
                path=pin["path"],
                sha256=pin["sha256"],
                build_id=pin["build_id"],
                role=pin["role"],
                aliases=_strings(pin["aliases"], 32),
            )
        )
    resolutions = obj["needed_resolutions"]
    _require(
        type(resolutions) is list and len(resolutions) <= 4096,
        "REQUIREMENTS_RUNTIME_REJECTED",
    )
    edges: list[tuple[str, str, str]] = []
    for edge in resolutions:
        parts = _strings(edge, 3)
        _require(len(parts) == 3, "REQUIREMENTS_RUNTIME_REJECTED")
        edges.append((parts[0], parts[1], parts[2]))
    inventory = RuntimeInventoryV5c3(
        files=tuple(files),
        needed_resolutions=tuple(edges),
        loader_configuration_paths=_strings(obj["loader_configuration_paths"], 256),
        import_paths=_strings(obj["import_paths"], 64),
        provider_configuration_paths=_strings(obj["provider_configuration_paths"], 256),
    )
    validate_runtime_inventory_structure_v5c3(inventory)
    return inventory


def parse_requirements_v5d1(raw: bytes) -> PreflightRequirementsV5d1:
    """Parse expectations as data; parsing supplies no independent approval."""
    _require(
        type(raw) is bytes and 0 < len(raw) <= _MAX_REQUIREMENTS_BYTES,
        "REQUIREMENTS_LIMIT_REJECTED",
    )
    try:
        obj = _keys(
            json.loads(raw, object_pairs_hook=_object),
            {"profile", "domain", "disk_chains", "runtime_inventory"},
        )
        _require(
            obj["profile"] == "stage9-observer-kvm-requirements-v5d1",
            "REQUIREMENTS_SCHEMA_REJECTED",
        )
        domain = _keys(
            obj["domain"],
            {"domain_uuid", "memory_bytes", "vcpu_count", "cpu_period_us", "cpu_quota_us"},
        )
        uuid = domain["domain_uuid"]
        _require(
            type(uuid) is str
            and re.fullmatch(r"[0-9a-f]{8}-(?:[0-9a-f]{4}-){3}[0-9a-f]{12}", uuid) is not None,
            "REQUIREMENTS_DOMAIN_REJECTED",
        )
        chains = obj["disk_chains"]
        _require(type(chains) is list and 1 <= len(chains) <= 8, "REQUIREMENTS_DISKS_REJECTED")
        disk_chains: list[tuple[ExpectedImageV5d1, ...]] = []
        all_paths: set[str] = set()
        for chain in chains:
            _require(type(chain) is list and 1 <= len(chain) <= 16, "REQUIREMENTS_DISKS_REJECTED")
            images: list[ExpectedImageV5d1] = []
            for value in chain:
                pin = _keys(value, {"path", "sha256", "format"})
                path = _path(pin["path"])
                _require(path not in all_paths, "REQUIREMENTS_DISKS_REJECTED")
                all_paths.add(path)
                _require(
                    type(pin["sha256"]) is str
                    and re.fullmatch(r"[0-9a-f]{64}", pin["sha256"]) is not None
                    and pin["format"] in {"raw", "qcow2"},
                    "REQUIREMENTS_DISKS_REJECTED",
                )
                images.append(ExpectedImageV5d1(path, pin["sha256"], pin["format"]))
            disk_chains.append(tuple(images))
        requirements = DomainRequirementsV5d1(
            domain_uuid=uuid,
            disk_paths=tuple(chain[0].path for chain in disk_chains),
            memory_bytes=_positive_int(domain["memory_bytes"], 1024**4),
            vcpu_count=_positive_int(domain["vcpu_count"], 64),
            cpu_period_us=_positive_int(domain["cpu_period_us"], 1000000),
            cpu_quota_us=_positive_int(domain["cpu_quota_us"], 64000000),
            disk_formats=tuple(chain[0].format for chain in disk_chains),
            disk_backing_chains=tuple(
                tuple(image.path for image in chain) for chain in disk_chains
            ),
            disk_backing_formats=tuple(
                tuple(image.format for image in chain) for chain in disk_chains
            ),
        )
        return PreflightRequirementsV5d1(
            requirements,
            tuple(disk_chains),
            _inventory(obj["runtime_inventory"]),
            hashlib.sha256(raw).hexdigest(),
        )
    except PreflightRefusalV5d1:
        raise
    except (ValueError, TypeError, KeyError, RecursionError, UnicodeError) as exc:
        raise PreflightRefusalV5d1("REQUIREMENTS_SCHEMA_REJECTED") from exc


def collect_preflight_v5d1() -> PreflightReportV5d1:
    """Use only fixed native boundaries; the report always refuses execution."""
    identity = verify_controller_identity_v5d1()
    reasons = list(identity.reason_codes)
    reasons.extend(
        ("LOCAL_EXECUTION_OBSERVATION_DISABLED_V5D1", "EXPERIMENT_AUTHORIZATION_MISSING")
    )
    try:
        raw = read_protected_file_v5d1(REQUIREMENTS_PATH)
        requirements = parse_requirements_v5d1(raw)
    except (ValueError, OSError) as exc:
        reason = str(exc)
        reasons.append(
            reason if re.fullmatch(r"[A-Z][A-Z0-9_]{1,127}", reason) else "REQUIREMENTS_UNAVAILABLE"
        )
        return PreflightReportV5d1(identity, tuple(dict.fromkeys(reasons)))
    approved = identity.policy_signature_verified and any(
        pin.path == str(REQUIREMENTS_PATH)
        and pin.role == "domain-requirements"
        and pin.sha256 == requirements.sha256
        and pin.expected_sha256 == requirements.sha256
        for pin in identity.measurements
    )
    if not approved:
        reasons.append("REQUIREMENTS_INDEPENDENT_APPROVAL_MISSING")
        return PreflightReportV5d1(identity, tuple(dict.fromkeys(reasons)), requirements.sha256)
    if set(identity.reason_codes) - _DIAGNOSTIC_IDENTITY_GAPS:
        reasons.append("CONTROLLER_POLICY_MEASUREMENTS_REJECTED")
        return PreflightReportV5d1(identity, tuple(dict.fromkeys(reasons)), requirements.sha256)
    tool_pins = [
        pin
        for pin in identity.measurements
        if pin.path == "/usr/bin/qemu-img"
        and pin.role == "qemu-tool"
        and pin.sha256 == pin.expected_sha256
    ]
    if len(tool_pins) != 1:
        reasons.append("IMAGE_TOOL_APPROVAL_MISSING")
        return PreflightReportV5d1(identity, tuple(dict.fromkeys(reasons)), requirements.sha256)
    inspection: DomainInspectionV5d1 | None = None
    pending_images: list[ImageMeasurementV5d1] = []
    image_evidence_complete = False
    observations: list[str] = []
    reader: LibvirtDomainReaderV5d1 | None = None
    try:
        reader = LibvirtDomainReaderV5d1()
        first = reader.read(requirements.domain.domain_uuid)
        observations.append(_observation_digest(first))
        # Reject a running/mismatched guest before attempting disk inspection.
        inspection = inspect_domain_pair_v5d1(first, first, requirements.domain)
        if inspection.config_matches:
            for chain in requirements.disk_chains:
                image = measure_image_v5d1(chain, approved_qemu_sha256=tool_pins[0].expected_sha256)
                reasons.extend(image.reason_codes)
                pending_images.append(image)
            image_evidence_complete = all(
                image.status == "IMAGE_MEASURED_LOCAL" for image in pending_images
            )
            if not image_evidence_complete:
                reasons.append("IMAGE_MEASUREMENT_BLOCKED")
        else:
            reasons.append("IMAGE_MEASUREMENT_BLOCKED")
        second = reader.read(requirements.domain.domain_uuid)
        observations.append(_observation_digest(second))
        inspection = inspect_domain_pair_v5d1(first, second, requirements.domain)
        reasons.extend(inspection.reason_codes)
        reasons.extend(inspection.missing_evidence)
        image_evidence_complete = image_evidence_complete and inspection.config_matches
        # Rebind approval and requirements after the native reads, too.
        if read_protected_file_v5d1(REQUIREMENTS_PATH) != raw:
            reasons.append("REQUIREMENTS_CHANGED_DURING_PREFLIGHT")
            image_evidence_complete = False
        final_identity = verify_controller_identity_v5d1()
        reasons.extend(final_identity.reason_codes)
        if _identity_binding(final_identity) != _identity_binding(identity):
            reasons.append("CONTROLLER_IDENTITY_CHANGED_DURING_PREFLIGHT")
            image_evidence_complete = False
        if set(final_identity.reason_codes) - _DIAGNOSTIC_IDENTITY_GAPS:
            image_evidence_complete = False
    except KeyboardInterrupt:
        image_evidence_complete = False
        reasons.append("NATIVE_PREFLIGHT_INTERRUPTED")
    except Exception as exc:
        # Native errors are diagnostics, never a partial successful receipt.
        image_evidence_complete = False
        reasons.append("NATIVE_PREFLIGHT_OBSERVATION_INCOMPLETE")
        reason = str(exc)
        if re.fullmatch(r"[A-Z][A-Z0-9_]{1,127}", reason):
            reasons.append(reason)
    finally:
        if reader is not None:
            try:
                reader.close()
            except KeyboardInterrupt:
                image_evidence_complete = False
                reasons.append("NATIVE_PREFLIGHT_INTERRUPTED")
            except Exception:
                image_evidence_complete = False
                reasons.append("LIBVIRT_READONLY_CLOSE_FAILED")
    # Publish every chain together only after native reads, approval rebinding
    # and reader cleanup succeed. Blocked diagnostics remain in reason_codes.
    images = tuple(pending_images) if image_evidence_complete else ()
    if pending_images and not image_evidence_complete:
        reasons.append("IMAGE_MEASUREMENT_BLOCKED")
    reasons.extend(("RUNTIME_EXTERNAL_TRACE_MISSING", "RUNTIME_TREE_CAPTURE_MISSING"))
    return PreflightReportV5d1(
        identity=identity,
        reason_codes=tuple(dict.fromkeys(reasons)),
        requirements_sha256=requirements.sha256,
        domain=inspection,
        images=images,
        observation_sha256=tuple(observations),
    )


def _observation_digest(snapshot: Any) -> str:
    return hashlib.sha256(
        json.dumps(asdict(snapshot), sort_keys=True, separators=(",", ":"), allow_nan=False).encode(
            "utf-8"
        )
    ).hexdigest()


def _identity_binding(identity: IdentityReportV5d1) -> tuple[Any, ...]:
    # Lazy libvirt imports can resolve a missing-observation diagnostic. Only
    # changed policy, environment or measured artifacts establish actual drift.
    return (
        identity.policy_sha256,
        identity.policy_signature_verified,
        identity.authority_id,
        identity.policy_revision,
        identity.environment_sha256,
        identity.measurements,
    )


class ReadOnlyKVMCollectorV5d1:
    """Concrete V5c3 interface whose lifecycle methods remain unauthorized.

    Labels provide compatibility only. V5c3 rejects this origin before invoking
    callbacks; even direct invocation cannot construct an incomplete accepted
    ControllerObservationV5c3 or execute or stop an existing domain.
    """

    collector_id = "stage9-readonly-kvm-preflight-v5d1"
    origin: Literal["CONTROLLER_OBSERVED", "SYNTHETIC", "GUEST_REPORTED"] = "CONTROLLER_OBSERVED"

    def preflight(self) -> PreflightReportV5d1:
        return collect_preflight_v5d1()

    def inspect(self) -> ControllerObservationV5c3:
        raise PreflightRefusalV5d1("COLLECTOR_IDENTITY_UNVERIFIED")

    def execute(
        self, request_raw: bytes, source: bytes, timeout_s: float
    ) -> ProcessObservationV5c3:
        raise PreflightRefusalV5d1("EXECUTION_NOT_AUTHORIZED")

    def stop(self) -> bool:
        raise PreflightRefusalV5d1("VM_LIFECYCLE_NOT_AUTHORIZED")


def prepared_collector_v5d1() -> PreparedVMCollectorV5c3:
    """Type-check the existing protocol without changing its semantics."""
    return ReadOnlyKVMCollectorV5d1()
