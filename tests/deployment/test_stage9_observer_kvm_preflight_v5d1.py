"""SYNTHETIC orchestration tests; no libvirt host or CPython experiment."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, replace
from types import SimpleNamespace
from typing import Any

import pytest

from deployment import stage9_observer_kvm_preflight_v5d1 as preflight
from deployment.stage9_observer_collector_identity_v5d1 import (
    IdentityArtifactMeasurementV5d1,
    IdentityReportV5d1,
)
from deployment.stage9_observer_kvm_domain_v5d1 import (
    DomainInspectionV5d1,
    DomainSnapshotV5d1,
)
from deployment.stage9_observer_kvm_image_v5d1 import ImageArtifactV5d1, ImageMeasurementV5d1
from scripts import stage9_observer_kvm_preflight_v5d1 as cli

pytestmark = [pytest.mark.security, pytest.mark.unit]
_UUID = "12345678-1234-1234-1234-123456789abc"


def _requirements() -> dict[str, Any]:
    return {
        "profile": "stage9-observer-kvm-requirements-v5d1",
        "domain": {
            "domain_uuid": _UUID,
            "memory_bytes": 536870912,
            "vcpu_count": 1,
            "cpu_period_us": 100000,
            "cpu_quota_us": 100000,
        },
        "disk_chains": [[{"path": "/srv/observer/base.raw", "sha256": "a" * 64, "format": "raw"}]],
        "runtime_inventory": {
            "files": [
                {
                    "path": f"/runtime/file{index}",
                    "sha256": "b" * 64,
                    "build_id": None,
                    "role": "stdlib",
                    "aliases": [],
                }
                for index in range(5)
            ],
            "needed_resolutions": [],
            "loader_configuration_paths": [],
            "import_paths": ["/runtime"],
            "provider_configuration_paths": [],
        },
    }


def _raw() -> bytes:
    return json.dumps(_requirements()).encode()


def _identity(raw: bytes | None = None) -> IdentityReportV5d1:
    measurements = ()
    if raw is not None:
        digest = hashlib.sha256(raw).hexdigest()
        measurements = (
            IdentityArtifactMeasurementV5d1(
                path=str(preflight.REQUIREMENTS_PATH),
                role="domain-requirements",
                sha256=digest,
                size=len(raw),
                device=1,
                inode=2,
                mtime_ns=3,
                ctime_ns=4,
                expected_sha256=digest,
            ),
            IdentityArtifactMeasurementV5d1(
                path="/usr/bin/qemu-img",
                role="qemu-tool",
                sha256="f" * 64,
                expected_sha256="f" * 64,
                size=4,
                device=1,
                inode=9,
                mtime_ns=3,
                ctime_ns=4,
            ),
        )
    return IdentityReportV5d1(
        policy_sha256="c" * 64,
        policy_signature_verified=raw is not None,
        measurements=measurements,
        environment_sha256="d" * 64,
        reason_codes=(
            "INDEPENDENT_LAUNCH_ATTESTATION_MISSING",
            "LOADED_EXECUTION_CLOSURE_UNVERIFIED",
        ),
    )


def _snapshot() -> DomainSnapshotV5d1:
    return DomainSnapshotV5d1(
        _UUID,
        -1,
        5,
        0,
        False,
        True,
        "<domain/>",
        "<domain/>",
        524288,
        524288,
        1,
        (),
    )


def _inspection(matches: bool = True) -> DomainInspectionV5d1:
    return DomainInspectionV5d1(
        matches,
        () if matches else ("DOMAIN_RUNNING",),
        ("MEASUREMENT_CHANNEL_UNVERIFIED",),
        ("/srv/observer/base.raw",),
        (),
        (),
        536870912,
        1,
        "e" * 64,
        "e" * 64,
    )


@pytest.fixture
def native(monkeypatch: pytest.MonkeyPatch) -> Any:
    raw = _raw()
    state = SimpleNamespace(
        identity=_identity(raw),
        raw=raw,
        reads=0,
        closed=False,
        images=0,
        snapshot=_snapshot(),
        config_matches=True,
        changed=False,
        fail_read=False,
        fail_read_after=0,
        fail_close=False,
        interrupt_close=False,
    )

    class SyntheticReader:
        def read(self, uuid: str) -> DomainSnapshotV5d1:
            assert uuid == _UUID
            state.reads += 1
            if state.fail_read or (state.fail_read_after and state.reads >= state.fail_read_after):
                raise ValueError("LIBVIRT_UNAVAILABLE")
            if state.changed and state.reads > 1:
                return replace(state.snapshot, active=True, state=1)
            return state.snapshot

        def close(self) -> None:
            state.closed = True
            if state.interrupt_close:
                raise KeyboardInterrupt
            if state.fail_close:
                raise OSError

    def inspect(first: Any, second: Any, expected: Any) -> DomainInspectionV5d1:
        assert expected.domain_uuid == _UUID
        result = _inspection(state.config_matches)
        if first != second:
            return replace(
                result, config_matches=False, reason_codes=("DOMAIN_CHANGED_DURING_PREFLIGHT",)
            )
        return result

    def measure(chain: Any, *, approved_qemu_sha256: str) -> ImageMeasurementV5d1:
        state.images += 1
        assert approved_qemu_sha256 == "f" * 64
        assert chain[0].path == "/srv/observer/base.raw"
        return ImageMeasurementV5d1("IMAGE_MEASURED_LOCAL", ())

    monkeypatch.setattr(preflight, "verify_controller_identity_v5d1", lambda: state.identity)
    monkeypatch.setattr(preflight, "read_protected_file_v5d1", lambda path: state.raw)
    monkeypatch.setattr(preflight, "LibvirtDomainReaderV5d1", SyntheticReader)
    monkeypatch.setattr(preflight, "inspect_domain_pair_v5d1", inspect)
    monkeypatch.setattr(preflight, "measure_image_v5d1", measure)
    return state


def test_parse_complete_expected_runtime_without_claiming_capture() -> None:
    result = preflight.parse_requirements_v5d1(_raw())
    assert len(result.runtime_inventory.files) == 5
    assert result.disk_chains[0][0].format == "raw"
    assert result.sha256 == hashlib.sha256(_raw()).hexdigest()


@pytest.mark.parametrize(
    "mutation",
    [
        lambda obj: obj.update(profile="v5c3"),
        lambda obj: obj.update(authority="operator-approved"),
        lambda obj: obj["domain"].update(domain_uuid="trusted-vm-name"),
        lambda obj: obj["domain"].update(vcpu_count=True),
        lambda obj: obj["domain"].update(memory_bytes=0),
        lambda obj: obj["domain"].update(cpu_quota_us=float("inf")),
        lambda obj: obj["disk_chains"][0][0].update(path="/srv/../etc/passwd"),
        lambda obj: obj["disk_chains"][0][0].update(path="/srv/link\n"),
        lambda obj: obj["disk_chains"][0][0].update(format="host_device"),
        lambda obj: obj["disk_chains"][0][0].update(sha256="self-declared"),
        lambda obj: obj.update(disk_chains=[]),
        lambda obj: obj["disk_chains"].append(obj["disk_chains"][0]),
        lambda obj: obj["runtime_inventory"].update(files=[]),
        lambda obj: obj["runtime_inventory"].update(needed_resolutions=[["a"]]),
        lambda obj: obj["runtime_inventory"]["files"][0].update(role="unknown"),
        lambda obj: obj["runtime_inventory"]["files"][0].update(aliases="alias"),
    ],
)
def test_malformed_manifest_fails_closed(mutation: Any) -> None:
    obj = _requirements()
    mutation(obj)
    with pytest.raises(ValueError):
        preflight.parse_requirements_v5d1(json.dumps(obj).encode())


@pytest.mark.parametrize(
    "raw", [b"", b"[]", b"{", b"\xff", b'{"profile":1,"profile":2}', b"x" * 262145]
)
def test_malformed_or_unbounded_json_fails_closed(raw: bytes) -> None:
    with pytest.raises(ValueError):
        preflight.parse_requirements_v5d1(raw)


def test_synthetic_native_reads_never_qualify_host(native: Any) -> None:
    report = preflight.collect_preflight_v5d1()
    assert native.reads == 2 and native.images == 1 and native.closed
    assert report.domain is not None and report.domain.config_matches
    assert len(report.observation_sha256) == 2
    assert report.host_preflight == "NOT_TESTED"
    assert report.execution == "EXECUTION_BLOCKED"
    assert report.execution_authorization == "EXECUTION_NOT_AUTHORIZED"
    assert report.qualification == "BLOCKED"
    assert "MEASUREMENT_CHANNEL_UNVERIFIED" in report.reason_codes
    assert "RUNTIME_EXTERNAL_TRACE_MISSING" in report.reason_codes
    assert "HOST_PREFLIGHT_VERIFIED" not in json.dumps(asdict(report))


@pytest.fixture
def multi_chain_native(native: Any, monkeypatch: pytest.MonkeyPatch) -> Any:
    requirements = _requirements()
    requirements["disk_chains"].append(
        [{"path": "/srv/observer/second.raw", "sha256": "b" * 64, "format": "raw"}]
    )
    native.raw = json.dumps(requirements).encode()
    native.identity = _identity(native.raw)
    native.outcomes = [
        ImageMeasurementV5d1(
            "IMAGE_MEASURED_LOCAL",
            (),
            (
                ImageArtifactV5d1(
                    chain[0]["path"], chain[0]["sha256"], "raw", 1, 1, index, 1, 1, True, None, 0
                ),
            ),
        )
        for index, chain in enumerate(requirements["disk_chains"], start=1)
    ]

    def measure(chain: Any, *, approved_qemu_sha256: str) -> ImageMeasurementV5d1:
        outcome = native.outcomes[native.images]
        native.images += 1
        assert approved_qemu_sha256 == "f" * 64
        if isinstance(outcome, BaseException):
            raise outcome
        assert not outcome.artifacts or outcome.artifacts[0].path == chain[0].path
        return outcome

    monkeypatch.setattr(preflight, "measure_image_v5d1", measure)
    return native


def test_complete_multi_chain_result_publishes_every_artifact(multi_chain_native: Any) -> None:
    report = preflight.collect_preflight_v5d1()
    assert multi_chain_native.images == 2 and multi_chain_native.closed
    assert report.images == tuple(multi_chain_native.outcomes)
    assert [item.artifacts[0].path for item in report.images] == [
        "/srv/observer/base.raw",
        "/srv/observer/second.raw",
    ]


@pytest.mark.parametrize("blocked_index", [0, 1])
def test_one_blocked_chain_discards_all_chains_and_preserves_diagnostics(
    multi_chain_native: Any, blocked_index: int
) -> None:
    previous = multi_chain_native.outcomes[blocked_index]
    # Even a malformed downstream blocked result cannot smuggle positive
    # records through the aggregate boundary.
    multi_chain_native.outcomes[blocked_index] = replace(
        previous,
        status="IMAGE_MEASUREMENT_BLOCKED",
        reason_codes=("IMAGE_MEASUREMENT_BLOCKED", "IMAGE_DIGEST_MISMATCH"),
    )
    report = preflight.collect_preflight_v5d1()
    assert multi_chain_native.images == 2 and multi_chain_native.closed
    assert "IMAGE_MEASUREMENT_BLOCKED" in report.reason_codes
    assert "IMAGE_DIGEST_MISMATCH" in report.reason_codes
    assert report.images == ()


@pytest.mark.parametrize(
    ("failure", "reason"),
    [
        (KeyboardInterrupt(), "NATIVE_PREFLIGHT_INTERRUPTED"),
        (OSError("SYNTHETIC_CHAIN_UNAVAILABLE"), "NATIVE_PREFLIGHT_OBSERVATION_INCOMPLETE"),
    ],
)
def test_interrupt_or_error_during_aggregation_discards_earlier_chain(
    multi_chain_native: Any, failure: BaseException, reason: str
) -> None:
    multi_chain_native.outcomes[1] = failure
    report = preflight.collect_preflight_v5d1()
    assert multi_chain_native.images == 2 and multi_chain_native.closed
    assert reason in report.reason_codes
    assert "IMAGE_MEASUREMENT_BLOCKED" in report.reason_codes
    assert report.images == ()


@pytest.mark.parametrize(
    ("failure", "reason"),
    [
        ("domain", "DOMAIN_CHANGED_DURING_PREFLIGHT"),
        ("read", "NATIVE_PREFLIGHT_OBSERVATION_INCOMPLETE"),
        ("requirements", "REQUIREMENTS_CHANGED_DURING_PREFLIGHT"),
        ("identity", "CONTROLLER_IDENTITY_CHANGED_DURING_PREFLIGHT"),
        ("identity-reason", "IDENTITY_ARTIFACT_HASH_MISMATCH"),
        ("close", "LIBVIRT_READONLY_CLOSE_FAILED"),
        ("close-interrupted", "NATIVE_PREFLIGHT_INTERRUPTED"),
    ],
)
def test_late_preflight_failure_discards_all_measured_chains(
    multi_chain_native: Any,
    failure: str,
    reason: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    if failure == "domain":
        multi_chain_native.changed = True
    elif failure == "read":
        multi_chain_native.fail_read_after = 2
    elif failure == "requirements":
        reads = iter((multi_chain_native.raw, multi_chain_native.raw + b" "))
        monkeypatch.setattr(preflight, "read_protected_file_v5d1", lambda path: next(reads))
    elif failure in {"identity", "identity-reason"}:
        identity = multi_chain_native.identity
        final = (
            replace(identity, policy_sha256="f" * 64)
            if failure == "identity"
            else replace(identity, reason_codes=identity.reason_codes + (reason,))
        )
        identities = iter((identity, final))
        monkeypatch.setattr(preflight, "verify_controller_identity_v5d1", lambda: next(identities))
    elif failure == "close":
        multi_chain_native.fail_close = True
    else:
        multi_chain_native.interrupt_close = True
    report = preflight.collect_preflight_v5d1()
    assert multi_chain_native.images == 2 and multi_chain_native.closed
    assert reason in report.reason_codes
    assert "IMAGE_MEASUREMENT_BLOCKED" in report.reason_codes
    assert report.images == ()


@pytest.mark.parametrize("failure", ["signature", "hash", "role", "path", "expected_hash"])
def test_false_authority_stops_before_native_reads(native: Any, failure: str) -> None:
    identity = native.identity
    pin = identity.measurements[0]
    if failure == "signature":
        identity = replace(identity, policy_signature_verified=False)
    else:
        changes = {
            "hash": {"sha256": "0" * 64},
            "role": {"role": "controller-code"},
            "path": {"path": "/tmp/unapproved.json"},
            "expected_hash": {"expected_sha256": "0" * 64},
        }[failure]
        identity = replace(identity, measurements=(replace(pin, **changes),))
    native.identity = identity
    report = preflight.collect_preflight_v5d1()
    assert "REQUIREMENTS_INDEPENDENT_APPROVAL_MISSING" in report.reason_codes
    assert native.reads == 0 and native.images == 0


@pytest.mark.parametrize(
    "reason",
    [
        "IDENTITY_ARTIFACT_HASH_MISMATCH",
        "IDENTITY_POLICY_CHANGED",
        "IDENTITY_ENVIRONMENT_MISMATCH",
        "IDENTITY_ARTIFACT_UNREADABLE",
        "IDENTITY_PATH_NOT_PROTECTED",
        "IDENTITY_POLICY_CLOSURE_INCOMPLETE",
    ],
)
def test_known_invalid_controller_measurements_block_native_code(native: Any, reason: str) -> None:
    native.identity = replace(
        native.identity, reason_codes=native.identity.reason_codes + (reason,)
    )
    report = preflight.collect_preflight_v5d1()
    assert "CONTROLLER_POLICY_MEASUREMENTS_REJECTED" in report.reason_codes
    assert native.reads == 0 and native.images == 0


def test_missing_or_mismatched_image_tool_approval_blocks_native_reads(native: Any) -> None:
    native.identity = replace(native.identity, measurements=native.identity.measurements[:1])
    report = preflight.collect_preflight_v5d1()
    assert "IMAGE_TOOL_APPROVAL_MISSING" in report.reason_codes
    assert native.reads == 0 and native.images == 0


def test_missing_manifest_is_refusal(native: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    def missing(path: Any) -> bytes:
        raise OSError("not a diagnostic code")

    monkeypatch.setattr(preflight, "read_protected_file_v5d1", missing)
    assert "REQUIREMENTS_UNAVAILABLE" in preflight.collect_preflight_v5d1().reason_codes
    assert native.reads == 0


def test_changed_domain_retains_refusal(native: Any) -> None:
    native.changed = True
    report = preflight.collect_preflight_v5d1()
    assert "DOMAIN_CHANGED_DURING_PREFLIGHT" in report.reason_codes
    assert native.closed and report.execution == "EXECUTION_BLOCKED"


def test_running_domain_does_not_measure_disk(native: Any) -> None:
    native.config_matches = False
    report = preflight.collect_preflight_v5d1()
    assert "IMAGE_MEASUREMENT_BLOCKED" in report.reason_codes
    assert native.images == 0 and native.closed


@pytest.mark.parametrize("failure", ["fail_read", "fail_close"])
def test_native_errors_never_produce_success(native: Any, failure: str) -> None:
    setattr(native, failure, True)
    report = preflight.collect_preflight_v5d1()
    assert native.closed
    assert report.execution == "EXECUTION_BLOCKED"
    assert any("INCOMPLETE" in reason or "CLOSE_FAILED" in reason for reason in report.reason_codes)


def test_manifest_changed_during_measurement(native: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    reads = iter((_raw(), _raw() + b" "))
    monkeypatch.setattr(preflight, "read_protected_file_v5d1", lambda path: next(reads))
    assert (
        "REQUIREMENTS_CHANGED_DURING_PREFLIGHT" in preflight.collect_preflight_v5d1().reason_codes
    )


def test_identity_changed_during_measurement(native: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    identities = iter((native.identity, replace(native.identity, policy_sha256="f" * 64)))
    monkeypatch.setattr(preflight, "verify_controller_identity_v5d1", lambda: next(identities))
    assert (
        "CONTROLLER_IDENTITY_CHANGED_DURING_PREFLIGHT"
        in preflight.collect_preflight_v5d1().reason_codes
    )


def test_cli_json_and_exit_preserve_execution_block(native: Any, capsys: Any) -> None:
    assert cli.main([]) == 2
    obj = json.loads(capsys.readouterr().out)
    assert obj["production"] == "NO-GO" and obj["stage3b"] == "NOT AUTHORIZED"


def test_cli_cannot_select_authority_or_mutate_vm(capsys: Any) -> None:
    with pytest.raises(SystemExit, match="2"):
        cli.main(["--policy", "/tmp/synthetic.json"])
    capsys.readouterr()
    with pytest.raises(SystemExit, match="0"):
        cli.main(["--help"])


def test_v5c3_collector_methods_always_refuse() -> None:
    collector = preflight.prepared_collector_v5d1()
    for origin in ("SYNTHETIC", "CONTROLLER_OBSERVED", "GUEST_REPORTED"):
        collector.origin = origin
        with pytest.raises(preflight.PreflightRefusalV5d1, match="COLLECTOR_IDENTITY_UNVERIFIED"):
            collector.inspect()
        with pytest.raises(preflight.PreflightRefusalV5d1, match="EXECUTION_NOT_AUTHORIZED"):
            collector.execute(b"{}", b"inert", 1)
        with pytest.raises(preflight.PreflightRefusalV5d1, match="VM_LIFECYCLE_NOT_AUTHORIZED"):
            collector.stop()


def test_collector_preflight_does_not_create_observation(native: Any) -> None:
    assert preflight.ReadOnlyKVMCollectorV5d1().preflight().execution == "EXECUTION_BLOCKED"


def test_lazy_import_diagnostic_is_not_artifact_drift(
    native: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    identities = iter(
        (
            native.identity,
            replace(native.identity, reason_codes=("LOADED_EXECUTION_CLOSURE_UNVERIFIED",)),
        )
    )
    monkeypatch.setattr(preflight, "verify_controller_identity_v5d1", lambda: next(identities))
    report = preflight.collect_preflight_v5d1()
    assert "CONTROLLER_IDENTITY_CHANGED_DURING_PREFLIGHT" not in report.reason_codes


@pytest.mark.parametrize("origin", ["CONTROLLER_OBSERVED", "SYNTHETIC"])
def test_existing_v5c3_orchestrator_cannot_promote_readonly_adapter(
    tmp_path: Any, monkeypatch: pytest.MonkeyPatch, origin: Any
) -> None:
    from tests.deployment import test_stage9_observer_pyc_execution_v5c3 as previous

    case = previous.case.__wrapped__(tmp_path, monkeypatch)
    collector = preflight.ReadOnlyKVMCollectorV5d1()
    collector.origin = origin
    result = case.reconstruct(collector=collector)
    assert result.consistency == "EXECUTION_BLOCKED"
    assert result.reconstruction == "UNVERIFIED"
    assert result.qualification == "BLOCKED"
    assert not result.files
    if origin == "CONTROLLER_OBSERVED":
        assert "COLLECTOR_IDENTITY_UNVERIFIED" in result.reason_codes
