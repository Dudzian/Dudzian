"""Synthetic controller mechanism tests over actual immutable fixture DEBs.

The opaque PYC and its independently specified digest predate these tests. The
runtime-byte validator is replaced only inside the synthetic mechanism fixture:
these tests neither possess nor execute the pinned Ubuntu CPython. Static ELF
validation and trusted-host compiler tests live in their separate test modules.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Callable

import pytest

from deployment import (
    stage9_observer_pyc_execution_v5c3 as execution,
    stage9_observer_pyc_reconstruction_v5c2 as declaration,
    stage9_observer_pyc_source_hash_v5c2 as source_hash,
    stage9_observer_runtime_identity_v5c3 as runtime,
)
from deployment.canonical_json import canonical_json_bytes
from deployment.stage9_observer_pyc_v5c1 import SourceBindingV5c1
from tests.deployment.test_stage9_observer_pyc_source_hash_v5c2 import (
    SourceHashCase,
    _build_case,
)

pytestmark = [pytest.mark.security, pytest.mark.unit]

_SOURCE = b"value = 37\n"
_SOURCE_HASH = "0bc3e3a8ab4ff6bb"
_PYC_SHA256 = "7f1e07689792df18b6cae867be8efd86cb8742819a0e7e493b64465dcbdd30d0"
_PYC = bytes.fromhex("cb0d0d0a03000000" + _SOURCE_HASH) + (
    b"INERT BODY; NOT A MARSHALLED CODE OBJECT\n"
)


def _sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _runtime_digest(inventory: Any) -> str:
    return _sha(canonical_json_bytes(json.loads(json.dumps(asdict(inventory)))))


def _requests_digest(requests: tuple[Any, ...]) -> str:
    document = json.loads(json.dumps([asdict(request) for request in requests]))
    return _sha(canonical_json_bytes(document))


class SyntheticCollector:
    """In-memory test double, explicitly incapable of exact execution evidence."""

    collector_id = "synthetic-unit-test-collector"
    origin = "SYNTHETIC"

    def __init__(self, observation: Any) -> None:
        self.observation = observation
        self.inspect_calls = 0
        self.execute_calls = 0
        self.stop_calls = 0
        self.stop_result = True
        self.outputs = [_PYC]
        self.mutate_process: Callable[[Any], Any] = lambda result: result
        self.execution_error: BaseException | None = None
        self.inspection_error: BaseException | None = None
        self.after_observation: Any = None
        self.requests: list[dict[str, Any]] = []
        self.sources: list[bytes] = []
        self.runtime_accesses = (("/usr/bin/python3.12", source_hash.TARGET_INTERPRETER_SHA256),)
        self.mapped_elf_paths = ("/usr/bin/python3.12",)

    def inspect(self) -> Any:
        self.inspect_calls += 1
        if self.inspection_error is not None:
            raise self.inspection_error
        if self.inspect_calls > 1 and self.after_observation is not None:
            return self.after_observation
        return self.observation

    def execute(self, request_raw: bytes, source: bytes, timeout_s: float) -> Any:
        self.execute_calls += 1
        if self.execution_error is not None:
            raise self.execution_error
        request = json.loads(request_raw)
        self.requests.append(request)
        self.sources.append(source)
        assert timeout_s > 0
        result = execution.ProcessObservationV5c3(
            run_id=request["run_id"],
            nonce=request["nonce"],
            arguments=execution.ARGUMENTS,
            environment=execution.ENVIRONMENT,
            input_sha256=_sha(source),
            output=self.outputs[min(self.execute_calls - 1, len(self.outputs) - 1)],
            exit_code=0,
            stdout=b"synthetic collector: no interpreter executed\n",
            stderr=b"",
            elapsed_s=0.01,
            interrupted=False,
            interruption_reason=None,
            complete=True,
            runtime_accesses=self.runtime_accesses,
            runtime_accesses_complete=True,
            mapped_elf_paths=self.mapped_elf_paths,
        )
        return self.mutate_process(result)

    def stop(self) -> bool:
        self.stop_calls += 1
        return self.stop_result


@dataclass
class ExecutionCase:
    deb: SourceHashCase
    request: Any
    runtime_inventory: Any
    collector: SyntheticCollector

    def reconstruct(self, **kwargs: Any) -> Any:
        return execution.reconstruct_pyc_set_v5c3(
            inventory_raw=kwargs.pop("inventory_raw", self.deb.inventory_raw),
            archives=kwargs.pop("archives", self.deb.archives),
            requests=kwargs.pop("requests", (self.request,)),
            expected_requests_sha256=kwargs.pop(
                "expected_requests_sha256", _requests_digest((self.request,))
            ),
            expected_inventory_sha256=kwargs.pop(
                "expected_inventory_sha256", _sha(self.deb.inventory_raw)
            ),
            repository_commit=kwargs.pop("repository_commit", "c" * 40),
            expected_runtime_inventory=kwargs.pop(
                "expected_runtime_inventory", self.runtime_inventory
            ),
            expected_runtime_inventory_sha256=kwargs.pop(
                "expected_runtime_inventory_sha256",
                _runtime_digest(self.runtime_inventory),
            ),
            collector=kwargs.pop("collector", self.collector),
            **kwargs,
        )


@pytest.fixture
def case(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> ExecutionCase:
    deb = _build_case(tmp_path / "debs")
    source = SourceBindingV5c1(**deb.operation()["source"])
    package = declaration.ReconstructionPackageV5c2(
        "python3.12-minimal",
        "3.12.3-1ubuntu0.17",
        "amd64",
        _sha(b"independently pinned synthetic interpreter DEB"),
        source.snapshot_sha256,
    )
    library = declaration.ReconstructionLibraryV5c2(
        "/usr/lib/x86_64-linux-gnu/libc.so.6",
        _sha(b"synthetic library"),
        "12" * 20,
        replace(package, package="libc6", version="2.39-0ubuntu8"),
    )
    policy = declaration.PycReconstructionPolicyV5c2(
        interpreter_sha256=source_hash.TARGET_INTERPRETER_SHA256,
        interpreter_build_id=source_hash.TARGET_BUILD_ID,
        interpreter_package=package,
        required_libraries=(library,),
        source=source,
        parser_version="CPython-3.12.3",
        encoding_configuration="pep263-utf8-default-v1",
        co_filename="/module.py",
        optimization=0,
        invalidation="checked_hash",
        arguments=execution.ARGUMENTS,
        environment=execution.ENVIRONMENT,
        isolation_requirements=execution.ISOLATION_REQUIREMENTS,
        tool_image_id="synthetic-noble-tool-image",
        tool_image_sha256=_sha(b"synthetic tool image"),
        expected_image_sha256=_sha(b"separately pinned expected image"),
        expected_pyc_sha256=_PYC_SHA256,
    )
    request = execution.ReconstructionRequestV5c3(
        policy=policy,
        expected_source_hash_hex=_SOURCE_HASH,
        target_path="/cache/module.cpython-312.pyc",
    )
    inventory = runtime.RuntimeInventoryV5c3(
        files=(
            runtime.RuntimeFilePinV5c3(
                "/usr/bin/python3.12",
                source_hash.TARGET_INTERPRETER_SHA256,
                source_hash.TARGET_BUILD_ID,
                "interpreter",
            ),
            runtime.RuntimeFilePinV5c3(
                "/stage9/tool/compile_worker.py", _sha(b"synthetic worker"), None, "tool"
            ),
            runtime.RuntimeFilePinV5c3(
                "/usr/lib/python3.12/py_compile.py", _sha(b"synthetic py_compile"), None, "stdlib"
            ),
            runtime.RuntimeFilePinV5c3("/etc/ld.so.preload", None, None, "configuration"),
            runtime.RuntimeFilePinV5c3("/etc/ld.so.cache", None, None, "configuration"),
        ),
        needed_resolutions=(),
        loader_configuration_paths=(),
        import_paths=("/usr/lib/python3.12",),
    )
    isolation = execution.IsolationObservationV5c3(
        vm_id="synthetic-disposable-vm",
        disk_id="synthetic-unique-working-disk",
        network_interfaces=(),
        shared_paths=(),
        secret_names=(),
        host_sockets=(),
        production_access=False,
        read_only_sources=True,
        separate_output=True,
        disposable_disk=True,
        stop_available=True,
        limits=execution.ExecutionLimitsV5c3(),
        enforced_limits=execution.RESOURCE_CONTROLS,
    )
    observation = execution.ControllerObservationV5c3(
        isolation=isolation,
        tool_image_id=policy.tool_image_id,
        tool_image_sha256=policy.tool_image_sha256,
        runtime_files=(),
        complete=True,
        runtime_tree_complete=True,
        runtime_tree_immutable=True,
    )
    # This replacement is deliberately limited to mechanism tests. It supplies
    # no actual target bytes and cannot promote the SYNTHETIC collector origin.
    monkeypatch.setattr(execution, "validate_runtime_identity_v5c3", lambda *a, **k: ())
    collector = SyntheticCollector(observation)
    collector.runtime_accesses = tuple(
        (pin.path, pin.sha256) for pin in inventory.files if pin.sha256 is not None
    )
    return ExecutionCase(deb, request, inventory, collector)


def _statuses_remain_blocked(result: Any) -> None:
    assert result.stage_9 == "IN_PROGRESS"
    assert result.qualification == "BLOCKED"
    assert result.observer_3a == "BLOCKED"
    assert result.stage_3b == "NOT AUTHORIZED"
    assert result.production == "NO-GO"
    assert result.external_authority_acceptance == "UNVERIFIED"
    assert result.independently_bound_evidence == "UNVERIFIED"
    assert result.accepted_reconstruction_evidence == "UNVERIFIED"
    assert result.reconstruction == "UNVERIFIED"
    assert result.exact_target_test == "NOT_TESTED"
    assert result.authenticated_time == "UNVERIFIED"
    assert result.durable_anti_rollback == "BLOCKED"
    assert result.runtime_qualification == "BLOCKED"
    assert result.runtime_closure == "UNVERIFIED"


def _refused(result: Any) -> None:
    assert result.consistency not in {
        "SYNTHETIC_MECHANISM_MATCH",
        "LOCAL_EXECUTION_OBSERVATION",
    }, result
    assert result.reason_codes
    assert result.files == ()
    _statuses_remain_blocked(result)


def test_actual_synthetic_deb_and_static_expected_output_are_measured(case: ExecutionCase) -> None:
    result = case.reconstruct()
    assert result.consistency == "SYNTHETIC_MECHANISM_MATCH", result.reason_codes
    assert result.reason_codes == ()
    assert len(result.files) == 1
    record = result.files[0]
    assert record.source_sha256 == _sha(_SOURCE)
    assert record.source_size == len(_SOURCE)
    assert record.output.sha256 == _PYC_SHA256
    assert record.output.size == len(_PYC)
    assert record.output.magic_hex == "cb0d0d0a"
    assert record.output.flags == 3
    assert record.output.source_hash_hex == _SOURCE_HASH
    assert record.output.opaque_body_sha256 == _sha(_PYC[16:])
    assert record.comparison == "OUTPUT_BYTES_MATCH"
    assert record.collector_origin == "SYNTHETIC"
    assert record.clock == record.integrity == "LOCAL_UNAUTHENTICATED"
    assert record.guest_report_origin == "GUEST_REPORTED"
    assert record.record_complete is True
    assert record.repository_commit == "c" * 40
    assert record.arguments == execution.ARGUMENTS
    assert record.environment == execution.ENVIRONMENT
    assert record.run_id and len(record.nonce) == 64
    assert record.started_local and record.ended_local
    assert case.collector.sources == [_SOURCE]
    assert case.collector.inspect_calls >= 2
    assert case.collector.execute_calls == case.collector.stop_calls == 1
    _statuses_remain_blocked(result)


def test_absent_prepared_vm_blocks_exact_execution(case: ExecutionCase) -> None:
    result = case.reconstruct(collector=None)
    assert result.consistency == "EXECUTION_BLOCKED"
    assert case.collector.execute_calls == 0
    _refused(result)


@pytest.mark.parametrize(
    "collector_id", ["synthetic-unit-test-collector", "approved-vm-controller"]
)
def test_self_declared_controller_origin_is_blocked(case: ExecutionCase, collector_id: str) -> None:
    case.collector.collector_id = collector_id
    case.collector.origin = "CONTROLLER_OBSERVED"
    result = case.reconstruct()
    assert result.consistency == "EXECUTION_BLOCKED"
    assert result.reason_codes == ("COLLECTOR_IDENTITY_UNVERIFIED",)
    assert result.diagnostic_records == ()
    assert case.collector.inspect_calls == 0
    assert case.collector.execute_calls == 0
    assert case.collector.stop_calls == 0
    _refused(result)
    payload = json.loads(execution.serialize_reconstruction_result_v5c3(result))["payload"]
    assert payload["consistency"] == "EXECUTION_BLOCKED"
    assert payload["exact_target_test"] == "NOT_TESTED"
    assert payload["files"] == []


def test_changing_origin_during_synthetic_run_cannot_promote_evidence(case: ExecutionCase) -> None:
    def claim_controller(process: Any) -> Any:
        case.collector.origin = "CONTROLLER_OBSERVED"
        return process

    case.collector.mutate_process = claim_controller
    result = case.reconstruct()
    assert result.consistency == "SYNTHETIC_MECHANISM_MATCH"
    assert result.files[0].collector_origin == "SYNTHETIC"
    assert case.collector.execute_calls == case.collector.stop_calls == 1
    _statuses_remain_blocked(result)


def test_manually_claimed_local_observation_cannot_be_exported(case: ExecutionCase) -> None:
    claimed = replace(case.reconstruct(), consistency="LOCAL_EXECUTION_OBSERVATION")
    with pytest.raises(execution.ExecutionRefusalV5c3) as caught:
        execution.serialize_reconstruction_result_v5c3(claimed)
    assert caught.value.execution_reason == "RESULT_STATUS_REJECTED"


@pytest.mark.parametrize("origin", ["GUEST_REPORTED", "unknown", "", None])
def test_guest_or_unknown_report_cannot_authorize_a_run(case: ExecutionCase, origin: Any) -> None:
    case.collector.origin = origin
    _refused(case.reconstruct())
    assert case.collector.execute_calls == 0


def test_json_process_report_is_not_a_controller_adapter(case: ExecutionCase) -> None:
    forged = {"origin": "CONTROLLER_OBSERVED", "complete": True, "output_sha256": _PYC_SHA256}
    _refused(case.reconstruct(collector=forged))


def test_v5c2_declaration_does_not_become_execution_record(case: ExecutionCase) -> None:
    claimed = declaration.PycReconstructionEvidenceV5c2(
        case.request.policy,
        _PYC_SHA256,
        "forged-run",
        "forged-evidence",
        "trusted-signed-proof-claim",
        False,
        None,
    )
    _refused(case.reconstruct(requests=(claimed,)))
    assert case.collector.execute_calls == 0


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("network_interfaces", ("eth0",)),
        ("shared_paths", ("/home/operator",)),
        ("secret_names", ("production-secret",)),
        ("host_sockets", ("/run/libvirt/libvirt-sock",)),
        ("production_access", True),
        ("read_only_sources", False),
        ("separate_output", False),
        ("disposable_disk", False),
        ("stop_available", False),
        ("enforced_limits", ("cpu", "memory", "wall_time")),
        ("vm_id", ""),
        ("disk_id", ""),
    ],
)
def test_isolation_failure_refuses_before_execute(
    case: ExecutionCase, field: str, value: Any
) -> None:
    observation = case.collector.observation
    case.collector.observation = replace(
        observation, isolation=replace(observation.isolation, **{field: value})
    )
    _refused(case.reconstruct())
    assert case.collector.execute_calls == 0


@pytest.mark.parametrize("field", ["cpu_count", "memory_bytes", "wall_time_s", "process_count"])
def test_unenforced_resource_setting_refuses(case: ExecutionCase, field: str) -> None:
    observation = case.collector.observation
    limits = replace(observation.isolation.limits, **{field: 2})
    case.collector.observation = replace(
        observation, isolation=replace(observation.isolation, limits=limits)
    )
    _refused(case.reconstruct())
    assert case.collector.execute_calls == 0


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("interpreter_sha256", "a" * 64),
        ("interpreter_build_id", "a" * 40),
        ("parser_version", "CPython-3.11.9"),
        ("invalidation", "unchecked_hash"),
        ("optimization", 3),
        ("arguments", ("/usr/bin/python3.11", "-c", "compile('x', '', 'exec')")),
        ("environment", (("LD_PRELOAD", "/unapproved.so"),)),
        ("isolation_requirements", ("trust-guest-report",)),
    ],
)
def test_wrong_policy_cannot_invoke_collector(case: ExecutionCase, field: str, value: Any) -> None:
    request = replace(case.request, policy=replace(case.request.policy, **{field: value}))
    _refused(
        case.reconstruct(requests=(request,), expected_requests_sha256=_requests_digest((request,)))
    )
    assert case.collector.execute_calls == 0


@pytest.mark.parametrize("field", ["package", "version", "architecture", "snapshot_sha256"])
def test_interpreter_package_identity_is_checked(case: ExecutionCase, field: str) -> None:
    package = replace(case.request.policy.interpreter_package, **{field: "wrong"})
    policy = replace(case.request.policy, interpreter_package=package)
    request = replace(case.request, policy=policy)
    _refused(
        case.reconstruct(requests=(request,), expected_requests_sha256=_requests_digest((request,)))
    )
    assert case.collector.execute_calls == 0


@pytest.mark.parametrize(
    "field",
    [
        "tool_image_id",
        "tool_image_sha256",
        "complete",
        "runtime_tree_complete",
        "runtime_tree_immutable",
    ],
)
def test_incomplete_or_wrong_tool_observation_refuses(case: ExecutionCase, field: str) -> None:
    value: Any = "wrong" if field in {"tool_image_id", "tool_image_sha256"} else False
    case.collector.observation = replace(case.collector.observation, **{field: value})
    _refused(case.reconstruct())
    assert case.collector.execute_calls == 0


def test_runtime_inventory_cannot_be_substituted_under_old_digest(case: ExecutionCase) -> None:
    changed = replace(case.runtime_inventory, import_paths=("/host/lib/python3.12",))
    _refused(case.reconstruct(expected_runtime_inventory=changed))
    assert case.collector.execute_calls == 0


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("co_filename", "/other.py"),
        ("optimization", 1),
        ("expected_image_sha256", "a" * 64),
        ("expected_pyc_sha256", "a" * 64),
    ],
)
def test_declaration_substitution_is_refused_under_independent_request_pin(
    case: ExecutionCase, field: str, value: Any
) -> None:
    request = replace(case.request, policy=replace(case.request.policy, **{field: value}))
    _refused(case.reconstruct(requests=(request,)))
    assert case.collector.execute_calls == 0


def test_target_path_substitution_is_refused_under_independent_request_pin(
    case: ExecutionCase,
) -> None:
    request = replace(case.request, target_path="/other/module.cpython-312.pyc")
    _refused(case.reconstruct(requests=(request,)))
    assert case.collector.execute_calls == 0


def test_snapshot_declaration_substitution_is_refused_under_independent_request_pin(
    case: ExecutionCase,
) -> None:
    policy = case.request.policy
    substituted = replace(
        policy,
        source=replace(policy.source, snapshot_sha256="a" * 64),
        interpreter_package=replace(policy.interpreter_package, snapshot_sha256="a" * 64),
    )
    _refused(case.reconstruct(requests=(replace(case.request, policy=substituted),)))
    assert case.collector.execute_calls == 0


@pytest.mark.parametrize("mutation", ["archive", "inventory-pin", "no-source", "source", "pep552"])
def test_source_chain_failure_prevents_execution(case: ExecutionCase, mutation: str) -> None:
    kwargs: dict[str, Any] = {}
    if mutation == "archive":
        archive = case.deb.archives[0]
        kwargs["archives"] = (replace(archive, content=archive.content[:-1] + b"x"),)
    elif mutation == "inventory-pin":
        kwargs["expected_inventory_sha256"] = "f" * 64
    elif mutation == "no-source":
        kwargs["archives"] = ()
    elif mutation == "source":
        source = replace(case.request.policy.source, sha256="f" * 64)
        policy = replace(case.request.policy, source=source)
        kwargs["requests"] = (replace(case.request, policy=policy),)
    else:
        kwargs["requests"] = (replace(case.request, expected_source_hash_hex="00" * 8),)
    if "requests" in kwargs:
        kwargs["expected_requests_sha256"] = _requests_digest(kwargs["requests"])
    _refused(case.reconstruct(**kwargs))
    assert case.collector.execute_calls == 0


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("run_id", "forged-run"),
        ("nonce", "00" * 32),
        ("arguments", ("/usr/bin/python3.11",)),
        ("environment", (("LD_LIBRARY_PATH", "/host/lib"),)),
        ("input_sha256", "f" * 64),
        ("exit_code", 1),
        ("exit_code", None),
        ("output", None),
        ("complete", False),
        ("interrupted", True),
        ("elapsed_s", 31),
        ("elapsed_s", float("nan")),
        ("elapsed_s", float("inf")),
        ("runtime_accesses_complete", False),
        ("runtime_accesses", ()),
        ("runtime_accesses", (("/usr/bin/python3.12", source_hash.TARGET_INTERPRETER_SHA256),)),
        ("runtime_accesses", (("/host/lib/unknown.py", "a" * 64),)),
        ("runtime_accesses", (("/usr/bin/python3.12", "a" * 64),)),
        ("runtime_accesses", (("/usr/lib/libunapproved.so", "a" * 64),)),
        ("mapped_elf_paths", ()),
        ("mapped_elf_paths", ("/usr/lib/libunapproved.so",)),
    ],
)
def test_process_measurement_failure_retains_no_positive_record(
    case: ExecutionCase, field: str, value: Any
) -> None:
    case.collector.mutate_process = lambda process: replace(process, **{field: value})
    _refused(case.reconstruct())
    assert case.collector.stop_calls == 1


@pytest.mark.parametrize(
    "output",
    [
        _PYC[:-1] + b"x",
        b"wrong-magic" + _PYC[11:],
        _PYC[:4] + bytes.fromhex("01000000") + _PYC[8:],
        _PYC[:8] + b"12345678" + _PYC[16:],
        b"short",
    ],
)
def test_full_measured_output_is_not_replaced_by_claimed_digest(
    case: ExecutionCase, output: bytes
) -> None:
    case.collector.outputs = [output]
    result = case.reconstruct()
    _refused(result)
    assert result.consistency == "RECONSTRUCTION_MISMATCH"
    assert case.collector.stop_calls == 1


@pytest.mark.parametrize("field", ["stdout", "stderr"])
def test_oversized_untrusted_stream_is_not_retained_in_diagnostics(
    case: ExecutionCase, field: str
) -> None:
    unsafe = b"x" * 65537
    case.collector.mutate_process = lambda process: replace(process, **{field: unsafe})
    result = case.reconstruct()
    _refused(result)
    assert result.consistency == "EXECUTION_FAILED"
    assert result.reason_codes == ("PROCESS_STREAM_LIMIT_EXCEEDED",)
    assert result.diagnostic_records == ()
    exported = execution.serialize_reconstruction_result_v5c3(result)
    assert unsafe not in exported
    assert json.loads(exported)["payload"]["diagnostic_records"] == []
    assert case.collector.stop_calls == 1


@pytest.mark.parametrize("mutation", ["files", "path", "resolutions"])
def test_runtime_inventory_is_bounded_before_recursive_serialization(
    case: ExecutionCase, monkeypatch: pytest.MonkeyPatch, mutation: str
) -> None:
    from deployment.stage9_observer_contract import ContractRefusal

    inventory = case.runtime_inventory
    if mutation == "files":
        inventory = replace(inventory, files=inventory.files * 103)
    elif mutation == "path":
        pin = replace(inventory.files[0], path="/" + "x" * 4097)
        inventory = replace(inventory, files=(pin,) + inventory.files[1:])
    else:
        inventory = replace(
            inventory,
            needed_resolutions=(("/usr/bin/python3.12", "libc.so.6", "/unknown/libc.so.6"),) * 4097,
        )

    def forbidden(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("unbounded inventory reached recursive serialization")

    monkeypatch.setattr(execution, "asdict", forbidden)
    with pytest.raises(ContractRefusal):
        execution.runtime_inventory_sha256_v5c3(inventory)


@pytest.mark.parametrize("error", [TimeoutError(), InterruptedError(), RuntimeError("failure")])
def test_process_exception_stops_vm_and_discards_evidence(
    case: ExecutionCase, error: BaseException
) -> None:
    case.collector.execution_error = error
    _refused(case.reconstruct())
    assert case.collector.stop_calls == 1


def test_changed_postrun_isolation_invalidates_output(case: ExecutionCase) -> None:
    observation = case.collector.observation
    case.collector.after_observation = replace(
        observation,
        isolation=replace(observation.isolation, network_interfaces=("unexpected-nic",)),
    )
    _refused(case.reconstruct())
    assert case.collector.stop_calls == 1


def test_failed_vm_stop_cannot_preserve_positive_result(case: ExecutionCase) -> None:
    case.collector.stop_result = False
    _refused(case.reconstruct())
    assert case.collector.stop_calls == 1


def test_reported_short_runtime_cannot_override_controller_clock(
    case: ExecutionCase, monkeypatch: pytest.MonkeyPatch
) -> None:
    clock = {"now": 100.0}
    monkeypatch.setattr(execution.time, "monotonic", lambda: clock["now"])
    execute = case.collector.execute

    def slow(request_raw: bytes, source: bytes, timeout_s: float) -> Any:
        receipt = execute(request_raw, source, timeout_s)
        clock["now"] = 131.0
        return receipt

    monkeypatch.setattr(case.collector, "execute", slow)
    result = case.reconstruct()
    _refused(result)
    assert "EXECUTION_TIMEOUT" in result.reason_codes
    assert result.consistency == "EXECUTION_FAILED"
    assert case.collector.stop_calls == 1


def test_second_output_failure_discards_first_positive_measurement(case: ExecutionCase) -> None:
    second = replace(case.request, target_path="/cache/second.cpython-312.pyc")
    case.collector.outputs = [_PYC, _PYC[:-1] + b"x"]
    requests = (case.request, second)
    result = case.reconstruct(
        requests=requests, expected_requests_sha256=_requests_digest(requests)
    )
    _refused(result)
    assert result.diagnostic_records
    assert all(record.comparison == "SET_INVALIDATED" for record in result.diagnostic_records)
    assert case.collector.execute_calls == 2
    assert case.collector.stop_calls == 1


def test_complete_declared_set_has_distinct_run_bindings(case: ExecutionCase) -> None:
    second = replace(case.request, target_path="/cache/second.cpython-312.pyc")
    requests = (case.request, second)
    result = case.reconstruct(
        requests=requests, expected_requests_sha256=_requests_digest(requests)
    )
    assert result.consistency == "SYNTHETIC_MECHANISM_MATCH", result.reason_codes
    assert len(result.files) == 2
    assert len({record.run_id for record in result.files}) == 2
    assert len({record.nonce for record in result.files}) == 2
    assert {record.target_path for record in result.files} == {
        case.request.target_path,
        second.target_path,
    }
    assert all(record.comparison == "OUTPUT_BYTES_MATCH" for record in result.files)
    assert case.collector.execute_calls == 2
    assert case.collector.stop_calls == 1
    _statuses_remain_blocked(result)


def test_runtime_validator_refusal_blocks_before_execution(
    case: ExecutionCase, monkeypatch: pytest.MonkeyPatch
) -> None:
    from deployment.stage9_observer_contract import ContractRefusal

    def reject(*args: Any, **kwargs: Any) -> None:
        raise ContractRefusal("LIBRARY_MISSING")

    monkeypatch.setattr(execution, "validate_runtime_identity_v5c3", reject)
    _refused(case.reconstruct())
    assert case.collector.execute_calls == 0


def test_no_source_bytecode_or_host_subprocess_execution(
    case: ExecutionCase, monkeypatch: pytest.MonkeyPatch
) -> None:
    import builtins
    import importlib.util
    import marshal
    import os
    import subprocess

    def forbidden(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("orchestrator attempted host execution or deserialization")

    with monkeypatch.context() as guards:
        for module, name in (
            (builtins, "compile"),
            (importlib.util, "source_from_cache"),
            (marshal, "loads"),
            (os, "system"),
            (subprocess, "Popen"),
            (subprocess, "run"),
        ):
            guards.setattr(module, name, forbidden)
        result = case.reconstruct()
    assert result.consistency == "SYNTHETIC_MECHANISM_MATCH", result.reason_codes
    _statuses_remain_blocked(result)


def test_execution_export_has_independently_recomputed_local_digest(
    case: ExecutionCase,
) -> None:
    case.collector.mutate_process = lambda process: replace(
        process, stdout=b"\x00\xff\x80guest output", stderr=b"\xfeerror\r\n"
    )
    result = case.reconstruct()
    raw = execution.serialize_reconstruction_result_v5c3(result)
    exported = json.loads(raw)
    assert exported["format"] == "stage9-observer-pyc-execution-export-v5c3"
    assert exported["integrity_origin"] == "LOCAL_UNAUTHENTICATED"
    payload = exported["payload"]
    assert exported["payload_sha256"] == _sha(canonical_json_bytes(payload))
    assert payload["consistency"] == "SYNTHETIC_MECHANISM_MATCH"
    assert payload["exact_target_test"] == "NOT_TESTED"
    assert payload["accepted_reconstruction_evidence"] == "UNVERIFIED"
    record = payload["files"][0]
    assert bytes.fromhex(record["stdout_hex"]) == b"\x00\xff\x80guest output"
    assert bytes.fromhex(record["stderr_hex"]) == b"\xfeerror\r\n"
    assert "stdout" not in record and "stderr" not in record and "elapsed_s" not in record
    assert record["reported_elapsed_ns"] == 10_000_000
    assert record["record_complete"] is True


def test_failed_set_export_has_no_positive_records(case: ExecutionCase) -> None:
    case.collector.mutate_process = lambda process: replace(process, exit_code=17)
    result = case.reconstruct()
    _refused(result)
    exported = json.loads(execution.serialize_reconstruction_result_v5c3(result))
    payload = exported["payload"]
    assert payload["files"] == []
    assert payload["diagnostic_records"]
    assert all(
        record["comparison"] == "SET_INVALIDATED" for record in payload["diagnostic_records"]
    )
    assert exported["payload_sha256"] == _sha(canonical_json_bytes(payload))
