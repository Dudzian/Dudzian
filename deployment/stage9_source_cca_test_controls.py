"""Bounded TEST ONLY refusals around fresh live Source/CCA positive controls.

Physical file/service stimuli, faulted expectations, and injected observations
are reported separately. None of them supplies production approval or an
independent physical-host attestation.
"""

from __future__ import annotations

from contextlib import contextmanager
import hashlib
import json
from pathlib import Path
from typing import Any, Callable, ContextManager, Iterator, Protocol, cast

from deployment import stage9_trusted_installation_identity_gate as gate
from deployment.stage9_source_cca_test_package import HarnessRefusal
from deployment.stage9_trusted_installation_contract import RoleExpectation
from deployment.stage9_trusted_installation_files import PinnedFile
from deployment.stage9_trusted_installation_runtime import observe_service

_ROLES = frozenset({"root_source", "history_source", "cca"})
_BLOCKED = {
    "status": "BLOCKED",
    "qualification": "NOT_QUALIFIED",
    "production_emission": "BLOCKED",
}


class ControlRun(Protocol):
    resources: list[dict[str, Any]]

    @property
    def manifest_path(self) -> str | Path: ...

    @property
    def pin_path(self) -> str | Path: ...

    def assert_owned_file(self, path: str) -> Any: ...

    def write_owned(self, path: str, data: bytes, mode: int | None = None) -> None: ...

    def chmod_owned(self, path: str, mode: int) -> None: ...

    def replace_owned(
        self,
        path: str,
        replacement: bytes | None = None,
        symlink_target: str | None = None,
    ) -> ContextManager[None]: ...

    def test_dropin(self, role: dict[str, Any], data: bytes) -> ContextManager[None]: ...

    def stop(self, role: dict[str, Any]) -> None: ...

    def start_role(self, role: dict[str, Any]) -> None: ...


def _json(value: Any) -> bytes:
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n"
    ).encode()


def report_sha256(report: dict[str, Any]) -> str:
    """Hash canonical redacted report bytes, never environment or log contents."""
    return hashlib.sha256(_json(report)).hexdigest()


def _blocked(report: dict[str, Any]) -> bool:
    return all(report.get(key) == value for key, value in _BLOCKED.items())


def require_matched(report: dict[str, Any]) -> None:
    """A positive comparison must include all roles and keep every blocker."""
    roles = report.get("role_checks")
    if (
        not _blocked(report)
        or report.get("snapshot") != "UNTRUSTED_LIVE_OBSERVATION"
        or report.get("installation_checks") != "MATCHED"
        or report.get("observed_failures") != []
        or type(roles) is not dict
        or set(roles) != _ROLES
        or any(type(item) is not dict or item.get("checks") != "MATCHED" for item in roles.values())
        or not report.get("blockers")
    ):
        raise HarnessRefusal("CONTROL_POSITIVE_FAILED")


def _positive(run: ControlRun) -> dict[str, Any]:
    result = gate.inspect_installation(str(run.manifest_path), approval_pin_path=str(run.pin_path))
    require_matched(result)
    return cast(dict[str, Any], result)


def _read(path: str | Path, *, max_bytes: int = 65536) -> bytes:
    with PinnedFile(str(path), max_bytes=max_bytes) as checked:
        return cast(bytes, checked.data)


@contextmanager
def _expectation_fault(
    run: ControlRun, change: Callable[[dict[str, Any]], None], *, rebind: bool = True
) -> Iterator[None]:
    """Change only this run's synthetic expectations and restore exact bytes."""
    original = _read(run.manifest_path)
    original_pin = _read(run.pin_path, max_bytes=4096)
    document = json.loads(original)
    change(document)
    changed = _json(document)
    pin = json.loads(original_pin)
    pin["manifest_sha256"] = hashlib.sha256(changed).hexdigest()
    try:
        run.write_owned(str(run.manifest_path), changed)
        if rebind:
            run.write_owned(str(run.pin_path), _json(pin))
        yield
    finally:
        # Restore both resources even if the first restoration is refused.
        try:
            run.write_owned(str(run.manifest_path), original)
        finally:
            run.write_owned(str(run.pin_path), original_pin)


@contextmanager
def _permission_fault(run: ControlRun, role: dict[str, Any]) -> Iterator[None]:
    path = role["executable"]
    try:
        run.chmod_owned(path, 0o777)
        yield
    finally:
        run.chmod_owned(path, 0o755)


@contextmanager
def _stopped(run: ControlRun, role: dict[str, Any]) -> Iterator[None]:
    try:
        run.stop(role)
        yield
    finally:
        run.start_role(role)


def _injected_report(run: ControlRun, stimulus: str) -> dict[str, Any]:
    """Fault a bounded snapshot only after the real observer succeeded.

    These cases exercise the real gate's snapshot validation and cross-sweep
    comparisons, without claiming that a physical PID or namespace was changed.
    """
    counts: dict[str, int] = {}

    def observer(role: RoleExpectation, descriptor: int) -> dict[str, Any]:
        snapshot = observe_service(role, descriptor)
        counts[role.role] = counts.get(role.role, 0) + 1
        result = {**snapshot, "namespaces": dict(snapshot["namespaces"])}
        if stimulus == "pid_starttime" and counts[role.role] == 2:
            result["start_time_ticks"] += 1
        elif stimulus == "shared_namespace":
            result["namespaces"]["net"] = 1
        elif stimulus == "incomplete_observation":
            result.pop("groups")
        return cast(dict[str, Any], result)

    with (
        PinnedFile(str(run.manifest_path), max_bytes=65536) as manifest,
        PinnedFile(str(run.pin_path), max_bytes=4096) as pin,
    ):
        result = gate._inspect(
            manifest.data,
            pin.data,
            file_factory=PinnedFile,
            service_observer=observer,
            synthetic=True,
            reserved_identities=(manifest.identity, pin.identity),
        )
        manifest.recheck()
        pin.recheck()
        return cast(dict[str, Any], result)


def _evaluate(
    run: ControlRun,
    name: str,
    evidence: str,
    expected: frozenset[str],
    fault: Callable[[], ContextManager[None]] | None = None,
) -> dict[str, Any]:
    before = _positive(run)
    if fault is None:
        negative = _injected_report(run, name)
    else:
        with fault():
            negative = gate.inspect_installation(
                str(run.manifest_path), approval_pin_path=str(run.pin_path)
            )
    # A wrong refusal, disappearance of a baseline, or incomplete restoration
    # blocks the test. A cleanup message is never substituted for this read.
    after = _positive(run)
    failures = negative.get("observed_failures")
    if (
        not _blocked(negative)
        or negative.get("installation_checks") == "MATCHED"
        or type(failures) is not list
        or not failures
        or not set(failures).issubset(expected)
        or (
            evidence == "OBSERVATION_FAULT_TEST"
            and negative.get("snapshot") != "SYNTHETIC_TEST_INPUT"
        )
    ):
        raise HarnessRefusal("CONTROL_NEGATIVE_FAILED")
    return {
        "name": name,
        "evidence": evidence,
        "result": "REFUSED",
        "positive_before": "MATCHED",
        "positive_after": "MATCHED",
        "positive_before_sha256": report_sha256(before),
        "negative_report_sha256": report_sha256(negative),
        "positive_after_sha256": report_sha256(after),
        "gate": negative,
    }


def run_controls(run: ControlRun) -> list[dict[str, Any]]:
    """Run thirteen refusals; mutate only disposable resources owned by run."""
    if (
        type(run.resources) is not list
        or len(run.resources) != 3
        or {item.get("role") for item in run.resources} != _ROLES
    ):
        raise HarnessRefusal("CONTROL_RUNTIME_INVALID")
    role = next(item for item in run.resources if item["role"] == "root_source")
    other = next(item for item in run.resources if item["role"] == "history_source")
    # The replacement is another reviewed harmless worker and is never started.
    replacement = _read(other["executable"], max_bytes=64 * 1024 * 1024)

    def change_role(key: str, value: Any) -> Callable[[dict[str, Any]], None]:
        def change(document: dict[str, Any]) -> None:
            target = next(item for item in document["roles"] if item["role"] == "root_source")
            target[key] = value

        return change

    def changed_digest(document: dict[str, Any]) -> None:
        target = next(item for item in document["roles"] if item["role"] == "root_source")
        digest = target["executable"]["sha256"]
        target["executable"]["sha256"] = ("1" if digest[0] == "0" else "0") + digest[1:]

    def changed_identity(document: dict[str, Any]) -> None:
        target = next(item for item in document["roles"] if item["role"] == "root_source")
        target["uid"] = 65001
        target["gid"] = 65001

    specifications: list[tuple[str, str, frozenset[str], Callable[[], ContextManager[None]]]] = [
        (
            "executable_replacement",
            "PHYSICAL_FILE_FAULT_TEST",
            frozenset({"INSTALLATION_DIGEST_MISMATCH", "EXECUTABLE_OBJECT_MISMATCH"}),
            lambda: run.replace_owned(role["executable"], replacement=replacement),
        ),
        (
            "digest",
            "FAULTED_TEST_EXPECTATION",
            frozenset({"INSTALLATION_DIGEST_MISMATCH"}),
            lambda: _expectation_fault(run, changed_digest),
        ),
        (
            "dropin",
            "PHYSICAL_SYSTEMD_FAULT_TEST",
            frozenset({"UNEXPECTED_SYSTEMD_DROPINS"}),
            lambda: run.test_dropin(role, b"[Service]\nEnvironment=CH_S9_TEST_ONLY=1\n"),
        ),
        (
            "uid_gid",
            "FAULTED_TEST_EXPECTATION",
            frozenset({"NSS_IDENTITY_MISMATCH"}),
            lambda: _expectation_fault(run, changed_identity),
        ),
        (
            "unauthorized_group",
            "FAULTED_TEST_EXPECTATION",
            frozenset({"MANIFEST_SUPPLEMENTARY_GROUPS_FORBIDDEN"}),
            lambda: _expectation_fault(run, change_role("supplementary_groups", [65534])),
        ),
        (
            "symlink",
            "PHYSICAL_FILE_FAULT_TEST",
            frozenset({"FILE_METADATA_UNSAFE", "FILE_UNAVAILABLE"}),
            lambda: run.replace_owned(role["executable"], symlink_target=other["executable"]),
        ),
        (
            "permissions",
            "PHYSICAL_FILE_FAULT_TEST",
            frozenset({"FILE_METADATA_UNSAFE"}),
            lambda: _permission_fault(run, role),
        ),
        (
            "pin_binding",
            "FAULTED_TEST_EXPECTATION",
            frozenset({"APPROVAL_PIN_MISMATCH"}),
            lambda: _expectation_fault(run, changed_digest, rebind=False),
        ),
        (
            "unavailable_process",
            "PHYSICAL_PROCESS_FAULT_TEST",
            frozenset({"SYSTEMD_IDENTITY_MISMATCH", "PROCESS_UNAVAILABLE"}),
            lambda: _stopped(run, role),
        ),
        (
            "cgroup",
            "FAULTED_TEST_EXPECTATION",
            frozenset({"SYSTEMD_IDENTITY_MISMATCH", "PROCESS_CGROUP_MISMATCH"}),
            lambda: _expectation_fault(
                run, change_role("control_group", "/unexpected.slice/" + role["unit"])
            ),
        ),
    ]
    reports = [_evaluate(run, *specification) for specification in specifications]
    for name, expected in (
        ("pid_starttime", "SERVICE_CHANGED_DURING_CHECK"),
        ("shared_namespace", "SHARED_PROCESS_NAMESPACE"),
        ("incomplete_observation", "INVALID_SERVICE_OBSERVATION"),
    ):
        reports.append(_evaluate(run, name, "OBSERVATION_FAULT_TEST", frozenset({expected})))
    return reports
