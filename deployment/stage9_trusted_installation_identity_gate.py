"""Read-only Source/CCA installation comparison, never production authorization.

A protected digest pin binds expectations to separate supplied bytes. Its local
ownership cannot prove independent approval, so this first-stage gate has no
qualified trust provider and always retains NOT_QUALIFIED / production BLOCKED.
"""

from __future__ import annotations

from contextlib import ExitStack
import hmac
import json
import re
import sys
from typing import Any, Callable

from deployment.stage9_trusted_installation_contract import (
    MAX_MANIFEST_BYTES,
    InstallationRefusal,
    Manifest,
    RoleExpectation,
    parse_manifest,
)
from deployment.stage9_trusted_installation_files import FileRefusal, PinnedFile
from deployment.stage9_trusted_installation_runtime import RuntimeRefusal, observe_service

_PIN_SCHEMA = "CryptoHunter.SourceCCAInstallationApprovalPinV1"
_MAX_PIN_BYTES = 4096
_MAX_UNIT_BYTES = 128 * 1024
_MAX_EXECUTABLE_BYTES = 64 * 1024 * 1024
_IDENTIFIER = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}\Z")
_CODES = frozenset(
    {
        "INDEPENDENT_APPROVAL_PIN_MISSING",
        "APPROVAL_PIN_INVALID",
        "APPROVAL_PIN_MISMATCH",
        "INPUT_PATHS_NOT_INDEPENDENT",
        "INSTALLATION_DIGEST_MISMATCH",
        "INVALID_SERVICE_OBSERVATION",
        "SHARED_PROCESS_IDENTITY",
        "SHARED_PROCESS_NAMESPACE",
        "SERVICE_CHANGED_DURING_CHECK",
        "BACKEND_OBSERVATION_FAILED",
        "CLI_ARGUMENT_INVALID",
    }
)
_BLOCKERS = (
    "INDEPENDENT_MANIFEST_APPROVAL_PROVENANCE_NOT_QUALIFIED",
    "PHYSICAL_HOST_SYSTEMD_PROCFS_AND_NSS_TRUST_NOT_QUALIFIED",
    "SOCKET_BOUND_SO_PEERPIDFD_AUTHENTICATION_NOT_QUALIFIED",
    "CROSS_ROLE_EFFECTIVE_ACCESS_AND_EXTERNAL_PRINCIPAL_ISOLATION_NOT_QUALIFIED",
    "RUNTIME_CODE_IMPORTS_LIBRARIES_AND_MOUNT_PROVENANCE_NOT_QUALIFIED",
    "CUSTODY_KEYRING_AND_CREDENTIAL_ISOLATION_NOT_QUALIFIED",
    "POSTGRESQL_COMMITTED_CLOSE_SOURCE_LEASE_AND_CCA_DURABILITY_NOT_QUALIFIED",
    "TRANSACTION_ADDENDUM_A_C_INTEGRATION_REVIEW_PENDING",
    "PRODUCTION_SIGN_BIND_ENROLLMENT_CCA_PUBLICATION_AND_STAGE10_BLOCKED",
)


class _GateRefusal(ValueError):
    def __init__(self, code: str) -> None:
        self.code = code if type(code) is str and code in _CODES else "BACKEND_OBSERVATION_FAILED"
        super().__init__(self.code)


def _result(*, synthetic: bool) -> dict[str, Any]:
    return {
        "schema": "CryptoHunter.SourceCCAInstallationIdentityGateV1",
        "status": "BLOCKED",
        "qualification": "NOT_QUALIFIED",
        "production_emission": "BLOCKED",
        "snapshot": "SYNTHETIC_TEST_INPUT" if synthetic else "UNTRUSTED_LIVE_OBSERVATION",
        "installation_checks": "NOT_QUALIFIED",
        "trust": "NOT_QUALIFIED",
        "role_checks": {},
        "observed_failures": [],
        "blockers": list(_BLOCKERS),
    }


def _code(exc: Exception) -> str:
    # An arbitrary backend exception's text or `code` attribute is untrusted.
    if isinstance(exc, (InstallationRefusal, FileRefusal, RuntimeRefusal, _GateRefusal)) and type(
        exc
    ) in (InstallationRefusal, FileRefusal, RuntimeRefusal, _GateRefusal):
        return type(exc)(exc.code).code
    return "BACKEND_OBSERVATION_FAILED"


def _pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise _GateRefusal("APPROVAL_PIN_INVALID")
        result[key] = value
    return result


def _constant(_value: str) -> Any:
    raise _GateRefusal("APPROVAL_PIN_INVALID")


def _parse_pin(raw: bytes | None, manifest: Manifest) -> None:
    if raw is None:
        raise _GateRefusal("INDEPENDENT_APPROVAL_PIN_MISSING")
    if type(raw) is not bytes or not 0 < len(raw) <= _MAX_PIN_BYTES:
        raise _GateRefusal("APPROVAL_PIN_INVALID")
    try:
        pin = json.loads(raw.decode("utf-8"), object_pairs_hook=_pairs, parse_constant=_constant)
    except (UnicodeDecodeError, ValueError, RecursionError) as exc:
        raise _GateRefusal("APPROVAL_PIN_INVALID") from exc
    if (
        type(pin) is not dict
        or set(pin) != {"schema", "version", "installation_id", "approval_id", "manifest_sha256"}
        or type(pin["schema"]) is not str
        or pin["schema"] != _PIN_SCHEMA
        or type(pin["version"]) is not int
        or pin["version"] != 1
        or type(pin["installation_id"]) is not str
        or type(pin["approval_id"]) is not str
        or not _IDENTIFIER.fullmatch(pin["approval_id"])
        or type(pin["manifest_sha256"]) is not str
        or not re.fullmatch(r"[0-9a-f]{64}", pin["manifest_sha256"])
    ):
        raise _GateRefusal("APPROVAL_PIN_INVALID")
    if pin["installation_id"] != manifest.installation_id or not hmac.compare_digest(
        pin["manifest_sha256"], manifest.sha256
    ):
        raise _GateRefusal("APPROVAL_PIN_MISMATCH")


def _snapshot(value: Any, role: RoleExpectation) -> dict[str, Any]:
    if type(value) is not dict or set(value) != {
        "pid",
        "start_time_ticks",
        "uid",
        "gid",
        "groups",
        "namespaces",
    }:
        raise _GateRefusal("INVALID_SERVICE_OBSERVATION")
    if (
        type(value["pid"]) is not int
        or not 1 < value["pid"] < 2**31
        or type(value["start_time_ticks"]) is not int
        or not 0 < value["start_time_ticks"] < 2**64
        or type(value["uid"]) is not int
        or value["uid"] != role.uid
        or type(value["gid"]) is not int
        or value["gid"] != role.gid
        or type(value["groups"]) is not list
        or len(value["groups"]) > 1
        or any(type(item) is not int or item != role.gid for item in value["groups"])
        or type(value["namespaces"]) is not dict
        or set(value["namespaces"]) != {"mnt", "ipc", "net"}
        or any(
            type(item) is not int or not 0 < item < 2**64 for item in value["namespaces"].values()
        )
    ):
        raise _GateRefusal("INVALID_SERVICE_OBSERVATION")
    return {
        **value,
        "groups": value["groups"].copy(),
        "namespaces": value["namespaces"].copy(),
    }


def _inspect(
    raw: bytes,
    pin_raw: bytes | None,
    *,
    file_factory: Callable[..., Any],
    service_observer: Callable[..., Any],
    synthetic: bool = True,
    reserved_identities: tuple[tuple[int, int], ...] = (),
) -> dict[str, Any]:
    """Private synthetic harness; injected observations can never grant authority."""
    result = _result(synthetic=synthetic)
    active_role = None
    try:
        manifest = parse_manifest(raw)
        _parse_pin(pin_raw, manifest)
        result["trust"] = "PROTECTED_PIN_MATCHED_PROVENANCE_UNQUALIFIED"
        with ExitStack() as held:
            files = []
            executables = {}
            first_pass = {}
            identities = set(reserved_identities)
            for role in manifest.roles:
                active_role = role.role
                result["role_checks"][active_role] = {"checks": "NOT_QUALIFIED", "issues": []}
                for expectation, limit in (
                    (role.unit_file, _MAX_UNIT_BYTES),
                    (role.executable, _MAX_EXECUTABLE_BYTES),
                ):
                    pinned = held.enter_context(
                        file_factory(
                            expectation.path,
                            uid=expectation.uid,
                            gid=expectation.gid,
                            mode=expectation.mode,
                            max_bytes=limit,
                            retain_data=False,
                        )
                    )
                    files.append(pinned)
                    if pinned.identity in identities:
                        raise _GateRefusal("INPUT_PATHS_NOT_INDEPENDENT")
                    identities.add(pinned.identity)
                    if not hmac.compare_digest(pinned.sha256, expectation.sha256):
                        raise _GateRefusal("INSTALLATION_DIGEST_MISMATCH")
                    if expectation is role.executable:
                        executables[role.role] = pinned
                first_pass[role.role] = _snapshot(
                    service_observer(role, executables[role.role].fd), role
                )
            if len({snapshot["pid"] for snapshot in first_pass.values()}) != 3:
                raise _GateRefusal("SHARED_PROCESS_IDENTITY")
            for namespace in ("mnt", "ipc", "net"):
                if len({item["namespaces"][namespace] for item in first_pass.values()}) != 3:
                    raise _GateRefusal("SHARED_PROCESS_NAMESPACE")
            # Repeat across all roles while retaining every installed file.
            # This detects observed churn, not an atomic or socket-bound snapshot.
            for role in manifest.roles:
                active_role = role.role
                if first_pass[role.role] != _snapshot(
                    service_observer(role, executables[role.role].fd), role
                ):
                    raise _GateRefusal("SERVICE_CHANGED_DURING_CHECK")
            active_role = None
            for pinned in files:
                pinned.recheck()
        # Closing descriptors may also fail; only successful exits reach MATCHED.
        for role in manifest.roles:
            result["role_checks"][role.role]["checks"] = "MATCHED"
        result["installation_checks"] = "MATCHED"
    except Exception as exc:
        code = _code(exc)
        result["observed_failures"].append(code)
        # All role results stay nonqualified if the combined comparison fails.
        for item in result["role_checks"].values():
            item["checks"] = "NOT_QUALIFIED"
        if result["role_checks"] and active_role in result["role_checks"]:
            result["role_checks"][active_role]["issues"].append(code)
    return result


def inspect_installation(
    manifest_path: str | None, *, approval_pin_path: str | None = None
) -> dict[str, Any]:
    """Read separate protected expectations and compare the installed services.

    No pin is generated from this host, and no qualified trust provider exists
    here. Even locally protected matching bytes cannot qualify production.
    """
    result = _result(synthetic=False)
    try:
        if manifest_path is None:
            raise InstallationRefusal("MANIFEST_MISSING")
        if approval_pin_path is None:
            raise _GateRefusal("INDEPENDENT_APPROVAL_PIN_MISSING")
        if manifest_path == approval_pin_path:
            raise _GateRefusal("INPUT_PATHS_NOT_INDEPENDENT")
        with (
            PinnedFile(manifest_path, max_bytes=MAX_MANIFEST_BYTES) as manifest_file,
            PinnedFile(approval_pin_path, max_bytes=_MAX_PIN_BYTES) as pin_file,
        ):
            if manifest_file.identity == pin_file.identity:
                raise _GateRefusal("INPUT_PATHS_NOT_INDEPENDENT")
            manifest = parse_manifest(manifest_file.data)
            installed_paths = {
                item.path for role in manifest.roles for item in (role.unit_file, role.executable)
            }
            if installed_paths.intersection({manifest_path, approval_pin_path}):
                raise _GateRefusal("INPUT_PATHS_NOT_INDEPENDENT")
            result = _inspect(
                manifest_file.data,
                pin_file.data,
                file_factory=PinnedFile,
                service_observer=observe_service,
                synthetic=False,
                reserved_identities=(manifest_file.identity, pin_file.identity),
            )
            manifest_file.recheck()
            pin_file.recheck()
    except Exception as exc:
        result["installation_checks"] = "NOT_QUALIFIED"
        result["trust"] = "NOT_QUALIFIED"
        result["observed_failures"].append(_code(exc))
        for item in result["role_checks"].values():
            item["checks"] = "NOT_QUALIFIED"
    return result


def main(argv: list[str] | None = None) -> int:
    args = sys.argv[1:] if argv is None else argv
    # Do not echo user paths or malformed values through argparse diagnostics.
    fields: dict[str, str] = {}
    valid = type(args) is list and len(args) in (2, 4)
    if valid:
        for index in range(0, len(args), 2):
            key, value = args[index : index + 2]
            if (
                key not in ("--manifest", "--approval-pin")
                or key in fields
                or type(value) is not str
            ):
                valid = False
                break
            fields[key] = value
    if valid and "--manifest" in fields:
        result = inspect_installation(
            fields["--manifest"], approval_pin_path=fields.get("--approval-pin")
        )
    else:
        result = _result(synthetic=False)
        result["observed_failures"].append("CLI_ARGUMENT_INVALID")
    print(json.dumps(result, sort_keys=True))
    # Matching checks have no production approval meaning and remain a refusal.
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
