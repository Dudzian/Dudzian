"""Bounded Stage 9 observer data contracts; parsing never establishes trust.

This module performs no I/O and never executes inspected code. Parsed contract
records hold immutable bytes. Structural acceptance is always UNVERIFIED.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Literal, NoReturn, cast

from deployment.canonical_json import canonical_json_bytes

PURPOSE = "STAGE9_OBSERVER_TEST_ONLY"
ACCEPTANCE_DOMAIN = b"CryptoHunter.Stage9ObserverAcceptanceV1\x00"
MAX_INTEGER = 2**53 - 1
MAX_MANIFEST_BYTES = 4 * 1024 * 1024
MAX_DOCUMENT_BYTES = 65536
MAX_CONTEXT_BYTES = 16384
MAX_DEPTH = 16
MAX_FILES = 32768
MAX_SOURCES = 128
MAX_REASON_CODES = 64
MAX_PATH_BYTES = 4096
REASON_CODES = frozenset(
    {
        "FORMAT_REJECTED",
        "LIMIT_EXCEEDED",
        "TRUST_CONTEXT_MISSING",
        "ACCEPTANCE_REJECTED",
        "LAUNCH_POLICY_MISMATCH",
        "RUNTIME_LAUNCH_UNVERIFIED",
        "HOST_SCOPE_UNVERIFIED",
        "IMPORT_POLICY_REJECTED",
        "CLOSURE_INCOMPLETE",
        "ROLLBACK_REJECTED",
        "ARTIFACT_MISMATCH",
        "UNSUPPORTED_SOURCE",
        "PROVENANCE_INCOMPLETE",
        "FILE_POLICY_REJECTED",
        "FILE_CHANGED",
        "REPORT_REJECTED",
        "CHANNEL_REJECTED",
    }
)


class ContractRefusal(ValueError):
    """An input-independent diagnostic; no input or exception text is exposed."""

    def __init__(self, code: str = "FORMAT_REJECTED") -> None:
        self.code = code if type(code) is str and code in REASON_CODES else "FORMAT_REJECTED"
        super().__init__(self.code)


def _deny(code: str = "FORMAT_REJECTED") -> NoReturn:
    raise ContractRefusal(code)


def _require(condition: bool, code: str = "FORMAT_REJECTED") -> None:
    if not condition:
        _deny(code)


def _string(value: Any, maximum: int = MAX_MANIFEST_BYTES) -> str:
    if type(value) is not str:
        _deny()
    if any(ord(c) == 0 or 0xD800 <= ord(c) <= 0xDFFF for c in value):
        _deny()
    if len(value.encode("utf-8")) > maximum:
        _deny("LIMIT_EXCEEDED")
    return cast(str, value)


def _integer(value: Any, minimum: int = 0, maximum: int = MAX_INTEGER) -> int:
    if type(value) is not int or not minimum <= value <= maximum:
        _deny()
    return cast(int, value)


def _record(value: Any, fields: set[str]) -> dict[str, Any]:
    if type(value) is not dict or set(value) != fields:
        _deny()
    return cast(dict[str, Any], value)


def _list(value: Any, maximum: int = MAX_FILES, minimum: int = 0) -> list[Any]:
    if type(value) is not list:
        _deny()
    if len(value) > maximum:
        _deny("LIMIT_EXCEEDED")
    if len(value) < minimum:
        _deny()
    return cast(list[Any], value)


def _identifier(value: Any) -> str:
    value = _string(value, 128)
    _require(bool(value) and value.isascii())
    return cast(str, value)


def _hex(value: Any, length: int) -> str:
    value = _string(value)
    _require(re.fullmatch(r"[0-9a-f]{%d}" % length, value) is not None)
    return cast(str, value)


def _digest(value: Any) -> str:
    value = _hex(value, 64)
    _require(value != "0" * 64)
    return cast(str, value)


def _enum(value: Any, allowed: set[str] | frozenset[str]) -> str:
    value = _string(value)
    _require(value in allowed)
    return cast(str, value)


def _equal(value: Any, expected: Any) -> None:
    _require(type(value) is type(expected))
    if type(expected) is dict:
        _require(set(value) == set(expected))
        for key in expected:
            _equal(value[key], expected[key])
    elif type(expected) is list:
        _require(len(value) == len(expected))
        for actual, wanted in zip(value, expected):
            _equal(actual, wanted)
    else:
        _require(value == expected)


def _path(value: Any, absolute: bool = True) -> str:
    value = _string(value, MAX_PATH_BYTES)
    _require(bool(value) and "\\" not in value and all(ord(c) >= 32 for c in value))
    _require(value.startswith("/") is absolute)
    if absolute and value == "/":
        return cast(str, value)
    components = value[1:].split("/") if absolute else value.split("/")
    _require(all(c not in {"", ".", ".."} for c in components))
    return cast(str, value)


def _timestamp(value: Any) -> datetime:
    value = _string(value, 20)
    _require(
        re.fullmatch(r"[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}Z", value) is not None
    )
    try:
        return datetime.strptime(value, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
    except ValueError:
        raise ContractRefusal() from None


def _schema(value: dict[str, Any], name: str, purpose: bool = False) -> None:
    _equal(value["schema"], name)
    _equal(value["version"], 1)
    if purpose:
        _equal(value["purpose"], PURPOSE)


def _reason_codes(value: Any) -> list[str]:
    result = [_enum(x, REASON_CODES) for x in _list(value, MAX_REASON_CODES)]
    _require(len(set(result)) == len(result))
    return result


def _blocked_fields() -> dict[str, Any]:
    return {
        "qualification": "BLOCKED",
        "full_confirmation": False,
        "coexistence_authorized": False,
        "preflight_exception": False,
        "physical_test_status": "PHYSICAL_TEST_PENDING",
        "status": "IN_PROGRESS",
        "production_status": "PRODUCTION_NO_GO",
        "runtime_attestation": "MISSING",
        "host_pid_namespace": "UNVERIFIED",
    }


def _preflight(raw: bytes) -> None:
    """Bound depth, integer tokens and container cardinality before the JSON decoder.

    This lexical pass has a stack of at most 16 integers, never recurses and
    allocates no input-sized token lists. JSON syntax is checked by json.loads.
    """
    stack: list[int] = []
    index = 0
    while index < len(raw):
        char = raw[index]
        if char in b" \t\r\n:":
            index += 1
            continue
        if char == 44:
            if stack:
                stack[-1] += 1
                _require(stack[-1] <= MAX_FILES, "LIMIT_EXCEEDED")
            index += 1
            continue
        if char in (93, 125):
            _require(bool(stack))
            stack.pop()
            index += 1
            continue
        if char in (91, 123):
            _require(len(stack) < MAX_DEPTH, "LIMIT_EXCEEDED")
            stack.append(1)
            index += 1
        elif char == 34:
            start = index
            index += 1
            while index < len(raw) and raw[index] != 34:
                index += 2 if raw[index] == 92 else 1
            _require(index < len(raw))
            index += 1
        else:
            start = index
            while index < len(raw) and raw[index] not in b" \t\r\n,]}:":
                index += 1
                _require(index - start <= 16, "LIMIT_EXCEEDED")
            token = raw[start:index]
            _require(
                token in (b"true", b"false") or re.fullmatch(rb"0|[1-9][0-9]*", token) is not None
            )
    _require(not stack)


def _pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        _string(key)
        _require(key not in result)
        result[key] = value
    return result


def _reject_number(value: str) -> NoReturn:
    _deny()


def _parse_integer(value: str) -> int:
    _require(len(value) <= 16, "LIMIT_EXCEEDED")
    return _integer(int(value))


def _decode(raw: bytes, maximum: int = MAX_DOCUMENT_BYTES) -> Any:
    _require(type(raw) is bytes)
    _require(bool(raw))
    _require(len(raw) <= maximum, "LIMIT_EXCEEDED")
    _require(not raw.startswith(b"\xef\xbb\xbf"))
    _preflight(raw)
    try:
        value = json.loads(
            raw.decode("utf-8", errors="strict"),
            object_pairs_hook=_pairs,
            parse_int=_parse_integer,
            parse_float=_reject_number,
            parse_constant=_reject_number,
        )
    except (ValueError, UnicodeError, RecursionError) as exc:
        if isinstance(exc, ContractRefusal):
            raise ContractRefusal(exc.code) from None
        raise ContractRefusal() from None
    pending = [value]
    while pending:
        item = pending.pop()
        if type(item) is str:
            _string(item)
        elif type(item) is dict:
            pending.extend(item.values())
        elif type(item) is list:
            pending.extend(item)
        elif type(item) is int:
            _integer(item)
        elif type(item) is not bool:
            _deny()
    return value


@dataclass(frozen=True, slots=True)
class ParsedContract:
    """Immutable structural data, explicitly carrying no authorization."""

    schema: str
    raw: bytes
    canonical: bytes
    sha256: str
    authorization: Literal["UNVERIFIED"] = field(default="UNVERIFIED", init=False)

    def as_dict(self) -> dict[str, Any]:
        return cast(dict[str, Any], json.loads(self.canonical))


def decode_json(raw: bytes) -> Any:
    """Strict bounded JSON data only; does not validate a contract or trust."""
    return _decode(raw, MAX_MANIFEST_BYTES)


def _checkpoint(value: Any) -> None:
    value = _record(value, {"sequence", "state_sha256"})
    _integer(value["sequence"])
    _digest(value["state_sha256"])


def _validate_acceptance(value: Any) -> None:
    value = _record(value, {"payload", "payload_digest", "threshold", "signatures"})
    payload = _record(
        value["payload"],
        {
            "schema",
            "version",
            "purpose",
            "manifest_sha256",
            "policy_sha256",
            "root_set_id",
            "sequence",
            "revocation_checkpoint",
            "issued_at",
            "expires_at",
        },
    )
    _schema(payload, "CryptoHunter.Stage9ObserverAcceptanceV1", purpose=True)
    _digest(payload["manifest_sha256"])
    _digest(payload["policy_sha256"])
    _identifier(payload["root_set_id"])
    _integer(payload["sequence"])
    _checkpoint(payload["revocation_checkpoint"])
    _require(
        _timestamp(payload["issued_at"]) < _timestamp(payload["expires_at"]), "ACCEPTANCE_REJECTED"
    )
    _equal(value["threshold"], 2)
    _digest(value["payload_digest"])
    _require(
        value["payload_digest"] == hashlib.sha256(canonical_json_bytes(payload)).hexdigest(),
        "ACCEPTANCE_REJECTED",
    )
    signers: list[str] = []
    for signature in _list(value["signatures"], maximum=MAX_FILES, minimum=2):
        signature = _record(signature, {"signer_id", "signature_hex"})
        signers.append(_identifier(signature["signer_id"]))
        _hex(signature["signature_hex"], 128)
    _require(signers == sorted(set(signers)), "ACCEPTANCE_REJECTED")


def acceptance_signed_message(raw: bytes) -> bytes:
    """Return the specified signing preimage, without verifying any signature."""
    value = _decode(raw)
    _validate_acceptance(value)
    return ACCEPTANCE_DOMAIN + bytes.fromhex(value["payload_digest"])


def _parse_contract(
    raw: bytes,
    schema: str,
    validator: Callable[[Any], None],
    *,
    maximum: int = MAX_DOCUMENT_BYTES,
    canonical_required: bool = False,
) -> ParsedContract:
    value = _decode(raw, maximum)
    validator(value)
    canonical = canonical_json_bytes(value)
    _require(not canonical_required or raw == canonical)
    return ParsedContract(schema, raw, canonical, hashlib.sha256(raw).hexdigest())


def parse_acceptance(raw: bytes) -> ParsedContract:
    """Parse signature syntax and signing digest; signatures remain unverified."""
    return _parse_contract(raw, "CryptoHunter.Stage9ObserverAcceptanceV1", _validate_acceptance)


def parse_launch_policy(raw: bytes) -> ParsedContract:
    """Parse canonical launch declarations, without manifest closure/enforcement."""
    return _parse_contract(
        raw,
        "CryptoHunter.Stage9ObserverLaunchPolicyV1",
        _validate_launch_policy,
        canonical_required=True,
    )


def _reviewer_keys(value: Any) -> list[dict[str, Any]]:
    result = []
    signers: list[str] = []
    reviewers: set[str] = set()
    public_keys: set[str] = set()
    for key in _list(value, minimum=2):
        key = _record(key, {"signer_id", "reviewer_id", "public_key_hex"})
        signer = _identifier(key["signer_id"])
        reviewer = _identifier(key["reviewer_id"])
        public_key = _hex(key["public_key_hex"], 64)
        _require(reviewer not in reviewers and public_key not in public_keys, "ACCEPTANCE_REJECTED")
        signers.append(signer)
        reviewers.add(reviewer)
        public_keys.add(public_key)
        result.append(key)
    _require(signers == sorted(set(signers)), "ACCEPTANCE_REJECTED")
    return result


def parse_acceptance_policy(raw: bytes) -> NoReturn:
    """Refuse unfinished V1 schemas; see the schema-gap review document.

    Validate only normative fields before refusing. The missing target,
    allowed_sources and limits wire schemas must never be accepted as opaque
    dictionaries or replaced by locally invented authorization semantics.
    """
    value = _record(
        _decode(raw),
        {
            "schema",
            "version",
            "purpose",
            "root_set_id",
            "reviewer_keys",
            "threshold",
            "allowed_sources",
            "minimum_sequence",
            "revocation_checkpoint",
            "valid_until",
            "target",
            "limits",
            "launch_policy_sha256",
        },
    )
    _schema(value, "CryptoHunter.Stage9ObserverAcceptancePolicyV1", purpose=True)
    _identifier(value["root_set_id"])
    _reviewer_keys(value["reviewer_keys"])
    _equal(value["threshold"], 2)
    _integer(value["minimum_sequence"])
    _checkpoint(value["revocation_checkpoint"])
    _timestamp(value["valid_until"])
    _digest(value["launch_policy_sha256"])
    _deny()


def parse_manifest(raw: bytes) -> NoReturn:
    """Refuse until exact manifest nested schemas have been reviewed.

    No manifest, inventory, provenance or dependency graph is accepted by this
    interim implementation. Checking its header does not validate its closure.
    """
    value = _record(
        _decode(raw, MAX_MANIFEST_BYTES),
        {
            "schema",
            "version",
            "purpose",
            "manifest_id",
            "sequence",
            "source_revision",
            "target",
            "sources",
            "files",
            "transforms",
            "runtime",
            "imports",
            "launch_policy_sha256",
        },
    )
    _schema(value, "CryptoHunter.Stage9ObserverManifestV1", purpose=True)
    _identifier(value["manifest_id"])
    _integer(value["sequence"])
    _hex(value["source_revision"], 40)
    _list(value["sources"], maximum=MAX_SOURCES)
    _list(value["files"], maximum=MAX_FILES)
    _digest(value["launch_policy_sha256"])
    _deny()


def _launch_ids(value: Any, *, minimum: int = 0) -> list[str]:
    values = _list(value, minimum=minimum)
    result = [_identifier(item) for item in values]
    _require(len(set(result)) == len(result), "LAUNCH_POLICY_MISMATCH")
    return result


def _launch_exact_text(value: Any) -> str:
    text = _string(value, maximum=128)
    _require(
        bool(text)
        and text.isascii()
        and all(33 <= ord(char) <= 126 for char in text)
        and not any(char in text for char in "*?[]"),
        "FORMAT_REJECTED",
    )
    return text


def _launch_profile(value: Any) -> None:
    profile = _record(value, {"profile_file_id", "profile_sha256", "enforcement"})
    _identifier(profile["profile_file_id"])
    _digest(profile["profile_sha256"])
    _equal(profile["enforcement"], "REQUIRED")


def _validate_launch_policy(value: Any) -> None:
    """Validate V1 launch declarations, without authenticating or enforcing them."""
    launch = _record(
        value,
        {
            "schema",
            "version",
            "purpose",
            "target",
            "interpreter",
            "elf",
            "imports",
            "environment",
            "cwd",
            "allowed_fds",
            "namespace_scope",
            "sandbox",
            "file_policy",
            "runtime_dependencies",
            "resources",
            "reporting",
            "run_context",
        },
    )
    _schema(launch, "CryptoHunter.Stage9ObserverLaunchPolicyV1", purpose=True)
    target = _record(
        launch["target"], {"os", "distribution", "distribution_version", "architecture"}
    )
    for key, target_value in (
        ("os", "Linux"),
        ("distribution", "Ubuntu"),
        ("distribution_version", "24.04"),
        ("architecture", "x86_64"),
    ):
        _equal(target[key], target_value)

    interpreter = _record(
        launch["interpreter"],
        {
            "file_id",
            "version",
            "package_version",
            "abi",
            "build_id",
            "libpython_file_ids",
            "entrypoint_file_id",
            "argv",
            "frozen_modules",
        },
    )
    interpreter_id = _identifier(interpreter["file_id"])
    entrypoint_id = _identifier(interpreter["entrypoint_file_id"])
    _require(interpreter_id != entrypoint_id, "LAUNCH_POLICY_MISMATCH")
    python_version = _launch_exact_text(interpreter["version"])
    components = python_version.split(".")
    _require(
        len(components) == 3
        and components[:2] == ["3", "12"]
        and components[2].isascii()
        and components[2].isdecimal()
        and (components[2] == "0" or not components[2].startswith("0")),
        "LAUNCH_POLICY_MISMATCH",
    )
    for field in ("package_version", "abi", "build_id"):
        _launch_exact_text(interpreter[field])
    libpython_ids = _launch_ids(interpreter["libpython_file_ids"])
    _require(interpreter_id not in libpython_ids, "LAUNCH_POLICY_MISMATCH")
    argv = _list(interpreter["argv"], maximum=5, minimum=5)
    _path(argv[0])
    _path(argv[4])
    _require(argv[0] != argv[4], "LAUNCH_POLICY_MISMATCH")
    for index, argument in enumerate(("-I", "-S", "-B"), start=1):
        _equal(argv[index], argument)
    _equal(interpreter["frozen_modules"], "BUILD_FIXED")

    elf = _record(
        launch["elf"],
        {
            "loader_file_id",
            "library_file_ids",
            "dependency_edges",
            "loader_config_file_ids",
            "absent_paths",
            "hwcaps_file_ids",
            "dlopen_file_ids",
            "resolution_root_ids",
            "rpath_runpath_policy",
        },
    )
    _identifier(elf["loader_file_id"])
    for field in (
        "library_file_ids",
        "loader_config_file_ids",
        "hwcaps_file_ids",
        "dlopen_file_ids",
    ):
        _launch_ids(elf[field])
    _launch_ids(elf["resolution_root_ids"], minimum=1)
    absent_paths = [_path(item) for item in _list(elf["absent_paths"])]
    _require(len(set(absent_paths)) == len(absent_paths), "LAUNCH_POLICY_MISMATCH")
    edges_seen: set[tuple[str, str]] = set()
    for item in _list(elf["dependency_edges"]):
        edge = _record(item, {"requester_file_id", "needed", "resolved_file_id"})
        requester = _identifier(edge["requester_file_id"])
        needed = _string(edge["needed"], maximum=4096)
        _require(bool(needed))
        resolved = _identifier(edge["resolved_file_id"])
        _require(requester != resolved, "LAUNCH_POLICY_MISMATCH")
        pair = (requester, needed)
        _require(pair not in edges_seen, "LAUNCH_POLICY_MISMATCH")
        edges_seen.add(pair)
    _equal(elf["rpath_runpath_policy"], "IN_IMAGE_ONLY")

    imports = _record(
        launch["imports"],
        {"roots", "modules_sha256", "embedded_modules_policy", "external_imports"},
    )
    _launch_ids(imports["roots"], minimum=1)
    _digest(imports["modules_sha256"])
    _equal(imports["embedded_modules_policy"], "EXACT_BUILD_INVENTORY")
    _equal(imports["external_imports"], "FORBIDDEN")
    environment = _record(launch["environment"], {"inherit", "values"})
    _equal(environment["inherit"], False)
    environment_values = _record(environment["values"], {"LC_ALL"})
    _equal(environment_values["LC_ALL"], "C")
    cwd = _record(launch["cwd"], {"directory_file_id"})
    _identifier(cwd["directory_file_id"])

    fds = _list(launch["allowed_fds"], maximum=2, minimum=2)
    for item, fd_values in zip(
        fds,
        (
            {
                "fd": 3,
                "role": "REPORT_WRITE",
                "kind": "PIPE",
                "access": "WRITE",
                "binding": "TRUSTED_RUN_REPORT_PIPE",
            },
            {
                "fd": 4,
                "role": "RUN_CONTEXT_READ",
                "kind": "SEALED_MEMFD",
                "access": "READ",
                "binding": "TRUSTED_LAUNCHER_CONTEXT",
            },
        ),
    ):
        fd = _record(item, set(fd_values))
        for field, expected_value in fd_values.items():
            _equal(fd[field], expected_value)

    scope = _record(
        launch["namespace_scope"],
        {"pid", "user", "mount", "procfs", "binding_schema", "binding_source", "visibility"},
    )
    for field, expected_value in (
        ("pid", "HOST_INITIAL"),
        ("user", "HOST_INITIAL"),
        ("procfs", "AUTHENTICATED_HOST_PROCFS"),
        ("binding_schema", "HostScopeBindingV1"),
        ("binding_source", "TRUSTED_RUN_CONTEXT"),
        ("visibility", "COMPLETE"),
    ):
        _equal(scope[field], expected_value)
    _enum(scope["mount"], {"HOST", "PRIVATE_FAITHFUL_HOST_VIEW"})

    sandbox = _record(
        launch["sandbox"],
        {
            "uid",
            "gid",
            "supplementary_groups",
            "capabilities",
            "no_new_privileges",
            "seccomp",
            "lsm",
            "network",
            "devices",
            "deny_operations",
        },
    )
    _integer(sandbox["uid"], minimum=1)
    _integer(sandbox["gid"], minimum=1)
    _equal(sandbox["supplementary_groups"], [])
    capabilities = _record(
        sandbox["capabilities"], {"bounding", "permitted", "effective", "inheritable", "ambient"}
    )
    for capability_set in capabilities.values():
        _equal(capability_set, [])
    _equal(sandbox["no_new_privileges"], True)
    _equal(sandbox["network"], "DENY")
    _equal(sandbox["devices"], "DENY")
    _launch_profile(sandbox["seccomp"])
    _launch_profile(sandbox["lsm"])
    denied = {
        "NEW_EXEC",
        "NEW_PROCESS",
        "MOUNT",
        "SETNS",
        "UNSHARE",
        "PTRACE",
        "PROCESS_VM",
        "PIDFD_GETFD",
        "SIGNAL",
        "NETWORK",
        "DEVICE",
        "WRITE_OUTSIDE_REPORT",
    }
    operations = [
        _enum(item, denied) for item in _list(sandbox["deny_operations"], maximum=12, minimum=12)
    ]
    _require(set(operations) == denied, "LAUNCH_POLICY_MISMATCH")

    file_policy = _record(
        launch["file_policy"],
        {
            "files_sha256",
            "ancestry",
            "acl",
            "capability_xattrs",
            "other_xattrs",
            "regular_nlink",
            "links",
            "extra_entries",
            "mutable_runtime",
            "integrity_profile_file_id",
            "integrity_profile_sha256",
        },
    )
    _digest(file_policy["files_sha256"])
    _identifier(file_policy["integrity_profile_file_id"])
    _digest(file_policy["integrity_profile_sha256"])
    for field, expected_value in (
        ("ancestry", "PROTECTED"),
        ("acl", "FORBIDDEN"),
        ("capability_xattrs", "FORBIDDEN"),
        ("other_xattrs", "PROFILE_ONLY"),
        ("regular_nlink", 1),
        ("links", "FORBIDDEN"),
        ("extra_entries", "FORBIDDEN"),
        ("mutable_runtime", "FORBIDDEN"),
    ):
        _equal(file_policy[field], expected_value)
    dependencies = _record(
        launch["runtime_dependencies"],
        {"files_sha256", "runtime_sha256", "imports_sha256", "unlisted_dependencies"},
    )
    for field in ("files_sha256", "runtime_sha256", "imports_sha256"):
        _digest(dependencies[field])
    _equal(dependencies["unlisted_dependencies"], "FORBIDDEN")
    _require(
        dependencies["files_sha256"] == file_policy["files_sha256"]
        and dependencies["imports_sha256"] == imports["modules_sha256"],
        "LAUNCH_POLICY_MISMATCH",
    )

    resources = _record(
        launch["resources"],
        {
            "run_timeout_s",
            "receive_timeout_s",
            "memory_max_bytes",
            "fd_limit",
            "process_limit",
            "core_dump_bytes",
        },
    )
    for field, expected_value in (
        ("run_timeout_s", 300),
        ("receive_timeout_s", 30),
        ("memory_max_bytes", 268435456),
        ("process_limit", 1),
        ("core_dump_bytes", 0),
    ):
        _equal(resources[field], expected_value)
    _integer(resources["fd_limit"], minimum=8, maximum=64)
    reporting = _record(
        launch["reporting"], {"schema", "fd", "framing", "max_payload_bytes", "nonce_source", "eof"}
    )
    for field, expected_value in (
        ("schema", "ObserverResultV1"),
        ("fd", 3),
        ("framing", "U32BE_UTF8_JSON"),
        ("max_payload_bytes", 65536),
        ("nonce_source", "TRUSTED_RUN_CONTEXT"),
        ("eof", "REQUIRED"),
    ):
        _equal(reporting[field], expected_value)
    context = _record(
        launch["run_context"], {"schema", "fd", "max_bytes", "storage", "seals", "access", "source"}
    )
    for field, expected_value in (
        ("schema", "ObserverRunContextV1"),
        ("fd", 4),
        ("max_bytes", 16384),
        ("storage", "MEMFD"),
        ("seals", ["WRITE", "GROW", "SHRINK", "SEAL"]),
        ("access", "READ_ONLY"),
        ("source", "TRUSTED_LAUNCHER"),
    ):
        _equal(context[field], expected_value)
