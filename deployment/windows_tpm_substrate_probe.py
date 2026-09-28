"""Disposable, fail-closed Windows TPM 2.0/TBS acceptance probe.

This module is deliberately isolated from the product root-of-trust and MSI
paths.  It owns only an NV index which it selected as unused during this run.
Static tests may exercise the codec and state machine, but only this module
running against a physical TPM is allowed to publish PASS results.
"""

from __future__ import annotations

import argparse
import ctypes
from ctypes import wintypes
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import secrets
import struct
import subprocess
import sys
import tempfile
from typing import Any, Callable, Iterable


OUTPUT_NAMES = (
    "CAN_CREATE_REQUIRED_NV_INDEX",
    "CAN_READ_REQUIRED_NV_INDEX",
    "CAN_INCREMENT_REQUIRED_NV_INDEX",
    "CAN_OPEN_POLICY_SESSION",
    "CAN_SATISFY_SELECTED_POLICY",
    "CAN_SURVIVE_SERVICE_RESTART",
    "CAN_DETECT_DISK_STATE_BEHIND_COUNTER",
)
OUTPUT_VALUES = frozenset({"PASS", "FAIL", "BLOCKED", "NOT_RUN"})

TPM_ST_NO_SESSIONS = 0x8001
TPM_ST_SESSIONS = 0x8002
TPM_CC_NV_UNDEFINE_SPACE = 0x00000122
TPM_CC_NV_INCREMENT = 0x00000134
TPM_CC_NV_DEFINE_SPACE = 0x0000012A
TPM_CC_NV_READ = 0x0000014E
TPM_CC_NV_READ_PUBLIC = 0x00000169
TPM_CC_GET_CAPABILITY = 0x0000017A
TPM_CC_START_AUTH_SESSION = 0x00000176
TPM_CC_POLICY_NV = 0x00000149
TPM_CC_POLICY_COMMAND_CODE = 0x0000016C
TPM_CC_POLICY_OR = 0x00000171
TPM_CC_POLICY_GET_DIGEST = 0x00000189
TPM_CC_FLUSH_CONTEXT = 0x00000165
TPM_RH_OWNER = 0x40000001
TPM_RH_NULL = 0x40000007
TPM_RS_PW = 0x40000009
TPM_CAP_HANDLES = 0x00000001
TPM_HT_NV_INDEX = 0x01
TPM_ALG_SHA256 = 0x000B
TPM_ALG_NULL = 0x0010
TPM_SE_POLICY = 0x01
TPM_EO_EQ = 0x0000
TPMA_NV_OWNERWRITE = 0x00000002
TPM_NT_COUNTER = 0x1
TPMA_NV_TPM2_NT_SHIFT = 4
TPMA_NV_TPM2_NT_COUNTER = TPM_NT_COUNTER << TPMA_NV_TPM2_NT_SHIFT
TPMA_NV_OWNERREAD = 0x00020000
NV_ATTRIBUTES = TPMA_NV_OWNERWRITE | TPMA_NV_TPM2_NT_COUNTER | TPMA_NV_OWNERREAD
TBS_COMMAND_LOCALITY_ZERO = 0
TBS_COMMAND_PRIORITY_NORMAL = 200
TBS_SUCCESS = 0
TPM_VERSION_20 = 2
TBS_CONTEXT_INCLUDE_TPM20 = 0x4
WINDOWS_11_MINIMUM_BUILD = 22000
MAX_RESPONSE = 16 * 1024
NV_OWNER_FIRST = 0x01800000
NV_OWNER_LAST = 0x01BFFFFF

TPM_RC_SUCCESS = 0x000
TPM_RC_ATTRIBUTES = 0x082
TPM_RC_HANDLE = 0x08B
TPM_RC_AUTH_FAIL = 0x08E
TPM_RC_SCHEME = 0x092
TPM_RC_INSUFFICIENT = 0x09A
TPM_RC_POLICY_FAIL = 0x09D
TPM_RC_FAILURE = 0x101
TPM_RC_NV_RANGE = 0x146
TPM_RC_NV_AUTHORIZATION = 0x149
TPM_RC_NV_UNINITIALIZED = 0x14A
TPM_RC_NV_SPACE = 0x14B
TPM_RC_NV_DEFINED = 0x14C

RC_NAMES = {
    TPM_RC_SUCCESS: "TPM_RC_SUCCESS",
    TPM_RC_ATTRIBUTES: "TPM_RC_ATTRIBUTES",
    TPM_RC_HANDLE: "TPM_RC_HANDLE",
    TPM_RC_AUTH_FAIL: "TPM_RC_AUTH_FAIL",
    TPM_RC_SCHEME: "TPM_RC_SCHEME",
    TPM_RC_INSUFFICIENT: "TPM_RC_INSUFFICIENT",
    TPM_RC_POLICY_FAIL: "TPM_RC_POLICY_FAIL",
    TPM_RC_FAILURE: "TPM_RC_FAILURE",
    TPM_RC_NV_RANGE: "TPM_RC_NV_RANGE",
    TPM_RC_NV_AUTHORIZATION: "TPM_RC_NV_AUTHORIZATION",
    TPM_RC_NV_UNINITIALIZED: "TPM_RC_NV_UNINITIALIZED",
    TPM_RC_NV_SPACE: "TPM_RC_NV_SPACE",
    TPM_RC_NV_DEFINED: "TPM_RC_NV_DEFINED",
}


class ProbeError(RuntimeError):
    """A bounded probe failure with a stable machine-readable reason."""

    def __init__(self, reason: str, detail: str) -> None:
        self.reason = reason
        self.detail = detail
        super().__init__(f"{reason}: {detail}")


def u16(value: int) -> bytes:
    return struct.pack(">H", value)


def u32(value: int) -> bytes:
    return struct.pack(">I", value)


def tpm2b(value: bytes) -> bytes:
    if len(value) > 0xFFFF:
        raise ValueError("TPM2B payload too large")
    return u16(len(value)) + value


def read_u16(data: bytes, offset: int = 0) -> tuple[int, int]:
    if offset + 2 > len(data):
        raise ValueError("truncated UINT16")
    return struct.unpack_from(">H", data, offset)[0], offset + 2


def read_u32(data: bytes, offset: int = 0) -> tuple[int, int]:
    if offset + 4 > len(data):
        raise ValueError("truncated UINT32")
    return struct.unpack_from(">I", data, offset)[0], offset + 4


def read_tpm2b(data: bytes, offset: int = 0) -> tuple[bytes, int]:
    size, offset = read_u16(data, offset)
    end = offset + size
    if end > len(data):
        raise ValueError("truncated TPM2B")
    return data[offset:end], end


def decode_tpm_rc(rc: int) -> str:
    """Decode the stable base error while retaining the full formatted RC."""
    if rc == 0:
        return "TPM_RC_SUCCESS"
    base = rc & 0x0BF if rc & 0x080 else rc & 0x17F
    name = RC_NAMES.get(rc) or RC_NAMES.get(base) or "TPM_RC_UNKNOWN"
    return f"{name}(0x{rc:08X})"


def command_packet(code: int, handles: Iterable[int], parameters: bytes, *, auth: bool) -> bytes:
    handle_bytes = b"".join(u32(handle) for handle in handles)
    if auth:
        password = u32(TPM_RS_PW) + tpm2b(b"") + b"\x00" + tpm2b(b"")
        body = handle_bytes + u32(len(password)) + password + parameters
        tag = TPM_ST_SESSIONS
    else:
        body = handle_bytes + parameters
        tag = TPM_ST_NO_SESSIONS
    return u16(tag) + u32(10 + len(body)) + u32(code) + body


def response_parameters(response: bytes) -> bytes:
    tag, offset = read_u16(response)
    size, offset = read_u32(response, offset)
    rc, offset = read_u32(response, offset)
    if size != len(response):
        raise ValueError("TPM response size does not match header")
    if rc:
        raise ProbeError("TPM_COMMAND_REJECTED", decode_tpm_rc(rc))
    if tag == TPM_ST_SESSIONS:
        parameter_size, offset = read_u32(response, offset)
        return response[offset : offset + parameter_size]
    if tag != TPM_ST_NO_SESSIONS:
        raise ValueError("unexpected TPM response tag")
    return response[offset:]


def reconcile_pending(pending: int, observed: int, increment: Callable[[], int]) -> tuple[int, int]:
    """Recover one lost response without ever issuing a duplicate increment."""
    if observed == pending:
        return observed, 0
    if observed + 1 == pending:
        after = increment()
        if after != pending:
            raise ProbeError("COUNTER_RECONCILIATION_MISMATCH", f"expected={pending}, got={after}")
        return after, 1
    raise ProbeError("COUNTER_RECONCILIATION_GAP", f"pending={pending}, observed={observed}")


def may_cleanup(selected: int | None, created_by_run: bool, handle: int) -> bool:
    """The cleanup guard is intentionally stricter than TPM authorization."""
    return created_by_run and selected is not None and selected == handle


def blank_evidence(revision: str, timestamp: str) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "source_revision": revision,
        "timestamp_utc": timestamp,
        "live_physical_tpm_execution": False,
        "windows_build": None,
        "windows_edition": None,
        "tpm_present": None,
        "tpm_ready": None,
        "tpm_manufacturer": None,
        "tpm_version": None,
        "tbs_available": False,
        "tpm2_capable": None,
        "selected_nv_index": None,
        "nv_public": None,
        "nv_name": None,
        "nv_name_recomputed": None,
        "nv_identity_transition": None,
        "new_process": None,
        "commands": [],
        "responses": [],
        "policy_digests": {},
        "counter_transitions": [],
        "lost_response_cases": {},
        "cleanup": {"nv": "NOT_RUN", "service": "NOT_RUN"},
        "outputs": {name: {"status": "NOT_RUN", "reason": None} for name in OUTPUT_NAMES},
    }


def validate_evidence(evidence: dict[str, Any]) -> None:
    required = {
        "schema_version",
        "source_revision",
        "timestamp_utc",
        "windows_build",
        "tpm_manufacturer",
        "tpm_version",
        "tbs_available",
        "selected_nv_index",
        "nv_public",
        "nv_name",
        "commands",
        "responses",
        "policy_digests",
        "counter_transitions",
        "lost_response_cases",
        "cleanup",
        "outputs",
    }
    missing = required - evidence.keys()
    if missing:
        raise ValueError(f"evidence missing fields: {sorted(missing)}")
    if tuple(evidence["outputs"]) != OUTPUT_NAMES:
        raise ValueError("evidence outputs are not the seven canonical outputs")
    for output in evidence["outputs"].values():
        if output["status"] not in OUTPUT_VALUES:
            raise ValueError("invalid output status")
        if output["status"] in {"FAIL", "BLOCKED"} and not output["reason"]:
            raise ValueError("FAIL/BLOCKED output requires a reason")
    for command in evidence["commands"]:
        expected = {
            "command_code",
            "request_size",
            "response_size",
            "response_code",
            "decoded_response_code",
        }
        if not expected <= command.keys():
            raise ValueError("incomplete TPM command evidence")


@dataclass
class TbsTransport:
    evidence: dict[str, Any]
    dll: Any = None
    context: wintypes.HANDLE | None = None

    def open(self) -> None:
        if os.name != "nt":
            raise ProbeError("WINDOWS_REQUIRED", "TBS probe can run only on Windows")
        self.dll = ctypes.WinDLL("tbs.dll")

        class ContextParams2(ctypes.Structure):
            _fields_ = [("version", ctypes.c_uint32), ("flags", ctypes.c_uint32)]

        params = ContextParams2(TPM_VERSION_20, TBS_CONTEXT_INCLUDE_TPM20)
        context = wintypes.HANDLE()
        try:
            create_context = self.dll.Tbsi_Context_Create
        except AttributeError as exc:
            raise ProbeError(
                "TBS_CONTEXT_CREATE_ENTRYPOINT_MISSING",
                "tbs.dll does not export Tbsi_Context_Create",
            ) from exc
        create_context.argtypes = [ctypes.c_void_p, ctypes.POINTER(wintypes.HANDLE)]
        create_context.restype = ctypes.c_uint32
        status = create_context(ctypes.byref(params), ctypes.byref(context))
        if status != TBS_SUCCESS:
            raise ProbeError("TBS_CREATE_CONTEXT_FAILED", f"TBS status=0x{status:08X}")
        self.context = context
        self.evidence["tbs_available"] = True
        self.dll.Tbsip_Context_Close.argtypes = [wintypes.HANDLE]
        self.dll.Tbsip_Context_Close.restype = ctypes.c_uint32

    def close(self) -> None:
        if self.dll is not None and self.context:
            status = self.dll.Tbsip_Context_Close(self.context)
            self.evidence["cleanup"]["tbs_context"] = (
                "PASS" if status == TBS_SUCCESS else f"FAIL:TBS_STATUS_0x{status:08X}"
            )
            self.context = None

    def submit(self, code: int, request: bytes) -> bytes:
        if self.dll is None or not self.context:
            raise ProbeError("TBS_CONTEXT_NOT_OPEN", "cannot submit without a TBS context")
        request_buffer = (ctypes.c_ubyte * len(request)).from_buffer_copy(request)
        response_buffer = (ctypes.c_ubyte * MAX_RESPONSE)()
        response_size = ctypes.c_uint32(MAX_RESPONSE)
        self.dll.Tbsip_Submit_Command.argtypes = [
            wintypes.HANDLE,
            ctypes.c_uint32,
            ctypes.c_uint32,
            ctypes.c_void_p,
            ctypes.c_uint32,
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_uint32),
        ]
        self.dll.Tbsip_Submit_Command.restype = ctypes.c_uint32
        status = self.dll.Tbsip_Submit_Command(
            self.context,
            TBS_COMMAND_LOCALITY_ZERO,
            TBS_COMMAND_PRIORITY_NORMAL,
            request_buffer,
            len(request),
            response_buffer,
            ctypes.byref(response_size),
        )
        response = bytes(response_buffer[: response_size.value]) if status == TBS_SUCCESS else b""
        rc = struct.unpack_from(">I", response, 6)[0] if len(response) >= 10 else None
        item = {
            "command_code": f"0x{code:08X}",
            "request_size": len(request),
            "response_size": len(response),
            "response_code": None if rc is None else f"0x{rc:08X}",
            "decoded_response_code": "NO_TPM_RESPONSE" if rc is None else decode_tpm_rc(rc),
            "tbs_status": f"0x{status:08X}",
        }
        self.evidence["commands"].append(item)
        self.evidence["responses"].append(dict(item))
        if status != TBS_SUCCESS:
            raise ProbeError(
                "TBS_SUBMIT_FAILED", f"command=0x{code:08X}, TBS status=0x{status:08X}"
            )
        if rc:
            raise ProbeError("TPM_COMMAND_REJECTED", f"command=0x{code:08X}, {decode_tpm_rc(rc)}")
        return response


class NvProbe:
    def __init__(self, transport: TbsTransport) -> None:
        self.t = transport

    def handles(self) -> set[int]:
        found: set[int] = set()
        first = TPM_HT_NV_INDEX << 24
        while True:
            request = command_packet(
                TPM_CC_GET_CAPABILITY, (), u32(TPM_CAP_HANDLES) + u32(first) + u32(64), auth=False
            )
            data = response_parameters(self.t.submit(TPM_CC_GET_CAPABILITY, request))
            if len(data) < 5:
                raise ProbeError("MALFORMED_CAPABILITY", "short handles response")
            more = data[0] != 0
            capability, offset = read_u32(data, 1)
            if capability != TPM_CAP_HANDLES:
                raise ProbeError("WRONG_CAPABILITY", f"received={capability}")
            count, offset = read_u32(data, offset)
            page = []
            for _ in range(count):
                handle, offset = read_u32(data, offset)
                page.append(handle)
                found.add(handle)
            if not more or not page:
                return found
            first = page[-1] + 1

    def define(self, handle: int) -> None:
        public = u32(handle) + u16(TPM_ALG_SHA256) + u32(NV_ATTRIBUTES) + tpm2b(b"") + u16(8)
        params = tpm2b(b"") + tpm2b(public)
        request = command_packet(TPM_CC_NV_DEFINE_SPACE, (TPM_RH_OWNER,), params, auth=True)
        response_parameters(self.t.submit(TPM_CC_NV_DEFINE_SPACE, request))

    def read_public(self, handle: int) -> tuple[bytes, bytes]:
        request = command_packet(TPM_CC_NV_READ_PUBLIC, (handle,), b"", auth=False)
        data = response_parameters(self.t.submit(TPM_CC_NV_READ_PUBLIC, request))
        public, offset = read_tpm2b(data)
        name, _ = read_tpm2b(data, offset)
        return public, name

    def read(self, handle: int) -> int:
        request = command_packet(TPM_CC_NV_READ, (TPM_RH_OWNER, handle), u16(8) + u16(0), auth=True)
        data = response_parameters(self.t.submit(TPM_CC_NV_READ, request))
        value, _ = read_tpm2b(data)
        if len(value) != 8:
            raise ProbeError("INVALID_COUNTER_SIZE", f"received={len(value)}")
        return int.from_bytes(value, "big")

    def increment(self, handle: int) -> None:
        request = command_packet(TPM_CC_NV_INCREMENT, (TPM_RH_OWNER, handle), b"", auth=True)
        response_parameters(self.t.submit(TPM_CC_NV_INCREMENT, request))

    def undefine(self, handle: int) -> None:
        request = command_packet(TPM_CC_NV_UNDEFINE_SPACE, (TPM_RH_OWNER, handle), b"", auth=True)
        response_parameters(self.t.submit(TPM_CC_NV_UNDEFINE_SPACE, request))

    def start_policy(self) -> tuple[int, bytes, bytes]:
        nonce = secrets.token_bytes(32)
        params = (
            tpm2b(nonce)
            + tpm2b(b"")
            + bytes((TPM_SE_POLICY,))
            + u16(TPM_ALG_NULL)
            + u16(TPM_ALG_SHA256)
        )
        request = command_packet(
            TPM_CC_START_AUTH_SESSION, (TPM_RH_NULL, TPM_RH_NULL), params, auth=False
        )
        data = response_parameters(self.t.submit(TPM_CC_START_AUTH_SESSION, request))
        session, offset = read_u32(data)
        nonce_tpm, _ = read_tpm2b(data, offset)
        return session, nonce, nonce_tpm

    def policy_command_code(self, session: int, code: int) -> None:
        request = command_packet(TPM_CC_POLICY_COMMAND_CODE, (session,), u32(code), auth=False)
        response_parameters(self.t.submit(TPM_CC_POLICY_COMMAND_CODE, request))

    def policy_nv(self, session: int, handle: int, value: int) -> None:
        params = tpm2b(value.to_bytes(8, "big")) + u16(0) + u16(TPM_EO_EQ)
        request = command_packet(
            TPM_CC_POLICY_NV, (TPM_RH_OWNER, handle, session), params, auth=True
        )
        response_parameters(self.t.submit(TPM_CC_POLICY_NV, request))

    def policy_digest(self, session: int) -> bytes:
        request = command_packet(TPM_CC_POLICY_GET_DIGEST, (session,), b"", auth=False)
        data = response_parameters(self.t.submit(TPM_CC_POLICY_GET_DIGEST, request))
        digest, _ = read_tpm2b(data)
        return digest

    def policy_or(self, session: int, branches: list[bytes]) -> None:
        params = u32(len(branches)) + b"".join(tpm2b(branch) for branch in branches)
        request = command_packet(TPM_CC_POLICY_OR, (session,), params, auth=False)
        response_parameters(self.t.submit(TPM_CC_POLICY_OR, request))

    def flush(self, handle: int) -> None:
        request = command_packet(TPM_CC_FLUSH_CONTEXT, (handle,), b"", auth=False)
        response_parameters(self.t.submit(TPM_CC_FLUSH_CONTEXT, request))


def _revision() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"], text=True, capture_output=True, check=False
    )
    return result.stdout.strip() if result.returncode == 0 else "UNKNOWN"


def _powershell_json(script: str) -> dict[str, Any]:
    result = subprocess.run(
        ["powershell.exe", "-NoProfile", "-NonInteractive", "-Command", script],
        text=True,
        capture_output=True,
        timeout=30,
        check=False,
    )
    if result.returncode != 0:
        raise ProbeError("WINDOWS_QUALIFICATION_FAILED", result.stderr.strip()[:500])
    value = json.loads(result.stdout)
    return value if isinstance(value, dict) else {}


def require_windows_11_client(build: Any, product_type: Any) -> int:
    """Validate stable CIM values rather than a localized Windows caption."""
    try:
        numeric_build = int(build)
        numeric_product_type = int(product_type)
    except (TypeError, ValueError) as exc:
        raise ProbeError(
            "PHYSICAL_WINDOWS_11_REQUIRED", "Windows build and ProductType must be numeric"
        ) from exc
    if numeric_product_type != 1 or numeric_build < WINDOWS_11_MINIMUM_BUILD:
        raise ProbeError(
            "PHYSICAL_WINDOWS_11_REQUIRED",
            f"requires Windows client build >= {WINDOWS_11_MINIMUM_BUILD}; "
            f"build={numeric_build}, ProductType={numeric_product_type}",
        )
    return numeric_build


def qualify_windows(evidence: dict[str, Any]) -> None:
    info = _powershell_json(
        "$o=Get-CimInstance Win32_OperatingSystem; $t=Get-Tpm; "
        "$w=Get-CimInstance -Namespace root/cimv2/Security/MicrosoftTpm -Class Win32_Tpm; "
        "[ordered]@{Edition=$o.Caption;Build=$o.BuildNumber;ProductType=$o.ProductType;"
        "Present=$t.TpmPresent;Ready=$t.TpmReady;"
        "Manufacturer=$w.ManufacturerIdTxt;Version=$w.ManufacturerVersion;Spec=$w.SpecVersion}|ConvertTo-Json -Compress"
    )
    numeric_build = require_windows_11_client(info.get("Build"), info.get("ProductType"))
    evidence.update(
        windows_edition=info.get("Edition"),
        windows_build=numeric_build,
        windows_product_type=int(info["ProductType"]),
        tpm_present=info.get("Present"),
        tpm_ready=info.get("Ready"),
        tpm_manufacturer=info.get("Manufacturer"),
        tpm_version=info.get("Version"),
        tpm2_capable="2.0" in str(info.get("Spec", "")),
    )
    if not evidence["tpm_present"] or not evidence["tpm_ready"] or not evidence["tpm2_capable"]:
        raise ProbeError("TPM2_NOT_READY", "physical TPM 2.0 must be present and ready")


def _set(evidence: dict[str, Any], name: str, status: str, reason: str | None = None) -> None:
    evidence["outputs"][name] = {"status": status, "reason": reason}


def _choose_unused(existing: set[int], randbelow: Callable[[int], int] = secrets.randbelow) -> int:
    """Select only from the TCG owner NV range after enumerating occupied handles."""
    width = NV_OWNER_LAST - NV_OWNER_FIRST + 1
    for _ in range(256):
        candidate = NV_OWNER_FIRST + randbelow(width)
        if candidate not in existing:
            return candidate
    raise ProbeError("NO_UNUSED_NV_INDEX", "could not select a non-colliding application index")


def qualify_new_counter(
    nv: Any,
    handle: int,
    transitions: list[dict[str, Any]],
    mark_created: Callable[[], None],
    mark_read: Callable[[], None],
) -> tuple[bytes, bytes, int, int, dict[str, Any]]:
    """Create and qualify a fresh counter in the only valid command order.

    ``mark_created`` runs immediately after DefineSpace succeeds so cleanup
    remains authorized even if a later qualification command fails.  This
    helper does not set any CAN_* result; recording fakes therefore cannot
    publish live TPM evidence.
    """
    nv.define(handle)
    mark_created()
    pre_write_public, pre_write_name = nv.read_public(handle)

    # TPMA_NV_WRITTEN is clear after DefineSpace.  The first increment is the
    # initializing write; an NV_Read before it may return NV_UNINITIALIZED.
    nv.increment(handle)
    initialized = nv.read(handle)
    mark_read()
    transitions.append({"purpose": "initialization", "before": None, "after": initialized})

    # TPMA_NV_WRITTEN is now set and is part of TPMS_NV_PUBLIC.  Consequently
    # this second Name, rather than the pre-write Name, is the stable identity
    # which persistence checks must carry across processes and service starts.
    post_write_public, post_write_name = nv.read_public(handle)

    nv.increment(handle)
    after = nv.read(handle)
    transitions.append({"purpose": "monotonic_transition", "before": initialized, "after": after})
    if after != initialized + 1:
        raise ProbeError("COUNTER_NOT_MONOTONIC", f"before={initialized}, after={after}")
    pre_write_recomputed = u16(TPM_ALG_SHA256) + hashlib.sha256(pre_write_public).digest()
    post_write_recomputed = u16(TPM_ALG_SHA256) + hashlib.sha256(post_write_public).digest()
    if pre_write_name != pre_write_recomputed:
        raise ProbeError("PRE_WRITE_NV_NAME_INVALID", "returned Name does not match public area")
    if post_write_name != post_write_recomputed:
        raise ProbeError("POST_WRITE_NV_NAME_INVALID", "returned Name does not match public area")
    identity_transition = {
        "pre_write_public": pre_write_public.hex(),
        "pre_write_name": pre_write_name.hex(),
        "pre_write_name_recomputed": pre_write_recomputed.hex(),
        "post_write_public": post_write_public.hex(),
        "post_write_name": post_write_name.hex(),
        "post_write_name_recomputed": post_write_recomputed.hex(),
        "name_changed_after_first_write": pre_write_name != post_write_name,
    }
    return post_write_public, post_write_name, initialized, after, identity_transition


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _read_child(handle: int, expected_name: str, output: Path) -> int:
    evidence = blank_evidence(_revision(), datetime.now(timezone.utc).isoformat())
    transport = TbsTransport(evidence)
    try:
        transport.open()
        public, name = NvProbe(transport).read_public(handle)
        value = NvProbe(transport).read(handle)
        _write_json(
            output, {"handle": handle, "name": name.hex(), "public": public.hex(), "value": value}
        )
        return 0 if name.hex() == expected_name else 1
    finally:
        transport.close()


def _new_process_read(
    handle: int, name: str, value: int, public: str, output: Path
) -> dict[str, Any]:
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--read-existing",
        hex(handle),
        name,
        str(output),
    ]
    try:
        result = subprocess.run(
            command, timeout=60, check=False, capture_output=True, text=True
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise ProbeError(
            "NEW_PROCESS_EXECUTION_FAILED",
            json.dumps({"exception": type(exc).__name__, "output_exists": output.exists()}),
        ) from exc

    diagnostic: dict[str, Any] = {
        "returncode": result.returncode,
        "stdout_tail": result.stdout[-500:],
        "stderr_tail": result.stderr[-500:],
        "output_exists": output.exists(),
    }
    child: dict[str, Any] | None = None
    if output.exists():
        try:
            loaded = json.loads(output.read_text(encoding="utf-8"))
            if isinstance(loaded, dict):
                child = loaded
                diagnostic["child"] = child
        except (OSError, json.JSONDecodeError) as exc:
            diagnostic["output_error"] = type(exc).__name__

    def fail(reason: str) -> None:
        raise ProbeError(reason, json.dumps(diagnostic, sort_keys=True))

    if child is not None:
        if child.get("handle") != handle:
            fail("NEW_PROCESS_HANDLE_MISMATCH")
        if child.get("name") != name:
            fail("NEW_PROCESS_NAME_MISMATCH")
        if child.get("value") != value:
            fail("NEW_PROCESS_VALUE_MISMATCH")
        if child.get("public") != public:
            fail("NEW_PROCESS_PUBLIC_MISMATCH")
    if result.returncode != 0 or child is None:
        fail("NEW_PROCESS_EXECUTION_FAILED" if child is None else "NEW_PROCESS_READ_FAILED")
    return diagnostic


def _scm_restart_check(handle: int, name: str, scratch: Path, evidence: dict[str, Any]) -> None:
    """Run two disposable LocalSystem service instances and compare TPM reads.

    The helper executable is this script hosted by the active Python runtime;
    pywin32 supplies the required service dispatcher.  The random service is
    always deleted by the caller's finally block.
    """
    service = f"CryptoHunterTpmAcceptance_{secrets.token_hex(6)}"
    evidence["cleanup"]["service_name"] = service
    outputs = [scratch / "service-1.json", scratch / "service-2.json"]
    bin_path = f'"{sys.executable}" "{Path(__file__).resolve()}" --scm-worker {service} {handle} {name} "{scratch}"'
    create = subprocess.run(
        ["sc.exe", "create", service, "binPath=", bin_path, "start=", "demand"],
        capture_output=True,
        text=True,
        check=False,
    )
    if create.returncode != 0:
        raise ProbeError("SCM_SERVICE_CREATE_FAILED", create.stdout[-500:] + create.stderr[-500:])
    try:
        for sequence, output in enumerate(outputs, 1):
            started = subprocess.run(
                ["sc.exe", "start", service, str(sequence)],
                capture_output=True,
                text=True,
                check=False,
            )
            if started.returncode != 0:
                raise ProbeError(
                    "SCM_SERVICE_START_FAILED", f"start={sequence}, exit={started.returncode}"
                )
            for _ in range(120):
                if output.exists():
                    break
                import time

                time.sleep(0.25)
            if not output.exists():
                raise ProbeError("SCM_SERVICE_OUTPUT_MISSING", f"start={sequence}")
            for _ in range(120):
                query = subprocess.run(
                    ["sc.exe", "query", service], capture_output=True, text=True, check=False
                )
                if "STOPPED" in query.stdout:
                    break
                import time

                time.sleep(0.25)
            else:
                raise ProbeError("SCM_SERVICE_DID_NOT_STOP", f"start={sequence}")
        first, second = (json.loads(path.read_text(encoding="utf-8")) for path in outputs)
        if first != second or first["name"] != name:
            raise ProbeError(
                "SCM_RESTART_IDENTITY_MISMATCH", "NV identity/value changed across service starts"
            )
    finally:
        subprocess.run(["sc.exe", "stop", service], capture_output=True, check=False)
        deleted = subprocess.run(["sc.exe", "delete", service], capture_output=True, check=False)
        evidence["cleanup"]["service"] = "PASS" if deleted.returncode == 0 else "FAIL"


def _scm_worker(service_name: str, handle: int, name: str, scratch: Path) -> int:
    import servicemanager
    import win32event
    import win32service
    import win32serviceutil

    class Worker(win32serviceutil.ServiceFramework):
        _svc_name_ = service_name
        _svc_display_name_ = service_name

        def __init__(self, args: list[str]) -> None:
            super().__init__(args)
            self.stop_event = win32event.CreateEvent(None, 0, 0, None)
            self.sequence = args[-1] if args and args[-1] in {"1", "2"} else "1"

        def SvcStop(self) -> None:
            self.ReportServiceStatus(win32service.SERVICE_STOP_PENDING)
            win32event.SetEvent(self.stop_event)

        def SvcDoRun(self) -> None:
            _read_child(handle, name, scratch / f"service-{self.sequence}.json")

    servicemanager.Initialize()
    servicemanager.PrepareToHostSingle(Worker)
    servicemanager.StartServiceCtrlDispatcher()
    return 0


def run_probe(output: Path) -> int:
    timestamp = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
    evidence = blank_evidence(_revision(), timestamp)
    transport = TbsTransport(evidence)
    selected: int | None = None
    created = False
    session: int | None = None
    fatal_reason: str | None = None
    active_output = OUTPUT_NAMES[0]
    scratch = Path(tempfile.mkdtemp(prefix="cryptohunter-tpm-acceptance-"))
    try:
        if os.name != "nt" or platform.system() != "Windows":
            raise ProbeError("PHYSICAL_WINDOWS_11_REQUIRED", "static CI is not live TPM evidence")
        qualify_windows(evidence)
        transport.open()
        evidence["live_physical_tpm_execution"] = True
        nv = NvProbe(transport)
        existing = nv.handles()
        selected = _choose_unused(existing)
        evidence["selected_nv_index"] = f"0x{selected:08X}"

        def mark_created() -> None:
            nonlocal created, active_output
            created = True
            _set(evidence, OUTPUT_NAMES[0], "PASS")
            active_output = OUTPUT_NAMES[1]

        def mark_read() -> None:
            nonlocal active_output
            _set(evidence, OUTPUT_NAMES[1], "PASS")
            active_output = OUTPUT_NAMES[2]

        public, name, _g, after, identity_transition = qualify_new_counter(
            nv, selected, evidence["counter_transitions"], mark_created, mark_read
        )
        evidence["nv_identity_transition"] = identity_transition
        evidence["nv_public"] = public.hex()
        evidence["nv_name"] = name.hex()
        evidence["nv_name_recomputed"] = (
            u16(TPM_ALG_SHA256) + hashlib.sha256(public).digest()
        ).hex()
        _set(evidence, OUTPUT_NAMES[2], "PASS")

        active_output = OUTPUT_NAMES[3]
        session, nonce_caller, nonce_tpm = nv.start_policy()
        evidence["policy_session"] = {
            "session_handle": f"0x{session:08X}",
            "hash_algorithm": "TPM_ALG_SHA256",
            "nonce_caller_size": len(nonce_caller),
            "nonce_tpm_size": len(nonce_tpm),
        }
        _set(evidence, OUTPUT_NAMES[3], "PASS")
        active_output = OUTPUT_NAMES[4]
        nv.policy_nv(session, selected, after)
        evidence["policy_digests"]["after_policy_nv"] = nv.policy_digest(session).hex()
        nv.policy_command_code(session, TPM_CC_NV_INCREMENT)
        branch = nv.policy_digest(session)
        evidence["policy_digests"]["after_policy_command_code"] = branch.hex()
        evidence["policy_digests"]["policy_or"] = {
            "status": "NOT_RUN",
            "reason": "ONLY_ONE_REAL_CANDIDATE_BRANCH_AVAILABLE",
            "branches": [branch.hex()],
        }
        evidence["policy_digests"]["approvedPolicy"] = None
        evidence["policy_digests"]["policyRef"] = ""
        evidence["policy_digests"]["keySign_name"] = None
        evidence["policy_digests"]["verification_ticket"] = None
        evidence["policy_digests"]["final_policy_digest"] = None
        _set(
            evidence,
            OUTPUT_NAMES[4],
            "BLOCKED",
            "POLICY_AUTHORIZE_COMMAND_PATH_NOT_IMPLEMENTED",
        )

        active_output = OUTPUT_NAMES[5]
        evidence["new_process"] = _new_process_read(
            selected, name.hex(), after, public.hex(), scratch / "new-process.json"
        )
        _scm_restart_check(selected, name.hex(), scratch, evidence)
        _set(evidence, OUTPUT_NAMES[5], "PASS")

        active_output = OUTPUT_NAMES[6]
        disk_state = scratch / "disk-generation.json"
        _write_json(disk_state, {"generation": after})
        pending_a = after + 1

        def increment_and_read() -> int:
            nv.increment(selected)
            return nv.read(selected)

        case_a_after, case_a_count = reconcile_pending(
            pending_a, nv.read(selected), increment_and_read
        )
        evidence["lost_response_cases"]["case_a"] = {
            "pending": pending_a,
            "observed_before": after,
            "observed_after": case_a_after,
            "increment_command_count": case_a_count,
        }
        case_b_after, case_b_count = reconcile_pending(
            pending_a, nv.read(selected), increment_and_read
        )
        evidence["lost_response_cases"]["case_b"] = {
            "pending": pending_a,
            "observed_before": pending_a,
            "observed_after": case_b_after,
            "increment_command_count": case_b_count,
        }
        disk_generation = json.loads(disk_state.read_text(encoding="utf-8"))["generation"]
        detected = disk_generation < nv.read(selected)
        evidence["disk_state_model"] = {
            "generation": disk_generation,
            "classification": "DISK_STATE_BEHIND_COUNTER" if detected else "CURRENT",
        }
        if not detected:
            raise ProbeError("STALE_DISK_NOT_DETECTED", "disk generation did not trail TPM counter")
        _set(evidence, OUTPUT_NAMES[6], "PASS")
    except ProbeError as exc:
        fatal_reason = exc.reason
        if exc.reason.startswith("NEW_PROCESS_"):
            try:
                evidence["new_process"] = json.loads(exc.detail)
            except json.JSONDecodeError:
                evidence["new_process"] = {"detail": exc.detail[:1000]}
        if evidence["outputs"][active_output]["status"] == "NOT_RUN":
            _set(evidence, active_output, "BLOCKED", exc.reason)
    except Exception as exc:
        fatal_reason = "UNEXPECTED_PROBE_ERROR"
        if evidence["outputs"][active_output]["status"] == "NOT_RUN":
            _set(evidence, active_output, "FAIL", fatal_reason)
        evidence["unexpected_error"] = {"type": type(exc).__name__, "message": str(exc)[:1000]}
    finally:
        if session is not None:
            try:
                NvProbe(transport).flush(session)
            except Exception as exc:
                evidence["cleanup"]["policy_session"] = f"FAIL:{type(exc).__name__}"
        if selected is not None and may_cleanup(selected, created, selected):
            try:
                NvProbe(transport).undefine(selected)
                removed = selected not in NvProbe(transport).handles()
                evidence["cleanup"]["nv"] = "PASS" if removed else "FAIL:HANDLE_STILL_PRESENT"
            except Exception as exc:
                evidence["cleanup"]["nv"] = f"FAIL:{type(exc).__name__}:{str(exc)[:300]}"
        transport.close()
        try:
            for child in scratch.iterdir():
                child.unlink()
            scratch.rmdir()
            evidence["cleanup"]["scratch"] = "PASS"
        except OSError as exc:
            evidence["cleanup"]["scratch"] = (
                f"FAIL:{exc.winerror if hasattr(exc, 'winerror') else exc.errno}"
            )
        cleanup_failures = [
            name
            for name in ("nv", "service", "scratch", "tbs_context")
            if str(evidence["cleanup"].get(name, "")).startswith("FAIL")
        ]
        if cleanup_failures:
            fatal_reason = fatal_reason or f"CLEANUP_FAILED:{','.join(cleanup_failures)}"
        validate_evidence(evidence)
        _write_json(output, evidence)
        for name in OUTPUT_NAMES:
            print(f"{name} = {evidence['outputs'][name]['status']}")
    all_pass = all(evidence["outputs"][name]["status"] == "PASS" for name in OUTPUT_NAMES)
    # Candidate policy is deliberately expected to remain BLOCKED until the
    # live evidence proves a non-circular PolicyAuthorize construction.
    return 0 if all_pass and fatal_reason is None else 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path, default=Path("dist/windows/tpm-substrate-evidence.json")
    )
    parser.add_argument("--read-existing", nargs=3, metavar=("HANDLE", "NAME", "OUTPUT"))
    parser.add_argument("--scm-worker", nargs=4, metavar=("SERVICE", "HANDLE", "NAME", "SCRATCH"))
    args = parser.parse_args(argv)
    if args.read_existing:
        handle, name, output = args.read_existing
        return _read_child(int(handle, 0), name, Path(output))
    if args.scm_worker:
        service, handle, name, scratch = args.scm_worker
        return _scm_worker(service, int(handle, 0), name, Path(scratch))
    return run_probe(args.output)


if __name__ == "__main__":
    raise SystemExit(main())
