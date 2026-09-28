from __future__ import annotations

from datetime import datetime, timezone
import ctypes
import inspect
import struct
from types import SimpleNamespace

import pytest

from deployment import windows_tpm_substrate_probe as probe
from deployment.windows_tpm_substrate_probe import (
    NV_OWNER_FIRST,
    NV_OWNER_LAST,
    OUTPUT_NAMES,
    TPM_CC_NV_INCREMENT,
    TPM_RH_OWNER,
    _choose_unused,
    blank_evidence,
    command_packet,
    decode_tpm_rc,
    may_cleanup,
    qualify_new_counter,
    read_tpm2b,
    reconcile_pending,
    response_parameters,
    tpm2b,
    validate_evidence,
)


def test_big_endian_codec_and_bounds() -> None:
    encoded = tpm2b(b"acceptance")
    assert encoded == b"\x00\nacceptance"
    assert read_tpm2b(encoded) == (b"acceptance", len(encoded))
    with pytest.raises(ValueError, match="truncated TPM2B"):
        read_tpm2b(b"\x00\x04abc")


def test_tpm_library_constant_values_used_by_wire_codec() -> None:
    expected = {
        "TPM_ST_NO_SESSIONS": 0x8001,
        "TPM_ST_SESSIONS": 0x8002,
        "TPM_CC_NV_UNDEFINE_SPACE": 0x00000122,
        "TPM_CC_NV_INCREMENT": 0x00000134,
        "TPM_CC_NV_DEFINE_SPACE": 0x0000012A,
        "TPM_CC_NV_READ": 0x0000014E,
        "TPM_CC_NV_READ_PUBLIC": 0x00000169,
        "TPM_CC_GET_CAPABILITY": 0x0000017A,
        "TPM_CC_START_AUTH_SESSION": 0x00000176,
        "TPM_CC_POLICY_NV": 0x00000149,
        "TPM_CC_POLICY_COMMAND_CODE": 0x0000016C,
        "TPM_CC_POLICY_OR": 0x00000171,
        "TPM_CC_POLICY_GET_DIGEST": 0x00000189,
        "TPM_CC_FLUSH_CONTEXT": 0x00000165,
        "TPM_RH_OWNER": 0x40000001,
        "TPM_RH_NULL": 0x40000007,
        "TPM_ALG_SHA256": 0x000B,
        "TPM_ALG_NULL": 0x0010,
        "TPMA_NV_OWNERWRITE": 0x00000002,
        "TPMA_NV_TPM2_NT_COUNTER": 0x00000010,
        "TPMA_NV_OWNERREAD": 0x00020000,
    }
    assert {name: getattr(probe, name) for name in expected} == expected


def test_tbs_transport_uses_canonical_context_entrypoint(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[tuple[str, object]] = []

    class Function:
        def __init__(self, name: str, implementation: object) -> None:
            self.name = name
            self.implementation = implementation
            self.argtypes: object = None
            self.restype: object = None

        def __call__(self, *args: object) -> int:
            calls.append((self.name, args))
            return self.implementation(*args)  # type: ignore[operator]

    def create(params: object, output: object) -> int:
        words = ctypes.cast(params, ctypes.POINTER(ctypes.c_uint32))
        assert (words[0], words[1]) == (
            probe.TPM_VERSION_20,
            probe.TBS_CONTEXT_INCLUDE_TPM20,
        )
        ctypes.cast(output, ctypes.POINTER(probe.wintypes.HANDLE)).contents.value = 0x1234
        return probe.TBS_SUCCESS

    def submit(
        context: object,
        locality: object,
        priority: object,
        request: object,
        request_size: object,
        response: object,
        response_size: object,
    ) -> int:
        packet = struct.pack(">HII", probe.TPM_ST_NO_SESSIONS, 10, 0)
        ctypes.memmove(response, packet, len(packet))
        ctypes.cast(response_size, ctypes.POINTER(ctypes.c_uint32)).contents.value = len(packet)
        return probe.TBS_SUCCESS

    fake_dll = SimpleNamespace(
        Tbsi_Context_Create=Function("Tbsi_Context_Create", create),
        Tbsip_Submit_Command=Function("Tbsip_Submit_Command", submit),
        Tbsip_Context_Close=Function("Tbsip_Context_Close", lambda context: probe.TBS_SUCCESS),
    )
    monkeypatch.setattr(probe, "os", SimpleNamespace(name="nt"))
    monkeypatch.setattr(probe.ctypes, "WinDLL", lambda name: fake_dll, raising=False)
    evidence = blank_evidence("revision", "timestamp")
    transport = probe.TbsTransport(evidence)

    transport.open()

    assert [name for name, _ in calls] == ["Tbsi_Context_Create"]
    assert transport.context is not None
    assert transport.context.value == 0x1234
    assert evidence["tbs_available"] is True
    assert all(item["status"] == "NOT_RUN" for item in evidence["outputs"].values())
    assert transport.submit(0x123, b"request") == struct.pack(">HII", 0x8001, 10, 0)
    transport.close()
    assert [name for name, _ in calls] == [
        "Tbsi_Context_Create",
        "Tbsip_Submit_Command",
        "Tbsip_Context_Close",
    ]
    assert evidence["cleanup"]["tbs_context"] == "PASS"
    assert "Tbs" + "CreateContext" not in inspect.getsource(probe.TbsTransport)


def test_tbs_close_failure_is_preserved_in_evidence() -> None:
    class FailingClose:
        def __call__(self, context: object) -> int:
            return 0x8028400A

    evidence = blank_evidence("revision", "timestamp")
    transport = probe.TbsTransport(
        evidence=evidence,
        dll=SimpleNamespace(Tbsip_Context_Close=FailingClose()),
        context=probe.wintypes.HANDLE(1),
    )

    transport.close()

    assert evidence["cleanup"]["tbs_context"] == "FAIL:TBS_STATUS_0x8028400A"


def test_missing_canonical_tbs_context_entrypoint_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(probe, "os", SimpleNamespace(name="nt"))
    monkeypatch.setattr(probe.ctypes, "WinDLL", lambda name: SimpleNamespace(), raising=False)
    transport = probe.TbsTransport(blank_evidence("revision", "timestamp"))

    with pytest.raises(probe.ProbeError, match="TBS_CONTEXT_CREATE_ENTRYPOINT_MISSING"):
        transport.open()


@pytest.mark.parametrize(
    ("build", "product_type"),
    ((26100, 1), (22631, 1)),
)
def test_windows_11_client_qualification_accepts_supported_builds(
    build: int, product_type: int
) -> None:
    assert probe.require_windows_11_client(build, product_type) == build


@pytest.mark.parametrize(
    ("build", "product_type"),
    ((19045, 1), (26100, 3)),
)
def test_windows_11_client_qualification_rejects_old_or_server_builds(
    build: int, product_type: int
) -> None:
    with pytest.raises(probe.ProbeError, match="PHYSICAL_WINDOWS_11_REQUIRED"):
        probe.require_windows_11_client(build, product_type)


def test_session_command_marshalling_is_big_endian() -> None:
    packet = command_packet(TPM_CC_NV_INCREMENT, (TPM_RH_OWNER, 0x01801234), b"", auth=True)
    tag, size, code = struct.unpack_from(">HII", packet)
    assert tag == 0x8002
    assert size == len(packet)
    assert code == TPM_CC_NV_INCREMENT
    assert packet[10:18] == bytes.fromhex("4000000101801234")
    assert bytes.fromhex("40000009") in packet


def test_response_unmarshal_and_rc_decode() -> None:
    success = struct.pack(">HII", 0x8001, 14, 0) + b"data"
    assert response_parameters(success) == b"data"
    assert decode_tpm_rc(0) == "TPM_RC_SUCCESS"
    assert "TPM_RC_NV_DEFINED" in decode_tpm_rc(0x14C)
    assert "0x00000BAD" in decode_tpm_rc(0xBAD)


@pytest.mark.parametrize(
    ("raw", "expected_name"),
    (
        (0x092, "TPM_RC_SCHEME"),
        (0x09A, "TPM_RC_INSUFFICIENT"),
        (0x09D, "TPM_RC_POLICY_FAIL"),
        (0x146, "TPM_RC_NV_RANGE"),
        (0x149, "TPM_RC_NV_AUTHORIZATION"),
        (0x14A, "TPM_RC_NV_UNINITIALIZED"),
        (0x14B, "TPM_RC_NV_SPACE"),
        (0x14C, "TPM_RC_NV_DEFINED"),
    ),
)
def test_tpm_rc_decoder_uses_canonical_base_codes(raw: int, expected_name: str) -> None:
    decoded = decode_tpm_rc(raw)
    assert decoded.startswith(expected_name + "(")
    assert f"0x{raw:08X}" in decoded


def test_tpm_rc_decoder_strips_fmt1_index_but_preserves_full_raw_rc() -> None:
    raw_session_one_auth_fail = 0x0000098E

    decoded = decode_tpm_rc(raw_session_one_auth_fail)

    assert decoded.startswith("TPM_RC_AUTH_FAIL(")
    assert "0x0000098E" in decoded


def test_lost_response_reconciliation_issues_exactly_one_increment() -> None:
    calls = 0

    def increment() -> int:
        nonlocal calls
        calls += 1
        return 8

    assert reconcile_pending(8, 7, increment) == (8, 1)
    assert calls == 1
    assert reconcile_pending(8, 8, increment) == (8, 0)
    assert calls == 1
    with pytest.raises(Exception, match="COUNTER_RECONCILIATION_GAP"):
        reconcile_pending(10, 8, increment)


def test_cleanup_requires_exact_run_owned_handle() -> None:
    assert may_cleanup(0x01801234, True, 0x01801234)
    assert not may_cleanup(0x01801234, False, 0x01801234)
    assert not may_cleanup(0x01801234, True, 0x01801235)
    assert not may_cleanup(None, True, 0x01801234)


def test_owner_selector_is_bounded_and_skips_occupied_handles() -> None:
    assert NV_OWNER_FIRST == 0x01800000
    assert NV_OWNER_LAST == 0x01BFFFFF
    width = NV_OWNER_LAST - NV_OWNER_FIRST + 1
    samples = iter((0, width - 1, 17))
    occupied = {NV_OWNER_FIRST, NV_OWNER_LAST}

    selected = _choose_unused(occupied, lambda upper: next(samples))

    assert selected == NV_OWNER_FIRST + 17
    assert NV_OWNER_FIRST <= selected <= NV_OWNER_LAST
    assert selected < 0x01C00000
    for offset in range(0, width, 65537):
        candidate = _choose_unused(set(), lambda upper, value=offset: value)
        assert NV_OWNER_FIRST <= candidate <= NV_OWNER_LAST
        assert candidate < 0x01C00000


def test_fresh_counter_qualification_never_reads_before_initial_increment() -> None:
    class RecordingNv:
        def __init__(self) -> None:
            self.events: list[str] = []
            self.read_values = iter((0x1234_5678, 0x1234_5679))

        def define(self, handle: int) -> None:
            self.events.append("define")

        def read_public(self, handle: int) -> tuple[bytes, bytes]:
            self.events.append("read_public")
            return b"public", b"name"

        def increment(self, handle: int) -> None:
            self.events.append("increment")

        def read(self, handle: int) -> int:
            self.events.append("read")
            return next(self.read_values)

    fake = RecordingNv()
    transitions: list[dict[str, object]] = []
    milestones: list[str] = []

    public, name, initialized, after = qualify_new_counter(
        fake,
        NV_OWNER_FIRST,
        transitions,
        lambda: milestones.append("created"),
        lambda: milestones.append("read"),
    )

    assert fake.events == ["define", "read_public", "increment", "read", "increment", "read"]
    assert fake.events[:2] != ["define", "read"]
    assert milestones == ["created", "read"]
    assert (public, name) == (b"public", b"name")
    assert after == initialized + 1
    assert transitions == [
        {"purpose": "initialization", "before": None, "after": initialized},
        {"purpose": "monotonic_transition", "before": initialized, "after": after},
    ]


def test_initial_evidence_is_static_not_live_proof() -> None:
    evidence = blank_evidence("abc", datetime.now(timezone.utc).isoformat())
    validate_evidence(evidence)
    assert tuple(evidence["outputs"]) == OUTPUT_NAMES
    assert {item["status"] for item in evidence["outputs"].values()} == {"NOT_RUN"}
    assert evidence["live_physical_tpm_execution"] is False


def test_evidence_rejects_unexplained_block() -> None:
    evidence = blank_evidence("abc", datetime.now(timezone.utc).isoformat())
    evidence["outputs"][OUTPUT_NAMES[0]] = {"status": "BLOCKED", "reason": None}
    with pytest.raises(ValueError, match="requires a reason"):
        validate_evidence(evidence)
