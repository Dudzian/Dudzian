from __future__ import annotations

from typing import Any

from deployment.windows_stage9_two_branch_probe import (
    RESOURCE_NAMES,
    apply_cleanup_result,
    mandatory_cleanup,
)
from deployment.windows_tpm_substrate_probe import ProbeError


class FakeNv:
    def __init__(self, *, flush_failure: int | None = None, undefine_failure: bool = False) -> None:
        self.flush_failure = flush_failure
        self.undefine_failure = undefine_failure
        self.flushed: list[int] = []
        self.undefined: list[int] = []

    def flush(self, handle: int) -> None:
        self.flushed.append(handle)
        if handle == self.flush_failure:
            raise ProbeError("TEST_FLUSH_FAILURE", hex(handle))

    def undefine(self, handle: int) -> None:
        self.undefined.append(handle)
        if self.undefine_failure:
            raise ProbeError("TEST_UNDEFINE_FAILURE", hex(handle))


class FakeTransport:
    def __init__(self, *, close_failure: bool = False) -> None:
        self.close_failure = close_failure

    def close(self) -> None:
        if self.close_failure:
            raise OSError("test TBS close failure")


def cleanup_evidence() -> dict[str, Any]:
    return {
        "cleanup": {name: "NOT_CREATED" for name in (*RESOURCE_NAMES, "nv", "tbs_context")},
        "cleanup_handles": {name: None for name in (*RESOURCE_NAMES, "nv")},
    }


def resources() -> dict[str, int | None]:
    return {name: 0x03000000 + offset for offset, name in enumerate(RESOURCE_NAMES)}


def test_successful_mandatory_cleanup_allows_zero_exit() -> None:
    evidence = cleanup_evidence()
    nv = FakeNv()
    complete = mandatory_cleanup(
        evidence,
        nv,  # type: ignore[arg-type]
        FakeTransport(),  # type: ignore[arg-type]
        resources(),
        0x018F0001,
        True,
        True,
    )
    assert complete is True
    assert apply_cleanup_result(evidence, 0, complete) == 0
    assert set(evidence["cleanup"].values()) == {"PASS"}
    assert len(nv.flushed) == len(RESOURCE_NAMES)
    assert nv.undefined == [0x018F0001]


def test_session_or_key_flush_failure_is_fatal() -> None:
    evidence = cleanup_evidence()
    handles = resources()
    failed_name = "normal_session"
    complete = mandatory_cleanup(
        evidence,
        FakeNv(flush_failure=handles[failed_name]),  # type: ignore[arg-type]
        FakeTransport(),  # type: ignore[arg-type]
        handles,
        0x018F0001,
        True,
        True,
    )
    result = apply_cleanup_result(evidence, 0, complete)
    assert result != 0
    assert evidence["cleanup"][failed_name] == "FAIL"
    assert evidence["failure"]["reason"] == "CLEANUP_FAILED"


def test_nv_undefine_failure_is_fatal() -> None:
    evidence = cleanup_evidence()
    complete = mandatory_cleanup(
        evidence,
        FakeNv(undefine_failure=True),  # type: ignore[arg-type]
        FakeTransport(),  # type: ignore[arg-type]
        resources(),
        0x018F0001,
        True,
        True,
    )
    result = apply_cleanup_result(evidence, 0, complete)
    assert result != 0
    assert evidence["cleanup"]["nv"] == "FAIL"
    assert evidence["failure"]["reason"] == "CLEANUP_FAILED"


def test_uncreated_resources_are_explicit() -> None:
    evidence = cleanup_evidence()
    empty = {name: None for name in RESOURCE_NAMES}
    complete = mandatory_cleanup(
        evidence,
        None,
        FakeTransport(),  # type: ignore[arg-type]
        empty,
        None,
        False,
        False,
    )
    assert complete is True
    assert all(evidence["cleanup"][name] == "NOT_CREATED" for name in RESOURCE_NAMES)
    assert evidence["cleanup"]["nv"] == "NOT_CREATED"
    assert evidence["cleanup"]["tbs_context"] == "NOT_CREATED"


def test_tbs_close_success_is_caller_owned_pass() -> None:
    evidence = cleanup_evidence()
    complete = mandatory_cleanup(
        evidence,
        None,
        FakeTransport(),  # type: ignore[arg-type]
        {name: None for name in RESOURCE_NAMES},
        None,
        False,
        True,
    )
    assert complete is True
    assert evidence["cleanup"]["tbs_context"] == "PASS"
    assert apply_cleanup_result(evidence, 0, complete) == 0


def test_tbs_close_failure_is_fatal() -> None:
    evidence = cleanup_evidence()
    complete = mandatory_cleanup(
        evidence,
        None,
        FakeTransport(close_failure=True),  # type: ignore[arg-type]
        {name: None for name in RESOURCE_NAMES},
        None,
        False,
        True,
    )
    result = apply_cleanup_result(evidence, 0, complete)
    assert complete is False
    assert evidence["cleanup"]["tbs_context"] == "FAIL"
    assert result != 0
    assert evidence["failure"]["reason"] == "CLEANUP_FAILED"
