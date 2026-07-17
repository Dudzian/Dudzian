from __future__ import annotations
import copy, json
from typing import Any
import pytest
import ui.pyside_app.preview_block_p_desktop_exe_build_readiness_contract as contract
from ui.pyside_app.preview_block_p_desktop_exe_build_readiness_matrix import (
    build_preview_block_p_desktop_exe_build_readiness_matrix,
)


class HashBomb:
    hash_calls = 0

    def __hash__(self) -> int:
        type(self).hash_calls += 1
        raise RuntimeError


def test_real_18_5_builder_is_accepted(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = 0
    sources: list[dict[str, Any]] = []
    snapshots: list[dict[str, Any]] = []

    def builder() -> dict[str, Any]:
        nonlocal calls
        calls += 1
        s = build_preview_block_p_desktop_exe_build_readiness_matrix()
        sources.append(s)
        snapshots.append(copy.deepcopy(s))
        return s

    monkeypatch.setattr(
        contract, "build_preview_block_p_desktop_exe_build_readiness_matrix", builder
    )
    p = contract.build_preview_block_p_desktop_exe_build_readiness_contract()
    assert calls == 1 and sources[0] == snapshots[0] and contract._source_accepted(sources[0])
    assert (
        p["status"] == contract.STATUS
        and len(p["build_readiness_contract_rows"]) == 17
        and len(p["build_readiness_acceptance_rules"]) == 6
        and contract._integrity(p)
    )


def test_hash_bomb_fails_closed(monkeypatch: pytest.MonkeyPatch) -> None:
    s = copy.deepcopy(contract._source_template())
    s["readiness_rows"][0]["source_requirement_ids"][0] = HashBomb()
    HashBomb.hash_calls = 0
    assert not contract._source_accepted(s)
    monkeypatch.setattr(
        contract, "build_preview_block_p_desktop_exe_build_readiness_matrix", lambda: s
    )
    p = contract.build_preview_block_p_desktop_exe_build_readiness_contract()
    assert p == contract._blocked() and HashBomb.hash_calls == 0


def test_blocked_is_exact() -> None:
    p = contract._blocked()
    assert contract._integrity(p) and json.dumps(p)
    p["extra"] = False
    assert not contract._integrity(p)


@pytest.mark.parametrize(
    "key,value", [("ready_for_build_execution", True), ("integrity_valid", False)]
)
def test_tampered_source_is_blocked(monkeypatch: pytest.MonkeyPatch, key: str, value: bool) -> None:
    s = copy.deepcopy(contract._source_template())
    s[key] = value
    monkeypatch.setattr(
        contract, "build_preview_block_p_desktop_exe_build_readiness_matrix", lambda: s
    )
    assert (
        contract.build_preview_block_p_desktop_exe_build_readiness_contract() == contract._blocked()
    )


def test_exact_plain_accepts_matching_cyclic_lists() -> None:
    actual: list[object] = []
    trusted: list[object] = []
    actual.append(actual)
    trusted.append(trusted)
    assert contract._exact_plain(actual, trusted) is True


def test_exact_plain_rejects_different_cyclic_lists() -> None:
    actual: list[object] = []
    trusted: list[object] = []
    actual.append(actual)
    trusted.extend([trusted, "extra"])
    assert contract._exact_plain(actual, trusted) is False


def test_exact_plain_accepts_matching_cyclic_dicts() -> None:
    actual: dict[str, Any] = {}
    trusted: dict[str, Any] = {}
    actual["self"] = actual
    trusted["self"] = trusted
    assert contract._exact_plain(actual, trusted) is True


def test_exact_plain_rejects_different_cyclic_dicts() -> None:
    actual: dict[str, Any] = {}
    trusted: dict[str, Any] = {}
    actual["self"] = actual
    trusted["self"] = trusted
    trusted["extra"] = False
    assert contract._exact_plain(actual, trusted) is False


def test_blocked_rejects_all_top_level_mutations() -> None:
    blocked = contract._blocked()
    assert blocked["source_18_5_accepted"] is False
    for field in blocked:
        payload = copy.deepcopy(blocked)
        payload.pop(field)
        assert contract._integrity(payload) is False
    for field in (
        "source_18_5_accepted",
        "build_readiness_contract_artifact_complete",
        "ready_for_block_p_7",
        "build_ready",
        "packaging_authorized",
        "build_authorized",
        "artifact_creation_authorized",
        "integrity_valid",
    ):
        payload = copy.deepcopy(blocked)
        payload[field] = not payload[field]
        assert contract._integrity(payload) is False


@pytest.mark.parametrize("field,value", [("source_18_4_accepted", False), ("status", "wrong")])
def test_focused_source_tampering_returns_canonical_blocked(
    monkeypatch: pytest.MonkeyPatch, field: str, value: object
) -> None:
    source = copy.deepcopy(contract._source_template())
    source[field] = value
    monkeypatch.setattr(
        contract, "build_preview_block_p_desktop_exe_build_readiness_matrix", lambda: source
    )
    payload = contract.build_preview_block_p_desktop_exe_build_readiness_contract()
    assert payload == contract._blocked()
    assert contract._integrity(payload) is True


def test_main_sections_reject_missing_extra_and_reordered_fields() -> None:
    sections = [
        "block_p_desktop_exe_build_readiness_matrix_reference",
        "source_matrix_preservation",
        "build_readiness_contract_summary",
        "build_readiness_contract_principles",
        "capability_contract_state",
        "fail_closed_build_readiness_contract_decision",
        "non_execution_contract_evidence",
        "contract_boundaries",
        "source_boundaries",
    ]
    for name in sections:
        payload = copy.deepcopy(contract._nominal())
        section = payload[name]
        section.pop(next(iter(section)))
        assert contract._integrity(payload) is False
        payload = copy.deepcopy(contract._nominal())
        payload[name]["unexpected_test_field"] = False
        assert contract._integrity(payload) is False
        payload = copy.deepcopy(contract._nominal())
        payload[name] = dict(reversed(list(payload[name].items())))
        assert contract._integrity(payload) is False
