from __future__ import annotations

import copy
import json
from typing import Any, cast

import pytest

from ui.pyside_app import (
    preview_block_p_desktop_exe_build_readiness_contract as source_contract_module,
)
from ui.pyside_app import preview_block_p_desktop_exe_build_readiness_read_model as read_model
from ui.pyside_app.preview_block_p_desktop_exe_build_readiness_contract import (
    build_preview_block_p_desktop_exe_build_readiness_contract,
)


def test_does_not_call_upstream_private_nominal(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = 0

    def fail_private_nominal() -> dict[str, Any]:
        nonlocal calls
        calls += 1
        raise RuntimeError("18.7 must not call upstream private nominal factory")

    monkeypatch.setattr(source_contract_module, "_nominal", fail_private_nominal)
    trusted = read_model._trusted_source_template()
    nominal = read_model._nominal()
    blocked = read_model._blocked()

    assert trusted["status"] == read_model.SOURCE_STATUS
    assert read_model._integrity(nominal) is True
    assert read_model._integrity(blocked) is True
    assert calls == 0


def test_local_template_matches_real_18_6_builder_once_and_is_immutable() -> None:
    calls = 0
    sources: list[dict[str, Any]] = []

    def real_source_wrapper() -> dict[str, Any]:
        nonlocal calls
        calls += 1
        source = cast(dict[str, Any], build_preview_block_p_desktop_exe_build_readiness_contract())
        sources.append(source)
        return source

    real_source = real_source_wrapper()
    snapshot = copy.deepcopy(real_source)
    trusted = read_model._trusted_source_template()

    assert calls == 1
    assert read_model._exact_plain(real_source, trusted) is True
    assert read_model._source_accepted(real_source) is True
    assert real_source == snapshot


def test_public_builder_uses_real_18_6_builder_once_and_returns_nominal(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = 0
    snapshots: list[dict[str, Any]] = []

    def real_source_wrapper() -> dict[str, Any]:
        nonlocal calls
        calls += 1
        source = cast(dict[str, Any], build_preview_block_p_desktop_exe_build_readiness_contract())
        snapshots.append(copy.deepcopy(source))
        return source

    monkeypatch.setattr(
        read_model,
        "build_preview_block_p_desktop_exe_build_readiness_contract",
        real_source_wrapper,
    )
    payload = read_model.build_preview_block_p_desktop_exe_build_readiness_read_model()

    assert calls == 1
    assert len(snapshots) == 1
    assert read_model._exact_plain(snapshots[0], read_model._trusted_source_template()) is True
    assert read_model._source_accepted(snapshots[0]) is True
    assert payload["status"] == read_model.STATUS
    assert payload["source_18_6_accepted"] is True
    assert read_model._integrity(payload) is True
    assert json.dumps(payload)


def test_counts_preservation_decision_and_zero_authorization() -> None:
    source = build_preview_block_p_desktop_exe_build_readiness_contract()
    payload = read_model._nominal(source)
    summary = payload["build_readiness_read_model_summary"]
    assert len(payload["readiness_clause_read_rows"]) == 17
    assert len(payload["acceptance_rule_read_rows"]) == 6
    assert summary["unique_requirement_count"] == 8
    assert summary["unique_blocker_count"] == 12
    assert summary["unique_evidence_count"] == 12
    for key in (
        "satisfied_readiness_clause_count",
        "observed_readiness_count",
        "validated_readiness_count",
        "satisfied_readiness_count",
        "ready_readiness_count",
        "satisfied_acceptance_rule_count",
    ):
        assert summary[key] == 0
    for key in (
        "build_readiness_contract_satisfied",
        "build_ready",
        "packaging_authorized",
        "build_authorized",
        "artifact_creation_authorized",
        "release_authorized",
        "runtime_authorized",
        "orders_authorized",
    ):
        assert summary[key] is False
    decision = payload["fail_closed_readiness_decision_view"]
    assert decision["contract_satisfied"] is False
    assert decision["build_ready"] is False
    assert decision["only_source_only_18_8_handoff_allowed"] is True
    for key in decision:
        if key.endswith("by_18_6"):
            assert decision[key] is False


def test_rows_and_rules_preserve_source_order_and_links() -> None:
    source = build_preview_block_p_desktop_exe_build_readiness_contract()
    payload = read_model._nominal(source)
    assert [r["readiness_id"] for r in payload["readiness_clause_read_rows"]] == [
        r["readiness_id"] for r in source["build_readiness_contract_rows"]
    ]
    assert [r["contract_clause_id"] for r in payload["readiness_clause_read_rows"]] == [
        r["contract_clause_id"] for r in source["build_readiness_contract_rows"]
    ]
    for out, src in zip(
        payload["readiness_clause_read_rows"], source["build_readiness_contract_rows"], strict=True
    ):
        for key in ("source_requirement_ids", "source_blocker_ids", "required_evidence_ids"):
            assert out[key] == src[key]
        assert out["contract_clause_satisfied"] is False
        assert out["build_authorization_granted"] is False
    assert [r["rule_id"] for r in payload["acceptance_rule_read_rows"]] == [
        r["rule_id"] for r in source["build_readiness_acceptance_rules"]
    ]


def test_capability_maps_all_blocked() -> None:
    payload = read_model._nominal(build_preview_block_p_desktop_exe_build_readiness_contract())
    for mapping in payload["capability_read_model_state"].values():
        assert all(value == "blocked" for value in mapping.values())


@pytest.mark.parametrize(
    "mutate",
    [
        lambda s: s.__setitem__("source_18_5_accepted", False),
        lambda s: s.__setitem__("build_readiness_contract_artifact_complete", False),
        lambda s: s.__setitem__("ready_for_block_p_7", False),
        lambda s: s.__setitem__("integrity_valid", False),
        lambda s: s.__setitem__("status", "wrong"),
        lambda s: s.__setitem__("extra", False),
        lambda s: s.pop("integrity_valid"),
        lambda s: (
            s.clear()
            or s.update(dict(reversed(list(read_model._trusted_source_template().items()))))
        ),
    ],
)
def test_source_tampering_uses_public_builder_and_returns_canonical_blocked(
    mutate: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = 0
    source = copy.deepcopy(read_model._trusted_source_template())
    mutate(source)
    assert read_model._source_accepted(source) is False

    def builder() -> dict[str, Any]:
        nonlocal calls
        calls += 1
        return source

    monkeypatch.setattr(
        read_model, "build_preview_block_p_desktop_exe_build_readiness_contract", builder
    )
    payload = read_model.build_preview_block_p_desktop_exe_build_readiness_read_model()

    assert calls == 1
    assert payload == read_model._blocked()
    assert read_model._integrity(payload) is True
    assert json.dumps(payload)


def test_canonical_blocked_18_6_source_returns_canonical_blocked_18_7(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = 0
    blocked_18_6 = source_contract_module._blocked()
    assert read_model._source_accepted(blocked_18_6) is False

    def builder() -> dict[str, Any]:
        nonlocal calls
        calls += 1
        return blocked_18_6

    monkeypatch.setattr(
        read_model, "build_preview_block_p_desktop_exe_build_readiness_contract", builder
    )
    payload = read_model.build_preview_block_p_desktop_exe_build_readiness_read_model()

    assert calls == 1
    assert payload == read_model._blocked()
    assert read_model._integrity(payload) is True
    assert json.dumps(payload)


def test_canonical_blocked_exact_integrity_and_authorizations() -> None:
    blocked = read_model._blocked()
    assert blocked == read_model._blocked()
    assert read_model._integrity(blocked) is True
    assert json.dumps(blocked)
    extra = copy.deepcopy(blocked)
    extra["extra"] = False
    assert read_model._integrity(extra) is False
    for key in read_model.BLOCKED_FIELDS:
        missing = copy.deepcopy(blocked)
        missing.pop(key)
        assert read_model._integrity(missing) is False
    assert read_model._integrity(dict(reversed(list(blocked.items())))) is False
    for key in (
        "build_ready",
        "packaging_authorized",
        "build_authorized",
        "artifact_creation_authorized",
    ):
        changed = copy.deepcopy(blocked)
        changed[key] = True
        assert read_model._integrity(changed) is False


def test_comparator_cycles_and_deep_graphs() -> None:
    a: list[Any] = []
    b: list[Any] = []
    a.append(a)
    b.append(b)
    assert read_model._exact_plain(a, b) is True
    c: list[Any] = [False]
    c.append(c)
    assert read_model._exact_plain(c, b) is False
    da: dict[str, Any] = {}
    db: dict[str, Any] = {}
    da["self"] = da
    db["self"] = db
    assert read_model._exact_plain(da, db) is True
    dc: dict[str, Any] = {"self": None}
    dc["self"] = dc
    dc["x"] = False
    assert read_model._exact_plain(dc, db) is False
    left: dict[str, Any] = {"value": 0}
    right: dict[str, Any] = {"value": 0}
    for _ in range(1500):
        left = {"next": left}
        right = {"next": right}
    assert read_model._exact_plain(left, right) is True
    mismatch: dict[str, Any] = {"value": 1}
    for _ in range(1500):
        mismatch = {"next": mismatch}
    assert read_model._exact_plain(left, mismatch) is False
