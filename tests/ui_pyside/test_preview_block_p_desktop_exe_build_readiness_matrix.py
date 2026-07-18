from __future__ import annotations
import copy
import json
from typing import Any
import pytest
from ui.pyside_app import preview_block_p_desktop_exe_build_readiness_matrix as matrix
from ui.pyside_app import (
    preview_block_p_desktop_exe_build_readiness_contract as contract,
)
from ui.pyside_app import (
    preview_block_p_desktop_exe_build_readiness_read_model as read_model,
)
from ui.pyside_app.preview_block_p_desktop_exe_packaging_read_model import (
    build_preview_block_p_desktop_exe_packaging_read_model,
)


def _payload() -> dict[str, Any]:
    return matrix.build_preview_block_p_desktop_exe_build_readiness_matrix()


def _blocked(payload: dict[str, Any]) -> None:
    assert (
        payload["source_18_4_accepted"],
        payload["status"],
        payload["readiness_rows"],
        payload["ready_for_build_execution"],
        payload["integrity_valid"],
    ) == (False, matrix.BLOCKED_STATUS, [], False, True)


def test_nominal_matrix_is_exact_plain_and_fail_closed() -> None:
    payload = _payload()
    assert matrix._integrity(payload) is True
    assert len(payload["readiness_rows"]) == 17
    assert all(value == "blocked" for value in payload["capability_build_readiness_state"].values())
    json.dumps(payload, sort_keys=True)


@pytest.mark.parametrize("source", [None, [], {}, {"ready_for_block_p_5": True}])
def test_forged_non_dict_or_incomplete_source_is_blocked(
    monkeypatch: pytest.MonkeyPatch, source: object
) -> None:
    monkeypatch.setattr(
        matrix, "build_preview_block_p_desktop_exe_packaging_read_model", lambda: source
    )
    _blocked(_payload())


@pytest.mark.parametrize(
    "field,value",
    [
        ("schema_version", "wrong"),
        ("block", "X"),
        ("step", "18.5"),
        ("status", "wrong"),
        (
            "status",
            "blocked_for_functional_preview_18_5_block_p_desktop_exe_packaging_read_model_source_not_accepted",
        ),
        ("packaging_read_model_artifact_complete", False),
        ("ready_for_block_p_5", False),
    ],
)
def test_forged_scalar_source_is_blocked(
    monkeypatch: pytest.MonkeyPatch, field: str, value: object
) -> None:
    source = build_preview_block_p_desktop_exe_packaging_read_model()
    source[field] = value
    monkeypatch.setattr(
        matrix, "build_preview_block_p_desktop_exe_packaging_read_model", lambda: source
    )
    _blocked(_payload())


def test_source_builder_called_once(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = 0
    source = build_preview_block_p_desktop_exe_packaging_read_model()

    def fake() -> dict[str, Any]:
        nonlocal calls
        calls += 1
        return source

    monkeypatch.setattr(matrix, "build_preview_block_p_desktop_exe_packaging_read_model", fake)
    assert _payload()["source_18_4_accepted"] is True
    assert calls == 1


@pytest.mark.parametrize(
    "mutate",
    [
        lambda p: p.__setitem__("ready_for_build_execution", True),
        lambda p: p.__setitem__("status", "ready"),
        lambda p: p.__setitem__("source_18_4_accepted", False),
        lambda p: p.__setitem__("integrity_valid", False),
        lambda p: p["capability_build_readiness_state"].__setitem__("build", "ready"),
        lambda p: p["capability_build_readiness_state"].pop("build"),
        lambda p: p["boundaries"].__setitem__("network_opened", True),
        lambda p: p["boundaries"].pop("source_only"),
        lambda p: p.__setitem__("extra", True),
        lambda p: p["readiness_rows"].reverse(),
        lambda p: p["readiness_rows"][0].__setitem__("ready", True),
        lambda p: p["readiness_rows"][0].__setitem__("source_blocker_ids", ["wrong"]),
    ],
)
def test_integrity_rejects_payload_tampering(mutate: object) -> None:
    payload = _payload()
    mutate(payload)
    assert matrix._integrity(payload) is False  # type: ignore[operator]


class TextSubclass(str):
    pass


class ListSubclass(list[str]):
    pass


class DictSubclass(dict[str, object]):
    pass


def test_payload_nested_mutation_does_not_contaminate_18_6_or_18_7() -> None:
    rows_before = copy.deepcopy(matrix.READINESS_ROWS)
    payload = matrix.build_preview_block_p_desktop_exe_build_readiness_matrix()

    payload["readiness_rows"][0]["source_requirement_ids"][0] = TextSubclass(
        payload["readiness_rows"][0]["source_requirement_ids"][0]
    )

    assert matrix.READINESS_ROWS == rows_before
    source = contract.build_preview_block_p_desktop_exe_build_readiness_contract()
    assert source["status"] == contract.STATUS
    assert "capability_contract_state" in source
    assert read_model._source_integrity(source) is True
    output = read_model.build_preview_block_p_desktop_exe_build_readiness_read_model()
    assert output["status"] == read_model.STATUS
    assert output["source_18_6_accepted"] is True
    assert read_model._integrity(output) is True


@pytest.mark.parametrize(
    "field", ["readiness_id", "readiness_state", "readiness_result", "failure_policy"]
)
def test_integrity_rejects_readiness_string_subclasses(field: str) -> None:
    payload = matrix.build_preview_block_p_desktop_exe_build_readiness_matrix()
    payload["readiness_rows"][0][field] = TextSubclass(payload["readiness_rows"][0][field])
    assert matrix._integrity(payload) is False


@pytest.mark.parametrize(
    "mutate",
    __import__(
        "tests.ui_pyside.test_preview_block_p_desktop_exe_packaging_read_model",
        fromlist=["HANDOFF_MUTATORS"],
    ).HANDOFF_MUTATORS,
)
def test_builder_rejects_forged_handoff(monkeypatch: pytest.MonkeyPatch, mutate: Any) -> None:
    source = build_preview_block_p_desktop_exe_packaging_read_model()
    mutate(source)
    monkeypatch.setattr(
        matrix, "build_preview_block_p_desktop_exe_packaging_read_model", lambda: source
    )
    _blocked(_payload())


@pytest.mark.parametrize(
    "mutate",
    [
        lambda p: p.__setitem__("readiness_rows", ListSubclass(p["readiness_rows"])),
        lambda p: p.__setitem__(
            "capability_build_readiness_state", DictSubclass(p["capability_build_readiness_state"])
        ),
        lambda p: p["readiness_rows"].__setitem__(0, DictSubclass(p["readiness_rows"][0])),
        lambda p: p["readiness_rows"][0]["source_requirement_ids"].__setitem__(
            0, TextSubclass(p["readiness_rows"][0]["source_requirement_ids"][0])
        ),
    ],
)
def test_integrity_rejects_container_subclasses(mutate: Any) -> None:
    payload = _payload()
    mutate(payload)
    assert matrix._integrity(payload) is False
