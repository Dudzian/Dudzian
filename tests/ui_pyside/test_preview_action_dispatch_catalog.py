from __future__ import annotations

from dataclasses import FrozenInstanceError, asdict

import pytest

from ui.pyside_app import preview_action_dispatch_catalog as catalog_module
from ui.pyside_app.preview_action_dispatch_audit import ACCEPTED_INTENT_NOT_EXECUTED
from ui.pyside_app.preview_action_dispatch_catalog import (
    CATALOG_KIND,
    CATALOG_SCHEMA_VERSION,
    SOURCE_PANEL,
    PaperRuntimeActionDispatchCatalog,
    build_paper_runtime_action_dispatch_catalog,
)
from ui.pyside_app.preview_action_dispatch_contract import (
    ALLOWED_PAPER_RUNTIME_ACTIONS,
    build_paper_runtime_action_dispatch_contract,
)

FORBIDDEN_CATALOG_TERMS = (
    "live",
    "testnet",
    "order",
    "account",
    "fetch",
    "export",
    "secret",
    "cloud",
)


def _catalog() -> PaperRuntimeActionDispatchCatalog:
    return build_paper_runtime_action_dispatch_catalog()


def test_catalog_contains_exactly_all_allowed_paper_actions_once_in_stable_order() -> None:
    catalog = _catalog()

    assert catalog.schema_version == CATALOG_SCHEMA_VERSION
    assert catalog.catalog_kind == CATALOG_KIND
    assert tuple(item.action for item in catalog.actions) == ALLOWED_PAPER_RUNTIME_ACTIONS
    assert catalog.allowed_actions == ALLOWED_PAPER_RUNTIME_ACTIONS
    assert catalog.action_count == len(ALLOWED_PAPER_RUNTIME_ACTIONS)
    assert len({item.action for item in catalog.actions}) == len(ALLOWED_PAPER_RUNTIME_ACTIONS)


def test_each_catalog_item_has_accepted_audit_envelope_without_execution() -> None:
    catalog = _catalog()

    assert catalog.execution_allowed is False
    assert catalog.execution_performed is False
    assert catalog.safe_to_bind_from_ui is True
    for item in catalog.actions:
        assert item.audit_status == ACCEPTED_INTENT_NOT_EXECUTED
        assert item.audit_envelope.requested_action == item.action
        assert item.audit_envelope.audit_status == ACCEPTED_INTENT_NOT_EXECUTED
        assert item.safe_to_bind_from_ui is True
        assert item.execution_allowed is False
        assert item.execution_performed is False
        assert item.audit_envelope.execution_allowed is False
        assert item.audit_envelope.execution_performed is False
        assert item.blocked_reason == ""
        assert item.refusal_reason == ""


def test_catalog_and_items_are_paper_only_local_only_with_runtime_mode_paper() -> None:
    catalog = _catalog()

    assert catalog.runtime_mode == "paper"
    assert catalog.paper_only is True
    assert catalog.local_only is True
    for item in catalog.actions:
        assert item.runtime_mode == "paper"
        assert item.paper_only is True
        assert item.local_only is True


def test_requires_operator_confirmation_matches_dispatch_evidence() -> None:
    for item in _catalog().actions:
        evidence = build_paper_runtime_action_dispatch_contract(item.action)
        assert item.requires_operator_confirmation is evidence.requires_operator_confirmation
        assert item.audit_envelope.operator_confirmation is evidence.requires_operator_confirmation


def test_catalog_excludes_rejected_live_testnet_order_account_export_secret_actions() -> None:
    catalog = _catalog()
    catalog_text = " ".join(
        [
            *catalog.allowed_actions,
            *(item.action for item in catalog.actions),
            *(item.source_control for item in catalog.actions),
        ]
    ).lower()

    for term in FORBIDDEN_CATALOG_TERMS:
        assert term not in catalog_text


def test_source_panel_and_source_control_are_deterministic_text() -> None:
    first = _catalog()
    second = _catalog()

    assert (
        tuple(item.source_panel for item in first.actions) == (SOURCE_PANEL,) * first.action_count
    )
    assert tuple(item.source_control for item in first.actions) == tuple(
        item.source_control for item in second.actions
    )
    assert tuple(item.source_control for item in first.actions) == (
        "paper-runtime-start",
        "paper-runtime-stop",
        "paper-runtime-pause",
        "paper-runtime-resume",
        "paper-runtime-snapshot-refresh",
    )
    assert all(isinstance(item.source_panel, str) for item in first.actions)
    assert all(isinstance(item.source_control, str) for item in first.actions)


def test_labels_and_descriptions_do_not_imply_live_execution_or_real_orders() -> None:
    forbidden_terms = ("live", "real", "order", "execute", "executed", "submission", "submit")

    for item in _catalog().actions:
        text = f"{item.label} {item.description}".lower()
        assert "paper" in text
        assert "local" in text
        assert "intent" in text
        assert all(term not in text for term in forbidden_terms)


def test_catalog_output_is_deterministic() -> None:
    first = _catalog()
    second = _catalog()

    assert first == second
    assert asdict(first) == asdict(second)


def test_catalog_and_nested_mappings_are_immutable_and_copy_safe() -> None:
    catalog = _catalog()

    with pytest.raises(FrozenInstanceError):
        catalog.action_count = 0  # type: ignore[misc]
    with pytest.raises(FrozenInstanceError):
        catalog.actions[0].label = "changed"  # type: ignore[misc]
    with pytest.raises(TypeError):
        catalog.boundary_checks["execution_disabled"] = False  # type: ignore[index]
    with pytest.raises(TypeError):
        catalog.actions[0].boundary_checks["execution_disabled"] = False  # type: ignore[index]

    copied_actions = list(catalog.actions)
    copied_actions.clear()
    reread = _catalog()
    assert reread.action_count == len(ALLOWED_PAPER_RUNTIME_ACTIONS)
    assert tuple(item.action for item in reread.actions) == ALLOWED_PAPER_RUNTIME_ACTIONS


def test_catalog_reuses_dispatch_audit_envelope_helper(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[str] = []
    real_builder = catalog_module._audit.build_paper_runtime_action_dispatch_audit_envelope

    def spy_builder(requested_action_or_evidence: object, **kwargs: object) -> object:
        calls.append(getattr(requested_action_or_evidence, "normalized_action", ""))
        return real_builder(requested_action_or_evidence, **kwargs)

    monkeypatch.setattr(
        catalog_module._audit,
        "build_paper_runtime_action_dispatch_audit_envelope",
        spy_builder,
    )

    catalog = _catalog()

    assert calls == list(ALLOWED_PAPER_RUNTIME_ACTIONS)
    assert all(item.audit_envelope.dispatch_evidence is not None for item in catalog.actions)


def test_boundary_checks_are_preserved_from_audit_envelopes() -> None:
    catalog = _catalog()

    assert catalog.boundary_checks["allowed_actions_complete"] is True
    assert catalog.boundary_checks["accepted_intents_only"] is True
    assert catalog.boundary_checks["execution_disabled"] is True
    assert catalog.boundary_checks["execution_not_performed"] is True
    for item in catalog.actions:
        assert item.boundary_checks == item.audit_envelope.boundary_checks
        assert item.boundary_checks["allowlisted_action"] is True
        assert item.boundary_checks["fail_closed"] is False
