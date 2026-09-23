"""S9D-C25-R1-FIX regressions for typed M0.8 source trust and journal restore."""

from dataclasses import replace
from fractions import Fraction

import pytest

from bot_core.accounting import (
    ACCOUNTING_RULE_VERSION,
    AccountingAuthority,
    AccountingAuthorityError,
    AssetReference,
    AtomicAccountingState,
    CoreAcceptedAccountingFactProjection,
    InMemoryAccountingCarrier,
    InMemoryAccountingFactCarrier,
    LedgerEntry,
)
from bot_core.m09_kill_switch_authority import (
    CoreAcceptedContentAuthority,
    CoreAcceptedContentBinding,
    InMemoryCoreAcceptedContentCarrier,
)
from bot_core.persistence.fingerprints import canonical_json_sha256

U1 = "01890f47-5f2d-7a31-8123-123456789abc"
U2 = "01890f47-5f2d-7a31-8123-123456789abd"
WS, PORT, XACC = f"ws_{U1}", f"port_{U1}", f"xacc_{U1}"
BTC = AssetReference("BTC", "BTC", "binance", "EXACT")


def economic(
    identity=f"evt_{U1}",
    *,
    source_type="deposit",
    quantity="2",
    workspace=WS,
    portfolio=PORT,
    environment="PAPER",
    account=XACC,
    asset=BTC,
    effective="2025-01-01T00:00:00Z",
):
    payload = {
        "audit_event_id": identity,
        "source_type": source_type,
        "workspace_id": workspace,
        "portfolio_id": portfolio,
        "environment": environment,
        "effective_at_utc": effective,
        "provenance": "EXTERNAL",
        "exchange_account_id": account,
        "asset_reference": vars_asset(asset),
        "quantity": quantity,
        "capital_flow_kind": "EXTERNAL_CONTRIBUTION"
        if source_type == "deposit"
        else "EXTERNAL_WITHDRAWAL",
        "basis_valuation_unit": vars_asset(AssetReference("USDT", "USDT", "binance", "EXACT")),
        "unit_cost_basis": "10",
    }
    payload["source_fingerprint_sha256"] = canonical_json_sha256(payload)
    return payload


def vars_asset(asset):
    return {
        "venue_asset_code": asset.venue_asset_code,
        "canonical_display_code": asset.canonical_display_code,
        "asset_namespace": asset.asset_namespace,
        "mapping_status": asset.mapping_status,
    }


def source_authority(*payloads):
    content, content_writer = CoreAcceptedContentAuthority.compose(
        InMemoryCoreAcceptedContentCarrier()
    )
    source, source_writer = CoreAcceptedAccountingFactProjection.compose(
        InMemoryAccountingFactCarrier(), content_membership=content
    )
    for payload in payloads:
        content_writer.accept(
            CoreAcceptedContentBinding(
                payload["audit_event_id"], payload["source_fingerprint_sha256"]
            )
        )
        source_writer.accept(payload)
    return source


def accounting(source, state=None):
    carrier = InMemoryAccountingCarrier(state)
    view, writer = AccountingAuthority.compose(carrier, source_authority=source)
    return carrier, view, writer


def query(view, **changes):
    values = dict(
        workspace_id=WS,
        portfolio_id=PORT,
        environment="PAPER",
        exchange_account_id=XACC,
        asset_reference=BTC,
    )
    values.update(changes)
    return view.resolve_internal_quantity(**values)


def test_exact_deposit_and_withdrawal_are_derived_not_caller_postings():
    deposit, withdrawal = economic(), economic(f"evt_{U2}", source_type="withdrawal")
    source = source_authority(deposit, withdrawal)
    carrier, view, writer = accounting(source)
    first = writer.accept_source(deposit["audit_event_id"])
    second = writer.accept_source(withdrawal["audit_event_id"])
    assert first.batch_fingerprint_sha256 != ""
    assert query(view).status == "INTERNAL_HISTORY_PRESENT"
    assert query(view).quantity == Fraction(0)
    assert [entry.account_role for entry in view.journal()] == [
        "OWNED_AVAILABLE",
        "EXTERNAL_CAPITAL",
        "EXTERNAL_CAPITAL",
        "OWNED_AVAILABLE",
    ]
    assert carrier.read().store_revision == 2


@pytest.mark.parametrize("source_type", ["deposit", "withdrawal"])
def test_raw_unaccepted_economic_fact_is_rejected(source_type):
    payload = economic(source_type=source_type)
    source = source_authority()
    carrier, _, writer = accounting(source)
    with pytest.raises(AccountingAuthorityError, match="TRUSTED_CONTEXT_FAILURE"):
        writer.accept_source(payload["audit_event_id"])
    assert carrier.read() == AtomicAccountingState()


@pytest.mark.parametrize(
    "change",
    [
        {"quantity": "999"},
        {"source_type": "withdrawal"},
        {"workspace_id": f"ws_{U2}"},
        {"portfolio_id": f"port_{U2}"},
        {"environment": "LIVE"},
        {"exchange_account_id": f"xacc_{U2}"},
        {"asset_reference": vars_asset(AssetReference("XBT", "BTC", "binance", "EXPLICIT_ALIAS"))},
        {"effective_at_utc": "2025-02-01T00:00:00Z"},
    ],
)
def test_pre_enrolled_identity_and_fingerprint_cannot_authorize_changed_economics(change):
    trusted = economic()
    content, content_writer = CoreAcceptedContentAuthority.compose(
        InMemoryCoreAcceptedContentCarrier()
    )
    content_writer.accept(
        CoreAcceptedContentBinding(trusted["audit_event_id"], trusted["source_fingerprint_sha256"])
    )
    source, owner = CoreAcceptedAccountingFactProjection.compose(
        InMemoryAccountingFactCarrier(), content_membership=content
    )
    forged = {**trusted, **change}  # deliberately retains the syntactically valid trusted SHA
    with pytest.raises(AccountingAuthorityError, match="TRUSTED_CONTEXT_FAILURE"):
        owner.accept(forged)
    carrier, _, _ = accounting(source)
    assert carrier.read() == AtomicAccountingState()


@pytest.mark.parametrize(
    "blocked",
    [
        "fill",
        "internal_transfer",
        "capital_reservation",
        "capital_release",
        "reconciliation_correction",
    ],
)
def test_sources_without_required_production_upstream_context_are_explicitly_unsupported(blocked):
    payload = economic(source_type=blocked)
    source = source_authority()
    _, _, _ = accounting(source)
    with pytest.raises(AccountingAuthorityError, match="UNSUPPORTED_ACCOUNTING_SEMANTICS"):
        # A nominal identity/fingerprint is not a Fill, command, or terminal-event authority.
        CoreAcceptedAccountingFactProjection._validate_payload(payload)


def test_restore_revalidates_typed_upstream_and_empty_authority_rejects_state():
    payload = economic()
    source = source_authority(payload)
    carrier, _, writer = accounting(source)
    writer.accept_source(payload["audit_event_id"])
    persisted = carrier.read()
    empty = source_authority()
    with pytest.raises(AccountingAuthorityError, match="CONTRACT_INCONSISTENT"):
        AccountingAuthority(InMemoryAccountingCarrier(persisted), source_authority=empty)
    assert carrier.read() == persisted


def test_identical_postings_keep_batch_fingerprint_but_get_globally_unique_entry_ids():
    first, second = economic(), economic(f"evt_{U2}")
    source = source_authority(first, second)
    carrier, view, writer = accounting(source)
    a = writer.accept_source(first["audit_event_id"])
    b = writer.accept_source(second["audit_event_id"])
    entries = view.journal()
    assert a.batch_fingerprint_sha256 == b.batch_fingerprint_sha256
    assert entries[0].ledger_entry_id != entries[2].ledger_entry_id
    assert entries[1].ledger_entry_id != entries[3].ledger_entry_id
    assert len({entry.ledger_entry_id for entry in entries}) == len(entries)
    assert [entry.append_sequence for entry in entries] == [1, 2, 3, 4]
    assert carrier.read().store_revision == 2


def test_exact_replay_preserves_ids_and_does_not_mutate_carrier():
    payload = economic()
    source = source_authority(payload)
    carrier, view, writer = accounting(source)
    accepted = writer.accept_source(payload["audit_event_id"])
    state, ids = carrier.read(), tuple(entry.ledger_entry_id for entry in view.journal())
    assert writer.accept_source(payload["audit_event_id"]) == accepted
    assert carrier.read() == state
    assert tuple(entry.ledger_entry_id for entry in view.journal()) == ids


def test_restore_explicitly_rejects_duplicate_ledger_entry_id():
    payload = economic()
    source = source_authority(payload)
    carrier, _, writer = accounting(source)
    writer.accept_source(payload["audit_event_id"])
    state = carrier.read()
    forged = replace(
        state,
        journal=(
            state.journal[0],
            replace(state.journal[1], ledger_entry_id=state.journal[0].ledger_entry_id),
        ),
    )
    with pytest.raises(AccountingAuthorityError, match="CONTRACT_INCONSISTENT"):
        AccountingAuthority(InMemoryAccountingCarrier(forged), source_authority=source)


def test_current_projection_ignores_observation_time_and_missing_is_scope_only():
    deposit, withdrawal = (
        economic(),
        economic(f"evt_{U2}", source_type="withdrawal", effective="2025-03-01T00:00:00Z"),
    )
    source = source_authority(deposit, withdrawal)
    carrier, view, writer = accounting(source)
    assert query(view).status == "MISSING_INTERNAL_HISTORY"
    writer.accept_source(deposit["audit_event_id"])
    # An external observation dated 2024 would still see this accepted Jan-2025
    # entry in the current internal projection; observation time is not an API
    # argument and cannot turn PRESENT back into MISSING.
    jan_entry_with_earlier_observation = query(view)
    assert jan_entry_with_earlier_observation.status == "INTERNAL_HISTORY_PRESENT"
    assert jan_entry_with_earlier_observation.quantity == 2
    assert query(view, environment="TESTNET").status == "MISSING_INTERNAL_HISTORY"
    before = carrier.read()
    carrier.fail_next = True
    with pytest.raises(OSError, match="INJECTED_CARRIER_FAILURE"):
        writer.accept_source(withdrawal["audit_event_id"])
    assert carrier.read() == before
    writer.accept_source(withdrawal["audit_event_id"])
    # Parity oracle: even for an external observation as_of Feb, the accepted
    # future-effective Mar withdrawal participates in the current projection.
    zero = query(view)
    assert zero.status == "INTERNAL_HISTORY_PRESENT" and zero.quantity == 0


def test_negative_resulting_owned_balance_is_rejected_without_mutation():
    withdrawal = economic(source_type="withdrawal")
    source = source_authority(withdrawal)
    carrier, _, writer = accounting(source)
    with pytest.raises(AccountingAuthorityError, match="INVALID_ACCOUNTING_TRANSITION"):
        writer.accept_source(withdrawal["audit_event_id"])
    assert carrier.read() == AtomicAccountingState()


def test_raw_ledger_and_caller_quantity_have_no_acceptance_api():
    source = source_authority()
    _, view, _ = accounting(source)
    assert not hasattr(view, "accept_balance")
    assert not hasattr(view, "accept_entry")
    assert not hasattr(view, "accept_batch")
    assert len(LedgerEntry.__dataclass_fields__) == 23
    assert ACCOUNTING_RULE_VERSION == "ACCOUNTING_SPOT_FIFO_V1"
