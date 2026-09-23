from __future__ import annotations

import hashlib
import inspect
import json
from dataclasses import replace
from pathlib import Path
from typing import get_args, get_type_hints

import pytest

import bot_core.instruments as public
import bot_core.instruments.paper_canonical_metadata as source

FROZEN_FIELDS = (
    "instrument_id",
    "workspace_id",
    "source_exchange_id",
    "market_type",
    "instrument_type",
    "venue_symbol",
    "display_symbol",
    "base_asset_reference",
    "quote_asset_reference",
    "settlement_asset_reference",
    "trading_status",
    "price_tick",
    "quantity_step",
    "min_quantity",
    "max_quantity",
    "min_notional",
    "max_notional",
    "contract_size",
    "contract_value_currency",
    "derivative_settlement_type",
    "expiry_at_utc",
    "strike_price",
    "option_side",
    "accepted_source_catalog_snapshot_id",
    "metadata_version",
    "observed_at_utc",
    "effective_at_utc",
    "stale_after_utc",
    "source_adapter_family_id",
)
M05 = json.loads(
    (
        Path(__file__).parents[2]
        / "docs/architecture/cryptohunter_product_architecture/exchange_accounts_and_instruments.json"
    ).read_text(encoding="utf-8")
)
NULLABLE_FIELDS = {
    "settlement_asset_reference",
    "min_quantity",
    "max_quantity",
    "min_notional",
    "max_notional",
    "contract_size",
    "contract_value_currency",
    "derivative_settlement_type",
    "expiry_at_utc",
    "strike_price",
    "option_side",
}


def _asset(code: str = "BASE") -> source._AssetReference:
    return source._AssetReference(code, code, "generic_testnet_venue", "EXACT")


def _spot(**changes: object) -> source._CanonicalPaperInstrumentMetadata:
    record = source._CanonicalPaperInstrumentMetadata(
        instrument_id="instr_private_candidate",
        workspace_id="workspace_private",
        source_exchange_id="generic_testnet_venue",
        market_type="SPOT",
        instrument_type="SPOT_PAIR",
        venue_symbol="PRIVATE",
        display_symbol="PRIVATE",
        base_asset_reference=_asset(),
        quote_asset_reference=_asset("QUOTE"),
        settlement_asset_reference=None,
        trading_status="TRADING",
        price_tick="0.01",
        quantity_step="0.1",
        min_quantity=None,
        max_quantity=None,
        min_notional=None,
        max_notional=None,
        contract_size=None,
        contract_value_currency=None,
        derivative_settlement_type=None,
        expiry_at_utc=None,
        strike_price=None,
        option_side=None,
        accepted_source_catalog_snapshot_id="ascat_private",
        metadata_version=1,
        observed_at_utc="2026-01-01T00:00:00Z",
        effective_at_utc="2026-01-01T00:00:00Z",
        stale_after_utc="2027-01-01T00:00:00Z",
        source_adapter_family_id="generic_testnet_adapter_family",
    )
    return replace(record, **changes)


def _perpetual(**changes: object) -> source._CanonicalPaperInstrumentMetadata:
    values = {
        "market_type": "PERPETUAL",
        "instrument_type": "PERPETUAL_CONTRACT",
        "settlement_asset_reference": _asset("SETTLE"),
        "contract_size": "1",
        "contract_value_currency": "QUOTE",
        "derivative_settlement_type": "LINEAR",
    }
    values.update(changes)
    return replace(_spot(), **values)


def test_frozen_field_names_types_and_nullability_have_exact_parity() -> None:
    hints = get_type_hints(source._CanonicalPaperInstrumentMetadata)
    assert tuple(M05["instrument_contract"]["record_fields"]) == FROZEN_FIELDS
    assert tuple(source._CanonicalPaperInstrumentMetadata.__dataclass_fields__) == FROZEN_FIELDS
    actual_nullable = {name for name, hint in hints.items() if type(None) in get_args(hint)}
    assert actual_nullable == NULLABLE_FIELDS
    assert hints["contract_value_currency"] == str | None
    assert hints["metadata_version"] is int
    assert set(get_args(hints["market_type"])) == {
        "SPOT",
        "MARGIN",
        "PERPETUAL",
        "DELIVERY_FUTURES",
        "OPTIONS",
    }
    assert set(get_args(hints["instrument_type"])) == {
        "SPOT_PAIR",
        "MARGIN_PAIR",
        "PERPETUAL_CONTRACT",
        "DELIVERY_FUTURE",
        "OPTION",
    }
    assert set(get_args(hints["trading_status"])) == {
        "TRADING",
        "HALTED",
        "SUSPENDED",
        "DELISTED",
        "UNKNOWN",
    }


def test_registry_is_stable_empty_and_public_surface_hides_nominal_constructors() -> None:
    assert source._RELEASE_OWNED_ENTRIES == ()
    assert source.canonical_paper_instruments() == ()
    assert source.PAPER_CANONICAL_METADATA_CONTENT_STATUS == (
        "NO_APPROVED_CANONICAL_PAPER_INSTRUMENT_ENTRIES"
    )
    assert not hasattr(public, "AssetReference")
    assert not hasattr(public, "CanonicalPaperInstrumentMetadata")
    assert public.__all__ == [
        "PAPER_CANONICAL_METADATA_SOURCE_FINGERPRINT",
        "PAPER_CANONICAL_METADATA_SOURCE_ID",
        "PAPER_CANONICAL_METADATA_SOURCE_VERSION",
        "PAPER_CANONICAL_METADATA_CONTENT_STATUS",
        "canonical_paper_instruments",
        "resolve_canonical_paper_instrument",
    ]


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"instrument_id": "wrong"}, "instr prefix"),
        ({"market_type": "MARGIN"}, "market/instrument pair"),
        ({"price_tick": "1.0"}, "positive decimal"),
        ({"metadata_version": 0}, "positive integer"),
        ({"stale_after_utc": "2025-01-01T00:00:00Z"}, "timestamp graph"),
    ],
)
def test_release_validator_rejects_invalid_scalar_semantics(change: dict, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        source._validate_release_entry(_spot(**change))


def test_release_validator_rejects_malformed_asset_and_derivative_tuple() -> None:
    with pytest.raises(ValueError, match="malformed AssetReference"):
        source._validate_release_entry(_spot(base_asset_reference={"venue_asset_code": "X"}))
    with pytest.raises(ValueError, match="incomplete derivative tuple"):
        source._validate_release_entry(_perpetual(contract_value_currency=None))
    with pytest.raises(ValueError, match="dated derivative fields"):
        source._validate_release_entry(_perpetual(expiry_at_utc="2026-02-01T00:00:00Z"))


def test_release_validator_accepts_empty_registry_and_rejects_duplicate_ids() -> None:
    source._validate_release_entries(())
    candidate = _spot()
    with pytest.raises(ValueError, match="duplicate instrument_id"):
        source._validate_release_entries((candidate, candidate))


def _recompute(entries: tuple[source._CanonicalPaperInstrumentMetadata, ...]) -> str:
    payload = {
        "source_id": source.PAPER_CANONICAL_METADATA_SOURCE_ID,
        "source_version": source.PAPER_CANONICAL_METADATA_SOURCE_VERSION,
        "content_status": source.PAPER_CANONICAL_METADATA_CONTENT_STATUS,
        "entries": [source._entry_projection(entry) for entry in entries],
    }
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode(
        "utf-8"
    )
    return hashlib.sha256(encoded).hexdigest()


def test_fingerprint_exactly_binds_ordered_actual_release_projection() -> None:
    assert source.PAPER_CANONICAL_METADATA_SOURCE_FINGERPRINT == _recompute(())
    assert _recompute(()) == _recompute(())
    candidate = _spot()
    baseline = source._fingerprint_release_projection("id", "1", "status", (candidate,))
    assert baseline != source._fingerprint_release_projection("id", "2", "status", (candidate,))
    assert baseline != source._fingerprint_release_projection(
        "id", "1", "status", (replace(candidate, display_symbol="changed"),)
    )
    nested = replace(
        candidate,
        base_asset_reference=replace(
            candidate.base_asset_reference, canonical_display_code="changed"
        ),
    )
    assert baseline != source._fingerprint_release_projection("id", "1", "status", (nested,))
    other = replace(candidate, instrument_id="instr_other", venue_symbol="OTHER")
    assert source._fingerprint_release_projection("id", "1", "status", (candidate, other)) != (
        source._fingerprint_release_projection("id", "1", "status", (other, candidate))
    )


def test_every_canonical_field_and_nested_asset_change_fingerprint() -> None:
    candidate = _spot()
    baseline = source._fingerprint_release_projection("id", "1", "status", (candidate,))
    for field in FROZEN_FIELDS:
        value = getattr(candidate, field)
        if field in {"base_asset_reference", "quote_asset_reference"}:
            changed = replace(candidate, **{field: replace(value, venue_asset_code="changed")})
        elif field == "metadata_version":
            changed = replace(candidate, metadata_version=2)
        elif field == "settlement_asset_reference":
            changed = replace(candidate, settlement_asset_reference=_asset("changed"))
        elif value is None:
            changed = replace(candidate, **{field: "changed"})
        else:
            changed = replace(candidate, **{field: f"{value}-changed"})
        assert baseline != source._fingerprint_release_projection("id", "1", "status", (changed,))


def test_caller_nominal_instance_and_matching_self_hash_cannot_enroll() -> None:
    caller_record = _spot()
    caller_hash = source._fingerprint_release_projection(
        source.PAPER_CANONICAL_METADATA_SOURCE_ID,
        source.PAPER_CANONICAL_METADATA_SOURCE_VERSION,
        source.PAPER_CANONICAL_METADATA_CONTENT_STATUS,
        (caller_record,),
    )
    assert source.resolve_canonical_paper_instrument(caller_record.instrument_id) is None
    assert source.resolve_canonical_paper_instrument(caller_hash) is None


def test_no_runtime_writer_or_caller_controlled_input_surface() -> None:
    public_functions = {
        name
        for name, value in vars(source).items()
        if not name.startswith("_") and inspect.isfunction(value)
    }
    assert public_functions == {"canonical_paper_instruments", "resolve_canonical_paper_instrument"}
    assert tuple(inspect.signature(source.canonical_paper_instruments).parameters) == ()
    assert tuple(inspect.signature(source.resolve_canonical_paper_instrument).parameters) == (
        "instrument_id",
    )
