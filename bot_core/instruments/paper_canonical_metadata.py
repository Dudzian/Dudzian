"""Release-owned, read-only canonical PAPER instrument metadata boundary.

Membership is established only by source-controlled inclusion in
``_RELEASE_OWNED_ENTRIES``.  Validation and hashing protect that release projection;
neither operation can promote caller content.  Catalog acceptance/history is not
implemented here.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass as _dataclass
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
from hashlib import sha256
from types import MappingProxyType
from typing import Final, Literal, Mapping

PAPER_CANONICAL_METADATA_SOURCE_ID: Final = "cryptohunter.product.paper-instrument-metadata"
PAPER_CANONICAL_METADATA_SOURCE_VERSION: Final = "1.0.0"
PAPER_CANONICAL_METADATA_CONTENT_STATUS: Final = (
    "NO_APPROVED_CANONICAL_PAPER_INSTRUMENT_ENTRIES"
)

MarketType = Literal["SPOT", "MARGIN", "PERPETUAL", "DELIVERY_FUTURES", "OPTIONS"]
InstrumentType = Literal[
    "SPOT_PAIR", "MARGIN_PAIR", "PERPETUAL_CONTRACT", "DELIVERY_FUTURE", "OPTION"
]
TradingStatus = Literal["TRADING", "HALTED", "SUSPENDED", "DELISTED", "UNKNOWN"]
MappingStatus = Literal["EXACT", "EXPLICIT_ALIAS"]
DerivativeSettlementType = Literal["LINEAR", "INVERSE"]
OptionSide = Literal["CALL", "PUT"]


@_dataclass(frozen=True, slots=True)
class _AssetReference:
    venue_asset_code: str
    canonical_display_code: str
    asset_namespace: str
    mapping_status: MappingStatus


@_dataclass(frozen=True, slots=True)
class _CanonicalPaperInstrumentMetadata:
    """Complete frozen M0.5 shape; an instance is not membership evidence."""

    instrument_id: str
    workspace_id: str
    exchange_id: str
    environment: Literal["PAPER"]
    market_type: MarketType
    instrument_type: InstrumentType
    venue_symbol: str
    display_symbol: str
    base_asset_reference: _AssetReference
    quote_asset_reference: _AssetReference
    settlement_asset_reference: _AssetReference | None
    trading_status: TradingStatus
    price_tick: str
    quantity_step: str
    min_quantity: str | None
    max_quantity: str | None
    min_notional: str | None
    max_notional: str | None
    contract_size: str | None
    contract_value_currency: str | None
    derivative_settlement_type: DerivativeSettlementType | None
    expiry_at_utc: str | None
    strike_price: str | None
    option_side: OptionSide | None
    catalog_snapshot_id: str
    metadata_version: int
    observed_at_utc: str
    effective_at_utc: str
    stale_after_utc: str
    source_adapter_family_id: str


_MARKET_INSTRUMENT_PAIRS: Final = {
    "SPOT": "SPOT_PAIR",
    "MARGIN": "MARGIN_PAIR",
    "PERPETUAL": "PERPETUAL_CONTRACT",
    "DELIVERY_FUTURES": "DELIVERY_FUTURE",
    "OPTIONS": "OPTION",
}
_TRADING_STATUSES: Final = {"TRADING", "HALTED", "SUSPENDED", "DELISTED", "UNKNOWN"}
_DECIMAL_RE: Final = re.compile(r"^(0|[1-9][0-9]*)(\.[0-9]*[1-9])?$")
_TIMESTAMP_RE: Final = re.compile(
    r"^[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}(\.[0-9]{1,9})?Z$"
)


def _nonempty(value: object) -> bool:
    return isinstance(value, str) and bool(value)


def _decimal(value: object, *, positive: bool = False) -> bool:
    if not isinstance(value, str) or _DECIMAL_RE.fullmatch(value) is None:
        return False
    try:
        number = Decimal(value)
    except (InvalidOperation, ValueError):
        return False
    return number > 0 if positive else number >= 0


def _timestamp(value: object) -> tuple[datetime, int] | None:
    if not isinstance(value, str) or _TIMESTAMP_RE.fullmatch(value) is None:
        return None
    base = value[:-1]
    fraction = ""
    if "." in base:
        base, fraction = base.split(".", 1)
    try:
        parsed = datetime.strptime(base, "%Y-%m-%dT%H:%M:%S").replace(tzinfo=timezone.utc)
    except ValueError:
        return None
    return parsed, int((fraction + "0" * 9)[:9]) if fraction else 0


def _validate_asset_reference(reference: object) -> None:
    if not isinstance(reference, _AssetReference):
        raise ValueError("malformed AssetReference")
    if not all(
        _nonempty(value)
        for value in (
            reference.venue_asset_code,
            reference.canonical_display_code,
            reference.asset_namespace,
        )
    ) or reference.mapping_status not in {"EXACT", "EXPLICIT_ALIAS"}:
        raise ValueError("malformed AssetReference")
    if reference.asset_namespace != "paper_simulated_venue":
        raise ValueError("AssetReference namespace mismatch")


def _validate_release_entry(entry: object) -> None:
    """Validate one build-time entry against frozen M0.5 record semantics."""

    if not isinstance(entry, _CanonicalPaperInstrumentMetadata):
        raise ValueError("invalid release entry type")
    required_strings = (
        entry.instrument_id,
        entry.workspace_id,
        entry.exchange_id,
        entry.environment,
        entry.market_type,
        entry.instrument_type,
        entry.venue_symbol,
        entry.display_symbol,
        entry.catalog_snapshot_id,
        entry.source_adapter_family_id,
    )
    if not all(_nonempty(value) for value in required_strings):
        raise ValueError("required Instrument scalar is empty")
    if not entry.instrument_id.startswith("instr_"):
        raise ValueError("instrument_id must use frozen instr prefix")
    if entry.exchange_id != "paper_simulated_venue" or entry.environment != "PAPER":
        raise ValueError("release entry is outside canonical PAPER scope")
    if entry.source_adapter_family_id != "paper_simulation_adapter_family":
        raise ValueError("source adapter family mismatch")
    if _MARKET_INSTRUMENT_PAIRS.get(entry.market_type) != entry.instrument_type:
        raise ValueError("invalid market/instrument pair")
    if entry.venue_symbol.strip() != entry.venue_symbol:
        raise ValueError("venue_symbol is not exact")
    for reference in (entry.base_asset_reference, entry.quote_asset_reference):
        _validate_asset_reference(reference)
    if entry.settlement_asset_reference is not None:
        _validate_asset_reference(entry.settlement_asset_reference)
    if entry.trading_status not in _TRADING_STATUSES:
        raise ValueError("invalid trading status")
    if type(entry.metadata_version) is not int or entry.metadata_version <= 0:
        raise ValueError("metadata_version must be a positive integer")
    observed, effective, stale = map(
        _timestamp, (entry.observed_at_utc, entry.effective_at_utc, entry.stale_after_utc)
    )
    if observed is None or effective is None or stale is None or not (observed <= effective < stale):
        raise ValueError("invalid timestamp graph")
    if not _decimal(entry.price_tick, positive=True) or not _decimal(
        entry.quantity_step, positive=True
    ):
        raise ValueError("invalid positive decimal")
    for value in (
        entry.min_quantity,
        entry.max_quantity,
        entry.min_notional,
        entry.max_notional,
        entry.contract_size,
        entry.strike_price,
    ):
        if value is not None and not _decimal(value):
            raise ValueError("invalid canonical decimal")
    for minimum, maximum in (
        (entry.min_quantity, entry.max_quantity),
        (entry.min_notional, entry.max_notional),
    ):
        if maximum is not None and (minimum is None or Decimal(minimum) > Decimal(maximum)):
            raise ValueError("invalid min/max constraint")

    derivative_values = (
        entry.contract_size,
        entry.contract_value_currency,
        entry.derivative_settlement_type,
        entry.expiry_at_utc,
        entry.strike_price,
        entry.option_side,
    )
    if entry.instrument_type in {"SPOT_PAIR", "MARGIN_PAIR"}:
        if any(value is not None for value in derivative_values):
            raise ValueError("cash instrument has derivative fields")
        settlement = entry.settlement_asset_reference
        if settlement is not None and (
            settlement.venue_asset_code != entry.quote_asset_reference.venue_asset_code
        ):
            raise ValueError("cash settlement asset must equal quote asset")
        return
    if (
        not _decimal(entry.contract_size, positive=True)
        or entry.settlement_asset_reference is None
        or not _nonempty(entry.contract_value_currency)
        or entry.derivative_settlement_type not in {"LINEAR", "INVERSE"}
    ):
        raise ValueError("incomplete derivative tuple")
    if entry.instrument_type == "PERPETUAL_CONTRACT":
        if any(value is not None for value in (entry.expiry_at_utc, entry.strike_price, entry.option_side)):
            raise ValueError("perpetual has dated derivative fields")
    elif entry.instrument_type == "DELIVERY_FUTURE":
        if _timestamp(entry.expiry_at_utc) is None or any(
            value is not None for value in (entry.strike_price, entry.option_side)
        ):
            raise ValueError("invalid delivery future tuple")
    elif (
        _timestamp(entry.expiry_at_utc) is None
        or not _decimal(entry.strike_price, positive=True)
        or entry.option_side not in {"CALL", "PUT"}
    ):
        raise ValueError("invalid option tuple")


def _validate_release_entries(entries: tuple[_CanonicalPaperInstrumentMetadata, ...]) -> None:
    seen_ids: set[str] = set()
    identity_to_id: dict[tuple[str, str, str, str], str] = {}
    for entry in entries:
        _validate_release_entry(entry)
        if entry.instrument_id in seen_ids:
            raise ValueError("duplicate instrument_id")
        seen_ids.add(entry.instrument_id)
        identity = (entry.exchange_id, entry.environment, entry.market_type, entry.venue_symbol)
        previous = identity_to_id.setdefault(identity, entry.instrument_id)
        if previous != entry.instrument_id:
            raise ValueError("immutable identity tuple collision")


def _entry_projection(entry: _CanonicalPaperInstrumentMetadata) -> dict[str, object]:
    def asset(value: _AssetReference | None) -> dict[str, str] | None:
        if value is None:
            return None
        return {
            "venue_asset_code": value.venue_asset_code,
            "canonical_display_code": value.canonical_display_code,
            "asset_namespace": value.asset_namespace,
            "mapping_status": value.mapping_status,
        }

    return {
        name: asset(value) if name.endswith("_asset_reference") else value
        for name, value in ((name, getattr(entry, name)) for name in entry.__dataclass_fields__)
    }


def _fingerprint_release_projection(
    source_id: str,
    source_version: str,
    content_status: str,
    entries: tuple[_CanonicalPaperInstrumentMetadata, ...],
) -> str:
    """Hash ordered exact JSON; no Unicode normalization or membership side effect."""

    payload = {
        "source_id": source_id,
        "source_version": source_version,
        "content_status": content_status,
        "entries": [_entry_projection(entry) for entry in entries],
    }
    encoded = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return sha256(encoded).hexdigest()


# Do not populate until product-owned PAPER instruments and stable IDs are approved.
_RELEASE_OWNED_ENTRIES: Final[tuple[_CanonicalPaperInstrumentMetadata, ...]] = ()
_validate_release_entries(_RELEASE_OWNED_ENTRIES)
_BY_ID: Final[Mapping[str, _CanonicalPaperInstrumentMetadata]] = MappingProxyType(
    {entry.instrument_id: entry for entry in _RELEASE_OWNED_ENTRIES}
)
PAPER_CANONICAL_METADATA_SOURCE_FINGERPRINT: Final = _fingerprint_release_projection(
    PAPER_CANONICAL_METADATA_SOURCE_ID,
    PAPER_CANONICAL_METADATA_SOURCE_VERSION,
    PAPER_CANONICAL_METADATA_CONTENT_STATUS,
    _RELEASE_OWNED_ENTRIES,
)


def canonical_paper_instruments() -> tuple[object, ...]:
    """Enumerate immutable records compiled into this release."""

    return _RELEASE_OWNED_ENTRIES


def resolve_canonical_paper_instrument(instrument_id: str) -> object | None:
    """Resolve exact release membership; caller objects and hashes are not inputs."""

    return _BY_ID.get(instrument_id)
