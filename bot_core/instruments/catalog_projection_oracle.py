"""Total executable oracle for the canonical M0.5 -> M0.6 catalog chain.

Structural validation never mints source acceptance. Producer membership evidence
is resolved separately; runtime Catalog acceptance remains unimplemented.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
from typing import Any, Callable, Mapping

_CANONICAL_UTC = re.compile(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d{1,6})?Z\Z")
_MARKET_TYPES = frozenset({"SPOT", "MARGIN", "PERPETUAL", "DELIVERY_FUTURES", "OPTIONS"})
SOURCE_FIELDS = frozenset(
    {
        "accepted_source_catalog_snapshot_id",
        "source_exchange_id",
        "market_type",
        "source_adapter_family_id",
        "source_adapter_implementation_id",
        "source_adapter_release_id",
        "source_adapter_version",
        "accepted_source_producer_membership_id",
        "source_producer_generation",
        "source_producer_membership_fingerprint",
        "upstream_snapshot_or_retrieval_id",
        "observed_at_utc",
        "effective_at_utc",
        "stale_after_utc",
        "previous_snapshot_id",
        "completeness_status",
        "completeness_evidence",
        "acceptance_status",
        "member_source_product_metadata_versions",
        "content_fingerprint",
    }
)
SOURCE_MEMBER_FIELDS = frozenset(
    {"source_exchange_id", "market_type", "venue_symbol", "source_metadata_version_id"}
)
PROJECTION_FIELDS = frozenset(
    {
        "workspace_catalog_projection_id",
        "workspace_id",
        "accepted_source_catalog_snapshot_id",
        "instrument_ids",
        "member_bindings",
        "created_at_utc",
        "content_fingerprint",
    }
)
PROJECTION_MEMBER_FIELDS = frozenset(
    {
        "instrument_id",
        "source_exchange_id",
        "market_type",
        "venue_symbol",
        "instrument_metadata_version",
        "source_metadata_version_id",
    }
)
INSTRUMENT_FIELDS = frozenset(
    {
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
    }
)
SOURCE_FINGERPRINT_FIELDS = tuple(
    sorted(SOURCE_FIELDS - {"accepted_source_catalog_snapshot_id", "content_fingerprint"})
)
PROJECTION_FINGERPRINT_FIELDS = (
    "workspace_id",
    "accepted_source_catalog_snapshot_id",
    "instrument_ids",
    "member_bindings",
    "created_at_utc",
)


def _nonempty(value: Any, *, prefix: str | None = None) -> bool:
    return type(value) is str and bool(value) and (prefix is None or value.startswith(prefix + "_"))


def _timestamp(value: Any) -> datetime | None:
    """Parse only the frozen canonical UTC grammar, never permissive ISO offsets."""
    if type(value) is not str or _CANONICAL_UTC.fullmatch(value) is None:
        return None
    try:
        parsed = (
            datetime.strptime(value, "%Y-%m-%dT%H:%M:%SZ")
            if "." not in value
            else datetime.strptime(value, "%Y-%m-%dT%H:%M:%S.%fZ")
        )
    except ValueError:
        return None
    return parsed.replace(tzinfo=timezone.utc)


def _json_safe(value: Any) -> bool:
    if value is None or type(value) in {str, bool, int}:
        return True
    if type(value) is float:
        return math.isfinite(value)
    if type(value) is list:
        return all(_json_safe(item) for item in value)
    if type(value) is dict:
        return all(type(key) is str and _json_safe(item) for key, item in value.items())
    return False


def _fingerprint(domain: str, fields: tuple[str, ...], record: Mapping[str, Any]) -> str | None:
    try:
        payload = {field: record[field] for field in fields}
        if not _json_safe(payload):
            return None
        members = payload.get("member_source_product_metadata_versions")
        if isinstance(members, list):
            payload["member_source_product_metadata_versions"] = sorted(
                members,
                key=lambda item: (
                    item["source_exchange_id"],
                    item["market_type"],
                    item["venue_symbol"],
                    item["source_metadata_version_id"],
                ),
            )
        bindings = payload.get("member_bindings")
        if isinstance(bindings, list):
            payload["member_bindings"] = sorted(bindings, key=lambda item: item["instrument_id"])
        raw = (
            domain
            + "\n"
            + json.dumps(
                payload,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=False,
                allow_nan=False,
            )
        )
    except (KeyError, TypeError, ValueError):
        return None
    return hashlib.sha256(raw.encode()).hexdigest()


def validate_accepted_source_catalog_snapshot(record: Any) -> bool:
    """Validate a trusted record structurally, including PARTIAL/REJECTED history."""
    if not isinstance(record, dict) or set(record) != SOURCE_FIELDS or not _json_safe(record):
        return False
    observed, effective, stale = map(
        _timestamp,
        (
            record.get("observed_at_utc"),
            record.get("effective_at_utc"),
            record.get("stale_after_utc"),
        ),
    )
    previous = record.get("previous_snapshot_id")
    if (
        not _nonempty(record.get("accepted_source_catalog_snapshot_id"), prefix="ascat")
        or record.get("completeness_status") not in {"COMPLETE", "PARTIAL"}
        or record.get("acceptance_status") not in {"VALID", "REJECTED"}
        or type(record.get("market_type")) is not str
        or record["market_type"] not in _MARKET_TYPES
        or not all(
            _nonempty(record.get(key))
            for key in (
                "source_exchange_id",
                "source_adapter_family_id",
                "source_adapter_implementation_id",
                "source_adapter_release_id",
                "source_adapter_version",
                "upstream_snapshot_or_retrieval_id",
            )
        )
        or not _nonempty(record.get("accepted_source_producer_membership_id"), prefix="aspm")
        or type(record.get("source_producer_generation")) is not int
        or record["source_producer_generation"] < 1
        or type(record.get("source_producer_membership_fingerprint")) is not str
        or re.fullmatch(r"[0-9a-f]{64}", record["source_producer_membership_fingerprint"]) is None
        or observed is None
        or effective is None
        or stale is None
        or not observed <= effective < stale
        or (previous is not None and not _nonempty(previous, prefix="ascat"))
        or previous == record.get("accepted_source_catalog_snapshot_id")
        or not isinstance(record.get("completeness_evidence"), dict)
    ):
        return False
    members = record.get("member_source_product_metadata_versions")
    if not isinstance(members, list) or not members:
        return False
    keys: set[tuple[str, str, str]] = set()
    for member in members:
        if not isinstance(member, dict) or set(member) != SOURCE_MEMBER_FIELDS:
            return False
        if (
            not _nonempty(member.get("source_exchange_id"))
            or type(member.get("market_type")) is not str
            or member["market_type"] not in _MARKET_TYPES
            or not _nonempty(member.get("venue_symbol"))
            or not _nonempty(member.get("source_metadata_version_id"))
        ):
            return False
        key = (member["source_exchange_id"], member["market_type"], member["venue_symbol"])
        if key in keys or key[:2] != (record["source_exchange_id"], record["market_type"]):
            return False
        keys.add(key)
    return record.get("content_fingerprint") == _fingerprint(
        "cryptohunter.m0.5.accepted_source_catalog_snapshot.v2",
        SOURCE_FINGERPRINT_FIELDS,
        record,
    )


def validate_accepted_source_snapshot_membership_evidence(
    snapshot: Any, source_producer_membership_authority: Any
) -> bool:
    """Resolve exact historical authority without requiring current ACTIVE status."""
    try:
        from .source_producer_membership import SourceProducerMembershipAuthority

        if not validate_accepted_source_catalog_snapshot(snapshot):
            return False
        identity = {
            name: snapshot[name]
            for name in (
                "source_exchange_id",
                "market_type",
                "source_adapter_family_id",
                "source_adapter_implementation_id",
                "source_adapter_release_id",
                "source_adapter_version",
            )
        }
        if not isinstance(source_producer_membership_authority, SourceProducerMembershipAuthority):
            return False
        return (
            source_producer_membership_authority.resolve_historical(
                snapshot["accepted_source_producer_membership_id"],
                snapshot["source_producer_generation"],
                snapshot["source_producer_membership_fingerprint"],
                identity,
                snapshot["effective_at_utc"],
            )
            is not None
        )
    except (KeyError, TypeError, ValueError):
        return False


def validate_accepted_source_catalog_lineage(
    record: Any, accepted_source_catalog_snapshots_by_id: Mapping[str, Any]
) -> bool:
    if not validate_accepted_source_catalog_snapshot(record):
        return False
    seen = {record["accepted_source_catalog_snapshot_id"]}
    child = record
    predecessor_id = child["previous_snapshot_id"]
    while predecessor_id is not None:
        if predecessor_id in seen:
            return False
        predecessor = accepted_source_catalog_snapshots_by_id.get(predecessor_id)
        if (
            not isinstance(predecessor, dict)
            or predecessor.get("accepted_source_catalog_snapshot_id") != predecessor_id
            or not validate_accepted_source_catalog_snapshot(predecessor)
            or (predecessor["source_exchange_id"], predecessor["market_type"])
            != (record["source_exchange_id"], record["market_type"])
            or _timestamp(predecessor["effective_at_utc"]) >= _timestamp(child["effective_at_utc"])
        ):
            return False
        seen.add(predecessor_id)
        child = predecessor
        predecessor_id = predecessor["previous_snapshot_id"]
    return True


def _valid_binding(binding: Any) -> bool:
    return (
        isinstance(binding, dict)
        and set(binding) == PROJECTION_MEMBER_FIELDS
        and _json_safe(binding)
        and _nonempty(binding.get("instrument_id"))
        and _nonempty(binding.get("source_exchange_id"))
        and type(binding.get("market_type")) is str
        and binding["market_type"] in _MARKET_TYPES
        and _nonempty(binding.get("venue_symbol"))
        and type(binding.get("instrument_metadata_version")) is int
        and binding["instrument_metadata_version"] > 0
        and _nonempty(binding.get("source_metadata_version_id"))
    )


def validate_instrument_record_structure(record: Any) -> bool:
    """Full frozen M0.5 Instrument validator shared by current and history maps."""
    if not isinstance(record, dict) or set(record) != INSTRUMENT_FIELDS or not _json_safe(record):
        return False
    observed, effective, stale = map(
        _timestamp,
        (
            record.get("observed_at_utc"),
            record.get("effective_at_utc"),
            record.get("stale_after_utc"),
        ),
    )
    if not (
        _nonempty(record.get("instrument_id"))
        and _nonempty(record.get("workspace_id"))
        and _nonempty(record.get("source_exchange_id"))
        and type(record.get("market_type")) is str
        and record["market_type"] in _MARKET_TYPES
        and _nonempty(record.get("venue_symbol"))
        and _nonempty(record.get("accepted_source_catalog_snapshot_id"), prefix="ascat")
        and type(record.get("metadata_version")) is int
        and record["metadata_version"] > 0
        and observed is not None
        and effective is not None
        and stale is not None
        and observed <= effective < stale
        and _nonempty(record.get("instrument_type"))
        and _nonempty(record.get("display_symbol"))
        and _nonempty(record.get("source_adapter_family_id"))
        and record.get("trading_status")
        in {"TRADING", "HALTED", "SUSPENDED", "DELISTED", "UNKNOWN"}
    ):
        return False
    allowed_types = {
        "SPOT": "SPOT_PAIR",
        "MARGIN": "MARGIN_PAIR",
        "PERPETUAL": "PERPETUAL_CONTRACT",
        "DELIVERY_FUTURES": "DELIVERY_FUTURE",
        "OPTIONS": "OPTION",
    }
    if record["instrument_type"] != allowed_types[record["market_type"]]:
        return False
    if record["venue_symbol"].strip() != record["venue_symbol"]:
        return False

    asset_fields = {
        "venue_asset_code",
        "canonical_display_code",
        "asset_namespace",
        "mapping_status",
    }
    for name in ("base_asset_reference", "quote_asset_reference"):
        ref = record.get(name)
        if (
            not isinstance(ref, dict)
            or set(ref) != asset_fields
            or not all(_nonempty(ref.get(field)) for field in asset_fields - {"mapping_status"})
            or ref.get("mapping_status") not in {"EXACT", "EXPLICIT_ALIAS"}
            or ref["asset_namespace"] != record["source_exchange_id"]
        ):
            return False
    settlement = record.get("settlement_asset_reference")
    if settlement is not None and (
        not isinstance(settlement, dict)
        or set(settlement) != asset_fields
        or not all(_nonempty(settlement.get(field)) for field in asset_fields - {"mapping_status"})
        or settlement.get("mapping_status") not in {"EXACT", "EXPLICIT_ALIAS"}
        or settlement["asset_namespace"] != record["source_exchange_id"]
    ):
        return False

    def decimal(value: Any, *, positive: bool = False) -> bool:
        if (
            type(value) is not str
            or re.fullmatch(r"(0|[1-9][0-9]*)(\.[0-9]*[1-9])?", value) is None
        ):
            return False
        try:
            parsed = Decimal(value)
        except (InvalidOperation, ValueError):
            return False
        return parsed > 0 if positive else parsed >= 0

    if not decimal(record.get("price_tick"), positive=True) or not decimal(
        record.get("quantity_step"), positive=True
    ):
        return False
    for field in (
        "min_quantity",
        "max_quantity",
        "min_notional",
        "max_notional",
        "contract_size",
        "strike_price",
    ):
        if record.get(field) is not None and not decimal(record[field]):
            return False
    for minimum, maximum in (("min_quantity", "max_quantity"), ("min_notional", "max_notional")):
        if record.get(maximum) is not None and (
            record.get(minimum) is None or Decimal(record[minimum]) > Decimal(record[maximum])
        ):
            return False
    derivative = (
        "contract_size",
        "contract_value_currency",
        "derivative_settlement_type",
        "expiry_at_utc",
        "strike_price",
        "option_side",
    )
    kind = record["instrument_type"]
    if kind in {"SPOT_PAIR", "MARGIN_PAIR"}:
        return not any(record.get(field) is not None for field in derivative)
    if (
        not decimal(record.get("contract_size"), positive=True)
        or settlement is None
        or not _nonempty(record.get("contract_value_currency"))
        or record.get("derivative_settlement_type") not in {"LINEAR", "INVERSE"}
    ):
        return False
    if kind == "PERPETUAL_CONTRACT":
        return all(
            record.get(field) is None for field in ("expiry_at_utc", "strike_price", "option_side")
        )
    if kind == "DELIVERY_FUTURE":
        return (
            _timestamp(record.get("expiry_at_utc")) is not None
            and record.get("strike_price") is None
            and record.get("option_side") is None
        )
    return (
        _timestamp(record.get("expiry_at_utc")) is not None
        and decimal(record.get("strike_price"), positive=True)
        and record.get("option_side") in {"CALL", "PUT"}
    )


def validate_instrument_history_map(
    history: Any, current_instruments_by_id: Any | None = None
) -> bool:
    """Validate every full ordered record, including unrelated one-record histories."""
    if type(history) is not dict or (
        current_instruments_by_id is not None and type(current_instruments_by_id) is not dict
    ):
        return False
    for instrument_id, records in history.items():
        if not _nonempty(instrument_id) or not isinstance(records, list) or not records:
            return False
        identity = None
        previous_version = 0
        for record in records:
            if (
                not validate_instrument_record_structure(record)
                or record["instrument_id"] != instrument_id
                or record["metadata_version"] <= previous_version
            ):
                return False
            current_identity = tuple(
                record[field]
                for field in ("workspace_id", "source_exchange_id", "market_type", "venue_symbol")
            )
            if identity is not None and current_identity != identity:
                return False
            identity = current_identity
            previous_version = record["metadata_version"]
        if current_instruments_by_id is not None and instrument_id in current_instruments_by_id:
            current = current_instruments_by_id[instrument_id]
            if not validate_instrument_record_structure(current):
                return False
            current_identity = tuple(
                current[field]
                for field in ("workspace_id", "source_exchange_id", "market_type", "venue_symbol")
            )
            if (
                current["instrument_id"] != instrument_id
                or current_identity != identity
                or current["metadata_version"] <= previous_version
            ):
                return False
    return True


def _instrument_source_binding(record: dict[str, Any], snapshots: Mapping[str, Any]) -> bool:
    source = snapshots.get(record["accepted_source_catalog_snapshot_id"])
    return (
        isinstance(source, dict)
        and record.get("source_adapter_family_id") == source.get("source_adapter_family_id")
        and any(
            (member["source_exchange_id"], member["market_type"], member["venue_symbol"])
            == (record["source_exchange_id"], record["market_type"], record["venue_symbol"])
            for member in source["member_source_product_metadata_versions"]
        )
    )


def _valid_paper_permissions(value: Any) -> bool:
    return type(value) is frozenset and all(
        type(item) is tuple
        and len(item) == 2
        and _nonempty(item[0])
        and type(item[1]) is str
        and item[1] in _MARKET_TYPES
        for item in value
    )


def validate_canonical_context_scalars(
    workspace_ids_by_portfolio_id: Any,
    paper_source_product_permissions: Any,
    validation_time_utc: Any,
) -> bool:
    """Total preflight for non-record canonical trusted-context fields."""
    return (
        type(workspace_ids_by_portfolio_id) is dict
        and all(
            _nonempty(portfolio_id) and _nonempty(workspace_id)
            for portfolio_id, workspace_id in workspace_ids_by_portfolio_id.items()
        )
        and _valid_paper_permissions(paper_source_product_permissions)
        and _timestamp(validation_time_utc) is not None
    )


def _projection_shape(projection: Any) -> bool:
    if (
        not isinstance(projection, dict)
        or set(projection) != PROJECTION_FIELDS
        or not _json_safe(projection)
    ):
        return False
    ids, bindings = projection.get("instrument_ids"), projection.get("member_bindings")
    if (
        not _nonempty(projection.get("workspace_catalog_projection_id"), prefix="wcat")
        or not _nonempty(projection.get("workspace_id"))
        or not _nonempty(projection.get("accepted_source_catalog_snapshot_id"), prefix="ascat")
        or _timestamp(projection.get("created_at_utc")) is None
        or not isinstance(ids, list)
        or not ids
        or not all(_nonempty(iid) for iid in ids)
        or len(ids) != len(set(ids))
        or not isinstance(bindings, list)
        or len(bindings) != len(ids)
        or not all(_valid_binding(binding) for binding in bindings)
    ):
        return False
    binding_ids = [binding["instrument_id"] for binding in bindings]
    return (
        len(binding_ids) == len(set(binding_ids))
        and set(binding_ids) == set(ids)
        and projection.get("content_fingerprint")
        == _fingerprint(
            "cryptohunter.m0.5.workspace_catalog_projection.v1",
            PROJECTION_FINGERPRINT_FIELDS,
            projection,
        )
    )


def _validate_projection_members(
    projection: dict[str, Any],
    source_snapshot: dict[str, Any],
    resolve: Callable[[dict[str, Any]], Any],
) -> bool:
    source_members = {
        (
            member["source_exchange_id"],
            member["market_type"],
            member["venue_symbol"],
            member["source_metadata_version_id"],
        )
        for member in source_snapshot["member_source_product_metadata_versions"]
    }
    for binding in projection["member_bindings"]:
        instrument = resolve(binding)
        product = tuple(
            binding[key]
            for key in (
                "source_exchange_id",
                "market_type",
                "venue_symbol",
                "source_metadata_version_id",
            )
        )
        if (
            product not in source_members
            or not isinstance(instrument, dict)
            or instrument.get("instrument_id") != binding["instrument_id"]
            or instrument.get("workspace_id") != projection["workspace_id"]
            or instrument.get("accepted_source_catalog_snapshot_id")
            != projection["accepted_source_catalog_snapshot_id"]
            or instrument.get("source_adapter_family_id")
            != source_snapshot.get("source_adapter_family_id")
            or tuple(
                instrument.get(key) for key in ("source_exchange_id", "market_type", "venue_symbol")
            )
            != product[:3]
            or instrument.get("metadata_version") != binding["instrument_metadata_version"]
        ):
            return False
    return True


def validate_workspace_catalog_projection(
    projection: Any, source_snapshot: Any, instruments_by_id: Mapping[str, Any]
) -> bool:
    """Validate a current projection; all malformed nested input returns False."""
    if (
        not _projection_shape(projection)
        or not validate_accepted_source_catalog_snapshot(source_snapshot)
        or projection["accepted_source_catalog_snapshot_id"]
        != source_snapshot["accepted_source_catalog_snapshot_id"]
        or projection["workspace_catalog_projection_id"]
        == projection["accepted_source_catalog_snapshot_id"]
        or not isinstance(instruments_by_id, Mapping)
    ):
        return False
    return _validate_projection_members(
        projection, source_snapshot, lambda binding: instruments_by_id.get(binding["instrument_id"])
    )


def _validate_map_identities(projections: Mapping[str, Any], snapshots: Mapping[str, Any]) -> bool:
    return (
        isinstance(projections, Mapping)
        and isinstance(snapshots, Mapping)
        and all(
            _nonempty(key, prefix="wcat")
            and isinstance(value, dict)
            and value.get("workspace_catalog_projection_id") == key
            and _projection_shape(value)
            for key, value in projections.items()
        )
        and all(
            _nonempty(key, prefix="ascat")
            and isinstance(value, dict)
            and value.get("accepted_source_catalog_snapshot_id") == key
            and validate_accepted_source_catalog_snapshot(value)
            for key, value in snapshots.items()
        )
        and all(
            validate_accepted_source_catalog_lineage(value, snapshots)
            for value in snapshots.values()
        )
    )


def validate_workspace_catalog_projection_graph(
    projection: Any,
    snapshots: Mapping[str, Any],
    resolve: Callable[[dict[str, Any], dict[str, Any]], Any],
) -> bool:
    """Apply identical referential closure to referenced and unrelated projections."""
    if not _projection_shape(projection):
        return False
    source = snapshots.get(projection["accepted_source_catalog_snapshot_id"])
    return (
        isinstance(source, dict)
        and validate_accepted_source_catalog_lineage(source, snapshots)
        and _validate_projection_members(
            projection, source, lambda binding: resolve(projection, binding)
        )
    )


def validate_canonical_catalog_context_graph(
    *,
    workspace_catalog_projections_by_id: Any,
    accepted_source_catalog_snapshots_by_id: Any,
    instruments_by_id: Any,
    instrument_history_by_id: Any,
    source_producer_membership_authority: Any,
) -> bool:
    """Validate every canonical trusted record independently of Universe presence."""
    from .source_producer_membership import SourceProducerMembershipAuthority

    if (
        type(workspace_catalog_projections_by_id) is not dict
        or type(accepted_source_catalog_snapshots_by_id) is not dict
        or type(instruments_by_id) is not dict
        or type(instrument_history_by_id) is not dict
        or type(source_producer_membership_authority) is not SourceProducerMembershipAuthority
        or not _validate_map_identities(
            workspace_catalog_projections_by_id, accepted_source_catalog_snapshots_by_id
        )
        or not all(
            validate_accepted_source_snapshot_membership_evidence(
                snapshot, source_producer_membership_authority
            )
            for snapshot in accepted_source_catalog_snapshots_by_id.values()
        )
        or not all(
            _nonempty(key)
            and isinstance(record, dict)
            and record.get("instrument_id") == key
            and validate_instrument_record_structure(record)
            for key, record in instruments_by_id.items()
        )
        or not validate_instrument_history_map(instrument_history_by_id, instruments_by_id)
    ):
        return False
    if any(
        not _instrument_source_binding(record, accepted_source_catalog_snapshots_by_id)
        for records in instrument_history_by_id.values()
        for record in records
    ) or any(
        not _instrument_source_binding(record, accepted_source_catalog_snapshots_by_id)
        for record in instruments_by_id.values()
    ):
        return False

    def resolve(projection: dict[str, Any], binding: dict[str, Any]) -> Any:
        current = instruments_by_id.get(binding["instrument_id"])
        if (
            isinstance(current, dict)
            and current.get("metadata_version") == binding["instrument_metadata_version"]
            and current.get("accepted_source_catalog_snapshot_id")
            == projection["accepted_source_catalog_snapshot_id"]
        ):
            return current
        matches = [
            record
            for record in instrument_history_by_id.get(binding["instrument_id"], [])
            if record.get("metadata_version") == binding["instrument_metadata_version"]
            and record.get("accepted_source_catalog_snapshot_id")
            == projection["accepted_source_catalog_snapshot_id"]
        ]
        return matches[0] if len(matches) == 1 else None

    return all(
        validate_workspace_catalog_projection_graph(
            projection, accepted_source_catalog_snapshots_by_id, resolve
        )
        for projection in workspace_catalog_projections_by_id.values()
    )


def _validate_trading_universe_source_graph_prevalidated_impl(
    universe: Any,
    account: Any,
    *,
    workspace_catalog_projections_by_id: Mapping[str, Any],
    accepted_source_catalog_snapshots_by_id: Mapping[str, Any],
    instruments_by_id: Mapping[str, Any],
    now_utc: str,
    paper_source_product_permissions: frozenset[tuple[str, str]] | None = None,
    instrument_history_by_id: Mapping[str, list[dict[str, Any]]] | None = None,
    historical: bool = False,
    activation_operability: bool,
) -> bool:
    """The single canonical Universe -> workspace projection -> source path."""
    now = _timestamp(now_utc)
    if (
        (activation_operability and now is None)
        or not isinstance(universe, dict)
        or not isinstance(account, dict)
        or (
            activation_operability
            and not _valid_paper_permissions(paper_source_product_permissions)
        )
    ):
        return False
    history = {} if instrument_history_by_id is None else instrument_history_by_id
    refs, requested = universe.get("source_catalog_snapshot_ids"), universe.get("instrument_ids")
    if (
        not isinstance(refs, list)
        or not refs
        or not all(_nonempty(ref, prefix="wcat") for ref in refs)
        or len(refs) != len(set(refs))
        or not isinstance(requested, list)
        or not requested
        or not all(_nonempty(iid) for iid in requested)
    ):
        return False

    def resolve(projection: dict[str, Any], binding: dict[str, Any]) -> Any:
        if not historical:
            current = instruments_by_id.get(binding["instrument_id"])
            if activation_operability or (
                isinstance(current, dict)
                and current.get("metadata_version") == binding["instrument_metadata_version"]
                and current.get("accepted_source_catalog_snapshot_id")
                == projection["accepted_source_catalog_snapshot_id"]
            ):
                return current
        matches = [
            record
            for record in history.get(binding["instrument_id"], [])
            if record.get("metadata_version") == binding["instrument_metadata_version"]
            and record.get("accepted_source_catalog_snapshot_id")
            == projection["accepted_source_catalog_snapshot_id"]
        ]
        return matches[0] if len(matches) == 1 else None

    admitted: set[str] = set()
    resolved: dict[str, dict[str, Any]] = {}
    for projection_id in refs:
        projection = workspace_catalog_projections_by_id.get(projection_id)
        if not isinstance(projection, dict) or projection.get("workspace_id") != account.get(
            "workspace_id"
        ):
            return False
        source = accepted_source_catalog_snapshots_by_id.get(
            projection.get("accepted_source_catalog_snapshot_id")
        )
        if not validate_workspace_catalog_projection_graph(
            projection, accepted_source_catalog_snapshots_by_id, resolve
        ):
            return False
        if activation_operability and (
            not isinstance(source, dict)
            or source["acceptance_status"] != "VALID"
            or source["completeness_status"] != "COMPLETE"
            or not (
                _timestamp(source["effective_at_utc"])
                <= now
                < _timestamp(source["stale_after_utc"])
            )
        ):
            return False
        admitted.update(projection["instrument_ids"])
        for binding in projection["member_bindings"]:
            resolved[binding["instrument_id"]] = resolve(projection, binding)
    if len(requested) != len(set(requested)) or not set(requested) <= admitted:
        return False
    for iid in requested:
        instrument = resolved.get(iid)
        if not isinstance(instrument, dict):
            return False
        if instrument.get("market_type") != account.get("market_type"):
            return False
        if (
            activation_operability
            and account.get("environment") == "PAPER"
            and (
                paper_source_product_permissions is None
                or (instrument.get("source_exchange_id"), instrument.get("market_type"))
                not in paper_source_product_permissions
            )
        ):
            return False
    return True


def validate_trading_universe_source_chain(
    universe: Any,
    account: Any,
    *,
    workspace_catalog_projections_by_id: Mapping[str, Any],
    accepted_source_catalog_snapshots_by_id: Mapping[str, Any],
    instruments_by_id: Mapping[str, Any],
    now_utc: str,
    paper_source_product_permissions: frozenset[tuple[str, str]] | None = None,
    instrument_history_by_id: Mapping[str, list[dict[str, Any]]] | None = None,
    historical: bool = False,
    source_producer_membership_authority: Any = None,
) -> bool:
    """Public activation validator; operability cannot be disabled by the caller."""
    history = {} if instrument_history_by_id is None else instrument_history_by_id
    if not validate_canonical_catalog_context_graph(
        workspace_catalog_projections_by_id=workspace_catalog_projections_by_id,
        accepted_source_catalog_snapshots_by_id=accepted_source_catalog_snapshots_by_id,
        instruments_by_id=instruments_by_id,
        instrument_history_by_id=history,
        source_producer_membership_authority=source_producer_membership_authority,
    ):
        return False
    return _validate_trading_universe_source_graph_prevalidated_impl(
        universe,
        account,
        workspace_catalog_projections_by_id=workspace_catalog_projections_by_id,
        accepted_source_catalog_snapshots_by_id=accepted_source_catalog_snapshots_by_id,
        instruments_by_id=instruments_by_id,
        instrument_history_by_id=history,
        historical=historical,
        now_utc=now_utc,
        paper_source_product_permissions=paper_source_product_permissions,
        activation_operability=True,
    )


def validate_trading_universe_source_graph(
    universe: Any,
    account: Any,
    *,
    workspace_catalog_projections_by_id: Mapping[str, Any],
    accepted_source_catalog_snapshots_by_id: Mapping[str, Any],
    instruments_by_id: Mapping[str, Any],
    instrument_history_by_id: Mapping[str, list[dict[str, Any]]] | None = None,
    historical: bool = False,
    source_producer_membership_authority: Any = None,
) -> bool:
    """Structural source graph; intentionally ignores current activation operability."""
    history = {} if instrument_history_by_id is None else instrument_history_by_id
    if not validate_canonical_catalog_context_graph(
        workspace_catalog_projections_by_id=workspace_catalog_projections_by_id,
        accepted_source_catalog_snapshots_by_id=accepted_source_catalog_snapshots_by_id,
        instruments_by_id=instruments_by_id,
        instrument_history_by_id=history,
        source_producer_membership_authority=source_producer_membership_authority,
    ):
        return False
    return _validate_trading_universe_source_graph_prevalidated_impl(
        universe,
        account,
        workspace_catalog_projections_by_id=workspace_catalog_projections_by_id,
        accepted_source_catalog_snapshots_by_id=accepted_source_catalog_snapshots_by_id,
        instruments_by_id=instruments_by_id,
        instrument_history_by_id=history,
        historical=historical,
        now_utc="1970-01-01T00:00:00Z",
        paper_source_product_permissions=None,
        activation_operability=False,
    )


def _validate_trading_universe_source_graph_prevalidated(
    universe: Any,
    account: Any,
    *,
    workspace_catalog_projections_by_id: Mapping[str, Any],
    accepted_source_catalog_snapshots_by_id: Mapping[str, Any],
    instruments_by_id: Mapping[str, Any],
    instrument_history_by_id: Mapping[str, list[dict[str, Any]]],
    historical: bool,
) -> bool:
    """Private per-Universe check used only after local global-graph validation."""
    return _validate_trading_universe_source_graph_prevalidated_impl(
        universe,
        account,
        workspace_catalog_projections_by_id=workspace_catalog_projections_by_id,
        accepted_source_catalog_snapshots_by_id=accepted_source_catalog_snapshots_by_id,
        instruments_by_id=instruments_by_id,
        instrument_history_by_id=instrument_history_by_id,
        historical=historical,
        now_utc="1970-01-01T00:00:00Z",
        paper_source_product_permissions=None,
        activation_operability=False,
    )


def validate_universe_source_membership(
    universe: Any,
    account: Any,
    *,
    workspace_catalog_projections_by_id: Mapping[str, Any],
    accepted_source_catalog_snapshots_by_id: Mapping[str, Any],
    instruments_by_id: Mapping[str, Any],
    now_utc: str,
    paper_source_product_permissions: frozenset[tuple[str, str]] | None = None,
    instrument_history_by_id: Mapping[str, list[dict[str, Any]]] | None = None,
    historical: bool = False,
    source_producer_membership_authority: Any = None,
) -> bool:
    """Canonical shared-validator name; delegates without a legacy catalog path."""
    return validate_trading_universe_source_chain(
        universe,
        account,
        workspace_catalog_projections_by_id=workspace_catalog_projections_by_id,
        accepted_source_catalog_snapshots_by_id=accepted_source_catalog_snapshots_by_id,
        instruments_by_id=instruments_by_id,
        now_utc=now_utc,
        paper_source_product_permissions=paper_source_product_permissions,
        instrument_history_by_id=instrument_history_by_id,
        historical=historical,
        source_producer_membership_authority=source_producer_membership_authority,
    )


def validate_activate_trading_universe(
    universe: Any, account: Any, validation_context: Mapping[str, Any], *, now_utc: str
) -> bool:
    """ACTIVATE_TRADING_UNIVERSE operation membership boundary."""
    if not isinstance(validation_context, Mapping):
        return False
    required = (
        "workspace_catalog_projections_by_id",
        "accepted_source_catalog_snapshots_by_id",
        "instruments_by_id",
        "paper_source_product_permissions",
        "source_producer_membership_authority",
    )
    if any(key not in validation_context for key in required):
        return False
    return validate_universe_source_membership(
        universe,
        account,
        workspace_catalog_projections_by_id=validation_context[
            "workspace_catalog_projections_by_id"
        ],
        accepted_source_catalog_snapshots_by_id=validation_context[
            "accepted_source_catalog_snapshots_by_id"
        ],
        instruments_by_id=validation_context["instruments_by_id"],
        instrument_history_by_id=validation_context.get("instrument_history_by_id"),
        now_utc=now_utc,
        paper_source_product_permissions=validation_context["paper_source_product_permissions"],
        source_producer_membership_authority=validation_context[
            "source_producer_membership_authority"
        ],
    )


def execute_instrument_operation(
    operation: Any,
    universe: Any,
    account: Any,
    validation_context: Mapping[str, Any],
    *,
    now_utc: str,
) -> bool:
    """Frozen executable operation entrypoint; unknown operations fail closed."""
    if operation != "ACTIVATE_TRADING_UNIVERSE":
        return False
    return validate_activate_trading_universe(
        universe, account, validation_context, now_utc=now_utc
    )
