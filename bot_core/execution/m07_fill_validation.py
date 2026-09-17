"""Structural validation for the frozen M0.7 Full Fill contract.

This module proves canonical shape and fingerprint integrity only.  It is not an
accepted economic fact, Instrument-history, Order, or lifecycle authority.
"""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import re
import unicodedata
from typing import Mapping

FILL_V1_FIELDS = frozenset({
    "fill_id", "order_id", "environment", "workspace_id", "portfolio_id",
    "exchange_account_id", "exchange_id", "instrument_id",
    "instrument_metadata_version", "execution_route_id", "venue_trade_id", "side",
    "executed_quantity", "execution_price", "executed_at_utc", "fee_kind",
    "fee_quantity", "fee_asset_reference", "fill_fingerprint_sha256",
})
FILL_FIELDS = FILL_V1_FIELDS  # historical public name
FILL_V1_FINGERPRINT_FIELDS = (
    "fill_id", "order_id", "environment", "workspace_id", "portfolio_id",
    "exchange_account_id", "exchange_id", "instrument_id",
    "instrument_metadata_version", "execution_route_id", "venue_trade_id", "side",
    "executed_quantity", "execution_price", "executed_at_utc", "fee_kind",
    "fee_quantity", "fee_asset_reference",
)
FILL_FINGERPRINT_FIELDS = FILL_V1_FINGERPRINT_FIELDS  # historical public name
FILL_V2_FINGERPRINT_DOMAIN = "cryptohunter.m0.7.full_fill.v2"
FILL_V2_FINGERPRINT_FIELDS = (
    "fill_id", "order_id", "environment", "workspace_id", "portfolio_id",
    "exchange_account_id", "exchange_id", "instrument_id",
    "instrument_metadata_version", "accepted_source_catalog_snapshot_id",
    "execution_route_id", "venue_trade_id", "side", "executed_quantity",
    "execution_price", "executed_at_utc", "fee_kind", "fee_quantity",
    "fee_asset_reference",
)
FILL_V2_FIELDS = frozenset((*FILL_V2_FINGERPRINT_FIELDS, "fill_fingerprint_sha256"))
_DECIMAL = re.compile(r"(?:0|[1-9][0-9]*)(?:\.[0-9]*[1-9])?\Z")
_ID = re.compile(r"[a-z][a-z0-9]*_[0-9a-f]{8}-[0-9a-f]{4}-7[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}\Z")
_TIME = re.compile(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d*[1-9])?Z\Z")
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_ASSET_FIELDS = {"venue_asset_code", "canonical_display_code", "asset_namespace", "mapping_status"}


class M07FillValidationError(ValueError):
    """Frozen structural Fill validation failure."""


def canonical_fill_fingerprint(fill: Mapping[str, object]) -> str:
    """Independently recompute the exact frozen fingerprint projection."""
    try:
        projected = {field: fill[field] for field in FILL_V1_FINGERPRINT_FIELDS}
    except KeyError as exc:
        raise M07FillValidationError("MALFORMED_FILL") from exc
    serialized = json.dumps(
        projected, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    )
    normalized = unicodedata.normalize("NFC", serialized)
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def canonical_fill_v2_fingerprint(fill: Mapping[str, object]) -> str:
    """Recompute the domain-separated structural v2 fingerprint."""
    try:
        projected = {field: fill[field] for field in FILL_V2_FINGERPRINT_FIELDS}
    except (KeyError, TypeError) as exc:
        raise M07FillValidationError("MALFORMED_FILL") from exc
    try:
        serialized = json.dumps(
            projected, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise M07FillValidationError("MALFORMED_FILL") from exc
    normalized = unicodedata.normalize("NFC", serialized)
    preimage = f"{FILL_V2_FINGERPRINT_DOMAIN}\0{normalized}".encode("utf-8")
    return hashlib.sha256(preimage).hexdigest()


def _id(value: object, prefix: str) -> bool:
    return type(value) is str and value.startswith(prefix + "_") and _ID.fullmatch(value) is not None


def _decimal(value: object, *, positive: bool) -> bool:
    return type(value) is str and _DECIMAL.fullmatch(value) is not None and (not positive or value != "0")


def _timestamp(value: object) -> bool:
    if type(value) is not str or _TIME.fullmatch(value) is None:
        return False
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00")).tzinfo == timezone.utc
    except ValueError:
        return False


def _sha256(value: object) -> bool:
    return type(value) is str and _SHA256.fullmatch(value) is not None


def _enum(value: object, values: tuple[str, ...]) -> bool:
    return type(value) is str and value in values


def _asset(value: object, namespace: object | None) -> bool:
    return (type(value) is dict and set(value) == _ASSET_FIELDS
            and all(type(value[name]) is str and bool(value[name]) for name in _ASSET_FIELDS)
            and _enum(value["mapping_status"], ("EXACT", "EXPLICIT_ALIAS"))
            and (namespace is None or value["asset_namespace"] == namespace))


def _validate_fields(item: dict[str, object], *, v2: bool) -> None:
    snapshot_valid = (not v2 or (
        type(item["accepted_source_catalog_snapshot_id"]) is str
        and item["accepted_source_catalog_snapshot_id"].startswith("ascat_")
        and len(item["accepted_source_catalog_snapshot_id"]) > len("ascat_")
    ))
    if (not snapshot_valid
            or not _id(item["fill_id"], "fill") or not _id(item["order_id"], "ord")
            or not _id(item["workspace_id"], "ws") or not _id(item["portfolio_id"], "port")
            or not _id(item["exchange_account_id"], "xacc") or not _id(item["instrument_id"], "instr")
            or not _id(item["execution_route_id"], "xroute")
            or not _enum(item["environment"], ("PAPER", "TESTNET", "LIVE"))
            or type(item["exchange_id"]) is not str or not item["exchange_id"]
            or type(item["venue_trade_id"]) is not str or not item["venue_trade_id"]
            or type(item["instrument_metadata_version"]) is not int
            or item["instrument_metadata_version"] <= 0
            or not _enum(item["side"], ("BUY", "SELL"))
            or not _decimal(item["executed_quantity"], positive=True)
            or not _decimal(item["execution_price"], positive=True)
            or not _timestamp(item["executed_at_utc"])
            or not _enum(item["fee_kind"], ("NONE", "CHARGE"))
            or not _decimal(item["fee_quantity"], positive=False)):
        raise M07FillValidationError("MALFORMED_FILL")
    if item["fee_kind"] == "NONE":
        legal_fee = item["fee_quantity"] == "0" and item["fee_asset_reference"] is None
    else:
        namespace = None if v2 else item["exchange_id"]
        legal_fee = item["fee_quantity"] != "0" and _asset(item["fee_asset_reference"], namespace)
    fingerprint = canonical_fill_v2_fingerprint(item) if v2 else canonical_fill_fingerprint(item)
    if (not legal_fee or not _sha256(item["fill_fingerprint_sha256"])
            or item["fill_fingerprint_sha256"] != fingerprint):
        raise M07FillValidationError("MALFORMED_FILL")


def validate_structural_fill(fill: object) -> Mapping[str, object]:
    """Validate raw structural Fill without granting accepted membership."""
    if type(fill) is not dict:
        raise M07FillValidationError("MALFORMED_FILL")
    item = dict(fill)
    fields = set(item)
    if fields == FILL_V1_FIELDS:
        v2 = False
    elif fields == FILL_V2_FIELDS:
        v2 = True
    else:
        raise M07FillValidationError("MALFORMED_FILL")
    _validate_fields(item, v2=v2)
    return item
