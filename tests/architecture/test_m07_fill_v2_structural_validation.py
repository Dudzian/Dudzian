from __future__ import annotations

from copy import deepcopy
import hashlib
import inspect
import json
from pathlib import Path
import unicodedata

import pytest

from bot_core.execution.m07_fill_validation import (
    FILL_V1_FIELDS,
    FILL_V2_FIELDS,
    FILL_V2_FINGERPRINT_DOMAIN,
    FILL_V2_FINGERPRINT_FIELDS,
    M07FillValidationError,
    canonical_fill_fingerprint,
    canonical_fill_v2_fingerprint,
    validate_structural_fill,
)

U = "01890f47-5f2d-7a31-8123-123456789abc"


def asset(code="BNB", namespace="binance"):
    return {
        "venue_asset_code": code,
        "canonical_display_code": code,
        "asset_namespace": namespace,
        "mapping_status": "EXACT",
    }


def fill(**changes):
    raw = {
        "fill_id": f"fill_{U}",
        "order_id": f"ord_{U}",
        "environment": "PAPER",
        "workspace_id": f"ws_{U}",
        "portfolio_id": f"port_{U}",
        "exchange_account_id": f"xacc_{U}",
        "exchange_id": "paper_simulated_venue",
        "instrument_id": f"instr_{U}",
        "instrument_metadata_version": 1,
        "accepted_source_catalog_snapshot_id": "ascat_00000000000000000001",
        "execution_route_id": f"xroute_{U}",
        "venue_trade_id": "trade-é",
        "side": "BUY",
        "executed_quantity": "0.8",
        "execution_price": "100",
        "executed_at_utc": "2025-01-01T00:00:00Z",
        "fee_kind": "CHARGE",
        "fee_quantity": "0.01",
        "fee_asset_reference": asset(),
        "fill_fingerprint_sha256": "",
    }
    raw.update(changes)
    raw["fill_fingerprint_sha256"] = canonical_fill_v2_fingerprint(raw)
    return raw


def reject(raw):
    with pytest.raises(M07FillValidationError, match="MALFORMED_FILL"):
        validate_structural_fill(raw)


def test_exact_contract_golden_vector_and_independent_oracle():
    raw = fill()
    projection = {key: raw[key] for key in FILL_V2_FINGERPRINT_FIELDS}
    encoded = unicodedata.normalize(
        "NFC", json.dumps(projection, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    )
    expected = hashlib.sha256((FILL_V2_FINGERPRINT_DOMAIN + "\0" + encoded).encode()).hexdigest()
    assert expected == "78ba98e2c021bb4b7a29933bc67d91a4686d47a37d9b3e83ecb0720f18ad289f"
    assert len(raw) == 20 and set(raw) == FILL_V2_FIELDS
    assert validate_structural_fill(raw) == raw
    raw["fill_fingerprint_sha256"] = "f" * 64
    assert canonical_fill_v2_fingerprint(raw) == expected


@pytest.mark.parametrize("snapshot", ["ascat_1", "ascat_fake", "ascat_00000000000000000001"])
def test_snapshot_syntax_positive_without_authority(snapshot):
    assert validate_structural_fill(fill(accepted_source_catalog_snapshot_id=snapshot))


@pytest.mark.parametrize(
    "snapshot", ["", "foo", "snapshot_1", "wcat_1", "ascat", "ascat_", "ASCat_1", 1]
)
def test_snapshot_syntax_negative(snapshot):
    reject(fill(accepted_source_catalog_snapshot_id=snapshot))


def test_prefix_is_derived_from_catalog_contract():
    path = (
        Path(__file__).parents[2]
        / "docs/architecture/cryptohunter_product_architecture/exchange_accounts_and_instruments.json"
    )
    prefix = json.loads(path.read_text())["accepted_source_catalog_snapshot_contract"]["id_prefix"]
    assert prefix == "ascat"
    assert fill()["accepted_source_catalog_snapshot_id"].startswith(prefix + "_")


def test_paper_namespace_mismatch_and_third_asset_are_structurally_legal():
    raw = fill(fee_asset_reference=asset("BNB", "binance"))
    assert raw["exchange_id"] != raw["fee_asset_reference"]["asset_namespace"]
    assert validate_structural_fill(raw)


def test_none_fee_and_closed_shapes():
    assert validate_structural_fill(
        fill(fee_kind="NONE", fee_quantity="0", fee_asset_reference=None)
    )
    for bad in [
        fill(fee_kind="NONE", fee_quantity="0.1", fee_asset_reference=None),
        fill(fee_kind="CHARGE", fee_quantity="0", fee_asset_reference=asset()),
    ]:
        reject(bad)
    raw = fill()
    raw["extra"] = 1
    reject(raw)
    raw = fill()
    del raw["side"]
    reject(raw)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("instrument_metadata_version", True),
        ("exchange_id", 1),
        ("executed_quantity", 1.0),
        ("fee_asset_reference", []),
        ("executed_quantity", "NaN"),
        ("execution_price", "Infinity"),
        ("executed_quantity", "+1"),
        ("executed_quantity", "01"),
        ("execution_price", "1."),
        ("fee_quantity", "-0"),
        ("executed_at_utc", "2025-01-01T00:00:00+00:00"),
    ],
)
def test_malformed_types_and_canonical_values(field, value):
    reject(fill(**{field: value}))


def test_asset_exact_types_and_mapping_status():
    class DictSubclass(dict):
        pass

    reject(fill(fee_asset_reference=DictSubclass(asset())))
    bad = asset()
    bad["mapping_status"] = 1
    reject(fill(fee_asset_reference=bad))


def test_every_input_is_bound_and_nfc_is_canonicalized():
    raw = fill()
    expected = raw["fill_fingerprint_sha256"]
    for field in FILL_V2_FINGERPRINT_FIELDS:
        changed = deepcopy(raw)
        value = changed[field]
        changed[field] = (
            {**value, "venue_asset_code": "ETH"}
            if isinstance(value, dict)
            else value + "x"
            if isinstance(value, str)
            else value + 1
        )
        assert canonical_fill_v2_fingerprint(changed) != expected
    decomposed = fill(venue_trade_id="trade-e\u0301")
    composed = fill(venue_trade_id="trade-é")
    assert decomposed["fill_fingerprint_sha256"] == composed["fill_fingerprint_sha256"]


def test_cross_version_isolation_and_legacy_regression():
    v2 = fill()
    old_hash = canonical_fill_fingerprint(v2)
    v2["fill_fingerprint_sha256"] = old_hash
    reject(v2)
    legacy = fill()
    del legacy["accepted_source_catalog_snapshot_id"]
    legacy["exchange_id"] = "binance"
    legacy["fill_fingerprint_sha256"] = canonical_fill_fingerprint(legacy)
    assert set(legacy) == FILL_V1_FIELDS and validate_structural_fill(legacy)
    upgraded = dict(legacy, accepted_source_catalog_snapshot_id="ascat_1")
    reject(upgraded)
    downgraded = fill()
    del downgraded["accepted_source_catalog_snapshot_id"]
    reject(downgraded)


def test_raw_only_signature_and_no_authority_dependencies():
    assert list(inspect.signature(validate_structural_fill).parameters) == ["fill"]
    source = inspect.getsource(inspect.getmodule(validate_structural_fill))
    for forbidden in ("CatalogRuntimeAcceptanceAuthority", "OrderAuthority", "sqlite3"):
        assert forbidden not in source


def test_implementation_status_is_deterministic_and_preserves_blockers():
    root = Path(__file__).parents[2] / "docs/architecture/cryptohunter_product_architecture"
    path = root / "m07_structural_full_fill_v2_status.json"
    status = json.loads(path.read_text())
    rendered = (
        "# M0.7 structural Full Fill v2 implementation status\n\n"
        "This is a deterministic projection of `m07_structural_full_fill_v2_status.json`; JSON is the source of truth.\n\n"
        f"```json\n{json.dumps(status, indent=2, ensure_ascii=False)}\n```\n"
    )
    assert path.with_suffix(".md").read_text() == rendered
    assert status["v2_field_count"] == len(FILL_V2_FIELDS) == 20
    assert status["v2_fingerprint_domain"] == FILL_V2_FINGERPRINT_DOMAIN
    assert status["validator_input_mode"] == "RAW_FILL_ONLY"
    assert not status["semantic_authority_enabled"]
    assert not status["historical_instrument_resolution_enabled"]
    assert not status["workspace_projection_resolution_enabled"]
    assert set(status["preserved_status"].values()) >= {"NOT_AVAILABLE", "BLOCKED", "OPEN"}
