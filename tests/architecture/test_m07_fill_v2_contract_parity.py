"""Executable parity checks between canonical M0.7 Fill versions and production."""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import pytest

from bot_core.execution.m07_fill_validation import (
    FILL_V1_FIELDS,
    FILL_V1_FINGERPRINT_FIELDS,
    FILL_V2_FIELDS,
    FILL_V2_FINGERPRINT_DOMAIN,
    FILL_V2_FINGERPRINT_FIELDS,
    M07FillValidationError,
    canonical_fill_fingerprint,
    canonical_fill_v2_fingerprint,
    validate_structural_fill,
)

DOCS = Path(__file__).parents[2] / "docs/architecture/cryptohunter_product_architecture"
M07 = json.loads((DOCS / "commands_events_order_lifecycle_and_idempotency.json").read_text())
M05 = json.loads((DOCS / "exchange_accounts_and_instruments.json").read_text())
VERSIONS = M07["fill_contract_versions"]
V2 = VERSIONS["v2"]
U = "01890f47-5f2d-7a31-8123-123456789abc"


def v2_fill(**changes: object) -> dict[str, object]:
    value: dict[str, object] = {
        "fill_id": f"fill_{U}",
        "order_id": f"ord_{U}",
        "environment": "PAPER",
        "workspace_id": f"ws_{U}",
        "portfolio_id": f"port_{U}",
        "exchange_account_id": f"xacc_{U}",
        "exchange_id": "paper_simulated_venue",
        "instrument_id": f"instr_{U}",
        "instrument_metadata_version": 1,
        "accepted_source_catalog_snapshot_id": "ascat_fake",
        "execution_route_id": f"xroute_{U}",
        "venue_trade_id": "trade-1",
        "side": "BUY",
        "executed_quantity": "1",
        "execution_price": "100",
        "executed_at_utc": "2025-01-01T00:00:00Z",
        "fee_kind": "CHARGE",
        "fee_quantity": "1",
        "fee_asset_reference": {
            "venue_asset_code": "BNB",
            "canonical_display_code": "BNB",
            "asset_namespace": "binance",
            "mapping_status": "EXACT",
        },
        "fill_fingerprint_sha256": "",
    }
    value.update(changes)
    value["fill_fingerprint_sha256"] = canonical_fill_v2_fingerprint(value)
    return value


def reject(value: object) -> None:
    with pytest.raises(M07FillValidationError, match="MALFORMED_FILL"):
        validate_structural_fill(value)


def test_code_and_canonical_v2_field_and_fingerprint_parity() -> None:
    assert V2["version"] == "v2"
    assert V2["status"] == "IMPLEMENTED_CANDIDATE_STRUCTURAL_ONLY"
    assert V2["field_count"] == len(V2["fact_fields"]) == len(FILL_V2_FIELDS) == 20
    assert V2["fact_fields"] == [*FILL_V2_FINGERPRINT_FIELDS, "fill_fingerprint_sha256"]
    assert set(V2["fact_fields"]) == FILL_V2_FIELDS
    fingerprint = V2["fingerprint"]
    assert fingerprint["input_fields"] == list(FILL_V2_FINGERPRINT_FIELDS)
    assert fingerprint["domain"] == FILL_V2_FINGERPRINT_DOMAIN
    assert fingerprint["algorithm"] == "SHA-256"
    assert fingerprint["preimage"] == "UTF-8(domain || 0x00 || NFC(canonical JSON projection))"
    assert fingerprint["excluded_fields"] == ["fill_fingerprint_sha256"]


def test_historical_v1_contract_is_exactly_preserved_and_identifiable() -> None:
    legacy = M07["fill_contract"]
    v1 = VERSIONS["v1"]
    assert v1["canonical_contract_pointer"] == "/fill_contract"
    assert v1["field_count"] == len(legacy["fact_fields"]) == len(FILL_V1_FIELDS) == 19
    assert legacy["fact_fields"] == [*FILL_V1_FINGERPRINT_FIELDS, "fill_fingerprint_sha256"]
    assert v1["fingerprint_input_count"] == len(legacy["fingerprint"]["input_fields"]) == 18
    assert v1["fingerprint_domain"] == "LEGACY_UNDOMAINED_CANONICAL_JSON_SHA256"
    assert "asset_namespace must equal Fill.exchange_id" in legacy["fee_semantics"]["CHARGE"]


def test_canonical_ascat_grammar_has_one_upstream_prefix_and_matches_runtime() -> None:
    schema = V2["field_schemas"]["accepted_source_catalog_snapshot_id"]
    upstream = M05["accepted_source_catalog_snapshot_contract"]["id_prefix"]
    assert schema == {
        "type": "exact_str",
        "semantic_type": "AcceptedSourceCatalogSnapshotId",
        "prefix": upstream + "_",
        "suffix": "non_empty",
        "nullable": False,
        "fingerprint_input": True,
        "syntax_grants_trust": False,
    }
    assert validate_structural_fill(v2_fill(accepted_source_catalog_snapshot_id=upstream + "_fake"))
    for malformed in (upstream, upstream + "_", "wcat_1", 1):
        reject(v2_fill(accepted_source_catalog_snapshot_id=malformed))


def test_canonical_v2_fee_is_shape_only_and_runtime_does_not_compare_namespace() -> None:
    fee = V2["fee_structural_policy"]
    assert fee["namespace_comparison"] == "NONE"
    assert fee["historical_instrument_resolution"] is False
    raw = v2_fill()
    assert raw["exchange_id"] != raw["fee_asset_reference"]["asset_namespace"]
    assert validate_structural_fill(raw)


def test_exact_set_dispatch_matches_canonical_version_selection() -> None:
    assert VERSIONS["version_selection"] == {
        "discriminator": "EXACT_CLOSED_FIELD_SET",
        "rules": [
            {"exact_field_set": "v1", "result": "VALIDATE_WITH_V1_STRUCTURAL_RULES"},
            {"exact_field_set": "v2", "result": "VALIDATE_WITH_V2_STRUCTURAL_RULES"},
            {"exact_field_set": "anything_else", "result": "MALFORMED_FILL"},
        ],
        "heuristic_detection": False,
    }
    v2 = v2_fill()
    assert validate_structural_fill(v2)
    v1 = deepcopy(v2)
    del v1["accepted_source_catalog_snapshot_id"]
    v1["exchange_id"] = "binance"
    v1["fill_fingerprint_sha256"] = canonical_fill_fingerprint(v1)
    assert validate_structural_fill(v1)
    hybrid = dict(v1, accepted_source_catalog_snapshot_id="ascat_1")
    reject(hybrid)
    downgraded = deepcopy(v2)
    del downgraded["accepted_source_catalog_snapshot_id"]
    reject(downgraded)
    extra = dict(v2, extra=True)
    reject(extra)


def test_status_points_to_the_canonical_v2_contract() -> None:
    status = json.loads((DOCS / "m07_structural_full_fill_v2_status.json").read_text())
    assert (
        status["canonical_contract_file"] == "commands_events_order_lifecycle_and_idempotency.json"
    )
    assert status["canonical_v1_pointer"] == "/fill_contract"
    assert status["canonical_v2_pointer"] == "/fill_contract_versions/v2"
    assert V2["trusted_full_fill_authority"] == "NOT_AVAILABLE"
    assert V2["semantic_fill_admission"] == "NOT_AVAILABLE"
