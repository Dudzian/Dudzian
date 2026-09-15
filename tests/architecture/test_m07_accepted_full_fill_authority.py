"""S9D-C25-R1A-FIX: honest structural boundary and upstream blockers."""
from copy import deepcopy
import hashlib
import json
import unicodedata

import pytest

from bot_core.accounting import AccountingAuthority, CoreAcceptedAccountingFactProjection, InMemoryAccountingCarrier, InMemoryAccountingFactCarrier
from bot_core.execution.m07_fill_validation import FILL_FIELDS, M07FillValidationError, canonical_fill_fingerprint, validate_structural_fill
from bot_core.m09_kill_switch_authority import CoreAcceptedContentAuthority, InMemoryCoreAcceptedContentCarrier
from bot_core.persistence.fingerprints import canonical_json_sha256

U1 = "01890f47-5f2d-7a31-8123-123456789abc"


def asset(code="BTC"):
    return {"venue_asset_code": code, "canonical_display_code": code, "asset_namespace": "binance", "mapping_status": "EXACT"}


def fill(**changes):
    value = {"fill_id": f"fill_{U1}", "order_id": f"ord_{U1}", "environment": "PAPER",
             "workspace_id": f"ws_{U1}", "portfolio_id": f"port_{U1}", "exchange_account_id": f"xacc_{U1}",
             "exchange_id": "binance", "instrument_id": f"instr_{U1}", "instrument_metadata_version": 1,
             "execution_route_id": f"xroute_{U1}", "venue_trade_id": "venue-1", "side": "BUY",
             "executed_quantity": "0.8", "execution_price": "100", "executed_at_utc": "2025-01-01T00:00:00Z",
             "fee_kind": "NONE", "fee_quantity": "0", "fee_asset_reference": None,
             "fill_fingerprint_sha256": ""}
    value.update(changes)
    value["fill_fingerprint_sha256"] = canonical_fill_fingerprint(value)
    return value


def accounting():
    content, _ = CoreAcceptedContentAuthority.compose(InMemoryCoreAcceptedContentCarrier())
    sources, _ = CoreAcceptedAccountingFactProjection.compose(InMemoryAccountingFactCarrier(), content_membership=content)
    return AccountingAuthority.compose(InMemoryAccountingCarrier(), source_authority=sources)


def test_exact_frozen_schema_and_independent_fingerprint_are_structurally_valid_only():
    raw = fill()
    assert set(raw) == FILL_FIELDS
    assert validate_structural_fill(raw) == raw
    forged = deepcopy(raw)
    forged["executed_quantity"] = "0.9"
    with pytest.raises(M07FillValidationError, match="MALFORMED_FILL"):
        validate_structural_fill(forged)


class FingerprintSubclass(str):
    pass


class AlwaysEqualFingerprint:
    def __eq__(self, other):
        return True

    def __ne__(self, other):
        return False


class AlwaysEqualZero:
    def __eq__(self, other):
        return other == "0"

    def __ne__(self, other):
        return other != "0"


@pytest.mark.parametrize("replacement", [
    lambda expected: FingerprintSubclass(expected),
    lambda _expected: AlwaysEqualFingerprint(),
    lambda expected: expected.upper(),
    lambda expected: expected[:63],
    lambda expected: expected + "0",
    lambda expected: "g" + expected[1:],
    lambda expected: ("0" if expected[0] != "0" else "1") + expected[1:],
])
def test_terminal_fingerprint_requires_exact_plain_lowercase_sha256_and_value(replacement):
    raw = fill()
    expected = raw["fill_fingerprint_sha256"]
    assert type(expected) is str and len(expected) == 64
    raw["fill_fingerprint_sha256"] = replacement(expected)
    with pytest.raises(M07FillValidationError, match="MALFORMED_FILL"):
        validate_structural_fill(raw)


def test_fingerprint_projection_is_exactly_frozen_18_fields_and_excludes_terminal_hash():
    raw = fill()
    projected = {field: raw[field] for field in (
        "fill_id", "order_id", "environment", "workspace_id", "portfolio_id",
        "exchange_account_id", "exchange_id", "instrument_id",
        "instrument_metadata_version", "execution_route_id", "venue_trade_id", "side",
        "executed_quantity", "execution_price", "executed_at_utc", "fee_kind",
        "fee_quantity", "fee_asset_reference",
    )}
    assert len(projected) == 18
    assert canonical_fill_fingerprint(raw) == canonical_json_sha256(projected)
    raw["fill_fingerprint_sha256"] = "f" * 64
    assert canonical_fill_fingerprint(raw) == canonical_json_sha256(projected)


def frozen_fill_fingerprint(raw):
    projected = {field: raw[field] for field in (
        "fill_id", "order_id", "environment", "workspace_id", "portfolio_id",
        "exchange_account_id", "exchange_id", "instrument_id",
        "instrument_metadata_version", "execution_route_id", "venue_trade_id", "side",
        "executed_quantity", "execution_price", "executed_at_utc", "fee_kind",
        "fee_quantity", "fee_asset_reference",
    )}
    serialized = json.dumps(projected, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(unicodedata.normalize("NFC", serialized).encode("utf-8")).hexdigest()


def test_unicode_fingerprint_uses_exact_frozen_nfc_and_rejects_non_nfc_digest():
    decomposed = fill(venue_trade_id="trade-e\u0301")
    composed = fill(venue_trade_id="trade-é")
    assert canonical_fill_fingerprint(decomposed) == frozen_fill_fingerprint(decomposed)
    assert canonical_fill_fingerprint(decomposed) == canonical_fill_fingerprint(composed)
    assert validate_structural_fill(decomposed) == decomposed

    projected = {field: decomposed[field] for field in decomposed if field != "fill_fingerprint_sha256"}
    without_nfc = hashlib.sha256(json.dumps(
        projected, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")).hexdigest()
    assert without_nfc != decomposed["fill_fingerprint_sha256"]
    decomposed["fill_fingerprint_sha256"] = without_nfc
    with pytest.raises(M07FillValidationError, match="MALFORMED_FILL"):
        validate_structural_fill(decomposed)


def test_unicode_fee_asset_reference_uses_same_frozen_nfc_algorithm():
    fee = asset("é")
    decomposed_fee = {**fee, "venue_asset_code": "e\u0301", "canonical_display_code": "e\u0301"}
    raw = fill(fee_kind="CHARGE", fee_quantity="0.1", fee_asset_reference=decomposed_fee)
    assert canonical_fill_fingerprint(raw) == frozen_fill_fingerprint(raw)
    assert validate_structural_fill(raw) == raw


@pytest.mark.parametrize("fee_quantity", [
    FingerprintSubclass("0"), AlwaysEqualZero(), "0.0", "00", "-0",
])
def test_none_fee_quantity_requires_plain_canonical_decimal_zero(fee_quantity):
    raw = fill()
    raw["fee_quantity"] = fee_quantity
    with pytest.raises(M07FillValidationError, match="MALFORMED_FILL"):
        validate_structural_fill(raw)


def test_plain_zero_none_and_canonical_positive_charge_remain_legal():
    assert validate_structural_fill(fill(fee_quantity="0"))["fee_quantity"] == "0"
    charged = fill(fee_kind="CHARGE", fee_quantity="0.10", fee_asset_reference=asset())
    # 0.10 is noncanonical; the exact canonical equivalent remains accepted.
    with pytest.raises(M07FillValidationError, match="MALFORMED_FILL"):
        validate_structural_fill(charged)
    assert validate_structural_fill(fill(
        fee_kind="CHARGE", fee_quantity="0.1", fee_asset_reference=asset()
    ))["fee_quantity"] == "0.1"


@pytest.mark.parametrize("field", ["environment", "side"])
def test_unhashable_malformed_enums_fail_closed_as_malformed_fill(field):
    raw = fill()
    raw[field] = []
    with pytest.raises(M07FillValidationError, match="MALFORMED_FILL"):
        validate_structural_fill(raw)


def test_fee_semantics_remain_exact_at_structural_boundary():
    charged = fill(fee_kind="CHARGE", fee_quantity="0.01", fee_asset_reference=asset())
    assert validate_structural_fill(charged)["fee_asset_reference"] == asset()
    for malformed in (fill(fee_kind="NONE", fee_quantity="0.01"),
                      fill(fee_kind="CHARGE", fee_quantity="0", fee_asset_reference=asset())):
        with pytest.raises(M07FillValidationError, match="MALFORMED_FILL"):
            validate_structural_fill(malformed)


def test_no_self_enrolling_instrument_or_fill_authority_is_exported():
    import bot_core.execution as execution

    assert not hasattr(execution, "InstrumentHistoryAuthority")
    assert not hasattr(execution, "AcceptedFillAuthority")
    assert not hasattr(execution, "CoreAcceptedFillProjection")


def test_raw_self_hashed_fill_cannot_reach_accounting_or_lifecycle_membership():
    raw = fill()
    assert validate_structural_fill(raw)
    authority, writer = accounting()
    assert not hasattr(authority, "accept_fill")
    assert not hasattr(writer, "accept_fill")


def test_accounting_spot_fifo_rule_never_publishes_incomplete_fill_postings():
    authority, writer = accounting()
    assert authority.journal() == ()
    assert not hasattr(writer, "accept_fill")
    assert all(entry.source_type != "fill" for entry in authority.journal())


def test_overfill_and_fifo_cannot_be_claimed_without_genuine_order_lifecycle_authority():
    first, second = fill(executed_quantity="0.8"), fill(fill_id="fill_01890f47-5f2d-7a31-8123-123456789abd", venue_trade_id="venue-2", executed_quantity="0.8")
    assert validate_structural_fill(first) and validate_structural_fill(second)
    # Structural validation deliberately cannot accept either fact or infer the
    # canonical Order quantity.  No mutation API exists until M0.5 membership
    # and M0.7 lifecycle/order projections are production authorities.
    _, writer = accounting()
    assert not hasattr(writer, "accept_fill")


def test_m05_discovery_records_policy_specific_missing_anchors():
    """The frozen PAPER and TESTNET policies must retain distinct blockers."""
    from pathlib import Path

    root = Path(__file__).parents[2]
    machine = json.loads((root / "docs/architecture/cryptohunter_product_architecture/audit_observability_alerts_and_updater.json").read_text(encoding="utf-8"))
    closure = machine["alert_model"]["executable_authority"]["s9d_c25_m08_reconciliation_authority_disposition"]
    dependency = closure["m07_full_fill_authority_boundary"]["m05_historical_instrument_dependency"]

    assert closure["m07_full_fill_authority"] == (
        "M07_FULL_FILL_AUTHORITY_BLOCKED_M05_HISTORICAL_INSTRUMENT_AND_M07_LIFECYCLE_AUTHORITY"
    )
    assert dependency["status"] == "MISSING_GENUINE_UPSTREAM_CATALOG_SOURCE_MEMBERSHIP_AUTHORITY"
    assert dependency["genuine_production_authority"] == "NOT_FOUND"
    assert dependency["coherent_reseal_disposition"].startswith("BLOCKED_FAIL_CLOSED")
    assert dependency["restore_disposition"].startswith("UNAVAILABLE")
    assert "nearest_missing_prerequisite" not in dependency

    discovery = dependency["production_catalog_source_authority_discovery"]
    assert discovery["source_identity_authenticator"]["status"] == "POLICY_SPECIFIC"
    assert discovery["catalog_snapshot_acceptor"]["status"] == "NOT_FOUND"
    assert discovery["accepted_catalog_carrier_history"]["status"] == "NOT_FOUND"
    assert discovery["scope_binding_evidence"]["status"] == "SCHEMA_ONLY_NOT_AUTHORITY"

    policy = dependency["m05_catalog_source_policy_disposition"]
    assert policy["PAPER"]["nearest_missing_prerequisite"] == (
        "MISSING_CORE_OWNED_CANONICAL_LOCAL_CONTRACT_METADATA_CATALOG_HISTORY_PRODUCER"
    )
    assert policy["PAPER"]["external_authenticated_adapter_membership_required"] == "NO"
    assert policy["TESTNET"]["nearest_missing_prerequisite"] == (
        "CORE_OWNED_DURABLE_AUTHENTICATED_ADAPTER_SOURCE_MEMBERSHIP"
    )
    assert policy["LIVE"]["frozen_exchange_entry_status"] == (
        "NO_FROZEN_ENABLED_LIVE_EXCHANGE_ENTRY"
    )


def test_m05_discovery_is_split_by_exact_frozen_exchange_policy():
    """PAPER local metadata and TESTNET adapter snapshots are distinct trust paths."""
    from pathlib import Path

    root = Path(__file__).parents[2]
    m05 = json.loads((root / "docs/architecture/cryptohunter_product_architecture/exchange_accounts_and_instruments.json").read_text(encoding="utf-8"))
    machine = json.loads((root / "docs/architecture/cryptohunter_product_architecture/audit_observability_alerts_and_updater.json").read_text(encoding="utf-8"))
    registry = m05["exchange_registry_contract"]
    assert registry["closed_build_time_registry"] is True
    assert registry["runtime_config_may_extend_registry"] is False

    enabled = {entry["exchange_id"]: entry for entry in registry["entries"] if entry["status"] == "ENABLED"}
    fields = (
        "exchange_id", "supported_environments", "adapter_family_id",
        "capability_discovery_policy", "instrument_catalog_discovery_policy",
        "account_identity_discovery_policy",
    )
    expected = [{field: entry[field] for field in fields} for entry in enabled.values()]
    dependency = machine["alert_model"]["executable_authority"]["s9d_c25_m08_reconciliation_authority_disposition"]["m07_full_fill_authority_boundary"]["m05_historical_instrument_dependency"]
    assert dependency["enabled_frozen_exchange_policy_matrix"] == expected

    paper = enabled["paper_simulated_venue"]
    assert paper["supported_environments"] == ["PAPER"]
    assert paper["capability_discovery_policy"] == "STATIC_BUILD_TIME"
    assert paper["instrument_catalog_discovery_policy"] == "LOCAL_CONTRACT_METADATA"
    assert paper["account_identity_discovery_policy"] == "LOCAL_SIMULATED_IDENTITY"
    testnet = enabled["generic_testnet_venue"]
    assert testnet["supported_environments"] == ["TESTNET"]
    assert testnet["instrument_catalog_discovery_policy"] == "ADAPTER_SNAPSHOT_REQUIRED"
    assert not any("LIVE" in entry["supported_environments"] for entry in enabled.values())

def test_raw_catalog_and_instrument_have_no_production_enrollment_or_restore_surface():
    """A coherent local reseal cannot attack an API which correctly remains absent."""
    import bot_core.execution as execution

    forbidden = {
        "M05PrevalidatedInstrumentHistory",
        "InstrumentHistoryAuthority",
        "AcceptedInstrumentHistory",
        "restore_instrument_history",
        "accept_instrument",
        "accept_catalog",
    }
    assert forbidden.isdisjoint(vars(execution))


def test_existing_adapter_registries_are_not_authority_for_adapter_snapshot_provenance():
    """Caller-mutable DI cannot prove an ADAPTER_SNAPSHOT_REQUIRED TESTNET source."""
    from bot_core.exchanges.manager import (
        Mode,
        get_native_adapter_info,
        register_native_adapter,
        unregister_native_adapter,
    )
    from pathlib import Path

    class CallerSuppliedFactory:
        pass

    native_id = "caller_supplied_exchange"
    try:
        register_native_adapter(
            exchange_id=native_id,
            mode=Mode.FUTURES,
            factory=CallerSuppliedFactory,
            source="caller/config.yaml",
            dynamic=True,
        )
        registration = get_native_adapter_info(exchange_id=native_id, mode=Mode.FUTURES)
        assert registration is not None
        assert registration.factory is CallerSuppliedFactory
        assert registration.dynamic is True
    finally:
        unregister_native_adapter(
            exchange_id=native_id, mode=Mode.FUTURES, allow_dynamic=True
        )

    bootstrap = (
        Path(__file__).parents[2] / "bot_core/runtime/bootstrap.py"
    ).read_text(encoding="utf-8")
    assert "def register_adapter_factory(" in bootstrap
    assert "_DEFAULT_ADAPTERS[normalized] = factory" in bootstrap
    assert '"register_adapter_factory"' in bootstrap
