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
    assert "policy-specific accepted upstream Catalog/source authority" in dependency["restore_disposition"]
    assert "nearest_missing_prerequisite" not in dependency

    discovery = dependency["production_catalog_source_authority_discovery"]
    assert discovery["source_identity_authenticator"]["status"] == "POLICY_SPECIFIC"
    assert discovery["catalog_snapshot_acceptor"]["status"] == "NOT_FOUND"
    assert discovery["accepted_catalog_carrier_history"]["status"] == "NOT_FOUND"
    assert discovery["scope_binding_evidence"]["status"] == "SCHEMA_ONLY_NOT_AUTHORITY"

    policy = dependency["m05_catalog_source_policy_disposition"]
    assert policy["PAPER"]["nearest_missing_prerequisite"] == (
        "EXPLICIT_M05_SOURCE_VENUE_EXECUTION_ENVIRONMENT_IDENTITY_MIGRATION"
    )
    assert policy["PAPER"]["external_authenticated_adapter_membership_required"].startswith(
        "PRODUCT_TARGET_YES_FROZEN_PAPER_POLICY_NO"
    )
    assert policy["PAPER"]["authority_status"] == (
        "FROZEN_POLICY_CONFLICT_DYNAMIC_EXCHANGE_CATALOG_REQUIRES_EXPLICIT_M05_MIGRATION"
    )
    assert policy["PAPER"]["canonical_metadata_discovery"]["approved_entry_count"] == 0
    discovery = policy["PAPER"]["canonical_metadata_discovery"]
    assert discovery["build_time_immutable_production_metadata"] == (
        "AVAILABLE_RELEASE_OWNED_EMPTY_INSTRUMENT_REGISTRY"
    )
    assert discovery["frozen_fields_without_production_owned_source"] == []
    assert "does not require or permit filling it" in discovery["membership_rule"]
    assert "venue_symbol" in discovery["caller_config_only_fields"]
    assert discovery["test_reference_samples_are_authority"] == "NO"
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


def test_m07_lifecycle_discovery_records_missing_genuine_root_and_exact_answers():
    from pathlib import Path

    root = Path(__file__).parents[2]
    machine = json.loads((root / "docs/architecture/cryptohunter_product_architecture/audit_observability_alerts_and_updater.json").read_text(encoding="utf-8"))
    boundary = machine["alert_model"]["executable_authority"]["s9d_c25_m08_reconciliation_authority_disposition"]["m07_full_fill_authority_boundary"]
    dependency = boundary["m07_order_lifecycle_dependency"]
    assert dependency["status"] == "MISSING_CORE_OWNED_ACCEPTED_ORDER_LIFECYCLE_ROOT"
    assert dependency["genuine_production_authority"] == "NOT_FOUND"
    assert set(dependency["answers"].values()) == {"NO"}
    assert dependency["public_fake_acceptance_surface"].startswith("ABSENT")
    assert dependency["restore_disposition"].startswith("UNAVAILABLE")
    assert dependency["downstream_disposition"].startswith("BLOCKED_FAIL_CLOSED")
    assert boundary["complete_accepted_history"].startswith("BLOCKED")


def test_raw_order_event_status_hash_and_persistence_cannot_self_enroll_lifecycle():
    import bot_core.execution as execution

    forbidden = {
        "OrderLifecycleAuthority", "PrevalidatedOrderLifecycle",
        "accept_order", "accept_event", "restore_order_lifecycle",
    }
    assert forbidden.isdisjoint(vars(execution))

    raw_order = {"order_id": f"ord_{U1}", "quantity": "1", "state": "FILLED"}
    raw_event = {"event_id": f"evt_{U1}", "order_id": raw_order["order_id"], "type": "ORDER_FILLED"}
    persistence_dto = {**raw_order, "event": raw_event, "integrity": canonical_json_sha256(raw_order)}
    assert raw_order["state"] == "FILLED"  # a terminal string remains only caller data
    assert persistence_dto["integrity"] == canonical_json_sha256(raw_order)
    assert not hasattr(execution, "accept_order")
    assert not hasattr(execution, "accept_event")


def test_corrected_paper_catalog_product_decision_is_fail_closed_discovery():
    """Dynamic PAPER availability is specified without pretending it is implemented."""
    import json
    from pathlib import Path

    from bot_core.instruments import paper_canonical_metadata as paper_metadata

    root = Path(__file__).parents[2]
    machine = json.loads((root / "docs/architecture/cryptohunter_product_architecture/audit_observability_alerts_and_updater.json").read_text(encoding="utf-8"))
    dependency = machine["alert_model"]["executable_authority"][
        "s9d_c25_m08_reconciliation_authority_disposition"
    ]["m07_full_fill_authority_boundary"]["m05_historical_instrument_dependency"]
    decision = dependency["paper_catalog_product_decision_discovery"]
    answers = decision["discovery_answers"]

    assert decision["status"] == "NOT_IMPLEMENTED"
    assert answers == {
        "A_static_product_owned_paper_pair_whitelist": "NO",
        "B_full_trusted_exchange_catalog_source": "YES",
        "C_current_frozen_m05_allows_without_change": "NO",
        "D_paper_simulated_venue_can_inherit_real_metadata": "NO",
        "E_real_exchange_id_can_be_preserved_with_paper_execution": "NO",
        "F_genuine_trusted_exchange_catalog_source_authority_exists": "NO",
        "G_caller_controlled_adapter_can_self_enroll": "NO",
        "H_trading_universe_is_manual_selection_boundary": "YES",
        "I_autonomous_selector_only_accepted_catalog_members": "YES",
        "J_ranking_grants_execution_authority": "NO",
        "K_new_listings_without_release": "YES_PRODUCT_TARGET_NO_CURRENT_LEGAL_PATH",
        "L_delisting_preserves_historical_resolution": "YES_REQUIRED_NOT_IMPLEMENTED",
    }
    assert paper_metadata._RELEASE_OWNED_ENTRIES == ()
    assert paper_metadata.canonical_paper_instruments() == ()
    for section in (
        "exchange_catalog_ingestion", "catalog_acceptance",
        "manual_trading_universe", "autonomous_trading_universe",
        "autonomous_candidate_ranking", "execution_authority_separation",
    ):
        assert decision[section]["status"] == "NOT_IMPLEMENTED"
    assert "cannot create Instrument" in decision["manual_trading_universe"]["authority"]
    assert "unknown IDs fail closed" in decision["manual_trading_universe"]["authority"]
    assert "self-enroll" in decision["autonomous_trading_universe"]["authority"]
    assert "never sufficient" in decision["execution_authority_separation"]["invariant"]
    assert "raw results must never be accepted directly" in (
        dependency["m05_catalog_source_policy_disposition"]["PAPER"]
        ["canonical_metadata_discovery"]["runtime_derived_metadata"]
    )
    assert decision["testnet_disposition"].startswith("UNCHANGED")
    assert decision["live_disposition"] == "UNCHANGED_NOT_ENABLED"
    assert decision["global_status_invariants"] == {"C25": "BLOCKED", "S9D": "OPEN"}


def test_explicit_m05_identity_migration_design_closes_exact_decisions():
    """The design separates source identity without claiming production authority."""
    from pathlib import Path

    root = Path(__file__).parents[2]
    machine = json.loads(
        (
            root
            / "docs/architecture/cryptohunter_product_architecture/"
            "audit_observability_alerts_and_updater.json"
        ).read_text(encoding="utf-8")
    )
    dependency = machine["alert_model"]["executable_authority"][
        "s9d_c25_m08_reconciliation_authority_disposition"
    ]["m07_full_fill_authority_boundary"]["m05_historical_instrument_dependency"]
    design = dependency["paper_catalog_product_decision_discovery"][
        "explicit_m05_source_venue_execution_environment_identity_migration_design"
    ]

    assert design["status"] == "DESIGN_ONLY_NOT_IMPLEMENTED"
    assert design["chosen_identity_model"]["source_product_identity"]["tuple"] == [
        "source_exchange_id",
        "market_type",
        "venue_symbol",
    ]
    assert "execution_environment" not in design["chosen_identity_model"][
        "source_product_identity"
    ]["tuple"]
    assert design["chosen_identity_model"]["workspace_instrument_identity"][
        "tuple"
    ] == [
        "workspace_id",
        "source_exchange_id",
        "market_type",
        "venue_symbol",
    ]
    assert design["chosen_identity_model"]["environment_in_identity"] is False
    assert [item["disposition"] for item in design["evaluated_alternatives"]] == [
        "CHOSEN_WITH_WORKSPACE_PROJECTION_CORRECTION",
        "CHOSEN_AS_BINDING_PATTERN_NOT_NEW_ENTITY",
        "REJECTED",
    ]
    assert design["paper_simulated_venue_disposition"][
        "instrument_source_identity_owner"
    ] is False
    assert "execution_environment is forbidden from source provenance" in design[
        "catalog_snapshot_contract"
    ]["scope"]
    assert design["source_authenticator"]["architecture_name"] == (
        "AcceptedSourceProducerMembership"
    )
    assert design["dynamic_listing_delisting_and_completeness"][
        "complete_proof_required"
    ] is True
    assert design["dynamic_listing_delisting_and_completeness"][
        "partial_policy"
    ].startswith("PARTIAL may be retained")
    assert design["stable_identity_map"]["historical_resolution"].endswith(
        "no current-metadata fallback."
    )
    assert design["stable_identity_map"]["mapping"].startswith(
        "workspace_id + immutable source-product identity tuple"
    )
    assert design["stable_identity_map"]["workspace_invariants"] == [
        "one immutable workspace_id owns each instrument_id",
        "same source product in Workspace A and Workspace B maps to instrument_A and "
        "instrument_B where instrument_A != instrument_B",
        "workspace cannot change",
        "source-product continuity preserves ID only within the same Workspace",
        "cross-Workspace lookup or membership fails closed",
    ]
    catalog = design["catalog_snapshot_contract"]["two_level_model"]
    assert catalog["AcceptedSourceCatalogSnapshot"]["scope"] == [
        "source_exchange_id",
        "market_type",
    ]
    assert catalog["WorkspaceCatalogProjection"]["scope"] == [
        "workspace_id",
        "accepted_source_catalog_snapshot_id",
    ]
    assert catalog["WorkspaceCatalogProjection"]["cross_workspace"].startswith(
        "DENIED"
    )
    assert {item["target"] for item in design["downstream_impact_audit"]} == {
        "exchange_id",
        "environment",
        "source_adapter_family_id",
        "instrument_id",
        "catalog_snapshot_id",
        "ExchangeAccount",
        "TradingUniverse",
        "ExecutionRoute",
        "OrderIntent",
        "Order",
        "Fill",
        "M0.8 ledger",
        "M0.9 lease",
        "M0.2 Instrument parent",
        "M0.2 Workspace->Instrument",
        "M0.6 workspace binding",
        "M0.2 Instrument identity_dimensions",
        "M0.6 Instrument exchange/environment equality",
    }
    assert design["compatibility_decision"]["fully_additive_possible"] is False
    assert design["legacy_paper_records"]["automatic_mapping"] is False
    assert design["implementation_disposition"] == {
        "production_migration": "NOT_IMPLEMENTED",
        "forbidden_in_this_iteration": [
            "network ingestion",
            "exchange-specific adapters or venue hardcoding",
            "Catalog acceptance runtime",
            "identity map persistence",
            "GUI",
            "manual universe writer",
            "autonomous selector/ranking",
            "lifecycle authority",
            "Full Fill",
            "reconciliation",
            "LIVE enablement",
        ],
        "blockers": {"C25": "BLOCKED", "S9D": "OPEN"},
    }
    assert design["final_answers"] == {
        "source_venue_separate_from_execution_environment": "YES",
        "paper_simulated_venue_remains_instrument_source_identity": "NO",
        "real_source_exchange_id_preserved_under_paper": "YES",
        "current_instrument_identity_tuple_requires_migration": "YES",
        "migration_fully_additive": "NO",
        "trusted_adapter_registration_equals_source_authority": "NO",
        "dynamic_plugin_may_claim_official_venue_authority": "NO",
        "complete_snapshot_proof_required_before_absence_delisting": "YES",
        "partial_snapshot_may_remove_membership": "NO",
        "stable_instrument_id_requires_durable_identity_map": "YES",
        "historical_metadata_versions_remain_resolvable": "YES",
        "trading_universe_works_without_instrument_copy": "YES",
        "legacy_paper_records_automatically_map_to_real_venue": "NO",
        "production_migration_implemented": "NO",
        "C25": "BLOCKED",
        "S9D": "OPEN",
        "canonical_instrument_has_exactly_one_workspace_parent": "YES",
        "two_workspaces_may_share_one_canonical_instrument_id": "NO",
        "two_workspaces_may_reference_same_trusted_source_product_or_catalog_fact": "YES",
        "source_product_equivalence_implies_canonical_instrument_identity_equivalence": "NO",
        "cross_workspace_instrument_resolution_fails_closed": "YES",
        "m02_workspace_ownership_requires_migration": "NO",
        "m06_same_workspace_invariant_preserved": "YES",
        "paper_reuses_workspace_owned_source_backed_instrument_without_copy": "YES",
        "m02_identity_migration_was_required": "YES",
        "m02_identity_migration_current_status": "MIGRATED_CANONICAL_1.44.0",
        "environment_in_current_canonical_instrument_identity": "NO",
        "source_exchange_id_is_current_canonical_source_identity": "YES",
        "m06_source_execution_migration_was_required": "YES",
        "m06_source_execution_migration_current_status": "MIGRATED_CANONICAL_1.44.0",
    }


def test_workspace_instrument_identity_is_distinct_over_one_shared_source_fact():
    """Design oracle: source reuse cannot collapse tenant-owned entities."""
    from pathlib import Path

    root = Path(__file__).parents[2]
    machine = json.loads(
        (
            root
            / "docs/architecture/cryptohunter_product_architecture/"
            "audit_observability_alerts_and_updater.json"
        ).read_text(encoding="utf-8")
    )
    dependency = machine["alert_model"]["executable_authority"][
        "s9d_c25_m08_reconciliation_authority_disposition"
    ]["m07_full_fill_authority_boundary"]["m05_historical_instrument_dependency"]
    design = dependency["paper_catalog_product_decision_discovery"][
        "explicit_m05_source_venue_execution_environment_identity_migration_design"
    ]

    source_product = ("binance", "SPOT", "BTCUSDT")
    workspace_a_key = ("wrk_A", *source_product)
    workspace_b_key = ("wrk_B", *source_product)
    # Conceptual fixtures assert the frozen mapping's declared cardinality; they
    # are deliberately not a production ID-map implementation.
    conceptual_identity_map = {
        workspace_a_key: "instr_A",
        workspace_b_key: "instr_B",
    }
    assert workspace_a_key[1:] == workspace_b_key[1:] == source_product
    assert conceptual_identity_map[workspace_a_key] != conceptual_identity_map[workspace_b_key]

    compatibility = design["m02_m06_compatibility"]
    assert compatibility["M0.2"]["ownership"] == {
        "status": "UNCHANGED",
        "instrument_parent": "Workspace",
        "workspace_to_instrument": "one_to_many_catalog",
        "ownership_migration_required": False,
        "shared_canonical_instrument_id_across_workspaces": False,
    }
    assert compatibility["M0.2"]["instrument_identity_dimensions"]["status"] == (
        "MIGRATED_CANONICAL_1.44.0"
    )
    assert compatibility["M0.6"]["same_workspace_invariant"] == (
        "Instrument.workspace_id == ExchangeAccount.workspace_id remains required"
    )
    assert compatibility["M0.6"]["foreign_instrument_disposition"] == (
        "TRUSTED_CONTEXT_INVALID"
    )
    assert design["trading_universe_compatibility"][
        "cross_workspace_selection"
    ].startswith("DENIED_FAIL_CLOSED")


def test_m02_frozen_instrument_dimensions_prove_explicit_migration_dependency():
    """The design must cite the real frozen M0.2 conflict, not only M0.12 claims."""
    from pathlib import Path

    root = Path(__file__).parents[2]
    docs = root / "docs/architecture/cryptohunter_product_architecture"
    m02 = json.loads((docs / "canonical_domain_vocabulary.json").read_text())
    machine = json.loads(
        (docs / "audit_observability_alerts_and_updater.json").read_text()
    )
    instrument = next(
        entity for entity in m02["entity_kinds"] if entity["canonical_name"] == "Instrument"
    )
    dependency = machine["alert_model"]["executable_authority"][
        "s9d_c25_m08_reconciliation_authority_disposition"
    ]["m07_full_fill_authority_boundary"]["m05_historical_instrument_dependency"]
    design = dependency["paper_catalog_product_decision_discovery"][
        "explicit_m05_source_venue_execution_environment_identity_migration_design"
    ]
    migration = design["m02_identity_dimensions_migration_dependency"]

    assert instrument["parent"] == migration["exact_frozen_evidence"]["parent"] == (
        "Workspace"
    )
    assert instrument["identity_dimensions"] == migration["exact_frozen_evidence"][
        "identity_dimensions"
    ] == ["source_exchange_id", "market_type", "venue_symbol"]
    assert migration["status"] == "MIGRATED_CANONICAL_1.44.0"
    assert migration["target_identity_dimensions"] == [
        "source_exchange_id",
        "market_type",
        "venue_symbol",
    ]
    assert migration["cutover_dependency"] == (
        "REQUIRED_BEFORE_AUTHORITATIVE_M0.5_CUTOVER"
    )
    assert design["m02_m06_compatibility"]["M0.2"]["ownership"][
        "ownership_migration_required"
    ] is False
    assert design["m02_m06_compatibility"]["M0.6"][
        "instrument_environment_exchange_checks"
    ] == "MIGRATED_CANONICAL_1.44.0_SOURCE_EXECUTION_ROLES_SEPARATED"
