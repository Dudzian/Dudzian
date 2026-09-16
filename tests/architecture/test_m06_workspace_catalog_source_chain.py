from copy import deepcopy
from inspect import signature

from bot_core.instruments.catalog_projection_oracle import (
    PROJECTION_FINGERPRINT_FIELDS,
    SOURCE_FINGERPRINT_FIELDS,
    _fingerprint,
    validate_canonical_catalog_context_graph,
    validate_instrument_history_map,
    validate_accepted_source_catalog_snapshot,
    validate_activate_trading_universe,
    execute_instrument_operation,
    validate_trading_universe_source_chain,
    validate_trading_universe_source_graph,
    validate_universe_source_membership,
    validate_workspace_catalog_projection,
)


def _projection_fingerprint(projection):
    projection["content_fingerprint"] = _fingerprint(
        "cryptohunter.m0.5.workspace_catalog_projection.v1",
        PROJECTION_FINGERPRINT_FIELDS,
        projection,
    )


def canonical_graph():
    source = {
        "accepted_source_catalog_snapshot_id": "ascat_1",
        "source_exchange_id": "binance",
        "market_type": "SPOT",
        "source_adapter_family_id": "binance_public_catalog",
        "source_adapter_implementation_id": "impl_ccxt_binance",
        "source_adapter_release_id": "release_2026_09_15",
        "source_adapter_version": "4.5.1",
        "upstream_snapshot_or_retrieval_id": "exchangeInfo:42",
        "observed_at_utc": "2026-09-15T00:00:00Z",
        "effective_at_utc": "2026-09-15T00:00:00Z",
        "stale_after_utc": "2026-09-15T01:00:00Z",
        "previous_snapshot_id": None,
        "completeness_status": "COMPLETE",
        "completeness_evidence": {"method": "TRUSTED_TEST_ORACLE", "page_count": 1},
        "acceptance_status": "VALID",
        "member_source_product_metadata_versions": [
            {
                "source_exchange_id": "binance",
                "market_type": "SPOT",
                "venue_symbol": "BTCUSDT",
                "source_metadata_version_id": "meta_42",
            }
        ],
        "content_fingerprint": "",
    }
    source["content_fingerprint"] = _fingerprint(
        "cryptohunter.m0.5.accepted_source_catalog_snapshot.v2",
        SOURCE_FINGERPRINT_FIELDS,
        source,
    )
    instrument = {
        "instrument_id": "instr_1",
        "workspace_id": "ws_1",
        "accepted_source_catalog_snapshot_id": "ascat_1",
        "source_exchange_id": "binance",
        "market_type": "SPOT",
        "venue_symbol": "BTCUSDT",
        "instrument_type": "SPOT_PAIR",
        "display_symbol": "BTC/USDT",
        "base_asset_reference": {
            "asset_namespace": "binance",
            "venue_asset_code": "BTC",
            "canonical_display_code": "BTC",
            "mapping_status": "EXACT",
        },
        "quote_asset_reference": {
            "asset_namespace": "binance",
            "venue_asset_code": "USDT",
            "canonical_display_code": "USDT",
            "mapping_status": "EXACT",
        },
        "settlement_asset_reference": None,
        "trading_status": "TRADING",
        "price_tick": "0.01",
        "quantity_step": "0.0001",
        "min_quantity": "0.0001",
        "max_quantity": "100",
        "min_notional": "5",
        "max_notional": None,
        "contract_size": None,
        "contract_value_currency": None,
        "derivative_settlement_type": None,
        "expiry_at_utc": None,
        "strike_price": None,
        "option_side": None,
        "metadata_version": 7,
        "observed_at_utc": "2026-09-15T00:00:00Z",
        "effective_at_utc": "2026-09-15T00:00:00Z",
        "stale_after_utc": "2026-09-15T01:00:00Z",
        "source_adapter_family_id": "binance_public_catalog",
    }
    projection = {
        "workspace_catalog_projection_id": "wcat_1",
        "workspace_id": "ws_1",
        "accepted_source_catalog_snapshot_id": "ascat_1",
        "instrument_ids": ["instr_1"],
        "member_bindings": [
            {
                "instrument_id": "instr_1",
                "source_exchange_id": "binance",
                "market_type": "SPOT",
                "venue_symbol": "BTCUSDT",
                "instrument_metadata_version": 7,
                "source_metadata_version_id": "meta_42",
            }
        ],
        "created_at_utc": "2026-09-15T00:01:00Z",
        "content_fingerprint": "",
    }
    projection["content_fingerprint"] = _fingerprint(
        "cryptohunter.m0.5.workspace_catalog_projection.v1",
        PROJECTION_FINGERPRINT_FIELDS,
        projection,
    )
    universe = {"source_catalog_snapshot_ids": ["wcat_1"], "instrument_ids": ["instr_1"]}
    account = {
        "workspace_id": "ws_1",
        "environment": "PAPER",
        "market_type": "SPOT",
        "exchange_id": "paper_simulated_venue",
    }
    return source, projection, instrument, universe, account


def validate_graph(source, projection, instrument, universe, account):
    return validate_trading_universe_source_chain(
        universe,
        account,
        workspace_catalog_projections_by_id={"wcat_1": projection},
        accepted_source_catalog_snapshots_by_id={"ascat_1": source},
        instruments_by_id={"instr_1": instrument},
        paper_source_product_permissions=frozenset({("binance", "SPOT")}),
        now_utc="2026-09-15T00:30:00Z",
    )


def test_distinct_namespaces_resolve_the_executable_canonical_chain():
    source, projection, instrument, universe, account = canonical_graph()
    assert "ascat_1" != "wcat_1"
    assert validate_accepted_source_catalog_snapshot(source)
    assert validate_workspace_catalog_projection(projection, source, {"instr_1": instrument})
    assert validate_graph(source, projection, instrument, universe, account)


def test_source_snapshot_id_cannot_stand_in_for_workspace_projection_id():
    source, projection, instrument, universe, account = canonical_graph()
    universe["source_catalog_snapshot_ids"] = ["ascat_1"]
    assert not validate_graph(source, projection, instrument, universe, account)


def test_workspace_projection_id_cannot_stand_in_for_source_snapshot_id():
    source, projection, instrument, universe, account = canonical_graph()
    instrument["accepted_source_catalog_snapshot_id"] = "wcat_1"
    assert not validate_graph(source, projection, instrument, universe, account)


def test_projection_fails_closed_for_workspace_product_and_metadata_mutations():
    for path, value in (
        (("instrument", "workspace_id"), "ws_B"),
        (("instrument", "source_exchange_id"), "kraken"),
        (("binding", "venue_symbol"), "ETHUSDT"),
        (("binding", "source_metadata_version_id"), "meta_41"),
    ):
        source, projection, instrument, universe, account = canonical_graph()
        target = instrument if path[0] == "instrument" else projection["member_bindings"][0]
        target[path[1]] = value
        if path[0] == "binding":
            projection["content_fingerprint"] = _fingerprint(
                "cryptohunter.m0.5.workspace_catalog_projection.v1",
                PROJECTION_FINGERPRINT_FIELDS,
                projection,
            )
        assert not validate_graph(source, projection, instrument, universe, account)


def test_legacy_catalog_cannot_activate_and_paper_policy_is_explicit():
    source, projection, instrument, universe, account = canonical_graph()
    legacy_catalogs_by_id = {
        "wcat_1": {
            "catalog_snapshot_id": "wcat_1",
            "status": "VALID",
            "exchange_id": "binance",
            "environment": "PAPER",
            "instrument_ids": ["instr_1"],
        }
    }
    # The canonical oracle has no catalogs_by_id argument: this record grants nothing.
    assert legacy_catalogs_by_id
    assert not validate_trading_universe_source_chain(
        universe,
        account,
        workspace_catalog_projections_by_id={},
        accepted_source_catalog_snapshots_by_id={"ascat_1": source},
        instruments_by_id={"instr_1": instrument},
        paper_source_product_permissions=frozenset({("binance", "SPOT")}),
        now_utc="2026-09-15T00:30:00Z",
    )
    assert not validate_trading_universe_source_chain(
        universe,
        account,
        workspace_catalog_projections_by_id={"wcat_1": projection},
        accepted_source_catalog_snapshots_by_id={"ascat_1": source},
        instruments_by_id={"instr_1": instrument},
        paper_source_product_permissions=None,
        now_utc="2026-09-15T00:30:00Z",
    )
    assert validate_graph(source, projection, instrument, universe, account)


def test_historical_resolution_never_falls_back_to_current_instrument():
    source, projection, instrument, universe, account = canonical_graph()
    wrong_history = deepcopy(instrument)
    wrong_history["metadata_version"] = 6
    assert not validate_trading_universe_source_chain(
        universe,
        account,
        workspace_catalog_projections_by_id={"wcat_1": projection},
        accepted_source_catalog_snapshots_by_id={"ascat_1": source},
        instruments_by_id={"instr_1": instrument},
        instrument_history_by_id={"instr_1": [wrong_history]},
        historical=True,
        paper_source_product_permissions=frozenset({("binance", "SPOT")}),
        now_utc="2026-09-15T00:30:00Z",
    )


def rehash_source(source):
    source["content_fingerprint"] = _fingerprint(
        "cryptohunter.m0.5.accepted_source_catalog_snapshot.v2",
        SOURCE_FINGERPRINT_FIELDS,
        source,
    )


def test_partial_rejected_and_stale_are_structural_but_not_activation_eligible():
    for field, value, now in (
        ("completeness_status", "PARTIAL", "2026-09-15T00:30:00Z"),
        ("acceptance_status", "REJECTED", "2026-09-15T00:30:00Z"),
        (None, None, "2026-09-15T01:00:00Z"),
    ):
        source, projection, instrument, universe, account = canonical_graph()
        if field:
            source[field] = value
            rehash_source(source)
        assert validate_accepted_source_catalog_snapshot(source)
        assert not validate_trading_universe_source_chain(
            universe,
            account,
            workspace_catalog_projections_by_id={"wcat_1": projection},
            accepted_source_catalog_snapshots_by_id={"ascat_1": source},
            instruments_by_id={"instr_1": instrument},
            now_utc=now,
            paper_source_product_permissions=frozenset({("binance", "SPOT")}),
        )


def test_malformed_nested_bindings_and_completeness_evidence_are_total():
    malformed = (
        ("instrument_id", []),
        ("instrument_id", {}),
        ("source_exchange_id", []),
        ("market_type", {}),
        ("venue_symbol", []),
        ("source_metadata_version_id", []),
        ("instrument_metadata_version", []),
        ("instrument_metadata_version", True),
    )
    for field, value in malformed:
        source, projection, instrument, _, _ = canonical_graph()
        projection["member_bindings"][0][field] = value
        assert not validate_workspace_catalog_projection(
            projection, source, {"instr_1": instrument}
        )
    source, projection, instrument, _, _ = canonical_graph()
    projection["member_bindings"] = [{"instrument_id": []}]
    assert not validate_workspace_catalog_projection(projection, source, {"instr_1": instrument})
    for evidence in ({"bad": {1}}, {"bad": (1,)}, {"bad": object()}, {"bad": float("nan")}):
        source, _, _, _, _ = canonical_graph()
        source["completeness_evidence"] = evidence
        assert not validate_accepted_source_catalog_snapshot(source)


def test_historical_resolution_exactly_selects_requested_ordered_version():
    source, projection, instrument, universe, account = canonical_graph()
    instrument["metadata_version"] = 8
    history = []
    for version in (5, 6, 7):
        record = deepcopy(instrument)
        record["metadata_version"] = version
        history.append(record)
    projection["member_bindings"][0]["instrument_metadata_version"] = 6
    projection["content_fingerprint"] = _fingerprint(
        "cryptohunter.m0.5.workspace_catalog_projection.v1",
        PROJECTION_FINGERPRINT_FIELDS,
        projection,
    )
    kwargs = dict(
        workspace_catalog_projections_by_id={"wcat_1": projection},
        accepted_source_catalog_snapshots_by_id={"ascat_1": source},
        instruments_by_id={"instr_1": instrument},
        historical=True,
        now_utc="2026-09-15T00:30:00Z",
        paper_source_product_permissions=frozenset({("binance", "SPOT")}),
    )
    assert validate_trading_universe_source_chain(
        universe, account, instrument_history_by_id={"instr_1": history}, **kwargs
    )
    assert not validate_trading_universe_source_chain(
        universe,
        account,
        instrument_history_by_id={"instr_1": [history[0], history[2]]},
        **kwargs,
    )


def test_map_key_identity_and_unrelated_entries_fail_closed():
    source, projection, instrument, universe, account = canonical_graph()
    assert not validate_trading_universe_source_chain(
        universe,
        account,
        workspace_catalog_projections_by_id={"wcat_A": projection},
        accepted_source_catalog_snapshots_by_id={"ascat_1": source},
        instruments_by_id={"instr_1": instrument},
        now_utc="2026-09-15T00:30:00Z",
        paper_source_product_permissions=frozenset({("binance", "SPOT")}),
    )


def test_source_predecessor_lineage_resolves_scope_chronology_and_cycles():
    source, projection, instrument, universe, account = canonical_graph()
    predecessor = deepcopy(source)
    predecessor["accepted_source_catalog_snapshot_id"] = "ascat_0"
    predecessor["observed_at_utc"] = "2026-09-14T23:00:00Z"
    predecessor["effective_at_utc"] = "2026-09-14T23:00:00Z"
    predecessor["stale_after_utc"] = "2026-09-15T00:30:00Z"
    rehash_source(predecessor)
    source["previous_snapshot_id"] = "ascat_0"
    rehash_source(source)
    kwargs = dict(
        workspace_catalog_projections_by_id={"wcat_1": projection},
        instruments_by_id={"instr_1": instrument},
        now_utc="2026-09-15T00:30:00Z",
        paper_source_product_permissions=frozenset({("binance", "SPOT")}),
    )
    assert validate_trading_universe_source_chain(
        universe,
        account,
        accepted_source_catalog_snapshots_by_id={"ascat_0": predecessor, "ascat_1": source},
        **kwargs,
    )
    predecessor["previous_snapshot_id"] = "ascat_1"
    rehash_source(predecessor)
    assert not validate_trading_universe_source_chain(
        universe,
        account,
        accepted_source_catalog_snapshots_by_id={"ascat_0": predecessor, "ascat_1": source},
        **kwargs,
    )
    assert not validate_trading_universe_source_chain(
        universe,
        account,
        workspace_catalog_projections_by_id={"wcat_1": projection},
        accepted_source_catalog_snapshots_by_id={"ascat_A": source},
        instruments_by_id={"instr_1": instrument},
        now_utc="2026-09-15T00:30:00Z",
        paper_source_product_permissions=frozenset({("binance", "SPOT")}),
    )
    unrelated = deepcopy(projection)
    unrelated["workspace_catalog_projection_id"] = "wcat_B"
    assert not validate_trading_universe_source_chain(
        universe,
        account,
        workspace_catalog_projections_by_id={"wcat_1": projection, "wcat_BAD": unrelated},
        accepted_source_catalog_snapshots_by_id={"ascat_1": source},
        instruments_by_id={"instr_1": instrument},
        now_utc="2026-09-15T00:30:00Z",
        paper_source_product_permissions=frozenset({("binance", "SPOT")}),
    )


def test_paper_permission_is_exact_source_and_market_tuple():
    source, projection, instrument, universe, account = canonical_graph()
    for permission in ({("kraken", "SPOT")}, {("binance", "PERPETUAL")}):
        assert not validate_trading_universe_source_chain(
            universe,
            account,
            workspace_catalog_projections_by_id={"wcat_1": projection},
            accepted_source_catalog_snapshots_by_id={"ascat_1": source},
            instruments_by_id={"instr_1": instrument},
            now_utc="2026-09-15T00:30:00Z",
            paper_source_product_permissions=frozenset(permission),
        )


def test_shared_direct_operation_and_context_paths_have_decision_parity():
    source, projection, instrument, universe, account = canonical_graph()
    context = {
        "workspace_catalog_projections_by_id": {"wcat_1": projection},
        "accepted_source_catalog_snapshots_by_id": {"ascat_1": source},
        "instruments_by_id": {"instr_1": instrument},
        "instrument_history_by_id": {},
        "paper_source_product_permissions": frozenset({("binance", "SPOT")}),
    }
    kwargs = dict(
        workspace_catalog_projections_by_id=context["workspace_catalog_projections_by_id"],
        accepted_source_catalog_snapshots_by_id=context["accepted_source_catalog_snapshots_by_id"],
        instruments_by_id=context["instruments_by_id"],
        now_utc="2026-09-15T00:30:00Z",
        paper_source_product_permissions=context["paper_source_product_permissions"],
    )
    assert validate_trading_universe_source_chain(universe, account, **kwargs)
    assert validate_universe_source_membership(universe, account, **kwargs)
    assert validate_activate_trading_universe(
        universe, account, context, now_utc="2026-09-15T00:30:00Z"
    )
    assert execute_instrument_operation(
        "ACTIVATE_TRADING_UNIVERSE",
        universe,
        account,
        context,
        now_utc="2026-09-15T00:30:00Z",
    )


def test_unrelated_projection_requires_full_referential_closure():
    for mutation in ("snapshot", "instrument", "product", "version", "workspace"):
        source, projection, instrument, universe, account = canonical_graph()
        unrelated = deepcopy(projection)
        unrelated["workspace_catalog_projection_id"] = "wcat_2"
        projections = {"wcat_1": projection, "wcat_2": unrelated}
        instruments = {"instr_1": instrument}
        if mutation == "snapshot":
            unrelated["accepted_source_catalog_snapshot_id"] = "ascat_missing"
        elif mutation == "instrument":
            unrelated["instrument_ids"] = ["instr_2"]
            unrelated["member_bindings"][0]["instrument_id"] = "instr_2"
        elif mutation == "product":
            unrelated["member_bindings"][0]["venue_symbol"] = "ETHUSDT"
        elif mutation == "version":
            unrelated["member_bindings"][0]["instrument_metadata_version"] = 6
        else:
            unrelated["workspace_id"] = "ws_2"
        unrelated["content_fingerprint"] = _fingerprint(
            "cryptohunter.m0.5.workspace_catalog_projection.v1",
            PROJECTION_FINGERPRINT_FIELDS,
            unrelated,
        )
        assert not validate_trading_universe_source_chain(
            universe,
            account,
            workspace_catalog_projections_by_id=projections,
            accepted_source_catalog_snapshots_by_id={"ascat_1": source},
            instruments_by_id=instruments,
            now_utc="2026-09-15T00:30:00Z",
            paper_source_product_permissions=frozenset({("binance", "SPOT")}),
        )


def test_malformed_unrelated_full_history_fails_every_public_path():
    source, projection, instrument, universe, account = canonical_graph()
    context = {
        "workspace_catalog_projections_by_id": {"wcat_1": projection},
        "accepted_source_catalog_snapshots_by_id": {"ascat_1": source},
        "instruments_by_id": {"instr_1": instrument},
        "instrument_history_by_id": {"ghost": [{"instrument_id": "ghost", "metadata_version": 1}]},
        "paper_source_product_permissions": frozenset({("binance", "SPOT")}),
    }
    kwargs = dict(
        workspace_catalog_projections_by_id=context["workspace_catalog_projections_by_id"],
        accepted_source_catalog_snapshots_by_id=context["accepted_source_catalog_snapshots_by_id"],
        instruments_by_id=context["instruments_by_id"],
        instrument_history_by_id=context["instrument_history_by_id"],
        paper_source_product_permissions=context["paper_source_product_permissions"],
        now_utc="2026-09-15T00:30:00Z",
    )
    assert not validate_trading_universe_source_chain(universe, account, **kwargs)
    assert not validate_universe_source_membership(universe, account, **kwargs)
    assert not execute_instrument_operation(
        "ACTIVATE_TRADING_UNIVERSE",
        universe,
        account,
        context,
        now_utc="2026-09-15T00:30:00Z",
    )


def test_history_metadata_versions_and_paper_permissions_are_total():
    for version in ("1", True, 0, -1):
        source, projection, instrument, universe, account = canonical_graph()
        bad = deepcopy(instrument)
        bad["metadata_version"] = version
        assert not validate_trading_universe_source_chain(
            universe,
            account,
            workspace_catalog_projections_by_id={"wcat_1": projection},
            accepted_source_catalog_snapshots_by_id={"ascat_1": source},
            instruments_by_id={"instr_1": instrument},
            instrument_history_by_id={"ghost": [bad | {"instrument_id": "ghost"}]},
            now_utc="2026-09-15T00:30:00Z",
            paper_source_product_permissions=frozenset({("binance", "SPOT")}),
        )
    malformed_permissions = (
        1,
        [],
        {},
        set(),
        {"binance"},
        frozenset({"binance"}),
        frozenset({("binance",)}),
        frozenset({("binance", "BAD_MARKET")}),
    )
    for permission in malformed_permissions:
        source, projection, instrument, universe, account = canonical_graph()
        assert not validate_trading_universe_source_chain(
            universe,
            account,
            workspace_catalog_projections_by_id={"wcat_1": projection},
            accepted_source_catalog_snapshots_by_id={"ascat_1": source},
            instruments_by_id={"instr_1": instrument},
            now_utc="2026-09-15T00:30:00Z",
            paper_source_product_permissions=permission,
        )


def test_frozen_operation_entrypoint_fail_closes_canonical_graph_mutations():
    for mutation in (
        "dangling_unrelated",
        "malformed_history",
        "malformed_permission",
        "workspace",
        "product",
        "metadata",
        "legacy_only",
    ):
        source, projection, instrument, universe, account = canonical_graph()
        context = {
            "workspace_catalog_projections_by_id": {"wcat_1": projection},
            "accepted_source_catalog_snapshots_by_id": {"ascat_1": source},
            "instruments_by_id": {"instr_1": instrument},
            "instrument_history_by_id": {},
            "paper_source_product_permissions": frozenset({("binance", "SPOT")}),
        }
        if mutation == "dangling_unrelated":
            unrelated = deepcopy(projection)
            unrelated["workspace_catalog_projection_id"] = "wcat_2"
            unrelated["accepted_source_catalog_snapshot_id"] = "ascat_missing"
            unrelated["content_fingerprint"] = _fingerprint(
                "cryptohunter.m0.5.workspace_catalog_projection.v1",
                PROJECTION_FINGERPRINT_FIELDS,
                unrelated,
            )
            context["workspace_catalog_projections_by_id"]["wcat_2"] = unrelated
        elif mutation == "malformed_history":
            context["instrument_history_by_id"] = {
                "ghost": [{"instrument_id": "ghost", "metadata_version": 1}]
            }
        elif mutation == "malformed_permission":
            context["paper_source_product_permissions"] = frozenset({"binance"})
        elif mutation == "workspace":
            instrument["workspace_id"] = "ws_2"
        elif mutation == "product":
            instrument["source_exchange_id"] = "kraken"
        elif mutation == "metadata":
            projection["member_bindings"][0]["source_metadata_version_id"] = "meta_41"
            projection["content_fingerprint"] = _fingerprint(
                "cryptohunter.m0.5.workspace_catalog_projection.v1",
                PROJECTION_FINGERPRINT_FIELDS,
                projection,
            )
        else:
            context["workspace_catalog_projections_by_id"] = {}
            context["catalogs_by_id"] = {"wcat_1": {"status": "VALID"}}
        assert not execute_instrument_operation(
            "ACTIVATE_TRADING_UNIVERSE",
            universe,
            account,
            context,
            now_utc="2026-09-15T00:30:00Z",
        )


def _history(instrument, versions):
    records = []
    for version in versions:
        record = deepcopy(instrument)
        record["metadata_version"] = version
        records.append(record)
    return records


def _global_graph(source, projection, instrument, history):
    return validate_canonical_catalog_context_graph(
        workspace_catalog_projections_by_id={"wcat_1": projection},
        accepted_source_catalog_snapshots_by_id={"ascat_1": source},
        instruments_by_id={"instr_1": instrument},
        instrument_history_by_id=history,
    )


def test_current_instrument_version_must_strictly_follow_history():
    source, projection, instrument, _, _ = canonical_graph()
    instrument["metadata_version"] = 5
    projection["member_bindings"][0]["instrument_metadata_version"] = 5
    _projection_fingerprint(projection)
    assert not _global_graph(source, projection, instrument, {"instr_1": _history(instrument, (6, 7))})

    source, projection, instrument, _, _ = canonical_graph()
    assert not _global_graph(
        source, projection, instrument, {"instr_1": _history(instrument, (5, 6, 7))}
    )

    instrument["metadata_version"] = 8
    projection["member_bindings"][0]["instrument_metadata_version"] = 8
    _projection_fingerprint(projection)
    assert _global_graph(
        source, projection, instrument, {"instr_1": _history(instrument, (5, 6, 7))}
    )


def test_current_instrument_cannot_rewrite_immutable_historical_identity():
    mutations = (
        {"workspace_id": "ws_2"},
        {
            "source_exchange_id": "kraken",
            "base_asset_reference": {
                "asset_namespace": "kraken",
                "venue_asset_code": "BTC",
                "canonical_display_code": "BTC",
                "mapping_status": "EXACT",
            },
            "quote_asset_reference": {
                "asset_namespace": "kraken",
                "venue_asset_code": "USDT",
                "canonical_display_code": "USDT",
                "mapping_status": "EXACT",
            },
        },
        {"market_type": "MARGIN", "instrument_type": "MARGIN_PAIR"},
        {"venue_symbol": "ETHUSDT"},
    )
    for mutation in mutations:
        _, _, instrument, _, _ = canonical_graph()
        history = {"instr_1": _history(instrument, (5, 6, 7))}
        current = deepcopy(instrument)
        current["metadata_version"] = 8
        current.update(mutation)
        assert not validate_instrument_history_map(history, {"instr_1": current})


def test_source_adapter_family_exact_binds_current_and_historical_instruments():
    source, projection, instrument, universe, account = canonical_graph()
    instrument["source_adapter_family_id"] = "wrong_family"
    kwargs = {
        "workspace_catalog_projections_by_id": {"wcat_1": projection},
        "accepted_source_catalog_snapshots_by_id": {"ascat_1": source},
        "instruments_by_id": {"instr_1": instrument},
    }
    assert not validate_canonical_catalog_context_graph(
        **kwargs, instrument_history_by_id={}
    )
    assert not validate_trading_universe_source_graph(universe, account, **kwargs)
    assert not validate_trading_universe_source_chain(
        universe,
        account,
        **kwargs,
        now_utc="2026-09-15T00:30:00Z",
        paper_source_product_permissions=frozenset({("binance", "SPOT")}),
    )

    source, projection, instrument, _, _ = canonical_graph()
    historical = deepcopy(instrument)
    historical["metadata_version"] = 6
    historical["source_adapter_family_id"] = "wrong_family"
    instrument["metadata_version"] = 8
    projection["member_bindings"][0]["instrument_metadata_version"] = 8
    _projection_fingerprint(projection)
    assert not _global_graph(source, projection, instrument, {"instr_1": [historical]})


def test_public_membership_apis_expose_no_trust_or_operability_bypass():
    for validator in (
        validate_trading_universe_source_graph,
        validate_trading_universe_source_chain,
        validate_universe_source_membership,
    ):
        parameters = signature(validator).parameters
        assert "_activation_operability" not in parameters
        assert "activation_operability" not in parameters
        assert "_trusted_graph_prevalidated" not in parameters
        assert "trusted_graph_prevalidated" not in parameters
        assert not any(parameter.kind.name == "VAR_KEYWORD" for parameter in parameters.values())


def test_public_membership_cannot_bypass_global_closure_or_map_preflight():
    source, projection, instrument, universe, account = canonical_graph()
    dangling = deepcopy(projection)
    dangling["workspace_catalog_projection_id"] = "wcat_bad"
    dangling["accepted_source_catalog_snapshot_id"] = "ascat_missing"
    _projection_fingerprint(dangling)
    kwargs = {
        "workspace_catalog_projections_by_id": {"wcat_1": projection, "wcat_bad": dangling},
        "accepted_source_catalog_snapshots_by_id": {"ascat_1": source},
        "instruments_by_id": {"instr_1": instrument},
        "now_utc": "2026-09-15T00:30:00Z",
        "paper_source_product_permissions": frozenset({("binance", "SPOT")}),
    }
    assert not validate_universe_source_membership(universe, account, **kwargs)
    kwargs["workspace_catalog_projections_by_id"] = 1
    assert not validate_universe_source_membership(universe, account, **kwargs)
