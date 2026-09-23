"""S9D-C25-R2 regressions for accepted observed-balance membership."""

from dataclasses import replace
from threading import Barrier, Event, Thread

import pytest

from bot_core.persistence.fingerprints import canonical_json_sha256
from bot_core.reconciliation import (
    AtomicObservedBalanceState,
    CoreAcceptedObservedBalanceFactProjection,
    InMemoryObservedBalanceCarrier,
    ObservedBalanceAuthorityError,
)

U1 = "01890f47-5f2d-7a31-8123-123456789abc"
U2 = "01890f47-5f2d-7a31-8123-123456789abd"


def observation(number=1, **changes):
    raw = {
        "workspace_id": f"ws_{U1}",
        "portfolio_id": f"port_{U1}",
        "environment": "PAPER",
        "exchange_account_id": f"xacc_{U1}",
        "asset_reference": {
            "venue_asset_code": "BTC",
            "canonical_display_code": "BTC",
            "asset_namespace": "binance",
            "mapping_status": "EXACT",
        },
        "observed_quantity": "2",
        "as_of_utc": "2025-01-01T00:00:00Z",
        "source_id": f"snap_{U1 if number == 1 else U2}",
    }
    raw.update(changes)
    raw["source_fingerprint_sha256"] = canonical_json_sha256(raw)
    return raw


def composed(state=None):
    carrier = InMemoryObservedBalanceCarrier(state)
    authority, owner = CoreAcceptedObservedBalanceFactProjection.compose(carrier)
    return carrier, authority, owner


@pytest.mark.parametrize(
    "status,authoritative",
    [
        ("EXACT", True),
        ("EXPLICIT_ALIAS", True),
        ("UNKNOWN", False),
        ("AMBIGUOUS", False),
    ],
)
def test_all_frozen_mapping_states_are_exact_accepted_facts(status, authoritative):
    raw = observation(
        asset_reference={**observation()["asset_reference"], "mapping_status": status}
    )
    carrier, authority, owner = composed()
    accepted = owner.publish(raw, semantics="BALANCE")
    assert accepted.mapping_authoritative is authoritative
    assert authority.resolve(raw["source_id"]) == accepted
    assert dict(accepted.content()) == raw
    assert carrier.read().store_revision == 1


def test_zero_and_unsupported_semantics_are_preserved_without_outcome_derivation():
    raw = observation(observed_quantity="0")
    _, authority, owner = composed()
    accepted = owner.publish(raw, semantics="UNSUPPORTED")
    assert accepted.observed_quantity == "0"
    assert accepted.semantics == "UNSUPPORTED"
    assert not hasattr(authority, "reconcile")


def test_raw_self_hash_and_source_id_do_not_create_membership():
    raw = observation()
    _, authority, _ = composed()
    assert authority.resolve(raw["source_id"]) is None
    assert not hasattr(authority, "accept")
    assert not hasattr(authority, "publish")
    with pytest.raises(ObservedBalanceAuthorityError, match="TRUSTED_CONTEXT_FAILURE"):
        authority.consume_accepted(
            raw["source_id"], raw["source_fingerprint_sha256"], lambda value: value
        )


@pytest.mark.parametrize(
    "change",
    [
        {"workspace_id": f"ws_{U2}"},
        {"portfolio_id": f"port_{U2}"},
        {"environment": "LIVE"},
        {"exchange_account_id": f"xacc_{U2}"},
        {"observed_quantity": "3"},
        {"as_of_utc": "2025-01-02T00:00:00Z"},
        {
            "asset_reference": {
                "venue_asset_code": "XBT",
                "canonical_display_code": "BTC",
                "asset_namespace": "binance",
                "mapping_status": "EXPLICIT_ALIAS",
            }
        },
        {
            "asset_reference": {
                "venue_asset_code": "BTC",
                "canonical_display_code": "BTC",
                "asset_namespace": "binance",
                "mapping_status": "UNKNOWN",
            }
        },
    ],
)
def test_valid_old_fingerprint_cannot_authorize_changed_exact_content(change):
    trusted = observation()
    forged = {**trusted, **change}
    carrier, _, owner = composed()
    with pytest.raises(ObservedBalanceAuthorityError, match="TRUSTED_CONTEXT_FAILURE"):
        owner.publish(forged, semantics="BALANCE")
    assert carrier.read() == AtomicObservedBalanceState()


@pytest.mark.parametrize(
    "mutation",
    [
        lambda raw: raw.update(extra=True),
        lambda raw: raw.pop("as_of_utc"),
        lambda raw: raw.update(as_of_utc="2025-02-30T00:00:00Z"),
        lambda raw: raw.update(observed_quantity="2.0"),
        lambda raw: raw.update(observed_quantity="-1"),
        lambda raw: raw.update(observed_quantity=2.0),
        lambda raw: raw["asset_reference"].update(mapping_status="GUESSED"),
    ],
)
def test_malformed_shapes_and_canonical_values_fail_closed(mutation):
    raw = observation()
    mutation(raw)
    if set(raw) == set(observation()):
        raw["source_fingerprint_sha256"] = canonical_json_sha256(
            {key: value for key, value in raw.items() if key != "source_fingerprint_sha256"}
        )
    with pytest.raises(ObservedBalanceAuthorityError, match="TRUSTED_CONTEXT_FAILURE"):
        composed()[2].publish(raw, semantics="BALANCE")


def test_exact_replay_is_idempotent_but_identity_changes_conflict():
    raw = observation()
    carrier, _, owner = composed()
    first = owner.publish(raw, semantics="BALANCE")
    assert owner.publish(raw, semantics="BALANCE") is first
    assert carrier.read().store_revision == 1
    changed = observation(observed_quantity="3")
    with pytest.raises(ObservedBalanceAuthorityError, match="OBSERVED_BALANCE_IDENTITY_CONFLICT"):
        owner.publish(changed, semantics="BALANCE")
    with pytest.raises(ObservedBalanceAuthorityError, match="OBSERVED_BALANCE_IDENTITY_CONFLICT"):
        owner.publish(raw, semantics="UNSUPPORTED")
    assert carrier.read().store_revision == 1


def test_history_shared_visibility_restart_and_exact_fingerprint_resolution():
    carrier, first_view, owner = composed()
    first, second = observation(), observation(2, observed_quantity="7")
    owner.publish(first, semantics="BALANCE")
    owner.publish(second, semantics="UNSUPPORTED")
    second_view = CoreAcceptedObservedBalanceFactProjection(carrier)
    assert first_view.resolve(first["source_id"]) is not None
    assert second_view.resolve(second["source_id"]).semantics == "UNSUPPORTED"
    with pytest.raises(ObservedBalanceAuthorityError, match="TRUSTED_CONTEXT_FAILURE"):
        second_view.resolve(first["source_id"], expected_fingerprint="0" * 64)
    restored = CoreAcceptedObservedBalanceFactProjection(
        InMemoryObservedBalanceCarrier(carrier.read())
    )
    assert restored.resolve(first["source_id"]).content() == first
    assert restored.resolve(second["source_id"]).content() == second


def test_carrier_failure_is_atomic_and_restore_rejects_coherently_resealed_corruption():
    raw = observation()
    carrier, _, owner = composed()
    carrier.fail_next = True
    with pytest.raises(OSError, match="INJECTED_CARRIER_FAILURE"):
        owner.publish(raw, semantics="BALANCE")
    assert carrier.read() == AtomicObservedBalanceState()
    accepted = owner.publish(raw, semantics="BALANCE")
    corrupt = replace(accepted, observed_quantity="9")
    with pytest.raises(ObservedBalanceAuthorityError, match="CORRUPT_OBSERVED_BALANCE_AUTHORITY"):
        CoreAcceptedObservedBalanceFactProjection(
            InMemoryObservedBalanceCarrier(AtomicObservedBalanceState(1, (corrupt,)))
        )


@pytest.mark.parametrize(
    "accepted_semantics,forged_semantics",
    [
        ("BALANCE", "UNSUPPORTED"),
        ("UNSUPPORTED", "BALANCE"),
    ],
)
def test_restore_rejects_persisted_semantics_rewrite(accepted_semantics, forged_semantics):
    raw = observation()
    carrier, _, owner = composed()
    accepted = owner.publish(raw, semantics=accepted_semantics)
    assert accepted.source_fingerprint_sha256 == raw["source_fingerprint_sha256"]
    forged = replace(accepted, semantics=forged_semantics)
    forged_state = AtomicObservedBalanceState(carrier.read().store_revision, (forged,))
    with pytest.raises(ObservedBalanceAuthorityError, match="CORRUPT_OBSERVED_BALANCE_AUTHORITY"):
        CoreAcceptedObservedBalanceFactProjection(InMemoryObservedBalanceCarrier(forged_state))


def test_opaque_non_snap_source_identity_survives_publish_consume_and_restart():
    raw = observation(source_id="venue-balance-primary")
    carrier, authority, owner = composed()
    accepted = owner.publish(raw, semantics="BALANCE")
    assert authority.resolve("venue-balance-primary") == accepted
    assert (
        authority.consume_accepted(
            "venue-balance-primary",
            raw["source_fingerprint_sha256"],
            lambda item: item,
        )
        == accepted
    )
    restored = CoreAcceptedObservedBalanceFactProjection(
        InMemoryObservedBalanceCarrier(carrier.read()),
    )
    assert restored.resolve("venue-balance-primary") == accepted


def test_reentrant_consume_holds_fence_and_can_resolve_same_membership():
    raw = observation()
    _, authority, owner = composed()
    owner.publish(raw, semantics="BALANCE")
    result = authority.consume_accepted(
        raw["source_id"],
        raw["source_fingerprint_sha256"],
        lambda accepted: authority.resolve(accepted.source_id),
    )
    assert result is not None and result.source_id == raw["source_id"]


def test_shared_carrier_serializes_cross_runtime_publication_and_keeps_history():
    carrier, _, _ = composed()
    views = [CoreAcceptedObservedBalanceFactProjection.compose(carrier) for _ in range(2)]
    raws = [observation(), observation(2)]
    barrier = Barrier(2)
    errors = []

    def publish(index):
        try:
            barrier.wait()
            views[index][1].publish(raws[index], semantics="BALANCE")
        except Exception as exc:  # pragma: no cover - diagnostic capture
            errors.append(exc)

    threads = [Thread(target=publish, args=(index,)) for index in range(2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert errors == []
    assert carrier.read().store_revision == 2
    assert all(views[0][0].resolve(raw["source_id"]) is not None for raw in raws)


def test_consume_fence_blocks_other_runtime_publication_until_callback_exits():
    carrier, runtime_a, owner_a = composed()
    runtime_b, owner_b = CoreAcceptedObservedBalanceFactProjection.compose(carrier)
    first, second = observation(), observation(2)
    owner_a.publish(first, semantics="BALANCE")
    callback_entered, release_callback, publication_finished = Event(), Event(), Event()

    def consume():
        def blocked_callback(accepted):
            callback_entered.set()
            assert release_callback.wait(timeout=5)
            return accepted

        runtime_a.consume_accepted(
            first["source_id"],
            first["source_fingerprint_sha256"],
            blocked_callback,
        )

    def publish():
        owner_b.publish(second, semantics="BALANCE")
        publication_finished.set()

    consumer_thread = Thread(target=consume)
    consumer_thread.start()
    assert callback_entered.wait(timeout=2)
    publisher_thread = Thread(target=publish)
    publisher_thread.start()
    assert not publication_finished.wait(timeout=0.1)
    release_callback.set()
    consumer_thread.join(timeout=2)
    publisher_thread.join(timeout=2)
    assert not consumer_thread.is_alive()
    assert not publisher_thread.is_alive()
    assert publication_finished.is_set()
    assert runtime_b.resolve(second["source_id"]) is not None


def test_otherwise_duplicate_content_under_distinct_source_ids_coexists():
    first = observation()
    second = observation(2)
    carrier, authority, owner = composed()
    owner.publish(first, semantics="BALANCE")
    owner.publish(second, semantics="BALANCE")
    assert carrier.read().store_revision == 2
    assert (
        authority.resolve(first["source_id"]).content()["asset_reference"]
        == authority.resolve(second["source_id"]).content()["asset_reference"]
    )
