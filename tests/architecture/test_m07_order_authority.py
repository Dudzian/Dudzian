from __future__ import annotations

import hashlib
import hmac
import json
import sqlite3
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

import bot_core.orders.authority as authority_module

from bot_core.orders import (
    AuthorityCorrupt,
    OrderAuthority,
    OrderAuthorityError,
    PRODUCTION_ORDER_AUTHORITY_DOMAIN,
    TEST_ORDER_AUTHORITY_DOMAIN,
    TestOrderAuthority,
    UpstreamAuthorityUnavailable,
    command_fingerprint_sha256,
    event_fingerprint_sha256,
    validate_order_command,
    validate_order_event,
    require_production_order_authority,
)


class _MemoryProductionCustody:
    keys: dict[str, bytes] = {}

    def persist(self, handle: str, key: bytes) -> None:
        self.keys[handle] = key

    def digest(self, handle: str, payload: bytes) -> bytes | None:
        key = self.keys.get(handle)
        return None if key is None else hmac.digest(key, payload, "sha256")


@pytest.fixture(autouse=True)
def _production_custody(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        authority_module, "production_order_authority_custody", _MemoryProductionCustody
    )


U = "018f0f3e-7b5a-7abc-8def-1234567890ab"


def ident(prefix: str, tail: str = U) -> str:
    return f"{prefix}_{tail}"


def command() -> dict[str, object]:
    return {
        "command_id": ident("cmd"),
        "operation_type": "SUBMIT_ORDER",
        "authority_context_id": ident("authctx"),
        "environment": "PAPER",
        "workspace_id": ident("ws"),
        "portfolio_id": ident("port"),
        "exchange_account_id": ident("xacc"),
        "strategy_instance_id": None,
        "source_type": "OPERATOR",
        "instrument_id": ident("instr"),
        "execution_route_id": ident("xroute"),
        "correlation_id": ident("corr"),
        "causation_id": None,
        "idempotency_key": ident("cmd"),
        "order_intent_id": ident("oint"),
        "order_id": ident("ord"),
        "side": "BUY",
        "order_type": "MARKET",
        "quantity": "1",
        "limit_price": None,
        "time_in_force": "GTC",
        "expire_at_utc": None,
    }


def next_event(
    authority: TestOrderAuthority, event_type: str, payload: dict[str, object], n: int
) -> dict[str, object]:
    prior = dict(authority.history(ident("ord"))[0])
    event = {
        **prior,
        "audit_event_id": ident("evt", f"018f0f3e-7b5a-7ab{n}-8def-1234567890ab"),
        "event_type": event_type,
        "aggregate_version": n,
        "safe_payload": payload,
        "event_fingerprint_sha256": "",
    }
    event["event_fingerprint_sha256"] = event_fingerprint_sha256(event)
    return event


def test_fingerprint_parity_with_frozen_executable_oracle() -> None:
    request = command()
    expected = {k: v for k, v in request.items() if k != "correlation_id"}
    import hashlib, unicodedata

    raw = unicodedata.normalize(
        "NFC", json.dumps(expected, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    ).encode()
    assert command_fingerprint_sha256(request) == hashlib.sha256(raw).hexdigest()
    validate_order_command(request)


@pytest.mark.parametrize(
    "mutation",
    [
        {"extra": 1},
        {"quantity": 1.0},
        {"quantity": "1.0"},
        {"quantity": True},
        {"command_id": "cmd_bad"},
    ],
)
def test_command_closed_schema_and_canonical_types(mutation: dict[str, object]) -> None:
    request = command()
    request.update(mutation)
    with pytest.raises(OrderAuthorityError, match="MALFORMED_REQUEST"):
        validate_order_command(request)


def test_production_submit_fails_before_any_fact(tmp_path: Path) -> None:
    path = tmp_path / "prod.sqlite"
    authority = OrderAuthority(path)
    with pytest.raises(UpstreamAuthorityUnavailable):
        authority.submit_order(command())
    assert (
        dict(authority.authority_head)["command_count"]
        == dict(authority.authority_head)["event_count"]
        == 0
    )
    authority.close()
    with pytest.raises(AuthorityCorrupt):
        TestOrderAuthority(path)
    assert PRODUCTION_ORDER_AUTHORITY_DOMAIN != TEST_ORDER_AUTHORITY_DOMAIN


def test_test_admission_replay_conflict_history_restart_and_restore(tmp_path: Path) -> None:
    path = tmp_path / "test.sqlite"
    authority = TestOrderAuthority(path)
    request = command()
    first = dict(authority.submit_order(request))
    assert dict(authority.submit_order(request)) == first
    changed = command()
    changed["quantity"] = "2"
    with pytest.raises(OrderAuthorityError, match="IDEMPOTENCY_CONFLICT"):
        authority.submit_order(changed)
    events = [
        ("ORDER_DISPATCHED", {"client_order_id": "client"}),
        ("ORDER_ACKNOWLEDGED", {"venue_order_id": "venue"}),
        ("ORDER_CANCEL_REQUESTED", {"reason_code": "OPERATOR_REQUEST"}),
        ("ORDER_CANCEL_REJECTED", {"reason_code": "VENUE_DENIED"}),
    ]
    for version, (kind, payload) in enumerate(events, 2):
        assert authority.append_event(next_event(authority, kind, payload, version)) == "ACCEPTED"
    assert dict(authority.resolve_current(ident("ord")))["state"] == "ACKNOWLEDGED"
    assert authority.resolve_event(ident("ord"), 99) is None
    history = authority.history(ident("ord"))
    assert len(history) == 5
    assert (
        authority.append_event(
            {k: (dict(v) if k == "safe_payload" else v) for k, v in history[-1].items()}
        )
        == "REPLAY_SUCCESS"
    )
    head = dict(authority.authority_head)
    authority.close()
    reopened = TestOrderAuthority(path)
    assert reopened.history(ident("ord")) == history and dict(reopened.authority_head) == head


def test_terminal_cannot_reopen_and_concurrent_next_version_linearizes(tmp_path: Path) -> None:
    path = tmp_path / "race.sqlite"
    seed = TestOrderAuthority(path)
    seed.submit_order(command())
    seed.append_event(next_event(seed, "ORDER_DISPATCHED", {"client_order_id": "c"}, 2))
    seed.close()
    a = TestOrderAuthority(path)
    b = TestOrderAuthority(path)
    left = next_event(a, "ORDER_ACKNOWLEDGED", {"venue_order_id": "v"}, 3)
    right = next_event(a, "ORDER_REJECTED", {"reason_code": "NO"}, 3)

    def put(pair: tuple[TestOrderAuthority, dict[str, object]]) -> str:
        try:
            return pair[0].append_event(pair[1])
        except OrderAuthorityError as exc:
            return exc.code

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(put, [(a, left), (b, right)]))
    assert results.count("ACCEPTED") == 1 and len(a.history(ident("ord"))) == 3
    winner = dict(a.resolve_current(ident("ord")))["state"]
    if winner == "REJECTED":
        late = next_event(a, "ORDER_ACKNOWLEDGED", {"venue_order_id": "late"}, 4)
        with pytest.raises(OrderAuthorityError, match="INVALID_LIFECYCLE_TRANSITION"):
            a.append_event(late)


def test_tamper_and_missing_table_fail_closed(tmp_path: Path) -> None:
    path = tmp_path / "tamper.sqlite"
    authority = TestOrderAuthority(path)
    authority.submit_order(command())
    authority.close()
    db = sqlite3.connect(path)
    db.execute("DROP TRIGGER event_no_update")
    db.execute("UPDATE event_journal SET aggregate_version=9")
    db.commit()
    db.close()
    with pytest.raises(AuthorityCorrupt):
        TestOrderAuthority(path)
    path2 = tmp_path / "missing.sqlite"
    authority = OrderAuthority(path2)
    authority.close()
    db = sqlite3.connect(path2)
    db.execute("DROP TABLE event_journal")
    db.commit()
    db.close()
    with pytest.raises(AuthorityCorrupt):
        OrderAuthority(path2)


def test_event_identity_conflict_terminal_and_replace_restore(tmp_path: Path) -> None:
    authority = TestOrderAuthority(tmp_path / "edge.sqlite")
    authority.submit_order(command())
    dispatched = next_event(authority, "ORDER_DISPATCHED", {"client_order_id": "c"}, 2)
    authority.append_event(dispatched)
    acknowledged = next_event(authority, "ORDER_ACKNOWLEDGED", {"venue_order_id": "v"}, 3)
    authority.append_event(acknowledged)
    conflict = dict(acknowledged)
    conflict["safe_payload"] = {"venue_order_id": "other"}
    conflict["event_fingerprint_sha256"] = event_fingerprint_sha256(conflict)
    with pytest.raises(OrderAuthorityError, match="EVENT_IDENTITY_CONFLICT"):
        authority.append_event(conflict)
    authority.append_event(
        next_event(
            authority,
            "ORDER_REPLACE_REQUESTED",
            {"replacement_order_id": ident("ord", "018f0f3e-7b5a-7abc-8def-1234567890ac")},
            4,
        )
    )
    authority.append_event(
        next_event(authority, "ORDER_REPLACE_REJECTED", {"reason_code": "DENIED"}, 5)
    )
    assert dict(authority.resolve_current(ident("ord")))["state"] == "ACKNOWLEDGED"
    authority.append_event(
        next_event(
            authority,
            "ORDER_FILLED",
            {
                "fill_id": ident("fill"),
                "venue_trade_id": "trade",
                "cumulative_executed_quantity": "1",
            },
            6,
        )
    )
    assert dict(authority.resolve_current(ident("ord")))["state"] == "FILLED"
    late = next_event(authority, "ORDER_CANCEL_REQUESTED", {"reason_code": "OPERATOR_REQUEST"}, 7)
    with pytest.raises(OrderAuthorityError, match="INVALID_LIFECYCLE_TRANSITION"):
        authority.append_event(late)


def _acknowledged(path: Path) -> TestOrderAuthority:
    authority = TestOrderAuthority(path)
    authority.submit_order(command())
    authority.append_event(next_event(authority, "ORDER_DISPATCHED", {"client_order_id": "c"}, 2))
    authority.append_event(next_event(authority, "ORDER_ACKNOWLEDGED", {"venue_order_id": "v"}, 3))
    return authority


def _fill_event(
    authority: TestOrderAuthority,
    kind: str,
    cumulative: str,
    version: int,
    *,
    fill_tail: str = U,
    trade: str = "T1",
) -> dict[str, object]:
    return next_event(
        authority,
        kind,
        {
            "fill_id": ident("fill", fill_tail),
            "venue_trade_id": trade,
            "cumulative_executed_quantity": cumulative,
        },
        version,
    )


def test_duplicate_fill_new_event_identity_is_no_effect_and_conflict_denied(tmp_path: Path) -> None:
    authority = _acknowledged(tmp_path / "fill-dedupe.sqlite")
    first = _fill_event(authority, "ORDER_PARTIALLY_FILLED", "0.5", 4)
    assert authority.append_event(first) == "ACCEPTED"
    duplicate = _fill_event(authority, "ORDER_PARTIALLY_FILLED", "0.5", 5)
    duplicate["audit_event_id"] = ident("evt", "018f0f3e-7b5a-7ab5-8def-1234567890ac")
    duplicate["event_fingerprint_sha256"] = event_fingerprint_sha256(duplicate)
    assert (
        authority.append_event(duplicate) == "REPLAY_SUCCESS"
        and len(authority.history(ident("ord"))) == 4
    )
    conflict = _fill_event(authority, "ORDER_PARTIALLY_FILLED", "0.6", 5)
    with pytest.raises(OrderAuthorityError, match="FILL_IDENTITY_CONFLICT"):
        authority.append_event(conflict)
    assert len(authority.history(ident("ord"))) == 4


@pytest.mark.parametrize(
    ("kind", "cumulative", "accepted"),
    [
        ("ORDER_PARTIALLY_FILLED", "1", False),
        ("ORDER_PARTIALLY_FILLED", "1.1", False),
        ("ORDER_FILLED", "0.9", False),
        ("ORDER_FILLED", "1", True),
        ("ORDER_FILLED", "1.1", False),
    ],
)
def test_frozen_fill_quantity_progression(
    kind: str, cumulative: str, accepted: bool, tmp_path: Path
) -> None:
    authority = _acknowledged(tmp_path / f"{kind}-{cumulative}.sqlite")
    event = _fill_event(authority, kind, cumulative, 4)
    if accepted:
        assert authority.append_event(event) == "ACCEPTED"
    else:
        with pytest.raises(OrderAuthorityError, match="FILL_PROGRESSION_CONFLICT"):
            authority.append_event(event)
        assert len(authority.history(ident("ord"))) == 3


def test_exact_production_type_boundary_denies_test_subclass(tmp_path: Path) -> None:
    production = OrderAuthority(tmp_path / "prod-boundary.sqlite")
    assert require_production_order_authority(production) is production
    test = TestOrderAuthority(tmp_path / "test-boundary.sqlite")
    assert isinstance(test, OrderAuthority)
    with pytest.raises(OrderAuthorityError, match="TRUSTED_CONTEXT_FAILURE"):
        require_production_order_authority(test)


def test_coherent_public_sha_sql_mint_without_mac_is_denied(tmp_path: Path) -> None:
    path = tmp_path / "mint.sqlite"
    authority = OrderAuthority(path)
    authority.close()
    request = command()
    fingerprint = command_fingerprint_sha256(request)
    outcome = {
        "command_id": request["command_id"],
        "operation_type": "SUBMIT_ORDER",
        "request_fingerprint_sha256": fingerprint,
        "outcome": "ACCEPTED",
        "denial_code": None,
        "order_id": request["order_id"],
        "accepted_audit_event_id": ident("evt"),
        "recorded_at_utc": "2024-01-01T00:00:00Z",
    }
    canonical_request = authority_module._canonical(request)
    canonical_outcome = authority_module._canonical(outcome)
    scope = authority_module._canonical(
        [request[k] for k in authority_module.CONTRACT["idempotency_contract"]["scope"]]
    )
    command_values = [
        1,
        request["command_id"],
        scope,
        canonical_request,
        fingerprint,
        canonical_outcome,
        "0" * 64,
    ]
    command_digest = authority_module._digest(command_values)
    event = TestOrderAuthority._planned_event(
        request, "test-exchange", "2024-01-01T00:00:00Z", ident("evt")
    )
    canonical_event = authority_module._canonical(event)
    event_values = [
        1,
        event["audit_event_id"],
        event["order_id"],
        1,
        canonical_event,
        event["event_fingerprint_sha256"],
        "0" * 64,
    ]
    event_digest = authority_module._digest(event_values)
    db = sqlite3.connect(path)
    db.execute(
        "INSERT INTO command_journal VALUES(?,?,?,?,?,?,?,?,?,?)",
        (*command_values, command_digest, "0" * 64, "0" * 64),
    )
    db.execute(
        "INSERT INTO event_journal VALUES(?,?,?,?,?,?,?,?,?,?)",
        (*event_values, event_digest, "0" * 64, "0" * 64),
    )
    db.execute(
        "UPDATE authority_heads SET command_count=1,command_head=?,command_authenticated_head=?,event_count=1,event_head=?,event_authenticated_head=?",
        (command_digest, "0" * 64, event_digest, "0" * 64),
    )
    db.commit()
    db.close()
    with pytest.raises(AuthorityCorrupt, match="authenticated admission"):
        OrderAuthority(path)


def test_coherent_public_rewrite_without_new_mac_is_denied(tmp_path: Path) -> None:
    path = tmp_path / "rewrite.sqlite"
    authority = TestOrderAuthority(path)
    authority.submit_order(command())
    authority.close()
    db = sqlite3.connect(path)
    db.row_factory = sqlite3.Row
    db.execute("DROP TRIGGER event_no_update")
    row = db.execute("SELECT * FROM event_journal").fetchone()
    event = json.loads(row["canonical_event"])
    event["safe_payload"]["quantity"] = "2"
    event["event_fingerprint_sha256"] = event_fingerprint_sha256(event)
    canonical = authority_module._canonical(event)
    values = [
        1,
        event["audit_event_id"],
        event["order_id"],
        1,
        canonical,
        event["event_fingerprint_sha256"],
        "0" * 64,
    ]
    digest = authority_module._digest(values)
    db.execute(
        "UPDATE event_journal SET canonical_event=?,event_fingerprint=?,record_digest=?",
        (canonical, event["event_fingerprint_sha256"], digest),
    )
    db.execute("UPDATE authority_heads SET event_head=?", (digest,))
    db.commit()
    db.close()
    with pytest.raises(AuthorityCorrupt, match="authenticated admission"):
        TestOrderAuthority(path)


def _rewrite_test_event_chain(path: Path, mutate: object) -> None:
    db = sqlite3.connect(path)
    db.row_factory = sqlite3.Row
    db.execute("DROP TRIGGER event_no_update")
    handle = db.execute("SELECT custody_handle FROM authority_marker").fetchone()[0]
    custody = authority_module.DeterministicTestOrderAuthoritySecretCustody()
    previous = authenticated = "0" * 64
    count = 0
    for row in db.execute("SELECT * FROM event_journal ORDER BY event_sequence").fetchall():
        event = json.loads(row["canonical_event"])
        mutate(event)
        event["event_fingerprint_sha256"] = event_fingerprint_sha256(event)
        canonical = authority_module._canonical(event)
        count += 1
        values = [
            count,
            event["audit_event_id"],
            event["order_id"],
            event["aggregate_version"],
            canonical,
            event["event_fingerprint_sha256"],
            previous,
        ]
        digest = authority_module._digest(values)
        auth_fields = [
            count,
            event["audit_event_id"],
            event["order_id"],
            event["aggregate_version"],
            canonical,
            event["event_fingerprint_sha256"],
            authenticated,
        ]
        payload = (
            authority_module.TEST_ORDER_AUTHENTICITY_PURPOSE.encode("ascii")
            + b"\x00"
            + authority_module.TEST_ORDER_AUTHORITY_DOMAIN.encode()
            + b"\x00EVENT\x00"
            + authority_module._canonical(auth_fields).encode()
        )
        mac = custody.digest(handle, payload).hex()
        db.execute(
            "UPDATE event_journal SET audit_event_id=?,order_id=?,canonical_event=?,event_fingerprint=?,previous_digest=?,record_digest=?,previous_authenticated_commitment=?,admission_mac=? WHERE event_sequence=?",
            (
                event["audit_event_id"],
                event["order_id"],
                canonical,
                event["event_fingerprint_sha256"],
                previous,
                digest,
                authenticated,
                mac,
                count,
            ),
        )
        previous = digest
        authenticated = mac
    db.execute(
        "UPDATE authority_heads SET event_count=?,event_head=?,event_authenticated_head=?",
        (count, previous, authenticated),
    )
    db.commit()
    db.close()


def test_restart_rejects_authenticated_but_impossible_fill_history(tmp_path: Path) -> None:
    path = tmp_path / "impossible.sqlite"
    authority = _acknowledged(path)
    authority.append_event(_fill_event(authority, "ORDER_PARTIALLY_FILLED", "0.5", 4))
    authority.close()

    def mutate(event: dict[str, object]) -> None:
        if event["event_type"] == "ORDER_PARTIALLY_FILLED":
            event["safe_payload"]["cumulative_executed_quantity"] = "1.1"

    _rewrite_test_event_chain(path, mutate)
    with pytest.raises(AuthorityCorrupt):
        TestOrderAuthority(path)


def test_restart_replay_exactly_binds_order_scope(tmp_path: Path) -> None:
    path = tmp_path / "scope.sqlite"
    authority = _acknowledged(path)
    authority.close()

    def mutate(event: dict[str, object]) -> None:
        if event["aggregate_version"] == 3:
            event["workspace_id"] = "ws_018f0f3e-7b5a-7abc-8def-1234567890ac"

    _rewrite_test_event_chain(path, mutate)
    with pytest.raises(AuthorityCorrupt):
        TestOrderAuthority(path)


def test_restart_replay_cross_binds_command_outcome_to_request(tmp_path: Path) -> None:
    path = tmp_path / "outcome.sqlite"
    authority = TestOrderAuthority(path)
    authority.submit_order(command())
    authority.close()
    db = sqlite3.connect(path)
    db.row_factory = sqlite3.Row
    db.execute("DROP TRIGGER command_no_update")
    row = db.execute("SELECT * FROM command_journal").fetchone()
    outcome = json.loads(row["canonical_outcome"])
    outcome["order_id"] = "ord_018f0f3e-7b5a-7abc-8def-1234567890ac"
    canonical = authority_module._canonical(outcome)
    values = [
        1,
        row["command_id"],
        row["scope_key"],
        row["canonical_request"],
        row["command_fingerprint"],
        canonical,
        "0" * 64,
    ]
    digest = authority_module._digest(values)
    handle = db.execute("SELECT custody_handle FROM authority_marker").fetchone()[0]
    auth_fields = [
        1,
        row["command_id"],
        row["scope_key"],
        row["canonical_request"],
        row["command_fingerprint"],
        canonical,
        "0" * 64,
    ]
    payload = (
        authority_module.TEST_ORDER_AUTHENTICITY_PURPOSE.encode("ascii")
        + b"\x00"
        + authority_module.TEST_ORDER_AUTHORITY_DOMAIN.encode()
        + b"\x00COMMAND\x00"
        + authority_module._canonical(auth_fields).encode()
    )
    mac = (
        authority_module.DeterministicTestOrderAuthoritySecretCustody()
        .digest(handle, payload)
        .hex()
    )
    db.execute(
        "UPDATE command_journal SET canonical_outcome=?,record_digest=?,admission_mac=?",
        (canonical, digest, mac),
    )
    db.execute(
        "UPDATE authority_heads SET command_head=?,command_authenticated_head=?", (digest, mac)
    )
    db.commit()
    db.close()
    with pytest.raises(AuthorityCorrupt, match="outcome order identity"):
        TestOrderAuthority(path)


def test_test_domain_mac_cannot_launder_into_production_store(tmp_path: Path) -> None:
    test_path = tmp_path / "source-test.sqlite"
    test = TestOrderAuthority(test_path)
    test.submit_order(command())
    test.close()
    prod_path = tmp_path / "target-prod.sqlite"
    prod = OrderAuthority(prod_path)
    prod.close()
    source = sqlite3.connect(test_path)
    source.row_factory = sqlite3.Row
    command_row = source.execute("SELECT * FROM command_journal").fetchone()
    event_row = source.execute("SELECT * FROM event_journal").fetchone()
    source.close()
    db = sqlite3.connect(prod_path)
    db.execute("INSERT INTO command_journal VALUES(?,?,?,?,?,?,?,?,?,?)", tuple(command_row))
    db.execute("INSERT INTO event_journal VALUES(?,?,?,?,?,?,?,?,?,?)", tuple(event_row))
    db.execute(
        "UPDATE authority_heads SET command_count=1,command_head=?,command_authenticated_head=?,event_count=1,event_head=?,event_authenticated_head=?",
        (command_row[7], command_row[9], event_row[7], event_row[9]),
    )
    db.commit()
    db.close()
    with pytest.raises(AuthorityCorrupt, match="authenticated admission"):
        OrderAuthority(prod_path)


def test_wrong_or_missing_record_mac_fails_closed(tmp_path: Path) -> None:
    path = tmp_path / "wrong-mac.sqlite"
    authority = TestOrderAuthority(path)
    authority.submit_order(command())
    authority.close()
    db = sqlite3.connect(path)
    db.execute("DROP TRIGGER event_no_update")
    db.execute("UPDATE event_journal SET admission_mac='' ")
    db.execute("UPDATE authority_heads SET event_authenticated_head='' ")
    db.commit()
    db.close()
    with pytest.raises(AuthorityCorrupt, match="authenticated admission"):
        TestOrderAuthority(path)


@pytest.mark.parametrize(
    ("field", "replacement", "message"),
    [
        ("quantity", "2", "initial plan quantity mismatch"),
        ("side", "SELL", "initial plan side mismatch"),
        ("order_type", "LIMIT", "initial plan order_type mismatch"),
    ],
)
def test_authenticated_initial_plan_payload_must_match_submit_order(
    field: str, replacement: str, message: str, tmp_path: Path
) -> None:
    path = tmp_path / f"plan-{field}.sqlite"
    authority = TestOrderAuthority(path)
    authority.submit_order(command())
    authority.close()

    def mutate(event: dict[str, object]) -> None:
        if event["event_type"] == "ORDER_PLANNED":
            event["safe_payload"][field] = replacement

    _rewrite_test_event_chain(path, mutate)
    with pytest.raises(AuthorityCorrupt, match=message):
        TestOrderAuthority(path)


def test_authenticated_coherent_whole_event_scope_rewrite_still_binds_command(
    tmp_path: Path,
) -> None:
    path = tmp_path / "whole-scope.sqlite"
    authority = _acknowledged(path)
    authority.close()
    replacement = "ws_018f0f3e-7b5a-7abc-8def-1234567890ac"

    def mutate(event: dict[str, object]) -> None:
        event["workspace_id"] = replacement

    _rewrite_test_event_chain(path, mutate)
    with pytest.raises(AuthorityCorrupt, match="initial event workspace_id mismatch"):
        TestOrderAuthority(path)


def test_accepted_outcome_must_resolve_exact_initial_event(tmp_path: Path) -> None:
    path = tmp_path / "wrong-initial.sqlite"
    authority = TestOrderAuthority(path)
    authority.submit_order(command())
    authority.close()

    def mutate(event: dict[str, object]) -> None:
        event["audit_event_id"] = "evt_018f0f3e-7b5a-7abc-8def-1234567890ac"

    _rewrite_test_event_chain(path, mutate)
    with pytest.raises(AuthorityCorrupt, match="no referenced initial event"):
        TestOrderAuthority(path)


def test_initial_event_must_resolve_exact_accepted_command(tmp_path: Path) -> None:
    path = tmp_path / "wrong-command.sqlite"
    authority = TestOrderAuthority(path)
    authority.submit_order(command())
    authority.close()

    def mutate(event: dict[str, object]) -> None:
        event["command_id"] = "cmd_018f0f3e-7b5a-7abc-8def-1234567890ac"

    _rewrite_test_event_chain(path, mutate)
    with pytest.raises(AuthorityCorrupt, match="command_id mismatch"):
        TestOrderAuthority(path)


def test_fill_cannot_expand_authenticated_plan_beyond_submit_quantity(tmp_path: Path) -> None:
    path = tmp_path / "expanded-plan-fill.sqlite"
    authority = _acknowledged(path)
    authority.append_event(_fill_event(authority, "ORDER_FILLED", "1", 4))
    authority.close()

    def mutate(event: dict[str, object]) -> None:
        if event["event_type"] == "ORDER_PLANNED":
            event["safe_payload"]["quantity"] = "2"
        if event["event_type"] == "ORDER_FILLED":
            event["safe_payload"]["cumulative_executed_quantity"] = "2"

    _rewrite_test_event_chain(path, mutate)
    with pytest.raises(AuthorityCorrupt, match="initial plan quantity mismatch"):
        TestOrderAuthority(path)
