"""Durable M0.7 Order authority mechanics.

The frozen JSON contract is the source of truth.  Production semantic admission is
intentionally unavailable until its upstream authorities exist; the separate test
authority is the only deterministic fixture admission surface.
"""

from __future__ import annotations

import hashlib
import hmac
import secrets
import json
import re
import sqlite3
import threading
import unicodedata
from collections.abc import Mapping
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path
from types import MappingProxyType
from typing import Any, Final

from .custody import (
    DeterministicTestOrderAuthoritySecretCustody,
    ORDER_AUTHENTICITY_ALGORITHM,
    PRODUCTION_ORDER_AUTHENTICITY_PURPOSE,
    TEST_ORDER_AUTHENTICITY_PURPOSE,
    OrderAuthoritySecretCustody,
    production_order_authority_custody,
)

_CONTRACT_PATH = (
    Path(__file__).parents[2]
    / "docs/architecture/cryptohunter_product_architecture"
    / "commands_events_order_lifecycle_and_idempotency.json"
)
CONTRACT: Final[dict[str, Any]] = json.loads(_CONTRACT_PATH.read_text(encoding="utf-8"))
PRODUCTION_ORDER_AUTHORITY_DOMAIN: Final = "cryptohunter.m0.7.order-authority.production.v1"
TEST_ORDER_AUTHORITY_DOMAIN: Final = "cryptohunter.m0.7.order-authority.test.v1"
VALID_PREFIX_ROLLBACK_THREAT: Final = "NOT_DETECTABLE_WITH_LOCAL_CHAIN_AND_HEAD"
SEMANTIC_SUBMIT_ORDER_AVAILABLE: Final = False

_ID = re.compile(
    r"^(?P<prefix>[a-z]+)_[0-9a-f]{8}-[0-9a-f]{4}-7[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$"
)
_DECIMAL = re.compile(r"^(0|[1-9][0-9]*)(\.[0-9]*[1-9])?$")
_TIMESTAMP = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d*[1-9])?Z$")
_SHA = re.compile(r"^[0-9a-f]{64}$")
_ZERO = "0" * 64


class OrderAuthorityError(RuntimeError):
    """Stable fail-closed error carrying a frozen or storage denial code."""

    def __init__(self, code: str, detail: str = "") -> None:
        self.code = code
        super().__init__(f"{code}: {detail}" if detail else code)


class UpstreamAuthorityUnavailable(OrderAuthorityError):
    def __init__(self) -> None:
        super().__init__(
            "TRUSTED_CONTEXT_FAILURE",
            "Workspace, ExchangeAccount, Instrument, and execution-route authorities unavailable",
        )


class AuthorityCorrupt(OrderAuthorityError):
    def __init__(self, detail: str) -> None:
        super().__init__("CONTRACT_INCONSISTENT", detail)


def _canonical(value: Any) -> str:
    def normalize(item: Any) -> Any:
        if type(item) is str:
            return unicodedata.normalize("NFC", item)
        if type(item) is dict:
            if any(type(k) is not str for k in item):
                raise OrderAuthorityError("MALFORMED_REQUEST", "JSON keys must be strings")
            return {normalize(k): normalize(v) for k, v in item.items()}
        if type(item) is list:
            return [normalize(v) for v in item]
        if item is None or type(item) in (bool, int):
            return item
        raise OrderAuthorityError("MALFORMED_REQUEST", "non-canonical JSON value")

    return json.dumps(normalize(value), ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _digest(value: Any) -> str:
    return hashlib.sha256(_canonical(value).encode("utf-8")).hexdigest()


def _validate_timestamp(value: Any, code: str) -> None:
    if (
        type(value) is not str
        or not _TIMESTAMP.fullmatch(value)
        or unicodedata.normalize("NFC", value) != value
    ):
        raise OrderAuthorityError(code, "non-canonical timestamp")
    try:
        parsed = datetime.fromisoformat(value[:-1] + "+00:00")
    except ValueError as exc:
        raise OrderAuthorityError(code, "invalid timestamp") from exc
    if parsed.tzinfo != UTC:
        raise OrderAuthorityError(code, "timestamp is not UTC")


def _validate_field(name: str, value: Any, schema: Mapping[str, Any], code: str) -> None:
    kind = schema["type"]
    if kind == "id":
        match = _ID.fullmatch(value) if type(value) is str else None
        if match is None or match.group("prefix") != schema["prefix"]:
            raise OrderAuthorityError(code, f"invalid {name}")
    elif kind in ("string", "non_empty_string"):
        if (
            type(value) is not str
            or unicodedata.normalize("NFC", value) != value
            or (kind == "non_empty_string" and not value)
        ):
            raise OrderAuthorityError(code, f"invalid {name}")
    elif kind == "enum":
        values = schema.get(
            "values", CONTRACT["event_contract"][schema["registry"]] if "registry" in schema else ()
        )
        if type(value) is not str or value not in values:
            raise OrderAuthorityError(code, f"invalid {name}")
    elif kind == "constant":
        if type(value) is not str or value != schema["value"]:
            raise OrderAuthorityError(code, f"invalid {name}")
    elif kind == "decimal":
        if (
            type(value) is not str
            or not _DECIMAL.fullmatch(value)
            or (code == "MALFORMED_REQUEST" and value == "0")
        ):
            raise OrderAuthorityError(code, f"invalid canonical decimal {name}")
    elif kind == "positive_integer":
        if type(value) is not int or value <= 0:
            raise OrderAuthorityError(code, f"invalid {name}")
    elif kind == "timestamp":
        _validate_timestamp(value, code)
    elif kind == "sha256_hex":
        if type(value) is not str or not _SHA.fullmatch(value):
            raise OrderAuthorityError(code, f"invalid {name}")
    elif kind == "event_safe_payload":
        if type(value) is not dict:
            raise OrderAuthorityError(code, "safe_payload must be a plain object")
    else:
        raise AuthorityCorrupt(f"unknown frozen field type {kind}")


def _closed_validate(
    value: Mapping[str, Any],
    fields: list[str],
    schemas: Mapping[str, Any],
    nullable: list[str],
    code: str,
) -> None:
    if type(value) is not dict or set(value) != set(fields):
        raise OrderAuthorityError(code, "closed schema field mismatch")
    for name in fields:
        item = value[name]
        if item is None:
            if name not in nullable:
                raise OrderAuthorityError(code, f"{name} is not nullable")
        else:
            _validate_field(name, item, schemas[name], code)


def validate_order_command(request: Mapping[str, Any]) -> Mapping[str, Any]:
    """Validate and return an immutable canonical command copy."""
    if type(request) is not dict or type(request.get("operation_type")) is not str:
        raise OrderAuthorityError("MALFORMED_REQUEST", "plain command object required")
    spec = CONTRACT["command_registry"].get(request["operation_type"])
    if spec is None:
        raise OrderAuthorityError("MALFORMED_REQUEST", "unknown operation_type")
    _closed_validate(
        request,
        spec["request_fields"],
        spec["field_schemas"],
        spec["nullable_fields"],
        "MALFORMED_REQUEST",
    )
    if request["idempotency_key"] != request["command_id"]:
        raise OrderAuthorityError("MALFORMED_REQUEST", "idempotency_key must equal command_id")
    source_has_strategy = request["source_type"] == "STRATEGY_INSTANCE"
    if source_has_strategy != (request["strategy_instance_id"] is not None):
        raise OrderAuthorityError("MALFORMED_REQUEST", "strategy source binding")
    op = request["operation_type"]
    if op == "SUBMIT_ORDER":
        if (request["order_type"] == "MARKET") != (request["limit_price"] is None):
            raise OrderAuthorityError("MALFORMED_REQUEST", "limit_price/order_type mismatch")
    if op in ("SUBMIT_ORDER", "REPLACE_ORDER"):
        if (request["time_in_force"] == "GTD") != (request["expire_at_utc"] is not None):
            raise OrderAuthorityError("MALFORMED_REQUEST", "expire_at_utc/time_in_force mismatch")
    if op == "REPLACE_ORDER" and request["replacement_order_id"] == request["order_id"]:
        raise OrderAuthorityError("MALFORMED_REQUEST", "replacement identity must differ")
    return _freeze(json.loads(_canonical(request)))


def validate_command_outcome(outcome: Mapping[str, Any]) -> Mapping[str, Any]:
    """Validate the exact immutable public command-outcome shape."""
    fields = CONTRACT["command_outcome_contract"]["fields"]
    if type(outcome) is not dict or set(outcome) != set(fields):
        raise OrderAuthorityError("CONTRACT_INCONSISTENT", "command outcome field mismatch")
    for name in ("command_id",):
        _validate_field(
            name, outcome[name], {"type": "id", "prefix": "cmd"}, "CONTRACT_INCONSISTENT"
        )
    if (
        type(outcome["operation_type"]) is not str
        or outcome["operation_type"] not in CONTRACT["command_registry"]
    ):
        raise OrderAuthorityError("CONTRACT_INCONSISTENT", "invalid outcome operation")
    _validate_field(
        "request_fingerprint_sha256",
        outcome["request_fingerprint_sha256"],
        {"type": "sha256_hex"},
        "CONTRACT_INCONSISTENT",
    )
    if (
        type(outcome["outcome"]) is not str
        or outcome["outcome"] not in CONTRACT["command_outcome_contract"]["outcomes"]
    ):
        raise OrderAuthorityError("CONTRACT_INCONSISTENT", "invalid command outcome")
    if outcome["denial_code"] is not None and (
        type(outcome["denial_code"]) is not str
        or outcome["denial_code"] not in CONTRACT["failure_taxonomy"]
    ):
        raise OrderAuthorityError("CONTRACT_INCONSISTENT", "invalid denial code")
    for name, prefix in (("order_id", "ord"), ("accepted_audit_event_id", "evt")):
        if outcome[name] is not None:
            _validate_field(
                name, outcome[name], {"type": "id", "prefix": prefix}, "CONTRACT_INCONSISTENT"
            )
    _validate_timestamp(outcome["recorded_at_utc"], "CONTRACT_INCONSISTENT")
    return _freeze(json.loads(_canonical(outcome)))


def command_fingerprint_sha256(request: Mapping[str, Any]) -> str:
    validate_order_command(request)
    value = {
        k: v
        for k, v in request.items()
        if k not in CONTRACT["idempotency_contract"]["fingerprint_excluded_fields"]
    }
    return _digest(value)


def validate_order_event(event: Mapping[str, Any]) -> Mapping[str, Any]:
    """Validate a complete frozen event envelope and its public fingerprint."""
    envelope = CONTRACT["event_contract"]["envelope_schema"]
    _closed_validate(
        event,
        envelope["fields"],
        envelope["field_schemas"],
        envelope["nullable_fields"],
        "MALFORMED_EVENT",
    )
    spec = CONTRACT["event_contract"]["event_schema_registry"].get(event["event_type"])
    if spec is None:
        raise OrderAuthorityError("MALFORMED_EVENT", "unknown event_type")
    _closed_validate(
        event["safe_payload"],
        spec["safe_payload_fields"],
        spec["field_schemas"],
        spec["nullable_fields"],
        "MALFORMED_EVENT",
    )
    if event_fingerprint_sha256(event) != event["event_fingerprint_sha256"]:
        raise OrderAuthorityError("EVENT_FINGERPRINT_MISMATCH")
    return _freeze(json.loads(_canonical(event)))


def event_fingerprint_sha256(event: Mapping[str, Any]) -> str:
    excluded = CONTRACT["event_contract"]["envelope_schema"]["fingerprint_excluded_fields"]
    return _digest({k: v for k, v in event.items() if k not in excluded})


def _freeze(value: Any) -> Any:
    if type(value) is dict:
        return MappingProxyType({k: _freeze(v) for k, v in value.items()})
    if type(value) is list:
        return tuple(_freeze(v) for v in value)
    return value


_TRANSITIONS = {x["event"]: x for x in CONTRACT["order_lifecycle"]["transitions"]}
_TERMINAL = frozenset(CONTRACT["order_lifecycle"]["terminal_states"])
_RECONCILE = CONTRACT["order_lifecycle"]["dynamic_target_rules"][
    "RESOLVE_TRUSTED_RECONCILIATION_FACT"
]["trusted_fact_targets"]
_REQUIRED_TABLES = frozenset(
    {"authority_marker", "authority_heads", "command_journal", "event_journal"}
)


class OrderAuthority:
    """Production durable kernel; semantic SUBMIT_ORDER intentionally fails closed."""

    domain = PRODUCTION_ORDER_AUTHORITY_DOMAIN
    _test_admission = False
    authenticity_purpose = PRODUCTION_ORDER_AUTHENTICITY_PURPOSE

    def __init__(self, database: str | Path) -> None:
        self._path = str(database)
        self._custody = self._make_custody()
        self._custody_handle = ""
        self._lock = threading.RLock()
        self._db = sqlite3.connect(
            self._path, timeout=30, isolation_level=None, check_same_thread=False
        )
        self._db.row_factory = sqlite3.Row
        self._db.execute("PRAGMA foreign_keys=ON")
        self._db.execute("PRAGMA busy_timeout=30000")
        if self._path != ":memory:":
            self._db.execute("PRAGMA journal_mode=WAL")
        try:
            self._bootstrap_or_verify()
        except AuthorityCorrupt:
            raise
        except (sqlite3.Error, KeyError, TypeError, ValueError) as exc:
            raise AuthorityCorrupt(
                "durable Order authority bootstrap rejected corrupt state"
            ) from exc

    def _make_custody(self) -> OrderAuthoritySecretCustody:
        return production_order_authority_custody()

    def close(self) -> None:
        self._db.close()

    def __enter__(self) -> OrderAuthority:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def submit_order(self, request: Mapping[str, Any]) -> Mapping[str, Any]:
        validate_order_command(request)
        if request["operation_type"] != "SUBMIT_ORDER":
            raise OrderAuthorityError("MALFORMED_REQUEST", "SUBMIT_ORDER required")
        raise UpstreamAuthorityUnavailable()

    def resolve_current(self, order_id: str) -> Mapping[str, Any] | None:
        self._validate_id(order_id, "ord")
        with self._lock:
            self._verify_integrity()
            history = self._history_rows(order_id)
            if not history:
                return None
            state, pre = self._replay(history)
            return _freeze(
                {
                    "order_id": order_id,
                    "aggregate_version": len(history),
                    "state": state,
                    "pre_request_state": pre,
                }
            )

    def resolve_event(self, order_id: str, aggregate_version: int) -> Mapping[str, Any] | None:
        self._validate_id(order_id, "ord")
        if type(aggregate_version) is not int or aggregate_version <= 0:
            raise OrderAuthorityError("MALFORMED_REQUEST", "positive aggregate version required")
        with self._lock:
            self._verify_integrity()
            row = self._db.execute(
                "SELECT canonical_event FROM event_journal WHERE order_id=? AND aggregate_version=?",
                (order_id, aggregate_version),
            ).fetchone()
            return None if row is None else _freeze(json.loads(row[0]))

    def history(self, order_id: str) -> tuple[Mapping[str, Any], ...]:
        self._validate_id(order_id, "ord")
        with self._lock:
            self._verify_integrity()
            return tuple(
                _freeze(json.loads(row["canonical_event"])) for row in self._history_rows(order_id)
            )

    @property
    def authority_head(self) -> Mapping[str, Any]:
        with self._lock:
            self._verify_integrity()
            row = self._db.execute("SELECT * FROM authority_heads WHERE singleton=1").fetchone()
            return _freeze(dict(row))

    def _bootstrap_or_verify(self) -> None:
        tables = {
            r[0]
            for r in self._db.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"
            )
        }
        marker = "authority_marker" in tables
        durable = bool(tables & _REQUIRED_TABLES)
        if not marker and durable:
            raise AuthorityCorrupt("authority marker missing from nonempty authority state")
        if marker:
            if not _REQUIRED_TABLES <= tables:
                raise AuthorityCorrupt("initialized authority is missing a required table")
            row = self._db.execute(
                "SELECT domain,purpose,algorithm,custody_handle,schema_version FROM authority_marker WHERE singleton=1"
            ).fetchone()
            if (
                row is None
                or tuple(row)[:3]
                != (self.domain, self.authenticity_purpose, ORDER_AUTHENTICITY_ALGORITHM)
                or row[4] != 2
            ):
                raise AuthorityCorrupt("authority domain, purpose, algorithm, or schema mismatch")
            self._custody_handle = row[3]
            self._verify_integrity()
            return
        if tables:
            raise AuthorityCorrupt("database is not an empty Order authority store")
        self._custody_handle = "ordauthk_" + secrets.token_hex(24)
        try:
            self._custody.persist(self._custody_handle, secrets.token_bytes(32))
        except Exception as exc:
            raise AuthorityCorrupt("Order authority secret custody unavailable") from exc
        self._db.executescript(
            """
        BEGIN IMMEDIATE;
        CREATE TABLE authority_marker(singleton INTEGER PRIMARY KEY CHECK(singleton=1), domain TEXT NOT NULL, purpose TEXT NOT NULL, algorithm TEXT NOT NULL, custody_handle TEXT NOT NULL, schema_version INTEGER NOT NULL);
        CREATE TABLE authority_heads(singleton INTEGER PRIMARY KEY CHECK(singleton=1), command_count INTEGER NOT NULL, command_head TEXT NOT NULL, command_authenticated_head TEXT NOT NULL, event_count INTEGER NOT NULL, event_head TEXT NOT NULL, event_authenticated_head TEXT NOT NULL);
        CREATE TABLE command_journal(command_sequence INTEGER PRIMARY KEY, command_id TEXT NOT NULL UNIQUE, scope_key TEXT NOT NULL, canonical_request TEXT NOT NULL, command_fingerprint TEXT NOT NULL, canonical_outcome TEXT NOT NULL, previous_digest TEXT NOT NULL, record_digest TEXT NOT NULL UNIQUE, previous_authenticated_commitment TEXT NOT NULL, admission_mac TEXT NOT NULL UNIQUE);
        CREATE TABLE event_journal(event_sequence INTEGER PRIMARY KEY, audit_event_id TEXT NOT NULL UNIQUE, order_id TEXT NOT NULL, aggregate_version INTEGER NOT NULL, canonical_event TEXT NOT NULL, event_fingerprint TEXT NOT NULL, previous_digest TEXT NOT NULL, record_digest TEXT NOT NULL UNIQUE, previous_authenticated_commitment TEXT NOT NULL, admission_mac TEXT NOT NULL UNIQUE, UNIQUE(order_id, aggregate_version));
        CREATE TRIGGER command_no_update BEFORE UPDATE ON command_journal BEGIN SELECT RAISE(ABORT,'immutable command journal'); END;
        CREATE TRIGGER command_no_delete BEFORE DELETE ON command_journal BEGIN SELECT RAISE(ABORT,'immutable command journal'); END;
        CREATE TRIGGER event_no_update BEFORE UPDATE ON event_journal BEGIN SELECT RAISE(ABORT,'immutable event journal'); END;
        CREATE TRIGGER event_no_delete BEFORE DELETE ON event_journal BEGIN SELECT RAISE(ABORT,'immutable event journal'); END;
        INSERT INTO authority_marker VALUES(1,'"""
            + self.domain
            + """','"""
            + self.authenticity_purpose
            + """','"""
            + ORDER_AUTHENTICITY_ALGORITHM
            + """','"""
            + self._custody_handle
            + """',2);
        INSERT INTO authority_heads VALUES(1,0,'"""
            + _ZERO
            + """','"""
            + _ZERO
            + """',0,'"""
            + _ZERO
            + """','"""
            + _ZERO
            + """');
        COMMIT;
        """
        )
        self._verify_integrity()

    def _authenticated_commitment(self, kind: str, fields: list[Any]) -> str:
        payload = (
            self.authenticity_purpose.encode("ascii")
            + b"\x00"
            + self.domain.encode("utf-8")
            + b"\x00"
            + kind.encode("ascii")
            + b"\x00"
            + _canonical(fields).encode("utf-8")
        )
        result = self._custody.digest(self._custody_handle, payload)
        if result is None:
            raise AuthorityCorrupt("Order authority authentication key unavailable")
        return result.hex()

    def _verify_integrity(self) -> None:
        try:
            self._verify_integrity_raw()
        except AuthorityCorrupt:
            raise
        except (
            OrderAuthorityError,
            sqlite3.Error,
            KeyError,
            TypeError,
            ValueError,
            AssertionError,
        ) as exc:
            raise AuthorityCorrupt("durable Order authority replay rejected corrupt state") from exc

    def _verify_integrity_raw(self) -> None:
        tables = {
            r[0]
            for r in self._db.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"
            )
        }
        if not _REQUIRED_TABLES <= tables:
            raise AuthorityCorrupt("required authority table missing")
        marker = self._db.execute(
            "SELECT domain,purpose,algorithm,custody_handle,schema_version FROM authority_marker WHERE singleton=1"
        ).fetchall()
        if len(marker) != 1 or tuple(marker[0]) != (
            self.domain,
            self.authenticity_purpose,
            ORDER_AUTHENTICITY_ALGORITHM,
            self._custody_handle,
            2,
        ):
            raise AuthorityCorrupt("authority marker mismatch")
        heads = self._db.execute("SELECT * FROM authority_heads WHERE singleton=1").fetchall()
        if len(heads) != 1:
            raise AuthorityCorrupt("authority head missing")
        command_head = command_auth = _ZERO
        command_count = 0
        accepted_submits: dict[str, tuple[dict[str, Any], dict[str, Any]]] = {}
        for row in self._db.execute("SELECT * FROM command_journal ORDER BY command_sequence"):
            command_count += 1
            if (
                row["command_sequence"] != command_count
                or row["previous_digest"] != command_head
                or row["previous_authenticated_commitment"] != command_auth
            ):
                raise AuthorityCorrupt("command ordering or previous commitment mismatch")
            request = json.loads(row["canonical_request"])
            outcome = json.loads(row["canonical_outcome"])
            validate_command_outcome(outcome)
            validate_order_command(request)
            fingerprint = command_fingerprint_sha256(request)
            expected_scope = _canonical(
                [request[k] for k in CONTRACT["idempotency_contract"]["scope"]]
            )
            if (
                _canonical(request) != row["canonical_request"]
                or fingerprint != row["command_fingerprint"]
                or request["command_id"] != row["command_id"]
                or row["scope_key"] != expected_scope
            ):
                raise AuthorityCorrupt("command canonical payload, scope, or fingerprint mismatch")
            if (
                outcome["command_id"] != request["command_id"]
                or outcome["operation_type"] != request["operation_type"]
                or outcome["request_fingerprint_sha256"] != fingerprint
            ):
                raise AuthorityCorrupt("command outcome is not bound to request")
            if (
                request["operation_type"] == "SUBMIT_ORDER"
                and outcome["order_id"] != request["order_id"]
            ):
                raise AuthorityCorrupt("command outcome order identity mismatch")
            data = [
                row["command_sequence"],
                row["command_id"],
                row["scope_key"],
                row["canonical_request"],
                row["command_fingerprint"],
                row["canonical_outcome"],
                row["previous_digest"],
            ]
            command_head = _digest(data)
            auth_fields = [
                row["command_sequence"],
                row["command_id"],
                row["scope_key"],
                row["canonical_request"],
                row["command_fingerprint"],
                row["canonical_outcome"],
                row["previous_authenticated_commitment"],
            ]
            expected_mac = self._authenticated_commitment("COMMAND", auth_fields)
            if command_head != row["record_digest"] or not hmac.compare_digest(
                expected_mac, row["admission_mac"]
            ):
                raise AuthorityCorrupt("command public digest or authenticated admission mismatch")
            command_auth = row["admission_mac"]
            if outcome["outcome"] == "ACCEPTED" and request["operation_type"] == "SUBMIT_ORDER":
                if request["command_id"] in accepted_submits:
                    raise AuthorityCorrupt("duplicate accepted SUBMIT_ORDER command identity")
                accepted_submits[request["command_id"]] = (request, outcome)
        event_head = event_auth = _ZERO
        event_count = 0
        by_order: dict[str, list[sqlite3.Row]] = {}
        events_by_id: dict[str, dict[str, Any]] = {}
        for row in self._db.execute("SELECT * FROM event_journal ORDER BY event_sequence"):
            event_count += 1
            if (
                row["event_sequence"] != event_count
                or row["previous_digest"] != event_head
                or row["previous_authenticated_commitment"] != event_auth
            ):
                raise AuthorityCorrupt("event ordering or previous commitment mismatch")
            event = json.loads(row["canonical_event"])
            validate_order_event(event)
            if (
                _canonical(event) != row["canonical_event"]
                or event["audit_event_id"] != row["audit_event_id"]
                or event["order_id"] != row["order_id"]
                or event["aggregate_version"] != row["aggregate_version"]
                or event["event_fingerprint_sha256"] != row["event_fingerprint"]
            ):
                raise AuthorityCorrupt("event indexed envelope mismatch")
            data = [
                row["event_sequence"],
                row["audit_event_id"],
                row["order_id"],
                row["aggregate_version"],
                row["canonical_event"],
                row["event_fingerprint"],
                row["previous_digest"],
            ]
            event_head = _digest(data)
            auth_fields = [
                row["event_sequence"],
                row["audit_event_id"],
                row["order_id"],
                row["aggregate_version"],
                row["canonical_event"],
                row["event_fingerprint"],
                row["previous_authenticated_commitment"],
            ]
            expected_mac = self._authenticated_commitment("EVENT", auth_fields)
            if event_head != row["record_digest"] or not hmac.compare_digest(
                expected_mac, row["admission_mac"]
            ):
                raise AuthorityCorrupt("event public digest or authenticated admission mismatch")
            event_auth = row["admission_mac"]
            by_order.setdefault(row["order_id"], []).append(row)
            events_by_id[event["audit_event_id"]] = event
        actual_heads = (
            heads[0]["command_count"],
            heads[0]["command_head"],
            heads[0]["command_authenticated_head"],
            heads[0]["event_count"],
            heads[0]["event_head"],
            heads[0]["event_authenticated_head"],
        )
        if actual_heads != (
            command_count,
            command_head,
            command_auth,
            event_count,
            event_head,
            event_auth,
        ):
            raise AuthorityCorrupt("authority head rollback or mismatch")
        for rows in by_order.values():
            self._replay(rows)
        referenced_plans: set[str] = set()
        for request, outcome in accepted_submits.values():
            event_id = outcome["accepted_audit_event_id"]
            event = events_by_id.get(event_id)
            if event is None:
                raise AuthorityCorrupt("accepted SUBMIT_ORDER has no referenced initial event")
            self._verify_submit_order_plan_binding(request, outcome, event)
            if event_id in referenced_plans:
                raise AuthorityCorrupt("initial ORDER_PLANNED is referenced more than once")
            referenced_plans.add(event_id)
        planned_ids = {
            event_id
            for event_id, event in events_by_id.items()
            if event["event_type"] == "ORDER_PLANNED"
        }
        if planned_ids != referenced_plans:
            raise AuthorityCorrupt("orphan or multiply-bound ORDER_PLANNED event")

    @staticmethod
    def _verify_submit_order_plan_binding(
        request: Mapping[str, Any], outcome: Mapping[str, Any], event: Mapping[str, Any]
    ) -> None:
        if event["event_type"] != "ORDER_PLANNED" or event["aggregate_version"] != 1:
            raise AuthorityCorrupt("accepted SUBMIT_ORDER does not reference initial ORDER_PLANNED")
        if outcome["accepted_audit_event_id"] != event["audit_event_id"]:
            raise AuthorityCorrupt("accepted outcome audit identity mismatch")
        for name in ("command_id", "order_id"):
            if request[name] != outcome[name] or outcome[name] != event[name]:
                raise AuthorityCorrupt(f"SUBMIT_ORDER outcome/event {name} mismatch")
        for name in (
            "environment",
            "workspace_id",
            "portfolio_id",
            "exchange_account_id",
            "instrument_id",
            "execution_route_id",
            "correlation_id",
            "causation_id",
        ):
            if event[name] != request[name]:
                raise AuthorityCorrupt(f"SUBMIT_ORDER initial event {name} mismatch")
        for name in ("side", "order_type", "quantity"):
            if event["safe_payload"][name] != request[name]:
                raise AuthorityCorrupt(f"SUBMIT_ORDER initial plan {name} mismatch")

    def _history_rows(self, order_id: str) -> list[sqlite3.Row]:
        return list(
            self._db.execute(
                "SELECT * FROM event_journal WHERE order_id=? ORDER BY aggregate_version",
                (order_id,),
            )
        )

    @staticmethod
    def _fill_identity(
        event: Mapping[str, Any],
    ) -> tuple[str, tuple[str, str, str, str], tuple[str, ...]]:
        payload = event["safe_payload"]
        external = (
            event["environment"],
            event["exchange_account_id"],
            event["exchange_id"],
            payload["venue_trade_id"],
        )
        economics = tuple(
            str(event[name])
            for name in (
                "order_id",
                "environment",
                "workspace_id",
                "portfolio_id",
                "exchange_account_id",
                "exchange_id",
                "instrument_id",
                "execution_route_id",
            )
        ) + (payload["venue_trade_id"], payload["cumulative_executed_quantity"])
        return payload["fill_id"], external, economics

    def _fill_duplicate_decision(
        self, rows: list[sqlite3.Row], candidate: Mapping[str, Any]
    ) -> str | None:
        fill_id, external, economics = self._fill_identity(candidate)
        for row in rows:
            prior = json.loads(row["canonical_event"])
            if prior["event_type"] not in ("ORDER_PARTIALLY_FILLED", "ORDER_FILLED"):
                continue
            old_fill, old_external, old_economics = self._fill_identity(prior)
            if fill_id == old_fill or external == old_external:
                return "REPLAY_SUCCESS" if economics == old_economics else "FILL_IDENTITY_CONFLICT"
        return None

    def _replay(self, rows: list[sqlite3.Row]) -> tuple[str, str | None]:
        state: str | None = None
        pre: str | None = None
        scope: tuple[Any, ...] | None = None
        ordered: Decimal | None = None
        cumulative = Decimal("0")
        seen_fills: dict[str, tuple[str, ...]] = {}
        seen_external: dict[tuple[str, str, str, str], tuple[str, ...]] = {}
        scope_fields = CONTRACT["event_contract"]["envelope_schema"]["scope_fields"]
        order_id: str | None = None
        for expected, row in enumerate(rows, 1):
            event = json.loads(row["canonical_event"])
            if event["aggregate_version"] != expected:
                raise OrderAuthorityError("EVENT_VERSION_GAP")
            current_scope = tuple(event[name] for name in scope_fields)
            if scope is None:
                scope = current_scope
                order_id = event["order_id"]
            elif current_scope != scope or event["order_id"] != order_id:
                raise OrderAuthorityError("ORDER_SCOPE_MISMATCH")
            if event["event_type"] == "ORDER_PLANNED":
                if expected != 1:
                    raise OrderAuthorityError("INVALID_LIFECYCLE_TRANSITION")
                ordered = Decimal(event["safe_payload"]["quantity"])
            elif event["event_type"] in ("ORDER_PARTIALLY_FILLED", "ORDER_FILLED"):
                if ordered is None:
                    raise OrderAuthorityError("FILL_PROGRESSION_CONFLICT")
                fill_id, external, economics = self._fill_identity(event)
                if fill_id in seen_fills or external in seen_external:
                    previous = seen_fills.get(fill_id, seen_external.get(external))
                    code = "REPLAY_SUCCESS" if previous == economics else "FILL_IDENTITY_CONFLICT"
                    raise OrderAuthorityError(code)
                new_cumulative = Decimal(event["safe_payload"]["cumulative_executed_quantity"])
                if new_cumulative <= cumulative or new_cumulative > ordered:
                    raise OrderAuthorityError("FILL_PROGRESSION_CONFLICT")
                if event["event_type"] == "ORDER_PARTIALLY_FILLED" and not (
                    Decimal("0") < new_cumulative < ordered
                ):
                    raise OrderAuthorityError("FILL_PROGRESSION_CONFLICT")
                if event["event_type"] == "ORDER_FILLED" and new_cumulative != ordered:
                    raise OrderAuthorityError("FILL_PROGRESSION_CONFLICT")
                seen_fills[fill_id] = economics
                seen_external[external] = economics
                cumulative = new_cumulative
            state, pre = self._next_state(state, pre, event)
        if state is None:
            raise OrderAuthorityError("INVALID_LIFECYCLE_TRANSITION")
        return state, pre

    def _next_state(
        self, state: str | None, pre: str | None, event: Mapping[str, Any]
    ) -> tuple[str, str | None]:
        transition = _TRANSITIONS.get(event["event_type"])
        source = "NONE" if state is None else state
        if transition is None or source not in transition["sources"] or (state in _TERMINAL):
            raise OrderAuthorityError("INVALID_LIFECYCLE_TRANSITION")
        rule = transition.get("target_rule")
        if rule == "RESTORE_PRE_REQUEST_STATE":
            if pre not in ("ACKNOWLEDGED", "PARTIALLY_FILLED"):
                raise OrderAuthorityError("INVALID_LIFECYCLE_TRANSITION")
            return pre, None
        if rule == "RESOLVE_TRUSTED_RECONCILIATION_FACT":
            target = _RECONCILE.get(event["safe_payload"]["trusted_fact_kind"])
            if target is None:
                raise OrderAuthorityError("TRUSTED_CONTEXT_FAILURE")
            return target, None
        target = transition["target"]
        return target, state if target in ("CANCEL_PENDING", "REPLACE_PENDING") else pre

    @staticmethod
    def _validate_id(value: Any, prefix: str) -> None:
        match = _ID.fullmatch(value) if type(value) is str else None
        if match is None or match.group("prefix") != prefix:
            raise OrderAuthorityError("MALFORMED_REQUEST", "invalid identity")


def require_production_order_authority(authority: object) -> OrderAuthority:
    """Executable production trust boundary; subclasses and test authorities are denied."""
    if type(authority) is not OrderAuthority:
        raise OrderAuthorityError(
            "TRUSTED_CONTEXT_FAILURE", "exact production OrderAuthority required"
        )
    return authority


class TestOrderAuthority(OrderAuthority):
    """Distinct test trust domain with deterministic fixture-only admission."""

    __test__ = False
    domain = TEST_ORDER_AUTHORITY_DOMAIN
    _test_admission = True
    authenticity_purpose = TEST_ORDER_AUTHENTICITY_PURPOSE

    def _make_custody(self) -> OrderAuthoritySecretCustody:
        return DeterministicTestOrderAuthoritySecretCustody()

    def submit_order(
        self,
        request: Mapping[str, Any],
        *,
        exchange_id: str = "test-exchange",
        recorded_at_utc: str = "2024-01-01T00:00:00Z",
        audit_event_id: str | None = None,
    ) -> Mapping[str, Any]:
        validated = validate_order_command(request)
        if request["operation_type"] != "SUBMIT_ORDER":
            raise OrderAuthorityError("MALFORMED_REQUEST")
        _validate_timestamp(recorded_at_utc, "MALFORMED_REQUEST")
        fingerprint = command_fingerprint_sha256(request)
        scope = _canonical([request[k] for k in CONTRACT["idempotency_contract"]["scope"]])
        with self._lock:
            self._db.execute("BEGIN IMMEDIATE")
            try:
                self._verify_integrity()
                old = self._db.execute(
                    "SELECT command_fingerprint,canonical_outcome FROM command_journal WHERE command_id=?",
                    (request["command_id"],),
                ).fetchone()
                if old:
                    if old[0] != fingerprint:
                        raise OrderAuthorityError("IDEMPOTENCY_CONFLICT")
                    self._db.execute("COMMIT")
                    return _freeze(json.loads(old[1]))
                if audit_event_id is None:
                    audit_event_id = "evt_" + request["command_id"][4:]
                event = self._planned_event(request, exchange_id, recorded_at_utc, audit_event_id)
                outcome = {
                    "command_id": request["command_id"],
                    "operation_type": "SUBMIT_ORDER",
                    "request_fingerprint_sha256": fingerprint,
                    "outcome": "ACCEPTED",
                    "denial_code": None,
                    "order_id": request["order_id"],
                    "accepted_audit_event_id": audit_event_id,
                    "recorded_at_utc": recorded_at_utc,
                }
                validate_command_outcome(outcome)
                self._insert_command(request, fingerprint, scope, outcome)
                self._insert_event(event)
                self._db.execute("COMMIT")
                return _freeze(outcome)
            except Exception:
                self._db.execute("ROLLBACK")
                raise

    def append_event(self, event: Mapping[str, Any]) -> str:
        validate_order_event(event)
        with self._lock:
            self._db.execute("BEGIN IMMEDIATE")
            try:
                self._verify_integrity()
                old = self._db.execute(
                    "SELECT event_fingerprint FROM event_journal WHERE audit_event_id=?",
                    (event["audit_event_id"],),
                ).fetchone()
                if old:
                    if old[0] != event["event_fingerprint_sha256"]:
                        raise OrderAuthorityError("EVENT_IDENTITY_CONFLICT")
                    self._db.execute("COMMIT")
                    return "REPLAY_SUCCESS"
                rows = self._history_rows(event["order_id"])
                current = len(rows)
                if event["event_type"] in ("ORDER_PARTIALLY_FILLED", "ORDER_FILLED"):
                    duplicate = self._fill_duplicate_decision(rows, event)
                    if duplicate == "REPLAY_SUCCESS":
                        self._db.execute("COMMIT")
                        return duplicate
                    if duplicate == "FILL_IDENTITY_CONFLICT":
                        raise OrderAuthorityError(duplicate)
                if event["aggregate_version"] <= current:
                    raise OrderAuthorityError("STALE_EVENT")
                if event["aggregate_version"] > current + 1:
                    raise OrderAuthorityError("EVENT_VERSION_GAP")
                if not rows and event["event_type"] != "ORDER_PLANNED":
                    raise OrderAuthorityError("INVALID_LIFECYCLE_TRANSITION")
                if rows:
                    first = json.loads(rows[0]["canonical_event"])
                    for field in CONTRACT["event_contract"]["envelope_schema"]["scope_fields"]:
                        if event[field] != first[field]:
                            raise OrderAuthorityError("ORDER_SCOPE_MISMATCH")
                    state, pre = self._replay(rows)
                    self._next_state(state, pre, event)
                    if event["event_type"] in ("ORDER_PARTIALLY_FILLED", "ORDER_FILLED"):
                        planned = json.loads(rows[0]["canonical_event"])
                        ordered = Decimal(planned["safe_payload"]["quantity"])
                        previous = Decimal("0")
                        for row in rows:
                            prior = json.loads(row["canonical_event"])
                            if prior["event_type"] in ("ORDER_PARTIALLY_FILLED", "ORDER_FILLED"):
                                previous = Decimal(
                                    prior["safe_payload"]["cumulative_executed_quantity"]
                                )
                        cumulative = Decimal(event["safe_payload"]["cumulative_executed_quantity"])
                        if cumulative <= previous or cumulative > ordered:
                            raise OrderAuthorityError("FILL_PROGRESSION_CONFLICT")
                        if event["event_type"] == "ORDER_PARTIALLY_FILLED" and not (
                            Decimal("0") < cumulative < ordered
                        ):
                            raise OrderAuthorityError("FILL_PROGRESSION_CONFLICT")
                        if event["event_type"] == "ORDER_FILLED" and cumulative != ordered:
                            raise OrderAuthorityError("FILL_PROGRESSION_CONFLICT")
                self._insert_event(event)
                self._db.execute("COMMIT")
                return "ACCEPTED"
            except Exception:
                self._db.execute("ROLLBACK")
                raise

    def _insert_command(
        self, request: Mapping[str, Any], fingerprint: str, scope: str, outcome: Mapping[str, Any]
    ) -> None:
        head = self._db.execute(
            "SELECT command_count,command_head,command_authenticated_head FROM authority_heads WHERE singleton=1"
        ).fetchone()
        seq = head[0] + 1
        values = [
            seq,
            request["command_id"],
            scope,
            _canonical(dict(request)),
            fingerprint,
            _canonical(dict(outcome)),
            head[1],
        ]
        digest = _digest(values)
        auth_fields = [
            seq,
            request["command_id"],
            scope,
            values[3],
            fingerprint,
            values[5],
            head[2],
        ]
        mac = self._authenticated_commitment("COMMAND", auth_fields)
        self._db.execute(
            "INSERT INTO command_journal VALUES(?,?,?,?,?,?,?,?,?,?)",
            (*values, digest, head[2], mac),
        )
        self._db.execute(
            "UPDATE authority_heads SET command_count=?,command_head=?,command_authenticated_head=? WHERE singleton=1",
            (seq, digest, mac),
        )

    def _insert_event(self, event: Mapping[str, Any]) -> None:
        head = self._db.execute(
            "SELECT event_count,event_head,event_authenticated_head FROM authority_heads WHERE singleton=1"
        ).fetchone()
        seq = head[0] + 1
        canonical = _canonical(dict(event))
        values = [
            seq,
            event["audit_event_id"],
            event["order_id"],
            event["aggregate_version"],
            canonical,
            event["event_fingerprint_sha256"],
            head[1],
        ]
        digest = _digest(values)
        auth_fields = [
            seq,
            event["audit_event_id"],
            event["order_id"],
            event["aggregate_version"],
            canonical,
            event["event_fingerprint_sha256"],
            head[2],
        ]
        mac = self._authenticated_commitment("EVENT", auth_fields)
        self._db.execute(
            "INSERT INTO event_journal VALUES(?,?,?,?,?,?,?,?,?,?)", (*values, digest, head[2], mac)
        )
        self._db.execute(
            "UPDATE authority_heads SET event_count=?,event_head=?,event_authenticated_head=? WHERE singleton=1",
            (seq, digest, mac),
        )

    @staticmethod
    def _planned_event(
        request: Mapping[str, Any], exchange_id: str, timestamp: str, audit_event_id: str
    ) -> dict[str, Any]:
        event = {
            "audit_event_id": audit_event_id,
            "event_type": "ORDER_PLANNED",
            "order_id": request["order_id"],
            "aggregate_version": 1,
            "correlation_id": request["correlation_id"],
            "causation_id": request["causation_id"],
            "command_id": request["command_id"],
            "environment": request["environment"],
            "workspace_id": request["workspace_id"],
            "portfolio_id": request["portfolio_id"],
            "exchange_account_id": request["exchange_account_id"],
            "exchange_id": exchange_id,
            "instrument_id": request["instrument_id"],
            "execution_route_id": request["execution_route_id"],
            "occurred_at_utc": timestamp,
            "safe_payload": {
                "side": request["side"],
                "order_type": request["order_type"],
                "quantity": request["quantity"],
            },
            "event_fingerprint_sha256": "",
        }
        event["event_fingerprint_sha256"] = event_fingerprint_sha256(event)
        validate_order_event(event)
        return event
