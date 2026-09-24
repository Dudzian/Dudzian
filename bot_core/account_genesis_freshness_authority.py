"""Fail-closed high-level AccountGenesis FreshnessAuthority for PRODUCTION_LOCAL.

The adapter deliberately exposes one semantic operation.  Its composition is
fixed at construction: AF_UNIX verifier IPC followed by a peer-authenticated,
SERIALIZABLE PostgreSQL compare-and-advance.  It is not a signer, verifier,
preparation API, SQL facade, or predecessor-advancement state machine.

PRODUCTION_LOCAL does not resist host root or a PostgreSQL superuser, provide
whole-host rollback independence or HSM/non-exportable custody, and is not
SERVER_READY.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import base64
import os
import re
import socket
import struct
import time

import psycopg
from psycopg.types.json import Jsonb

from bot_core.postgresql_freshness_authority import (
    PREDECESSOR_CONFLICT_SQLSTATE,
    PREPARATION_FIELDS,
    canonical_json_bytes,
    parse_canonical_json,
)

_MAX_FRAME = 4 * 1024 * 1024
_HEX = re.compile(r"[0-9a-f]{64}\Z")


class FreshnessAuthorityOutcome(str, Enum):
    CAS_ACCEPTED = "CAS_ACCEPTED"
    ALREADY_ACCEPTED_EXACT = "ALREADY_ACCEPTED_EXACT"
    CAS_CONFLICT = "CAS_CONFLICT"
    INVALID_AUTHENTICATION = "INVALID_AUTHENTICATION"
    INVALID_DOCUMENT = "INVALID_DOCUMENT"
    UNAVAILABLE = "UNAVAILABLE"
    PREPARATION_OUTCOME_UNKNOWN = "PREPARATION_OUTCOME_UNKNOWN"
    FRESHNESS_OUTCOME_UNKNOWN = "FRESHNESS_OUTCOME_UNKNOWN"
    CORRUPT = "CORRUPT"


class _PreparationTransportState(str, Enum):
    NOT_CONNECTED = "not_connected"
    CONNECTED = "connected"
    REQUEST_STARTED = "request_started"
    RESPONSE_RECEIVED = "response_received"


@dataclass(frozen=True, slots=True)
class FreshnessAuthorityResult:
    outcome: FreshnessAuthorityOutcome
    decision_sequence: int | None = None
    retained_receipt: bytes | None = None
    preparation_id: str | None = None


@dataclass(frozen=True, slots=True)
class ProductionLocalFreshnessAuthorityConfig:
    verifier_socket_path: str
    postgres_socket_directory: str
    postgres_port: int
    database: str = "freshness_gate"

    def __post_init__(self) -> None:
        if type(self.verifier_socket_path) is not str or not os.path.isabs(
            self.verifier_socket_path
        ):
            raise ValueError("verifier_socket_path must be absolute")
        if type(self.postgres_socket_directory) is not str or not os.path.isabs(
            self.postgres_socket_directory
        ):
            raise ValueError("postgres_socket_directory must be absolute")
        if type(self.postgres_port) is not int or not 1 <= self.postgres_port <= 65535:
            raise ValueError("postgres_port must be an exact port")
        if (
            type(self.database) is not str
            or re.fullmatch(r"[a-z_][a-z0-9_]{0,62}", self.database) is None
        ):
            raise ValueError("database must be a safe fixed identifier")


@dataclass(frozen=True, slots=True)
class _Prepared:
    preparation_id: str
    binding: bytes
    binding_value: dict[str, object]


class _NativeSerializationFailure:
    """Internal marker requiring a fresh exact SERIALIZABLE transaction."""


class ProductionLocalFreshnessAuthority:
    """One fixed verify → durable prepare → SERIALIZABLE CAS operation."""

    def __init__(self, config: ProductionLocalFreshnessAuthorityConfig) -> None:
        if type(config) is not ProductionLocalFreshnessAuthorityConfig:
            raise TypeError("exact production-local configuration required")
        self._config = config

    def authenticate_and_advance(
        self,
        proposer_authentication: bytes,
        authoritative_document: bytes,
        finalization_receipt: bytes,
    ) -> FreshnessAuthorityResult:
        """Authenticate and attempt exactly the supplied frozen candidate.

        Inputs are copied immediately.  This method never creates a successor,
        changes a request identity, or accepts a caller-created preparation.
        """
        if any(
            type(value) is not bytes
            for value in (proposer_authentication, authoritative_document, finalization_receipt)
        ):
            return FreshnessAuthorityResult(FreshnessAuthorityOutcome.INVALID_DOCUMENT)
        proposer = bytes(proposer_authentication)
        document = bytes(authoritative_document)
        receipt = bytes(finalization_receipt)
        prepared_or_result = self._request_preparation(proposer, document, receipt)
        if isinstance(prepared_or_result, FreshnessAuthorityResult):
            return prepared_or_result
        return self._compare_and_advance(prepared_or_result, document, receipt)

    def _request_preparation(
        self, proposer: bytes, document: bytes, receipt: bytes
    ) -> _Prepared | FreshnessAuthorityResult:
        first = self._request_preparation_once(proposer, document, receipt)
        if (
            isinstance(first, FreshnessAuthorityResult)
            and first.outcome is FreshnessAuthorityOutcome.PREPARATION_OUTCOME_UNKNOWN
        ):
            # Only an exact retry is permitted: all three snapshots are reused.
            retry = self._request_preparation_once(proposer, document, receipt)
            if (
                isinstance(retry, FreshnessAuthorityResult)
                and retry.outcome is FreshnessAuthorityOutcome.UNAVAILABLE
            ):
                # Once authority execution may have happened, an outage cannot
                # prove that the durable preparation did not commit.
                return first
            return retry
        return first

    def _request_preparation_once(
        self, proposer: bytes, document: bytes, receipt: bytes
    ) -> _Prepared | FreshnessAuthorityResult:
        if not hasattr(socket, "AF_UNIX"):
            return FreshnessAuthorityResult(FreshnessAuthorityOutcome.UNAVAILABLE)
        request = canonical_json_bytes(
            {
                "proposer_authentication": _b64(proposer),
                "authoritative_document": _b64(document),
                "finalization_receipt": _b64(receipt),
            }
        )
        state = _PreparationTransportState.NOT_CONNECTED
        try:
            with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as client:
                client.connect(self._config.verifier_socket_path)
                state = _PreparationTransportState.CONNECTED
                # Set before sendall: even a partial frame may have reached the
                # peer, so absence of a response is conservatively ambiguous.
                state = _PreparationTransportState.REQUEST_STARTED
                client.sendall(struct.pack("!I", len(request)) + request)
                length = struct.unpack("!I", _receive_exact(client, 4))[0]
                if length < 2 or length > _MAX_FRAME:
                    raise ValueError("invalid verifier response frame")
                raw_response = _receive_exact(client, length)
                state = _PreparationTransportState.RESPONSE_RECEIVED
        except OSError:
            outcome = (
                FreshnessAuthorityOutcome.UNAVAILABLE
                if state
                in {_PreparationTransportState.NOT_CONNECTED, _PreparationTransportState.CONNECTED}
                else FreshnessAuthorityOutcome.PREPARATION_OUTCOME_UNKNOWN
            )
            return FreshnessAuthorityResult(outcome)
        except (ValueError, struct.error):
            return FreshnessAuthorityResult(FreshnessAuthorityOutcome.CORRUPT)
        try:
            response = parse_canonical_json(raw_response)
        except (ValueError, TypeError, UnicodeError):
            return FreshnessAuthorityResult(FreshnessAuthorityOutcome.CORRUPT)
        if type(response) is not dict or type(response.get("outcome")) is not str:
            return FreshnessAuthorityResult(FreshnessAuthorityOutcome.CORRUPT)
        outcome = response["outcome"]
        if outcome == "INVALID" and set(response) == {"outcome"}:
            return FreshnessAuthorityResult(FreshnessAuthorityOutcome.INVALID_AUTHENTICATION)
        if outcome == "UNAVAILABLE" and set(response) == {"outcome"}:
            return FreshnessAuthorityResult(FreshnessAuthorityOutcome.UNAVAILABLE)
        if outcome == "OUTCOME_UNKNOWN" and set(response) == {"outcome"}:
            return FreshnessAuthorityResult(FreshnessAuthorityOutcome.PREPARATION_OUTCOME_UNKNOWN)
        if outcome != "PREPARED" or set(response) != {"outcome", "preparation_id", "binding"}:
            return FreshnessAuthorityResult(FreshnessAuthorityOutcome.CORRUPT)
        preparation_id = response["preparation_id"]
        try:
            binding = _unb64(response["binding"])
            parsed = parse_canonical_json(binding)
        except (TypeError, ValueError, UnicodeError):
            return FreshnessAuthorityResult(FreshnessAuthorityOutcome.CORRUPT)
        if (
            type(preparation_id) is not str
            or _HEX.fullmatch(preparation_id) is None
            or type(parsed) is not dict
            or set(parsed) != PREPARATION_FIELDS
            or parsed.get("preparation_id") != preparation_id
        ):
            return FreshnessAuthorityResult(FreshnessAuthorityOutcome.CORRUPT)
        return _Prepared(preparation_id, binding, parsed)

    def _connect_runtime(self) -> psycopg.Connection[object]:
        connection = psycopg.connect(
            host=self._config.postgres_socket_directory,
            port=self._config.postgres_port,
            dbname=self._config.database,
            user="freshness_runtime",
            sslmode="disable",
        )
        facts = connection.execute(
            "SELECT session_user,current_user,inet_client_addr(),inet_server_addr(),"
            "coalesce((SELECT ssl FROM pg_catalog.pg_stat_ssl "
            "WHERE pid=pg_catalog.pg_backend_pid()),false)"
        ).fetchone()
        if facts != ("freshness_runtime", "freshness_runtime", None, None, False):
            connection.close()
            raise RuntimeError("wrong runtime database principal or transport")
        connection.rollback()
        return connection

    def _compare_and_advance(
        self, prepared: _Prepared, document: bytes, receipt: bytes
    ) -> FreshnessAuthorityResult:
        # A native serialization failure is not semantic evidence of a stale
        # predecessor.  Retry within a small fixed bound using the identical
        # frozen evidence in new connections and SERIALIZABLE transactions.
        for attempt in range(3):
            result = self._compare_and_advance_once(prepared, document, receipt)
            if not isinstance(result, _NativeSerializationFailure):
                return result
            if attempt < 2:
                time.sleep(0.01)
        return FreshnessAuthorityResult(
            FreshnessAuthorityOutcome.FRESHNESS_OUTCOME_UNKNOWN,
            preparation_id=prepared.preparation_id,
        )

    def _compare_and_advance_once(
        self, prepared: _Prepared, document: bytes, receipt: bytes
    ) -> FreshnessAuthorityResult | _NativeSerializationFailure:
        try:
            connection = self._connect_runtime()
        except (psycopg.Error, OSError, RuntimeError):
            return FreshnessAuthorityResult(
                FreshnessAuthorityOutcome.UNAVAILABLE, preparation_id=prepared.preparation_id
            )
        sent = False
        try:
            connection.execute("SET TRANSACTION ISOLATION LEVEL SERIALIZABLE")
            sent = True
            row = connection.execute(
                "SELECT * FROM freshness_authority.compare_and_advance(%s,%s,%s,%s)",
                (prepared.preparation_id, Jsonb(prepared.binding_value), document, receipt),
            ).fetchone()
            connection.commit()
        except psycopg.Error as exc:
            try:
                connection.rollback()
            except psycopg.Error:
                pass
            if exc.sqlstate == PREDECESSOR_CONFLICT_SQLSTATE:
                outcome = FreshnessAuthorityOutcome.CAS_CONFLICT
            elif exc.sqlstate == "40001":
                return _NativeSerializationFailure()
            elif exc.sqlstate == "28000":
                outcome = FreshnessAuthorityOutcome.INVALID_AUTHENTICATION
            elif exc.sqlstate in {"22023", "23000", "23505"}:
                outcome = FreshnessAuthorityOutcome.CORRUPT
            elif isinstance(exc, (psycopg.OperationalError, psycopg.InterfaceError)) and sent:
                outcome = FreshnessAuthorityOutcome.FRESHNESS_OUTCOME_UNKNOWN
            else:
                outcome = FreshnessAuthorityOutcome.UNAVAILABLE
            return FreshnessAuthorityResult(outcome, preparation_id=prepared.preparation_id)
        finally:
            connection.close()
        if (
            row is None
            or len(row) != 3
            or row[0] not in {"CAS_ACCEPTED", "ALREADY_ACCEPTED_EXACT"}
            or type(row[1]) is not int
            or type(row[2]) is not dict
        ):
            return FreshnessAuthorityResult(
                FreshnessAuthorityOutcome.CORRUPT, preparation_id=prepared.preparation_id
            )
        retained = canonical_json_bytes(row[2])
        if retained != receipt:
            return FreshnessAuthorityResult(
                FreshnessAuthorityOutcome.CORRUPT, preparation_id=prepared.preparation_id
            )
        return FreshnessAuthorityResult(
            FreshnessAuthorityOutcome(row[0]), row[1], retained, prepared.preparation_id
        )


def _receive_exact(connection: socket.socket, count: int) -> bytes:
    value = bytearray()
    while len(value) < count:
        part = connection.recv(count - len(value))
        if not part:
            raise OSError("truncated local frame")
        value.extend(part)
    return bytes(value)


def _b64(value: bytes) -> str:
    return base64.urlsafe_b64encode(value).rstrip(b"=").decode("ascii")


def _unb64(value: object) -> bytes:
    if (
        type(value) is not str
        or not value
        or "=" in value
        or re.fullmatch(r"[A-Za-z0-9_-]+", value) is None
    ):
        raise ValueError("invalid base64url")
    result = base64.urlsafe_b64decode(value + "=" * (-len(value) % 4))
    if _b64(result) != value:
        raise ValueError("noncanonical base64url")
    return result
