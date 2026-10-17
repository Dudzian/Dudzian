"""Qualify service SQL only across the authenticated native local pipe boundary.

Wire dictionaries are never public receipt inputs. The sole issuer connects to
both independently pinned SCM processes, validates raw fixed-query results while
their SQL sessions remain open, and only then issues an opaque local receipt.
"""

from __future__ import annotations

from contextlib import ExitStack
import copy
import importlib
import os
from pathlib import PureWindowsPath
import re
import secrets
from types import SimpleNamespace
import time
from typing import Any

from deployment import windows_stage9_evidence_pipe as pipe
from deployment import windows_stage9_postgresql_passive as passive
from deployment import windows_stage9_service_sql as service_sql
from deployment.windows_installer.contract import CONTRACT
from deployment.windows_postgresql_auth_contract import RUNTIME_ROLE, VERIFIER_ROLE

_SERVICES = (CONTRACT.backend_service, CONTRACT.verifier_service)
_SQLSTATE = re.compile(r"[0-9A-Z]{5}")


def _fail(message: str) -> passive.PassivePostgreSQLError:
    return passive.PassivePostgreSQLError(message, status="FAIL")


def _require_enrollment(observation: dict[str, Any]) -> None:
    # This committed-enrollment reader belongs to the administrator client.
    # Services use a separate admission boundary and never read its protected
    # ownership records or borrow the qualifier's identification token.
    from deployment import windows_stage9_install_qualification as qualification

    try:
        qualification._confirmed_legal_enrollment(observation)
    except Exception:
        raise passive.PassivePostgreSQLError(
            "canonical accepted and committed enrollment is unavailable"
        ) from None


def _inspect_host(observation: dict[str, Any]) -> dict[str, Any]:
    """Administrator-only host checks; never invoked by either service."""
    if os.name != "nt":
        raise passive.PassivePostgreSQLError("native Windows inspector is unavailable")
    passive._native_x64()
    api = importlib.import_module("win32api")
    registry = importlib.import_module("winreg")
    scm = importlib.import_module("win32service")
    with registry.OpenKey(
        registry.HKEY_LOCAL_MACHINE,
        r"SOFTWARE\Microsoft\Cryptography",
        0,
        registry.KEY_READ | registry.KEY_WOW64_64KEY,
    ) as key:
        guid = registry.QueryValueEx(key, "MachineGuid")[0]
    host = {
        "hostname": api.GetComputerName(),
        "machine_guid": guid,
        "hardware_uuid": passive._hardware_uuid(),
    }
    paths = {
        "program_files": passive._known_folder(0x26),
        "program_data": passive._known_folder(0x23),
    }
    if host != observation.get("host_identity") or any(
        not passive._same_path(value, observation.get("paths", {}).get(key))
        for key, value in paths.items()
    ):
        raise _fail("native host or canonical product paths changed")
    name = CONTRACT.postgresql_service
    manager = scm.OpenSCManager(None, None, scm.SC_MANAGER_CONNECT)
    handle = None
    try:
        handle = scm.OpenService(manager, name, scm.SERVICE_QUERY_CONFIG | scm.SERVICE_QUERY_STATUS)
        config = scm.QueryServiceConfig(handle)
        status = scm.QueryServiceStatusEx(handle)
        expected = str(PureWindowsPath(paths["program_files"]) / "PostgreSQL" / f"{name}.exe")
        if (
            config[0] != 16
            or config[7] != rf"NT SERVICE\{name}"
            or config[3] != f'"{expected}"'
            or status["CurrentState"] != 4
            or type(status["ProcessId"]) is not int
            or status["ProcessId"] <= 0
        ):
            raise passive.PassivePostgreSQLError("canonical running PostgreSQL service unavailable")
        process = passive._process_details(status["ProcessId"])
        rows = [row for row in observation["services"] if row["name"] == name]
        if len(rows) != 1:
            raise _fail("PostgreSQL service observation differs")
        row = rows[0]
        if (
            row.get("account") != config[7]
            or row.get("command_line") != config[3]
            or row.get("state") != "Running"
            or row.get("second_state") != 4
            or row.get("process_id") != process["process_id"]
            or row.get("second_process_id") != process["process_id"]
            or not passive._same_path(process["process_path"], expected)
            or not passive._same_path(row.get("process_path"), expected)
            or abs(
                passive._timestamp(row.get("process_started"))
                - passive._timestamp(process["process_started"])
            )
            > 0.001
        ):
            raise _fail("PostgreSQL SCM process lifetime differs from inspection")
    finally:
        if handle is not None:
            scm.CloseServiceHandle(handle)
        scm.CloseServiceHandle(manager)
    listeners = passive._native_listeners()
    observed = observation.get("listeners")
    if not isinstance(observed, list) or len(observed) != 1 or len(listeners) != 1:
        raise _fail("canonical private PostgreSQL listener is unavailable")
    listener = listeners[0]
    expected_pg = str(PureWindowsPath(paths["program_files"]) / "PostgreSQL/bin/postgres.exe")
    if (
        listener["address"] != CONTRACT.postgresql_host
        or listener["port"] != CONTRACT.postgresql_port
        or listener["parent_process_id"] != process["process_id"]
        or not passive._same_path(listener["process_path"], expected_pg)
        or passive._timestamp(listener["process_started"])
        < passive._timestamp(process["process_started"])
        or any(
            observed[0].get(key) != listener[key]
            for key in ("address", "port", "process_id", "parent_process_id")
        )
        or not passive._same_path(observed[0].get("process_path"), expected_pg)
        or abs(
            passive._timestamp(observed[0].get("process_started"))
            - passive._timestamp(listener["process_started"])
        )
        > 0.001
    ):
        raise _fail("native PostgreSQL listener lifetime differs from inspection")
    return {
        "host_identity": host,
        "paths": paths,
        "listeners": listeners,
        "postgresql_service": process,
    }


class _CapturedSQLError(Exception):
    def __init__(self, sqlstate: str | None) -> None:
        self.sqlstate = sqlstate


def _batch(batch: Any, limit: int) -> list[tuple[Any, ...]]:
    if type(batch) is not dict:
        raise _fail("authenticated SQL batch has an invalid schema")
    if set(batch) == {"sqlstate"}:
        state = batch["sqlstate"]
        if state is not None and (type(state) is not str or not _SQLSTATE.fullmatch(state)):
            raise _fail("authenticated SQLSTATE is malformed")
        raise _CapturedSQLError(state)
    if set(batch) != {"rows"} or type(batch["rows"]) is not list or len(batch["rows"]) > limit:
        raise _fail("authenticated SQL rows exceed the fixed schema")
    if any(type(row) is not list or len(row) > 13 for row in batch["rows"]):
        raise _fail("authenticated SQL row shape differs")
    return [tuple(row) for row in batch["rows"]]


class _CapturedConnection:
    """Private adapter usable only after a native authenticated exchange."""

    def __init__(self, payload: dict[str, Any]) -> None:
        self.info = SimpleNamespace(backend_pid=payload["backend_pid"])
        self.batches = copy.deepcopy(payload["queries"])
        self.current: Any = None

    def cursor(self) -> _CapturedConnection:
        return self

    def __enter__(self) -> _CapturedConnection:
        return self

    def __exit__(self, *_args: Any) -> None:
        return None

    def execute(self, sql: str) -> None:
        name = service_sql.QUERY_NAMES.get(sql)
        if name is None or not self.batches[name]:
            raise _fail("authenticated SQL trace is incomplete")
        self.current = self.batches[name].pop(0)

    def fetchmany(self, limit: int) -> list[tuple[Any, ...]]:
        return _batch(self.current, limit - 1)


def _validate_payload(
    payload: Any, request: dict[str, Any], peer_identity: dict[str, Any], binding: dict[str, Any]
) -> tuple[dict[str, dict[str, str]], dict[str, Any]]:
    if type(payload) is not dict or set(payload) != {
        "version",
        "nonce",
        "collection_id",
        "service",
        "process",
        "backend_pid",
        "queries",
        "probes",
        "post_probe",
    }:
        raise _fail("authenticated service SQL response schema differs")
    if (
        type(payload["version"]) is not int
        or payload["version"] != 1
        or payload["nonce"] != request["nonce"]
        or payload["collection_id"] != request["collection_id"]
        or payload["service"] != peer_identity["name"]
        or type(payload["backend_pid"]) is not int
        or payload["backend_pid"] <= 0
    ):
        raise _fail("authenticated response challenge or service binding differs")
    process = payload["process"]
    if (
        type(process) is not dict
        or set(process)
        != {
            "name",
            "identity",
            "role",
            "certificate_identity",
            "process_id",
            "process_path",
            "process_started",
            "token_sid",
            "paths",
        }
        or any(
            process.get(key) != peer_identity[key]
            for key in ("name", "process_id", "token_sid", "process_path", "process_started")
        )
    ):
        raise _fail("authenticated response process lifetime differs from native peer")
    role = RUNTIME_ROLE if payload["service"] == CONTRACT.backend_service else VERIFIER_ROLE
    if (
        process.get("role") != role
        or process.get("identity") != payload["service"]
        or process.get("certificate_identity") != payload["service"]
    ):
        raise _fail("authenticated service role differs")
    for key in ("program_files", "program_data"):
        if not passive._same_path(process.get("paths", {}).get(key), binding["paths"][key]):
            raise _fail("service canonical host paths differ")
    queries = payload["queries"]
    if (
        type(queries) is not dict
        or set(queries) != set(service_sql.QUERY_COUNTS)
        or any(
            type(queries[name]) is not list or len(queries[name]) != count
            for name, count in service_sql.QUERY_COUNTS.items()
        )
    ):
        raise _fail("authenticated fixed SQL trace differs")
    # Validate every batch, including those skipped by a denied catalog branch.
    for batches in queries.values():
        for batch in batches:
            try:
                _batch(batch, 64)
            except _CapturedSQLError:
                pass
    fixed = {
        **binding,
        "identity": payload["service"],
        "role": role,
        "certificate_identity": payload["service"],
    }
    witness: dict[str, Any] = {}
    components = passive._collect_sql(_CapturedConnection(payload), fixed, witness=witness)
    post = payload["post_probe"]
    if type(post) is not dict or set(post) != {"identity", "ssl", "control", "data"}:
        raise _fail("post-probe SQL session binding is incomplete")
    for name in post:
        try:
            rows = _batch(post[name], 1)
        except _CapturedSQLError as exc:
            outcome = passive._query_error(exc)
            if components["cross_role_authentication"]["status"] != "FAIL":
                components["cross_role_authentication"] = outcome
            # A lost own session cannot establish continuity of any PASS.
            for component, result in components.items():
                if result["status"] == "PASS":
                    components[component] = outcome
            continue
        try:
            first = _batch(queries[name][0], 1)
        except _CapturedSQLError:
            continue
        if rows != first:
            raise _fail("PostgreSQL instance or own SQL session changed during role probes")
    other = VERIFIER_ROLE if role == RUNTIME_ROLE else RUNTIME_ROLE
    probes = payload["probes"]
    if type(probes) is not dict or set(probes) != {other, "postgres"}:
        raise _fail("cross-role authentication matrix is incomplete")
    denial = True
    for probe in probes.values():
        if type(probe) is not dict or set(probe) != {"status", "sqlstate"}:
            raise _fail("cross-role probe schema differs")
        if probe["sqlstate"] is not None and (
            type(probe["sqlstate"]) is not str or not _SQLSTATE.fullmatch(probe["sqlstate"])
        ):
            raise _fail("cross-role probe SQLSTATE is malformed")
        if probe["status"] == "ACCEPTED":
            if probe["sqlstate"] is not None:
                raise _fail("accepted cross-role probe has an incompatible SQLSTATE")
            components["cross_role_authentication"] = passive._result(
                "FAIL", "a service credential authenticated as a forbidden PostgreSQL role"
            )
            return components, witness
        if probe["status"] == "DENIED" and probe["sqlstate"] in {"28000", "28P01"}:
            continue
        if probe["status"] != "BLOCKED":
            raise _fail("cross-role denial classification differs")
        denial = False
    if components["cross_role_authentication"]["status"] == "FAIL":
        return components, witness
    if denial and all(
        components[name]["status"] == "PASS"
        for name in ("effective_configuration", "instance_identity", "service_authentication")
    ):
        components["cross_role_authentication"] = passive._result(
            "PASS",
            "own service credential received explicit SQLSTATE authentication denials for both forbidden roles",
        )
    else:
        components["cross_role_authentication"] = passive._result(
            "BLOCKED", "complete instance-bound authentication denials are unavailable"
        )
    return components, witness


def _merge(
    evidence: list[tuple[dict[str, dict[str, str]], dict[str, Any]]],
) -> dict[str, dict[str, str]]:
    if len(evidence) != 2:
        return {
            name: passive._result("BLOCKED", "both genuine running service identities are required")
            for name in passive.COMPONENTS
        }
    if all(item[0]["instance_identity"]["status"] == "PASS" for item in evidence):
        if not evidence[0][1] or evidence[0][1] != evidence[1][1]:
            raise _fail(
                "service evidence belongs to different PostgreSQL instances or postmaster lifetimes"
            )
    components = {}
    for name in passive.COMPONENTS:
        outcomes = [item[0][name] for item in evidence]
        status = next(
            (
                state
                for state in ("FAIL", "BLOCKED", "UNVERIFIED")
                if any(item["status"] == state for item in outcomes)
            ),
            "PASS",
        )
        components[name] = passive._result(status, "; ".join(item["detail"] for item in outcomes))
    if components["instance_identity"]["status"] != "PASS":
        for name in ("service_authentication", "cross_role_authentication"):
            if components[name]["status"] == "PASS":
                components[name] = passive._result(
                    "BLOCKED",
                    "same PostgreSQL instance is not established for both service identities",
                )
    components["active_hba"] = passive._result(
        "BLOCKED", "parsed HBA and sampled denials do not prove the complete active HBA rule set"
    )
    return components


def collect_service_postgresql(observation: dict[str, Any]) -> Any:
    """Issue fresh local authority only from two native authenticated live peers."""
    receipt = object.__new__(passive._LivePostgreSQLReceipt)
    fingerprint = None
    validator = None
    created = time.monotonic()
    components = passive._blocked_components("authenticated service SQL collection is unavailable")
    evidence: list[tuple[dict[str, dict[str, str]], dict[str, Any]]] = []
    try:
        fingerprint = passive._binding_fingerprint(observation)
        _require_enrollment(observation)
        binding = _inspect_host(observation)
        issued = time.monotonic()
        expires = issued + service_sql.MAX_REQUEST_AGE
        collection_id = secrets.token_hex(32)
        identities = []
        nonces: list[str] = []
        with ExitStack() as stack:
            peers = []
            for name in _SERVICES:
                try:
                    peers.append(stack.enter_context(pipe.connect_service(name, observation)))
                except Exception:
                    raise passive.PassivePostgreSQLError(
                        "authenticated evidence from running service " + name + " is unavailable"
                    ) from None
            for peer in peers:
                request: dict[str, Any] = {
                    "version": 1,
                    "nonce": secrets.token_hex(32),
                    "collection_id": collection_id,
                    "issued": issued,
                    "expires": expires,
                }
                nonces.append(request["nonce"])
                payload = peer.request(request)
                if not issued <= time.monotonic() < expires:
                    raise passive.PassivePostgreSQLError("authenticated service response expired")
                peer.assert_current()
                evidence.append(_validate_payload(payload, request, peer.identity, binding))
                identities.append(copy.deepcopy(peer.identity))
            components = _merge(evidence)
            if _inspect_host(observation) != binding:
                raise _fail("host or PostgreSQL process lifetime changed during collection")
            for peer in peers:
                peer.assert_current()
            # ACK only after all independent checks, requiring successful server
            # cleanup confirmation before any local receipt can carry authority.
            for peer, nonce in zip(peers, nonces):
                peer.finish(nonce)
            for peer in peers:
                peer.assert_current()
            if not issued <= time.monotonic() < expires:
                raise passive.PassivePostgreSQLError(
                    "authenticated service acknowledgement expired"
                )
        comparison = copy.deepcopy({key: observation[key] for key in passive._BINDING_FIELDS})

        def validate() -> None:
            if _inspect_host(comparison) != binding:
                raise _fail("host or PostgreSQL lifetime changed after collection")
            for name, identity in zip(_SERVICES, identities):
                pipe.verify_service(name, comparison, identity)

        validator = validate
        if passive._binding_fingerprint(observation) != fingerprint:
            raise _fail("host observation changed during authenticated collection")
    except Exception as exc:
        outcome = passive._query_error(exc)
        status = (
            "FAIL"
            if any(x["status"] == "FAIL" for x in components.values())
            or any(x["status"] == "FAIL" for item, _witness in evidence for x in item.values())
            else outcome["status"]
        )
        components = passive._blocked_components(outcome["detail"], status)
    passive._RECEIPTS[receipt] = {
        "observation_id": id(observation),
        "fingerprint": fingerprint,
        "created": created,
        "pid": os.getpid(),
        "identity": list(_SERVICES),
        "role": [RUNTIME_ROLE, VERIFIER_ROLE],
        "components": components,
        "validator": validator,
        "source": "live_service_bound_postgresql",
    }
    return receipt
