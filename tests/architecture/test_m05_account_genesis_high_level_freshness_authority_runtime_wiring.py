from __future__ import annotations

import ast
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

import pytest

from bot_core.account_genesis_freshness_authority import (
    FreshnessAuthorityOutcome,
    FreshnessAuthorityResult,
    ProductionLocalFreshnessAuthority,
    ProductionLocalFreshnessAuthorityConfig,
)
import bot_core.account_genesis_freshness_authority as runtime_module
from bot_core.postgresql_freshness_authority import (
    PRODUCTION_LOCAL_REVIEWED_IDENTIFIERS,
    PostgreSQLFreshnessAuthorityProvisioning,
)


MODULE = Path("bot_core/account_genesis_freshness_authority.py")


@pytest.mark.parametrize("field,value", [
    ("schema", "freshness_authority_alt"),
    ("schema_owner_role", "freshness_schema_owner_alt"),
    ("function_owner_role", "freshness_function_owner_alt"),
    ("admin_role", "freshness_admin_alt"),
    ("verifier_role", "freshness_crypto_verifier_alt"),
    ("runtime_role", "freshness_runtime_alt"),
    ("reader_role", "freshness_reader_alt"),
])
def test_production_local_postgresql_identifiers_are_exact_reviewed_defaults(field, value):
    defaults = PostgreSQLFreshnessAuthorityProvisioning()
    values = asdict(defaults)
    assert tuple(values.values()) == PRODUCTION_LOCAL_REVIEWED_IDENTIFIERS
    values[field] = value
    with pytest.raises(ValueError, match="identifiers are frozen"):
        PostgreSQLFreshnessAuthorityProvisioning(**values)


def test_public_boundary_is_one_semantic_operation_and_has_closed_outcomes(tmp_path):
    public = {name for name in dir(ProductionLocalFreshnessAuthority) if not name.startswith("_")}
    assert public == {"authenticate_and_advance"}
    assert {item.value for item in FreshnessAuthorityOutcome} == {
        "CAS_ACCEPTED", "ALREADY_ACCEPTED_EXACT", "CAS_CONFLICT",
        "INVALID_AUTHENTICATION", "INVALID_DOCUMENT", "UNAVAILABLE",
        "PREPARATION_OUTCOME_UNKNOWN", "FRESHNESS_OUTCOME_UNKNOWN", "CORRUPT",
    }
    authority = ProductionLocalFreshnessAuthority(ProductionLocalFreshnessAuthorityConfig(
        str(tmp_path / "absent.sock"), str(tmp_path), 5432,
    ))
    assert authority.authenticate_and_advance(b"{}", b"{}", b"{}").outcome is (
        FreshnessAuthorityOutcome.UNAVAILABLE
    )


def test_configuration_rejects_tcp_relative_and_caller_connection_material():
    with pytest.raises(ValueError):
        ProductionLocalFreshnessAuthorityConfig("relative.sock", "/run/postgresql", 5432)
    with pytest.raises(ValueError):
        ProductionLocalFreshnessAuthorityConfig("/run/verifier.sock", "localhost", 5432)
    fields = set(ProductionLocalFreshnessAuthorityConfig.__dataclass_fields__)
    assert fields == {"verifier_socket_path", "postgres_socket_directory", "postgres_port", "database"}
    forbidden = {"password", "dsn", "host", "user", "role", "provider", "key_id", "principal"}
    assert fields.isdisjoint(forbidden)


def test_source_has_fixed_peer_principal_serializable_cas_and_no_bypass_api():
    source = MODULE.read_text(encoding="utf-8")
    tree = ast.parse(source)
    public_methods = {
        node.name
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and not node.name.startswith("_")
    }
    assert public_methods == {"authenticate_and_advance"}
    assert 'user="freshness_runtime"' in source
    assert 'sslmode="disable"' in source
    assert "SET TRANSACTION ISOLATION LEVEL SERIALIZABLE" in source
    assert "freshness_authority.compare_and_advance" in source
    for forbidden in (
        "SET ROLE", "SET SESSION AUTHORIZATION", "execute_sql", "def sign(",
        "def verify(", "def prepare(", "def cas(", "N+2", "prepare_verified_freshness_candidate",
    ):
        assert forbidden not in source


def test_preparation_unknown_is_sticky_when_exact_retry_is_unavailable(monkeypatch, tmp_path):
    authority = ProductionLocalFreshnessAuthority(ProductionLocalFreshnessAuthorityConfig(
        str(tmp_path / "verifier.sock"), str(tmp_path), 5432,
    ))
    outcomes = iter([
        FreshnessAuthorityResult(FreshnessAuthorityOutcome.PREPARATION_OUTCOME_UNKNOWN),
        FreshnessAuthorityResult(FreshnessAuthorityOutcome.UNAVAILABLE),
    ])
    monkeypatch.setattr(ProductionLocalFreshnessAuthority, "_request_preparation_once",
                        lambda *_: next(outcomes))
    result = authority.authenticate_and_advance(b"p", b"d", b"r")
    assert result.outcome is FreshnessAuthorityOutcome.PREPARATION_OUTCOME_UNKNOWN


def test_preparation_unknown_can_only_be_resolved_by_exact_prepared_retry(monkeypatch, tmp_path):
    authority = ProductionLocalFreshnessAuthority(ProductionLocalFreshnessAuthorityConfig(
        str(tmp_path / "verifier.sock"), str(tmp_path), 5432,
    ))
    prepared = SimpleNamespace(preparation_id="a" * 64)
    outcomes = iter([
        FreshnessAuthorityResult(FreshnessAuthorityOutcome.PREPARATION_OUTCOME_UNKNOWN),
        prepared,
    ])
    monkeypatch.setattr(ProductionLocalFreshnessAuthority, "_request_preparation_once",
                        lambda *_: next(outcomes))
    monkeypatch.setattr(ProductionLocalFreshnessAuthority, "_compare_and_advance", lambda self, value, document, receipt:
                        SimpleNamespace(outcome=FreshnessAuthorityOutcome.CAS_ACCEPTED,
                                        prepared=value, exact=(document, receipt)))
    result = authority.authenticate_and_advance(b"p", b"d", b"r")
    assert result.outcome is FreshnessAuthorityOutcome.CAS_ACCEPTED
    assert result.prepared is prepared and result.exact == (b"d", b"r")


def test_sqlstate_domains_do_not_overlap_and_messages_are_not_parsed():
    runtime = MODULE.read_text(encoding="utf-8")
    database = Path("bot_core/postgresql_freshness_authority.py").read_text(encoding="utf-8")
    verifier = Path("bot_core/freshness_semantic_verifier.py").read_text(encoding="utf-8")
    assert "substitution or consumed preparation' USING ERRCODE='23000'" in database
    assert "lifecycle divergence or inactive' USING ERRCODE='28000'" in database
    assert "predecessor mismatch' USING ERRCODE='{PREDECESSOR_CONFLICT_SQLSTATE}'" in database
    assert "ERRCODE='40001'" not in database
    assert 'exc.sqlstate in {"22023", "28000"}' in verifier
    assert 'exc.sqlstate in {"22023", "23000", "23505"}' in runtime
    assert "exc.sqlstate == PREDECESSOR_CONFLICT_SQLSTATE" in runtime
    assert 'exc.sqlstate == "40001"' in runtime
    assert "str(exc)" not in runtime and "str(exc)" not in verifier


def test_native_serialization_failure_retries_exact_evidence_in_new_attempt(monkeypatch, tmp_path):
    authority = ProductionLocalFreshnessAuthority(ProductionLocalFreshnessAuthorityConfig(
        str(tmp_path / "verifier.sock"), str(tmp_path), 5432,
    ))
    prepared = SimpleNamespace(preparation_id="b" * 64)
    calls = []

    def _attempt(self, value, document, receipt):
        calls.append((value, document, receipt))
        if len(calls) == 1:
            return runtime_module._NativeSerializationFailure()
        return FreshnessAuthorityResult(FreshnessAuthorityOutcome.ALREADY_ACCEPTED_EXACT, 7, receipt)

    monkeypatch.setattr(ProductionLocalFreshnessAuthority, "_request_preparation",
                        lambda *_: prepared)
    monkeypatch.setattr(ProductionLocalFreshnessAuthority, "_compare_and_advance_once", _attempt)
    result = authority.authenticate_and_advance(b"p", b"document", b"receipt")
    assert result.outcome is FreshnessAuthorityOutcome.ALREADY_ACCEPTED_EXACT
    assert calls == [(prepared, b"document", b"receipt")] * 2


def test_bounded_native_serialization_failures_end_outcome_unknown(monkeypatch, tmp_path):
    authority = ProductionLocalFreshnessAuthority(ProductionLocalFreshnessAuthorityConfig(
        str(tmp_path / "verifier.sock"), str(tmp_path), 5432,
    ))
    prepared = SimpleNamespace(preparation_id="c" * 64)
    calls = []
    monkeypatch.setattr(ProductionLocalFreshnessAuthority, "_compare_and_advance_once",
                        lambda *args: calls.append(args[1:]) or
                        runtime_module._NativeSerializationFailure())
    result = authority._compare_and_advance(prepared, b"document", b"receipt")
    assert result.outcome is FreshnessAuthorityOutcome.FRESHNESS_OUTCOME_UNKNOWN
    assert len(calls) == 3
