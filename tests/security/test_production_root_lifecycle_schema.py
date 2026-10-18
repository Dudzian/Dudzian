"""Installation admission must fail before any database mutation."""

from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import Mock

import psycopg
import pytest

from bot_core.postgresql_production_root_lifecycle_schema import (
    PostgreSQLProductionRootLifecycleProvisioning,
    install_production_root_lifecycle_authority,
)


def _config() -> PostgreSQLProductionRootLifecycleProvisioning:
    return PostgreSQLProductionRootLifecycleProvisioning(
        "production_lifecycle",
        "production_lifecycle_owner",
        "production_lifecycle_issuer",
        "production_lifecycle_admin",
        "installation-01",
        "production.example",
    )


def _bindings() -> tuple[dict[str, object], ...]:
    return tuple(
        {
            "semantic_role": role,
            "authority_identity": f"authority-{index}",
            "identity": f"credential-{index}",
            "public_material_identity": "sha256:" + str(index) * 64,
            "key_version": 1,
            "lifecycle_generation": 1,
        }
        for index, role in enumerate(
            ("ROOT_PROOF_SIGNING", "HISTORY_ATTESTATION_SIGNING", "TRUST_ACTIVATION"),
            1,
        )
    )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("schema", "caller.schema"),
        ("schema", "production;SET ROLE administrator"),
        ("issuer_role", "production_lifecycle_admin"),
        ("schema_owner_role", "production_lifecycle_issuer"),
        ("installation_id", ""),
        ("trust_domain", "\0production"),
    ],
)
def test_installation_configuration_rejects_alias_or_untrusted_identifier(
    field: str, value: str
) -> None:
    with pytest.raises(ValueError):
        replace(_config(), **{field: value})


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("semantic_role", "TEST_ONLY"),
        ("identity", ""),
        ("public_material_identity", "private-seed"),
        ("key_version", True),
        ("lifecycle_generation", 0),
        ("lifecycle_generation", 2**63 - 1),
        ("private_seed", "caller-secret"),
    ],
)
def test_malformed_public_binding_never_reaches_database(field: str, value: object) -> None:
    connection = Mock(info=SimpleNamespace(transaction_status=psycopg.pq.TransactionStatus.INTRANS))
    bindings = _bindings()
    bindings[0][field] = value
    with pytest.raises(ValueError):
        install_production_root_lifecycle_authority(
            connection, _config(), bindings, bootstrap_operation_id="bootstrap-01"
        )
    connection.execute.assert_not_called()


@pytest.mark.parametrize("field", ["authority_identity", "identity", "public_material_identity"])
def test_cross_role_identity_or_material_alias_rejected_before_database(field: str) -> None:
    connection = Mock(info=SimpleNamespace(transaction_status=psycopg.pq.TransactionStatus.INTRANS))
    bindings = _bindings()
    bindings[1][field] = bindings[0][field]
    with pytest.raises(ValueError, match="independent"):
        install_production_root_lifecycle_authority(
            connection, _config(), bindings, bootstrap_operation_id="bootstrap-01"
        )
    connection.execute.assert_not_called()


@pytest.mark.parametrize(
    "observation",
    [
        ("bootstrap", "other", True, "serializable", 160015),
        ("bootstrap", "bootstrap", False, "serializable", 160015),
        ("bootstrap", "bootstrap", True, "read committed", 160015),
        ("bootstrap", "bootstrap", True, "serializable", 170000),
        ("production_lifecycle_admin", "production_lifecycle_admin", True, "serializable", 160015),
    ],
)
def test_installer_rejects_non_direct_or_wrong_bootstrap_authority(
    observation: tuple[object, ...],
) -> None:
    connection = Mock(info=SimpleNamespace(transaction_status=psycopg.pq.TransactionStatus.INTRANS))
    connection.execute.return_value.fetchone.return_value = observation
    with pytest.raises(ValueError):
        install_production_root_lifecycle_authority(
            connection, _config(), _bindings(), bootstrap_operation_id="bootstrap-01"
        )
    assert connection.execute.call_count == 1
