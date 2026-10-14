"""Explicit UNIT-ONLY lifecycle seam for historical read-only issuer tests.

These tests verify retained Root Proof / signed attempt interpretation after a
locally completed downgrade. The real DB-first authority and close/commit order
are separately exercised by test_postgresql_custody_lifecycle and
test_legacy_local_custody_isolation against PostgreSQL 16.

This deliberately scoped seam NEVER registers a genuine authorization in the
production membership registry and NEVER exercises the legacy mutation API.
It cannot be used to claim PostgreSQL or production lifecycle qualification.
"""

from __future__ import annotations

import uuid
from pathlib import Path
from unittest.mock import patch

from bot_core import (
    local_signing_custody as custody,
    local_signing_lifecycle_evidence as journal,
    postgresql_custody_lifecycle as database_lifecycle,
)
from bot_core.root_proof_issuer_substrate import ProviderRole, SecurityProfileIdentity
from tests.security.test_local_signing_lifecycle_evidence import _descriptor


def simulate_qualified_local_downgrade_for_reader_test(
    directory: Path,
    *,
    security: SecurityProfileIdentity,
    target: custody.SigningKeyLifecycle,
) -> None:
    """Use real journal/fsync/CAS but stub ONLY committed-DB validation in this unit case."""
    role = ProviderRole.ROOT_PROOF_SIGNING
    previous = custody._read_record(custody._record_path(directory, role), security, role)
    operation_id = "reader-test-" + uuid.uuid4().hex
    descriptor = _descriptor(directory, security, previous, operation_id, target)
    token = database_lifecycle.DurableCloseAuthorization(descriptor)

    def verify_authorization(candidate: object, raw: bytes) -> None:
        if (
            candidate is not token
            or type(candidate) is not database_lifecycle.DurableCloseAuthorization
            or type(raw) is not bytes
            or raw != descriptor
        ):
            raise database_lifecycle.CustodyLifecycleConflict(
                "unit-only exact close authorization required"
            )

    with patch.object(
        database_lifecycle, "validate_durable_close_authorization", verify_authorization
    ):
        receipt = journal.durable_transition_local_signing_credential_lifecycle(
            directory,
            security=security,
            role=role,
            credential_identity=previous.credential_identity(),
            expected_lifecycle_generation=previous.lifecycle_generation,
            target=target,
            lifecycle_operation_id=operation_id,
            operation_descriptor=descriptor,
            close_authorization=token,
        )
    assert receipt.metadata.lifecycle_state is target
    assert receipt.metadata.lifecycle_generation == previous.lifecycle_generation + 1
