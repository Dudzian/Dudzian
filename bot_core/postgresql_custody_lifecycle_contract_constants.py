"""Frozen V3 custody-enrollment catalog identifiers for unprivileged readers.

No migration/admin imports are permitted here. The lifecycle schema re-exports
these exact constants to keep SQL generation and live qualification identical.
"""

SQL_CONTRACT_VERSION = 3
ENROLLMENT_MIGRATION_ID = "DURABLE_CUSTODY_V2_SIGNER_ENROLLMENT_V3"
