"""Frozen executable for the PRODUCTION_LOCAL semantic verifier service.

There is deliberately no command-line configuration surface.  Deployment
coordinates are reviewed constants and PostgreSQL authentication is kernel
peer authentication, never a password or caller-selected role.
"""

from __future__ import annotations

import os
import sys

from bot_core.linux_service_ipc import (
    ReadOnlyVerifierQualifier,
    capture_process_identity,
    load_installed_qualification_policy,
)
from bot_core.linux_service_postgresql import observe_service_connection

from bot_core.freshness_semantic_verifier import (
    FreshnessSemanticVerifier,
    PostgreSQLPreparationWriter,
    PostgreSQLRetainedCredentialResolver,
    PostgreSQLVerifierConnection,
    serve_local_unix_socket,
)

POSTGRES_SOCKET_DIRECTORY = "/run/postgresql"
POSTGRES_PORT = 5432
POSTGRES_DATABASE = "freshness_gate"
VERIFIER_SOCKET_PATH = "/run/cryptohunter/freshness/verifier.sock"
VERIFIER_USER = "os_freshness_crypto_verifier"
RUNTIME_USER = "os_freshness_runtime"
IPC_GROUP = "freshness_verifier_ipc"
AUTHORITY_IDENTITY = "semantic-verifier-v1"


def main() -> None:
    """Build only the reviewed verifier graph and serve its one operation."""
    if sys.platform != "linux":
        raise PermissionError("verifier service requires native Linux")
    import grp
    import pwd

    verifier_uid = pwd.getpwnam(VERIFIER_USER).pw_uid
    runtime_uid = pwd.getpwnam(RUNTIME_USER).pw_uid
    ipc_gid = grp.getgrnam(IPC_GROUP).gr_gid
    if os.geteuid() != verifier_uid or verifier_uid == runtime_uid:
        raise PermissionError("verifier must run as its distinct frozen OS principal")
    connection = PostgreSQLVerifierConnection(
        POSTGRES_SOCKET_DIRECTORY, POSTGRES_PORT, POSTGRES_DATABASE
    )
    # Connect before binding: a broken peer mapping or unavailable substrate
    # must never produce a ready-looking IPC socket.
    with connection.connect():
        pass
    verifier = FreshnessSemanticVerifier(
        PostgreSQLRetainedCredentialResolver(connection),
        PostgreSQLPreparationWriter(connection),
        AUTHORITY_IDENTITY,
    )
    installation_id = load_installed_qualification_policy()
    readonly_qualifier = None
    if installation_id is not None:
        process = capture_process_identity(os.getpid())

        def observe_postgresql(challenge_id: str, installation_binding: str) -> dict:
            # This connection is opened by the actual verifier process through
            # its existing frozen factory. No SQL/role/DSN comes from IPC.
            with connection.connect() as database_connection:
                database_connection.rollback()
                return observe_service_connection(
                    database_connection,
                    challenge_id=challenge_id,
                    installation_id=installation_binding,
                    role="verifier",
                    expected_uid=verifier_uid,
                    expected_gid=process["gid"],
                    expected_os_principal=VERIFIER_USER,
                    expected_database_role="freshness_crypto_verifier",
                    expected_executable_sha256=process["exe_sha256"],
                    expected_cgroup=process["cgroup"],
                )

        readonly_qualifier = ReadOnlyVerifierQualifier(installation_id, observe_postgresql)
    serve_local_unix_socket(
        verifier,
        VERIFIER_SOCKET_PATH,
        allowed_peer_uid=runtime_uid,
        socket_gid=ipc_gid,
        parent_mode=0o750,
        readonly_qualifier=readonly_qualifier,
    )


if __name__ == "__main__":
    main()
