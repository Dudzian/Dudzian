"""Frozen executable for the PRODUCTION_LOCAL semantic verifier service.

There is deliberately no command-line configuration surface.  Deployment
coordinates are reviewed constants and PostgreSQL authentication is kernel
peer authentication, never a password or caller-selected role.
"""

from __future__ import annotations

import grp
import os
import pwd

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
    serve_local_unix_socket(
        verifier,
        VERIFIER_SOCKET_PATH,
        allowed_peer_uid=runtime_uid,
        socket_gid=ipc_gid,
        parent_mode=0o750,
    )


if __name__ == "__main__":
    main()
