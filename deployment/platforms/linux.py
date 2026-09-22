"""Linux adapter identity; the existing systemd/POSIX qualifier is preserved."""

SYSTEMD_UNIT = "cryptohunter-freshness-verifier.service"
AUTHENTICATION = "PostgreSQL peer plus pg_ident over a protected Unix socket"


def qualify_live() -> None:
    # Lazy import keeps Windows/macOS adapter imports independent of POSIX modules.
    from bot_core.freshness_deployment_qualification import (
        qualify_production_local_deployment,
    )

    qualify_production_local_deployment()

