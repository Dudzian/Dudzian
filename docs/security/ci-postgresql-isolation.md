# PostgreSQL authentication in disposable CI fixtures

The PostgreSQL 16 services in `ci.yml` and `quality-security.yml` are disposable
SQL authorization and durability fixtures on GitHub-hosted Ubuntu runners. Their
published ports bind to `127.0.0.1`, including the separate `fsync=off` negative
control and the independent durable cluster. They contain synthetic identities
and no production keys, credentials, or enrollment.

These services retain host `trust` authentication because the current fixtures
create fresh passwordless `LOGIN` roles and open independent connections with
those roles as `session_user`. For example,
`tests/security/test_postgresql_security_definer_principal_semantics.py` proves
the real caller identity, cross-principal function denial and `SET ROLE` denial;
the authenticated issuer-history and lifecycle-qualification tests use the same
model. Switching only the service to SCRAM would reject those connections before
their SQL authorization assertions. A password-backed alternative would require
explicit credentials for every ephemeral role and corresponding negative tests.
No production authentication conclusion may be drawn from these fixtures.

Loopback publishing restricts access through the runner's host port. It does not
authenticate local runner processes or isolate a service from other containers
on the same Docker network: those peers can reach the container port and claim
any `LOGIN` identity under `trust`. The runner and its job-owned containers are
therefore one trusted test boundary, destroyed with the job. Do not reuse this
configuration on a shared/self-hosted runner, attach an untrusted container to
that network, expose a wildcard host port, or use it for production storage.

The native Linux peer-auth proof remains separate. It creates real OS principals
and its own native cluster in
`tests/security/test_freshness_semantic_verifier_postgresql.py`. Production-local
authentication refusal tests also retain their isolated native clusters. The
container's `trust` configuration does not replace or qualify those checks, and
does not change production deployment authentication or Stage 9 readiness.
