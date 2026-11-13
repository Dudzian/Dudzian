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


## Version and image update policy

The reviewed container qualification line is PostgreSQL **16**, with patch
releases accepted by the integration contract using the numeric server major.
Production providers retain their own version checks (for example, the
entitlement provider requires PostgreSQL 16 or newer) and all schema, role,
durability and transaction qualification. Passing this CI line does not qualify
another major release or a production deployment.

All PostgreSQL containers in `ci.yml` and `quality-security.yml`, including the
independent durable and `fsync=off` negative controls, use the same explicit patch
and immutable official-image index digest. The current reviewed reference is
`postgres:16.16@sha256:30551eca978d79ccf49d037bc297b5670baea61a04ceca42ff7fd3100163d3ce`.
The index was resolved from `registry-1.docker.io/v2/library/postgres/manifests/16.16`
on 2026-11-13; its Linux amd64 manifest is
`sha256:2d5c77ae9b0e0944ffb92f8d8fc2fa2edb215cb0976da8bac34df917af417e87`.
A mutable major or patch tag alone must not silently advance the fixture.

Update patch and digest together in a reviewed PR after checking official image
metadata and release notes. Re-run the targeted PostgreSQL schema, Root Proof,
History Successor and entitlement qualification, including durability-refusal
and independent-cluster controls, before the wider CI gates. Preserve `fsync`,
`synchronous_commit`, `full_page_writes`, serializable transaction checks,
loopback bindings and the distinct negative-control clusters. Never roll back a
security patch or accept unknown schema metadata merely to obtain a green test.
Native PostgreSQL fixtures remain separate and report their actual server
version; this container pin does not pin native package-manager binaries.
