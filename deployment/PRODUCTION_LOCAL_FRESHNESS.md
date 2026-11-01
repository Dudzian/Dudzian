# AccountGenesis FreshnessAuthority — PRODUCTION_LOCAL deployment

This profile has one online topology:

`AccountGenesis runtime (os_freshness_runtime)` →
`/run/cryptohunter/freshness/verifier.sock` →
`semantic verifier (os_freshness_crypto_verifier)` →
`/run/postgresql, freshness_crypto_verifier` → durable preparation; then the
runtime connects independently as `freshness_runtime` for the SERIALIZABLE CAS.

The safe installer now requires an explicit read-only preflight, a separately
approved signed inventory, an existing PostgreSQL 16/main cluster and prepared
OS identities. It publishes only a fresh or identical managed substrate and
does not enable, reload or start services, provision keys, or migrate an existing
deployment. Follow the [operator procedure](../docs/runbooks/STAGE9_LINUX_SAFE_INSTALLER.md)
for prerequisites, refusal states and recovery. Schema, retained-key provisioning
and applying configuration remain separately reviewed offline ceremonies.
Runtime code uses the same frozen socket and database coordinates; there is no
second runtime daemon or duplicated authority state.

The runtime directory is recreated by systemd as verifier-owned,
`freshness_verifier_ipc` group-owned mode `0750`.  The socket is mode `0660`
with the same owner/group.  Both exact principals are group members, but
`SO_PEERCRED` admits only `os_freshness_runtime`.  Startup validates the peer
database session before binding.  Only an exact, verifier-owned stale socket is
removed; symlinks, regular files, foreign sockets, and substituted parents fail
closed.

The isolated systemd command uses the fixed, protected
`deployment/freshness_verifier_entrypoint.py` launcher. It imports the actual
verifier from the fixed installation path without executing the broad package
initializer; `WorkingDirectory` alone does not provide imports under `-I`.

Live qualification remains a separate operator action after independently
approved startup. The qualifier module must be available through a separately
trusted interpreter/package installation; invoking `-I -m` from the copied
working directory alone is insufficient:

```console
cd /usr/lib/cryptohunter
/usr/bin/python3 -I -s -m bot_core.freshness_deployment_qualification
```

It emits only `PASS` or `FAIL CLOSED`. The known reviewed AccountGenesis source
pin drift still blocks full qualification; the installer does not change that
pin. It checks distinct principals and IPC
group membership, exact unit bytes and effective unit properties, live runtime
directory/socket metadata, code-side provenance, active service state,
PostgreSQL >=16 durability and local-only transport settings, exact effective
HBA and ident catalog views, and the complete existing physical/function/schema
qualification.  Unit, package and configuration artifacts are root-owned and
not writable by either online principal.  `-I -s` and `PYTHONNOUSERSITE=1`
exclude caller-controlled Python paths and user packages.

This closes only the deployment substrate.  The frozen flags remain
`classification = UNKNOWN`, `finding_scope = CURRENT_TREE_ONLY`, and
`formal_project_advancement = WITHHELD`; `FreshnessAuthority_implemented`,
`PRODUCTION_LOCAL_RUNTIME_AVAILABLE`, `ROOT_PROOF_ISSUER_IMPLEMENTED`, and
`production_substrate_implemented` remain false pending the separately frozen
FINAL_COMMIT/readiness contract.

## Explicit limitations

This is **not SERVER_READY**. Host root and PostgreSQL superuser are outside the
runtime guarantee. Coordinated whole-host rollback is not resisted. There is no
administrator key-extraction resistance, HSM, or non-exportable key custody.
