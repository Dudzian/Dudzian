# CI regression repair after PR #3265

Scope: `Dudzian/Dudzian`, merge
`b15434b0977c8a4b22d1039cd9acfbb7db3c5ee5`; first parent
`fc15b98298c5430286dd2206ec7eeb8f20e2497b`. This is a separate repair PR.
It does not authorize merge, deployment, runtime rebuilding, trust-material
issuance, installation, VM admission or physical Stage 9/10 qualification.

## Causes and evidence

### Stage 10 workflow contract

PR #3265 correctly added an isolation conjunction to the existing manual
`workflow_dispatch && inputs.run_stage10 == true` condition. All three Stage 10
lifecycle jobs still require explicit boolean opt-in. A `stage9-safe` dispatch
excludes the full jobs even if either privileged flag is true; its own preflight
rejects those flags. Evidence dependencies, runner constraints and fail-closed
preflight are unchanged.

The old exact-string assertion omitted this conjunction. The original test
passes at the first parent and fails at the merge. Only the test contract changes;
the production workflow guard is retained. New tests exercise the actual YAML
conditions and dispatch-preflight subprocess with absent, malformed and forbidden
input combinations.

### PostgreSQL checkpoint catalog qualification

The checkpoint qualifier and DDL are identical at the first parent and merge.
The original qualifier has this reproduced matrix against disposable official
containers:

| Source | PostgreSQL 16.15 | PostgreSQL 16.16 |
| --- | --- | --- |
| First parent | PASS | FAIL |
| Merge | PASS | FAIL |

A rolled-back DDL probe compared every actual catalog tuple with
`_REVIEWED_COLUMNS` and `expected_columns`. Table, ordinal, column name, type,
nullability, default, generated/identity fields, collation name/namespace and
determinism, storage, compression, locality, inheritance count and statistics
target all match. The sole differing field is `attoptions`: 16.15 returns `NULL`;
16.16 returns the singleton `pg_attver=<server-assigned WAL position>`.

PostgreSQL 16.16 backpatches attribute versioning for **CVE-2026-6460**. Its
[catalog design](https://github.com/postgres/postgres/blob/REL_16_16/src/backend/catalog/README.attver.md)
documents `pg_attver` as reserved internal metadata which users cannot set using
`ALTER`. Its
[serializer](https://github.com/postgres/postgres/blob/REL_16_16/src/backend/access/common/reloptions.c)
writes canonical unsigned 64-bit decimal; zero removes the option. Legacy or
upgraded columns can retain no option.

The repair explicitly recognizes only `NULL` or exactly one canonical
`pg_attver=1..18446744073709551615` on the reviewed PostgreSQL 16.16+ minor line.
All other column fields are still compared exactly. Unknown/user options,
duplicates, noncanonical/overflow values and this option on unreviewed engine
series fail closed. Nothing is learned automatically from the database or a
mutable manifest. The existing physical fingerprint and all other qualification
checks remain intact. Negative integration tests alter storage, compression,
statistics, user options, nullability, defaults and column inventory while also
updating the stored fingerprint; independent local qualification still refuses.

This is environment drift exposing an existing overly strict metadata assumption,
not a DDL change introduced by #3265.

### PostgreSQL patch assertion and deterministic images

The entitlement test's `version()` substring expected 16.15. It also fails on the
first parent against 16.16. The production provider's existing minimum is 16;
the reviewed container CI line is major 16. The assertion now uses numeric major
16 and keeps `fsync`, `synchronous_commit`, serializable isolation and metadata
checks; it additionally asserts `full_page_writes=on`.

All 18 PostgreSQL image references in the two affected workflows use the same
reviewed patch and immutable official-image index digest:

`postgres:16.16@sha256:30551eca978d79ccf49d037bc297b5670baea61a04ceca42ff7fd3100163d3ce`.

The independent durable and unsafe controls, loopback bindings and production
authentication contracts are unchanged. The
[image policy](../security/ci-postgresql-isolation.md) requires reviewed updates
of both patch and digest with schema/security regression testing.

Historical evidence shows floating `postgres:16` moved from 16.15 index
`f9a7b9ebf6f56201f949db745e47bc423c0f0f8294bafe991cc10bf74b029657` to
16.16 index `63e83b5a3dc36a917dccb76fe1bd2ac51f1ec9c1deb47ff20d4f04233bf6eae9`.
The historical failing index and the new explicit 16.16 index resolve to the same
Linux amd64 manifest
`2d5c77ae9b0e0944ffb92f8d8fc2fa2edb215cb0976da8bac34df917af417e87`.

### Stage 9 dynamic execution

PR #3265 introduced two literal `exec(compile(...))` calls. The security review
found that both already consumed authenticated snapshots, rather than arbitrary
unverified input: the helper/project sources pass the protected root inventory,
revision, exact object graph, SHA-256 and stable nofollow-FD checks; external
runtime imports additionally bind to signed manifest records and checked origins.

The repair replaces the generic execution API with the standard importlib
`SourceLoader` lifecycle over copied module-name/origin/source snapshots.
Overridden `get_code` avoids the inherited filesystem and bytecode paths;
unknown modules, filenames and `.pyc` reads are refused. The helper uses the same
mechanism. Synthetic package paths stay empty. Signature, inventory, identity,
FD, import-hook/cache, package metadata and native runtime checks remain.

The AST graph no longer has a blanket exception for `exec`. No Semgrep rule,
ignore or suppression changes. New tests reject module/path substitution and
bytecode fallback, preserve original source after caller mutation, and forbid
filesystem reopening during snapshot execution. An independent source review
confirmed the standard loader calls the overridden `get_code` and compiles with
`dont_inherit=True`.

## Historical CI evidence

- [First-parent Quality run 43013215029](https://github.com/Dudzian/Dudzian/actions/runs/43013215029): PASS with PostgreSQL 16.15.
- [Merge Quality run 43326333483](https://github.com/Dudzian/Dudzian/actions/runs/43326333483): stale Stage 10/version assertions, two Semgrep calls, and checkpoint qualification failures on 16.16.
- [Merge CI run 43326333581](https://github.com/Dudzian/Dudzian/actions/runs/43326333581): Ubuntu 8283 PASS / 28 SKIP with one failure, the stale Stage 10 assertion.

The merge Quality failures include Atomic Root Proof 174 setup errors,
History Successor 78 setup errors and Semantic Issuer 3 failures/24 setup errors.
Their underlying exception is the same checkpoint catalog mismatch, not separate
Root Proof or semantic-contract regressions.

## Validation and limits

Final executed results are recorded in the repair PR description. Local logs,
JUnit, coverage and command/exit metadata are retained under
`work/ci-regressions/` and are not committed as product artifacts. Historical
failures above are reproductions, not failures of the repaired code.

No tests are skipped/xfail-marked or gates weakened by this repair. Existing
platform-specific skips must be reported explicitly. Local Win32 mypy and
emulated tests do not establish native Windows/macOS qualification. The reviewed
new catalog option is intentionally limited to PostgreSQL 16; another major
requires separate review. Loader tests assume a trusted CPython/stdlib bootstrap,
protected inventory and independently authenticated runtime; they do not qualify
a VM, host or production installation. Existing Stage 9 production/readiness
blocks remain in force.
