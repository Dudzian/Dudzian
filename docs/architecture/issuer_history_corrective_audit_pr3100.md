# Corrective audit of issuer history — PR #3100

Target: [PR #3100](https://github.com/Dudzian/Dudzian/pull/3100),
`codex/stage9-durable-issuer-history`. Audit date: 2026-10-11 UTC.
The audit makes two narrow corrections to the existing component and adds
regressions and a manual benchmark. It does not replace the implementation.

## Corrected defects and before/after behavior

### Principal guard construction

`_function_sources` previously constructed the administrator guard with
`runtime_guard.replace(config.runtime_role, config.admin_role)`. For
`hist`, `hist_owner`, `hist_admin`, this also changed the expected owner from
`hist_owner` to `hist_admin_owner`. A directly authenticated, authorized
administrator therefore failed `provision_stream` with SQLSTATE `42501`.

`_principal_guard(*, login_role, schema_owner_role)` now builds the direct-login
and durability checks from their exact identities. `_function_sources` invokes
it independently for runtime and administrator. The required owner remains
unchanged. PostgreSQL version, `fsync`, `synchronous_commit`, direct `session_user`
and SECURITY DEFINER `current_user` checks remain mandatory.

Two real PostgreSQL regressions exercise literal `hist/hist_owner/hist_admin`
and UUID-qualified overlapping names. Both failed with the original function
source and pass with the correction. They prove successful admin provisioning
and runtime reading, denied wrong-role ACLs, and intrinsic principal rejection
even after a temporary EXECUTE grant in an isolated test schema. The fixture
now cleans up only after its atomic installation succeeds; existing roles are
preserved on installation failure.

### Malformed retained byte classification

The SQL canonical validator's UTF-8 conversion could raise `22021` for invalid
UTF-8 or literal NUL. The adapter treated that generic database error as
`HistoryStorageUnavailable`, causing reconciliation to return `UNAVAILABLE`.
The frozen history contract requires malformed retained history to be
`CORRUPT`. Both outcomes denied authority, but the classification was wrong.

Only the `convert_from` operation inside `validate_canonical_json(bytea)` now
catches `character_not_in_repertoire` / `untranslatable_character` and reports
the existing corruption code `23514`. The adapter's unchanged error boundary
therefore raises `HistoryContractError` and reconciliation returns `CORRUPT`.
Other database errors retain their original behavior, including uncertain
writes as `HistoryOperationIndeterminate`.

Two real PostgreSQL regressions corrupt an intermediate record with invalid
UTF-8 or literal NUL. Current head, complete history, verification, reads on
both sides of the bad record, new append and exact replay all reject corruption;
reconciliation returns `CORRUPT` and no successor is published. Valid escaped
NUL, controls, Unicode, large integers and deep canonical payload parity still
pass.

## Exact corrective changes

| File | Functions or scope |
|---|---|
| `bot_core/postgresql_issuer_history_schema.py` | Added `_principal_guard`; changed `_function_sources` to build independent guards |
| `bot_core/postgresql_issuer_history_canonical_sql.py` | `canonical_function_sources`: narrow encoding exception block in `validate_canonical_json(bytea)` |
| `tests/security/test_postgresql_authenticated_issuer_history_integration.py` | `_isolated_history` accepts explicit overlapping roles and guards failed-install cleanup; `test_overlapping_principal_names_preserve_admin_and_runtime_boundaries` has two cases |
| `tests/security/test_postgresql_issuer_history_security.py` | `test_invalid_intermediate_encoding_is_corrupt_for_reads_writes_and_reconciliation`, two encoding cases |
| `scripts/benchmark_issuer_authenticated_history.py` | New manual tool; `_fixture`, `_seed`, `_stats`, `_difference`, `_measure`, `_worker`, `main` and local payload/connection/RSS helpers; no normal CI invocation |
| `docs/architecture/issuer_history_production_local.md` | Link to this audit |
| This document | Findings, measurements, validation, CI and limitations |

The production adapter, oracle, issuer composition, entitlement authority and
signing custody implementation are unchanged by this corrective commit.

## Integration with current main

The branch was created from `570c5b8c3a6687c0bdb93f2af11bcdf840f83649`.
Fetched main is `8ed20bdd89655bc8a62af962f0a6087eea4dc864`.
Main subsequently merged documentation PRs #3097, #3098 and #3099: eight paths
cover Stage 10 runner/MSI handoff, Stage 11/12 sequencing, multi-account beta/UI
gates and Zonda retirement. Those paths are disjoint from the component's diff.

The complete effective PR diff and a non-writing merge-tree were inspected.
There is no integration conflict. The newer main documents survive the combined
tree; behind-main direct comparison differences do not represent PR deletions.
Frozen M0.5 contracts, history oracle, semantic runtime, Stage 9 status and
deployment/readiness implementations are unchanged. No merge or force push is
performed.

## Performance method and measurements

Measurements use PostgreSQL 16.15 on a managed Linux host, Python 3.12.14,
loopback connections, warm buffers, `fsync=on` and `synchronous_commit=on`.
The execution cgroup allows two CPU equivalents and 8 GiB memory. There are
three uninstrumented samples per operation and separate memory probes.
The benchmark runs outside normal fast CI, with other local PostgreSQL tests
and expensive quality commands stopped.

Each retained size uses a fresh child process, UUID schema and separate roles.
Offline test setup COPYs records built by frozen `build_record` and canonical
bytes, atomically updates the matching head, then requires complete runtime
qualification and authoritative verification before timing. This avoids
quadratic fixture construction. It supplies no production cache, checkpoint,
authority shortcut or issuance data. Timed operations use the unchanged adapter.
Only loopback DSNs are accepted; remote `hostaddr` and connection-service
overrides are rejected. Cleanup does not overwrite an existing installation.

Seconds, median of three samples; brackets show the observed minimum/maximum:

| Retained prefix | Authoritative `current_head` | New append | Exact replay |
|---:|---:|---:|---:|
| 100 | 0.245 [0.237–0.259] | 0.897 [0.875–0.955] | 0.273 [0.256–0.278] |
| 1,000 | 3.009 [2.262–3.350] | 8.558 [8.336–9.930] | 2.119 [2.112–2.123] |
| 10,000 | 21.089 [21.073–21.355] | 121.800 [92.590–124.677] | 31.261 [25.402–33.544] |

Separate instrumented memory probes, in MiB (1,048,576 bytes):

| Retained prefix | Traced Python peak, read / append | Worker peak RSS, read / append | Instrumented elapsed, read / append |
|---:|---:|---:|---:|
| 100 | 0.69 / 0.87 | 187.94 / 188.06 | 0.365 / 1.076 s |
| 1,000 | 6.34 / 7.97 | 203.33 / 205.33 | 2.369 / 8.857 s |
| 10,000 | 63.34 / 79.40 | 331.40 / 374.97 | 30.434 / 108.262 s |

The canonical payload is 87/88/89 bytes at the three sizes; the tail record is
730/736/742 bytes. Fixture setup plus three authoritative verification operations
takes 0.870/8.014/66.489 seconds, excluded from latency samples.

Observed database workload:

| Prefix | Record tuples visited, one read / first append | Records relation incl. indexes | Server-only read / shared hit blocks |
|---:|---:|---:|---:|
| 100 | 200 / 603 | 0.41 MiB | 0.216 s / 613 |
| 1,000 | 2,000 / 6,003 | 3.62 MiB | 2.033 s / 1,011 |
| 10,000 | 20,000 / 60,003 | 34.35 MiB | 26.610 s / 5,378 |

The query planner uses sequential scans at the two smaller sizes and index scans
at 10,000; both visit the complete retained prefix. Normal appends insert one
record and update one head. Separate server read EXPLAINs show zero physical
read blocks, temporary blocks or WAL. Global append WAL deltas range from
2,896–2,920 bytes at 100, 2,984–170,128 at 1,000, and 3,152–21,264 at 10,000;
those include background/checkpoint effects and are not per-operation WAL
attribution. The server-only EXPLAIN is taken after the samples at prefix N+4.

Across a 100-fold larger prefix, median read latency grows 85.9-fold and append
latency 135.7-fold. Python traced memory grows approximately linearly. Host-load
averages and sample ranges are retained because the host is shared; these are
local observations with no production SLA. The 10,000-record append takes
roughly 1.5–2.1 minutes and holds the stream lock during repeated validation.
This is a material bottleneck, not acceptable throughput qualification evidence.
Warm read measurements and server-only EXPLAIN indicate the dominant cost is
server canonical-chain validation rather than physical disk reads. Python CPU
for an uninstrumented 10,000-record read is 0.61–0.69 seconds and for append
1.65–2.20 seconds, much smaller than the total wall time.

The smallest proposed optimization is a private SQL multi-member extractor:
validate each raw canonical object once per record and return exact bytea member
slices for reuse inside that same `verify_stream` invocation. Currently each
`canonical_object_member` revalidates the complete object, including repeated
extraction of the same event identity fields. The proposal retains a visit to
every record, exact member inventories, raw canonical-byte equality, both domain
digests, sequence/predecessor/event/head relations, CAS and role/durability/schema
checks. It creates no cross-operation cache or checkpoint. Existing public raw
SQL ports must continue to reject malformed input independently.

This optimization is proposed, not implemented or qualified in this corrective
PR. It requires source-attested private helper/ACL review, the full canonical
negative/parity matrix, intermediate-corruption/race/crash regressions and the
same three-size benchmark. Removing the repeated full-history snapshots or
trusting an unsigned cached head is outside this proposal. The current complete
validation remains in place until equivalence is independently demonstrated.

SQL source inspection confirms four complete `verify_stream` passes and two
Python snapshots for a successful append. An authoritative head read performs
one SQL full-chain verification followed by canonical envelope materialization
and Python verification. Even an exact replay verifies the complete retained
chain. Table reads also include building the complete returned snapshot.

`pg_stat_statements` is unavailable and server configuration is not changed.
The report includes isolated schema table/I/O counters, separately labelled
global database/WAL deltas, and server-only `EXPLAIN (ANALYZE, BUFFERS, WAL)`.
Global deltas can include observer and background work. Python RSS includes
imports and prior worker activity; tracemalloc excludes libpq and PostgreSQL
backend allocation. No backend peak-memory or disk/cold-cache qualification is
claimed. Appends add a few records during samples; their exact prefix sizes are
reported in JSON.

Reproduction, from the repository root against an isolated local PostgreSQL
test server:

```sh
python -m scripts.benchmark_issuer_authenticated_history \
  --sizes 100 1000 10000 --samples 3 \
  --output work/history-benchmark.json
```

The optional DSN is read from `DUDZIAN_BENCHMARK_POSTGRES_DSN` and is not printed
in the report. Reports bind the dirty-working-copy flag and SHA-256 hashes of
the three runtime source files; the base HEAD alone does not identify the
corrected working copy used for measurements.

## Validation and error/durability review

| Scope | Corrective result |
|---|---|
| Original guard source against both new PostgreSQL regressions | Expected 2 failures, reproducing the defect |
| Corrected literal and UUID-overlap PostgreSQL regressions | 2 passed, 1.34 s |
| Full independent PostgreSQL security suite | 34 passed, 12.25 s |
| Complete added CI history gate | 657 passed, 42.38 s; no failures or skips in this selection |
| Frozen-contract / oracle / substrate selection | 806 passed, 1.49 s |
| Configured Linux and Windows mypy | 282 source files each; passed |
| Ruff lint, expanded E/F/B/I and format | Passed |
| Blocking custom Semgrep and rule fixtures | Zero findings; 5/5 fixtures passed |
| Secret scan | Unchanged offline Betterleaks baseline: zero leaks |
| Changed-line coverage | 90% (474 executable changed lines, 47 missing); unchanged 70% requirement |

The complete gate includes 31 adapter unit cases, 32 real PostgreSQL integration
cases, 34 security cases, the existing reference oracle and relevant frozen
contracts. Combined Python branch-aware coverage is 88%; adapter 84%, schema
90%, canonical SQL helper 82%, oracle 90%. These selected scopes overlap and
must not be added together. SQL execution is tested against PostgreSQL; the
coverage percentages describe Python instrumentation.

Reviewed test evidence includes:

- `test_lost_commit_response_is_indeterminate_until_exact_identity_retry`:
  committed-response loss, exact retry and conflicting payload rejection.
- `test_separate_processes_racing_same_predecessor_have_one_cas_winner`:
  actual competing processes and exactly one successor.
- `test_process_kill_at_transaction_cuts_recovers_exactly_committed_history`:
  kill before append, after append before commit, and after commit.
- `test_complete_history_and_attested_head_survive_fresh_processes`:
  restart, retained event identity and verified public head.
- `test_corruption_fails_closed_for_every_authoritative_read_and_append`,
  malformed-encoding regressions and independent non-tail corruption tests:
  prefix corruption blocks reads, new writes and exact replay.
- Live ACL, principal, role, durability, function-source and physical-schema
  mutation tests, including coordinated metadata spoofing.
- Cross-stream, issuer, environment, trust-domain, product and epoch rejection;
  trusted Ed25519 verification rejects modified signatures and head replacement.
- Raw SQL noncanonical JSON with recomputed digests and strict record/head/hex
  envelopes; legitimate canonical Unicode/control/integer values remain valid.

The prior unchanged entitlement/credentials/issuance-authority PostgreSQL
regression selection passed 369 tests in 2794.44 s. It is recorded as prior
evidence and is not repeated for changes confined to the new history component.
No test, assertion, scanner rule, coverage threshold or job timeout is weakened.
Manual hardware, destructive and network qualification remain outside these
existing fast selections. Client process-kill tests do not establish PostgreSQL
server power-loss durability. Windows type checks do not establish native
Windows PostgreSQL or hardware qualification.

## GitHub checks and existing deployment gaps

Initial snapshot at 00:13:51 UTC captured all 81 visible checks. The refresh on
2026-10-11, before publishing the correction, confirms all workflows for
published base head `08c17217` finished: 93 checks, 78 passed, 11 skipped and four
failed. No cancelled checks remain. Skips are existing scheduled, advisory,
release-only or dependency-blocked paths. The updated PR description records
fresh checks for the published corrective head; results for `08c17217` do not
qualify the correction automatically.

| Workflow | Completed result for `08c17217` |
|---|---|
| [Quality and Security](https://github.com/Dudzian/Dudzian/actions/runs/38096417504) | Passed; added history gate 653 passed in 24.26 s; hosted changed-line coverage 90% against 70% |
| [Marketplace Catalog](https://github.com/Dudzian/Dudzian/actions/runs/38096417490) | Passed |
| [CodeQL](https://github.com/Dudzian/Dudzian/actions/runs/38096417541) | Passed Python and JavaScript/TypeScript |
| [PR CI](https://github.com/Dudzian/Dudzian/actions/runs/38096417537) | Passed |
| [Push CI](https://github.com/Dudzian/Dudzian/actions/runs/38096400621) | Failed: Ubuntu fast pytest step exceeded its unchanged 45-minute timeout; dependent Linux integration skipped |
| [Platform deployment tracks](https://github.com/Dudzian/Dudzian/actions/runs/38096417482) | Failed Linux/macOS/Windows evidence gates |

[Linux job 114343127991](https://github.com/Dudzian/Dudzian/actions/runs/38096417482/job/114343127991)
and [macOS job 114343127969](https://github.com/Dudzian/Dudzian/actions/runs/38096417482/job/114343127969)
report `MISSING_CURRENT_EVIDENCE` and `*_PRODUCTION_READY=NOT_READY`.
Their unchanged commands invoke `python -m deployment.platform_readiness
--platform LINUX` / `MACOS` without current revision/run-bound evidence.
Repeating those commands locally with the audited revision, provider and run
reproduces exit 1. The workflow, validator and status blobs are identical at
approved base, PR head and current main. These are existing qualification gaps;
this audit does not manufacture PASS evidence or broaden the PR to fix them.

[Windows release gate 114351995354](https://github.com/Dudzian/Dudzian/actions/runs/38096417482/job/114351995354)
also reports `WINDOWS_PRODUCTION_READY=NOT_READY`: missing reboot recovery,
long-running lifecycle, update restart and failed-update safety evidence. Stage
10 remains intentionally skipped. These are separately gated physical/lifecycle
requirements and are not supplied by this PostgreSQL history change.

[Push Ubuntu job 114343491317](https://github.com/Dudzian/Dudzian/actions/runs/38096400621/job/114343491317)
reports “Run fast pytest suite has timed out after 45 minutes.” The log has no
failed assertion; its last emitted module is `test_probe_keyring.py` at 75%,
and the new history modules ran earlier. The same-head PR Ubuntu/CI run passed.
Code-regression causality is not established. The push integration gate is
incomplete; the timeout is recorded and not extended or hidden.

Accessible branch metadata reports unprotected main, status-check enforcement
off, empty required contexts and no inherited rulesets. The separate protection
administration endpoint returns 403. Project quality gates remain mandatory
under AGENTS.md and this audit request. Pending hosted checks have no successful
final result and are not presented as green.

## Remaining boundaries and recommendation

Root Proof issuance remains disabled, entitlement BIND is unavailable through
this component, PostgreSQL stores no private signing keys, and no authority or
activation bypass is added. Frozen contracts, status flags, the Windows `10/15`
count and Stage 10 destructive gates are unchanged.

Independent durable checkpoint and historical acceptance, complete issuance
reconciliation, qualified signing custody, native deployment/storage evidence
and privileged coordinated rollback protection remain separate requirements.
Changed source-attested validator bodies cause old installed schemas to fail
qualification; this component does not silently upgrade or overwrite them.
Any existing installation needs a separately reviewed offline upgrade plan.

Recommendation: **BLOCKED** for acceptance/merge while the published corrective
head's hosted checks are incomplete, the existing deployment evidence gates are
unresolved, and throughput requirements have not been assessed against the
measured bottleneck. The two confirmed implementation defects are corrected and
their focused verification is complete. No production readiness is asserted and
no automatic merge is performed. The PR remains available for review of these
corrections and the narrowly scoped performance proposal.
