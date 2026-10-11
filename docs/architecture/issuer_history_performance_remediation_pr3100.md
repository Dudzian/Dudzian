# Issuer history performance remediation — PR #3100

Date: 2026-10-11 UTC. Target: [PR #3100](https://github.com/Dudzian/Dudzian/pull/3100),
`codex/stage9-durable-issuer-history`. Baseline corrective commit:
`e006182531f67a9a55b4059493129cd446d92cf6`.
The [complete corrective audit](issuer_history_corrective_audit_pr3100.md)
and its measurements remain unchanged. This is a narrow continuation of the
existing component, with no replacement implementation or automatic merge.

## Root cause and change

`verify_stream` repeatedly called `canonical_object_member`, which validated
the entire object before walking its members again. For each retained record
in each SQL verification pass, the old path validated the whole record twice
and its event identity eleven times, including inventory, required-string and
physical-projection checks. Stream, head and credential extraction repeated
similar work. The baseline warm EXPLAIN showed no physical reads or temporary
I/O; Python CPU was small relative to wall time. Server canonical parsing was
the demonstrated bottleneck.

The new private `canonical_object_members(bytea,text[]) -> bytea[]` validates
one exact, sorted inventory and returns the original raw member slices in that
order. It checks UTF-8 once, validates each top-level key and child once through
the existing iterative lexer, and requires the closing delimiter at the exact
end of input. There is no JSON/jsonb conversion of canonical bytes. All slices
are local to one invocation; nothing persists or authorizes a later operation.

`verify_stream` reuses these arrays for stream, record, event, head and credential
checks. The private `validate_stream` returns its seven already-validated
slices instead of void. The fixed mappings were reviewed independently against
all five exact inventories. Whole-record reconstruction still compares every
physical field with the exact canonical bytes; both domain-separated digests,
sequence, predecessor, event identity, stream projection and head relations
remain checked.

| Changed file | Scope |
|---|---|
| `bot_core/postgresql_issuer_history_canonical_sql.py` | New private fused validator/extractor and reviewed return inventory |
| `bot_core/postgresql_issuer_history_schema.py` | Invocation-local reuse in `verify_stream`; private stream return and catalog declaration |
| `tests/security/test_postgresql_issuer_history_security.py` | 44 additional canonical, inventory, ACL, manifest and malicious-chain cases |
| `tests/security/test_postgresql_authenticated_issuer_history_integration.py` | Five historical signed-head and oracle regressions |
| `scripts/benchmark_issuer_authenticated_history.py` | Independent append lock observation after the unchanged primary samples |
| Component documentation and this report | Method, measured evidence, limits and review conclusion |

The production Python adapter, reference oracle, public SQL authority ports,
table/index definitions, principal guards and durability policy are unchanged
from the corrective commit. Both e006 fixes remain: independently constructed
runtime/admin guards, and the narrow UTF-8/literal-NUL exception classification.

## Security equivalence and qualification

Canonical ordering uses decoded bytea keys, preserving Python scalar ordering
and escaped NUL without PostgreSQL text normalization. Empty/duplicate/misordered
keys, alternate escapes, floats, exponents, negative zero, malformed UTF-8,
literal NUL, missing/extra members and trailing bytes still fail closed. Large
integers are scanned without conversion to bounded SQL numbers. Nested values
still use the existing iterative container validator.

Every retained record and retained head is inspected. New append, exact replay
and authoritative reads retain all existing corruption checks. SQL validates
head shape and exact history/credential bindings; cryptographic verification and
lifecycle authority remain with the existing trusted signing provider. This
does not add a SQL claim of signature authority.

Four complete SQL verification passes and two Python snapshots remain in a
successful append. The first snapshot admits the predecessor and replay/CAS
decision. The raw SQL append must independently validate its prefix, and its
post-write check completes candidate bindings that its pre-insert checks do not
fully establish. The final Python snapshot strictly decodes returned envelopes
and requires the exact expected successor. Removing these passes would require
additional equivalent checks and a separate transactional proof. This change
removes repeated parsing within each pass, without removing those decisions.

The stream row lock, exact predecessor CAS, same-identity retry semantics,
direct-login role checks, owner/ACL/physical-schema/source qualification,
`fsync` and synchronous commit remain mandatory. Uncertain writes still become
`HistoryOperationIndeterminate`; the adapter neither retries automatically nor
infers absence from an uncertain response.

The helper is included in the locally reviewed function inventory and installer
source fingerprint manifest. Its bytea[] result and the private stream result
are catalog-qualified. The reviewed ACL grants private helper execution only
to the owner; superusers remain outside the privilege-isolation guarantee.
No new runtime/admin/PUBLIC authority port is granted. New tests reject grant drift
and coordinated helper-body/persisted-manifest tampering against local sources.
Installation remains fresh-only. Old function inventories, sources and return
types fail the updated qualifier; installed production schemas are neither
silently migrated nor overwritten. Existing installations need a separately
reviewed offline upgrade plan.

## Regression and quality results

| Check | Result for this working copy |
|---|---|
| Complete PostgreSQL/history/frozen quality selection | **706 passed in 52.41 s**, no failures or skips |
| Independent PostgreSQL security suite | **78 passed in 21.34 s**, including 44 new cases |
| New historical-head integration cases | **Five passed in 4.41 s** |
| Frozen oracle/substrate/readiness selection | **806 passed in 1.47 s** |
| Ruff baseline, expanded E/F/B/I and formatting | Passed; seven history/benchmark Python files formatted |
| Configured mypy, Linux and win32 | Passed, 282 source files each; focused benchmark file also passed |
| Blocking Semgrep and rule fixtures | Zero findings over 991 Python targets; 5/5 fixture tests passed |
| Changed executable-line coverage | **90%**, 474 changed lines / 45 missing; unchanged 70% requirement |
| Combined Python branch-aware coverage | 89%; adapter 85%, schema 90%, helper 82%, oracle 90% |
| Offline Betterleaks, unchanged baseline | Zero leaks; final working-tree scan passed |

The 706 selection includes 31 adapter unit, 37 real PostgreSQL integration and
78 independent security cases, plus existing contracts/oracle tests. Selections
overlap and are not summed. Coverage describes Python instrumentation; SQL
behavior is exercised against real PostgreSQL.

The 49 new cases cover exact bytes for controls, escaped NUL, Unicode ordering,
nested structures and 500+ digit integers; malformed descendants and selectors;
complete inventories; private helper ACL/source attestation; and malicious
intermediate payloads whose complete chain and matching head hashes have been
recomputed. Those chains still reject reads, new append and exact replay and
reconcile as `CORRUPT`. Existing 700-level nesting and malformed intermediate
UTF-8/literal-NUL regressions also pass.

Historical-head tests compare the existing oracle with independently accepted
checkpoint state, including NUL-bearing stream/provider identities and revoked
historical credentials. Correctly signed but wrongly bound old heads for stream,
sequence, digest or credential block full/tail reads, new append and exact replay;
no successor is published. Existing wrong-signature, credential, role, physical
schema, stale-CAS, conflicting retry, competing-process and transaction-cut kill
tests all remain enabled and pass in the complete gate.

The prior unchanged entitlement/credential/issuance-authority regression of
369 tests in 2794.44 s remains historical evidence; it was not rerun for changes
confined to this component. No assertion, scanner baseline, coverage scope,
threshold, job timeout or frozen contract was weakened. Client-process cuts
do not qualify PostgreSQL server power loss; win32 mypy does not qualify native
Windows PostgreSQL/storage.

## Comparable measurements

Measurements use the existing manual benchmark, Python 3.12.14, PostgreSQL
16.15, loopback connections, warm buffers, `fsync=on` and
`synchronous_commit=on`. Both series used the same managed Linux kernel,
two-CPU-equivalent / 8 GiB cgroup and retained test-server configuration. The
server container was restarted on session resumption and warmed with fresh
fixtures; this is not a cold-buffer or server-restart performance comparison.
NumPy 1.26.4 was pinned before the original baseline and remains unchanged.
Other expensive local tests and scans were stopped during measurements.

Each size uses a fresh worker and UUID-isolated schema/roles. Offline COPY uses
frozen `build_record` and canonical bytes, followed by authoritative runtime
qualification and complete verification before timing. No production data,
cache, checkpoint or authority shortcut is introduced. Three separately timed,
uninstrumented operations use fresh connections; these are consecutive samples
in one isolated fixture, not independent machines. Reads use N records, the
three new appends use N/N+1/N+2 predecessors, and exact replays use N+3. Separate
memory probes read N+3 and append N+4. The fixture has no optional signed-head
rows; its benchmark does not qualify attestation-heavy capacity. EXPLAIN and the final complete read use
N+4, exactly as in the baseline. The new independent lock probe follows them
and appends N+5, so it cannot contaminate primary measurements.

Before latency/memory/EXPLAIN values are the original corrective audit samples,
not a new substitute baseline. Their runtime source hashes match e006 exactly.
The after series was measured at 10:13–10:18 UTC with dirty base HEAD e006;
the [compact evidence](issuer_history_performance_measurements_pr3100.json)
binds actual source hashes, all three samples, prefix sizes, CPU/load, memory,
relation counters, complete EXPLAIN plans and separate lock bounds. The adapter
hash is identical before/after. A base commit alone does not identify a measured
dirty working copy.

Seconds; median with observed minimum–maximum in brackets. Speedup is the ratio
of original median to after median:

| Records | Operation | Before | After | Median speedup |
|---:|---|---:|---:|---:|
| 100 | Authoritative head read | 0.245 [0.237–0.259] | 0.132 [0.086–0.142] | 1.86× |
| 100 | New append | 0.897 [0.875–0.955] | 0.269 [0.237–0.413] | 3.34× |
| 100 | Exact replay | 0.273 [0.256–0.278] | 0.087 [0.083–0.092] | 3.12× |
| 1,000 | Authoritative head read | 3.009 [2.262–3.350] | 0.933 [0.856–1.087] | 3.22× |
| 1,000 | New append | 8.558 [8.336–9.930] | 3.105 [3.044–3.640] | 2.76× |
| 1,000 | Exact replay | 2.119 [2.112–2.123] | 0.734 [0.592–1.090] | 2.89× |
| 10,000 | Authoritative head read | 21.089 [21.073–21.355] | 5.714 [5.610–6.470] | 3.69× |
| 10,000 | New append | 121.800 [92.590–124.677] | 27.990 [21.328–31.376] | 4.35× |
| 10,000 | Exact replay | 31.261 [25.402–33.544] | 5.521 [5.513–6.880] | 5.66× |

All nine original/after sample ranges are disjoint. This demonstrates a material
improvement for this workload, without a production SLA. Shared-host load and
scheduling remain uncontrolled: observed one-minute load before/after ranges
are 1.134–1.146 / 0.702–0.726, 0.902–0.952 / 0.748–0.804, and
0.871–1.000 / 1.011–1.084 at the three sizes. Raw sample ranges are retained;
the factors are local observations rather than universal throughput claims.

Separate instrumented probes, MiB; each cell is read / append:

| Records | Python traced peak before → after | Worker high-water RSS before → after | Instrumented elapsed before → after, seconds |
|---:|---:|---:|---:|
| 100 | 0.69 / 0.87 → 0.69 / 0.87 | 187.94 / 188.06 → 170.19 / 170.31 | 0.365 / 1.076 → 0.123 / 0.310 |
| 1,000 | 6.34 / 7.97 → 6.34 / 7.97 | 203.33 / 205.33 → 183.70 / 186.70 | 2.369 / 8.857 → 0.858 / 4.162 |
| 10,000 | 63.34 / 79.40 → 63.34 / 79.40 | 331.40 / 374.97 → 305.21 / 354.37 | 30.434 / 108.262 → 8.163 / 26.653 |

Traced Python memory is effectively unchanged and grows with the full prefix.
The lower observed process RSS is not attributed to SQL extraction: RSS includes
imports, allocator/high-water history and fixture activity, and the series ran
in separate processes/sessions. Tracemalloc excludes libpq and PostgreSQL
backend allocation; backend peak memory was not measured. At 10,000 records,
after uninstrumented Python user+system CPU is 0.706–0.823 s for read and
1.398–1.719 s for append, still much smaller than wall time. The original
reported CPU remains in the evidence; this is not a claim of constant Python CPU.

SQL evidence, measured after the comparable samples at N+4:

| Records | Record tuples visited, read / first append, both series | Records relation including indexes, both | Server EXPLAIN read before → after | Shared hits before → after |
|---:|---:|---:|---:|---:|
| 100 | 200 / 603 | 0.41 MiB | 215.839 → 53.239 ms | 613 → 627 |
| 1,000 | 2,000 / 6,003 | 3.62 MiB | 2032.885 → 488.983 ms | 1,011 → 1,023 |
| 10,000 | 20,000 / 60,003 | 34.35 MiB | 26610.165 → 4813.048 ms | 5,378 → 5,389 |

Server-only read improvements are 4.05× / 4.16× / 5.53×. All plans report zero
physical read blocks, temporary blocks and WAL. Both versions use sequential
scans for the two smaller fixtures and index scans at 10,000. Read visits remain
2N; successful append visits remain 6N+3, including full snapshot materialization.
Source review, separately labelled from runtime counters, confirms one SQL
full-chain pass / one Python snapshot for read and exact replay and four / two
for new append. Relation counters do not directly count function invocations.

Canonical payload sizes remain 87/88/89 bytes and tail records 730/736/742 bytes.
Fixture setup plus three complete authoritative verifications decreases from
0.870/8.014/66.489 to 0.307/2.253/19.947 s, excluded from primary timings.
Normal append still inserts one record and updates one head. After global append
WAL deltas are 2,896–2,920 / 2,928–2,952 / 3,016–3,096 bytes, compared with
2,896–2,920 / 2,984–170,128 / 3,152–21,264 before. Global counters can include
observer, checkpoint and background work; the lower noisy ranges are not
per-operation WAL savings. `pg_stat_statements` remains unavailable, and no
server configuration was changed to collect telemetry.

### Append lock duration

Each version has one additional uncontended N+4 → N+5 append. The current
benchmark observer is loaded with the exact detached e006 runtime for the
baseline probe. All three imported runtime file paths/hashes, reviewed old
installer/manifest, server settings, Python/platform/cgroup metadata and fully
verified prefix are checked. The baseline seeds all N+4 events offline and
performs three warm authoritative verifications; the after fixture produced its
last four events through the three primary appends and separate memory append.
Both use the same frozen event/payload builders. These are fresh supplemental
baseline lock measurements; the original latency samples are unchanged.

An independent bootstrap connection polls `pg_locks` / `pg_stat_activity` for
the exact streams relation OID, runtime role, unique application name and one
backend PID. It bounds acquisition/release of the granted `RowShareLock` marker
using query-start/end uncertainty. That relation lock starts before the row
lock is acquired and lasts through transaction end. PostgreSQL does not expose
every uncontended row lock directly; these are transaction-marker bounds,
not exact row-lock acquisition/wait/held telemetry. The nominal poll interval
is 10 ms, and observer queries can perturb this separate probe.

| Initial records | Retained probe prefix | Before marker lower–upper | After marker lower–upper | Separate append wall before → after |
|---:|---:|---:|---:|---:|
| 100 | 104 | 0.915932–0.939021 s | 0.229049–0.252417 s | 0.940907 → 0.254141 s |
| 1,000 | 1,004 | 10.687827–10.719418 s | 2.227085–2.253478 s | 10.745532 → 2.266502 s |
| 10,000 | 10,004 | 101.592842–101.616013 s | 21.716390–21.759671 s | 101.627651 → 21.792625 s |

Both probes require one exact committed successor and a separate complete
authoritative read afterward. At 10,000, baseline/after have 9,147/1,934 positive
polls and maximum observation-query gaps of 358.363/90.279 ms. Full bounds,
poll counts, backend IDs and final verification are in the evidence. These are
single probe intervals, not a three-sample lock distribution or median speedup.
The approximately 21.7 s marker after optimization remains a contention cost.

Reproduction against an isolated local PostgreSQL 16 test server, from the
repository root; the optional DSN is taken from
`DUDZIAN_BENCHMARK_POSTGRES_DSN` and is not printed:

```sh
python -m scripts.benchmark_issuer_authenticated_history \
  --sizes 100 1000 10000 --samples 3 \
  --output work/history-benchmark.json
```

To reproduce baseline lock evidence or rerun its complete latency series, use
a separate detached e006 worktree with the current manual benchmark script
copied into its `scripts` directory. Keep all three runtime files at e006 and
verify their hashes against the evidence. This exercises the old reviewed
installer/manifest in fresh test schemas; it does not upgrade any installation.
Our supplemental baseline run measures only the separate lock probe after
seeding N+4 and requiring three authoritative verification operations; it leaves
the original three latency samples intact.

## Branch integration and hosted CI

The approved base is `570c5b8c3a6687c0bdb93f2af11bcdf840f83649`.
Latest fetched main is `8ed20bdd89655bc8a62af962f0a6087eea4dc864`.
Its eight additional documentation paths are disjoint from the effective PR
diff. A non-writing merge-tree is clean and retains those main documents.
Thirty-three frozen-contract, oracle, runtime and deployment/readiness paths
were verified unchanged. No merge or force push is performed.

Prepublication snapshot at **10:28:25 UTC**, exact published corrective head
e006: all 86 checks fetched, **60 success / six skipped / two failure /
18 running**. No queued, cancelled or timed-out conclusion was present.
No code-scope failure was observed. This snapshot qualifies neither the local
performance working copy nor its subsequently published commit automatically.

| Workflow for e006 | Snapshot result |
|---|---|
| [Quality and Security](https://github.com/Dudzian/Dudzian/actions/runs/38130003511) | Running; Ruff, formatting, Linux/win32 mypy, secrets and property tests passed; Account INITIAL_BINDING running, history serial gate still pending |
| [CodeQL](https://github.com/Dudzian/Dudzian/actions/runs/38130003501) | Passed |
| [Marketplace Catalog](https://github.com/Dudzian/Dudzian/actions/runs/38130003507) | Passed |
| [PR CI](https://github.com/Dudzian/Dudzian/actions/runs/38130003504) | Running |
| [Push CI](https://github.com/Dudzian/Dudzian/actions/runs/38130001058) | Running |
| [Platform deployment tracks](https://github.com/Dudzian/Dudzian/actions/runs/38130003551) | Running; two evidence jobs failed |

The only observed failures are
[Linux job 114442185588](https://github.com/Dudzian/Dudzian/actions/runs/38130003551/job/114442185588)
and [macOS job 114442185629](https://github.com/Dudzian/Dudzian/actions/runs/38130003551/job/114442185629),
with `MISSING_CURRENT_EVIDENCE` and production-ready `NOT_READY`. Their workflow,
validators and status sources are unchanged. Windows native physical/lifecycle
evidence remains separately required; a running/skipped check is not readiness
evidence. The older original-head push timeout is preserved in the corrective
audit and is not presented as a performance-head failure.

The performance commit triggers fresh hosted checks. Their exact-head status
and links are recorded in the updated [PR description](https://github.com/Dudzian/Dudzian/pull/3100);
pending checks are not reported as successful. No timeout or deployment gate is
changed to hide these existing gaps. Main has no required status contexts in
accessible branch metadata, but AGENTS.md and the requested quality/evidence
gates remain mandatory.

## Remaining costs, limits and recommendation

Full-prefix SQL scans, four successful-append checks, two complete Python
snapshots, canonical byte materialization, hashing, transport and Python
decoding remain linear in retained history. Memory is still proportional to the
complete returned prefix. Building an entire stream through successive normal
appends therefore remains cumulatively quadratic in retained record count.
This is the intended integrity boundary, not a trusted
cache or a partial-history shortcut. Historical-attestation-heavy workloads,
large payloads, multiple contending streams, cold buffers, remote latency and
native deployment storage remain unmeasured by this small warm fixture.

Production signing custody, independent durable checkpoint/historical acceptance,
complete issuance reconciliation and privileged coordinated rollback protection
remain separate requirements. PostgreSQL receives no private signing keys.
RootProof issuance and entitlement BIND remain inactive. Windows remains
**10/15**; readiness/status flags and destructive Stage 10 gates are unchanged.
No hardware, reboot or destructive qualification was executed.

Conclusion: **BLOCKED** for acceptance/merge until the performance head's hosted
checks complete and the existing deployment evidence gates are resolved. The
local performance remediation and independent integrity regressions are complete;
the measured improvement is reported without inventing a production SLA or
promoting readiness. PR #3100 remains available for review; it is not merged.
