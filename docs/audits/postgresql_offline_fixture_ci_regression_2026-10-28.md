# PostgreSQL CI: disposable offline fixture session lifecycle

Review baseline: `5f8f5688f41330236f84650949e4571f9173e9a1`, 2026-10-28.
This fix changes a TEST-only helper and regression tests. Production migration,
privileges, qualification states and Stage 9 safety gates are unchanged.
The [Dell ELF report PR #3219](https://github.com/Dudzian/Dudzian/pull/3219)
remains separate.

## Confirmed failure and forensic limit

[Run 40716440677, job 122970984826](https://github.com/Dudzian/Dudzian/actions/runs/40716440677/job/122970984826)
failed at the immediate post-drain activity check in
`test_offline_fixture_drains_stale_test_sessions_without_touching_owner`:
`RuntimeError: disposable TEST database still has active sessions`.
The termination results contained no false row on that path. CI reported
756 passed, 17 skipped, 1 failed and 25469 deselected in 1312.57 seconds.
Its effective checkout was `8771b8ef5705461e6f2daaa378f1334d29818169`,
Ubuntu 24.04.5 / CPython 3.11.17 / psycopg 3.3.6 / PostgreSQL 16.15.

The CI log did not record the remaining PID or backend type. Its particular
backend cannot now be identified. Local controlled reproduction establishes a
real PostgreSQL mechanism yielding the identical exception; it does not prove
that the original CI backend was an autovacuum worker.

## Root cause established by source and real reproduction

The helper assumed that one successful termination pass followed by one empty
activity query would suffice after `ALLOW_CONNECTIONS=false`. Closing that gate
excludes new ordinary client connections, but autovacuum workers explicitly
override it. A real worker may enter after the drain statement sampled activity.
Conversely, a sampled worker may exit naturally before the termination function
checks its PID. The original helper mishandled both lifecycle windows.

PostgreSQL 16.15 source, pinned at
`7d3e000c5961a544302072058a1184e9a588837b`:

- [autovacuum.c](https://github.com/postgres/postgres/blob/7d3e000c5961a544302072058a1184e9a588837b/src/backend/postmaster/autovacuum.c): the worker's `InitPostgres` call ignores `datallowconn`.
- [signalfuncs.c](https://github.com/postgres/postgres/blob/7d3e000c5961a544302072058a1184e9a588837b/src/backend/storage/ipc/signalfuncs.c): nonzero termination timeout waits for process disappearance; a missing PostgreSQL backend reports a warning and false. True is stronger than merely sending SIGTERM.
- [backend_status.c](https://github.com/postgres/postgres/blob/7d3e000c5961a544302072058a1184e9a588837b/src/backend/utils/activity/backend_status.c): activity is cached within a transaction and cleared at commit/abort; `pg_stat_clear_snapshot()` explicitly discards the local snapshot.

The real fixture already uses `autocommit=True`. A separate PostgreSQL control
experiment showed that an explicit transaction retained a terminated PID until
`pg_stat_clear_snapshot()`, whereas the actual autocommit arrangement immediately
observed its disappearance. A stale transaction snapshot was **not** established
as this helper's cause. Explicit snapshot clearing in the fix ensures fresh
sampling; it does not substitute for handling actual live sessions.

The [bounded local reproduction evidence](postgresql_offline_fixture_ci_regression_2026-10-28.json)
records these observations and log digests. It is development test evidence,
not physical, runtime or production qualification. The isolated loopback cluster
used PostgreSQL 16.15, durable settings enabled and development CPython 3.11.16 /
psycopg 3.3.6. Its accelerated autovacuum schedule was confined to that disposable
test cluster; no production service or other user's database was inspected.

Controlled old-helper trace, with a genuine stale client:

```text
first drain: termination=true
migration owner=1471; new PID=1475, backend_type=autovacuum worker
fresh activity: PID1475 active, autovacuum: VACUUM ANALYZE public.probe
RuntimeError: disposable TEST database still has active sessions
finally: datallowconn=true
```

A stress probe exposed the second window:

```text
PGNOTICE: PID 4095 is not a PostgreSQL backend process
termination results: true, false
pg_stat_clear_snapshot(); fresh scoped activity: []
```

An initial bounded-drain revision still treated every false as failure: its
accelerated fresh-database run had **98 passes and 2 failures**. That was not
accepted as validation. A diagnostic rerun reproduced the warning above after
82 passes, proving the already-departed-backend case and prompting the final
PID-specific check.

## Final behavior and safety boundary

The TEST-only helper performs at most **three drain passes**, with a **five-second
retry budget**. Each drain/verification phase sets a server statement timeout
from the remaining budget; the per-backend termination timeout uses that same
remaining value. Deadline expiry, including after an empty activity result,
refuses. This is not a hard five-second wall-clock guarantee for connection
establishment, scheduler delays or finally-block cleanup; those use their own
connection timeout and must report failure rather than manufacture success.
No sleep, unbounded retry or increased timeout is introduced.

Every termination and activity query retains the exact TEST database predicate
and excludes the migration owner's PID. Termination results now identify their
sampled PID. A false result is acceptable only after fresh activity proves that
the same PID is absent. A false result with a still-present PID, a nonboolean
result, a SQL/permission/timeout error, persistent/repeated arrivals or an expired
budget refuses. A different late worker cannot be hidden by an earlier departed
PID; it must be drained within the same bounds. The helper yields only after
the **entire fresh scoped activity set is empty**.

The closed-gate operation is inside `try/finally`, so even a lost acknowledgement
after the server commits the close triggers a restoration attempt. Drain/body
failures also restore `ALLOW_CONNECTIONS=true`. A failed restoration propagates;
it is not recorded as successful cleanup.

This helper remains in the test module. It is not imported by production.
[`migrate_production_root_lifecycle_completion_block_v2`](../../bot_core/postgresql_production_root_lifecycle_qualification.py)
still requires a separately committed closed gate, an independent fresh session
check and every other production precondition. It does not terminate sessions
or modify the gate. A worker arriving after the helper yields can still make
that independent production check refuse. No active session is authorized by
the fixture, and no production precondition is weakened.

## Validation and repeatability

The exact failed real test was run first: PASS. One hundred unforced fresh
baseline fixtures also passed; these did not establish absence of the bug.
The controlled real-worker schedule reproduced the old exception, then passed
with the final helper:

```text
first drain: PID4195=true; late real autovacuum worker=4201
second drain: PID4201=true
yield: migration owner4197 alive; other-database client4196 alive
stale client4195 absent; finally datallowconn=true
```

The final helper passed **100/100 fresh isolated database fixtures in 52.76
seconds** under the accelerated one-second autovacuum schedule. A separate
real-session control captured an exited PID before the actual termination call:
PostgreSQL warned that PID5256 was no longer a backend and returned false;
fresh activity was empty, the owner and other-database connection remained live,
and the gate was restored. Injected false/error responses with a genuinely live
session still refused; continuously visible real sessions exhausted three
passes and refused. These controls do not disable the independent production
session check.

The [unit fault matrix](../../tests/security/test_postgresql_offline_migration_fixture.py)
has **16 passing cases** covering late worker arrival, already-gone false,
different late PID after a false, real-survivor refusal, nonboolean/permission/
timeout errors, three-pass exhaustion, deadline expiry before/after fresh-empty
verification, gate-close acknowledgement loss, cleanup failure, body failure,
owner/other-database exclusion and intentional no-drain negative setup.
The existing real migration negative test remains enabled, including the
requirement to reject an old runtime connection.

Run the focused checks before broader regression:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 DUDZIAN_TEST_MODE=1 DUDZIAN_ALLOW_LONG_POLL=0 \
python -m pytest -p asyncio -p pytest_timeout \
  tests/security/test_postgresql_production_root_lifecycle_qualification.py::test_offline_fixture_drains_stale_test_sessions_without_touching_owner \
  tests/security/test_postgresql_offline_migration_fixture.py -q
```

The final exact failed test plus its unit matrix passed **17 tests in 0.89
seconds**. Broader PostgreSQL 16 regression passed **153 tests, 1 skipped, in
37.46 seconds** across these five modules:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 DUDZIAN_TEST_MODE=1 DUDZIAN_ALLOW_LONG_POLL=0 \
python -m pytest -p asyncio \
  tests/security/test_postgresql_production_root_lifecycle_qualification.py \
  tests/security/test_postgresql_offline_migration_fixture.py \
  tests/security/test_production_root_lifecycle_schema.py \
  tests/security/test_production_root_lifecycle_identity.py \
  tests/security/test_production_root_lifecycle_races.py \
  -q --timeout=90 --maxfail=1 --durations=10
```

For this run `DUDZIAN_TEST_POSTGRES_DSN` and `CRYPTOHUNTER_TEST_POSTGRES_DSN` pointed to the disposable loopback
PostgreSQL 16 cluster; `DUDZIAN_TEST_POSTGRES_RESTART_CONTAINER_ID` explicitly
selected its dedicated task container. Do not reuse a production DSN or restart
controller. The single skip was
`test_non_linux_installation_and_service_entrypoints_fail_closed`, intentionally
inapplicable on Linux. Actual active-session, closed/uncommitted-gate, retained
pin/catalog and durable restart/replay negative tests passed.

Mandatory exact `python -m pre_commit run ruff-format --all-files`, Tier A Ruff,
explicit changed-Python formatting, offline Betterleaks of all four changed
files, local document links, evidence JSON and diff whitespace checks passed.
No production code changed. The delivered PR records exact-HEAD hosted CI;
pending, cancelled or skipped runs are never counted as PASS. No whole-workflow
rerun, manual hardware test or production migration is dispatched by this fix.

## Severity and merge recommendation

| Priority | Disposition |
| --- | --- |
| P0 | No new confirmed production security defect in this bounded review. |
| P1 | Existing Stage 9 independent image/runtime/host qualification blockers remain; this fixture fix does not resolve them. |
| P2 | Confirmed TEST-fixture lifecycle defect causing CI nondeterminism; both reproduced admission/exit windows are addressed. |
| P3 | PID-specific diagnosis, repeatability and evidence attribution improved; original CI backend identity remains unknown. |

Conditional merge is appropriate only after the exact final HEAD's applicable
format, lint, secret and PostgreSQL regression checks pass and the independent
review accepts the scoped change. Preserve expected production-readiness
refusals and classify any remaining CI failure separately. No automatic merge
is performed. Stage 9, Stage 3A, production qualification, Root Proof, LIVE and
Stage 41B states remain unchanged; the ELF report remains an independent PR.
