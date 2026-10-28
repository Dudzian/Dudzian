# Stage 9 / 3A: CI remediation review, 2026-10-28

This report concerns the existing [PR #3215](https://github.com/Dudzian/Dudzian/pull/3215).
It records inspected historical failures separately from new local validation.
Neither a passing unit test nor a merge of the inactive Observer implementation
would supply production qualification or independent contract approval.

## Source and diagnostic scope

The branch incorporates main `52ea173221572364e21263433a2623c77af8ed3a`
(merged PR #3217) through the history-preserving merge `ba2efb31`. The merge had
no conflicts. The platform policy, evaluator, platform workflow and associated
platform tests remain those of main: Windows PRIMARY, Linux SUPPORTED, macOS
DEFERRED. The quality workflow and mypy configuration retain the union of the
Observer and platform scopes; no existing check, threshold or timeout was
removed or relaxed.

The review used the complete logs of every failed job in these runs, together
with the complete run job inventories: 40 jobs in CI, 9 in Quality and Security,
12 in Platform deployment tracks, and 40 in the comparison CI run. The inventory
requests used `per_page=100`, so each inventory fit in one page. The ten failed
job logs, rather than only failure annotations or summary excerpts, were read.

| Run | Relationship | Historical conclusion |
| --- | --- | --- |
| [CI 40661128123](https://github.com/Dudzian/Dudzian/actions/runs/40661128123) | #3215, before remediation | Three failed platform pytest jobs; seven optional/skipped jobs. |
| [Quality and Security 40661127992](https://github.com/Dudzian/Dudzian/actions/runs/40661127992) | #3215, before remediation | Quality ratchet failed; combined coverage was skipped after its dependency failed. |
| [Platform deployment tracks 40661128067](https://github.com/Dudzian/Dudzian/actions/runs/40661128067) | #3215, old platform-policy base | Linux, macOS native readiness and Windows release gate failed on missing evidence. |
| [CI 40660607511](https://github.com/Dudzian/Dudzian/actions/runs/40660607511) | Merged #3217, comparison only | The same DailyTrend failure on three platforms and the same Ubuntu/macOS post-summary timeout pattern. |

No history of #3217 was modified. The comparison establishes that the DailyTrend
defect and timeout pattern predated the Observer changes; it does not establish
that every potential failure on a new HEAD is historical.

## Every failed #3215 job

Times below are UTC on 2026-10-28. Counts are historical log observations, not
results for the remediated HEAD.

| Job / log | First failed operation and historical result | Classification | Correction or required disposition |
| --- | --- | --- | --- |
| [Ubuntu pytest, 122741119799](https://github.com/Dudzian/Dudzian/actions/runs/40661128123/job/122741119799) | DailyTrend nested `pytest.approx` raises `TypeError`. Summary at 10:13:00: 1 failed, 20388 passed, 108 skipped, 1471 deselected. The step then remains alive until the 120-minute timeout at 11:57:47. | `CODE_TEST_FAILURE` and a separate `TIMEOUT` | Repair the numeric expectation and session resource shutdown. A new full hosted run must confirm the complete process exits. |
| [Windows pytest, 122741119918](https://github.com/Dudzian/Dudzian/actions/runs/40661128123/job/122741119918) | Same DailyTrend `TypeError`. 1 failed, 18317 passed, 2145 skipped, 1491 deselected; exits 1 at 10:59:32. No post-summary timeout. | `CODE_TEST_FAILURE` | Numeric expected quantity and two independent tolerance comparisons; production trading/risk behavior unchanged. |
| [macOS pytest, 122741120378](https://github.com/Dudzian/Dudzian/actions/runs/40661128123/job/122741120378) | Same DailyTrend `TypeError`. Summary at 10:28:16: 1 failed, 18369 passed, 2093 skipped, 1491 deselected. Timeout at 11:01:26 after the 45-minute budget. | `CODE_TEST_FAILURE` and a separate `TIMEOUT` | Same assertion repair and POSIX session cleanup; keep shared-code macOS tests required despite deferred native production support. |
| [Quality ratchet, 122724124693](https://github.com/Dudzian/Dudzian/actions/runs/40661127992/job/122724124693) | `test_context_provenance_does_not_retain_unused_contexts` assumes the global weak registry shrinks by exactly one: `assert 0 == (4 - 1)`. 1 failed, 3978 passed, 3 skipped; exits 1 at 09:36:47. | `CODE_TEST_FAILURE` in an order/GC-sensitive test | Verify the lifetime and exact weak registration of one context, preserve authorization of a live context, and reject a forged projection. Production provenance implementation is unchanged. |
| [Linux deployment, 122724125255](https://github.com/Dudzian/Dudzian/actions/runs/40661128067/job/122724125255) | At 09:37:11, readiness evaluation has no current Core, systemd integration, clean-install or reboot/lifecycle/update evidence; exits 1. | `MISSING_CURRENT_EVIDENCE`, `EXPECTED_QUALIFICATION_BLOCK` | Preserve `LINUX_PRODUCTION_READY=NOT_READY`. No physical Dell qualification or synthetic replacement evidence. |
| [macOS native deployment, 122724125216](https://github.com/Dudzian/Dudzian/actions/runs/40661128067/job/122724125216) | At 09:38:22, old-base evaluation lacks Core, launchd, package/permissions/auth/IPC and reboot/lifecycle/update evidence; exits 1 with `MACOS_PRODUCTION_READY=NOT_READY`. | `MISSING_CURRENT_EVIDENCE`, `EXPECTED_QUALIFICATION_BLOCK` under the old policy | Incorporate #3217's DEFERRED policy. The current job runs blocking deployment compatibility tests and does not require or emit native production qualification. Historical evidence is unchanged. |
| [Windows release gate, 122798878606](https://github.com/Dudzian/Dudzian/actions/runs/40661128067/job/122798878606) | At 11:54:02, exactly four lifecycle items have no current evidence: reboot recovery, long-running lifecycle, update restart and failed-update safe behavior. Exits 1 with `WINDOWS_PRODUCTION_READY=NOT_READY`. | `MISSING_CURRENT_EVIDENCE`, `EXPECTED_QUALIFICATION_BLOCK` | Preserve the gate. Same-run Core, SCM and clean-install artifacts were downloaded successfully and consumed; manual Stage-10 qualification was not executed. |

The inspected failed operations provide no confirmed
`ACTUAL_SECURITY_REGRESSION`. No failed historical job was established to be an
independent GitHub runner outage or dependency-download
`CI_INFRASTRUCTURE_FAILURE`. A timeout remains a failure, even if pytest has
already printed a test summary.

## Comparison on merged #3217

| Comparison job | Observed test result | Process result |
| --- | --- | --- |
| [Ubuntu 122724117922](https://github.com/Dudzian/Dudzian/actions/runs/40660607511/job/122724117922) | Same nested-approx DailyTrend failure; 1 failed, 20452 passed, 108 skipped, 1471 deselected. Summary 11:18:39. | Still alive until 13:04:11, 120-minute timeout. |
| [Windows 122724117979](https://github.com/Dudzian/Dudzian/actions/runs/40660607511/job/122724117979) | Same failure; 1 failed, 18381 passed, 2145 skipped, 1491 deselected. | Exits 1 at 11:53:13, without post-summary timeout. |
| [macOS 122724117995](https://github.com/Dudzian/Dudzian/actions/runs/40660607511/job/122724117995) | Same failure; 1 failed, 18433 passed, 2093 skipped, 1491 deselected. Summary 09:26:51. | Still alive until 09:56:31, 45-minute timeout. |

The logs show pytest **9.2.0** in the executing test environment. An earlier
installation pin in a separate preparation step does not describe the final
pytest selected after test dependencies are installed. The deterministic
DailyTrend reproduction and fix were therefore also checked using 9.2.0.

## Corrected tests and shutdown behavior

### DailyTrend quantity after risk adjustment

`tests/test_controller_daily_trend.py` now computes
`expected_qty = (0.03 * total_equity) / 20_120.0` as a number, asserts it is
positive and smaller than the originally requested quantity, and compares both
the filled quantity and ledger quantity against it separately with the existing
`rel=1e-6`. The nested `ApproxScalar` is removed. No risk limit, order-sizing
algorithm, controller implementation or paper-execution implementation changed.

The single test was run first. The subsequent controller/risk/paper selection
passed 44 tests. These are new local checks for the fix, not hosted CI results.

### Production Trust provenance lifecycle

`gc.collect()` may remove several previously unreachable weak keys, so a global
registry count is not a deterministic assertion about one context. A controlled
probe with an unrelated unreachable cycle reproduced the count mismatch while
the disposable context was correctly collected. Source inspection found weak
context keys and snapshots that do not retain their context key.

The revised test deliberately introduces an unrelated cycle, retains a snapshot
value, and tracks the disposable context and its exact weak registry key. After
collection, both weak references must be dead and that exact registration must
be absent. An active context must still satisfy
`require_verified_production_trust_context`, and a copied/forged projection must
still be rejected. This verifies the security property rather than relaxing a
global count comparison. Production `ProductionTrustContext` code was not
modified. Five repeated copies of the complete provenance module under four
xdist workers passed **120 tests** in a new local run.

### Pytest summary followed by a live process

The historical Ubuntu/macOS logs show that pytest printed its summary but did
not return to the runner before the timeout. Windows returned normally with
exit 1. No final thread stacks were captured in those historical runs, so their
logs alone cannot identify every worker that remained alive.

A bounded reproduction using the actual repository session fixtures and a
test-owned `DatabaseManager` background loop printed **1 passed**, then failed
to exit within the parent's five-second observation window. The parent stopped
only its own disposable child. This demonstrates that the shutdown defect also
affects an entirely green test session; it is not dependent on DailyTrend's
assertion failure. The owned loop is non-daemon on POSIX, and Python waits for
non-daemon threads before ordinary `atexit` callbacks. The previous pytest
session finalizer closed it only on Windows.

`tests/conftest.py` now performs the existing bounded database cleanup on every
OS. Its explicit fixture dependency keeps the database alive until dependent
stream/router cleanup has completed, then closes active managers, waits for
aiosqlite workers and shuts down the loop. Cleanup exceptions are logged and
re-raised. Production database code, daemon flags and CI timeout budgets are
unchanged.

`tests/test_pytest_session_shutdown.py` starts the real owned loop in disposable
pytest children and requires actual process termination, exact exit 0 for a
passing child and exact exit 1 for a deliberately failing child. A dependent
component finalizer also starts a loop, exercising teardown order. A separate
negative test ensures cleanup failure is not suppressed. The new three-test
selection passes locally; the native database selection also passed. These
regressions establish a corrected contributing shutdown gap. A new hosted
full-suite run is still needed before claiming all historical timeout causes
are exhausted.

The managed local executor initially denied IPC/socket wakeups, which prevented
`asyncio.call_soon_threadsafe` from waking the owned loop and caused additional
local database timeouts. Repeating with the executor's network permission,
which permits the required local socket wakeups, passed. This local sandbox
restriction is an infrastructure distinction, not evidence of an identical
GitHub runner restriction. No host-process scan, timeout increase, ignored exit
code, disabled test or fake early PASS was used.

## Native evidence remains a separate gate

The Linux job invokes the evaluator without physical evidence. Its missing
Core/systemd/install/lifecycle items are genuine missing current evidence, so
the expected result remains NOT_READY. Supplying fake documents would defeat
the gate and was not attempted.

For Windows, the failed run successfully downloaded Core artifacts for all
three OSes, Windows SCM evidence and Windows clean-install evidence. Their
revision/run-bound aggregation succeeded; the release evaluation passed the
other native items and reported only:

- `WINDOWS_REBOOT_RECOVERY`
- `WINDOWS_LONG_RUNNING_LIFECYCLE`
- `WINDOWS_UPDATE_RESTART`
- `WINDOWS_FAILED_UPDATE_SAFE_BEHAVIOR`

The producing lifecycle path is the explicitly opted-in manual/self-hosted
Stage-10 workflow and its `windows_stage10_live.ps1` / `platform_evidence.py`
producer. Its prepare/reboot/qualify jobs are intentionally skipped on a PR,
and the corresponding artifact download is conditioned on that manual opt-in.
The observed failure therefore does not demonstrate a lost existing artifact,
wrong-SHA binding or broken consumer. No Stage-10 job was triggered, no
qualification receipt was fabricated, and Windows remains NOT_READY.

After incorporating #3217, macOS still runs Core/shared-code and deployment
compatibility checks. Its current deployment job does not demand physical
launchd/package/reboot qualification for Windows/Linux v1. DEFERRED does not
mark shared-code failures successful and does not rewrite historical macOS
results as PASS or NOT_APPLICABLE.

## Commit scope and preserved authorization boundary

The changes are split into the main synchronization, numeric DailyTrend test
repair, deterministic provenance test, POSIX pytest cleanup repair and the
Observer native-shadow correction. The latter is reviewed in the
[final contract review and activation plan](../architecture/stage9_observer_3a_final_review.md).
The PR remains an inactive synthetic Observer contract proposal pending
independent approval and any subsequent separately reviewed activation change.

All Stage 9 qualification flags remain:

```text
Stage 9 = IN_PROGRESS
qualification = BLOCKED
full_confirmation = false
coexistence_authorized = false
preflight_exception = false
physical_test_status = PHYSICAL_TEST_PENDING
production_status = PRODUCTION_NO_GO
runtime_attestation = MISSING
host_pid_namespace = UNVERIFIED
```

No sudo/root, host permissions change, historical installer/payload execution,
physical qualification, process scan, Stage 41B modification, privileged helper,
launcher, Root Proof change, production enrollment, Stage 10, LIVE activation or
change to existing `--evidence` semantics forms part of this remediation.

## New validation of the remediated source

These are new local results on Linux, Python 3.11.16 and pytest 9.2.0, for source
through `4cc16d9c`; only the two final report documents follow that source commit.
They are not new Windows/macOS native qualification or hosted workflow results.

| Check | New result |
| --- | --- |
| Combined relevant pytest selection | **1507 passed, 0 failed, 3 skipped**, 187.93 seconds; the process actually returned exit 0. |
| Observer and four architecture-boundary modules | Included in that run: 869 Observer cases plus the existing 110 boundary cases. |
| DailyTrend/risk/paper | Single formerly failing test passed first; four-file selection passed 44 tests, then was included in the final combined run. |
| Provenance repeated/parallel | Five copies of the full module, four workers: 120 passed; final combined run includes it again. |
| Pytest shutdown/native database | Three subprocess/error-propagation regressions and native database tests included; no post-summary hang in the final run. |
| Platform V2 | Eight-file selection passed 434 tests with two existing skips; included again in the combined run. Policy/evaluator/workflow/platform tests match integrated main. |
| Observer coverage | 994/997 lines = **99.70%**; 229/232 branches = **98.71%**; branch-inclusive display 99%. |
| Combined measured Observer/platform coverage | Branch-inclusive display **97%**; this is the selected two-module scope, not whole-repository coverage. |
| Diff coverage versus `origin/main` | 994/997 changed measured lines = **99.70%**; unchanged `--fail-under=70` passed. Test files are omitted by the existing coverage configuration. |
| Ruff | Exact configured baseline and separate platform gate passed; additional E/F/I on Observer module/test and the new shutdown test passed. |
| Formatting | Exact `python -m pre_commit run ruff-format --all-files` passed; explicit formatter checks for all six changed Python files also passed. |
| mypy | Full configured Linux and `--platform win32`: success in **359 source files each**, using mypy 1.10.0 with installed project dependencies. Win32 is a type-check target on Linux. |
| Semgrep | Five blocking local rules, full configured `bot_core core deployment scripts` scope: **0 findings**, 1085 targets; rule fixtures **5/5 passed**. Existing ignore/large-file rules still apply; this is not a claim that every repository byte was scanned. |
| Betterleaks | Full working-tree offline scan and staged hook: **0 new findings** against the unchanged baseline. |
| Whitespace | `git diff --check` passed. |

The exact three skipped cases are:

- `tests/deployment/test_windows_frozen_production_trust_integration.py::test_installed_frozen_production_trust_lifecycle`: native Windows frozen proof.
- `tests/deployment/test_windows_acceptance.py::test_windows_path_identity_runtime_contract`: requires Windows PowerShell.
- `tests/deployment/test_windows_stage10_artifact_preflight.py::test_operator_supplied_real_public_package_uses_full_verifier`: requires an operator-supplied genuine public Production Trust package; no package was synthesized.

There are **no remaining failed tests in the final local selection**. The full
hosted general suites, native platforms, PostgreSQL qualification shards and
required aggregate checks are not replaced by this selection. New workflow
statuses and the exact published HEAD belong in the PR description, which can
be updated after this report's commit without a self-referential SHA or a
second CI-triggering documentation commit.

Reproduction of the final combined scope (with installed test dependencies and
local IPC permitted):

```sh
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 DUDZIAN_TEST_MODE=1 DUDZIAN_ALLOW_LONG_POLL=0 \
python -m pytest -ra -p asyncio -p pytest_cov.plugin \
  tests/test_stage9_observer_contract.py \
  tests/architecture/test_stage9_independent_cca_boundary.py \
  tests/architecture/test_stage9_external_provisioning_architecture.py \
  tests/architecture/test_stage9_custody_lifecycle_import_isolation.py \
  tests/architecture/test_stage9_atomic_root_proof_commit_boundary.py \
  tests/deployment/test_windows_stage9_production_trust_provenance.py \
  tests/deployment/test_windows_production_trust_install_command.py \
  tests/deployment/test_windows_frozen_production_trust_integration.py \
  tests/test_controller_daily_trend.py tests/test_risk_engine.py \
  tests/test_paper_execution.py tests/test_execution_paper.py \
  tests/test_pytest_session_shutdown.py tests/test_database_manager_native.py \
  tests/deployment/test_platform_readiness.py \
  tests/deployment/test_platform_evidence.py \
  tests/deployment/test_core_test_plan.py \
  tests/deployment/test_windows_acceptance.py \
  tests/deployment/test_windows_stage10_preflight.py \
  tests/deployment/test_windows_stage10_artifact_preflight.py \
  tests/architecture/test_cryptohunter_architecture_baseline_freeze.py \
  tests/architecture/test_cryptohunter_environment_and_product_capabilities.py \
  --cov=deployment.stage9_observer_contract --cov=deployment.platform_readiness \
  --cov-branch --cov-config=.coveragerc --cov-report=term-missing \
  --cov-report=xml:coverage.xml -q
diff-cover coverage.xml --compare-branch origin/main --fail-under=70
```

## Separate readiness decisions

| Decision | Disposition and reason |
| --- | --- |
| `CODE_READY` | **YES for review of the inactive candidate**, subject to required CI and independent code review. Confirmed test defects and import shadowing are fixed; selected local checks pass. This is not an approval of the contract. |
| `CI_READY` | **NOT ESTABLISHED** until current-HEAD required hosted checks finish successfully. Historical red runs remain historical failures; physical-evidence gates can correctly remain blocked. No branch protection is bypassed. |
| `CONTRACT_APPROVED` | **NO**. Independent acceptance of the full profile and exact-target ELF decision remains outstanding. |
| `STAGE_3A_COMPLETE` | **NO**. Independent contract approval and the separately reviewed standard-parser activation remain open. |
| `PRODUCTION_QUALIFIED` | **NO**. Runtime attestation, authenticated host scope and native lifecycle evidence are missing; qualification remains BLOCKED. |

No open verified P0 implementation finding was identified. The schema P1 groups
have complete candidate implementations but open approval gates; exact-target
ELF compatibility is an explicit P1 contract-approval blocker. The concrete
DailyTrend/provenance/shutdown/import findings are corrected within their stated
scope. No additional verified P2/P3 code finding was established. Missing
physical evidence and unfinished CI must not be relabelled as successful tests.
