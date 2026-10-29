# Platform CI evidence remediation — 2026-10-29

Base: `main` **3946695a8c7a37b09e9c494bb07caf68ccbbf81f**. Related PR:
[#3222](https://github.com/Dudzian/Dudzian/pull/3222), audited source HEAD
`670147fdaa1dc1061fe7b7a0119f1ee7de5277b1`.
The remote main and open PRs were checked before editing: main was still this
base; the returned open PR was #3222, whose documentation-only changes do not
overlap the remediation paths. Work uses a separate branch/worktree/PR.
The new PR, immutable remediation HEAD, raw diff and exact-HEAD Actions snapshot
are published in its PR description and delivery report after commit creation.

The confirmed Linux defect is missing transfer of already-produced Core
evidence. The Windows readiness failure is a correct refusal: the PR ran no
manual Stage 10, and two required update mechanisms remain unimplemented.
Separate producer schema/identity and final-snapshot defects found by source
review are repaired without manufacturing qualification evidence.

Codebase Memory MCP/CLI is unavailable. Conclusions use actual full job logs,
targeted repository search and direct source reads; no exhaustive graph/index
coverage claim is made. No physical, destructive, reboot or 24-hour test was
run. Synthetic tests below do not qualify an installation or authorize LIVE.

## Full-log comparison and failure classification

Full decoded logs of both failed jobs in each of three runs were inspected,
including checkout, setup, artifact download, aggregation and refusal output.

| Run / PR | Source HEAD | Actual PR merge checkout | Linux job | Windows job |
| --- | --- | --- | --- | --- |
| [40792867502 / #3222](https://github.com/Dudzian/Dudzian/actions/runs/40792867502) | `670147fdaa1dc1061fe7b7a0119f1ee7de5277b1` | `45c90a1fcb650083a2db8f9369b6bb444f6cbe95` | `123182093043` | `123203598434` |
| [40759804926 / #3221](https://github.com/Dudzian/Dudzian/actions/runs/40759804926) | `42d14dbdfa155d3b82a80d2db09df6e71ef47610` | `242df74f96a9601af865e41fdcae537ae106b53b` | `123068337689` | `123152935779` |
| [40716454800 / #3218](https://github.com/Dudzian/Dudzian/actions/runs/40716454800) | `8771b8ef5705461e6f2daaa378f1334d29818169` | `16472edaa9f4f9f55a038efa04c5b695ef7e64d5` | `122915369758` | `123012816475` |

`GITHUB_SHA` denotes the actually checked-out merge revision on these PR runs.
Markers must match that execution revision, not be rewritten to the branch
HEAD. Each run API response returned twelve jobs from the latest attempt's
first page; this is not a claim about every attempt or all repository checks.
Artifact archives were not independently downloaded/replayed in this audit.

| Category | Observed cause and disposition |
| --- | --- |
| Code regression in #3222 | Not established. The relevant workflow/producers/consumer/live script are unchanged by that documentation PR. The immediately previous run has the same failure signatures. |
| Evidence transfer defect | Confirmed, pre-existing Linux bug. The job had no Core dependencies, downloads, aggregation or `--evidence`; readiness received an empty evidence list even though all three Core jobs succeeded in the audited and immediately preceding run. Fixed here. |
| Stale/substituted artifact | Not the observed current cause. Linux supplied none; current/previous Windows downloads and aggregation succeeded. Invalid revision/run/provider/plan/OS/schema cases are now regression-tested. Same-run artifact reuse after failed jobs is separately refused by a current job-status guard. |
| Incomplete artifact set | Older Windows job failed during aggregation: `evidence\\core-macOS.json` missing, while `core-macos` job `122915370214` was cancelled. This was correct refusal, before readiness. The cancelled job's log returned 404 `BlobNotFound`; its cancellation cause is unknown. |
| Missing physical qualification | Three Linux live requirements remain absent. On PR events all three manual/self-hosted Stage-10 jobs are intentionally skipped. Two Windows lifecycle results consequently remain absent. |
| Deliberate production refusal | Current/previous Windows gate lists reboot, long-running lifecycle, update restart and failed-update-safe behavior, then `WINDOWS_BETA_RELEASE_READY=NOT_READY`. The last two have no implemented producer in the reviewed scope and are explicitly `NOT_IMPLEMENTED` in policy. No gate is relaxed. |
| Infrastructure failure | No current infrastructure root cause is established by these job logs. Older macOS cancellation is observed, with reason unknown. Do not relabel it as passing Core or a proven infrastructure defect. |

All three historical Linux jobs printed the same four blockers: Core plus
`LINUX_SYSTEMD_INTEGRATION`, `LINUX_CLEAN_INSTALL` and
`LINUX_REBOOT_LIFECYCLE_UPDATE`. Current and previous Windows jobs aggregated
Core successfully, then printed the same four Windows blockers above. This
distinguishes the Linux transfer bug from a failure of the Core test suites.

## Implementation changes

The [workflow](../../.github/workflows/platform-deployment.yml) now makes Linux
depend on all three Core jobs. Both always-running release consumers first
require `needs.core-{linux,windows,macos}.result == success`. Failed, cancelled,
skipped or absent current status refuses before downloading artifacts; old
artifacts from an earlier attempt of the same run cannot override that result.
Linux downloads three exact artifact names from the default **current run**
into separate OS directories, invokes existing `platform_evidence aggregate-core`
and passes the resulting matrix to readiness with explicit merge revision,
provider and run ID. There is no previous-run search, docs-only exception,
status conversion, required-step `continue-on-error` or branch-policy change.

The [producer](../../deployment/platform_evidence.py) continues to require the
canonical plan ID/digest and exact Linux/Windows/macOS marker set, matching
execution revision/provider/run. It now also refuses non-integer schema
versions and empty/non-string marker identity fields, and records the actual
aggregator OS using the existing canonical host helper. Core markers are
success-only: `core_test_plan.execute_plan` publishes one only after the full
required pytest plan exits successfully with zero failures/errors/skips.
Injecting an extra `status=FAIL` field is malformed, not an alternate PASS wire.

Stage-10 normalization now rejects boolean/float versions in receipt/contract
and absent/empty/whitespace current runner name or architecture. Matching
malformed receipt fields could previously pass those comparisons, including a
null name matching an absent environment value. Synthetic baseline replay
confirmed these old cases emitted a PASS slice; they now refuse without output.
Exact revision, run, provider, runner and MachineGuid comparisons remain.

The [native producer](../../deployment/windows_stage10_live.ps1) previously
checked each loop sample's PID/start time but collected its final snapshot
without the same comparison. It now calls a shared
[soak identity guard](../../deployment/windows_stage10_soak_identity.ps1) for
loop samples **and the final snapshot before emitting proofs**. A changed PID,
changed start time, nonpositive/non-integer PID or missing identity refuses.
The [synthetic PowerShell regression](../../deployment/windows_stage10_soak_identity_regression.ps1)
executes only this pure guard and is a required Windows PowerShell contract
step; it neither invokes Stage 10 nor touches a service. Minimum 24 hours,
monotonic elapsed measurement, manual opt-in, self-hosted labels and timeouts
are unchanged. These repairs do not prove continuous host/process integrity.

## Four Windows requirements and operator-only next work

| Requirement | Current producer, transport and consumer | Remaining evidence |
| --- | --- | --- |
| `WINDOWS_REBOOT_RECOVERY` | Native prepare/qualify binds revision/run, runner name/arch, MachineGuid, changed boot and current-run backend provenance. Continuity is uploaded before reboot; receipt/evidence/transcript are uploaded after qualification and consumed by the same workflow run. | An authorized real run on the approved machine; none was performed here or on the audited PR. |
| `WINDOWS_LONG_RUNNING_LIFECYCLE` | Same producer requires at least 24 hours, samples service/readiness/backend identity, uses Stopwatch and now checks the final instance. Python producer validates native receipt identity/provenance/proofs; readiness validates normalized revision/run/provider/class/status. | The real full-duration qualification and retained originals, not unit tests. |
| `WINDOWS_UPDATE_RESTART` | Policy explicitly says `NOT_IMPLEMENTED`: secure updater execution contract unresolved. No producer for this acceptance ID was found in the targeted deployment/workflow search. | Separately reviewed implementation and an actual qualified update/restart run. No existing command can truthfully emit this PASS. |
| `WINDOWS_FAILED_UPDATE_SAFE_BEHAVIOR` | Policy explicitly says `NOT_IMPLEMENTED`: no fabricated M1 rollback authority. The two-item lifecycle contract does not contain this acceptance ID. | Reviewed safe-failure/recovery and failure-injection contract, followed by authorized actual evidence. Do not add a blanket PASS field to the lifecycle script. |

Before the native prepare/qualify jobs may proceed, require approval
of the exact revision, reboot window and Windows x64 machine; protected runner
registration with labels `self-hosted, Windows, X64, cryptohunter-stage10`;
the **same runner name, architecture and MachineGuid across reboot**; no
competing maintenance; and current-run Stage-9 installer manifest/clean-install
receipt. The installed backend hash/version must match that run's qualified
manifest, with the required enrollment/runtime state established through the
approved operator process. Stage 10 does not itself install/enroll that host.
SCM service must already run automatically, with protected readiness matching
its PID, `freshness_runtime`, corehost lock and denied cross-key/role checks.

Only after those prerequisites and explicit operator authorization:

```sh
gh workflow run platform-deployment.yml --repo Dudzian/Dudzian --ref <approved-ref> -f run_stage10=true
```

The existing workflow runs Core/SCM/clean-install prerequisites and performs:

```powershell
./deployment/windows_stage10_live.ps1 -Mode prepare-reboot -State stage10-continuity.json -QualifiedManifest stage10-qualified/installer-manifest.json -CleanInstallReceipt stage10-qualified/windows-clean-install-receipt.json
# Continuity is uploaded first; reboot is scheduled after job cleanup:
shutdown.exe /r /t 75 /d p:4:1 /c "CryptoHunter Stage-10 qualification"
# After its barrier and an actual reboot of that same approved host:
./deployment/windows_stage10_live.ps1 -Mode qualify -State stage10/stage10-continuity.json -SoakHours 24 -Output stage10/windows-stage10-receipt.json
python -m deployment.platform_evidence windows-stage10 --receipt stage10/windows-stage10-receipt.json --output stage10/windows-stage10-evidence.json
```

These are explanatory counterparts of the reviewed manual workflow, not an
alternative to its environment/provenance/authorization. Preserve continuity,
raw receipt, normalized evidence, transcript and failures. Successful lifecycle
qualification could close **only two items**, leaving both update blockers.
No arbitrary wall-clock receipt TTL is invented: stale-receipt tests reject
different revisions/runs; a broader independent freshness policy is separate.

Linux still needs real current `systemd` integration, clean-install and
reboot/lifecycle/update evidence. There is no implemented Linux live producer
identified by this targeted review from which to give a truthful one-command
qualification recipe. A reviewed host procedure/producer and actual original
evidence are required; synthetic platform tests cannot fill those entries.

## Betterleaks

The original local ignored cache
`tests/__pycache__/test_runtime_bootstrap.cpython-311-pytest-9.1.1.pyc` is an
untracked, 348426-byte generated file dated 2026-10-06. Static extraction found
the exact `_RISK_TLS_KEY`/`_RISK_TLS_CERT` source-fixture bytes in that cache,
without executing it or printing keys. The source fixture at
`tests/test_runtime_bootstrap.py:500` is already narrowly reviewed as BL-018;
its matching self-signed certificate is `CN=test.local` and tests assert its
fingerprint. The duplicate binary-cache path is not that source baseline entry.

The original full scan's failure remains a valid local observation. No cache,
security test, global baseline, scanner rule or ignore was changed to hide it.
A separate clean worktree was used, with reports/tools outside the checkout.
Its full scan of `.` with unchanged Betterleaks 1.9.0 and baseline passes.
Repeatable comparison requires a fresh checkout/worktree of the selected
commit, rather than deleting inconvenient files from the original checkout.

Hosted [security job 123182093012](https://github.com/Dudzian/Dudzian/actions/runs/40792867468/job/123182093012)
for the **related audited HEAD** succeeded: release/pinned archive checksums
both OK, full-directory scan approximately 61.60 MB, `no leaks found` at
2026-10-29 00:22:54 UTC. Its isolated checkout runs no pytest or cache download
before scanning. This validates the existing hosted scope, not the new PR's
checks. Scanner rules, baseline and scan steps need no changes; the quality
workflow only adds the new regression test paths to its existing test command.

## Regression and quality evidence

The focused suites cover valid current Core, absent/malformed/substituted
markers, wrong revision/run/provider/plan, missing/duplicate/unknown OS,
injected failure status, required pytest failure/bad JUnit, invalid schema or
identity, actual aggregator OS and unsupported hosts. They also execute the
current-job guard for each OS in both gates with success/failure/cancelled/
skipped/absent status. Correct Core removes **only Core** from Linux's blockers.
Existing and new Windows cases reject absent Stage 10, stale revision/run,
machine/runner mismatch, invalid schema/identity and failed/incomplete proofs;
valid synthetic lifecycle evidence still leaves production/update requirements.

Final focused result: **450 passed, 1 skipped in 12.34s**. The skip is the
existing genuine Production Trust package test described below. The nine
selected files are the platform evidence/readiness/Core suites, Stage-10
identity/provenance/preflight/artifact/runner suites and workflow regression.
The exact all-files format hook, Tier A Ruff plus the changed producer, and
explicit changed Python file formatting checks pass without unrelated changes.

```sh
python -m pre_commit run ruff-format --all-files
python -m ruff check --no-cache bot_core core scripts tests deploy stage6_samples ui deployment/platform_evidence.py
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 DUDZIAN_TEST_MODE=1 DUDZIAN_ALLOW_LONG_POLL=0 python -m pytest -q -rs tests/deployment/test_platform_evidence.py tests/deployment/test_platform_readiness.py tests/deployment/test_core_test_plan.py tests/deployment/test_windows_stage10_evidence_identity.py tests/deployment/test_windows_stage10_runtime_provenance.py tests/deployment/test_windows_stage10_preflight.py tests/deployment/test_windows_stage10_artifact_preflight.py tests/deployment/test_windows_stage10_runner_preflight.py tests/scripts/test_platform_core_evidence_workflow.py
python -m mypy
python -m mypy --platform win32
python -m mypy --platform win32 deployment/platform_evidence.py
semgrep scan --config .semgrep/cryptohunter.yml --metrics=off --disable-version-check --error bot_core core deployment scripts
semgrep scan --test --config .semgrep/cryptohunter.yml .semgrep/cryptohunter.py --metrics=off --disable-version-check --jobs 1
betterleaks dir --baseline-path .betterleaks-baseline.json --redact=100 --no-banner --no-color .
```

Task-local tool/cache paths were selected outside the checkout; no system
Python or security policy was replaced. Local
tools: development CPython 3.12.14, Ruff 0.14.11, pre-commit 4.5.1, mypy 1.10.0,
Semgrep CE 1.179.0, Betterleaks 1.9.0. Configured mypy Linux and win32 cover 359
files each. Supplementary producer-only win32 type checking passes; its native
Linux check has three pre-existing `winreg` attribute errors, reproduced
unchanged on the base. No suppressions or configured-scope reduction were made.
The unchanged blocking Semgrep rules and their five fixtures pass. Remote
advisory rules, CodeQL, dependency audit and full Tier B are not claimed locally.

The synthetic PowerShell guard passed locally on PowerShell 7.5.4/Linux
(one valid case, eight refusals). The downloaded archive matched Microsoft's
published release checksum. This is not Windows PowerShell 5.1 execution or
physical qualification; the new required hosted contract step covers that
shell separately. One existing Python test requires the genuine operator-
supplied public Production Trust package; its skip is reported, not filled
with a fabricated package. Exact-HEAD hosted results remain separate.

## Priority register and conditional merge

| Priority / ID | Assessment | Disposition |
| --- | --- | --- |
| P0 | No newly confirmed P0 defect from this bounded review; not a proof of absence. | Preserve every physical/production authorization boundary. |
| P1 / CI-01 | Linux dropped valid current Core evidence. | Fixed wiring/validation; verify exact-HEAD Actions eventually removes only the Core blocker. |
| P1 / CI-02 | Producer accepted malformed schema/current runner identity; aggregator mislabelled its own host; always-running gates needed current status protection. | Fixed with refusal regressions; not the observed cause of audited Windows refusal. |
| P1 / CI-03 | Final native service snapshot was not compared with soak instance before proofs. | Fixed guard and synthetic regression; real qualification still required. |
| P1 / CI-04 | Missing Linux live evidence; Windows lifecycle not run; update requirements lack implemented producers. | OPEN, production blocking. No invented PASS or branch-protection exception. |
| P2 | Prior cancelled macOS job cause unknown; supplementary Linux winreg typing limitation; independent freshness/host trust still external. | Disclosed; no blanket suppression, arbitrary TTL or infrastructure claim. |
| P3 | Distinguish source HEAD from merge checkout, scoped/local/hosted scans and synthetic/live evidence. | Traceable report and exact-HEAD PR snapshot; historical reports unchanged. |

Conditional recommendation: review the repair as an evidence-flow change only,
after changed-code tests/style/types/security and required exact-HEAD checks
have their actual dispositions. Expected physical-readiness refusals remain
failures, not successful CI. If required branch checks are red, do not merge
or bypass protection; obtain the missing evidence/implementation under the
existing policy. This task neither changes that policy nor performs merge.

```text
stage_9=IN_PROGRESS
qualification=BLOCKED
CONTRACT_APPROVED=NO
STAGE_3A_COMPLETE=NO
PRODUCTION_QUALIFIED=NO
production_status=PRODUCTION_NO_GO
runtime_attestation=MISSING
host_pid_namespace=UNVERIFIED
```
