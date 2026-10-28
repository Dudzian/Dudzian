# Stage 9 / 3A — post-merge CI evidence

Status: `PROPOSED_UNAPPROVED`. This is an observation of GitHub Actions,
not independent approval or target-runtime qualification.

## Revision and observation boundary

The reviewed merge is PR [#3215](https://github.com/Dudzian/Dudzian/pull/3215),
commit `ae712aea85e4cbc372bdf226d6a5bc5be02f4949`. The public Actions list and
the individual run pages bind the runs below to that exact commit, `main`,
event `push`, created at **2026-10-28 14:52:18 UTC**. These are post-merge
runs, not the preceding PR-head checks.

The initial run-list snapshot was read at approximately 15:00 UTC; job states
were read through the connected GitHub job API at **15:00:38 UTC**. A queued
aggregate run can already contain running or successful jobs. An unfinished
run has no final conclusion; it must not be described as PASS or FAIL.

The connected combined-status endpoint returned an empty `statuses` list.
That endpoint did not establish check success. The connected commit-workflow
wrapper filters to `pull_request` and its first page, so it was not used to
infer the absence or success of post-merge `push` runs. The ordinary CLI REST
request was forbidden in this environment. Public GitHub HTTPS pages supplied
the exact push-run identities, then connected job/log tools supplied job
evidence. No credentials were replaced, no workflow was rerun, and no gate
was changed.

## Observed runs

| Run (exact merge SHA; push) | Aggregate snapshot | Job evidence at 15:00:38 UTC |
| --- | --- | --- |
| [CI #9845 / 40710218843](https://github.com/Dudzian/Dudzian/actions/runs/40710218843) | Queued | Dependency vulnerability scan, Linux/macOS wheelhouse and native Windows x64 security (no MSI): success. Windows wheelhouse, SBOM, marketplace job: queued. Downstream tests and gates still pending. |
| [Quality and Security #563 / 40710218918](https://github.com/Dudzian/Dudzian/actions/runs/40710218918) | Queued | PostgreSQL remainder, semantic issuer preflight and atomic Root Proof commit: running. Quality ratchet, History Successor qualification and Semgrep/secrets: queued. Full-history advisory secrets: skipped. Combined coverage gates pending dependencies. |
| [CodeQL #563 / 40710218870](https://github.com/Dudzian/Dudzian/actions/runs/40710218870) | In progress | JavaScript/TypeScript: success; Python analysis: running. |
| [Windows Build #502 / 40710218884](https://github.com/Dudzian/Dudzian/actions/runs/40710218884) | In progress | Focused tests and Windows probe: success; onedir build: running; artifact smoke test and publication pending. |
| [Style preflight #279 / 40710218948](https://github.com/Dudzian/Dudzian/actions/runs/40710218948) | Success | `Ruff format (all tracked files)` success, job `122893155904`. |
| [Marketplace Catalog #5071 / 40710218931](https://github.com/Dudzian/Dudzian/actions/runs/40710218931) | Success | Preset signatures/Markdown and catalog signatures/QA coverage: success. |
| [Stress Lab Report #2466 / 40710218851](https://github.com/Dudzian/Dudzian/actions/runs/40710218851) | Success | Stress report generation and artifact upload: success. |

The public list also displayed a successful dependency-graph run
`40710232704`, created at 14:52:24 UTC, but its list entry did not expose an
exact-commit link. It is not counted as exact-SHA qualification evidence.
The list's first page contained all seven linked merge-SHA runs above; this
is not an assertion that no additional manual, delayed or later-attempt run
can exist. Re-read the run identities and attempts before approval.

## Classification

| Category | Finding and consequence |
| --- | --- |
| Confirmed code regression | None identified in the observed completed jobs. Most code/test gates are unfinished, so absence of a regression is **not established**. No code fix is justified solely by a queue or missing conclusion. |
| Confirmed infrastructure failure | None identified in the observed completed jobs. Runner notices about macOS capacity and Ubuntu migration are notices, not failures. A queued job may be waiting for capacity; its cause is not proved by the status. |
| Timeout | None observed at the snapshot. `quality-security.yml` retains 120-minute quality/atomic/history jobs and a 180-minute semantic job. A still-running job before its deadline is not a timeout. Later `timed_out` conclusions require job logs and are distinct from assertion failures. |
| Missing qualification evidence | Exact Ubuntu 24.04 x86_64 / CPython 3.12 build, loader/library inventory, independent provenance and physical qualification remain missing. Successful ordinary CI cannot supply these absent materials. |
| Expected production block | Linux/Windows production readiness and Stage 9 remain blocked under their existing policy. Missing required authentic native/install/lifecycle evidence must remain a refusal; do not weaken a gate to turn it green. No post-merge production-readiness job is claimed to have failed here. |
| Expected skip | Full-history secrets is advisory/scheduled; its skipped job does not mean the blocking working-tree secrets job passed. Stage 10 is manual/self-hosted and must not be dispatched for this review. |
| Pending outcome | CI and Quality/Security have no complete result at the latest snapshot below. Required validation remains pending independently of local tests. |

**Job refresh, 2026-10-28 15:03:27 UTC:** CI's SBOM job `122893156717`
is now successful. Both CodeQL jobs are now successful (Python job
`122893156542`, JavaScript/TypeScript job `122893156027`). The other listed
running/queued jobs remain unfinished, with no failed step reported by the
job API. This refresh supersedes the individual job states in the initial
table; it does not change the earlier timestamp of its aggregate run-list
snapshot. In particular, CI and Quality/Security still have not passed.

**Latest refresh, 2026-10-28 15:08:48 UTC:** the public `main` Actions list
still exposes the same seven exact-merge-SHA push runs, with no additional
exact-SHA run on its first page. Windows Build is now successful, including
its artifact smoke test; CodeQL is successful as an aggregate run. Quality
and Security is now in progress: all six ordinary quality/security/serial
qualification jobs are running, while full-history advisory secrets remains
skipped and combined coverage gates await dependencies. CI remains queued,
with SBOM, vulnerability audit, Linux/macOS wheelhouses and native Windows
security successful; Windows wheelhouse and marketplace validation remain
queued. The connected job API reports no failed steps in the enumerated
jobs. This latest snapshot supersedes earlier run/job states, but still
does not establish CI/Quality completion or any target qualification.

## The successful formatter does not qualify the target

The downloaded [formatter job log](https://github.com/Dudzian/Dudzian/actions/runs/40710218948/job/122893155904)
contains:

```text
2026-10-28T14:54:36.8797747Z 26.04.1
2026-10-28T14:54:36.8800332Z Image: ubuntu-26.04
2026-10-28T14:54:36.8800914Z Version: 20261025.179.1
2026-10-28T14:54:46.8386061Z ae712aea85e4cbc372bdf226d6a5bc5be02f4949
2026-10-28T14:54:46.9695612Z Successfully set up CPython (3.11.17)
2026-10-28T14:54:59.8441118Z ruff format..............................................................Passed
```

`ubuntu-latest` resolved to Ubuntu **26.04.1**, not the required 24.04, in
this job. Its setup used CPython **3.11.17**, not 3.12. The source workflows
also select Python 3.11. This is valid formatting evidence for this commit;
it is not ELF compatibility evidence for the Observer target. Other jobs'
actual images must be read individually before making image-specific claims.

## Source-checked safety and follow-up

The workflow definitions at the reviewed commit are the source for job
scope and gates:

- [CI](../../.github/workflows/ci.yml) has separate `push` and `pull_request`
  concurrency groups and retains ordinary test, type, audit and integration gates.
- [Quality and Security](../../.github/workflows/quality-security.yml) retains
  its 70% changed-code coverage checks, serial PostgreSQL jobs, security scans
  and existing timeouts. Pending coverage is not a measured percentage.
- [Style preflight](../../.github/workflows/style-preflight.yml) runs the exact
  all-files pre-commit formatting scope; this post-merge PASS does not replace
  formatting the new PR.
- [Platform deployment](../../.github/workflows/platform-deployment.yml) is
  triggered by PR/manual dispatch, not ordinary `push`. The production trust
  artifact is required for Windows MSI promotion. Stage 10 requires explicit
  manual input and a self-hosted runner. No such dispatch was performed.
- [Platform readiness](../../deployment/platform_readiness.py) preserves
  authentic revision-bound native requirements including Windows reboot/long
  lifecycle and Linux systemd, clean install and reboot/update evidence.

Codebase Memory was unavailable in this executor (no corresponding connected
tools or executable), so this bounded CI review used direct workflow/source
inspection and live run/job/log evidence. It makes no graph-index completeness
claim.

Before a decision, refresh these same exact-SHA runs and latest attempts,
enumerate any additional runs, retrieve failed job logs if failures appear,
and classify each first failing step. Preserve a code regression as a separate
logical fix with a regression test. Keep infrastructure failures, timeouts,
missing evidence and intended production refusals separate. Do not rerun full
CI repeatedly or infer GitHub success from local PASS.

`CODE_READY`, `CONTRACT_REVIEW_READY`, `CONTRACT_APPROVED`,
`STAGE_3A_COMPLETE`, and `PRODUCTION_QUALIFIED` are not awarded by this CI
snapshot. `PROPOSED_UNAPPROVED`, `BLOCKED`, `PRODUCTION_NO_GO`, and
`PHYSICAL_TEST_PENDING` remain in force.
