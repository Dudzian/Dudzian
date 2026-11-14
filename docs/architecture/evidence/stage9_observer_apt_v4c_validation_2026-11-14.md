# Observer 3A V4c validation — 2026-11-14

This record describes a nonproduction foundation, not independent acceptance.
The [contract and remaining approval register](../stage9_observer_apt_v4c.md)
define what the evidence establishes and what remains blocked before V5.

## Base and remote status

Before editing, `git ls-remote origin refs/heads/main` returned
`4d43ef1ae8687c86ec8a41d3659f3e3e6b0c22f6`, matching the requested base.
GitHub reported PR #3270 merged at `2026-11-14T13:54:09Z` with that merge SHA.
The starting checkout was clean but older; the isolated branch
`feat/stage9-observer3a-v4c-trust-foundation` was created from fetched `origin/main`.

At the first post-merge Actions observation, Style preflight and dependency graph
update were complete/successful; Windows Build and Marketplace Catalog were in
progress, and CI, Quality and Security, CodeQL and Stress Lab were queued.
A subsequent observation showed Windows Build, Marketplace Catalog and Stress
Lab successful, CodeQL in progress, CI and Quality and Security still queued.
**An overall post-merge CI PASS is not claimed.** Relevant run IDs:

| Workflow | Run ID |
| --- | --- |
| Style preflight | 43602076365 |
| CI (push: main) | 43602076417 |
| Quality and Security | 43602076275 |
| CodeQL | 43602076366 |
| Windows Build | 43602076325 |
| Stress Lab Report | 43602076324 |
| Marketplace Catalog | 43602076482 |

Codebase Memory MCP and its CLI were unavailable in this environment. Targeted
repository searches and direct source review were used instead; there is no
index-coverage or exhaustive architecture claim.

## Implemented scope and review

Three new modules implement bounded canonical APT acceptance/checkpoint/time
contracts, TEST-only Ed25519 verification, immutable snapshot transitions and
composition with unchanged V4b results. CI additions include these modules in
existing Ruff, format, mypy Linux/Win32 and branch-coverage scopes, and include
their tests in the existing quality ratchet. No thresholds, security policies,
readiness gates or runtime authorization paths change.

Independent source review identified two gaps during development; both were
fixed before final validation:

1. A signed acceptance claim issued after the supplied evaluation time must be
   refused, even if both times fall inside the policy window.
2. A predecessor state cannot bind its own fields merely by supplying a hash.
   The composition now requires the previous signed checkpoint, verifies it
   under the separate TEST checkpoint anchor, hashes its exact bytes, and
   compares its complete state projection before proposing a transition.

Regression tests cover future issuance, signed predecessor corruption, altered
state fields and rotation grants, and a recomputed digest paired with a
substituted state. Final independent review found no unresolved blocking issue
in the new source. This is source review of this bounded foundation, not approval
of roots, clock, storage, toolchain or deployment.

## Local environment and completed checks

All commands ran as nonroot UID 1000 on Linux with CPython 3.12.14.
The disposable validation environment used Ruff 0.14.11, mypy 1.10.0,
cryptography 50.0.2, Zstandard 0.25.0 and pytest 9.2.0.
Only synthetic test documents, in-memory TEST Ed25519 keys and inert package
fixtures were used. No inspected package, installer, Stage 3A/3B or privileged
qualification was executed.

| Check | Completed result |
| --- | --- |
| Existing V1–V4b package/chain/crypto/reproduction regressions | 918 passed; zero skips/failures |
| Existing Stage9 Observer contract regressions | 869 passed; zero skips/failures |
| Existing CI contract regressions | 181 passed; zero skips/failures |
| V4c contracts/transitions/composition | 373 passed: 71 trust, 89 transition, 213 integration; zero skips/failures |
| V4c branch-inclusive coverage | 99% aggregate: trust 99%, transition 100%, integration 97% |
| Changed executable line coverage / unchanged 70% gate | 98% (535/541 lines); passed |
| Full workflow Ruff baseline | Passed |
| New V4c source/tests strict Ruff E/F/B/I and format | Passed |
| Configured mypy Linux / Win32 | Passed, 402 source files each |
| Exact mandatory `python -m pre_commit run ruff-format --all-files` | Passed |
| Full `python -m pre_commit run --all-files` | Ruff, format, configured mypy and staged offline Betterleaks passed |
| Configured blocking Semgrep | Five rules, 1,134 Python targets, zero findings |
| Semgrep rule fixtures | 5/5 passed |
| Maintained community Semgrep on three V4c modules | 200 rules, zero findings/errors |
| Betterleaks 1.9.0 working-tree scan | Passed with unchanged configuration/baseline |
| pip-audit canonical desktop lock | 105 dependencies; zero vulnerabilities/skips |
| pip-audit exact Zstandard 0.25.0 lock | One dependency; zero vulnerabilities/skips |
| Test environment `pip check` | Passed |

Configured mypy ran inside the repository's isolated cached pre-commit hook
environment. Optional NumPy stubs in the broader test environment triggered an
internal mypy 1.10 cache failure; using the exact hook dependencies resolved the
tooling failure without changing source scope, options or suppressions.

The unchanged Betterleaks scan found synthetic fixture text in a pre-existing
ignored CPython 3.11 bytecode cache. Its final rerun additionally found synthetic
credential-URI test parameter IDs in generated pytest cache and CI JUnit output.
These generated artifacts were reversibly quarantined outside the repository;
the full working-tree scan then passed. No source, scanner exception or baseline
edit was made.

Genuine V4b crypto regressions used standalone gpgv 2.4.7 extracted locally from
the package pinned in the existing V4b validation evidence; no investigated DEB
was installed or executed. `STAGE9_TEST_GPGV` selected that verifier. This
establishes test execution, not independent tool approval.

## Reproduction

Use a nonroot environment with the repository's test extra and pinned quality
tools. Select a provisioned local TEST gpgv via `STAGE9_TEST_GPGV` if needed.

```sh
python -m pytest -q -o addopts='' \
  tests/deployment/test_stage9_observer_offline.py \
  tests/deployment/test_stage9_observer_zstd.py \
  tests/deployment/test_stage9_observer_tar_v3.py \
  tests/deployment/test_stage9_observer_package_v3.py \
  tests/deployment/test_stage9_observer_apt_v4a.py \
  tests/deployment/test_stage9_observer_apt_v4b.py \
  tests/deployment/test_stage9_observer_apt_v4b_crypto.py \
  tests/deployment/test_stage9_observer_gpgv_v4b.py \
  tests/deployment/test_stage9_real_deb_v2.py \
  tests/deployment/test_stage9_real_deb_v3.py \
  tests/test_stage9_observer_contract.py
python -m pytest -q -o addopts='' \
  tests/deployment/test_stage9_observer_apt_v4c.py \
  tests/deployment/test_stage9_observer_checkpoint_v4c.py \
  tests/deployment/test_stage9_observer_trust_v4c.py \
  --cov=deployment.stage9_observer_apt_v4c \
  --cov=deployment.stage9_observer_checkpoint_v4c \
  --cov=deployment.stage9_observer_trust_v4c --cov-branch
python -m pre_commit run ruff-format --all-files
python -m pre_commit run --all-files
python -m mypy --config-file=pyproject.toml --platform=win32
```

Validation logs, JUnit, scanner JSON and coverage XML are intermediate local
artifacts under ignored `work/`, with the above generated fixture artifacts
quarantined in `/workspace/work/`; the Draft PR records the published SHA and the
remote workflow status. The source, tests and this evidence are committed.

## Fixed authorization outcome

```text
Stage 9 = IN_PROGRESS
Observer 3A qualification = BLOCKED
Stage 3B = NOT AUTHORIZED
Production = NO-GO

External authority acceptance = UNVERIFIED
Authenticated time = UNVERIFIED
Durable anti-rollback = BLOCKED
Runtime qualification = BLOCKED
```

Even the complete correctly signed TEST bundle refuses production authorization.
There is no production root enrollment, clock adapter, current checkpoint
service, atomic durable state writer, automatic key rotation, network key/policy
retrieval or V5 implementation in this change.
