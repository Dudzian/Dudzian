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

## P2: full V4b → V4c integration evidence

The P2 follow-up starts from the verified PR #3271 HEAD
`9008efeb3acfa1aebe5d1563b5ee72a8c69387b3`. Before editing, the remote head,
local head and clean checkout matched; the existing diff contained eleven
files, 3,118 additions and one deletion. All eight exact-head PR workflows were
still queued. Their pending state is separate from completed local checks.

The follow-up changes only the existing V4c test module and this evidence
record. It adds a genuine local `gpgv` path from an inert Release → Packages →
DEB fixture through `verify_apt_signature_v4b()` into
`verify_apt_acceptance_v4c()`. The returned V4b object is passed unchanged;
the real integration path neither constructs `AptResultV4b` nor mocks the
signature verifier. The original constructed-result unit cases remain useful
separate tests. A shared TEST-material factory signs acceptance, evaluation-time
evidence and both candidate/predecessor checkpoints around either input path.

Disposable GnuPG TEST keys sign the exact dated Release bytes. Separate
in-memory Ed25519 TEST keys sign the V4c contracts. The successful path checks
complete V4b chain evidence, a technically valid immutable proposal, unchanged
predecessor/V4b inputs, and all fixed blocked authorization/qualification fields.
No investigated package or installer is executed.

Negative cases cover Release replacement after signing, an invalid APT
signature, an APT signer outside the context policy, substituted V4b context,
invalid V4c acceptance/checkpoint signatures, rollback, mismatched predecessor
hash and missing/mismatched evaluation-time evidence. Stage-specific reason
assertions distinguish the intended failures; every V4c refusal contains no
proposed state. Release/signature declaration hashes are refreshed after
tampering so the APT negative cases exercise real crypto/policy validation.

The available local verifier was GnuPG `gpgv` 2.4.7, SHA-256
`3f29dddc10e4089aeac5b2675313f4e5fe822abe5ab3bf7898d238c32d758304`;
GnuPG `gpg` 2.4.7 generated only disposable TEST signing material. This tool
selection is test provenance, not independent tool approval. Tests require a
nonroot Linux environment and provisioned local tools; unavailable requirements
are reported as SKIPPED, with no mock fallback and no runtime download.

Neither the supplied Python result object nor this successful test is an
independently authenticated proof of execution. Root acceptance, authenticated
current time, checkpoint currentness, durable state and runtime qualification
remain open exactly as recorded in the contract. This follow-up closes a P2
integration-evidence gap; it does not change the implementation, V1–V4b
contracts, policies, Linux/Windows gates or V5 scope.

Completed P2 local validation (Linux, UID 1000):

| Check | Result |
| --- | --- |
| Genuine V4b → V4c integration | 12 passed: one positive and eleven negative cases; zero skips/failures |
| Complete V4c suite | 385 passed: 225 composition, 89 transitions, 71 trust; zero skips/failures |
| V1–V4b/Observer contracts and existing CI contracts | 1,968 passed; zero skips/failures |
| Total relevant regression cases | 2,353 passed; zero skips/failures |
| V4c branch-inclusive coverage | 99% aggregate; implementation unchanged |
| Full workflow Ruff baseline and strict test E/F/B/I/format | Passed |
| Configured mypy Linux/Win32 | Passed, 402 files each |
| Exact `python -m pre_commit run ruff-format --all-files` | Passed |
| Full all-files pre-commit, including staged offline Betterleaks | Passed |
| Configured Semgrep / rule fixtures | Five rules, 1,134 Python targets, zero findings; 5/5 fixtures passed |
| Community Semgrep on three V4c modules | 200 rules; zero findings/errors |
| Betterleaks working tree / final staged set | Passed; configuration and baseline unchanged |
| pip-audit desktop / Zstandard lock | 105 / one dependencies; zero vulnerabilities/skips |
| Test environment `pip check` | Passed |

A separate controlled missing-tool probe selected only the positive E2E test
with a deliberately nonexistent `STAGE9_TEST_GPGV`: one explicit SKIPPED,
224 deselected, with the missing local TEST gpgv prerequisite in the reason.
This prerequisite probe is not part of the successful native regression counts.
All twelve genuine integration cases executed with the available verifier.
Fixture-local platform/tool checks preserve the portable V4c unit tests.

For a provisioned nonroot Linux test environment, the targeted reproduction is:

```sh
STAGE9_TEST_GPGV=/absolute/path/to/local/gpgv python -m pytest \
  tests/deployment/test_stage9_observer_apt_v4c.py \
  -k real_v4b_to_v4c -q -rs -o addopts=''
```

P2 validation logs, JUnit, scanner JSON and coverage XML are intermediate local
artifacts under `/workspace/work/p2-validation/`, outside the repository. The
final independently reviewed diff has no unresolved scoped P0/P1/P2 finding;
the requested integration-evidence P2 is closed. The external acceptance,
authenticated time, durable rollback protection and runtime requirements remain
open by design, with every authorization status above unchanged. The existing
PR remains Draft, without merge; its published-head Actions snapshot is recorded
separately in the PR description and is not inferred from local PASS results.
