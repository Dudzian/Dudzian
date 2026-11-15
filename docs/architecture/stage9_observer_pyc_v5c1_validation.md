# V5c1 validation and baseline preflight

Date: 2026-11-15 UTC. Base main:
`68e03d48343c98cdafe0c0ea20033730d4541ea1` (merged PR #3273).
Branch: `codex/stage9-observer-v5c1-pyc`.

The [technical contract](stage9_observer_pyc_v5c1.md) contains the exact schema,
trust boundary, missing reconstruction-proof register and V5c2 requirements.
This report distinguishes local verifier validation from hosted CI, real
CPython reconstruction and platform qualification. There was no merge.

## Mandatory preflight

Before implementation, remote main matched the base above and the paginated
open-PR inventory was empty. The repository started clean on an older local
checkout; the new branch was created from fetched `origin/main`. Codebase
Memory tools and CLI were unavailable, so the relevant V1, V5a, V5b, immutable
member reader, runtime/build materials, schema and CI source were inspected
directly. No conclusion was inferred from a missing structural index.

Merged PR #3273 has exact head
`ec329c282e0ff436286e2c0534148053856bbade`. The preflight was refreshed at
**01:29:32 UTC**, with unchanged main and no open PRs. Observed checks:

| Baseline check | Observed result |
| --- | --- |
| [PR CodeQL](https://github.com/Dudzian/Dudzian/actions/runs/43661203047) | SUCCESS, both Python and JavaScript/TypeScript jobs |
| [PR Style preflight](https://github.com/Dudzian/Dudzian/actions/runs/43661203009) | SUCCESS |
| [Native Windows security](https://github.com/Dudzian/Dudzian/actions/runs/43661203053/job/132650699356) | SUCCESS; 29 native PASS, separate emulated set 191 PASS |
| [Exact-head push Ubuntu suite](https://github.com/Dudzian/Dudzian/actions/runs/43661129307/job/132651333190) | SUCCESS; 30265 PASS, 169 SKIP, 1539 deselected |
| [PR broad Windows suite](https://github.com/Dudzian/Dudzian/actions/runs/43661203053/job/132651678849) | IN_PROGRESS; no final result claimed |
| [PR broad Ubuntu suite](https://github.com/Dudzian/Dudzian/actions/runs/43661203053/job/132651678878) | SUCCESS; 30265 PASS, 169 SKIP, 1539 deselected; separate from the completed push run |
| [PR Quality and Security](https://github.com/Dudzian/Dudzian/actions/runs/43661203076) | IN_PROGRESS; Semgrep/secrets job SUCCESS, history-secret advisory job SKIPPED, other jobs running |
| [PR main CI aggregate](https://github.com/Dudzian/Dudzian/actions/runs/43661203053) | Pending; individual queued/running jobs prevent a final PASS |
| [PR platform tracks](https://github.com/Dudzian/Dudzian/actions/runs/43661203063) | Pending aggregate; Linux live-evidence integration refused, core OS, evidence-contract and Windows deployment integration jobs succeeded, later Windows installation jobs pending |
| Merge-main CI, Quality/Security and CodeQL | Pending at the preflight refresh; no final PASS claimed |

The native Windows result above is a test of its existing bounded Windows
security scope, not production Windows qualification. Ubuntu's skips and
deselections are not passes. No completed scoped regression was observed at
this checkpoint; pending jobs remain an explicit validation gap.

Known platform qualification blockers were checked against
`.github/workflows/platform-deployment.yml` and the
[V5b validation report](stage9_observer_materialization_v5b_validation.md).
Windows requires its operator-supplied Production Trust/live-evidence
artifacts; Linux requires live integration/readiness evidence. Earlier
expected denials include Linux systemd/clean-install/reboot evidence and
Windows reboot/long-running/update/failure-recovery evidence. They remain
blocking. This preflight does not label a new failure as expected without
matching its actual logs. The final snapshot's
[Linux integration failure](https://github.com/Dudzian/Dudzian/actions/runs/43661203063/job/132651673935)
explicitly reports `LINUX_SYSTEMD_INTEGRATION`, `LINUX_CLEAN_INSTALL` and
`LINUX_REBOOT_LIFECYCLE_UPDATE` as `MISSING_CURRENT_EVIDENCE` under
`LIVE_LINUX_INTEGRATION`, with `LINUX_PRODUCTION_READY=NOT_READY` and exit 1.
This matches the unchanged workflow's lines 798–805 and
`deployment/platform_readiness.py` evidence requirements; it is an existing
qualification denial, not an observed V5b regression. It was not suppressed
or relabeled PASS. Broad Windows remains running. Physical Linux and Stage 10
execution gates were not dispatched.

## Local validation

Functional and type checks used host CPython **3.11.16**. This host only runs
the trusted verifier and test tools; it never compiles investigated source,
deserializes the investigated body or runs the declared target interpreter.
The fixtures are actual immutable synthetic DEBs and explicitly pinned opaque
bytecode data, not qualification of real CPython output.

| Check | Result |
| --- | --- |
| New V5c1 positive/negative tests | 120 PASS, 0 SKIP |
| V1–V5b and real-DEB-script regression tests | 2474 PASS, 0 SKIP, actual `gpgv` exercised |
| Combined final functional run | 2594 PASS, 0 SKIP |
| CI workflow, platform, quality inventory and security-scan contracts | 241 PASS, 0 SKIP |
| Full configured workflow Ruff baseline | PASS |
| Expanded new-module Ruff `E,F,I,UP,B,SIM` and test Ruff `E,F,B,I` | PASS |
| Explicit workflow format scope | PASS, 341 files already formatted |
| Configured mypy Linux | PASS, 406 source files |
| Configured mypy `--platform win32` | PASS, 406 source files; type check, not native Windows execution |
| Exact `python -m pre_commit run ruff-format --all-files` | PASS, exit 0, no formatter changes |
| Full pre-commit | PASS; Ruff, ruff-format, configured mypy and staged offline Betterleaks |
| V5c1 statement/branch coverage | 99%; 290/291 statements, 20/20 branch outcomes |
| Diff coverage against `origin/main`, existing minimum 70% | PASS, 99.7%; 290/291 changed measured lines covered |
| Full offline Betterleaks with unchanged reviewed baseline | PASS, no leaks |
| Blocking custom Semgrep rules | PASS, zero findings; five rule fixtures PASS |
| New implementation/tests/contract: maintained plus custom Semgrep | PASS, 205 applicable rules, zero findings/errors |
| Full maintained Semgrep repository scan | Completed; 25 advisory findings in unchanged files, nine timeouts and one partial parse; not a clean or complete analysis |
| Canonical desktop lock `pip-audit` | PASS, 104 dependencies, zero known vulnerabilities |

The missing statement is the defensive `OSError` refusal handler. Negative
tests cover malformed headers, reserved flags, unsupported invalidation,
replaced bytecode/source, every source binding, independent pins, compiler
claim inconsistencies, filenames/paths, metadata, missing/extra/ambiguous
inputs, resource exhaustion, elapsed timeout and interruption both before and
after staging a first file. Forbidden-operation tests replace host file I/O,
network, subprocess, marshal and compilation/execution calls with failures.

The independently pinned positive fixture deliberately contains an opaque
non-code body. Complete artifact/hash/header agreement leaves per-file and
global `RECONSTRUCTION_UNVERIFIED`, `SOURCE_HASH_UNVERIFIED` and all authority
and qualification blockers unchanged. Empty declarations return `NOT_DECLARED`.

An initial regression run had 2444 PASS and 30 SKIP because `gpgv` was missing.
Those skips were not counted as successes. An unprivileged validation helper
was then prepared using signed Debian Trixie InRelease verification with the
existing archive keyring and `sqv`, signed Packages index hash/size, and exact
DEB hash/size checks. The extracted-only tool is
`gpgv_2.4.7-21+deb13u1+b5_amd64.deb`, 241100 bytes, SHA-256
`104e4c57b98f0b883aa94c68fbe7ec7b48bcdec7c0f9dedf2a51a5054aaeb5ae`.
`STAGE9_TEST_GPGV` selected it for the final 2474/2594-case runs. It was never
installed; preparing a trusted test tool is separate from executing any
investigated artifact.

The first Betterleaks scan found only an ignored generated pytest cache that
contained an existing private-key fixture literal. That cache was preserved
outside the repository and the full working-tree scan repeated with the
unchanged reviewed baseline. No source secret, suppression or baseline entry
was added. A combined maintained scan of changed files also reports two XML
warnings on unchanged pre-existing lines 14/776 of the CI contract test; its
only source change adds a module inventory entry. New implementation and tests
scan without findings. These existing advisory findings are not new blocking
scanner regressions.

The full maintained scan ran 219 applicable rules across 1223 files. All 25
findings are in unchanged sources. Nine rule timeouts across five unchanged
Python files and a partial parse of the existing
`scripts/diagnostics/stage9_hybrid_seccomp_probe.c` limit its completeness.
These advisory limitations remain visible; the dedicated new-artifact scan
completed with zero findings/errors and the separate blocking custom-rule
scope completed with zero findings. An exploratory changed-file scan used
`--error` with community rules and exited 1 for the two existing XML findings;
that result is not hidden or reported as a zero-finding PASS. The repository's
existing community-advisory/blocking-custom policy was preserved.

No existing test, scanner, import rule, scope, threshold, timeout, safety gate
or reconstruction blocker was disabled or weakened. No real CPython
reconstruction or installed platform qualification was performed.

## Reproduction

Use the repository test dependencies and pinned tool versions described in
[the toolchain](../quality-security-toolchain.md). CI runs this module in the
existing Observer regression/coverage scope. For isolated local execution:

```sh
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q \
  -p asyncio -p pytest_cov.plugin --confcutdir=tests/deployment \
  tests/deployment/test_stage9_observer_pyc_v5c1.py \
  --cov=deployment.stage9_observer_pyc_v5c1 --cov-branch --cov-report=xml
diff-cover coverage.xml --compare-branch origin/main --fail-under=70
python -m mypy
python -m mypy --platform win32
python -m pre_commit run --all-files
python -m pre_commit run ruff-format --all-files
python -m ruff format --check deployment/stage9_observer_pyc_v5c1.py \
  tests/deployment/test_stage9_observer_pyc_v5c1.py
```

Pre-commit's generic formatter excludes `deployment/`; the explicit workflow
format list includes this module and its tests. Both gates are required. A
hosted draft's checks must be assessed on its own exact HEAD, separately from
the baseline and local results recorded here.

```text
Stage 9 = IN_PROGRESS
Observer 3A qualification = BLOCKED
Stage 3B = NOT AUTHORIZED
Production = NO-GO
External authority acceptance = UNVERIFIED
Authenticated time = UNVERIFIED
Durable anti-rollback = BLOCKED
Runtime qualification = BLOCKED
PYC_RECONSTRUCTION = UNVERIFIED
```
