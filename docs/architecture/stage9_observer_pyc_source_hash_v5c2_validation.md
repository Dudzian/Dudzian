# V5c2 validation and preflight

Date: 2026-11-15 UTC. Base: `9d96a6d4ecf2d02e0d89126e389b2c771ed434f0`.
Branch: `codex/stage9-observer-v5c2-source-hash`. No merge is authorized or
performed. The [technical contract](stage9_observer_pyc_source_hash_v5c2.md),
[source-material report](evidence/stage9_observer_pyc_source_hash_v5c2_sources.md)
and [future reconstruction contract](stage9_observer_pyc_reconstruction_v5c2.md)
describe the precise evidence boundary.

## Baseline preflight

The local checkout was clean on an older branch. `origin/main` was fetched,
and the new branch was created directly from the base above. Repository
`AGENTS.md`, quality instructions, source and workflows were inspected.
Codebase Memory tools and CLI were unavailable; direct targeted source review
was used without making structural-index completeness claims. Both connected
GitHub PR inventory and the open-pulls API returned no open PRs.

PR #3274 merged at 09:13:17 UTC with head
`c6ea5a18d2e30bd1c4e2674d973c32f88e64d6c8`. Its completed
[CI](https://github.com/Dudzian/Dudzian/actions/runs/43663477091),
[Quality/Security](https://github.com/Dudzian/Dudzian/actions/runs/43663477070),
[CodeQL](https://github.com/Dudzian/Dudzian/actions/runs/43663477096) and
[Style](https://github.com/Dudzian/Dudzian/actions/runs/43663477085) passed.
Actual Ubuntu logs report 30385 PASS, 169 SKIP, 1539 deselected; Windows logs
report 26105 PASS, 4150 SKIP, 1559 deselected. Native Windows security has
29 PASS, with 191 PASS in a separate emulated set. Skips are not passes.

The [baseline platform run](https://github.com/Dudzian/Dudzian/actions/runs/43663477140)
failed unchanged qualification requirements. Actual Linux logs name missing
`LINUX_SYSTEMD_INTEGRATION`, `LINUX_CLEAN_INSTALL` and
`LINUX_REBOOT_LIFECYCLE_UPDATE`; actual Windows logs name missing
`WINDOWS_REBOOT_RECOVERY`, `WINDOWS_LONG_RUNNING_LIFECYCLE`,
`WINDOWS_UPDATE_RESTART` and `WINDOWS_FAILED_UPDATE_SAFE_BEHAVIOR`. The logs
match `deployment/platform_readiness.py` and existing platform-workflow gates.
These remain failed qualification checks; they are not labeled PASS or hidden.
No new scoped V5c1 regression was observed in completed baseline checks.

At 09:22:46 UTC, post-merge main
[CodeQL](https://github.com/Dudzian/Dudzian/actions/runs/43698184261) and
[Style](https://github.com/Dudzian/Dudzian/actions/runs/43698184199) passed;
native Windows security, lint and mypy jobs in
[main CI](https://github.com/Dudzian/Dudzian/actions/runs/43698184223) passed.
Broad Windows/Ubuntu and
[Quality/Security](https://github.com/Dudzian/Dudzian/actions/runs/43698184181)
were still running. Main CI's aggregate API said QUEUED while individual jobs
were running, so no aggregate main PASS was inferred. Pending checks were
retained as gaps; local V1–V5c1 regressions separately passed below.

## Local functional and quality results

Trusted verifier/test tooling uses host CPython **3.11.16**, Ruff **0.14.11**
and mypy **1.10.0**. It is not the target Ubuntu interpreter. Source review
identifies the exact Ubuntu archives, all active patches and the hash path;
20 static vectors come from a separately compiled hash-only C reference.
The exact target CPython was never invoked and its differential validation is
**NOT_TESTED**. No studied Python was compiled, imported or executed.

| Check | Observed result |
| --- | --- |
| Independent primitive/vector tests | 53 PASS |
| Actual V5b bytes → source-hash integration tests | 108 PASS |
| Future reconstruction structure tests | 55 PASS |
| New V5c2 tests total | 216 PASS, 0 SKIP |
| Baseline V1–V5c1/real-DEB regressions | 2594 PASS, 0 SKIP |
| Final combined V1–V5c2 and real-DEB regression run | 2810 PASS, 0 SKIP |
| CI/platform/quality/security/safe workflow contracts | 273 PASS, 0 SKIP |
| New wrapper and primitive statement/branch coverage | 100% each |
| Three new modules' statement/branch coverage | 99%; 486/488 statements, all 62 branch outcomes |
| Diff coverage against `origin/main`, unchanged minimum 70% | PASS, 99%; 486/488 changed measured lines |
| Full workflow Ruff baseline and strict new-code checks | PASS |
| Explicit workflow format scope | PASS, 347 files |
| Configured mypy Linux and `--platform win32` | PASS, 409 source files each; type checks, not native Windows |
| Exact `python -m pre_commit run ruff-format --all-files` | PASS, exit 0, no formatter changes |
| Full pre-commit | PASS, aggregate exit 0; Ruff, format, configured mypy and staged Betterleaks |
| Full offline Betterleaks, unchanged reviewed baseline | PASS, no leaks |
| Blocking custom Semgrep and five rule fixtures | PASS, zero findings/errors; 5/5 fixtures |
| Three new modules and tests under maintained + custom Semgrep | PASS, 205 applicable rules, zero findings/errors |
| Full maintained Semgrep advisory scan | Completed, 219 applicable rules/1227 files; 25 findings in unchanged files; seven timeouts and one partial parse limit completeness |

The two unexecuted statements are a defensive generic exception handler in
the reconstruction structure validator; all branch outcomes were exercised.
Tests explicitly cover invalid bytecode bodies with correct headers,
source/header byte replacement, wrong key/endian/algorithm, every source and
compiler identity, material substitution, missing/extra declarations, source
name collisions, independent pins, invalidation, wrong contract version,
data/memory/time limits, interruption after processing starts, second-file
failure and complete removal of partial positive evidence. Host hash,
filesystem/network/subprocess and code-execution facilities are replaced by
failures in the integration boundary test. Complete reconstruction declarations
and fictitious proof extensions cannot authorize reconstruction.

V4b tests use actual `gpgv`, extracted as trusted test tooling without package
installation. The verified Debian tool package is
`gpgv_2.4.7-21+deb13u1+b5_amd64.deb`, 241100 bytes, SHA-256
`104e4c57b98f0b883aa94c68fbe7ec7b48bcdec7c0f9dedf2a51a5054aaeb5ae`.
It is separate from the investigated Ubuntu artifacts. Betterleaks 1.9.0 was
checked against the existing workflow's archive SHA-256.

An initial full Betterleaks scan found only generated pytest node IDs with a
credential-URI test parameter and a bytecode-cache copy of an existing private
key fixture. Those generated caches were preserved outside the repo, then the
full scanner passed with the unchanged baseline. No source finding, new scanner
exclusion, suppression or secret-baseline entry was added. Final tests disable
bytecode and pytest cache writes.

The full maintained Semgrep scan is advisory under the unchanged workflow.
All 25 findings are outside the changed paths. Seven rule timeouts affect five
unchanged Python files, and a partial parse affects the existing
`scripts/diagnostics/stage9_hybrid_seccomp_probe.c`. This is not a clean or
complete analysis. The separate blocking custom-rule scope and dedicated new
implementation/test scope completed without findings or errors.

## Reproduction and hosted boundary

With the repository test/tool dependencies and trusted `gpgv` available:

```sh
PYTHONDONTWRITEBYTECODE=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
STAGE9_TEST_GPGV=/path/to/trusted/gpgv python -m pytest -q \
  -p no:cacheprovider -p asyncio -p pytest_cov.plugin \
  --confcutdir=tests/deployment tests/test_stage9_observer_contract.py \
  tests/deployment/test_stage9_observer*.py \
  tests/deployment/test_stage9_real_deb*.py \
  --cov=deployment.stage9_observer_pyc_source_hash_v5c2 \
  --cov=deployment.stage9_observer_source_hash_algorithm_v5c2 \
  --cov=deployment.stage9_observer_pyc_reconstruction_v5c2 \
  --cov-branch --cov-report=xml
diff-cover coverage.xml --compare-branch origin/main --fail-under=70
python -m mypy
python -m mypy --platform win32
python -m pre_commit run ruff-format --all-files
python -m pre_commit run --all-files
```

The pre-commit formatter excludes `deployment/`, so the explicit workflow
format list is also required and was run. Hosted checks must identify the
new draft's exact HEAD separately from the baseline. CodeQL and native Ubuntu/
Windows results on the new HEAD are pending until CI supplies them; this host
does not establish those native results. Physical/reboot/privileged qualification
workflows were not dispatched. No skips, coverage thresholds, limits, scanner
policies, authority/checkpoint storage or production contracts were weakened.

The next substage is actual reconstruction only after separate acceptance of
isolation and execution-evidence origins. V5c2 does not perform it, accept a
declaration as proof, or authorize Stage 3B. All project status invariants in
the technical contract remain unchanged, including `PYC_RECONSTRUCTION =
UNVERIFIED`, Observer qualification `BLOCKED` and Production `NO-GO`.
