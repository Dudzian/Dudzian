# Observer 3A V4a — actual synthetic validation, 2026-11-14

Base: current remote `main`, `d762d7b813ce987f72c7e942569aae99822ca1d8`.
Branch: `feat/stage9-observer-apt-v4a`, created from that base after fetching.
Remote branch-name checks and GitHub searches for V4a, offline APT and Observer
3A found no existing V4a implementation PR to duplicate. This is a bounded
search observation, not an exhaustive repository-history claim.

The [technical contract](../stage9_observer_apt_v4a.md) defines the exact profile.
This report concerns local synthetic fixtures and source/tool checks. No actual
APT repository, physical host, historical operator capture or production runtime
was measured. Historical evidence and V1–V3 source/contracts were not edited.

Validated implementation SHA-256:
`79351d12ab9fedc040b766c430c1b47c54f00a7326bf2c1e823b4ba68618b1e9`.
Validated test-file SHA-256:
`f94201a918ac2df025dbbb9bdf0608e1440073d9f3173bd08872ddcd852a6f8a`.
These identify the tested files, without authenticating this checkout or tools.
The published commit HEAD is recorded separately in the Draft PR and delivery
report; documentation cannot contain its own final commit hash.

## Implemented comparison and refusals

The public frozen declaration and `verify_apt_provenance_v4a` compare selected
Release bytes, one Release-relative SHA256 index row, selected encoded index
bytes, exact six-field package stanzas, whole-DEB identities and existing V3
control/data inventories. The same no-follow descriptor spans DEB hashing and
archive comparison. Shared byte, decoded-byte, record, retained-memory and time
budgets cover the complete chain; the input tree is closed and checked again.
No successful partial package set survives a refusal or cleanup failure.

Accepted metadata is a deliberately narrow ASCII/LF UTF-8 subset: local
`Release` with one SHA256 row, and `Packages` or a single ordinary `Packages.gz`
member with flags zero and no trailing bytes. Packages has exactly Package,
Version, Architecture, Filename, SHA256 and Size. DEB filenames remain flat V3
basenames. General archive Release files, pool paths, additional algorithms,
metadata fields, InRelease/PGP, other index compression and guessing refuse.
This is an operational comparison slice, not general APT support.

An independent bounded source review identified one P2 implementation issue:
a deadline could expire during successful reader close after the last check.
The implementation now checks the budget after close and discards staged
measurements on expiry. A public-interface regression exercises that case.
No confirmed P0/P1/P2 implementation defect remained in the reviewed V4a scope;
this is not a repository-wide absence proof or independent tool acceptance.

## Tests actually run

Environment: unprivileged Linux, CPython 3.11.17, pytest 9.2.0, coverage.py 7.17.0,
existing repository `.venv`. No dependency lock or installed test environment
was changed. Intermediate logs/caches are local `work/apt-v4a-validation/`
artifacts and are not committed.

| Executed scope | Result |
| --- | --- |
| New public V4a suite | 193 passed |
| Existing Observer contract/offline/Zstandard/TAR V3/package V3 and V2/V3 reproduction suites | 1,387 passed unchanged |
| Final combined invocation of those eight suites | **1,580 passed**, 29.27 seconds, zero skipped/failures |
| Existing CI/workflow inventory contract tests | 138 passed, 23.37 seconds |
| New module branch-inclusive coverage | **100%**: 276/276 statements, 44/44 branches |
| Changed-code coverage against `origin/main` | **100%**, 276 changed executable lines; unchanged 70% gate passed |

The V4a cases cover complete raw/gzip index chains and V3 raw/gzip/Zstandard
DEBs; every link's hash/size/content; all six identity fields; duplicate and
conflicting fields/records/paths; traversal, symlinks, hardlinks, aliases and
encoding; unsupported algorithms/compression; bounded input, expansion, memory,
records and time; missing/extra inputs; short reads, disappearance, descriptor
changes, final-inventory and close failure; atomic failure and fixed trust
statuses. Exact-cap and cap-minus-one cases test shared index/TAR budgets.
Recomputed outer Release/index/DEB hashes with incorrect inner payload or
control identity still refuse. Partial preads are accumulated correctly.

The final combined command was:

```sh
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONDONTWRITEBYTECODE=1 \
COVERAGE_FILE=work/apt-v4a-validation/.coverage \
.venv/bin/python -m coverage run --branch \
  --source=deployment.stage9_observer_apt_v4a,deployment.stage9_observer_package_v3,deployment.stage9_observer_tar_v3,deployment.stage9_observer_ubuntu,deployment.stage9_observer_offline,deployment.stage9_observer_contract,scripts.verify_stage9_real_deb_v2,scripts.verify_stage9_real_deb_v3 \
  -m pytest -q -p no:cacheprovider \
  --basetemp=work/apt-v4a-validation/final-pytest \
  tests/test_stage9_observer_contract.py \
  tests/deployment/test_stage9_observer_offline.py \
  tests/deployment/test_stage9_observer_zstd.py \
  tests/deployment/test_stage9_observer_tar_v3.py \
  tests/deployment/test_stage9_observer_package_v3.py \
  tests/deployment/test_stage9_real_deb_v2.py \
  tests/deployment/test_stage9_real_deb_v3.py \
  tests/deployment/test_stage9_observer_apt_v4a.py
```

Coverage XML and `diff_cover.diff_cover_tool --compare-branch=origin/main
--fail-under=70` passed. Separate CI contract tests were
`tests/scripts/test_history_reproposal_quality_ci.py`,
`tests/scripts/test_stage9_safe_ci.py`,
`tests/scripts/test_s9_fv_01_junit_workflow.py` and
`tests/scripts/test_general_ci_platform_contract.py`.

## Quality and security tools actually run

The new module/test join existing configured mypy, explicit Ruff/format,
pytest and branch-coverage scopes. Previous inventory, thresholds, timeouts and
manual/hardware gates remain intact.

| Check | Actual outcome |
| --- | --- |
| Exact `python -m pre_commit run ruff-format --all-files` | Passed twice using `.venv/bin` on PATH and a writable workspace `PRE_COMMIT_HOME`; new files staged first; no unrelated tracked formatting changes |
| Ruff 0.14.11, full workflow-defined baseline | Passed |
| Explicit new module/test Ruff `E,F,B,I` and `ruff format --check` | Passed |
| Configured mypy 1.10.0 Linux and Win32, without filename narrowing | Passed, 397 source files each |
| YAML/TOML scope assertions and diff whitespace | Passed |
| Semgrep CE 1.179.0 custom blocking rules | Passed: five rules, 1,129 Python targets, zero findings; explicit new-module scan also zero |
| Semgrep custom rule fixtures | Passed, 5/5 |
| Maintained Semgrep community rules, explicit new V4a module | Passed: 200 applicable rules, one target, zero findings |
| Maintained community full configured source scope | Advisory/partial: exit 0, 219 applicable rules, 1,214 targets, 25 findings outside changed files, eight timeouts and one partial C parsing diagnostic; not a clean scan |
| Betterleaks 1.9.0 working tree and existing staged wrapper | Passed, zero findings, unchanged baseline/config; staged scan repeated after final documents |
| pip-audit 2.10.1 canonical desktop lock | Passed: 104 dependencies, zero known vulnerabilities, zero skipped |
| pip-audit exact offline Zstandard lock | Passed: one dependency, zero known vulnerabilities, zero skipped |
| Existing `.venv` `python -m pip check` | Passed |

Security commands follow [the maintained toolchain](../../quality-security-toolchain.md):

```sh
semgrep scan --config .semgrep/cryptohunter.yml --metrics=off --disable-version-check --error bot_core core deployment scripts
semgrep scan --test --config .semgrep/cryptohunter.yml .semgrep/cryptohunter.py --metrics=off --disable-version-check --jobs 1
betterleaks dir --baseline-path .betterleaks-baseline.json --redact=100 --no-banner --no-color --report-format json --report-path work/apt-v4a-validation/betterleaks-working-tree.json .
python scripts/quality/betterleaks_hook.py
python -m pip_audit --requirement deploy/packaging/requirements-desktop.lock --cache-dir work/apt-v4a-validation/audit-cache --format json --output work/apt-v4a-validation/desktop-audit.json
python -m pip_audit --requirement deployment/stage9-offline-verifier-cp311-manylinux-x86_64.lock --cache-dir work/apt-v4a-validation/audit-cache --format json --output work/apt-v4a-validation/zstd-audit.json
python -m pip check
```

Semgrep ran in an isolated tooling environment under `work/`; Betterleaks's
downloaded release matched the workflow-pinned SHA-256 and upstream checksums.
TLS and proxy settings were preserved. Initial audit cache errors from the
read-only home were resolved with a writable cache, without changing controls.
The first secret scan found only a downloaded wheel-cache JWT example and an
existing ignored generated fixture `.pyc`; those disposable generated artifacts
were deleted and the identical unchanged-baseline scan passed. No source finding
was suppressed. Codebase Memory tools/binary were unavailable; direct source
inspection and tests supplied evidence without structural-completeness claims.

The broad community scan began before the new module existed; its separate
final scan covers the exact module SHA-256 above with zero findings/errors.
The broad scan's partial parsing diagnostic concerns the unchanged
`scripts/diagnostics/stage9_hybrid_seccomp_probe.c`. Its existing exclusions,
file-size limits and timeouts were preserved. The 25 findings were not assigned
new severities or remediated in this scoped change; their presence and scan
timeouts prevent any repository-wide clean-security claim.

Local scans may access tool registries/advisory services during validation.
The V4a verification function has no such access; its synthetic authority test
forbids network and execution boundaries. Known-advisory scans do not authenticate
the toolchain or prove the absence of vulnerabilities. Remote exact-HEAD CI is
reported separately in the PR; a local pass is not an overall CI pass.

## Open risks and authorization boundary

| Severity / scope | Remaining limit |
| --- | --- |
| P0, bounded implementation review | None confirmed; no exhaustive absence or production qualification claim. |
| P1, source qualification | Publisher/signature authenticity, external roots, signer/subkey/algorithm policy, time/freshness and rollback are unimplemented V4b obligations. Agreeing attacker-supplied materials can intentionally yield local MATCH. |
| P1, runtime/image qualification | Application budgets are cooperative, not hard memory/syscall isolation; filesystem snapshots do not eliminate TOCTOU. Tool/host acceptance, runtime closure and V5 image/metadata mapping remain independent gates. |
| P2, coverage/usability | The provisioned slice rejects ordinary broader archive metadata and pool paths. No real repository, historical host evidence, multi-platform runtime or physical qualification was replayed. Broad advisory scan timeouts/findings and remote CI remain separately assessed. |

V4b is left for separately approved roots, lifecycle/time/checkpoint policy and
the pinned isolated `gpgv` adapter. No signature adapter or test root was needed
for V4a. V5 retains archive-to-image transformations, target metadata and image
acceptance. APT archive-key authority never becomes Observer reviewer authority.

```text
signature_verification=NOT_PERFORMED
source_authenticity=UNVERIFIED
acceptance=UNVERIFIED
stage_9=IN_PROGRESS
qualification=BLOCKED
STAGE_3A_COMPLETE=NO
PRODUCTION_QUALIFIED=NO
production_status=PRODUCTION_NO_GO
Stage 3B=NOT AUTHORIZED
```

The requested publication is a Draft PR only. No merge, Stage 3B activation,
installation, launcher or production qualification is performed.
