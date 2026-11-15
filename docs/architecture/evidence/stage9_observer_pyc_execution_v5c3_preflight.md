# V5c3 current baseline CI report
Observed 2026-11-15T14:28:13.458915+00:00; local unauthenticated UTC clock. All run/job API pages inspected. No workflow dispatch, rerun or other remote mutation was performed.

This report supersedes the initial 09:46 UTC preflight snapshot. That earlier snapshot was accurate only while the relevant jobs were pending. Completed results now prove a new Windows V5c2 fixture-byte regression; baseline CI is not green.

## Confirmed new Windows regression

| Event / exact commit | Failed Windows job | Actual log summary |
| --- | --- | --- |
| Branch push / `35d8d1720c8d2083d5558922523e473688e939ce` | [132769398560](https://github.com/Dudzian/Dudzian/actions/runs/43699494348/job/132769398560) | 10:25:14 UTC: 1 failed, 26320 passed, 4150 skipped, 1559 deselected |
| PR #3275 / head `35d8d1720c8d2083d5558922523e473688e939ce` | [132770460555](https://github.com/Dudzian/Dudzian/actions/runs/43699570465/job/132770460555) | 11:04:07 UTC: 1 failed, 26320 passed, 4150 skipped, 1559 deselected |
| Main push / `cca8e8370d0124bb651a4722942843b14b559330` | [132779534621](https://github.com/Dudzian/Dudzian/actions/runs/43700382835/job/132779534621) | 11:31:16 UTC: 1 failed, 26320 passed, 4150 skipped, 1559 deselected |

All three failure summaries and exact assertion values were independently read using the connected GitHub decoded-job-log tool. Failing test: `tests/deployment/test_stage9_observer_source_hash_algorithm_v5c2.py::test_reference_provenance_and_material_identities_are_pinned`. The pinned C reference was checked out with CRLF:

```text
expected LF SHA-256 = 3b3686ea5c9187d5ee36ff4f69db159775b5e9f4462eecb95d1195d280e7c488
actual CRLF SHA-256 = 65512132c96579b8df11f8877f62fc1db3a9073fc205cc6dfb89907299c54d06
```

The current working change adds exactly one file-specific `.gitattributes` rule:

```gitattributes
tests/fixtures/stage9_observer/pyc_source_hash_v5c2/cpython_siphash13_reference.c text eol=lf
```

The C reference bytes, frozen vectors and expected digest are unchanged. The root agent separately reproduced the byte conversion with disposable Git checkouts and observed 216 local V5c2 regressions passing. Those local checks do not establish native Windows success on the new V5c3 HEAD. Hosted native Windows validation of the new HEAD remains pending until its own CI runs complete.

## Ubuntu, Quality, CodeQL and Windows build

- Branch, PR and main broad Ubuntu jobs all SUCCESS, together with their three PostgreSQL qualification shards.
- PR Ubuntu [132770460628](https://github.com/Dudzian/Dudzian/actions/runs/43699570465/job/132770460628) and main Ubuntu [132779534629](https://github.com/Dudzian/Dudzian/actions/runs/43700382835/job/132779534629): actual decoded logs each report **30601 passed, 169 skipped, 1539 deselected**, plus three warnings. Skips are not passes.
- PR Quality/Security [43699570439](https://github.com/Dudzian/Dudzian/actions/runs/43699570439) and main Quality/Security [43700382815](https://github.com/Dudzian/Dudzian/actions/runs/43700382815): SUCCESS, including combined coverage jobs.
- PR CodeQL [43699570445](https://github.com/Dudzian/Dudzian/actions/runs/43699570445) and main CodeQL [43700382822](https://github.com/Dudzian/Dudzian/actions/runs/43700382822): SUCCESS for both Python and JavaScript/TypeScript.
- PR Style [43699570463](https://github.com/Dudzian/Dudzian/actions/runs/43699570463), main Style [43700382936](https://github.com/Dudzian/Dudzian/actions/runs/43700382936), main Windows Build [43700382902](https://github.com/Dudzian/Dudzian/actions/runs/43700382902): SUCCESS.
- Narrow native Windows security jobs succeeded separately; actual initial logs for PR/main each contained 29 native passes and 191 emulated passes. These do not replace the failed broad Windows job.
- Scheduled main CI [43703009584](https://github.com/Dudzian/Dudzian/actions/runs/43703009584) is SUCCESS under a separate event, but its broad Ubuntu and Windows test jobs were **SKIPPED**. It does not repair or supersede the failed main-push Windows test.

## Existing platform qualification refusals

Current PR #3275 platform run [43699570490](https://github.com/Dudzian/Dudzian/actions/runs/43699570490) is FAILURE for the following independently retrieved actual log reasons:

- Linux [132770262616](https://github.com/Dudzian/Dudzian/actions/runs/43699570490/job/132770262616), 10:02:42 UTC: `LINUX_SYSTEMD_INTEGRATION`, `LINUX_CLEAN_INSTALL`, `LINUX_REBOOT_LIFECYCLE_UPDATE` = `MISSING_CURRENT_EVIDENCE`; `LINUX_PRODUCTION_READY=NOT_READY`; exit 1.
- Windows release [132790475344](https://github.com/Dudzian/Dudzian/actions/runs/43699570490/job/132790475344), 11:08:10 UTC: missing `WINDOWS_REBOOT_RECOVERY`, `WINDOWS_LONG_RUNNING_LIFECYCLE`, `WINDOWS_UPDATE_RESTART`, `WINDOWS_FAILED_UPDATE_SAFE_BEHAVIOR`; `WINDOWS_BETA_RELEASE_READY=NOT_READY`; exit 1.

These are the same required evidence names observed in PR #3274 baseline platform jobs [132661596554](https://github.com/Dudzian/Dudzian/actions/runs/43663477140/job/132661596554) and [132676735660](https://github.com/Dudzian/Dudzian/actions/runs/43663477140/job/132676735660). Required evidence is confirmed in `deployment/platform_readiness.py` lines 68–94. The workflow keeps physical Linux and destructive/reboot Stage 10 execution behind explicit manual inputs. No such workflow was dispatched. These platform failures remain FAILED qualification; they are distinct from the new byte-integrity test regression above.

## All observed workflows

| Scope | Event | Workflow | Run | Status | Conclusion |
| --- | --- | --- | --- | --- | --- |
| PR #3275 head | pull_request | CodeQL | [43699570445](https://github.com/Dudzian/Dudzian/actions/runs/43699570445) | completed | success |
| PR #3275 head | pull_request | Windows Stage 10 — obserwator ciągłości | [43699570502](https://github.com/Dudzian/Dudzian/actions/runs/43699570502) | completed | success |
| PR #3275 head | pull_request | Style preflight | [43699570463](https://github.com/Dudzian/Dudzian/actions/runs/43699570463) | completed | success |
| PR #3275 head | pull_request | Windows Stage 10 proof validation | [43699570458](https://github.com/Dudzian/Dudzian/actions/runs/43699570458) | completed | success |
| PR #3275 head | pull_request | Marketplace Catalog | [43699570476](https://github.com/Dudzian/Dudzian/actions/runs/43699570476) | completed | success |
| PR #3275 head | pull_request | Platform deployment tracks | [43699570490](https://github.com/Dudzian/Dudzian/actions/runs/43699570490) | completed | failure |
| PR #3275 head | pull_request | Quality and Security | [43699570439](https://github.com/Dudzian/Dudzian/actions/runs/43699570439) | completed | success |
| PR #3275 head | pull_request | CI (pull_request: 3275/merge) | [43699570465](https://github.com/Dudzian/Dudzian/actions/runs/43699570465) | completed | failure |
| PR #3275 head | push | CI (push: codex/stage9-observer-v5c2-source-hash) | [43699494348](https://github.com/Dudzian/Dudzian/actions/runs/43699494348) | completed | failure |
| Merged main | schedule | DR Checks | [43711198943](https://github.com/Dudzian/Dudzian/actions/runs/43711198943) | completed | success |
| Merged main | schedule | exchange-report | [43709736215](https://github.com/Dudzian/Dudzian/actions/runs/43709736215) | completed | success |
| Merged main | schedule | CI (schedule: main) | [43703009584](https://github.com/Dudzian/Dudzian/actions/runs/43703009584) | completed | success |
| Merged main | schedule | Cleanup old artifacts | [43702942819](https://github.com/Dudzian/Dudzian/actions/runs/43702942819) | completed | success |
| Merged main | dynamic | Configured Graph Update: pip in /. #1705399063 | [43700388205](https://github.com/Dudzian/Dudzian/actions/runs/43700388205) | completed | success |
| Merged main | push | CodeQL | [43700382822](https://github.com/Dudzian/Dudzian/actions/runs/43700382822) | completed | success |
| Merged main | push | CI (push: main) | [43700382835](https://github.com/Dudzian/Dudzian/actions/runs/43700382835) | completed | failure |
| Merged main | push | Marketplace Catalog | [43700382826](https://github.com/Dudzian/Dudzian/actions/runs/43700382826) | completed | success |
| Merged main | push | Stress Lab Report | [43700382834](https://github.com/Dudzian/Dudzian/actions/runs/43700382834) | completed | success |
| Merged main | push | Style preflight | [43700382936](https://github.com/Dudzian/Dudzian/actions/runs/43700382936) | completed | success |
| Merged main | push | Windows Build | [43700382902](https://github.com/Dudzian/Dudzian/actions/runs/43700382902) | completed | success |
| Merged main | push | Quality and Security | [43700382815](https://github.com/Dudzian/Dudzian/actions/runs/43700382815) | completed | success |

## All observed jobs

### PR #3275 head: CodeQL (43699570445, pull_request)

| Job | Status | Conclusion |
| --- | --- | --- |
| [CodeQL (python)](https://github.com/Dudzian/Dudzian/actions/runs/43699570445/job/132767522905) | completed | success |
| [CodeQL (javascript-typescript)](https://github.com/Dudzian/Dudzian/actions/runs/43699570445/job/132767523104) | completed | success |

### PR #3275 head: Windows Stage 10 — obserwator ciągłości (43699570502, pull_request)

| Job | Status | Conclusion |
| --- | --- | --- |
| [Obserwator procesu (ubuntu-latest)](https://github.com/Dudzian/Dudzian/actions/runs/43699570502/job/132767522981) | completed | success |
| [Obserwator procesu (windows-latest)](https://github.com/Dudzian/Dudzian/actions/runs/43699570502/job/132767523166) | completed | success |
| [Obserwator procesu (macos-latest)](https://github.com/Dudzian/Dudzian/actions/runs/43699570502/job/132767523227) | completed | success |

### PR #3275 head: Style preflight (43699570463, pull_request)

| Job | Status | Conclusion |
| --- | --- | --- |
| [Ruff format (all tracked files)](https://github.com/Dudzian/Dudzian/actions/runs/43699570463/job/132767523196) | completed | success |

### PR #3275 head: Windows Stage 10 proof validation (43699570458, pull_request)

| Job | Status | Conclusion |
| --- | --- | --- |
| [Odmowy Stage 10 (windows-latest)](https://github.com/Dudzian/Dudzian/actions/runs/43699570458/job/132767522851) | completed | success |
| [Odmowy Stage 10 (macos-latest)](https://github.com/Dudzian/Dudzian/actions/runs/43699570458/job/132767523018) | completed | success |
| [Odmowy Stage 10 (ubuntu-latest)](https://github.com/Dudzian/Dudzian/actions/runs/43699570458/job/132767523041) | completed | success |

### PR #3275 head: Marketplace Catalog (43699570476, pull_request)

| Job | Status | Conclusion |
| --- | --- | --- |
| [Validate marketplace presets](https://github.com/Dudzian/Dudzian/actions/runs/43699570476/job/132767523010) | completed | success |

### PR #3275 head: Platform deployment tracks (43699570490, pull_request)

| Job | Status | Conclusion |
| --- | --- | --- |
| [linux-evidence-contract](https://github.com/Dudzian/Dudzian/actions/runs/43699570490/job/132767523525) | completed | success |
| [core-macos](https://github.com/Dudzian/Dudzian/actions/runs/43699570490/job/132767523718) | completed | success |
| [core-windows](https://github.com/Dudzian/Dudzian/actions/runs/43699570490/job/132767523725) | completed | success |
| [macOS deployment compatibility (production deferred)](https://github.com/Dudzian/Dudzian/actions/runs/43699570490/job/132767523751) | completed | success |
| [windows-deployment-contract](https://github.com/Dudzian/Dudzian/actions/runs/43699570490/job/132767523755) | completed | success |
| [core-linux](https://github.com/Dudzian/Dudzian/actions/runs/43699570490/job/132767523756) | completed | success |
| [linux-physical-challenge](https://github.com/Dudzian/Dudzian/actions/runs/43699570490/job/132767524315) | completed | skipped |
| [linux-physical-validation](https://github.com/Dudzian/Dudzian/actions/runs/43699570490/job/132767524397) | completed | skipped |
| [Stage 9 safe checks (Python ${{ matrix.python }})](https://github.com/Dudzian/Dudzian/actions/runs/43699570490/job/132767524574) | completed | skipped |
| [windows-deployment-integration](https://github.com/Dudzian/Dudzian/actions/runs/43699570490/job/132770254841) | completed | success |
| [linux-deployment-integration](https://github.com/Dudzian/Dudzian/actions/runs/43699570490/job/132770262616) | completed | failure |
| [linux-production-gate](https://github.com/Dudzian/Dudzian/actions/runs/43699570490/job/132770263970) | completed | skipped |
| [windows-clean-install-integration](https://github.com/Dudzian/Dudzian/actions/runs/43699570490/job/132772423494) | completed | success |
| [windows-stage10-prepare-and-schedule-reboot](https://github.com/Dudzian/Dudzian/actions/runs/43699570490/job/132790475343) | completed | skipped |
| [windows-release-gate](https://github.com/Dudzian/Dudzian/actions/runs/43699570490/job/132790475344) | completed | failure |
| [windows-stage10-reboot-barrier](https://github.com/Dudzian/Dudzian/actions/runs/43699570490/job/132790475749) | completed | skipped |
| [windows-stage10-qualify-after-reboot](https://github.com/Dudzian/Dudzian/actions/runs/43699570490/job/132790476093) | completed | skipped |

### PR #3275 head: Quality and Security (43699570439, pull_request)

| Job | Status | Conclusion |
| --- | --- | --- |
| [PostgreSQL atomic Root Proof commit](https://github.com/Dudzian/Dudzian/actions/runs/43699570439/job/132767522973) | completed | success |
| [Quality ratchet (PostgreSQL remainder)](https://github.com/Dudzian/Dudzian/actions/runs/43699570439/job/132767523150) | completed | success |
| [Semgrep and working-tree secrets](https://github.com/Dudzian/Dudzian/actions/runs/43699570439/job/132767523174) | completed | success |
| [History Successor Reproposal qualification](https://github.com/Dudzian/Dudzian/actions/runs/43699570439/job/132767523184) | completed | success |
| [Semantic issuer preflight](https://github.com/Dudzian/Dudzian/actions/runs/43699570439/job/132767523237) | completed | success |
| [Quality ratchet](https://github.com/Dudzian/Dudzian/actions/runs/43699570439/job/132767523258) | completed | success |
| [Full Git history secrets (advisory baseline)](https://github.com/Dudzian/Dudzian/actions/runs/43699570439/job/132767523836) | completed | skipped |
| [PostgreSQL atomic Root Proof and history coverage gate](https://github.com/Dudzian/Dudzian/actions/runs/43699570439/job/132787811455) | completed | success |
| [Quality ratchet (combined coverage gate)](https://github.com/Dudzian/Dudzian/actions/runs/43699570439/job/132791534779) | completed | success |

### PR #3275 head: CI (pull_request: 3275/merge) (43699570465, pull_request)

| Job | Status | Conclusion |
| --- | --- | --- |
| [Marketplace catalog validation](https://github.com/Dudzian/Dudzian/actions/runs/43699570465/job/132767524056) | completed | success |
| [Native Windows x64 security (no MSI)](https://github.com/Dudzian/Dudzian/actions/runs/43699570465/job/132767524175) | completed | success |
| [Generate SBOM](https://github.com/Dudzian/Dudzian/actions/runs/43699570465/job/132767524281) | completed | success |
| [Dependency vulnerability scan](https://github.com/Dudzian/Dudzian/actions/runs/43699570465/job/132767524291) | completed | success |
| [Prepare wheelhouse (ubuntu-latest)](https://github.com/Dudzian/Dudzian/actions/runs/43699570465/job/132767524307) | completed | success |
| [Prepare wheelhouse (macos-latest)](https://github.com/Dudzian/Dudzian/actions/runs/43699570465/job/132767524308) | completed | success |
| [Prepare wheelhouse (windows-latest)](https://github.com/Dudzian/Dudzian/actions/runs/43699570465/job/132767524342) | completed | success |
| [Lint and Test](https://github.com/Dudzian/Dudzian/actions/runs/43699570465/job/132770460528) | completed | success |
| [Python tests (Windows)](https://github.com/Dudzian/Dudzian/actions/runs/43699570465/job/132770460555) | completed | failure |
| [Licensing HWID Tolerance](https://github.com/Dudzian/Dudzian/actions/runs/43699570465/job/132770460560) | completed | success |
| [Mypy (core packages)](https://github.com/Dudzian/Dudzian/actions/runs/43699570465/job/132770460583) | completed | success |
| [Release Quality Gates](https://github.com/Dudzian/Dudzian/actions/runs/43699570465/job/132770460604) | completed | success |
| [Python tests (Ubuntu PostgreSQL qualification, issuer-durability)](https://github.com/Dudzian/Dudzian/actions/runs/43699570465/job/132770460609) | completed | success |
| [gRPC Decision Feed Integration](https://github.com/Dudzian/Dudzian/actions/runs/43699570465/job/132770460623) | completed | success |
| [Python tests (Ubuntu)](https://github.com/Dudzian/Dudzian/actions/runs/43699570465/job/132770460628) | completed | success |
| [Python tests (Ubuntu PostgreSQL qualification, history-reproposal)](https://github.com/Dudzian/Dudzian/actions/runs/43699570465/job/132770460642) | completed | success |
| [Python tests (Ubuntu PostgreSQL qualification, remaining-security)](https://github.com/Dudzian/Dudzian/actions/runs/43699570465/job/132770460668) | completed | success |
| [Frontend parity runtime proof](https://github.com/Dudzian/Dudzian/actions/runs/43699570465/job/132770460673) | completed | success |
| [UI QML Tests](https://github.com/Dudzian/Dudzian/actions/runs/43699570465/job/132770460682) | completed | success |
| [UI Packaging](https://github.com/Dudzian/Dudzian/actions/runs/43699570465/job/132770460700) | completed | success |
| [UI Native Tests](https://github.com/Dudzian/Dudzian/actions/runs/43699570465/job/132770460719) | completed | success |
| [Python tests (macOS)](https://github.com/Dudzian/Dudzian/actions/runs/43699570465/job/132770460745) | completed | success |
| [Licensing security/durability (windows-latest)](https://github.com/Dudzian/Dudzian/actions/runs/43699570465/job/132770460787) | completed | success |
| [Licensing security/durability (windows-latest, lppi-durability)](https://github.com/Dudzian/Dudzian/actions/runs/43699570465/job/132770460788) | completed | success |
| [Licensing security/durability (macos-latest)](https://github.com/Dudzian/Dudzian/actions/runs/43699570465/job/132770460792) | completed | success |
| [Licensing security/durability (windows-latest, pdsa-production)](https://github.com/Dudzian/Dudzian/actions/runs/43699570465/job/132770460804) | completed | success |
| [Licensing security/durability (windows-latest, cha-root-proof)](https://github.com/Dudzian/Dudzian/actions/runs/43699570465/job/132770460816) | completed | success |
| [Licensing security/durability (windows-latest, cha-signed-attempt)](https://github.com/Dudzian/Dudzian/actions/runs/43699570465/job/132770460820) | completed | success |
| [Licensing security/durability (ubuntu-latest)](https://github.com/Dudzian/Dudzian/actions/runs/43699570465/job/132770460824) | completed | success |
| [Licensing security/durability (windows-latest, cha-core)](https://github.com/Dudzian/Dudzian/actions/runs/43699570465/job/132770460850) | completed | success |
| [gRPC Feed Soak (cron)](https://github.com/Dudzian/Dudzian/actions/runs/43699570465/job/132770461635) | completed | skipped |
| [Performance Benchmarks](https://github.com/Dudzian/Dudzian/actions/runs/43699570465/job/132770461714) | completed | skipped |
| [Marketplace Catalog Packaging](https://github.com/Dudzian/Dudzian/actions/runs/43699570465/job/132782535411) | completed | success |
| [Exchange HyperCare Report](https://github.com/Dudzian/Dudzian/actions/runs/43699570465/job/132782535435) | completed | success |
| [Marketing bundle parity](https://github.com/Dudzian/Dudzian/actions/runs/43699570465/job/132789991858) | completed | success |
| [Loopback E2E](https://github.com/Dudzian/Dudzian/actions/runs/43699570465/job/132790480838) | completed | skipped |
| [UI / QML tests (Windows)](https://github.com/Dudzian/Dudzian/actions/runs/43699570465/job/132790481338) | completed | skipped |
| [UI / QML tests (macOS)](https://github.com/Dudzian/Dudzian/actions/runs/43699570465/job/132790699619) | completed | success |
| [Release marketplace assets](https://github.com/Dudzian/Dudzian/actions/runs/43699570465/job/132792284277) | completed | skipped |
| [Integration tests (Linux)](https://github.com/Dudzian/Dudzian/actions/runs/43699570465/job/132799059652) | completed | success |

### PR #3275 head: CI (push: codex/stage9-observer-v5c2-source-hash) (43699494348, push)

| Job | Status | Conclusion |
| --- | --- | --- |
| [Dependency vulnerability scan](https://github.com/Dudzian/Dudzian/actions/runs/43699494348/job/132767293056) | completed | success |
| [Marketplace catalog validation](https://github.com/Dudzian/Dudzian/actions/runs/43699494348/job/132767293267) | completed | success |
| [Prepare wheelhouse (macos-latest)](https://github.com/Dudzian/Dudzian/actions/runs/43699494348/job/132767293268) | completed | success |
| [Native Windows x64 security (no MSI)](https://github.com/Dudzian/Dudzian/actions/runs/43699494348/job/132767293274) | completed | success |
| [Generate SBOM](https://github.com/Dudzian/Dudzian/actions/runs/43699494348/job/132767293353) | completed | success |
| [Prepare wheelhouse (windows-latest)](https://github.com/Dudzian/Dudzian/actions/runs/43699494348/job/132767293358) | completed | success |
| [Prepare wheelhouse (ubuntu-latest)](https://github.com/Dudzian/Dudzian/actions/runs/43699494348/job/132767293363) | completed | success |
| [Release Quality Gates](https://github.com/Dudzian/Dudzian/actions/runs/43699494348/job/132769398504) | completed | success |
| [Mypy (core packages)](https://github.com/Dudzian/Dudzian/actions/runs/43699494348/job/132769398505) | completed | success |
| [Licensing HWID Tolerance](https://github.com/Dudzian/Dudzian/actions/runs/43699494348/job/132769398512) | completed | success |
| [Lint and Test](https://github.com/Dudzian/Dudzian/actions/runs/43699494348/job/132769398522) | completed | success |
| [Python tests (Windows)](https://github.com/Dudzian/Dudzian/actions/runs/43699494348/job/132769398560) | completed | failure |
| [Python tests (Ubuntu)](https://github.com/Dudzian/Dudzian/actions/runs/43699494348/job/132769398604) | completed | success |
| [UI Native Tests](https://github.com/Dudzian/Dudzian/actions/runs/43699494348/job/132769398607) | completed | success |
| [UI QML Tests](https://github.com/Dudzian/Dudzian/actions/runs/43699494348/job/132769398619) | completed | success |
| [UI Packaging](https://github.com/Dudzian/Dudzian/actions/runs/43699494348/job/132769398631) | completed | success |
| [Licensing security/durability (windows-latest, cha-core)](https://github.com/Dudzian/Dudzian/actions/runs/43699494348/job/132769398769) | completed | success |
| [Python tests (Ubuntu PostgreSQL qualification, issuer-durability)](https://github.com/Dudzian/Dudzian/actions/runs/43699494348/job/132769398774) | completed | success |
| [Python tests (macOS)](https://github.com/Dudzian/Dudzian/actions/runs/43699494348/job/132769398777) | completed | success |
| [Python tests (Ubuntu PostgreSQL qualification, history-reproposal)](https://github.com/Dudzian/Dudzian/actions/runs/43699494348/job/132769398793) | completed | success |
| [Licensing security/durability (windows-latest, pdsa-production)](https://github.com/Dudzian/Dudzian/actions/runs/43699494348/job/132769398802) | completed | success |
| [gRPC Decision Feed Integration](https://github.com/Dudzian/Dudzian/actions/runs/43699494348/job/132769398804) | completed | success |
| [Licensing security/durability (macos-latest)](https://github.com/Dudzian/Dudzian/actions/runs/43699494348/job/132769398822) | completed | success |
| [Frontend parity runtime proof](https://github.com/Dudzian/Dudzian/actions/runs/43699494348/job/132769398824) | completed | success |
| [Licensing security/durability (windows-latest, lppi-durability)](https://github.com/Dudzian/Dudzian/actions/runs/43699494348/job/132769398844) | completed | success |
| [Python tests (Ubuntu PostgreSQL qualification, remaining-security)](https://github.com/Dudzian/Dudzian/actions/runs/43699494348/job/132769398846) | completed | success |
| [Licensing security/durability (windows-latest, cha-signed-attempt)](https://github.com/Dudzian/Dudzian/actions/runs/43699494348/job/132769398859) | completed | success |
| [Licensing security/durability (ubuntu-latest)](https://github.com/Dudzian/Dudzian/actions/runs/43699494348/job/132769398869) | completed | success |
| [Licensing security/durability (windows-latest, cha-root-proof)](https://github.com/Dudzian/Dudzian/actions/runs/43699494348/job/132769398883) | completed | success |
| [Licensing security/durability (windows-latest)](https://github.com/Dudzian/Dudzian/actions/runs/43699494348/job/132769398909) | completed | success |
| [gRPC Feed Soak (cron)](https://github.com/Dudzian/Dudzian/actions/runs/43699494348/job/132769399931) | completed | skipped |
| [Performance Benchmarks](https://github.com/Dudzian/Dudzian/actions/runs/43699494348/job/132769400033) | completed | skipped |
| [Marketplace Catalog Packaging](https://github.com/Dudzian/Dudzian/actions/runs/43699494348/job/132771421546) | completed | success |
| [Exchange HyperCare Report](https://github.com/Dudzian/Dudzian/actions/runs/43699494348/job/132771421583) | completed | success |
| [UI / QML tests (macOS)](https://github.com/Dudzian/Dudzian/actions/runs/43699494348/job/132773090745) | completed | success |
| [Marketing bundle parity](https://github.com/Dudzian/Dudzian/actions/runs/43699494348/job/132779029363) | completed | success |
| [Loopback E2E](https://github.com/Dudzian/Dudzian/actions/runs/43699494348/job/132781056755) | completed | skipped |
| [UI / QML tests (Windows)](https://github.com/Dudzian/Dudzian/actions/runs/43699494348/job/132781056901) | completed | skipped |
| [Release marketplace assets](https://github.com/Dudzian/Dudzian/actions/runs/43699494348/job/132783213253) | completed | skipped |
| [Integration tests (Linux)](https://github.com/Dudzian/Dudzian/actions/runs/43699494348/job/132792314515) | completed | success |

### Merged main: DR Checks (43711198943, schedule)

| Job | Status | Conclusion |
| --- | --- | --- |
| [Weekly DR failover snapshots](https://github.com/Dudzian/Dudzian/actions/runs/43711198943/job/132802912132) | completed | success |
| [DR weekly failover (runbook)](https://github.com/Dudzian/Dudzian/actions/runs/43711198943/job/132802912346) | completed | success |
| [Cloud DR synthetic probes](https://github.com/Dudzian/Dudzian/actions/runs/43711198943/job/132802912452) | completed | success |
| [Alerting DR failover validation](https://github.com/Dudzian/Dudzian/actions/runs/43711198943/job/132802913577) | completed | skipped |

### Merged main: exchange-report (43709736215, schedule)

| Job | Status | Conclusion |
| --- | --- | --- |
| [Generate exchange signal-quality snapshots](https://github.com/Dudzian/Dudzian/actions/runs/43709736215/job/132798472783) | completed | success |

### Merged main: CI (schedule: main) (43703009584, schedule)

| Job | Status | Conclusion |
| --- | --- | --- |
| [Native Windows x64 security (no MSI)](https://github.com/Dudzian/Dudzian/actions/runs/43703009584/job/132778011013) | completed | success |
| [Prepare wheelhouse (windows-latest)](https://github.com/Dudzian/Dudzian/actions/runs/43703009584/job/132778011202) | completed | success |
| [Dependency vulnerability scan](https://github.com/Dudzian/Dudzian/actions/runs/43703009584/job/132778011255) | completed | success |
| [Generate SBOM](https://github.com/Dudzian/Dudzian/actions/runs/43703009584/job/132778011293) | completed | success |
| [Prepare wheelhouse (ubuntu-latest)](https://github.com/Dudzian/Dudzian/actions/runs/43703009584/job/132778011401) | completed | success |
| [Prepare wheelhouse (macos-latest)](https://github.com/Dudzian/Dudzian/actions/runs/43703009584/job/132778011417) | completed | success |
| [Marketplace catalog validation](https://github.com/Dudzian/Dudzian/actions/runs/43703009584/job/132778012046) | completed | skipped |
| [gRPC Feed Soak (cron)](https://github.com/Dudzian/Dudzian/actions/runs/43703009584/job/132786046509) | completed | success |
| [Performance Benchmarks](https://github.com/Dudzian/Dudzian/actions/runs/43703009584/job/132786046611) | completed | success |
| [Lint and Test](https://github.com/Dudzian/Dudzian/actions/runs/43703009584/job/132786047237) | completed | skipped |
| [gRPC Decision Feed Integration](https://github.com/Dudzian/Dudzian/actions/runs/43703009584/job/132786047427) | completed | skipped |
| [Python tests (Ubuntu)](https://github.com/Dudzian/Dudzian/actions/runs/43703009584/job/132786047572) | completed | skipped |
| [UI Native Tests](https://github.com/Dudzian/Dudzian/actions/runs/43703009584/job/132786047589) | completed | skipped |
| [Release Quality Gates](https://github.com/Dudzian/Dudzian/actions/runs/43703009584/job/132786047593) | completed | skipped |
| [Python tests (macOS)](https://github.com/Dudzian/Dudzian/actions/runs/43703009584/job/132786047657) | completed | skipped |
| [Mypy (core packages)](https://github.com/Dudzian/Dudzian/actions/runs/43703009584/job/132786047695) | completed | skipped |
| [Frontend parity runtime proof](https://github.com/Dudzian/Dudzian/actions/runs/43703009584/job/132786047701) | completed | skipped |
| [Python tests (Windows)](https://github.com/Dudzian/Dudzian/actions/runs/43703009584/job/132786047855) | completed | skipped |
| [Licensing HWID Tolerance](https://github.com/Dudzian/Dudzian/actions/runs/43703009584/job/132786047898) | completed | skipped |
| [UI QML Tests](https://github.com/Dudzian/Dudzian/actions/runs/43703009584/job/132786047940) | completed | skipped |
| [Marketing bundle parity](https://github.com/Dudzian/Dudzian/actions/runs/43703009584/job/132786047964) | completed | skipped |
| [Exchange HyperCare Report](https://github.com/Dudzian/Dudzian/actions/runs/43703009584/job/132786048019) | completed | skipped |
| [Python tests (Ubuntu PostgreSQL qualification, ${{ matrix.shard }})](https://github.com/Dudzian/Dudzian/actions/runs/43703009584/job/132786048040) | completed | skipped |
| [UI / QML tests (Windows)](https://github.com/Dudzian/Dudzian/actions/runs/43703009584/job/132786048052) | completed | skipped |
| [UI Packaging](https://github.com/Dudzian/Dudzian/actions/runs/43703009584/job/132786048055) | completed | skipped |
| [UI / QML tests (macOS)](https://github.com/Dudzian/Dudzian/actions/runs/43703009584/job/132786048171) | completed | skipped |
| [Licensing security/durability (${{ matrix.label }})](https://github.com/Dudzian/Dudzian/actions/runs/43703009584/job/132786048177) | completed | skipped |
| [Marketplace Catalog Packaging](https://github.com/Dudzian/Dudzian/actions/runs/43703009584/job/132786048287) | completed | skipped |
| [Release marketplace assets](https://github.com/Dudzian/Dudzian/actions/runs/43703009584/job/132786048332) | completed | skipped |
| [Loopback E2E](https://github.com/Dudzian/Dudzian/actions/runs/43703009584/job/132786048333) | completed | skipped |
| [Integration tests (Linux)](https://github.com/Dudzian/Dudzian/actions/runs/43703009584/job/132786048429) | completed | skipped |

### Merged main: Cleanup old artifacts (43702942819, schedule)

| Job | Status | Conclusion |
| --- | --- | --- |
| [delete-old-artifacts](https://github.com/Dudzian/Dudzian/actions/runs/43702942819/job/132777808468) | completed | success |

### Merged main: Configured Graph Update: pip in /. #1705399063 (43700388205, dynamic)

| Job | Status | Conclusion |
| --- | --- | --- |
| [update-pip-graph](https://github.com/Dudzian/Dudzian/actions/runs/43700388205/job/132770025697) | completed | success |

### Merged main: CodeQL (43700382822, push)

| Job | Status | Conclusion |
| --- | --- | --- |
| [CodeQL (python)](https://github.com/Dudzian/Dudzian/actions/runs/43700382822/job/132770006917) | completed | success |
| [CodeQL (javascript-typescript)](https://github.com/Dudzian/Dudzian/actions/runs/43700382822/job/132770006938) | completed | success |

### Merged main: CI (push: main) (43700382835, push)

| Job | Status | Conclusion |
| --- | --- | --- |
| [Marketplace catalog validation](https://github.com/Dudzian/Dudzian/actions/runs/43700382835/job/132770091199) | completed | success |
| [Native Windows x64 security (no MSI)](https://github.com/Dudzian/Dudzian/actions/runs/43700382835/job/132770091318) | completed | success |
| [Prepare wheelhouse (ubuntu-latest)](https://github.com/Dudzian/Dudzian/actions/runs/43700382835/job/132770091339) | completed | success |
| [Prepare wheelhouse (macos-latest)](https://github.com/Dudzian/Dudzian/actions/runs/43700382835/job/132770091345) | completed | success |
| [Prepare wheelhouse (windows-latest)](https://github.com/Dudzian/Dudzian/actions/runs/43700382835/job/132770091352) | completed | success |
| [Generate SBOM](https://github.com/Dudzian/Dudzian/actions/runs/43700382835/job/132770091365) | completed | success |
| [Dependency vulnerability scan](https://github.com/Dudzian/Dudzian/actions/runs/43700382835/job/132770091440) | completed | success |
| [Python tests (Windows)](https://github.com/Dudzian/Dudzian/actions/runs/43700382835/job/132779534621) | completed | failure |
| [Python tests (Ubuntu)](https://github.com/Dudzian/Dudzian/actions/runs/43700382835/job/132779534629) | completed | success |
| [Release Quality Gates](https://github.com/Dudzian/Dudzian/actions/runs/43700382835/job/132779534663) | completed | success |
| [Licensing HWID Tolerance](https://github.com/Dudzian/Dudzian/actions/runs/43700382835/job/132779534672) | completed | success |
| [UI Packaging](https://github.com/Dudzian/Dudzian/actions/runs/43700382835/job/132779534692) | completed | success |
| [Lint and Test](https://github.com/Dudzian/Dudzian/actions/runs/43700382835/job/132779534696) | completed | success |
| [UI QML Tests](https://github.com/Dudzian/Dudzian/actions/runs/43700382835/job/132779534734) | completed | success |
| [UI Native Tests](https://github.com/Dudzian/Dudzian/actions/runs/43700382835/job/132779534740) | completed | success |
| [Python tests (macOS)](https://github.com/Dudzian/Dudzian/actions/runs/43700382835/job/132779534742) | completed | success |
| [Python tests (Ubuntu PostgreSQL qualification, issuer-durability)](https://github.com/Dudzian/Dudzian/actions/runs/43700382835/job/132779534753) | completed | success |
| [Python tests (Ubuntu PostgreSQL qualification, remaining-security)](https://github.com/Dudzian/Dudzian/actions/runs/43700382835/job/132779534768) | completed | success |
| [Python tests (Ubuntu PostgreSQL qualification, history-reproposal)](https://github.com/Dudzian/Dudzian/actions/runs/43700382835/job/132779534771) | completed | success |
| [Mypy (core packages)](https://github.com/Dudzian/Dudzian/actions/runs/43700382835/job/132779534773) | completed | success |
| [gRPC Decision Feed Integration](https://github.com/Dudzian/Dudzian/actions/runs/43700382835/job/132779534774) | completed | success |
| [Frontend parity runtime proof](https://github.com/Dudzian/Dudzian/actions/runs/43700382835/job/132779534786) | completed | success |
| [Licensing security/durability (windows-latest, pdsa-production)](https://github.com/Dudzian/Dudzian/actions/runs/43700382835/job/132779534815) | completed | success |
| [Licensing security/durability (windows-latest)](https://github.com/Dudzian/Dudzian/actions/runs/43700382835/job/132779534834) | completed | success |
| [Licensing security/durability (ubuntu-latest)](https://github.com/Dudzian/Dudzian/actions/runs/43700382835/job/132779534857) | completed | success |
| [Licensing security/durability (windows-latest, lppi-durability)](https://github.com/Dudzian/Dudzian/actions/runs/43700382835/job/132779534881) | completed | success |
| [Licensing security/durability (windows-latest, cha-signed-attempt)](https://github.com/Dudzian/Dudzian/actions/runs/43700382835/job/132779534894) | completed | success |
| [Licensing security/durability (windows-latest, cha-core)](https://github.com/Dudzian/Dudzian/actions/runs/43700382835/job/132779534912) | completed | success |
| [Licensing security/durability (windows-latest, cha-root-proof)](https://github.com/Dudzian/Dudzian/actions/runs/43700382835/job/132779534919) | completed | success |
| [Licensing security/durability (macos-latest)](https://github.com/Dudzian/Dudzian/actions/runs/43700382835/job/132779534974) | completed | success |
| [Performance Benchmarks](https://github.com/Dudzian/Dudzian/actions/runs/43700382835/job/132779536095) | completed | skipped |
| [gRPC Feed Soak (cron)](https://github.com/Dudzian/Dudzian/actions/runs/43700382835/job/132779536322) | completed | skipped |
| [Marketplace Catalog Packaging](https://github.com/Dudzian/Dudzian/actions/runs/43700382835/job/132789763707) | completed | success |
| [Exchange HyperCare Report](https://github.com/Dudzian/Dudzian/actions/runs/43700382835/job/132789763758) | completed | success |
| [UI / QML tests (macOS)](https://github.com/Dudzian/Dudzian/actions/runs/43700382835/job/132791958958) | completed | success |
| [Marketing bundle parity](https://github.com/Dudzian/Dudzian/actions/runs/43700382835/job/132795652634) | completed | success |
| [Release marketplace assets](https://github.com/Dudzian/Dudzian/actions/runs/43700382835/job/132795820063) | completed | skipped |
| [Loopback E2E](https://github.com/Dudzian/Dudzian/actions/runs/43700382835/job/132797280250) | completed | skipped |
| [UI / QML tests (Windows)](https://github.com/Dudzian/Dudzian/actions/runs/43700382835/job/132797280276) | completed | skipped |
| [Integration tests (Linux)](https://github.com/Dudzian/Dudzian/actions/runs/43700382835/job/132808881584) | completed | success |

### Merged main: Marketplace Catalog (43700382826, push)

| Job | Status | Conclusion |
| --- | --- | --- |
| [Validate marketplace presets](https://github.com/Dudzian/Dudzian/actions/runs/43700382826/job/132770006714) | completed | success |

### Merged main: Stress Lab Report (43700382834, push)

| Job | Status | Conclusion |
| --- | --- | --- |
| [Generate Stress Lab artefact](https://github.com/Dudzian/Dudzian/actions/runs/43700382834/job/132770006665) | completed | success |

### Merged main: Style preflight (43700382936, push)

| Job | Status | Conclusion |
| --- | --- | --- |
| [Ruff format (all tracked files)](https://github.com/Dudzian/Dudzian/actions/runs/43700382936/job/132770006992) | completed | success |

### Merged main: Windows Build (43700382902, push)

| Job | Status | Conclusion |
| --- | --- | --- |
| [Build CryptoHunter Windows x64](https://github.com/Dudzian/Dudzian/actions/runs/43700382902/job/132770006946) | completed | success |

### Merged main: Quality and Security (43700382815, push)

| Job | Status | Conclusion |
| --- | --- | --- |
| [PostgreSQL atomic Root Proof commit](https://github.com/Dudzian/Dudzian/actions/runs/43700382815/job/132770006813) | completed | success |
| [Quality ratchet](https://github.com/Dudzian/Dudzian/actions/runs/43700382815/job/132770006952) | completed | success |
| [Quality ratchet (PostgreSQL remainder)](https://github.com/Dudzian/Dudzian/actions/runs/43700382815/job/132770006971) | completed | success |
| [History Successor Reproposal qualification](https://github.com/Dudzian/Dudzian/actions/runs/43700382815/job/132770006995) | completed | success |
| [Semantic issuer preflight](https://github.com/Dudzian/Dudzian/actions/runs/43700382815/job/132770007012) | completed | success |
| [Semgrep and working-tree secrets](https://github.com/Dudzian/Dudzian/actions/runs/43700382815/job/132770007019) | completed | success |
| [Full Git history secrets (advisory baseline)](https://github.com/Dudzian/Dudzian/actions/runs/43700382815/job/132770007665) | completed | skipped |
| [PostgreSQL atomic Root Proof and history coverage gate](https://github.com/Dudzian/Dudzian/actions/runs/43700382815/job/132793283876) | completed | success |
| [Quality ratchet (combined coverage gate)](https://github.com/Dudzian/Dudzian/actions/runs/43700382815/job/132796439223) | completed | success |
