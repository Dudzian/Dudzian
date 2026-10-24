# CI audit — baseline PR #3204

Inspected both requested runs with authenticated GitHub connector GETs, including pagination via `github_fetch .../jobs?per_page=100&page=1`. Each run has exactly 40 jobs (`total_count=40`, 40 returned); 37 executed, 3 skipped. All 74 executed-job logs were downloaded and directly searched. Raw logs and normalized inventories remain under `work/ci-remediation/logs/` and are not intended as repository changes.

## Commit and scope evidence

- PR: https://github.com/Dudzian/Dudzian/pull/3204 (open, draft).
- Head: `2a14d8db54c0ed506b1734e5f8765da21445eb66`; base main: `bae4fcb661de1de91f112347ac07f4220ed7e346`.
- Only one Markdown file, 56 added lines; confirms operator-reported synthetic physical Stage41B evidence, not independent attestation or production authorization.
- PR run https://github.com/Dudzian/Dudzian/actions/runs/40080330799 checks merge `2b41fd97b73ef98690667a64ba53e20af823f81c`.
- Push run https://github.com/Dudzian/Dudzian/actions/runs/40080304463 checks head `2a14d8db54c0ed506b1734e5f8765da21445eb66`.
- Existing failures therefore are not caused by the documentation addition.

## Failures

The only failed job in each requested run is **Dependency vulnerability scan** (job120832357017 and job120832436810), step `Run pip-audit (canonical desktop release lock)`. Dependency findings and artifact-access limits are described in [the remediation report](ci-remediation-2026-10-24.md). Other executed jobs completed successfully; this does not establish absence of warnings or skipped scenarios.

## Warning baseline, each run separately

| Category | Observed baseline per run | Evidence |
| --- | ---: | --- |
| Node DEP0005 | 29 messages, across 28 jobs | Artifact download; actions/download-artifact@v8 resolved to SHA9000827ccba6bdab643e8b6fd33ac0654aef8333 |
| AnyIO shutdown exception | 3 | Ubuntu, Windows and macOS main Python jobs, after successful pytest |
| NumPy Mean of empty slice | 27 warnings in each of 3 main Python jobs (81 total) | pytest groups each repeated category into one printed message; raw printed messages=3 |
| Explicit PytestCollectionWarning | 77 printed messages across 7 jobs | 11 each in main Python3, PostgreSQL3 and Linux integration1 |
| Additional suppressed pytest warning payloads | 44 warnings in 4 successful jobs | Release Quality Gates and QML3 jobs report11 each with existing `--disable-warnings`; category not independently proved from their suppressed text |
| SET LOCAL outside transaction | 42 | issuer-durability3, history-reproposal13, remaining-security26 |
| PostgreSQL old directory cleanup | 2 related warnings | remaining-security `base/20188`; must correlate fault injection/cleanup as the remediation report does |
| Marketing parity skipped | 2 emitted workflow annotations | Missing successful current stress-lab evidence and no configured mirror |
| ProcDump unavailable | 1 | Windows QML diagnostic capability unavailable; tests still run |
| pip root | 1 | Disposable loopback test container installs as root |

Main pytest warnings total38 per platform = collection11 + NumPy27. Total pytest warning counts across all summaries are202 per run; 44 warning payloads are hidden by existing UI/release CLI options, not counted as independently identified collection messages.

All 29 expected artifact SHA256 values per run match the29 computed download digests (multiset comparison per job); no digest mismatch was observed. The string `digest-mismatch: error` is a configured fail-closed policy, not a mismatch event. Exact upstream Node source finding and no available released fix are documented in [the remediation report](ci-remediation-2026-10-24.md).

## Skips and shards

Three skipped jobs per run are expected event separation: Performance Benchmarks, gRPC Feed Soak (cron), Release marketplace assets. PostgreSQL selection is split into disjoint issuer-durability/history-reproposal/remaining-security `-k` partitions under `external_postgresql`; main suites exclude that marker and integration/QML/soak/performance/retraining. These deselections preserve dedicated execution and are not missing-test failures.

| Suite | Passed | Skipped | Deselected | pytest warnings |
| --- | ---: | ---: | ---: | ---: |
| Ubuntu main | 22874 | 122 | 1454 | 38 |
| Windows main | 21171 | 1786 | 1474 | 38 |
| macOS main | 21361 | 1596 | 1474 | 38 |
| PostgreSQL issuer-durability | 249 | 17 | 24184 | 11 |
| PostgreSQL history-reproposal | 80 | 17 | 24353 | 11 |
| PostgreSQL remaining-security | 923 | 18 | 23509 | 11 |
| Linux integration | 183 | 28 | 41778 | 11 |
| QML Ubuntu | 16366 | 11 | 25631 | 11 |
| QML macOS/Windows | 16363 | 17 | 25592 | 11 |
| Frontend parity PySide | 16307 | 6 | 0 | 0 |

The main Python and PostgreSQL commands omit `-ra`; exact historical per-test skip reasons cannot be reconstructed from their text logs alone. JUnit artifacts exist, but `download_workflow_artifact` returns File Service ZIP references whose host is explicitly denied by runtime network policy. Generic GitHub fetch does not support binary archives, and `download_file` is Sediment-only. This remains an evidence gap; do not describe all122/1786/1596 or PostgreSQL skips as justified solely from platform counts.

Explicit QML Ubuntu `-rA` details prove11 skips: missing Playwright1, missing FastAPI1, genuine binding-loop/load-warning conversion to skip3 (SetupWizard, StrategyConfigurator, TicketDialog), six empty parameter matrix entries. The three QML warning skips were real coverage loss and have now been repaired; actual Qt6.7.2 regression validation executes all three scenarios (four tests because both TicketDialog input names are covered). They were not legitimate missing-platform skips. Six empty matrices are vacuous missing/reorder cases when canonical list is empty; surrounding valid/invalid contract matrices still execute.

Marketing parity returns a *skipped* report when current stress-lab evidence/mirror is missing; its successful job is not parity PASS and never production readiness proof. No report or gate was rewritten.

## Additional UI issues identified

- `ui/qml/views/StrategyConfigurator.qml` binds child dependency properties through same-named unqualified fallbacks, creating self-reference when explicit parent dependency is null. Fix direct root dependency bindings, preserve public input properties.
- `ui/qml/support/TicketDialog.qml` cycles `diagnosticsControllerContext` and `diagnosticsController`; parent SupportCenter already injects explicit controller. Preserve writable input names while removing cycle, tighten tests.
- Frontend parity reports11 distinct pre-existing QML binding-loop locations in MainWindow/DockManager/ModeWizard/IconButton. These are now repaired with explicit root controller/theme references, with a strict real-Qt bootstrap regression. Root-id shadowing and a missing required modelData declaration in newly reached DockManager delegates were also repaired; source locations remain in the baseline warning inventory.
- Optional `tests/ui/test_strategy_workbench_demo.py` requires Playwright and launches Chromium only to round-trip one text value; missing dependency prevents its whole QML scenario from running. Do not count it as tested.
- `tests/ui/test_runbook_panel.py::test_guardrail_report_endpoint_fastapi` uses `importorskip` for absent FastAPI; missing optional HTTP stack differs from platform skip.

## Safety and interpretation

No native physical Mac is available here. Successful hosted macOS Python tests are not production hardware qualification. Stage9/Stage10 statuses, contracts, readiness blockers and test-only evidence classifications must remain unchanged. Expected platform `MISSING_CURRENT_EVIDENCE` / `PRODUCTION_READY=NOT_READY` blocks from other workflows should be retained, not made green through fixtures.

## Complete job inventory

### Run 40080330799

| Job | Result | Log |
| --- | --- | --- |
| [Dependency vulnerability scan](https://github.com/Dudzian/Dudzian/actions/runs/40080330799/job/120832357017) | failure | saved |
| [Prepare wheelhouse (windows-latest)](https://github.com/Dudzian/Dudzian/actions/runs/40080330799/job/120832357100) | success | saved |
| [Native Windows x64 security (no MSI)](https://github.com/Dudzian/Dudzian/actions/runs/40080330799/job/120832357173) | success | saved |
| [Marketplace catalog validation](https://github.com/Dudzian/Dudzian/actions/runs/40080330799/job/120832357176) | success | saved |
| [Prepare wheelhouse (macos-latest)](https://github.com/Dudzian/Dudzian/actions/runs/40080330799/job/120832357182) | success | saved |
| [Generate SBOM](https://github.com/Dudzian/Dudzian/actions/runs/40080330799/job/120832357201) | success | saved |
| [Prepare wheelhouse (ubuntu-latest)](https://github.com/Dudzian/Dudzian/actions/runs/40080330799/job/120832357237) | success | saved |
| [Mypy (core packages)](https://github.com/Dudzian/Dudzian/actions/runs/40080330799/job/120833576929) | success | saved |
| [Licensing HWID Tolerance](https://github.com/Dudzian/Dudzian/actions/runs/40080330799/job/120833576951) | success | saved |
| [Python tests (Ubuntu)](https://github.com/Dudzian/Dudzian/actions/runs/40080330799/job/120833576961) | success | saved |
| [Lint and Test](https://github.com/Dudzian/Dudzian/actions/runs/40080330799/job/120833576975) | success | saved |
| [Python tests (Windows)](https://github.com/Dudzian/Dudzian/actions/runs/40080330799/job/120833576990) | success | saved |
| [Python tests (macOS)](https://github.com/Dudzian/Dudzian/actions/runs/40080330799/job/120833577016) | success | saved |
| [UI Native Tests](https://github.com/Dudzian/Dudzian/actions/runs/40080330799/job/120833577028) | success | saved |
| [Frontend parity runtime proof](https://github.com/Dudzian/Dudzian/actions/runs/40080330799/job/120833577037) | success | saved |
| [Release Quality Gates](https://github.com/Dudzian/Dudzian/actions/runs/40080330799/job/120833577062) | success | saved |
| [Python tests (Ubuntu PostgreSQL qualification, history-reproposal)](https://github.com/Dudzian/Dudzian/actions/runs/40080330799/job/120833577078) | success | saved |
| [gRPC Decision Feed Integration](https://github.com/Dudzian/Dudzian/actions/runs/40080330799/job/120833577085) | success | saved |
| [Licensing security/durability (windows-latest)](https://github.com/Dudzian/Dudzian/actions/runs/40080330799/job/120833577093) | success | saved |
| [Licensing security/durability (windows-latest, pdsa-production)](https://github.com/Dudzian/Dudzian/actions/runs/40080330799/job/120833577097) | success | saved |
| [Python tests (Ubuntu PostgreSQL qualification, remaining-security)](https://github.com/Dudzian/Dudzian/actions/runs/40080330799/job/120833577103) | success | saved |
| [Licensing security/durability (windows-latest, lppi-durability)](https://github.com/Dudzian/Dudzian/actions/runs/40080330799/job/120833577107) | success | saved |
| [Licensing security/durability (ubuntu-latest)](https://github.com/Dudzian/Dudzian/actions/runs/40080330799/job/120833577110) | success | saved |
| [Python tests (Ubuntu PostgreSQL qualification, issuer-durability)](https://github.com/Dudzian/Dudzian/actions/runs/40080330799/job/120833577122) | success | saved |
| [Licensing security/durability (windows-latest, cha-core)](https://github.com/Dudzian/Dudzian/actions/runs/40080330799/job/120833577130) | success | saved |
| [UI QML Tests](https://github.com/Dudzian/Dudzian/actions/runs/40080330799/job/120833577140) | success | saved |
| [Licensing security/durability (macos-latest)](https://github.com/Dudzian/Dudzian/actions/runs/40080330799/job/120833577161) | success | saved |
| [UI Packaging](https://github.com/Dudzian/Dudzian/actions/runs/40080330799/job/120833577168) | success | saved |
| [Licensing security/durability (windows-latest, cha-root-proof)](https://github.com/Dudzian/Dudzian/actions/runs/40080330799/job/120833577180) | success | saved |
| [Licensing security/durability (windows-latest, cha-signed-attempt)](https://github.com/Dudzian/Dudzian/actions/runs/40080330799/job/120833577272) | success | saved |
| [Performance Benchmarks](https://github.com/Dudzian/Dudzian/actions/runs/40080330799/job/120833578089) | skipped | not created (job skipped) |
| [gRPC Feed Soak (cron)](https://github.com/Dudzian/Dudzian/actions/runs/40080330799/job/120833578347) | skipped | not created (job skipped) |
| [UI / QML tests (macOS)](https://github.com/Dudzian/Dudzian/actions/runs/40080330799/job/120837913812) | success | saved |
| [Marketplace Catalog Packaging](https://github.com/Dudzian/Dudzian/actions/runs/40080330799/job/120839160460) | success | saved |
| [Exchange HyperCare Report](https://github.com/Dudzian/Dudzian/actions/runs/40080330799/job/120839160479) | success | saved |
| [UI / QML tests (Windows)](https://github.com/Dudzian/Dudzian/actions/runs/40080330799/job/120847644802) | success | saved |
| [Loopback E2E](https://github.com/Dudzian/Dudzian/actions/runs/40080330799/job/120847644872) | success | saved |
| [Marketing bundle parity](https://github.com/Dudzian/Dudzian/actions/runs/40080330799/job/120848093443) | success | saved |
| [Release marketplace assets](https://github.com/Dudzian/Dudzian/actions/runs/40080330799/job/120848501945) | skipped | not created (job skipped) |
| [Integration tests (Linux)](https://github.com/Dudzian/Dudzian/actions/runs/40080330799/job/120857375394) | success | saved |

### Run 40080304463

| Job | Result | Log |
| --- | --- | --- |
| [Prepare wheelhouse (ubuntu-latest)](https://github.com/Dudzian/Dudzian/actions/runs/40080304463/job/120832436597) | success | saved |
| [Prepare wheelhouse (windows-latest)](https://github.com/Dudzian/Dudzian/actions/runs/40080304463/job/120832436714) | success | saved |
| [Marketplace catalog validation](https://github.com/Dudzian/Dudzian/actions/runs/40080304463/job/120832436770) | success | saved |
| [Native Windows x64 security (no MSI)](https://github.com/Dudzian/Dudzian/actions/runs/40080304463/job/120832436779) | success | saved |
| [Generate SBOM](https://github.com/Dudzian/Dudzian/actions/runs/40080304463/job/120832436793) | success | saved |
| [Dependency vulnerability scan](https://github.com/Dudzian/Dudzian/actions/runs/40080304463/job/120832436810) | failure | saved |
| [Prepare wheelhouse (macos-latest)](https://github.com/Dudzian/Dudzian/actions/runs/40080304463/job/120832436873) | success | saved |
| [Lint and Test](https://github.com/Dudzian/Dudzian/actions/runs/40080304463/job/120834173197) | success | saved |
| [Release Quality Gates](https://github.com/Dudzian/Dudzian/actions/runs/40080304463/job/120834173232) | success | saved |
| [Python tests (Ubuntu PostgreSQL qualification, issuer-durability)](https://github.com/Dudzian/Dudzian/actions/runs/40080304463/job/120834173258) | success | saved |
| [UI Packaging](https://github.com/Dudzian/Dudzian/actions/runs/40080304463/job/120834173261) | success | saved |
| [Python tests (Windows)](https://github.com/Dudzian/Dudzian/actions/runs/40080304463/job/120834173267) | success | saved |
| [UI Native Tests](https://github.com/Dudzian/Dudzian/actions/runs/40080304463/job/120834173268) | success | saved |
| [Python tests (macOS)](https://github.com/Dudzian/Dudzian/actions/runs/40080304463/job/120834173274) | success | saved |
| [Licensing HWID Tolerance](https://github.com/Dudzian/Dudzian/actions/runs/40080304463/job/120834173285) | success | saved |
| [Python tests (Ubuntu PostgreSQL qualification, history-reproposal)](https://github.com/Dudzian/Dudzian/actions/runs/40080304463/job/120834173286) | success | saved |
| [UI QML Tests](https://github.com/Dudzian/Dudzian/actions/runs/40080304463/job/120834173288) | success | saved |
| [Frontend parity runtime proof](https://github.com/Dudzian/Dudzian/actions/runs/40080304463/job/120834173296) | success | saved |
| [Python tests (Ubuntu PostgreSQL qualification, remaining-security)](https://github.com/Dudzian/Dudzian/actions/runs/40080304463/job/120834173301) | success | saved |
| [gRPC Decision Feed Integration](https://github.com/Dudzian/Dudzian/actions/runs/40080304463/job/120834173315) | success | saved |
| [Mypy (core packages)](https://github.com/Dudzian/Dudzian/actions/runs/40080304463/job/120834173317) | success | saved |
| [Python tests (Ubuntu)](https://github.com/Dudzian/Dudzian/actions/runs/40080304463/job/120834173328) | success | saved |
| [Licensing security/durability (windows-latest, lppi-durability)](https://github.com/Dudzian/Dudzian/actions/runs/40080304463/job/120834173372) | success | saved |
| [Licensing security/durability (windows-latest, cha-core)](https://github.com/Dudzian/Dudzian/actions/runs/40080304463/job/120834173376) | success | saved |
| [Licensing security/durability (macos-latest)](https://github.com/Dudzian/Dudzian/actions/runs/40080304463/job/120834173390) | success | saved |
| [Licensing security/durability (windows-latest, cha-signed-attempt)](https://github.com/Dudzian/Dudzian/actions/runs/40080304463/job/120834173408) | success | saved |
| [Licensing security/durability (windows-latest, cha-root-proof)](https://github.com/Dudzian/Dudzian/actions/runs/40080304463/job/120834173432) | success | saved |
| [Licensing security/durability (windows-latest)](https://github.com/Dudzian/Dudzian/actions/runs/40080304463/job/120834173437) | success | saved |
| [Licensing security/durability (windows-latest, pdsa-production)](https://github.com/Dudzian/Dudzian/actions/runs/40080304463/job/120834173456) | success | saved |
| [Licensing security/durability (ubuntu-latest)](https://github.com/Dudzian/Dudzian/actions/runs/40080304463/job/120834173531) | success | saved |
| [gRPC Feed Soak (cron)](https://github.com/Dudzian/Dudzian/actions/runs/40080304463/job/120834174092) | skipped | not created (job skipped) |
| [Performance Benchmarks](https://github.com/Dudzian/Dudzian/actions/runs/40080304463/job/120834174573) | skipped | not created (job skipped) |
| [Marketplace Catalog Packaging](https://github.com/Dudzian/Dudzian/actions/runs/40080304463/job/120835314783) | success | saved |
| [Exchange HyperCare Report](https://github.com/Dudzian/Dudzian/actions/runs/40080304463/job/120835314798) | success | saved |
| [UI / QML tests (macOS)](https://github.com/Dudzian/Dudzian/actions/runs/40080304463/job/120839362033) | success | saved |
| [Loopback E2E](https://github.com/Dudzian/Dudzian/actions/runs/40080304463/job/120846207046) | success | saved |
| [UI / QML tests (Windows)](https://github.com/Dudzian/Dudzian/actions/runs/40080304463/job/120846207113) | success | saved |
| [Marketing bundle parity](https://github.com/Dudzian/Dudzian/actions/runs/40080304463/job/120847707779) | success | saved |
| [Release marketplace assets](https://github.com/Dudzian/Dudzian/actions/runs/40080304463/job/120848660258) | skipped | not created (job skipped) |
| [Integration tests (Linux)](https://github.com/Dudzian/Dudzian/actions/runs/40080304463/job/120857744480) | success | saved |


## Related workflows for exact head

Authenticated REST `actions/runs?head_sha=2a14d8db54c0ed506b1734e5f8765da21445eb66&per_page=100` returns7 workflow runs, saved in `logs/related-workflows.json`: Marketplace Catalog SUCCESS40080330801, Style preflight SUCCESS40080330837, CodeQL SUCCESS40080330772, Quality and Security SUCCESS40080330773, requested CI PR/push FAILURE, Platform deployment tracks FAILURE40080330793.

Three platform failures are directly confirmed from complete raw logs and `deployment/platform_readiness.py:200–216`, which returns1 whenever required evidence blockers remain:

- [Linux deployment](https://github.com/Dudzian/Dudzian/actions/runs/40080330793/job/120832356303): CORE_REQUIRED_SUITES and live Linux systemd/clean-install/reboot lifecycle missing current evidence; LINUX_PRODUCTION_READY=NOT_READY.
- [macOS deployment](https://github.com/Dudzian/Dudzian/actions/runs/40080330793/job/120832356334): CORE_REQUIRED_SUITES and live launchd/package permissions/auth IPC/reboot lifecycle missing; MACOS_PRODUCTION_READY=NOT_READY.
- [Windows release gate](https://github.com/Dudzian/Dudzian/actions/runs/40080330793/job/120844120774): current core/SCM/clean-install artifacts present, but reboot recovery, long-running lifecycle, update restart and failed-update safe behavior evidence absent; WINDOWS_PRODUCTION_READY=NOT_READY.
- Three Stage10 prepare/reboot/qualification jobs remain skipped; no manual/destructive/self-hosted gate was invoked.

These are expected fail-closed readiness refusals, not dependency/test regressions. They must remain red until genuine current evidence exists. The hosted macOS Python PASS is not a substitute for physical macOS qualification.

[Quality ratchet](https://github.com/Dudzian/Dudzian/actions/runs/40080330773/job/120832356440) raw log additionally confirms exactly3 obsolete WPS433 warnings at requested locations3895–3897; Ruff itself succeeds. Its baseline unit/critical suite3119pass3skip, CHA reservation35pass, issuer-history1057pass, JCS10pass, attempt reservation313pass, signed attempt326pass12skip. Full unrelated Quality and Security job logs were not exhaustively downloaded; requested two main workflows were.


## Completed QML remediation addendum

See [the remediation report](ci-remediation-2026-10-24.md) for exact source changes, actual Qt runtime evidence, public-input preservation and remaining limits. Final targeted actual PySide6/Qt6.7.2 run: **9 passed, zero skips, zero pytest warnings in19.52s**, Linux/Python3.12 only. It covers the three formerly skipped scenarios (four tests), real app bootstrap/controllers/delegates, standalone mode wizard, recommendation persistence and layout roundtrip. MainWindow and ModeWizard regressions now reject every emitted QML load warning rather than accepting binding loops. This is focused evidence, not a claim that hosted Windows/macOS or all QML tests passed on the new commit.

The separately recovered workbench demo test remains a real failure due pre-existing removed demo APIs; see [remaining blockers](ci-remediation-2026-10-24.md#f-remaining-blockers). No synthetic replacement API, scheduler activation or skip was introduced to turn it green.
