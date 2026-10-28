# PLAT-01B — platform release policy V2

Status: implementation of the approved PLAT-01 Windows/Linux product scope;
release qualification remains evidence-dependent. This policy does not perform
or authorize physical tests, enrollment, Root Proof issuance, Stage 10 or LIVE.
It advances the roadmap's platform-scope alignment, not a product milestone exit.

## Three independent concepts

| Concept | Contract field / evaluator | Meaning |
| --- | --- | --- |
| Product delivery intent | `release_priority` | Windows PRIMARY, Linux SUPPORTED, macOS DEFERRED; these are scope decisions, not test outcomes. |
| Compatibility coverage | `compatibility_platforms`, Core test plan and CI | Windows, Linux and macOS continue running required shared-code compatibility tests. Hosted compatibility PASS does not qualify a native installation. |
| Native production qualification | retained `release_gates`, acceptance evidence and `production_ready` | Each included platform requires its own complete current-revision execution evidence plus shared Core evidence. A deferred platform always refuses qualification. |

`platforms` remains the inventory of all three known platforms, including the
deferred one. It is not a supported-release list. Native macOS mechanisms,
acceptance records, evidence classes and `MACOS_PRODUCTION_READY` requirements
are retained. No macOS item is rewritten to PASS or NOT_APPLICABLE. A future
macOS release needs a separately approved project, tests, native qualification
and a reviewed policy/evaluator update; a caller cannot reactivate it by editing
one priority value.

## Closed release profiles

| Profile | Claimed native platforms | Required qualification |
| --- | --- | --- |
| `WINDOWS_BETA` | Windows | All frozen Windows gates and all required cross-OS Core evidence. Linux-native PASS is not invented or required. |
| `WINDOWS_LINUX_BETA` | Windows and Linux | The complete independent gates for both platforms, with the same Core requirement. |
| `CRYPTOHUNTER_V1` | Windows and Linux | Both native platforms must qualify. A Windows-only beta does not finish full v1. |

`release_ready` and `release_blocking_items` evaluate exactly these closed
profiles. Unknown profile/platform names, missing policy fields, unknown
statuses, weakened native gate inventories, altered evidence classes and
unsupported policy versions fail closed. `DEFERRED` is not an acceptance status
and never means `MACOS_PRODUCTION_READY`.

The existing `--platform WINDOWS|LINUX|MACOS` CLI keeps its platform-specific
`<PLATFORM>_PRODUCTION_READY` label. The new `--release-profile` selector emits
`<PROFILE>_RELEASE_READY`, so a Windows-only beta result cannot be mistaken for
full Windows/Linux v1 qualification.

The first beta may be Windows-only; Linux may enter the beta after its own
qualification. A physical Mac is not a Windows/Linux native-release prerequisite.
The hosted macOS compatibility matrix remains mandatory for shared-code quality.
The supported Windows/Linux native requirements, Core suites and all existing
security/quality thresholds remain in force.

## Explicit V1 to V2 policy migration

The previous policy uses `schema_version=1` and declares macOS SUPPORTED. V2
changes only product-scope metadata and adds its explicit interpretation:

1. Set policy `schema_version=2`.
2. Keep `platforms=[WINDOWS,LINUX,MACOS]` and `reference_platform=WINDOWS`.
3. Set the exact priorities to PRIMARY/SUPPORTED/DEFERRED.
4. Add `compatibility_platforms=[WINDOWS,LINUX,MACOS]`.
5. Add the three closed `release_profiles` listed above.
6. Preserve acceptance records, native gates, semantic invariants, platform
   matrix, mechanism descriptions, historical evidence text and conservative
   legacy flags. No evidence is promoted by this migration.

The current evaluator accepts the complete V2 policy only. A historical V1
policy fails with `UNSUPPORTED_POLICY_VERSION` rather than acquiring V2 release
semantics implicitly. Historical repository revisions retain their original
policy and evaluator for interpretation/replay; the new file does not rewrite
historical reports. Any historical decision must retain that original policy
revision, not be recomputed as a V2 authorization merely because evidence exists.
The evidence envelope is independently versioned and **remains V1**:
`deployment/platform_evidence.schema.json`, runner markers, probe identities and
revision/provider/run bindings are unchanged. Current-revision checks still
reject stale historical evidence.

Policy validation freezes complete native gate inventories and evidence-class
ownership independently of caller-supplied JSON. A provided policy cannot omit
Core, turn native requirements into static checks, replace a release with an
empty platform list, raise a conservative legacy flag, or claim an arbitrary
new release platform. Validation establishes policy structure, not the provenance
of the policy file or permission to publish an actual release.

## CI and evidence consumers

The Core manifest `deployment/core_required_suites_v1.json` and its
`deployment.core_test_plan` executor are unchanged. Linux, Windows and macOS
runner markers are still mandatory. `deployment.platform_evidence` aggregates
exactly those three current-revision, same-provider/run/plan markers, and refuses
missing/failed/duplicate markers. This keeps macOS shared-code regressions
blocking for Windows/Linux releases without calling them physical qualification.

In `.github/workflows/platform-deployment.yml`, the existing macOS deployment
job now runs blocking deployment/Core-plan compatibility contract tests and is
labelled as deferred production. It does not emit native macOS PASS. The Windows
release job selects `WINDOWS_BETA`; it keeps its existing Core and Windows
installation/lifecycle evidence dependencies. The Linux native gate remains
required for a release that includes Linux. No new `continue-on-error`, skip,
xfail or tolerated failed shared test is introduced.

## Source-inspected consumer inventory

This inventory was obtained with repository `rg` searches for
`platform_readiness`, `release_priority`, `MACOS_PRODUCTION_READY`, native gate
names and supported-platform declarations, followed by direct source reads.
Codebase Memory was unavailable; no index-completeness claim is made.

| Location | Role and disposition |
| --- | --- |
| `deployment/platform_readiness.json` | Canonical V2 policy; all native acceptance evidence and gates retained. |
| `deployment/platform_readiness.py` | Loads/validates V2, enforces exact scope/profile/native requirements and consumes unchanged V1 evidence. |
| `deployment/platform_evidence.py`, `platform_evidence.schema.json` | Revision-bound producers and frozen V1 envelope; no scope migration or reduced Core marker set. |
| `deployment/core_test_plan.py`, `core_required_suites_v1.json` | Three-OS compatibility plan, including macOS; retained. |
| `deployment/windows_stage10_artifact_preflight.py` | Imports `load_contract` and `_validated_results`; must continue consuming V1 Windows evidence under complete V2 policy without Stage 10 activation. |
| `deployment/windows_acceptance.py` | Independent Windows acceptance reports remain NOT_READY without their requirements; no deferred-platform shortcut. |
| `.github/workflows/platform-deployment.yml` | Closed Windows release profile; retained Windows/Linux native gates and blocking macOS/Core compatibility. |
| `.github/workflows/quality-security.yml`, `pyproject.toml` | Additive lint/format/type/coverage integration for the policy evaluator/tests; ratchets unchanged. |
| `tests/deployment/test_platform_readiness.py`, `test_platform_evidence.py` | Policy migration, release selection, native evidence and negative cases. |
| `tests/deployment/test_windows_acceptance.py`, `test_windows_stage10_preflight.py`, `test_core_test_plan.py` | Existing direct or adjacent contract consumers; run as regression scope. |
| `product_delivery_roadmap.md`, `windows_first_deployment_audit.md` | Current planning/architecture wording aligned and linked to V2. |
| `docs/audits/ci-remediation-2026-10-24-baseline.md`, `docs/architecture/issuer_history_corrective_audit_pr3100.md` | Historical audit findings; left intact as evidence of the earlier revision, not restated as current scope. |
| `docs/architecture/cryptohunter_product_architecture/audit_observability_alerts_and_updater.{json,md}` | Existing `supported_platforms` belongs to a cross-platform SQLite component compatibility contract, not product release selection. Its Windows/macOS/Linux technical inventory is retained. |
| `ui/pyside_app/preview_block_p_desktop_exe_packaging_source_inventory.py` | Preview source/packaging inventory includes all three OS names; it does not establish release/native qualification and remains unchanged. |
| security tests for unsupported custody platforms | Runtime security fallback contracts, not release scope; retained unchanged. |

Historical macOS NOT_READY reports remain correct for their original scope.
Current macOS compatibility can pass while native production remains deferred.
Neither state changes Stage 9 `IN_PROGRESS`, qualification `BLOCKED`, the false
confirmation/authorization/exception flags, `PHYSICAL_TEST_PENDING`,
`PRODUCTION_NO_GO`, runtime attestation `MISSING` or host PID scope `UNVERIFIED`.
