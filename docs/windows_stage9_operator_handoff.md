# Stage 9 operator-approved MSI handoff — S9-S10-PRE-01B.2A

PRE-01B.2A extends PRE-01A and PRE-01B.1 with a separately invoked manual MSI
executor and separate passive base-install/runtime qualification. The delivered
code and hermetic tests do **not** establish qualification on native Windows,
approve a physical target/operator or perform an MSI installation. Physical
host approval and actual installation remain separate future decisions. There
is no Actions runner, CI execution channel or workflow hookup for the executor.

[stage9_current_status.json](../deployment/stage9_current_status.json) remains
the current-status authority: `stage_9 = IN_PROGRESS`,
`legal_production_enrollment = NOT_PERFORMED`, and production readiness is false.
Keep `run_stage10 = false`; Stage 10 remains `BLOCKED` / `NOT_RUN`. Historical
hosted clean-install or TPM `LIVE_PASS` receipts do not close those gates.

## Custody and provenance

The only eligible artifact is `windows-stage9-qualified-msi` from the canonical
`.github/workflows/platform-deployment.yml` in `Dudzian/Dudzian`, bound to an
independently approved full commit SHA, `main` revision, manual dispatch, run
ID/attempt, artifact ID and original ZIP SHA-256. Preserve the exact MSI,
`installer-manifest.json`, `windows-clean-install-receipt.json` and
`windows-clean-install-evidence.json`. Diagnostics, desktop ZIPs, another run,
PR artifacts, alternate branches and local rebuilds cannot substitute.

Reuse [the PRE-01A artifact validator](../deployment/windows_stage10_artifact_preflight.py)
and the existing Stage 9 clean-install and public Production Trust validators.
The artifact identity includes the exact MSI SHA-256, manifest SHA-256,
canonical installed-payload map, source revision, GitHub provider/run,
Production Trust ceremony, public package manifest digest and independently
approved public-trust run. The public-trust publication run may legitimately
be different from the MSI qualification run. None of this admits private
ceremony or target keys into the transfer bundle.

Preserve the original ZIP bytes and authenticated retrieval record. A metadata
JSON file or a successful offline fixture is not proof that GitHub supplied
it. Obtain expected pins independently of the archive and validate the exact
artifact/run/attempt through an authenticated, minimum-permission GitHub read
session. No GitHub access or ambiguous identity leaves the live handoff
`BLOCKED`; do not select a different successful run as a fallback.

The completed-run protocol continues to require successful completion of the
canonical run. The active-run protocol instead requires authenticated evidence
that `windows-deployment-contract`, `windows-deployment-integration` and
`windows-clean-install-integration` in that exact run attempt have **completed
successfully**, together with independent artifact metadata and matching
repository, workflow, event, revision and run identity. An active overall run
is expected while later jobs remain pending; an uploaded ZIP alone never
establishes successful qualification. Missing jobs, incomplete/failed producer
jobs, incomplete metadata collection, mixed attempts or an unrelated artifact
block the handoff. An active-run validation does not assert the eventual
success of the overall workflow or authorize any Stage 10 job.

[`qualified_handoff`](../deployment/windows_stage9_handoff.py) performs fresh
authenticated `gh api` GET requests for the exact artifact/run and every page
of that run attempt's jobs. It requires consistent totals and unique job IDs,
matching job revisions/attempts, and artifact creation/update times within the
successful clean-install producer's interval; the artifact API has no direct
run-attempt field. It repeats GitHub provenance checks after file validation.
The command uses the PRE-01A artifact arguments plus the independently pinned
required `--expected-run-attempt`:

```powershell
python -m deployment.windows_stage9_handoff `
  --allowed-root C:\Stage9Handoff `
  --archive C:\Stage9Handoff\qualified.zip `
  --artifact-metadata C:\Stage9Handoff\artifact-metadata.json `
  --workflow-run-metadata C:\Stage9Handoff\workflow-run-metadata.json `
  --production-trust-package $ApprovedTrustDirectory `
  --expected-artifact-id $ApprovedArtifactId `
  --expected-archive-sha256 $ApprovedArchiveSha256 `
  --expected-revision $ApprovedRevision `
  --expected-run-id $ApprovedRunId `
  --expected-run-attempt $ApprovedRunAttempt `
  --expected-production-trust-run-id $ApprovedTrustRunId `
  --expected-production-trust-manifest-sha256 $ApprovedTrustManifestSha256 `
  --output C:\Stage9Handoff\reports\qualified-handoff.json
```

Create the staging/report directories first; the trust-package directory must
have its canonical ceremony ID basename. Populate all approved variables from
the independent review, not the archive. The CLI reads staged files and GitHub;
it neither downloads nor executes payload binaries or authenticates the host
operator. The original PRE-01A offline completed-run CLI remains a rehearsal
and cannot substitute for this authenticated handoff.

Stage, hash and report under an explicitly approved operator-controlled
directory, separate from Program Files, ProgramData and runner configuration.
Restrict its writers, retain original inputs and use fresh report filenames.
Reject links/reparse points, traversal and unexpected archive members. Rehash
the original archive and every consumed file immediately before each use,
after transfers, before plan preparation and again at the future installation
boundary. An earlier `PASS` report is not a substitute for current bytes.
Hashing alone does not prevent a concurrent writer. The executor's native
materialization boundary must retain protected file handles through installer
completion; that implementation still requires adversarial qualification on
the actual Windows target before operational use.

## Approval data and operator authentication

[windows_stage9_operator_approval.py](../deployment/windows_stage9_operator_approval.py)
separates strict approval-data validation from actual operator authentication.
An approval record has exactly these fields:

| Fields | Required binding |
| --- | --- |
| `schema_version`, `approval_id` | Supported schema and explicit UUID approval identity |
| `repository`, `source_revision` | `Dudzian/Dudzian` and full approved 40-character commit SHA |
| `ci_run_id`, `ci_run_attempt`, `artifact_id` | Exact qualified run, positive attempt and artifact identity |
| `archive_sha256`, `msi_sha256`, `manifest_sha256` | Exact original ZIP, MSI and manifest SHA-256 values |
| `production_trust` | Exact validated ceremony/public-package/run binding |
| `host_identity` | Independently approved `hostname`, `machine_guid`, `hardware_uuid` |
| `operator_sid` | Operator Windows SID selected through independent policy |
| `issued_at`, `expires_at` | Explicit UTC issue/expiry times; at most 24 hours validity |
| `scope` | Exactly `transfer-qualified-msi`, `prepare-clean-install`, `verify-installed-payload` |

Unknown fields, including `approved=true`, fail closed. Expiry, mismatched host,
operator, artifact, run/attempt, revision, scope or Production Trust binding
cannot be overridden by a boolean or a prior report. This scope does not
authorize enrollment, service changes, repair/uninstall/reset, reboot, TPM/NV
mutation, key operations or Stage 10.

Schema validity only establishes that the data describes the intended
operation. Operator authentication requires a fresh native Windows integrated
token observation for the independently approved SID and host, in an elevated
interactive non-impersonated operator session. The approved execution design
must keep that session outside CI; a Windows token does not prove human
confirmation of a particular install. The approval file must not select
its own trusted operator or host policy. Token/session observation establishes
the identity of the current caller; it does not establish product readiness,
delegate additional rights or authorize installation by itself. Mocked
sessions and test adapters are rehearsal inputs, not authenticated live
operator sign-off; the authentication API accepts no session snapshot. Protect
approval/policy records and retain the independent
review record; no private signing-key system is introduced as a parallel
product trust authority.

The [installation preparation coordinator](../deployment/windows_stage9_install_preparation.py)
can produce `status=PREPARED_REVIEW_ONLY` or a blocked/failure result. Even a
prepared plan keeps `overall_status=BLOCKED` and
`installation_execution_authorized=false`. There is no `--execute` installation
switch and no callback from this preparation path into `msiexec`. PRE-01B.2A's
separate executor requires its own short-lived execution authorization and
explicit confirmation of the exact execution-plan digest; `prepare-clean-install`
approval never grants installation rights.
The fresh handoff also returns `receipt_sha256` and `evidence_sha256`; use its
receipt digest as `expected_receipt_sha256` in passive post-install validation.
Deriving that expectation from an unqualified replacement receipt is forbidden.

Native collectors resolve Windows PowerShell through `GetSystemDirectoryW` and
reject redirected system paths; neither the working directory nor `PATH`
selects the authentication/inspection executable. The native implementation
still requires qualification on Windows before operational use.

## Separate manual installation executor

[`windows_stage9_install_executor.py`](../deployment/windows_stage9_install_executor.py)
provides `review_installation_plan` and `execute_installation` for use from the
protected, independently verified executor checkout. Neither function is a CI
entry point. Keep this checkout and its Python dependencies separate from the
downloaded artifact tree; transfer credentials and an artifact cannot choose
the elevated code or policy. Use a dedicated native Windows x64 host without
an Actions runner, a separate transfer identity and an approved elevated
interactive operator session. Establish those controls before invoking any
future live operation.

The independently protected host-policy file has exactly `schema_version=1`,
`host_identity`, `operator_sids`, `artifact_root`, `staging_root`,
`executor_root` and `selection`. The selection independently pins every
`expected_*` handoff input: archive/artifact/revision, exact run/attempt and
public Production Trust identity. Policy and host inventory are approved
outside downloaded artifacts. The executor reads the policy from its protected
path and validates its ACL tree, the trusted checkout, original artifacts and
staging, including their ancestors. Unknown writers, unsafe ACLs, reparse
points, aliases and hardlinks block execution.

`review_installation_plan(preparation_approval, policy_path=...,
handoff_inputs=..., staging_root=...)` performs fresh qualification and native
host/operator checks and returns a reviewable plan digest. Protect the review
record and independently authorize that exact digest. The execution record
retains all strict PRE-01B.1 identity and provenance fields, but has the exact
scope `install-exact-qualified-msi-on-clean-host`, a distinct `approval_id`,
`plan_sha256`, `preparation_approval_sha256` and `policy_sha256`. Its UTC
validity is at most ten minutes. It binds the independent host/operator policy,
preparation approval, qualified ZIP/MSI/manifest/receipt/evidence, Production
Trust and GitHub run/attempt. A saved review digest cannot replace fresh
qualification or authorize a different operation.

`execute_installation(preparation_approval, execution_authorization,
policy_path=..., handoff_inputs=..., staging_root=...)` separately validates the
execution record and reauthenticates the operator through the native elevated
interactive token. It requires the operator to type `INSTALL <plan_sha256>`
in the actual Windows console. An environment variable, `approved=true`,
supplied session snapshot or stored PASS is not confirmation. After consent,
it repeats expiry, clean-host, policy, operator and authenticated handoff
checks before the installer boundary. Existing installations, enrollment,
residuals or an incomplete inventory remain blocked without repair, uninstall,
reset or takeover.

The native boundary materializes only the exact qualified ZIP member into a
fresh protected operation directory, checks the open MSI's SHA-256 and holds
native handles denying write/delete sharing through installer completion.
Original inputs, policy and execution code remain pinned during the operation.
The trusted Windows system directory supplies `msiexec.exe`; PATH, CWD and
environment lookup cannot select it. Arguments are separate process arguments,
without a shell: `/i <exact-msi> /qn /norestart REBOOT=ReallySuppress ALLUSERS=1
MSIINSTALLPERUSER=0` and `/L*V <protected-log>`. Retain the full MSI log, exact
command, plan and authorization digests, return code and failure report under
protected staging.

| Installer result | Recorded meaning and required action |
| --- | --- |
| `0` | Installer completed; independently run passive base-install qualification |
| `3010` | Installer reports reboot required; retain that result and request a separate decision, without rebooting |
| `1641` | Unwanted reboot initiation; stop and retain diagnostics for a separate investigation |
| Other error, cancellation, interruption or integrity loss | Retain the failure report/log; diagnose independently without an automatic retry |

If Ctrl-C interrupts the operator's wait after process creation, record the
interruption and keep the protected handles while waiting for that single
existing installer process to finish. Do not kill it, start another attempt or
reboot. Classify its actual exit code and verify pinned identities afterward;
an interruption cannot establish that installation never happened. Preserve
the immutable `installation-intent.json` snapshot from before process handoff
alongside the final `installation-report.json`. If a final report write fails, retain earlier journal evidence and
report the outcome/reporting uncertainty without overwriting that evidence.

Neither a successful installer return code nor a hermetic process-boundary
test establishes installed payload qualification, legal enrollment or Stage
10 readiness. This task never invokes the real installer.

## Host inspection and installation verification

Read-only host diagnostics must distinguish a proven clean target from an
existing or unknown installation. Observe MSI registration/product version,
canonical `%ProgramFiles%\CryptoHunter` and `%ProgramData%\CryptoHunter`,
ownership/enrollment markers, and every canonical SCM service: account,
executable path, quoting/arguments, dependency, startup, state, PID and recovery.
Record PostgreSQL, backend and freshness-verifier observations and all access
errors. A partial inventory cannot establish absence. Existing enrollment,
existing installation, unexplained residual data or an unknown state remains
`BLOCKED` pending a separate migration/recovery decision. Do not adopt,
uninstall, repair, reset or remove it; do not read private-key contents.

Native MSI inspection combines UpgradeCode-related registration with
`ProductsEx('', 'S-1-1-0', 7)` and read-only, stable cached-MSI product/upgrade
identities across installed user contexts. Incomplete access or an unreadable
cache blocks the host. Windows omits other users' unmanaged products that are
only advertised from this API; this is not an assertion about every possible
advertisement. No advertised-only state authorizes an installation here.

[`run_host_inspection`](../deployment/windows_stage9_host_inspection.py)
collects native observations only with `live=True`; supplied snapshots and
`inspect_existing_installation` are explicitly rehearsal observations. The
native MSI inventory reads registration directly and does not invoke
`Win32_Product`, which can trigger repair. Unknown registration, enrollment or
canonical state cannot establish a clean target.

Runner ACL inspection must cover the canonical root and its ancestors, all
protected descendants including `bin`, `_work`, `_update`, executable files and
configuration. Review explicit and inherited allow/deny ACEs, owners, group
principals, inheritance state and mutation rights including child deletion.
An unapproved writable group is unsafe even if its members are not listed in
the snapshot. Reject junctions, symlinks/reparse points, unknown ACE semantics,
unreadable nodes, inconsistent paths and changes during enumeration. A safe
root DACL does not hide an unsafe child. Conservative validation may reject
otherwise legitimate customized ACLs; a `FAIL` is not permission to repair
them. A stable observation during inspection also cannot prevent subsequent
authorized changes: preserve isolation and revalidate before future use.

[`validate_acl_tree`](../deployment/windows_runner_acl.py) consumes a complete
tree observation; its native `Collect-RunnerAclTree` collector pins parent,
root and descendant handles without delete sharing and compares two
inventories, including file identity, final paths, size/change times, owners,
DACLs and flags. It rejects hardlinks and noncanonical/NULL DACLs. Disabled
inheritance is acceptable only when the observed complete ACL is safe. Limits
of 50,000 nodes, depth 256 and the collector timeout fail closed. This is a
conservative mutation-grant/owner policy, not a general Windows effective-token
access calculator: an unfamiliar writable group is rejected rather than
trusted through guessed membership. Ordinary broad ancestor grants may need a
separate approved ACL-remediation decision before the host can qualify.

[`windows_stage9_install_qualification.py`](../deployment/windows_stage9_install_qualification.py)
provides two separate passive entry points, reusing the existing host, payload,
SCM and installer authorities. The PRE-01B.1 `verify_post_install` boundary
remains available for its existing callers; it does not authorize installation.

**A. `qualify_clean_base_installation`:** rehash the exact qualified MSI,
manifest and clean receipt, then verify the entire installed file set and all
SHA-256 values against `installed_files`, exact MSI identity/product version,
canonical Program Files/ProgramData layout and protected ACLs. Verify SCM
accounts, executable paths/arguments, dependencies and recovery. Inspect the
private PostgreSQL payload, `127.0.0.1:55432`, canonical HBA/identity mapping and
static configuration boundaries. Backend and verifier must remain
demand-start/stopped. This phase does not require an enrolled, running backend
or attempt to start one.

Static PostgreSQL files alone cannot establish the server's effective HBA,
TLS or SQL role separation. The existing qualification helpers include
service-token mechanisms with side effects; they are not a passive SQL
authority. An approved, native read-only service-token adapter is still
unavailable here, so both phases include a `postgresql_effective_sql` gate that
remains `BLOCKED`. Successful base-static and stopped-service checks do not
close that gate or claim actual SQL role separation; native clean-base overall
qualification therefore also remains `BLOCKED`. Base qualification still has
no legal-enrollment or backend-readiness requirement. Access errors and missing
authorities cannot become successful empty observations. Keep
`freshness_runtime` and `freshness_crypto_verifier` separated without changing
roles, passwords, service state or ACLs.

**B. `qualify_enrolled_runtime`:** first require accepted legal production
enrollment from the existing authority. That authority currently reports
`LEGAL_PRODUCTION_ENROLLMENT_NOT_COMPLETED`; runtime qualification therefore
remains `BLOCKED` and never tries to start services to create readiness.
When the legal and native authorities become available through separate work,
qualification must bind
[`qualify_runtime_provenance`](../deployment/windows_stage10_runtime_provenance.py)
to the current run, native SCM/PID and actual running processes, then verify
effective service-token SQL roles, mTLS and readiness (`freshness_runtime`,
`corehost_lock`, `cross_key_denied`, `cross_roles_denied`). Stale JSON, an
executable hash, caller-supplied PASS or enrollment marker alone cannot supply
those authorities. No fixture closes these live gates.

Both entry points reject snapshots in live mode. Saved observations are
rehearsals and remain `UNVERIFIED`, even when their static checks match the
contract. Neither phase installs, starts services, enrolls, repairs or changes
PostgreSQL/ACL state.

Keep the pre-enrollment base and enrolled runtime phases explicit. The actual
[`Product.wxs`](../deployment/windows_installer/Product.wxs) and
[`prove_services`](../deployment/windows_stage9_clean_install.py) contract
requires PostgreSQL automatic/running, backend demand-start/stopped and
verifier demand-start/stopped after a clean base install. A base may therefore
pass its own payload/SCM checks while live backend readiness remains `BLOCKED`.
Only the separately completed legal enrollment/readiness path may justify the
backend runtime transition. Do not start it or set Automatic merely to satisfy
Stage 10. Older narrative statements that backend is automatic do not override
the current source contract.

Review side effects before reusing qualification commands:
`qualify-mtls` starts/stops the verifier, `qualify-logging` restarts the backend,
and aggregate installer qualification configures recovery. They are outside
read-only inspection. The existing Stage 9 clean-install driver itself builds,
installs, uninstalls and cleans acceptance-owned state; never invoke it against
the persistent target for this handoff.

## Infrastructure decision: User owner and enforceable isolation

The current repository owner is a GitHub `User`. Organization runner groups
are not available as an isolation boundary for this personal repository.
Runner labels select jobs; they do not grant isolation from other workflows
which can request those labels. PRE-01B.1 changes no GitHub settings, ownership,
runner registration or workflow routing.

| Design | Security boundary | Complexity and migration cost | Current CI compatibility |
| --- | --- | --- | --- |
| Transfer to an approved Organization, or design an Organization-controlled execution repository | An appropriate GitHub plan plus dedicated runner group restricted to the approved repository and exact trusted manual workflow/ref; dedicated host identity and OS protection remain required | Transfer requires owner/operator approval and review of repository URLs, canonical provenance/Trust bindings, permissions, secrets/variables, branch/environment protections and external integrations. A separate execution repository avoids transfer but needs a reviewed cross-repository artifact and run-authentication protocol | Hosted qualification can remain; persistent lifecycle jobs need a separately reviewed manual route. The current PR-triggered `platform-deployment.yml` is not a manual-only trusted workflow merely because individual jobs are conditional |
| Dedicated host with manual execution while ownership remains `User` | Do not register an Actions runner on the target. Give CI no host execution credentials or remote execution channel. An authenticated operator transfers only approved public artifacts into protected staging, then uses a protected reviewed validator and separate elevated interactive session | No repository migration; requires OS accounts/ACLs, staging protection, independent identity/approval records and manually retained evidence. More operator work; future reboot continuity still needs a separate design | Canonical hosted Stage 9 CI continues unchanged. No PR code, downloaded workflow scripts or general Actions jobs run on the persistent host. Existing self-hosted Stage 10 jobs remain unavailable |

**PRE-01B.2A implementation design:** retain hosted Stage 9 qualification and
use the dedicated-host manual design. Physical host provisioning, operator
approval and native qualification are still pending. CI must have no path to
execute on the host; runner labels do not supply that protection. The
operator must independently authenticate GitHub, verify all pins and obtain a
protected reviewed checkout of the validator; unreviewed checkout code or
downloaded artifacts must not execute in the elevated session. Separate
non-administrator transfer from approved elevated installation, limit staging
writers, retain logs and leave the host unavailable to Actions jobs.

This recommendation covers artifact/install handoff only. The current Stage 10
prepare job has no operator installation hold and downloads clean-install
evidence only. Do not dispatch it to obtain a paused installation window.

**Recommended eventual Actions lifecycle architecture:** an approved
Organization-controlled route with verified plan support for selected
repositories/workflows, a dedicated runner group and a separate manual-only
trusted workflow pinned to the reviewed protected ref. Verify the actual
GitHub enforcement and negative routing tests before registering/enabling the
runner. A separate execution repository is a larger protocol change: do not
change the canonical repository constant or accept foreign artifacts to make
it work. Review repository-transfer and organization-group policy against
[GitHub runner-group documentation](https://docs.github.com/en/actions/how-tos/manage-runners/self-hosted-runners/manage-access-to-self-hosted-runners-using-groups)
and [repository transfer documentation](https://docs.github.com/en/repositories/creating-and-managing-repositories/transferring-a-repository)
at decision time; documentation alone does not prove available plan features.

Separate workflow or execution-repository routing also creates a run-identity
problem: current Stage 10 provenance requires the qualified artifact from the
same canonical run. A future architecture must explicitly preserve that
binding or approve and test an equivalent cross-workflow protocol before
changing the guard. PRE-01B.1 does not solve this by accepting another run or
rewriting receipt identities, and PRE-01B.2 does not enable Stage 10 routing.

Neither option authorizes migration, enabling a runner or live installation
now. Record the actual host, responsible owner and independent sign-off as
open prerequisites. PRE-01A's organization-group validator must continue to
block the current User-owned repository; a manual target is not a fictitious
organization runner and does not earn a runner `PASS`.

## Evidence and remaining operational decisions

Fixtures establish validator behavior only. This change does not establish an
actual approved Windows target, authenticated operator session, qualified live
archive, live full-tree ACL, installed MSI, database observation or runtime
readiness. Retain concrete check reasons and distinguish `FAIL` (contradiction),
`BLOCKED` (missing prerequisite/access) and `UNVERIFIED` (not observed from the
required authority). A passing unit test never promotes those conditions.

| Area | PRE-01B.2A evidence and remaining limit |
| --- | --- |
| Code implementation | Separate execution authorization, protected materialization/process boundary and passive qualification phases are implemented; preparation authority is unchanged |
| Hermetic validation | Substituted process/native boundaries test refusals and installer outcomes; retain the exact test/quality results in the PR, without treating them as live evidence |
| Native Windows qualification | Not performed; actual token, ACL, handle-sharing, link/race, system-image, cancellation and logging behavior must be independently qualified |
| Physical host/operator | Not approved by this change; actual identity/inventory, isolated accounts, protected policy/checkout/staging and operator sign-off remain required |
| Actual MSI installation | Not performed; requires a separate decision, fresh short-lived exact-plan authorization and native console confirmation |
| Passive qualification | Both phases lack approved passive SQL authority and remain `BLOCKED`; enrolled runtime additionally requires legal enrollment |
| Stage 9 / Stage 10 | Stage 9 `IN_PROGRESS`; legal enrollment `NOT_PERFORMED`; `run_stage10=false`; Stage 10 `BLOCKED` / `NOT_RUN` |

Before any future installation decision, approve and observe the physical
manual target, retain independent operator/policy/code review records, and
qualify the native boundary including attempted concurrent replacement and
unsafe ACL/link cases. Close host-access gaps with read-only diagnostics; an
existing/unknown/enrolled installation requires a separate migration/recovery
decision. No automatic cleanup or takeover is introduced.

Only after a separate installation decision may an approved elevated operator
review a fresh plan, obtain its short-lived execution authorization, explicitly
confirm the exact digest and install the qualified MSI. Preserve the full log,
return-code classification and before/after evidence. Then run base-install
qualification independently and retain each blocked gate. Do not retry or
restart automatically. Future legal enrollment and runtime qualification need
their own approved authority and decision.

Keep `run_stage10=false`. Reboot, 24-hour soak, TPM/NV/key operations and actual
Stage 10 dispatch remain outside this work and require separate explicit
authorization after the production gates close. Full long-running matrices
belong to GitHub CI. CI changes only add hermetic tests and quality scope for
the new modules; no workflow calls the native executor or reaches an actual
installation target.

The handoff is not ready for actual installation until these required live
controls and approvals are implemented and verified. Installation completion,
when separately authorized, still does not constitute Stage 10 `PASS`.
