# Stage 9 operator-approved MSI handoff — S9-S10-PRE-01B.1

PRE-01B.1 prepares and hermetically tests custody, approval, host inspection and
installation verification boundaries. It does **not** install an MSI or make
the persistent target operational. The installation coordinator prepares a
reviewable plan; there is no production installation executor or automatic CI
installation route in this change.

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
Hashing does not prevent a concurrent privileged writer; PRE-01B.2 must prove
the protected staging/materialization and execution boundary on the target.

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
switch and no callback from this preparation path into `msiexec`. A future
production executor still requires separately approved implementation,
protected materialization and explicit confirmation of the exact plan digest.
The fresh handoff also returns `receipt_sha256` and `evidence_sha256`; use its
receipt digest as `expected_receipt_sha256` in passive post-install validation.
Deriving that expectation from an unqualified replacement receipt is forbidden.

Native collectors resolve Windows PowerShell through `GetSystemDirectoryW` and
reject redirected system paths; neither the working directory nor `PATH`
selects the authentication/inspection executable. The native implementation
still requires qualification on Windows before operational use.

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

`verify_post_install` in the same host module prepares a separate read-only
post-install boundary. Its
checks must use the qualified artifact and existing installer contracts:

1. Verify the entire installed file set and each SHA-256 against
   `installed_files`, exact product version/MSI identity and manifest/receipt
   binding. Reuse
   [`qualify_runtime_provenance`](../deployment/windows_stage10_runtime_provenance.py)
   for the backend/current-run binding.
2. Verify SCM accounts, canonical paths/arguments, dependency, startup and
   recovery using the installer/service-recovery contract. Check Program
   Files/ProgramData canonical layout and protected ACL boundaries.
3. Verify the private PostgreSQL version, `127.0.0.1:55432`, canonical HBA and
   identity mapping, and separation of `freshness_runtime` and
   `freshness_crypto_verifier`. Missing database or host access is a blocked
   prerequisite, never a successful empty result.
4. For legally enrolled runtime readiness, bind the backend readiness PID to
   the actually running SCM process and require `freshness_runtime`,
   `corehost_lock`, `cross_key_denied` and `cross_roles_denied`. Stale JSON or a
   matching executable hash does not establish a running process.

The present native verifier can inspect static database configuration,
listeners, SCM and readiness markers, but has no approved service-token
read-only SQL adapter to authenticate effective HBA/role observations. Actual
live post-install qualification therefore remains `BLOCKED` at that gate;
fixtures cannot supply the missing live authority. PRE-01B.2 must review and
complete this adapter without starting services or exporting private keys.
The current verifier also keeps any existing/unknown enrollment blocked; it
has no authority to classify an enrolled installation as legally ready.
PRE-01B.2 must compose the existing legal-enrollment authority before accepting
that separate phase, rather than bypassing the enrollment boundary.

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

**Recommended immediate architecture:** retain hosted Stage 9 qualification and
use the dedicated-host manual design for a separately approved PRE-01B.2
implementation/rehearsal. This is an enforceable execution boundary when CI
cannot execute on the host, not a claim that runner labels protect it. The
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
now. Record the selected design, responsible owner and independent sign-off as
open prerequisites. PRE-01A's organization-group validator must continue to
block the current User-owned repository; a manual target is not a fictitious
organization runner and does not earn a runner `PASS`.

## Evidence and exact next task — S9-S10-PRE-01B.2

Fixtures establish validator behavior only. This change does not establish an
actual approved Windows target, authenticated operator session, qualified live
archive, live full-tree ACL, installed MSI, database observation or runtime
readiness. Retain concrete check reasons and distinguish `FAIL` (contradiction),
`BLOCKED` (missing prerequisite/access) and `UNVERIFIED` (not observed from the
required authority). A passing unit test never promotes those conditions.

PRE-01B.2 must complete this bounded operational handoff, under a separate
operator decision:

1. Select and implement one reviewed infrastructure design above; establish
   actual host ownership/identity, execution isolation, accounts and protected
   staging. Do not enable a runner or migrate a repository as an incidental
   implementation detail.
2. Implement/review the native authenticated retrieval, protected
   materialization and separately invoked elevated MSI executor. Validate a
   fresh exact-plan confirmation and approval immediately before execution,
   rehash every input, preserve exact command/MSI log/outcome and reject MSI
   reboot-required outcomes without scheduling a reboot. Rehearse failures
   hermetically before any live execution decision.
3. Observe the actual target with the reviewed read-only collectors, close
   access/ACL/identity gaps and independently review before/after evidence.
   Unknown/existing/enrolled hosts stay blocked until a separate decision;
   PRE-01B.2 does not introduce automatic cleanup or takeover.
4. Only after explicit elevated-session authorization, install the exact
   qualified MSI and verify the full installed payload, product/MSI identity,
   SCM, recovery, Program Files/ProgramData ACLs and private PostgreSQL against
   existing contracts. A successful base install records backend/verifier
   demand-start/stopped and does not claim enrolled runtime readiness.
5. Keep legal enrollment and its backend/readiness transition independently
   gated. Once legally authorized and completed, collect runtime/SCM PID,
   database/role and readiness proofs with the existing authorities. Missing
   legal enrollment keeps those dependent checks blocked.
6. Retain operator sign-off and test/quality evidence, then reassess the
   remaining Stage 9 closure and lifecycle prerequisites. Keep `run_stage10`
   false throughout this task. Reboot, 24-hour soak, TPM/NV/key operations and
   actual Stage 10 dispatch remain outside PRE-01B.2 and require their own
   explicit authorization after the production gates close.

The handoff is not ready for actual installation until these required live
controls and approvals are implemented and verified. Installation completion,
when separately authorized, still does not constitute Stage 10 `PASS`.
