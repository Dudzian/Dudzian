# Windows Stage 10 read-only preflight — S9-S10-PRE-01A

This runbook covers the first part of the mandatory
[S9-S10-PRE-01 handoff](architecture/cryptohunter_product_architecture/product_delivery_roadmap.md#windows-stage-9--stage-10-preflight--s9-s10-pre-01).
It prepares evidence for review. It does not authorize installation or lifecycle
qualification and cannot produce Stage 10 `PASS`.

## Current gates and the delivery gap

[stage9_current_status.json](../deployment/stage9_current_status.json) remains
the current-status authority: Stage 9 is `IN_PROGRESS`, legal production
enrollment is `NOT_PERFORMED`, production readiness is false, and Stage 10 is
`BLOCKED_UNTIL_LEGAL_ENROLLMENT`. Historical clean-install or TPM `LIVE_PASS`
does not close these gates. This preparation does not change that status file.

The canonical [platform workflow](../.github/workflows/platform-deployment.yml)
promotes `windows-stage9-qualified-msi` only after its hosted Windows
clean-install proof and revision-bound evidence succeed. That artifact contains
the exact MSI, `installer-manifest.json`,
`windows-clean-install-receipt.json`, and
`windows-clean-install-evidence.json`.

The current Stage 10 prepare job downloads **`windows-clean-install-evidence`**.
It neither downloads `windows-stage9-qualified-msi` nor installs that MSI on
the dedicated self-hosted runner. Its runtime provenance check assumes an
already installed product. The hosted clean-install proof also uninstalls its
product and performs acceptance-owned cleanup. Neither proof establishes an
installation on the persistent target. **The target installation gap remains
BLOCKED and must be resolved in S9-S10-PRE-01B before any live dispatch.**

The existing Stage 10 jobs remain behind manual `workflow_dispatch` and the
explicit `run_stage10` opt-in. Leave that input false. Do not invoke
`windows_stage10_live.ps1` during this preflight: its workflow path schedules
a reboot and its qualification mode runs the live 24-hour lifecycle.

## Read-only boundary and report meanings

Allowed work consists of reading approved GitHub metadata, reading public
qualification artifacts, hashing files, inspecting runner/OS/service
configuration and writing preflight reports to a separate evidence directory.
Creating that directory or downloading public artifacts is a transfer operation,
not permission to mutate the product host. Never place reports under the
artifact root, Program Files, ProgramData product directories or runner
configuration directories.

Create the separate staging/report directories before invoking the tools. Each
report must use a new filename; do not reuse an input path or replace a prior
report. Keep the previous report and original inputs for the review record.

No MSI install/uninstall/repair, reboot, service start/stop/reconfiguration,
runner registration/reconfiguration, TPM/NV operation, production enrollment,
private-key operation or soak is performed. Do not run the Stage 9 clean-install
driver against the target: it builds, installs, uninstalls and purges its
acceptance-owned state. Some existing `qualify-*` commands also mutate state:
`qualify-mtls` starts/stops the verifier, `qualify-logging` restarts the backend,
and aggregate installer qualification configures recovery. They are outside
this rehearsal.

Each check records its name, status and concrete reasons. Use the following
meanings consistently:

| State | Meaning |
| --- | --- |
| `PASS` | The stated check was verified from its required source; this applies only to that check. |
| `FAIL` | Observed input contradicts the required identity, hash, policy or configuration. Stop and retain diagnostic evidence. |
| `BLOCKED` | A known prerequisite, approved input or permitted access is missing. The dependent operation cannot proceed. |
| `UNVERIFIED` | The condition has not been observed on the actual target or through the required authority. A fixture or operator assertion is insufficient. |

A configuration/snapshot rehearsal can exercise failure handling but never
verifies the actual runner. No actual host access means `UNVERIFIED`/`BLOCKED`,
never runner `PASS`. The aggregate preparation report records Stage 10 as
`NOT_RUN`; even verified artifact and runner checks leave current Stage 9,
enrollment and installation gates `BLOCKED`. `PREPARED` is an eventual reviewed
handoff milestone, not a Stage 10 acceptance result. Preflight reports are not
`windows-stage10` receipts or `LIVE_WINDOWS_INTEGRATION` evidence.

## Dedicated runner review

Do not assume the runner already exists. The operator identifies a persistent
physical Windows x64 machine, host owner, GitHub runner registration, service
identity and approved installation/elevation owner. Record a stable machine
identity and runner identity together; computer name or a label alone is
insufficient. Redact or restrict access to host-identifying reports as needed.

Review these conditions from actual read-only observations:

1. Native Windows x64 OS and process architecture, the expected machine
   identity, runner name and repository/organization registration all match.
   Confirm physical ownership/inventory separately; an OS platform string is
   not proof of a dedicated physical machine.
2. Authenticated GitHub runner metadata contains all four required exact labels:
   `self-hosted`, `Windows`, `X64`, `cryptohunter-stage10`. Extra labels must not
   create a route for general workloads; this validator accepts exactly the
   four-label set. Local environment variables do not prove GitHub labels or
   registration.
3. The runner is installed as the expected Windows service, is running and
   configured for automatic startup, and its canonical service executable,
   account and directory match the approved inventory. Inspect startup
   configuration without changing it or rebooting. Actual recovery after boot
   remains unverified until separately authorized lifecycle execution.
4. The current inspection token has required read access to host identity, SCM
   configuration and runner configuration. Record the separate authorized
   elevation path for the future per-machine MSI operation; do not grant rights
   or exercise installation privileges during preflight. Use minimum GitHub
   artifact/runner read permissions and keep registration credentials and keys
   out of reports.
5. The host and runner group are dedicated. Inspect GitHub runner-group access
   restrictions and the allowed trusted workflow/ref policy. Do not permit
   pull-request or fork-controlled code, broad repository access, arbitrary
   workflows or general PR jobs on this runner. Labels alone provide no access
   control. If the installed GitHub plan/configuration cannot enforce the
   required restriction, keep the runner unavailable for jobs and report
   `BLOCKED`; do not substitute a claim of isolation.
6. Record whether CryptoHunter services, MSI registration or product directories
   already exist. The observation does not authorize adoption, repair, reset or
   replacement. An enrolled or ambiguous installation blocks the future clean
   handoff pending a separate reviewed decision.

The runner CLI can validate planned configuration or a captured snapshot:

```powershell
python -m deployment.windows_stage10_runner_preflight `
  --config C:\Stage10Preflight\runner-config.json `
  --snapshot C:\Stage10Preflight\runner-snapshot.json `
  --output C:\Stage10Preflight\reports\runner-rehearsal.json
```

On the actual approved target, `--live` replaces `--snapshot` to collect local
read-only observations. Review the schema and check reasons in
[windows_stage10_runner_preflight.py](../deployment/windows_stage10_runner_preflight.py).
A supplied snapshot remains rehearsal data even when every field looks valid.
Missing GitHub policy/registration evidence remains unverified independently
of a successful local OS or SCM check.

The approved runner configuration has exactly these keys; obtain values from
the operator's host inventory and authenticated GitHub registration:

| Keys | Required values |
| --- | --- |
| `schema_version`, `repository` | `1`, `Dudzian/Dudzian` |
| `hostname`, `machine_guid`, `hardware_uuid` | Expected host name and both stable UUID identities |
| `runner_id`, `runner_name`, `runner_group_id` | Positive GitHub IDs and exact registered name |
| `runner_service_name`, `runner_root`, `runner_service_account` | Exact canonical service name, fully qualified Windows directory and dedicated service account |
| `trusted_workflows` | Explicit repository workflow paths pinned to `@refs/heads/main` |

The live collector runs under the non-administrator dedicated runner service
identity, with PowerShell and read-only `gh` access for organization runner,
group, repository membership and workflow metadata. Future installation
elevation belongs to a separate approved operator. The isolation validator
requires a manual-only trusted workflow and a group restricted to the single
approved runner/repository. The current PR-triggered
`platform-deployment.yml` cannot satisfy this manual-only policy merely because
its Stage 10 jobs have conditional guards. Resolving the trusted dispatch route
and group restriction is a PRE-01B prerequisite; do not weaken the policy or
make the physical runner available to PR code to obtain a green report.

## Qualified MSI custody and provenance

The transfer owner first approves the repository `Dudzian/Dudzian`, canonical
workflow identity, exact full source revision, exact CI run ID/attempt, artifact
ID and artifact name. Obtain expected identity through authenticated GitHub
metadata and the reviewed release decision; never derive expected values from
the downloaded manifest itself. Require a successful canonical qualification
for that selected revision/run. Do not select "latest", a different successful
run, an expired artifact, a PR-produced substitute, diagnostics, or a local
rebuild. The future Stage 10 execution must bind to this same canonical run;
rerunning the workflow requires qualifying the newly selected run again.

1. Download the exact `windows-stage9-qualified-msi` archive and preserve its
   original bytes. Preserve authenticated artifact/run metadata and the
   expected `sha256:` artifact digest. Verify the original downloaded archive
   digest before extraction. A digest of a re-created ZIP or an extracted
   directory is not the artifact digest.
2. Use a fresh operator-controlled staging directory with restricted write
   access. Reject archive traversal, absolute/drive paths, alternate data
   streams, reparse points/symlinks and unexpected wrappers/files. The MSI
   basename must be the manifest's allowed relative MSI file, and all evidence
   must belong to the same approved artifact root. Keep output reports outside
   that root. Do not execute any downloaded binary during rehearsal.
3. Validate the manifest structure, exact MSI SHA-256, canonical installed-file
   map/payload SHA-256 and production executable bindings. A payload map hash
   proves internal manifest consistency; read-only preflight does not extract
   or run the MSI to establish target installed-file bytes. Stage 9's existing
   qualification and later target installed-payload verification supply those
   separate proofs.
4. Reuse the existing Stage 9 clean-install evidence validator to bind the exact
   manifest/MSI, source revision, CI provider/run, x64 Windows qualification,
   successful proof set and reviewed compiler/PostgreSQL contract. Validate the
   supplied evidence against the same qualified receipt; malformed JSON,
   missing values, stale/mixed run or revision, substituted files and changed
   hashes fail closed.
5. Bind the receipt and evidence to the reviewed Production Trust ceremony ID,
   public package manifest SHA-256 and public-trust artifact run ID. Obtain the
   canonical public package separately using that reviewed identity. Use the
   existing public package/Stage 9 validation boundaries; never copy private
   ceremony or target keys. Public-trust artifact run ID is its own explicit
   lineage and need not equal the MSI qualification run ID.
6. Retain the original archive, authenticated metadata, approved identity,
   reports and file hashes in the evidence store. Rehash after each transfer
   and immediately before a later approved install. Operator sign-off identifies
   host, run/revision, artifact ID/digest, exact MSI hash, manifest hash and
   public-trust binding. Any mismatch or expired/missing source evidence stops
   the handoff; no fallback build or inferred `PASS` is allowed.

An artifact-only successful validation does not establish runner existence,
installed product state, production enrollment or Stage 10 readiness.

The artifact CLI reads the original ZIP directly. Keep the archive, metadata,
canonical public-trust package and report below the explicit `--allowed-root`;
it does not install or execute the archived MSI. Metadata copies are the exact
GitHub REST artifact object from
`GET /repos/Dudzian/Dudzian/actions/artifacts/{artifact_id}` and workflow-run
object from `GET /repos/Dudzian/Dudzian/actions/runs/{run_id}`. Require
`windows-stage9-qualified-msi`, an unexpired artifact, its `sha256:` digest and
canonical download URL, matching workflow run/head revision, `main`, the
canonical platform workflow, `workflow_dispatch`, and completed success. An
offline metadata file is not itself evidence of authenticated acquisition;
retain the approved retrieval record and independently selected pins.

Populate the variables below from that reviewed selection, including the
separately approved public-trust run and public package manifest digest. The
trust package directory must have the canonical ceremony ID basename. Do not
set these variables from unvalidated archive contents:

```powershell
python -m deployment.windows_stage10_artifact_preflight `
  --allowed-root C:\Stage10Preflight `
  --archive C:\Stage10Preflight\qualified.zip `
  --artifact-metadata C:\Stage10Preflight\artifact-metadata.json `
  --workflow-run-metadata C:\Stage10Preflight\workflow-run-metadata.json `
  --production-trust-package $ApprovedTrustDirectory `
  --expected-artifact-id $ApprovedArtifactId `
  --expected-archive-sha256 $ApprovedArchiveSha256 `
  --expected-revision $ApprovedRevision `
  --expected-run-id $ApprovedRunId `
  --expected-production-trust-run-id $ApprovedTrustRunId `
  --expected-production-trust-manifest-sha256 $ApprovedTrustManifestSha256 `
  --output C:\Stage10Preflight\reports\artifact-preflight.json
```

See the exact strict validation and report contract in
[windows_stage10_artifact_preflight.py](../deployment/windows_stage10_artifact_preflight.py).

This rehearsal requires a completed successful canonical workflow run. The
future same-run Stage 10 prepare job executes while that run is still active,
so this completed-run check is not yet a deployable same-active-run handoff.
PRE-01B must review how to authenticate successful Stage 9 job/artifact
qualification within the active canonical run, or an explicitly approved
alternative protocol with equivalent run binding. Do not remove the success
check or quietly use an older completed run to fill that gap.

## Rehearsal and review record

Run the artifact validator against approved downloaded public artifacts and run
the runner CLI in snapshot/configuration mode. Retain both reports and the
aggregate preflight report. Offline fixtures are useful for testing rejection
paths; label them as rehearsal inputs and never present them as observed host
or authenticated GitHub evidence.

Combine the two reports without promoting them to live qualification:

```powershell
python -m deployment.windows_stage10_preflight `
  --runner-report C:\Stage10Preflight\reports\runner-rehearsal.json `
  --artifact-report C:\Stage10Preflight\reports\artifact-preflight.json `
  --output C:\Stage10Preflight\reports\windows-stage10-preflight.json
```

The aggregate command deliberately exits `1` with a retained `BLOCKED` or
`FAIL` report and offers no override. Imported summaries are preparation inputs,
not authenticated observations. Omitting either input records it as
`UNVERIFIED`; malformed or unsafe input cannot produce `PASS`. The report
always records `stage10_qualification: NOT_RUN` and
`stage10_execution_authorized: false`. The checked-in
[PRE-01A preparation report](windows_stage10_preflight_pre01a_report.json)
records the available repository facts and absent live runner/artifact inputs;
it is not evidence that production artifacts or a physical host were tested.

The review record must include confirmed conditions and unresolved conditions
with owners. For the present repository state, the minimum honest assessment is:

| Condition | Current interpretation |
| --- | --- |
| Read-only tools and positive/negative tests | `PASS` only after the recorded targeted checks pass. |
| Actual qualified archive and approved GitHub metadata | `UNVERIFIED` until that exact downloaded input is validated. |
| Dedicated physical runner, labels, service persistence, permissions and isolation | `UNVERIFIED` without actual host and GitHub control-plane access. |
| Stage 9 closure and completed legal production enrollment | `BLOCKED` by current status authority. |
| Exact qualified MSI installed on target | `BLOCKED`: the current Stage 10 workflow does not deliver/install it. |
| Reboot recovery and 24-hour lifecycle | `BLOCKED` / `NOT_RUN`; this preflight cannot prove either. |

Record denied access and missing inputs explicitly, rather than treating an
empty observation set as successful. No full multi-hour CI matrix is required
for this rehearsal: run the targeted pure tests, applicable Ruff/type/security
checks and exact `python -m pre_commit run ruff-format --all-files` gate. Keep
the existing Stage 9 tests and at least 70% changed-code branch coverage gate.
Hardware, destructive and self-hosted lifecycle jobs retain separate opt-in.

## Exact next task — S9-S10-PRE-01B

**Implement and review the operator-approved target-host transfer and elevated
installation of the exact current-run `windows-stage9-qualified-msi`, with
post-install provenance/readiness checks and positive/negative tests. Keep live
Stage 10 disabled pending independent Stage 9 closure, legal production
enrollment and lifecycle authorization.**

The PRE-01B implementation must:

1. Close the delivery gap with an explicit reviewed handoff that downloads the
   selected qualified MSI and all three evidence files, validates them before
   use, binds approval to host and immutable run/artifact identities, then
   revalidates immediately before an explicitly approved elevated installation.
   Resolve the completed-run rehearsal versus active same-run execution
   metadata protocol and trusted manual-only runner workflow route explicitly.
   Preserve MSI logs, exact command/outcome and before/after observations.
   Installation is a separate mutation and is not executed by PRE-01A or PR CI.
2. Detect existing MSI registration, canonical services, Program Files,
   ProgramData, ownership records and enrollment markers. Never silently
   uninstall, adopt, overwrite or reset them. Ordinary uninstall preserves
   ProgramData; the separate pre-enrollment reset is restricted to a proven
   authority-free committed base and is forbidden once enrollment begins.
   An enrolled or unknown host requires a separate reviewed migration/recovery
   decision; do not inspect or transfer private-key contents.
3. Verify the installed file set and SHA-256 values against the qualified
   manifest, installed product version and exact MSI/manifest receipt binding.
   Reuse `windows_stage10_runtime_provenance.qualify_runtime_provenance` for
   the current-run backend/version binding rather than creating another trust
   authority.
4. Verify SCM executable paths, quoting/arguments, service SIDs/accounts,
   PostgreSQL dependency and recovery configuration against the installer
   contract. Verify canonical Program Files/ProgramData layout and protected
   ACLs using the existing Stage 9 qualification boundaries, without exposing
   private keys or adding permissive grants.
5. Verify private PostgreSQL version, loopback `127.0.0.1:55432`, expected HBA and
   identity mapping, mTLS role separation and readiness. Check the backend
   readiness PID matches its live SCM process and requires `freshness_runtime`,
   `corehost_lock`, `cross_key_denied` and `cross_roles_denied`; stale readiness
   files cannot establish a running service. Tests must reject mismatched
   backend/version, service path, role, PID, ACL or database configuration.
   Review side effects before invoking existing qualification commands.
6. Handle the intentional pre-enrollment/runtime transition explicitly. Current
   `windows_stage9_clean_install.prove_services` expects PostgreSQL automatic
   and running, but backend/verifier demand-start and stopped. The Stage 10
   live script requires backend automatic and running. A clean MSI install
   therefore cannot prove live readiness. Do not start the backend or change
   its start mode to bypass missing legal enrollment; complete the approved
   enrollment/readiness path independently before lifecycle dispatch.
7. Require independent operator sign-off on the actual target and retained
   reports. Rehearse download, validation, approval denial and mismatch paths
   without installation. Preserve all Stage 9, coverage and security gates.
   Do not add automatic Stage 10 execution to PR jobs or set `run_stage10=true`.

PRE-01B installation completion still does not constitute Stage 10 `PASS`.
Actual same-host reboot, continuity and bounded 24-hour evidence remain a later,
separately approved lifecycle operation after the production gates are closed.
