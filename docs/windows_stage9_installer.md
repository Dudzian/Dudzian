# Windows Stage 9 — canonical production MSI

## Inventory and decision

The existing desktop/PyInstaller ZIP pipeline remains legacy desktop
packaging.  It is not evidence for `WINDOWS_CLEAN_INSTALL`.  The native path
authority remains `deployment.platforms.windows.resolve_paths`; the frozen
backend SCM identity and recovery policy remain in
`windows_service_recovery`; Stage-4 DACL mutation/qualification remain split
between `windows_dacl_provision` and `windows_dacl_qualification`; and the
Stage-8 PostgreSQL/mTLS authority remains
`postgresql_freshness_authority` plus the Stage-8 probe.  The production MSI
must not package `windows_test_service.py` or the Stage-8 scratch service.

The canonical artifact is a WiX 7 per-machine x64 MSI.  PyInstaller produces
four self-contained executables: the real CoreHost service, the narrowly
scoped verifier, a PostgreSQL SCM host which owns its child process tree, and
the privileged installer provisioner.  The MSI embeds those executables and
the checksum-verified PostgreSQL 17 payload; installation is offline.

## WiX 7 OSMF EULA CI gate

The project owner has not yet recorded conscious acceptance of the WiX 7 OSMF
EULA.  Therefore the clean-install workflow fails explicitly with
`WINDOWS_CLEAN_INSTALL = BLOCKED_ON_WIX7_EULA_ACCEPTANCE`; it must not classify
the result as an MSI installation failure or publish Stage-9 runtime proof.
The blocked attempt is classified as `EXE_BUILD = NOT_RUN`,
`WIX_COMPILE = BLOCKED_ON_EULA`, `MSI_CREATED = NO`, `MSI_INSTALL = NOT_RUN`,
and `CLEAN_INSTALL_PROBE = NOT_RUN`.

Only after the owner records that acceptance may the workflow gate be replaced
by an auditable command-line acceptance at the build invocation:
`wix build -acceptEula wix7 ...`.  CI must not rely on ephemeral per-user
`wix eula accept wix7` state.

The verifier is demand-start because the frozen semantic verifier performs a
bounded privileged verification operation rather than continuous product
work.  PostgreSQL and backend are automatic; backend declares an SCM
dependency on PostgreSQL.

## Frozen layout and endpoint

Immutable files are under `%ProgramFiles%\CryptoHunter`.  Config, State, Logs,
Runtime, Updates, PostgreSQL Data, and Security are under
`%ProgramData%\CryptoHunter`.  No mutable directory is removed by ordinary
uninstall.  The private database listens only on `127.0.0.1:55432`; no firewall
rule is installed.

## Transaction order

WiX copies files and creates all three service objects using `ServiceInstall`.
Only after `InstallServices`, the deferred, non-impersonating provisioner
resolves each live service SID, installs and qualifies protected DACLs,
generates per-machine PKI, initializes PGDATA, provisions and qualifies the
existing FreshnessAuthority schema and final certificate authentication, and
configures the frozen 1s/5s/30s recovery sequence.  Standard `StartServices`
then starts the final services.  A paired rollback action may delete only
resources carrying the current install's ownership marker.  Pre-existing
services, ProgramData, PGDATA, or PKI are conflicts, never adopted.

Mutable transaction authority is never written under Program Files.  A
protected transaction journal beside the ProgramData product root moves from
`PROVISIONING` to `PROVISIONED`; an MSI commit custom action publishes the
durable `COMMITTED` ownership record inside ProgramData.  Rollback removes
only journaled resources in the first two states, while normal uninstall
preserves the committed record and user data.

Private keys are generated on the target and never enter the MSI or manifest.
Final `pg_hba.conf` preserves the Stage-8 source form `cert map=stage8_cert`;
PostgreSQL reports the implicit `clientcert=verify-full` only in its parsed representation,
and `pg_ident.conf` maps `CryptoHunterBackend` to `freshness_runtime` and
`CryptoHunterFreshnessVerifier` to `freshness_crypto_verifier` exactly.

Stage 10 update behavior, GUI lifecycle, reboot qualification, and soak are
explicitly outside this installer.

`WINDOWS_STAGE9_ITEMS` intentionally remains outside the legacy
`windows_acceptance.RESULT_NAMES`: the latter owns the Stage 1–8 acceptance
harness, while `windows_stage9_clean_install` is the sole authority which can
create a clean-MSI receipt.  The evidence producer consumes the same Stage-9
constant and cannot infer PASS from the older acceptance run.

## Canonical first-run authority

The installer and backend never read a bootstrap state from Config and never
invent product identities.  Materialization is composed over
`DurableFirstRunBootstrapCoordinator` and scope lookup over
`DurableFirstRunBootstrapRegistry.current_pre_state()`.  The missing product
integration is represented explicitly by `WindowsExternalProvisioningHandoff`;
the default loader fails closed until a real `external_product_provisioning_boundary`
adapter supplies the accepted claim, binding, initial StateStore metadata and
external recovery ports.  Acceptance fixtures must implement this same
interface and therefore cannot bypass canonical fingerprint, membership or
current-designation checks.
