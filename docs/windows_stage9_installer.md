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

The canonical build runs on CPython 3.11.9.  The cross-platform desktop lock
was generated on the Python 3.11 Linux lock authority, so pip correctly
discarded the `platform_system == 'Windows'` project requirement while
compiling it.  `requirements-windows-runtime.lock` is the reviewed native
supplement and pins `pywin32==312`; both locks are installed with `--no-deps`
before the project is installed without dependency resolution.

## Build reproducibility and artifact identity

The canonical builder freezes the payload inputs rather than claiming
byte-for-byte reproducibility of the complete MSI.  `pins.json` is authority
for CPython 3.11.9, PyInstaller 6.5.0, `PYTHONHASHSEED=0`, and the fixed release
epoch `SOURCE_DATE_EPOCH=1790962380` (`2026-10-02T17:33:00Z`).  Every
PyInstaller process receives those environment values from the builder and
uses `--noupx`, so caller environment variables and an opportunistically
available UPX binary cannot alter the frozen executable payload.

The manifest records the observed Python and PyInstaller versions (which are
checked against the pins), the reproducibility profile, and
`installed_payload_sha256`.  That payload identity is SHA-256 over the
UTF-8 canonical JSON map from relative installed paths to their file hashes,
with sorted keys and fixed compact separators.  It permits payload comparison
between builds without treating the complete MSI hash as a reproducibility
digest.  `msi.sha256` remains the identity of one exact installer artifact.

The production rule is **not rebuild-and-trust, but prove-and-promote**.  After
the canonical clean-install, qualification, uninstall, and revision-bound
evidence succeed, CI uploads the same, unmodified MSI with its manifest,
receipt, and evidence as `windows-stage9-qualified-msi`.  A later persistent
installation must use that qualified artifact for the intended final source
revision; it must not use a local rebuild presumed to have the same MSI hash.

Before any PyInstaller invocation the builder imports the centrally reviewed
Win32 surface: `servicemanager`, `win32api`, `win32con`, `win32event`,
`win32job`, `win32process`, `win32security`, `win32service`,
`win32serviceutil`, and `win32timezone`.  Each of the four generated warning
files is then rejected if it reports one of those required modules missing.
Optional backend warnings remain permitted.  The production CoreHost graph
reaches NumPy, so NumPy 1.26.4's wheel-owned `numpy.libs/libopenblas*.dll` is
required and explicitly collected.  Finally every packaged executable runs
the read-only `--build-smoke` path, which imports NumPy and the Win32 surface
without contacting SCM, installing the product, or performing enrollment.

## WiX 7 OSMF EULA acceptance in CI

The project owner recorded conscious acceptance of the WiX 7 OSMF EULA on
2026-09-28.  Every canonical compiler execution records that authorization in
the CI log and supplies `-acceptEula wix7` directly to `wix build`.  The build
is therefore self-contained and does not read or write persistent per-user
acceptance state.

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
