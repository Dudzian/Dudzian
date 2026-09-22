# Windows-first deployment architecture audit

This pre-deployment refactor does not advance S9D/M1 and does not claim that
FreshnessAuthority is implemented. The canonical machine-readable source is
`deployment/platform_readiness.json`; statuses describe evidence, not intent.
Requirements (`release_gates`, invariants and platform matrix) are distinct from
per-item execution evidence (`status`, `evidence`, `evidence_class`). A static
proof class can never satisfy a separately required live-integration item.

## Discovery inventory

| Component | Current owner | Current platform assumption | Target | Current tests | Current blocker |
|---|---|---|---|---|---|
| Trading/runtime, scheduler, health | `bot_core` / `core` | mostly portable; several CLI signal handlers | CORE | runtime and architecture suites | per-OS execution still required |
| CoreHost lock | `bot_core.runtime.core_host` | explicit `msvcrt`/`fcntl` branches | CORE through narrow lock boundary | process-lock/recovery tests | Windows crash/reboot/locked-handle evidence absent |
| SQLite StateStore | persistence layer | SQLite/WAL, host filesystem | CORE | persistence, backup and recovery suites | Windows forced-termination evidence absent |
| Freshness semantic/CAS/receipt | `bot_core` and frozen M05 contracts | semantics portable | CORE | architecture/security/PostgreSQL tests | frozen implementation flags remain false |
| Freshness live qualification | `bot_core.freshness_deployment_qualification` | `pwd`, `grp`, systemctl, `/etc`, `/run`, peer/pg_ident | LINUX | static and live qualification tests | live PID-1 systemd unavailable here |
| Verifier IPC | verifier service | Unix socket, `SO_PEERCRED`, POSIX group | LINUX | verifier/deployment tests | Windows authenticated named-pipe design not implemented |
| PostgreSQL production-local | deployment templates | PostgreSQL 16 Unix socket, peer, pg_ident | LINUX | schema/qualification tests | Windows install/auth/backup/ordering not frozen |
| systemd service | deployment unit | systemd RuntimeDirectory, restart and sandbox | LINUX | exact static unit tests | live restart/reboot evidence absent |
| Existing install script | offline administrator | root, POSIX accounts and Linux paths | LINUX | static deployment tests | not a Windows installer |
| Preview/desktop packaging | scripts/PyInstaller | Windows build artifacts, not a 24/7 service | WINDOWS (non-production today) | packaging contracts and workflows | no production SCM installer |
| Updater | scripts/core update contracts | mixed desktop execution; M1 unresolved | CORE state + OS executor | updater/architecture tests | trusted Windows handoff/rollback not implemented |
| GUI/backend lifecycle | PySide UI and runtime commands | no canonical independently installed backend service | WINDOWS boundary | UI/runtime contract tests | split-service acceptance is currently not applicable |
| Logging | core/runtime modules | file/console depending entry point | CORE events + OS sink | observability tests | Windows persistent bounded service sink absent |
| CI | GitHub Actions | broad Windows-first main job plus mixed matrices | ALL tracks | workflow contract tests | deployment integrations were not independent |
| macOS deployment | build scripts only | packaging, no launchd production service | MACOS | packaging tests | launchd/security/auth lifecycle not implemented |

## Frozen ownership and mechanisms

Core owns business semantics, configuration schemas, persistence semantics,
health/readiness events, recovery and update state. It must not import any
platform adapter. `deployment.platforms.contracts` defines narrow service,
security/path, process-supervision and updater-execution protocols.

Windows is the reference platform. Its canonical target is a genuine SCM
service named `CryptoHunterBackend`, running as the virtual least-privilege
identity `NT SERVICE\CryptoHunterBackend`, never the interactive user or
LocalSystem. Immutable binaries belong under Program Files. Machine config,
state, logs, runtime and staging are separate children of ProgramData; GUI
per-user state belongs under LocalAppData. The installer alone may provision
DACLs. Runtime qualification reads them and fails closed; it never repairs.

The intended Windows IPC is a local ACL-protected named pipe bound to the
service identity. This is not yet security-qualified. PostgreSQL must observe
an authenticated Windows principal with distinct verifier/runtime/admin roles.
Unauthenticated localhost and caller-provided role strings are forbidden.
SSPI is a candidate, not a frozen implementation decision: packaging,
PostgreSQL distribution support and exact database mapping need review first.

Linux retains the complete systemd, POSIX account/mode, Unix socket,
`SO_PEERCRED`, peer/pg_ident, `/etc`, `/run`, restart and sandbox hardening. It
is now explicitly Linux acceptance evidence and is not a Windows gate.

macOS has a distinct launchd boundary. It must eventually use native paths,
identity/permissions, IPC authentication, lifecycle and update execution; no
systemd emulation or unexecuted PASS is allowed.

## Release and evidence policy

Each platform gate includes core evidence plus only its own acceptance items.
`NOT_IMPLEMENTED`, `NOT_TESTED`, `UNVERIFIED_ENVIRONMENT_LIMITATION` and
`FAIL` all keep a gate false. `NOT_APPLICABLE` is descriptive and is not used
to satisfy a required item. Linux/macOS results can never influence the
Windows gate. Legacy global booleans remain conservative and are explicitly
not inputs to these platform gates.

Live Windows SCM, DACL, reboot, file-locking, installer and long-running items
may pass only on Windows. Live systemd may pass only with systemd as PID 1.
Live launchd may pass only on macOS. Unit tests with fake boundaries establish
code behavior, not live integration evidence.

The Windows path proof is deliberately split: `WINDOWS_PATH_LAYOUT_STATIC_CONTRACT`
uses `PureWindowsPath` only to verify the frozen layout, while
`WINDOWS_NATIVE_PATH_INTEGRATION` requires execution on a real Windows filesystem.
Likewise, `LINUX_INSTALLER_STATIC_CONTRACT` proves reviewed installer structure,
whereas `LINUX_CLEAN_INSTALL` requires a privileged live Linux installation.

CI contract jobs may be green after static checks. Deployment integration and
release jobs invoke `python -m deployment.platform_readiness --platform <OS>`;
they exit non-zero unless every item required by that platform gate has PASS
evidence. The Windows release job depends on all three core jobs and Windows
deployment evidence only, never Linux systemd or macOS launchd integration.

## Revision-bound execution evidence

Live and cross-OS PASS values are never consumed from the repository baseline.
`deployment/platform_evidence.schema.json` freezes the evidence envelope and
`deployment.platform_evidence` is the only CI producer. Evidence binds the
source revision, CI run, runner OS/architecture, timestamp, probe and result.
The release consumer rejects stale revisions, wrong platform/OS, unknown items,
unsupported classes, malformed documents and every duplicate result.

Each core job writes only a revision-bound runner marker after its executable
tests pass. The Windows release job aggregates exactly Linux, Windows and macOS
markers into one `CORE_REQUIRED_SUITES` result. Deployment integration results
are separate and cannot be substituted by Linux systemd or macOS launchd.

The first Windows vertical slice is a reviewed pywin32 SCM harness. Its probe
installs the exact `CryptoHunterBackend` service using the canonical virtual
service identity, verifies SCM configuration, starts it, observes a deterministic
health marker, stops it gracefully, starts it again, and always removes it. Only
successful completion emits PASS for install/start/graceful-stop/manual-restart.
This repository change does not claim those results: no remote Windows run was
executed from the current sandbox, so the declared baseline remains unchanged.

## Canonical core acceptance plan

`deployment/core_required_suites_v1.json` is the sole selector manifest for
`CORE_REQUIRED_SUITES_V1`. It includes CoreHost lifecycle/recovery, StateStore,
schedulers, network/health, configuration validation, platform-neutral trading
runtime, update state and common architecture/security contracts. Every suite
is required on Linux, Windows and macOS. The frozen policy requires zero pytest
failures, errors, skips, xfails and unexpected passes.

Systemd/launchd/SCM deployment, external exchange credentials/services, GUI,
live PostgreSQL and OS-only behavior are explicitly excluded and remain owned
by their integration tracks. Each core job invokes `deployment.core_test_plan`,
not an inline workflow selector list. A marker is written only after the entire
manifest succeeds and binds schema, revision, runner OS, plan ID, deterministic
SHA-256 plan digest, GitHub run ID and provider.

Core aggregation requires exactly three markers and rejects an extra marker,
duplicate OS, stale revision, stale plan, different run, wrong provider or an
incomplete Linux/Windows/macOS set. Production evidence trusts provider identity
`https://github.com`; `LOCAL_REVIEWED_EXECUTION` can be used to exercise the
plan locally but is not accepted as GitHub release evidence.

Native host names are normalized once by `deployment.host_identity`: Python's
`Linux` and `Windows` remain unchanged, while `Darwin` maps to the canonical
evidence label `macOS`. Any other host fails closed. Core execution and the
Windows SCM producer share this helper; marker/evidence schemas continue to use
only `Linux`, `Windows`, and `macOS` rather than native `Darwin`.
