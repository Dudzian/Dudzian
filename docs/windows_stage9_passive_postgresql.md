# Stage 9 authenticated service-bound PostgreSQL evidence — S9-S10-PRE-01B.2D

The passive adapter and service-channel implementation keep each evidence
authority separate. Fixed service-owned SQL capture and authenticated local
delivery are reviewable software; no production service endpoint is operational
under the current authority contracts. Inspector-side committed enrollment
verification and service-side committed runtime admission are both required.
The latter has no approved service-accessible adapter and always fails closed.
The clean-base path still has no running backend or verifier process and never
contacts these endpoints. Missing admission authority, identities and catalog
permissions remain `BLOCKED`.
`active_hba` and the complete `postgresql_effective_sql` gate remain `BLOCKED`;
native Windows qualification of the new channel is `NOT_RUN`.

## Why clean-base cannot supply both service identities

The clean-base validator
[`_base_services`](../deployment/windows_stage9_install_qualification.py)
requires `CryptoHunterBackend` and `CryptoHunterFreshnessVerifier` to remain
Demand/Stopped, with no process. The private-key DACLs installed by
[`_pki`](../deployment/windows_installer/provision.py) permit only SYSTEM and
the owning service SID; an administrator inspection session does not become
that service by selecting its PostgreSQL role or certificate filename.

The frozen
[`final_hba_lines` and `ident_lines`](../deployment/windows_postgresql_auth_contract.py)
admit the two separate certificate identities at `127.0.0.1:55432` and reject
other connections. During installation,
[`_configure_database`](../deployment/windows_installer/provision.py) retains
an already authenticated bootstrap administrator connection only long enough
to qualify the final parser output and reject a fresh administrator login.
That connection is closed and its bootstrap postmaster is stopped. It is not
an inspection authority available after a clean installation.

Consequently, starting either service, creating a helper under its account,
duplicating or impersonating a service token, reading its key as SYSTEM,
changing a key ACL, reopening bootstrap access, or substituting an administrator
session would change the authorized scope. The passive adapter performs none
of these operations. Stage-8 scratch services and
`provision.qualify_mtls()`, which starts the verifier, are not passive
alternatives.

The enrolled-runtime channel does not change this clean-base boundary. Its
server uses temporary identification-only impersonation solely to authenticate
the inspecting pipe client, then calls `RevertToSelf` and verifies the absence
of a thread token before service SQL can run. It never executes SQL under the
inspector's identity or impersonates either service on the inspector's thread.

## Collection API and trust boundary

[`windows_stage9_postgresql_passive.py`](../deployment/windows_stage9_postgresql_passive.py)
provides three narrow entry points:

- `collect_passive_postgresql(observation)` creates a process-local live
  receipt from native collection. The observation supplies the host, paths,
  services and listener binding to compare; it cannot assert a SQL identity.
- `summarize_passive_postgresql(receipt, observation)` exports safe JSON
  diagnostics. This summary is suitable for a report, not for authorizing a
  later live check.
- `require_passive_postgresql_component(receipt, observation, component)`
  checks one capability and raises a status-bearing failure when its live
  authority or required facts are unavailable or contradictory.

The direct collector has no connection, DSN, role, certificate-path, process-name or
administrator-session parameter. It derives the permitted role from the
current native Windows process. Before connecting, it requires native Windows
x64, the exact canonical virtual-service TokenUser SID, no impersonated thread
token, and agreement between the current PID, executable, process creation
time and SCM service instance. A service SID present merely as a group does
not make an administrator or SYSTEM token the service identity.

The collector uses the owning service's fixed certificate and key paths only.
It never opens the other service's private key to test access. The connection
uses the fixed loopback endpoint and `freshness_gate` database, `verify-full`
TLS and a fixed CA, through the native psycopg connection implementation.
It does not accept a caller-provided connection or SQL-result callback.

Only fixed, allowlisted `SELECT` statements collect evidence. Startup options
enforce read-only transactions, a 5-second statement timeout, a 1-second lock
timeout, a 5-second idle-transaction timeout and a 15-second transaction
timeout; connection establishment is limited to 5 seconds. The private
connection also imposes a 5-second client socket-wait deadline, independently
of the server's statement timeout. A driver without the distinct polling
interval and absolute-timeout API is rejected before credential-path access.
The desktop dependency lock pins psycopg 3.3.6, whose wait implementation
supports that deadline. No SQL changes roles, grants, configuration,
application data or server lifecycle. No service, MSI, TPM or NV operation is
part of collection.

The receipt is sealed in an in-process registry, bound to the collecting
process and host observation, and expires after 60 seconds. Native process,
SCM and listener observations and SQL instance observations are checked
through collection. Different host bindings, instance changes, expired or
copied receipts, dictionaries, readiness markers, and saved report summaries
cannot substitute for the original live receipt. An opaque receipt is a
short-lived implementation capability, not a transferable authorization or
an enrollment credential. The authenticated service channel issues receipts
through the same registry and component consumers; its wire payloads and saved
summaries are never receipt inputs.

## Authenticated delivery from already running services

The transport and SQL implementation below can collect only after the distinct
production admission requirements are satisfied. Its hermetic tests exercise
the mechanics with emulated authorities; they cannot authorize a real endpoint.
The implementation separates native transport, service-owned SQL capture and
inspector-owned qualification:

- [`windows_stage9_evidence_pipe.py`](../deployment/windows_stage9_evidence_pipe.py)
  owns the two fixed local named pipes. Each endpoint reserves its name for its
  lifetime with `FILE_FLAG_FIRST_PIPE_INSTANCE`, rejects remote clients and
  uses a protected DACL for SYSTEM, enabled administrators and its owning
  service SID. Inspector access excludes `FILE_CREATE_PIPE_INSTANCE`; handles
  are not inheritable. The client pins each canonical SCM service process,
  primary TokenUser SID, executable and creation time, and compares the native
  pipe-server PID with that pinned process. A service SID present only as a
  token group cannot satisfy the identity check.
- [`windows_stage9_service_sql.py`](../deployment/windows_stage9_service_sql.py)
  runs inside that service process and checks only its own token, process and
  SCM entry. It derives the owning credential paths locally and accepts no
  observation, SQL text, DSN, role selector or connection factory from the
  request. Fixed read-only queries capture effective configuration, the own
  authenticated session, catalog rows and the PostgreSQL instance witness.
  Both forbidden-role probes use that same service's own certificate. The
  own SQL session stays open through validation and the inspector's ACK;
  successful connection cleanup is confirmed before a receipt can carry PASS.
- [`windows_stage9_service_evidence.py`](../deployment/windows_stage9_service_evidence.py)
  performs administrator host/listener inspection independently, connects to
  both pinned peers and validates their exact fixed-query traces. It requires
  matching system identifier, data directory and postmaster lifetime, checks
  the own session/instance again after the role probes and rechecks native
  host and service identities. It issues an opaque process-local receipt only
  through this authenticated collection path. Dictionaries, previously
  captured responses and report summaries cannot invoke a receipt issuer.

Requests contain only the version, independent random nonce, shared collection
ID and monotonic issue/expiry times. A collection has a 30-second lifetime;
frames are limited to 256 KiB and reject duplicate keys and nonfinite numbers.
The endpoint rejects replayed nonces, bounds its replay cache and allows one
request per connection. Pipe connection is limited to five seconds; message
I/O and disconnect waits are cancellable and bounded by the request deadline.
An invalid client cannot authorize SQL or terminate the endpoint's request
loop. Failure to revert the identification token terminates the endpoint
thread before a handler can run. A missing, malformed or late cleanup ACK
cannot issue passing evidence.

The receipt remains bound to its original observation object, host fingerprint,
inspector PID and 60-second receipt lifetime. Every subsequent capability check
also revalidates the native PostgreSQL lifetime and both service process
identities. A stopped or restarted service, reused PID, changed executable,
changed listener or PostgreSQL restart revokes the original live authority.

### Lifecycle and legal enrollment authority

[`service_base.run_service`](../deployment/windows_installer/service_base.py)
keeps `START_PENDING` until the existing product worker explicitly reports
`RUNNING`. Only the canonical backend and verifier then start their evidence
thread. The thread attempts `_confirmed_service_evidence_enrollment` before
exposing the endpoint and again before every SQL request. This service-local
gate resolves the existing canonical handoff, which currently fails closed.
Even if a handoff claim were accepted, it would not prove committed enrollment:
the gate still raises
`SERVICE_LOCAL_COMMITTED_ENROLLMENT_AUTHORITY_UNAVAILABLE`. No approved adapter
currently supplies committed runtime admission readable by each owning
service, so neither production endpoint can be exposed.

The administrator's `_confirmed_legal_enrollment` verifier remains on the
inspector side. Its committed records and `State`/`ProductionTrust` access
assumptions cannot be transferred to the verifier service. The service-local
gate does not call that administrator qualifier, enumerate its protected
files or treat accepted claim data as a committed record. Local file presence,
file contents and readiness markers are not substitute admission authorities.
No new role, ACL grant, privilege or alternate enrollment authority is added;
production integration remains partial until an approved service-accessible
committed admission contract exists.

Endpoint failures emit only a fixed `BLOCKED` warning and do not grant worker
readiness. Stop, shutdown and worker failure cancel the endpoint and close its
pipe and held SQL resources. The lifecycle waits for endpoint cleanup before
reporting `STOPPED`; an unclean thread cannot be reported as a completed stop.
The frozen dependency contract includes `win32file` and `win32pipe`; the
existing MSI dependency-closure tests cover that import surface. This source
contract is not evidence that an installed Windows executable ran successfully.

The enrolled installation-qualification path selects
`collect_service_postgresql(observation)` only after resolving the
administrator's canonical committed enrollment authority. The public collector
also performs that administrator qualification before opening either pipe;
a caller cannot bypass admission by invoking the collector directly. The
service-side gate remains an independent requirement. Clean-base collection
continues to use the direct passive collector with both application services
Demand/Stopped. Neither collector starts, restarts, stops or reconfigures a
service.

### PostgreSQL 17.11 client certificate DN rendering

The installer [`_certificate`](../deployment/windows_installer/provision.py)
creates a subject containing exactly one CN. For these certificates, PostgreSQL
17.11 exposes `/CN=CryptoHunterBackend` or
`/CN=CryptoHunterFreshnessVerifier` in `pg_stat_ssl.client_dn`, including the
leading slash. The tagged source traces this value from
[`pg_stat_ssl`](https://github.com/postgres/postgres/blob/REL_17_11/src/backend/catalog/system_views.sql#L983)
through
[`pg_stat_get_activity`](https://github.com/postgres/postgres/blob/REL_17_11/src/backend/utils/adt/pgstatfuncs.c#L572)
and
[`backend SSL status`](https://github.com/postgres/postgres/blob/REL_17_11/src/backend/utils/activity/backend_status.c#L348)
through
[`be_tls_get_peer_subject_name`](https://github.com/postgres/postgres/blob/REL_17_11/src/backend/libpq/be-secure-openssl.c#L1516)
to
[`X509_NAME_to_cstring`](https://github.com/postgres/postgres/blob/REL_17_11/src/backend/libpq/be-secure-openssl.c#L1639),
which prefixes every attribute with `/` and escapes its value. This differs
from a whole-subject RFC4514/RFC2253 rendering such as `CN=CryptoHunterBackend`.

The adapter compares the entire observed DN to the exact expected `/CN=...`
value. Plain `CN=...`, another identity, extra attributes, duplicate CNs and
CN values containing a matching prefix or suffix cannot satisfy this check.
It does not extract a CN or accept a suffix match.

Verification inspected the exact `REL_17_11` source and reproduced its OpenSSL
rendering loop on ephemeral certificates generated by the isolated installer
`_certificate` function; both identities produced the slash-prefixed values
above. No PostgreSQL server, service or MSI was started. This source/rendering
check does not establish installed-server behavior or native Windows
qualification.

## Separate evidence and statuses

| Component | Meaning and limit |
| --- | --- |
| `effective_configuration` | Effective listener, port, TLS and server-version settings from the permitted SQL session. Static configuration files alone do not supply this component. |
| `instance_identity` | SQL control-system/data-directory identity tied to the native PostgreSQL service and listener. Insufficient catalog authority leaves this component `BLOCKED`. |
| `parsed_hba` | Exact `pg_hba_file_rules` parser rows checked using the frozen authentication contract. Catalog permission denial remains `BLOCKED`. |
| `service_authentication` | The direct collector proves only its current service's session. The authenticated channel requires both genuine service peers, each own session/current user, endpoint and TLS certificate identity, bound to the same PostgreSQL instance. |
| `role_catalog` | Read-only role attributes and membership checks for the required separation. Catalog consistency cannot authenticate the other Windows service. |
| `active_hba` | Remains `BLOCKED`: parser rows alone do not supply a complete fresh authentication matrix for the active rules. |
| `cross_role_authentication` | The paired channel can supply PASS only when both own sessions and the same PostgreSQL instance are proven and each own credential receives explicit authentication denials for the other service role and `postgres`. Missing peers or inconclusive probes remain `BLOCKED`; forbidden-role acceptance is `FAIL`. The direct single-service collector remains `BLOCKED`. |

PostgreSQL's `pg_hba_file_rules` reports the rules currently parsed from files.
On Windows, changes apply to subsequent new connections without the usual
reload step; the adapter nevertheless keeps parser evidence separate from
the observed authentication outcomes. It does not reload configuration or
promote parser consistency to a complete active-HBA PASS.

The backend and verifier startup checks in
[`backend_service._run`](../deployment/windows_installer/backend_service.py)
and
[`verifier_service._run`](../deployment/windows_installer/verifier_service.py)
now use the shared
[`probe_postgresql_role_denial`](../deployment/windows_postgresql_auth_contract.py)
helper. Only a PostgreSQL authentication SQLSTATE of `28000` or `28P01`
establishes rejection. Timeout, TLS failure, unavailable server, missing
SQLSTATE and other errors fail closed before readiness publication; an
unexpected successful connection is closed and rejected. The service-bound
collector applies the same explicit-denial rule and preserves forbidden-role
acceptance as `FAIL`, including when connection cleanup also fails.

Readiness JSON and `cross_roles_denied` flags remain diagnostic and are not
reused as SQL authority. Actual fresh probes under both genuine service tokens
must be bound to successful own sessions and the same live PostgreSQL instance.
Role-catalog consistency cannot replace those probes. Successful sampled
denials and parser consistency still cannot prove the complete active-HBA
matrix, so `active_hba` is always `BLOCKED` in this implementation.

The host-inspection and installation-qualification reports retain the original
combined `postgresql_effective_sql` requirement and expose the partial component
results separately. A partial PASS never supplies a combined PASS. Missing
access, sessions, identities, permissions or complete reads are `BLOCKED`;
evidence that contradicts the required contract is `FAIL`. Saved observations
are rehearsal inputs, so consistent checks become `UNVERIFIED`, never live
PASS. Malformed inputs cannot establish a component PASS. The safe report
summary is identified separately from static files and readiness markers.

The fixed SQL contract classifies missing or incompatible catalog objects,
functions, schemas, settings or types as `FAIL` for SQLSTATEs `42703`, `42P01`,
`42883`, `3F000`, `42704`, `42804` and `42846`. A returned required row with
the wrong width is also `FAIL`; an absent required row remains `BLOCKED`.
Permission denial (`42501`), unavailable authentication, connection failures,
timeouts and unclassified errors remain `BLOCKED`. No SQLSTATE class is
blanket-classified as a schema failure. Reports use fixed diagnostic text;
server messages, diagnostic details, connection strings and credentials are
not exported. A later unavailable read cannot downgrade an observed `FAIL`.
The authenticated transport preserves these classifications. Catalog permission
denial cannot be promoted by a second service's successful query; the pair
requires each relevant authority. No grants, role changes, bootstrap access,
key ACL changes, schema changes or configuration reloads are added to obtain
missing evidence.

## Qualification still required

The implementation and hermetic negative-path tests are reviewable software
evidence. This Linux execution environment cannot establish native Windows
x64 token behavior, service/key isolation, effective installed PostgreSQL
configuration or production readiness. The native hardware binding also
requires an accessible SMBIOS 2.6-or-newer UUID; an unsupported or unavailable
identity blocks collection. Emulated native collectors and SQL responses do
not establish those facts. No native or production qualification is claimed
here.

Fixed SQL capture, authenticated transport and paired receipt validation are
implemented, with lifecycle attachment guarded by unavailable committed
service-local admission. Production integration is incomplete. Before native
qualification, both the administrator's committed enrollment verifier and a
reviewed service-accessible committed runtime admission adapter must be
available under the existing identity and ACL contracts. An accepted handoff
claim alone satisfies neither missing committed service admission nor native
evidence. This change does not widen protected-file access to make it work.

Native Windows x64 qualification is **NOT_RUN**. After those production
admission requirements are satisfied, it requires a separately authorized
installed host with both canonical services already running. The qualification
plan is:

1. Read the real pipe DACL and demonstrate that an ordinary user, a filtered
   administrator and a remote client cannot collect; verify SYSTEM/enabled
   administrator access, noninheritable handles, exclusion of client
   `FILE_CREATE_PIPE_INSTANCE`, first-instance squatting rejection and no
   service-key ACL changes.
2. Verify real SCM/pipe PID, TokenUser, executable and creation-time agreement
   for each peer. Reject an impersonating thread, a service SID present only
   as a group, a substituted endpoint, stopped service, restart/PID reuse and
   changed host/listener/PostgreSQL lifetime before and after collection.
3. Exercise duplicate/replayed nonces, another collection ID, expired/future
   requests, malformed/oversized frames and replay-cache exhaustion. Verify
   identification-only caller inspection and unconditional `RevertToSelf`
   before any SQL or key access; failed reversion must stop the endpoint.
4. Observe both real own-role mTLS sessions and the same system identifier,
   data directory and postmaster lifetime. Run each credential's other-role
   and `postgres` probes. Only explicit `28000`/`28P01` can pass that component;
   unavailable server, timeout, TLS failure and missing SQLSTATE stay
   `BLOCKED`, while actual forbidden-role acceptance is `FAIL`.
5. Exercise success, rejected clients, denied catalog reads, disconnect before
   ACK, invalid/missing/late ACK, SQL/pipe timeout, service stop/shutdown and
   worker failure. Independently verify SQL sessions, pipe/process/token/SCM
   handles and endpoint threads are cleaned up; confirm completion only after
   the own SQL session closes, and no `STOPPED` report precedes cleanup.
6. Repeat receipt consumption after each service and PostgreSQL restart, and
   after expiry, observation copying and saved-summary replay. Verify revoked
   receipts cannot supply any live capability. Check the frozen dependency
   closure in the actual installed Windows executables.

These are required native observations, not results inferred from the hermetic
tests. Catalog reads for HBA and control-system evidence still need explicitly
approved read authority if the frozen roles cannot perform them. The collector
does not grant that authority or start a service to obtain it. No native ACL,
peer authentication, installed-service, SQL or cleanup PASS is claimed here.

The existing fast quality job includes the three new modules in Ruff, explicit
format checks, configured Linux/Windows mypy and branch coverage, and adds the
pipe, SQL, receipt, inspector-admission, lifecycle, denial and dependency-closure tests to its
existing pure-logic selection. Timeouts, hardware opt-ins and the 70% diff
coverage gate are unchanged. Local verification passed 667 targeted tests,
configured mypy across 326 files, repository/changed-source Ruff and direct
format checks; diff coverage was 88%, above the existing 70% gate. With all new
test files staged, the exact mandatory
`python -m pre_commit run ruff-format --all-files` passed on the final scope
without formatter changes. Betterleaks 1.9.0 matched both the official release checksum
and the repository's pinned SHA256; the official wrapper scanned the full
22-file changed tracked/untracked scope and found no leaks. Semgrep CE 1.179.0
ran the five custom blocking rules on 1,035 production files with zero findings;
all five rule-fixture tests passed. Tool caches and the isolated Semgrep
environment remained outside the repository. The maintained community-rules
advisory baseline and full CI matrix were not run locally. These software
checks do not supply native Windows, installed-service or production admission
evidence.

Until those authorities and native evidence exist, their capabilities and the
combined SQL gate remain `BLOCKED`. The
[`stage9_current_status.json`](../deployment/stage9_current_status.json)
authority is unchanged: Stage 9 remains `IN_PROGRESS`, legal production
enrollment remains `NOT_PERFORMED`, production readiness remains false, and
Stage 10 remains `BLOCKED` / `NOT_RUN` with `run_stage10 = false`. Shipping
this adapter does not authorize enrollment, installation, service starts or
Stage 10.
