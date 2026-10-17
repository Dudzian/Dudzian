# Stage 9 passive PostgreSQL SQL qualification — S9-S10-PRE-01B.2C

The passive adapter supplies a narrow collection mechanism and reports each
available authority separately. It does not close the complete
`postgresql_effective_sql` gate. A clean installation still has no running
backend or verifier process from which their genuine service authentication
can be observed. Missing identities and catalog permissions remain `BLOCKED`.

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

The collector has no connection, DSN, role, certificate-path, process-name or
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
an enrollment credential.

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
| `service_authentication` | The current canonical service process authenticated as its own expected role, with session/current user, endpoint and TLS evidence. It proves only that service's session. |
| `role_catalog` | Read-only role attributes and membership checks for the required separation. Catalog consistency cannot authenticate the other Windows service. |
| `active_hba` | Remains `BLOCKED`: parser rows alone do not supply a complete fresh authentication matrix for the active rules. |
| `cross_role_authentication` | Remains `BLOCKED`: one permitted service session cannot supply both real service identities and their negative authentication matrix. |

PostgreSQL's `pg_hba_file_rules` reports the rules currently parsed from files.
On Windows, changes apply to subsequent new connections without the usual
reload step; the adapter nevertheless keeps parser evidence separate from
the observed authentication outcomes. It does not reload configuration or
promote parser consistency to a complete active-HBA PASS.

Existing backend and verifier startup checks catch broad `psycopg.Error`
exceptions in
[`backend_service._run`](../deployment/windows_installer/backend_service.py)
and
[`verifier_service._run`](../deployment/windows_installer/verifier_service.py).
Their readiness JSON and `cross_roles_denied` flags are not reused as SQL
authority: a timeout, TLS failure or unavailable server is not evidence that
PostgreSQL rejected a forbidden identity. Role-catalog checks similarly do
not replace actual authentication attempts under each real service token.

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

## Qualification still required

The implementation and hermetic negative-path tests are reviewable software
evidence. This Linux execution environment cannot establish native Windows
x64 token behavior, service/key isolation, effective installed PostgreSQL
configuration or production readiness. The native hardware binding also
requires an accessible SMBIOS 2.6-or-newer UUID; an unsupported or unavailable
identity blocks collection. Emulated native collectors and SQL responses do
not establish those facts. No native or production qualification is claimed
here.

The next step requires separate authorization for collection within each
canonical service process after the accepted legal runtime authority permits
those services to run. A reviewed service-local collection and authenticated
delivery boundary must preserve the two identities and bind both observations
to the same live host and PostgreSQL instance. The complete matrix must
distinguish actual authentication rejection from connection failure. Catalog
reads for HBA and control-system evidence also need an explicitly approved
read authority if the frozen roles cannot perform them. This adapter neither
grants that authority nor starts a service to obtain it.

Until those authorities and native evidence exist, their capabilities and the
combined SQL gate remain `BLOCKED`. The
[`stage9_current_status.json`](../deployment/stage9_current_status.json)
authority is unchanged: Stage 9 remains `IN_PROGRESS`, legal production
enrollment remains `NOT_PERFORMED`, production readiness remains false, and
Stage 10 remains `BLOCKED` / `NOT_RUN` with `run_stage10 = false`. Shipping
this adapter does not authorize enrollment, installation, service starts or
Stage 10.
