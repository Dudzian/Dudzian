# Stage 9 / Stage 41B: read-only coexistence diagnostic

Status: **IN_PROGRESS / PRODUCTION_NO_GO / PHYSICAL_TEST_PENDING**.
Qualification: **BLOCKED**. `no_other_installations` remains unchanged.

## Physical evidence supplied by the operator

Ubuntu 24.04, Dell Latitude 5490; installations:

- `782816540373d7b31f426dab645ad115320d1234`
- `26e17e7c23ea6cc1b265d57bfda1a1e6589d9e3f`

For each installation the operator confirmed 1022 receipt-listed files matching
SHA-256, 154 directories, root:root ownership, safe permissions, no symlinks and
no detected Stage 41B accounts, groups, services or runtime paths. Source/CCA
preflight passed except `no_other_installations`. These are supplied observations,
not an independently reproduced measurement in this PR. A locally consistent
receipt is insufficient to authenticate an installation.

The subsequent physical process observations supplied for this hardening are:

| First scanner result | Value |
| --- | ---: |
| `checked_pid_count` | 85 |
| `unavailable_pid_count` | 180 |
| `unstable_pid_count` | 30 |
| `matched_pid_count` | 0 |
| `stable_visible_snapshot` | false |
| `full_confirmation` | false |

The installation metadata report contained 2354 observed entries and
`stable=true`, with `integrity_verified=false` and `provenance_verified=false`.
These are metadata observations, not verification of installation contents.

A second, independent `/proc` classification reported:

| Mapping observation | Process classification | UID scope | Count |
| --- | --- | --- | ---: |
| Empty | Kernel thread | Foreign UID | 142 |
| Nonempty | Other process | Own UID | 86 |
| Permission denied | Other process | Foreign UID | 36 |
| Permission denied | Other process | Own UID | 3 |
| **Total** | | | **267** |

Thus 142 empty mapping observations belonged to kernel threads, while 39 ordinary
processes had actual access denials. The observations came from separate runs;
their PID populations need not agree, and scanner categories can overlap. Zero
matches establishes no absence claim. This PR has not rerun the physical Dell
diagnostic; synthetic regression results are not physical isolation evidence.

## Decision and scope

A complete independent proof of host isolation is unavailable. This change
therefore implements **diagnostic reporting only**. Neither matching files,
operator acknowledgement, an empty process match list nor directory ownership
can authorize coexistence. There is no override, allowlist directory exception,
authorization flag, physical execute operation or cleanup operation. Existing
Source/CCA execution, cleanup, privileged import protection and preflight gates
are unchanged. The separate CLI always exits 2 and never calls an installer.

A future admission path would require complete independent positive evidence for
all boundaries below, an externally authenticated and protected administrative
observer, a closed resource inventory, host visibility and a separately reviewed
operator authorization bound to the exact evidence and Source/CCA plan. This
report cannot be replayed as such authorization. No admission path is implemented.

## File integrity versus provenance

`deployment/stage9_stage41b_coexistence.py` uses current reviewed verifier code,
never imports or executes the historical source, interpreter or installed wheel.
For each exact commit it performs two inventories around the current
`prepare_stage41b_environment.verify_installation` verifier. That verifier checks
the receipt, Git commit/tree/blob Merkle proof against the pinned full commit,
installed source, wheel lock/archive/member bytes and venv/interpreter settings.
The surrounding inventory also checks protected root:root ancestors and entries,
rejects symlinks, special files, unsafe permissions and hardlinked regular files,
and reads bounded no-follow file descriptors, comparing metadata before/after.
Enumeration errors and observed changes yield `UNCERTAIN`.

`integrity=PASS` describes consistency verified by this current verifier.
`provenance=PASS` additionally requires exact comparison with an independently
prepared external inventory whose SHA-256 the operator obtains through a trusted
channel. A mismatch yields `provenance=BLOCKED`; no/malformed evidence yields no
positive provenance result. Git integrity binds the selected source and lock to
a commit; it alone does not authenticate generated venv files or the provenance
of that commit. The external pin supplies the independent trust anchor.

The external JSON has exactly this structure (placeholders are not usable evidence):

```json
{
  "schema": 1,
  "installations": {
    "782816540373d7b31f426dab645ad115320d1234": {
      "files": {"installation.json": "<sha256>", "source/path.py": "<sha256>"},
      "directories": [".", "source"]
    },
    "26e17e7c23ea6cc1b265d57bfda1a1e6589d9e3f": {
      "files": {"installation.json": "<sha256>", "source/path.py": "<sha256>"},
      "directories": [".", "source"]
    }
  }
}
```

Supply **every** file and directory, with sorted directory names. Unlike the local
receipt, this inventory includes `installation.json` itself (1022 receipt-listed
files would mean 1023 entries if no additional files exist). Obtain expected
bytes from independently reviewed source/artifacts, a trusted build record or an
independently verified reconstruction of the exact installation. Do not generate
this trust anchor by hashing the Dell tree or copying its receipt, then present
that same hash as independent evidence. The tool checks the supplied pin and bytes;
it cannot establish how the operator obtained them. No external inventory or pin
for these physical installations was available during this implementation.

## Resource observations

The diagnostic reuses Source/CCA's exact name and numeric NSS lookups, strict NSS
parser, subordinate-ID checks, unit unused-state checks and reserved paths,
including cgroup paths. It repeats these checks and the existing Stage 41B
inventory to detect observed changes. It collects both `list-unit-files --all`
and `list-units --all`, including inactive units, Stage 41B runtime paths and
accounts/groups. It correlates discovered legacy UID/GID values with aliases and
primary groups under unrelated names. Collisions and residual units block.

The result remains `UNCERTAIN` when no conflict is found: known-name discovery
is not a closed inventory, NSS enumeration may omit remote identities, historical
dynamic IDs may no longer be discoverable, and arbitrary unit contents, aliases,
socket/port, mount, IPC and other reservations are not independently exhausted.
Truncation, inaccessible observations and unstable inventories never establish
absence. `source_cca_checks`, `legacy_numeric_ids`, `legacy_identity_aliases`,
`legacy` and explicit limitations explain the observed boundary.

## Process observations

The separate scanner examines every enumerated visible PID and thread, without
filtering by service name or UID. It records stat/status/cgroup, executable,
`cwd`, `root`, open descriptor targets and device/inode identities, plus mappings.
It matches both installation paths and device/inode identities, including aliases,
renamed files and deleted mapped/open files; Stage 41B cgroups also match.
Process correlation uses a separate protected `lstat` walk, not `_inventory`,
file hashing or receipt parsing. Directories need search/list permission; private
root:root 0600 artifacts need no content-read permission for their device/inode
to be observed. The walk retains accessible metadata when another entry/subtree
is unavailable and does not follow unsafe directories. Installation metadata is
sampled before and after PID observation. Changed or incomplete metadata prevents
a stable observation; available identities and pathnames can still identify a
positive process reference. Metadata is never evidence of file integrity or
provenance: both `integrity_verified` and `provenance_verified` in the
`installation_metadata` section remain false. Full content verification is a
separate boundary and remains `UNCERTAIN` when private content is unreadable.

Each source is observed independently. A denied `maps` read does not prevent
inspection of `cgroup`, `exe`, `cwd`, `root` or available descriptors. A failed
descriptor does not erase earlier descriptor targets or their inode identities;
an unavailable thread does not erase observations of another thread. Malformed
mapping entries do not stop matching valid entries or independent sources. Both
PID sweeps examine available positive evidence, including partial observations,
and the second sweep also examines newly visible PIDs. A reference found at any
point remains `BLOCKED`, even if a later source, thread, sweep or installation
metadata observation fails. An observation gap never becomes a negative result.

Kernel threads are identified only through the Linux `PF_KTHREAD=0x00200000`
flag in a validated `/proc/PID/stat` record. Parsing respects process names
containing spaces and parentheses, checks the numeric PID against its procfs
path, validates flags and `starttime`, and compares identity around observation.
Inconsistent, malformed or incomplete records cannot establish kernel-thread
classification. A recognized kernel thread with empty `maps` has its own
category; neither an empty mapping file nor a process name can establish that
category. This classification does not waive missing sources or prove isolation.
Status validation separately requires `Pid`, `Tgid`, all four UID/GID values and
the `Groups` field, with the expected thread and process IDs and unsigned 32-bit
credential values. A well-formed record naming another PID reports an identity
change separately from malformed structure. Missing or malformed
status fields remain an observation gap even when stat identity is readable.

Each sweep retains at most 65536 observed device/inode pairs for correlation
with installation metadata discovered at the end of observation. This preserves
earlier inode-only matches when installation metadata was initially unavailable.
Exhausting that cache reports `PROCESS_INODE_LIMIT`; matching against already
known installation identities continues even after the cache limit is reached.

### Process sweep budgets

Review of merge `d20a1a411f3fe543c84c364bfcdc332fb9961e88` confirmed that
`PROCESS_SWEEP_LIMIT` came from adding `len(repr(snapshot))` for successive PIDs
and comparing the cumulative character count with 64 MiB. This was not a measure
of retained fingerprints or RSS. The crossing PID and every later skipped PID
each added an event; the counter reset for the second independent sweep. Thus
258 events cannot be interpreted as 258 distinct PIDs. A synthetic 130-PID run
crossing at its second PID in both passes reproduces exactly 258 events with
only four snapshot calls and 129 unique unavailable PIDs. The Dell aggregates
do not identify its crossing PID, actual peak memory, or the relative shares of
maps, threads and descriptors. Measurements from different runs remain separate.

The former observer retained raw maps and link records for every thread within
one PID. Matching, fingerprinting and budget accounting repeatedly constructed
full `repr` strings; escaping could further amplify payload size. Even though
raw reads had a limit, counting their cumulative representations stopped later
PIDs after raw snapshots had already been released.

The observer now consumes maps in chunks of at most 64 KiB, hashes bindings and
operational data incrementally, and retains only compact thread identities,
digests, bounded inode sets and fixed issue codes. Metadata reads and pending map
lines are independently capped at 64 KiB. Descriptor enumeration is streamed;
its order-independent digest includes descriptor identity, target and stat
metadata with multiplicity. It does not retain a list of raw link targets.
Task enumeration remains bounded at 1024 entries and descriptors at 65536 per
thread. Reaching any such observation limit remains explicit uncertainty.

`PROCESS_LIMIT` bounds per-PID input/work to 64 MiB, including byte reads and
charged descriptor operations. Exhausting it reports `PROCESS_PID_LIMIT` and
source-specific reasons, retains previous matches and partial observations, and
allows the next PID to be observed. Source, thread and descriptor caps also
report `PROCESS_PID_LIMIT`. It is not an exact allocator measurement.

`SWEEP_LIMIT` independently bounds compact-state reservations to 64 MiB per
pass: 2048 bytes per PID, 1024 per observed thread and 192 per retained inode pair.
These conservative accounting units replace raw-traffic accounting. A PID that
cannot be started within the reservation budget is explicitly unavailable and
skipped. If its completed compact observation exceeds the remaining reservation,
the already detected match and counters survive, but the excess comparison
fingerprint is discarded and `PROCESS_SWEEP_LIMIT` is reported. A separate
per-PID inode correlation record survives when its reservation fits, even if no
full comparison fingerprint can be retained. The first pass
remains available while the second runs, so these are two independent bounded
stores, not a claim of a 64 MiB total RSS ceiling. PID inventories, installation
metadata, aggregate sets and one bounded in-flight observation use additional
memory. The global limit has not been increased.

The PID base reservation covers the correlation holder and fingerprint header;
each retained inode pair is charged exactly once to the same sweep budget.
These reservations are committed immediately after reference collection, before
fingerprinting or further validation can fail. Thread/fingerprint reservations
are charged separately only when they fit. Discarding the full fingerprint does
not release or erase its funded inode record. Final installation metadata is
correlated against these independent records from both sweeps; identity and
binding comparisons use only retained full fingerprints. The per-sweep 65536
inode cap still applies, with no additional global inode cache.

As before, a full fingerprint crossing exhausts that pass and later PIDs are
explicitly skipped. An exhaustion flag replaces accounting beyond the budget;
discarded fingerprint storage is not charged as retained memory. This also
preserves the bound on aggregate thread-count tracking: beyond admitted thread
reservations, at most one crossing PID per pass contributes its bounded thread
set (at most 1024 entries). The existing observed-thread counter remains a count
of unique observed PID/TID pairs. Correlation records and retained fingerprints
together stay within the unchanged per-pass reservation limit.

The inode cache is also bounded independently. Known installation inode matches
are checked immediately even after the cache is full; unretained identities
cannot be correlated with metadata discovered later and `PROCESS_INODE_LIMIT`
makes that late-correlation gap explicit, including when the sweep has room for
a PID holder but not even one inode pair. No raw process payload or exception
text enters JSON.

Synthetic regression measurements on Python 3.12 in the validation container
used `tracemalloc` around observation, after fixture construction:

| Workload | Input processed | Peak traced allocation |
| --- | --- | --- |
| 768 PID population, two sweeps | 75,497,472 bytes per sweep | 2,588,444 bytes |
| 49 threads streaming shared maps | 13,848,576 bytes | 319,997 bytes |
| 8192 distinct descriptor targets | 33,151,914 bytes | 81,130 bytes |

The population case supplies synthetic legacy snapshots; the thread and
descriptor cases exercise the real compact observer through local path adapters.
The descriptor fixture constructs its paths and materializes their lazy parts
before tracing, excluding process-global `pathlib` name-interning table resizes
caused by fixture setup. Target strings are still generated during observation;
the peak and retained-memory assertions are unchanged. Other interpreter
allocation during observation is included. These are regression measurements,
not RSS ceilings or physical Dell results. They do not qualify procfs visibility,
permissions or host isolation.

Stability is assessed separately for process identity and installation
references. PID/`starttime` changes, thread membership changes, credentials,
device/inode identities, permissions and reference-bearing data remain relevant.
Ordinary CPU/activity counters are excluded only where their field meaning can
be identified reliably. Unparsed or incomplete data remains uncertain. Changed
identity and changed observation content are reported separately from access
denials, vanished sources, empty data, malformed data and limit exhaustion.

Process counters describe unique PIDs observed across both sweeps:

| Field | Meaning |
| --- | --- |
| `checked_pid_count` | At least one source was retained; the observation may be partial. |
| `per_pid_limit_count` | Unique PIDs reaching a per-PID, source, thread or descriptor observation limit in either sweep. |
| `sweep_budget_limit_count` | Unique PIDs affected by a compact-state sweep budget limit in either sweep, including a crossing observation whose fingerprint cannot be retained. |
| `skipped_pid_count` | Unique PIDs skipped before observation in at least one sweep because that sweep lacked a PID reservation; a PID may still have been observed in the other sweep. |
| `unavailable_pid_count` | At least one source was denied, missing, empty, malformed or limited, apart from separately classified empty kernel maps. |
| `unstable_pid_count` | A PID/source disappeared, the PID set changed, or identity/reference-bearing content changed. |
| `matched_pid_count` | Any retained observation positively referenced an installation or Stage 41B cgroup. |
| `kernel_thread_pid_count` | The same coherent leader PID/`starttime`/`PF_KTHREAD` identity was observed across both sweeps. |
| `empty_kernel_maps_pid_count` | A leader meeting that two-sweep kernel identity check had empty mappings during at least one sweep. |
| `identity_unstable_pid_count` | A parsed PID/`starttime`/kernel-thread identity changed. |
| `bindings_unstable_pid_count` | The normalized reference/security observation changed. |
| `operationally_changed_pid_count` | Only identified operational fields changed between sweeps, while normalized observations agreed. |

Categories overlap: a PID with a positive partial observation and a later denial
remains both matched and unavailable. Kernel counters require coherent leader
identity across both sweeps; changed or missing identities do not qualify for
those counters. Other source gaps can still coexist with a kernel classification.
`observed_pid_count` aliases the checked count; `observed_thread_count` counts
unique observed PID/TID pairs. `stable_process_identities` and
`stable_installation_bindings` are conservative indicators: incomplete or unstable
process observations, partial/failed enumeration or uncertain procfs visibility
keep both false, and changed/incomplete installation metadata also prevents
stable bindings.

Sorted unavailable/unstable/matched PID lists and `reason_counts` explain gaps.
Reasons use fixed source codes such as `MAPS_PERMISSION_DENIED`, `FD_NOT_FOUND`,
`MAPS_EMPTY`, `STAT_MALFORMED`, `THREAD_LIMIT`, `STAT_IDENTITY_CHANGED` and
`PROCESS_CONTENT_CHANGED`; generic I/O errors have a separate `IO_ERROR` suffix.
The counts describe events across both sweeps rather than unique PIDs, and each
reason count saturates at 65536. Internal reason collection also aggregates fixed
codes with saturated counts instead of retaining an unbounded error list. An
empty `maps` read bracketed by a coherent kernel identity is classified separately
rather than reported as `MAPS_EMPTY`; its aggregate PID counter additionally
requires matching leader identity across both sweeps. Other missing or denied
sources remain explicit. The public report contains no raw `/proc` data, process
environments, pathnames, exception messages or potential secrets.

`stable_visible_snapshot=true` describes stable, complete visible observations
under the scanner's checks. It grants no qualification. `full_confirmation` is
always false. `/proc` visibility checks do not
externally attest the host PID/mount namespace. Sequential observations are not
an atomic host snapshot; copied data, already-closed scripts and short-lived
processes can escape the observed references. These limitations keep the final
qualification blocked even when all visible snapshots are stable and empty of
matches. Existing Source/CCA process observations are not upgraded or reused as
proof of Stage 41B isolation.

## Invocation and privilege boundary

Run the **current reviewed checkout** as an ordinary account, with bytecode
writes disabled, without using either historical installation's Python:

```bash
python -B -m scripts.diagnose_stage9_stage41b_coexistence \
  --run-id 012345abcdef --uid-base 62000 \
  --evidence /trusted/external-inventory.json \
  --evidence-sha256 <independently-reviewed-sha256>
```

Omit both evidence arguments to diagnose missing independent evidence. JSON is
printed to stdout. An unreadable protected receipt or process is expected to
produce uncertainty. No permissions are changed to obtain access.

The CLI refuses root before importing observer modules or reading the host.
There is deliberately no privileged mode and no flag asserting that startup was
trusted. Any future privileged read still requires verification **before Python
startup** of the protected interpreter, control code, dependencies, import roots
and startup artifacts through an external trusted mechanism. In-process metadata
checks or a hash of already-executed code cannot discharge this requirement.
Do not run the historical installation code as root or relax receipt permissions.

## Validation

Synthetic tests cover exact independent pins, malformed/duplicate manifests,
file and receipt replacement, current Git/wheel verification, changes during
reads, inode aliases, deleted mappings, cgroups, unrelated identities, numeric
UID collisions, inactive units, denied/missing/hidden procfs, process churn and
root CLI refusal. Process regressions additionally cover validated kernel threads
with empty mappings, ordinary empty mappings, independent matching after maps or
executable denials, disappearing descriptors, inaccessible sibling threads,
PID/`starttime` changes, operational-counter changes, malformed flags and mapping
records, sticky first-pass positives and fully unavailable observations. Stable
synthetic observations still assert blocked qualification and false full
confirmation. New modules enter configured Linux/Windows mypy; tests and
coverage enter the existing quality job without changing hardware gates,
timeouts or coverage thresholds. Existing Ruff and security CI remains enabled.
Physical qualification, independent external artifacts, complete administrative
observations and operator admission authorization remain outstanding.

## CI contract regression versus platform qualification

Review of PR #3210 HEAD `c5db034bffbd187ae395ae3d2856aba9ca277d97`
confirmed the same Python CI failure on Ubuntu, Windows and macOS: the history
partition contract omitted the two coexistence test files and two coverage
modules already present in the workflow. The contract now lists those entries
explicitly. Full set equality, uniqueness checks and all earlier entries remain.
No workflow assignment changes: atomic runs 75 unique test files with 63 unique
coverage modules; history retains its four test files and 64 unique coverage
modules. Both jobs cover the diagnostic module and CLI independently.

The separate [platform run](https://github.com/Dudzian/Dudzian/actions/runs/40367928000)
failed with `MISSING_CURRENT_EVIDENCE`: Linux lacks systemd, clean-install and
restart evidence; macOS lacks launchd, permissions/authentication/IPC and
lifecycle evidence; Windows lacks reboot recovery, long-running lifecycle,
update/restart and failed-update safety evidence. These are unsatisfied production
qualification conditions, not regressions introduced by the coexistence code.
All three platform readiness results remain `NOT_READY`; their fail-closed gates
and evidence requirements are unchanged. A passing Python contract does not
satisfy those production gates or qualify the physical Dell.
