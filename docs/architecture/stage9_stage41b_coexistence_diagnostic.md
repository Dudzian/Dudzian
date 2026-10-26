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
It repeats snapshots, PID enumeration and installation inventories. Reads and
enumerations have limits; any missing, empty, denied, malformed, oversized or
changed required observation gives `UNCERTAIN`, never an absence certificate.
Kernel threads or zombies with missing executable/mappings are not silently
excluded. A detected old-file reference yields a blocking finding.

`stable_visible_snapshot=true` is only an observation of two equal visible
snapshots. `full_confirmation` is always false. `/proc` visibility checks do not
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
root CLI refusal. New modules enter configured Linux/Windows mypy; tests and
coverage enter the existing quality job without changing hardware gates,
timeouts or coverage thresholds. Existing Ruff and security CI remains enabled.
Physical qualification, independent external artifacts, complete administrative
observations and operator admission authorization remain outstanding.
