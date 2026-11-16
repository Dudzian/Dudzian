# Stage 9 / Observer 3A — V5d2 operator deployment preparation

This package prepares a reproducible, reviewable operator deployment. It does
not install software, provision infrastructure, approve policy, qualify a host,
authorize an experiment, or change V5d1/V5c3 acceptance semantics. All deployment
actions below belong to a future independently reviewed operator procedure;
none is performed by the inventory or by this preparation task.

Contract-review baseline: `Dudzian/Dudzian` main at
`213226f6222f8d1ece05987afe038c82ed61923d` (merged PR #3277).

```text
Stage 9 = IN_PROGRESS
Observer 3A = BLOCKED
Stage 3B = NOT AUTHORIZED
Production = NO-GO
HOST_PREFLIGHT_VERIFIED = NOT_CLAIMED
EXECUTION_AUTHORIZATION = NOT_AUTHORIZED
PYC_RECONSTRUCTION = UNVERIFIED
Exact CPython execution = NOT_TESTED
```

## Retained source contracts

The implementation, rather than a successful inventory or an installation
recipe, defines the accepted profile:

| Source | Deployment requirement or boundary |
| --- | --- |
| [`stage9_observer_collector_identity_v5d1.py`](../../deployment/stage9_observer_collector_identity_v5d1.py) | Fixed external anchors/policy, protected-file rules, strict signed-policy schema, measured code paths, environment fingerprint, unconditional missing independent-launch evidence. |
| [`stage9_observer_kvm_preflight_v5d1.py`](../../deployment/stage9_observer_kvm_preflight_v5d1.py) | Fixed requirements file, complete expected runtime structure, policy binding, atomic native/image evidence, blocked execution and qualification. |
| [`stage9_observer_kvm_domain_v5d1.py`](../../deployment/stage9_observer_kvm_domain_v5d1.py) | Only read-only `qemu:///system`, domain lookup by approved UUID, narrow stopped-domain XML profile, missing effective enforcement evidence. |
| [`stage9_observer_kvm_image_v5d1.py`](../../deployment/stage9_observer_kvm_image_v5d1.py) | Protected immutable image chains, fixed `/usr/bin/qemu-img`, held-descriptor checks, bounded metadata inspection and no partial image success. |
| [`stage9_observer_kvm_preflight_v5d1.md`](stage9_observer_kvm_preflight_v5d1.md) | Separate Ubuntu host, external approval ceremony, reviewed launch, native read-only procedure and remaining qualification dependencies. |
| [`stage9_observer_pyc_execution_v5c3.py`](../../deployment/stage9_observer_pyc_execution_v5c3.py), [`runtime_identity_v5c3.py`](../../deployment/stage9_observer_runtime_identity_v5c3.py), and [V5c3 architecture](stage9_observer_pyc_execution_v5c3.md) | Synthetic-only collector admission, exact runtime/worker inputs, independent enforcement, full external runtime trace and opaque output collection; no real-execution admission path. |
| [`stage9_closure_matrix_v5d1.json`](stage9_closure_matrix_v5d1.json) | Separate software, physical observation, identity, isolation, runtime, trace, authorization and acceptance gates; no change is justified by this preparation. |
| [`pyproject.toml`](../../pyproject.toml) | Host Python `>=3.11`, complete project dependencies including `cryptography>=50.0.1,<51`, and optional `observer-kvm-preflight` extra with `defusedxml==0.7.1`. |

Contract regressions are in `tests/deployment/test_stage9_observer_` modules
for collector identity, KVM domain/image/preflight, runtime identity V5c3 and
PYC execution V5c3. Their synthetic signatures, mocked immutable files and
fake libvirt results are software tests, never operator approval artifacts or
physical receipts. V5c3 rejects `CONTROLLER_OBSERVED` before collector callbacks
and cannot emit `LOCAL_EXECUTION_OBSERVATION`; V5d2 preserves that refusal.

## Observed laptop and permitted diagnostic role

The operator reports a Dell Latitude 5490, Ubuntu 24.04.5 LTS, kernel
7.0.0-43-generic, x86_64, read/write KVM-device access, cgroup v2, QEMU/libvirt,
successful read-only local system libvirt access, and a clean checkout at the
baseline above. Six existing CryptoHunter domains are registered and stopped.
These are reported diagnostic observations, not independently accepted
qualification evidence, and were not repeated on physical hardware here.

The operator reports that `/opt/cryptohunter`, its `observer-v5d1` installation
and venv/interpreter, and `/etc/cryptohunter/observer-v5d1` are absent. Consequently
`operator-anchors.json`, `approved-policy.json` and `domain-requirements.json`
are unavailable. The native controller deployment is therefore blocked before
any question of physical acceptance.

The laptop can collect the unprivileged V5d2 inventory and retain absence,
access and metadata diagnostics. KVM access and cgroup availability do not
establish isolation, trusted launch or independence. Formal use requires a
separate Ubuntu observer host operated under an independently reviewed local
security policy, independence of the expectation/approval source and launch
witness, separation from the investigated guest, and independently accepted
host/enforcement evidence. There is no approved hardware-model, Ubuntu release,
kernel, libvirt or QEMU version matrix in the retained contracts; this plan
does not invent one or infer approval from the reported versions. The laptop
must not automatically become that independent observer host.

The six existing domains are not disposable qualification targets merely
because they are stopped. Do not select, reuse, redefine, start, stop, attach
storage to, or modify any of them. A separate approved disposable domain and
its UUID must be supplied by independent provisioning.

## Inventory use and interpretation

Run `bash scripts/stage9_operator_v5d2_inventory.sh` from the reviewed checkout
using an ordinary unprivileged operator account. The inventory observes host,
repository, dependency and fixed deployment-path metadata. It does not launch
the protected Python interpreter, import the controller, inspect image contents,
validate signed policy contents, or access signing keys. Its read-only libvirt
probe uses only the local `qemu:///system` URI. There is no policy, URI, trust
path, image path or approval override.

Use an independently reviewed Bash/runtime and trusted host tools. A hostile
shell startup or substituted command cannot be authenticated by this script.
The Git, package-query and libvirt subprocesses receive cleared environments
with only explicit settings: inherited Git trace destinations and libvirt
logging/authentication overrides cannot select files. Git disables configured
Trace2 outputs, global/system configuration, optional locks and filesystem
monitoring. Libvirt directs error logging to stderr and authentication/history
paths to `/dev/null`, with stdin closed to interactive input. No password is
supplied; failed access remains a refusal. These `/dev/null` sinks are
output/authentication controls, not alternative trust authorities. They do not
attest all native authentication backends: before inventory, separately review
that the fixed local read-only socket uses local peer identity without
production credentials. For example, a SASL/GSSAPI configuration may otherwise
consult a default Kerberos cache. If that local diagnostic boundary cannot be
established, do not run the native probe or treat its availability as tested;
retain the blocker for operator review. This narrow diagnostic review does not
qualify the host or approve an observer security policy.

An optional `--repository PATH` selects only the Git checkout to observe; it
does not select installation, trust or libvirt paths. Output consists of
tab-separated `ITEM`, `STATUS`, `DETAIL` records followed by the fixed safety
status lines. Exit 0 means collection completed, including missing or blocked
prerequisites; malformed arguments exit 64. Linux/x86_64 observations do not
establish an approved Ubuntu distribution or release. The distro therefore
remains `UNVERIFIED`. Tool discoverability and package-database records do not
establish an approved dependency version or Python importability.

A symlink at the documented `venv/bin/python` location is reported `BLOCKED`
pending independent launcher review. This conservative diagnostic does not
change V5d1's artifact rules: the required interpreter pin is the actual
`/proc/self/exe` target, and the launcher alias separately needs review. These
metadata observations neither accept nor reject a signed deployment policy.

| Inventory status | Interpretation and response |
| --- | --- |
| `PRESENT` | Requested existence/access/metadata observation succeeded; authenticity and approval remain separate. |
| `MISSING` | Required object/tool was not found; return it to the operator as a deployment prerequisite. |
| `INACCESSIBLE` | Inspection could not obtain required access; retain uncertainty without treating it as absence or broadening permissions. |
| `UNVERIFIED` | Metadata or existence does not establish provenance, approved versions, signatures, launch identity or completeness. |
| `BLOCKED` | Observed unsafe or unsupported state, or missing mandatory independent evidence, prevents advancement. |
| `NOT_TESTED` | The inventory did not perform the named measurement or operation. |

A complete inventory is a collection result, never a deployment-readiness
decision. Unknown access errors remain unknown; metadata checks are subject to
concurrent replacement and cannot replace V5d1's held-descriptor validation.
No inventory status may become `HOST_PREFLIGHT_VERIFIED`. Retain stdout,
stderr and exit status using an operator-chosen, already prepared receipt
destination; the script itself creates no evidence directory or file. Preserve
the script's exact bytes/hash and repository revision with the receipt. A
nonzero/incomplete collection remains an incomplete diagnostic. No failure
permits privilege elevation, installation, trust creation or VM mutation.

## A. Repository and release verification — preparation

1. Independently obtain the release and verify its exact commit, provenance and
   contents through the operator's approved acquisition process. Bind the
   reviewed baseline above, final V5d2 commit, source distribution, dependency
   declarations, launcher, code closure and test receipts in the review record.
2. Inspect checkout cleanliness and differences against the approved release.
   The diagnostic may read Git metadata but never checks out, fetches or pulls.
   A writable checkout supplies review material; it is not a trusted controller
   installation and must never be executed as root to establish trust.
3. Have a separate reviewer accept the proposed deployment contents. Local Git
   SHA values prove object identity, not release authority or independent
   acceptance. Missing provenance, mismatched content or unresolved differences
   stop deployment preparation at this step.

Expected result: an independently reviewed exact release/dependency/launcher
proposal, with no installed trust or physical qualification claim.

## B. Protected installation preparation — separate operator action

The documented installation layout is:

```text
/opt/cryptohunter/observer-v5d1/
  deployment/                         reviewed imported controller modules
  scripts/stage9_observer_kvm_preflight_v5d1.py
  venv/bin/python                     reviewed host verifier launch entry
  venv/...                            reviewed Python/native package closure
/etc/cryptohunter/observer-v5d1/
  operator-anchors.json
  approved-policy.json
  domain-requirements.json
```

`/opt/cryptohunter/observer-v5d1` is the retained documented deployment location,
not a caller-selectable trust fallback. V5d1 can measure approved actual
installed code paths; the external `/etc` authority and requirements paths and
`/usr/bin/qemu-img` remain fixed. A different installation requires its own
explicitly reviewed launcher and exact signed artifact paths; the inventory
continues to describe the documented `/opt` layout.

Every measured artifact is a regular, root-owned, single-link file; every
ancestor directory, including filesystem root, is root-owned and lacks group
or other write permission. Every component is free of symlinks. The contract
checks UID 0 and `mode & 022 == 0`; it does not prescribe one exact numeric mode
such as 0600/0700 or prohibit all group/other read access. Operator readability
and directory traversal must already be available. Root compromise remains
outside what these metadata checks can authenticate.

The minimum explicit controller-code list is `deployment/canonical_json.py`,
`deployment/stage9_observer_collector_identity_v5d1.py`,
`deployment/stage9_observer_kvm_domain_v5d1.py`,
`deployment/stage9_observer_kvm_image_v5d1.py`,
`deployment/stage9_observer_kvm_preflight_v5d1.py`, and
`scripts/stage9_observer_kvm_preflight_v5d1.py`. Imported repository `.py`
modules under `deployment` and `scripts` must also be pinned. Installing only
these six files is not a complete Python/import/native closure proof.

The retained contracts do not supply an approved privileged installer,
ownership-transition ceremony, immutable-storage provisioning tool or complete
loaded-dependency attester. A separately reviewed administrator must prepare
and verify the final installation outside this task. Do not improvise `sudo`,
directory creation, permission changes or a checkout-as-installation fallback.
An ordinary venv may use symlink launchers; a symlink in an artifact path fails
the retained protection rule. The actual `/proc/self/exe` target is the required
interpreter artifact; the launcher path and its startup behavior additionally
require independent review. Do not hide a symlink observation or relax the
validator to make an installation pass.

Expected result: protected installed bytes and documented independent custody.
Missing installation or unreviewed layout remains `BLOCKED`.

## C. Python and host dependencies — independent verification

Use a trusted host verifier Python compatible with `requires-python >=3.11`.
This is distinct from the investigated guest `/usr/bin/python3.12`; do not
execute that investigated interpreter to validate deployment. The reviewed
environment must satisfy the complete project dependency declaration and the
`observer-kvm-preflight` extra, including `cryptography>=50.0.1,<51` and exactly
`defusedxml==0.7.1`. Test/type-check tooling additionally declares
`types-defusedxml==0.7.0.20240218`. Importing a few packages successfully does
not establish the complete dependency closure or trusted package provenance.

The retained preparation guidance selects
`python -m pip install '.[observer-kvm-preflight]'` from independently verified
source material using the operator's approved package acquisition process.
This is future environment preparation, not an inventory/preflight command or
authorization to install into a protected path. Review all resolved versions,
package digests, native libraries and startup files before a separate protected
installation handoff. Do not replace the complete dependency declaration with
an invented minimal environment or silently use host fallback packages.

Required host facilities are Linux local KVM/libvirt, a libvirt daemon, the
reviewed system `libvirt-python` binding and its `libvirtmod` extension, QEMU
with `/usr/bin/qemu-system-x86_64`, and protected `/usr/bin/qemu-img`. Ubuntu
package names commonly corresponding to these components are
`python3-libvirt`, `libvirt-daemon-system`, `qemu-system-x86` and `qemu-utils`;
these names are acquisition hints, not a retained package-version allowlist.
The inventory also uses available Bash, coreutils, Git and `virsh` for its
read-only observations. The V5d1 native controller uses Python libvirt directly;
presence of `virsh` alone does not establish availability to the protected venv.

Verify system binding/Python ABI compatibility and approved import resolution
in the exact launch environment. Do not automatically add system site packages,
alter `sys.path`, or choose another interpreter to fix imports. Such changes
alter the reviewed environment and require explicit independent review. Missing
bindings, parser, crypto dependency or mismatched versions block native
collection; the inventory reports environment authenticity as `UNVERIFIED`.

## D. Disposable domain provisioning — independent operator task

Supply an independently prepared persistent, shut-off (`SHUTOFF=5`) KVM domain,
selected by its approved canonical lower-case UUID. It must have no managed-save
state or libvirt snapshots. Read-only access must already be granted to local
`qemu:///system`; access denial is a refusal. Never broaden socket permissions,
switch to `qemu:///session`, use a remote URI or invoke lifecycle operations.

The exact XML profile and all rejected semantics are maintained in
[V5d1's supported-domain table](stage9_observer_kvm_preflight_v5d1.md#supported-domain-and-requirements-profile).
Its deployment-critical settings include:

- HVM `arch=x86_64`, exactly `machine=pc-q35-8.2`, one explicit `hd` boot device,
  and emulator `/usr/bin/qemu-system-x86_64`.
- Exact independently expected memory/currentMemory, static vCPU count and
  CPU period/quota. Defaults are 512 MiB, one vCPU and 100000/100000 microseconds;
  explicitly approved values may differ within parser bounds.
- CPU `mode=host-model`, `check=full`; explicit topology sockets/dies/clusters/
  threads 1, cores equal to expected vCPUs; one explicit `clock offset=utc`.
- Explicit `destroy` for each poweroff/reboot/crash policy, USB controller
  `model=none`, balloon `model=none`.
- Only expected file-backed raw/restricted-QCOW2 disks, `snapshot=no`, explicit
  empty `readonly`, virtio `vda`, `vdb`, etc. in expected order; driver `name=qemu`,
  `cache=none`, `io=native`, `discard=ignore`, `detect_zeroes=off`; exact approved
  absolute source and complete backing paths/formats.
- No network interfaces, passthrough/host devices, shared directories, channels,
  sockets, consoles/serial, graphics, RNG, TPM, shared memory, vsock, QEMU
  extensions, NVRAM, unsupported addresses/confinement/resource semantics,
  unknown nodes or duplicate singleton settings. Current/persistent reviewed
  settings must agree.

Distribution-generated XML may fail this deliberately narrow profile. Preserve
the refusal and return it for separate provisioning review; do not redefine a
domain to clear it. Matching XML proves requested configuration only. Effective
QEMU identity, sockets, network isolation, namespaces/cgroups/LSM/seccomp,
capabilities, devices and CPU/RAM/process/wall-time enforcement remain missing.
A stopped VM cannot prove future process enforcement; a host `pids.max` does
not bound guest processes. Measurement-channel trust and whole-VM cleanup
authority are separate unresolved gates, not permission to add a channel or
start a VM during this procedure.

## E. Immutable images and runtime expectations — independent preparation

For each disk provide the complete expected top-to-base chain, using canonical
absolute paths and exact independently approved SHA-256 values/formats. Every
analysis-copy chain member must be a root-owned, regular, single-link protected
file with protected ancestors, no symlink components, and Linux
`FS_IMMUTABLE_FL` already set on storage that supports the implemented ioctl.
Independently establish source provenance, a lifecycle fence and absence of
privileged writers. A stopped VM, read-only mount/descriptor or identical stat
samples does not satisfy the immutable-file rule.

Supported formats are raw and restricted QCOW2 v2/v3, no encryption, internal
snapshots, incompatible feature bits, external data files or nonstandard v3
headers. Every backing path must be canonical absolute, expected, acyclic and
present in complete ordered chains. Missing flags, unsafe paths, unsupported
formats or uncertain protection remain `IMAGE_MEASUREMENT_BLOCKED`. Preparing
copies/flags/storage is a separate reviewed task; do not convert, flatten,
repair, mount, snapshot or modify images during diagnostics.

An approved external extractor must separately capture exact interpreter,
loader, libraries/native modules, standard-library/import tree and
loader/provider configuration bytes from the same immutable view. Independently
bind aliases, package versions and common snapshot provenance. Whole-image
hashes, guest file lists and host `DT_NEEDED` analysis do not provide this
complete `RuntimeInventoryV5c3`. Full actual runtime access/mapping tracing and
trusted output extraction remain unimplemented qualification blockers.

## F. Independent operator policy approval — external decision

A separate reviewer must approve the exact installation/tool identities,
environment constraints, disposable-domain UUID, image chains and complete
expected runtime inventory. Record reviewer identity, scope, decision,
contract/release revision, expectation provenance and content digests. Do not
derive accepted expectations solely from the running collector's proposals.
Presence of JSON files, valid syntax or a matching checksum is not approval.

The three protected fixed files have strict versioned JSON with unknown and
duplicate keys rejected. Hexadecimal digests/keys/signatures are lower case.

| Object | Exact fields and constraints |
| --- | --- |
| `operator-anchors.json` | `profile="stage9-observer-controller-anchors-v5d1"`, `authority_id`, 32-byte `public_key_hex`, positive integer `approved_revision`, `approved_policy_sha256`. |
| `approved-policy.json` | Exactly `policy` and 64-byte `signature_hex`. |
| Nested `policy` | `profile="stage9-observer-controller-policy-v5d1"`, matching `authority_id`, matching `revision`, `artifacts`, `environment_sha256`. |
| `artifacts` | 6–1024 unique entries, each exactly canonical absolute `path`, `role`, full `sha256`; all six roles required: `controller-code`, `interpreter`, `libvirt-code`, `libvirt-extension`, `qemu-tool`, `domain-requirements`. Exactly the fixed requirements path has the requirements role. |
| `domain-requirements.json` | Exactly `profile="stage9-observer-kvm-requirements-v5d1"`, `domain`, `disk_chains`, `runtime_inventory`; at most 256 KiB. |
| `domain` | Exactly `domain_uuid`, `memory_bytes`, `vcpu_count`, `cpu_period_us`, `cpu_quota_us`; canonical UUID, positive integer values, never booleans. Upper bounds: 1 TiB, 64 vCPUs, 1000000 period and 64000000 quota microseconds. |
| `disk_chains` | 1–8 arrays of 1–16 exactly `{path,sha256,format}` records each; globally unique canonical absolute paths, full SHA-256, format `raw` or `qcow2`. |
| `runtime_inventory` | Exactly `files`, `needed_resolutions`, `loader_configuration_paths`, `import_paths`, `provider_configuration_paths`. 5–512 files, each exactly `path`, `sha256`, `build_id`, `role`, `aliases`; existing V5c3 roles/optional configuration-absence/Build-ID/loader-alias rules apply. Each resolution is `[consumer_path,needed_name,resolved_path]`; 1–64 import paths and bounded configuration/resolution lists. |

Use the source validators for all remaining bounds and structural constraints;
passing structure does not approve completeness or origin. The signed policy
must cover the actual kernel-selected `/proc/self/exe` target, required/imported
controller modules, loaded libvirt Python/native files, fixed QEMU image tool
and the exact raw-byte requirements digest. Formatting requirements JSON changes
its raw-byte hash and requires renewed independent approval. V5d1 also bounds
each measured policy artifact to 256 MiB.

The environment fingerprint binds implementation/version, architecture,
UID/GID/groups, `sys.path`, isolation/site/environment flags and selected
startup-related environment values. Review the underlying values from the
exact approved launch independently; copying an unexplained fingerprint into
a policy is self-approval. Missing independent approval stays `BLOCKED` even
when every file is present and metadata appears safe.

## G. Ed25519 verification and trust custody — external authority

The independent authority keeps its signing private key outside the repository
and the investigated guest; the inventory and deployment preparation never
access it. The external directory contains only public verification material,
signed policy and approved requirements. Do not generate ephemeral test keys,
self-sign a deployment policy, or copy synthetic test artifacts there.

The anchor selects both the exact policy revision and
`SHA256(canonical_json_bytes(policy))`. Verify the signature against the
canonical JSON bytes of exactly:

```text
{"domain":"stage9-observer-controller-policy-signature-v5d1","policy":policy}
```

Use the reviewed `deployment.canonical_json.canonical_json_bytes`, not generic
JSON serialization or signing the digest alone. A different validly signed
policy under the same key is not accepted unless the exact anchor digest and
revision select it. Missing anchors, signature mismatch, pin mismatch,
unprotected/symlink/hardlink paths or changed bytes are refusals. Return them
to the independent custodian; no command-line replacement authority exists.

Root ownership does not authenticate a root-compromised host. This local
operator trust ceremony does not change production trust authorities, rotate
production keys or implement durable anti-rollback. Local timestamps and
revision pins provide neither authenticated freshness nor durable rollback
protection.

## H. Trusted controller launch — separately reviewed diagnostic

Before any installed Python runs, an independent launch witness must review
the launcher/interpreter, package/native dependency closure, site startup,
imports, sanitized environment and loaded execution identity. On-disk hashes
and a valid policy cannot prove how already loaded code started. The retained
identity implementation always reports `COLLECTOR_IDENTITY_UNVERIFIED`,
`INDEPENDENT_LAUNCH_ATTESTATION_MISSING` and
`LOADED_EXECUTION_CLOSURE_UNVERIFIED`; it has no accepted launch-attestation
admission path. A normal `-I -B` venv launch also retains
`SYSTEM_SITE_BOOTSTRAP_UNVERIFIED`. Document this blocker without adding an
attestation flag, self-assertion or undocumented no-site fallback.

After separate approval for this diagnostic launch and completed preparation,
use the fixed launcher in the next section. It runs the trusted host verifier,
never the investigated CPython/bytecode. Approval to collect a diagnostic does
not qualify the host or authorize target execution. If the exact reviewed
launch cannot be established safely, stop before invoking it.

## I. Read-only native collection — later diagnostic observation

This future command is retained from V5d1 for an already prepared protected
installation and pre-existing receipt directory, with no symlink components
or competing writers. It is not executed by V5d2 inventory or this task:

```bash
umask 077
set -C
env -i LANG=C.UTF-8 LC_ALL=C.UTF-8 TZ=UTC PATH=/usr/bin:/bin \
  /opt/cryptohunter/observer-v5d1/venv/bin/python -I -B -c \
  'import runpy,sys; sys.path.insert(0,"/opt/cryptohunter/observer-v5d1"); runpy.run_path("/opt/cryptohunter/observer-v5d1/scripts/stage9_observer_kvm_preflight_v5d1.py",run_name="__main__")' \
  > preflight-v5d1.json
v5d1_preflight_exit=$?
sha256sum preflight-v5d1.json
```

Use a receipt name that does not already exist. Noclobber/refused redirection
is a failed collection, not permission to overwrite evidence. Record the exact
command, exit status and report hash with original stdout/stderr bytes. Do not
run this inside a shell configuration that discards exit 2 before recording it.
The explicit import path is part of the independently reviewed launcher.

Normal report exit is **2**, including complete native/image reads; `--help`
alone exits 0 and collects no observation. JSON must preserve
`execution=EXECUTION_BLOCKED`, `qualification=BLOCKED`,
`host_preflight=NOT_TESTED`, and
`execution_authorization=EXECUTION_NOT_AUTHORIZED`. A process/import
error, missing receipt, malformed JSON or unexpected status is incomplete
evidence; do not reinterpret it as successful collection or readiness.

The native controller checks only the approved UUID over read-only
`qemu:///system`, brackets image reads with observations, rechecks policy and
requirements, and closes the connection. Any mismatch, incomplete native read,
changed identity, interrupted collection or cleanup failure discards the
complete positive image collection. There is no accepted partial-chain success.
Retain refusal codes and missing-enforcement fields unchanged. Never clear
them by editing receipts, running lifecycle operations, modifying images,
changing trust/permissions or wrapping JSON as a synthetic collector.

## J. Evidence retention and independent acceptance

Retain original inventory/native receipts, stdout/stderr, exit codes, exact
commands, repository/installation/launcher identities, artifact and receipt
SHA-256 values, environment, approved contract revision, reviewer/verifier
identity, expectation provenance and all negative/unknown findings. Bind each
receipt to the exact reviewed revision and collection environment. Local hashes
provide integrity comparison, not authenticated origin or trusted time.

Software validation commands/results and final revision are retained with the
focused draft PR. Synthetic negative-path tests prove inventory behavior only;
skips remain skips. Physical-host measurements, native CI and independent
acceptance are separate evidence streams. Do not copy this preparation's
software results into physical closure gates.

The closure matrix remains unchanged. The independent reviewer must separately
resolve identity-policy and launch gates, host/domain/image physical observations,
effective isolation and measurement channel, complete runtime bytes and trace,
whole-VM stop/cleanup authority, reconstruction plan, concrete experiment
authorization, later real-observation admission, provenance registry, acceptance
authority, authenticated freshness and durable anti-rollback. The read-only
controller cannot supply the missing lifecycle, execution or tracing mechanisms.
Only independently accepted evidence in the exact named scope could support a
later separately authorized matrix update; this task supplies none.

Until those blockers are resolved through separate review, the result is a
deployment-preparation package ready for independent software review with
physical-host qualification unclaimed. Stage 9 remains `IN_PROGRESS`, Observer
3A `BLOCKED`, Stage 3B `NOT AUTHORIZED`, Production `NO-GO`,
`HOST_PREFLIGHT_VERIFIED=NOT_CLAIMED` and
`EXECUTION_AUTHORIZATION=NOT_AUTHORIZED`. Stop after preparing the draft PR;
do not merge, install, start an experiment or change production trust.
