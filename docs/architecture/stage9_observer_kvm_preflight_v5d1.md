# Stage 9 / Observer 3A — KVM preflight V5d1

V5d1 provides a concrete read-only controller for an operator's local
`qemu:///system` libvirt connection. It reads an existing domain by UUID,
compares observations with independently maintained requirements, measures
eligible immutable local images and returns a refusal when required evidence
is absent. It does not start, stop, redefine, attach devices to, or take over a
VM. The controller is non-production. A successful parser or mocked libvirt
test does not qualify a host.

The [closure matrix](stage9_closure_matrix_v5d1.json) is the maintained register
of closure prerequisites. The [validation report](stage9_observer_kvm_preflight_v5d1_validation.md)
records implementation and local checks separately from measurements that
require the operator host.

```text
Stage 9 = IN_PROGRESS
Observer 3A = BLOCKED
Stage 3B = NOT AUTHORIZED
Production = NO-GO
PYC_RECONSTRUCTION = UNVERIFIED
Exact CPython execution = NOT_TESTED
```

## Evidence and authorization boundaries

| Result | Meaning |
| --- | --- |
| `IMPLEMENTED` | The repository contains the working read-only mechanism. |
| `SYNTHETIC_TESTED` | Automated tests exercised fake libvirt or synthetic files. This does not satisfy a physical closure gate. |
| `HOST_PREFLIGHT_VERIFIED` | Reserved for independently reviewed measurements on the required operator host. This change does not claim that result. |
| `EXECUTION_NOT_AUTHORIZED` | No separate authorization exists for a concrete target experiment. |
| `EXECUTION_BLOCKED` | Required identity, isolation, immutable runtime or measurement evidence is missing. |
| `NOT_TESTED` | The specified physical observation or exact target execution was not performed. |

The V5c3 `PreparedVMCollectorV5c3`, `ControllerObservationV5c3` and
`RuntimeInventoryV5c3` contracts retain their original semantics and completeness
requirements. A preflight report is not a `ControllerObservationV5c3` that can
satisfy execution admission. The controller cannot implement a successful
`execute` or `stop` through this read-only scope. V5c3 continues to reject a
`CONTROLLER_OBSERVED` claim before collector callbacks with
`COLLECTOR_IDENTITY_UNVERIFIED`; synthetic adapters retain synthetic results.
V5d1 does not add an admission path for `LOCAL_EXECUTION_OBSERVATION`.

A domain name, collector ID, class name, callable signature, nonce, self-declared
origin or JSON checksum proves no authority. A matching file SHA-256 verifies
bytes against an expectation; it establishes neither independent approval of
that expectation nor the identity of code already loaded into a process.
The reporting process is not an independent witness of its own launch.

## Operator host and independent policy approval

Use a separate Ubuntu host operated under an independently reviewed local
security policy. Required components are local KVM/libvirt, the system libvirt
Python binding, QEMU image tooling, a trusted host Python environment with the
project's declared dependencies, and storage supporting a genuinely immutable
measurement view. The optional `observer-kvm-preflight` extra pins the hardened
host XML parser to `defusedxml==0.7.1`; the system `libvirt-python` binding,
libvirt daemon and QEMU tooling are separately reviewed host prerequisites.
During the separate unprivileged environment preparation, the dependency
selection is `python -m pip install '.[observer-kvm-preflight]'`, from the
independently verified source distribution using the operator's approved
package acquisition process. That is preparation guidance, not a preflight
command; no dependency installation is performed by the controller. Test/type
checking additionally uses the declared `types-defusedxml` test/quality pin.
Provisioning and configuration are separate operator tasks; the controller
performs no installation, elevation or service changes.

Grant only the existing read-only libvirt access appropriate to the operator.
An access denial is a refusal. Do not broaden socket permissions, use `sudo`
inside the procedure, or substitute a remote/alternate URI. Do not reuse a
production or existing Stage 9 domain. An independently prepared disposable
domain must already exist; identify it by its externally approved UUID.

The host's operator approval artifacts live outside the checkout under
`/etc/cryptohunter/observer-v5d1/`. The identity API reads fixed authority and
approved-policy locations. The process cannot supply a replacement key or
self-approved policy on the command line. Trust files and their ancestors must
be protected against replacement and symlinks. Root ownership and restrictive
permissions are checked as local prerequisites; they do not authenticate a
root-compromised host.

Approval is a deliberate external ceremony:

1. The operator independently obtains and verifies the release/commit, code
   closure, trusted launcher/interpreter, libvirt/QEMU tooling, Python package
   closure and relevant host configuration. The material proposed by the
   running collector is comparison input, not the source of accepted hashes.
2. A separate reviewer approves the versioned policy, domain requirements,
   image identities and environment constraints. Record reviewer identity,
   decision, contract revision, source provenance and exact content digests.
3. An independently held Ed25519 signing key signs the identity policy in its
   versioned domain. Install only public verification material and signed
   approval artifacts in the protected external directory. Keep the private
   key outside the repository and outside the measurement host's guest domain.
   V5d1 does not generate keys, install anchors, rotate trust or persist a
   rollback decision.
4. Launch through the separately reviewed host mechanism. Independently attest
   loaded code, interpreter, native dependencies, startup/import configuration
   and execution environment. Policy signature and on-disk hashes alone do
   not complete this launch proof. The V5d1 identity report keeps this missing
   requirement explicit and execution remains blocked.

Operator approval is distinct from production root-of-trust management. This
procedure does not change existing production authorities or durable
anti-rollback state. Local wall-clock timestamps and revision labels do not
supply authenticated freshness or rollback resistance.

## Identity artifact format

The protected files use strict versioned JSON. Unknown and duplicate keys are
rejected. Hexadecimal digests, public keys and signatures use lower case.
The source validator is
`deployment/stage9_observer_collector_identity_v5d1.py`.

| Artifact | Exact fields |
| --- | --- |
| `operator-anchors.json` | `profile` = `stage9-observer-controller-anchors-v5d1`, `authority_id`, `public_key_hex` (32-byte Ed25519 public key), `approved_revision` (positive integer), `approved_policy_sha256` |
| `approved-policy.json` | `policy`, `signature_hex` (64-byte Ed25519 signature) |
| `policy` object | `profile` = `stage9-observer-controller-policy-v5d1`, `authority_id`, `revision`, `artifacts`, `environment_sha256` |
| Each `artifacts` entry | Exactly `path`, `role`, `sha256`; canonical absolute path and exact full SHA-256 |

The anchor's `approved_revision` must equal the policy revision, and
`approved_policy_sha256` is SHA-256 of the exact
`canonical_json_bytes(policy)` representation. The Ed25519 message is the
canonical JSON encoding of:

```text
{"domain":"stage9-observer-controller-policy-signature-v5d1","policy":policy}
```

Use the repository's reviewed canonical serializer, not generic JSON spacing
or a digest-only signature convention. The approval authority pins the exact
policy digest as well as the public key and revision; merely signing another
policy with the same key does not select it as approved. The local revision
pin does not establish an independent durable anti-rollback service.

Required artifact roles are `controller-code`, `interpreter`, `libvirt-code`,
`libvirt-extension`, `qemu-tool`, `domain-requirements`. Include the actual
kernel-selected `/proc/self/exe` target, `/usr/bin/qemu-img`, loaded libvirt
Python/native files, all required and imported repository modules, and the
fixed requirements path. Additional approved code files can be listed.
Every artifact and ancestor is root-owned without group/other write access;
symlinks and multi-link regular files are refused. The policy must cover
actual installed paths, not paths from a different checkout.

The environment fingerprint covers implementation/version, architecture,
UID/GID/groups, `sys.path`, Python isolation/site/environment flags, and the
selected startup-related environment values. The diagnostic CLI reports this
fingerprint even if no policy exists. Obtain it using the exact approved launch
and independently review the underlying values before incorporating it into
a signed policy. Copying an unreviewed collector's fingerprint into a policy
is not independent acceptance. The regular protected-venv diagnostic launch
uses `-I -B`; site startup remains explicitly
`SYSTEM_SITE_BOOTSTRAP_UNVERIFIED`. No diagnostic launch authenticates itself.

## Concrete local Ubuntu preflight procedure

This is a read-only diagnostic procedure. Administrative provisioning of the
protected installation, external policy, existing disposable domain and
immutable analysis images must already have been independently completed.
It does not authorize the target interpreter experiment.

1. Independently inspect the exact reviewed installation under
   `/opt/cryptohunter/observer-v5d1`, including its protected venv, dependency
   files and startup configuration. Use actual approved deployment paths if
   the operator's installation differs; the policy must pin those exact paths.
   Do not run an unreviewed writable checkout as root. Ensure the operator has
   existing read-only access to local system libvirt.
2. Prepare the independent domain requirements and complete expected runtime
   inventory described above. Verify the UUID against the operator registry,
   every immutable image/chain identity against approved source material, and
   all code/tool digests through the external review process. Provision the
   signed approval artifacts at the fixed paths; the CLI has no override for
   a key, policy, manifest, URI, VM UUID or image-tool command.
3. Select an already prepared operator-owned receipt directory, with no
   symlink path components or competing writers. From that directory run:

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

   The fixed explicit import path is part of the independently reviewed
   launcher. `-I` ignores environment-based Python path settings; `-B` disables
   cache writes. This command executes the trusted host verifier only. It
   does not invoke the investigated Ubuntu interpreter or bytecode. The
   pre-existing site/dependency bootstrap remains unverified, as recorded in
   the identity report.
4. The normal report exit code is **2**, including when native configuration
   and image reads complete. It preserves `EXECUTION_BLOCKED`. `--help`
   exits 0 without collecting observations. No execution or policy-selection
   flags exist. An import/process failure or non-JSON output is an incomplete
   observation, never a pass. Retain exit code and receipt SHA-256 alongside
   the original bytes and the exact approved installation/policy identities.
5. Review every reason code and missing-evidence field. Missing policy,
   libvirt access, immutable image proof, complete native observation or
   stable before/after identity must remain a refusal. Preserve both positive
   configuration observations and their missing enforcement evidence.
   Never clear a refusal by editing the report or wrapping it as a synthetic
   adapter. No V5d1 report authorizes V5c3 execution.
6. Update the closure matrix using independently reviewed receipts. Retain
   artifact SHA-256, environment, verifier identity, approved contract and
   expectation provenance for each gate. Physical host qualification requires
   its own review and missing-enforcement resolution; the CLI deliberately
   keeps `host_preflight=NOT_TESTED` and cannot self-award
   `HOST_PREFLIGHT_VERIFIED`.

Do not use `virsh start`, QMP/guest-agent execution, image repair/conversion,
mounting, `sudo`, or configuration changes to complete this procedure.
A failed requirement is returned to the operator for separate review.

## What the concrete preflight observes

The native boundary opens the fixed local URI read-only and selects the domain
by UUID. Observations retain actual domain identity and state, persistent and
applicable live configuration, and the XML bytes/digests needed to identify
what was inspected. Repeated observations surround the measurement interval;
a changed UUID, state, configuration, disk identity or chain invalidates the
preflight. Two matching samples do not prove the absence of an intermediate
change; a protected immutable view and lifecycle fence are separate evidence.

Domain XML parsing uses the pinned hardened parser and rejects DTDs,
entities and external references. Configuration checks cover interfaces and
other networking routes,
passthrough/host devices, local disks and backing declarations, shared
filesystems, channels and host sockets, memory/CPU settings and quotas,
process limits, removable devices, consoles, and mechanisms that expose host
resources. Unrecognized features cannot silently become accepted isolation.
A source path taken from domain XML is not automatically approved for opening.
It must agree with independently supplied, strictly validated requirements.

The domain XML expresses requested configuration. It does not independently
prove effective QEMU process state, host policy enforcement or lack of host
access. In particular, a stopped VM cannot supply proof of limits and security
state for a future QEMU process. The report must retain missing evidence for:

- Actual QEMU process identity and argv, loaded executable/modules, open file
  descriptors, socket connections and effective device graph.
- Active host network namespace, routes, sockets, filtering and policy controls
  preventing guest network access, including non-interface transports.
- Effective UID/GID, namespaces, cgroups and limits, LSM policy and labels,
  seccomp configuration, capabilities and device access control.
- Memory/CPU enforcement, bounded whole-experiment wall time, and a process
  limit covering every relevant process. A host QEMU `pids.max` alone does not
  limit guest processes inside the VM.
- Absence of secret/production access through host mounts, sockets, inherited
  descriptors, credentials, firmware/NVRAM, removable media or side channels
  allowed by the selected threat model.
- An independently approved measurement channel and a separately authorized
  whole-VM termination/cleanup capability. Listing a channel in XML does not
  attest the endpoint or make guest messages trustworthy.

Acquiring privileged host evidence requires an explicitly approved separate
observer. The read-only controller does not elevate or alter the host to
obtain it. Missing access or evidence is reported as a blocking condition.

## Supported domain and requirements profile

The current profile intentionally accepts only a persistent, shut-off KVM
`x86_64` HVM domain with no managed-save image or libvirt snapshots. Memory,
current memory, vCPU count, CPU period and CPU quota must match the independent
requirements. Defaults are 512 MiB, one vCPU and period/quota 100000/100000.
Supported local disks use raw or restricted QCOW2. The P2 remediation narrows
the accepted XML to the following explicit semantic profile, checked for both
current and persistent configuration. A name on an XML allowlist does not
authorize arbitrary values or omitted safety settings.

| XML setting | Accepted semantics |
| --- | --- |
| `domain` | `type=kvm`; optional XML `id` must be `-1`, consistent with the stopped-domain observation. |
| `name`, `uuid`, `title`, `description` | Name is a bounded simple identifier; UUID matches independent requirements; optional title/description are inert text without control characters. These fields cannot contain nested configuration. |
| `memory`, `currentMemory` | Both match independent memory bytes; units are only `bytes`, `KiB`, `MiB`, `GiB` (`KiB` if omitted). |
| `vcpu`, `cputune/period`, `cputune/quota` | Count/period/quota match independent requirements; placement is static (also its accepted omitted default), and optional `current` equals the count. |
| `os/type`, `os/boot` | HVM, `arch=x86_64`, exactly `machine=pc-q35-8.2`; exactly one explicit boot device `hd`. No machine alias, firmware or alternate boot source is inferred safe. |
| `features/acpi`, `features/apic` | Optional empty enable markers only; no attributes, nested settings or text. Their absence grants no enforcement evidence. |
| `cpu`, `cpu/topology` | Exactly `mode=host-model`, `check=full`; `match` and `migratable` are unsupported in this mode and refused. One explicit topology has sockets/dies/clusters/threads `1` and cores equal to the independent vCPU count. Actual host CPU enforcement remains unverified. |
| `clock` | Exactly one explicit empty element with `offset=utc`. |
| `on_poweroff`, `on_reboot`, `on_crash` | All three must occur exactly once, contain exactly `destroy`, and have no attributes or child elements. Omission, `restart`, `preserve`, other known policies and unknown text are refused. There is no automatic restart in this disposable-domain profile. |
| `devices/emulator` | Exactly `/usr/bin/qemu-system-x86_64`, subject to the separate controller/tool identity boundary. |
| `disk`, `driver`, `source`, `target`, `readonly` | File-backed disk only, explicit `snapshot=no` and one empty `readonly`; `driver` has `name=qemu`, approved `raw`/`qcow2` type, and explicit `cache=none`, `io=native`, `discard=ignore`, `detect_zeroes=off`. Source matches the canonical independently expected path; target is virtio `vda`, `vdb`, etc. in independent disk order. No other driver option or omitted driver policy is accepted. |
| `backingStore`, `format`, `source` | Declared file backing paths and raw/QCOW2 formats match the independently expected complete ordered chain. Optional index must equal chain depth; an empty terminal store has no attributes or children. Image metadata is checked separately. |
| `controller`, `memballoon` | One explicit USB controller with `model=none` (optional index only `0`), and one explicit balloon with `model=none`. Other controller profiles are unverified and refused. |
| Device `alias` | Optional unique bounded simple identifier; no child configuration. |
| `address`, `seclabel`/labels, `resource`/partition | Refused even though their names are recognized: independently approved address, confinement or host partition semantics are unavailable. No label or resource value can self-establish host enforcement. |

Other attributes, values, mixed text, nested extensions and duplicate singleton
elements are refused. Only disk siblings may repeat. The current and persistent
trees must agree on every reviewed setting, ignoring indentation and attribute
ordering only. A semantic difference adds `DOMAIN_LIVE_PERSISTENT_MISMATCH`;
it never yields `config_matches=True`.

Lifecycle omissions use `DOMAIN_LIFECYCLE_POLICY_MISSING`; invalid or duplicate
lifecycle uses `DOMAIN_LIFECYCLE_POLICY_REJECTED` (duplicates additionally use
`DUPLICATE_DOMAIN_CONFIGURATION`). Driver policies use `DISK_DRIVER_REJECTED`.
Other profile refusals include `DOMAIN_VALUE_UNREVIEWED`,
`BOOT_POLICY_UNREVIEWED`, `CPU_PROFILE_UNREVIEWED`, `CPU_TOPOLOGY_UNREVIEWED`,
`CLOCK_POLICY_UNREVIEWED`, `DISK_TARGET_UNREVIEWED`,
`DISK_LIFECYCLE_UNVERIFIED`, `CONTROLLER_PROFILE_UNVERIFIED`,
`DEVICE_ALIAS_UNREVIEWED`, `DEVICE_ADDRESS_PROFILE_UNVERIFIED`,
`HOST_CONFINEMENT_PROFILE_UNVERIFIED` and `RESOURCE_PARTITION_UNVERIFIED`.
Existing device, backing, resource-limit and unknown-extension diagnostics
remain applicable. A refusal code is diagnostic, not an approved alternative.

Network interfaces, passthrough, host devices, shared directories, channels,
consoles/serial, graphics, RNG, TPM, shared memory, vsock, QEMU extensions,
NVRAM and unrecognized nodes are refused. An ordinary distribution-generated
definition may therefore fail this strict profile. The preflight must not edit
that definition to make it pass. Prepare a separate conforming domain through
the operator's independently reviewed provisioning procedure.

`domain-requirements.json` has exactly these top-level keys:

| Key | Required value |
| --- | --- |
| `profile` | `stage9-observer-kvm-requirements-v5d1` |
| `domain` | Exactly `domain_uuid`, `memory_bytes`, `vcpu_count`, `cpu_period_us`, `cpu_quota_us`; UUID is lower-case canonical text and numeric fields are positive integers, never booleans |
| `disk_chains` | One to eight arrays, each containing one to sixteen `{path, sha256, format}` records ordered top image to base; formats `raw` or `qcow2`, globally unique canonical absolute paths and lower-case full SHA-256 values |
| `runtime_inventory` | The complete `RuntimeInventoryV5c3` structure described below; these are independently expected pins, not observations of captured runtime bytes |

`runtime_inventory` has exactly `files`, `needed_resolutions`,
`loader_configuration_paths`, `import_paths` and
`provider_configuration_paths`. Each of its five to 512 `files` has exactly
`path`, `sha256`, `build_id`, `role`, `aliases`; roles are the existing
`interpreter`, `library`, `loader`, `stdlib`, `tool`, `configuration` values.
Only the existing V5c3 optional configuration-absence and Build-ID semantics
apply. Each resolution is `[consumer_path, needed_name, resolved_path]`.
Configuration paths refer to declared files. At least one import path is
required. The existing bounded V5c3 validator checks structure; it does not
approve provenance, completeness or the actual image contents.

Unknown or duplicate JSON keys, malformed/nonfinite values, out-of-range
sizes, duplicate paths and incomplete runtime structures are refused. The
requirements file is limited to 256 KiB. Its exact raw-byte SHA-256 is an
artifact in the independently signed controller policy, so rewriting its
format or content requires a new deliberate external approval.

## Immutable image measurement

Measurements bind a local artifact's validated absolute path, file identity,
size and SHA-256, format and complete supported backing-chain identities to
the domain and independently approved expectation. All path components must
be safe against symlink traversal and replacement. No manifest-supplied command
or arbitrary tool argument is executed. Image tools are fixed host components
and must themselves be approved.

A stopped guest, a read-only file descriptor and identical before/after stat
fields are insufficient to guarantee an immutable image view. The implemented
Linux profile requires an independently prepared analysis copy
of every chain member with the kernel `FS_IMMUTABLE_FL` flag already set. Every
file and ancestor directory must be root-owned and not writable by group or
other users; files must be regular, single-link objects with no symlink path
components. The operator must separately establish the provenance, lifecycle
fence and absence of privileged writers. A read-only mount or matching stat
fields alone does not satisfy this implemented immutable-file check. Missing
immutable flags, unsupported storage or uncertain state yield
`IMAGE_MEASUREMENT_BLOCKED`. The controller neither mounts a filesystem nor creates snapshots, edits image headers, repairs an
image or follows an unverified backing path.

QCOW2 backing files are dependencies. V5d1 supports only canonical absolute
backing paths, ordered from the top image to the base in the independent
manifest. Relative references are refused. Every chain member must be opened
and validated through held descriptors before a tool can inspect it;
unexpected, cyclic, missing, changed or escaping references are refused.
The supported formats are raw and restricted QCOW2 v2/v3. Internal snapshots,
encryption, incompatible feature bits, external data files and nonstandard
v3 headers are refused. A fixed protected `/usr/bin/qemu-img` is used only for
bounded metadata inspection of a held file descriptor, without force sharing
or arbitrary manifest arguments. Each chain member is hashed through its
held descriptor while the immutable-file prerequisite holds; before/after
object metadata, immutable flags and a reopened protected pathname detect
substitution. The receipt binds all chain members, not merely the top image.
The actual QEMU tool's independently signed expected SHA-256 is propagated to
the image boundary. Its held executable descriptor is hashed and its identity
checked before execution; a caller cannot substitute an unapproved tool.
Where the implemented profile rejects a feature, prepare a separately
reviewed supported image outside this procedure; do not flatten, convert or
repair it during preflight.

Image evidence is published atomically. Per-image records remain private until
every member of the declared chain has passed its hash, backing metadata,
immutable flag, held-descriptor and protected-path rechecks. Every refused
chain returns `IMAGE_MEASUREMENT_BLOCKED` with an empty `artifacts` tuple,
including a failure after an earlier member was measured successfully.
`reason_codes` retain the diagnostic cause; earlier positive records are not
retained as partial diagnostics.

The preflight similarly stages all disk chains until every chain and the final
domain, requirements and controller-identity bindings have passed, and the
read-only libvirt connection has closed successfully. A failed chain, late
drift, incomplete native observation, interruption or cleanup failure discards
the complete `images` collection. Observation hashes and refusal codes remain
diagnostics; they do not constitute complete image evidence. A fully measured
local image collection still cannot authorize execution or qualify the host.
`IMAGE_MEASUREMENT_INTERRUPTED`, `IMAGE_DESCRIPTOR_CLOSE_FAILED` and
`NATIVE_PREFLIGHT_INTERRUPTED` distinguish interruption and descriptor cleanup
failures. Existing digest, backing-chain, immutable-view, path-change, timeout
and QEMU-tool refusal codes are preserved. No partial-success record type is
introduced. Descriptors are closed even after a previous close fails, and
metadata-helper errors and interruptions still kill/reap the helper and close
its output and executable descriptors before image evidence can be published.

The domain XML's `backingStore` can override image metadata on drivers
supporting that input; an explicit empty `backingStore` means no backing.
Declared chains must therefore agree with the independently expected ordered
paths and formats. An explicit empty declaration cannot silently mask an
expected backing dependency. The real chain is separately checked from image
metadata. See the libvirt [backing-store contract](https://libvirt.org/formatdomain.html#hard-drives-floppy-disks-cdroms).

A whole-image digest does not establish a complete guest runtime inventory.
To populate `RuntimeInventoryV5c3`, an approved external extractor must read
exact interpreter, loader, ELF library, native module, standard-library and
configuration bytes from the same immutable view, record path aliases and
package/snapshot provenance, and independently pin the complete import tree.
No host fallback library or guest-provided file list may complete that proof.

## Complete runtime observation design

V5d1 does not implement a trusted full execution trace. This is a distinct
qualification dependency and remains blocked. A future accepted mechanism
must witness actual accesses outside the investigated interpreter and bind
them to the same VM, immutable image, approved worker and experiment.

A host trace of QEMU syscalls observes host block-image reads, not guest
pathname opens or guest virtual-memory mappings. It cannot reconstruct the
required guest interpreter dependency trace on its own. Plain guest `strace`,
`/proc` dumps or a worker's list of imported modules are guest reports unless
a separately approved measurement substrate establishes their integrity and
completeness under the stated threat model.

Two possible designs require separate implementation and acceptance:

1. **Approved guest measurement substrate.** Measure and approve the guest
   kernel, initramfs, configuration, observer binary and dependencies, policy,
   boot chain and channel before starting the target. The observer must have
   an isolation boundary from the target, enforce that boundary using the
   approved kernel/LSM policy, and publish a complete authenticated stream to
   an independent host receiver. IMA appraisal/measurement can bind executed
   and read objects only for a carefully reviewed policy and hook coverage;
   an IMA log by itself is not a complete file-access or mapping trace.
   Kernel audit/eBPF/fanotify/LSM instrumentation must cover the required
   hooks, provide event-loss detection, and be inaccessible to the target.
   Guest kernel trust and trustworthy boot measurement are explicit
   prerequisites, not inferred from a channel name or signed guest report.
2. **Approved hypervisor/VMI observer.** Pin and review a hypervisor extension
   or out-of-guest introspection component, its kernel-specific semantic
   profile and event coverage. The observer must resolve guest task identity,
   descriptors, mounts, file objects, page mappings and relevant namespace
   changes without relying on target assertions. Guest-memory races, kernel
   version drift, unsupported filesystems and lost observations must fail
   closed. Access to a QEMU monitor or guest memory is not an acceptance proof.

The accepted mechanism must account for all of the following:

| Observed dependency | Required outside-process evidence |
| --- | --- |
| Interpreter and descendants | PID lifetime/start identity, exec/fork/clone scope, actual executable bytes, exact argv and sanitized environment, exit/termination and independent elapsed time |
| Runtime file reads | `open*`/descriptor lifecycle, reads including `pread*`, mappings and page-backed accesses, inherited/passed descriptors, mount/path aliases, device/inode identity, exact content from a frozen view |
| ELF and native modules | Actual successful executable mappings and unloads, ELF identities/Build IDs and hashes, `dlopen` dependencies, deleted/memfd objects and anonymous executable mappings; refuse unidentifiable code |
| Loader/provider inputs | Actual loader, preload/cache/config files, lookup failures and successful resolutions, OpenSSL/provider settings and inputs; static `DT_NEEDED` is comparison material only |
| Python import and data inputs | Builtin/frozen code bound to the interpreter, all selected `.py`, existing `.pyc`, zip/import hooks, encodings, native modules and standard-library data; path lists alone are incomplete |
| Source/request/output | Independently frozen source and request, first output creation through close, actual output bytes from the same image/view, writer identity, no replacement/alias ambiguity, and expected output from an independent plan |
| Trace completeness | Observer start before target launch, end after final output capture, sequence/count integrity, bounded buffers, explicit dropped-event/overflow handling, coverage of every process and mapping, and refusal on gaps |

Reading a digest after execution cannot establish which bytes were read if an
object was mutable during the run. An unlink, rename, changed backing file,
memfd, descriptor transfer or missed short-lived mapping must retain identity
or block the result. Measurements must correlate events to immutable object
identities and byte content, rather than re-open a pathname later and assume
it denotes the same object.

Output retrieval also needs an accepted external mechanism. It must read the
actual `.pyc` from the completed immutable output view, bind it to the observed
writer and experiment, measure opaque bytes independently, and prove there
was no competing writer. Guest stdout containing a digest is insufficient.
No `.pyc` body is imported, deserialized with `marshal.loads`, or executed.

## Preconditions for the first authorized exact CPython experiment

The following are still required in addition to repository tests:

1. An independently approved controller identity policy and trust custody,
   plus accepted proof of the actual controller launch, loaded code closure
   and host execution environment.
2. A separately prepared disposable local VM, stable UUID/configuration/image
   binding and complete independent evidence that every isolation requirement
   is effectively enforced for the full experiment.
3. Immutable complete disk and runtime views, a fully accepted
   `RuntimeInventoryV5c3`, exact Ubuntu interpreter/loader/library/compiler
   inputs and package/snapshot/source/expected-PYC bindings.
4. An accepted external runtime tracing and output collection mechanism with
   complete event coverage and fail-closed loss detection, and a reviewed
   whole-VM termination/cleanup controller outside this read-only component.
5. Separate explicit authorization for one concrete experiment, its exact
   inputs, limits, approved environment and stop conditions. V5d1 supplies no
   such authorization and never starts the target interpreter.
6. A reviewed later contract/admission change before any real observation
   status can be emitted. Independently bound evidence, acceptance authority,
   authenticated freshness and durable anti-rollback remain additional gates;
   local preflight success does not imply their acceptance.
