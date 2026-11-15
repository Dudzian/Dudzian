# Stage 9 / Observer 3A — isolated reconstruction V5c3

V5c3 supplies a fixed-path compilation worker, bounded static runtime inspection,
and an outside-guest collection interface that measures actual returned PYC
bytes. It does not provision, start or qualify a VM. The current task's exact
execution is **EXECUTION_BLOCKED** and the exact-interpreter test is
**NOT_TESTED**. A functioning synthetic mechanism is not an observation of the
target interpreter. The [validation report](stage9_observer_pyc_execution_v5c3_validation.md)
records the checks and remaining evidence.

## Boundaries and evidence levels

The host verifier reads DEB/TAR/source/ELF/PYC as bounded data. It never imports
the investigated module, runs its bytecode, calls `marshal.loads`, invokes the
target interpreter or uses `ldd`. V5b supplies frozen source bytes from the
selected TAR record. V5c2's independent SipHash-1-3 computes the PEP 552 source
hash before execution; the expected complete PYC SHA-256 is a separate input.
V5c2 declarations are not used as measured output.

The caller separately pins the entire reconstruction request set, including
source/snapshot, target path, `co_filename`, optimization and expected output,
as well as the V3 and runtime inventories. Changing a declaration without
changing its independently maintained pin is refused. V3 itself contains no
APT snapshot authority; these checks do not promote V4c acceptance.

Collector identity has no separately approved verification mechanism in this
version. Every `CONTROLLER_OBSERVED` claim is therefore refused with
`EXECUTION_BLOCKED` / `COLLECTOR_IDENTITY_UNVERIFIED` before any `inspect`,
`execute` or `stop` callback. Only a plain-string `SYNTHETIC` label permits
mechanism tests, whose successful result is always `SYNTHETIC_MECHANISM_MATCH`.
The admitted label is captured once; changing it during a synthetic run cannot
promote the result or relabel its records as controller observations.

A Python protocol implementation, collector name, process declaration, nonce,
JSON document or SHA-256 report checksum authenticates no collector and proves
no actual VM execution or hypervisor enforcement. This version cannot emit or
export `LOCAL_EXECUTION_OBSERVATION`; `exact_target_test` stays `NOT_TESTED`.
A prepared VM or a new adapter alone cannot unlock that status. A separately
approved collector-identification mechanism must first be implemented and
reviewed; no caller-provided name, label, flag or report can enable it.

| Level | Meaning | V5c3 treatment |
| --- | --- | --- |
| A, local execution observation | An outside controller observes a qualified disposable VM and measures input, process and output | BLOCKED: no separately approved collector-identification mechanism; this version never emits this status |
| Guest report | Data printed by the worker inside the VM | Always `GUEST_REPORTED`; cannot attest the VM or substitute for independently collected bytes |
| B, independently bound execution evidence | Run provenance, nonce, measurements and integrity linked to an independently maintained registry | `UNVERIFIED`; no registry or signing authority is introduced |
| C, accepted reconstruction evidence | Evidence formally accepted under a separately approved policy and authority | `UNVERIFIED`; no automatic promotion from A or B |

Every result retains Stage 9 `IN_PROGRESS`, Observer 3A `BLOCKED`, Stage 3B
`NOT AUTHORIZED`, production `NO-GO`, external authority and authenticated time
`UNVERIFIED`, durable anti-rollback and runtime qualification `BLOCKED`.

## Required outside-controller procedure

This is the future procedure after a separately approved collector-identification
mechanism is implemented. It cannot bypass the current synthetic-only API.
All source, runtime, isolation, process and output checks remain required.

1. Select an already prepared disposable offline KVM/libvirt VM. Inspect its
   actual hypervisor configuration and effective runtime state. Do not modify
   existing Stage 9 VMs, install DEBs on the host, run maintainer scripts or
   create privileged infrastructure as part of this procedure.
2. Before launching the interpreter, independently inspect the VM's disk or
   immutable image and capture exact runtime bytes. Check the interpreter
   SHA-256, Build ID, package/version/architecture and APT snapshot against the
   independent local inventory. Measure the loader, its configuration/cache,
   every selected library and compiler import input. Resolve aliases/symlinks
   to their inventoried final paths; never use a host fallback library. The
   controller must verify and freeze the complete runtime tree, including all
   import candidates and existing bytecode, rather than capture a partial list.
3. Require zero network interfaces/access, no forwarded secrets, no production
   credentials, no host Docker/libvirt sockets, no user-data shared directories,
   a separate disposable disk/snapshot, read-only source/tool mounts and a
   distinct empty output directory. Verify the VM can be stopped as a whole.
4. Enforce and observe CPU, RAM, wall-clock time and process-count limits from
   outside the guest. A requested limit or in-guest declaration is insufficient.
   Keep enforcement active for the complete run and output capture. Stop the
   whole VM on timeout, interruption, failed measurement or collection error.
5. After collector identification is separately approved and implemented,
   supply the pinned V5c2 policy, V5b archives and inventory, independently
   expected source hash and PYC digest, exact repository commit and reviewed
   collector. The orchestrator validates source provenance, generates a fresh
   run ID/nonce and supplies the canonical request and exact source bytes.
6. Put the versioned worker at `/stage9/tool/compile_worker.py`, the source at
   `/stage9/input/source.py` and the request at `/stage9/input/request.json`.
   Pin the worker's actual SHA-256 and entire tool image independently before
   execution. Launch this exact argument vector inside the VM:

   ```text
   /usr/bin/python3.12 -I -S -B /stage9/tool/compile_worker.py
   ```

   Supply exactly `LANG=C.UTF-8`, `LC_ALL=C.UTF-8`, `TZ=UTC`, with no inherited
   `LD_*`, `PYTHON*`, OpenSSL/provider or other variables. `-I -S -B` is an
   import/startup restriction, not a substitute for pinning the runtime and
   existing bytecode. It does not prevent reading an existing `.pyc`.
7. The worker calls the inventoried `py_compile.compile` with explicit input,
   output `/stage9/output/reconstructed.pyc`, `dfile`, `optimize` and
   `PycInvalidationMode.CHECKED_HASH`. Compilation does not evaluate module code.
   Export the PYC as opaque data. Independently capture process arguments,
   environment, exit status, bounded stdout/stderr and output bytes. Collect a
   complete runtime file-access trace and mapped ELF paths outside the guest;
   all accessed runtime paths and digests must occur in the independent
   inventory. Missing traces, unknown imports or unpinned mappings refuse the
   result. Frozen/builtin code belongs to the measured interpreter; reads of
   investigated source/request/output are separately measured data inputs and
   must not be misclassified as runtime imports. Neither this trace nor static
   `DT_NEEDED` edges alone proves effective symbol/provider resolution; runtime
   closure qualification remains explicitly `UNVERIFIED`.
8. The host measures full output SHA-256, size, magic `cb0d0d0a`, flags `3` and
   eight source-hash bytes. It hashes the remaining body without deserializing
   it. Compare against the independently declared expected full PYC digest.
   A mismatch is `RECONSTRUCTION_MISMATCH`. Never derive the expectation from
   this run's output or the worker's claimed digest.
9. Stop the disposable VM, retain the complete failure/observation record and
   publish positive per-file results only if the whole declared set succeeded.
   A failed file, missing record or failed cleanup removes all positive files.
   Timestamps use the local unauthenticated clock; run IDs/nonces do not provide
   authenticated time, replay protection or durable anti-rollback by themselves.

`serialize_reconstruction_result_v5c3` exports the versioned result as canonical
JSON. Streams retain their exact bytes as hexadecimal, elapsed durations are
integral nanoseconds and the payload has a local SHA-256 integrity checksum.
This checksum does not authenticate the collector, record origin or timestamp.
There is deliberately no JSON-to-trusted-collector import operation.

## Runtime dependencies and inventory gaps

The exact interpreter profile is Ubuntu Noble amd64 `python3.12-minimal`
`3.12.3-1ubuntu0.17`, SHA-256
`e50d468e8b0adfb05733f5b87b3cff34829c4a8c1aea50c865aa8bdfe4bb150f`,
Build ID `337d65cf00021797985cc9f77c0cc334a9fbeb38`.
Its historical DEB SHA-256 is
`d452689b9660845345a4c3e05e4ad82c082d5474e04031b7aa47f1a6d5610a6e`.
These retained identities are comparison material, not new acceptance of an
archive authority or proof of a runtime observed in this task.

The retained [exact ELF graph](evidence/stage9_observer_3a_cpython_elf_graph_v1.json)
lists the interpreter's ordered `DT_NEEDED` as `libm.so.6`, `libz.so.1`,
`libexpat.so.1`, `libc.so.6`, with `PT_INTERP`
`/lib64/ld-linux-x86-64.so.2`. Its selected startup closure is:

| Final package path under `/usr/lib/x86_64-linux-gnu/` | SHA-256 | Build ID |
| --- | --- | --- |
| `ld-linux-x86-64.so.2` | `c20a2dc8917c755f02b94049356320fe1f62ac7d9f8994731f807d9df39302da` | `0d77dbd4379bf0b5af3e05be7fb6af1e4cea7c96` |
| `libc.so.6` | `3a15d66867d83762c7f2f1e37359cb8f6c5743edb369c65285cb0b1c4f7498bf` | `a4a7992a8e66555c8141ab2a08a8465ff6e0ea65` |
| `libm.so.6` | `fce00b6f25f459cf4ae0b7fae4257909a1a8a86f9149e7c029a5d617baf1ccd0` | `d53fcbff9f855463606b25cfbe2a2e10856ea3a7` |
| `libz.so.1.3` | `f61677c7bdb7c2aa2a243ea781bfd16afcb361cede7b4ef76127a9d8eb9a5242` | `2f47d490170763b8b8e0064675d234bfede6c1db` |
| `libexpat.so.1.9.1` | `286682ecbc5e59a638963b1a4e6351e65eb32fcf4bdcb9cb7569b6a61fe06a8d` | `8ae4253012c773ce4f3ceb73efbd6dbd47fd9510` |

This table is the static startup list, **not a complete compilation runtime
closure**. All library packages and versions, DEB/TAR records and their common
accepted snapshot must be bound in the supplied local inventory. The historical
archive report's InRelease digest is
`89ad2f985a8959b207f598f833feb5f852d145be5bd4efcb97c4934ecb45723e`;
it must not be substituted for a differently defined V3 snapshot digest.

The worker's direct standard-library imports are `hashlib`, `importlib.util`,
`json`, `os`, `pathlib`, `py_compile`, `re`, `stat`, `sys` and `typing` (plus
the future-annotations declaration). Their transitive imports, startup/frozen
code, encodings, `py_compile`/`importlib` compiler implementation, native extension
selection and existing bytecode must all come from the pinned tool image.
`hashlib` and `json` can select `_hashlib` and `_json`; selecting `_hashlib`
also adds `libcrypto.so.3` and relevant provider/configuration inputs. Do not
infer the effective selection or closure from this import list. Exact target
import tracing has **NOT_TESTED** status. No complete independently accepted
compiler import/provider inventory was supplied in this task. Missing or
uninventoried inputs must block execution; static `DT_NEEDED` inspection does
not establish runtime closure.

## What is needed to unblock exact execution

- A separately approved and implemented mechanism that verifies collector
  identity independently of its labels, name, process declarations, nonces and
  report checksums. The current API has no real-observation admission path.
- An already prepared VM meeting every isolation requirement and an operator
  authorized to use it, with independently observed enforcement evidence.
- A reviewed outside-guest collector implementation bound to that VM and image,
  including bounded process capture, immutable inputs, output extraction and
  reliable whole-VM stop/cleanup.
- Exact interpreter, loader, library and compiler-import bytes bound to an
  accepted local package/snapshot inventory, including loader/provider configs.
- The independently pinned reconstruction policy, source record and expected
  complete PYC SHA-256, plus the approved worker and tool-image digests.
- Separately, a registry/integrity policy for level B and an approved authority
  and acceptance policy for level C. Neither is supplied by V5c3.
