# Stage 9 / 3A — CPython evidence, security findings and remaining measurements V1

Review date: **2026-10-29 UTC**. Repository source baseline:
`cad53867231a1a6e49ea770c07043b417af84449` (`main`). This additive review
preserves the historical [contract audit](stage9_observer_3a_contract_audit.md),
[Dell ELF report](stage9_observer_3a_dell_local_elf_report_v1.md) and
[Ubuntu provenance report](stage9_observer_3a_ubuntu_provenance_report_v1.md).
It does not approve a build, change a security policy or qualify execution.

## Evidence boundary

**`BUILD_SPECIFIC_UNVERIFIED` / `OPERATOR_REPORTED_AND_CHAT_REVIEWED`:** the
operator reports a static Dell measurement on **2026-10-29**, archive
`stage9-elf-triage-rNlpy9.tar.gz`, SHA-256
`0a26027f9bf94149fe2b36d3fbba1fdb6e5393e2016c9bd9956463b154005b0f`.
The original archive was not present among the session's supplied attachments,
shared files or library files when checked. Codex read the uploaded request
and compared its claims with repository evidence; it did not unpack this
archive or independently verify its internal manifest in this environment.
An independently inspected downloaded package is a separate evidence item,
not a replay of this Dell archive.

**Review correction, 2026-10-29 — `OPERATOR_REPORTED_AND_CHAT_REVIEWED` /
`BUILD_SPECIFIC_UNVERIFIED`:** the original request's **13/13** statement was
imprecise. The independent archive review in the operator conversation reports
**15 regular files total: 14 checked files and the manifest file**; the manifest
has **14 entries, 14/14 verified, 0 mismatches**. Its reported SHA-256 is
`f52859c04f703bfe6ba36190229e7b18ac1543df5c443d589a549835177c3fdb`;
the local Dell readelf version is **2.42**. The archive SHA-256 above is
unchanged. These are supplied correction results; Codex has not obtained the
archive or replayed them. The exact historical request file is unchanged.

| Operator-reported object | SHA-256 | Reported Build ID | Classification of the Dell account |
| --- | --- | --- | --- |
| `python3.12` | `e50d468e8b0adfb05733f5b87b3cff34829c4a8c1aea50c865aa8bdfe4bb150f` | `337d65cf00021797985cc9f77c0cc334a9fbeb38` | `BUILD_SPECIFIC_UNVERIFIED`; `OPERATOR_REPORTED_AND_CHAT_REVIEWED` |
| `_hashlib` | `8dc6cda6dd9495b95a2bfb5b26e8cf06b72f36dc0213d958b9277a958287f0fe` | `33c38727eb4140c55f853fd9e534856569024762` | `BUILD_SPECIFIC_UNVERIFIED`; `OPERATOR_REPORTED_AND_CHAT_REVIEWED` |
| `_json` | `ca8e35c58f7bc8e27258c4af43651c941413bc6318903621253485b0395758a3` | `f24fcec743f7d42ed6c90ccc0ef455c4d4ab99a9` | `BUILD_SPECIFIC_UNVERIFIED`; `OPERATOR_REPORTED_AND_CHAT_REVIEWED` |

**`VERIFIED_FROM_STATIC_ELF`:** independent static inspection of the extracted
Ubuntu `3.12.3-1ubuntu0.17` package members reproduces all three SHA-256s and
Build IDs above, their stated `DT_NEEDED` lists, interpreter `ET_EXEC` and
`PT_INTERP`, GNU_RELRO, nonexecutable stack and absence of interpreter
RPATH/RUNPATH/BIND_NOW. The retained
[static table/ELF JSON](evidence/stage9_observer_3a_cpython_static_tables_v1.json)
records the inspected paths and results. These independent package-byte facts
corroborate the supplied Dell metadata; the physical Dell account and archive
remain `OPERATOR_REPORTED_AND_CHAT_REVIEWED`.

**`VERIFIED_FROM_EXACT_BUILD`:** the separate
[archive verification](evidence/stage9_observer_3a_cpython_archive_sources_v1.md)
checks the current **2026-10-29 noble-updates** signature and index-to-package
hash/size chain for the four exact binary archives and three source/control
files. This is a new cloud acquisition, not replay or acceptance of the
historical operator snapshot. The key fingerprint is corroborated by official
sources; formal project acceptance of the bootstrap/root and freshness policy
remains pending. Build IDs and matching digests do not prove reproducible
source-to-binary reconstruction or authenticate the physical host.

**`VERIFIED_FROM_STATIC_ELF`:** dereferencing the actual `PyImport_Inittab`
pointer yields **60 named entries: 58 nonnull initializer pointers and the
intentional `builtins`/`sys` null-initializer entries**. Frozen table pointers
yield **3 bootstrap, 15 stdlib and 11 test entries**, with embedded payload
hashes and sentinels retained in the JSON. The exact package's generated
`config.c` and `cpython/import.h` agree with this table layout and builtin
names. Registration evidence therefore comes from table records, not merely
from the reported 56 `PyInit_*` symbols or three frozen-table symbol names.
An independent source cross-check matched names and order **60/60** against the
exact developer package's generated `config.c`, and **3/3, 15/15, 11/11** against
the exact source's `Python/frozen.c`. The table decoder's ELF64 little-endian
PT_LOAD mapping, 16-byte `_inittab` and 32-byte `_frozen` layouts agree with
package headers; it follows exported table pointers and checks finite null
sentinels. This is static data decoding, not execution of frozen payloads.
Availability remains separate from a particular import being selected or run.

## Security interpretation and graph boundary

| Subject / label | Consequence and required evidence |
| --- | --- |
| Main executable / `VERIFIED_FROM_STATIC_ELF` | The exact extracted executable is `ET_EXEC`, consistent with a fixed-address, non-PIE executable in this x86_64 Ubuntu profile. Do not claim PIE hardening from distribution defaults. Main executable address randomization is reduced; this does not establish that shared libraries, stack or heap lack ASLR. The historical linker invocation/reproducibility and selected-profile hardening disposition remain separate. |
| Relocations and stack / `VERIFIED_FROM_STATIC_ELF` | Independently inspected GNU_RELRO plus absence of dynamic BIND_NOW is insufficient to claim full RELRO. Pages covered by RELRO may be protected after relocation while lazy PLT binding can leave relevant GOT slots writable. A nonexecutable stack is a separate mitigation. Inspect PT_GNU_RELRO, PT_GNU_STACK, DT_BIND_NOW/DF_BIND_NOW/DF_1_NOW and relocations for each required object. Loader environment forcing eager binding would be a separate effective policy claim, not a property inferred from these tags. |
| `_json` API symbols / `VERIFIED_FROM_STATIC_ELF`; `SUPPORTED_BY_UPSTREAM_SOURCE` | Exact `_json` has no DT_NEEDED entries and lists undefined CPython API requirements including `PyDict_Next`, `PyModule_AddObjectRef` and `_Py_NoneStruct`. It requires resolution from the executable/global link scope. CPython's shared-module loader uses `dlopen` and resolves the initializer; see the pinned CPython `Python/dynload_shlib.c` reference in the [ELF review](stage9_observer_3a_elf_qualification.md#primary-source-checks-and-local-observations). Check undefined/versioned symbols, executable exports, relocation types and selected dlopen flags. Do not invent a `_json → libpython` DT_NEEDED edge or a reverse dependency merely to express symbol lookup. No-DT_NEEDED does not mean self-contained. |
| Registration / `VERIFIED_FROM_STATIC_ELF` | Actual pointer/table/sentinel records confirm the builtin and frozen memberships above, including `_datetime`; exact package generated `config.c` agrees. Symbol-only inference is superseded for these exact bytes. The source alias table, generated importer data and startup frozen-selection policy must still be distinguished from a frozen record being selected in a particular run. No target execution is claimed. |
| OpenSSL / `OBSERVER_DESIGN_UNDECIDED` | `_hashlib` loading can initialize OpenSSL and consult configuration/provider machinery. Pin exact libcrypto/provider builds, compiled default paths, configuration and includes, permitted environment and API behavior. The existing openssl.cnf digest/excerpt does not prove absence of all other configuration or API-selected providers. File presence does not require legacy.so, `_ssl` or libssl in the Observer runtime; removal needs proof that required, fallback, initialization and error paths cannot need them. |
| Loader search / `NOT_YET_PROVEN` | Absence of interpreter RPATH/RUNPATH does not close search. Inspect every object's tags; bind the execution-visible PT_INTERP path, cache/preload, defaults, requesting-object context, environment, hwcaps and already-loaded SONAME reuse. ld.so.conf usually feeds cache creation, while the runtime consumes the selected cache. A copied loader at another path does not redirect the kernel's PT_INTERP lookup. |
| Import and other data / `NOT_YET_PROVEN` | Bind source files, package parents, frozen bytecode containers, encoding modules, configuration and required data to accepted digests and protected ancestry. Decide locale/encoding, NSS and OpenSSL configuration paths from the selected design and builds. A .py file is executable input, and labeling a loadable candidate as data does not make it inert inside an import root. Data closure is separate from ELF adjacency. |
| Graph versus execution / `NOT_YET_PROVEN` | A finite static graph records request relationships; it does not choose actual filesystem paths, prove load order or symbol scope, authenticate files, or enforce integrity during delayed loads. Keep PT_INTERP, DT_NEEDED, native imports, dlopen and symbol lookup distinct in the analysis. The historical 12-object/19-DT_NEEDED subgraph is acyclic as reported; this is not evidence of complete permitted closure. |

The [upstream source reference matrix](stage9_observer_3a_elf_qualification.md#primary-source-checks-and-local-observations)
provides semantic support for CPython dynamic imports, glibc search/reuse and
OpenSSL provider loading. **`SUPPORTED_BY_UPSTREAM_SOURCE`** means support for
those mechanisms only. The cited upstream versions are not substituted for the
exact Ubuntu patched builds.

**`OBSERVER_DESIGN_UNDECIDED`:** the current DAG restriction remains intact.
Direct source inspection of
[`_proposed_acyclic`](../../deployment/stage9_observer_contract.py#L1202) and
[`_validate_proposed_runtime`](../../deployment/stage9_observer_contract.py#L1466)
confirms an iterative acyclicity check over the declared binary graph,
`IN_IMAGE_ONLY`, explicit builtin/frozen containers and source/native import
records. It does not inspect actual ELF contents or enforce loader behavior.
Codebase Memory tools/executable were unavailable in this session; these
conclusions use actual source, with no structural coverage or exhaustive
repository claim. A required `libcrypto --dlopen→ legacy.so --DT_NEEDED→
libcrypto` relation would require an explicit architectural decision, but that
first edge and its necessity have not been proved for this Observer. Do not
implement SCC, drop a required edge or exclude a required backend to force a
DAG result.

## Findings register

Severity describes the affected approval boundary, not evidence strength.
Earlier CA-/DELL- findings remain open unless separately and explicitly
resolved. This document neither retroactively changes their severity nor
promotes a reported observation into trusted evidence.

| ID / priority | Label | Finding and disposition |
| --- | --- | --- |
| CPY-P0-00 / P0 | `NOT_YET_PROVEN` | No new confirmed P0 defect was established in this bounded static review. This is not proof that no P0 exists and provides no activation permission. |
| CPY-P1-01 / P1 | `NOT_YET_PROVEN` | Exact package membership, generated config and linked module tables now narrow the build gap. Independent acceptance, complete expected runtime and reproducible source/patch/config/toolchain-to-output linkage remain missing; inspected artifacts do not supply external acceptance. Keep CA-P1-01 / DELL-P1-01 blocking image compatibility and final 3A approval until the exact inventory, provenance and contextual closure are reconciled. |
| CPY-P1-02 / P1 | `NOT_YET_PROVEN` | Static import/ELF adjacency does not prove effective protected launch, complete scope or lifetime integrity. CA-P1-03 / DELL-P1-02 remain blocking for runtime qualification; host PID namespace and attestation remain unverified/missing. |
| CPY-P1-03 / P1 | `OBSERVER_DESIGN_UNDECIDED` | Entrypoint/worker, cryptographic acceptance verifier/backend, provider/configuration policy and exact selected image require independent design decisions. Parser-only imports cannot close the future Observer's dependencies. CA-P1-02's external reviewers, roots, signatures, time/checkpoint and rollback requirements remain separate blockers. |
| CPY-P2-01 / P2 | `BUILD_SPECIFIC_UNVERIFIED` | The Dell triage archive and its 14-entry manifest have not been independently read by Codex. The supplied review correction reports 14/14 verified and 15 regular files total, superseding the original imprecise 13/13 claim. Preserve the operator-reported status and obtain retained bytes before claiming Codex replay. Exact package table records independently resolve the earlier symbol-only membership question; they do not authenticate the Dell capture. Any remaining required-module selection/completeness gap belongs to CPY-P1-01. |
| CPY-P2-02 / P2 | `VERIFIED_FROM_STATIC_ELF` | Independently confirmed non-PIE and GNU_RELRO without BIND_NOW require an explicit hardening disposition for the selected profile. No exploit, existing policy violation or severity P0 is proved by these tags alone. Do not silently alter link flags or security policy in this documentation change. |
| CPY-P2-03 / P2 | `OBSERVER_DESIGN_UNDECIDED` | Effective OpenSSL configuration/provider selection and loader/import search remain unresolved. Required dynamic loads or alternate crypto backends must be reconciled before DAG acceptance. Presence of 47 libraries is neither proof of necessity nor permission to omit them. |
| CPY-P2-04 / P2 | `NOT_YET_PROVEN` | Earlier CA-P2-01–04 remain open: advisory offline canonical/bundle scope, document precedence, actual resource enforcement and replay/receipt/crash protocols. This static build review supplies none of those mechanisms. Validate this PR's actual HEAD separately from historical CI results. |
| CPY-P3-01 / P3 | `NOT_YET_PROVEN` | Hashes, Build IDs and an internally consistent manifest identify/relate bytes; their origin, independent expected selection and reviewer approval remain separate. CA-P3-01's frozen dataclass limitation also persists: hostile code inside the verifier process is inside its trust boundary. |

## Missing data and evidence

All rows identify a remaining boundary, rather than asking for a fresh physical
measurement by default. Exact archive/build artifacts should be examined in
Codex first, without executing their contents.

| Label | Needed evidence | Preferred source / why it matters |
| --- | --- | --- |
| `BUILD_SPECIFIC_UNVERIFIED` | Original triage archive bytes, manifest paths/sizes, exact commands, tool versions, stdout/stderr/exit codes and collection time/context | Retained operator archive first; lets Codex replay the supplied corrected 14/14 check, reconcile 15 regular files including the manifest, and distinguish table content from symbol sightings. Manifest SHA-256 and readelf 2.42 are supplied review results, not a Codex archive read. |
| `BUILD_SPECIFIC_UNVERIFIED` | Formal archive trust/time acceptance, historical build logs/.buildinfo and reproducible source/patch/toolchain-to-output linkage | Current signed indexes, exact package/source hash checks and generated configuration are available in this review. They do not replace accepted roots or reconstruct the historical linker invocation. |
| `NOT_YET_PROVEN` | Bidirectional reconciliation of confirmed builtin/frozen records and exact source aliases/configuration with the selected Observer's startup/import policy | Linked tables and exact config/ABI materials now exist; remaining work is the complete intended closure and accepted selection, not another symbol count or target execution. |
| `OBSERVER_DESIGN_UNDECIDED` | Entrypoint/startup/error-path scope, crypto verification backend, permitted algorithms/providers/configuration and unsupported behavior | Independent Trusted Observer design decision; no Dell scan can settle intended functionality. |
| `NOT_YET_PROVEN` | Ordered dynamic records, undefined/exported/versioned symbols, canonical package members/host aliases, actual search contexts and complete required data/config inventory | Exact required objects first; only then any narrowly missing host-layout observations. Do not substitute ldd or a runtime trace. |
| `NOT_YET_PROVEN` | Independent expected image selection, authenticated host/run/launcher/channel scope, immutable ancestry and lifetime protection, enforced resources | Separately reviewed acceptance/qualification evidence; static package inspection cannot discharge these requirements. |

## Minimal further static Dell work, only if needed

1. **`NOT_YET_PROVEN`: reuse before remeasurement.** Transfer the already
   retained `stage9-elf-triage-rNlpy9.tar.gz` through the approved evidence
   channel and independently check its external SHA-256 and internal manifest.
   Transfer existing Ubuntu raw signed metadata/package/build evidence where
   available. This is evidence recovery, not a request to run Python again.
2. **`BUILD_SPECIFIC_UNVERIFIED`: fill only residual file-identity gaps.** If
   exact package bytes/configuration and the retained archive cannot settle a
   required fact, the operator may use the existing trusted-tool,
   clean-environment, bounded
   [read-only procedure](stage9_observer_3a_dell_readonly_diagnostics.md) on
   specifically named system-package files. Preserve before/after identity and
   hashes, full `readelf -W -h -l -d -n -V` output, and symbol/relocation output
   only where required for the registration or ABI question. Stop on missing
   tools, unreadability, drift, timeout or limits; do not install or repair.
3. **`NOT_YET_PROVEN`: capture only necessary configuration/layout.** If
   unresolved, collect exact canonical paths/aliases and package ownership for
   required objects; static existence/digests and bounded contents for selected
   loader/OpenSSL configuration/includes/cache and import data. Do not print
   secrets or dump process environments. These captures describe the sampled
   host layout and do not prove effective future loader resolution.

No new Dell execution is necessary merely to count existing symbols, fetch
Ubuntu build sources, settle architectural choices or recheck hashes already
available as retained bytes. No step here authorizes target CPython, imports,
Observer, ldd, loader `--list`/tracing, installation, system modification,
physical qualification, Stage 3B or a contract/security-policy change.

## Disposition

**`NOT_YET_PROVEN`:** an additive documentation-only merge can be considered
conditionally after raw diff, manifest/link checks, required format gate and
applicable CI are reviewed for its actual HEAD. Such a merge records findings;
it does not accept the target runtime. Final 3A approval and production remain
blocked by the findings above. No automatic merge is authorized.

```text
stage_9=IN_PROGRESS
qualification=BLOCKED
CONTRACT_APPROVED=NO
STAGE_3A_COMPLETE=NO
PRODUCTION_QUALIFIED=NO
production_status=PRODUCTION_NO_GO
runtime_attestation=MISSING
host_pid_namespace=UNVERIFIED
```
