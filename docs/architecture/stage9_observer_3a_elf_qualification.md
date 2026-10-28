# Stage 9 / 3A — ELF model compatibility qualification

Review baseline: `ae712aea85e4cbc372bdf226d6a5bc5be02f4949`, 2026-10-28.
Status: **PROPOSED_UNAPPROVED; exact-target compatibility UNVERIFIED.**

## Decision for this change

Retain the existing DAG restriction in the inactive proposed parser. It is a
restriction of this candidate profile, not a validity rule of ELF or glibc.
It cannot be retained **as a qualified Ubuntu runtime contract** until an
independently accepted inventory proves that the exact required runtime fits
it. Conversely, evidence that glibc supports cycles does not prove this
Observer's required Ubuntu runtime needs a cycle. No parser behavior, launcher,
runtime activation, physical test or Stage 3B verifier is added here.

The current evidence does not settle that compatibility question. The options
for independent decision are (a) prove the complete selected runtime is within
the existing restriction, or (b) approve an explicitly incompatible, typed and
bounded graph revision, such as the candidate below, against a real required
counterexample. Merely dropping `_proposed_acyclic` would be unsafe.

## Source assessment and edge meanings

Reviewed directly: proposed profile sections 3.5–3.6, the final review's ELF
finding, `deployment/stage9_observer_contract.py::_validate_proposed_runtime`,
`_proposed_acyclic`, `_validate_launch_policy`, and the graph/import cases in
`tests/test_stage9_observer_contract.py`. Codebase Memory was unavailable in
this session; no structural-index coverage or exhaustive repository claim is
made. Source inspection is the basis for these conclusions.

The present edge is exactly `{requester_file_id, needed, resolved_file_id}`.
The requester and result must be declared binary IDs; `(requester, needed)`
is unique; some interpreter-to-loader edge must exist. The iterative graph
check rejects self-loops and every cycle across all declared binary roles.
Proposed bundle validation also binds the ordered edges into launch policy.
These are valuable structural properties. They do not check the binary's
program headers, dynamic tags, request type, search operation, load order,
symbol versions, namespace state, or whether a required edge was omitted.
The existence of an arbitrary interpreter-to-loader edge does not establish
that its `needed` is the actual `PT_INTERP` string. Regular-role set equality
does not establish reachability or completeness against an expected build.

| Subject | Required exact interpretation and present evidence gap |
| --- | --- |
| `PT_INTERP` | A program-header pathname names the interpreter loaded by the kernel at exec. It is not a SONAME search or a `DT_NEEDED` entry. It resolves in the execution mount namespace, including when the executable itself was obtained through an FD. Bind its exact bytes to the selected loader and visible absolute path. A loader file under `/observer` does not redirect an unchanged `/lib64/ld-linux-x86-64.so.2`. The chosen runtime must provide an authenticated mapping or a reproducibly transformed interpreter with accepted provenance. |
| Recursive `DT_NEEDED` | Preserve every occurrence and its dynamic-tag order for every reachable ELF object, not just the interpreter. Resolve dependencies of the loader, libpython, native extensions, optional objects and shared descendants using the pinned loader's actual rules. An already mapped object can satisfy a request. Symbol lookup/relocation order is additional state; graph reachability alone is insufficient. |
| libpython | Do not assume it exists or is a shared dependency. Establish the executable's link mode and actual dependency strings; bind each supported shared libpython's bytes, ABI and build. Builtin/frozen modules bind to the actual interpreter/libpython containers. Undefined `Py*` symbols in an extension are not by themselves `DT_NEEDED` edges back to libpython. |
| Loader | Inspect its own ELF metadata. Do not invent a reverse loader-to-interpreter or loader-to-libc edge because the loader performs loading. Such an edge confuses a loading actor with a dependency request. Exact Ubuntu loader build/patches and default paths remain unknown. |
| RPATH/RUNPATH | Record exact tag values, token expansion and requesting/loading context. `DT_RUNPATH` is not inherited like `DT_RPATH`; a single flat list of directories cannot emulate both. Reject empty/relative/escaping/unsupported expansions in a qualified profile, or specify and authenticate their complete semantics. `IN_IMAGE_ONLY` is presently a declaration, not enforcement. |
| Dynamic `dlopen` | Strings in `DT_NEEDED` cannot enumerate computed calls. CPython native imports, `ctypes`, NSS, locale conversion modules and configured crypto providers are potential paths requiring build/application/config analysis. Record the actual calling object, exact request, flags and supported namespace state. A bounded list of allowed files is necessary but cannot establish that all possible requests were discovered. |
| Native extensions | Inspect every allowed extension's own dependency closure and ABI/symbol requirements, including dependencies located in wheel-relative directories. Bind module name/path/digest, accepted suffixes, parent packages, embedded inventory and import roots. Inventory of installed `.so` files does not establish the Observer's required import set; error paths and deferred imports matter. |
| Loader configuration | Account for the actual startup namespace view of `/etc/ld.so.cache`, `/etc/ld.so.preload`, loader defaults, and any application config consumed at runtime. `/etc/ld.so.conf` and its includes normally feed cache generation, rather than being a replacement for the cache's observed resolution. Pin runtime-consumed files, or authenticate exact absence and prohibit fallback. `LD_*`, audit/filter mechanisms and externally supplied FDs must not add objects. |
| glibc/hwcaps | Pin loader build and supported CPU/hwcaps selection. Distinguish alternative files with the same SONAME; CPU-dependent selection, cache and IFUNC behavior must be within the approved profile. An unordered union of variants is insufficient to prove which bytes execute. Unsupported CPU/profile/configuration must refuse. |
| Cycles/shared dependencies | A shared descendant is one immutable object with multiple incoming edges, not a cycle or duplicate file. glibc remembers loaded/visited maps; cyclic dependency graphs are not universally invalid. A supported SCC needs a finite member/edge set plus deterministic resolution state, not recursive expansion without a visited set. |
| Provenance versus ELF | Keep typed F/S/T provenance acyclic. A runtime dependency cycle does not authorize a transform/source provenance cycle. The reciprocal assertion that a transform outputs a file is not a new provenance dependency. Runtime link topology, load events and provenance are different relations. |
| No dependencies outside image | Prove every successful resolution maps to one accepted immutable object and every attempted unlisted resolution fails. This includes kernel interpreter lookup, all loader search candidates and later opens. A canonical manifest can assert this without making it true. Protected contents, namespace mapping and lifetime enforcement are separately required. |

The three-field edge conflates at least three materially different relations
and does not encode their context. This is an open **P1 resolution-semantics
and exact-build qualification gap**, alongside the missing authenticated
inventory. It is not a reason to add an ad hoc accepting wire variant now.
Current tests correctly prove declared DAG rejection, including optional
binary roles, and reject config files as binary nodes. They cannot validate
actual resolution because the parser deliberately performs no ELF/host I/O.

## Primary-source checks and local observations

The following upstream sources were retrieved and read for semantic review.
The URLs use immutable commits where available; content SHA-256 records the
retrieved file. HTTPS/tag resolution and these locally computed hashes do
**not** authenticate Ubuntu packages, accepted build materials, or reviewer
approval. Upstream versions are semantic references, not a claim about the
target's installed patch version.

| Source | Ref / retrieved file SHA-256 | Relevant behavior |
| --- | --- | --- |
| [glibc `elf/dl-deps.c`](https://github.com/bminor/glibc/blob/ef321e23c20eebc6d6fb4044425c00e6df27b05f/elf/dl-deps.c) | glibc-2.39; `47a1cb0f16d2073e2992af1a17c5aa269ffdd08b606a5a7e627bbfe6988d8fb1` | `_dl_map_object_deps` iterates the dependency list, remembers mapped dependencies, marks each enqueued object and avoids repeatedly adding it. `DT_FILTER`/`DT_AUXILIARY` are additional mechanisms which a closed profile must reject or explicitly model. |
| [glibc `elf/dl-load.c`](https://github.com/bminor/glibc/blob/ef321e23c20eebc6d6fb4044425c00e6df27b05f/elf/dl-load.c) | glibc-2.39; `26345f234ef918c866e40bb86524bdfd4c3beac487af81b3f51668379e671ae0` | `_dl_map_object` checks already loaded names/SONAMEs before searching. RPATH inheritance, RUNPATH, environment, cache and defaults are distinct paths. This supports finite reuse of shared/cyclic objects; it does not require all graphs to be DAGs. |
| [glibc `elf/dso-sort-tests-1.def`](https://github.com/bminor/glibc/blob/ef321e23c20eebc6d6fb4044425c00e6df27b05f/elf/dso-sort-tests-1.def) | glibc-2.39; `697df14ce553fbd78cea30210a8926b3c3ef10a7d41cd6c20fe9b9d0a717414e` | `tst-bz28937` declares `a->a2` and `a2->a`, with expected results for both loader sorting algorithms. The upstream regression explicitly acknowledges indeterminate destructor order. This is a real loader-supported dependency-cycle test definition, not evidence of a cycle in the selected Observer. No upstream test was executed in this task. |
| [CPython `Python/dynload_shlib.c`](https://github.com/python/cpython/blob/f6650f9ad73359051f3e558c2431a109bc016664/Python/dynload_shlib.c) | v3.12.3; `6b78ca8ca5ebe73deba0576ea17b4df48434741d16b8ef34952498638c4cc220` | `_PyImport_FindSharedFuncptr` calls `dlopen(pathname, interpreter flags)` and then obtains the module initializer. Native imports therefore add dynamic resolution not described by the Python executable's `DT_NEEDED` list. |
| [OpenSSL `crypto/provider_core.c`](https://github.com/openssl/openssl/blob/85cf92f55d9e2ac5aacf92bedd33fb890b9f8b4c/crypto/provider_core.c) | openssl-3.0.13; `9ae2d4300afd90bc8c32c09b1dc2541ad9938a83e9d181ad5d30f6ad2ef56e6b` | Provider initialization chooses an application/configuration/environment/default module path and uses `DSO_load`. Providers require an explicit policy decision; declaring libcrypto alone is insufficient. |

The [Linux loader manual](https://man7.org/linux/man-pages/man8/ld.so.8.html)
independently documents pathname versus name resolution, RPATH/RUNPATH,
dynamic string tokens, cache/default search and hwcaps. The
[dlopen manual](https://man7.org/linux/man-pages/man3/dlopen.3.html) documents
calling-object search context, recursive dependencies, flags and object reuse.
These references explain why type and context cannot be discarded.

Read-only static observation on **this cloud host**, not the Dell: Debian 13,
amd64; binutils 2.44-3; libc6 2.41-12+deb13u3; libssl3t64 3.5.7-1~deb13u2;
system Python 3.13.5-2+deb13u4. `readelf -W -d`, `readelf -W -l`,
`dpkg-query`, `readlink` and `sha256sum` were used. No inspected target ELF,
historical installation, loader, provider or Observer was executed; `ldd`
was not used.

| Local file | SHA-256 | Static observation |
| --- | --- | --- |
| `/usr/lib/x86_64-linux-gnu/libc.so.6` | `fa430b8f298f817a266046af84a77533185ad6fc4406c7d3787b5a0a0c207826` | `DT_NEEDED=ld-linux-x86-64.so.2`. |
| `/usr/lib/x86_64-linux-gnu/ld-linux-x86-64.so.2` | `438c546d8e8cc48496bf3a95f753051afd9db66a629a74e31a9ded71586b56e0` | No `DT_NEEDED` entry in the inspected dynamic section. There is no observed libc↔loader cycle. |
| `/usr/lib/x86_64-linux-gnu/libcrypto.so.3` | `8bb5f3fdffe280d4453eb79a4663c2c47af70b7c247fe2e94e2da703cee1fd3d` | `DT_NEEDED` names libz.so.1, libzstd.so.1 and libc.so.6. |
| `/usr/lib/x86_64-linux-gnu/ossl-modules/legacy.so` | `533bb6758a00a87a2d7835bcd3ebca58e0def2c1096ec582dea180adceeb7404` | `DT_NEEDED` names libcrypto.so.3 and libc.so.6. |

This provides a concrete **conditional mixed-edge cycle**: if an application
requires dynamically loading that legacy provider from libcrypto, the graph
contains libcrypto —dlopen→ legacy.so —DT_NEEDED→ libcrypto. The second edge
is locally observed; the first is a supported provider mechanism whose actual
selection was not executed or observed here. This is not evidence that the
Observer requires the legacy provider, that Ubuntu's provider has identical
linkage, or that the Dell has any such loaded object. Disabling or excluding a
provider is only a viable DAG-profile choice after proving no required/error
path needs it. Do not claim a target cycle from this example.

## Candidate alternative: bounded typed resolution graph

This section is an **unimplemented alternative for independent design review**.
It is not an extension accepted under the current V1 parser. A separate
version/profile and migration decision must cover manifest, launch policy,
digest/signing bindings, limits and tests before any implementation/activation.

1. **Objects and contexts.** Every node is one accepted regular ELF file ID
   binding exact bytes, size, absolute execution-visible path, source proof,
   architecture/ABI and build identity. Declare one base link namespace,
   the pinned loader, ordered startup roots and a finite list of resolution
   contexts. Each context binds exact loader/config/cache digests or required
   absence, CPU/hwcaps profile, caller, ordered already-loaded object list and
   load flags. No free-form context, second namespace, `dlmopen`, deep binding,
   audit/preload/filter/auxiliary mechanism or lookup hook is supported by this
   initial alternative; unexpected use refuses. The authenticated namespace
   mapping must preserve the host observation view and privilege boundary.
2. **Edges.** Use disjoint closed variants. `interp` has executable ID, exact
   `PT_INTERP` pathname and loader ID, exactly one for the executable.
   `needed` has requester ID, dynamic-tag ordinal, exact raw dependency string,
   context ID and resolved ID. `dlopen` has calling-object ID, approved callsite
   identity derived from reviewed build materials, exact request bytes, exact
   flags, context ID and resolved ID. Relative/null/generated/wildcard requests
   are unsupported. Type and ordinal distinguish equal strings with different
   meanings; no conversion between variants is allowed. ELF symbol-resolution
   requirements and versioned symbols are separately checked against the same
   ordered object scope; SCC membership is not a symbol provider decision.
3. **Bounds.** Proposed ceilings: 32768 objects, 32768 edges total, 128 contexts,
   and 64 members in a nontrivial SCC; each edge/context/member list is unique
   and bounded before graph construction. Existing 4 MiB manifest, 64 KiB
   launch-policy, depth, path, memory and elapsed-time limits continue to apply,
   whichever is stricter. These values are candidate decisions, not evidence
   that a real target fits. Exceeding any limit refuses, without truncation,
   fallback or automatic raising of a limit.
4. **Deterministic topology.** Build the complete directed object graph from
   typed edges, traverse iteratively with a visited set and compute SCCs in
   O(V+E) graph work and O(V+E) space. Reject absent/incorrect node references
   before traversal. Sort node IDs and edge keys bytewise for reproducible
   output; the component ID is its least member ID, and components/members are
   listed in that order. The condensation graph is a DAG. A self-loop or SCC
   is allowed only if its exact members and internal typed edges are separately
   listed in the accepted profile; undeclared or enlarged SCCs refuse. Graph
   work bounds do not replace JSON/content/resolution-worker limits.
5. **Resolution, not SCC ordering.** Do not load objects in arbitrary component
   or lexicographic order. Preserve pinned loader startup/dependency ordering
   and the approved finite transition order for later dlopen events. Repeated
   loads must resolve to the exact same immutable object in the same namespace.
   All approved contexts must give the declared result. If load-order/SONAME
   reuse can select different bytes, the profile refuses rather than choosing
   a convenient graph path. Cyclic constructor/destructor behavior needs
   build-specific review and controlled qualification, never inference from
   an acyclic condensation graph.
6. **Completeness.** Start with the interpreter, loader and every required
   builtin/frozen/native import and dynamic request in independently accepted
   build/application materials. Inspect every reachable object's ELF records
   and every resulting file exactly; reconcile both directions against the
   manifest. Require no missing or unexplained extra edge, object, context,
   loadable candidate or config. Every possible successful load of approved
   code, including error paths, must be accounted for. Static string search,
   one trace, package dependency lists, an SBOM alone, or the graph's fixed
   point cannot prove computed dlopen completeness. Unsupported/unbounded
   dynamic behavior blocks qualification; prohibit it or change the reviewed
   design explicitly. Account for non-ELF data such as encoding/locale/NSS
   configuration separately, without converting these files into ELF nodes.
7. **No substitution or fallback.** Accept only independently authenticated
   expected digests; protect the exact objects and path ancestry before and
   throughout execution. A verified FD for Python does not pin `PT_INTERP`
   or later loader opens. Prevent replacement via writable mounts, aliases,
   links, cache, same-SONAME objects, hwcaps variants, inherited state and
   post-verification mutation. Pin or exclude all search candidates and make
   host/default/external paths inaccessible to resolution without falsifying
   host-scope observations. Missing accepted files must refuse even if an
   ABI-compatible host library exists. Stage 3A can specify these proof
   obligations; it cannot establish runtime enforcement by declaration.

### Threat assessment for admitting cycles

| Threat | Required restriction / evidence |
| --- | --- |
| Cycle conceals unexamined dependencies or causes unbounded traversal | Full visited-set traversal and bidirectional reconciliation with authenticated build inventory; explicit object/edge/context/SCC bounds; no early success at first revisit. |
| Approved SCC becomes a blanket allowlist | Pin exact internal typed edges, external edges and context outputs; an added member/edge or different result refuses. |
| Same SONAME binds to attacker or different first-loaded object | Pin ordered startup/load contexts and all loadable candidates; prove loader object reuse chooses the accepted bytes. Hashing a different same-named file is insufficient. |
| Search path/token/hwcaps/cache diverts a cyclic back-edge | Pin exact search semantics and CPU/configuration context; refuse ambiguity, missing-file fallback and unlisted variants. |
| Symbol interposition or constructor order changes behavior | Review symbol versions/ordered scope and build-specific initialization; forbid unsupported loader flags/namespaces. SCC condensation alone does not preserve semantics. |
| Cycles weaken provenance or authenticate themselves | Keep provenance DAG separate; expected inventory, roots and approval are external to the submitted artifact. No runtime edge supplies approval authority. |
| Verified bytes change before a later dlopen | Independently qualified immutable image/lifetime protection covering all late-loaded files; drift makes observations incomplete and qualification blocked. |

## Test and evidence matrix

Rows labelled **required** are planned obligations, not executed tests or
qualification results. Existing synthetic tests exercise only the current
inactive DAG proposal. This change implements no SCC parser.

| Case | Required positive evidence | Required negative evidence / disposition |
| --- | --- | --- |
| Current graph algorithm | Existing iterative long DAG/shared-node fixture parses as proposed synthetic data. | Existing provenance/ELF/optional-role cycles and config-as-binary reject. These do not prove real closure. |
| Typed headers/tags | Exact build's `PT_INTERP`, ordered `DT_NEEDED`, correct libpython/no-libpython variant. | Wrong tag type/string/ordinal, invented loader reverse edge, missing/extra dependency or non-ELF target refuses. |
| Shared objects and SCCs | Required candidate SCC with exact internal/back edges; diamond shares one file; deterministic output under input permutations. | Missing back-edge, undeclared self-loop, altered/enlarged component, duplicate IDs and 65-member SCC refuse. |
| Bounded computation | Maximum accepted bounded graph/context with measured memory/time below the retained ceilings. | Over-limit nodes/edges/contexts, malformed/deep input, timeout/worker exhaustion refuse; no partial-success output. |
| RPATH/RUNPATH | Accepted transitive RPATH and direct-only RUNPATH contexts match pinned loader build. | Inherited RUNPATH assumption, empty/relative/escaping tokens, unlisted default/cache paths refuse. |
| dlopen/native imports | Every required extension and approved finite dynamic context resolves to accepted files; repeated load reuses the same node. | Unknown caller/request/flags, required error-path import omitted, null request, dlmopen/deepbind, plugin/config injection refuse. |
| CPU/cache/namespace | Exact approved variants and execution-visible `PT_INTERP` mapping in the faithful host view. | Wrong CPU/hwcaps, modified cache/preload, external bind mount or same-SONAME first-load substitution refuses. |
| Artifact/lifetime integrity | Independently accepted bytes and provenance; retained image protection across delayed loads. | Rename/overwrite, alias/link, changed inode/mount, accepted-file removal with host replacement refuse. |
| Completeness | Independently reviewed full inventory plus artifact inspection and separately authorized controlled resolution evidence reconcile. | Host-generated list, declarations, package dependencies, single successful trace, DAG/SCC success alone remain UNVERIFIED. |
| Authority | Existing standard parsers continue to refuse and proposed output stays synthetic/unapproved. | No SCC input, local digest, signed host list or test result can set CONTRACT_APPROVED/STAGE_3A_COMPLETE/PRODUCTION_QUALIFIED. |

## Exact-target qualification still required

The Dell diagnostic procedure supplies **local observations**, not independent
expected materials or permission to run them here. Independent reviewers need
the selected Ubuntu 24.04 amd64 CPython 3.12 package/build identity, executable,
loader, libpython decision, stdlib/embedded/native module inventory, complete
system/private library and config/data closure, CPU profile, and exact image
layout. Authenticate Ubuntu materials through externally accepted archive keys,
signed snapshot `InRelease` (or Release/Release.gpg), its Packages digests and
the exact `.deb` SHA-256 chain. Pin suite, pocket, architecture, versions,
snapshot freshness and expiry decisions; do not trust a package name/version
from the host database as that chain. If a custom interpreter/image is selected,
require separately authenticated builder/source/toolchain/configuration
materials and accepted reproducible transformation/output evidence, including
embedded modules and any patched interpreter path. Reviewer keys, accepted
inventory digest, review decision and revocation/checkpoint state must be rooted
outside the submitted inventory; two matching host-generated lists do not
provide that independence.

Only then can the review decide whether the required exact dependency graph
fits the current DAG restriction or requires a versioned SCC alternative.
Artifact verification and actual launch/resolution/lifetime proofs belong to
separately authorized later work. Until those gates close:

```text
Stage 9 = IN_PROGRESS
qualification = BLOCKED
production_status = PRODUCTION_NO_GO
runtime_attestation = MISSING
host_pid_namespace = UNVERIFIED
full_confirmation = false
coexistence_authorized = false
preflight_exception = false
physical_test_status = PHYSICAL_TEST_PENDING
```
