# Stage 9 / 3A — minimal CPython runtime and native import scope V1

Revision **V1**, source analysis **2026-10-28 UTC**, completed
**2026-10-29 UTC**. Source baseline:
`main` **3946695a8c7a37b09e9c494bb07caf68ccbbf81f**, including merged
[PR #3221](https://github.com/Dudzian/Dudzian/pull/3221).
Disposition: **source-grounded qualification scope; incomplete;
PROPOSED_UNAPPROVED; qualification BLOCKED**.

The existing contract parser justifies a preliminary application import set,
not a complete Trusted Observer runtime. Its current source also imports
`math`, `typing` and `dataclasses`: a SHA-256/JSON-only native list would miss
important import-time and error-path candidates. No final module allowlist,
complete runtime manifest, safe-removal claim or contract approval follows.

## 1. Evidence and method

The exact uploaded [operator request and new inventory account](evidence/stage9_observer_3a_minimal_cpython_source_v1.txt)
is retained without editing. It contains a summary, not raw terminal evidence.
Its size and SHA-256 are recorded in the [validation notes](evidence/stage9_observer_3a_minimal_cpython_validation_v1.md)
and the new [document integrity manifest](evidence/stage9_observer_3a_minimal_cpython_v1.sha256).
Retention identifies the submitted text; it does not authenticate its claims.

The operator reports comparing `/usr/lib/python3.12/lib-dynload` against
package-declared paths: initially **46 expected / 47 actual**, with `_gdbm`
outside the two-package scope. The new ownership account is:

```text
MODULE: _gdbm.cpython-312-x86_64-linux-gnu.so
OWNER: python3-gdbm:amd64
VERSION: 3.12.3-0ubuntu1
SOURCE: python3-stdlib-extensions
STATUS: ii
SHA256: fd4cfafba8307a1dabd0543400d94c875e08e7d2a4b795620ae86c6f041c7d28
```

After including `python3-gdbm`, the operator reports:

```text
EXPECTED_MODULES: 47
ACTUAL_MODULES:   47
ANOMALIES:        0
STABILITY:        PASS
RESULT: EXTENDED_NATIVE_MODULE_INVENTORY_MATCH
```

This is **operator-reported local path/package-ownership consistency**. It is
neither a byte comparison of all 47 extensions nor proof of the future
Observer's complete runtime. The 47 per-module rows, two original package
names, exact commands, raw outputs and collection times are not supplied in
this new input. The `_gdbm` hash is reported local evidence, not a newly
independently verified DEB-member match. Its package version differs from the
reported core CPython package revision `3.12.3-1ubuntu0.17`; that fact alone
establishes neither ABI compatibility nor an error.

The earlier [Ubuntu provenance report](stage9_observer_3a_ubuntu_provenance_report_v1.md)
records operator-reported **7/7 DEBs and 12/12 selected ELF byte matches**.
Those twelve include only three named CPython extension objects (`_hashlib`,
`_ssl`, `_json`), alongside the executable, loader and libraries/provider.
Do not add 12 and 47, extend the twelve-object evidence to the remaining
extensions, or treat path/owner equality as byte or target completeness.

Directly reviewed governing sources:

- [Trusted Observer contracts and implementation plan](stage9_trusted_observer_contracts_and_implementation_plan.md), especially §§2, 3.2, 4 and 7.
- [Proposed V1 profile](stage9_observer_3a_schema_proposal.md), especially §§3.5–3.6 and the unapproved execution boundary.
- [Eight-contract audit](stage9_observer_3a_contract_audit.md), including CA-P1-01/02/03 and CA-P2-01–04.
- [ELF qualification](stage9_observer_3a_elf_qualification.md), including typed resolution semantics and the retained DAG restriction.
- [Dell local ELF report](stage9_observer_3a_dell_local_elf_report_v1.md), especially `_datetime`, OpenSSL and local-versus-expected inventory.
- The Ubuntu provenance report above, [contract parser](../../deployment/stage9_observer_contract.py), [canonicalizer](../../deployment/canonical_json.py), package initializer and [synthetic tests](../../tests/test_stage9_observer_contract.py).

Codebase Memory MCP and its CLI were unavailable. Targeted repository search
and direct source reads replace graph discovery here; no index-coverage result
or exhaustive repository absence proof is claimed. The
[CPython semantic source map](evidence/stage9_observer_3a_cpython_semantic_sources_v1.md)
labels upstream references separately from an accepted Ubuntu build. Local
development tools are used for documentation/regression checks only. No
system interpreter was executed to authenticate its own module inventory;
no inspected installation payload, Observer, Stage 3B, launcher or physical
Dell test was run. Production code and historical manifests are unchanged.

## 2. Implementation boundary


| Component | Source-backed state | Runtime consequence |
| --- | --- | --- |
| Existing contract parser | `deployment/stage9_observer_contract.py:1–16` describes no I/O and immutable bytes. `_decode` at 265–296 performs strict bounded JSON; `ParsedContract` 299–310 permanently sets authorization UNVERIFIED; `_parse_contract` 370–382 canonicalizes/hashes supplied bytes. `parse_acceptance` 385–392 parses envelope syntax without verifying signatures (`acceptance_signed_message` 364–367 only returns signing preimage). | Direct imports establish only the parser's source requirements. No filesystem scanner, authenticated launcher, artifact verifier or actual host observation follows from parsing. |
| Unapproved structural review profile | Standard `parse_acceptance_policy` 424–457 and `parse_manifest` 460–491 end `_deny()`. Standard result 1019–1071 ends `_deny()`. `ProposedContract` 2019–2025 is PROPOSED_UNAPPROVED, synthetic=True, inherits UNVERIFIED. `parse_proposed_bundle` 2115–2139 cross-checks seven documents and explicitly authenticates no signature/proof/artifact/launcher/host. | Passing review-profile unit tests cannot approve current design, create production acceptance or prove runtime closure. |
| Shared canonicalizer | `deployment/canonical_json.py:1–45`: direct json/math/typing imports; strict canonical byte serialization and UTF-16BE key sorting. | Keep its actual import closure, including math despite floats later being refused. Source documentation “dependency-free” means no third-party dependencies, not literally no runtime dependencies. |
| Proposed Trusted Observer | Plan `docs/architecture/stage9_trusted_observer_contracts_and_implementation_plan.md:60–67, 69–86, 462–510`; observer reads untrusted installations as data in its own minimal runtime; trusted start/scope/lifetime protection are distinct trust boundaries. | Full observer imports remain architecture-dependent; parser imports are a lower bound only if parser is reused. Do not expand existing diagnostic/scanner import closure automatically into target. |
| Proposed offline verifier | Plan 604–609 names `deployment/stage9_observer_offline.py` as future module with Git/wheel/APT/transform verification, narrow adapters, import/ELF data analysis. Narrow existing envelope API is future work (623). Targeted `rg --files deployment scripts tests` found contract/tests/fixtures but not the proposed offline module; not a repository-wide absence theorem. | Offline crypto/gpgv/jsonschema/archive/toolchain dependencies are separate from the minimal observer. Do not import full Windows release verifier or Source/CCA harness merely for reuse (619–628). |
| Future launcher and mechanisms | Plan 23–25, 297–305, 470–485, 495–510, 706–712. LaunchPolicyV1 is a declaration, not an installed launcher. Entire pre-Python runtime, protected image, namespace, FD origin, sandbox, lifetime verified→executed relation need separate approval/evidence. | Launcher library choices are not parser imports and not settled by -I -S -B. No launcher/helper implementation authorized by this analysis. |
| Other existing observations | `deployment/stage9_native_linux_cross_observer_consistency.py:1–4` declares untrusted sequential Source/CCA observations; `_NATIVE_OBSERVERS` 19–25 delegates to existing probes and `_BLOCKERS` (around 65–72) explicitly retains untrusted/unqualified status. Plan 624–628 limits reuse after future closure review. | Existing diagnostics are not the designed Trusted Observer; their presence must not be called a completed Observer implementation or approved minimal-runtime dependency set. |

Other governing sources:

- `stage9_observer_3a_schema_proposal.md:3–10,457–479`: unapproved proposal;
  standard parsers refuse; proposed acceptance compares declarations only.
- `stage9_observer_3a_contract_audit.md:34–58`: eight contract kinds, seven bundle
  inputs, offline output is not the eighth authenticated bundle input.
- Contract audit 71–103: stdlib isolation, declarations do not establish CPython
  bootstrap/suffixes/native-init closure or resource/channel enforcement.
- Contract audit 151–180: exact build inventories, actual resolution, lifetime,
  scope, visibility, replay and output authenticity are external prerequisites.


## 3. Import categories and direct source requirements

The categories answer different questions and may overlap. No module becomes
required merely because a system inventory contains it.

| Category | Meaning in this report |
| --- | --- |
| `STATICALLY_REQUIRED` | Direct unconditional import in the current parser/canonicalizer. Required for loading that unchanged component; Observer inclusion depends on D1. |
| `TRANSITIVE_CANDIDATE` | A source-backed indirect import or branch to investigate, including import-time accelerators, fallback and error paths. |
| `BUILD_SPECIFIC_UNVERIFIED` | Exact accepted-build presence, source/frozen/builtin/native placement, ABI or container binding is not established. Applies to every CPython candidate here. |
| `OBSERVER_DESIGN_UNDECIDED` | Requirement depends on future Observer/verifier/launcher responsibilities or operation choices. |
| `NOT_JUSTIFIED_FOR_TARGET` | Inspected current sources provide no inclusion argument; this is not a safe-removal finding. |

Provisional representation uses `builtin`, `frozen`, `source`, `native` or
`unresolved`. Where source/frozen or builtin/native alternatives are shown,
the final target classification is **unresolved** pending accepted build
materials. Native is reserved for file-backed C extensions, not all C code.


`STATICALLY_REQUIRED` here means required by loading the current source component.
It does not establish that the future Observer will use that component or that a
module is present in an independently accepted target CPython build. All imports
below occur unconditionally at module level; use of an imported name may be on
an error path even when the import itself is unconditional.

| Module | Repository source / line | Component | Provisional representation | Use / call site | Category | Exact accepted CPython build presence |
| --- | --- | --- | --- | --- | --- | --- |
| `__future__` | `deployment/stage9_observer_contract.py:7`, `deployment/canonical_json.py:3` | parser + serializer | source (future compiler directive also binds imported feature) | postponed annotations; do not assume all typing imports disappear | `STATICALLY_REQUIRED` | unverified |
| `hashlib` | `deployment/stage9_observer_contract.py:9` | parser | source | SHA-256 of canonical acceptance payload L353, input contract L382, scope binding L918, projections L1784-1789 and launch policy L1812 | `STATICALLY_REQUIRED` | unverified |
| `json` | `deployment/stage9_observer_contract.py:10` | parser | source package | bounded decode L272, `ParsedContract.as_dict` L310 | `STATICALLY_REQUIRED` | unverified |
| `re` | `deployment/stage9_observer_contract.py:11` | parser | source package + builtin `_sre` candidate | fixed-pattern `fullmatch` L103, L109, L157, L242, L834, L1146, L1196 | `STATICALLY_REQUIRED` | unverified |
| `dataclasses` | `deployment/stage9_observer_contract.py:12` | parser | source | decorators L299, L2018, L2079; immutable record fields L307, L2022-2025, L2090-2094 | `STATICALLY_REQUIRED` | unverified |
| `datetime` | `deployment/stage9_observer_contract.py:13` | parser | source facade; native `_datetime` vs source `_pydatetime` unresolved | numerical calendar constructor + `timezone.utc` L154-170; no `strptime` or `strftime` in this file | `STATICALLY_REQUIRED` | unverified |
| `typing` | `deployment/stage9_observer_contract.py:14` | parser | source + builtin `_typing` candidate | `Any`, `Callable`, `Literal`, `NoReturn` annotations; `cast` runtime call e.g. L76/L82 | `STATICALLY_REQUIRED` | unverified |
| `deployment.canonical_json` | `deployment/stage9_observer_contract.py:16` | serializer shared by parser | repository source | canonical JSON bytes and hashes L353/L380 and projection checks | `STATICALLY_REQUIRED` | repository source established; target image inclusion unverified |
| `json` | `deployment/canonical_json.py:4` | serializer | source package | `json.dumps` for strings L21 | `STATICALLY_REQUIRED` | unverified |
| `math` | `deployment/canonical_json.py:4` | serializer | native extension candidate (not `cmath`) | imported eagerly, `math.isfinite` only in float rejection branch L26-29 | `STATICALLY_REQUIRED` | unverified |
| `typing` | `deployment/canonical_json.py:5` | serializer | source + builtin `_typing` candidate | `Any` annotation | `STATICALLY_REQUIRED` | unverified |

The implicit parent package `deployment` is repository source:
`deployment/__init__.py:1` contains only a docstring. Importing the child package
does not justify importing the rest of deployment or application dependencies.
This is the entire direct-import list of the two inspected modules, not the
complete runtime closure. The serializer's docstring says “Dependency-free”;
its actual source still requires stdlib `json`, `math`, `typing` and `__future__`.


## 4. Preliminary indirect import map

The semantic reference is upstream CPython 3.12.3 commit
`f6650f9ad73359051f3e558c2431a109bc016664`. The retained
[source map and retrieval hashes](evidence/stage9_observer_3a_cpython_semantic_sources_v1.md)
provide immutable links and line anchors. It is **not** the independently
accepted Ubuntu-patched build `3.12.3-1ubuntu0.17`.

Every module named in the tables below is `TRANSITIVE_CANDIDATE` and
`BUILD_SPECIFIC_UNVERIFIED`. Source means a reference source file was inspected;
builtin/native placement follows reference templates provisionally. Exact
accepted target presence is **UNVERIFIED for every row**. The component path
starts at parser/canonicalizer imports from §3; unrelated API-local paths are
labelled conditional rather than silently counted as required.


| Modules / provisional representation | Upstream source anchors (all commit above) | Path condition / use |
| --- | --- | --- |
| `json.decoder`, `json.encoder`, `codecs` — source | `Lib/json/__init__.py:106`, `:107`, `:108` | eager import of `json` |
| `json.scanner` — source | `Lib/json/decoder.py:5` | eager through JSON decoder |
| `_json` — native | `Lib/json/decoder.py:7`, `Lib/json/scanner.py:5`, `Lib/json/encoder.py:6`, `:10`, `:14`; `Modules/Setup.stdlib.in:37` | conditional C accelerator probes, fallback implementations exist; decide support/denial, do not silently drop based on “optional” |
| `enum`, `re._compiler`, `re._parser`, `functools`, `copyreg` — source; `_sre` — builtin | `Lib/re/__init__.py:124-127`, `:338`; compiler `Lib/re/_compiler.py:13-16` additionally imports `re._constants`, `re._casefix` | eager regex setup |
| `unicodedata` — native; `warnings` — source/frozen unresolved | `Lib/re/_parser.py:343`, `:403`; `Lib/re/__init__.py:238`, `:301`; parser warning paths L558/L576/L603/L810/L1041 | lazy named Unicode escape support (`\\N{...}`) and warning/deprecation paths; current fixed parser patterns use neither caller patterns nor LOCALE flags; not justified as unconditional direct requirement |
| `re`, `copy`, `types`, `inspect`, `keyword`, `functools`, `abc` — source/frozen unresolved; `sys`, `itertools`, `_thread` — builtin | `Lib/dataclasses.py:1-11` | eager dataclasses imports, even though only a subset of APIs is used |
| `weakref`, `copyreg`, `types` — source/frozen unresolved | `Lib/copy.py:51-53` | eager through `dataclasses -> copy` |
| `_weakref` — builtin; `_weakrefset`, `_collections_abc` — source/frozen unresolved; `sys`, `itertools` — builtin | `Lib/weakref.py:12-26` | eager weakref initialization; deferred deepcopy/atexit/gc paths L183/L442/L572/L649 require separate path review |
| `abc`, `ast`, `dis`, `collections.abc`, `enum`, `importlib.machinery`, `linecache`, `os`, `re`, `tokenize`, `token`, `types`, `functools`, `keyword`, `operator`, `collections` — source/frozen unresolved; `itertools`, `sys`, `builtins` — builtin | `Lib/inspect.py:144-162` | eager through dataclasses; CLI-only `argparse` L3338 has no current parser use |
| `_ast` — builtin candidate; `contextlib` — source | `Lib/ast.py:27-31` | eager through inspect; AST parsing support is not proof the Observer inspects/executes target code |
| `io`, `opcode`, `collections`, `types` — source/frozen unresolved; `_opcode` — native | `Lib/dis.py:3-9`, `Lib/opcode.py:19`; `Modules/Setup.stdlib.in:39` | eager through inspect/dis; `_opcode` import is a guarded accelerator probe, so retain conditional character |
| `functools`, `os`, `tokenize` — source/frozen unresolved; `sys` — builtin | `Lib/linecache.py:8-11` | eager through inspect; file reads performed by later APIs require separate runtime policy |
| `codecs`, `collections`, `functools`, `io`, `re`, `token` — source/frozen unresolved; `builtins`, `itertools`, `sys`, `_tokenize` — builtin candidate | `Lib/tokenize.py:27-42` | eager through linecache; `_tokenize` target classification requires build record rather than guessing suffix |
| `abc`, `collections`, `collections.abc`, `copyreg`, `contextlib`, `functools`, `operator`, `re`, `types`, `warnings` — source/frozen unresolved; `sys`, `_typing` — builtin | `Lib/typing.py:22-45` | eager despite postponed annotations; `inspect` import at L1800 is lazy, but inspect already justified by dataclasses |
| `os`, `_collections_abc`, `collections`, `functools`, `types`, `abc` — source/frozen unresolved; `sys` — builtin | `Lib/contextlib.py:2-8` | eager through typing |
| `abc`, `collections`, `reprlib`, `types` — source/frozen unresolved; `_thread` — builtin; `_functools` — builtin | `Lib/functools.py:17-22`, `:226`, `:266`, `:342`, `:642` | eager base imports and guarded accelerator imports; do not interpret function-local singledispatch imports as all necessary |
| `_collections_abc`, `keyword`, `operator`, `reprlib` — source/frozen unresolved; `sys`, `itertools`, `_weakref`, `_collections` — builtin | `Lib/collections/__init__.py:29-54`, `:340`, `:351`, `:540` | eager plus guarded native acceleration; `heapq` L631 and `copy` L1192 are API-local |
| `_abc` — builtin; `_py_abc` — source | `Lib/abc.py:85`, `:89`; `Modules/Setup.bootstrap.in:28` | guarded C implementation / source fallback |
| `_operator` — builtin | `Lib/operator.py:414`, `:418`; `Modules/Setup.bootstrap.in:31` | guarded native implementation import |
| `builtins`, `sys` — builtin; `types`, `operator`, `functools` — source/frozen unresolved | `Lib/enum.py:1-5`; `Lib/types.py:4` imports `sys` | eager enum/type setup; enum warnings L400/L1210 are conditional |
| `_datetime` — native; `_pydatetime` — source | `Lib/datetime.py:1-6`; `Modules/Setup.stdlib.in:55` | C implementation, with ImportError fallback to source; choose supported branch explicitly |
| `time`, `sys` — builtin; `math` — native; `operator` — source/frozen unresolved | `Lib/_pydatetime.py:11-14` | eager only when datetime fallback selected; `_strptime` L2089 is lazy |
| `time` — builtin; `_strptime` — source | `Modules/_datetimemodule.c:1723`, `:1754`, `:1772`, `:5217` | Python imports/attribute lookups inside C methods for strftime/current time/struct_time/strptime. No eager import in `_datetime_exec` L6713-6917; repo timestamp constructor does not call these methods |
| `locale`, `calendar`, `re`, `datetime` — source; `time`, `_thread` — builtin | `Lib/_strptime.py:13-22` | branch only if strptime becomes allowed; current parser uses numeric UTC constructor, not strptime |
| `encodings`, `encodings.aliases`, `encodings.utf_8`, `encodings.utf_16_be`, `codecs` — source/frozen unresolved; `_codecs` — builtin | `Lib/encodings/__init__.py:31-33`, `Lib/encodings/utf_8.py:9`, `Lib/encodings/utf_16_be.py:9`, `Lib/codecs.py:10-16`, `:1129` | startup codec lookup plus runtime UTF-16-BE key sorting. `canonical_json.py:39` is a concrete use of utf-16-be, not merely a bootstrap guess |
| `logging` — source | `Lib/hashlib.py:247-248` | error path when a supported hash constructor is missing; it logs exception during hashlib import. Logging's further closure has not been enumerated here |
| `linecache`, `tracemalloc`, `traceback`, `re` — source; `_warnings` — builtin | `Lib/warnings.py:41`, `:56`, `:153`, `:225`, `:382`, `:539`, `:565` | warnings/error-reporting paths and guarded native acceleration; target launcher exception policy must bound them |

This preliminary table deliberately exposes more than only hash/JSON native
accelerators. It is not a resolved execution closure; individual source/frozen
identities, further imports, embedded modules and callback paths remain open.


### Hashlib import-time backend and error paths


| Candidate | Upstream evidence | Consequence |
| --- | --- | --- |
| `_hashlib` — native | `Lib/hashlib.py:169-178`; `_hashlib` functions L182/L190 | eagerly attempted during hashlib import; OpenSSL backend/provider/ELF closure decision remains open |
| `_blake2` — native | `Lib/hashlib.py:73-80`, `:103`, `:126-129`, `:241-248` | preferred even with OpenSSL because hashlib initializes ALL supported algorithms at module import. SHA256-only caller does not eliminate this import candidate |
| `_sha2` — native | `Lib/hashlib.py:95`, `:99`, `:140-141` | fallback for SHA224/256/384/512, including provider constructor failure; CPython 3.12 uses `_sha2` (do not invent `_sha256`/`_sha512` from earlier versions) |
| `_sha1`, `_md5`, `_sha3` — native | `Lib/hashlib.py:89`, `:92`, `:107`, `:113`, `:241-248` | fallback candidates for eager algorithm initialization, although existing parser only requests SHA256 |

`__always_supported` lists algorithms L58-61; the loop L241-248 resolves and
binds every advertised constructor before the parser calls its first SHA256.
The OpenSSL branch also probes a constructor with `usedforsecurity=False`;
the non-OpenSSL fallback imports and returns the function without necessarily
creating a digest object for every name. Removing unrelated algorithms'
modules without resolving initialization/fallback/error semantics is unsupported.
Use of hashing does not itself justify `_ssl`, TLS, CA bundles or sockets.


## 5. Native modules: qualification candidates and unjustified additions

This table is a triage scope, not a minimal-image manifest. `native` describes
an implementation only when it is a file-backed extension; C implementations
can instead be builtin. No placement or presence is confirmed for an exact
independently accepted build. Names below are not asserted to be all, or
necessarily any, of the 47 directory entries unless explicitly observed.

| Module | Existing-code rationale / relevant path | Classification and next qualification step |
| --- | --- | --- |
| `math` | Direct unconditional canonicalizer import at line 4; `isfinite` at line 27 occurs while refusing floats. | `STATICALLY_REQUIRED` **for unchanged canonicalizer**; C placement `BUILD_SPECIFIC_UNVERIFIED`. Qualify actual container/extension and ELF closure even if ordinary valid inputs use no math function. |
| `_datetime` | Indirect through parser line 13. Numeric UTC timestamp validation at lines 154–170. Reported symbol and no standalone file suggest embedded implementation, with registration still unverified. | `TRANSITIVE_CANDIDATE` / `BUILD_SPECIFIC_UNVERIFIED`. Authenticate accepted registration/container or extension and fallback path. Do not invent `_datetime.so`, a separate embedded-module digest or a thirteenth ELF byte match. |
| `_hashlib` | Reference hashlib imports it and selects OpenSSL-backed constructors; parser uses SHA-256, e.g. lines 353 and 382. Earlier Dell table reports `DT_NEEDED=libcrypto.so.3,libc.so.6`. | `TRANSITIVE_CANDIDATE`. Prioritize exact patched hashlib/C implementation and provider/error paths. A historical file match establishes neither actual use nor complete crypto closure. |
| `_blake2` | Reference hashlib deliberately uses its BLAKE2 implementation and initializes all always-supported constructors at module import, beyond the caller's SHA-256 calls. | `TRANSITIVE_CANDIDATE`; high-priority source/build/ELF review. Narrow algorithm usage is insufficient grounds to drop it. C placement and presence remain unverified. |
| `_sha2`, `_sha1`, `_md5`, `_sha3` | Reference hashlib's constructor selection/fallbacks cover SHA-224/256/384/512, SHA-1, MD5, SHA-3/SHAKE. Import-time initialization and OpenSSL failure can reach them. | `TRANSITIVE_CANDIDATE`, with build/backend behavior unresolved. `_sha2` is the CPython 3.12 reference name; do not substitute a different version's `_sha256`. Account for required failures as well as successful hashing. |
| `_json` | Reference json decoder/scanner/encoder attempt accelerator imports; parser uses `loads` and canonicalizer uses `dumps` for strings. Earlier Dell table reports no direct `DT_NEEDED`. | `TRANSITIVE_CANDIDATE`. Select and authenticate accelerator/fallback behavior and extension init/ABI. No direct `DT_NEEDED` is not proof of process-ABI or initialization independence. |
| `_opcode` | dataclasses imports inspect, inspect imports dis, dis imports opcode, and opcode conditionally imports `_opcode` in the reference sources. | `TRANSITIVE_CANDIDATE`; analyze actual accepted source and C registration/ELF. It can be reached before any contract input despite no app-level bytecode inspection. |
| `_typing` | Reference typing imports it. Upstream `Modules/Setup.bootstrap.in:24` lists it among statically required modules. | `TRANSITIVE_CANDIDATE` / provisional **builtin**, not an assumed shared extension or a claimed member of the 47. Accepted Ubuntu build table/container still required. |
| `_sre`, `_collections`, `_functools`, `_operator`, `_weakref`, `_abc`, `_thread`, `itertools`, `time` | C-backed support under re/dataclasses/typing/copy/collections/functools/abc and datetime, plus bootstrap. Source paths appear in §4 / semantic map. | `TRANSITIVE_CANDIDATE` and/or `BUILD_SPECIFIC_UNVERIFIED`. Review as embedded/container candidates or actual extensions according to selected build, not as a fabricated `.so` list. |
| `_ssl` | Historically sampled extension, with libssl/libcrypto/libc dependencies. No ssl, socket or TLS requirement in inspected parser/canonicalizer. | `NOT_JUSTIFIED_FOR_TARGET` on current sources; `OBSERVER_DESIGN_UNDECIDED` only if a future operation requires it. `_hashlib`/libcrypto does not imply `_ssl`/libssl. Historical evidence remains useful. |
| `_gdbm` | New path/owner/version/source/status/hash report. No dbm persistence requirement in inspected sources. | `NOT_JUSTIFIED_FOR_TARGET` on current sources; future persistence design remains undecided. Package ownership is explained locally; usefulness to Observer and signed archive/member byte provenance remain unresolved. |
| `_ctypes`, `_socket`, `_sqlite3`, `_bz2`, `_lzma`, `zlib`, `pyexpat`, process/multiprocessing helpers | Illustrative operation/archive/IPC families, not a claim about individual membership in the 47. No corresponding direct requirement in inspected parser/canonicalizer. | `NOT_JUSTIFIED_FOR_TARGET` on current sources; broader future component paths are `OBSERVER_DESIGN_UNDECIDED`. Do not pull offline verifier/harness dependencies into Observer by association. |
| Other unnamed directory entries | No full 47-row account supplied; only `_hashlib`, `_ssl`, `_json` from earlier module observations and `_gdbm` from this input are individually evidenced as local files here. | `unresolved` evidence. A per-module decision for the remaining entries requires the actual names, paths, build mapping and source rationale. No invented all-47 assessment, exclusion or deletion. |

The smallest justified **next qualification scope** is therefore:

1. The selected interpreter, kernel loader and complete recursive startup
   library closure; the reported executable already needs libc/libm/libz/
   libexpat. Omitting Python modules `zlib` or `pyexpat` would not remove those
   executable `DT_NEEDED` obligations.
2. The unchanged parser/canonicalizer's source modules and package parents,
   `math`, datetime strategy, JSON accelerator/fallback, hashlib's complete
   import-time/backend/error strategy, and source-proven C support under
   dataclasses/typing/re. Prioritize concrete file-backed candidates only after
   accepted build tables determine their placement.
3. Under an accepted `_hashlib` strategy, libcrypto and any required provider/
   configuration/deferred-load closure. No automatic `_ssl`, libssl or legacy
   provider inclusion.
4. Full selected-build builtin/frozen/bootstrap inventory and required runtime
   data, then Observer operations and their success/refusal/error/reporting
   paths once designed. This is deliberately not a claimed four-module allowlist.

No system module is removed. `NOT_JUSTIFIED_FOR_TARGET` means that current
sources supply no inclusion argument; it is not proof of safe exclusion from
all future required execution paths or permission to change installed packages.


## 6. CPython bootstrap and runtime data obligations

These obligations apply before and beyond the parser's explicit imports.
All placements below remain `BUILD_SPECIFIC_UNVERIFIED`; a semantic source
reference cannot confirm presence in the exact independently accepted build.
The required *available* builtin/frozen set is the complete set of the chosen
build, even when some entries are not called by application code.

| Area | Source-grounded qualification requirement | Missing evidence / consequence |
| --- | --- | --- |
| Builtins | Derive names, C initializers, build flags and interpreter/libpython container bindings from accepted `Modules/Setup*`, generated inittab/config tables and outputs. Candidate mechanisms include `sys`, `builtins`, `_imp`, `_io`, `_thread`, `_warnings`, `_weakref`, `_codecs`, `marshal`, `posix`, `_sre`, `_collections`, `_functools`, `_operator`, `_abc`, `time`; this is an investigative list, not an expected inventory. | No accepted full table is supplied. `sys.builtin_module_names`, symbol presence or a locally emitted list cannot nominate its own expected set. A C implementation can be embedded or a shared extension according to the build. |
| Frozen modules | Bind every available frozen name, alias/package property, generated frozen byte array and build provenance to the actual binary container. Include bootstrap and error paths, not just application imports. | Frozen `__file__` may describe a source path without execution of those file bytes. Do not invent a per-module source file/hash for embedded code. Frozen profile changes require renewed target/manifest/launch acceptance. |
| importlib bootstrap | Review accepted `_frozen_importlib`, `_frozen_importlib_external`, `_imp`, import initialization, finder/loader setup and frozen/source fallbacks. Distinguish standard bootstrap finder machinery from forbidden custom hooks and zip-based application imports. | Availability of bootstrap support such as `zipimport` does not authorize a zip import root. Conversely a banned zip root does not prove the build can omit embedded bootstrap machinery. Full available embedded inventory remains required. |
| Import roots and path initialization | Review the exact patched getpath/pathconfig and frozen configuration, executable location, prefixes, compiled paths, `pyvenv.cfg` treatment, all ordered source/native roots and explicit source package parents, including `deployment/__init__.py`. | `-I -S -B` is a requested argv, not an authenticated `sys.path`. `-I` alone does not disable system site; `-B` only disables bytecode writes. Existing `.pyc`, default zip paths, CWD/home and host roots cannot silently supply missing code. A missing optional default path and a permitted load source are different facts. |
| ABI and extension suffixes | Pin SOABI, extension suffix order, architecture, Python ABI/build, compiler/configuration and import flags from accepted build materials. Compare actual finder behavior with V1's `.so` and `.<target ABI>.so` lexical model; `.abi3.so`/native-package shadows remain refused. | The reported `cpython-312-x86_64-linux-gnu` filename suffix is not independent ABI/build acceptance. Do not import `sysconfig`/execute the target to bootstrap the authority for its own expected ABI. Static config/header/build records are candidates for review. |
| `_datetime` | Current `_timestamp` uses numeric fields and `timezone.utc` (parser lines 154–170), with `ValueError` refusal. Review accepted `datetime.py`, `_pydatetime`, C initialization/registration and all methods reachable from the selected application. | A reported `PyInit__datetime` symbol is evidence to investigate, not proof of builtin registration or absence of a separate extension. If embedded, bind interpreter/libpython containers. Lazy `time`/`_strptime` paths must be checked; the current numeric constructor does not itself justify `strptime`, zoneinfo or timezone databases. |
| Native initialization | Review each selected module's `PyInit_*`, single/multiphase initialization, slots, Python-level import calls, C API/ABI requirements, global state and failure handling. | A `.so` path or `DT_NEEDED` table cannot show all imports/file opens made by initialization. Symbols, C source and accepted generated tables must agree; no inspected extension is imported in this analysis. |
| Dynamic loading | Review accepted CPython `dynload_shlib.c`, importdl and extension loader: exact pathname, calling context, flags, initializer lookup, dependency reuse and errors. Include native imports as `dlopen` behavior alongside any libc/OpenSSL activity. | Python import allowlisting does not constrain every C-level load. `ctypes`, plugins and extra namespaces cannot be admitted through an unspecified fallback. Computed or unbounded behavior blocks closure until constrained by an approved design. |
| Encodings and data | Inventory encoding bootstrap (`encodings`, aliases, UTF-8 and any actually reachable codec), filesystem/error output encodings, accepted locale coercion/UTF-8-mode behavior, C locale/gconv and any data files. Canonical JSON uses UTF-8 and UTF-16BE key encoding. | `LC_ALL=C` and a run without `locale-archive` mappings do not prove all required paths lack locale/gconv data. Determine which UTF codecs are C handlers versus Python modules in this exact build. No timezone/CA/NSS data is automatically required by the parser. |
| Configuration and environment | Account for actual runtime-consumed loader cache/preload, OpenSSL config/includes/provider paths, locale/NSS files, profile bytes, import root ancestry and fixed FD/CWD/env. Pin bytes or authenticated absence according to selected behavior. | `/etc/ld.so.conf` generally feeds cache generation; it is not interchangeable with a runtime-consumed cache. Static build records (`pyconfig.h`, `Makefile`, `_sysconfigdata*`) are evidence, not automatically runtime data files. Required runtime configuration must be selected explicitly. |
| Refusal and fatal errors | Review import failures, decoder/Unicode errors, crypto-provider/algorithm errors, memory/recursion/resource failures, exception formatting, shutdown and report serialization. Parser fixed-code `ContractRefusal` limits its own exposed text; it does not eliminate CPython's internal paths. | Examples needing investigation include `logging` imported by hashlib on initialization failures and traceback/linecache/tokenize/source-file reads during exception rendering. Do not add these wholesale or declare them safely removable; determine which paths are reachable or forced to terminate by the approved profile. |

Dataclass initialization in the upstream reference generates trusted methods
using `exec`. V1 prohibits execution of *inspected installation code* and custom
import hooks; it does not establish a blanket prohibition of CPython internal
code generation. A later sandbox policy must account for actual trusted-code
behavior without using this distinction to authorize untrusted `eval`/`exec`.
Postponed annotations also do not remove explicit runtime `typing` imports
or calls to `cast`. A type-only use does not cancel an unconditional import.

There is no independently accepted build yet against which a complete
builtin/frozen/source/native map can be marked CONFIRMED. The upstream
semantic source map is a research aid; acceptance of the Ubuntu-patched
sources, build recipes, generated tables, containers and data remains open.


## 7. ELF/loader plan and library queue


For every justified executable/native/shared/provider candidate, retain read-only inspection using trusted tools: `readelf -W -h -l -d -V --dyn-syms`, with build identity notes and `sha256sum`, `stat`, `readlink`/directory ancestry evidence. This is a future plan for operator review, not executed Dell commands here. Do not use `ldd`, execute a loader, import an extension or invoke a target executable.

| Concern | Required retained evidence / disposition |
| --- | --- |
| PT_INTERP | Record executable program-header bytes and exact absolute kernel-visible path, independently accepted loader bytes and authenticated mount/alias mapping; transformations require accepted reproducible recipe/provenance. A loader under `/observer` does not redirect `/lib64/...`. |
| DT_NEEDED | Preserve every occurrence and order for every reachable binary, recurse to fixed accepted inventory with identity/visited tracking. Preserve request token vs resolved object separately. Static executable zlib/expat needs survive module decisions. |
| SONAME | Record DT_SONAME, duplicates/same-SONAME alternatives, actual alias/member/canonical path identities; don't infer SONAME from basename or dpkg owner. Account for already-loaded object reuse/order. |
| RPATH / RUNPATH | Exact tags, strings, tokens and call context per object, inherited RPATH vs direct-only RUNPATH, accepted expansion/context; empty/relative/out-of-image or unsupported tokens block. |
| Symbols | Exact version requirements/definitions and dynamic symbols for every selected object, undefined Py* binding in accepted process, architecture/ABI compatibility, IFUNC/relocation/scope/load order. No-DT_NEEDED extension still relies on process ABI. |
| Resolution paths | Selected pinned loader build, cache/preload/default paths, effective mount namespace, search order and prior mapped objects, no external fallback, env/audit/filter restrictions. `/etc/ld.so.conf` inputs do not substitute for actual runtime cache semantics. |
| dlopen | Finite accepted calling-object/request/flags/namespace list covering CPython imports, native init, error/fallback paths and provider/config-driven loads. Computed dlopen completeness is not provable from string search or DT_NEEDED alone; unsupported paths block. |
| OpenSSL | Determine whether hashlib uses _hashlib/OpenSSL or accepted non-OpenSSL CPython alternative; pin config/provider initialization, default vs legacy selection, API-based loads, module paths/environment, provider artifact closure. No need for TLS implied by SHA-256. Existing config excerpt doesn't prove no legacy/API/alternate config load. |
| hwcaps | Pin CPU profile, loader hwcaps search/cache policy and exact variants, same SONAME behavior and IFUNC assumptions. Union of all variants doesn't establish chosen executed bytes. |
| Fallback | Enumerate Python implementation fallback, missing extension/module, provider/hash failure, cache/default/host path fallback and already-loaded substitution. Required missing accepted artifact must fail closed; cannot erase a requirement to fit DAG. |

Source anchors: ELF qualification lines 43–64, 75–88; Dell report 93–123, 169–185; Ubuntu report 262–305. Current untyped graph checks parser 1693–1709 include all declared binary roles and require an interpreter→loader edge, but do not authenticate PT_INTERP metadata, tag order/kind/context or complete closure. `_proposed_acyclic` at 1202–1218 remains unchanged. Do not implement SCC, delete edges, classify loading actor as reverse dependency, or prune a real dependency to obtain DAG. The existing 12-node/19-DT_NEEDED partial static DAG is conditional, not a statement about a complete selected runtime. Hypothetical libcrypto→legacy→libcrypto mixed cycle remains hypothetical until a required actual dynamic path is established. If a required real cycle is confirmed, retain BLOCKED and seek separate design decision; no unilateral model change.


### Libraries requiring further analysis


- **Current candidate executable startup closure:** pinned `ld-linux-x86-64.so.2`, `libc.so.6`, `libm.so.6`, `libz.so.1` resolved to exact libz.so.1.3 object, `libexpat.so.1` resolved to exact libexpat.so.1.9.1 object. Already reported library matches are useful but do not complete headers/symbols/path/config/lifetime context.
- **Conditional hash closure:** `_hashlib`→`libcrypto.so.3` plus all build/config-selected provider/data/dynamic closure. A non-OpenSSL CPython backend strategy could change this requirement only after source/build analysis and explicit design selection.
- **Conditional JSON/native support closure:** `_json`, math and actual selected `_datetime`/other native support containers, with any newly discovered DT_NEEDED; do not invent missing library names before static inspection.
- **Design-unjustified historical measurements:** `_ssl`, `libssl.so.3`, `legacy.so` are retained observations, not automatic minimal-image constituents. If a future required path justifies them, include their actual recursive/contextual closure without deleting back-edges.
- **Future verifier crypto:** choice of Ed25519 implementation may bring OpenSSL/libsodium/cryptography/cffi/Rust/etc. None is presently justified solely by parser imports; actual implementation/build choice and its closure are OBSERVER_DESIGN_UNDECIDED.
- **_gdbm path:** local gdbm owner doesn't justify `_gdbm` nor an assumed libgdbm/libgdbm_compat dependency in the target. Inspect actual ELF only if required or as bounded host inventory evidence; don't infer dependencies from package/name.


## 8. Missing architectural decisions


All recommendations below are proposals derived from sources, not approved
architectural decisions. Existing V1 restrictions are preserved.

| ID | Variants | Risk | Required evidence | Dependencies | Recommendation justified by current sources |
| --- | --- | --- | --- | --- | --- |
| D1 Runtime/application boundary | Reuse current parser+canonicalizer in observer; narrower separately reviewed observer entrypoint; keep parsing/crypto primarily in offline verifier/receiver. | Mistaking parser lower bound for observer closure; dragging large offline/harness TCB into observer; divergent contract semantics after rewriting. | Full required observer success/refusal/error/reporting behavior, chosen entrypoint source, call/import graphs; explicit component API boundary and compatibility review. | Drives all remaining decisions. | Preserve shared canonical JSON and separation of offline tools; do not import Windows release verifier or Source/CCA harness (plan 619,623–628). No source justifies choosing a completed observer import set now. |
| D2 Exact CPython build and layout | Independently selected Ubuntu 3.12 binary packages; independently reproducible custom/minimized 3.12 build. A different architecture/version lies outside current proposal. | Host packages define convenience, not accepted image; mixed binary/source package revisions; alternate build changes builtin/frozen/native split and ABI; deleting embedded modules cannot be modeled as dropping `.so`. | Accepted package roots/snapshot/DEBs or complete source/recipe/toolchain/patch/build outputs; build flags/Modules/Setup/generated inittab/frozen tables; binary/container hashes and exact image layout; external reviewer selection. | D1, D3–D10, external source/contract acceptance. | Prefer separate CPython 3.12, stdlib-only/source `.py` as current proposal (plan 202–205,464–468); do not recommend system-copy inventory as approval. |
| D3 Builtin/frozen/bootstrap and import roots | Exact available inventory of chosen build with BUILD_FIXED; source modules where selected build uses them; changed frozen/builtin build requires reviewed target revision. | Self-enumeration circularity; wrong frozen `__file__`; unmodeled startup module, path fallback, site/.pth/customize, existing `.pyc`; sys.path cannot be inferred solely from flags. | Accepted full tables and container bindings, pathconfig/getpath/importlib semantics and source/byte transforms for exact distro build, ordered roots and accepted source package parents; separately qualified negative resolution tests later. | D2, D8/D10 and later launcher design. | Preserve full exact available embedded inventory per plan 189–200, fixed `-I -S -B` / source `.py` / no zip or namespace/custom hooks. Static `.so` minimization does not reduce embedded inventory. |
| D4 Hash backend and providers | Exact selected build's hashlib/OpenSSL path including required provider/config; reviewed non-OpenSSL CPython backend/build; otherwise no unsupported fallback. | hashlib exposes backend imports beyond application's SHA-256; error/FIPS/provider behavior, legacy provider dynamic loading, hidden dependencies and mixed cycles; removing backend changes behavior. | Exact patched hashlib and C backend/build materials; required algorithm semantics/error paths; OpenSSL build and provider/config policy, complete DT_NEEDED/dlopen/symbol closure and deterministic resolution. | D1–D3, D7, D8. | Qualify SHA-256's real selected backend, do not assume `_ssl` or legacy provider is required because `_hashlib` uses libcrypto; do not claim safe exclusion of any provider without required/error-path evidence (ELF review 80–81,104–112). |
| D5 Native accelerators/fallbacks including datetime | Selected standard build's available accelerator+fallback imports; deliberately absent accelerator with separately reviewed source fallback/build; builtin `_datetime` vs native `_datetime` placement according to accepted tables. | ImportError fallback changes import closure and semantics; accelerator import happens despite narrow application use; `_datetime` symbol alone insufficient proof of builtin registration. | Patched exact stdlib sources and dependency/error map; native initializer tables/exports and binary bindings; both success/refusal and fallback obligations for selected profile. | D1–D4; architecture changes cannot be introduced as inventory cleanup. | Treat `_json`, `_datetime`, `_hashlib` etc as candidates pending exact build confirmation; keep `_datetime` container-bound if accepted materials prove builtin, not invent a missing .so (local report 52–54,75). |
| D6 Required observer OS operations | Narrow read-only fd/procfs/stat/pidfd primitives; selected reuse of reviewed probe helpers; no broad diagnostic or harness imports. | A broad helper pulls subprocess, ctypes, NSS/user lookup/plugins; insufficient privileges yield incomplete observations; adding privileges invalidates current V1 sandbox. | Complete allowed read operations, path/FD/syscall/error/permission/time/resource specification, source closure; prove faithful host scope and complete visibility under accepted minimal nonzero uid/gid/no capabilities. | D1, D2,D3,D8,D9. | Use numeric metadata and avoid account/DNS lookup if unneeded (plan 487–490). Missing permission/scope retains BLOCKED; no present source settles exact OS module list. |
| D7 ELF model compatibility | Demonstrate complete required runtime fits current DAG; later independently approved incompatible typed/bounded revision if a required counterexample proves necessary. | Omitting required edges/objects to force acyclicity; conflating PT_INTERP with SONAME or dlopen; assuming parser declaration equals loader resolution. | Complete exact required objects/edges/config/data, loader state/order/symbol and provider semantics; record actual cycle if any. | D2–D6,D8. | Retain existing DAG; implement no SCC; no dependency removal merely to pass graph (ELF review 8–20,31–64; parser 1202–1218,1680–1709). |
| D8 Startup, path mapping and lifetime protection | Separately reviewed authenticated image/namespace mapping with enforced appraisal/integrity; reproducibly transformed interpreter with accepted provenance when needed; static launcher still needs kernel trust. | Verified interpreter FD does not pin PT_INTERP or later opens; `/observer` copy does not change `/lib64` PT_INTERP; read-only bind mount and fs-verity alone insufficient; alias/path/CPU variant substitution. | Concrete executable→kernel interpreter mapping, closed resolution roots, pinned/absent cache/preload, immutable objects+ancestry, accepted profiles and kernel/launcher evidence over lifetime; no fallback. | D2–D7,D9; entire stage4 prerequisite. | Preserve fixed environment `LC_ALL=C`, explicit CWD/FD/import roots; defer actual mechanism choice to separately approved launcher stage (plan 316–325,470–510; ELF review 45–56). |
| D9 Error/reporting/time/nonce/state | Minimal bounded protocol and numeric UTC timestamps; future receiver/launcher owns nonce generation and durable single-use state; additional timezone/logging/traceback features require explicit justification. | Lazy imports and locale/NSS/encoding/cfg on refusals; traceback exception formatting imports; unauthenticated FD/receipt or clock/replay state; uncontrolled logging. | Success/refusal/error graph incl exceptions and CPython internal imports, FD3/FD4 framing/seals/origin, receiver state/time/checkpoint and failure handling; precise runtime data set. | D1–D8, separately reviewed receiver/trust protocols. | Preserve narrow fixed-code refusals and existing protocol; do not infer traceback/logging/zoneinfo/sqlite modules from design words. Contract audit P2-03/P2-04 and plan 321,327–328 require future enforcement evidence. |
| D10 External acceptance and measurement independence | Independently accepted Ubuntu build materials or custom build proofs plus external reviewer/key/checkpoint policy; local static observations only for comparison. | Matching 47 local paths or even selected archive byte matches becomes self-authenticating expected inventory; source count/package labels replace content proof. | Outside-rooted source/build/inventory/profile acceptance and revocation/freshness policy, exact raw observations/hash chains and independently repeated qualified artifact checks later. | All decisions; offline verifier and pre-Python launcher require their own TCB evidence. | Keep local observation/byte consistency/independent acceptance distinct; new 47 result narrows local ownership anomaly only, never runtime closure or attestation. |


## 9. Minimal next Dell measurements


1. Preserve original static inventory procedure and raw before/after captures, exact collection UTC/tools versions/digests and sorted 47-row names/absolute paths, module stems/suffixes, regular/symlink/type/size/identity/ownership/version/source/status. Retain which two original packages declared 46 paths; do not invent their identities from totals. Bind `_gdbm` new row and package file-list discrepancy explanation; state local path/ownership equality only. Hash selected source/config files and capture read stability; endpoint equality alone not lifetime proof.
2. For already justified first-tier candidates, retain complete static ELF outputs + exact file/alias identities: interpreter and loader plus startup library closure; math and source-selected native/builtin containers; `_hashlib`/`_json` if source establishes candidacy. Needn't fully qualify all 47 before deciding actual required paths; raw per-file inventory can still be collected once. Separately acquire all 47 digests/DEB member comparisons only if claiming all 47 byte identity later, not as implicit existing result.
3. Acquire independently accepted source/build materials (`Modules/config.c`, frozen tables/generated inputs, Modules/Setup*, generated pyconfig/SOABI/suffix settings, build recipe/patches/toolchain and outputs) for exact Ubuntu/custom build; use trusted archive extraction/static inspection, not target `python -c`/`sys` inventory. Preserve accepted stdlib Python source hashes for full transitive map. Symbol table checks of target can corroborate `_datetime` but can't settle registration alone.
4. Preserve signed archive chain and exact member/path/size/hash mappings for any **newly selected** package/extension; if _gdbm is only inventory evidence, flag its package chain pending rather than requiring it in image. Acceptance of Ubuntu archive roots/freshness must be independent; do not mix old python3.12 revision with separate stdlib-extensions version as a claim of shared build identity.
5. Capture static runtime-consumed config/absence, loader/cache/preload view and OpenSSL config/provider path evidence only as inputs to chosen profile, plus CPU/hwcaps identity needed for resolution analysis. Avoid replacing preserved historical APT metadata with fresh data as if it retroactively authenticated old collection.

These measurements do not close effective launch/run/host-scope, real resolution, lifetime protection or runtime attestation. Later physical/runtime qualification needs separate authorization; no such run is requested here.


## 10. Priority register and impact on Stage 3A


Priority labels describe blockers/evidence tasks, not a new production defect in
an intentionally inactive parser.

| ID | Severity | Finding and source | Disposition / impact |
| --- | --- | --- | --- |
| P0 | No newly confirmed P0 | No new P0 defect is established by this bounded source review; this is not proof of absence. | Preserve every existing refusal/activation block. No Observer or Stage 3B launch follows from this report. |
| MR-P1-01 | P1 | No complete, source-required Observer entrypoint or operation set: parser only; D1/D6. | Blocks final minimal module/runtime selection and CA-P1-03 closure. |
| MR-P1-02 | P1 | Exact independently accepted CPython builtin/frozen/native/source inventory, ABI/pathconfig/containers not established; plan 189–200, audit CA-P1-01/03. | 47 system extensions do not close it; static accepted build-material review required before final selection. |
| MR-P1-03 | P1 | Hash/backend provider/error path and native initialization closure unresolved; ELF 50–53, audit 155–158. | Blocks exact needed libraries/config/data and DAG compatibility. |
| MR-P1-04 | P1 | Complete required ELF graph/loader-resolution semantics and fit in retained DAG unresolved; ELF 8–20,58–64. | No SCC implementation or dependency deletion; preserve CA-P1-01. |
| MR-P1-05 | P1 | Pre-Python trust/lifetime integrity, initial host namespace/scope/attestation absent; plan 682–689,706–712. | Unchanged runtime_attestation=MISSING and host_pid_namespace=UNVERIFIED. Out of scope to implement or remeasure physically. |
| MR-P1-06 | P1 | Independent reviewer/build/keys/freshness acceptance and future offline crypto verification not supplied (audit CA-P1-02; schema 467–479). | Manifest SHA-256 is document identity only; CONTRACT_APPROVED remains NO. |
| MR-P2-01 | P2 | 47 names, exact collection commands/raw transcripts, package path/member sets and per-file bytes not supplied in new account. | Request minimal static evidence on Dell; explicitly no 47/47 byte claim. Status of `_gdbm` changes local ownership anomaly only. |
| MR-P2-02 | P2 | Error/fallback/import/init/resource/data paths require exact-source reconciliation and later separately authorized accepted-runtime tests (audit CA-P2-03). | Plan focused static material collection first; do not invoke 3B. |
| MR-P2-03 | P2 | Offline canonical/raw binding difference and replay/time/receipt protocol remain CA-P2-01/02/04. | Refer to existing audit, do not alter parser semantics or silently promote future states. |
| MR-P3-01 | P3 | Traceable source locations/versioned report/evidence manifest and scope vocabulary needed. | Add report and new separate SHA-256 manifest; historical manifests unchanged. No status promotion. |

All requested final status values remain exactly:

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

The reported 47/47 local path/package ownership result addresses only the
specific previously unexplained `_gdbm` local package owner. It does not close
CA-P1-01 (independent exact inventory/resolution), CA-P1-03 (full application /
bootstrap / data / ELF / lifetime closure), CA-P1-02 (external acceptance), or
any Stage4 runtime/scope/attestation gate. The selected previous 12 ELF byte
matches remain a different evidence set, not all 47 modules or complete runtime.


All historical P1 findings remain open: CA-P1-01 (accepted exact build and
resolution), CA-P1-02 (external roots/reviewers/real signature/freshness),
CA-P1-03 (closure and enforcement), DELL-P1-01 (expected image) and DELL-P1-02
(host/run/launcher). The new account explains the separately reported local
`_gdbm` ownership discrepancy. DELL-P2-01 remains open for its original
path/alias/registration/provider/configuration/capture-evidence gaps.
Historical CA-P2-01–04, DELL-P2-02 and P3 trust/immutability distinctions
are unchanged. No inferred severity downgrade or Stage 3A completion is made.

## 11. Integrity and validation

The new [SHA-256 manifest](evidence/stage9_observer_3a_minimal_cpython_v1.sha256)
pins this report, exact input, semantic source map, validation note, reviewed
repository sources and the three unchanged historical manifests. It excludes
itself. These are document/source identities, not a fabricated runtime inventory
or authentication of absent operator-local binaries. Historical manifests are
neither edited nor regenerated.

See the [validation note](evidence/stage9_observer_3a_minimal_cpython_validation_v1.md)
for actual commands and limits. The mandatory exact all-files formatting hook,
Tier A Ruff and the **869 existing synthetic tests passed**. The full local
Betterleaks scan returned one pre-existing ignored compiled-test fixture
finding, so that full scan is **not PASS**. Final documentation-path scanning
and integrity results are recorded separately there. These checks establish
source/document consistency and parser regression behavior; none qualifies
the Dell runtime or provides independent acceptance.

This packet supports a separate documentation PR because it adds concrete
source-backed import-time, codec and native/build-placement findings, and a
bounded next-evidence plan. Hosted checks must be evaluated for that PR's exact
HEAD; historical CI, queued/running jobs and successful local tests are not
hosted PASS evidence. The known full-local-scan finding remains explicitly
reported for reviewer disposition. No merge or automatic merge is performed.
Recommendations in §8 remain proposals for independent review.
