# Stage 9 / 3A — exact Ubuntu CPython build material reconciliation V1

Review date **2026-10-29 UTC**; repository baseline `main`
`cad53867231a1a6e49ea770c07043b417af84449`. Scope: Ubuntu source package
`python3.12 3.12.3-1ubuntu0.17`, binary architecture `amd64`. This is a static
package/source/ELF review. No extracted interpreter, native extension, package
maintainer script, production Observer or Dell program was executed.

## Evidence boundary and exact materials

`VERIFIED_FROM_EXACT_BUILD`: the [archive source review](evidence/stage9_observer_3a_cpython_archive_sources_v1.md)
and [verification records](evidence/stage9_observer_3a_cpython_archive_verification_v1.json)
record a new independent cloud acquisition: current Ubuntu `noble-updates`
InRelease signature, four compressed/uncompressed index hash/size checks,
four exact-version DEBs, and the exact `.dsc`, orig and Debian tar archives.
The signing fingerprint is externally corroborated; the project has not
approved that root/channel as an Observer acceptance authority. This label
means the identified exact-version archive bytes were inspected under that
explicit trust boundary. It does not claim reproducible compilation, project
approval, Dell collection authenticity or actual runtime selection.

The three runtime DEB hashes match the earlier operator report. The additional
`libpython3.12-dev` DEB is analysis material, not a required runtime dependency.
Current archive metadata is separate from the operator's historical snapshot.
The operator's triage tar is `OPERATOR_REPORTED_AND_CHAT_REVIEWED`; it is not
the source of the independent package inspection here.

`VERIFIED_FROM_EXACT_BUILD`: source extraction used host `dpkg-source
--no-check -x python3.12_3.12.3-1ubuntu0.17.dsc`, after the outer signed Sources
chain's file hash/size checks. `--no-check` does not claim verification of the
uploader's `.dsc` signature. Quilt applied **82 active patches**, in exact
`debian/patches/series` order. The [member inventory](evidence/stage9_observer_3a_cpython_build_members_v1.json)
retains SHA-256 and sizes of selected extracted/generated build configuration,
reviewed C/Python source, and every active patch with affected paths. Source
paths below refer to that applied tree; package member paths refer to the
exact DEB payload, not the cloud host's installed Python.

| Material and exact location | Evidence and use |
| --- | --- |
| `libpython3.12-dev`: `/usr/lib/python3.12/config-3.12-x86_64-linux-gnu/config.c` | `VERIFIED_FROM_EXACT_BUILD`; generated `_PyImport_Inittab`, not the upstream template; SHA-256 `af12a9a892f0bfed75bb53877dbe6353150dc276c5b20a97d01e4bdf665058be`. |
| Same directory: `Setup`, `Setup.local`, `Setup.bootstrap`, `Setup.stdlib`, `makesetup` | `VERIFIED_FROM_EXACT_BUILD`; actual shipped configuration; `Setup.local` and bootstrap static entries take precedence over shared stdlib candidates. |
| Same directory: `Makefile`; minimal package `_sysconfigdata__x86_64-linux-gnu.py` | `VERIFIED_FROM_EXACT_BUILD`; installed extension-development configuration, intentionally taken from the shared build and sanitized. Neither is a complete command log for `/usr/bin/python3.12`. |
| `libpython3.12-dev`: `/usr/include/x86_64-linux-gnu/python3.12/pyconfig.h` | `VERIFIED_FROM_EXACT_BUILD`; installed static-build header selected by packaging rules, distinct from the shared Makefile configuration. |
| Applied `Python/frozen.c`, `Tools/build/freeze_modules.py`, `Tools/build/deepfreeze.py` | `VERIFIED_FROM_EXACT_BUILD` source material; describes frozen generation and registration. Actual executable table entries are separately `VERIFIED_FROM_STATIC_ELF`. |
| Generated `Python/frozen_modules/*.h`, `Python/deepfreeze/*.c` | `BUILD_SPECIFIC_UNVERIFIED` as retained generated build-source files: not shipped in these source archives, whose directories contain README files. The compiled frozen table/data in the exact executable can still be inspected statically. |
| `debian/rules`, patch series and changelog | `VERIFIED_FROM_EXACT_BUILD`; distribution configuration and patch selection, rather than upstream 3.12.3 defaults. |

## Confirmed builtin registrations

`VERIFIED_FROM_EXACT_BUILD` and `VERIFIED_FROM_STATIC_ELF`: **60 named entries**
in the generated `config.c` table match the **60 entries** statically decoded
from the exact executable's initial `PyImport_Inittab` pointer/table. See
[ELF table evidence](evidence/stage9_observer_3a_cpython_static_tables_v1.json).
There are **58 non-null initializer callbacks** and the two special entries
`builtins` and `sys` with null callbacks. The final `{0, 0}` is a sentinel.

```text
_abc _ast _bisect _blake2 _codecs _collections _csv _datetime
_elementtree _functools _heapq _imp _io _locale _md5 _opcode
_operator _pickle _posixsubprocess _random _sha1 _sha2 _sha3 _signal
_socket _sre _stat _statistics _string _struct _symtable _thread
_tokenize _tracemalloc _typing _warnings _weakref
array atexit binascii builtins cmath errno faulthandler fcntl gc grp
itertools marshal math posix pwd pyexpat select spwd sys syslog time
unicodedata zlib
```

The **56 `PyInit_*` symbols** are consistent with these entries but do not
establish registration: `marshal` uses `PyMarshal_Init`, `_warnings` uses
`_PyWarnings_Init`, and the two special entries have no initializer. The proof
here is the name-to-callback registration table and matching executable table,
not symbol presence. `math`, `_datetime`, `_typing`, `_opcode`, all builtin
hash implementations, `zlib` and `pyexpat` are consequently not inferred `.so`
requirements for this executable. Builtin C code can still call system APIs
and create data or ELF dependencies; builtin does not mean dependency-free.

`VERIFIED_FROM_EXACT_BUILD`: `debian/rules:535`–`546` builds `Setup.local` from
the distribution minimal-module inventory, then runs `makesetup` in this order:
`Setup.local`, `Setup.bootstrap`, `Setup.stdlib`, upstream `Setup`. The shipped
`Setup.local` has **29 static extension entries**, followed by **22 bootstrap
entries** and **9 template/special entries** in `config.c`. The generated
Makefile's `MODBUILT_NAMES` includes shared build products too; it must not be
used alone as the builtin inventory. `Setup.stdlib`'s `*shared*` candidates also
do not override earlier static registrations.

## Confirmed frozen registrations and generated importer data

`VERIFIED_FROM_STATIC_ELF`: the exact executable contains these null-terminated
frozen tables, decoded through the exported table pointers with file-backed
`PT_LOAD` virtual-address mapping and the exact `_frozen` layout. The retained
JSON records names, code pointers, sizes, package flags and `get_code` pointers.
`Python/frozen.c:97`–`142` in the applied Ubuntu source independently agrees:

| Table | Confirmed names |
| --- | --- |
| Bootstrap, 3 | `_frozen_importlib`, `_frozen_importlib_external`, `zipimport` |
| Stdlib, 15 | `abc`, `codecs`, `io`, `_collections_abc`, `_sitebuiltins`, `genericpath`, `ntpath`, `posixpath`, `os.path`, `os`, `site`, `stat`, `importlib.util`, `importlib.machinery`, `runpy` |
| Test, 11 | `__hello__`, `__hello_alias__`, `__phello_alias__`, `__phello_alias__.spam`, `__phello__`, `__phello__.__init__`, `__phello__.ham`, `__phello__.ham.__init__`, `__phello__.ham.eggs`, `__phello__.spam`, `__hello_only__` |

`VERIFIED_FROM_EXACT_BUILD`: applied `Python/frozen.c:144`–`160` supplies alias
semantics, including `_frozen_importlib` → `importlib._bootstrap`, external
bootstrap → `importlib._bootstrap_external`, and `os.path` → `posixpath`.
The aliases are source-verified here; no independent exported ELF alias table
was available. Test table presence does not make the tests required Observer
imports. `encodings`, `json`, `hashlib`, `datetime`, `typing`, `dataclasses` and
`logging` are not in these frozen tables; their exact `.py` files and native
backends must be considered separately.

`VERIFIED_FROM_EXACT_BUILD`: `Python/initconfig.c:768`–`772` selects frozen
stdlib by default in a non-debug installed build; `-X frozen_modules`, build
mode and configuration can change selection (`Python/import.c:1822`–`1837`).
`OBSERVER_DESIGN_UNDECIDED`: the launch configuration and acceptance of any
frozen-disabled source fallback. Table availability proves embedding, not the
actual launch's import resolution. Importer bootstrap still requires separate
startup/configuration analysis, including external encoding data.

## ABI, extension loading and build-variant reconciliation

| Finding | Status and interpretation |
| --- | --- |
| `SOABI=cpython-312-x86_64-linux-gnu`, `EXT_SUFFIX=.cpython-312-x86_64-linux-gnu.so`, `MULTIARCH=x86_64-linux-gnu`, empty `ABIFLAGS` | `VERIFIED_FROM_EXACT_BUILD`, shipped Makefile lines 43, 134, 154, 168 and sysconfig data; actual installed extension filenames agree. |
| Accepted suffix candidates | `VERIFIED_FROM_EXACT_BUILD` for applied `Python/dynload_shlib.c:37`–`51`: `.cpython-312-x86_64-linux-gnu.so`, `.abi3.so`, `.so` in source order given this SOABI and undefined `ALT_SOABI`. `BUILD_SPECIFIC_UNVERIFIED` for independent decoding of the executable's suffix table, which is not exported in the dynamic symbol table. A manifest must bind the permitted paths/suffixes rather than accept every candidate automatically. |
| Installed header | `VERIFIED_FROM_EXACT_BUILD`: `Py_DEBUG` and `Py_ENABLE_SHARED` undefined; `HAVE_DLOPEN`, `HAVE_DYNAMIC_LOADING`, `HAVE_LIBDL`, `HAVE_DECL_RTLD_NOW` defined; 8-byte pointers/long/size_t/time_t; `WITH_PYMALLOC`, `WITH_FREELISTS`, `WITH_DECIMAL_CONTEXTVAR` defined. `HAVE_LIBDL` does not imply an independent `libdl.so` `DT_NEEDED`. |
| Shared build configuration | `VERIFIED_FROM_EXACT_BUILD`: shipped `CONFIG_ARGS` includes `--enable-shared`; `PY_ENABLE_SHARED=1`. `debian/rules:896`–`907` installs the static interpreter/header but replaces sysconfig data using the shared build; lines 993–996 replace the installed Makefile using the shared build. This is intentional distribution packaging, not evidence of executable `DT_NEEDED=libpython`. |
| Static interpreter versus static ELF | `VERIFIED_FROM_STATIC_ELF`: actual `/usr/bin/python3.12` is dynamically linked `ET_EXEC` with four `DT_NEEDED` names and no shared libpython edge. The packaging term `build-static` means embedded libpython; it does not mean an ELF with no dynamic loader. `debian/rules:137`, 442–454 and 896–907 agree with that distinction. |
| Available native modules | `VERIFIED_FROM_EXACT_BUILD`: the selected minimal+stdlib DEBs contain **46 regular `lib-dynload/*.so` module files**. This is a scoped package inventory, not an assertion about the Dell's entire 47-library disk inventory or the Observer's required set. See the separate import-closure map before selecting modules. |
| Native import mechanics | `VERIFIED_FROM_EXACT_BUILD`: `Python/dynload_shlib.c:55`–`112` uses `dlopen(pathname, interpreter flags)` then `dlsym` for the initializer. `Include/internal/pycore_import.h:84`–`97` selects `RTLD_NOW` when available, and the installed header declares it. Runtime mutation of flags remains possible. Absence of executable `DT_BIND_NOW` does not prove extension imports use lazy relocation. |

## Distribution compiler/linker configuration and patches

`VERIFIED_FROM_EXACT_BUILD`: common configure arguments in `debian/rules:368`–
`380` include `/usr`, multiarch libdir, IPv6, loadable SQLite extensions,
`bdb:gdbm`, computed gotos, system expat, DTrace, OpenSSL default suites,
`--without-ensurepip`, and the distribution wheel directory. Availability of
SQLite extension loading is a potential path if SQLite is later selected; the
present contract does not require importing every configured module.

`VERIFIED_FROM_EXACT_BUILD`: installed development flags include
`-D_FORTIFY_SOURCE=3`, `-fstack-protector` (not `-strong`),
`-fstack-clash-protection`, `-fcf-protection`, frame pointers,
`-Werror=format-security`, hidden C visibility, `-Wl,-Bsymbolic-functions`,
`-Wl,-z,relro`, and export-dynamic linkage. These are observed strings in
sanitized development configuration, not a complete final interpreter command.

`VERIFIED_FROM_EXACT_BUILD`: Ubuntu's build recipe enables PGO for native
amd64/Linux and LTO for amd64 unless build options disable them
(`debian/rules:210`–`245`), while `sysconfig_substflags` deliberately strips
LTO/profile-use flags and changes `-O3` to `-O2` in installed configuration
(lines 250–261). It also sets `hardening=-pie` in its buildflags recipes
(lines 83–90). `BUILD_SPECIFIC_UNVERIFIED`: without the actual build log,
`.buildinfo`, build environment/options and reproduction, do not assert exact
PGO/LTO invocation, compiler package revision or linker command from these
recipes. Actual ELF properties take precedence over intended flags.

`VERIFIED_FROM_STATIC_ELF`: the exact executable embeds `[GCC 13.3.0]`,
`3.12.3`, and `Aug 31 2026` strings. They are build-information literals, not
an authenticated compiler log or a physical measurement timestamp. The
`reproducible-buildinfo.diff` patch and `debian/rules:37`–`39` explicitly derive
the embedded date/time from the changelog, explaining why it need not equal
the archive publication date. The executable has no `.comment` section to
provide an additional compiler record.

`VERIFIED_FROM_EXACT_BUILD`: all **82 active patches** were accounted for by
name, size, SHA-256 and touched paths. Relevant reviewed differences include:

- `setup-modules.diff`: distribution system-library linkage entries, including
  expat, OpenSSL and builtin-hash implementations selected by `Setup.local`.
- `link-opt.diff`: Linux shared-link/export flags `-O1 -Bsymbolic-functions`.
- `multiarch.diff`, `destshared-location.diff`, `sysconfigdata-name.diff`,
  `sysconfig-debian-schemes.diff`: import/install/configuration layout changes.
- `reproducible-buildinfo.diff`, `profiled-build.diff`: build-info and profile
  recipe behavior.
- `CVE-2026-2297.patch`: changes `Lib/importlib/_bootstrap_external.py` so
  sourceless bytecode loading uses `io.open_code`; relevant to allowed data
  paths and integrity even though its frozen generated result needs a build
  reproduction to map back to exact source bytes.
- The `.17` changelog adds `CVE-2026-4360.patch` and
  `CVE-2026-15308.patch` for tarfile/HTML parser. It is not equivalent to
  pristine upstream CPython 3.12.3. The complete selected series remains in
  the evidence inventory rather than being reduced to these examples.

## Build identity: what is established and what is missing

`VERIFIED_FROM_STATIC_ELF`: exact signed-index-bound binary package extraction
independently produces executable SHA-256
`e50d468e8b0adfb05733f5b87b3cff34829c4a8c1aea50c865aa8bdfe4bb150f`
and Build ID `337d65cf00021797985cc9f77c0cc334a9fbeb38`. The independently
extracted `_hashlib` and `_json` SHA-256 and Build IDs also equal the supplied
operator values. This ties the downloaded Ubuntu members to the reported
object identities, conditional on the report; it does not independently
observe the Dell filesystem or authenticate the absent triage archive.

`BUILD_SPECIFIC_UNVERIFIED`: `.buildinfo`, build logs, authenticated exact
compiler/binutils/dependency inputs, generated frozen/deepfreeze source,
complete compilation/link command lines, and a reproducible rebuild matching
the executable are not retained. A Build ID identifies an ELF note; by itself
it neither authenticates source nor proves that two different payloads could
not carry the same chosen note. Full SHA-256 byte equality is the stronger
comparison here. Source and binary package co-membership of the same published
source version is established under the archive chain; deterministic
source-to-binary derivation is a separate unresolved proof.

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
