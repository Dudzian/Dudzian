# Stage 9 / 3A: Dell Ubuntu 24.04 read-only ELF observations

Status: **PROPOSED_UNAPPROVED**. Prepared 2026-10-28 against contract source
`ae712aea85e4cbc372bdf226d6a5bc5be02f4949`. This procedure has **not** been run
on the Dell Latitude 5490. Only the physical operator may decide to run it.
It does not launch the Observer, a historical installation, a loader, a helper
or a production test. It cannot qualify the host or complete Stage 3A.

## Operator boundary

Use an existing ordinary user session on the Dell. No command below needs
`sudo`, installs a package, changes permissions, writes an output file, changes
the repository or edits host configuration. Output goes to the terminal. Do
not add redirects, `tee`, package-manager operations or execution of files
discovered by inspection. If a tool is absent, a file is unreadable, a command
times out or a limit is exceeded, record that observation as incomplete and
stop that part; do not repair the system during collection.

The operator must already trust the installed diagnostic utilities and the
session used to invoke them. Static inspection runs these utilities, never the
inspected ELF. Cleaning command environment variables does not authenticate the
utility, its own loader, `/etc/ld.so.preload`, the kernel or an already compromised
host. Local digests cannot establish that initial trust. A host of uncertain
integrity needs separately authorized evidence collection from a trusted system.
Do not substitute tools from a repository, virtual environment, working
directory or historical installation.

Use a trusted interactive Bash session with startup-file effects, aliases,
functions, traps and shell options already understood and trusted. Environment
cleaning cannot undo code already loaded or executed in that shell. Do not
start an untrusted shell or try to repair a suspect session with this procedure.

Before any external command, define this wrapper in that same trusted Bash
session. These are shell-local settings and a function definition, not host
configuration or a separate shell launch:

```bash
PATH=/usr/bin:/bin
LC_ALL=C

diagnostic_command() (
  builtin exec -c /usr/bin/env -i PATH=/usr/bin:/bin LC_ALL=C "$@"
)
```

The Bash **builtin `exec -c`** clears the environment passed to `/usr/bin/env`
before that executable's dynamic loader starts. A bare `/usr/bin/env -i ...`
would clean only the environment of its later child: inherited `LD_PRELOAD`,
`LD_AUDIT` or other loader settings could already affect `env` itself. An
absolute executable path alone does not prevent that. The subshell function
keeps the operator's existing shell alive and returns the tool's failure status.

The only environment entries deliberately passed onward are the literal
`PATH=/usr/bin:/bin` and `LC_ALL=C`. This removes inherited loader controls
(`LD_*`, including tracing/search/preload/audit controls, `GLIBC_TUNABLES`,
`GCONV_PATH`, `LOCPATH`), Python settings (`PYTHON*`, including `PYTHONPATH`,
`PYTHONHOME`, `PYTHONSTARTUP` and `PYTHONINSPECT`), shell-startup/exported-function
entries (`BASH_ENV`, `ENV`, exported functions), locale and tool-specific
settings (for example `LANG`, other `LC_*` and `DPKG_*`). This is an environment
allowlist, not a claim that a finite variable denylist covers every loader or
tool. No inherited export needs to be unset or printed, including readonly
variables in the current shell.

Use `diagnostic_command` for **every external invocation below**, including
inside `$(...)`. Commands launched by the fixed system `timeout` inherit that
same environment. Shell builtins, tests, assignments, globbing and function
calls run in the already trusted Bash; they do not start diagnostic executables.
Do not run later blocks in another session without this definition, replace the
wrapper with bare `env -i`, or add inherited environment entries. Only use the
fixed trusted system tools shown here; the optional Python command retains its
separate trust and operator-decision prerequisites.

This does not authenticate Bash, `env`, the tools, their libraries or the kernel.
It does not neutralize system-level `/etc/ld.so.preload`, loader cache/config
changes, a previously compromised shell or file substitution. All initial trust
requirements above remain mandatory; no diagnostic result creates trust.

Verify the named tools exist without installing missing ones:

```bash
for tool in /usr/bin/env /usr/bin/timeout /usr/bin/readelf /usr/bin/sha256sum \
  /usr/bin/stat /usr/bin/readlink /usr/bin/dpkg /usr/bin/dpkg-query \
  /usr/bin/uname /usr/bin/cat /usr/bin/sed /usr/bin/find /usr/bin/sort; do
  if [ -x "$tool" ]; then printf 'AVAILABLE %s\n' "$tool";
  else printf 'MISSING %s\n' "$tool"; fi
done
```

The resource limits here are collection limits, not approved Observer profile
limits. `timeout` bounds elapsed time; the file-size/count checks below bound
the chosen inputs. They do not promise a hard memory bound for every installed
version of an ELF parser. Use only the explicit system-package files below;
do not feed an arbitrary submitted binary to these commands. Do not use `ldd`,
`LD_TRACE_LOADED_OBJECTS`, `ld-linux --list`, Python imports or a discovered
executable to infer closure.

## 1. Host and installed package identity

```bash
diagnostic_command /usr/bin/uname -m
diagnostic_command /usr/bin/dpkg --print-architecture
diagnostic_command /usr/bin/cat /etc/os-release
diagnostic_command /usr/bin/dpkg-query \
  -W -f='${binary:Package}\t${Version}\t${Architecture}\t${db:Status-Abbrev}\n' \
  'python3.12*' 'libpython3.12*' 'libc6*' 'binutils*'
```

Expected scope is `x86_64`, Debian architecture `amd64`, Ubuntu `24.04` and
installed (`ii`) packages. An unexpected value is a scope mismatch; do not
coerce it. `dpkg-query` is local database evidence, not an authenticated package
receipt. No matches for an optional package is an observation, not permission
to install it. Record the complete Ubuntu package revision, including epoch
and Ubuntu suffix, separately from upstream CPython `3.12.x`.

Identify the fixed system interpreter without PATH-based Python selection:

```bash
diagnostic_command /usr/bin/stat \
  -c 'LINK %n type=%F uid=%u gid=%g mode=%a size=%s mtime=%y ctime=%z' \
  /usr/bin/python3.12
diagnostic_command /usr/bin/readlink -e /usr/bin/python3.12
diagnostic_command /usr/bin/dpkg-query -S /usr/bin/python3.12
```

Stop if this is missing, resolves to a historical installation, a user-writable
tree, a virtual environment, `/usr/local`, or an unexpected package. A symlink
in the host package layout may be observed here; it is **not** authorization
to include a symlink in the proposed image, whose file contract forbids it.
For each returned canonical path, record package ownership again for that exact
path. Do not silently replace the intended Observer interpreter with the host
interpreter: its eventual approved image/build may be a different artifact.

## 2. Static file inspector

Paste this function into the trusted session. Defining it has no filesystem
effect. It inspects only one explicitly reviewed absolute path at a time;
there is no dependency execution or automatic resolution. It refuses a target
over 64 MiB and uses a 10-second limit per expensive command. A refusal or
timeout is missing evidence, never a successful empty result.

```bash
inspect_system_elf() {
  local target_path canonical_path file_size
  target_path=$1
  case "$target_path" in
    /usr/bin/python3.12|/usr/lib/*|/lib/*|/lib64/*) ;;
    *) printf 'REFUSED scope %s\n' "$target_path"; return 2 ;;
  esac
  canonical_path=$(diagnostic_command /usr/bin/readlink -e -- "$target_path") || return 2
  case "$canonical_path" in
    /usr/bin/python3.12|/usr/lib/*) ;;
    *) printf 'REFUSED canonical scope %s\n' "$canonical_path"; return 2 ;;
  esac
  if [ ! -f "$canonical_path" ] || [ ! -r "$canonical_path" ]; then
    printf 'INCOMPLETE unreadable/nonregular %s\n' "$canonical_path"; return 2
  fi
  file_size=$(diagnostic_command /usr/bin/stat -Lc '%s' -- "$canonical_path") || return 2
  case "$file_size" in ''|*[!0-9]*) return 2 ;; esac
  if [ "$file_size" -gt 67108864 ]; then
    printf 'INCOMPLETE size limit %s\n' "$canonical_path"; return 2
  fi
  printf '\nOBSERVED target=%s canonical=%s\n' "$target_path" "$canonical_path"
  diagnostic_command /usr/bin/dpkg-query -S "$canonical_path" || return 2
  diagnostic_command /usr/bin/stat -Lc 'BEFORE dev=%d ino=%i size=%s uid=%u gid=%g mode=%a mtime=%y ctime=%z' \
    -- "$canonical_path" || return 2
  diagnostic_command /usr/bin/timeout 10s \
    /usr/bin/sha256sum -- "$canonical_path" || return 2
  diagnostic_command /usr/bin/timeout 10s \
    /usr/bin/readelf -W -h -l -d -n -V -- "$canonical_path" || return 2
  diagnostic_command /usr/bin/timeout 10s \
    /usr/bin/sha256sum -- "$canonical_path" || return 2
  diagnostic_command /usr/bin/stat -Lc 'AFTER dev=%d ino=%i size=%s uid=%u gid=%g mode=%a mtime=%y ctime=%z' \
    -- "$canonical_path" || return 2
}

inspect_system_elf /usr/bin/python3.12
```

Compare both hashes and all BEFORE/AFTER identity fields. Any change invalidates
that observation. Equal snapshots reduce accidental drift; they cannot prove
absence of a malicious swap between separate opens or authenticate a running
process. The function does not inspect every ancestor's ownership, authenticate
package content or create a TOCTOU-resistant file binding.

Read these `readelf` fields without reinterpreting them as actual resolution:

| Metadata | Meaning of this observation |
| --- | --- |
| ELF class, machine, type | ABI/container metadata; inspect every later candidate too. |
| `INTERP` / “Requesting program interpreter” | Kernel-selected loader path requested by this executable. It is not an ordinary SONAME search and not an extra `DT_NEEDED`. |
| Every `NEEDED` entry | Ordered direct request made by this object. Retain exact request bytes and requester identity. A request is not a resolved file. |
| `RPATH`, `RUNPATH`, flags, `SONAME` | Per-object search/identity inputs; retain absence explicitly. RUNPATH is direct-dependency scope, not inherited transitively as RPATH can be. |
| Build ID | Toolchain identifier in ELF notes; neither a digest nor authentication. |
| Version needs/definitions | Symbol-version compatibility input; matching SONAME and architecture alone is insufficient. |

Copy the literal `PT_INTERP` path only after reviewing it as an expected system
path, then call `inspect_system_elf` with that literal. On a conventional amd64
Ubuntu installation the request is often `/lib64/ld-linux-x86-64.so.2`; this is
an example, **not** an approved expected value. Inspect the loader's own dynamic
section even if it has no `NEEDED`. Also inspect any `libpython3.12.so.*`
actually requested by this binary; an installed libpython package does not
prove this interpreter uses it, and libpython must not be invented when absent.

For every object's direct `NEEDED`, enumerate its candidate package files,
review the candidate paths, then call the function on each relevant exact
file. Repeat until each newly observed object has been inspected once; keep a
visited set by canonical path **and digest**. Preserve shared edges and cycles
instead of dropping repeated destinations. A finite traversal is not proof
that the candidate chosen is the loader's actual choice. For example, finding
`libc.so.6 -> ld-linux-x86-64.so.2` and another reference to that loader does
not justify a fictitious reverse edge. Report the actual tags for each object.

Do not resolve a SONAME by choosing the first result from `find`, by the host's
current cache alone, or by executing the loader. Unknown/ambiguous resolution
stays `UNVERIFIED`. Record missing candidates and search inputs; do not add
host paths to an image allowlist to make closure appear complete.

## 3. Loader configuration and hwcaps observations

Read configuration as data; never `source` it. Inspect only existing regular,
readable system files of at most 1 MiB with the following function:

```bash
inspect_loader_text() {
  local config_path config_size
  config_path=$1
  case "$config_path" in
    /etc/ld.so.conf|/etc/ld.so.preload|/etc/ld.so.conf.d/*.conf) ;;
    *) printf 'REFUSED config scope\n'; return 2 ;;
  esac
  if [ ! -e "$config_path" ]; then printf 'ABSENT %s\n' "$config_path"; return; fi
  if [ -L "$config_path" ] || [ ! -f "$config_path" ] || [ ! -r "$config_path" ]; then
    printf 'INCOMPLETE config type/readability %s\n' "$config_path"; return 2
  fi
  config_size=$(diagnostic_command /usr/bin/stat -c '%s' -- "$config_path") || return 2
  if [ "$config_size" -gt 1048576 ]; then printf 'INCOMPLETE config size\n'; return 2; fi
  diagnostic_command /usr/bin/sha256sum -- "$config_path" || return 2
  diagnostic_command /usr/bin/timeout 5s \
    /usr/bin/sed -n '1,200l' -- "$config_path" || return 2
  if [ "$(diagnostic_command /usr/bin/sed -n '201=' -- "$config_path")" = 201 ]; then
    printf 'INCOMPLETE text output truncated %s\n' "$config_path"
  fi
}

inspect_loader_text /etc/ld.so.conf
inspect_loader_text /etc/ld.so.preload
config_count=0
for config_path in /etc/ld.so.conf.d/*.conf; do
  [ -e "$config_path" ] || continue
  config_count=$((config_count + 1))
  if [ "$config_count" -gt 64 ]; then printf 'INCOMPLETE config count\n'; break; fi
  inspect_loader_text "$config_path"
done
if [ -f /etc/ld.so.cache ] && [ -r /etc/ld.so.cache ]; then
  diagnostic_command /usr/bin/sha256sum /etc/ld.so.cache
  diagnostic_command /usr/bin/stat -Lc '%n type=%F size=%s uid=%u gid=%g mode=%a mtime=%y' /etc/ld.so.cache
else printf 'ABSENT_OR_UNREADABLE /etc/ld.so.cache\n'; fi
```

`sed`'s `l` command escapes control characters in displayed configuration.
Includes outside the named directory, unreadable files, symlinks and more than
64 files remain gaps. The cache hash identifies observed bytes; this procedure
does not parse or authenticate its contents and never runs `ldconfig` to
regenerate it. `LD_PRELOAD`/`LD_AUDIT`, loader command-line options, tunables,
secure-execution mode, cache state and included configuration are further
resolution inputs. Do not print the complete session environment: it may
contain secrets, and it would still not prove a future launch environment.

For each already reviewed system resolution directory, list hwcaps variants
without executing them. The fixed common multiarch location is only a probe:

```bash
if [ -d /usr/lib/x86_64-linux-gnu/glibc-hwcaps ]; then
  diagnostic_command /usr/bin/timeout 10s \
    /usr/bin/find /usr/lib/x86_64-linux-gnu/glibc-hwcaps -maxdepth 2 -type f \
    -name '*.so*' -print
else printf 'ABSENT /usr/lib/x86_64-linux-gnu/glibc-hwcaps\n'; fi
```

Review at most 256 returned candidates; more requires a separately scoped
collection, not truncation labelled complete. Inspect relevant variants with
the ELF function. Absence under this one directory does not establish global
absence. The Dell model does not establish CPU feature masks or glibc-hwcaps
selection. The accepted image must specify whether variants are excluded or
the exact admitted variants and selection inputs. No host fallback is allowed.

## 4. Native modules and CPython build observations

List package-owned candidate native modules without importing them:

```bash
diagnostic_command /usr/bin/timeout 10s \
  /usr/bin/dpkg-query -L libpython3.12-stdlib:amd64
if [ -d /usr/lib/python3.12/lib-dynload ]; then
  diagnostic_command /usr/bin/timeout 10s \
    /usr/bin/find /usr/lib/python3.12/lib-dynload -maxdepth 1 -type f -name '*.so' -print
else printf 'ABSENT /usr/lib/python3.12/lib-dynload\n'; fi
```

Review at most 256 candidates and inspect each relevant regular ELF using the
function. A `dpkg-query -L` path may be missing on disk, and `find` may discover
a file absent from the package database. Neither result alone identifies an
available Python import or a verified image member. Collect both discrepancies.
Third-party extensions from the proposed wheel set need separate approved
wheel/archive inventory; do not scan or import historical installations or
arbitrary `site-packages`. This procedure deliberately does not discover them
by executing `pip`, `site`, `sysconfig`, `importlib` or an extension initializer.

If already installed, inspect these package-controlled files as text (never
import a `_sysconfigdata` Python file or source a Makefile):

```bash
if [ -f /usr/include/python3.12/patchlevel.h ]; then
  diagnostic_command /usr/bin/dpkg-query -S /usr/include/python3.12/patchlevel.h
  diagnostic_command /usr/bin/sha256sum /usr/include/python3.12/patchlevel.h
  diagnostic_command /usr/bin/sed -n '1,100l' /usr/include/python3.12/patchlevel.h
else printf 'ABSENT /usr/include/python3.12/patchlevel.h\n'; fi
diagnostic_command /usr/bin/dpkg-query \
  -W -f='${binary:Package}\t${Version}\t${source:Package}\t${source:Version}\n' \
  python3.12-minimal:amd64 libpython3.12-minimal:amd64 libpython3.12-stdlib:amd64
```

`PY_VERSION` in headers is the header's declared upstream version. It is not
proof of the inspected executable's version. Installed package versions and
ELF Build IDs likewise do not prove compiler flags, PGO/LTO, frozen/builtin
module tables, SOABI, extension suffix ordering or libpython linkage. Those
facts must come from authenticated build materials tied to exact file hashes.

An **optional separate operator decision**, after independently authenticating
the system interpreter **and its loader/runtime closure**, may obtain the
interpreter's own version/build string with the following fixed command:

```bash
diagnostic_command /usr/bin/timeout 5s \
  /usr/bin/python3.12 -I -S -B -VV
```

This optional command executes the trusted system interpreter. It is not part
of the default static collection and must not be used if that trust condition
is unmet. It executes no Observer entrypoint or supplied script; `-I -S -B`
do not authenticate CPython or its dependencies. If omitted, report the exact
running interpreter version/build as **UNVERIFIED**, alongside the observed
package/header values. Do not run a historical interpreter to fill that gap.

Static ELF tags cannot enumerate arbitrary `dlopen` requests. CPython native
imports, `ctypes`, OpenSSL providers, NSS/locale/gconv modules and native-library
plugins may load additional bytes according to code/configuration/runtime
state. Import names, builtins and frozen modules cannot be exhaustively proved
from filenames. No diagnostic above claims otherwise. Static source/build
review must independently enumerate admitted requests and configurations, or
exclude that behavior in the proposed profile; unknown dynamic requests block
closure. Actual enforcement/measurement is separately authorized Stage 3B work.

## 5. Evidence handoff and independently expected inventory

Keep three records separate:

1. **Local observations:** host/package versions, lexical and canonical paths,
   file hashes and identity snapshots, ELF tags, candidate edges, configuration,
   missing tools/paths, timeouts, ambiguities and unexamined dynamic behavior.
2. **Independently expected materials:** accepted build inventory and immutable
   artifacts obtained and authenticated outside this host's self-report.
3. **Comparison result:** differences and unresolved resolution/closure claims.
   Matching hashes do not remove missing attestation or host namespace proof.

For official Ubuntu inputs, the build/inventory custodian must identify the
exact Ubuntu Noble archive pocket, architecture, package version and retained
snapshot. Obtain that snapshot's `InRelease` (or `Release` and detached
`Release.gpg`) plus the exact `Packages`/source indices and `.deb` artifacts.
Validate Release signatures against Ubuntu archive key fingerprints pinned by
the independent approval process, using a verifier/keyring obtained through
that process; a key downloaded beside the package or read only from this Dell
is not a new trust anchor. Check signed distribution/pocket/architecture,
index hashes, expiry/freshness and the approved historical-snapshot policy.
The verified index must bind each exact package filename/version/architecture,
size and SHA-256; check the downloaded `.deb` bytes against that entry before
extracting them as inert data in a separate review environment. Compare exact
members/hashes to the local observations. APT package signatures authenticate
archive publication, not the Observer's complete runtime policy.

The approved inventory must originate from the controlled build pipeline that
assembles the **actual Observer image** from these authenticated package inputs,
pinned CPython/build sources, wheels and reviewed transforms. A distro package
list is insufficient if materialization, relocation, stripping, copying or
other transforms change bytes. The custodian supplies an authenticated record
binding source revision, exact CPython build/configuration, artifact digests,
transform/toolchain inputs, final file paths/modes/owners/hashes, import and
embedded-module inventories, loader configuration, every admitted ELF edge
and dynamic-load request, CPU/hwcaps policy, and explicit absence/fallback rules.
Verify its signature/attestation using independently pinned build identity and
reviewer trust roots, freshness/revocation/checkpoints and exact output digest.
Where reproducible builds are claimed, a separate controlled rebuild and
byte comparison is additional evidence; it must not be invented from a local
list or from matching package names.

Reviewers must reconcile that expected inventory with this host's observations
and the chosen image resolution model. No local list is an independent
attestation. No command here emits an approval, signature or positive
qualification result. File `AcceptanceV1` syntax is not signature verification.

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

Next action is the operator's separately chosen read-only collection and an
independent custodian's authenticated exact-build inventory. Neither authorizes
Stage 3B, a privileged helper, launcher activation, Root Proof changes, legal
enrollment, Stage 10 or LIVE.
