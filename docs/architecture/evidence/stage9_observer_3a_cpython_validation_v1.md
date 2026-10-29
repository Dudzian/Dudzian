# Exact CPython reconciliation V1 — integrity and development checks

Date **2026-10-29 UTC**. Source baseline:
`cad53867231a1a6e49ea770c07043b417af84449`. These are cloud development
checks and static reads. No extracted target program/shared object, Dell
CPython, Observer, Stage 3B, hardware qualification, package maintainer script,
`ldd` or target loader diagnostic was executed. No production source,
security policy, tests, workflow, historical evidence or status file changed.

## Exact material checks

`VERIFIED_FROM_EXACT_BUILD`: current Ubuntu signature/index/source/package
checks are recorded in the [archive evidence](stage9_observer_3a_cpython_archive_sources_v1.md)
and adjacent JSON. Signature exit 0, expected full fingerprint; **4/4** index
representations, **4/4** exact CPython DEBs and **3/3** source files match
signed-index SHA-256 and sizes. The additional
[library-package records](stage9_observer_3a_cpython_library_packages_v1.json)
verify **4/4** exact runtime DEBs against the same authenticated binary index.
They were downloaded using host Python urllib through inherited proxy/TLS,
hash/size checked before `dpkg-deb --extract`. The original CPython acquisition
used curl and dpkg-source as documented separately. Package extraction never
installed or ran payload code. Root/bootstrap/time acceptance remains pending.

`VERIFIED_FROM_STATIC_ELF`: GNU readelf **2.44** and host-only data decoding
examined **55** ELF objects: interpreter, 46 available core extensions and
eight library/provider files. This is an inventory, not a selected-runtime
closure for all 46 modules. [Decoder source](stage9_observer_3a_cpython_static_extractor_v1.txt)
is retained as text for review/replay, alongside [decoded data](stage9_observer_3a_cpython_static_tables_v1.json)
and [raw selected output](stage9_observer_3a_cpython_readelf_v1.txt).

The decoder checks ELF64 little-endian identity, file-backed PT_LOAD address
mapping, bounded 16-byte builtin / 32-byte frozen records, and null sentinels.
Builtin names/order **60/60** and callback symbols **60/60** match generated
`config.c`, counting the two intentional null special entries. A separate
reader checked the parser layout against exact headers and confirmed frozen
names/order **3/3**, **15/15**, **11/11** against `Python/frozen.c`.
Frozen payload bytes are hashed without unmarshalling/executing them.
`PyImport_FrozenModules` is zero in PT_LOAD zero-fill initialization;
this says nothing about future runtime mutation. FrozenAliases and native
suffix tables were not independently decoded from ELF.

The preliminary selected graph has **12** nodes, **19** ordered DT_NEEDED
edges and one executable PT_INTERP edge. A finite traversal found it acyclic.
This validates only that subset and these relation types, without changing
the repository DAG model or implementing SCC. All twelve object SHA-256 values
match the historical report's strings. Physical Dell bytes were not reread.

`OPERATOR_REPORTED_AND_CHAT_REVIEWED` / `BUILD_SPECIFIC_UNVERIFIED`: the missing
Dell archive's SHA-256 remains an operator claim. **Review correction,
2026-10-29:** the original request's **13/13** statement was imprecise; the
independent original-archive review in the operator conversation reports
**14 manifest entries, 14/14 verified, 0 mismatches, 15 regular files total
(14 checked files plus the manifest file)**. The reported manifest-file SHA-256
is `f52859c04f703bfe6ba36190229e7b18ac1543df5c443d589a549835177c3fdb`;
local Dell readelf is **2.42**, distinct from the cloud readelf **2.44** above.
Archive SHA-256 remains
`0a26027f9bf94149fe2b36d3fbba1fdb6e5393e2016c9bd9956463b154005b0f`.
Codex still lacks that archive and has not replayed these corrected checks.
The exact historical uploaded text, including its original erroneous count,
is unchanged: **5366 bytes**, SHA-256
`5f0ef91f0078ea530558b8fdc873008bf3f73d1375b7d40359904c02db9939d8`.
Retaining it authenticates neither its author nor the physical measurements.

## Mandatory formatting and Tier A equivalents

The default tool Python lacked pre_commit/Ruff; the existing onboarding
environment lacked pre_commit. A task-only venv under `/workspace/work/`
provided pre-commit **4.7.0**, with configured Ruff **0.14.11** hook. Initial
unavailable-tool attempts were resolved; the mandatory hook then exited **0**:

```sh
PRE_COMMIT_HOME=/workspace/work/pre-commit-cache /workspace/work/quality-venv/bin/python -m pre_commit run ruff-format --all-files
```

Result: `ruff format ... Passed`, no source formatting changes. No hook,
scope, exclusion or scanner rule was changed. Host tooling Python **3.11.16**
is separate from the extracted Ubuntu 3.12.3 target. Other host data-analysis
scripts used tool Python **3.12.14**; neither is target inventory evidence.

PowerShell was unavailable. Applicable constituent commands from
`scripts/quality/run_fast.ps1` were run directly:

```sh
/workspace/.onboarding/venv/bin/python -m ruff check --no-cache bot_core core scripts tests deploy stage6_samples ui
/workspace/.onboarding/venv/bin/python -m mypy
```

Both exit **0**: Ruff `All checks passed!`; mypy `Success: no issues found in
359 source files`, with informational unchecked-function-body notes. The
task-only venv initially lacked mypy; the repository's existing onboarding
environment supplied it. The changed-path offline secret hook is recorded
below. Full Tier B, Semgrep, dependency audit and coverage are not claimed
as local checks for this documentation-only change.

## Existing contract regression tests

```sh
/workspace/.onboarding/venv/bin/python -m pytest -q tests/test_stage9_observer_contract.py
```

Exit **0**, **869 passed in 4.83s**, pytest **9.1.1**. These are existing
rootless synthetic parser tests, including refusal/negative cases; no new
tests were added for documentation. They execute development code in the
cloud, not inspected installation payloads or a physical Observer. They
establish regression behavior, not runtime closure or production qualification.

## Offline Betterleaks and document integrity

Stable Betterleaks **1.9.0** Linux asset SHA-256 matched the repository workflow's
pinned `f8b185a39ffcece2a1ca82bf3a4e7435cd81963ffd16b7a9128daf75f35f6de7`
before extraction. The scanner binary is in task scratch, outside Git.

```sh
/workspace/work/betterleaks/betterleaks dir --baseline-path .betterleaks-baseline.json --redact=100 --no-banner --no-color --report-format json --report-path /workspace/work/cpython-betterleaks-working-tree.json .
```

Exit **0**, approximately **126.88 MB** scanned in **3.04s**, no leaks found.
This full-tree scan ran before this validation note and packet manifest were
added; final staged-path scanning covers those final additions too.

`VERIFIED_FROM_EXACT_BUILD` for document identities only: historical manifests
remain unchanged and verify **22/22** (review R1), **4/4** (Dell ELF), **6/6**
(Ubuntu provenance) and **19/19** (minimal CPython scope). The new packet
manifest excludes itself and binds all new text/JSON files, exact repository
source/plan inputs and these unchanged historical manifests. Its successful
verification, UTF-8/JSON decoding, relative-link/fence/whitespace/status checks,
unchanged source comparison and staged-path secret scan are required before
opening the original documentation PR. Packet checks at original HEAD
`1932f5af941ca5a436888d8691c6d71af75d9f49`: **26/26** manifest entries,
**41** relative file links, **5** JSON documents, exact uploaded-text equality,
balanced Markdown fences and unchanged bytes for **12** source/plan/historical
manifest inputs. Markdown/JSON trailing whitespace checks pass; exact uploaded
text is preserved including its missing final newline. Readelf transcript
trailing blanks alone are normalized, explicitly marked in its header.
The final staged-path Betterleaks hook and `git diff --cached --check` pass.

## PR review correction checks

The correction to the original count and the full Windows SCM investigation
are recorded in the [review audit](../../audits/pr3224_archive_correction_windows_scm_2026-10-29.md)
and [SCM evidence JSON](stage9_observer_3a_windows_scm_review_v1.json).
The corrected PR contains **17 documentation/evidence files**, including
**6 JSON documents**. Its regenerated manifest has **36 entries**: the
16 non-manifest packet files, 16 unchanged reviewed repository source/plan
inputs and four unchanged historical manifests. It excludes itself.
The original request's exact bytes/hash and all historical manifests remain
unchanged. The SCM implementation repair is on separate
[PR #3225](https://github.com/Dudzian/Dudzian/pull/3225), HEAD
`4b873a0d80640e3408c153103044c6c462abff25`.

The mandatory all-files ruff-format hook, Tier A Ruff, **869 existing contract
tests (3.96 s)**, full-tree and final changed-path Betterleaks scans, SHA-256 checks for
all 36 packet entries and the four historical manifests, relative-link,
UTF-8/JSON/fence/whitespace checks are repeated for the correction. They must
all pass before the update is pushed; the correction's local results are PASS.
The full-tree scan examined approximately **126.95 MB** with no leaks; the
final scoped scan covers subsequent document-only additions/updates.
Hosted Actions for the new HEAD are
checked separately; the old Windows SCM failure is retained as a failure and
is not waived by a local PASS or by a later run. The final review response
reports the actual updated HEAD, checks and merge disposition.

## Merge and qualification limits

`NOT_YET_PROVEN`: hosted exact-HEAD checks are not implied by local PASS.
Evaluate the PR's current Actions/status results after publication; queued or
running checks are pending, not PASS. A separate documentation merge is
conditionally reviewable after clean raw diff, final integrity/scanner checks
and applicable hosted gates. No automatic merge is performed.

Project acceptance of keys/freshness/build expectations, complete selected
import/ELF/data/config closure, actual source-to-binary reproduction and
physical launch/lifetime/host/run attestations remain outside these checks.
No status promotion follows:

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
