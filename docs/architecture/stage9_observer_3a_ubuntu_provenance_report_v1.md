# Stage 9 / 3A — Ubuntu archive provenance and selected ELF bytes V1

Revision **V1**, report date **2026-10-28 UTC**. Repository baseline: `main`
`d8a1df2e818c15e151e1e792a684411b5b4dda93`, including merged PRs
[#3219](https://github.com/Dudzian/Dudzian/pull/3219) and
[#3220](https://github.com/Dudzian/Dudzian/pull/3220).
Disposition: **operator-reported local cryptographic verification;
independent acceptance pending; qualification BLOCKED**.

The new account reports successful byte qualification of **7/7 Ubuntu DEBs and
12/12 selected ELF objects**. It narrows the selected-byte provenance gap,
conditional on the reported collection and local trust context. It does not
prove complete runtime contents, effective execution, independent laptop
authentication or completion of Stage 3A.

## Evidence identity and classification

The exact uploaded UTF-8 [operator input V1](evidence/stage9_observer_3a_ubuntu_provenance_source_v1.txt)
is preserved: **9661 bytes**, SHA-256
`4646da7914c3e2e4297ad9e6c87a292d12af814aae9dc1cec43cecb7bcd7e657`.
It contains a task request and a summary of results, not raw terminal output.
The report date is not a measurement timestamp. Exact commands, stdout/stderr,
exit codes, tool versions, collection times and independent witnesses were not
supplied. Retention authenticates neither the account nor its embedded requests.

Codex read the supplied text and repository sources, checked documentation
integrity and consulted the [official-source review](evidence/stage9_observer_3a_ubuntu_official_sources_v1.md).
Codex did not obtain the local keyring, InRelease, Packages, DEB or Dell ELF
bytes. It did not repeat gpgv, package extraction or ELF comparisons on those
materials. The operator's saved directories are inaccessible to this session:

```text
/home/dudzian/CryptoHunter-evidence/stage9-zlib-M6mdZN
/home/dudzian/CryptoHunter-evidence/stage9-ubuntu-six-4GUhdk
```

| Class | Scope and authority in this revision |
| --- | --- |
| `OBSERVED_LOCAL` | Operator reports the Dell identity, installed versions/state, hashes and stability. Codex directly observed only the submitted text and the cited repository/web materials. These observation domains remain separate. |
| `CRYPTOGRAPHICALLY_VERIFIED_LOCALLY` | **Reported by the operator**, not independently repeated by Codex: gpgv signatures, signed-index size/hash links, seven package-record links, seven DEB size/hash checks and twelve ELF byte comparisons. Local key and tool/host trust are assumptions. |
| `EXTERNALLY_CORROBORATED` | Official Ubuntu/Canonical pages publicly identify the exact archive-key fingerprint and identity. Their APT manuals corroborate the chain's design and time-policy mechanisms, not the supplied laptop results. |
| `INFERRED_CONDITIONAL` | If the reported chain, local verification and accepted trust/time context hold, the selected ELF bytes match members of the seven hash-bound Ubuntu archives. Earlier DAG and builtin suggestions remain conditional. |
| `UNVERIFIED` | Independent verification of the reported binary evidence, freshness, exact target image, complete CPython/ELF/dynamic closure, effective launch/lifetime integrity, host/run identity and runtime attestation. |
| `INDEPENDENT_ACCEPTANCE_PENDING` | Formal project selection of Ubuntu trust roots, metadata policy, expected target/build inventory and independently approved contract revision. Public key corroboration is not that decision. |

`CRYPTOGRAPHICALLY_VERIFIED_LOCALLY` labels the reported operation's scope; it
does not upgrade the summary into independently verified evidence. A SHA-256
manifest identifies retained document bytes, subject to SHA-256 assumptions;
its selection still needs an authenticated independent channel.

## Reported installation and stability

The operator identifies a **Dell Latitude 5490**, **Ubuntu 24.04.5 LTS**,
**amd64**, interpreter `/usr/bin/python3.12`, Python package revision
**3.12.3-1ubuntu0.17**. All seven listed packages reportedly had dpkg status
`ii`. These are local observations, not an independently authenticated OS,
device identity, approved image selection or installed-file completeness check.

The reported dpkg database SHA-256 before and after collection is identical:

```text
e3e783901bd414ff51acc4114d467b6d5817612ccc64639bcb9a4194f75fcbe4
```

The source does not specify the exact dpkg database pathname/extent hashed.
Endpoint equality establishes reported equality of that sampled database
content only. It does not prove every installed file remained unchanged, exclude
changes and restoration between samples, or provide lifetime runtime integrity.

## Reported archive signature and index chain

Both InRelease files reportedly pass `gpgv` under local keyring SHA-256
`80a36b0a6de2f69f49d2df75ef473ccde121e9e190b9ea01d20a4f63778d5c31`.
The reported signer fingerprint is
`F6ECB3762474EDA9D21B7022871920D1991BC93C`,
**Ubuntu Archive Automatic Signing Key (2018)**.

| Pocket / InRelease | Reported date | InRelease SHA-256 |
| --- | --- | --- |
| `noble-security/InRelease` | 2026-10-27 | `03b4a340344bee27fe5de728cf9060b9a30a102a85ca30f25c209980c0d4f184` |
| `noble-updates/InRelease` | 2026-09-14 | `fccd5d3f5d46cfcd9d2c05ffd0720de11fa77c9b8e3b4e47d1874da83c0360b1` |

| Signed-index logical pathname | Reported size (bytes) | Packages SHA-256 |
| --- | ---: | --- |
| `noble-security/main/binary-amd64/Packages` | 6241173 | `6aca30ff476fab7b627369a756b02b1ac59cf59b01776f8c9b0b144dec301ce7` |
| `noble-updates/main/binary-amd64/Packages` | 6812353 | `3658d119bd64ea79e24a0462c8d66110b53c4dc3da2e5bd0217ca68cd9e7e5be` |

Each index reportedly passed GPG linkage, SHA-256, size and stability checks:
`TWO_SIGNED_INDEX_HASHES_MATCH`. Preserve these as the supplied logical
uncompressed index names; no particular local compressed APT-cache file,
decompression command or byte representation has been independently inspected.

The chain being documented is:

```text
project-accepted Ubuntu key + accepted time/snapshot policy [PENDING]
  -> reported gpgv verification of the exact InRelease bytes
  -> signed SHA-256 + size of each exact Packages index
  -> exact name/version/architecture package records and DEB SHA-256 + size
  -> matching downloaded DEB bytes
  -> extracted member bytes matching the 12 sampled Dell ELF objects
```

Each arrow needs retained byte-level evidence and a trustworthy verification
context for independent replay. The summary omits raw package stanzas,
Filename/Size fields, per-record pocket attribution, full archive URLs,
member paths/types, extraction logs and original file identity captures. No
DEB size or per-package security/updates attribution is invented here. The
reported 7/7 record match is not 14 independently verified records.

APT archive signatures authenticate Release metadata and its downstream hash
chain. The seven DEBs are **hash-bound by signed archive metadata**; this is not
a claim that each DEB carries an independently verified GPG package signature.
See official-source U5 (`apt-secure(8)`).

## Exact seven-package inventory

All rows are operator-reported `CRYPTOGRAPHICALLY_VERIFIED_LOCALLY`: exact
record match, downloaded DEB size/hash PASS. Result:
`SEVEN_SIGNED_PACKAGE_RECORDS_MATCH`. The operator reports downloads from the
official Ubuntu archive, no package installation and archive inspection without
executing contained code. Codex has not independently confirmed those actions.

| Package | Reported installed version | DEB SHA-256 |
| --- | --- | --- |
| `python3.12-minimal` | `3.12.3-1ubuntu0.17` | `d452689b9660845345a4c3e05e4ad82c082d5474e04031b7aa47f1a6d5610a6e` |
| `libpython3.12-minimal` | `3.12.3-1ubuntu0.17` | `d646ad7112b5adec21ba0e1af015f04ae8ca7f5efdc619622547dd6673c4c14b` |
| `libpython3.12-stdlib` | `3.12.3-1ubuntu0.17` | `45d3f530ba1f9d6e879ad46b92046fabab13fe50a82450e4f65557a3bbad1489` |
| `libc6` | `2.39-0ubuntu8.9` | `ff5557d99b51f761c4b7c92368b9cc45565eda17df9bf9eb4b134d09825008be` |
| `zlib1g` | `1:1.3.dfsg-3.1ubuntu2.3` | `376249c8de139568221dbb242318cc8cb8f02299e3a62b7432948e1754fbddf9` |
| `libexpat1` | `2.6.1-2ubuntu0.6` | `494b8e672f722130c6bca6a7bc4cc31a43ca891a31d60d868bfdd699a3c20b13` |
| `libssl3t64` | `3.0.13-0ubuntu3.16` | `219f43b1cd836a4da550938db5fda93160d269d39a4edcf1f1ce698a470db797` |

The zlib epoch `1:` is part of the version and is retained. Seven archive
matches do not establish that all package payload files, dependencies or the
whole installation were compared. No `.deb` binaries are added to Git.

## Exact twelve-object ELF inventory

All rows are operator-reported local byte-comparison PASS against corresponding
DEB members. SHA-256 values match the same twelve ELF rows in the
[historical Dell report V1](stage9_observer_3a_dell_local_elf_report_v1.md).
Object labels remain labels where canonical absolute paths were not supplied;
no exact package-member mapping or filesystem alias proof is fabricated.

| Selected object | SHA-256 |
| --- | --- |
| `python3.12` | `e50d468e8b0adfb05733f5b87b3cff34829c4a8c1aea50c865aa8bdfe4bb150f` |
| `_hashlib` | `8dc6cda6dd9495b95a2bfb5b26e8cf06b72f36dc0213d958b9277a958287f0fe` |
| `_ssl` | `80c755c80688b92e0f32af9168a605f1283256077f35d30f6cf518252f2a6660` |
| `_json` | `ca8e35c58f7bc8e27258c4af43651c941413bc6318903621253485b0395758a3` |
| `ld-linux-x86-64.so.2` | `c20a2dc8917c755f02b94049356320fe1f62ac7d9f8994731f807d9df39302da` |
| `libc.so.6` | `3a15d66867d83762c7f2f1e37359cb8f6c5743edb369c65285cb0b1c4f7498bf` |
| `libm.so.6` | `fce00b6f25f459cf4ae0b7fae4257909a1a8a86f9149e7c029a5d617baf1ccd0` |
| `libz.so.1.3` | `f61677c7bdb7c2aa2a243ea781bfd16afcb361cede7b4ef76127a9d8eb9a5242` |
| `libexpat.so.1.9.1` | `286682ecbc5e59a638963b1a4e6351e65eb32fcf4bdcb9cb7569b6a61fe06a8d` |
| `libcrypto.so.3` | `6866ed711dab8927e1bf82bf1cd30e828bdfc34f3cc1e83b1c9eb01ce6c37d9a` |
| `libssl.so.3` | `a40e7b67489bbd257e432c700ff8f6a1ad796425b5eccecdc2930127fdd86624` |
| `legacy.so` | `e639c7121825222dc720bd38aa1fca3ddcedd7d1213dc72ce006b8ccf541b547` |

Reported totals: **7/7 DEBs PASS; 12/12 ELF PASS; stability PASS**. There is
no additional verified ELF row for `_datetime` or `/etc/ssl/openssl.cnf`.
The latter's historical standalone digest remains local configuration evidence,
outside this DEB-member verification. None of the 12 matching object hashes
proves that all required modules or future loads are present and controlled.

## Official key corroboration and project root acceptance

Official Ubuntu security documentation prints the full fingerprint beside
**Ubuntu Archive Automatic Signing Key (2018)**. Official Chisel documentation
also associates that fingerprint with the Noble archive example. Sources U1/U2
were opened on **2026-10-28 UTC**, independently of the operator's summary.
This supports `EXTERNALLY_CORROBORATED` for the published key identity only.
The retained [source matrix](evidence/stage9_observer_3a_ubuntu_official_sources_v1.md)
provides exact URLs, short excerpts and the APT references U3–U6.

These pages share the Ubuntu/Canonical publisher trust domain; their agreement
is not multiple independent institutional approvals. It does not establish the
bytes of the local keyring or authenticate the laptop, its tools or archive
snapshot. A keyring hash without an authenticated expected keyring is only an
identifier. Formal acceptance remains `INDEPENDENT_ACCEPTANCE_PENDING`:
designated independent authority must pin accepted key bytes/full fingerprint,
trusted acquisition channel, permitted suites/architecture/use, revocation and
rotation treatment, review time and signed external decision. Ubuntu archive
keys do not automatically become Observer acceptance-reviewer keys.

## Freshness assessed separately from signature validity

At report date 2026-10-28, `noble-updates`' reported 2026-09-14 date is **44
calendar days earlier**; `noble-security`'s 2026-10-27 date is **one calendar
day earlier**. Full signed timestamps and capture times are missing, so these
are not exact elapsed ages. Both pockets' freshness remains **UNVERIFIED**.
The older updates document warrants investigation; its date alone establishes
neither a security failure nor acceptability/currentness. A valid historical
signature does not prevent replay and is not proof of a fresh mirror state.

Official Noble APT manuals document Check-Date/Check-Valid-Until defaults and
global/per-source time controls. In particular Max-ValidTime defaults to zero
("valid forever"); actual signed Valid-Until fields or stricter configuration
may still impose expiry. No such default is asserted to be the Dell's effective
policy. `gpgv` verification is not evidence that APT freshness checks ran.

Required for a decision: exact signed Date and Valid-Until or demonstrated
absence, signature creation times, trusted UTC collection/validation time and
clock source, archive/mirror identity and retrieval timestamps, effective global
and per-source APT time options plus APT_CONFIG/CLI overrides, and the project's
accepted maximum-age/replay/historical-snapshot policy. Compare independent
authenticated contemporaneous metadata if that policy requires it. Preserve
these historical bytes; fetching today's index would not retroactively validate
the reported snapshot or prove it was current when measured.

## Effect on historical findings and severity

This is an additive reassessment of the
[contract audit](stage9_observer_3a_contract_audit.md),
[review package R1](stage9_observer_3a_review_package.md),
[ELF qualification analysis](stage9_observer_3a_elf_qualification.md) and
[Dell report V1](stage9_observer_3a_dell_local_elf_report_v1.md).
Historical documents, findings and qualification states remain unchanged.

| ID / retained priority | Newly documented portion | Still blocking / required disposition |
| --- | --- | --- |
| CA-P1-01 / P1 | Selected package/ELF origin is supported by the operator-reported signed hash chain; published archive-key identity is externally corroborated. | Independently accepted exact Observer build/image and complete recursive contextual ELF/CPython inventory remain absent. Neither independent replay nor root/freshness acceptance is complete. **PARTIALLY DOCUMENTED; OPEN, BLOCKING.** |
| CA-P1-02 / P1 | Full Ubuntu archive-key identity now has official public corroboration. | External root enrollment, distinct reviewer authority, actual Observer Ed25519 acceptance, revocation/time/checkpoints and durable rollback protection remain unresolved. An archive signature does not approve contracts. **PARTIALLY DOCUMENTED; OPEN, BLOCKING.** |
| CA-P1-03 / P1 | The reported bytes of twelve sampled objects are linked to package members at collection. | Import/bootstrap/data/dlopen closure, effective launch and continuous file protection are unmeasured. Matching endpoint database hashes are not lifetime enforcement. **PARTIALLY DOCUMENTED; OPEN, BLOCKING.** |
| DELL-P1-01 / P1 | Selected-byte package provenance is stronger than the historical local hashes alone. | The selected Dell files do not define or authenticate the exact expected Observer image; complete ELF/import/dynamic closure is missing. **PARTIALLY DOCUMENTED; OPEN, BLOCKING.** |
| DELL-P1-02 / P1 | Public key identity and reported local verification clarify one provenance link. | Independent host/run/launcher identity, accepted authorities/signatures, actual scope and continuous integrity remain absent. **OPEN, BLOCKING.** |
| DELL-P2-01 / P2 | Summary now records package hashes, selected member equality and endpoint stability. | Raw capture provenance, exact member/canonical paths, SONAME/alias evidence, builtin `_datetime` registration and effective provider/configuration context remain unresolved. Configuration file was not included in this DEB comparison. **PARTIALLY DOCUMENTED; OPEN**; required closure escalates to the existing P1 boundary. |

| Priority | Bounded assessment for this revision |
| --- | --- |
| P0 | No new confirmed P0 defect established by the supplied summary and targeted source review. This is not proof of absence. |
| P1 | All five P1 entries above retain their blocking boundaries. No downgrade or closure is justified by selected-byte matches. |
| P2 | DELL-P2-01 and historical CA-P2-01–04 / DELL-P2-02 remain open: offline canonical/bundle exception, precedence, resource enforcement and replay/receipt/crash protocols. Metadata freshness also needs an explicit acceptance decision; age alone is not a newly confirmed security defect. Hosted validation is assessed for this PR's exact HEAD, separately from historical DELL-P2-03. |
| P3 | Historical CA-P3-01 and DELL-P3-01 remain: frozen dataclasses do not isolate hostile in-process code; report labels and digest links do not confer evidence authority. |

## Compatibility with all eight Observer V1 contracts

Direct source inspection used the unchanged
[parser](../../deployment/stage9_observer_contract.py),
[canonicalizer](../../deployment/canonical_json.py) and
[V1 proposal](stage9_observer_3a_schema_proposal.md). Codebase Memory MCP and
its executable are unavailable; there is no graph-coverage or exhaustive-source
claim. Named validators, the synthetic-only bundle boundary and the findings
above were checked directly against source.

| Contract | Effect of this new evidence and remaining boundary |
| --- | --- |
| `ObserverManifestV1` | `_proposed_sources` accepts `ubuntu_apt_snapshot` declarations; `_validate_proposed_runtime` and `_proposed_acyclic` validate declared references/graphs. This packet is not a complete manifest or an authenticated expected build, and supplies no artifact-verification implementation. |
| `AcceptancePolicyV1` | `_validate_proposed_acceptance_policy` binds target/source records, limits and reviewers. Ubuntu fingerprint corroboration neither enrolls a root nor supplies approved independent reviewers, time or policy. |
| `ObserverResultV1` | `_validate_proposed_observer_result` retains synthetic/incomplete restrictions and immutable blocks. An operator diagnostic summary is not a run/result or host observation. |
| `LaunchPolicyV1` | `_validate_manifest_launch` compares projections and pins; selected package bytes do not prove effective argv/environment, loader mapping, sandbox/resources or lifetime protection. |
| `ObserverRunContextV1` | `_validate_run_context` checks supplied run/nonce/scope/pins, not trusted launcher identity, one-time state or authenticated channel. This packet creates none. |
| `HostScopeBindingV1` | `_validate_scope_binding` constrains scope declarations, not initial host PID namespace or complete procfs visibility. `host_pid_namespace=UNVERIFIED` persists. |
| `OfflineVerificationResultV1` | `_validate_offline_result` remains advisory/synthetic with acceptance UNVERIFIED. Its existing raw/canonical and bundle-binding exceptions are unchanged; this report is not that result. |
| `AcceptanceV1` | `_validate_acceptance` checks envelope/signing-preimage structure; actual Ed25519 verification and externally accepted keys/freshness remain external. Ubuntu GPG archive signatures are not Observer acceptance signatures. |

`parse_proposed_bundle` binds **seven input documents**, excluding advisory
offline output, and expressly authenticates no signature, artifact, launcher or
host. No new contract, acceptance envelope, signature or attestation is emitted.

The historical report's 12-node / 19-`DT_NEEDED` partial DAG remains a conditional
derivation from the earlier observations. New byte matches supply no new edge,
context, trace or completeness proof. Preserve both the current runtime DAG
restriction and the separate provenance DAG. `legacy.so` equality does not show
that libcrypto loads it. A hypothetical `libcrypto -> legacy -> libcrypto`
mixed dynamic/static cycle still does not justify SCC implementation. A required
real cycle in the accepted target would need new confirmed evidence and a
separately approved model/version; no edge may be omitted to force a DAG.

Similarly the earlier `PyInit__datetime` symbol and absence of a standalone
module do not establish builtin registration. The matched executable identifies
a container only. Authenticate build registration/expected inventory before
representing `_datetime` as a builtin through `embedded_modules` and its exact
build/container binding. No fabricated module digest or thirteenth ELF is added.

## Next evidence, in dependency order

1. Retain and independently acquire original command transcripts, tool versions,
   exact timestamps and byte identities. Supply the exact keyring/InRelease/
   Packages/DEB bytes through an approved evidence channel, package stanzas with
   sizes/URLs/pocket mapping, and member-to-Dell path/identity comparisons.
   Independent review may reproduce nonexecuting archive verification; its
   authorization and trustworthy tools/context must be explicit. Binary DEBs
   are not implicitly approved for inclusion in Git.
2. Independently accept the Ubuntu root/key lifecycle and the metadata validity,
   age/replay/snapshot policy; evaluate both preserved InRelease files against
   trusted time and effective collection policy. Public sources alone do not
   close these decisions.
3. Authenticate one exact expected Observer image, approved build inputs/recipe/
   toolchain/transforms and complete output inventory. Reconcile it with the
   selected Dell bytes without treating the laptop as its own trust root.
4. Establish complete CPython source/native/builtin/frozen/bootstrap/deferred/
   error-path inventory, ABI suffixes/import roots and `_datetime` registration,
   bound to the accepted exact build and containers.
5. Establish complete recursive ELF headers, tag order/SONAME/aliases/identities,
   symbol/version compatibility and every approved load context; qualify actual
   execution-visible PT_INTERP, RPATH/RUNPATH, cache/default search, CPU/hwcaps,
   preloads/audit, namespaces and already-loaded object reuse.
6. Bound native initialization, imports, dlopen, OpenSSL configuration/providers,
   NSS/locale/encoding and error paths. Determine whether legacy is required or
   effectively excluded, and whether all permitted edges fit the retained DAG.
7. Separately prove the exact authenticated launch/run context, nonce/channel,
   effective sandbox/resource policy, permitted file/search candidates and
   continuous integrity through all delayed loads. Host scope and runtime
   attestation remain distinct missing evidence.
8. Designated independent reviewers must resolve historical P1/P2 decisions,
   pin the contract/material revision, validate real signatures and external
   revocation/checkpoint/time/replay state, and record approval outside the
   submitted bundle. Controlled physical/runtime qualification, if later
   authorized, must not be replaced by synthetic parser tests.

## Integrity, validation and merge recommendation

The new [V1 manifest](evidence/stage9_observer_3a_ubuntu_provenance_v1.sha256)
pins this report, the exact input, official-source note, new validation note
and both unchanged historical manifests. It excludes itself and does not list
operator-local binary hashes as if those files were present in Git.
The historical manifests recursively identify their original evidence. Verify
all three from the repository root:

```bash
sha256sum --check docs/architecture/evidence/stage9_observer_3a_review_r1.sha256
sha256sum --check docs/architecture/evidence/stage9_observer_3a_dell_local_elf_report_v1.sha256
sha256sum --check docs/architecture/evidence/stage9_observer_3a_ubuntu_provenance_v1.sha256
```

Actual local commands, outcomes and limitations are retained in the
[validation note](evidence/stage9_observer_3a_ubuntu_validation_v1.md).
New-PR number, immutable HEAD, raw diff and timestamped exact-HEAD GitHub Actions
states belong in the delivery report once the PR exists. Historical CI does
not validate this revision; queued, running, skipped or cancelled is not PASS.
No validation here substitutes for an independent measurement.

**Conditional merge recommendation:** this additive documentation packet may be
considered after content review, manifest/link validation and assessment of all
required exact-HEAD checks. Any unavailable format gate requires draft status
under AGENTS.md until the separate Style preflight passes. Failure or unfinished
CI needs explicit disposition; production-readiness refusal remains enforced.
Merging documentation would not close any qualification/approval blocker.
No merge or automatic merge is performed by this task.

```text
stage_9=IN_PROGRESS
qualification=BLOCKED
CONTRACT_APPROVED=NO
STAGE_3A_COMPLETE=NO
PRODUCTION_QUALIFIED=NO
production_status=PRODUCTION_NO_GO
runtime_attestation=MISSING
host_pid_namespace=UNVERIFIED
full_confirmation=false
coexistence_authorized=false
preflight_exception=false
physical_test_status=PHYSICAL_TEST_PENDING
```

This revision changes documentation only. Observer, Stage 3B and physical Dell
tests were not run; production code, contracts, workflows, historical evidence,
qualification statuses and DAG behavior are unchanged.
