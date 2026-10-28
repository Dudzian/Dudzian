# Stage 9 / 3A: eight-contract security review

Review date: 2026-10-28. Source baseline:
`ae712aea85e4cbc372bdf226d6a5bc5be02f4949` (merged PR #3215).
Status: **PROPOSED_UNAPPROVED**. This is a source-based recommendation for
independent review, not approval, attestation, or authorization to activate.

The reviewed sources are the [base design](stage9_trusted_observer_contracts_and_implementation_plan.md),
[V1 proposal](stage9_observer_3a_schema_proposal.md),
[previous final review](stage9_observer_3a_final_review.md),
[parser](../../deployment/stage9_observer_contract.py),
[canonicalizer](../../deployment/canonical_json.py), and
[synthetic tests](../../tests/test_stage9_observer_contract.py), including their
fixtures. References below name validator/test symbols so the review can be
checked against the pinned source. Codebase Memory MCP and its executable were
not available in this environment; the audit used targeted source inspection.
There is no graph-index coverage result or claim of a repository-wide proof.

## Representation and contract coverage

`_decode`, `_preflight`, `_record`, `_integer`, `_path`, `_identifier` and
`_reason_codes` implement strict bounded syntax before domain validation.
Duplicate decoded keys, null, floats, non-finite values, booleans used as
integers, invalid UTF-8, surrogate/NUL, unknown/missing fields and unsupported
variants refuse. Reason vocabulary is the fixed 17-code set. These properties
establish a data boundary, not identity or truth of the declarations.

The shared canonicalizer sorts object keys by UTF-16BE code units, emits UTF-8
without ASCII escaping, retains array order and permits no float. Observer
decoding additionally forbids null and negative integers. This exact profile
must be used by independent implementations; ordinary sorted-key JSON is not
an interchangeable signing representation. There is no Unicode normalization.

| Contract | Implemented structural rules and source | Evidence boundary / review decision |
| --- | --- | --- |
| `ObserverManifestV1` | `_validate_proposed_manifest` and `_validate_proposed_runtime`: exact nested variants; unique IDs/paths; typed source/file/transform references; reciprocal transform outputs; explicit image ancestry; regular-file role union; separate provenance and ELF cycle checks; source/native/embedded import bindings. Proposed API requires canonical raw bytes. | Inventory closure means closure of supplied declarations. The parser does not inspect binaries, packages, archive members, reconstruction proofs, effective imports or the expected build inventory. Exact-target ELF compatibility remains P1. |
| `AcceptancePolicyV1` | `_validate_proposed_acceptance_policy`: exact full target and source records, 22 fixed limits, distinct sorted signer IDs, distinct reviewer IDs/raw keys, threshold two, sequence/checkpoint/window and launch pin. Bundle checks ordered equality to manifest sources/target. | A supplied policy cannot nominate its own trusted reviewers or authenticate its own root set. Identity independence, key validity, revocation and fresh checkpoint/time are external prerequisites. |
| `ObserverResultV1` | `_validate_proposed_observer_result` and `_validate_proposed_observer_sequence`: ten counters, three exact booleans, event overlap constraints, cumulative retention, fixed historical commit pair, bounded run window, required missing-proof codes and all immutable blocks. Proposed API requires canonical raw bytes. | These are synthetic assertions. Zero matches, true stability, a valid frame or increasing counters do not prove complete observation. Single-result parsing has no expected run; bundle/channel verification is a separate boundary. |
| `LaunchPolicyV1` | `_validate_launch_policy` and `_validate_manifest_launch`: exact target/build/argv/environment/FD/sandbox/resource/profile records; ordered ELF/import projections and files/runtime/import digests; profile regular-file/data-role binding; canonical raw bytes. | The policy describes requested enforcement. It neither launches a process nor proves credentials, memory/deadline limits, filesystem integrity, sandbox, effective loader or imports. |
| `ObserverRunContextV1` | `_validate_run_context`: exact fields, bounded canonical bytes, embedded scope validation, matching nonce and canonical scope digest. Bundle compares document pins, run ID/nonce, scope and result bindings. | Digest equality and sealed-memfd declarations cannot authenticate a launcher. Randomness, one-time use, FD provenance and trusted receiver state are not supplied by JSON. |
| `HostScopeBindingV1` | `_validate_scope_binding`: exact boot/nonce/starttime/namespace/procfs records, positive inode/mount IDs, matching mount namespace, sorted allowed mount options and fixed claimed view; canonical raw bytes. | Device/inode numbers, `/`, `HOST_INITIAL_COMPLETE`, a local signature or readable namespace file do not establish initial host namespace or faithful host scope. Authentication-evidence digest is opaque. Host scope remains `UNVERIFIED`. |
| `OfflineVerificationResultV1` | `_validate_offline_result`: closed fields; bounded digests/reasons; `synthetic=true`, acceptance exactly `UNVERIFIED`, fixed blocks, mandatory `RUNTIME_LAUNCH_UNVERIFIED`. `artifact_consistency` permits MATCH/MISMATCH/INCOMPLETE. | Advisory derived output, not an eighth input to `parse_proposed_bundle`. `parse_offline_result` currently permits noncanonical raw bytes and retains both raw and canonical forms. It does not compare its document pins to a supplied bundle or verify artifacts. See CA-P2-01. |
| `AcceptanceV1` | `_validate_acceptance` / `parse_acceptance`: canonical whole envelope, exact payload, canonical payload hash, sorted unique signer IDs and 64-byte signature encodings, threshold two. Bundle checks policy membership, raw document hashes, root/checkpoint/sequence and synthetic window. | Ed25519 verification is absent by design. A well-formed zero/forged signature remains unverified. All supplied signatures must later verify under externally accepted keys; merely finding two good signatures must not hide an invalid or revoked extra signer. |

Raw manifest and policy hashes pin the exact received bytes. Launch and scope
pins equal raw hashes because their parsers require canonical bytes. Acceptance
signs exactly `b"CryptoHunter.Stage9ObserverAcceptanceV1\x00"` followed by the
32 raw SHA-256 bytes of its canonical payload. Threshold and signature array
are envelope fields outside that signing preimage; the payload's policy pin
must bind the independently accepted threshold/key policy. No ASCII-hex
preimage, normalization, second hash or invented signature is authorized.

`parse_proposed_bundle` reparses **seven** raw input documents: manifest,
policy, launch policy, acceptance, scope, run context and result. It compares
the references they can actually express. Offline output has three document
pins but no run/scope/acceptance pin, and is not part of this input set. Do not
describe seven-way consistency as validation of eight mutually authenticated
documents. Standalone parsers cannot check references to absent documents.

## Threat model and retained boundaries

An adversary may control all submitted bytes, names, file labels, paths,
signatures, digests, claimed clock/checkpoint, scope and synthetic observations.
Relevant attacks include schema confusion, decoder resource exhaustion,
canonicalization disagreement, substituted source/build/profile materials,
import shadowing, omitted loader dependencies, self-selected roots, reviewer
aliasing, rollback/replay, counterfeit launcher/pipe and suppressed positive
observations. Syntax and cross-reference checks constrain these inputs; they
do not authenticate a host, issuer or execution.

The parser imports only standard-library modules and the shared canonicalizer;
`test_contract_architecture_keeps_only_stdlib_and_shared_canonicalizer` checks
this isolation. Its file paths are data, not paths that the parser opens.
Artifact/toolchain inspection, trusted launch, host observation, signature and
trust-context verification remain separate work. No control-plane message,
privileged helper, operator-supplied execution argument or activation mechanism
is introduced by this review.

Import defenses cover declared standard source/native candidates, explicit
source-package parents, forbidden `.pyc` discovery and native/package shadow
forms. They cannot prove a CPython build's real suffixes, `sys.path`, bootstrap,
frozen availability or behavior of extension initialization. `-I -S -B` is a
fixed requested argv, not a proof of any of those properties. An executable
file labelled `data` is inert only if actual import/loader/file-access policy
prevents its execution throughout the run. Acyclic provenance does not imply
acyclic ELF or complete runtime closure; see the
[ELF qualification review](stage9_observer_3a_elf_qualification.md).

The parser enforces document/cardinality/depth limits and the explicit import
comparison ceiling `(files + imports) * import_roots <= 65536`; graph traversal
is iterative. It does **not** enforce its own process memory limit, artifact
deadline, bundle deadline or worker lifetime. The 22 policy values declare the
future enforcement contract. A parser result, benchmark or fixture timeout
cannot establish the required 256 MiB worker limit or 300-second trusted
deadline. Decoding, canonicalizing, copying and repeated bundle validation all
consume memory and time; exact-target resource measurements are still needed.

Incomplete reads, permissions, limits, identity/binding drift and missing scope
must retain INCOMPLETE/BLOCKED and preserve prior matches. Sequence comparison
protects two caller-supplied synthetic snapshots from internal regression; it
is not persistent replay protection or permission to send two frames. The
trusted receiver must derive EOF/exit/time and authenticate producer/run/pipe;
the framing helper's caller-provided arguments are fixture assertions only.

## Open findings and explicit assumptions

No additional verified P0 or accepting-standard-parser defect was established
in these inspected sources. The following register records approval decisions
and evidence gaps, including assumptions that must not remain implicit. P1
items block their stated approval/activation boundary; absence of their future
implementations is not a new regression in an intentionally inactive parser.

| ID / severity | Finding or assumption | Required disposition |
| --- | --- | --- |
| CA-P1-01 | Exact Ubuntu 24.04 x86_64 CPython 3.12 build/runtime inventory and ELF resolution are not independently accepted. Declared graph closure is not completeness evidence. | Obtain authenticated expected build materials and exact-target resolution analysis; accept current restriction or review a bounded revised model. Cross-reference ELF review, without weakening current parser. |
| CA-P1-02 | Reviewer independence, policy-root enrollment, Ed25519 validation, trusted time, checkpoint freshness and durable rollback state are outside all eight JSON schemas. Names and raw-key uniqueness do not establish them. | Independent approval must identify external authorities, pinned profile/policy revision, key lifecycle and authenticated offline checkpoint/time policy. Activation cannot infer these facts from submitted records. |
| CA-P1-03 | Import/bootstrap/data/ELF closure and effective launch/file-integrity protection are assumed by declarations. Neither bytes nor lifetime enforcement are measured here. | Authenticate expected available modules, native initialization/dlopen/error-path materials and all permitted files; separately verify effective enforcement and no substitution/fallback. Preserve production blocks meanwhile. |
| CA-P2-01 | Offline canonicalization and binding scope differ from the seven bundle inputs. The prior review's broad “whole documents” wording must not imply a canonical requirement implemented for offline output. | Explicitly review the current advisory-output exception: `sha256` on ParsedContract denotes raw input while `canonical` is a separate representation. Never interchange those pins. A future canonical offline wire or authenticated receipt requires an approved revision and negative tests; leave parser semantics unchanged in this review. |
| CA-P2-02 | Base design §3.5 describes future offline acceptance VERIFIED/REJECTED and result COMPLETE; proposed §5 intentionally narrows the current synthetic profile to UNVERIFIED/INCOMPLETE. | Record precedence: proposed synthetic restrictions govern this candidate; future states in the base design are not accepting variants in 3A. Do not activate them through a supposedly compatible parser update. |
| CA-P2-03 | Resource ceilings are exact schema constants but parser process memory/time enforcement is external. No maximum accepted-bundle resource measurement is supplied by source. | Before artifact verification/activation, measure adversarial bounded inputs on the accepted runtime and demonstrate enforced limits, termination and refusal with no partial success. Do not equate a declared constant with enforcement. |
| CA-P2-04 | Real replay state, receipt authentication, crash recovery and trusted-clock/checkpoint availability are not specified as executable protocols by these contracts. | Define in separately reviewed receiver/trust work: one-time nonce consumption, crash/restore behavior, boot/session scope, stale/unknown state refusal and authenticated export receipt. Synthetic monotonicity tests do not discharge this item. |
| CA-P3-01 | Python frozen dataclasses prevent routine accidental mutation, not hostile code already executing in the same interpreter. | Keep the trusted verifier process in the threat boundary; an untrusted in-process plugin or constructor must never be an authority channel. No immutability claim creates a privilege boundary. |

The following details are explicit prerequisites rather than facts proven by
successful parsing. They supplement the finding register and expose remaining
deployment assumptions for an independent reviewer:

1. **Document precedence/versioning.** The new normative candidate narrows the
   base design only where stated. Independent approval must pin all governing
   document digests and resolve any disagreement before activation. A future
   SCC/typed-edge, import or offline-wire change needs an explicit compatibility
   decision; unchanged `version=1` alone cannot make it compatible.
2. **Cryptographic assumptions.** SHA-256 equality relies on collision and
   second-preimage resistance and identifies bytes, not origin. Git SHA-1 OID
   syntax/existence is not source selection approval. File/source/proof hashes
   need authenticated material selection; keys and signatures need validated
   encodings and the approved verifier, not regex membership.
3. **Authority identities.** Two reviewer IDs and keys must map to genuinely
   separate approved reviewers through a record maintained outside this
   bundle. Ubuntu archive/CI keys are not automatically Observer reviewers.
   Revocation, rotation and test-root separation need independently rooted
   records; no key list can enroll itself.
4. **Offline freshness.** Checkpoint equality proves consistency only.
   Independent state supplies the minimum sequence, accepted revocation head,
   freshness/expiry policy and trusted time. Local clock/checkpoint rollback,
   unavailable state and restored snapshots must fail closed. Exact age and
   recovery policy need acceptance outside these declarations.
5. **Nonce/receiver state.** A 64-hex value does not prove 256 random bits.
   Trusted generation, unique run association, durable atomic single use,
   reboot/restore treatment and receipt identity must be established elsewhere.
6. **Build inventories.** Available builtin/frozen modules include bootstrap
   and error paths, with container bindings derived from accepted build
   inputs/recipe/output. Host self-enumeration or locally generated inventory
   is comparison evidence, never the expected independent inventory.
7. **Actual resolution.** The approved CPython build, ABI/suffix set, fixed
   path configuration, native extension initialization, loader search/context,
   CPU variants and permitted dlopen/data behavior must match the closed
   profile. Unsupported machinery refuses; it receives no runtime fallback.
8. **File and principal protection.** UID/GID/mode declarations do not prove
   effective ownership, namespace mapping, ACL/xattr policy, protected ancestry,
   immutable mounts or continuous identity/content protection. An accepted
   integrity profile must define and demonstrate those mechanisms, including
   replacement between verification and execution. Role overlaps are not proof
   that a named regular file has the asserted binary/source content.
9. **Scope authentication.** Host anchor/source IDs and authentication-evidence
   hashes require independently accepted producer/evidence semantics. Allowed
   procfs option strings and `HOST_INITIAL_COMPLETE` do not prove actual full
   visibility; omission of an option cannot serve as evidence of its absence.
10. **Observation limits.** A sandbox with no capabilities may be unable to
    inspect the host completely. Permissions/missing PIDs and namespace gaps
    remain incomplete; adding privileges or weakening sandboxing is not an
    authorized remedy in 3A. Positive observations survive later failure.
11. **Opaque inputs.** Known metadata digest exclusions examine declared
    digest fields only. Hidden references inside profiles/proofs/ELF/archive
    bytes need later bounded content verification. A digest cannot inspect
    those bytes or prove reconstruction/absence of cycles in another graph.
12. **Output authenticity.** MATCH in synthetic offline output is a permitted
    advisory declaration, not authenticated artifact verification. No source
    pin, report hash, successful pipe framing or frozen Python result object
    supplies an independently authenticated producer or export receipt.

## Test evidence and approval exit conditions

The source suite contains positive/negative tests for every structural area
above: `test_proposed_manifest_full_synthetic_inventory`, closed-record and
typed-reference mutations; transform/provenance cycles; ancestry/role closure;
source-package parents/native shadowing; exact policy/key/limit records;
canonical/signing-domain vectors; result counter/window/sequence retention;
cross-document substitutions; frame size/EOF/deadline; immutable blocked labels;
and standard parser refusal. These are a test inventory, not a claim that any
particular CI or local run passed. Actual HEAD test/coverage/CI evidence belongs
to the review package and PR report.

A focused source-baseline experiment prepended a space and appended LF to the
offline fixture, then called `parse_offline_result`: it accepted the bytes,
retained `raw != canonical`, and kept `authorization=UNVERIFIED`,
`acceptance=UNVERIFIED`, `qualification=BLOCKED`. No target payload, host scan,
physical procedure or production observer was executed. This confirms the
CA-P2-01 distinction without treating it as an authorized wire change.

Independent approval must accept or revise the whole pinned wire profile,
decide the ELF model from authenticated exact-target materials, disposition
every P1/P2 decision above and record reviewer authority outside the submitted
bundle. Codex cannot supply its own independent approval. Standard parser
activation needs a later reviewed change with the full existing negative
matrix, accepted canonical vectors and no fallback/activation flag. That change
must not confuse structural acceptance with signature/trust/artifact/runtime
verification or production qualification.

No result of this review authorizes Stage 3B execution, trusted launcher or
privileged helper activation, Root Proof changes, enrollment, Stage 10 or LIVE.
The following status remains mandatory:

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
