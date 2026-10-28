# Stage 9 / 3A: review disposition and remaining approval gate

Base design: `236fa623e1e88eaedf634d59d8bd3de8d38452de` (PR #3214).
Initial implementation reviewed at `1dcf1c9df46154e7729b7fb2e6fd45c2920a1fb0`
in PR #3215. The following is a contract-review disposition, not independent
approval, host qualification, or a historical test result for a later HEAD.

## P1/P2 disposition

The [complete proposed normative profile](stage9_observer_3a_schema_proposal.md)
now defines exact fields, types, variants, limits, relationships and failure
rules for all previously unspecified records. It is normative **within the
proposal under review**, not an independently approved replacement contract.

| Finding | Proposed correction | Approval status |
| --- | --- | --- |
| P1-1: manifest nested records | Full target/source/file/provenance/transform/runtime/import/embedded schemas; explicit image root, inventory closure, ordered projections and two structural DAGs. | Complete candidate; independent review pending. |
| P1-2: acceptance policy | Full target, exact ordered source inventory matching, 22 exact limit constants, reviewer/key uniqueness, threshold 2, version/checkpoint/window and document bindings. | Complete candidate; independent review pending. |
| P1-3: observer result | Ten named bounded counters with event units and overlap rules, three exact booleans, consistency/sequence rules, missing-proof reasons and immutable qualification blocks. | Complete candidate; independent review pending. |
| P2-1: diagnostic codes | Exactly 17 refusal codes; status words are excluded from that enum. | Normative proposed vocabulary; independent review pending. |
| P2-2: dependency cycles | Precisely directed provenance graph, separately checked ELF graph; no claim that a digest or structural DAG inspected opaque artifact references. | Scope distinction explicit; content verification remains 3B. |
| P2-3: identifiers | Full token grammar `[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}`; rejects CR/LF/ESC/DEL/all controls, whitespace and aliases; no normalization. | Explicit tightening, compatible with existing synthetic IDs. |
| P2-4: canonical acceptance | The entire envelope must already be canonical; payload digest and `domain + 32 digest bytes` are unchanged. | Explicit wire tightening; public Windows wrapper precedent documented. |

This table distinguishes a technically specified correction from its approval.
It does not label a newly written security decision independently accepted.
There is no remaining intentionally unspecified field inside the candidate
profile. Unsupported transform kinds, cyclic ELF profiles or broader source
catalogues fail closed instead of receiving implicit semantics.

## Working implementation boundary

Standard parse_manifest, parse_acceptance_policy, parse_observer_result and
parse_report_frame remain refusing entrypoints until the contract is reviewed
and independently approved in a later change. The user-authorized, separately
labelled working commit can expose explicit parse_proposed_* APIs, proposed
synthetic bundle checks and synthetic sequence comparison for review tests.
Their immutable result is permanently synthetic=true,
review_status=PROPOSED_UNAPPROVED and authorization=UNVERIFIED. No environment
flag, trust-on-first-use root, production launcher or evidence-export path
activates these records as authority.

The structural candidate checks declared file/source/transform references,
identifier/path uniqueness, provenance and ELF cycles, import/embedded bindings,
manifest/launch projections, policy source matching, envelope/root/checkpoint/
sequence/window syntax and run/result nonce/boot/scope/digest consistency.
Those checks never prove artifact contents, actual build inventory, Python/ELF
resolution, runtime enforcement, signatures, reviewer independence, a trusted
clock/revocation head, authenticated producer/channel or a physical host scope.
A SHA-256 equality proves neither provenance nor examination of hidden content.
The proposed static resolver additionally checks its file/import-to-root work
bound before nested comparisons: `(files + imports) * import_roots <= 65536`.
This defensive domain ceiling gives LIMIT_EXCEEDED rather than unbounded
comparison work; it introduces no new field in the 22-field policy limits.

Contract work performs no host scan, network/subprocess/host I/O, inspected-code
execution, privileged helper/launcher action, sudo/root/capability operation,
Stage 41B installation change or modification of existing --evidence semantics.
No CI threshold, Root Proof, enrollment, Stage 10 or LIVE condition is relaxed.

## Qualification and readiness

Every proposed synthetic result keeps runtime_attestation=MISSING,
host_pid_namespace=UNVERIFIED, qualification=BLOCKED,
full_confirmation=false, coexistence_authorized=false,
preflight_exception=false, physical_test_status=PHYSICAL_TEST_PENDING,
status=IN_PROGRESS and production_status=PRODUCTION_NO_GO.
Offline acceptance remains UNVERIFIED. Syntactically valid signatures, zero
observed matches or a complete transport frame cannot change these blocks.

Stage 3A is **not closed before independent contract review and approval**.
The remaining P1 approval gate is acceptance or revision of all three complete
candidate record groups, followed by a separately reviewed standard-parser
activation change and its negative tests. Actual artifacts, signature/trust
context verification, runtime attestation and authenticated host PID scope
remain distinct Stage 3B/qualification work. Synthetic fixtures are never those
proofs. New HEAD test/coverage/CI results belong in the PR report; earlier 347
passing tests and 98% coverage are historical only.
