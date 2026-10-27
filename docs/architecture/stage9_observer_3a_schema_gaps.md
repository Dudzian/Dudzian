# Stage 9 / 3A: schema gaps and fail-closed implementation

Base: `236fa623e1e88eaedf634d59d8bd3de8d38452de` (PR #3214), verified against
GitHub `main` before implementation. This document proposes a correction; it
is not an independently approved schema or trust anchor.

## Blocking specification gaps

The approved [design](stage9_trusted_observer_contracts_and_implementation_plan.md)
requires exact fields at every level, but does not define all wire records:

| Priority | Location in the approved design | Missing normative definition |
| --- | --- | --- |
| P1 | 3.2, manifest table | Exact fields/types/variants for full `target`, `sources`, `files`, `transforms`, and `runtime`; source-kind literals and provenance/reference edges. |
| P1 | 3.3, acceptance policy | Shapes and matching rules for `allowed_sources`, `limits`, and the full policy `target`. |
| P1 | 3.5, observer result | Exact `counters` and `observations` fields, units, domain bounds, and consistency relationships. |
| P2 | 3.1 and 8 | A normative exhaustive reason-code enum. Threat examples include both refusal codes and status words. |
| P2 | 3.4, metadata back-references | The boundary between structural acyclicity and discovering a forbidden reference inside opaque profile/artifact bytes. Digests alone cannot establish the latter. |

No P0 contradiction was identified in the inspected sections. The issue is an
unfinished wire schema. Accepting an arbitrary nested dictionary or choosing
new field names and policy matching rules in code would violate the requested
closed contract and the instruction not to invent security semantics.

## Smallest proposed correction

Approve one normative exact-field addendum to sections 3.2, 3.3 and 3.5 before
accepting those records. A [concrete candidate addendum](stage9_observer_3a_schema_proposal.md)
is supplied for review only and is not used by any accepting validator. It must settle the following narrow decisions:

1. Choose the full target record, including exact CPython package/build/ABI/profile
   keys, and define its projection to the already specified launch target.
2. Enumerate each source variant and exact artifact/proof reference. Specify
   whether `allowed_sources` repeats the selected source records exactly or is
   a larger catalogue with explicit digest-and-version matching rules.
3. Define regular/directory records, permission representation, provenance
   variants, directory size/provenance, and the image ancestry boundary.
4. Enumerate transform kinds and their exact parameters. Any unlisted kind
   remains unsupported. Distinguish an output assertion from a dependency edge
   so that an output/provenance pair does not create an artificial cycle.
5. Define runtime fields and ordered projections to interpreter/ELF/import
   launch records. Specify which graphs must be acyclic and which role lists
   cover each file. Keep opaque artifact-content checks in Stage 3B.
6. Give the policy limit keys and state whether a policy can tighten V1 ceilings;
   no field may raise the approved ceilings or choose a fallback implicitly.
7. Define the counter names/units/bounds and observations shape. Specify the
   relationships for positive match retention, instability and incompleteness.
   Keep overlapping counter categories; do not require summing them as PID totals.

For reason codes, the proposed closed initial enum is the 17 explicit refusal
names in section 8, also exposed as `REASON_CODES` in the implementation.
`INCOMPLETE`, `MISMATCH`, `UNVERIFIED`, and `PERMISSION_DENIED` are not silently
promoted from status words into additional refusal codes. The existing
`FORMAT_REJECTED` is used for unfinished schemas; no new diagnostic is invented.
Path validation uses a lexical POSIX profile: absolute paths, no empty/dot/dotdot
components, repeated/trailing slashes except `/`, backslashes or control
characters, and the approved 4096-byte UTF-8 ceiling. This does not read or
resolve the host filesystem. Identifiers retain the stated nonempty ASCII
128-character ceiling without trimming or inventing an authorization grammar.
Linux sandbox UID/GID declarations use 1..2^32−2: zero, the reserved invalid ID
and values that could truncate to zero in a future launcher are refused. This
is a domain restriction on Linux credentials, not a credential-changing action.

## Implemented and deliberately unavailable behavior

`deployment.stage9_observer_contract` implements bounded JSON decoding and
structural byte parsers for acceptance envelopes, canonical launch declarations,
canonical host scope/run contexts and synthetic offline results. It validates
nested known fields, digest/signature representations, the signing preimage,
reviewer uniqueness as a policy fragment, run/scope digest and nonce agreement,
exact launch constants and its internal digest projections. Parsed records hold
immutable bytes and carry permanently `authorization=UNVERIFIED`.

`parse_manifest`, `parse_acceptance_policy`, `parse_observer_result`, and
`parse_report_frame` always refuse, including superficially plausible records.
They check available normative fields and never accept opaque unfinished parts.
They are explicitly not completed validators for these contracts. No manifest
closure, file/source/transform reference graph, interpreter-to-manifest comparison,
acceptance-policy quorum authorization, or completed report is claimed.
`parse_launch_policy` is structural only; file existence, file type, profile
contents, ELF closure, exact build and manifest projection remain unavailable.

`decode_synthetic_frame` checks one bounded U32BE/UTF-8/JSON frame, EOF, process
exit and timing supplied by a synthetic caller. It returns **untrusted bytes**,
not an accepted report. A correctly framed `{}` is valid transport data and is
still rejected by `parse_report_frame`. There is no real receiver, replay store,
process observation, host authentication or authenticated export receipt.
`compare_synthetic_run_context` compares complete declared contexts; supplying
both JSON documents cannot establish an independent trust anchor.

No cryptocurrency/exchange code, inspected installer, historical payload or
privileged helper is executed. No host scan, `/proc` access, network operation,
subprocess, root/capability action, launcher, artifact verification, signature
verification, `--evidence` change, CI threshold change or physical qualification
is introduced by the contract module.

## Qualification and next-stage readiness

Every accepted synthetic offline result keeps `runtime_attestation=MISSING`,
`host_pid_namespace=UNVERIFIED`, `qualification=BLOCKED`,
`full_confirmation=false`, `coexistence_authorized=false`,
`preflight_exception=false`, `physical_test_status=PHYSICAL_TEST_PENDING`,
`status=IN_PROGRESS`, and `production_status=PRODUCTION_NO_GO`.
Acceptance remains `UNVERIFIED`; parsing signatures does not verify them.

Stage 3A is **partial and blocked by the P1 schema gaps**. Its exact-field
parsers and transport model are reviewable, but it is not ready to claim the
requested complete contract implementation or to integrate Stage 3B verifiers.
First approve the missing schemas, complete their validators/cross-record
negative tests, and review that diff. Independent attestation and authenticated
host PID scope remain separate qualification blockers after that work.
