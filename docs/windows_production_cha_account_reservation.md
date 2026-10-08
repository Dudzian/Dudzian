# Stage 9: initial candidate account reservation

This child follows the merged CHA logical-operation boundary (#3082). Its frozen
contract is `architecture/cryptohunter_product_architecture/stage9_account_initial_binding_contract.json`.
Parent, upstream CHA, and historical M0.5 contract bytes remain unchanged.

`VerifiedCHALogicalOperation` → durable `INITIAL_BINDING_COMMITTED` →
`VerifiedAccountGenesisInitialBinding` is the complete boundary. The installed
`CryptoHunterAccountAuthority` is implemented by
`deployment.windows_production_cha_account_reservation`; its fixed state and
nonblocking lock sit beside the installed CHA operation state. The public entry
points accept only the upstream capability, never a caller-selected ID, timestamp,
state path, request DTO, or source tuple.

The internally derived closed RFC8785/JCS UTF-8 request has exactly these fields:
`schema_version`, `request_domain`, `purpose`, `environment`, `pdsa_trust_domain`,
`intended_action`, `product_scope`, `logical_operation_id`,
`provisioning_operation_id`, `account_id`, `reservation_relation`,
`root_proof_handoff`, and `cha_state_sha256`. Its SHA-256 covers every field,
including the internally minted candidate account. Environment comes from the
exact guarded CHA/LPPI source; the source digest covers complete retained CHA
bytes, including signed LPPI lineage. No timestamp is part of this semantic request.

The child preserves all seven semantic slots already frozen by the historical
M0.5 operation-identity/request-binding contract. That artifact's exact schema
remains DESIGN_BLOCKED in its original bytes; this child freezes the exact Stage9
schema. It selects the historical candidate request-domain literal
`cryptohunter.m0.5.account-genesis-request.v1` and reuses the frozen issuer's
`ACCOUNT_GENESIS_BOOTSTRAP`, `CryptoHunter`, and `EXACT_OPERATION_ACCOUNT`
vocabulary. No earlier semantic requirement is removed.

The closed `root_proof_handoff` object binds timing
`FROZEN_AFTER_INITIAL_BINDING_BEFORE_PREPARED` and authenticity model
`CHA_AUTHENTICATED_EXACT_INITIAL_BINDING_DIGEST_AND_REFERENCE`. This is an
expectation for downstream authenticated CHA resolution; it neither creates a
proof ID nor issues or admits a proof. Satisfying this handoff later does not
change the initial request.

INITIAL_BINDING retains exact request bytes and their fingerprint, full guarded
CHA state, candidate account, and the assigned internal UTC instant. Its separate
JCS integrity fingerprint detects ordinary field corruption. Neither public hash
provides authentication, authority, freshness, or rollback protection; coherent
privileged rewrites remain possible. No additional reservation ID is minted:
`(pdsa_trust_domain, logical_operation_id)` is the recovery key. This is an
installed initial singleton, not a general subject-to-account business rule.

One captured internal UTC instant feeds `bot_core.uuid7`: integer uint48 Unix
milliseconds and independent CSPRNG 12+62 bits, version 7 and RFC variant 10.
There is no second UUID implementation. After exact upstream requalification and
snapshot capture, the owner captures UTC, mints the candidate in memory, builds
the account-bound request, calculates its fingerprint, and atomically commits
INITIAL_BINDING. Capability issuance follows successful durable commit. A crash
before that commit publishes no candidate or request; retry may make the first
reservation. Once durable state exists, retry/restart recovers the exact retained
account, request bytes, and fingerprint without reminting. The loader requires a
retained binding and never mints. Changed request/source or a second operation
fails closed. Every property/guard requalifies current upstream CHA, then rereads
and compares exact retained bytes and all identity/request bindings.

Safe-path/reparse checking and the durable publisher are existing hardened
primitives. Dedicated locks use POSIX nonblocking flock or Windows nonblocking
msvcrt locking. State/lock reject hardlinks and nonregular files; state is bounded
at 16 MiB and canonical. POSIX fences temporary/published files and the directory;
Windows publication uses atomic MoveFileExW with WRITE_THROUGH and file flush.
A failed publication rereads retained state and raises without issuance. Subsequent
retry/load reasserts all fences for the retained winner. Temporary files are
removed. The machine-protected installed authority directory remains a precondition.
Privileged coherent offline rewrites after restart are not cryptographically
excluded: **disk rollback protection = NOT PROVIDED BY THIS LAYER**.

Candidate account ≠ genuine account. Reservation ≠ authorization.
INITIAL_BINDING ≠ PREPARED and ≠ final COMMITTED AccountGenesis.
Root-proof admission and external freshness CAS remain required downstream.
There is no root proof consumption, membership, device identity, Protected
Freshness, Secret Resource, external Windows handoff, or Stage 10 lifecycle here.
The next separate PR is root-proof admission / PREPARED AccountGenesis.

Runtime/durability tests use TEST_ONLY NCrypt/TBS simulation with real ECDSA,
upstream provenance guards, canonical records and local filesystem fences.
Architecture tests check frozen hashes, scope, API, current readiness, and semantic
parity against the historical required slots. Negative parity cases remove each
slot or mapped field from the request/fingerprint/schema. Runtime tests reject
account disagreement even after recomputing public hashes, scope/handoff mutation,
unknown fields, and the former six-field request; retry and both sides of the
durable commit boundary preserve the required publication/recovery semantics.
Serial licensing CI discovers `test_cha_account_reservation*.py` via the existing
`test_cha_*.py` selector. Quality coverage also includes both production modules.
Hosted tests do not establish physical Windows/TPM qualification, and no legal
production enrollment is performed. Readiness remains false / NOT_READY.

## Validation of the semantic request correction

Local hosted Linux serial validation: 492 passed, zero failures/errors/skips,
in 28m19s. This consists of 35 reservation runtime/durability cases, 25 child
architecture cases, 358 cases from the four required historical contracts
(request binding 32, physical persistence 61, independent issuer 224, root-proof
admission 41), and 74 upstream CHA runtime/durability regression cases.

Ruff baseline and expanded E/F/B/I, formatting, mypy full configured 268-file
scope for Linux and win32, and git diff --check pass. Blocking Semgrep scans
977 targets with five existing rules: zero findings/errors. Checksum-verified
Betterleaks v1.9.0 working-tree scan reports zero leaks with unchanged baseline.
Changed executable-line coverage against the pre-correction HEAD
`a3c531ed7594b43eebf55561853289265f7b5f0e`: 5/5 (100%). Whole-PR security
diff coverage against its base `a28a5d6957182dce8c119f97943dd9ab067c970b`:
246/272 (90.44%). Both retain the 70% gate. Capability line/branch coverage
is 100%; owner line coverage is 88.60% and combined line/branch coverage is 85%.
All 60 protected parent/upstream/historical/status/provenance/UUID artifacts
remain byte-identical to the pre-correction HEAD. No gate or suppression changed.

This is hosted simulation and Linux filesystem evidence. No physical Windows/TPM
qualification or legal production enrollment was performed, and no downstream
readiness was promoted. Remote PR CI results are separate from local checks.
