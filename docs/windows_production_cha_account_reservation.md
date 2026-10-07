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

The internally derived closed JCS request binds the initial installed lifecycle
purpose, trust domain, ago, prvop, and a CHA-state SHA256 integrity fingerprint.
INITIAL_BINDING separately retains its exact request bytes, full guarded CHA
state (including retained signed LPPI lineage), candidate account ID, and captured
UTC instant. A JCS fingerprint of the entire binding also detects ordinary
field corruption, including a valid but changed account UUID. Public hashes are
not authority and cannot prevent coherent privileged rewrites. No additional reservation ID is
minted: `(pdsa_trust_domain, logical_operation_id)` is the recovery key. This is
an installed initial singleton, not a general subject-to-account business rule.

One captured internal UTC instant feeds `bot_core.uuid7`: integer uint48 Unix
milliseconds and independent CSPRNG 12+62 bits, version 7 and RFC variant 10.
There is no second UUID implementation. Minting in memory does not publish an
account. One atomic durable INITIAL_BINDING is the first retained state. Exact
retry returns the retained winner, never a second account candidate. The loader
requires a retained binding and never mints. Changed request/source or a second
operation fails closed. Every property/guard requalifies current upstream CHA,
then rereads and compares exact retained bytes and all identity/request bindings.

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
Architecture tests check frozen hashes, scope, API and current readiness.
Serial licensing CI discovers `test_cha_account_reservation*.py` via the existing
`test_cha_*.py` selector. Quality coverage also includes both production modules.
Hosted tests do not establish physical Windows/TPM qualification, and no legal
production enrollment is performed. Readiness remains false / NOT_READY.

## Validation of this implementation

Local hosted Linux validation: focused reservation/boundary suites 69 passed;
additional path/provenance cases 7 passed (all 32 new runtime/durability cases
covered across these runs); upstream CHA/LPPI runtime/native/durability 249 passed;
Stage9 architecture and CI contracts 221 passed. Architecture suites overlap
between runs; these counts are not a unique aggregate.

Ruff baseline and expanded E/F/B/I, formatting, mypy full configured 268-file
scope for Linux and win32, and git diff --check pass. Blocking Semgrep scans
977 targets with five existing rules: zero findings/errors. Checksum-verified
Betterleaks v1.9.0 working-tree scan reports zero leaks with unchanged baseline.
Security diff coverage: 255/281 executable lines (90.75%), threshold 70 retained.
Capability line/branch coverage is 100%; owner line coverage is 88.50% and
combined line/branch coverage is 85%.

This is hosted simulation and Linux filesystem evidence. No physical Windows/TPM
qualification or legal production enrollment was performed, and no downstream
readiness was promoted. Remote PR CI results are separate from local checks.
