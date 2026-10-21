# Stage 9 — S9-RPI-36: TEST-only native Linux fixture readiness declarations

**PURE CONFIGURATION CHECK · NOT INSTALLED · NOT_QUALIFIED · PRODUCTION NO-GO**

## Purpose and trust boundary

S9-RPI-34 defines 90 directed cross-role denial cases among six Source/CCA
and signer/issuer TEST role labels. S9-RPI-35 adds 18 positive owner-control
cases and treats missing targets and non-permission failures as inconclusive.

Before running these tests, a physical **TEST-only** fixture design must
have separate low-privilege principals and correctly protected resources.
S9-RPI-36 is a small **declarative topology consistency validator**: it checks
the shape of six intended TEST role entries and basic least-privilege claims.
It **does not** create host accounts, inspect UID/GID state, open files,
read keyrings, execute Linux syscalls, use sudo, install units, run PostgreSQL,
provision credentials or approve any installation.

The labels describe a proposed TEST fixture; they are **not** production
principal names, service assignments or a binding to the Dell computer.

## Fixed closed layout

The six role names come from the existing Stage 9 108-case classifier.
The fixture claims, for each role, an independent, non-root numeric UID and
primary GID. `effective_gids` must be exactly the single-element list
containing that primary GID — a **strict proposed TEST profile**, not a
generic Linux requirement. Reused UIDs or intersecting effective groups
between roles are rejected. Identifiers are not printed.

Each role's **declared** protected TEST directory must be owned by that
role's UID/GID and have mode `0700`. Its **declared** regular TEST data file
must have the same owner and mode `0600`. Four explicit true/false claims
must each be strictly `true`:

- `symlink_free_claim`: no link in the future protected path chain;
- `acl_absent_claim`: no unexpected POSIX ACL entries;
- `unprivileged_claim`: planned actor processes cannot override DAC;
- `private_parent_chain_claim`: no publicly traversable ancestor chain
  inadvertently bypasses the protected directory boundary.

All four are **claims**, not verified properties. The validator cannot
inspect a real inode, assess capabilities, validate an ACL or distinguish
separate UIDs sharing another Linux authority. The eventual physical fixture
must independently verify these things and perform actual syscall attempts.

## Closed input and failure precedence

`inspect_declared_fixture(record)` accepts exactly:

- `schema`: `CryptoHunter.Stage9TestFixtureReadinessV1`;
- `snapshot`: `SYNTHETIC_TEST_INPUT` or
  `UNTRUSTED_LIVE_OBSERVATION`;
- `roles`: a list of up to six unique, closed role declarations.

Each entry contains the precise 14 documented fields in the module:
`role`, `uid`, `gid`, `effective_gids`,
`directory_owner_uid`, `directory_owner_gid`, `directory_mode`,
`file_owner_uid`, `file_owner_gid`, `file_mode` and the four
boolean claims listed above. Reject extra/unknown keys, duplicate role
labels, wrong types, malformed group lists and oversized input.

Refusal codes use the following priority after strict shape validation:
`DECLARED_TEST_RESOURCE_OR_PRIVILEGE_GAP` (unsafe mode, owner, UID or
assertion), `DECLARED_ROLE_IDENTITY_COLLISION`,
`INCOMPLETE_TEST_FIXTURE`.
Malformed input is `INVALID_TEST_FIXTURE` with zero partial observations.

For all six self-consistent declarations the ONLY output is
`UNTRUSTED_TEST_LAYOUT_DECLARATIONS_ONLY` with
`NO_INDEPENDENT_PROCESS_OR_FILESYSTEM_EVIDENCE`. It is not an installation
approval, proof of group isolation, valid cross-role denial or Stage 9 GO.

Every result is permanently `BLOCKED / NOT_QUALIFIED /
production_emission=BLOCKED / topology=PROPOSED_NOT_APPROVED`. The report
includes only fixed status labels and aggregate counts, never the submitted
UID/GID numbers, directory locations, group names or free-text values.

## How to use

From a reviewed checkout, the default CLI prints the blocked no-fixture
summary and exits with **code 2**; it does not inspect the host:

```bash
python -m deployment.stage9_native_linux_test_fixture_readiness
```

A future, independently reviewed TEST configuration generator may call
`inspect_declared_fixture(...)` in-process with isolated numeric TEST
identities. It must not mistake synthetic `UNTRUSTED_LIVE_OBSERVATION`
strings for genuine Linux host provenance. **Do not save real credentials,
copy production keys or fill the declaration with personal host paths.**

## Next native Linux work / hardware boundary

The planned Dell Latitude will need native Ubuntu for **physical** isolation
checks. A separate TEST-only fixture design and operator review must first
specify temporary account creation, safe local resource layout, full path and
inode/ACL pinning, effective UID/GID and capability verification,
positive owner-control attempts, negative cross-role operations, artifact
handling, reproducible cleanup and restoration of any changed host settings.
This PR implements none of those actions and should not run against a
production installation.

**The Dell is not needed to review or merge this pure contract, but it WILL
be needed for actual native physical denial tests.** No SSH, Tailscale,
headless server conversion, sudo, systemd changes, CI bypass or credentials
are added. Private-repository CI limitations do not turn unexecuted tests
into successful validation.
