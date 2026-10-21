# S9-RPI-34 — TEST-only cross-role denial matrix

**PURE DECLARED TEST CONTRACT · BLOCKED · NOT_QUALIFIED · PRODUCTION NO-GO**

## Scope

Previous Stage 9 diagnostics check capability hints and untrusted service metadata,
not actual permission denial between installed principals. This addition defines
the **negative test coverage** that a future, separately reviewed Linux TEST-only
fixture must attempt. It does not perform system calls, create accounts, install
units, change Linux permissions, use sudo or authorize production.

The six abstract roles are sourced from the existing preflight:
`root_source`, `history_source`, `cca`, `root_signer`,
`history_signer`, `issuer`.

Every actor must attempt `read`, `write` and `traverse` against each
of the **other five roles' protected TEST resources**: 6 × 5 × 3 = **90**
ordered, distinct negative cases. These names are not approved systemd
units, real paths, numeric identities or production installation pins.

## Strict declarative interface

The Python function `inspect_declared_matrix(record)` accepts exactly:

- `schema`: `CryptoHunter.Stage9TestOnlyCrossRoleDenialMatrixV1`
- `snapshot`: `SYNTHETIC_TEST_INPUT` or `UNTRUSTED_LIVE_OBSERVATION`
- `cells`: at most 90 dictionaries with exactly the fields
  `actor`, `owner`, `operation`, `observation`
- Each `observation`: `CLAIMED_DENIED`, `CLAIMED_ALLOWED`, `NOT_OBSERVED`

Unknown, duplicate, self-owned, malformed, oversized or secret-bearing inputs
are rejected with a fixed reason code, without reflecting input data. Missing
cells are not considered a complete negative test.

This **synthetic** example creates no host observation:

```python
from deployment import stage9_native_linux_test_denial_matrix_contract as matrix

claims = {
    "schema": matrix._SCHEMA,
    "snapshot": "SYNTHETIC_TEST_INPUT",
    "cells": [
        {
            "actor": actor,
            "owner": owner,
            "operation": operation,
            "observation": "CLAIMED_DENIED",
        }
        for actor, owner, operation in matrix.expected_cases()
    ],
}
result = matrix.inspect_declared_matrix(claims)
assert result["qualification"] == "NOT_QUALIFIED"
assert result["matrix"]["observation"] == "UNTRUSTED_DECLARED_DENIALS_ONLY"
```

The module's no-argument CLI prints only a blocked, no-observations summary
and always returns exit code **2**:

```bash
python -m deployment.stage9_native_linux_test_denial_matrix_contract
```

## Refusal and trust boundaries

Every result remains `BLOCKED / NOT_QUALIFIED / production_emission=BLOCKED /
PROPOSED_NOT_APPROVED`. The output contains only fixed labels and aggregate
counts, never user-supplied filenames, credentials, UIDs, private data or
exception text.

- Any `CLAIMED_ALLOWED`: `DECLARED_CROSS_ROLE_ACCESS`.
- Any missing or `NOT_OBSERVED` cell: `INCOMPLETE_TEST_DENIAL_MATRIX`.
- Invalid shape, role, duplicate or unknown outcome: `INVALID_TEST_REPORT`.
- Exactly 90 `CLAIMED_DENIED` cells: `UNTRUSTED_DECLARED_DENIALS_ONLY`
  and `NO_INDEPENDENT_HOST_ATTESTATION` — **never a trusted PASS**.

A future physical TEST fixture must independently bind Linux processes to
effective UID/GID/supplementary group membership, pin each protected TEST
resource and full path, check actual syscalls and distinguish permission denial
from missing paths, and test positive owner access as a control. This pure
contract does **none** of those things. Even a complete `UNTRUSTED_LIVE_OBSERVATION`
declaration cannot supply effective denial proof, protected install pins,
authenticated `SO_PEERPIDFD`, native keyring restrictions, PostgreSQL committed
close, CCA durability or production Root Proof authorization.

## Hardware and CI

No extra computer is required for **coding or review** of this pure contract.
Physical negative access tests **will require the Dell after Ubuntu has been
installed** and after a TEST-only fixture has been separately reviewed.
Do not provision the proposed production service names or handle real keys
based on this matrix.

S9-RPI-34 adds regression coverage and existing Ruff/Stage 9 CI inventory
entries without changing any security, release or test gates. Private-repository
CI failures do not count as executed or passed tests. No SSH/Tailscale,
server conversion, sudo, network access or filesystem mutation is added.
