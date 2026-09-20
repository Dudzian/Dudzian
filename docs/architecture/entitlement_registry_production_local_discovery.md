# EntitlementRegistry `PRODUCTION_LOCAL` — discovery result

## Result

`ENTITLEMENT_REGISTRY_SEMANTIC_CONTRACT_INSUFFICIENT`

This is a `CURRENT_TREE_ONLY` finding.  Its provenance remains:

* `classification = UNKNOWN`
* `finding_scope = CURRENT_TREE_ONLY`
* `formal_project_advancement = WITHHELD`

No PostgreSQL entitlement-registry provider is implemented by this finding.
In particular, it does not change either of the existing negative runtime
claims:

* `ROOT_PROOF_ISSUER_IMPLEMENTED = false`
* `PRODUCTION_LOCAL_RUNTIME_AVAILABLE = false`

## Artifacts inspected

The frozen architecture selects a dedicated PostgreSQL service/container and
PostgreSQL `SERIALIZABLE` conditional CAS for the production-local registry.
It also assigns the runtime port to `EntitlementRegistryProvider`.  The current
Python protocol, however, types the subject as `str`, both state values as
`object`, the CAS result as `bool`, and every read result as `object`:

```python
def authoritative_state(self, subject_id: str) -> object: ...
def compare_and_swap_bind(self, expected: object, successor: object) -> bool: ...
def historical_state(self, subject_id: str, generation: int) -> object: ...
```

The semantic issuer artifact describes a bootstrap entitlement at prose/data
contract level: an issuer-generated entitlement identity, a positive immutable
generation, `UNBOUND` or `BOUND(...)` state, `ACTIVE`/`REVOKED`/`SUPERSEDED`
lifecycle, and exact-retry behavior.  It does not freeze closed runtime record
types or map those concepts onto the three protocol arguments and results.
The implementation foundation also explicitly records that the semantic
issuer, `NEW_BIND` flow, issuance state machine, and PostgreSQL entitlement
registry are not implemented.

The physical persistence artifact cannot fill that gap.  It labels its local
transaction contract as abstract with the exact schema and adapter unselected;
its CAS generation contract concerns an AccountGenesis-specific external
multi-lineage anchor document, whose writer is unavailable, rather than a
closed `EntitlementRegistryProvider` state record.

## Missing frozen invariants

Implementing a provider now would require inventing security semantics.  The
following items must be frozen before implementation:

1. The exact immutable runtime classes returned by `authoritative_state` and
   `historical_state`, including every field, closed enum, canonical encoding,
   digest, and validation rule.
2. The exact registry subject and lookup cardinality: whether `subject_id` is
   the authoritative entitlement ID, a lookup handle, a composite including
   environment/trust domain, or another identity, plus its canonical format.
3. The complete initial/genesis record and the authority operation that creates
   it, including initial generation and lifecycle/state combinations.
4. Whether entitlement supersession and `UNBOUND -> BOUND` are generations of
   one chain or separate lineages, and the exact legal transition graph.
5. The exact predecessor value accepted by `compare_and_swap_bind`: full state,
   generation, revision, digest, decision identity, or a defined combination.
6. The exact successor value, including which party creates entitlement,
   generation, registry revision, root-proof identity, and any signed bytes.
7. The exact relationship between `entitlement_generation`, authority
   `cas_revision`/`authoritative_state_revision`, and historical lookup
   `generation`.
8. Typed outcomes for stale predecessor, missing subject, corruption,
   unavailability, PostgreSQL serialization failure (`40001`), and exact replay;
   the current `bool` result cannot preserve all required classifications.
9. Exact lost-response/idempotency ownership and equality: which complete tuple
   identifies an exact retry and what value/evidence the registry must return.
10. Historical semantics for revoked and superseded generations, including
    whether a generation stores lifecycle, binding decision, or both, and what
    `historical_state` returns for a missing generation.
11. The authoritative boundary between registry-local immutable history and
    the separate authenticated-history/checkpoint roles, including which
    digest or reference (if any) belongs in registry state.
12. Provisioning and migration APIs and their typed credentials: a runtime-only
    three-method protocol cannot create the pre-account entitlement while
    preserving the required authority separation.

Until those invariants exist, database keys, uniqueness constraints, CAS SQL,
history-chain checks, corruption validation, and retry behavior cannot be
proven to implement the frozen meaning rather than a newly invented one.

## Dependency discovery

The repository bootstrap mechanism is an editable install with project extras
(`python -m pip install -e ".[dev]"`).  The manifest currently declares
SQLAlchemy with asyncio support and `aiosqlite`, but no PostgreSQL DBAPI driver
(`psycopg`, `psycopg2`, or `asyncpg`).  Consequently there is no repository-
selected PostgreSQL driver to use or version to verify.  A driver selection
must be added to the reviewed project manifest together with the concrete
adapter; choosing one in this stopped stage would pre-empt the missing frozen
contract and provider design.
