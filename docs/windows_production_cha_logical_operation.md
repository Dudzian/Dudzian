# Produkcyjna initial CHA logical operation

`establish_installed_cha_logical_operation(upstream)` w
`deployment/windows_production_cha_operation.py` konsumuje exact
`VerifiedLPPIAuthenticatedProvisioningOperation`, przypisuje wewnętrzne
`ago_<canonical-lowercase-UUIDv7>` i wydaje `VerifiedCHALogicalOperation`
wyłącznie po durable immutable `prvop↔ago` commit. Account reservation i genesis
pozostają następnym odrębnym etapem.

## Audyt źródła i wcześniejszych kontraktów

Brakującą granicą po #3081 było przejście od committed authenticated LPPI
operation do CHA-owned logical operation. Existing LPPI ma durable `prvop`,
exact PDSA package/subject/enrollment i podpis current ACTIVE successor, ale jego
subordinate contract kończy się na `LPPI_AUTHENTICATED_OPERATION_BINDING_COMMITTED`
i jawnie wyklucza CHA/ago/bijection. External frozen parent już przypisuje CHA
ownership `ago`, mapping registry key i `ONE_TO_ONE_BIJECTION`, assignment przed
account reservation oraz immutable same-source retry.

Historyczne M0.5 dokumenty identity/reservation/uniqueness opisywały brak genuine
operation owner i exact schema. Topology resolution później wybrało CHA jako
issuer/owner/coordinator; Stage9 parent następnie zamroził `ago` syntax i ordering
przed account reservation. Ich wcześniejsze `DESIGN_BLOCKED` i `NOT_FROZEN`
pozostają historycznymi zapisami. Entity `account_id`, logical operation i subject
business uniqueness są osobnymi pojęciami. Ten PR nie domyka reservation,
canonical AccountGenesis request, root-proof closure ani subject/account cardinality.

| Źródło | Zachowane wymagania / supersession |
| --- | --- |
| [`m05_account_genesis_operation_identity_request_binding_contract.json`](architecture/cryptohunter_product_architecture/m05_account_genesis_operation_identity_request_binding_contract.json) | Caller command/account/proof ID nie jest operation identity; same operation zachowuje request. Earlier owner/syntax/timing blockers superseded przez topology i Stage9 parent; exact AccountGenesis request schema pozostaje downstream. |
| [`m05_cryptohunter_account_genesis_reservation_state_model.json`](architecture/cryptohunter_product_architecture/m05_cryptohunter_account_genesis_reservation_state_model.json) | One operation → at most one account candidate i recovery identity pozostają wymaganiami przyszłej reservation/genesis granicy; ten mapping nie ustanawia PREPARED ani genuine account. |
| [`m05_cryptohunter_account_genesis_uniqueness_idempotency_design.json`](architecture/cryptohunter_product_architecture/m05_cryptohunter_account_genesis_uniqueness_idempotency_design.json) | Entity/operation/business uniqueness są oddzielne; ten PR nie wybiera subject/account cardinality ani global account singleton. |
| [`m05_account_genesis_authority_topology_resolution_after_binding_freeze.json`](architecture/cryptohunter_product_architecture/m05_account_genesis_authority_topology_resolution_after_binding_freeze.json) | `ownership_resolution` supersedes earlier owner resolution; sole coordinator to CHA, independent root proof nadal konieczny przed genuine genesis. |

Legacy `external_provisioning.py` zawiera `Stage9AccountGenesisAuthority`,
`ProvisioningRepository`, `Stage9ProvisioningService` i `LPPI_TEST_ONLY` flow.
Legacy reserve sam mintuje `prvop`; `_uuid7` używa float `time.time()*1000`
oraz 48-bit mask. Zwykłe SQLite rows nie wymagają committed LPPI capability.
Tych obiektów nie należy routować do nowej production authority. Są zachowane
dla compatibility/tests. Istniejący `cha_attempt_store.py` dotyczy innej granicy
root-proof attempts i nie jest automatycznym issuerem nowej Stage9 operation.

Reused primitives to corrected shared UUIDv7 i hardened file replacement oraz
cross-process locking profile wcześniejszego LPPI lifecycle. Current source code
jest podstawą audytu; structural Codebase Memory nie był dostępny w tym środowisku.

## Contract i source binding

[`stage9_cha_logical_operation_contract.json`](architecture/cryptohunter_product_architecture/stage9_cha_logical_operation_contract.json)
i
[`stage9_cha_logical_operation_freeze.json`](architecture/cryptohunter_product_architecture/stage9_cha_logical_operation_freeze.json)
pinują exact parent SHA-256 i exact upstream LPPI child SHA-256. Historical bytes
obu kontraktów pozostają niezmienione. Nowy freeze wybiera record schema,
`mapping_generation=1`, UTC millisecond precision, shared instant UUID timestamp,
retained source snapshot, crash/retry i capability reconstruction.

Minimalny derived source tuple zawiera environment, trust domain, exact package
digest, subject, enrollment, `prvop`, LPPI binding generation i exact binding digest.
Durable `lppi_operation_state_raw_hex` zachowuje kompletne exact canonical source
bytes z guarded private LPPI snapshot, w tym ACTIVE signer lineage i signature.
Source tuple jest wyprowadzany, a jego pola nie są nowe caller inputs.
Sam JSON/snapshot/hash/transport binding nie daje authority.

Canonical durable record ma dokładnie dziewięć pól:

```text
schema_version = CHAInitialLogicalOperationV1
status
history
mapping_generation = 1
pdsa_trust_domain
provisioning_operation_id
logical_operation_id
assigned_at_utc
lppi_operation_state_raw_hex
```

Parser wymaga RFC8785 JCS UTF-8 oraz exact field set; odrzuca boolean jako integer,
unknown/missing/duplicate fields, noncanonical encoding, malformed IDs i daty.
UUIDv7 profile używa integer UTC milliseconds i CSPRNG, bez float rounding lub
maskowania invalid clock. Jeden captured assignment instant jest retained i
wiąże exact `assigned_at_utc` z timestamp `ago`.

## Durable lifecycle i recovery

```text
NO_CHA_OPERATION
→ AGO_RESERVED
→ PRVOP_AGO_BIJECTION_COMMITTED
→ VerifiedCHALogicalOperation
```

Production APIs przyjmują tylko upstream capability. Caller nie wybiera `ago`,
zegara, source fields, state path lub signer. Przed dotknięciem lock files
upstream musi przejść production guard. Reserve zapisuje jedyny `ago`, timestamp
i exact upstream snapshot. Commit zmienia tylko status/history. Publication
wymaga trwałego commit oraz niezależnej source i retained-state verification.

| Cut-point | Recovery |
| --- | --- |
| Przed durable AGO reservation | Pierwszy reserve jest bezpieczny; nic nie zostało opublikowane. |
| Po durable AGO reservation | Zachowaj exact `ago`, instant i source; nie mintuj ponownie. |
| Przed mapping commit | Requalify current upstream i finalizuj tę samą pinned reservation. |
| Po commit przed response | Zwróć exact committed mapping po reverification. |
| Lost response | Ten sam exact mapping, zero remint. |
| Restart | Odtwórz upstream jego production loaderem; load committed CHA lub resume reserved przez establish. |

`load_installed_cha_logical_operation(upstream)` nie tworzy state ani nie mintuje
UUID. Incomplete reservation nie ustanawia loader-issued capability. Persistence
failure nie publikuje authority; następny retry rereads durable state przed
wyborem mint/resume.
Retry i load utrwalają te same exact committed bytes przed publication, aby
ponowić physical durability fence po ambiguous earlier replacement failure;
nie zmieniają mappingu ani source.

Fixed `initial-cha-logical-operation.json` jest one-entry registry initial
installed lifecycle. Same exact `(pdsa_trust_domain, prvop)` oraz full source
snapshot zwracają jedyny immutable `ago`. Inny source w RESERVED/COMMITTED jest
conflict; jego `prvop` nie przejmuje winnera i nie wskazuje retained `ago`.
Reverse uniqueness wynika z jednego immutable mapping record w installed owner
namespace; layer nie dostarcza cross-host global registry. Caller nie może
wybrać alternatywnego directory, aby obejść first-winner rule.

`initial-cha-logical-operation.lock` jest nonblocking cross-process exclusion
dla krótkich reserve/commit. Expensive upstream/native requalification odbywa
się poza write lock. Równoległy proces otrzymuje winnera albo BUSY/retry.
Fsync file precedes atomic replace; POSIX directory fsync lub Windows
`MOVEFILE_WRITE_THROUGH` zapewniają replacement durability. Malformed, oversized,
symlink/reparse/hardlink state failuje closed.

`disk rollback protection = NOT PROVIDED BY THIS LAYER`.
Machine directory protection musi pochodzić z installed State ownership.
Independent Protected Freshness / TPM NV pozostaje kolejnym mechanizmem.

## Capability i obecny status

Opaque `VerifiedCHALogicalOperation` wymaga exact type i private issuance registry.
Copy, subclass, forged `object.__new__`, raw row i persistent JSON nie odtwarzają
provenance. `require_verified_cha_logical_operation` i każda property ponownie
weryfikują upstream LPPI oraz exact committed retained bytes, mapping i source.
`source_binding` udostępnia bezpiecznie verified public LPPI transport; jego
samodzielne użycie nie przyznaje account/genesis authority.
`source_tuple` zwraca świeżą kopię ośmiu frozen derived fields po tych samych
current upstream/retained CHA checks; dict pozostaje danymi transportowymi.

Nie powstaje nowy CHA signer/key/secret. Authority wynika z verified upstream,
CHA ownership, durable immutable mapping i verifier-issued provenance.
Końcowa granica to `VerifiedCHALogicalOperation`; account_id nie jest minted,
account reservation/genesis są `NOT_STARTED`, membership jest `BLOCKED`,
Protected Freshness i Secret Resource są `NOT_STARTED`. Windows handoff i
Stage10 nie są uruchamiane.

Canonical current status pozostaje w
[`deployment/stage9_current_status.json`](../deployment/stage9_current_status.json):
Stage9 `IN_PROGRESS`, production provisioning `false`, Windows `NOT_READY`,
legal production enrollment `NOT_PERFORMED`, Stage10 production lifecycle
`NOT_STARTED`. Root material/ceremony status nie jest cofany do historical
parent snapshot. Hosted tests nie zastępują native Windows qualification.
