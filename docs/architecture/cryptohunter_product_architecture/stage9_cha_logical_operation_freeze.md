# Stage 9 — initial CHA logical operation freeze

Status: `FROZEN`. Normatywny subordinate contract to
[`stage9_cha_logical_operation_contract.json`](stage9_cha_logical_operation_contract.json),
a jego exact-byte SHA-256 chroni
[`stage9_cha_logical_operation_freeze.json`](stage9_cha_logical_operation_freeze.json).
Parent external provisioning i upstream LPPI operation contract pozostają
byte-identical. Implementation jest architektonicznie autoryzowana;
`production_provisioning_ready=false`.

Parent SHA-256:
`6538a75d8cf3a6e3cdab8f4a9022277a21d98e1a61923aa8d8eae9b80ae1d67d`.
Upstream LPPI SHA-256:
`f75494b163137374d39d9147b5e328aa30e772d2887255deb8980cb442457ca2`.
Exact subordinate digest znajduje się w freeze JSON, zamiast tworzyć drugie
źródło hash metadata.

Parent zamraża CHA jako owner/issuer `ago_<canonical-lowercase-UUIDv7>`,
`ONE_TO_ONE_BIJECTION`, registry key `(pdsa_trust_domain, provisioning_operation_id)`
i assignment przed account reservation. Ten child domyka exact durable record,
initial `mapping_generation=1`, millisecond UTC, UUID timestamp correspondence,
retained source, dwa trwałe stany i odtworzenie capability. Kończy się dokładnie
na `VerifiedCHALogicalOperation` po `PRVOP_AGO_BIJECTION_COMMITTED`.

## Source i minimalny record

Production input to wyłącznie exact
`VerifiedLPPIAuthenticatedProvisioningOperation` po
`require_verified_lppi_authenticated_operation`. Raw JSON, row, `prvop`, binding,
podpis, copy, subclass i `object.__new__` nie ustanawiają source authority.

Canonical source tuple jest wyprowadzany z reverified LPPI operation:
`environment`, `pdsa_trust_domain`, `pdsa_package_digest_sha256`,
`provisioning_subject_id`, `enrollment_reference`, `provisioning_operation_id`,
`binding_generation`, `lppi_authenticated_operation_binding_digest_sha256`.
Binding digest to SHA-256 exact canonical LPPI payload; sam hash nie autoryzuje.

Durable record zawiera dokładnie `schema_version`, `status`, `history`,
`mapping_generation`, `pdsa_trust_domain`, `provisioning_operation_id`,
`logical_operation_id`, `assigned_at_utc`, `lppi_operation_state_raw_hex`.
Schema to `CHAInitialLogicalOperationV1`, generation to exact integer `1`.
Ostatnie pole zachowuje pełny, exact canonical, guarded LPPI private snapshot:
package/subject/enrollment, generation, binding bytes/digest, signature oraz
ACTIVE signer binding/profile/fingerprint/custody. Nie tworzymy nowego signer
tuple ani dodatkowych kopii jego pól. Snapshot transport sam nie daje authority.
Ten minimalny source binding nie jest canonical AccountGenesis request; ten
ostatni będzie obejmował przyszłe reservation/account/root-proof semantics.

## Trwałość, retry i bijekcja

```text
LPPI_AUTHENTICATED_OPERATION_BINDING_COMMITTED
→ CHA_AGO_ASSIGNED / AGO_RESERVED
→ PRVOP_AGO_BIJECTION_COMMITTED
→ VerifiedCHALogicalOperation
```

CHA przechwytuje jeden internal UTC instant; wspólny `bot_core/uuid7.py` używa
integer Unix milliseconds, unsigned 48-bit range bez maskowania, CSPRNG rand_a
12-bit i rand_b 62-bit, version `7` i RFC variant `10`. `assigned_at_utc` ma
dokładnie trzy fractional digits i exact timestamp UUIDv7. Caller nie podaje
UUID, czasu, source fields ani state path.

Pierwszy durable write przypina `ago`, instant i pełny source snapshot.
Drugi zmienia tylko status/history. Capability może powstać dopiero po durable
commit. Crash po reservation wraca do tego samego `ago`; crash po commit przed
response i lost response zwracają exact retained mapping bez remint. Loader
restores capability wyłącznie z nowo zweryfikowaną upstream capability i existing
committed CHA record; nie mintuje ani nie finalizuje incomplete reservation.

Registry jest one-entry singletonem initial installed CHA lifecycle pod fixed
`initial-cha-logical-operation.json` w tym samym machine state directory co LPPI.
Registry key pozostaje `(pdsa_trust_domain, provisioning_operation_id)`.
Ten sam exact source zwraca jedyny retained `ago`; inny source w RESERVED lub
COMMITTED jest conflict i nie mutuje winnera. Jeden immutable record zapewnia
forward i reverse uniqueness w tym installed owner namespace. Ten layer nie
jest cross-host global registry i nie wybiera subject/account cardinality.

Nonblocking cross-process state lock chroni krótkie reserve/commit. Drogie
upstream/native requalification odbywa się poza write lock. Racing process
obserwuje ten sam winner albo BUSY/retry. File fsync poprzedza atomic replace;
POSIX directory fsync lub Windows `MoveFileExW` z `MOVEFILE_WRITE_THROUGH`
domyka zapis. Unsafe symlink/reparse/hardlink, malformed/noncanonical state,
unknown/missing/duplicate fields oraz oversized state failują closed.
Retry/load ponownie utrwala te same exact committed bytes przed publication,
aby domknąć ambiguous previous replacement fence; nie zmienia mappingu.

`disk rollback protection = NOT PROVIDED BY THIS LAYER`.
Istniejący installer-owned protected machine directory jest persistence
precondition; TPM NV / independent Protected Freshness pozostaje osobną granicą.

## Capability i zakres

`VerifiedCHALogicalOperation` wymaga exact type i private issuance registry.
Public constructor, copy, subclass, `object.__new__` oraz persistent record nie
dają provenance. Każdy consequential use rekwalifikuje upstream LPPI, odczytuje
committed CHA state, sprawdza exact retained bytes, `prvop↔ago` i exact source.
Properties udostępniają `provisioning_operation_id`, `logical_operation_id` i
bezpiecznie zweryfikowany public `source_binding`; `source_tuple` zwraca świeżą
kopię ośmiu derived semantic fields. Oba transport results same nie są authority.

CHA nie tworzy nowego software signing key ani lokalnego root secret.
Authority wynika z current verified LPPI capability, CHA ownership, durable
immutable mapping oraz verifier-issued provenance. Stare M0.5 artefakty zachowują
historyczne bytes/statusy; later topology/Stage9 supersession nie oznacza, że
wcześniej miały frozen owner/schema. Legacy `Stage9AccountGenesisAuthority`,
`ProvisioningRepository` i `Stage9ProvisioningService` pozostają compatibility /
`LPPI_TEST_ONLY` flow i nie stanowią authority nowej ścieżki.

`ACCOUNT_ID=NOT_MINTED`, `ACCOUNT_RESERVATION=NOT_STARTED`,
`ACCOUNT_GENESIS=NOT_STARTED`, `PROVISIONING_MEMBERSHIP=BLOCKED`,
`PROTECTED_FRESHNESS=NOT_STARTED`, `SECRET_RESOURCE=NOT_STARTED`.
`stage_9=IN_PROGRESS`, `production_provisioning_ready=false`,
`windows_production_ready=NOT_READY`,
`LEGAL_PRODUCTION_ENROLLMENT=NOT_PERFORMED`,
`stage_10_production_lifecycle_live=NOT_STARTED`.
Hosted/mock tests nie są Windows native qualification.
