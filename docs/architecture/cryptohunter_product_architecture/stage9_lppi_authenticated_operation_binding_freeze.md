# Stage 9 — LPPI authenticated provisioning operation freeze

Status: `FROZEN`. Normatywny subordinate contract znajduje się w
[`stage9_lppi_authenticated_operation_binding_contract.json`](stage9_lppi_authenticated_operation_binding_contract.json),
a exact-byte hash chroni
[`stage9_lppi_authenticated_operation_binding_freeze.json`](stage9_lppi_authenticated_operation_binding_freeze.json).
Nadrzędny `stage9_external_provisioning_architecture_contract.json` pozostaje
byte-identical. Implementacja tego initial operation block jest architektonicznie
autoryzowana, a `production_provisioning_ready=false`.

| Artefakt | SHA-256 |
| --- | --- |
| Subordinate operation contract | `f75494b163137374d39d9147b5e328aa30e772d2887255deb8980cb442457ca2` |
| Parent Stage 9 contract | `6538a75d8cf3a6e3cdab8f4a9022277a21d98e1a61923aa8d8eae9b80ae1d67d` |

Parent zamraża właściciela `prvop`, purpose domain, dokładnie dziewięć payload
fields, signature domain, ACTIVE key match oraz kolejność protocol. Subordinate
domyka initial `schema_version=1`, `binding_generation=1`, millisecond UTC,
exact UUID timestamp correspondence i użycie purpose domain w durable identity.
Nie zmienia parent ani nie ustanawia CHA, membership lub successor operation.

Canonical payload zawiera dokładnie: `schema_version`, `environment`,
`pdsa_trust_domain`, `pdsa_package_digest_sha256`, `provisioning_subject_id`,
`enrollment_reference`, `provisioning_operation_id`, `binding_generation`,
`created_at_utc`. Środowisko wynosi `PRODUCTION`, a trust domain
`PDSA_PRODUCTION_2_OF_3_ED25519`. Generation i schema to exact integer `1`;
boolean jest odrzucany. Parser wymaga RFC 8785 JCS UTF-8 i odrzuca unknown,
missing, duplicate fields oraz noncanonical bytes.

## Audyt UUIDv7 i purpose domain

Poprawiony PDSA authorization profile w
`bot_core/licensing/pdsa_enrollment_authorization.py` zawiera
`_reservation_epoch_milliseconds` i `_mint_uuidv7`. Pierwszy używa integer
`timedelta` arithmetic i odrzuca timestamp spoza unsigned 48-bit zakresu.
Drugi ustawia UUID version `7`, RFC variant `10` oraz losuje niezależne CSPRNG
`rand_a` 12-bit i `rand_b` 62-bit. Operation issuer stosuje ten profile bez
obcinania reservation instant do sekund przed wygenerowaniem UUID.

Inne odnalezione minty mają odrębne znaczenie. `pdsa_enrollment_challenge.py`
najpierw obcina issue timestamp do pełnych sekund i wiąże tak timestamp
`pchal`; tego profile nie stosuje się do nowego `prvop`.
`external_provisioning._uuid7` używa float `time.time()*1000` oraz 48-bit mask,
a legacy `ProvisioningRepository.reserve` ma osobne źródło czasu payload.
Istniejący `LPPI_TEST_ONLY` flow nie ustanawia production authority.
`runtime.runtime_session.new_runtime_session_id` używa `time_ns`, 74-bit CSPRNG
i mask, lecz tworzy process-local session, bez durable operation semantics.
Canonical UUID validators w pre-enrollment, CHA attempt store i persistence
nie są issuerami production operation identity.

LPPI przechwytuje jeden rzeczywisty wewnętrzny UTC reservation instant.
Dla `2026-10-06T12:00:00.789123Z` UUIDv7 zachowuje epoch milliseconds
`1791288000789`, a binding ma `created_at_utc=2026-10-06T12:00:00.789Z`.
Wire timestamp ma dokładnie trzy cyfry fractional seconds; verifier porównuje
jego exact millisecond instant z timestamp UUIDv7. Caller nie wybiera UUID,
`prvop`, timestamp ani wartości źródłowych binding.

`CryptoHunter.Stage9.ProvisioningOperation.v1` nie jest hash input do UUID.
Purpose jest retained i oddziela reservation identity digest:
`SHA256(UTF8(purpose) || 0x00 || JCS_UTF8(exact reservation tuple))`.
Tuple obejmuje trust domain, package digest, subject, enrollment oraz exact
ACTIVE authority binding digest, algorithm profile, fingerprint i custody.
Zmiana któregokolwiek elementu w tym samym initial lifecycle failuje closed;
reservation nie tworzy listy konkurencyjnych identifiers.

## Durable operation i signer authority

Trwały model to `NO_OPERATION → PRVOP_RESERVED → AUTHENTICATED_OPERATION_COMMITTED`.
`BINDING_SIGNED` jest etapem w pamięci przed durable commit. Rezerwacja exact
`prvop`, canonical payload i ACTIVE identity poprzedza native sign lub publication.
Native signing odbywa się poza operation write lock. Pierwszy zaakceptowany
strict minimal low-S DER jest retained przed udostępnieniem operation authority.
Po commit payload, timestamp, generation, signature i signer identity są immutable.
Retry i restart zachowują exact `prvop`, payload bytes i committed signature bytes.
Crash po sign przed commit może ponownie podpisać tę samą rezerwację, gdy wynik
nie został wcześniej opublikowany ani ustanowiony jako authority.

Source authority pochodzi wyłącznie z verifier-issued
`VerifiedActiveLPPIAuthorityKey` z production lifecycle. Każdy consequential
sign wymaga `require_verified_active_lppi_authority_key` i live requalification
exact native successor. Sam ACTIVE status, public key, fingerprint, key name,
copied capability lub JSON record nie daje authority. Purpose-specific
`WindowsLPPIAuthorityKey.sign_authenticated_operation_binding` używa operation
domain `CryptoHunter.Stage9.LPPIAuthenticatedProvisioningOperationBinding.v1`.
Native sign dodatkowo wymaga `require_reserved_operation_binding`, exact durable
payload i private current-process/current-thread signing-lock owner. Standalone
native sign bez tego ownera jest odrzucany, więc commit nie może ścigać się
z niezależnym podpisem. Operation domain jest odrębny od authority PoP, continuity i membership domain.

Po durable commit trusted verifier wydaje opaque
`VerifiedLPPIAuthenticatedProvisioningOperation`. Exact type, prywatny registry,
immutable snapshot, retained exact-match, current ACTIVE guard i signature
revalidation wykluczają copied, subclassed i reconstructed capabilities.
Restart loader odtwarza/reverifikuje ACTIVE successor i exact retained tuple oraz
podpis przed wydaniem nowej capability. Durable JSON sam nie jest authority.

`ProductionMembershipSignerUnavailable` pozostaje zablokowany; ten block nie
wywołuje CHA, nie mintuje `ago_*`, nie tworzy account ani membership.
Następny odrębny blok wymaga CHA `logical_operation_id`, `prvop↔ago` bijection
i account genesis. Freshness, Secret Resource, Windows handoff i Stage 10
pozostają poza zakresem.

`stage_9=IN_PROGRESS`, `production_provisioning_ready=false`,
`windows_production_ready=NOT_READY`,
`stage_10_production_lifecycle_live=NOT_STARTED`,
`stage_10_prerequisite=BLOCKED_UNTIL_LEGAL_ENROLLMENT`.
`WINDOWS_NATIVE_LPPI_AUTHENTICATED_OPERATION_QUALIFICATION=NOT_RUN`;
`LEGAL_PRODUCTION_ENROLLMENT=NOT_PERFORMED`. Hosted/mock tests nie są physical
production qualification.
