# Produkcyjny initial LPPI authenticated provisioning operation

`establish_installed_lppi_authenticated_operation(active_authority)` w
`deployment/windows_production_lppi_operation.py` kończy kolejny production block
Stage 9. Wymaga prawdziwego `VerifiedActiveLPPIAuthorityKey` i po durable commit
wydaje `VerifiedLPPIAuthenticatedProvisioningOperation`. Caller nie przekazuje
`prvop`, zegara, raw key, fingerprintu, key path, trust roots, mappingu payload
ani callbacku podpisującego.

## Audyt punktu wyjścia

Dotychczasowy `external_provisioning.py` zawiera legacy test-only provisioning
flow z `LPPI_TEST_ONLY`, software `TestOnlyMembershipSigner` oraz repozytorium
operacji/CHA/membership. Dane tego flow nie stanowią production operation
authority. `ProductionMembershipSignerUnavailable` pozostaje fail-closed:
production membership wymaga autoryzacji PDSA i CHA account commit.

Production successor lifecycle z #3071 wydaje opaque
`VerifiedActiveLPPIAuthorityKey` przez
`establish_installed_lppi_authority_from_artifacts(...)` lub istniejący verified
package acceptance boundary. `require_verified_active_lppi_authority_key(...)`
sprawdza exact type, prywatny registry, immutable snapshot, current retained
ACTIVE state, package lineage i native CNG/TPM requalification. Publiczny ACTIVE
record i sam `status=ACTIVE` nie są signing authority. Przed tym blockiem
production flow nie utrwalał własnego `prvop` ani canonical authenticated
operation binding.

Każda granica wymaga bieżącej rekwalifikacji. Odczyty źródła w obrębie tej samej
weryfikacji korzystają z jednego immutable snapshotu uzyskanego przez ten guard;
snapshot nie jest cache'em pomiędzy wywołaniami. Niezależne guards przed i po
native sign, przed commit oraz przy każdym użyciu operation capability pozostają.
Niepoprawna authority jest odrzucana przed utworzeniem operation lock files.

Audyt UUIDv7 wybrał poprawiony PDSA authorization profile:
`_reservation_epoch_milliseconds` oraz `_mint_uuidv7` z
`pdsa_enrollment_authorization.py`. Profile używa integer UTC arithmetic,
unsigned 48-bit Unix milliseconds oraz CSPRNG `rand_a` 12-bit i `rand_b` 62-bit.
Challenge mint obcina timestamp do sekund, legacy `_uuid7` używa float czasu
i bitowej maski, a runtime session mint jest process-local. Nowa durable
operation korzysta z poprawionego authorization profile.

## Contract closure

Parent `stage9_external_provisioning_architecture_contract.json#/operation_identity`
pozostaje byte-identical; SHA-256 wynosi
`6538a75d8cf3a6e3cdab8f4a9022277a21d98e1a61923aa8d8eae9b80ae1d67d`.
Subordinate
[`stage9_lppi_authenticated_operation_binding_contract.json`](architecture/cryptohunter_product_architecture/stage9_lppi_authenticated_operation_binding_contract.json)
oraz
[`stage9_lppi_authenticated_operation_binding_freeze.json`](architecture/cryptohunter_product_architecture/stage9_lppi_authenticated_operation_binding_freeze.json)
domykają wyłącznie initial operation. Architecture guard porównuje exact parent
bytes, subordinate digest, dziewięć pól, domain/profile, initial wartości,
durable record fields, status progression i protocol ordering z runtime.

Canonical `LPPIAuthenticatedProvisioningOperationBindingV1` ma dokładnie:

```text
schema_version
environment
pdsa_trust_domain
pdsa_package_digest_sha256
provisioning_subject_id
enrollment_reference
provisioning_operation_id
binding_generation
created_at_utc
```

`schema_version=1` i `binding_generation=1` to exact integers; boolean jest
odrzucany. `environment=PRODUCTION`, trust domain
`PDSA_PRODUCTION_2_OF_3_ED25519`. RFC 8785 JCS UTF-8 odrzuca unknown, missing
i duplicate fields, noncanonical bytes oraz niewłaściwe wire types.

LPPI przechwytuje jeden rzeczywisty wewnętrzny reservation instant.
Dla `2026-10-06T12:00:00.789123Z` UUIDv7 zawiera `1791288000789`, a wire
`created_at_utc` wynosi `2026-10-06T12:00:00.789Z`. Timestamp ma dokładnie trzy
fractional digits; verifier wymaga real Gregorian UTC date i exact równości
UUID milliseconds z `created_at_utc`.

`CryptoHunter.Stage9.ProvisioningOperation.v1` nie wpływa na UUID bits.
Purpose jest retained i uczestniczy w reservation digest:

```text
SHA256(UTF8("CryptoHunter.Stage9.ProvisioningOperation.v1")
       || 0x00
       || JCS_UTF8(exact reservation tuple))
```

Tuple obejmuje trust domain, exact package digest, subject, enrollment,
`lppi_authority_key_binding_digest_sha256`, authority algorithm profile,
fingerprint i custody. Domain oddziela durable operation identity/conflict
namespace od innych celów bez arbitralnego hashowania purpose do UUID.

## Source i signing boundary

Binding pobiera trust domain, package digest, subject i enrollment z exact
verified production package stojącego za bieżącym ACTIVE successor.
`provisioning_operation_id` pochodzi wyłącznie z durable LPPI reservation.
Reconstructed binding lub mapping caller nie ustanawia authority.

Native method `WindowsLPPIAuthorityKey.sign_authenticated_operation_binding`
przyjmuje exact canonical operation binding i current ACTIVE capability.
Przed sign wymaga current ACTIVE guard, exact native key identity i exact
durably reserved payload przez `require_reserved_operation_binding`. Guard
wymaga również private signing-lock owner związany z current process i current
thread. Standalone native sign poza tym locked flow jest odrzucany; owner nie
przechodzi do innego wątku ani po fork. Matching obejmuje authority algorithm profile,
fingerprint, custody oraz retained CNG Unique Name i TPMT_PUBLIC kwalifikowane
przez production lifecycle. Pre-enrollment key, drugi P-256 key, TEST_ONLY
signer i stany poprzedzające ACTIVE nie mogą podpisać operation authority.

Signed bytes to:

```text
UTF8("CryptoHunter.Stage9.LPPIAuthenticatedProvisioningOperationBinding.v1")
|| 0x00
|| SHA256(JCS_UTF8(canonical payload))
```

Podpis jest `ECDSA-P256-SHA256`, strict minimal ASN.1 DER, low-S. Verification
odrzuca high-S, nonminimal DER, trailing bytes i invalid signature. Operation
nie używa authority PoP, continuity, package acceptance ani membership domain.
Public API nie udostępnia arbitrary-domain signing.

## Durable lifecycle i immutability

```text
NO_OPERATION
→ PRVOP_RESERVED
→ BINDING_SIGNED (wyłącznie pamięć)
→ AUTHENTICATED_OPERATION_COMMITTED
```

Frozen protocol kończy się na
`LPPI_AUTHENTICATED_OPERATION_BINDING_COMMITTED`; CHA nie jest wywoływane.
Rekord `State/LPPIAuthority/initial-authenticated-operation.json` jest
singletonem z exact schema `LPPIInitialAuthenticatedOperationV1`. Retains:

- schema, status i history;
- purpose domain i reservation digest;
- trust/package/subject/enrollment i exact ACTIVE binding digest/profile/fingerprint/custody;
- exact canonical ACTIVE binding bytes;
- `prvop`, generation i millisecond created timestamp;
- exact canonical operation payload bytes i SHA-256 digest;
- exact signature DER bytes po commit.

Atomic replacement następuje po file fsync; POSIX dodatkowo fsyncuje directory,
a Windows używa `MoveFileExW` z `MOVEFILE_WRITE_THROUGH`.
`initial-authenticated-operation.lock` zabezpiecza krótkie reserve/finalize,
a `initial-authenticated-operation-signing.lock` chroni cały expensive live/sign
etap. Oba są nonblocking cross-process locks. Live ACTIVE qualification i native
signing odbywają się poza state write lock. Równoległe rozpoczęcie widzi jeden
retained `prvop`, exact result albo `LPPI_AUTHENTICATED_OPERATION_BUSY`.

Po pierwszej reservation retry zachowuje ten sam `prvop`. Inny package,
subject, enrollment, trust domain lub authority w tym samym lifecycle zwraca
`LPPI_AUTHENTICATED_OPERATION_CONFLICT`. Po commit nie można zmienić payload,
identity, generation, timestamp, signature ani signer snapshot. Committed
operation zwraca exact retained bytes bez kolejnego native sign.

Installer-owned machine state wymaga istniejącej ochrony katalogu `State`.
Ten primitive nie implementuje authenticated rollback-proof storage ani
Protected Freshness. Publiczny JSON nie jest samodzielną authority.

| Cut-point | Retry / recovery |
| --- | --- |
| Przed durable `prvop` reservation | Można bezpiecznie zarezerwować pierwszy identifier; nic nie zostało opublikowane. |
| Po reservation | Użyj tego samego durable `prvop`, payload i reservation instant. |
| Przed signing | Resume exact reservation po current ACTIVE requalification. |
| Podczas native signing | Fail-closed lub ponów tę samą rezerwację; bez capability przed commit. |
| Po native sign, przed durable signature commit | Można ponownie podpisać exact payload, gdy poprzedni wynik nie został opublikowany ani ustanowiony jako authority. |
| Po full commit, przed response | Zwróć exact retained payload i signature bytes; nie podpisuj ponownie. |
| Lost response po commit | Exact same result po retained signature i current ACTIVE verification. |
| Process restart | Odtwórz/reverifikuj ACTIVE przez istniejące artefakty, następnie verify retained operation; bez remint. |

## Restart i capability provenance

Restart rozpoczyna się od existing production artifact loader:
`establish_installed_lppi_authority_from_artifacts(...)`. Loader rozwiązuje
installed Production Trust, weryfikuje exact retained package/request/challenges,
TPM exchange, custody oraz local pre-enrollment possession i odzyskuje exact
native successor zgodnie z #3071 ownership gates. Nie rekonstruuje ACTIVE
capability z publicznego JSON.

Po uzyskaniu rzeczywistego current ACTIVE caller wywołuje
`load_installed_lppi_authenticated_operation(active_authority)`.
Loader odczytuje durable committed operation, exact-matche tuple i retained
ACTIVE binding, ponownie weryfikuje operation signature oraz wydaje nową opaque
capability przez prywatny registry. Loader nie tworzy reservation ani operation.
Jeżeli record pozostaje `PRVOP_RESERVED`, trusted caller używa
`establish_installed_lppi_authenticated_operation(active_authority)` do resume
tej samej rezerwacji; loader committed authority failuje closed.

`VerifiedLPPIAuthenticatedProvisioningOperation` nie ma publicznego konstruktora,
przechowuje immutable snapshot w private registry i wymaga exact type.
`copy.copy`, subclass, `object.__new__` i copied retained row nie dają authority.
Każde consequential użycie przez
`require_verified_lppi_authenticated_operation(...)` rekwalifikuje current
ACTIVE, sprawdza retained bytes i tuple oraz rewaliduje signature. Tampering
ACTIVE fingerprint, custody, binding digest, CNG Unique Name lub TPMT_PUBLIC
odbiera capability authority przez current ACTIVE gates.

## Kolejny dependency block i status

Kolejny block konsumuje `VerifiedLPPIAuthenticatedProvisioningOperation` i
wymaga CHA `logical_operation_id` / `ago_*`, immutable `prvop↔ago` bijection
oraz account genesis. Membership pozostaje zależny od CHA account commit.
Ten block nie tworzy account ID, membership, Protected Freshness, Secret Resource,
Windows handoff ani Stage 10.

```text
WINDOWS_NATIVE_LPPI_AUTHENTICATED_OPERATION_QUALIFICATION = NOT_RUN
LEGAL_PRODUCTION_ENROLLMENT = NOT_PERFORMED
stage_9 = IN_PROGRESS
production_provisioning_ready = false
windows_production_ready = NOT_READY
stage_10_production_lifecycle_live = NOT_STARTED
stage_10_prerequisite = BLOCKED_UNTIL_LEGAL_ENROLLMENT
```

Hosted i mock tests nie stanowią physical Windows/TPM qualification.
