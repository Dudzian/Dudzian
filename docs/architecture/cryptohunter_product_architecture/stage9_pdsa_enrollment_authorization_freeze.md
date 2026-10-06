# Stage 9 — initial issuance PDSAEnrollmentAuthorizationPackageV1

Status: `FROZEN`. Kanoniczne źródło prawdy stanowi
[`stage9_pdsa_enrollment_authorization_contract.json`](stage9_pdsa_enrollment_authorization_contract.json),
a jego dokładne bajty chroni
[`stage9_pdsa_enrollment_authorization_freeze.json`](stage9_pdsa_enrollment_authorization_freeze.json).
Nadrzędny kontrakt Stage 9 oraz wcześniejszy subordinate challenge contract zachowują
dotychczasowe bajty i SHA-256. Implementacja jest architektonicznie autoryzowana;
`production_provisioning_ready = false`, Stage 9 pozostaje `IN_PROGRESS`.

| Artefakt | Schema | SHA-256 |
| --- | --- | --- |
| `stage9_pdsa_enrollment_authorization_contract.json` | `cryptohunter.stage9_pdsa_enrollment_authorization_contract.v1` | `c1a6c039f2dfc446b323bb7b1912c18111fd4672365384566b5d605b5feeb4be` |
| Parent `stage9_external_provisioning_architecture_contract.json` | `cryptohunter.stage9_external_provisioning_architecture.v1` | `6538a75d8cf3a6e3cdab8f4a9022277a21d98e1a61923aa8d8eae9b80ae1d67d` |

Parent definiuje dokładnie 23 canonical payload fields, Ed25519 PDSA 2-of-3,
signature domain, exact request/device binding, mandatory expiration i retained
atomic issuance. Nie określa formatu `enrollment_reference`, initial wartości
generation/version/lineage, initial predecessor, policy wyznaczania expiry ani
liczby i kolejności wpisów w signature block. Ten subordinate domyka wyłącznie
initial issuance. Nie ustanawia replacement ani recovery semantics.

`enrollment_reference` ma format `penr_<canonical-lowercase-UUIDv7>`.
PDSA niezależnie mintuje oba identyfikatory przy użyciu CSPRNG i zapisuje je
w durable reservation przed podpisaniem. Caller nie dostarcza identyfikatorów.
Oba UUIDv7 używają tych samych captured reservation milliseconds:
`floor(actual internal UTC reservation instant * 1000)`, z niezależną randomness.
`issued_at_utc` pochodzi z tego samego captured instant, ale wire encoding obcina
go do pełnych sekund. Dla `2026-10-06T12:00:00.789123Z` oba UUIDv7 zachowują
timestamp `1791288000789`, a package zapisuje `2026-10-06T12:00:00Z`.
`authorization_generation`, `authorization_version` i `lineage_generation`
wynoszą `1`; `predecessor_package_digest_or_null` wynosi `null`. Inna lineage
jest odrzucana. Package zachowuje istniejący wire literal
`CryptoHunter.Stage9.PDSAEnrollmentAuthorizationPackage.v1` oraz envelope
`payload`/`signatures`. Signature block zawiera dokładnie dwa odrębne zaufane
`key_id` z exact trzech aktualnie autoryzowanych kluczy PDSA, uporządkowane
leksykograficznie; trzeci podpis jest odrzucany. Każda z par `K1+K2`, `K1+K3`
i `K2+K3` spełnia threshold `2`. Kolejność lexical określa canonical envelope,
a wybór dostępnej pary należy wyłącznie do trusted signing authority.

Expiry wynosi minimum deadline podpisanego retained `PDSAEnrollmentChallengeV1`
oraz deadline retained verified `TPMEnrollmentChallengeV1`. `ProductionTPMChallengeStore.issue`
już ogranicza TPM challenge do wcześniejszego deadline PDSA challenge i podpisanego
curated TPM endorsement. Źródłem jest retained exchange z rzeczywistej
verifier-issued capability; caller JSON nie ustanawia tej authority. Nie przyjęto
stałego package TTL ani nowego okresu siedmiu dni. Internal timestamp reservation
i source deadline są immutable; retry nie odnawia ważności. Final commit ponownie
sprawdza current Production Trust, live TPM exchange, request PoP i source validity.

Lifecycle to `RESERVED → SIGNED → COMMITTED`. `RESERVED` zachowuje exact request,
identyfikatory, canonical payload, deadline, exact Production Trust identity
i public key set, wszystkie trzy authorized signer IDs oraz threshold `2`.
Nie wymaga wyboru pary przed signingiem. Signing działa poza SQLite write lock.
Internal issuer-deployment service otrzymuje exact message, `authorized_signer_ids`
obejmujące exact reserved trzy klucze, `required_threshold = 2` oraz idempotency key
`SHA256(exact reserved canonical payload)`. Caller transport nie wybiera callbacka,
signera, threshold, public keys ani endpointu. Trusted service wybiera dowolne dwa
distinct eligible keys i trwale zapisuje pierwszą skuteczną odpowiedź z exact
selected signer IDs oraz signature bytes przed jej zwróceniem. Local issuer
weryfikuje oba podpisy, lexical order oraz niezmienioną exact reserved/current
Production Trust identity i key set; trust rotation/mismatch failuje closed.
Lost response, issuer crash przed local `SIGNED` lub service restart odtwarzają
exact tę samą odpowiedź. Jeśli service wybrał `K2+K3`, odzyskanie dostępności `K1`
nie pozwala zmienić pary ani signature bytes na retry.
`SIGNED` zachowuje exact selected pair, envelope i digest; retry `SIGNED`/`COMMITTED`
wymaga tej samej pary i exact package bytes. Dopiero `COMMITTED` atomowo zapisuje
package oraz consumption/accepted request/receipt i zezwala na odpowiedź.
Lost response i restart zwracają exact retained package bytes bez nowych identity.
Retained fields określają semantyczną gwarancję exact danych. Implementacja może
użyć exact immutable join do retained challenge row i odczytu zwalidowanych pól
z exact canonical payload zamiast duplikować każdy element w osobnej kolumnie.
Acceptance-only receipt z wcześniejszego flow sam nie pozwala dopisać package.
Deployment musi uwzględnić istniejący stan sprzed remediation. Populated legacy
issuance schema bez exact reserved Production Trust identity i pełnego public
key set failuje closed (`PDSA_AUTHORIZATION_ISSUANCE_SCHEMA_MISMATCH`); zachowuje
pliki i rekordy, bez automatycznej rekonstrukcji lub rebinding do current trust.
Przebudowa pustego legacy issuance schema nie odtwarza żadnej authority.

Audyt legacy rozróżnia trzy aktywne kontrakty. `bot_core/licensing/enrollment.py`
buduje licensing `PDSAEnrollmentPackageV1` z entitlements, `license` i K_PSA;
używają go `authority.py` oraz `verification.py`. Deployment schema
`deployment/stage9_pdsa_enrollment_package_v1.schema.json` nadal obsługuje
root-of-trust `CryptoHunter.PDSAEnrollmentPackageV1` z target TPM i `state_enrollment_digest`;
używa go `windows_stage9_root_of_trust_freeze.verify_pdsa_enrollment_package`,
a installer zawiera ten schema. Żaden z tych artefaktów nie jest przemianowany
ani reinterpretowany jako nowa authorization package. Reuse obejmuje zaufaną
canonical serialization, Production Trust i istniejący production pre-enrollment.

`ProductionProvisioningPackageVerifier` był już obecny w `external_provisioning.py`,
lecz wymagał domknięcia strict production encoding, initial lineage, signature
ordering oraz exact authenticated target binding. Dotychczasowy production
pre-enrollment kończył się w `PDSAChallengeStore.consume_authenticated_request`:
`CONSUMED` wraz z exact request bytes/digest i acceptance receipt, bez subject
i package. Nowy issuance rozszerza ten sam issuer-owned atomic boundary.

Ten block nie generuje LPPI successor key, `prvop`, konta ani membership.
`ProductionMembershipSignerUnavailable` pozostaje fail-closed. Real public trust
i off-host quorum signing service są niezbędnymi production deployment inputs.
Exact durable service reply po lost response/restart wymaga kwalifikacji realnego
off-host serwisu; lokalny RPC verifier i test service nie ustanawiają tej gwarancji.
Po pozytywnym review remediation można architektonicznie przejść do odrębnego
bloku `LPPIAuthorityKeyBindingV1` i successor TPM-backed key z continuity/PoP oraz
custody verification. Legal production enrollment nadal wymaga live deployment i physical
custody qualification; Stage 9 pozostaje `IN_PROGRESS`, bez gotowości produkcyjnej.
**PRODUCTION MATERIAL REQUIRED — DO NOT GENERATE SUBSTITUTE KEYS.**
