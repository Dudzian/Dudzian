# Stage 9 — propozycja domknięcia external product provisioning

## Kontrola dokumentu

| Pole | Wartość |
|---|---|
| status | `SUPERSEDED — HISTORICAL REVISION 4 PROPOSAL / NOT CURRENT CANONICAL` |
| zakres | domknięcie upstream dla `external_product_provisioning_boundary` |
| implementacja produkcyjna | `SUPERSEDED BY FROZEN CONTRACT; ARCHITECTURALLY AUTHORIZED, ACTIVATION NOT READY` |
| Stage 0–8 | `DONE` (bez zmiany) |
| Stage 9 | `IN PROGRESS` (bez zmiany) |
| Stage 10 | `NOT_STARTED` |

> **Supersession:** current machine-checkable authority is `stage9_external_provisioning_architecture_contract.json`, frozen by `stage9_external_provisioning_architecture_freeze.json`. Statements below that say `DESIGN_BLOCKED`, `NOT_FOUND`, `NOT ACCEPTED`, `NOT FROZEN`, or `NOT AUTHORIZED` are retained as historical proposal rationale and are not current canonical values. Production composition remains fail-closed until the real ceremony public package exists.

Ten dokument był wynikiem inventory ówczesnego HEAD. Jako zachowany proposal nie superseduje
zamrożonych kontraktów i nie jest current authority. Wymagany architecture change, canonical JSON
oraz freeze manifest istnieją teraz w artefaktach wskazanych powyżej; production activation nadal
wymaga realnego publicznego outputu ceremony. MSI pozostaje wyłącznie orkiestratorem:
`MSI != account authority`, `MSI != identity mint owner` i
`MSI != provisioning membership authority`.

## 1. Inventory aktualnego HEAD

### Aktualne, wykonywalne granice

* M0.3 wskazuje `external_product_provisioning_boundary` jako istniejący przed Core trust
  anchor. Bootstrapper jedynie przenosi opaque reference, Core jedynie konsumuje i waliduje,
  a publiczny hash dowodzi integralności treści, nie membership ani autentyczności.
* `FirstRunBootstrapClaim` wiąże `account_id`, `device_installation_id`,
  `intended_operator_id`, generation/revision, okno ważności, challenge i provisioning context.
  `ProvisioningMembershipBinding` wiąże fingerprint claimu z kompletną treścią i dokładnym
  `authority_source=external_product_provisioning_boundary`. Ani jeden typ nie definiuje
  emitenta, podpisu ani ownera mintowania `account_id`.
* `FirstRunBootstrapAuthority` jest wyłącznie semantycznym compare-and-consume dla
  `INITIAL_SECURITY_ESTABLISHMENT_ONLY`. Nie mintuje konta, urządzenia, operatora ani
  membership. Sprawdza accepted claim i membership w wstrzykniętym `ProvisioningBoundary`,
  zgodność z bieżącym Core state oraz jednorazowe generation/revision/challenge.
* `DurableFirstRunBootstrapRegistry` i `DurableFirstRunBootstrapCoordinator` zapewniają
  trwały M0.11 carrier, current designation oraz atomowe materializowanie/consume w
  StateStore. Przechowują fakty już zaakceptowane; nie mogą stworzyć upstream authority.
* M0.10 ustanawia pierwszą `OperatorIdentity` i oznacza pierwsze urządzenie jako `TRUSTED`
  dopiero po zaakceptowanym bootstrap transition. Jest to initial-security authority, nie
  account/provisioning authority.
* `CoreHostStartupRecoveryCoordinator` składa istniejących ownerów recovery i failuje
  closed. Nie jest właścicielem genesis, external freshness ani zewnętrznego sekretu.
* `ProtectedFreshnessAuthorityPort` jest portem do osobnego
  `EXTERNAL_PRODUCT_PROTECTED_STATE_BOUNDARY`. Current designation i monotonic lifecycle
  `UNINITIALIZED -> PREPARED -> COMMITTED` żyją poza restorable StateStore. M0.3 jawnie
  zabrania mintowania membership przez Core, M0.11, Bootstrapper, backup i first-run claim.
* `SecretExternalResourcePort` definiuje `begin/reconcile/cleanup`; koordynator M0.11
  przechowuje lifecycle handoffu i wymusza reconciliation-before-retry, ale repo nie zawiera
  produkcyjnego ownera zewnętrznego magazynu sekretów.
* Stage-9 `WindowsExternalProvisioningHandoff` jest tylko fail-closed protokołem kompozycji.
  Żąda accepted claim, membership, initial StateStore metadata oraz dwóch niezależnych portów.
  Default loader celowo nie dostarcza providera.

### Kontrakty superseded i current

| Obszar | Wcześniejszy wynik | Późniejszy current kontrakt | Skutek dla Stage 9 |
|---|---|---|---|
| owner finalnej decyzji account genesis | `NOT_FOUND / DESIGN_BLOCKED` w discovery/modelu genesis | `m05_account_genesis_authority_topology_resolution_after_binding_freeze.json`: `A_SINGLE_ACCOUNT_GENESIS_COORDINATOR`, owner `CryptoHunterAccountAuthority` | ownership decyzji CHA jest zamrożony, lecz commit nadal blokuje brak external root proof |
| operation identity dla account genesis | identity source `DESIGN_BLOCKED` | kontrakt operation binding zamraża wymagania i progression, ale nadal wybiera `F_DESIGN_BLOCKED`; późniejsze root-proof contracts używają `logical_operation_id`, nie ustanawiając jego issuer-a | replay invariants są znane, lecz canonical operation identity nadal nie jest zamknięta |
| root-proof semantics | brak primitive | independent pre-account RootProofIssuer, podpisane request/proof i retained registry są zamrożone | model kryptograficzny istnieje, ale produkcyjne entitlement/provisioning i runtime nadal nie są dostępne |
| local root-proof substrate | `NOT_FOUND` | wybrano PostgreSQL registry + odseparowaną lokalną Ed25519 custody + niezależny checkpoint | wybór substrate nie jest implementacją i nie ustanawia provisioning membership |
| AccountGenesis freshness | design blocked | wybrano PostgreSQL SERIALIZABLE CAS/history/receipt + osobną Ed25519 custody; authentication gate przeszedł | dotyczy freshness account genesis, nie wolno automatycznie rozszerzać go na M0.3 protected restore freshness ani provisioning head |
| account-ID mint/reservation | wcześniejszy projekt przypisywał przyszłej authority, następnie reconciliation jawnie go otworzyło | `m05_cryptohunter_account_root_of_trust_reconciliation.json` oraz model genesis nadal mówią `UNRESOLVED`, `NOT_FROZEN`, model E `DESIGN_BLOCKED` | brak superseding wyboru A/B/C/D i brak pozwolenia na implementację |
| account/first-device ordering | brak | nadal `DESIGN_BLOCKED`; ani `ACCOUNT_FIRST`, ani `ATOMIC_ACCOUNT_PLUS_FIRST_DEVICE` nie został wybrany | propozycja poniżej musi dokonać wyboru przed kodem |
| produkcyjny ProvisioningBoundary | brak | nadal `NOT_FOUND / NOT_AVAILABLE`, `implementation_allowed=false` | blocker pozostaje aktualny |
| protected freshness provider | implementation-neutral external owner | nadal `EXTERNAL_PRODUCT_PROTECTED_STATE_BOUNDARY`, bez produkcyjnego adaptera | osobny dependency; nie może zniknąć w Windows handoff |
| secret resource provider | protocol i durable lifecycle | brak produkcyjnego adaptera | osobny dependency; testowy `SecretPort` nie jest providerem |

Późniejsze kontrakty M0.5 supersedują część dawnych ustaleń o **wewnętrznym** account-genesis
(topologia, root proof, operation binding i wybrane substrate). Nie supersedują jednak ostatniego
reconciliation w zakresie mintowania `account_id`, account/device ordering ani produkcyjnego
`external_product_provisioning_boundary`. Zatem architecture jest nadal design-blocked w zakresie
potrzebnym Stage 9.

## 2. Current canonical status

```text
ACCOUNT_ID_MINT_OWNER = NOT_FROZEN / NOT_FOUND
FIRST_ACCOUNT_DEVICE_ORDERING = DESIGN_BLOCKED
ACCOUNT_GENESIS_ATOMICITY = CHA SEMANTIC OWNER FROZEN; END-TO-END ACCOUNT+DEVICE NOT_FROZEN
ACCOUNT_GENESIS_IDEMPOTENCY = REQUIREMENTS FROZEN; OPERATION IDENTITY/OWNER AND PROVISIONING IDEMPOTENCY NOT_FROZEN
PROVISIONING_AUTHORITY_OWNER = ABSTRACT external_product_provisioning_boundary ONLY; PRODUCTION OWNER NOT_FOUND
PROVISIONING_MEMBERSHIP_DURABILITY = NOT_PROVEN / PRODUCTION REGISTRY NOT_FOUND
PROVISIONING_AUTHENTICATION = NOT_FOUND
PROVISIONING_ROLLBACK_PROTECTION = NOT_FOUND
PRODUCTION_PROVISIONING_IMPLEMENTATION = NOT_AVAILABLE
IMPLEMENTATION_ALLOWED = FALSE
```

Exact source contracts:

* `ACCOUNT_ID_MINT_OWNER`, ordering i end-to-end atomicity: sekcje
  `/account_id_mint_owner`, `/account_id_reservation_problem`, `/account_vs_device_ordering` i
  `/account_genesis_semantics` w
  `m05_cryptohunter_account_root_of_trust_reconciliation.json`, potwierdzone przez
  `/selected_or_blocked_mint_model`, `/selected_or_blocked_ordering_model` i
  `/implementation_allowed` w `m05_cryptohunter_account_genesis_authority_model.json`.
* Wewnętrzny owner oraz wymagania idempotency CHA: `/decision`, `/ownership_resolution` w
  `m05_account_genesis_authority_topology_resolution_after_binding_freeze.json` oraz
  `/selected_or_blocked_model`, `/canonical_request`, `/idempotency_progression` w
  `m05_account_genesis_operation_identity_request_binding_contract.json`. Ten drugi kontrakt
  nadal wybiera `F_DESIGN_BLOCKED`, więc nie jest superseding wyborem issuer-a identity.
* Provisioning authority, durability, authentication, rollback i availability: sekcje
  `/m03_existing_authorities`, `/production_implementation_search`, `/authentication`,
  `/rollback_freshness`, `/impact_on_current_account_design` w reconciliation oraz
  `/first_run_bootstrap_authority` w `process_topology_and_lifecycle.json`.
* Protected freshness owner: `/restore_freshness_authority_contract` w
  `process_topology_and_lifecycle.json` i `/protected_freshness_authority` w M0.11
  `persistence_versioning_migrations_backup_and_recovery.json`.
* Secret resource: `SecretExternalResourcePort` w `bot_core/persistence/secret_handoff.py`;
  istniejący kontrakt określa protokół, nie produkcyjny authority/provider.

## 3. Porównanie modeli

| Kryterium | A — pre-admission reservation | B — external authority mints account + first device | C — atomic Account Authority + provisioning issuer | D — local product root poza Core |
|---|---|---|---|---|
| authority owner | CHA dla rezerwacji i genesis | jeden external provisioning owner | CHA jest jedynym finalnym genesis ownerem; niezależny issuer wystawia root/device evidence | lokalny machine authority jest samodzielnym ownerem |
| mint `account_id` | CHA w durable reservation | external authority | CHA w durable `PREPARED` dla authority-issued operation | local product root |
| mint `device_installation_id` | późniejszy provisioning issuer | external authority | provisioning issuer rezerwuje ID w tej samej operacji | local product root |
| `intended_operator_id` | osobny późniejszy issuer; ryzyko rozjazdu | external authority | issuer wiąże operatora; CHA commit exact-binds ten fakt bez ustanawiania M0.10 identity | local product root |
| ordering | `ACCOUNT_FIRST` | `ATOMIC_ACCOUNT_PLUS_FIRST_DEVICE` | `ATOMIC_ACCOUNT_PLUS_FIRST_DEVICE` jako jedna logiczna decyzja, koordynowana przez CHA | zwykle atomic lokalnie, ale duplikuje CHA |
| dwa różne pierwsze konta | potrzebny subject key przed kontem; nadal niejasny | external serializer wybiera | ten sam stable provisioning subject i operation registry daje jednego winnera | lokalny serializer daje winnera tylko na jednej maszynie |
| dwa urządzenia dla tego samego konta | first-device CAS po genesis | external CAS | CHA/issuer protocol ma jeden first-device slot | lokalny CAS; późniejszy multi-device wymaga migracji authority |
| restart | reservation journal CHA | registry external | oba durable journals + recovery tej samej operation | lokalny journal |
| replay | stable CHA operation | stable external operation | authority-issued `provisioning_operation_id` mapowany 1:1 do CHA `logical_operation_id` | stable local operation |
| rollback resistance | wymaga niezależnego head | wymaga external head | niezależne provisioning head i CHA freshness; dwa różne znaczenia | machine monotonic head/checkpoint |
| multi-device | naturalne po account-first, lecz wymaga nowego issuer protocol | możliwe tylko jeśli external owner ma globalny registry | naturalne: późniejszy issuer dowodzi istniejącego account root i tworzy wyłącznie device membership | trudne poza maszyną bez federacji/migracji root |
| Windows offline | po genesis tak | zależy od external deployment | tak po domknięciu; tier lokalny może wykonać first provisioning offline | tak |
| złożoność bezpieczeństwa | średnia, ale nierozwiązany pre-account subject | średnia implementacyjnie, wysoki konflikt ownership | najwyższa protokołowo, lecz jasne ownership i crash semantics | niska dla jednej maszyny, wysoka przy bezpiecznej ewolucji |
| M0.3/M0.5/M0.11 | M0.3 pasuje, lecz nie zamyka wspólnego claimu | koliduje z zamrożonym sole CHA genesis ownerem | najlepsza: zachowuje consumer M0.3, sole CHA owner i carriers M0.11 | koliduje/duplikuje CHA oraz grozi ukryciem protected/secret ports |

Ocena bezpieczeństwa:

* **A** nie jest kompletne bez istniejącej przed kontem, stabilnej tożsamości subject oraz
  osobnego protokołu first-device. Samo przesunięcie mintowania do CHA nie usuwa circularity.
* **B** jest spójne samo w sobie, ale łamie późniejszy frozen wybór
  `CryptoHunterAccountAuthority` jako jedynego finalnego genesis decision ownera.
* **C** ma więcej kroków, ale nie więcej ownerów niż rzeczywiście potrzeba: CHA decyduje o
  koncie, niezależny provisioning issuer autoryzuje product subject/device, a M0.3 konsumuje.
* **D** jest atrakcyjne local-first, lecz jako samodzielny model duplikuje CHA i zamyka drogę
  do bezpiecznego multi-device. Lokalny product root może być **deployment substrate issuer-a
  w C**, ale nie osobnym account authority.

## 4. Rekomendacja: model C

Revision 4 zachowuje **C — ATOMIC ACCOUNT AUTHORITY + PROVISIONING ISSUER PROTOCOL** oraz
wybór:

```text
ATOMIC_ACCOUNT_PLUS_FIRST_DEVICE
```

Względem proposal v1 revision 3:

1. ustanawia pre-account ownera `provisioning_subject_id`, zamiast pozwalać LPPI rezerwować
   niezakorzenioną identity;
3. wybiera signed offline enrollment package jako produkcyjną first-install ceremony;
4. wybiera `LPPI_INSTALLED_BY_CRYPTOHUNTER_MSI`, ale oddziela instalację kodu od ustanowienia
   trust;
5. rozdziela candidate i genuine device/operator identity;
6. zachowuje CHA jako jedynego requestera RootProofIssuer;
7. zamyka LPPI state machine oraz recovery matrix bez distributed SQL transaction;
8. wybiera TPM 2.0 NV jako lokalny anti-rollback anchor i podpisaną zewnętrzną recovery ceremony
   jako fallback po utracie TPM;
9. wybiera konkretne profile dla Product Protected State Authority oraz Secret Resource
   Authority;
10. umieszcza account/device provisioning w `POST_INSTALL_FIRST_RUN_ENROLLMENT`, przy backendzie
   `DEMAND_START` i niedopuszczonym CoreHost.

Dokument nadal jest proposalem. Żaden wybór poniżej nie jest current canonical authority przed
architecture change, walidatorami i aktualizacją freeze manifestu.

## 5. Exact authority dla `provisioning_subject_id`

Current frozen finding pozostaje punktem wyjścia:

```text
EXTERNAL_SUBJECT_IDENTITY
authority owner = NOT_FOUND
caller controllability = MUST NOT
stability across retries = NOT_PROVEN
stability across restart = NOT_PROVEN
```

Revision 4 proponuje nowego, jednoznacznego ownera:

```text
provisioning_subject_id owner = Product Deployment Security Authority (PDSA)
schema = psub_<canonical lowercase UUIDv7>
issuance = assigned random identifier, never derived from Windows/user/hardware/account data
issuance time = offline enrollment-package issuance, before product enrollment and before account
```

* **Kto mintuje:** odseparowany PDSA działający na offline provisioning workstation/tooling,
  dopiero po zatwierdzeniu pre-enrollment request z docelowego TPM.
  LPPI, MSI, użytkownik, administrator, CHA, Core ani TPM nie mogą mintować subject identity.
* **Z jakiej pre-account authority:** z PDSA trust domain istniejącego przed kontem i przed
  instalacją produktu. Root public key PDSA jest pinned w podpisanym release policy LPPI; private
  signing key pozostaje poza laptopem i poza MSI.
* **Random/derived/assigned:** PDSA przypisuje losowy canonical UUIDv7. Timestamp UUIDv7 nie
  rozstrzyga kolejności ani uniqueness. ID nie jest wyprowadzane z SID-u Windows, TPM EK,
  hardware fingerprint, e-maila, nazwy użytkownika ani caller input.
* **Authentication:** domain-separated Ed25519 signature PDSA nad exact canonical enrollment
  manifestem, zawierającym `psub`, immutable `enrollment_reference`, product/profile,
  environment=`PRODUCTION`, issuance sequence, validity policy, docelowy TPM EK/AK attestation
  fingerprint, LPPI pre-enrollment public-key fingerprint i manifest digest. Pinned root, signed
  key lifecycle oraz retained issuance record muszą się zgadzać; SHA-256 jest tylko content
  digest. TPM evidence samo nie jest subject authority — dopiero podpis PDSA zatwierdza binding.
* **Durability:** authoritative issuance record i status subject są w append-only offline PDSA
  issuance registry. Laptop przechowuje signed package/receipt w LPPI registry oraz mirror
  związany z TPM NV head. StateStore i jego backup nie są subject authority.
* **Restart:** LPPI rozwiązuje current subject z verified registry + TPM head; nie pyta sieci.
* **Reinstall:** reinstall nie mintuje subject. Ten sam signed package albo signed PDSA recovery
  package wskazuje ten sam `enrollment_reference` i `psub`. LPPI wymaga zgodności z zachowanym
  TPM anchor oraz odtworzonym authority backup. Brak obu oznacza fail-closed external recovery,
  nigdy auto-enrollment.
* **No self-mint:** package bez podpisu pinned PDSA, package podpisany caller key, nowy root,
  zmienione `psub` lub TOFU są odrzucane. MSI nie może dopisać root key do allow-listy.
* **Exact retry:** caller przekazuje wyłącznie opaque `enrollment_reference` z signed package.
  LPPI unique-indexuje `(trust_domain, enrollment_reference)` i zwraca istniejący subject oraz
  operation. Retry nie może podać ani zmienić `psub`.

PDSA retained registry rozstrzyga również cardinality: jeden ACTIVE production enrollment subject
ma najwyżej jeden account-genesis slot. Ponowne wydanie package dla tego samego subject jest
jawnie signed `REPLACEMENT` z predecessor package ID; nie jest nowym subject.

## 6. First-install bootstrap ceremony

### Porównanie

| Model | First authenticator i pre-existing root | Offline / SaaS | Fresh laptop | Self-mint resistance | CI/test equivalent | Reinstall recovery |
|---|---|---|---|---|---|---|
| A — signed offline package | PDSA podpisuje; root pinned w signed LPPI release policy | pełne offline; brak SaaS | tak, po dostarczeniu package out-of-band | tak, jeśli root jest immutable dla profilu i private key poza hostem | osobny TEST root i generator fixture, nigdy production key/store | ten sam package + authority backup/TPM head albo signed recovery package |
| B — authenticated online | zdalna Enrollment Authority; root w kliencie | wymaga sieci podczas enrollment, nie podczas runtime; zwykle service dependency | tak | tak przy mTLS/attestation, ale operator endpointu staje się krytyczny | izolowany test endpoint/root | zdalny authoritative lookup |
| C — hardware-rooted local | OEM/enterprise TPM endorsement chain | offline, ale wymaga kwalifikowanego hardware i wiarygodnej EK policy | tylko wspierany sprzęt | lokalny admin nie mintuje EK, lecz ownership osoby/produktu nie wynika z samego TPM | vTPM z TEST EK hierarchy | motherboard replacement wymaga zewnętrznej ceremony |
| D — hybrid | external enrollment + TPM | offline po enrollment; opcjonalny remote checkpoint | tak | najwyższa, ale większa operacyjna złożoność | TEST CA + vTPM | remote/offline recovery authority |

### Wybrany production model

Wybrany zostaje **A — SIGNED OFFLINE PROVISIONING/ENROLLMENT PACKAGE**, uzupełniony TPM dopiero
jako lokalną custody/rollback protection po uwierzytelnieniu package. To nadal model A ceremony,
nie model D: zdalny serwis nie uczestniczy obowiązkowo ani w enrollment, ani w normalnym runtime.

Start od zupełnie świeżego Windows:

1. Użytkownik otrzymuje oficjalnie podpisany MSI. MSI nie zawiera package ani private
   root/signing key.
2. MSI instaluje nieenrolled kod LPPI i service definition. Nie przyjmuje package w MSI custom
   action i nie tworzy membership.
3. LPPI generuje non-exportable TPM/CNG pre-enrollment key i eksportuje wyłącznie signed/quoted
   enrollment request zawierający EK/AK evidence, public key, product/profile i fresh PDSA
   challenge. Ten request nie ustanawia trust, subject, operation ani device identity.
4. PDSA poza laptopem weryfikuje attestation, challenge i product authorization, mintuje `psub`
   oraz immutable `enrollment_reference`, zapisuje issuance w retained registry i podpisuje exact
   PRODUCTION package związany z tym TPM/request key.
5. Użytkownik dostarcza package out-of-band. Dowolny lokalny administrator może przesłać jego
   bytes do LPPI, lecz transport caller nie staje się authority.
6. LPPI sam weryfikuje pinned PDSA chain, signature, environment, product/profile, validity,
   issuance sequence i brak conflicting enrollment. Dopiero wtedy generuje non-exportable local
   authority key (odrębny od pre-enrollment key, chyba że frozen key-lifecycle contract jawnie
   autoryzuje promotion) i wiąże public key/attestation w signed enrollment acceptance.
7. LPPI zapisuje subject registry i inicjalizuje TPM NV anchor. Package związany z innym TPM,
   request key lub challenge jest odrzucany, co zapobiega klonowaniu package na drugi laptop.
   Następnie LPPI rozpoczyna model C.
8. Po `MEMBERSHIP_COMMITTED` provisioner może ustawić/uruchomić backend. CoreHost wcześniej nie
   jest dopuszczony do startu.

```text
ENROLLMENT TIME = offline PDSA package verification + LPPI/TPM enrollment + model C
NORMAL RUNTIME = local LPPI registry + TPM anchor + local PostgreSQL; zero cloud dependency
```

### Offline challenge round trip

Challenge nie jest nonce'em wymyślonym przez laptop. PDSA tworzy i podpisuje artefakt
`PDSAEnrollmentChallengeV1` zawierający `challenge_id=pchal_<uuidv7>`, niezależny 256-bitowy
losowy nonce, `PRODUCTION` trust domain, oczekiwany product/profile, policy generation, issuance i
expiry UTC oraz signature/key identity. Ważność wynosi dokładnie 7 dni; czas jest tylko bramką
ważności, nigdy winner selection. PDSA zapisuje przed wydaniem immutable rekord `ISSUED`.

```text
PDSA -> signed one-time challenge artifact
offline transfer -> laptop / LPPI
LPPI -> TPM-bound quoted PreEnrollmentRequestV1 exact-bound to challenge ID, nonce and bytes digest
offline transfer -> PDSA
PDSA -> verify ISSUED + unexpired + unused challenge, TPM quote/key attestation and request
PDSA -> atomically persist CONSUMED(request_digest, package_id, package_digest)
PDSA -> issue/return the one signed enrollment package
```

PDSA najpierw durably zapisuje `CONSUMED` oraz canonical package bytes, a dopiero potem zwraca
package. Utrata odpowiedzi nie otwiera nowego challenge: retry identycznego request digest zwraca
te same package ID i bytes. Ten sam challenge z innym requestem jest permanentnym
`CHALLENGE_REPLAY_CONFLICT`. Consumed challenge nigdy nie wraca do `ISSUED`. Niezużyty challenge
po expiry przechodzi w terminalny `EXPIRED`; wymaga nowego, podpisanego challenge PDSA. Caller-
generated ID/nonce, zmiana choć jednego bajtu challenge, brak retained record albo TEST challenge
w PRODUCTION są odrzucane.

### Root-of-root release policy

Revision 4 wybiera następujący chain niezależny od samego MSI:

```text
RELEASE_POLICY_SIGNER = CryptoHunter Release Policy Signing Authority (offline subordinate Ed25519)
RELEASE_POLICY_TRUST_ROOT = versioned 2-of-3 CryptoHunter Product Release Root Ed25519 key set
ROOT_STORAGE = three separately controlled offline HSM-equivalent custodians; public set compiled in verifier
VERIFICATION_CODE_OWNER = CryptoHunter Product Security / Release Engineering
ROOT_ROTATION = higher-generation RootSetTransitionV1 signed by 2 current roots
ROOT_REVOCATION = higher-generation ReleasePolicyRevocationV1 signed by 2 non-revoked current roots
TEST/PRODUCTION_SEPARATION = disjoint root sets, signer keys, policy IDs, domains and package registries
```

LPPI binary zawiera publiczny genesis root set i minimalny verifier; digest root setu oraz verifiera
jest częścią Authenticode-covered binary, ale **Authenticode potwierdza jedynie Windows package
publishera**. CryptoHunter release-policy authority jest osobnym chainem. PDSA challenge i
enrollment package exact-bindują dozwolony verifier measurement, production root-set ID,
`release_policy_generation` oraz policy digest. Dzięki temu sam publisher/MSI nie może podmienić
PDSA root i uznać go za trusted.

LPPI akceptuje policy tylko gdy jej podpis rozwiązuje się przez pinned root set, generation jest
nie mniejsza niż TPM-anchored accepted release-policy floor, environment/root-set ID są
PRODUCTION, signer nie jest revoked, a PDSA artifact exact-binduje policy digest. Nieznany signer,
caller-selected PDSA root, downgrade, TEST root/policy oraz stale/revoked policy są odrzucane.
Fresh install nie może offline poznać revocation opublikowanej po posiadanym release; dlatego PDSA
nie wydaje challenge/package dla revoked measurement/policy i w package wskazuje aktualny minimum.
Po enrollment nowe revocation/policy wchodzą wyłącznie przez wyżej podpisany, monotonic update lub
recovery artifact; brak sieci podczas zwykłego startu pozostaje dozwolony.

## 7. LPPI deployment topology i WiX ordering

Wybrany wariant:

```text
LPPI_INSTALLED_BY_CRYPTOHUNTER_MSI
service name = CryptoHunterProvisioningAuthority
service account = NT SERVICE\CryptoHunterProvisioningAuthority
start mode = DEMAND_START
CoreHost/backend before enrollment = NOT STARTED / NOT ADMITTED
```

Instalacja kodu nie ustanawia trust. Docelowa kolejność WiX, do zamrożenia przed zmianą MSI:

1. `InstallFiles`: zweryfikowane release binaries i pinned **public** PDSA trust policy;
2. `CreateFolders`/ACL: osobne katalogi LPPI; backend i interactive user bez write do authority
   code/policy/store;
3. `InstallServices`: utworzenie `CryptoHunterProvisioningAuthority` jako DEMAND_START;
4. ustawienie unrestricted service SID type i zakończenie SCM service creation, dzięki czemu
   service SID istnieje i może zostać rozwiązany przed ACL zasobów;
5. nadanie exact ACL katalogom/pipe/TPM key handle dla service SID;
6. commit MSI bez account/device enrollment i bez uruchamiania CoreHost;
7. dopiero post-install provisioner żąda SCM start LPPI i łączy się przez named pipe, weryfikując
   server SID/process image/signature; pipe DACL dopuszcza service oraz lokalnego administratora
   wyłącznie do przesłania untrusted enrollment request/package, a authority wynika wyłącznie z
   package PDSA, nigdy z client token;
8. LPPI weryfikuje package i dopiero wtedy tworzy TPM-backed key/NV state;
9. account/device provisioning kończy się przed zmianą backendu na docelowy start i pierwszym
   CoreHost launch.

To usuwa chicken-and-egg przed `ProvisionMachine`: MSI nie potrzebuje canonical account/device
scope do instalacji LPPI. `ProvisionMachine` w przyszłym kontrakcie Stage 9 staje się
post-install orchestrator enrollment, nie deferred MSI authority action. Obecnego MSI ani
provisionera revision 2 nie zmienia.

Niezmienne zakazy:

```text
MSI != provisioning authority
MSI != account authority
MSI != subject mint authority
MSI != root proof issuer
```

MSI może instalować code/service/ACL i przekazać opaque reference. Nie może wygenerować trusted
root, zaakceptować caller key, użyć TOFU ani wydać membership.

## 8. Device i operator provenance

### `device_installation_id`

```text
schema = dev_<canonical lowercase UUIDv7>
reservation owner = LPPI operation registry
FIRST_DEVICE_SLOT authority = joint protocol; LPPI owns candidate slot, CHA owns commit admission
```

LPPI mintuje losowy candidate `dev` atomowo z `provisioning_operation_id` w `RESERVED`, po
uwierzytelnieniu PDSA subject. Candidate nie jest genuine DeviceInstallation i nie jest M0.10
`TRUSTED`. Staje się genuine admitted device identity dopiero, gdy CHA `GENESIS_COMMITTED` exact-
binds first-device commitment, a LPPI atomowo zapisze `MEMBERSHIP_COMMITTED`. M0.10 później i
wyłącznie ono ustanawia Core security/device `TRUSTED` state.

* Exact retry zwraca ten sam `dev`; zmiana device fields pod tym samym operation ID failuje.
* Reinstall tej samej instalacji używa signed recovery package/authority backup i tego samego
  `dev`; samo ponowne uruchomienie MSI nie mintuje urządzenia.
* Utrata authority state bez weryfikowalnej recovery evidence nie pozwala odzyskać ani remintować
  `dev`.
* `ADD_DEVICE` jest osobną, post-genesis operation class. Wymaga genuine account proof oraz
  account-owner/device-enrollment authorization, mintuje nowy `dev`, nigdy nie dotyka
  `FIRST_DEVICE_SLOT` i nigdy nie tworzy account genesis.

### `intended_operator_id`

```text
schema = op_<canonical lowercase UUIDv7>
candidate reservation owner = LPPI operation registry
final semantic owner = M0.10 initial-security authority
candidate ID == final OperatorIdentity ID
```

LPPI mintuje candidate `op` atomowo w tej samej `RESERVED` operation, nie przyjmuje caller ID.
Przed M0.10 jest to wyłącznie signed intended-operator binding: nie istnieje OperatorIdentity,
rola, authentication ani entitlement. Ciągłość candidate==final gwarantują: signed PDSA/LPPI
request, CHA commitment, immutable membership, M0.3 exact comparison oraz M0.10 atomic
initial-security creation wymagające exact ID. M0.10 tworzy genuine OperatorIdentity dopiero po
trwałym bootstrap consumption; conflict/już istniejący inny operator failuje closed. Mapping nie
jest potrzebny i jest zabroniony w first-run flow.

## 9. Frozen RootProofIssuer requester chain

Revision 4 zachowuje bez zmian frozen requester ownership:

```text
LPPI
-> signed provisioning evidence
-> CHA validates LPPI evidence
-> CHA builds root-proof issuance request
-> CHA signs with ACCOUNT_GENESIS_ROOT_PROOF_ISSUANCE_REQUESTER_V1
-> RootProofIssuer
-> proof
-> CHA validates proof
-> CHA finalizes account genesis
```

LPPI nie wywołuje `RootProofIssuer` jako requester, nie posiada requester credential i nie może
podpisać requestu w roli CHA. LPPI evidence jest inputem CHA, nie root proof. RootProofIssuer
odrzuca każdy submitter principal/key/role inny niż genuine allow-listed CHA. Claimant/PDSA/LPPI,
CHA requester, root-proof signing, provisioning head i freshness credentials nie mogą aliasować.

## 10. Exact LPPI state machine

Każdy transition jest jednym durable LPPI registry transaction: append immutable transition,
compare current predecessor, update current designation i — od `RESERVED` wzwyż — advance
provisioning head/TPM anchor według write protocol. Stan terminalny nie może być cofnięty.

| Stan | Durable owner | Dozwolony predecessor i efekt | Exact retry / crash | Terminalność |
|---|---|---|---|---|
| `UNSEEN` | brak rekordu; PDSA package pozostaje external evidence | brak current operation | retry lookup po enrollment reference nadal nie mutuje | nie |
| `RESERVED` | LPPI | `UNSEEN`; atomowo zapisuje operation, `psub`, candidate `dev`, `op` i canonical account-genesis request bez `account_id` | retry zwraca ten sam rekord; crash przed commit=`UNSEEN`, po commit=`RESERVED` | nie |
| `CHA_PREPARED` | LPPI projection; authoritative prepare jest CHA | `RESERVED`; exact CHA operation/reservation receipt zapisany | lost response rozwiązuje CHA po exact operation; nigdy nowy ID | nie |
| `ROOT_PROOF_ACCEPTED` | LPPI projection; proof authority pozostaje RootProofIssuer/CHA validation | `CHA_PREPARED`; exact proof ID/digest i CHA validation receipt | reread/revalidate; mismatch fail closed | nie |
| `GENESIS_COMMITTED` | LPPI projection; genuine genesis owner pozostaje CHA | `ROOT_PROOF_ACCEPTED`; authenticated CHA committed-genesis receipt | lost response rozwiązuje exact CHA result; account jest genuine, membership jeszcze niewidoczne | nie; recovery obowiązkowe |
| `MEMBERSHIP_COMMITTED` | LPPI provisioning authority | `GENESIS_COMMITTED`; atomic signed claim, membership, current designation i terminal operation result | exact retry zwraca same bytes/IDs; M0.3 dopiero teraz może resolve | tak, success |
| `FAILED_TERMINAL` | LPPI | tylko `RESERVED` lub `CHA_PREPARED`, i wyłącznie authenticated proof że CHA/root-proof side effect nie mógł commitować; zapisuje immutable reason | exact retry zwraca same failure; unknown outcome nie może tu wejść | tak, failure |

`ROOT_PROOF_ACCEPTED` nie może przejść do `FAILED_TERMINAL`, ponieważ proof mógł prowadzić do
commit; wymaga resolution. `GENESIS_COMMITTED` nigdy nie abortuje i musi być doprowadzony do tego
samego `MEMBERSHIP_COMMITTED`. Timeout nie jest dowodem failure.

Revision 4 nie tworzy drugiego CHA lifecycle. CHA używa istniejących frozen reservation,
`INITIAL_BINDING -> PREPARED -> COMMITTED`, operation-attempt, root-proof admission,
freshness CAS/finalization i receipt contracts. LPPI przechowuje authenticated references do tych
stanów, a nie ich kopie udające authority.

## 11. CHA/LPPI recovery matrix i atomic meaning

`ATOMIC_ACCOUNT_PLUS_FIRST_DEVICE` oznacza jedno externally visible outcome: M0.3 membership jest
niewidoczne przed committed account+first-device commitment. Nie oznacza distributed SQL.

| Cut / obserwacja | Authoritative read | Recovery | Zakaz |
|---|---|---|---|
| crash przed LPPI `RESERVED` | brak LPPI operation | retry z tym samym enrollment ref może utworzyć pierwszą operation | caller operation ID |
| `RESERVED`, brak CHA prepare | LPPI exact record | wyślij/retry ten sam canonical request do CHA | nowe acct/dev/op |
| CHA prepared, brak odpowiedzi | CHA attempt/current registry | resolve/reconcile, zapisz `CHA_PREPARED` | blind new attempt |
| root proof issued, LPPI stale | RootProofIssuer retained history + CHA attempt | CHA waliduje ten sam proof; LPPI zapisuje projection | LPPI jako requester |
| CHA commit unknown | CHA authenticated genesis resolver + freshness receipt | committed -> `GENESIS_COMMITTED`; proven pre-commit safe failure -> terminal; unavailable -> block | timeout jako abort |
| CHA committed, LPPI przed membership | CHA committed receipt | deterministycznie zbuduj te same claim/membership bytes i commit LPPI | drugi genesis/first device |
| LPPI membership committed, ack lost | LPPI current/history + TPM head | zwróć zapisany terminal result | ponowna publikacja innej treści |
| LPPI DB behind TPM | TPM generation/head wyższy | fail closed, restore exact authority backup lub PDSA recovery ceremony | rollback TPM |
| LPPI DB ahead TPM | incomplete anchor advance | recovery wykonuje exact pending NV compare/advance tylko dla signed retained transition; inaczej fail closed | uznanie DB za current |
| reinstall z zachowanym TPM i backupem | package, backup head i TPM head zgodne | restore LPPI authority, zweryfikuj, bez nowych IDs | bootstrap od zera |
| nowa płyta/TPM | stary anchor niedostępny | wyłącznie PDSA signed replacement/recovery package z predecessor head i retained registry check | lokalny reset jako recovery |

## 12. Provisioning operation identity i idempotency

```text
provisioning_operation_id owner = LPPI
schema = prvop_<canonical lowercase UUIDv7>
issuance = inside the same LPPI SERIALIZABLE reservation transaction as candidate dev/op
lookup key = (production trust_domain, authenticated enrollment_reference)
```

LPPI jest mint ownerem operation ID dopiero po weryfikacji package PDSA. Caller nie przekazuje ID
przy create/retry i nie może wymusić nowej operation. Exact retry zaczyna się od signed opaque
enrollment reference, rozwiązuje current subject i unique account-genesis slot, po czym zwraca
istniejące `prvop`. UUID clock/order nie wybiera winnera; winnerem jest jedyny durable record pod
unique authenticated lookup key.

Canonical request obejmuje co najmniej: schema/domain version, PRODUCTION trust domain, package
ID/digest, enrollment reference, `psub`, operation class, candidate `dev`/`op`, żądanie nowego
account genesis **bez caller/LPPI-selected `account_id`**, LPPI key ID/version, challenge i
predecessor provisioning head. CHA dopisuje minted `account_id` do własnej immutable reservation
i authenticated receipt; LPPI wiąże go dopiero w `CHA_PREPARED`. Request po reservation jest
immutable.

```text
same operation identity + same canonical request
-> recover/return same state, IDs and terminal decision

same operation identity + different canonical request
-> FAIL_CLOSED / IDEMPOTENCY_CONFLICT; zero mutation

same enrollment reference + proposed new operation while nonterminal or success exists
-> return existing operation
```

Wall-clock, PID, caller random retry ID i local startup order nigdy nie rozstrzygają replay.

## 13. TPM 2.0 anti-rollback i finalna topologia root-of-trust

Revision 4 wycofuje shorthand `(generation,digest)` CAS. TPM 2.0 nie jest arbitrary tuple CAS,
a sam standard TPM nie dowodzi, że wymagany command path jest dostępny dla produktu przez Windows.
W szczególności **nie zamrażamy** `TPMA_NV_PLATFORMCREATE` ani rzekomego
„controlled platform-provisioning boundary”. Normalne Windows 10/11 automatycznie provisionuje TPM;
CryptoHunter/MSI nie posiada z tego powodu platform hierarchy authorization ani owner authorization.

### Windows TPM/NV access inventory

| Element | Ustalenie dla zwykłego Windows 11 | Konsekwencja |
|---|---|---|
| TPM Base Services (`TBS`) | `Tbsi_Context_Create`/`Tbsip_Submit_Command` są transportem raw command bytes i arbitrem współdzielenia TPM; nie nadają hierarchy authorization. Windows stosuje własną command-blocking policy. | Sam sukces otwarcia TBS nie dowodzi prawa do `TPM2_NV_DefineSpace`, policy session ani increment. Bez zmiany systemowej allow-list probe musi przejść na konfiguracji domyślnej. |
| `TPM2_NV_DefineSpace` | Command wymaga autoryzacji `authHandle`; dla `TPM_RH_PLATFORM` tworzy index z `TPMA_NV_PLATFORMCREATE`, dla `TPM_RH_OWNER` owner-created index bez tego bitu. | Bit atrybutu nie jest sposobem uzyskania authority. Caller musi już móc autoryzować wskazaną hierarchy. |
| `TPM_RH_PLATFORM` | Platform hierarchy jest kontrolowana przez firmware/OS; produkt nie może zakładać znajomości ani ustawienia jej authorization. | Wariant A (`PLATFORMCREATE`) jest **odrzucony jako baseline** bez OEM/firmware contract. |
| `TPM_RH_OWNER` | Windows auto-provisioning przejmuje/provisionuje TPM i zarządza storage owner authorization; produkt nie może zakładać, że otrzyma reusable owner secret. | Wariant B jest wyłącznie kandydatem do realnego probe, nie zamrożonym prawem produktu. |
| auto-provisioning / owner clear | Wyłączenie auto-provisioning, manual takeover, clear lub przejęcie hierarchy zmienia systemowy trust lifecycle. | Żaden z tych kroków nie jest dopuszczalnym installer prerequisite ani remediation. |
| Windows privilege | Administrator/elevated service może być wymagany przez Windows/TBS dla uprzywilejowanych komend, ale token administratora nie jest hierarchy authorization. | Probe działa jako dedykowany elevated test process/service i raportuje zarówno Win32/TBS, jak i TPM response code; MSI custom action nie jest substytutem proof. |
| physical presence | Zdefiniowanie owner-created index co do zasady nie wymaga physical presence w TPM; platform/firmware policy może ją wymusić dla platform operations. | Baseline nie może wymagać UEFI promptu ani physical-presence ceremony. |
| command allow-list | Windows może blokować komendy niezależnie od TPM authorization. Registry override/zmiana allow-list jest poza produktem. | Każda wymagana komenda musi przejść bez systemowego override. |
| Microsoft Platform Crypto Provider | CNG/KSP udostępnia TPM-backed asymetryczne keys i operacje key custody; nie jest udokumentowanym API do arbitralnego product-owned NV countera. | KSP może chronić signing key, lecz nie zastępuje monotonic NV generation. |

Źródła normatywne do qualification: Microsoft Learn — [TPM Base Services](https://learn.microsoft.com/windows/win32/tbs/tpm-base-services-portal),
[TPM fundamentals / auto-provisioning](https://learn.microsoft.com/windows/security/hardware-security/tpm/trusted-platform-module-overview),
[TPM group policy and blocked commands](https://learn.microsoft.com/windows/security/hardware-security/tpm/trusted-platform-module-services-group-policy-settings),
[Microsoft Platform Crypto Provider](https://learn.microsoft.com/windows/win32/api/ncrypt/nf-ncrypt-ncryptopenstorageprovider),
oraz TCG — [TPM 2.0 Library Specification](https://trustedcomputinggroup.org/resource/tpm-library-specification/),
Part 3 (`NV_DefineSpace`, `PolicyNV`, `PolicyCommandCode`, `PolicySigned`, `PolicyAuthorize`,
`NV_Increment`, `NV_Read`). Link do KSP opisuje wspólny Windows CNG provider model; qualification
musi dodatkowo potwierdzić dokładny provider name i hardware properties, nie tylko nazwę providera.

### Porównanie creation authority i decyzja

| Wariant | Creation | Runtime | Ocena na normalnym consumer Windows |
|---|---|---|---|
| A — platform-created | `TPM_RH_PLATFORM`, `TPMA_NV_PLATFORMCREATE` | policy-only | **NIE WYBRANY**: produkt nie ma udowodnionej platform auth; OEM/UEFI boundary byłby innym deployment contract. |
| B — owner-created | `TPM_RH_OWNER`, bez `PLATFORMCREATE` | docelowo policy-only po define | **PREFEROWANY KANDYDAT MODELU A (counter + signed DB head)**, ale creation przez auto-provisioned Windows owner i policy command path wymagają realnego proof. |
| C — higher-level Windows primitive | Microsoft Platform Crypto Provider/CNG key | sign/unwrap, brak app counter API | Wspierany dla keys, ale nie daje udokumentowanego monotonic countera ani compare-and-increment dla application state. Nie jest równoważnym lokalnym anti-rollback substrate. |

### A. Existing architecture i wynik live proof

Wcześniejszy kandydat łączył trzy operacje (`increment`, `read`, `compare`) w jednym
`PolicyOR`, ale nie zamykał źródła authority dla sesji `S_cmp`, a read/compare nie są
alternatywnymi sposobami wykonania chronionego `NV_Increment`. Taki układ nie jest finalną
polityką. `PolicyOR` nie służy do agregowania command permissions i nie wolno go zastąpić
`PolicyOR(branch, branch)`.

Fizyczny Windows 11 / TPM 2.0 potwierdził owner-created counter, read, increment, policy session,
`PolicyNV`, restart persistence, stale-disk detection oraz rzeczywiste `LoadExternal`,
`VerifySignature` i `PolicyAuthorize`. Ticket miał `TPM_ST_VERIFIED`, `TPM_RH_OWNER` i niepusty
digest; finalny `PolicyGetDigest` był równy niezależnemu expected digest. Cleanup external key,
policy session, NV, service, scratch i TBS context przeszedł. Jest to **LIVE PROVEN feasibility**,
nie produkcyjny key ani produkcyjny policy vector. Acceptance key pozostaje disposable i nigdy
nie jest Product Release Root ani jego subordinate.

```text
NV_DEFINE_AUTHORITY = TPM_RH_OWNER (LIVE PROVEN, empty owner authorization on qualified host)
NV_PUBLIC = TPM_NT_COUNTER | TPMA_NV_POLICYWRITE | TPMA_NV_AUTHREAD | TPMA_NV_NO_DA
NV_AUTH_VALUE = empty; read/PolicyNV comparison is intentionally public, write is not
WINDOWS_API_PATH = TBS 2.0 raw-command path (LIVE PROVEN)
REQUIRES_PHYSICAL_PRESENCE = NO
ROOT_OF_TRUST_FREEZE = BLOCKED ON PRODUCTION KEY MATERIAL + CANONICAL VECTOR, NOT COMMAND FEASIBILITY
```

`TPMA_NV_OWNERWRITE` nie występuje w produkcyjnym template: po `DefineSpace` owner path nie może
ominąć write policy. `AUTHREAD` z pustym `authValue` jest celowe, ponieważ counter nie jest sekretem
i usuwa self-read circularity `NV.authPolicy -> PolicyNV -> NV.authPolicy`. Ujawnienie generation
nie daje prawa do increment. Owner-created index nie używa `TPMA_NV_PLATFORMCREATE`.

### B. Dependency graph (DAG)

```text
Product Release Root (offline, immutable 2-of-3 Ed25519 trust root)
├── signs ReleasePolicyV1
│   ├── authorizes exact PDSA verification-key set and profile
│   ├── authorizes one TPM-compatible Recovery Policy Key public area/Name
│   └── fixes NV template, hash algorithm, policyRef values and canonical encodings
└── never signs its own trust statement and never signs a runtime approvedPolicy

ReleasePolicyV1 + PDSA private authority (off-host)
└── PDSAEnrollmentPackageV1
    ├── binds psub + TPM/EK/AK evidence + LPPI pre-enrollment key
    ├── binds local Protected State Authority policy key Name (K_PSA.Name)
    └── binds ReleasePolicyV1 digest and NV template version

PDSAEnrollmentPackageV1 + K_PSA.Name + K_RECOVERY.Name
└── deterministic final NV authPolicy digest
    └── owner-created TPM monotonic NV counter
        ├── normal write branch: PolicyNV(g) -> PolicyCommandCode(Increment)
        │   -> PolicyAuthorize(K_PSA) -> PolicyOR
        └── recovery write branch: PolicyCommandCode(Increment) -> PolicyCpHash(exact command)
            -> PolicyAuthorize(K_RECOVERY) -> PolicyOR

NV generation + signed Protected State Authority history
└── authenticated LPPI current/pending head
    ├── LPPI evidence -> CHA validation -> CHA-only RootProofIssuer request
    │   -> RootProofIssuer retained proof -> CHA genesis decision
    ├── Product Protected State Authority (separate key/history/domain)
    └── Product Secret Resource Authority (separate KEK/ciphertext/domain)

CHA-only RootProofIssuer, Protected State Authority and Secret Resource Authority
└── are consumers/siblings of the enrolled root; none creates, signs or selects authPolicy

WindowsExternalProvisioningHandoff
└── remains unintegrated until ReleasePolicyV1, production Names and canonical vector exist
```

The edge direction is "must already exist/verify before". There is no edge from the NV counter,
LPPI, CHA, RootProofIssuer, Protected State history or Secret Resource state back to Product Release
Root, ReleasePolicyV1, PDSA identity, or either policy-key Name.

### C. Circularity analysis

The earlier `S_cmp` design had a concrete cycle:

```text
NV increment policy needs PolicyNV authorization S_cmp
S_cmp uses the same NV authPolicy
same NV authPolicy was being defined from the branches that included S_cmp
```

Revision 4 removes it rather than hiding it: `PolicyNV` reads the non-secret counter using an
ordinary password session and the fixed empty NV `authValue` permitted by `TPMA_NV_AUTHREAD`.
That session is not a policy branch and cannot authorize a write.

A second potential cycle would be `Product Release Root -> approve final digest -> final digest
contains Product Release Root approval`. It is also absent. The root signs the byte-complete
`ReleasePolicyV1`, including subordinate public material and policy grammar, but never the final
NV digest and never a statement containing its own signature. The final digest contains TPM Names
of `K_PSA` and `K_RECOVERY`, not an Ed25519 signature or package digest. The PDSA package is verified
**before** defining the index; the index does not authorize that package. No digest depends on a
value which itself must be authorized by that digest.

### D. Proposed non-circular final construction

There are exactly two write branches:

| Branch | Purpose | Inputs | Authority | Failure/recovery semantics |
|---|---|---|---|---|
| normal | one ordinary `g -> g+1` after a durable, signed pending head exists | canonical NV Name, observed `g`, fixed normal `policyRef`, enrolled `K_PSA.Name` | local TPM-bound PSA policy key, whose Name was bound by PDSA under the immutable release policy | mismatch, missing pending record, invalid signature or lost-response ambiguity fails closed; reread reconciles without a second increment |
| bootstrap/recovery | initialize the counter once, or perform one explicitly approved increment during off-host recovery when normal PSA custody/state is unavailable | exact `NV_Increment` cpHash, fixed recovery `policyRef`, immutable release `K_RECOVERY.Name`, signed enrollment/recovery case | offline Recovery Policy Key authorized by Product Release Root in `ReleasePolicyV1` | initializes the otherwise unreadable fresh counter, but provides no general reset/undefine; later ceremonies may only move forward and must restore/finalize the exact retained predecessor/pending head |

They are semantically and cryptographically distinct: different prerequisite commands, different
keys, different `policyRef`, and different operational authorities. Read is deliberately not a
third branch. `PolicyOR` appears once, only to make both genuine write paths satisfy the single NV
`authPolicy`.

### E. Exact TPM policy command sequence per branch

```text
NORMAL(g):
  StartAuthSession(TPM_SE_POLICY) -> S_normal
  PolicyNV(S_normal, authHandle=NV_COUNTER, nvIndex=NV_COUNTER,
           operandB=UINT64_BE(g), offset=0, operation=TPM_EO_EQ,
           authorization=password-session(empty NV authValue))
  PolicyCommandCode(S_normal, TPM2_CC_NV_Increment)
  approvedPolicy_normal(g) = PolicyGetDigest(S_normal)
  VerifySignature(K_PSA, SHA256(approvedPolicy_normal(g) || REF_NORMAL), sig) -> ticket
  PolicyAuthorize(S_normal, approvedPolicy_normal(g), REF_NORMAL, K_PSA.Name, ticket)
  PolicyOR(S_normal, [BRANCH_NORMAL, BRANCH_RECOVERY])
  NV_Increment(authHandle=NV_COUNTER, nvIndex=NV_COUNTER, authorization=S_normal)

BOOTSTRAP_OR_RECOVERY(exact command):
  StartAuthSession(TPM_SE_POLICY) -> S_recovery
  PolicyCommandCode(S_recovery, TPM2_CC_NV_Increment)
  PolicyCpHash(S_recovery, cpHashA=SHA256(canonical TPM2_NV_Increment command Names/parameters))
  approvedPolicy_recovery = PolicyGetDigest(S_recovery)
  VerifySignature(K_RECOVERY,
                  SHA256(approvedPolicy_recovery || REF_RECOVERY), sig) -> ticket
  PolicyAuthorize(S_recovery, approvedPolicy_recovery, REF_RECOVERY,
                  K_RECOVERY.Name, ticket)
  PolicyOR(S_recovery, [BRANCH_NORMAL, BRANCH_RECOVERY])
  NV_Increment(authHandle=NV_COUNTER, nvIndex=NV_COUNTER, authorization=S_recovery)

READ / COMPARE (not a PolicyOR branch):
  NV_Read(authHandle=NV_COUNTER, nvIndex=NV_COUNTER,
          authorization=password-session(empty NV authValue), size=8, offset=0)
```

Both `PolicyOR` calls use the identical ordered two-element list. Duplicate or semantically cloned
branches are invalid. `PolicyAuthorize` occurs once in each real branch, after branch-specific
constraints and before `PolicyOR`.

### F. Expected digest derivation and exact production root policy

For SHA-256, `Z = 00^32`, `CC(x) = UINT32_BE(TPM2_CC_x)`, all TPM2B/Name values use canonical TPM
wire bytes, and `NameNV` is recomputed from the final public area (including `NV_WRITTEN` handling
required by the TPM):

```text
argsNV(g) = SHA256(UINT64_BE(g) || UINT16_BE(0) || UINT16_BE(TPM_EO_EQ))
N1(g) = SHA256(Z || CC(PolicyNV) || argsNV(g) || NameNV)
N2(g) = SHA256(N1(g) || CC(PolicyCommandCode) || CC(NV_Increment))

R1 = SHA256(Z || CC(PolicyCommandCode) || CC(NV_Increment))
R2 = SHA256(R1 || CC(PolicyCpHash) || cpHashA)

BRANCH_NORMAL = SHA256(
    SHA256(Z || CC(PolicyAuthorize) || Name(K_PSA)) || REF_NORMAL)
BRANCH_RECOVERY = SHA256(
    SHA256(Z || CC(PolicyAuthorize) || Name(K_RECOVERY)) || REF_RECOVERY)

PRODUCTION_ROOT_POLICY_DIGEST = SHA256(
    Z || CC(PolicyOR) || BRANCH_NORMAL || BRANCH_RECOVERY)
```

This is the exact production root policy definition and is independent of `g`, disk head, LPPI
state, signatures and tickets. Its concrete 32-byte hex value is **per enrolled device**, because
`Name(K_PSA)` is per device. It MUST be materialized and independently recomputed in the enrollment
record before `NV_DefineSpace`; inventing a fleet-wide hex value before production public keys exist
would be false evidence. `N2(g)` and `R2` are mutable approved policies; `BRANCH_*` and the final
root digest are immutable for the lifetime of that NV index.

### G. Recovery semantics

The same branch performs the first increment after `DefineSpace`, because an unwritten TPM counter
cannot satisfy `PolicyNV(g)`. Its exact cpHash and enrollment record are approved by the offline
authority; after initialization all ordinary transitions use the normal branch.

Recovery never clears TPM, takes ownership, uses platform hierarchy, changes firmware, disables
auto-provisioning, or rewinds generation. PDSA/recovery operators validate the retained predecessor,
physical counter, signed pending head and device/subject binding. They issue one signature for
`R2 || REF_RECOVERY`; `PolicyCpHash` limits it to the exact counter command. After an ambiguous
response, recovery first reads NV: `g+1` finalizes the same pending record, while `g` permits at most
one retry. Any other gap fails closed. TPM replacement requires a separately signed replacement
package and new enrollment/index; it is not recovery of the old counter.

### H. Migration/update semantics

Mutable data are `g`, pending/current signed DB heads, transition/history records, session nonces,
tickets and branch signatures. Immutable-per-index data are the NV public template, hash algorithm,
`K_PSA.Name`, `K_RECOVERY.Name`, both refs, ordered branch list and final root digest.
`ReleasePolicyV1`, the Product Release Root identity and PDSA trust profile are immutable per release
policy version. Rotation of either TPM policy key, a ref, algorithm or template creates a new index
under an explicit predecessor-bound migration ceremony; it never edits the old `authPolicy`.
Ordinary software/data updates only append a signed pending head then advance the counter once.

### I. Threat-model consequences

The TPM enforces monotonicity, equality to observed `g` on the normal path, exact command code, the
pinned branch authority and the two-branch final digest. The signed DB history binds generation to
`head_digest`, predecessor, operation and transition; the TPM does not store `head_digest`. Thus
`disk.g < TPM.g` is detected and cannot be accepted, `disk.g == TPM.g` must match the signed chain,
and `disk.g > TPM.g` is only a recoverable exact pending transition, never proof that disk is
current. A local malicious administrator can still deny service, read the public counter, tamper
with processes or attempt credential use; full resistance requires separate WDAC/Secure Boot/
measured-boot enforcement and remains outside this guarantee.

### J. Minimal implementation plan

1. Publish production `ReleasePolicyV1` schema and provision real Product Release Root/PDSA/
   Recovery Policy Key public material outside the acceptance probe.
2. Add a pure canonical policy-vector generator and independent verifier for the equations above;
   reject duplicate branches, wrong ordering, non-final public areas and acceptance keys.
3. Capture a byte-complete enrollment vector containing both Names, refs, branches and final hex
   digest; cross-check it with TPM trial sessions.
4. Extend a disposable physical test with the **two genuine branches**, including empty-auth
   `AUTHREAD`, normal `PolicyNV`, recovery `PolicyCpHash`, and negative cross-branch tests.
5. Only after that evidence, implement the production NV owner and PSA/recovery ceremony. Keep
   `WindowsExternalProvisioningHandoff` and MSI unchanged until the production vector is frozen.

This construction is non-circular. Two-branch and canonical-vector physical evidence now exists,
but production material has not been provisioned. Therefore no production integration is authorized
by this proposal.

The first static implementation layer now consists of:

* `deployment/stage9_release_policy_v1.schema.json`, which constrains release-root/PDSA material,
  the recovery key, the PSA key **profile**, domain-separated refs, branch ordering and the exact NV
  template, but deliberately does not contain per-device `K_PSA`;
* `deployment/stage9_enrollment_policy_material_v1.schema.json`, which represents the separately
  PDSA-authorized per-device `K_PSA`, target EK/AK binding and exact release-policy digest;
* `deployment/windows_stage9_policy_vector.py`, a pure canonical generator with no TPM I/O or
  private-key generation;
* `deployment/windows_stage9_policy_vector_verifier.py`, which independently restates and verifies
  the digest, Name, cpHash and lifecycle equations rather than accepting generator output;
* `tests/fixtures/windows_stage9_policy_vector_v1.json`, containing visibly `TEST_ONLY` public
  fixture material and a deterministic byte-complete vector. It is not production authority
  material and is rejected if relabelled as disposable acceptance material.

`deployment/windows_stage9_two_branch_probe.py` and the canonical cross-check use ephemeral
TEST-only keys and execute both genuine command paths plus negative cross-authority and
reversed-`PolicyOR` cases. They do not provision an authority or authorize MSI/handoff integration.
Production public keys and signatures remain external inputs to the freeze ceremony; no production
private key is generated or stored here. The public-only production freeze schemas, verifier,
migration rules and manifest contract are defined in
`docs/windows_stage9_production_root_of_trust_freeze.md`.

## 14. Product Protected State Authority

Rekomendowany minimalny production profile przed Stage 9:

```text
form = dedicated Windows service (nie CoreHost i nie in-process Core adapter)
service = CryptoHunterProtectedStateAuthority
identity = NT SERVICE\CryptoHunterProtectedStateAuthority
start = AUTO przed backendem, fail-closed dependency
storage = osobny ACL-protected PostgreSQL schema + append-only history/current map
key custody = distinct non-exportable CNG key backed by TPM
rollback anchor = osobny TPM NV index/domain, nie LPPI i nie AccountGenesis
current map key = (account_id, device_installation_id, state_store_identity_fingerprint_sha256)
```

Dedicated service jest wybrany zamiast in-process adaptera, ponieważ current map musi przeżyć
Core/StateStore rollback i Core nie może posiadać credential pozwalającego mintować/zmieniać
membership. Core-visible adapter używa local mutually authenticated named pipe i ograniczonych
operations exact M0.3 lifecycle.

Startup service weryfikuje binary/policy, complete signed history, exact one current reference per
scope, pending transition oraz własny TPM NV generation/head. Następnie rozwiązuje legalny
`PREPARED` wyłącznie według exact local durable evidence rules M0.3/M0.11. Missing/fork/DB-behind,
TPM replacement lub mismatch blokują backend i wymagają signed PDSA recovery ceremony właściwej
dla **protected-state domain**.

StateStore backup jest wyłącznie candidate. Nie zawiera current map, private key, NV anchor ani
protected membership i nie może ich mintować. Restore może zostać zaakceptowany wyłącznie po
porównaniu z niezależnym protected head oraz legalnym PREPARE/FINALIZE flow. LPPI może dostarczyć
scope po membership, lecz nie jest właścicielem protected map.

## 15. Product Secret Resource Authority

### Decision matrix

| Kandydat | Service access / disclosure | Rotation i cleanup | Crash reconciliation | Backup / machine replacement | Unattended startup |
|---|---|---|---|---|---|
| DPAPI machine blob | łatwy service access, lecz machine scope może ujawnić innym uprzywilejowanym principals bez ścisłych ACL | aplikacyjne | trzeba zbudować registry | image rollback/DPAPI backup nie daje freshness; migracja trudna | tak |
| DPAPI user/service blob | związany z profilem service, lepsza izolacja | aplikacyjne | trzeba zbudować registry | profil/DPAPI recovery krytyczny | możliwy, lecz profile/load semantics wymagają qualification |
| Windows Credential Manager | API gotowe, ale service identity/session i credential visibility wymagają qualification | wspiera update/delete | ograniczone external outcome evidence | słaba przenośność/machine replacement | niepewne dla virtual service account bez testów |
| CNG/TPM-backed handle | najlepsze dla non-exportable keys; nie przechowuje arbitralnych API secrets bez envelope layer | silna key rotation, cleanup handle | potrzebny durable descriptor/provider query | TPM-bound; recovery wymaga rewrap/reissue | tak |
| dedicated local vault abstraction | może łączyć service-only ACL, encrypted blobs i CNG wrapping | jawny versioned lifecycle | może realizować exact `begin/reconcile/cleanup` | jawna non-authoritative backup/recovery policy | tak po qualification |

Rekomendacja: **dedicated local vault service/abstraction z envelope encryption, którego KEK jest
non-exportable CNG/TPM-backed handle**, uruchamiany jako
`NT SERVICE\CryptoHunterSecretResourceAuthority`. Arbitralne sekrety są szyfrowane per-secret DEK;
KEK unwrap jest dostępny tylko service SID. Core i interactive user otrzymują opaque references,
nie raw vault storage ani KEK.

Vault posiada osobny append-only operation registry mapujący exact handoff ID/fingerprint na
external outcome. `begin` jest create-if-absent z exact metadata, `reconcile` zwraca retained
outcome po crashu, a `cleanup` jest idempotentnym tombstone+crypto-erase DEK/handle. Rotation tworzy
nową wersję/ref, atomowo publikuje ją przez istniejący M0.11 SecretHandoff lifecycle, a stara wersja
jest czyszczona dopiero w `CLEANUP_PENDING`.

Vault backup zawiera tylko ciphertext, metadata i signed operation history; nie jest authority i
bez TPM KEK nie ujawnia sekretu. Machine replacement wymaga external secret reissuance albo
jawnego PDSA-authorized KEK migration/recovery profile — nigdy kopiowania plaintext key. Brak
credential/vault outcome blokuje unattended backend zamiast proszenia interactive usera.
Produkcja wymaga qualification service startup, ACL, CNG provider, TPM absence/failure, rotation,
lost response, cleanup i machine replacement.

### Local TPM/CNG algorithm profile

Revision 4 nie zakłada Ed25519 w Microsoft Platform Crypto Provider. Lokalne role mają osobne
keys/usages i następujący profil, podlegający Windows/TPM qualification na wspieranej macierzy:

| Rola | Algorytm | Użycie i zakaz aliasowania |
|---|---|---|
| LPPI local authority signing | TPM-backed CNG ECDSA P-256 + SHA-256 | podpis `ProvisioningHeadV1`, operation/membership receipts; osobny key name/version |
| Product Protected State Authority | TPM-backed CNG ECDSA P-256 + SHA-256 | protected history/current receipts; nigdy LPPI key |
| provisioning NV authorization | TPM-backed CNG ECDSA P-256 + SHA-256 | wyłącznie `PolicySigned/PolicyAuthorize` dla jednego provisioning NV Name; nie podpisuje domain records |
| protected-state NV authorization | osobny TPM-backed CNG ECDSA P-256 + SHA-256 | wyłącznie protected NV Name/domain |
| Secret vault KEK | non-exportable TPM-backed CNG RSA-3072, OAEP-SHA-256 | wrap/unwrap per-secret AES-256 DEK; nie jest signing key |
| PDSA/release policy offline | Ed25519 | software/HSM-equivalent offline domain; nie jest Windows TPM-backed |

TPM public area, provider name, key attestation, algorithm, curve/key size, usage flags, export
policy, PCR/policy binding oraz key ID/version są exact-bound w enrollment/authority record.
Provider fallback do software key, unattested key, algorytm substitution albo reuse jednego key
dla dwóch ról failują closed. Existing frozen CHA/RootProofIssuer Ed25519 domains nie są zmieniane.

### Realistyczny local threat boundary

| Actor/threat | Chronione założenie | Gwarancja / brak gwarancji |
|---|---|---|
| ordinary interactive user | brak admin/service ACL i vault access | nie może pisać authority stores/policies, używać keys ani czytać plaintext secret |
| local administrator transportujący bytes | może startować demand service i przesłać untrusted artifacts | nie staje się authority; PDSA signatures, exact domains i TPM binding decydują |
| malicious local administrator z arbitrary code execution | może patchować user-mode binaries, ACL, DB, hookować procesy i próbować podszyć provider | **poza gwarantowaną odpornością revision 3**, chyba że przyszły osobny contract zamrozi WDAC, Secure Boot, measured boot/code i remote/offline attestation enforcement |
| kernel/firmware/TPM compromise | pełna privileged execution lub złamany hardware root | poza threat boundary; wykrycie może skutkować revocation/recovery, nie gwarancją zapobiegania |
| physical disk rollback/clone | attacker cofa cały dysk bez TPM private keys/NV | wykrywany przez fizyczny NV counter; clone nie ma właściwego TPM Name/keys |

TPM key attestation dowodzi pochodzenia i właściwości key tylko po pełnej walidacji attestation
chain/public area/policy. Sama nazwa CNG KSP lub deklaracja providera nie jest dowodem hardware
custody. Bez WDAC/measured-code enforcement proposal nie twierdzi, że malicious local admin nie
uruchomi zmodyfikowanego klienta lub providera; authority verifiers nadal failują closed wobec
niezgodnych podpisów/TPM evidence, lecz nie obiecują ochrony całego skompromitowanego OS.

## 16. Production service topology oraz MSI lifecycle

MSI jest jedynym ownerem instalacji i usuwania wszystkich SCM definitions. First-run enrollment
nie wywołuje service-create/delete i nie tworzy untracked service objects.

| Service | Po MSI, przed enrollment | Podczas enrollment | Po terminalnym sukcesie | Uzasadnienie |
|---|---|---|---|---|
| `CryptoHunterPostgreSQL` | `AUTO`, `RUNNING` | `RUNNING` | `AUTO`, `RUNNING` | prywatna DB jest potrzebna durable authorities; sama DB nie nadaje trust |
| `CryptoHunterProvisioningAuthority` | `DEMAND`, `STOPPED` | provisioner startuje; `RUNNING` | `DEMAND`, normalnie `STOPPED`; start dla recovery/`ADD_DEVICE` | provisioning nie jest wymagany na każdy backend start |
| `CryptoHunterProtectedStateAuthority` | `DEMAND`, `STOPPED` | start po authenticated subject, przed StateStore genesis | `AUTO`, `RUNNING`, dependency backendu | current restore map musi być dostępna przed Core recovery |
| `CryptoHunterSecretResourceAuthority` | `DEMAND`, `STOPPED` | start przed secret handoff | `AUTO`, `RUNNING`, dependency backendu | unattended secret resolution/reconciliation |
| `CryptoHunterFreshnessVerifier` | `DEMAND`, `STOPPED` | start dokładnie na potrzeby CHA/freshness validation | `DEMAND`, `STOPPED` poza operacją | verifier nie jest stałym authority store ani normal-runtime daemonem |
| `CryptoHunterBackend` | `DEMAND`, `STOPPED`, enrollment gate deny | nie uruchamia się | `AUTO`, `RUNNING` dopiero po readiness commit | CoreHost nie może powstać bez canonical scope |

MSI instaluje code, SCM records, service SID types, dependencies, failure actions i bazowe ACL dla
wszystkich sześciu usług. Enrollment wyłącznie startuje dozwolone demand services, generuje lub
provisionuje keys/state, ustanawia authenticated configuration, kwalifikuje readiness i wykonuje
idempotent readiness commit zmieniający docelową start policy Protected/Secret/Backend. Zmiana SCM
start policy jest orchestration effect związanym z terminalnym enrollment receipt, nie authority.
Failed enrollment pozostawia backend stopped/demand i nie promuje częściowej readiness.

### MSI repair, maintenance i same-version reinstall

Repair/maintenance/same-version reinstall zachowują ten sam podział ownership: MSI jest jedynym
ownerem wszystkich sześciu SCM definitions, a first-run ani repair nie tworzą nowych service
objects. Repair może wyłącznie kwalifikować istniejące MSI-owned definitions, naprawić MSI-owned
binary/service metadata (w tym SID type, dependencies, failure actions i bazowe ACL) oraz zastosować
start policy wynikającą z **uwierzytelnionego** enrollment/readiness receipt.

```text
authenticated UNENROLLED
-> CryptoHunterBackend = DEMAND / STOPPED
-> CryptoHunterProtectedStateAuthority = DEMAND / STOPPED
-> CryptoHunterSecretResourceAuthority = DEMAND / STOPPED

authenticated ENROLLED + terminal readiness receipt
-> CryptoHunterBackend = expected production policy (AUTO; start tylko po dependency/readiness checks)
-> CryptoHunterProtectedStateAuthority = expected production policy (AUTO)
-> CryptoHunterSecretResourceAuthority = expected production policy (AUTO)

missing / invalid / conflicting receipt
-> fail closed; nigdy infer ENROLLED; nie nadpisuj authority state
```

Receipt verification obejmuje pinned trust domain/signer, schema/version, machine/subject binding,
terminal enrollment operation, exact service-policy profile oraz non-rollback current designation.
Sama obecność katalogu, DB, usługi, binary, TPM object, package albo preserved state **nie jest**
dowodem enrollment. Repair nie resetuje poprawnie enrolled systemu do `DEMAND`, nie mintuje receipt,
nie odtwarza trust z filesystem heuristics i nie uruchamia backendu przy niejednoznacznym stanie.
Zmiana start policy jest idempotentną projekcją authenticated receipt; nie jest nową authority
decision.

### Uninstall i rollback ownership

* Normal uninstall usuwa wszystkie sześć MSI-owned SCM definitions oraz installed binaries, ale
  zgodnie z Stage-9 policy zachowuje machine/user authority state, PostgreSQL data, LPPI history,
  TPM NV indices/keys, protected map, vault ciphertext i PDSA packages/receipts. Ich wipe jest
  osobnym, authenticated destructive recovery/decommission protocol, nie skutkiem uninstall.
* Enrollment nigdy nie tworzy ani nie usuwa SCM service objects, więc nie może pozostawić
  untracked service.
* Failed enrollment zachowuje pre-existing authenticated PDSA challenge/package/evidence i każdy
  committed authority record; usuwa wyłącznie bezpiecznie abortowalne ephemeral staging zgodnie ze
  state machine. Nie cofa TPM counter.
* Failed MSI rollback usuwa wyłącznie pliki, ACL/directories i SCM definitions utworzone przez tę
  MSI transaction. Nie usuwa pre-existing ProgramData, PDSA evidence, authority backup, TPM state
  ani zasobów oznaczonych jako existing-before-transaction.
* Reinstall wykrywa preserved state i wymaga exact authenticated recovery; nie adoptuje samej
  obecności service, directory, DB lub TPM object jako authority.

## 17. Stage-9 boundary: MSI versus first run

Zamrażany w proposal revision 3 wybór:

```text
account/device provisioning = POST-INSTALL FIRST-RUN ENROLLMENT
LPPI service before enrollment = DEMAND_START
backend service before enrollment = DEMAND_START / STOPPED
CoreHost before canonical account/device scope = MUST NOT START
```

MSI transaction instaluje kod, directories, ACL i service definitions, ale kończy się bez
account/device provisioning. Post-install provisioner uruchamia wyłącznie LPPI, przeprowadza
signed offline ceremony i model C, provisionuje osobne protected/vault services, materializuje
canonical StateStore scope, a dopiero po terminalnym sukcesie zezwala na start backend/CoreHost.
Nie wybieramy `RUNNING_IN_UNPROVISIONED_SUPERVISOR_MODE`, ponieważ obecny CoreHost wymaga
canonical account/device scope, a supervisor stworzyłby nowy runtime trust boundary.

Canonical clean-install w przyszłości musi otrzymać production-equivalent **TEST-isolated**
signed enrollment package, vTPM/qualified TPM fixture i osobne TEST roots/roles/stores. Nie może
używać production keys ani testowego `Boundary`/`SecretPort`. Do implementacji tych prerekwizytów
live job pozostaje nieuruchamiany.

## 18. Canonical contracts wymagające freeze

W kolejności zależności należy przygotować canonical JSON, deterministyczny MD i validators:

1. `product_release_policy_trust_contract` — 2-of-3 root set, signer, Authenticode separation,
   monotonic policy/revocation/rotation i TEST/PRODUCTION isolation;
2. `product_deployment_security_and_enrollment_authority_contract` — PDSA, `psub`, signed one-time
   challenge lifecycle, TPM-bound request/package, retained issuance/recovery registry i keys;
3. `external_product_provisioning_authority_contract` — LPPI topology, operation/request schema,
   candidate `dev/op`, state machine, signed membership i current resolver;
4. amendment `m05_cryptohunter_account_root_of_trust_reconciliation` — model C i closed external
   subject identity;
5. amendment `m05_cryptohunter_account_genesis_authority_model` — CHA account-ID mint owner,
   atomic first-device commitment i first-device slot;
6. amendment operation identity/request binding — bijekcja `prvop` do istniejącej CHA operation
   identity, exact retry/conflict i authenticated subject cardinality;
7. amendment root-proof admission/issuer — zachowanie CHA-only requester oraz LPPI evidence input;
8. `product_provisioning_tpm_rollback_anchor_contract` — native NV counter layout/policy,
   signed-head protocol, algorithms, recovery/replacement, endurance, backup i qualification;
9. M0.3 amendment — signed membership/current resolver, candidate-to-genuine device/operator
   binding oraz `ADD_DEVICE`, bez rozszerzania `FirstRunBootstrapAuthority`;
10. M0.11 amendment — external references/recovery projections bez reverse minting;
11. `product_protected_state_authority_substrate_contract` — osobny service/key/schema/NV domain;
12. `product_secret_resource_authority_substrate_contract` — vault envelope/CNG/TPM lifecycle;
13. `windows_stage9_first_run_enrollment_contract` — post-install ordering, service states, pipe
    authentication, readiness gates i production-equivalent TEST harness;
14. architecture baseline freeze manifest dopiero po pozytywnych cross-contract/adversarial tests.

Do punktu 14 current `DESIGN_BLOCKED`, `NOT_AVAILABLE` oraz `implementation_allowed=false`
pozostają wiążące i produkcyjny provider nie może powstać.

## 19. Mandatory adversarial cases

Freeze validators muszą odrzucić co najmniej:

* caller-chosen `psub`, `prvop`, `acct`, first `dev` lub `op`;
* enrollment package podpisany nieznanym/caller key, podmieniony pinned root, TOFU lub TEST package
  w PRODUCTION;
* ten sam enrollment reference z innym subject/request/package lineage;
* ten sam operation ID z innym canonical request;
* drugi account albo first device dla tego samego subject/slot, niezależnie od arrival time;
* LPPI bezpośrednio wołające RootProofIssuer lub używające CHA requester credential;
* publiczny SHA-256 traktowany jako authentication;
* crash/lost response prowadzący do nowego operation/account/device/operator ID;
* timeout CHA/root proof traktowany jako terminal abort;
* publikację membership przed authenticated CHA committed genesis;
* M0.10 trust/operator state ustanowione przez LPPI;
* DB+ProgramData rollback przy wyższym TPM NV head;
* DB ahead TPM bez exact pending transition;
* TPM clear/motherboard replacement jako automatyczne `UNSEEN`;
* StateStore backup mintujący LPPI/protected membership, current map albo vault authority;
* alias key/role/schema/head/generation między provisioning, AccountGenesis freshness, protected
  state i secret vault;
* MSI custom action mintujący trust lub uruchamiający CoreHost bez canonical scope;
* interactive user odczytujący vault plaintext/KEK;
* cleanup/rotation retry usuwający nowy sekret lub resurrecting stary reference;
* reinstall mintujący nowe IDs bez PDSA recovery evidence;
* `ADD_DEVICE` otwierające account genesis lub first-device slot;
* software KSP key lub niezweryfikowaną TPM deklarację zaakceptowaną jako attested hardware key;
* próbę użycia Ed25519 jako domniemanego Microsoft Platform Crypto Provider key;
* drugi podpisany DB head dla tej samej NV-counter generation;
* drugi `TPM2_NV_Increment` po lost response;
* ordinary NV digest slot traktowany jako atomowy razem z counterem;
* caller-minted, replayed, expired albo wcześniej consumed PDSA challenge;
* lost enrollment-package response powodujący nowe package ID lub ponowne consume challenge;
* nieznanego release-policy signera, caller-selected root, downgrade, revoked/stale policy lub
  TEST root/policy w PRODUCTION;
* traktowanie poprawnego Authenticode publishera jako PDSA/release-policy authority;
* first-run tworzący/usuwający SCM definitions albo uninstall usuwający preserved authority state;
* twierdzenie o odporności na malicious local admin bez osobno zamrożonego WDAC/measured-boot
  enforcement.

## 20. Minimalna kolejność implementacji po akceptacji

1. Zamrozić kontrakty 1–7 i ich vectors, bez Stage-9 code.
2. Zbudować TEST-isolated PDSA package tooling i TPM emulator qualification.
3. Zaimplementować/zakwalifikować LPPI registry, operation state machine i NV anchor.
4. Zintegrować LPPI evidence z istniejącym CHA-only RootProofIssuer chain.
5. Zamrozić i wdrożyć osobne Protected State Authority oraz Secret Resource Authority.
6. Zamrozić Windows post-install ordering i dopiero wtedy zmienić Stage-9 installer/provisioner.
7. Zaimplementować cienki `WindowsExternalProvisioningHandoff` delegujący do gotowych ownerów.
8. Wykonać fault-injection/reinstall/TPM replacement tests.
9. Dopiero po spełnieniu gate zdecydować o canonical live Windows clean-install.

Żaden krok nie rozpoczyna Stage 10 ani nie pozwala zadeklarować produkcyjnej gotowości.

### 20.1. Zachowane live evidence z fizycznego TPM

Najnowszy fizyczny Windows 11 / TPM 2.0 run superseduje wcześniejszy częściowy run, ale go nie
usuwa z historii zmian. Pełne formalne outputs wynoszą:

```text
CAN_CREATE_REQUIRED_NV_INDEX = PASS
CAN_READ_REQUIRED_NV_INDEX = PASS
CAN_INCREMENT_REQUIRED_NV_INDEX = PASS
CAN_OPEN_POLICY_SESSION = PASS
CAN_SATISFY_SELECTED_POLICY = PASS
CAN_SURVIVE_SERVICE_RESTART = PASS
CAN_DETECT_DISK_STATE_BEHIND_COUNTER = PASS
```

Rzeczywiste `TPM2_LoadExternal`, `TPM2_VerifySignature` i `TPM2_PolicyAuthorize` zwróciły
`TPM_RC_SUCCESS`. Finalny `PolicyGetDigest` był identyczny z niezależnie obliczonym expected
`PolicyAuthorize` digest. Owner-backed ticket miał `tag=TPM_ST_VERIFIED`,
`hierarchy=TPM_RH_OWNER` i niepusty digest. Cleanup external key, policy session, NV, disposable
service, scratch i TBS context zakończył się `PASS`. Wynik brzmi:

```text
PolicyAuthorize substrate feasibility = LIVE PROVEN
PolicyAuthorize candidate = LIVE PASS
```

Ten dowód pozostaje acceptance-only. Disposable key/material nie jest i nie może zostać
przeniesiony do production Product Release Root, PDSA, PSA ani Recovery Policy Key. Późniejszy
canonical cross-check potwierdził oba rzeczywiste branche, `PolicyOR`, cykl NV Name, `cpHash`,
fixture generation `7` oraz observed/runtime generation `13` i final generation `14`. Generator,
niezależny verifier i fizyczny TPM były zgodne; cleanup zakończył się PASS.

## 21. Formalny status po pełnym fizycznym Windows 11 / TPM 2.0 live run

```text
Canonical vector ↔ TPM cross-check
[██████████] LIVE PASS

ROOT-OF-TRUST CLOSURE
[█████████░] POLICY TOPOLOGY + CANONICAL TPM SEMANTICS LIVE PROVEN / PRODUCTION FREEZE REMAINS

STAGE 9
[█████████░] IN PROGRESS

CAŁY BLOK WINDOWS 0–14
[██████░░░░] 60.0% — 9/15 DONE

WINDOWS_PRODUCTION_READY = NOT_READY
STAGE10 = NOT_STARTED
PRODUCTION_ROOT_MATERIAL = NOT_PROVISIONED
```

Stage 0–8 pozostają frozen i niezmienione. Stage 9 nie jest `DONE`; Stage 10 nie został rozpoczęty.
MSI i `WindowsExternalProvisioningHandoff` pozostają bez zmian. Fizyczny canonical cross-check ma
pełny PASS, ale Stage 9 pozostaje `IN PROGRESS`. Publiczny freeze layer jest implementation-ready;
następny gate wymaga rzeczywistych publicznych trust anchors i podpisów z ceremony. **PRODUCTION
MATERIAL REQUIRED — DO NOT GENERATE SUBSTITUTE KEYS.**
