# Stage 9 — S9-RPI-05: independent custody completion qualification

## Wynik i zakres

S9-RPI-05 przygotowuje zamknięty format i odczytową kwalifikację evidence
zakończenia operacji custody. Nie instaluje Custody Completion Authority
(CCA), jej klucza, źródła uwierzytelnionych zdarzeń wykonania ani trwałego
decision log. Nie implementuje runtime podpisującego completion. Taka
implementacja bez zaufanego źródła podpisałaby jedynie oświadczenie o plikach,
a nie wykonanie operacji custody.

Zakres wykonawczy jest ograniczony do walidacji formatu, powiązania exact
operation/journal/metadata oraz raportowania brakujących gwarancji.
Projekt przyszłej topologii i podpisanej evidence poniżej wymaga review,
provisioningu, migracji i kwalifikacji fizycznej. Testowy raport, poprawny
podpis syntetyczny i lokalne członkostwo obiektu nie są production capability.

```text
independent_custody_completion_authority = NOT_QUALIFIED
authenticated_custody_mutation_source = NOT_QUALIFIED
independent_custody_durability_confirmation = NOT_QUALIFIED
production_custody_completion_publication = BLOCKED
CONTRACT_RECONCILIATION_PENDING_APPROVAL
semantic_root_proof_issuer_runtime = NOT_IMPLEMENTED / BLOCKED
production_provisioning_ready = false
ROOT_PROOF_ISSUANCE_AVAILABLE = false
ATOMIC_ISSUER_READY = NOT_READY
PRODUCTION_LIFECYCLE_ISOLATION = BLOCKED
public_root_proof_issuance = NO-GO
```

To opis granicy tej zmiany, bez aktualizacji canonical statusów.
Po pierwotnym audycie scalono #3156–#3158. Zapis
[decyzji właściciela A+C](decisions/stage9_root_proof_transaction_authority_owner_approval_v1.md)
potwierdza kierunek architektoniczny, lecz pozostawia
`CONTRACT_ADDENDUM_INTEGRATION_PENDING` i `PRODUCTION_NO_GO`.
Zachowujemy wymagany w tym zadaniu label
`CONTRACT_RECONCILIATION_PENDING_APPROVAL`; nie traktujemy kierunkowej zgody
jako formalnej integracji addendum, zgody na CCA/migrację lub capability.
`ProductionRootLifecycleBoundary.commit()` i
`publish_administrator_completion()` nadal bezwarunkowo odmawiają. SQL V2
`complete_close(text,text,bytea)` pozostaje odmawiającym tombstone, bez grantu
`EXECUTE` dla issuera, administratora lub PUBLIC. Nie powstaje alternatywny
port publication, produkcyjny SIGN, BIND, enrollment, aktywacja ani Stage 10.

## Audyt istniejących mechanizmów

Podstawą jest audyt `main` po #3155 i rzeczywiste źródła poniżej. Przed PR
gałąź oparto na aktualnym `c39cedf2` po #3158; istniejące osiem modułów
runtime nie zmieniło się między tymi bazami.

- [`production_signer_custody.py`](../../bot_core/production_signer_custody.py)
  i [`production_signer_ipc.py`](../../bot_core/production_signer_ipc.py);
- [`local_signing_lifecycle_evidence.py`](../../bot_core/local_signing_lifecycle_evidence.py)
  i [`local_signing_custody.py`](../../bot_core/local_signing_custody.py);
- [`production_root_lifecycle.py`](../../bot_core/production_root_lifecycle.py),
  [`postgresql_production_root_lifecycle_schema.py`](../../bot_core/postgresql_production_root_lifecycle_schema.py)
  i [`postgresql_production_root_lifecycle_qualification.py`](../../bot_core/postgresql_production_root_lifecycle_qualification.py);
- [`postgresql_custody_lifecycle.py`](../../bot_core/postgresql_custody_lifecycle.py)
  oraz właściwe testy security, plików, procesów i PostgreSQL.

Codebase Memory MCP nie był callable. Wnioski oparto na celowanych
wyszukiwaniach i bezpośrednich odczytach; nie deklarują kompletności
nieobecnego indeksu ani kwalifikacji instalacji.

| Mechanizm | Zapewniana właściwość | Czego nie dowodzi |
| --- | --- | --- |
| S9-RPI-01 production lifecycle | Immutable close, exact predecessor, slot/fence locks, osobny COMMIT `PENDING_CLOSED`, `close_xid`, direct-login/OID i peer authentication | Wykonania mutation custody ani niezależnego completion |
| S9-RPI-02 signer services | Oddzielne konta, storage/config/socket, live Linux peer identity, zamknięte `DIAGNOSTIC`/`QUALIFY` | Independent private control, wykonania operacji ani trwałości po restarcie |
| S9-RPI-03 local journal observation | Exact canonical journal `COMPLETE`, descriptor, before/after i aktualne metadata pod istniejącą blokadą; odrzucenie podmiany ścieżek i generation | Autorstwa journalu, wcześniejszego `fsync`, niezależnego źródła execution ani publication |
| Production catalog qualification | Exact sources/ACL/owners, `SECURITY DEFINER`, stały `search_path`, direct roles, catalog/physical pins i zamknięte alternatywne ścieżki | Kryptograficznej ochrony przed bootstrap superuserem lub przejęciem hosta |
| Isolated custody coordinator V2 | Testowy DB-first close, journal, recovery, MAC i completion pod blokadą | Niezależności od administratora posiadającego testowy MAC ani production authorization |
| S9-RPI-04 reconciliation | Jawny konflikt zakresu transakcji, propozycja C oraz późniejsza kierunkowa decyzja A+C | Formalnej versioned contract integration, kwalifikacji produkcyjnego coordinatora lub zniesienia NO-GO |

Dokumenty bazowe to [S9-RPI-01](stage9_production_root_proof_lifecycle_isolation.md),
[S9-RPI-02](stage9_independent_production_signers.md),
[S9-RPI-03](stage9_production_custody_completion_verifier.md) i
[S9-RPI-04](stage9_root_proof_transaction_authority_reconciliation.md).
Historyczne completion V1 pozostają `LEGACY_UNVERIFIED_COMPLETION`.
Ta zmiana nie awansuje ich, nie przepisuje receipt bytes i nie naprawia
historycznej decyzji na podstawie nowego raportu.

Sprawdzone frozen źródła pozostają bez zmian:

- [Independent issuer](cryptohunter_product_architecture/m05_account_genesis_independent_root_proof_issuer_contract.json):
  exact BIND/recovery, oryginalne bytes i zakaz ponownego podpisywania po
  utracie odpowiedzi;
- [Production substrate](cryptohunter_product_architecture/m05_account_genesis_root_proof_issuer_production_substrate_selection_contract.json):
  osobne authority, credentials, namespaces i lifecycle, jawne migracje,
  `NOT_FOUND=OUTCOME_UNKNOWN` oraz ograniczenia profilu lokalnego;
- [Implementation readiness](cryptohunter_product_architecture/m05_account_genesis_root_proof_issuer_implementation_readiness_contract.json)
  i [physical persistence](cryptohunter_product_architecture/m05_account_genesis_physical_persistence_crash_atomicity_contract.json):
  implementacja fragmentu issuera nie spełnia readiness ani niezależnych
  bramek CHA/FreshnessAuthority/AccountGenesis.

`PRODUCTION_LOCAL` dopuszcza software Ed25519 i nie wymaga HSM/TPM.
Nie deklaruje ochrony przed pełnym uprzywilejowanym przejęciem hosta,
ekstrakcją kluczy przez administratora ani skoordynowanym rollbackiem
PostgreSQL, lokalnych kluczy i checkpointu. Nie wolno nazwać tej topologii
`PRODUCTION_SERVER_READY` na podstawie zmiany etykiety.

## Wybrana topologia do osobnego review

Technicznie wykonalnym wariantem lokalnym jest dedykowana CCA z własnym
Linux UID/GID, chronioną konfiguracją, software key, namespace, lifecycle
i trwałym decision log. Jej credential służy wyłącznie do completion domain;
różni się od kluczy Root Proof, historii, trust activation, checkpointu,
requestera, claimanta i poświadczeń PostgreSQL. CCA nie eksportuje ani nie
odczytuje signer private keys. Administrator lifecycle i issuer writer nie
mają dostępu do klucza CCA, jej logu zapisu ani jej konfiguracji.

Każda z dwóch ról custody potrzebuje osobno kwalifikowanego mutation
executor/source: procesu lub sidecaru z przypiętym role-specific endpointem,
OS identity i własnym execution-authentication credential. Credential źródła
nie jest kluczem Root Proof lub historii. Root executor nie poświadcza
history operation, a history executor nie poświadcza root operation.
Executor posiada wyłącznie uprawnienia konieczne do swojego lifecycle
namespace. Prawo modyfikacji metadata nie może przy okazji otwierać dostępu
do sekretów innej roli lub do klucza CCA.

Zaufanym źródłem jest wykonawca posiadający cały przebieg kwalifikowanej
operacji, blokadę i zapis trwałych faz, a nie ogólny endpoint podpisujący
caller-supplied hash. Executor uwierzytelnia zdarzenie dopiero po exact
DB-first admission i zakończeniu protokołu opisanego poniżej. CCA weryfikuje
ten event, jego oryginalne bytes, własny odczyt źródła i committed close.
Podpis CCA jest ostatnim uwierzytelnieniem wyniku tego protokołu; nie tworzy
wiarygodności wejścia z samego faktu podpisania DTO.

Aktualna instalacja nie zawiera takiego executora/source ani CCA.
Manifest production lifecycle akceptuje dokładnie siedem wcześniejszych
principals. Nie dodajemy do niego pozornego ósmego konta, grantu ani
automatycznego re-pin. Istniejący reader wymaga bieżącego właściciela plików
custody; jego uruchomienie jako signer nie jest dowodem odrębnego CCA
principal. Przyszła CCA otrzymuje publiczne evidence przez osobny,
uwierzytelniony, przypięty kanał źródła; nie otrzymuje prawa czytania
prywatnego keyringu signera.

## Threat model — dziewięć granic

| Principal lub trust boundary | Dostęp i dozwolone działania | Rzeczywista gwarancja i ograniczenie |
| --- | --- | --- |
| 1. Administrator PostgreSQL lifecycle | Oddzielny minimalny direct-login z instalacyjnym OID i `peer:<unix_user>`; `begin_close`, exact resolution, odczyt historyczny i przygotowanie zamkniętego successora | Bez ownership, DML/DDL, memberships, signer secrets i CCA key. Może trwale zamknąć fence, lecz nie może samodzielnie stworzyć ważnego completion. CCA nie uznaje jego hasha lub forwarded DTO za execution evidence. |
| 2. Właściciel schematu PostgreSQL | `NOLOGIN`, właściciel tabel i reviewed `SECURITY DEFINER` helpers; stały `search_path` i minimalne ACL | Runtime nie może objąć jego roli. Przejęcie owner privileges lub funkcji może sfałszować SQL facts; zewnętrzny pin wykrywa drift przy nieprzejętym verifierze. Sam SQL nie stanowi niezależnej kryptograficznej evidence mutation. |
| 3. Issuer runtime / writer credential | Minimalny direct-login, odczyt/fence i diagnostyka signerów; zainstalowany proces wybiera fixed konfigurację | Bez admin, custody mutation, CCA signing i publication. `commit()` pozostaje zablokowany. Poprawny raport nie jest capability BIND. |
| 4. Root Proof signer | Własne UID/GID, konfiguracja, native-keyring namespace, custody i socket; obecnie wyłącznie `DIAGNOSTIC`/`QUALIFY` | Nie może podszywać się pod history role. Przyszły root execution event wymaga osobnego root source credential i kwalifikacji; istniejące public facts lub seed/public match tego nie zastępują. |
| 5. History Attestation signer | Analogicznie odrębne konto, storage, keyring i endpoint historii | Nie może poświadczać root role ani używać Root Proof key. Obserwacja peer identity potwierdza proces, nie wykonanie mutation lub private control. |
| 6. Custody Completion Authority | Projektowane osobne konto, completion key, root-owned pin/config, własny log; odczyt publicznej evidence i qualified committed close | Nie przyjmuje dowolnych hashy do podpisu, nie wykonuje root/history SIGN i nie czyta ich keys. Chroni przed lifecycle admin/issuer i cross-role substitution pod warunkiem nieprzejętego CCA/source/provisioningu. Dziś `NOT_QUALIFIED`. |
| 7. Bootstrap/provisioning administrator | Jawny, odrębny direct bootstrap superuser, review katalogu i pinów, instalacja kont/kluczy oraz kontrolowana migracja | Zaufany do provisioningu. Może zmienić role, funkcje i dane PG; nie wolno deklarować kryptograficznej niepodrabialności SQL wobec niego. Nie jest runtime CCA ani codziennym adminem lifecycle. |
| 8. Administrator hosta / root | Kontrola procesów, pamięci, filesystem, kont, usług, software keys i PostgreSQL | Pełne przejęcie znajduje się poza gwarancją lokalnej topologii. Separacja UID/GID ogranicza zwykłe principals; nie zatrzymuje root ani skoordynowanego rollbacku całego hosta. |
| 9. Zewnętrzny chroniony installation/trust pin | Niezależnie dostarczone oczekiwania installation/environment/trust domain, identities/OID/UID/GID, katalogu, credential/material/version i źródeł | Nie pochodzi z requestu ani z badanej bazy. Brak, drift lub nieautoryzowany nowy pin powoduje odmowę. Pin jest źródłem oczekiwań, nie dowodem wykonania lub wcześniejszego `fsync`; jego przejęcie unieważnia tę granicę. |

Nazwy usług nie dowodzą rozdzielenia. Fizyczna kwalifikacja musi porównać
rzeczywiste UID/GID, supplementary groups, memberships, role OID, effective
ACL, namespaces, dostępy do keyringu i wszystkie alternatywne endpointy.
Requester, claimant i CHA nie dostają CCA/source/custody/SQL publication port.
Kompromitacja zaufanego executora może spowodować fałszywe execution event;
w tym wariancie autentyczność event oznacza odpowiedzialność przypiętego,
zakwalifikowanego źródła, nie odporność na jego pełne przejęcie.

## Pięć odrębnych właściwości

1. **Obserwacja**: reader widzi dokładny istniejący `COMPLETE` i metadata pod
   blokadą, bez mutation/recovery/`fsync`.
2. **Source/execution authentication**: przypięty executor uwierzytelnia fakt
   wykonania dokładnego transition w swoim lifecycle namespace; publiczne
   pliki i caller assertion nie wystarczają.
3. **Trwałość**: kwalifikowany executor kończy wymagane synchronizacje,
   zachowuje event i potrafi po restarcie odczytać te same bytes; CCA osobno
   utrwala własną decyzję. Sam podpis lub odczyt nie poświadcza wcześniejszej
   trwałości, zachowania hardware ani odporności na host rollback.
4. **Publication authorization**: przyszły odrębny SQL boundary sam sprawdza
   uwierzytelnioną evidence, exact close i installation trust; Python DTO
   nie może otwierać obejścia przez bezpośrednie SQL.
5. **Późniejsze otwarcie fence**: osobny zatwierdzony protokół aktywacji lub
   sukcesji. Completion retirement nie oznacza powrotu starego klucza do
   `ACTIVE` i nie uruchamia signing.

Ta zmiana obejmuje pierwszy wynik oraz rygorystyczne wykazanie brakujących
przesłanek pozostałych. Każdy brak ma własny status; pozytywne sprawdzenie
formatu nie awansuje następnej właściwości.

## Exact evidence — stan wykonawczy i projekt przyszły

### Wykonywalny claim V1 — wyłącznie kwalifikacja

Odczytowy format kwalifikacyjny definiuje
[`independent_custody_completion_qualification.py`](../../bot_core/independent_custody_completion_qualification.py).
Nie jest to podpisane completion ani alternatywa dla produkcyjnego receipt.
Każde identity, installation, pin, key, DB coordinates i custody path
dostarczone w claim pozostają **niezaufanym oświadczeniem callera**, również
gdy wszystkie wartości są wzajemnie spójne i pliki istnieją. Verifier nie
uczy się z nich oczekiwanego trust anchor, nie tworzy CCA credentials,
nie weryfikuje live production close i nie posiada funkcji podpisującej.

```text
schema = cryptohunter.independent-custody-completion-qualification-claim
schema_version = 1
domain = CryptoHunter/IndependentCustodyCompletion/Qualification/V1
profile = PRODUCTION_LOCAL
```

Top-level ma dokładnie następujące pola; dopuszczalnych rozszerzeń nie ma:

```text
schema, schema_version, domain, profile,
installation_id, environment_id, trust_domain,
installation_sha256, external_trust_pin_sha256,
completion_authority_claim, database_identity_claim,
production_close_request_b64, production_close_request_sha256,
operation_descriptor_b64, operation_descriptor_sha256,
complete_journal_b64, complete_journal_sha256
```

`completion_authority_claim` ma dokładnie:

```text
authority_id, host_id, os_uid, os_gid, credential_identity, semantic_role,
key_version, lifecycle_generation, lifecycle_state,
public_key_b64, public_material_identity
```

Jego `semantic_role` to `CUSTODY_COMPLETION_ATTESTATION_V1`.
Public key ma 32 bytes i zgodną material identity; nie może być tym samym
materiałem co credential signera w journalu. Walidacja obsługuje wyłącznie
zadeklarowany prosty lifecycle: `ACTIVE` z generation 1, `VERIFY_ONLY` z 2
oraz `REVOKED` z 2 lub 3. To kontrola kształtu claims, bez potwierdzenia
tożsamości, posiadania klucza, eligibility lub trwałego lifecycle CCA.

`database_identity_claim` ma dokładnie:

```text
system_identifier, database_oid, database_name, cluster_name,
connection_host, connection_port, server_address, server_port, instance_id
```

`system_identifier` jest dodatnim canonical decimal string w zakresie
PostgreSQL uint64. `database_oid` mieści się w uint32, porty w 1–65535;
server address/port mogą być `null`, cluster name może być pusty.
`installation_id` musi równać się claimed database `instance_id`.
Te współrzędne są porównywane z descriptor bytes; nie są live odczytem DB.

Cały claim ma najwyżej 2 MiB. Embedded PostgreSQL request ma limit 64 KiB,
descriptor 256 KiB, journal 1 MiB. Kodowanie jest ścisłym canonical Base64;
każdy digest to 64 małe hex znaki i musi odpowiadać zachowanym bytes.
Outer claim, descriptor i journal muszą używać trusted canonical JSON;
PostgreSQL request zachowuje odrębne oryginalne UTF8 JSON bytes.
Decoder odrzuca duplicate fields, float, NaN/infinity, bool zamiast int,
niewłaściwą wersję/domenę/profil, brakujące lub dodatkowe pola i niezgodne
binding. Każdy JSON ma jawny limit zagnieżdżenia 16, sprawdzany przed
dekodowaniem. Claim nie ma pola podpisu ani pola `approved`/`production_ready`.

`parse_completion_qualification_claim()` tworzy
`CompletionQualificationClaim` zachowujący original `canonical_bytes`.
`qualify_completion_claim()` ponownie sprawdza format i exact binding.
`observe_completion_qualification()` dodatkowo porównuje claim journal
z realnym istniejącym V2 `COMPLETE` i current metadata pod shared custody
lock, z path/identity rechecks również przy wyjściu z context managera.
Nie tworzy katalogów, lock files, journalu, sekretów ani capabilities;
nie wykonuje `fsync`, recovery, reconnect lub mutation. Wybrana przez callera
ścieżka służy tylko obserwacji jego wskazanych plików.

Oba rodzaje oceny zwracają `CompletionQualificationObservation` z niezmiennymi
`status=NOT_QUALIFIED`, `production_emission=BLOCKED` i pełną listą blockers:

```text
COMPLETION_AUTHORITY_IDENTITY_NOT_INDEPENDENTLY_QUALIFIED
COMPLETION_CREDENTIAL_LIFECYCLE_NOT_QUALIFIED
EXTERNAL_TRUST_PIN_NOT_VERIFIED
LIVE_PRODUCTION_CLOSE_NOT_VERIFIED
CUSTODY_MUTATION_SOURCE_NOT_AUTHENTICATED
CUSTODY_EXECUTION_NOT_ATTESTED
PRIOR_DURABILITY_NOT_ATTESTED
COMPLETION_PUBLICATION_NOT_AUTHORIZED
CONTRACT_RECONCILIATION_PENDING_APPROVAL
```

Nieprawidłowe lub zmienione dane powodują `CompletionQualificationError`
albo odmowę istniejącego local custody readera. Poprawny claim i prawdziwy
`COMPLETE` nie usuwają żadnej pozycji tej listy.

### Przyszły podpisany protokół V1 — propozycja

Poniższa **przyszła podpisana evidence V1** jest projektem do review,
bez runtime i bez deklaracji zgodności produkcyjnej. Dwie domeny muszą być
rozłączne:

```text
source event: CryptoHunter/CustodyMutationExecution/V1\0
CCA evidence: CryptoHunter/IndependentCustodyCompletion/V1\0
```

Każdy podpis to Ed25519 nad właściwą stałą domeną i oryginalnymi canonical
payload bytes. Envelope ma dokładnie `schema`, `version`, `payload_b64`,
`credential_id`, `key_version`, `signature_b64`; żadne pole envelope nie
zmienia interpretacji podpisanych bytes. Wartości credential/version muszą
zgadzać się z payloadem i z niezależnym pinem. Kodowanie Base64 jest
kanoniczne RFC 4648 z paddingiem; verifier dekoduje rygorystycznie i porównuje
ponowne kodowanie. Klucz publiczny dostarczony przez callera nie jest
trust anchor. Nie ma fallbacku do MAC administratora ani innej domeny podpisu.

Payload CCA zawiera poniższe grupy, skonkretyzowane zamkniętymi listami
dalej w tej sekcji. Unknown field lub wersja oznaczają odmowę, zamiast
ignorowania rozszerzeń. Zatwierdzenie tego proponowanego schematu jest
osobnym prerekwizytem produkcji.

| Grupa | Obowiązkowa treść i źródło |
| --- | --- |
| Protocol | Schema identity, version `1`, cryptographic domain, profile `PRODUCTION_LOCAL`, rodzaj wyniku; stałe protokołu, nie caller |
| Installation | Installation ID, environment, trust domain, installation manifest digest i external trust-pin identity/version/digest; niezależnie chroniona instalacja |
| Physical authority | CCA service/authority ID, przypięte OS identity i service coordinates, executor/source authority ID i role-scoped namespace; dane instalacji porównane z live procesem, nie etykieta z DTO |
| Database | Pełne `system_identifier` jako canonical decimal string, database OID/name, cluster/service instance ID, fixed connection host/port, server coordinates oraz production authority/schema/catalog identity/version/digest; własny qualified odczyt CCA i external pin |
| Completion issuer | CCA credential identity, completion-only namespace, public material identity, key version, lifecycle generation i trust epoch; własny credential CCA porównany z pinem |
| Custody identity | Semantic role, authority identity, credential identity, key handle/version, custody lifecycle namespace, public key bytes/material identity, key version i exact predecessor generation; verified original descriptor/journal/installed source |
| Production close | Exact operation ID, przechowane authoritative PostgreSQL request bytes, ich format identity i `request_sha256`, committed `close_xid`, exact predecessor/state/generation i `PENDING_CLOSED`; własny synchronizowany odczyt odrębnej qualified SQL authority |
| Local operation | Oryginalne canonical descriptor bytes w Base64 oraz SHA256; descriptor ma własny format V2, pełne custody scope i DB identity |
| Journal | Oryginalne canonical bytes journalu `COMPLETE` w Base64 oraz SHA256, before/after metadata bytes i digests, stan i generation przed/po; qualified źródło oraz niezależny exact re-read |
| Source provenance | Oryginalny source event envelope i payload bytes, source credential/material/version/lifecycle, source sequence i previous event digest, exact execution ID; uwierzytelniony executor, nie administrator |
| Durable decision | CCA decision ID, monotonic sequence w jej namespace, previous decision digest, oryginalny operation binding i source event digest; trwały CCA decision log |
| Result | Exact terminal result oraz odrębne wyniki source, execution i durability qualification; publication i fence activation pozostają odrębnymi etapami |

Proponowane dokładne payload schema identities to
`cryptohunter.custody-mutation-execution` dla source oraz
`cryptohunter.independent-custody-completion` dla CCA; dla obu `version=1`.
Source envelope schema to `cryptohunter.signed-custody-mutation-execution`,
a CCA envelope schema to `cryptohunter.signed-independent-custody-completion`.
Powyższe nazwy i listy są projektem, nie istniejącym production formatem.
Nie zastępują executable qualification claim schema.

Zamknięte rekordy współdzielone w proponowanych payloadach:

| Rekord | Dokładna lista pól |
| --- | --- |
| `installation` | `installation_id`, `environment_id`, `profile`, `trust_domain`, `installation_sha256`, `trust_pin_id`, `trust_pin_version`, `trust_pin_sha256` |
| `authority` | `authority_id`, `service_id`, `host_id`, `os_uid`, `os_gid`, `endpoint`, `provider_namespace`, `credential_identity`, `semantic_role`, `public_key_b64`, `public_material_identity`, `key_version`, `lifecycle_generation`, `lifecycle_state`, `trust_epoch` |
| `database` | `system_identifier`, `database_oid`, `database_name`, `cluster_name`, `connection_host`, `connection_port`, `server_address`, `server_port`, `instance_id`, `authority_identity`, `schema_name`, `schema_version`, `catalog_sha256` |
| `custody` | `semantic_role`, `authority_identity`, `credential_identity`, `key_handle_or_version`, `custody_lifecycle_namespace`, `public_key_b64`, `public_material_identity`, `key_version`, `predecessor_generation`, `successor_generation`, `predecessor_state`, `target_state` |
| `close` | `operation_id`, `request_format`, `request_b64`, `request_sha256`, `close_xid`, `phase` |
| `close_observation` | `reader_role`, `reader_role_oid`, `observation_xid`, `qualified_catalog_sha256` |
| `originals` | `operation_descriptor_b64`, `operation_descriptor_sha256`, `complete_journal_b64`, `complete_journal_sha256`, `before_metadata_b64`, `before_metadata_sha256`, `after_metadata_b64`, `after_metadata_sha256` |
| `log_position` | `namespace`, `sequence`, `previous_entry_sha256`, `entry_id` |

`request_format` ma stałą wartość `PostgreSQL16/jsonb-text/UTF8`, a `phase`
to `PENDING_CLOSED`. `close.close_xid` i
`close_observation.observation_xid` są dodatnimi canonical decimal strings;
close musi być committed w innej transakcji niż observation. Wartości
`before_metadata_b64`/`after_metadata_b64` kodują canonical bytes właściwych
oryginalnych obiektów osadzonych w journalu, nie niezależną rekonstrukcję
podobnych public facts. Pierwszy `previous_entry_sha256` ma wartość 64 zer;
dalsze wskazują digest oryginalnego poprzedniego retained log entry.
Sequence ma monotoniczną unikalność w przypiętym namespace; bez zaufanego
floor nie zapewnia odporności na rollback.

Source payload ma dokładnie:

```text
schema, version, domain, installation, executor, database, custody,
close, close_observation, originals, execution_id, source_log_position,
execution_result, durability_result
```

`executor` jest rekordem `authority` przypiętym do właściwej roli źródła.
`source_log_position` jest `log_position`; `entry_id` równa się zachowanemu
`execution_id`. `execution_result` ma jedyną pozytywną wartość
`EXACT_LIFECYCLE_TRANSITION_COMPLETED`, a `durability_result` wyłącznie
`METADATA_AND_COMPLETE_SYNCED`. Source nie podpisuje pozytywnego eventu
dla `INTENT`, nieznanego wyniku sync lub obcej roli. Terminal conflict i
niepewność są własnymi zapisami protokołu recovery, bez nadawania im schema
pozytywnego execution event.

CCA payload ma dokładnie:

```text
schema, version, domain, installation, completion_authority, database,
custody, close, close_observation, originals, source_event_envelope_b64,
source_event_envelope_sha256, decision_id, decision_log_position,
source_authentication_result, execution_result, durability_result,
completion_result
```

`completion_authority` jest rekordem `authority` przypiętym do completion-only
credential. `decision_log_position` jest `log_position`, którego `entry_id`
równa się `decision_id`. CCA sprawdza source envelope signature i wszystkie
współdzielone binding: `installation`, `database`, `custody`, `close`,
`originals` muszą być identyczne. `close_observation` zachowuje osobny
qualified direct reader principal/OID oraz transaction ID CCA i executora;
każdy katalogowy digest musi odpowiadać wspólnemu database pinowi.
CCA zachowuje original source envelope;
nie tworzy zastępczego eventu dla tego samego wykonania.

Pozytywny przyszły wynik wymaga dokładnie
`source_authentication_result=QUALIFIED_INSTALLED_SOURCE`,
`execution_result=EXACT_LIFECYCLE_TRANSITION_COMPLETED`,
`durability_result=SOURCE_AND_CCA_RECORDS_SYNCED` oraz
`completion_result=AUTHENTICATED_CUSTODY_COMPLETION_RETAINED`.
To zamknięty schema pozytywnej retained evidence, nie pola, których ustawienie
przez callera kwalifikuje system. Brak wymaganej przesłanki nie może utworzyć
takiego payloadu. Publication eligibility, ACTIVE fence i readiness nie są
polami możliwymi do nadania przez ten dokument.

Future authenticated odpowiedź CCA ma dokładnie `schema`, `version`,
`operation_id`, `decision_id`, `evidence_envelope_b64`,
`evidence_envelope_sha256`, `result`; schema to
`cryptohunter.independent-custody-completion-ack`, version `1`, a pozytywny
result `DURABLE_ORIGINAL_RETURNED`. Jest przesyłana dopiero po trwałym
decision log przez qualified authenticated channel. Odbiorca osobno
porównuje envelope digest, podpis, original bytes i operation binding;
samo pole `result` lub nieuwierzytelniony transport nie jest ack. Nie
potrzeba ponownego podpisania retained evidence przy exact lookup.

Canonical JSON używa istniejącej trusted serialization. Decoder odrzuca
duplicate fields, niekanoniczne bytes, NaN/infinity, bool w miejscu int,
nieprawidłowe Unicode/encoding, niewłaściwe typy i zakresy. Rozmiary document,
embedded bytes i głębokość muszą mieć jawne limity przed kosztownym
dekodowaniem. Digests są identyfikatorami zachowanych oryginałów, nie
substytutem ich dostępności. Zewnętrzny removable plik lub sam URL nie
zastępuje trwałej retencji exact bytes.

### Dwa różne oryginały operacji

Obecny production `begin_close` przyjmuje dokładnie
`generation`, `identity`, `operation_id`, `predecessor_state`, `semantic_role`
i `target_state`. SQL przechowuje request jako `jsonb`; jego `request_sha256`
to SHA256 UTF8 reprezentacji PostgreSQL `request::text`. Nie zachowuje
oryginalnych wire bytes requestu callera. Przyszły odczyt musi pobrać exact
authoritative reprezentację przechowanego requestu wraz z format identity
i zweryfikować digest, zamiast twierdzić, że odtworzył utracone wire bytes.

Lokalny journal V2 zachowuje inne oryginalne canonical descriptor bytes:
`protocol_version`, `operation_id`, `fence_key`, `semantic_role`,
`credential_identity`, `public_key_hex`, `key_version`,
`lifecycle_generation`, `predecessor_state`, `target_state`, `custody_scope`
i `database_identity`. Nie są one tekstem production requestu ani jego hashem.

Future mapper musi preserve oba oryginały i potwierdzić przynajmniej:
identyczne operation ID i semantic role, SQL identity równą credential
identity, generation równą predecessor generation, dokładne stany oraz
pełne authority/material/version/namespace/profile/trust/DB binding ze
zgodnych installation i fence facts. Odrębny `fence_key` lokalnego testowego
protokołu nie staje się automatycznie production fence identity. Brak
jednoznacznego mapowania oznacza `NOT_QUALIFIED`; nie tworzymy production
admission przez adaptację izolowanego coordinatora.

Legalne przejścia to wyłącznie `ACTIVE → VERIFY_ONLY`, `ACTIVE → REVOKED`
i `VERIFY_ONLY → REVOKED`, z generation zwiększoną dokładnie o jeden.
Current metadata muszą odpowiadać dokładnemu `after_metadata`. Późniejszy
legalny descendant nie potwierdza tego completion. Inna rola, materiał,
namespace, key version, generation lub operation descriptor jest konfliktem.

## Projekt protokołu source, trwałości i recovery

1. Lifecycle administrator wykonuje `begin_close` i otrzymuje potwierdzenie
   osobnego COMMIT `PENDING_CLOSED`. Nie trzyma transakcji DB podczas
   mutation. Przy nieznanym wyniku wykonuje wyłącznie exact resolution.
2. Przyszły CCA/source odczytuje close z własnym przypiętym credential przez
   nowy, osobno reviewed minimalny port odczytu. Odczyt zachowuje qualified
   primary/catalog/physical identity, direct login i peer authentication,
   wymaga innej transakcji niż `close_xid`, synchronizuje się na slot/fence
   i sprawdza obydwa oryginały oraz dokładny predecessor. Forwarded response
   administratora nie wystarcza. Brak rekordu pozostaje wynikiem nieznanym.
3. Role-specific executor blokuje właściwy custody namespace. Pod blokadą
   porównuje rzeczywisty credential i predecessor, zapisuje exact immutable
   `INTENT` z before/after i descriptor oraz synchronizuje plik i właściwe
   katalogi. Konflikt operation ID nie może nadpisać starego intent.
4. Executor wykonuje dokładnie dozwolony lifecycle transition istniejącą
   trusted custody ścieżką, synchronizuje zmienione metadata i katalogi,
   następnie zapisuje `COMPLETE` oraz synchronizuje journal i katalog.
   Nie eksportuje root/history private keys i nie otwiera możliwości SIGN.
5. Ten sam executor tworzy source-authenticated execution event powiązany
   z oboma oryginałami operacji, exact journal bytes i before/after,
   qualified close, role/credential/generation i własną tożsamością. Event
   trafia do własnego trwałego, role-scoped event log; sukces i authenticated
   ack następują dopiero po trwałym zapisie. To nie jest ogólna usługa
   podpisywania danych przekazanych przez administratora.
6. CCA uwierzytelnia źródło i event z independently installed trust anchor,
   sprawdza live peer/role, exact journal/metadata oraz close. Sprawdzenia
   source obejmują zamknięty schema, executable/config provenance i brak
   alternatywnego writer/signing entrypoint. Sam PIN, UID lub hash programu
   nie dowodzi, że program wykonał mutation; to cały qualified execution
   protocol nadaje eventowi znaczenie.
7. CCA pod własną blokadą serializuje decision log. Idempotency key to
   `(installation_id, operation_id)`, a zachowany binding zawiera wszystkie
   identity/DB/role/credential/predecessor/descriptor/journal/source facts.
   Ten sam klucz z inną treścią daje trwały konflikt. Exact replay odczytuje
   oryginalną decyzję i oryginalne evidence bytes, bez ponownej mutation
   lub zastąpienia poprzedniego eventu. Dwa requests nie mogą wygrać różnymi
   decyzjami dla tego samego klucza.
8. CCA utrwala własne intent/decision, oryginalne podpisane payload/envelope
   i linkage zgodnie z osobno qualified log protocol. Awaria między etapami
   wymaga odczytu zachowanych faz; tworzenie nowego podpisu lub decision ID
   nie jest sposobem rozwiązania utraconej odpowiedzi. Po COMMIT/`fsync`
   CCA uwierzytelnia odpowiedź, a odbiorca porównuje original bytes,
   signature, pin i operation binding. Odpowiedź przed trwałą decyzją nie
   jest positive durable result.
9. Publication jest późniejszym, obecnie niedostępnym etapem: granica SQL
   musi sama zweryfikować niezależną evidence i exact pending close oraz
   zachować receipt bytes i replay semantics. Nie dodajemy jej w tym PR.
   Completion retirement zachowuje zakaz `REVOKED → ACTIVE`; ewentualna
   aktywacja successora jest osobnym protokołem i kwalifikacją.

Od finalnego exact re-read do potwierdzenia trwałego CCA decision source
musi zachować kwalifikowany custody lease/lock, który blokuje konkurencyjną
mutation tego credential. CCA nie zwalnia tej gwarancji po odebraniu DTO.
Jedna przyszła reviewed kolejność to: committed DB close bez otwartej
transakcji → source custody lock → CCA decision-log lock → krótki własny
qualified DB read z slot/fence lock → zakończenie odczytu DB → trwały CCA
decision i ack → zwolnienie CCA/source locks. Ponowny finalny re-read
odbywa się już pod obydwoma odpowiednimi lockami. Inne ścieżki nie mogą
brać ich w odwrotnej kolejności. Rotation/revoke source i CCA credentials
muszą uczestniczyć w ich własnym log/lifecycle serialization do końca
odpowiedniego source event lub CCA decision. Odrębny source event podpisany
przed legalną rotacją wymaga zachowanego uwierzytelnionego historical
eligibility, zamiast uznania aktualnej etykiety `ACTIVE` w tym evencie.
Brak któregoś z tych interlocków oznacza `NOT_QUALIFIED`.

Odczyt `COMPLETE` po restarcie potwierdza retencję zaobserwowanych bytes,
ale nie jest dowodem zachowania cache/storage podczas utraty zasilania.
Kwalifikacja deklarowanej durability wymaga realnej konfiguracji storage,
semantyki `fsync`/rename/directory sync i adekwatnych crash/restart tests.
Raport diagnostyczny nie może samodzielnie nadać statusu tych gwarancji.

Timeout, `NOT_FOUND`, zerwane połączenie, restart lub utrata odpowiedzi
oznaczają `INDETERMINATE`, jeżeli operacja mogła zostać wykonana. Odbiorca
rozstrzyga ten sam operation ID i descriptor na authoritative logach.
Nie wolno ponawiać nieodwracalnej mutation na podstawie samego braku
odpowiedzi, utworzyć replacement operation ani otworzyć fence.

## Lifecycle klucza CCA i rollback

Provisioning CCA i source keys wymaga oddzielnej ceremony. Installation pin
zawiera ich credential identities, public material, namespace, key version,
trust epoch i approved lifecycle. Private keys nie są parametrem requestu
ani polem publicznego manifestu. Credential aktywny wyłącznie dla jednej
domeny nie może być aliasem signer key, administrator credential lub
wspólnego testowego MAC. Procesy nie wykonują automatycznego TOFU, re-pin,
generacji brakującego klucza ani naprawy konfiguracji.

Rotacja/revoke CCA i executorów musi serializować eligibility z decyzją
i trwale zachować historyczne keys/eligibility/decision linkage. `VERIFY_ONLY`
nie wystawia nowej evidence; może weryfikować lub zwrócić dokładnie
zachowaną wcześniejszą decyzję według zatwierdzonego historical protocol.
Po `REVOKED` sam wcześniejszy podpis nie stanowi dowodu historycznego
uprawnienia. Potrzebna jest zachowana evidence sprzed kompromitacji lub
niezależnie od niej; bez niej wynik jest `NOT_QUALIFIED`/`BLOCKED`.

Sekwencja i hash chain logu wykrywają niespójność względem zachowanego
zaufanego floor. Same lokalne pliki, process cache, zegar i `fsync` nie
wykrywają skoordynowanego przywrócenia wszystkich lokalnych snapshotów.
Lokalny profil nie otrzymuje `independent_rollback_domain=true`. Utrata
zewnętrznego pina, rollback trust epoch lub brak potrzebnej historical
evidence powodują odmowę; wall clock nie wybiera prawidłowej historii.

## Czternaście przypadków crash, replay i współbieżności

Poniższa tabela określa wymagane zachowanie i granicę obecnego dowodu.
Nie jest listą wyników PASS nowych testów.

| Przypadek | Wymagane zachowanie | Obecna granica |
| --- | --- | --- |
| 1. Crash przed mutation | Close pozostaje `PENDING_CLOSED`; brak source event i completion; jedynie exact read resolution | Istniejąca granica DB-first; nowy runtime source/CCA `NOT_QUALIFIED` |
| 2. Crash po mutation, przed trwałym `COMPLETE` | `INTENT` i current after nie wystarczają do completion; bez automatycznej ponownej mutation | Read-only verifier odmawia `INTENT`; recovery executora wymaga osobnej kwalifikacji |
| 3. Trwały `COMPLETE` bez CCA evidence | Możliwa exact obserwacja; publication i fence pozostają zamknięte | Pozytywny odczyt nie zmienia brakujących source/execution/durability statuses |
| 4. Utrata odpowiedzi CCA | `INDETERMINATE`; exact lookup trwałej decyzji, oryginalne bytes; zero retry mutation | CCA log/ack/lookup jeszcze nie istnieje |
| 5. Restart jednego lub wszystkich procesów | Ponowna instalacyjna qualification; original durable source i CCA logs, brak polegania na cache/capability | Local observation może być powtórzona; restart niezależnej authority wymaga przyszłych real-process tests |
| 6. Exact retry | Identyczne bytes i binding, jedna retained decyzja; bez ponownego execution/SIGN | Format/read checks nie zastępują trwałego CCA idempotency log |
| 7. Ten sam operation ID, inny descriptor | Conflict; brak last-write-wins, nowej decyzji lub nadpisania journalu | Exact descriptor binding sprawdzany; przyszły globalny CCA conflict log niezakwalifikowany |
| 8. Inna role/credential/generation | Odmowa nawet przy podobnym public material lub końcowym stanie | Zamknięte format/descriptor/journal/metadata checks; brak awansu do source authentication |
| 9. Rotation/revoke w trakcie | Lock/CAS i source/CCA lifecycle eligibility zachowane do ich durable decision; stale generation odmawia | Lokalna obserwacja nie linearizuje wszystkich przyszłych logów i lifecycle; `NOT_QUALIFIED` |
| 10. Journal/symlink/inode/directory swap | Descriptor-based path pins, ownership/ACL i rechecks pod blokadą; odmowa przy zmianie | Istniejący odczyt real files zapewnia ograniczoną obserwację, bez source attestation |
| 11. Niewłaściwy OS principal/peer | Odmowa fixed identity/role/endpoint/pidfd checks, bez przyjmowania etykiety w DTO | S9-RPI-02 diagnostic peer checks istnieją; osobny CCA/source transport niezakwalifikowany |
| 12. Bezpośredni SQL completion | Odmowa nawet z poprawnym hashem lub lokalnym raportem | Python odmowa oraz SQL tombstone/ACL V2 zachowane |
| 13. Utrata/zmiana external pin | `NOT_QUALIFIED`/`BLOCKED`; żadnego TOFU lub automatycznego re-pin | Nieznany lub podmieniony anchor nie może zostać zaakceptowany z evidence |
| 14. Dwa równoczesne completion requests | Custody/source locks i CCA durable uniqueness: exact replay jednej decyzji albo trwały conflict | Weryfikacja read-only nie stanowi implementacji future durable decision arbitration |

## Status wymaganych dowodów

| Właściwość | Status S9-RPI-05 | Warunek zmiany statusu |
| --- | --- | --- |
| Zamknięty format i exact binding lokalnej evidence | Zakres implementacji odczytowej; wynik walidacji poniżej | Testy formatu i real files; nadal brak production capability |
| Exact journal `COMPLETE` i current metadata pod lock | Odczytowa obserwacja | Tylko wskazana operacja i aktualny exact generation; nie source authentication |
| Własna installed CCA identity/key/namespace/lifecycle | `NOT_QUALIFIED` | Osobna provisioning ceremony i fizyczna kwalifikacja rozłącznych principals |
| Authenticated role-scoped mutation executor/source | `NOT_QUALIFIED` | Reviewed execution protocol, keys/pins, own event log i real negative/process tests |
| Source execution i wcześniejsza trwałość mutation | `NOT_QUALIFIED` | Qualified executor, real writes/sync/storage/restart dowody i retained authenticated event |
| Własny uwierzytelniony odczyt committed production close przez CCA | `NOT_QUALIFIED` | Osobny minimalny read port, role i jawna migracja; preserve oba oryginały |
| CCA durable decision log, exact recovery i authenticated ack | `NOT_QUALIFIED` | Implementacja log protocol i fault/crash/restart/concurrency qualification |
| Klucz publiczny i external trust pin | CCA/source pin nieprovisionowany | Zewnętrznie zatwierdzona instalacja; brak automatycznego pinning z requestu |
| Publication odporne na direct SQL bypass | `BLOCKED` | Osobno approved contract/migration/CCA i verifier działający na granicy SQL |
| Produkcyjne otwarcie fence / activation / succession | `BLOCKED` | Osobny lifecycle/activation protokół i fizyczna kwalifikacja |
| Pełna niezależność od host root i host rollback | Poza gwarancją `PRODUCTION_LOCAL` | Inna osobno qualified security/rollback topology; nie zmiana nazwy profilu |
| Addendum C i atomowy production issuer | Zachowany `CONTRACT_RECONCILIATION_PENDING_APPROVAL` / `BLOCKED`; kierunek A+C przyjęty | Formalna versioned contract integration i pozostałe bramki S9-RPI-04 |
| Public Root Proof i całe production M0.5 | `NO-GO` | Wszystkie niezależne readiness predicates, CHA/Freshness i AccountGenesis closure |

## Fizyczna procedura kwalifikacji do przyszłego wdrożenia

1. Przejrzeć i zatwierdzić wersję kontraktu CCA, exact schema/domain,
   source protocol, lifecycle/recovery i scope trust model. Osobno rozstrzygnąć
   formalną integrację addendum C zgodnie z kierunkową decyzją A+C; utrwalić
   sources i digests bez przepisywania freeze.
2. Poza requesterem i runtime wybrać physical installation/service identity,
   environment, profile/trust domain oraz chroniony trust pin. Zweryfikować
   bootstrap principal; nie pobierać oczekiwanego katalogu z nieznanej bazy.
3. Ustanowić rzeczywiście rozłączne konta i credentials CCA, source root,
   source history i wcześniejszych services. Sprawdzić UID/GID/groups,
   namespace isolation, effective ACL, protected ancestry, unit mounts,
   hardlinks/symlinks i odmowę dostępu do obcych private keys/native keyring.
   Przeprowadzić realny procesowy negative test cross-role i peer substitution.
4. Zakwalifikować źródło od początku do końca: own executable/config,
   brak arbitrary-hash signing API, custody lock, exact committed admission,
   durable `INTENT`, metadata transition, synchronizacje, durable `COMPLETE`
   i own source-auth event. Sprawdzić każdą persistence cut oraz restart
   z original bytes, bez wstrzykniętej pozytywnej mutation assertion.
5. Zakwalifikować PostgreSQL 16 primary i pełną physical/service identity,
   durability settings, reviewed source/ACL/ownership, role OID i wszystkie
   login paths `pg_hba.conf`/`pg_ident.conf`/endpointy. Local socket i user name
   nie zastępują rzeczywistego peer authentication. Clone PGDATA może
   zachować system identifier/OID; nie jest dowodem jednego physical backendu.
6. Osobno review i wykonać konieczną migrację read/publication granicy z
   dokładnie uprzednio zatwierdzonego katalogu. Zachować evidence i legacy
   classification; brak runtime upgrade lub re-pin. Dotychczasowa migracja
   V1→V2 wymaga osobno committed `ALLOW_CONNECTIONS=false`, opróżnienia innych
   sesji i wcześniej otwartego bootstrap connection; helper nie wyłącza
   usług ani nie kończy sesji. Nie utożsamiać jej z przyszłą zgodą na CCA
   publication. Nowa migracja wymaga własnego reviewed protokołu quiescence,
   pełnej odmowy drift i zatwierdzonego zewnętrznego pina po COMMIT.
7. Zakwalifikować trwały CCA decision log i source/CCA key lifecycle:
   fault injection przed/po każdym sync/commit, timeout, lost response,
   restart procesów/DB, exact byte replay, konflikt dwóch requests, rotation
   i revoke. Decyzja i jej oryginalny signed envelope muszą przetrwać restart;
   testowy credential i mock filesystem nie kwalifikują produkcji.
8. Przed publication sprawdzić w rzeczywistym SQL odmowę admin/issuer/raw
   callers, brak hash-only/MAC/DTO bypass oraz pełną weryfikację podpisu,
   trust pin, exact close i original bytes na docelowej publication boundary.
   Dzisiaj poprawnym wynikiem tego kroku jest odmowa tombstone.
9. Zachować dokumentację instalacji, original evidence, exact polecenia,
   wyniki i zakres dowodów; jawnie wyliczyć niezakwalifikowane predicates.
   Dopiero osobne zatwierdzenie całości może zmienić readiness. Nie wykonuje
   się prób Stage 10, enrollmentu lub produkcyjnego BIND w ramach tej pracy.

## Walidacja tej zmiany

Walidację wykonano 2026-10-18 na bazie
`a3d01ac2afd9ab54469342f41e760cb38649ca32` po #3155. Osobny venv używał
Python 3.12; root Linux harness działał w jednorazowym `python:3.11-slim`
z domyślnymi capabilities, bez privileged container. Dwa izolowane klastry
PostgreSQL **16.15**, porty 57432/57434, miały włączone `fsync`,
`full_page_writes` i `synchronous_commit`. Kontroler restartu wskazywał
wyłącznie jednorazowy klaster testowy.

| Kontrola | Wykonany zakres i wynik |
| --- | --- |
| Nowy codec/observer | `python -m pytest -p asyncio -p pytest_cov.plugin tests/security/test_independent_custody_completion_qualification.py --cov=bot_core.independent_custody_completion_qualification --cov-branch`: **102 PASS**, bez skip/xfail |
| Pokrycie | 197/204 linii (**96.57%**), 83/90 gałęzi (**92.22%**); pomiar coverage z gałęziami **95%**; `diff-cover --compare-branch=origin/main --fail-under=70`: **96.6% PASS** |
| Signer diagnostics | IPC, custody, deployment i boundary: **160 PASS** |
| Real Linux process/DAC/peer | `test_production_signer_linux_processes.py` we wskazanym kontenerze: **6 PASS**; rzeczywiste testowe UID/GID, bez produkcyjnych kluczy |
| Production PostgreSQL lifecycle | Schema, qualification, identity i races: **136 PASS**, bez skip; SQL tombstone/ACL, migracja, direct SQL, drift i kontrolowany restart |
| Istniejący isolated custody coordinator | `test_postgresql_custody_lifecycle.py`: **31 PASS** w 404.88 s, bez skip; genuine INITIAL_BINDING fixture, realny admin crash, utrata odpowiedzi COMMIT, restart i wyścigi. To nadal testowy coordinator, nie nowa CCA |
| Lokalne journal/lock/admission | `test_local_signing_lifecycle_evidence.py`, `test_native_custody_locking.py`, `test_local_custody_admission_contract.py`: **258 PASS**, 10 istniejących Windows-only skips |
| Frozen/NO-GO | RPI-04 refusals, custody admin contract, substrate/readiness contracts i atomic boundary: **397 PASS** |
| Po rebase na #3158 | Zmienione upstream signer/identity tests oraz ścisły test inwentarza CI: **207 PASS**, jeden istniejący upstream non-Linux case pominięty na Linux. Dodano wyłącznie oczekiwane nazwy nowego testu/modułu do inwentarza, bez osłabienia asercji |
| Ruff i format | Pełny scope Ruff z `Quality and Security`: **PASS**; `python -m pre_commit run ruff-format --all-files`: **PASS po staging nowych plików**, bez zmian formattera; writable `PRE_COMMIT_HOME` w scratch |
| Mypy | Skonfigurowane `python -m mypy` i `python -m mypy --platform win32`, bez filename overrides: **PASS**, po 339 plików |
| Semgrep CE 1.179.0 | Blocking custom rules: **0 findings**, 1048 plików/5 reguł; fixtures **5/5 PASS**; jawny scan nowego modułu **0 findings** |
| Betterleaks 1.9.0 | Offline baseline scan po rebase: **0 nowych leaks**, 57.88 MB; checksum binarki sprawdzony. Snapshot zawiera tracked i nowe deliverables, wyłącza wyłącznie scratch `work/` z narzędziami/venv |
| Niezmienność blokad | Diff istniejących ośmiu modułów i frozen JSON względem `origin/main`: pusty; canonical readiness/Stage 9/Stage 10 bez zmian |

Testy używały `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1`, `DUDZIAN_TEST_MODE=1`
i `DUDZIAN_ALLOW_LONG_POLL=0`; IPC/procesy miały rzeczywisty dostęp do
lokalnych socketów. Dziesięć istniejących skips dotyczy Win32/NTFS: DACL,
junction/hardlink, opened-object/volume identity, share-delete i native
Windows timeout. Nie stanowią PASS kwalifikacji Windows; nie dodano
skip/xfail dla błędu Linux. `mypy --platform win32` nie zastępuje tych testów.
Osobny skip po rebase dotyczy upstream testu odmowy na systemie innym niż
Linux; jego wykonania na takim systemie tutaj nie deklarujemy.
Ruff zgłosił trzy wcześniejsze ostrzeżenia o `WPS433`; nie dodano suppressions.

Realny lokalny writer w nowych testach używa jawnego istniejącego
testowego close-validator seam. Ręcznie odtworzone identyczne `COMPLETE`
i metadata otrzymują ten sam `NOT_QUALIFIED`; fixture nie kwalifikuje native
keyring ani production close. Claim i raport nie przechodzą przez `commit()`
ani publication, a pole syntetycznego podpisu jest odrzucane. Test limitu
depth zabrania wywołania JSON decoder, dowodząc odmowy przed rekursywnym
dekodowaniem. Realny shared lock, restart procesu, concurrent odczyty oraz
path/inode/directory swaps nie nadają source authentication.

Obowiązują dotychczasowe wymagania: celowane testy, real Linux processes
tam, gdzie badane są OS/IPC, real files/journals/locks, konieczne regresje
na izolowanym PostgreSQL 16, Ruff, osobny pełny
`python -m pre_commit run ruff-format --all-files`, skonfigurowany mypy
Linux/Windows, właściwe security scans oraz kontrola niezmienionych frozen
contracts i canonical statusów. Nie zwiększamy timeoutów, nie zmieniamy
coverage i nie maskujemy braku dowodu przez `skip` lub `xfail`.

Celowana walidacja nie zastępuje pełnego CI. PR ma status `FULL_CI_PENDING`,
jeśli pełne CI nie zostało ukończone; brak automatycznego merge. Wykonany
test lokalny nie jest kwalifikacją fizycznego wdrożenia.

## Kolejność dalszych prac i warunki GO

Najpierw review topology/source/evidence i decyzja kontraktowa, następnie
implementacja odrębnego executora/source i CCA log/lifecycle, dopiero potem
osobny provisioning oraz kwalifikacja fizyczna. Zmiana SQL wymaga jawnej
migracji i odrębnie chronionego approved pina. Publication pozostaje
zablokowane do spełnienia wszystkich tych warunków i zatwierdzenia granicy
uwierzytelniającej evidence także dla bezpośredniego wywołania SQL.

Tekst C z [S9-RPI-04](proposals/stage9_root_proof_transaction_authority_addendum.md)
pozostaje historyczną propozycją; kierunek A+C został przyjęty w #3156,
lecz formalna integracja wersjonowanego kontraktu pozostaje otwarta.
Dotyczy on zakresu finalnej atomowej transakcji issuera;
nie zatwierdza CCA, provisioningu lub migracji. Przyjęcie addendum samo nie
odblokuje completion ani issuance. Zachowane pozostają wymagania jednego
qualified backendu i pełnego finalnego write set, wszystkich signer/trust
i requester/claimant fences do COMMIT, genuine attempt ingress, original
proof/history retention, legal cardinality/replacement, checkpoint acceptance,
private-control i profile isolation oraz niezależne bramki readiness.

Completion retirement nie jest produkcyjnym GO. RootProofIssuer nie wykonuje
CHA AttemptStore mutation, Freshness CAS ani AccountGenesis local closure;
checkpoint zachowuje własną transakcję i authority. Do osobnego spełnienia
tych wszystkich przesłanek obowiązuje **NO-GO dla publicznej emisji Root Proof**.
