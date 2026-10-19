# Stage 9 — S9-RPI-06: authenticated custody mutation source protocol

## Wynik i granica tej zmiany

S9-RPI-06 konkretyzuje zamknięty protokół `CustodyMutationExecutionV1`
z [S9-RPI-05](stage9_independent_custody_completion_authority.md). Nowy
[`authenticated_custody_mutation_source.py`](../../bot_core/authenticated_custody_mutation_source.py)
dekoduje oryginalny envelope, sprawdza exact binding i podpis Ed25519 względem
jawnie dostarczonej polityki weryfikacyjnej. Nie wykonuje mutation custody,
nie posiada klucza prywatnego, nie wystawia podpisów ani production capability.
Prawidłowy podpis potwierdza wyłącznie zgodność z publicznym kluczem w tej
polityce. Nie dowodzi pochodzenia polityki, kwalifikacji instalacji, wykonania
operacji, wcześniejszego `fsync` ani uprawnienia do publication.

Odrębny demonstrator w testach ma pokazać kontrolowaną operację procesu
z testowym credential i rzeczywistym zapisem. Nie jest importowalnym
produkcyjnym executorem ani adapterem production admission. Trzy projektowane
authority, ich produkcyjne klucze, transport i logi nie są provisionowane.
Nie dodajemy SQL kont, grantów, migracji, jednostek systemd ani nowych pól
do istniejącego installation manifest. Dokładny zakres wykonanego
demonstratora i testów wymieniono w sekcji walidacji; wymagania projektowe
nie są deklaracją wykonania.

```text
source_event_protocol = CLOSED_FORMAT_AND_SIGNATURE_VERIFICATION_ONLY
authenticated_custody_mutation_source = NOT_QUALIFIED
independent_custody_completion_authority = NOT_QUALIFIED
independent_custody_durability_confirmation = NOT_QUALIFIED
production_custody_completion_publication = BLOCKED
CONTRACT_ADDENDUM_INTEGRATION_PENDING
semantic_root_proof_issuer_runtime = NOT_IMPLEMENTED / BLOCKED
production_provisioning_ready = false
ROOT_PROOF_ISSUANCE_AVAILABLE = false
ATOMIC_ISSUER_READY = NOT_READY
PRODUCTION_LIFECYCLE_ISOLATION = BLOCKED
public_root_proof_issuance = NO-GO
```

To lokalny opis granic PR, bez zmiany canonical statusów. Wynik verifiera
zawsze zachowuje `NOT_QUALIFIED` i `BLOCKED`, również dla poprawnych podpisów,
realnego testowego `COMPLETE` i zgodnych policy snapshots. Nie staje się
`QUALIFIED_INSTALLED_SOURCE`. Dotychczasowy S9-RPI-05 qualification claim
pozostaje odrębnym formatem i nadal zachowuje własną pełną listę blockers.

## Audyt, freeze i bezpieczne ponowne użycie

Bazą jest `main` po #3159, commit
`a31d0589`, oraz bezpośrednie odczyty wskazanych źródeł. Codebase Memory MCP
nie jest callable w tej sesji; użyto celowanych `rg` i rzeczywistych plików.
Nie deklarujemy kompletności nieobecnego indeksu ani kwalifikacji deploymentu.

| Mechanizm i źródło | Co można wykorzystać | Granica, która pozostaje |
| --- | --- | --- |
| [S9-RPI-01](stage9_production_root_proof_lifecycle_isolation.md), [`production_root_lifecycle.py`](../../bot_core/production_root_lifecycle.py) | Immutable close, exact predecessor, osobny potwierdzony COMMIT `PENDING_CLOSED`, `close_xid`, slot/fence synchronization i direct-login qualification | Administrator SQL może zamknąć fence, lecz nie uwierzytelnia wykonania custody. `commit()` i `publish_administrator_completion()` odmawiają |
| [S9-RPI-02](stage9_independent_production_signers.md), [`production_signer_ipc.py`](../../bot_core/production_signer_ipc.py), [`production_signer_custody.py`](../../bot_core/production_signer_custody.py) | Fixed installation, Linux UID/GID, protected custody, peer/pidfd identity i rozłączne namespaces | Tylko `DIAGNOSTIC`/`QUALIFY`. Procesowa tożsamość i seed/public match nie dowodzą niezależnego private control ani source execution |
| [S9-RPI-03](stage9_production_custody_completion_verifier.md), [`local_signing_lifecycle_evidence.py`](../../bot_core/local_signing_lifecycle_evidence.py) | Canonical V2 descriptor/journal, exact before/after, existing read-only `COMPLETE` observation i path rechecks pod lockiem | Odczyt nie potwierdza autorstwa journalu ani wcześniejszej trwałości; `INTENT` i późniejszy descendant odmawiają |
| [`local_signing_custody.py`](../../bot_core/local_signing_custody.py) | Exact credential resolution, generation CAS, trzy jednokierunkowe przejścia, custody lock i istniejące zabezpieczenia otwartych plików | Mutation wymaga autentycznej admission. Nowy verifier nie tworzy jej, nie eksportuje sekretów i nie otwiera legacy transition API |
| [`postgresql_production_root_lifecycle_schema.py`](../../bot_core/postgresql_production_root_lifecycle_schema.py), [`postgresql_production_root_lifecycle_qualification.py`](../../bot_core/postgresql_production_root_lifecycle_qualification.py) | Dokładne function sources, ACL, ownerzy `NOLOGIN`, role OID, stały `search_path`, catalog/physical pins i peer authentication | SQL V2 `complete_close(text,text,bytea)` pozostaje odmawiającym tombstone; bootstrap superuser nadal jest zaufaną granicą provisioningu |
| [`postgresql_custody_lifecycle.py`](../../bot_core/postgresql_custody_lifecycle.py), [isolated V2](stage9_durable_custody_lifecycle_v2.md) | Regresje istniejącego DB-first test protocol, recovery, locks i exact original bytes | `ISOLATED_TEST_ONLY`, testowy MAC i istniejący testowy approval nie stają się production admission lub source trust anchor |
| [S9-RPI-04](stage9_root_proof_transaction_authority_reconciliation.md) i [decyzja A+C](decisions/stage9_root_proof_transaction_authority_owner_approval_v1.md) | Rozstrzygnięty kierunek jednego finalnego atomowego write set issuera | Formalna versioned contract integration pozostaje otwarta; custody, CHA, freshness, checkpoint i AccountGenesis zachowują własne authority/transakcje |
| [S9-RPI-05 codec](../../bot_core/independent_custody_completion_qualification.py) i [ADR](stage9_independent_custody_completion_authority.md) | Trusted canonical serialization, bounded strict decoding, exact request/descriptor/journal binding i jawne blockers | Qualification claim nie jest podpisanym source eventem; polityka, key lub installation podane przez callera nie są niezależnym dowodem |

Sprawdzone frozen źródła pozostają bez zmian:

- [Independent issuer](cryptohunter_product_architecture/m05_account_genesis_independent_root_proof_issuer_contract.json):
  exact BIND i byte-identical recovery; brak odpowiedzi nie zezwala na
  replacement, powtórne podpisanie ani wnioskowanie `UNBOUND`.
- [Production substrate](cryptohunter_product_architecture/m05_account_genesis_root_proof_issuer_production_substrate_selection_contract.json):
  rozłączne authority, credentials, namespaces i lifecycle; odrębne
  profile TEST/production, jawne migracje i `NOT_FOUND=OUTCOME_UNKNOWN`.
- [Implementation readiness](cryptohunter_product_architecture/m05_account_genesis_root_proof_issuer_implementation_readiness_contract.json):
  `/status/implementation_allowed=false`,
  `/implementation_gate/readiness_freeze_does_not_allow_implementation=true`
  i `/implementation_gate/non_production_implementation_mode=NONE_FROZEN`.
- [Physical persistence](cryptohunter_product_architecture/m05_account_genesis_physical_persistence_crash_atomicity_contract.json):
  własne granice CHA/FreshnessAuthority i
  `/implementation_allowed/design_validators_and_contract_tests_only=true`.

`NONE_FROZEN` nie zostaje zinterpretowany jako zgoda na testowy semantic issuer.
Nie jest też absolutnym zakazem każdego izolowanego harnessu subkomponentu:
[Atomic Commit V1](stage9_atomic_root_proof_commit_authority_v1.md) jawnie
odróżnia istniejący testowy mechanizm persistence od dostępnego semantic
RootProofIssuer przy zachowanym tym samym freeze. W tym PR świadomie
ograniczamy kod aplikacji do protocol verifiera. Demonstrator pozostaje
testem odpowiedzialnego wykonawcy we własnej kontrolowanej domenie;
nie implementuje trybu issuer runtime ani nie wykorzystuje flagi TEST do
otwarcia produkcyjnej ścieżki. Nie zmienia implementacyjnych gates żadnej
niezależnej authority.

Kierunek A+C jest zatwierdzony. Historyczny label
`CONTRACT_RECONCILIATION_PENDING_APPROVAL` w S9-RPI-05 nie oznacza, że decyzji
właściciela nadal brakuje. Aktualna zależność to
`CONTRACT_ADDENDUM_INTEGRATION_PENDING` i osobna produkcyjna kwalifikacja.
Nie integrujemy addendum w tym PR i nie traktujemy kierunkowej decyzji jako
zgody na CCA/source provisioning lub SQL migration.

## Docelowa topologia trzech authority

Poniższe nazwy są propozycją do osobnego review, a nie nowymi zainstalowanymi
principals. Root source, history source i CCA muszą być różne również od
obu signerów, lifecycle administratora, issuer writera, requestera, claimanta,
trust activation i checkpoint authority. Dla każdej instalacji przypina się
rzeczywiste numery UID/GID/OID; podobna nazwa konta niczego nie dowodzi.

| Authority | OS principal i katalogi | Credential i zakres dostępu | PostgreSQL i transakcja |
| --- | --- | --- | --- |
| Root Custody Executor/Source | Dedykowany `ch_root_custody_source`, własny UID/GID, `/etc/cryptohunter/root-custody-source/`, `/var/lib/cryptohunter/root-custody-source/`, `/run/cryptohunter/root-custody-source/` | Osobny Ed25519 execution-authentication credential, root-source namespace i własny log. Wyłącznie kwalifikowany root lifecycle; zero history/CCA secrets i zero Root Proof signing credential | Osobny minimalny direct-login reader `ch_root_source_reader`; odczyt exact committed close we własnej krótkiej transakcji, bez SQL completion/admin/write |
| History Custody Executor/Source | Inny `ch_history_custody_source`, inny UID/GID i analogiczne odrębne `history-custody-source` katalogi | Inny execution credential i namespace. Wyłącznie history lifecycle; zero root/CCA secrets i zero History Attestation signing credential | Inny `ch_history_source_reader`, bez memberships, DML/DDL i uprawnień do root-role close |
| Custody Completion Authority | Inny `ch_custody_completion`, własny UID/GID i katalogi `custody-completion` | Completion-only Ed25519 credential, własny decision log i namespace. Odczyt publicznej evidence przez authenticated source channel; zero signer/source private keys i zero mutation | Osobny `ch_completion_reader`; własny exact close read. Ewentualny przyszły publication credential wymaga odrębnego reviewed SQL boundary i migracji |

Konfiguracja i installation pins mają root-owned, niezapisywalne dla runtime
ancestry; proponowane katalogi konfiguracji `0750 root:<role_gid>` i pliki
`0640 root:<role_gid>`, bez dodatkowych ACL. Private state/key/log directory
ma `0700 <role_uid>:<role_gid>`, private files `0600`; exclusive groups bez
cross-role lub supplementary memberships. Native key storage musi stosować
równoważną rzeczywistą izolację, bez wspólnego session bus lub dostępnego
obcej roli keyringu. Private keys nie są polami manifestu ani requestu.
Nie generuje się automatycznie brakującego klucza ani nowego pina.

Sockets potrzebują osobnego, wąskiego interfejsu: runtime directory z
niezapisywalnym przez klienta ancestry i socket dostępny wyłącznie ustalonemu
klientowi. Nie należy dodawać CCA do private primary group źródła tylko po to,
aby udostępnić endpoint; konkretny socket access mechanism wymaga osobnego
review, z kontrolą efektywnego ACL i ponowną weryfikacją otwartego obiektu.
Obie strony sprawdzają live peer UID/GID, socket-associated pidfd, start time,
service executable/config provenance i właściwy namespace. DTO z PID/UID nie
zastępuje tych obserwacji. Nowy source transport nie powstaje w tym PR.

Minimalny source executor powinien móc mutować wyłącznie publiczne lifecycle
metadata w przypisanym custody namespace. Istniejąca kwalifikacja chronionego
storage wymaga bieżącego właściciela; samo dodanie source UID do grupy signera
lub nadanie prawa zapisu całego keyringu nie jest rozwiązaniem. Rozdzielenie
metadata writer od signer secret reader, właściciele plików i reguły blokady
wymagają osobno zatwierdzonej wersji/provisioningu. Source nie uzyskuje prawa
SIGN w zamian za uprawnienie do retirement. Nie poszerzamy obecnych modes,
owner checks ani siedmiu principals production installation manifest.

Usługi docelowe potrzebują minimalnych capabilities, `NoNewPrivileges`,
read-only system/config mounts, wąskiego `ReadWritePaths`, osobnych mount,
IPC i network namespaces oraz odmowy dostępu do katalogów innych ról.
Lista deklaracji systemd nie stanowi kwalifikacji: porównuje się rzeczywiste
UID/GID/groups, mounts, namespace inodes, effective ACL, keyring/session bus,
role OID, membership i wszystkie login/endpoint paths. Host root może
zmienić procesy i software keys; pozostaje poza gwarancją `PRODUCTION_LOCAL`.

## Exact source-event protocol V1

Source envelope ma dokładnie następujące pola:

```text
schema, version, payload_b64, credential_id, key_version, signature_b64
```

`schema=cryptohunter.signed-custody-mutation-execution`, `version=1`.
Podpis Ed25519 ma 64 bytes i obejmuje:

```text
b"CryptoHunter/CustodyMutationExecution/V1\0" + original_canonical_payload_bytes
```

Końcowe `\0` oznacza pojedynczy bajt NUL domeny podpisu. Nie dodaje się
drugiego separatora. CCA użyje innej domeny:
`b"CryptoHunter/IndependentCustodyCompletion/V1\0"`; source signature nie
może zostać użyta jako CCA signature, Root Proof lub History Attestation.
Envelope credential/version muszą odpowiadać `executor` i jawnej polityce.
Nie ma fallbacku MAC, hash-only signature ani self-signed trust anchor.

Canonical payload ma dokładnie:

```text
schema, version, domain, installation, executor, database, custody,
close, close_observation, originals, execution_id, source_log_position,
execution_result, durability_result
```

`schema=cryptohunter.custody-mutation-execution`, `version=1`.
`domain` to JSON string `CryptoHunter/CustodyMutationExecution/V1`, bez NUL;
interpretację podpisu wyznacza stała bajtowa protokołu z separatorem NUL.
Dodatkowe pola, zmiana wersji lub domeny odmawiają.
Zachowujemy dokładnie zamknięte rekordy z S9-RPI-05:

| Rekord | Dokładne pola |
| --- | --- |
| `installation` | `installation_id`, `environment_id`, `profile`, `trust_domain`, `installation_sha256`, `trust_pin_id`, `trust_pin_version`, `trust_pin_sha256` |
| `executor` (`authority`) | `authority_id`, `service_id`, `host_id`, `os_uid`, `os_gid`, `endpoint`, `provider_namespace`, `credential_identity`, `semantic_role`, `public_key_b64`, `public_material_identity`, `key_version`, `lifecycle_generation`, `lifecycle_state`, `trust_epoch` |
| `database` | `system_identifier`, `database_oid`, `database_name`, `cluster_name`, `connection_host`, `connection_port`, `server_address`, `server_port`, `instance_id`, `authority_identity`, `schema_name`, `schema_version`, `catalog_sha256` |
| `custody` | `semantic_role`, `authority_identity`, `credential_identity`, `key_handle_or_version`, `custody_lifecycle_namespace`, `public_key_b64`, `public_material_identity`, `key_version`, `predecessor_generation`, `successor_generation`, `predecessor_state`, `target_state` |
| `close` | `operation_id`, `request_format`, `request_b64`, `request_sha256`, `close_xid`, `phase` |
| `close_observation` | `reader_role`, `reader_role_oid`, `observation_xid`, `qualified_catalog_sha256` |
| `originals` | `operation_descriptor_b64`, `operation_descriptor_sha256`, `complete_journal_b64`, `complete_journal_sha256`, `before_metadata_b64`, `before_metadata_sha256`, `after_metadata_b64`, `after_metadata_sha256` |
| `source_log_position` (`log_position`) | `namespace`, `sequence`, `previous_entry_sha256`, `entry_id` |

`execution_result` ma wyłącznie pozytywną wartość
`EXACT_LIFECYCLE_TRANSITION_COMPLETED`, a `durability_result` wyłącznie
`METADATA_AND_COMPLETE_SYNCED`. Są to podpisane assertions źródła; parser nie
wykonuje kroków, których dotyczą. Nie przyjmuje schema pozytywnego eventu
dla `INTENT`, nieznanego sync outcome, braku close lub obcej roli.
`source_log_position.entry_id=execution_id`; sequence i predecessor digest
wiążą exact retained event z przypiętym namespace. Pierwszy predecessor
to 64 zera; następne wskazują SHA256 oryginalnego poprzedniego envelope.
Bez trusted floor lub własnego qualified logu sam hash chain nie chroni
przed rollbackiem lub replay.

`executor.semantic_role` to dokładnie `ROOT_CUSTODY_EXECUTION_V1` albo
`HISTORY_CUSTODY_EXECUTION_V1`. Odpowiadają im wyłącznie custody roles
`ROOT_PROOF_SIGNING` i `HISTORY_ATTESTATION_SIGNING`. Credential źródła ma
własny namespace/material/version, różny od credential signera i CCA.
`source_log_position.namespace=executor.provider_namespace`; sequence `1`
wymaga zerowego predecessor digest, a większa sequence niezerowego.
Obecny verifier przyjmuje wyłącznie asserted `ACTIVE` source i completion
credential. Nie implementuje historycznej akceptacji `VERIFY_ONLY`/`REVOKED`
ani niezależnie uwierzytelnionego credential lifecycle; protokół tych
przyszłych stanów opisano osobno poniżej.

Legalne custody transitions to dokładnie `ACTIVE → VERIFY_ONLY`,
`ACTIVE → REVOKED` i `VERIFY_ONLY → REVOKED`; successor generation jest
predecessor generation plus jeden. Current metadata muszą być dokładnym
journal `after_metadata`, a before/after bytes muszą być canonical bytes
obiektów rzeczywiście osadzonych w journalu. Inna credential, material,
rola, key version, namespace, generation, descriptor lub późniejszy legalny
descendant odmawiają. Executor credential nie może być signer credential
ani używać tego samego public key material.

Production close ma `request_format=PostgreSQL16/jsonb-text/UTF8` i
`phase=PENDING_CLOSED`. PostgreSQL przechowuje request jako `jsonb`, więc
oryginałem jest retained `request::text` UTF8 i jego digest, a nie utracone
wire bytes klienta. Descriptor V2 i journal mają odrębne canonical bytes.
Nie canonicalizuje się na nowo requestu SQL w celu otrzymania podobnego
hasha i nie zastępuje obu oryginałów jednym digestem. Mapper porównuje
operation ID, role, SQL identity z credential identity, exact generation
i stany oraz pełne custody/DB/profile/trust/authority scope. Isolated V2
`fence_key` nie staje się przez to production fence identity.

Wszystkie Base64 są canonical RFC 4648 z paddingiem; public key ma 32 bytes,
digest 64 małe znaki hex. JSON używa istniejącej trusted serialization.
Duplicate fields, floats, NaN/infinity, bool zamiast int, błędne UTF8/Unicode,
brakujące/nieznane pola i niekanoniczne bytes odmawiają. Rozmiar, Base64
i JSON depth są ograniczone przed kosztownym dekodowaniem. PostgreSQL system
identifier jest dodatnim canonical decimal string uint64, OID mieści się
w uint32, porty w 1–65535; transaction IDs są dodatnimi canonical decimal
strings do uint64. Pozostałe dodatnie liczby całkowite mają limit
`2**53-1`, UID/GID `2**32-2`. Exact limity są stałymi nowego modułu:

| Dane | Limit original bytes |
| --- | --- |
| Envelope | 4 MiB |
| Payload | 3 MiB |
| Explicit verification policy | 1 MiB |
| Journal | 1 MiB |
| Descriptor | 256 KiB |
| Close request | 64 KiB |
| Pojedynczy before/after metadata record | 64 KiB |
| JSON depth | 16, przed recursive decoding |

Nie można powiększyć tych limitów z requestu.

## Polityka verifiera i znaczenie wyniku

Verifier porównuje source event z jawną immutable polityką obejmującą
installation, executor, database, custody scope wynikający z descriptor,
exact operation i oczekiwane log position. Polityka musi być dostarczona
odrębnie od envelope; public key
w payloadzie lub zgodny `trust_pin_sha256` nie tworzą zaufania. Wejście callera
może zawsze sporządzić spójny payload i politykę dla własnego klucza, dlatego
wynik nie potwierdza niezależnego pochodzenia oczekiwań.

Read-only policy loader callback jest punktem integracji kontrolowanym przez
użytkownika verifiera. Nie jest instalatorem trust pinów ani production
authority. Verifier bierze immutable snapshot przy wejściu i ponownie
sprawdza go przy wyjściu; brak uprawnień, niemożność odczytu, zmiana bytes,
generation, key/version, revoke lub inne odstępstwo oznaczają odmowę.
Loader nie może potwierdzić własnego zaufania przez boolean `approved`.
Zgodne snapshots wykrywają zaobserwowany drift, lecz nie zapewniają locka
lifecycle do COMMIT, odporności na niezauważony ABA ani autentycznego
provisioningu. Te własności pozostają wymaganiami przyszłej authority.

Wyraźnie oddzielamy sześć wyników:

| Właściwość | Wynik możliwy w tym PR | Czego nadal potrzeba |
| --- | --- | --- |
| Original bytes i format | Exact bounded canonical schema i pełne embedded binding | Zachowania original bytes przez rzeczywisty qualified writer/log |
| Signature matching | Ed25519 odpowiada exact public key z jawnej polityki | Niezależnego pochodzenia installed pin i role/key lifecycle |
| Source identity/authorization | Porównanie zadeklarowanej tożsamości z policy snapshot | Rzeczywistego principal, peer/executable/config i kwalifikacji uprawnień |
| Execution | Walidacja spójności podpisanej assertion z journal/descriptor | Source posiadającego cały przebieg rzeczywistej operacji i admission |
| Durability | Spójne asserted result i dostępne oryginały | Real sync, storage, crash/restart i durable log qualification |
| Publication/issuance | Zawsze `BLOCKED` | CCA, niezależnego SQL verifiera, jawnej migracji i wszystkich pozostałych gates |

Brak funkcji podpisującej i mutation entrypointu w module jest granicą
implementacyjną. Syntetyczny testowy podpis jest celowo tylko materiałem
do sprawdzenia formatu i kryptografii. Verifier nie tworzy receipt, capability,
completion decision, production `ACTIVE` fence ani wejścia BIND.

Publiczne entrypointy są następujące:

```text
parse_custody_mutation_execution(raw: bytes) -> CustodyMutationExecutionV1
parse_source_verification_policy(raw: bytes) -> ExplicitSourceVerificationPolicy
verify_custody_mutation_execution(event, *, policy_loader)
    -> CustodyMutationSourceVerification
```

Oba wejściowe dataclasses są frozen i zachowują `canonical_bytes`.
Udostępnione decoded documents są ponownie walidowane; mutable dict zwrócony
callerowi nie zmienia original bytes. Frozen report ma stałe:

```text
status = NOT_QUALIFIED
production_emission = BLOCKED
signature = MATCHED_EXPLICIT_POLICY
source_authorization = NOT_QUALIFIED
execution = NOT_ATTESTED
durability = NOT_ATTESTED
```

Polityka ma schema `cryptohunter.explicit-custody-source-verification-policy`,
version `1` i dokładnie następujące pola:

```text
schema, version, installation, executor, database, completion_authority,
operation_descriptor_b64, close, close_observation, execution_id,
source_log_position
```

`completion_authority` jest zamkniętym rekordem `authority` z completion-only
role `CUSTODY_COMPLETION_ATTESTATION_V1`; służy sprawdzeniu non-aliasing
z source i custody signer. Nie provisionuje CCA i nie potwierdza jej
istnienia. Polityka przypina exact descriptor i oczekiwaną następną pozycję,
lecz verifier nie konsumuje zdarzenia i nie zapisuje floor. Ponowne użycie
tej samej polityki może ponownie sprawdzić ten sam event; trwała replay
arbitration i advancement oczekiwań należą do przyszłego właściciela logu.

`policy_loader()` nie otrzymuje envelope, ścieżki ani klucza z requestu.
Musi zwrócić dokładny `ExplicitSourceVerificationPolicy`; `None`, błąd
odczytu, obcy typ, nieprawidłowy dokument lub inny drugi snapshot odmawiają
przez `SourceMutationVerificationError`. Successful verification wykonuje
dwa odczyty. Nie ma domyślnego loadera z produkcyjnej instalacji.
Report zachowuje także policy fingerprint i blockers dla source provenance,
trust-anchor loadera, live close, execution, durability, durable replay,
CCA i publication. Nowy raport wskazuje
`CONTRACT_ADDENDUM_INTEGRATION_PENDING`: kierunek A+C jest zatwierdzony,
lecz formalna integracja addendum pozostaje odrębnym zadaniem. Historyczne
claims S9-RPI-05 zachowują dotychczasową listę blockers.

## Izolowany controlled-source demonstrator

[`_custody_source_demo.py`](../../tests/security/_custody_source_demo.py)
i [`test_authenticated_custody_source_processes.py`](../../tests/security/test_authenticated_custody_source_processes.py)
są odrębnym testowym harnessiem. Proces Linux wykonuje **syntetyczne fixture
transition** własnego rekordu metadata, a nie istniejący production custody
writer. Wykorzystuje wyłącznie istniejącą czystą serializację
`_OperationRecord`, `_document` i `_encode_record`; nie monkeypatchuje
admission i nie produkuje `close_authorization` dla rzeczywistego custody.
Zadeklarowany close pozostaje syntetycznym, niezakwalifikowanym SQL claim.

Child ma fixed role/config i jedną operation identity/namespace; normalny
request zawiera tylko `operation_id` i `target_state`. Proces sam tworzy
descriptor, `INTENT`, metadata, `COMPLETE` i envelope z zachowanych przez
siebie bytes. Wywołuje rzeczywiste zapisy, replace, file/directory `fsync`
i `flock`; retained source envelope trafia na dysk przed odpowiedzią.
Recovery odczytuje jego exact bytes zamiast ponownego podpisania. Nie ma
polecenia podpisania przekazanego DTO lub hasha.

Tymczasowy seed Ed25519 jest wyłącznie fixture source key; katalogi mają
`0700`, seed `0400`, mutable files `0600`. Procesy fixture działają pod tym
samym UID: modes i path checks nie izolują ich od złośliwego same-UID callera.
**Ten harness nie dowodzi niezależnego DAC principal źródła.** Rzeczywista
separacja kont/pidfd istniejących signerów jest sprawdzana osobnym dotychczasowym
Linux harnessiem; nie przenosi tej gwarancji automatycznie na nowy source.

Crash cuts przed `COMPLETE`, po `COMPLETE` i po retained event, restart,
concurrent request oraz tamper/lifecycle przypadki badają kontrolowany
fixture protocol. Nie są fizyczną kwalifikacją native keyringu, qualified
production close, source independence, wcześniejszego storage sync lub CCA.
Osobne zapisy markerów testowej lineage nie tworzą atomowej transakcji
filesystem. Przerwanie między zapisem artefaktu a jego markerem może
zakończyć się odmową wymagającą rozstrzygnięcia konfliktu. Demonstrator
nie obiecuje automatycznego recovery dla każdego możliwego punktu awarii;
wykonane testy dotyczą wskazanych barier i jawnych odmów podmian.
Wire schema `PRODUCTION_LOCAL` identyfikuje format protokołu; jego użycie
w syntetycznym izolowanym teście nie ustanawia produkcyjnej authority.
Wszystkie wyniki nowego verifiera pozostają `NOT_QUALIFIED`/`BLOCKED`.

## Minimalny niezależny read-only port committed close — projekt

Obecny admin `resolve_close` nie jest niezależnym portem source/CCA.
Projektowany port powstanie dopiero po osobnym review/migracji. Ten PR nie
nadaje nikomu nowych SQL uprawnień i nie zmienia przypiętego katalogu.

Każdy source i CCA używa własnego fixed connection/service endpointu,
direct-login principal/OID i `session_user=current_user` przy wejściu.
Qualification porównuje rzeczywisty `system_user=peer:<unix_user>`, UID/GID
usługi, wszystkie `pg_hba.conf`/`pg_ident.conf` i login paths, brak memberships,
DML/DDL, ownership oraz pośredniego dostępu przez inne definers. `SET ROLE`,
nazwa roli, local socket i forwarded wynik administratora nie wystarczają.
Schema owner jest `NOLOGIN`; minimalny reviewed helper ma fixed
`search_path=pg_catalog`, statyczny SQL i exact ACL/source inventory.

Własna synchronizowana transakcja odczytu musi:

1. Zakwalifikować PostgreSQL 16 primary, durability settings, installation
   instance, system identifier, database OID/name, service/server coordinates
   i catalog digest względem własnego, zewnętrznie chronionego pina.
   Klon PGDATA może zachować system identifier/OID, więc potrzeba całego
   physical/service binding. Oczekiwanego pina nie pobiera się z badanej DB.
2. Zsynchronizować slot/fence zgodnie z istniejącym `READ COMMITTED`
   resolution protocol, po zakończeniu konkurencyjnego wcześniejszego backendu;
   odczytać exact operation/request bytes, digest, role, credential,
   predecessor generation/states, `PENDING_CLOSED` i retained `close_xid`.
3. Zweryfikować inny observation transaction ID oraz potwierdzony status
   `pg_catalog.pg_xact_status(close_xid::xid8)='committed'`. Brak możliwości
   rozstrzygnięcia statusu, overflow/niepoprawny XID, niekwalifikowany backend
   lub niezgodny request odmawiają. Samo istnienie rekordu nie dowodzi close
   w odrębnej zatwierdzonej transakcji.
4. Zachować własne reader role/OID, observation XID i qualified catalog
   digest w `close_observation`; zwrócić oryginały, nie administrator hash.
   `NOT_FOUND`, timeout i disconnect zachowują nieznany outcome.

Read-only oznacza brak semantycznych zapisów authority; nie wyklucza
wymaganych synchronizacyjnych locks. Ich dokładny SQL i isolation protocol
wymagają review. Transaction read ends przed trwałą lokalną mutation;
DB-first close trwale utrzymuje zamknięty fence. Source następnie porównuje
exact predecessor pod custody lock. CCA wykonuje własny krótki finalny
re-read przy utrzymanym source lease i własnym decision-log lock.
Nie trzyma się transakcji administratora podczas filesystem mutation.

## Odpowiedzialny source, trwałość i recovery — protokół docelowy

Source nie udostępnia `sign(hash)`, `sign(payload)` ani endpointu
przyjmującego caller-supplied before/after/`COMPLETE` jako dowód wykonania.
Request wskazuje wyłącznie exact dopuszczoną operację. Zainstalowany proces
sam rozwiązuje fixed scope, własny qualified close read, credential i stan.
Administrator SQL nie posiada source key, write access do source logu
ani prawa zmienić jego konfiguracji.

1. Administrator potwierdza osobny COMMIT `PENDING_CLOSED`; unknown outcome
   rozstrzyga przez exact read, bez nowej mutation lub otwarcia fence.
2. Source kwalifikuje własny close, sprawdza niezmieniony lifecycle własnego
   execution credential i bierze exclusive custody lease. Sprawdza role,
   original descriptor, rzeczywisty before, generation i fixed namespace.
3. Pod lockiem zapisuje immutable `INTENT` z oboma oryginałami operacji
   i exact before/after; synchronizuje plik oraz właściwe katalogi.
   Existing operation ID z innymi bytes to konflikt, bez nadpisania.
4. Wykonuje jednokierunkową mutation istniejącą trusted custody ścieżką.
   Synchronizuje metadata i katalog, zapisuje exact `COMPLETE`, synchronizuje
   journal i katalog, ponownie sprawdza otwarte obiekty i after state.
   Nie odczytuje signer secret i nie generuje Root Proof/history signature.
5. Ten sam wykonawca tworzy source event ze swoich retained originals,
   własnego odczytu DB i potwierdzonych kroków wykonania. Source key eligibility
   pozostaje pod własnym lifecycle/log serialization do durable event.
   Podpis i envelope powstają raz; ich oryginał trafia do role-scoped logu,
   wraz z exact operation uniqueness i monotonic position. Po sync logu
   oraz wymaganych directory/linkage records wolno zwrócić authenticated ack.
6. Exact replay zwraca byte-identical retained envelope, bez ponownej mutation
   lub podpisu. Ten sam operation ID z innym descriptor, powtórzony event ID,
   zła sequence/predecessor, cross-role substitution lub niezgodny after
   daje trwały konflikt. Brak eventu po `COMPLETE` wymaga authoritative source
   recovery, nie dołączenia podpisu do cudzych zastanych plików.
7. CCA sprawdza source envelope, własny installed pin i peer/source provenance,
   obydwa oryginały operacji, retained event i własny qualified close.
   Source lease utrzymuje exact after do końca decyzji CCA. CCA zapisuje własną
   immutable decyzję i original signed evidence we własnym durable logu;
   dopiero po sync zwraca authenticated ack. CCA runtime pozostaje przyszły.

Proponowany lock order z S9-RPI-05 to: committed close bez otwartej
admin transaction → source custody lock → CCA decision-log lock → krótki
qualified DB read/slot lock → zakończenie DB read → durable CCA decision/ack
→ zwolnienie locks. Source log i source/CCA credential lifecycle muszą mieć
zgodny osobno reviewed order; żadna rotacja nie może brać tych locks odwrotnie.
Ta lista nie jest nową implementacją globalnej transakcji.

`ACTIVE` source key może uwierzytelniać nowe kwalifikowane wykonanie.
`VERIFY_ONLY` służy weryfikacji lub exact return uprzednio retained bytes;
nie tworzy nowego eventu. `REVOKED` jest terminalny. Sam dawny podpis nie
potwierdza historycznej eligibility po kompromitacji; potrzebna jest
zachowana niezależnie authenticated history/floor. Rotacja zachowuje stare
keys, epoch, generations i exact event linkage; nie przepisuje namespace
ani starych bytes. Source key i signer key mają odrębne lifecycle.

Hash chain, directory `fsync` i restart jednego procesu nie zapewniają
odporności na skoordynowany rollback całego hosta. Trusted log floor i pin
mają własny model retencji. W `PRODUCTION_LOCAL` nie deklarujemy takiej
odporności ani ochrony przed host root. Test process crash nie jest dowodem
zachowania dysku/cache podczas zaniku zasilania.

## Przypadki awarii i granice dowodu

Poniżej opisano wymagane zachowanie. Wykonanie testu formatu lub istniejącej
regresji nie oznacza wykonania jeszcze nieistniejącego production runtime.

| Przypadek | Wymagany wynik i recovery |
| --- | --- |
| Nieuprawnione podpisanie arbitrary hash/DTO | Brak takiego production API; kontrolowany wykonawca sam wyprowadza event z operacji. Poprawny obcy podpis/policy nie daje production qualification |
| Root/History substitution | Inny role-scoped executor/credential/material/namespace/custody lub endpoint odmawia, również przy identycznym końcowym stanie |
| Credential/key version/pin substitution | Odmowa exact policy binding; key z payloadu, requestu lub badanej DB nie jest anchor |
| Brak uprawnień lub odczytu trust anchor | Odmowa, bez default key, TOFU, re-pin, pliku zastępczego lub kontynuacji z cache |
| Replay wcześniejszej operacji | Exact same operation i retained log bytes mogą być historycznym lookup; inna operacja/position/generation odmawia. Stateless signature check sam nie zapewnia replay arbitration |
| Ten sam operation ID, inny descriptor | Immutable conflict; zero last-write-wins, nowego eventu i nowej mutation |
| Crash przed `INTENT` lub mutation | Close pozostaje zamknięty; brak pozytywnego eventu. Exact resolution istniejącej operacji |
| Crash po mutation, przed trwałym `COMPLETE` | `INTENT` i after-looking metadata nie są completion. Brak pozytywnego eventu; qualified recovery dokładnego protocol phase |
| Crash po `COMPLETE`, przed source event | Brak authenticated execution event. Recovery wymaga własnego retained execution lineage; sam późniejszy odczyt plików nie upoważnia do podpisu |
| Crash po podpisie, przed durable source log | Brak ack i nieznany outcome; recovery rozstrzyga retained bytes/phase, nie podpisuje na nowo na podstawie samej nieobecności pliku |
| Crash po durable source event, przed odpowiedzią | Exact lookup zwraca oryginalny envelope po ponownej qualification; bez execution, nowej sequence lub podpisu |
| Restart source/DB/CCA | Odczyt trwałych original logs i aktualnych pins, exact binding i qualification; process cache/DTO nie zastępują authoritative state |
| Równoczesne operacje | Custody lease, operation uniqueness i source log/lifecycle serialization dają jeden exact event albo trwały konflikt; CCA ma osobne durable decision uniqueness |
| Rotation/revoke podczas wykonania | Eligibility lock/CAS do durable source event, final recheck i history linkage. Stara generation albo revoke przed decyzją blokują nowy event |
| Journal/inode/symlink/directory swap | Protected open descriptors, ownership/ACL/ancestry i rechecks pod lease; odmowa zmiany przy wejściu i wyjściu, bez nowej mutation lub signing |
| Brak qualified committed PostgreSQL close | Odmowa; `NOT_FOUND` jest unknown, forwarded admin DTO lub close w tej samej transakcji nie wystarczają |
| Bezpośrednie SQL completion lub production commit | Dotychczasowe Python refusals i SQL V2 tombstone/ACL odmawiają nawet przy poprawnym podpisie i lokalnym raporcie |

## HARDWARE_DEPENDENCY

**Do następnego kroku projektowego i izolowanego demonstratora nie jest
potrzebny drugi fizyczny komputer, HSM ani TPM.** Wystarczy jeden Linux host
lub izolowane CI z rzeczywistymi procesami, odpowiednimi uprawnieniami do
disposable UID/GID, lokalnych Unix sockets i izolowanego PostgreSQL 16.
Mock procesu, uprawnień lub filesystem nie potwierdza odpowiadającej mu
właściwości. Brak wymaganych capabilities należy zgłosić jako niewykonany
zakres, bez ukrytego skip/xfail i bez nazywania go PASS.

Przyszła fizyczna kwalifikacja source/CCA wymaga natywnego Linux z prawdziwym
systemd/service hardening, namespaces, DAC/ACL i independently installed pins.
Ponowne użycie obecnego pidfd transportu wymaga działającego Linux 6.5+
`SO_PEERPIDFD`; sam stały numer opcji lub `pidfd_open(pid)` nie zastępują
socket-associated identity. Native SecretService/keyring, bus/session,
repository HWID envelope, dostęp z private network namespace i stabilność
restartów trzeba sprawdzić dla każdej rzeczywistej roli. Software Ed25519
pozostaje dopuszczalny w `PRODUCTION_LOCAL`.

Osobna maszyna, zewnętrzny independently protected checkpoint lub urządzenie
kryptograficzne stają się potrzebne wtedy, gdy wybrany silniejszy model
wymaga granicy niezależnej od wspólnego host root, ekstrakcji software keys
lub skoordynowanego restore wszystkich lokalnych logów/DB/pinów. HSM sam
nie zapewnia anti-rollback logu ani autentyczności input execution.
`PRODUCTION_SERVER_READY` wymaga rzeczywiście niezależnych security domains;
zmiana nazwy profilu i zakup drugiego komputera bez określonego protocolu
tego nie dowodzą. Nie rekomendujemy zakupu sprzętu bez takiego wymagania.

## Walidacja i warunki przejścia do CCA

Potwierdzone przez wykonany harness w tej sesji regresje bazowej granicy:

| Kontrola | Potwierdzony wynik | Znaczenie |
| --- | --- | --- |
| Existing production lifecycle PostgreSQL 16 | 136 PASS, 1 istniejący platform-specific skip | SQL/schema/qualification/identity/race boundary; nie nowy source reader, executor ani CCA |
| Existing root Linux signer process/DAC/peer harness | 6 PASS | Rzeczywiste testowe procesy i identity checks istniejącego transportu; nie installed native-keyring/source qualification |
| Existing non-PostgreSQL security regressions | 678 PASS, 10 istniejących Windows-only skips | Dotychczasowe signer/IPC/custody/evidence/admission; brak wykonania natywnych Windows cases na Linux |
| Frozen contracts i production refusal regressions | 397 PASS | Zachowanie frozen/status i odmów; bez nadania nowej production capability |
| Existing isolated custody lifecycle PostgreSQL 16 | 31 PASS | Rzeczywisty journal, COMMIT response loss, restart, recovery i races istniejącego isolated coordinatora |
| Nowy source protocol | 174 PASS | Zamknięty format, oryginały, podpisy i niezależne oczekiwania; raport nadal `NOT_QUALIFIED` |
| Nowy controlled TEST process harness | 38 PASS | Rzeczywiste procesy, crash/restart, exact event recovery, flock, własna signed lineage i odmowy podmian |
| Exact CI inventory i platform contracts | 39 PASS | Additive test/format/coverage inventories i dotychczasowe granice platform |
| Ruff oraz pełny `ruff-format --all-files` | PASS | Pełne skonfigurowane zakresy, bez zmian formattera |
| Configured mypy Linux i Win32 | PASS, po 340 plików | Bez filename overrides i bez zawężenia konfiguracji |
| Semgrep custom rules i ich fixtures | 0 findings, 5/5 fixtures PASS | Istniejący blocking scope i dodatkowy odczyt nowego modułu |
| Betterleaks 1.9.0 offline | 0 nowych leaks | Wszystkie 3750 tracked/staged ścieżki w snapshot, bez scratch `work/`; niezmieniona baseline |

Dokładne polecenia, nowe protocol/demonstrator tests, coverage, Ruff,
`python -m pre_commit run ruff-format --all-files`, skonfigurowany mypy
Linux/Win32, Semgrep, secret scanning i contract/inventory checks oraz ich
rzeczywiste wyniki należą do opisu PR. Coverage nowego modułu wynosi 95%
według łącznego pomiaru linii i gałęzi. Zmiany CI ograniczają się do
additive inventories nowych modułów/testów, w tym odpowiadających im kontroli
inwentarza. Istniejące timeouts, test assertions, próg coverage 70%, frozen
JSON, canonical statusy i production gates pozostają bez zmian.
Pełne CI pozostaje `FULL_CI_PENDING`, dopóki jego zakończenie nie jest
potwierdzone; nie czekamy na wielogodzinny suite i nie wykonujemy auto-merge.

Przed rzeczywistą CCA wymagane są łącznie:

1. Osobne review exact source protocol, trust/pin provenance, odpowiedzialnego
   executora, metadata/secret permission split, lifecycle i durable recovery;
   pełny source event nie może pochodzić z ogólnej usługi podpisującej.
2. Jawny provisioning trzech rozłącznych authority/credentials/namespaces,
   fizyczna kwalifikacja OS/keyring/process/peer i odmowy cross-role.
3. Reviewed minimalne independent committed-close read API, jawna migracja
   z exact approved catalogu i osobno zatwierdzony external pin; żadnego
   runtime upgrade, automatycznego re-pin ani adaptacji testowego admission.
4. Rzeczywisty source event log i CCA decision log z original-byte retention,
   exact recovery, lifecycle interlocks, crash cuts, restartem, concurrency,
   response loss, rotation/revoke i jasno ograniczoną rollback gwarancją.
5. Weryfikacja authenticated evidence także na SQL publication boundary,
   odporna na bezpośrednie wywołanie przez admina/issuera. Dzisiejszy poprawny
   wynik publication pozostaje odmową tombstone.
6. Formalna versioned integration A+C i niezależne issuer ingress,
   private-control/trust activation, requester/claimant fences oraz wszystkie
   CHA/FreshnessAuthority/CheckpointAuthority/AccountGenesis predicates.
   Custody retirement completion nie jest issuer COMMITTED ani account closure.
7. Odrębny przegląd i jawna decyzja GO/NO-GO rzeczywistego deploymentu.
   Pozytywny podpis, demonstrator, merge lub zakończone CI nie dają GO.

Production `commit()`, administracyjna publication, SQL `complete_close`,
SIGN/BIND/enrollment/activation i Stage 10 pozostają zablokowane.
