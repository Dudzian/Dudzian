# Stage 9: PostgreSQL V2 Root Proof preparation

Ten dokument opisuje odrębne, wewnętrzne adaptery przygotowania po migracji
`ATOMIC_ROOT_PROOF_AUTHORITY_V1_COMPONENT_INVENTORY_V2`. Nie zmienia zamrożonych
kontraktów M0.5, canonical Root Proof ani signed immutable attempt. Granica to:

`RESERVED → STAGED → qualified approval → atomic final COMMIT`.

`RESERVED` nie jest `COMMITTED`; `STAGED` nie jest `ISSUED`. Podpis i approval
nie wykonują BIND. Publiczny produkcyjny RootProofIssuer pozostaje niedostępny:

```text
ATOMIC_ISSUER_READY=NOT_READY
PRODUCTION_LIFECYCLE_ISOLATION=BLOCKED
ROOT_PROOF_ISSUANCE_AVAILABLE=FALSE
```

## Audyt rzeczywistych źródeł

Sprawdzono wszystkie jedenaście źródeł wskazanych w zadaniu:

- `bot_core/postgresql_root_proof_signing_intent.py` oraz
  `bot_core/postgresql_root_proof_signing_intent_schema.py`;
- `bot_core/postgresql_root_proof_retention.py` oraz
  `bot_core/postgresql_root_proof_retention_schema.py`;
- `bot_core/postgresql_root_proof_issuance_authority.py`;
- `bot_core/postgresql_atomic_root_proof_commit.py`;
- `bot_core/postgresql_atomic_root_proof_migration.py`;
- `bot_core/postgresql_atomic_root_proof_qualification.py`;
- `bot_core/postgresql_custody_lifecycle.py`;
- `bot_core/local_signing_custody.py`;
- `bot_core/root_proof_atomic_composition.py`.

ACL i zmiany inventory potwierdzono także w
`postgresql_atomic_root_proof_schema.py`, `postgresql_entitlement_registry.py`,
`postgresql_preaccount_credentials.py`, `postgresql_issuer_history_schema.py`
oraz `postgresql_custody_lifecycle_schema.py`. Codebase Memory nie jest dostępne
w tym środowisku. Ustalenia wynikają z celowanych odczytów źródeł, a nie z
założenia kompletności indeksu.

Migracja zachowuje źródła pierwotnych funkcji i aplikacyjne bajty. Nie występuje
ogólna zmiana original canonical payload ani dowolna zamiana signera. Dodaje
przejrzane funkcje, granty ownerów i deferred constraints; odbiera standalone
`append_bind`, `append_record` oraz `retain_head`. Dokładny legacy allowlist
celowo tego nie akceptuje. Usunięcie wywołań `qualify_*()` zgubiłoby dowód
inventory, ACL i direct-login identity, zamiast ustanowić kwalifikację V2.

W tabeli nazwy ról oznaczają pola autorytatywnych metadanych danego schematu.
`intent.metadata.issuer_writer_role` jest preparation principal i jest odrębny
od `atomic.provisioning.issuer_writer_role`, który wykonuje finalny commit.

| Legacy entrypoint | Rola SQL i uprawnienia | Powód odmowy po V2 | Nowy kwalifikowany adapter | Operacja SQL | Warunek bezpieczeństwa | Wymagane regresje |
| --- | --- | --- | --- | --- | --- | --- |
| `PostgreSQLRootProofSigningIntent.__init__`, `read_intent`, `reserve` | Direct-login `intent.metadata.issuer_writer_role`; `USAGE`, odczyt metadata, `EXECUTE read_intent` i `reserve_intent`; bez table DML | Legacy `_qualify` odrzuca dodatkowy schema `USAGE` coordinator ownera; `_qualify_functions` odrzuca dodatkowe `atomic_metadata`, `atomic_read_intent` i `atomic_intent_member`. Konstruktor dodatkowo rekwalifikuje legacy public authority. | `PostgreSQLRootProofSigningIntentV2` | Istniejące `read_intent(bytea,bytea)` i `reserve_intent(bytea,bytea,bytea,bytea,bytea,bytea,bytea)` | Genuine INITIAL_BINDING i retained signed attempt, dokładne public facts/request bytes, oryginalny enrolled ACTIVE snapshot i fizyczna baza; preparation nie może BIND | Nowy RESERVED po migracji, byte-identical retry, konflikt identity/generation/request, unknown write, crash przed STAGED, legacy rejection |
| `PostgreSQLRootProofRetention.__init__`, `read_prepared`, `retain_prepared` | Direct-login `retention.metadata.runtime_role`; `USAGE`, odczyt metadata, `EXECUTE read_prepared` i `retain_prepared`; bez table DML | Legacy `_qualify` odrzuca dodatkowy schema `USAGE`; `_qualify_functions` odrzuca `atomic_metadata`, `atomic_read_retention` i `atomic_retention_member`. Samo DTO authorization nie dowodzi genuine RESERVED. | `PostgreSQLRootProofRetentionV2` | Istniejące `read_prepared(bytea,bytea)` i `retain_prepared(bytea,bytea,bytea,bytea,bytea,bytea,bytea,bytea)` | Autorytatywny genuine RESERVED, dokładny Root Proof signature i linkage; snapshot/public key/generation pochodzą z oryginalnego RESERVED | Nowy STAGED, identyczne signature/payload/envelope przy retry, odrzucenie zmiany każdego z tych bajtów i innego ACTIVE signera, brak BIND/history/decision |
| `PostgreSQLRootProofIssuanceAuthority.requalify` i ponowny genuine preflight | Trzy oddzielne runtime loginy entitlement/requester/claimant; publiczne odczyty; requester/claimant bez podpisywania | `requalify()` wywołuje exact legacy qualifiers wszystkich trzech portów. Entitlement ma nowy schema ACL, helpers i deferred trigger, a grant `append_bind` został odwołany. Requester/claimant mają dodatkowy schema `USAGE` coordinator ownera i więcej funkcji niż jedyne legacy `apply(jsonb)`. | `PostgreSQLRootProofIssuanceAuthorityV2` z kwalifikowanymi portami publicznymi V2 | Istniejące publiczne odczyty entitlement i credentials | Scope, genuine binding/attempt identity, requester/claimant signatures, ACTIVE public credential generation/revision/material i zakaz aliasów; brak nowego prawa podpisywania lub BIND | Utworzenie genuine attempt po migracji, ponowna weryfikacja request bytes, public credential drift/close, wrong principal, rollback pozostałych autorytetów |
| `prepare_atomic_root_proof_commit` | Odczyty sześciu genuine legacy portów; funkcja Python nie wykonuje SQL BIND ani podpisu | Przed zmianą wymagał exact legacy typów i wywoływał ich rekwalifikację. Odczyt history też odmawia dodatkowych ACL, helper inventory i deferred link triggers; standalone history writes są odwołane. | Ten sam `prepare_atomic_root_proof_commit` z exact zgodnym zestawem portów V2 i `PostgreSQLAuthenticatedIssuerHistoryV2` | Tylko odczyt exact RESERVED/STAGED, entitlement predecessor i current history record | Przypięty V2 deployment, ponowna genuine weryfikacja i oryginalna projekcja history event; mieszane V1/V2 porty i detached DTO nie są capability | Post-migration STAGED rzeczywiście staje się genuine preparation, bez dopisania history; nowe preparation trafia do istniejącego approval i commit |
| `PostgreSQLAuthenticatedIssuerHistory` użyty przez legacy preparation | Direct-login `history.metadata.runtime_role`; read APIs pozostają SQL; `append_record` i `retain_head` tracą grant | Legacy kwalifikator odrzuca dodatkowy schema ACL, deferred triggers, fingerprint fizyczny i nowe function inventory/ACL | `PostgreSQLAuthenticatedIssuerHistoryV2` | Istniejące odczytowe history APIs | Odczyt nie nadaje runtime prawa do append record/head; niezmieniona history payload projection | Odczyt predecessor po migracji, brak standalone history writes, exact immutable replay po trwałym close |
| `PostgreSQLAtomicRootProofCommit.approve_prepared`, `commit`, `reconcile` | Oddzielny `test_admin_role` zatwierdza package; `atomic.issuer_writer_role` ma wyłącznie closed commit/read APIs i metadata | Te API nie odrzucają prawidłowego inventory V2: mają osobny administracyjny manifest. Dotychczas brakowało legalnego nowego Python preparation po migracji. | Istniejący `PostgreSQLAtomicRootProofCommit` | `approve_commit(jsonb)`, następnie wyłącznie `commit_root_proof(bytea)`; recovery `read_decision(bytea,bytea)` | Genuine registered preparation, exact protected approval, jeden kwalifikowany physical SERIALIZABLE transaction, enrollment i live fences | Nowy V2 STAGED → approval → COMMIT; exact replay bez podpisu; CAS/close/ACL refusal i pełny rollback |

Nie każde z jedenastu źródeł jest legacy API kwalifikującym schema. Poniższe
rozróżnienie zapobiega przypisaniu odmowy niewłaściwej warstwie:

- `postgresql_root_proof_signing_intent_schema.py` i
  `postgresql_root_proof_retention_schema.py` zawierają legacy closed inventory;
  ich kwalifikatory pozostają bez wyjątku dla V2.
- `postgresql_atomic_root_proof_migration.py` wymaga genuine legacy
  prequalification przed instalacją, porównuje original sources i application
  bytes, a następnie kwalifikuje nowy manifest V2. Nie jest runtime repair.
- `postgresql_atomic_root_proof_qualification.py` akceptuje jawnie
  administrowany atomic V1 / inventory V2, przypinając physical identity,
  źródła, owners, role OIDs oraz live catalog. Nie tworzy genuine attempt ani
  signing preparation.
- `postgresql_custody_lifecycle.py` ma oddzielną kwalifikację Durable Lifecycle
  V2 i Signer Enrollment SQL V3. Nie używa signing-intent legacy qualifier;
  DB-first close i exact enrolled fence pozostają autorytatywne.
- `local_signing_custody.py` nie kwalifikuje inventory PostgreSQL. Sprawdza
  autentyczny lokalny credential i lifecycle pod custody lock. Jego ACTIVE
  podpis nie jest pozwoleniem na DB BIND. Legacy Custody Isolation V1 zabrania
  lokalnego obejścia durable administration dla root/history custody.
- `root_proof_atomic_composition.py` jest odczytową diagnostyką legacy graph.
  Zachowuje `NOT_READY` i `authorizes_operations=False`; jego exact legacy type
  checks i implementacyjne production blockers nie są nową capability V2.

## Kwalifikacja i SQL separation

Adaptery mają oddzielne nazwy V2 i wymagają administracyjnie zakwalifikowanego
deploymentu. Nie ma `trust_v2`, `skip_qualification`, bypassu środowiskowego
ani wyboru nowego manifestu przez runtime. Każda operacja ponawia live
qualification przed autorytatywnym odczytem lub zapisem.

Wymagane dowody obejmują PostgreSQL 16 primary z `fsync`,
`synchronous_commit` i `full_page_writes`, physical database identity,
reviewed schema/function source inventory, exact owners i role OIDs, brak
memberships, ACL, `SECURITY DEFINER` z `search_path=pg_catalog`, live catalog
fingerprint oraz autentyczny enrollment. Zgodny DSN nie zastępuje physical
identity. Dane mutable, takie jak nowe RESERVED/STAGED lub decision, nie
wybierają nowego katalogowego pin.

Publiczne odczyty entitlement i credentials V2 kwalifikują dokładnie to
połączenie i tę transakcję `REPEATABLE READ READ ONLY`, która następnie
odczytuje dane. Każde wywołanie otwiera nowe połączenie i wykonuje pełną live
qualification przed pierwszym zapytaniem o publiczne fakty. Preparation
fingerprint obejmuje wszystkie osiem schematów w jednym pełnym przebiegu;
oryginalnych siedem schematów i globalne role/ACL nie są pomijane.

Oryginalne preparation SQL pozostaje pod swoimi direct-login guards. Nowy
kwalifikator nie wykonuje go jako atomic writer, przez `SET ROLE` ani przez
bootstrap. `migrate_root_proof_preparation_v2()` instaluje jawny kontrakt
`ROOT_PROOF_RESERVED_STAGED_PREPARATION_ACL_V2` po Durable Custody Lifecycle V2
i Signer Enrollment SQL V3. Tworzy osobny bezpiecznie nazwany preparation
schema należący do istniejącego atomic `NOLOGIN` ownera i dokładnie dwie
funkcje `SECURITY DEFINER`:

- `qualify_preparation_v2()` odczytuje physical identity, wymaga PostgreSQL 16
  i weryfikuje istniejący manifest oraz enrollment V3. `EXECUTE` dostają
  wyłącznie sześć dotychczasowych component runtime/preparation loginów;
- `lock_active_signer_v2(bytea)` porównuje oryginalny snapshot z exact enrolled
  ACTIVE root signerem i trzyma fence `FOR SHARE` do końca transakcji.
  `EXECUTE` dostają tylko intent oraz retention preparation loginy.

Nie ma szerokiego grantu `pg_control_system()` dla preparation. Migracja
przypina reviewed sources i ACL facade oraz jawnie aktualizuje atomic/lifecycle
catalog pins dla dodanych zewnętrznych grantów. Sprawdza zachowanie application
i lifecycle bytes. Osobna tabela `preparation.metadata` utrwala migration ID,
`ISOLATED_TEST_ONLY` oraz dokładny `predecessor_catalog_sha256`. `PRIMARY KEY`,
singleton `CHECK` i immutable triggers `ENABLE ALWAYS` chronią tę metadatę;
preparation principals nie dostają table SELECT/DML. Wąska facade zwraca
wyłącznie physical coordinates i dokładną metadatę do ponownej kwalifikacji.
Migracja zwraca publiczny, pozbawiony credentials
`PostgreSQLRootProofPreparationV2Deployment` i aktualny osobny lifecycle
deployment dla commit composition. `qualify_root_proof_preparation_v2()`
weryfikuje direct principal, role OIDs, closed privileges, reviewed sources,
live fingerprint i physical identity. Nie rozszerza issuer writera.

| Principal | Legalne capability po migration V2 preparation | Finalne prawa emisji |
| --- | --- | --- |
| `intent.metadata.issuer_writer_role` | Original `read_intent`/`reserve_intent`, `qualify_preparation_v2()`, `lock_active_signer_v2(bytea)` | Brak approval, BIND, history writes, decision i checkpoint |
| `retention.metadata.runtime_role` | Original `read_prepared`/`retain_prepared`, `qualify_preparation_v2()`, `lock_active_signer_v2(bytea)` | Brak approval, BIND, history writes, decision i checkpoint |
| Entitlement/requester/claimant runtime roles | Original publiczne odczyty i `qualify_preparation_v2()` | Brak podpisywania, BIND i finalnego commitu; V2 public ports mają `compare_and_swap=False` |
| `history.metadata.runtime_role` | Original odczyty history i `qualify_preparation_v2()` | Brak standalone append record/head i finalnego commitu |
| `atomic.provisioning.test_admin_role` | Existing qualified approval i isolated-test administration | Oddzielny verifier; brak prawa do finalnego `commit_root_proof` |
| `atomic.provisioning.issuer_writer_role` | Existing `commit_root_proof`, exact decision/approval reads i odczytowa qualification | Jedyny direct-login final commit principal; bez nowych preparation grantów, table DML lub arbitrary evidence ingress |
| Semantic schema owners | Existing `NOLOGIN` ownership i fixed definer helper execution | Niedostępne jako runtime login; brak role membership do przejęcia ownership |

Przygotowanie nie uzyskuje `EXECUTE commit_root_proof`, `approve_commit`,
`atomic_bind`, `atomic_append_record`, `atomic_retain_head`, standalone
`append_bind`, `append_record` ani `retain_head`. Nie uzyskuje table/column
DML, DDL, ownership, role membership ani nieprzejrzanych external definers.
Sam bootstrap jest użyty wyłącznie przy jawnej migracji/rekonstrukcji i nie
jest przechowywany w adapterach runtime.

## Genuine authorization V2 i recovery po restarcie

Jawny `resolve_root_proof_issuance_authorization_v2(binding, authority)` w
`cha_root_proof_attempt_reservation.py` przyjmuje wyłącznie exact
`PostgreSQLRootProofIssuanceAuthorityV2` z trzech kwalifikowanych component
connections, wspólnego publicznego preparation pin i przypiętego entitlement
subject. Tworzy istniejący genuine `VerifiedRootProofIssuanceAuthorization`,
a każde użycie ponownie odczytuje autentyczny retained INITIAL_BINDING,
entitlement/public credential facts i canonical evidence z tego samego
przypiętego providera. Default legacy resolver i production selector zachowują
swoją dotychczasową, zablokowaną granicę; nie ma env switch ani konfiguracji
odblokowującej produkcyjnego RootProofIssuer.

Read-only public capability V2 ma `compare_and_swap=False`; jej canonical
authorization evidence różni się od legacy capability z prawem BIND.
Dlatego nowa post-migration próba musi rzeczywiście uzyskać nową genuine
authorization V2, zostać zarezerwowana i podpisana przez dotychczasowy signed
attempt pipeline. Nie wolno przepisać ani reinterpretować immutable signed
attempt bytes przygotowanych dla innego authority evidence. Zamrożony request
schema, lineage i exact retry identity pozostają bez zmian.

Genuine signed-attempt preflight najpierw wywołuje
`_exact_binding_authorization()`, a następnie przekazuje świeżo zweryfikowany
snapshot do prywatnego `_qualified_pair_from_snapshot()` w tym samym
wywołaniu. Zachowuje custody identity, public key/material, podpisy
requester/claimant, zakaz aliasów oraz dokładny retained signed request.
Publiczne capability reads nadal ponawiają live source guard. Snapshot nie
staje się persistent cache ani upoważnieniem dla kolejnego wywołania;
nie zastępuje późniejszych `_verify_request()` i precommit `_inputs()`.

`requalify_root_proof_preparation_v2(bootstrap, lifecycle_deployment, *,
preparation_schema, connections)` odtwarza publiczny pin po restarcie procesu
w jawnej administracyjnej transakcji `SERIALIZABLE READ ONLY`. Najpierw
wymaga zakwalifikowanego existing lifecycle deployment; po jego utracie należy
użyć istniejącej osobnej lifecycle reconstruction. Sprawdza oryginalne
compiled V2/V3 functions, immutable database manifest, durable preparation
metadata i predecessor hash, pełny schema/table/index/constraint/trigger/function
inventory, role OIDs i closed ACL. Następnie reconnectuje każdy component
przez jego rzeczywisty direct login. Nie tworzy ani nie naprawia obiektów,
nie podpisuje, nie zatwierdza package i nie wywołuje finalnego commitu.

Po unknown preparation write i restarcie reconstructowany pin pozwala
odtworzyć genuine authorization dla tego samego retained binding/attempt i
odczytać `.read_intent()` lub `.read_prepared()` dla dokładnego oryginalnego
request. Exact retry może zwrócić wyłącznie oryginalne retained bytes.
Absence z nowego odczytu nie rozstrzyga samodzielnie wyniku wcześniejszego
nieznanego COMMIT; nie zezwala na automatyczny replacement ani zmieniony request.

## Oryginalne bajty i rozstrzygnięcie

`PostgreSQLRootProofSigningIntentV2.reserve()` odczytuje genuine INITIAL_BINDING
oraz signed immutable attempt z rzeczywistego źródła. Odtwarza exact request,
requester/claimant facts, entitlement identity/generation i scope. Pod SQL
attempt lock najpierw szuka retained RESERVED. Exact retry zwraca ten sam
root ID, payload, envelope i credential snapshot. Zmieniony request dla tego
samego attempt odmawia. Dla nowego wpisu Root Proof UUID7 jest mintowany raz
przez issuera; canonical payload pozostaje w zamrożonym schema.

`PostgreSQLRootProofRetentionV2.retain_prepared()` odczytuje genuine RESERVED
przez zakwalifikowany adapter i sprawdza oryginalny signature nad
`ROOT_PROOF_SIGNING_DOMAIN + canonical_payload`. STAGED zachowuje ten sam
root ID, request, authorization, payload i credential snapshot. Snapshot,
public key i lifecycle generation pochodzą z oryginalnego RESERVED. Aktualny
inny ACTIVE signer nie może go zastąpić. Jeśli STAGED już istnieje, sprawdzane
są wszystkie original bytes; retry nie podpisuje ani nie generuje nowego
envelope. Dostarczone sprzeczne envelope jest odmową.

Zapis RESERVED i zapis STAGED są oddzielnymi transakcjami preparation.
Zakończenie każdej z nich nie dopisuje entitlement BOUND, authenticated issuer
history, head, decision ani checkpoint. Awaria pomiędzy RESERVED a STAGED
pozostawia sam RESERVED. Awaria po STAGED i przed approval pozostawia
oryginalnego kandydata. Unknown result nie jest sukcesem ani dowodem braku
zapisu; nowy kwalifikowany odczyt i exact retry służą odzyskaniu original bytes.
Nie wolno po utracie odpowiedzi automatycznie utworzyć replacement attempt.

Przygotowanie pakietu commit ponownie weryfikuje genuine provenance i exact
RESERVED/STAGED. Buduje dotychczasowy history successor bez zapisu. Oddzielny
kwalifikowany test verifier sprawdza pakiet i autentyczny signed history head,
a następnie zapisuje immutable approval. W tej fazie entitlement nadal
pozostaje UNBOUND, history nie zawiera issuance record/head, a decision nie
istnieje.

Wyłącznie istniejący `commit_root_proof(bytea)` wykonuje finalny BIND oraz
history record, signed head i decision w jednej fizycznej transakcji
PostgreSQL `SERIALIZABLE`. Odczytuje protected approval i exact original
RESERVED/STAGED, blokuje public credential predecessors, entitlement lineage
oraz enrolled signer/trust fences i sprawdza history predecessor CAS.
Deferred link constraints wymagają jednej zgodnej decyzji. Odmowa wycofuje
cały finalny write set; nie wykonuje kompensującego UNBIND. Checkpoint,
AccountGenesis i FreshnessAuthority nie należą do tej transakcji.

## Lifecycle i immutable history

Nowa preparation wymaga enrolled ACTIVE root signer o dokładnym credential,
public key, key version i generation. Brak enrollment, niezgodny signer,
`PENDING_CLOSED`, `VERIFY_ONLY` i `REVOKED` odmawiają nowej emisji. Durable
Lifecycle V2 utrwala `PENDING_CLOSED` w PostgreSQL przed mutacją custody;
Signer Enrollment SQL V3 wiąże fence z oryginalnym local credential i material.
Po close nie wolno odtworzyć nowego signera z bieżącego stanu dla starego STAGED.

Odmowa nowej emisji nie usuwa starej decyzji. `commit`/`reconcile` dla
attempt z już trwale zachowaną decision `COMMITTED` zwracają immutable original decision i exact
payload/signature/envelope bez nowego podpisu i bez nowej historii. Lokalny
`verify_historical()` odmawia ustanawiania nowego zaufania przez REVOKED
credential; nie wolno zmieniać tej reguły, aby wykonać DB replay. Pozytywna,
już istniejąca decision zachowuje własne niezmienne powiązanie z dowodem.

## Kardynalność i dokładna późniejsza migracja

Ten PR zachowuje wszystkie UNIQUE. Intent ma globalne `UNIQUE(attempt_key)`,
`UNIQUE(root_proof_key)`, `UNIQUE(operation_key)`, `UNIQUE(entitlement_key)`
oraz `UNIQUE(account_key)`. Retention ma globalne attempt/root uniqueness
oraz `UNIQUE(scope_bytes, operation_key)`,
`UNIQUE(scope_bytes, entitlement_key)` i `UNIQUE(scope_bytes, account_key)`.
Istniejące envelope i status pozostają niezmienne. Konflikt drugiego attempt dla
tego samego entitlement odmawia w preparation, przed finalnym BIND.

Race dwóch różnych entitlement ze wspólnym history predecessor dowodzi
history CAS i rollback przegranego BIND. Race dwóch wywołań tego samego
attempt dowodzi pojedynczej decision i exact replay. Żaden z nich nie dowodzi
race dwóch różnych, jednocześnie legalnych STAGED attempts jednego entitlement.
Ten scenariusz wymaga osobnego replacement/supersession protocol.

Późniejsza migracja musi być osobnym reviewed kontraktem i administracyjną
migracją, proponowaną jako `ROOT_PROOF_PREPARATION_SUPERSESSION_V3`:

1. Przed zmianą administrator pod locks kwalifikuje obecny V2 catalog,
   physical identity, oryginalne bytes i wszystkie istniejące decyzje. Zapisuje
   immutable predecessor inventory i dokładne bytes inventory. Refuse przy
   niepełnym RESERVED/STAGED/decision linkage; brak backfill fikcyjnych podpisów.
2. Tworzy nowe wersjonowane tabele `intent_proposals_v3`,
   `prepared_proposals_v3`, `attempt_terminal_events_v3` oraz
   `entitlement_proposal_slots_v3`. Stare tabele, UNIQUE, envelopes i decyzje
   pozostają niezmienną historią. Istniejące kandydaty są importowane z exact
   lossless `BYTEA` oraz origin reference, bez regeneracji ID, payload,
   signature, envelope, snapshot lub entitlement lineage.
3. Nowe proposals mają globalny unikalny attempt/root i dokładne immutable
   linkage RESERVED→STAGED. Operation/entitlement/account przestają być UNIQUE
   między proposals wyłącznie w nowych tabelach i po przeglądzie nowej
   kardynalności. Slot pozostaje unikalny dla frozen scope i entitlement
   generation; jego revision jest CAS fence, nie podpisem i nie decyzją.
4. Każdy replacement wskazuje dokładny predecessor attempt, slot revision,
   lineage generation/revision i retained request identity. Append-only
   terminal event `SUPERSEDED` może powstać tylko przez oddzielne qualified SQL
   po autorytatywnym rozstrzygnięciu wcześniejszego commitu pod tym samym
   serialization lock. Missing decision po unknown COMMIT nie jest dowodem
   pozwalającym supersede. Żadna ścieżka nie aktualizuje original proof bytes.
5. Nowy kwalifikowany commit sprawdza terminal events oraz slot revision,
   blokuje ten sam entitlement lineage i decyduje o jednej zwycięskiej próbie.
   Dwie legalne STAGED proposals mogą rywalizować; tylko zwycięzca tworzy
   BIND/history/decision. Przegrany ma odrębny immutable terminal outcome,
   nie jest exact retry zwycięzcy i nie dziedziczy jego proof ID/signature.
6. Migracja ma nowe reviewed source inventory, owners/OIDs, ACL i manifest,
   oddzielne genuine Python capabilities i jawny recovery protocol. Nie
   rozszerza commit writera o arbitrary DML ani możliwość wyboru manifestu.
   Legacy V1/V2 qualifiers nadal odrzucają nowe inventory.
7. Przed dostępnością trzeba wykonać PostgreSQL 16 tests dwóch różnych
   attempts tego samego entitlement w niezależnych procesach, replacement
   przeciwko równoczesnemu commit/revoke/supersede, lost-response recovery,
   migration conservation i immutable history replay po close. Zmiana
   kardynalności bez tych dowodów nie jest dopuszczalna.

To specyfikacja przyszłego kroku. Ten PR nie tworzy wymienionych tabel,
nie usuwa UNIQUE i nie implementuje successor reproposal.

## Wymagane dowody i pozostałe blokery

Integration tests muszą wykonywać rzeczywiste PostgreSQL 16, sprawdzać
oryginalne bytes i transakcyjne snapshots oraz udowodnić brak wcześniejszego
RESERVED/STAGED dla nowego attempt po migracji. Ten właśnie nowy STAGED musi
przejść qualified approval i atomowy commit, a decyzja zachować jego exact
payload/signature/envelope. Negative paths porównują wszystkie pozostałe
authority snapshots; mocks nie dowodzą PostgreSQL rollback ani concurrency.

Wymagane są identity/generation/signature/envelope conflicts, wrong database
i principal, catalog/ACL drift, absence enrollment, trwały close, crash między
fazami oraz unknown preparation writes. Niezależne procesy służą scenariuszom
równoczesności. Wyniki konkretnego rerun i diff coverage należą do raportu PR;
opis testu nie stanowi deklaracji, że wykonanie przeszło.

Pozostają produkcyjne blokery: production custody/lifecycle isolation i
ochrona przed administracyjnym rollbackiem, kwalifikowana produkcyjna
kompozycja RootProofIssuer, reconciliation literalnego wymogu odrębnych
transactions z fizycznym atomic issuer commit, replacement cardinality i
successor reproposal, migracja już wcześniej wyemitowanej legacy historii,
AccountGenesis oraz checkpoint activation. Nowe preparation adaptery nie
zmieniają żadnego istniejącego production readiness gate.
