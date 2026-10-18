# S9-RPI-04 — propozycja addendum do granic transakcyjnych Root Proof

> **Aktualizacja 2026-10-18:** propozycja **A + C** została zaakceptowana kierunkowo przez właściciela projektu. [Dokument decyzji v1](../decisions/stage9_root_proof_transaction_authority_owner_approval_v1.md) definiuje jej zakres i warunki; poniższy tekst pozostaje historyczną wersją propozycji. Nadal wymagana jest formalna integracja wersjonowanego kontraktu i osobna kwalifikacja produkcyjna. `NO-GO` bez zmian.

Status: **CONTRACT_RECONCILIATION_PENDING_APPROVAL**. Data: 2026-10-18.
Podstawa audytu: `main` po #3154,
`7ed0e90321c9e24d6770f968e5bd7026bfa73cb7`.

Ten dokument jest propozycją do osobnego zatwierdzenia przez właściciela
projektu. Nie zmienia canonical JSON, ich projekcji Markdown ani historycznego
freeze. Nie stanowi capability, decyzji o dostępności, zgody na wdrożenie lub
polecenia wykonania migracji. ADR i macierz aktualnego kodu znajdują się w
[reconciliation](../stage9_root_proof_transaction_authority_reconciliation.md).

Rekomendacja: zatwierdzić **C jako wąskie addendum umożliwiające A**, po
niezależnym przeglądzie. A zachowuje atomowy write set issuera, lecz samo
rozdzielenie ownerów SQL nie spełnia literalnego wymagania odrębnych transakcji
dla każdego podległego persistence component. Do decyzji obowiązuje NO-GO.

## 1. Źródła normatywne i wykazana sprzeczność zakresu

Poniższe skróty wskazują canonical JSON; ścieżki są JSON Pointer, indeksy
tablic zaczynają się od zera. Sąsiadujące pliki `.md` deklarują, że są
deterministycznymi projekcjami JSON; nie są dodatkową zgodą na implementację.

- **I** — [independent issuer](../cryptohunter_product_architecture/m05_account_genesis_independent_root_proof_issuer_contract.json).
- **S** — [production substrate](../cryptohunter_product_architecture/m05_account_genesis_root_proof_issuer_production_substrate_selection_contract.json).
- **R** — [implementation readiness](../cryptohunter_product_architecture/m05_account_genesis_root_proof_issuer_implementation_readiness_contract.json).
- **P** — [physical persistence](../cryptohunter_product_architecture/m05_account_genesis_physical_persistence_crash_atomicity_contract.json).

| Źródło / JSON Pointer | Dokładna wartość |
| --- | --- |
| I `/bind_authorization_order/7` | `"8_ATOMICALLY_BIND_AND_PERSIST_IDENTICAL_SIGNED_PROOF_AND_HISTORY"` |
| I `/authority_model/semantic_authority_owner` | `"IndependentAccountGenesisRootProofIssuer"` |
| I `/authority_model/retained_issuance_history_owner` | `"independent root-proof issuer globally serialized registry"` |
| I `/cross_authority_atomicity/model` | `"NO_DISTRIBUTED_TRANSACTION; immutable exact-operation proof plus recoverable local phase machine and independent freshness CAS"` |
| I `/replay_and_consumption/consumption_owner` | `"NONE; CHA does not call issuer after issuance to consume"` |
| I `/lost_response/same_attempt_id_success` | `"return byte-identical root_proof_id, signed payload, signature and retained history for that exact attempt"` |
| I `/lost_response/requested_attempt_id_must_equal` | `"issuer_history.bound_issuance_attempt_id"` |
| S `/authority_separation/rule` | `"PHYSICAL COLOCATION != AUTHORITY REUSE"` |
| S `/authority_separation/separate_semantic_owners_schemas_roles_credentials_namespaces_transactions_and_lifecycles` | `true` |
| S `/authority_separation/required_separation/4` | `"transaction ownership"` |
| S `/authority_separation/CHA_direct_registry_write` | `false` |
| S `/authority_separation/issuer_direct_account_commit` | `false` |
| S `/authority_separation/Issuer_to_CHA_AttemptStore_write` | `false` |
| S `/authority_separation/CHA_to_Issuer_Registry_semantic_BIND_write` | `false` |
| R `/frozen_semantics_change_rule` | `"DO_NOT_REOPEN_WITHOUT_CONCRETE_EXPLOIT_OR_CONTRADICTION"` |
| R `/status/semantic_contract_frozen` | `true` |
| R `/status/implementation_readiness_frozen` | `true` |
| R `/implementation_gate/readiness_freeze_does_not_allow_implementation` | `true` |
| R `/implementation_gate/non_production_implementation_mode` | `"NONE_FROZEN"` |
| R `/status/implementation_allowed` | `false` |
| R `/production_M05_gate/issuer_implementation_allowed_alone_is_sufficient` | `false` |
| P `/authority_vs_storage_roles/transaction_coordinator` | `"CryptoHunterAccountAuthority"` |
| P `/selected_or_blocked_protocol/logical_authority_atomicity_equals_single_physical_transaction` | `false` |
| P `/result/selected_protocol` | `"INITIAL_BINDING -> PREPARED -> EXTERNAL_FRESHNESS_CAS -> LOCAL_FINAL_COMMIT"` |
| P `/selected_or_blocked_protocol/point_of_no_return` | `"exact authoritative freshness acceptance N -> N+1"` |
| P `/freshness_ownership_boundary/coordinator_may_unilaterally_advance_authoritative_generation` | `false` |
| P `/publishability_rule/authority_acceptance_before_local_final_commit` | `"NOT_PUBLISHABLE"` |

Konkretną podstawą rozpatrzenia R `/frozen_semantics_change_rule` jest kolizja
zakresu: jeśli „transactions” w S rozciąga się na osobne entitlement,
retention i history persistence, żadna sekwencja ich niezależnych COMMIT nie
wykonuje wymaganego atomowego kroku I. Jeżeli zakres dotyczył wyłącznie
niezależnych autorytetów protokołu, trzeba tę granicę jawnie zatwierdzić.
Wartość `true` nie definiuje takiego wyjątku, a owner-owned helper nie tworzy
nowej transakcji. Nie stwierdzamy, że wszelkie możliwe interpretacje freeze są
logicznie sprzeczne; stwierdzamy, że A nie dowodzi literalnej separacji
fizycznych transakcji podległych komponentów. R dopuszcza przedstawienie
konkretnego konfliktu do review; nie zatwierdza proponowanego rozwiązania.

P nie jest wyjątkiem uprawniającym do rozdzielenia write set issuera.
Jego wielofazowy protokół dotyczy utworzenia konta przez CHA, niezależnej
freshness CAS oraz własnego lokalnego closure. Issuer decision COMMITTED
nie oznacza AccountGenesis COMMITTED.

## 2. Proponowane brzmienie normatywne — do zatwierdzenia

> Separacja semantic owner, schema owner, runtime principal, administratora,
> credentials, namespaces, authority API i lifecycle pozostaje obowiązkowa.
> Separacja własności transakcji obowiązuje pomiędzy niezależnymi autorytetami:
> RootProofIssuer, CHA wraz z CHA AttemptStore i AccountGenesis account commit,
> FreshnessAuthority oraz CheckpointAuthority. Nie oznacza oddzielnego
> fizycznego COMMIT każdej podległej struktury persistence decyzji issuera.
>
> Wyłącznie finalna decyzja RootProofIssuer może objąć w jednej fizycznej
> transakcji SERIALIZABLE na jednym zakwalifikowanym PostgreSQL 16:
> entitlement CAS BIND wraz z jego registry history, exact signed proof
> retention/linkage, authenticated issuer record i signed head oraz immutable
> decision marker. Transakcję od otwarcia po COMMIT/ROLLBACK posiada jeden
> issuer-owned coordinator. Podlegli semantic schema owners udostępniają
> zamknięte, owner-owned persistence helpers; nie oddają ownership, DML/DDL,
> role membership ani prawa samodzielnego finalnego BIND issuer writerowi.
>
> Shared transaction nie nadaje coordinatorowi kompetencji provisioningu lub
> zmiany credentials, signing custody, checkpoint acceptance, freshness CAS,
> CHA attempt mutation ani account commit. Każdy z tych autorytetów zachowuje
> własne uwierzytelnienie, uprawnienia i odpowiednie własne transakcje.
> Credential owners mogą dostarczyć wyłącznie helpery exact reads i lifecycle
> locks trzymanych do finalnego COMMIT, bez mutation credential records.
>
> Signing i operacje zdalne kończą się poza finalną transakcją. Trwałe
> RESERVED i STAGED są przygotowaniem; STAGED proof przed BIND nie jest
> autorytatywnie issued proof. Finalna decyzja atomowo wiąże oryginalne,
> niezmienne bajty z wygrywającym BIND i history. Nie wolno zrealizować jej
> przez kolejne samodzielne COMMIT, kompensację lub późniejsze dopisanie
> brakującej części write set.

To jest najmniejszy proponowany wyjątek zakresu S `transactions`: dotyczy
wyłącznie podległego persistence jednej decyzji issuera. Nie proponuje
zmiany I atomicity, kardynalności, bind order, winner identity lub exact retry.
Nie nadaje nowego znaczenia P ani nie zastępuje jego protokołu. W tej zmianie
nie dopisuje się tego tekstu do canonical JSON. Ewentualny przyszły
versioned addendum i jawna migracja wymagają osobnej zatwierdzonej decyzji,
utrwalenia zakresu/sources/digestów oraz zachowania historycznych artefaktów.

## 3. Dokładny finalny write set A i granice

| Wewnątrz jednego finalnego COMMIT | Poza finalnym COMMIT |
| --- | --- |
| Entitlement generation/revision CAS, immutable BOUND i registry history | Provisioning, revoke/supersede i zmiany claimant/requester credentials |
| Chronione exact RESERVED/STAGED references oraz oryginalne payload/signature/envelope bytes związane z decyzją | Utworzenie RESERVED/STAGED i podpisanie root proof/head przed finalną transakcją |
| Authenticated issuer history record i signed head z exact predecessor CAS | Signing custody i operacje HSM/network; sam podpis nie jest decyzją |
| Immutable decision marker wskazujący bound attempt, tuple, root ID, original bytes i history | CheckpointAuthority append/acceptance i własny checkpoint transaction |
| Locks i exact ACTIVE fences dla requester, claimant, issuer signer, history signer i trust activation do COMMIT/ROLLBACK | CHA AttemptStore reservation/finalization/replacement, INITIAL_BINDING i PREPARED |
| Jedna pełna decyzja wszystkich nowych zapisów albo pełny rollback | FreshnessAuthority CAS/receipt oraz AccountGenesis LOCAL_FINAL_COMMIT |

Chroniona FK/referencja do immutable bytes w tej samej bazie może zachować
bajty bez nowej kopii STAGED. Digest, link do innej bazy lub usuwalny rekord
nie wystarcza. Nie wymaga się ponownego podpisania ani zmiany STAGED na
ISSUED przez UPDATE. Sam marker nie zastępuje weryfikacji podpisów, pełnego
history linkage i wymaganego checkpoint. Lifecycle i credential locks są
read/fence participation w transakcji decyzji, bez współdzielenia prawa
zmiany ich semantycznego stanu.

Aktualny isolated coordinator w
[`PostgreSQLAtomicRootProofCommit.commit()`](../../../bot_core/postgresql_atomic_root_proof_commit.py)
otwiera `BEGIN ISOLATION LEVEL SERIALIZABLE`, wywołuje closed SQL
`commit_root_proof(bytea)`, sprawdza exact bytes i ciągłość transakcji, a
następnie wywołuje jeden `conn.commit()`. W
[`postgresql_atomic_root_proof_schema.py`](../../../bot_core/postgresql_atomic_root_proof_schema.py)
ta funkcja kolejno wywołuje owner-owned `atomic_bind`,
`atomic_append_record`, `atomic_retain_head` i INSERT decision; są to
operacje tego samego backend transaction. `SECURITY DEFINER` zmienia
effective privilege, nie transaction owner ani punkt COMMIT. SQL function
return nie jest potwierdzeniem trwałego COMMIT.

[`AtomicRootProofTransactionOwner`](../../../bot_core/postgresql_atomic_root_proof_transaction.py)
obserwuje to samo połączenie/backend, xid, database i direct-login principal;
nie jest production authorization ani dowodem retencji row locks po
caller-owned savepoint rollback. Finalny coordinator musi posiadać cały
przebieg transakcji. Równe DSN, `system_identifier` i database OID nie dowodzą,
że kopie PGDATA są jednym backendem. Produkcyjny
[`ProductionRootLifecycleBoundary.commit()`](../../../bot_core/production_root_lifecycle.py)
nadal bezwarunkowo odmawia.

## 4. Odrębne autorytety i recovery

Issuer nie zapisuje CHA AttemptStore. Jego durable reservation, późniejsza
immutable signed finalization i replacement switch pozostają osobnymi
transakcjami CHA, w zamrożonej kolejności. Nie ma wspólnego issuer/CHA
COMMIT ani post-issuance consume callback.

CheckpointAuthority samodzielnie uwierzytelnia i zachowuje checkpoint
konkretnego history head; późniejszy checkpoint nie naprawia częściowego
issuer write set. Checkpoint signer/namespace/admin/rollback domain
pozostają odrębne. `PRODUCTION_LOCAL` nie daje ochrony przed skoordynowanym
rollback całego hosta; capability `PRODUCTION_SERVER_READY` wymaga
niezależnego rollback/security domain. Checkpoint nie jest account freshness.

FreshnessAuthority posiada własną decyzję `N -> N+1` i receipt. CHA może
zaproponować candidate, lecz dopiero exact authoritative acceptance stanowi
P point-of-no-return. Następny atomowy lokalny account commit należy do CHA:
operation COMMITTED, reservation CONSUMED_COMMITTED, terminal history,
exact receipt/history proof, resulting heads/accepted generation oraz
terminal idempotent result. Issuer BOUND nie publikuje konta i nie
autoryzuje unilateralnego przesunięcia freshness.

Utrata odpowiedzi finalnego COMMIT to OUTCOME_UNKNOWN/INDETERMINATE. Recovery
ma rozstrzygnąć ten sam authoritative marker i history oraz zwrócić
byte-identical proof dla faktycznie bound attempt. `NOT_FOUND`, timeout,
nowy lokalny current pointer lub semantycznie podobny request nie dowodzą
UNBOUND i nie uprawniają do remint, ponownego SIGN, drugiego BIND, UNBIND,
replacement ani drugiego event. Brak dostępnego autorytatywnego
rozstrzygnięcia wymaga FAIL_CLOSED.

## 5. B: osobne transakcje i ograniczenia PostgreSQL 2PC

Przy BIND COMMIT przed retention/history COMMIT crash pozostawia BOUND bez
pełnego proof/history; odwrotna kolejność pozostawia autorytatywne
proof/history bez wygrywającego BIND. Unikalność każdej tabeli, mutex,
SERIALIZABLE osobnych transakcji i owner-owned helpers nie zamykają tej
szczeliny. Rollback kolejnej transakcji nie cofa wcześniejszego COMMIT.
Kompensacja UNBIND byłaby zmianą immutable winner history. Wcześniejsze
STAGED bez BIND jest legalnym przygotowaniem wyłącznie dlatego, że nie ma
statusu autorytatywnej emisji.

Pełny distributed transaction manager z trwałą decyzją mógłby koordynować
prepared participants, lecz wymaga nowego protokołu i osobnej zgody wobec I
`NO_DISTRIBUTED_TRANSACTION`. PostgreSQL `PREPARE TRANSACTION` utrwala
przygotowany stan i jego blokady po utracie sesji; nie wykonuje globalnego
COMMIT. Potrzebne są co najmniej: konfiguracja `max_prepared_transactions`,
niepodrabialne global transaction IDs/participant set, trwały coordinator
decision log poza niejednoznaczną sesją, exact recovery i ograniczone prawa
`COMMIT PREPARED`/`ROLLBACK PREPARED`.

Po utracie coordinatora prepared transactions pozostają in-doubt i mogą
blokować BIND, lifecycle i maintenance; długie xid utrudniają vacuum.
Obowiązują globalny lock order, deadlock handling oraz zakaz samodzielnego
heurystycznego rollback participant po globalnej decyzji COMMIT. Nie można
wnioskować ABORT z timeoutu ani braku response/rekordu coordinatora.
Commit prepared participants następuje oddzielnymi poleceniami; obserwator
omijający protokół może zobaczyć część już committed i część jeszcze prepared.
2PC nie zapewnia jednoczesnej widoczności wielu zasobów bez dodatkowych
read/publication gates i rozstrzygnięcia globalnej decyzji. Nie czyni też
zewnętrznego SIGN lub checkpoint automatycznie transactional participant.

Brak tu zakwalifikowanego managera, decision log i prepared recovery, a
obecny coordinator używa jednego zwykłego COMMIT. B z niezależnymi COMMIT
jest niewystarczające; B z 2PC jest osobnym, kosztownym projektem, a nie
gotowym sposobem spełnienia obu literalnych zapisów. Nie zaleca się go dla
jednego issuer write set mieszczącego się w jednym PostgreSQL.

## 6. Decyzje do zatwierdzenia i rozłączne klasyfikacje

| Decyzja | Wymagane rozstrzygnięcie właściciela projektu |
| --- | --- |
| D1 | Czy przyjąć dokładnie ograniczony zakres `transactions` z sekcji 2 dla podległego persistence jednej decyzji issuera? |
| D2 | Czy jeden issuer-owned physical coordinator i closed owner-owned helpers są akceptowanym wykonaniem kroku I `/bind_authorization_order/7`, bez własności konta/freshness/checkpoint? |
| D3 | Czy immutable STAGED original bytes plus chronione atomowe linkage/marker spełniają final retention bez ponownego SIGN i zmiany STAGED? |
| D4 | Jak utrwalić zatwierdzone versioned addendum, jego relację do istniejącego freeze i przyszłą jawną migrację, bez przepisywania historii? |
| D5 | Czy utrzymać odrzucenie B/2PC i zachować wszystkie pozostałe bramki produkcyjne do odrębnego udowodnienia? |

`TECHNICALLY_FEASIBLE` dotyczy mechanizmu i dokładnie wskazanego zakresu
wykonanych dowodów; nie oznacza ukończonego produkcyjnego issuera.
`CONTRACTUALLY_APPROVED` wymaga odrębnej, utrwalonej decyzji D1–D5 i review;
sam merge ADR, test PASS, DTO, flag `approved`, konfiguracja lub test token
nie są taką zgodą. `PRODUCTION_QUALIFIED` wymaga niezależnie potwierdzonego
wdrożenia wszystkich produkcyjnych granic. Żaden stan nie wynika
automatycznie z pozostałych; dokument nie wystawia runtime capability.

Aktualny wynik kontraktowy: **CONTRACT_RECONCILIATION_PENDING_APPROVAL**;
produkcyjny: **NO-GO / BLOCKED**. Szczegółowy zakres wykonanych testów A
jest raportowany w ADR/PR. Opis mechanizmu i historyczne PASS nie są wynikiem
nowego uruchomienia ani kwalifikacją fizycznej instalacji.

## 7. Warunki przyszłego rozpatrzenia zniesienia blokady

Wszystkie poniższe warunki są konieczne; zatwierdzenie tego addendum samo
w sobie nie odblokowuje żadnej operacji:

1. Utrwalone zatwierdzenie D1–D5 wraz z dokładnym zakresem i wersją; canonical
   frozen JSON i historyczny freeze zachowane, jawna migracja osobno reviewed.
2. Dowód jednego PostgreSQL 16/backend i jednego finalnego COMMIT pełnego
   write set; crash przed/po COMMIT, restart oraz utrata odpowiedzi. Brak
   częściowego BOUND/decision/history; exact recovery bez ponownego SIGN.
3. Kwalifikacja exact catalog/ACL/function sources/owners, `SECURITY DEFINER`,
   stałego search_path, direct-login/OID, memberships i wszystkich pośrednich
   ścieżek; odmowa standalone BIND, helper EXECUTE, DML/DDL i SET ROLE bypass.
4. Rzeczywiste lifecycle fences i kompatybilny lock order wszystkich signer,
   requester/claimant, trust activation oraz custody transition paths; locks
   do finalnego COMMIT, zamknięcie po crash i niezależne completion evidence.
5. Genuine production attempt ingress, authenticated INITIAL_BINDING,
   claimant/requester authorization i retained history; brak caller-owned
   zgody. Pełne exact retry i NOT_FOUND/OUTCOME_UNKNOWN negative paths.
6. Review kardynalności RESERVED/STAGED/approval i legalnych replacementów:
   obecny V1 może blokować dwóch różnych finalnych kandydatów tego samego
   entitlement już na preparation. Race exact retry lub różnych entitlements
   nie zastępuje dowodu wymaganego distinct-attempt race/replacement/liveness.
7. Kwalifikacja rzeczywistego private control, OS/native-keyring i signer
   isolation, checkpoint/restore/anti-rollback dla deklarowanego profilu oraz
   wszystkich wymaganych providerów. Read-only journal lub testowy MAC nie
   zastępują niezależnej custody completion authority.
8. Odrębne spełnienie readiness predicates, kwalifikacja CHA i
   FreshnessAuthority oraz całego AccountGenesis closure. Zgodnie z R
   `/production_M05_gate/issuer_implementation_allowed_alone_is_sufficient=false`
   sama gotowość issuera nie wystarcza.

Do czasu osobnego spełnienia i zatwierdzenia tych bramek produkcyjne
Root Proof issuance, SIGN, BIND, enrollment, aktywacja kluczy i Stage 10
pozostają zablokowane. Nie proponuje się nowego publicznego write entrypointu
ani adaptowania izolowanego testowego coordinatora do produkcji.
