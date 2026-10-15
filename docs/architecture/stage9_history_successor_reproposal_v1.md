# Stage 9: Durable History Successor Reproposal V1

Ten protokół umożliwia ponowne zaproponowanie history successor dla tej samej
próby po rzeczywistym konflikcie predecessor CAS. Root Proof powstał wcześniej:
reproposal nie tworzy nowego proof, podpisu Root Proof, RESERVED, STAGED ani
signed immutable request. Publiczny issuer pozostaje zablokowany:

```text
ATOMIC_ISSUER_READY=NOT_READY
PRODUCTION_LIFECYCLE_ISOLATION=BLOCKED
ROOT_PROOF_ISSUANCE_AVAILABLE=FALSE
```

## Audyt V1 i granica zmiany

Przejrzano runtime, schema, migration i qualification atomowego commitu,
oba moduły authenticated history, preparation V2 i jego odczytowy kwalifikator
oraz dokumenty Atomic Commit V1 i PostgreSQL V2 preparation. Codebase Memory
nie jest dostępne w środowisku tej zmiany; ustalenia sprawdzono w źródłach.

| Blokada V1 | Rozwiązanie w nowym inventory |
| --- | --- |
| `approved_commits(attempt_key PRIMARY KEY, package)` zachowuje tylko oryginalny pakiet | Oryginalny approval pozostaje bez zmian. `history_proposals` dopisuje wersje związane z tym samym approval. |
| `approve_commit(jsonb)` odrzuca inny pakiet tej samej próby | Odrębne `approve_history_proposal(jsonb,bigint,text)` sprawdza immutable bazę i predecessor propozycji. |
| `commit_root_proof(bytea)` wybiera pakiet tylko według attempt | Nowe wejście wymaga exact candidate identity; chroniony selector wskazuje jedyną aktywną wersję. |
| `validate_decision()` porównuje decyzję wyłącznie z oryginalnym approval | Jawna migracja dodaje `validate_history_proposal_decision()` i przepina deferred trigger na walidację wybranego immutable candidate. |
| Brak decyzji w snapshot nie rozstrzyga COMMIT utraconego połączenia | Admission i final COMMIT modyfikują ten sam selector w transakcji SERIALIZABLE. Read-only `NOT_FOUND` pozostaje INDETERMINATE. |
| V2 `current_record()` sprawdza strukturę, ale sam nie uwierzytelnia podpisów head | Odrębny odczyt authenticated snapshot weryfikuje chain i zachowane heads przez autentyczny history provider. |

Źródła SQL V1 w `postgresql_atomic_root_proof_schema.py` pozostają niezmienione.
Także w bazie wszystkie funkcje poprzedniego inventory zachowują źródła i OIDs;
nowy validator ma odrębną nazwę. Jawnie zmienia się deferred trigger i granty
dwóch zamkniętych legacy wejść.
Nowy builder jawnie komponuje ich walidację oraz fizyczny write set; sprawdza
oczekiwane punkty kompozycji i odmawia po zmianie ich źródła. Nie wykonuje kodu
funkcji pobranego z bazy. Nowe sources, deklaracje, właściciele i ACL stanowią
osobne reviewed inventory.

## Identity, wersja i immutable lineage

`history_proposals` zachowuje `attempt_key`, dodatni `revision`, `candidate_id`,
`predecessor_candidate_id` oraz exact `package`. Pierwsza wersja nie ma
predecessora propozycji. Każda następna wskazuje autentyczną poprzednią wersję
tej samej próby. Klucze UNIQUE i foreign keys wiążą identity, wersję i lineage.
UPDATE, DELETE i TRUNCATE immutable evidence blokują zawsze aktywne triggers.

Identity `hp1_…` jest SHA-256 z rozdzieloną domeną V1, pakietem JSONB w
reprezentacji PostgreSQL, binarnym numerem wersji oraz identity predecessora.
To identity protokołu PostgreSQL; nie jest nowym canonical formatem Root Proof.
Exact retry tej samej wersji i zawartości zwraca zachowany candidate. Inny
pakiet pod tą samą wersją jest konfliktem, także po późniejszym supersession.

`history_proposal_current` jest chronionym selectorem, a nie approval. Zmiana
następuje tylko przez zamkniętą funkcję SQL, po porównaniu oczekiwanej wersji
i candidate identity. Stary candidate nie może wykonać nowego BIND.

Niezmienne są wszystkie identity próby, operacji, entitlement i generation,
account, proof, oryginalny canonical payload i podpis Root Proof, oba canonical
envelopes, oryginalny root signer snapshot, authorization i exact signed request.
Niezmienne pozostają także registry revision, BOUND projection, history stream,
event identity, event payload i event digest. Zmieniają się wyłącznie pozycja
w historii, authenticated predecessor/digest oraz wynikające record/head bytes
i nowa autentyczna history signature. Protokół V1 wymaga tego samego enrolled
history credential i generation; rotacja na inny history signer wymaga osobnej
wersji protokołu.

Historyczna autoryzacja obejmuje także publiczne capability providerów. Po
migracji odebranie standalone BIND zmienia entitlement `compare_and_swap`
z `true` na `false`. Nie wolno z tego powodu tworzyć nowej authorization ani
ponownie podpisać request. Wewnętrzny odczytowy bridge porównuje wszystkie
oryginalne fakty ze świeżo zakwalifikowanym providerem V2. Wyłącznie ten
konkretny historyczny grant może różnić się w kopii porównawczej. Oryginalny
authorization digest, evidence, attempt identity, request bytes i podpisy
pozostają dokładnie zachowane. Bridge ponownie weryfikuje genuine binding,
public credentials, prywatne requester/claimant custody oraz oryginalny
SQLite signed attempt w trybie read-only. Nie zmienia capability maps ani
retained providera i nie nadaje nowej capability.

## Admission, serializacja i unknown outcome

Admission ponownie kwalifikuje genuine evidence, oryginalne RESERVED/STAGED,
podpisy i live public credentials. Nową atestację tworzy wewnętrznie exact
`LocalHistoryAttestationSigningProvider` w stanie ACTIVE. API nie przyjmuje
caller-owned signature ani credential identity zastępującej providera.

SQL wymaga jednej zapisywalnej transakcji PostgreSQL 16 SERIALIZABLE. Admission
blokuje i aktualizuje selector przed sprawdzeniem immutable decyzji. Final COMMIT
wykonuje ten sam no-op UPDATE selectora. Ten zapis jest celowy: transakcja,
która rozpoczęła snapshot przed oczekiwaniem na wcześniejszy COMMIT, nie może
po zwolnieniu blokady uznać starego braku marker za wynik recovery. PostgreSQL
odrzuci ją jako serialization failure. Sam advisory lock nie zapewniłby tego
warunku dla starszego snapshotu.

Następnie SQL wymaga: braku zwycięskiej decyzji, rzeczywistego przesunięcia
head względem poprzedniego kandydata, dokładnego aktualnego signed predecessora,
ACTIVE/UNBOUND przy oryginalnej entitlement revision, oryginalnych envelopes
i request credentials oraz otwartych enrolled root/history/trust fences.
Weryfikuje canonical relacje record/head i wyprowadzone digests. Nie dopisuje
BIND, history event ani decision podczas admission.

Read-only recovery zwraca wyłącznie pozytywną immutable decyzję albo
INDETERMINATE. Utrata połączenia, SQL drift, revoked signer lub błąd kwalifikacji
nie uruchamia automatycznej reproposal. Klient jawnie przekazuje zachowaną
identity poprzedniej propozycji. Po utracie odpowiedzi approval odczytuje
chroniony candidate i może wysłać exact final COMMIT. Nie tworzy nowej próby.

## Finalny COMMIT i crash

Nowy commit zachowuje walidację V1 oryginalnego proof i pełny atomowy write set:
entitlement CAS BIND, registry history, issuance record, podpisany head i immutable
decision marker. Wszystkie zapisy należą do jednej fizycznej transakcji
SERIALIZABLE na jednym trwałym PostgreSQL 16 primary. Deferred constraints
nadal wiążą registry i history z decyzją tej samej transakcji.

Przed nową decyzją SQL blokuje lifecycle fences i requester/claimant credential
generations w ustalonym porządku. Close/revoke i final COMMIT korzystają z tych
samych autorytatywnych fences. Exact replay istniejącej decyzji zwraca zapisane
bajty także po późniejszym zamknięciu eligibility; nie wykonuje drugiego BIND.

Crash przed COMMIT admission cofa całą nową wersję i selector. Crash po nim
zachowuje oba. Crash podczas final COMMIT pozostawia cały write set albo żaden;
utrata odpowiedzi pozostaje unknown do kwalifikowanego recovery. Restart procesu
nie upoważnia do wymiany manifestu ani do rekonstrukcji z caller-owned pin.

## Migracja, kwalifikacja i uprawnienia

Jawna administracyjna migracja `DURABLE_HISTORY_SUCCESSOR_REPROPOSAL_V1` wymaga
kwalifikowanego Durable Custody Lifecycle V2 z Signer Enrollment SQL V3.
Archiwizuje poprzedni katalog i digest, utrwala nowe source hashes, a istniejące
approvals zapisuje jako immutable revision 1. Nie zmienia application evidence.
Odczytowa rekonstrukcja po restarcie ponownie porównuje compiled sources,
inventory, ACL, właścicieli, role OIDs, physical database identity i trwałość.

| Obiekt / principal | Zmiana |
| --- | --- |
| `history_proposals` | Nowy append-only journal wersji. |
| `history_proposal_current` | Nowy chroniony selector. |
| `history_reproposal_contract` | Immutable migration identity, predecessor archive i reviewed source hashes. |
| Atomic writer | EXECUTE exact candidate COMMIT, chronionych odczytów i qualification; bez DML, DDL, approval lub component helper EXECUTE. |
| Oddzielny test verifier | EXECUTE versioned admission i odczytów; bez finalnego COMMIT. |
| Component NOLOGIN owners | Wąskie `reproposal_*` read/validation facades dostępne tylko coordinator ownerowi; nie wykonują samodzielnego BIND/history/decision. |
| Stare publiczne `approve_commit` / `commit_root_proof` | Granty wycofane, aby stara ścieżka nie mogła ominąć selectora. |
| `validate_history_proposal_decision()` | Nowa funkcja deferred trigger wiążąca decyzję z aktywnym candidate; oryginalne `validate_decision()` pozostaje niezmienione. |
| Preparation principals | Zachowują dotychczasowe prawa V2; brak nowych finalnych praw emisji. |

Weryfikacja Ed25519 należy do genuine kwalifikowanego runtime verifiera,
tak jak w Atomic Commit V1. PostgreSQL sprawdza canonical relacje, digests,
immutable bytes i enrollment, lecz nie udaje samodzielnej weryfikacji Ed25519.
Oddzielny `test_admin_role` pozostaje zaufanym authority evidence ingress;
jego credential nie jest dostępny issuer writerowi ani publicznym preparerom.

Qualification obejmuje także zewnętrzny preparation schema. Publiczne V2
readers nie importują reproposal administratora, migracji ani runtime writera.
Nie ma flagi pomijania kwalifikacji, automatycznego repair lub nowego production
selector. Legacy custody mutation pozostaje zablokowane. UNIQUE RESERVED/STAGED
pozostają bez zmian: dwa attempts tego samego entitlement nie są zakresem tej
migracji. Checkpoint, AccountGenesis i FreshnessAuthority pozostają odrębne.

## Walidacja i pozostałe granice

Testy integracyjne korzystają z genuine podpisanych attempts, rzeczywistych
PostgreSQL 16 primary i autentycznego history custody. Scenariusze i wyniki
uruchomionych kontroli są raportowane w opisie Draft PR. Nowy kod i testy
wchodzą do istniejących Ruff, mypy Linux/Windows i seryjnego PostgreSQL joba;
próg diff coverage pozostaje 70%, a dotychczasowe testy i timeouty pozostają.

Ten protokół jest wewnętrzną izolowaną granicą testową. Nie ustanawia
produkcyjnego RootProofIssuer, kwalifikacji sprzętowej, produkcyjnego custody
rolloutu ani nowej kardynalności attempts per entitlement. Zachowuje zamrożone
schematy M0.5 oraz `_expected_issuance_history_payload`.
