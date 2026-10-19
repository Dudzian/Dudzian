# Stage 9 — S9-RPI-12: niezależny CCA close, transakcja i held-lock (TEST-only)

**Stan:** `TEST_ONLY_NON_AUTHORIZING`; **produkcja:** `NO-GO`.

Kontynuacja zatwierdzonych poprzedników [S9-RPI-09](stage9_independent_committed_close_reader_v1.md),
[S9-RPI-10](stage9_cca_independent_reader_postgresql_smoke.md) oraz
[S9-RPI-11](stage9_cca_reader_catalog_inspection.md).

## Problem i zakres

Poprzednie etapy rozdzielały inspekcję katalogu od odczytu
`PENDING_CLOSED`. Tymczasem przyszły `IndependentCommittedCloseReader`
musi utrzymać właściwy PostgreSQL slot lock podczas porównania z
**niezależnym źródłem**, a obserwacja ma nastąpić w odrębnej
`READ COMMITTED / READ WRITE` transakcji od close i źródła.

Nowy moduł
[`test_only_independent_cca_close_lease.py`](../../bot_core/test_only_independent_cca_close_lease.py)
jest wyłącznie demonstratorem tego **porządku wywołań**. Nie
implementuje interfejsu produkcyjnego `IndependentCommittedCloseReader`,
nie zna chronionego source loggera i nie może wytworzyć podpisanej
decyzji. Przyjmuje `IndependentCCAReaderCatalogPin` oraz
`SyntheticCloseExpectation`, stworzone z jednorazowych fixture'ów,
nigdy z caller-selected produkcyjnego inputu.

Kontekst `hold_synthetic_committed_close`:
1. W obrębie **już rozpoczętej transakcji wywołującego** potwierdza
   katalog, role OID, fizyczne coordinates i `SECURITY DEFINER`
   przez inspektor S9-RPI-11.
2. Wywołuje testowy `read_exact_close(text)`, który wykonuje
   `SELECT ... FOR UPDATE` na odpowiadającym lifecycle slocie.
3. Porównuje dokładny `operation_id`, `semantic_role`,
   `PENDING_CLOSED`, `close_xid`, `request_sha256` oraz oryginalne
   zachowane PostgreSQL16 `request::text` UTF-8 bytes z osobną
   syntetyczną oczekiwaną wartością. Waliduje także identity, generation,
   predecessor/target state, bez zastępowania zachowanych bajtów
   ponowną serializacją JSON.
4. Odrzuca aliasowanie `close_xid`, `source_observation_xid`
   i `observation_xid`. Kontroluje, że obserwacja oraz callback
   wywołującego pozostają w tej samej fizycznej transakcji.
5. Zwraca `SyntheticHeldCloseEvidence` zawsze z
   `status=NOT_QUALIFIED`, `production_authorization=BLOCKED` i
   `close_lease_qualified=false`.

Własność PostgreSQL row lock jest związana z **transakcją SQL**, a nie
samym wyjściem z Pythona `with`. Zamknięcie Python context managera
nie zwalnia blokady; następuje to dopiero przy `COMMIT`/`ROLLBACK`
transakcji należącej do wywołującego. Wzorzec future production wymaga
utrzymania tego lock aż do zakończenia dokładnego source/CCA porównania.

## Dowody testowe

[`test_postgresql_independent_cca_close_lease.py`](../../tests/security/test_postgresql_independent_cca_close_lease.py)
wykorzystuje **rzeczywisty PostgreSQL 16** i odrębne role z jednorazowej
bazy S9-RPI-10:
- pozytywne porównanie oryginalnych `jsonb::text` bytes, request SHA256,
  roli, original close XID, osobnego source i CCA observation XID,
- utrzymanie blokady przy dwóch próbach `UPDATE slots` przez administratora:
  przed wyjściem z context managera oraz po nim, ale przed `COMMIT`;
  dopiero po `COMMIT` aktualizacja jest dozwolona,
- fail-closed dla niezgodności request bytes, digest, close XID,
  identity i generation oraz niepoprawnych stanów,
- fail-closed dla aliasowania source/CCA XID, nieistniejącej operacji
  i niezgodnej przypiętej definicji SQL.

Nowy test jest wpisany do PostgreSQL atomic Root Proof CI, kontrolowanego
formatowania, coverage i inwentarza Stage 9. **Wyniki pozostają
niepotwierdzone do końca workflow.**

## Nieukończone gwarancje (BEZ GO)

To **syntetyczne źródło**, a nie authentically retained `source event`.
Brak protected policy loadera, instalacyjnych pinned SQL migration,
procesowego peer/auth UID/GID, niezależnego durable source lease,
historycznej eligibility, podpisania i trwałego logu decyzji CCA.
Nawet identyczny request jest tylko porównaniem bajtów, **nie**
dowodem rzeczywistej custody mutation.

Nadal istnieje TOCTOU między inspekcją katalogu a wywołaniem funkcji.
Produkcyjna ścieżka będzie musiała zabezpieczyć concurrent catalog DDL,
przeprowadzić niezależną instalacyjną kwalifikację, dowieść
autentyczności pinów, uzyskać source lease/credential eligibility i
przestrzegać atomowego lock order do trwałego zapisu decyzji. Tego PR
nie wolno promować jako installed/qualified CCA.

Nie zmieniono funkcji produkcyjnych, frozen JSON, `complete_close`,
publicznego SIGN/BIND ani `deployment/stage9_current_status.json`.

## Sprzęt

**DODATKOWY KOMPUTER TERAZ: NIE.** Testy syntetyczne działają na
izolowanym PostgreSQL 16 w GitHub Actions. Można je uruchomić
lokalnie, posiadając testową bazę z możliwością zarządzania
tymczasowymi rolami.

**PÓŹNIEJ: WYMAGANE NATYWNE LINUX/SYSTEMD** do rzeczywistej
kwalifikacji procesu, peer auth, systemd isolation, role OID/OS ACL,
protected keys, crash/restart durability i instalacyjnego catalog pin.
Jedna odpowiednio odizolowana maszyna Linux może wystarczyć w
ograniczonym modelu `PRODUCTION_LOCAL`, lecz nie dowodzi
niezależnego rollback/trust domain.
