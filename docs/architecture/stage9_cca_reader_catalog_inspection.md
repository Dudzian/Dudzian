# Stage 9 — S9-RPI-11: niezależny CCA reader, inspekcja katalogu i role OID

**Status:** `TEST_ONLY_NON_AUTHORIZING`; `PRODUCTION_NO_GO`.
Kontynuacja [S9-RPI-10](stage9_cca_independent_reader_postgresql_smoke.md)
oraz kontraktu [S9-RPI-09](stage9_independent_committed_close_reader_v1.md).

## Cel

S9-RPI-10 wykazał w syntetycznym środowisku własności transakcji,
`SECURITY DEFINER` i slot lock. Sam działający testowy reader nie chroni
jednak przed zmianą funkcji, grantu, schematu, OID czy podłączeniem do
obcej bazy. S9-RPI-11 dodaje **samodzielną, pasywną inspekcję** tych
faktów, porównując je z oczekiwaniami dostarczonymi z **zewnętrznej,
chronionej instalacji**.

Moduł:
[`postgresql_independent_cca_reader_inspection.py`](../../bot_core/postgresql_independent_cca_reader_inspection.py).
Rejestr testów:
[`test_postgresql_independent_cca_reader_inspection.py`](../../tests/security/test_postgresql_independent_cca_reader_inspection.py).

## Zabezpieczenia sprawdzane przez moduł

- Wymaga rzeczywistej transakcji `READ COMMITTED / READ WRITE`
  w PostgreSQL 16 i loginu `session_user == current_user ==`
  przypiętej roli CCA, bez `SET ROLE`.
- Weryfikuje `fsync=on`, `synchronous_commit=on`,
  `full_page_writes=on` i brak recovery mode. Prawdziwy system
  identifier, database OID/nazwa, cluster, host/port oraz installation
  ID muszą odpowiadać zewnętrznemu physical pin.
- Wymaga owner `NOLOGIN`, distinct reader/source `LOGIN`,
  dokładnych OID i braku członkostwa ról. Odrzuca uprawnienia
  `SUPERUSER`, `CREATEDB`, `CREATEROLE`, `REPLICATION`,
  `BYPASSRLS`.
- Sprawdza jeden zatwierdzony entrypoint `read_exact_close(text)`
  bez nadmiarowych funkcji/overloadów. Odrzuca inny OID, owner,
  podpis funkcji lub SHA-256 `pg_get_functiondef()`.
- Weryfikuje rzeczywiste ACL funkcji: bez `PUBLIC EXECUTE`,
  obcych grantów i `GRANT OPTION` dla CCA. Wymagany jest dokładny
  `SET search_path=pg_catalog,<trusted schema>,pg_temp` i zakaz
  `CREATE` dla obcych ról wewnątrz zaufanego schematu.
- Odrzuca direct SELECT/DML CCA w tym schemacie, brak `USAGE`
  lub nadane `CREATE` na schemacie oraz `CREATE`/`TEMPORARY`
  w bazie.

Inspekcja wykonuje jedynie odczyty katalogu. Wynik to zawsze
`status=NOT_QUALIFIED`, `production_authorization=BLOCKED`
oraz `reader_installed=false`. Nie istnieje metoda do podniesienia
statusu do `QUALIFIED`, odczyt `PENDING_CLOSED` czy autoryzacja
nowej operacji przez ten moduł.

## Zasięg dowodu i brakujące zależności

Testy używają **wyłącznie jednorazowej bazy i ról** z S9-RPI-10.
Oczekiwania testowe są tworzone przez superusera z fixture i
służą tylko do wykrywania manipulacji. **Nie są zaufanym pinem
produkcyjnym** i nie wolno ich generować z badanej bazy w runtime.

Nawet PASS testów nie dowodzi zaufanego upstream provisioning,
fizycznego peer authentication, kernel UID/GID/ACL, absence of alternate
write paths poza synthetic schema, poprawności rzeczywistego katalogu
lifecycle, jednego held custody/source lease, authorstwa source logów,
crash recovery, log durability, CCA signing, ani legalnego enrollment.
Nie sprawdza globalnego catalog closure wszystkich funkcji i uprawnień
instancji, ani nieruchomego lock order przez trwałą decyzję CCA.

Ten moduł nie instaluje produkcyjnej funkcji SQL, nowych loginów,
grantów, migracji, systemd units, key material, SIGN, BIND czy COMPLETE.
Nie zmieniono `deployment/stage9_current_status.json`, frozen
kontraktów ani żadnej drogi do Root Proof issuance lub Stage 10.
Pełna kwalifikacja nadal wymaga przyszłego reviewed SQL catalog,
zewnętrznego chronionego pinu i instalatora ze sprawdzeniem
rzeczywistych procesów/permissions.

## CI i sprzęt

Moduł i test są włączone do Stage 9 PostgreSQL atomic Root Proof
kwalifikacji oraz coverage, kontroli Ruff i zamkniętego inwentarza CI.
Nie oznaczaj wyników PASS, dopóki job nie skończy się sukcesem.

**HARDWARE_DEPENDENCY:** teraz **nie potrzeba drugiego komputera**;
testowa instancja PostgreSQL 16 i GitHub Linux CI wystarczą. Lokalnie
można testować na Windows z osobną bazą PostgreSQL 16, ale bez
natywnego Linux/systemd nie ma fizycznej kwalifikacji produkcyjnej.
Ta ostatnia wymaga chronionych ról, pins, ACL i live crash/recovery;
sam host Linux może wystarczyć dla ograniczonego `PRODUCTION_LOCAL`.
