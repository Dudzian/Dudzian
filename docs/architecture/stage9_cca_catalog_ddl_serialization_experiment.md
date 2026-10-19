# Stage 9 — S9-RPI-13: CCA reader i migracje DDL, syntetyczne blokady

**Stan:** `TEST_ONLY_NON_AUTHORIZING`. **Produkcja:** `PRODUCTION_NO_GO`.

Kontynuacja [S9-RPI-12](stage9_cca_close_transaction_lease_experiment.md)
po scaleniu PR #3167. Celem jest sprawdzenie, jakie gwarancje można,
a jakich **nie można**, uzyskać pomiędzy inspekcją katalogu
`SECURITY DEFINER`, wykonaniem `read_exact_close()` i zakończeniem
utrzymywanej transakcji.

## Model i obowiązujące ograniczenia

W PostgreSQL `pg_advisory_xact_lock_shared` i
`pg_advisory_xact_lock` są blokadami aplikacyjnymi: same `GRANT`,
`REVOKE`, `ALTER FUNCTION` i `CREATE OR REPLACE FUNCTION` **nie
muszą ich respektować**. Nie są to wymuszone uprawnieniami blokady
systemowego katalogu `pg_proc`, dowód pinów produkcyjnych ani
zabezpieczenie przed rolą administratora, która omija uzgodniony
protokół. Nie wolno promować ich do zabezpieczenia przed
niezaufanym DDL albo unqualified production caller.

[`test_only_cca_catalog_ddl_gate.py`](../../bot_core/test_only_cca_catalog_ddl_gate.py)
udostępnia dwa jawne, nieautoryzujące wywołania:

- `acquire_synthetic_reader_catalog_gate()` — shared advisory lock
  na czas transakcji odczytu.
- `acquire_synthetic_reviewed_ddl_gate()` — exclusive advisory lock
  na czas migracji, **wyłącznie dla współpracującego, reviewed DDL
  writera**. Nie nakłada grantów ani nie uruchamia migracji.

Para kluczy blokady jest deterministycznym, ograniczonym do testu
hashem domeny, fizycznej tożsamości, installation ID, schematu i
function OID z przypiętego `IndependentCCAReaderCatalogPin`.
Nie zapewnia kryptograficznego uwierzytelnienia ownera, migrującego
procesu ani niezależnego protected pin. Nie jest production lock namespace.

W [`hold_synthetic_committed_close()`](../../bot_core/test_only_independent_cca_close_lease.py)
gate zostaje zajęty przed pierwszą inspekcją katalogu.
Druga inspekcja następuje po SQL `read_exact_close()` oraz
porównaniu `PENDING_CLOSED`, a trzecia po zakończeniu callbacku
wywołującego, przed pomyślnym wyjściem z context managera.
Shared advisory lock i `FOR UPDATE` slot lock obowiązują aż do
`COMMIT`/`ROLLBACK` tej samej transakcji.

## Pokrycie testowe na PostgreSQL 16

[`test_postgresql_cca_catalog_ddl_gate.py`](../../tests/security/test_postgresql_cca_catalog_ddl_gate.py)
wykorzystuje izolowaną jednorazową bazę S9-RPI-10:
- reviewed DDL writer czeka na zakończenie całej transakcji readera,
  także po wyjściu z pythonowego kontekstu;
- exclusive gate migratora blokuje rozpoczęcie reader gate;
- nieskooperujący administrator wykonuje `GRANT PUBLIC EXECUTE`
  **po pierwszej inspekcji**, ale przed SQL readerem; ponowna
  inspekcja wykrywa zmianę i odmawia zwrotu dowodu;
- analogiczna zmiana **podczas callbacku** jest wykrywana
  przed pomyślnym zakończeniem `with`;
- graf importów produkcyjnego verifi­era CCA nie sięga testowego
  modułu advisory gate.

Nie ma testu, który udowadnia fizyczną niemożliwość obejścia
blokady przez samodzielne `GRANT`/DDL, ponieważ taka
niemożliwość **nie jest własnością blokad doradczych**.
Możliwy jest również `ABA` — szybka zmiana i przywrócenie
pierwotnej definicji pomiędzy inspekcjami. Pomyślne porównanie
SHA-256 dwóch snapshotów nie wyklucza zmian pośrednich.

## Zależności przed produkcyjnym GO

Produkcyjne rozwiązanie wymaga instalacyjnej migracji reviewed
i ochrony **wszystkich** dróg do DDL, dokładnego zewnętrznego pinu
oraz egzekwowalnej separacji principalów. Osobny peer/UID/GID
i katalog SQL muszą być kwalifikowane na natywnym Linux/systemd;
sam synthetic reviewer nie potwierdza braku alternate write ports.
Brak również niezależnego uwierzytelnionego source event, source
custody/credential eligibility lease, durable independent CCA
decision log oraz crash/restart recovery.

Ten PR nie tworzy produkcyjnych funkcji SQL, grantów, ról,
migracji, poświadczeń, SIGN/BIND/COMPLETE ani enrollment.
Nie zmienia frozen contracts czy
`deployment/stage9_current_status.json`. Całe Stage 9 pozostaje
`IN_PROGRESS / PRODUCTION_NO_GO`.

## Sprzęt

**DODATKOWY KOMPUTER: NIE POTRZEBNY teraz.** Testy działają
w istniejącym PostgreSQL 16 na GitHub Linux CI.
Fizyczna kwalifikacja production potrzebuje natywnego
Linux/systemd i rzeczywistych principalów/ACL/pinów.
Jeden właściwie izolowany host może obsłużyć ograniczony
`PRODUCTION_LOCAL`, ale nie udowadnia odrębnego rollback domain.
