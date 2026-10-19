# Stage 9 — S9-RPI-09: niezależny committed-close reader CCA (kontrakt V1)

**Stan:** `DESIGN_ONLY_UNQUALIFIED` · **Decyzja produktowa:** `PRODUCTION_NO_GO`

Po scaleniu [S9-RPI-08](contracts/stage9_root_proof_transaction_scope_addendum_v1.md)
następną krytyczną zależnością jest oddzielny, zaufany odczyt
`PENDING_CLOSED` w PostgreSQL 16 przez przyszłą Custody Completion
Authority (CCA). Ten PR wprowadza **zamknięty, maszynowo sprawdzany kontrakt
projektowy** i testy istniejących odmów. Nie instaluje readera SQL, nie
zmienia uprawnień ani statusu runtime.

[Kontrakt JSON V1](contracts/stage9_independent_committed_close_reader_v1.json)
ustala zakres niezależnego review. Pole `review_state` pozostaje
`DESIGN_ONLY_UNQUALIFIED` także po merge; sam dokument nie jest
instalacyjnym pinem ani capability.

## Dlaczego nie możemy użyć istniejącego administratora?

Obecne `read_operation(text,text,text)` jest funkcją
administratora lifecycle. Wymaga `READ COMMITTED` i blokady slotu, zanim
odczyta exact operation. Zwraca także historyczne
`LEGACY_UNVERIFIED_COMPLETION`, którego CCA nie może zaakceptować jako
`PENDING_CLOSED`. `complete_close(text,text,bytea)` jest celową odmową
`INDEPENDENT_CUSTODY_COMPLETION_AUTHORITY_REQUIRED`, SQLSTATE `42501`.

Przyszły reader CCA musi być odrębnym, minimalnym direct-login principal
z własnym OID, peer authentication, fizycznym pinem PostgreSQL, approved
catalog pin i **wyłącznie EXECUTE** na osobno zatwierdzonej funkcji SQL.
Konto CCA nie otrzymuje bezpośredniego SELECT/UPDATE/DML na tabelach,
uprawnień schema ownera ani alternatywnych write paths. Nie
wystarczy przepisać administratorowego `resolve_close()`, nazwać to
`independent` lub zaakceptować caller-provided credentials/role name.
Source reader, CCA reader, administrator, issuer writer, obaj signerzy i
właściciel schematu nie mogą się utożsamiać.

## Dokładny odczyt i transakcja

- Reader działa w **oddzielnej transakcji `READ COMMITTED`** po
  potwierdzonym COMMIT zamknięcia. Sam snapshot `NOT_FOUND`, wynik
  `timeout` albo in-flight COMMIT pozostają `OUTCOME_UNKNOWN`.
- Blokada exact slot/operation i weryfikacja role/identity/predecessor
  muszą obejmować niezależny odczyt oraz porównanie retained evidence;
  żywy lease/fence ma odrębną kontrolę i lock order.
- **Blokad wierszy nie można wykonywać w transakcji `READ ONLY`.**
  Przyszły backend transakcyjny wymaga `READ WRITE` wyłącznie ze względu
  na `SELECT ... FOR UPDATE/SHARE`; reviewed owner-owned `SECURITY DEFINER`
  wykonuje lock z własnym minimalnym prawem, a konto CCA ma jedynie
  ograniczony `EXECUTE`, bez uprawnienia do bezpośredniego DML. Bez
  faktycznego testu GRANT/REVOKE i rzeczywistego locka nie ma kwalifikacji.
- Przechowujemy oryginalne bytes **`request::text` zakodowane jako UTF-8
  przez PostgreSQL 16**. To nie są oryginalne bajty JSON z requestu klienta;
  re-serializacja po stronie Pythona nie zastąpi retention.
- Dokładnie sprawdzamy operation ID, rolę, identity, generation, before/after,
  oryginalne request bytes/digest, positive `close_xid`, osobne
  `observation_xid`, fizyczne DB/catalog piny i odrębność od source readera.
- Reader nie ma uprawnień do `COMPLETE`, BIND, issuer SIGN, mutation custody,
  enrollment, history issuance ani migracji katalogu. Brak evidence, drift,
  alias, role OID collision czy wynik niepewny odmawiają.
- Poprawnie dopasowane publiczne pola są **jedynie deklaracją formatu**;
  nie dowodzą instalacji, commit durability, provenance czy authorizacji
  CCA. Brak ścieżki, która zwraca `QUALIFIED`.

## Wymagany osobny etap instalacyjny

Przyszła implementacja musi przejść niezależny review i migrację katalogu:
minimalne role/OID, `SECURITY DEFINER` z reviewed source, kontrola
`READ WRITE` wyłącznie dla row locks i odmowa każdej bezpośredniej mutacji
przez CCA readera, direct peer login,
ochronę katalogu i zewnętrznych pinów, odmowę alternatywnych portów
DML/DDL, testy `READ COMMITTED` pod kontencją, COMMIT-response-loss,
crash/restart, exact `jsonb::text`, zamianę close/operation, role aliases
oraz próbę odczytu przez obcy principal. Trzeba też zintegrować
credential eligibility, source lease i trwały CCA log. Testy kontraktowe
nie zastępują tych dowodów.

Kontrakt wymaga następujących zabezpieczeń przyszłej funkcji
`SECURITY DEFINER`:

- Stały, zweryfikowany w review `search_path` musi być zadeklarowany na
  samej funkcji: najpierw `pg_catalog`, następnie wyłącznie zatwierdzone,
  chronione schematy przypiętego katalogu, a jawne `pg_temp` na końcu.
  Nie wolno korzystać z `$user`, `public`, niezaufanych schematów ani
  ścieżki wywołującego lub sesji. Pominięcie `pg_temp` jest niedopuszczalne,
  ponieważ PostgreSQL może wtedy przeszukiwać schemat tymczasowy przed
  pozostałymi schematami dla relacji i typów. Segment kontraktu
  `REVIEWED_INSTALLATION_PINNED_TRUSTED_SCHEMAS` oznacza zatwierdzone
  schematy, w których nieuprawnieni nie mają `CREATE`; nie jest nazwą SQL.
- `EXECUTE` musi zostać odebrane `PUBLIC` i wszystkim nieuprawnionym
  principalom, a następnie nadane wyłącznie właściwemu, instalacyjnie
  przypiętemu direct-login principalowi CCA, zweryfikowanemu przez OID
  oraz istniejące piny peer authentication, fizycznej bazy i katalogu.
  Szeroki grant ani dostęp odziedziczony przez nieuprawnione role nie
  spełniają kontraktu; dotyczy to także obejścia przez `SET ROLE` i starych
  grantów zachowanych przez `CREATE OR REPLACE`. Wyłączność dotyczy grantów
  `EXECUTE`, przy zachowaniu odrębnego, zaufanego właściciela funkcji.
- `CREATE`, `REVOKE` i `GRANT` muszą należeć do jednej atomowej transakcji
  instalacyjnej, z kompletem ograniczeń przed `COMMIT` i bez okna
  nieautoryzowanego `EXECUTE`. Błąd dowolnego kroku wymaga rollbacku
  całej instalacji.

Kwalifikacja musi odrzucać zmianę kolejności lub poluzowanie `search_path`,
pominięcie `pg_temp`, zachowanie nieuprawnionego `EXECUTE`, grant dla
nieprzypiętego lub niewłaściwego principala oraz publikację funkcji przed
zakończeniem ograniczania dostępu. Są to wymagania projektowe przyszłej
instalacji; ten kontrakt nie tworzy funkcji SQL, konta PostgreSQL ani migracji.

Testy w
[`test_stage9_independent_committed_close_reader_contract.py`](../../tests/architecture/test_stage9_independent_committed_close_reader_contract.py)
sprawdzają zamkniętą specyfikację, niebezpieczne mutacje, status Stage 9,
istniejące `ADMINISTRATOR_FUNCTIONS`/`ISSUER_FUNCTIONS`, tombstone SQL
i odmowę duplikatów JSON. Nie używają produkcyjnego SQL readera ani key
material. Frozen canonical JSON, `deployment/stage9_current_status.json`,
przywileje, CI thresholds i dotychczasowe produkcyjne odmowy pozostają bez zmian.

## HARDWARE_DEPENDENCY

**Ten etap można wykonać lokalnie na Windows 11** i w GitHub Linux CI,
bez drugiego komputera. Testy kontraktowe nie wykonują native systemd ani
destructive/reboot checks.

Późniejsza kwalifikacja zainstalowanego readera, CCA i źródeł będzie
wymagała natywnego Linux z systemd, rzeczywistych odrębnych UID/GID/OID,
ACL, key custody i kwalifikowanych PostgreSQL 16 credentials. Profil
`PRODUCTION_LOCAL` może być badany na jednym należycie przygotowanym
hoście; silniejsza ochrona przed pełnym host rollback wymaga osobnego
security domain i niezależnej decyzji architektonicznej.
