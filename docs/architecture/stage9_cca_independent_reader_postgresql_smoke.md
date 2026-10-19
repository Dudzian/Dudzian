# Stage 9 — S9-RPI-10: PostgreSQL 16 CCA reader, izolowany eksperyment

**Stan:** `TEST_ONLY_NON_AUTHORIZING`. **Produkcja:** `PRODUCTION_NO_GO`.

Kontynuacja [S9-RPI-09](stage9_independent_committed_close_reader_v1.md):
uruchamiamy *rzeczywisty PostgreSQL 16*, aby sprawdzić fizyczne własności
`READ COMMITTED`, `SECURITY DEFINER`, grantów oraz blokady slotu.
Poprzedni PR ustalił kontrakt; niniejszy PR wprowadza **wyłącznie
jednorazowy harness testowy**, nie produkcyjny reader i nie migrację.

## Zakres dowodu

[`tests/security/test_postgresql_independent_cca_reader_smoke.py`](../../tests/security/test_postgresql_independent_cca_reader_smoke.py)
tworzy losowo nazwaną, osobną bazę, zamknięty schema, prywatnego
owner `NOLOGIN`, direct-login CCA reader i dwóch nieuprawnionych
direct-login principalów (source reader oraz obca rola). Role istnieją
tylko w tym teście; fixture usuwa całą bazę i role po wykonaniu.

1. W **jednym transakcyjnym bootstrapie** test tworzy funkcję
   `SECURITY DEFINER`, odbiera `EXECUTE` od `PUBLIC`, udziela
   wyłącznego grantu CCA i dopiero wtedy publikuje COMMIT.
   `search_path` jest przypięty do `pg_catalog`, zaufanego
   dedykowanego schematu i `pg_temp` na końcu.
2. CCA w osobnej `READ COMMITTED / READ WRITE` transakcji wywołuje
   wyłącznie tę funkcję. Owner wykonuje `SELECT ... FOR UPDATE` na
   slot; blokada żyje aż do końca **transakcji wywołującego**.
3. Fixture utrwala `PENDING_CLOSED` w wcześniejszym, odrębnym
   committed xid. Odpowiedź zachowuje PostgreSQL `jsonb::text` UTF-8,
   jego SHA-256, dokładny operation ID i oddzielny observation xid;
   nie próbuje odtworzyć zewnętrznych client-wire JSON bytes.
4. Source i obca rola nie mogą wykonać funkcji; rola CCA nie może
   wykonywać nawet bezpośredniego SELECT na tabelach, tym bardziej UPDATE.
   Inspekcja `pg_proc` potwierdza `SECURITY DEFINER`, owner, pin
   `search_path` i ograniczony `EXECUTE`.
5. Funkcja odmawia niewłaściwej izolacji oraz nieistniejącej operacji.
   Osobne połączenie napotka `lock_timeout` przy próbie modyfikacji
   slotu, gdy CCA nadal trzyma blokadę; po zakończeniu transakcji
   aktualizacja ponownie jest możliwa.

Niniejszy test nie używa produkcyjnego SQL schema, nie modyfikuje
`postgresql_production_root_lifecycle_schema.py`, `complete_close`,
`read_operation`, frozen JSON, ACL istniejącej instalacji, enrollment,
signing lub tokenów CCA. Synthetic close jest wpisem superusera do
jednorazowej bazy; **nie dowodzi** autentyczności ani uprzedniej trwałości
wykonania realnej custody mutation.

## Co jeszcze blokuje realną kwalifikację

Nie występuje tutaj rzeczywisty protected installation pin, dynamiczne
źródło identities, historyczny credential lease, source log provenance,
role-OID pinning cross-process, OS peer attestation, obserwacja in-flight
COMMIT i crash/restart recovery, analiza pełnego katalogu SQL, executable
service, systemd ani signed CCA decision. Sam fakt, że synthetic
`SECURITY DEFINER` i row-lock działają, nie upoważnia do kopiowania tej
funkcji do instalacji produkcyjnej.

Wymagane są **osobna reviewed production migration, nowy katalog i
installer pin, real privilege/role qualification oraz live negative tests**
zobowiązań z [kontraktu S9-RPI-09](contracts/stage9_independent_committed_close_reader_v1.json).

Test został dodany do seryjnej kwalifikacji `PostgreSQL atomic Root Proof
commit`, listy ścisłych zależności i zakresu Ruff format; bez wyłączania
istniejących kontroli.

## HARDWARE_DEPENDENCY

**Teraz dodatkowy komputer nie jest potrzebny**: GitHub Linux CI posiada
izolowany PostgreSQL 16, a sam harness jest uruchamiany bez produkcyjnego
systemd. Próba odtworzenia tych testów lokalnie wymaga instancji testowej
PostgreSQL 16 z uprawnieniami do tworzenia/usuwania testowych baz i ról.

**Przed fizyczną kwalifikacją produkcyjną** potrzebny będzie natywny Linux
z systemd, PostgreSQL 16, rzeczywiste osobne UID/GID/role OID, protected
keys/pins, peering oraz testy instalacji i odzyskiwania po crash. Jeden
poprawnie odizolowany host może wystarczyć dla modelu
`PRODUCTION_LOCAL`; inny rollback domain wymaga dodatkowej decyzji
bezpieczeństwa. **Nie traktuj CI ani merge jako `PRODUCTION_GO`.**
