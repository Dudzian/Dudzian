# Stage 9: odczytowa Cross-Authority Reconciliation

`reconcile_root_proof_issuance_attempt(binding)` przyjmuje wyłącznie autentyczny
`VerifiedAccountGenesisInitialBinding`. Zaufany composition root wybiera
PostgreSQL Entitlement Registry, trwały CHA Attempt Store, PostgreSQL
Authenticated Issuer History oraz powiązany PostgreSQL Checkpoint Authority V2.
Wywołujący nie przekazuje autorytetów, kluczy, snapshotów ani evidence.

Moduł zwraca niemutowalny `RootProofIssuanceReconciliationReport`. Raport opisuje
identyfikatory, rewizje, digesty oraz przyczyny odmowy. Nigdy nie autoryzuje
emisji Root Proof, podpisania, Entitlement BIND, replacement, zmiany checkpointu
ani transportu. Także `EXACT_COMMITTED` opisuje zgodność zachowanych danych;
nie jest dowodem Account Genesis COMMITTED ani capability.

## Odczyty i weryfikacja

- InitialBinding jest ponownie weryfikowany przez istniejącą granicę CHA.
- CHA otwierany jest przez `SQLiteCHAAttemptStore.open_readonly`: istniejąca
  baza v6, `mode=ro`, `query_only`, bez tworzenia bazy, chmod i migracji.
  Weryfikowane są istniejące kontrakty schematu i retained chain. Odczyt WAL
  zachowuje widoczność zatwierdzonych danych.
- Rejestr udostępnia `authoritative_state` oraz `retained_history`; istniejąca
  walidacja obejmuje pełną lineage i rewizje. Zachowane podpisy requestu
  porównywane są z publicznymi, historycznymi credential generations.
- `PostgreSQLAuthenticatedIssuerHistory.authenticated_read_snapshot`
  sprawdza cały łańcuch, predecessor digests i wszystkie retained attestations
  przy użyciu dotychczasowych weryfikatorów podpisów.
- `PostgreSQLCheckpointAuthority.authenticated_read_snapshot` sprawdza
  zachowane rewizje, predecessor CAS, podpisy i potwierdzenia V2. Historycznie
  odwołany signer wymaga dokładnego potwierdzenia przyjęcia podczas ACTIVE,
  z prawidłowym porządkiem lifecycle generations.

Porównanie obejmuje environment, trust_domain, product_scope, entitlement,
account, logical operation, issuance attempt, InitialBinding i request.
Lokalna projekcja `ROOT_PROOF_ISSUANCE_BINDING_OBSERVATION` zawiera pełną
authorization, zachowany stan entitlementu, BindPredecessor oraz digest
immutable attempt. Służy tylko do odczytowego porównania istniejącego wpisu.
Nie definiuje nowego zamrożonego kontraktu emisji ani metody append.

## Znaczenie wyników

| Wynik | Znaczenie odczytowe |
| --- | --- |
| `NOT_FOUND` + `PRE_ISSUANCE_OBSERVED` | Zachowana podpisana próba ma stan `SIGNED_IMMUTABLE_DURABLE_NOT_SENT`, rejestr jest ACTIVE/UNBOUND, brak wpisu wydania dla tej operacji. To prawidłowy etap przed wydaniem. |
| `EXACT_COMMITTED` + `ALIGNED_COMMITTED_OBSERVED` | Dokładny BOUND tuple, retained history i checkpoint są zgodne. Raport nie nadaje uprawnień. |
| `NOT_FOUND` + `REFUSED` | Brakuje konkretnego wymaganego wpisu lub checkpointu. Brak danych nie dowodzi UNBOUND. |
| `STALE` | Uwierzytelniona historia wyprzedza zachowany checkpoint lub zachowany stan operacji jest nieaktualny. |
| `CONFLICT` | Poprawne strukturalnie dane odnoszą się do innej tożsamości lub niezgodnego tuple. |
| `CORRUPT` | Uszkodzony łańcuch, digest, podpis, CAS, schema lub relacja między history a checkpoint. |
| `UNAVAILABLE` | Wybrany autorytet lub wymagany weryfikator nie dostarcza danych. |
| `INDETERMINATE` | Zaobserwowano zmianę między odczytami albo wynik operacji pozostaje nieznany. |

Istotne dane są odczytywane ponownie. Różnica rewizji, tożsamości lub digestu
powoduje fail-closed. Zgodne próbki **nie dowodzą globalnej atomowości**:
transakcje kilku autorytetów nie tworzą wspólnego snapshotu, także gdy każda
lokalnie używa REPEATABLE READ. `atomic_global_snapshot=False` i
`authorizes_operations=False` pozostają jawne w każdym raporcie.

## Granice

Raport nie jest `RootProofIssuanceReconciliationEvidenceV1`. Zamrożony dowód
replacement wymaga osobnego uwierzytelnienia, aktualnego history fence i
autorytatywnego pozytywnego wyniku `AUTHORITATIVELY_UNBOUND`. NOT_FOUND,
pusty log i preflight nie spełniają tego warunku. CHA zachowuje digest
authorization evidence, ale nie jej canonical bytes; moduł nie odtwarza z tego
trwałego dowodu replacement.

Nie dodano uprawnień PostgreSQL. Moduł nie wykonuje emisji, BIND, append,
advance, signing, transportu ani Stage 10. Nie zmienia zamrożonych kontraktów,
statusów Stage 9 ani bramek kwalifikacji. Pełny skoordynowany restore poprawnego
prefiksu wszystkich magazynów po restarcie wymaga niezależnego świadka
anti-rollback; ten komponent nie podnosi statusu SERVER_READY.

Testy są celowane: rzeczywisty PostgreSQL 16 oraz durable SQLite, negatywne
ścieżki rekoncyliacji, snapshoty i istniejące regresje zmienionych granic.
Pełną macierz GitHub Actions pozostawiono do niezależnego wykonania dla PR.

## Instalacja i odtwarzanie walidacji

Prywatny, zaufany composition root wdrożenia musi zbudować runtime przez
`_compose_reconciliation_runtime(issuer_authority, history, checkpoint)` i zapisać
go w `_RUNTIME_CONFIGURATION` modułu. Kompozycja sprawdza dokładne typy,
powiązanie history–checkpoint i zakres autorytetów. Publiczny wywołujący nadal
przekazuje wyłącznie InitialBinding. Dla autentycznego InitialBinding brak
zainstalowanej konfiguracji zwraca `UNAVAILABLE` z odmową; nie wybiera providera
zastępczego. Sam komponent nie instaluje konfiguracji wdrożenia.

`test_root_proof_issuance_reconciliation.py` sprawdza prywatny silnik na czterech
rzeczywistych magazynach: PostgreSQL Entitlement Registry, durable SQLite CHA,
PostgreSQL Issuer History i PostgreSQL Checkpoint V2, z rzeczywistymi podpisami.
Wcześniejsza lineage InitialBinding jest wydzielona przez testowy seam
`_read_issuance_reconciliation_source`. Osobny
`test_root_proof_issuance_reconciliation_integration.py` wywołuje publiczne API
z pełną autentyczną lineage i aktywnymi wcześniejszymi weryfikatorami Stage 9.
Ten przypadek jest dołączony do istniejącego serialnego joba
`semantic-issuer-preflight`; testy silnika do serialnego joba `quality-ratchet`.

Po instalacji zależności testowych uruchom PostgreSQL 16 na lokalnym porcie
55432, z `fsync`, `synchronous_commit` i `full_page_writes` ustawionymi na `on`.
Z katalogu repozytorium:

```bash
export PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 DUDZIAN_TEST_MODE=1 DUDZIAN_ALLOW_LONG_POLL=0
export DUDZIAN_TEST_POSTGRES_DSN='host=127.0.0.1 port=55432 dbname=postgres user=postgres'
python -m pytest -p asyncio -p pytest_cov -q \
  tests/security/test_cha_attempt_store_readonly.py \
  tests/security/test_root_proof_issuance_reconciliation.py \
  -m 'not hardware and not destructive and not network'
python -m pytest -p asyncio -p pytest_cov -q \
  tests/security/test_root_proof_issuance_reconciliation_integration.py \
  -m 'not hardware and not destructive and not network'
python -m mypy
python -m mypy --platform win32
```

Pełna lineage ma koszt istniejących wielokrotnych weryfikacji, dlatego publiczny
przypadek pozostaje w jobie semantycznym. Rozszerzenia CI zachowują dotychczasowe
testy, coverage ratchet, security scans, filtry bezpieczeństwa i timeouty.
Nie dodano wyłączeń kontroli ani wyjątków od bramek.
