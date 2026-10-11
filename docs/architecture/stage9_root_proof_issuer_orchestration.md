# Stage 9: admission i bezpieczna orkiestracja RootProofIssuer

Publiczne API `orchestrate_root_proof_issuance(binding)` przyjmuje wyłącznie
autentyczny, opaque `VerifiedAccountGenesisInitialBinding`. Granica CHA ponownie
weryfikuje binding; raport preflight lub reconciliation, samodzielnie zbudowany
obiekt, surowe dane i provider wybrany przez wywołującego nie są capability.
Zaufany composition root ustala autorytety. Brak wymaganej kompozycji oznacza
odmowę, bez providera zastępczego i bez wygenerowanego signera produkcyjnego.

Admission wymaga zachowanej dokładnej próby
`SIGNED_IMMUTABLE_DURABLE_NOT_SENT`. Weryfikacja obejmuje retained canonical
request, immutable attempt identity, current pointer i fence, autentyczny
InitialBinding oraz podpisy requester i claimant przez istniejące weryfikatory.
Niepełne próby pozostają w zachowanych stanach; orkiestrator nie generuje nowego
identyfikatora, nie wykonuje replacement i nie promuje ich do signed immutable.

## Decyzja przed pierwszą mutacją

Przed pierwszą mutacją orkiestrator porównuje trzy kolejne odczyty autorytatywne
oraz na nowo odtwarza żywą authorization przez istniejącą granicę issuer preflight.
Weryfikuje dokładne
ACTIVE credentials, rozdzielenie ról i materiału kluczowego, pełny zakres,
provenance oraz ACTIVE/UNBOUND entitlement. Odczyty obejmują PostgreSQL
Entitlement Registry, durable CHA, authenticated Issuer History i Checkpoint V2.
Zmiana rewizji, current attempt, digestu, lifecycle albo wyniku odczytu odmawia
dalszego działania. Historyczny raport nie zastępuje żywej authorization.

Zgodne odczyty są punktami odczytowymi, nie wspólną transakcją magazynów.
Nie tworzą globalnego snapshotu ani uprawnienia do emisji. Nadal wymagane byłyby
świeże authorization i authority-owned CAS na faktycznej granicy mutacji.

## Recovery i granica emisji

Recovery porównuje dwa kolejne odczyty. Klasyfikator wykorzystuje dokładny
predecessor CAS z autorytatywnego
rejestru, zachowaną lineage i BOUND tuple oraz uwierzytelnioną historię
z Checkpoint V2 i historycznymi potwierdzeniami. Porównuje dokładną operację,
account, request, InitialBinding, issuance attempt i pochodzenie bindingu.
Wynik opisuje zgodność lub odmowę; nie zwraca Root Proof i nie nadaje uprawnień.
Rotacja credentials nie jest nową authorization, a `NOT_FOUND`, pusty log
lub obserwacja UNBOUND nie są dowodem pozwalającym na replacement.

Wszystkie punkty recovery są obserwacjami; wszystkie działania emisji pozostają
zablokowane:

| Punkt | Zachowany stan i decyzja |
| --- | --- |
| `BEFORE_FIRST_MUTATION` | Signed NOT_SENT, ACTIVE/UNBOUND i brak dopasowanego wpisu w odczytach; nadal potrzebna żywa authorization. |
| `UNKNOWN_PRIOR_OUTCOME` | `INDETERMINATE`; bez retry, replacement i zgadywania brakującego wyniku. |
| `REGISTRY_BOUND` | Dokładny BOUND tuple i zachowany ACTIVE/UNBOUND predecessor tej samej generacji. |
| `HISTORY_RECORDED` | Dokładny payload, history CAS i podpisany head; własny ogon jeszcze niezaakceptowany. |
| `CHECKPOINT_ACCEPTED_CHA_PENDING` | Własny history head ma potwierdzenie V2; CHA recovery jeszcze niezapisane. |
| `CHA_RECOVERY_RECORDED` | CHA ma stan EXACT_BOUND_RECOVERED i zgodne odczyty pozostałych autorytetów; nadal brak dowodu zachowanych proof bytes. |
| `UNRESOLVED` | Brak wystarczających lub spójnych danych; odmowa. |

Każdy etap po BOUND nadal wymaga oryginalnych signed Root Proof bytes;
`ROOT_PROOF_BYTES_NOT_RETAINED` i `CHA_FINALIZATION_BLOCKED` blokują finalizację.

Prymitywy Entitlement BIND, signing, authenticated history append i checkpoint
advance już istnieją. Nie ma jednak pełnej semantic composition realizującej
zamrożony krok
`8_ATOMICALLY_BIND_AND_PERSIST_IDENTICAL_SIGNED_PROOF_AND_HISTORY`:
authority-owned atomic retention identycznych signed Root Proof bytes razem
z bindingiem i historią. Dlatego wszystkie mutacje emisji pozostają zablokowane,
w tym BIND, signing, append, advance, transport, proof admission i account commit.
Signed request bytes ani generic history observation nie dowodzą zachowania
signed Root Proof. Raport reconciliation, także opisujący zgodność danych,
pozostaje obserwacją i nigdy nie jest grantem.

## Status i ograniczenia

`CODE_READY_FOR_REVIEW` dotyczy tego wycinka kodu dopiero po przejściu testów
i kontroli jakości. `ROOT_PROOF_ISSUANCE_AVAILABLE=False` oraz
`PRODUCTION_READY=False`; Stage 9 pozostaje nieukończony. Root Proof nie jest
wydany. Zamrożone kontrakty i `deployment/stage9_current_status.json` zachowują
dotychczasowe bajty i statusy.

Testy wykorzystują izolowany rzeczywisty PostgreSQL i durable SQLite oraz
istniejące testowe granice custody. Nie dowodzą provisioning, transportu ani
kwalifikacji produkcyjnej. Pełny skoordynowany restore poprawnego prefiksu
wszystkich magazynów po restarcie wymaga niezależnego świadka anti-rollback
poza tym hostem; zgodność lokalnych odczytów nie rozwiązuje tego ograniczenia.

## Walidacja i CI

Po instalacji zależności testowych skonfiguruj `DUDZIAN_TEST_POSTGRES_DSN`
do izolowanego rzeczywistego PostgreSQL 16, z `fsync`, `synchronous_commit`
i `full_page_writes` ustawionymi na `on`. Lokalna walidacja może używać
zewnętrznego testowego pluginu uwierzytelniającego SCRAM; nie jest on częścią
repozytorium ani konfiguracji CI. Z katalogu repozytorium:

```bash
export PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 DUDZIAN_TEST_MODE=1 DUDZIAN_ALLOW_LONG_POLL=0
: "${DUDZIAN_TEST_POSTGRES_DSN:?Ustaw DSN izolowanego testowego PostgreSQL}"
python -m pytest -p asyncio -p pytest_cov -q \
  tests/security/test_root_proof_issuer_orchestration.py \
  -m 'not hardware and not destructive and not network' \
  --cov=bot_core.root_proof_issuer_orchestration \
  --cov=bot_core.root_proof_issuer_recovery --cov-branch
python -m pytest -p asyncio -p pytest_cov -q \
  tests/security/test_root_proof_issuer_orchestration_integration.py \
  -m 'not hardware and not destructive and not network' \
  --cov=bot_core.root_proof_issuer_orchestration \
  --cov=bot_core.root_proof_issuer_recovery --cov-branch
python -m ruff check \
  bot_core/root_proof_issuer_orchestration.py bot_core/root_proof_issuer_recovery.py \
  tests/security/test_root_proof_issuer_orchestration.py \
  tests/security/test_root_proof_issuer_orchestration_integration.py
python -m ruff format --check \
  bot_core/root_proof_issuer_orchestration.py bot_core/root_proof_issuer_recovery.py \
  tests/security/test_root_proof_issuer_orchestration.py \
  tests/security/test_root_proof_issuer_orchestration_integration.py
python -m mypy
python -m mypy --platform win32
```

Testy silnika działają serialnie z rzeczywistym PostgreSQL w istniejącym
`quality-ratchet`; przypadek publicznego API z pełną autentyczną lineage jest
w istniejącym `semantic-issuer-preflight`. Oba moduły wchodzą do Ruff format,
configured mypy scope i coverage ratchet. Zachowane są dotychczasowe testy,
permissions, timeouty i bramki bezpieczeństwa, także Stage 10 i hardware opt-in.
