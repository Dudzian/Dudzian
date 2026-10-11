# Stage 9 — walidacja trwałego checkpointu i historycznych potwierdzeń V2

Walidacja dotyczy osobnego modułu checkpointu na bazie `main`
`64800572eeefb06f93e6ee80734c9a50809f2b00`, zawierającego scalone PR
#3100, #3101 i #3102. Architekturę, porty oraz granice zaufania opisuje
[dokument checkpointu](checkpoint_authority_production_local.md).

## Wyniki lokalne

Środowisko: Linux, Python 3.12.14, rzeczywisty PostgreSQL 16.15.
Podstawowy klaster ma `fsync=on` oraz `synchronous_commit=on`.
Negatywny test `fsync=off` używa odrębnego izolowanego klastra.

| Zestaw | PASS | FAIL/ERROR | SKIP |
| --- | ---: | ---: | ---: |
| Historia, checkpoint oraz trzy zamrożone kontrakty M0.5 | 782 | 0 | 0 |
| Ostateczny pełny plik checkpointu, po poprawkach overflow i izolacji raportów | 70 | 0 | 0 |
| Custody, substrate, readiness, admission i zamrożone kontrakty podpisanej próby oraz credentials | 475 | 0 | 0 |
| Oddzielny scenariusz SIGKILL serwera PostgreSQL i WAL recovery | 1 | 0 | 0 |

Pierwsze dwa wiersze współdzielą 68 testów. Łącznie zaliczono **1259 unikalnych
testów pytest** i jeden dodatkowy scenariusz awarii serwera. Nie sumujemy
powtórzeń, wcześniejszych bazowych 714 testów ani pomocniczych smoke tests.
Przebieg 782 testów trwał 151,95 s, ostateczne 70 testów 107,78 s,
a dodatkowe 475 testów 18,33 s. Przebiegi coverage miały jedno ostrzeżenie
o ponownym imporcie NumPy przez pandas; bez pomijania lub zmiany asercji.

Coverage dla ostatecznego kodu: **88% zmienionych linii** — 544 objęte
pomiarami linii, 61 brakujących, bramka pozostaje 70%. Branch-aware coverage
nowego adaptera wynosi 88%, a nowego modułu instalacji/kwalifikacji 85%.
Pomiar coverage.py dotyczy Python; nie jest pomiarem gałęzi PL/pgSQL.
Negatywne testy rzeczywistej bazy sprawdzają zachowanie funkcji SQL.

Ruff 0.14.11, formatowanie, rozszerzone E/F/B/I dla nowych plików,
configured mypy Linux i `win32` (po 284 pliki), blocking Semgrep 1.179.0
(993 pliki, zero znalezisk), jego 5/5 fixtures oraz offline Betterleaks 1.9.0
(zero znalezisk) zakończyły się sukcesem. Przypięty SHA256 archiwum
Betterleaks oraz checksum wydawcy sprawdzono przed uruchomieniem.

## Sprawdzone zachowania

- Odczyt po ponownym utworzeniu adaptera oraz uruchomieniu nowego procesu
  zachowuje UUID, rewizje, predecessor digests, evidence i dokładne podpisy.
- Dwa niezależne procesy mają jednego zwycięzcę expected-revision CAS.
  Replay bieżącego nagłówka nie dopisuje evidence. Resolver odtwarza dokładny
  wynik pierwotnej operacji zapisującej rewizję po utracie odpowiedzi.
- Przerwanie procesu przed commitem nie publikuje checkpointu ani evidence;
  przerwanie po commicie zachowuje oba. Brak odpowiedzi lub niedostępność bazy
  nie są traktowane jako nieobecność operacji.
- Korupcja środkowego evidence, checkpointu, podpisu i rekordu historii,
  brakujące evidence, rollback pointer, nieciągłość rewizji oraz zmiana UUID
  blokują odczyt, replay i dalszy zapis.
- ACTIVE, VERIFY_ONLY i historycznie REVOKED zachowują zamrożone reguły.
  Rotacja A→B, późniejsze odwołanie A i restart z retained portami A/B nie
  usuwają oryginalnego podpisu i exact-head acceptance A.
- Pełne stream/signer identities, role, namespace i key material bindings
  są sprawdzane; obce porty, DTO, fake oraz subclass nie nadają uprawnień.
- Zmiany SQL, matching lecz sfabrykowany manifest, typów wynikowych SQL,
  constraints z przeliczonym fingerprintem, ACL, membership, ownership,
  obiektów schematu i ustawień trwałości powodują fail-closed.
- Spójny rollback całego checkpoint prefix wykrywa wcześniej obserwująca
  instancja. Nowa instancja po takim restore nie ma niezależnego witness;
  test zachowuje tę granicę zamiast deklarować niezależny anti-rollback.

Osobny klaster testowy przeszedł `SIGKILL`, ponowne uruchomienie i automatyczne
WAL recovery. Odczyt zachował identyczną całą kopertę: UUID, trzy rewizje,
trzy podpisy i wszystkie acceptance digests. Log potwierdził interrupted
database, rozpoczęcie recovery i ukończenie redo. Scenariusz dotyczy awarii
procesu bazy na testowym nośniku; nie kwalifikuje utraty zasilania lub
niepoprawnego sprzętowego potwierdzania zapisów.

## Wykryte problemy i poprawki

1. Reconciliation interpretowało niedostępność nowego checkpointu jako
   `CORRUPT`. Dodano dokładne mapowanie `UNAVAILABLE` i `INDETERMINATE`
   oraz regresje; obce podklasy wyjątku nie nadają zaufanej klasyfikacji.
2. Kwalifikacja SQL wymagała uzupełnienia `RETURNS TABLE`: kontroluje teraz
   wszystkie typy wyjściowe, `proretset` i dokładny wynik funkcji.
3. Końcowa próbka lifecycle nie może być bool/string/float porównującym się
   jak integer. Nowy adapter wymaga dokładnego `int`; oracle nie zmieniono.
4. Overflow zapisanej sekwencji klasyfikowano jako niedostępność storage.
   Jawne odrzucenie SQL `NumericValueOutOfRange` jest teraz błędem integralności.
5. Mypy wykrył niewłaściwie wywnioskowany typ listy reviewed SQL types;
   poprawiono adnotację bez wyciszania kontroli.
6. Wynik nowego `advance` współdzielił stream z konfiguracją autorytetu.
   Zwracana jest teraz odłączona kopia, jak w reference oracle; mutacja
   zagnieżdżonego raportu nie może zmienić kolejnego odczytu ani replay.

Początkowy dodatkowy zestaw custody miał 50 ERROR z powodu braku `keyring`
w izolowanym venv. Po instalacji brakującej zależności wszystkie 475 testów
przeszły. Dwa początkowe błędy nowych fixture poprawiono zgodnie ze źródłem:
nazwa roli to `ROOT_PROOF_ISSUER_SIGNING`, a SELECT metadata jest dozwolonym
portem kwalifikacji. Nie zmieniono żadnej istniejącej asercji lub bramki.
Betterleaks wykrył testowy klucz w istniejącym ignorowanym bytecode;
wyłącznie ten wygenerowany artefakt przeniesiono poza drzewo repozytorium.
Pełny skan następnie przeszedł, przy niezmienionym baseline i wykluczeniach.

## Odtworzenie testów i granice CI

Nowy plik testów korzysta z istniejącego `DUDZIAN_TEST_POSTGRES_DSN`.
Przydatna minimalna komenda z gotowym PostgreSQL 16 na porcie 55432:

```sh
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 DUDZIAN_TEST_MODE=1 DUDZIAN_ALLOW_LONG_POLL=0 \
  python -m pytest -p asyncio tests/security/test_postgresql_checkpoint_authority.py -q
```

Negatywny test `fsync=off` wymaga
`CRYPTOHUNTER_TEST_POSTGRES_UNSAFE_DSN` wskazującego osobny klaster.
Bez niego wyłącznie ten test zgłasza SKIP; nie wyłącza fsync wspólnej bazy.
W lokalnej walidacji miał PASS. Obecny pojedynczy serwis PostgreSQL GitHub
Actions nie dostarcza drugiego klastra, więc tego PASS nie przypisujemy CI.

Istniejący serial quality-ratchet PostgreSQL został rozszerzony o nowy plik
testów i coverage dwóch modułów. Dodano je również do mypy i formatowania.
Timeouty, istniejące zestawy i bramka 70% pozostały niezmienione.
Wyniki GitHub Actions należy sprawdzać dla dokładnego opublikowanego SHA;
snapshot jest w opisie PR i raporcie końcowym, aby kolejny commit dokumentacji
nie unieważniał badanego SHA.

Niezależny domenowo anti-rollback, odporność po kompromitacji runtime DB
credentials, SERVER_READY, pełna orchestration registry/history/checkpoint/CHA,
capacity/SLA i fizyczna kwalifikacja nośnika pozostają **BLOCKED / niezakwalifikowane**.
Windows Stage 9 pozostaje **10/15 DONE**. Nie uruchomiono Stage 10,
destrukcyjnej kwalifikacji produkcyjnej, emisji Root Proof ani merge.
