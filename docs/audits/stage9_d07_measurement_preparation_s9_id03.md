# S9-ID-03 — przygotowanie pomiaru D07

6 listopada 2026 r. Baza: `5c8a600c3d4133362518513b80b6ac540d99dd19`
(main po PR #3258). **D07: PENDING. Stage 9: IN_PROGRESS / PRODUCTION_NO_GO.**
To przygotowanie diagnostyki TEST_ONLY, bez nowego ADR, implementacji K4-v2,
zmiany produkcji ani zatwierdzenia D01–D08. Recenzja kodu narzędzia nie jest
rekordem akceptacji właściciela i niezależnego recenzenta Stage 9.

## Sprawdzone źródła i istniejące możliwości

Przeczytano `AGENTS.md`, [kontrakt tożsamości](../architecture/stage9_verifier_identity_acceptance_contract_v1.md),
[S9-ID-02](stage9_python_verifier_feasibility_s9_id02.md),
[runbook VM](../runbooks/STAGE9_LINUX_VM_HARNESS.md),
`deployment/stage9_vm_{trial,environment,native}.py`, źródła verifiera/klienta
oraz materiały [PR #3257](https://github.com/Dudzian/Dudzian/pull/3257)
i [PR #3258](https://github.com/Dudzian/Dudzian/pull/3258).
API dyskusji obu PR zwróciło puste listy. Codebase Memory nie było dostępne
w katalogu narzędzi ani jako CLI; brak grafu i kontroli pokrycia indeksu.
Ustalenia poniżej oparto na bezpośredniej inspekcji wymienionych źródeł.

| Istniejący element | Co już umożliwia | Granica dowodu |
| --- | --- | --- |
| `stage9_vm_trial` | Kontrole runtime, publikacji, preflight, instalacji, systemd i PG/IPC; istniejący worker zatwierdzonego interpretera. | Statusy pojedynczych kontroli, bez serii czasów i rozdzielenia kosztów. |
| `stage9_vm_environment.require_approved_vm` | Weryfikację jednorazowej, odizolowanej VM i zgody związanej z maszyną, bootem, rewizją i środowiskiem. | Zgoda musi pochodzić od operatora; narzędzie jej nie tworzy. |
| `observe_systemd(start=True)` | Rzeczywisty start i kontrolę chronionej usługi po bramkach VM. | `Type=simple`/`active` nie oznacza gotowości listenera. |
| `observe_postgresql_ipc` | Rzeczywistą diagnostyczną wymianę i obserwację połączenia PG usługi. | Zachowuje `BLOCKED`; to nie pełne żądanie `verify_exact_candidate`. |
| Fixture i testy transportu | Rzeczywiste procesy AF_UNIX, PIDFD/procfs, ramki, utratę odpowiedzi i błędy protokołu przy modelowanej instalacji. | Całość MODEL_ONLY; brak natywnej izolacji dwóch UID i rzeczywistego writera PG. |
| `fixture_candidate` | Bazowy, niezmieniony verifier i rzeczywiste trzy weryfikacje Ed25519 poprawnego kandydata. | Resolver, writer, przygotowanie i trwały efekt są modelem. |

Scalenie #3257/#3258 nie wdrożyło ich patchy produkcyjnych. Nowa próba nie
aplikuje patchy z `tests/fixtures`; mierzy aktualny kod bazowy. Testy poprawności
i ich czas wykonania nie są wcześniejszym benchmarkiem D07.

## Minimalne uzupełnienie

Dodano wyłącznie bibliotekę diagnostyczną
[`stage9_vm_performance.py`](../../deployment/stage9_vm_performance.py)
i testy korzystające z istniejącej fixture. Nie ma nowego transportu, runnera VM,
polecenia natywnego, instalatora, źródła expected ani alternatywy K4.
Istniejące fazy i bramki harnessu pozostają bez zmian.

`Recorder.sample` mierzy czas zewnętrzny wywołania, CPU bieżącego procesu
i jego lifetime RSS HWM na Linux. `Recorder.profile` rozdziela przedziały
wskazanych istniejących funkcji według ich obiektów kodu; nie podmienia funkcji,
nie zmienia deadline i odmawia zastąpienia aktywnego profilera. `record` przyjmuje
jawne obserwacje startu, listenera, pierwszej odpowiedzi, IPC, PG, tożsamości
i cgroup wraz ze źródłem i jednostką. **Sam zapis etykiety nie dowodzi pochodzenia
danych**; do native potrzebne są surowe obserwacje i niezależna walidacja platformy.
Biblioteka nigdy automatycznie nie przyznaje `BASELINE_MEASURED`.

Grupowanie rozdziela scenariusz, temperaturę, metrykę, źródło, jednostkę
i zakres zasobów. Raport zachowuje wszystkie próbki, błędy i timeouty; kwantyle
nearest-rank zakończonych wywołań są oddzielone od przerwanych. Puste zbiory
nie dają zera. Czas powrotu `client_exchange` > **3 000 000 000 ns** jest
przekroczeniem, również przy błędzie lub timeout. Brak wyjątku `TimeoutError`
nie dowodzi braku timeoutu: klient zwraca również `OUTCOME_UNKNOWN`; adapter
obserwatora musi zapisać znaną przyczynę, a nie zgadywać jej z samego wyniku.

CPU to delta `process_time_ns` całego obserwowanego procesu; nie jest procentem
obciążenia jednostki systemd. RSS to szczyt od narodzin procesu, obejmujący
wcześniejsze importy i testy, nie przyrost ani szczyt konkretnego requestu.
Pomiar cgroup musi objąć również ExecStartPre i dzieci OpenSSL. Przedziały
profilera są inkluzywne, mogą się nakładać; oczekiwanie IPC zawiera pracę serwera.
Nie sumować ich i nie obliczać czasu PostgreSQL przez odejmowanie.

## Odtworzenie MODEL_ONLY

W zwykłym środowisku testowym repozytorium, bez uprawnień i opt-in native:

```bash
CRYPTOHUNTER_D07_MODEL_OUTPUT=/bezwzgledna/sciezka/nowy-model.json \
  python -m pytest -q tests/security/test_stage9_d07_model_measurement.py
python -m pytest -q tests/deployment/test_stage9_vm_performance.py
```

Plik wynikowy musi być nowy; test nie nadpisuje wcześniejszych danych. Próba
używa istniejącej `fixture_candidate`: cztery osobne serie po 101 wywołań
(poprawne, błędny podpis, błędny JSON, poprawne z profilerem). Pierwsze wywołanie
nowego obiektu ma temperaturę `unspecified`, dalsze 100 — `warm`.
**To nie zimny start procesu ani cache systemu/PG.** Korpus to jeden mały
kandydat na serię, a nie rozkład produkcyjnych rozmiarów do 4 MiB.
Generowanie kluczy i korpusu odbywa się przed odmierzanym przedziałem.
Przygotowania fixture mają syntetyczne identyfikatory i nie trafiają do PG.

Surowy JSON zawiera konfigurację interpretera, bibliotek i kernela, hash pliku
verifiera, n, wyniki/odmowy, CPU/RSS i komponenty profilowane. Dane źródłowe
mają etykiety `PROCESS` (rzeczywisty proces), `POSTGRESQL` (rzeczywista baza),
`MODEL` (model). Ta próba nie emituje `POSTGRESQL`.
Serii profilowanej nie łączyć z nieprofilowaną. Różnica pokazuje łączny wpływ
profilowania, kolejności i zmian obciążenia, a nie precyzyjnie skalibrowany narzut.

## Oddzielne wyniki i niepewność

| Klasa | Stan na tym etapie |
| --- | --- |
| BASELINE_MEASURED | Brak, n=0. Nie wskazano i nie zatwierdzono platformy native. |
| MODEL_ONLY | Odtwarzalna seria opisana wyżej oraz regresje agregacji i odmów. Liczby nie są budżetami usługi ani PostgreSQL. |
| NOT_RUN | Systemd→proces→listen→pierwsza odpowiedź, rzeczywiste PG/COMMIT, klient z pełnym K4 na dwóch UID, cgroup całej jednostki, zimne cache, utrata odpowiedzi i przeciążenie rzeczywistej usługi. Każda z tych serii n=0. |

Dokładne liczby lokalnego przebiegu i wyniki kontroli jakości znajdują się
poniżej oraz w dołączonym surowym JSON. Błędy/odmowy i timeouty
muszą być podane osobno; oczekiwane odrzucenie złego podpisu nie jest awarią
pomiaru. `NOT_RUN` nie oznacza zera błędów czy timeoutów.

100 ciepłych obserwacji daje tylko około jedną obserwację ogona 1%; nearest-rank
p99 nie ma tu wiarygodności budżetu produkcyjnego. Kolejne wywołania są zależne,
platforma współdzielona i bez zatwierdzonej alokacji CPU/RAM/dysku. Rozdzielczość
zegara nie jest dokładnością pomiaru; RSS HWM nie ma dokładności per request.
Dla niezależnych prób z zerem zdarzeń górna jednostronna granica 95% wynosi
`1 - 0.05**(1/n)`; nie stosować jej bez uzasadnienia niezależności. Dobór n
i osobnych serii/bloków wymaga pilotażu oraz uzgodnienia precyzji ogona.

## Lokalny przebieg MODEL_ONLY

Środowisko współdzielone: Linux 6.18.57 x86_64, CPython 3.12.14,
cryptography 50.0.2, psycopg 3.3.8 (biblioteka, bez połączenia z bazą),
3 widoczne logiczne CPU. Przydział vCPU/RAM/dysku i stan cache niezatwierdzone.
Cały proces to pytest z jego importami. Bazowy plik verifiera ma SHA-256
`64dd105d54a2eabd9be9164846fe642ca0a274b30ca5d0a83735a4436d9f39e1`.
Każda zwykła trójka wejść ma 401/687/701 bajtów; uszkodzony JSON ma 1 bajt.
Hashe korpusu, obserwatora i testu są zapisane w JSON; klucze fixture są losowe.

Ciepłe serie; czasy w ms, źródło **rzeczywisty proces**:

| Seria | n | p50 | p95 | p99 | max | Wynik |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| Poprawny kandydat, bez profilera | 100 | 0,553397 | 0,742368 | 1,359730 | 2,094690 | 100 przygotowań modelu |
| Błędny podpis | 100 | 0,464971 | 0,854879 | 1,729837 | 3,977784 | 100 oczekiwanych odmów, zero wywołań writera |
| Niepoprawny JSON | 100 | 0,004404 | 0,007647 | 0,008597 | 0,009666 | 100 oczekiwanych odmów, zero wywołań writera |
| Poprawny kandydat, z profilerem | 100 | 2,537059 | 4,487493 | 5,585337 | 5,780161 | 100 przygotowań modelu |
| Pojedyncze `_verify` w profilowanej serii | 300 | 0,109987 | 0,148944 | 0,503977 | 0,757859 | 3 rzeczywiste Ed25519 na request |

Dodatkowo wykonano po jednym pierwszym wywołaniu każdej serii (`unspecified`),
czyli łącznie 404 wywołania verifiera: 202 przygotowania modelu, 202 oczekiwane
odmowy, zero nieoczekiwanych wyjątków i zero timeoutów. Nie było tu pomiaru
`client_exchange`; nie ma podstaw do wniosku o przestrzeganiu 3,0 s przez klienta.
W surowym rejestratorze oczekiwane wyjątki semantyki są `error/interrupted`;
pole `outcomes` rozróżnia je od awarii próby. Szczyt RSS/HWM procesu pytest
wyniósł **73 756 672 B**; nie jest to zużycie pamięci usługi. CPU w JSON ma
jednostkę ns i dotyczy wyłącznie procesu pytest, bez dzieci i backendów PG.
Czasy resolvera/writera mają źródło **model** i nie stanowią pomiaru PostgreSQL.

[Surowy JSON, gzip](evidence/s9_id03_model_only.json.gz), 11 559 B archiwum:

- SHA-256 archiwum: `10c4f47dfe6e7daf3c39594d1e5bc9c63bbf0146e1b5dd378fedca029531347b`;
- SHA-256 rozpakowanego JSON: `bb7a59b961f64b8b16564c29b3b73c5e4bf0f0ffd3016e116d60c24f6e3fe44a`.

Archiwum zawiera surowe liczby i metadane bez payloadów, DSN i kluczy.
Można je odczytać przez `gzip.open(path, 'rt')` i `json.load`.
Narzut profilera jest wyraźny; nie odejmowano go od wyników i nie korygowano
kwantyli. Brak przedziałów ufności i kontroli współdzielonego obciążenia.

Kontrole jakości: testy instrumentu i istniejących bramek VM, kontrola zakresu
produkcji i kontraktu, Ruff (także niezależny hook `ruff-format --all-files`),
mypy nowego modułu dla Linux/Win32, Betterleaks 1.9.0 offline i niezależny
przegląd kodu. Zestaw instrumentu i bramek VM: **240 PASS**, w tym **30 nowych
testów**; pokrycie nowego modułu z gałęziami **84%**. Testy transportu pod
domyślnym sandboxem napotkały EPERM dla
AF_UNIX: 239 PASS, 48 SKIP, 5 FAIL. Ten sam zestaw bez uprawnień administratora,
z dostępem narzędzia do lokalnych socketów: **291 PASS, 1 SKIP** (warunkowy
test dodatkowych grup). Nie zmieniano hosta ani produkcyjnych plików; wynik
nie jest natywną kwalifikacją. Skanowanie i pełne skonfigurowane bramki CI
pozostają niezależnymi kontrolami Draft PR, bez restartu istniejących workflow.

## Następny pomiar — wyłącznie osobno zatwierdzona VM

1. Wskazać konkretną jednorazową VM: identyfikatory, snapshot i plan usunięcia,
   izolację sieci/udziałów, Ubuntu/kernel/systemd, pakiety, vCPU/RAM/dysk,
   wersję i konfigurację PostgreSQL oraz dokładny commit i korpus. Operator
   zatwierdza zakres zmian i obserwatora według istniejącego runbooka. Nie
   generować rekordu zgody ani nie traktować tego dokumentu jako zgody.
2. Użyć istniejącego harnessu, jego preflight, kontroli instalacji/runtime
   i native-workera. Nie restartować workflow ani nie uruchamiać tego na hoście
   fizycznym. Nie zmieniać Yama, seccomp, AppArmor, unitów ani ról PG na obecnym
   komputerze. Ewentualne zmiany na VM wymagają jawnie zatwierdzonego zakresu.
3. Zewnętrzny kontroler wiąże początek startu, narodziny ExecStartPre/ExecStart,
   boot/InvocationID/PID i moment listen oraz pierwszego wyniku. Listener
   obserwować pasywnie albo właściwą wymianą — pusty connect może zablokować
   sekwencyjny serwer oczekujący ramki. Zapisać przedział niepewności pollingu
   i rozdzielczość start ticks. `active` nie zastępuje listen.
4. Zmierzyć niezmienionego klienta do jego powrotu, włącznie z końcowym recheckiem;
   zachować 3,0 s, bez retry. Oddzielić poprawne odpowiedzi, odmowy K4, błędne
   dane, utratę odpowiedzi, częściową ramkę, zablokowany PG/COMMIT i stopniowane
   obciążenie. Rejestrować offered load, współbieżność, kolejkę i wszystkie
   wyniki; nie wyliczać ogona wyłącznie z sukcesów. Stary K4 może odmówić przed
   payload: taką odmowę mierzyć, a dodatnią ścieżkę pozostawić NOT_RUN/BLOCKED.
5. Zewnętrznie odczytać cgroup CPU/memory i `/proc/MainPID` RSS/HWM, bez zmiany
   polityk/accountingu w celu uzyskania danych. Osobne serie dla nowego procesu,
   ciepłych requestów i udokumentowanego stanu cache; nie czyścić cache hosta.
6. Wewnętrzne odcinki crypto/tożsamości/każdej sesji PG i COMMIT wymagają
   zatwierdzonego sposobu obserwacji. Profiler biblioteczny działa tylko we
   własnym procesie/wątku. Nie wstrzykuje się go do chronionej usługi. Osobno
   zatwierdzona diagnostyczna kopia closure jest innym wariantem niż dokładna
   podpisana usługa; jej wyników nie promować automatycznie do baseline.
   Bez takiej obserwacji odpowiednie komponenty pozostają NOT_RUN.

Po pomiarze można przedstawić do zatwierdzenia: budżet p99/max całej wymiany
z częstością przekroczeń 3,0 s, osobny budżet startu, CPU przy określonym offered
load oraz pamięć całej jednostki i RSS/HWM procesu. Rezerwy ustala się według
niepewności i docelowej platformy; teraz brak danych do liczbowych propozycji.
Następny krok to decyzja właściciela o konkretnej VM i metodzie obserwacji,
następnie pomiar w istniejącym harnessie. **D07 pozostaje PENDING.**
