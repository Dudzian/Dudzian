# Stage 9 VM — remediacja PR #3246, 2026-11-03

Zakres remediacji obejmuje nowe mechanizmy harnessu i dokładność raportu.
Historyczny [audyt początkowy](stage9_linux_vm_harness_2026-11-03.md) oraz
dowody TEST_ONLY pozostają bez zmian. Punktem wyjścia jest HEAD przeglądu
`39cd8e3475b59b16a8f84d16e40428fb9461b5d0`.

## Siedem awarii eksportera i tożsamość testowa

Job `linux-evidence-contract` 126236547858 zakończył się wynikiem
7 failed, 1533 passed, 2 skipped. Fixture wymuszała zapytanie o UID 1000,
ale nie zmieniała rzeczywistych właścicieli katalogów `tmp_path` runnera.
Produkcja słusznie odmawiała `OUTPUT_PARENT_UNPROTECTED`, zanim test dochodził
do zamierzonego eksportu, duplicate JSON, zmiany working tree, `refs/replace`,
niezgodnego HEAD, sanitizacji JSON albo restrykcyjnego umask.

Domyślna fixture zachowuje rzeczywisty nie-root UID runnera i właścicieli
jego katalogów. Siedem przypadków ma dodatkowe jawne modele UID 1000, 1001
i 4242: query UID i metadane właściciela zmieniają się razem; nie zmienia się
rzeczywistych credentials ani właścicieli filesystemu. Kontrole obcego UID,
trybów 0775/0777, sticky-bit, symlinków i odmowy root pozostają aktywne.
Kod `prepare_stage9_vm_bundle.py` nie wymagał zmiany produkcyjnej.

Osobny test obejmuje automatyczną konfigurację fixture przy modelowanym
runnerze root: pełny eksport w jawnym modelu rootless, następnie odmowa
`ROOTLESS_BUILDER_REQUIRED` dla query UID 0. Nie wykonano rzeczywistego
`setuid`, `chown` ani uprzywilejowanej próby. Lokalny rzeczywisty UID to 1000;
metadane katalogów `/` i `/tmp` w executorze są mapowane do 65534, dlatego
testy zachowują odrębny model tych dwóch granic jako root-owned na Ubuntu.
Nowy krok CI wypisuje rzeczywisty UID/GID i tryby/właścicieli katalogów
runnera. Ich wynik trzeba odczytać z joba bieżącego HEAD.

## Zakres porównania hosta

Przegląd `SNAPSHOT_PATHS`, `snapshot()` i `checked_preflight()` w
[`stage9_vm_trial.py`](../../deployment/stage9_vm_trial.py) potwierdził,
że digest obejmuje wybrane pliki i drzewa, a nie całą VM. Wykryto lukę
dotyczącą stagingu: instalator publikuje go jako rodzeństwo katalogu kodu
`/usr/lib/.cryptohunter-stage-<manifest_sha256>`. Samo rekurencyjne skanowanie
`/usr/lib/cryptohunter` nie obejmowało tych ścieżek. Remediacja rozszerza
pomiar wyłącznie o istniejące ścieżki z tym prefiksem: enumeruje nazwy
bezpośrednich dzieci `/usr/lib`, bez rekurencyjnego hashowania pozostałej
zawartości i bez aktywnego PGDATA. Regresja przed poprawką otrzymała
`DID NOT RAISE`; po poprawce utworzenie stagingu podczas preflight daje
`PREFLIGHT_MUTATED_HOST`. Odrębne testy obejmują zmianę/usunięcie stagingu,
odmowę symlinka zamiast jego katalogu nadrzędnego i brak dereferencji linku
do danych spoza zakresu.

| Warstwa dowodu | Co obejmuje | Czego nie ustanawia |
| --- | --- | --- |
| Manifest aplikacji v2 | Podpisane dokładne bajty, rewizja, pliki payloadu i binding runtime; instalator zwraca digest tych bajtów | Niezmienność całego hosta lub kwalifikacja produktu |
| Podpisany runtime | Własny manifest, zamknięcie pakietów i plików, interpreter oraz `external_files` | Pełny inwentarz dysku VM |
| Porównanie przed/po | Osiem pozycji `SNAPSHOT_PATHS` i staging `/usr/lib/.cryptohunter-stage-*`; treści, UID/GID, tryby, nlink, literalne linki | Dane PostgreSQL, stan managera/procesów, wejścia `/srv`, ACL/xattrs, wszystkie systemowe pliki |
| PostgreSQL i guard danych | Odrębne obserwacje sesji/peer/ACL oraz ręczne porównanie syntetycznych wierszy guard | Niezmienność wszystkich wierszy i plików PGDATA |
| Systemd/proces/IPC | Odrębne żywe obserwacje rzeczywistego managera i przypiętego procesu | Zastąpienie snapshotu hypervisora lub pełny dowód produktu |
| Snapshot hypervisora | Spójny punkt odtworzenia całej jednorazowej VM przygotowany przez operatora | Produkcyjna atestacja fizycznego hosta |

Równe digesty i `readonly=true` oznaczają zgodność wybranych obserwacji.
Nie wykluczają krótkotrwałych zapisów cofniętych pomiędzy odczytami ani
zmian poza zakresem pomiaru. Porównanie w `finally` obejmuje także odmowę
lub wyjątek instalatora; wykryta mutacja w mierzonym zakresie pozostaje `FAIL`.
[Runbook](../runbooks/STAGE9_LINUX_VM_HARNESS.md) precyzuje te ograniczenia.

## Użycie istniejącego autorytetu manifestu

Nie dodaje się nowego produkcyjnego walidatora. `_manifest()` w
[`production_local_installer.py`](../../deployment/production_local_installer.py)
weryfikuje Ed25519 dokładnych bajtów `--manifest` przed parsowaniem JSON,
a po walidacji zwraca ich SHA-256. `Plan`, receipt/journal i odpowiedź CLI
zachowują ten digest. Harness przekazuje wskazany manifest do tego istniejącego
interfejsu; nie deklaruje nowej ochrony przed równoległą mutacją przez root.

Istniejące testy `test_positive_preflight_is_readonly_even_for_unprivileged_test_identity`,
`test_manifest_bytes_are_signed_without_line_ending_normalization`,
`test_wrong_release_key_cannot_authorize_bundle` oraz testy CLI/instalacji
w `tests/deployment/test_production_local_installer.py` pokrywają ten łańcuch.
Pierwszy jawnie porównuje `Plan.manifest_sha256` z SHA-256 wejściowych bajtów;
CLI serializuje bezpośrednio pole tego samego `Plan`. Wybrany zestaw: 7 PASS.

## Walidacja i ograniczenia

Walidacja lokalna końcowych źródeł:

| Kontrola | Wynik |
| --- | --- |
| Eksporter, wszystkie odmowy, rzeczywisty UID i modele UID/root | 55 PASS |
| Siedem scenariuszy dawnej awarii dla runner-owner i trzech modeli UID | 28 PASS, wykonane dokładne asercje właściwych granic |
| Eksporter, VM Guard, VM Trial, VM Native, podpisany runtime i builder | 320 PASS, 1 SKIP (opt-in build Noble) |
| Dokładny zestaw pytest `linux-evidence-contract` uruchomiony lokalnie | 1567 PASS, 2 SKIP |
| Pozostałe wybrane regresje workflow i modeli Windows | 417 PASS, 2 SKIP |
| Ruff repo, szerszy lint zmienionych Pythonów i jawny format | PASS |
| Dokładny `python -m pre_commit run ruff-format --all-files` | PASS, bez dalszych zmian |
| Skonfigurowany mypy Linux i Win32 | PASS, po 391 źródeł |
| Pokrycie zmienionych linii produkcyjnych | 100%, istniejący próg 70% zachowany |
| Semgrep lokalnych reguł i Betterleaks zmienionych plików | 0 wykryć; baseline i wykluczenia bez zmian |

Zdalne workflow, w tym rzeczywisty runner UID oraz `linux-evidence-contract`,
Quality and Security, CodeQL i regresje wieloplatformowe, wymagają wyniku
bieżącego HEAD po wypchnięciu. Aktualny odczyt należy zachować w PR/raporcie,
bez przenoszenia statusów starszej rewizji.

Próba na zatwierdzonej Ubuntu 24.04 VM pozostaje `PENDING`; nie zastępuje się
jej kontenerem ani dodatkowymi modelami. Brak dostępnej VM nie jest nowym
błędem kodu lub samodzielną przesłanką blokady code review/merge harnessu.
Pełna kwalifikacja produktu i fizyczna atestacja produkcyjna pozostają poza
zakresem tej remediacji. Nie wykonuje się merge ani awansowania Stage 9.
