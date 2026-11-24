# Zależności npm i renderer — audyt #3280

## Zakres i zmiana

Zweryfikowano lockfile aktywnego projektu `ui/desktop` oraz kod tworzący
okno Electron. Audyt bazy `b143b570` wskazywał 8 podatnych pakietów
(4 high, 3 moderate, 1 low); wynik poprzedniej sesji „0” nie był aktualnym dowodem.

Podniesiono Electron do 43.8.10, Vite do 6.4.6 oraz Zod i zgodny resolver
formularzy do 4.6.7 / 5.9.2. Wymagany Node to co najmniej 22.12.
Reguły formularza przeniesiono bez zmiany kontraktu do wspólnego modułu,
aby testować ten sam schemat i resolver, których używa renderer.

Skrypt E2E wskazuje teraz właściwy plik konfiguracji. Serwer podglądu używa
katalogu wynikowego z konfiguracji Vite i nasłuchuje na `127.0.0.1`.
Nawigacja jest częścią fixture każdej strony; hook deklarowany w importowanym
module nie zapewniał jej kolejnemu plikowi testów w tym samym workerze.
Zachowano wszystkie dotychczasowe asercje scenariusza zakupu licencji.

## Walidacja lokalna

Linux, Node 24.19.0, npm 11.9.0. W katalogu `ui/desktop` wykonano:

| Polecenie | Exit | Wynik |
| --- | --- | --- |
| `npm ci --ignore-scripts` | 0 | Instalacja lockfile; bez pobierania i uruchamiania Electron |
| `npm run test:forms` | 0 | 18 testów rzeczywistego schematu i resolvera |
| `npm run build` | 0 | Build renderera i kompilacja main |
| `tsc --noEmit -p tsconfig.json` | 0 | Typy renderera |
| `tsc --noEmit -p tsconfig.main.json` | 0 | Typy main |
| `npm run test:e2e` | 0 | 5 renderer E2E, w tym 2 negatywne formularza |
| `npm audit --omit=dev --json` | 0 | 0 znanych podatności zależności produkcyjnych lockfile |
| `npm audit --json` | 1 | Wynik niezaliczony: pozostałe zależności narzędziowe |

Testy formularza odmawiają m.in. dla błędnego typu, braku wymaganych danych,
niepoprawnej alokacji, NaN i nieskończoności. E2E uruchomiono w Chromium
Playwright, z testowym interfejsem `desktopAPI`.

Pierwszy E2E zatrzymał się na błędnym katalogu podglądu. Po jego naprawie
zestaw dał 4 PASS / 1 FAIL z powodu braku nawigacji w fixture kolejnego pliku.
Po naprawie fixture wszystkie 5 przypadków przeszło. Tych wcześniejszych
niepowodzeń nie przedstawia się jako wyników zaliczonych.

## Wynik historyczny z 2026-11-18

Końcowy pełny audyt z 2026-11-18 wskazywał 4 pakiety narzędziowe z poziomem high,
związanych z publicznym [GHSA-fc5q-39xc-vjpf](https://github.com/advisories/GHSA-fc5q-39xc-vjpf).
Zależność `picomatch` w dostępnej wtedy wersji 4.0.7 była objęta advisory.
Nie zastosowano wyłączenia, obniżenia progu, nieudokumentowanego override
ani własnej kopii biblioteki. Pełne `npm audit` miało wtedy wynik **FAIL**.
Wynik `--omit=dev` dotyczy wyłącznie zależności produkcyjnych i nie zastępuje
kontroli narzędzi budowania ani testów bezpieczeństwa wykonania aplikacji.

## Rewalidacja z 2026-11-22

Na aktualnym lockfile PR #3289 pełne `npm audit --json` zwróciło exit 0:
0 zgłoszonych podatności dla wszystkich zależności, w tym dev
(394 zależności w inwentarzu).
Nie zmieniono lockfile, nie zastosowano override ani wyłączeń; to bieżący
wynik rejestru, który należy powtórzyć przed merge. Wersja Electron 43.8.10
spełnia wymaganie Node >=22.12 i korzysta z utrzymywanego ekstraktora.

Ponowny build ujawnił istniejącą także na main niezgodność wejścia aplikacji:
`package.main` wskazywał nieistniejący `dist/main/main.js`, kompilacja
zostawiała nierozwiązywany alias `@perf/profiler`, a ścieżka renderera
odnosiła się do innego katalogu. Ustalono jawny `rootDir`, wskazano rzeczywisty
plik main, uruchamianie przez `electron .` oraz względne ścieżki do modułu
profilera i renderera. Test `npm run test:desktop-build` wykonuje prawdziwy
build, ładuje emitowany main i jego zależności, sprawdza preload, renderer
oraz zachowanie `contextIsolation` i `nodeIntegration`. Zastępuje wyłącznie
API Electron; nie jest natywnym uruchomieniem. Na niezmienionym HEAD PR
test odmawiał z powodu brakującego wejścia, a po poprawce przechodzi.
Zestaw E2E wywołuje ten test przed uruchomieniem Playwright.

Ponowiono także pełny TypeScript main/renderera, 18 testów schematu/resolvera
i 5 E2E w Chromium, w tym odmowy błędnych danych. Wszystkie przeszły.
Oficjalny plik wykonywalny Electron pobrano, lecz próby uruchomienia zakończyły się
odmową konfiguracji systemowego sandboxu jeszcze przed wejściem aplikacji;
środowisko nie udostępnia też DISPLAY/Xvfb. Nie dodano flag wyłączających
sandbox. Natywny Electron ma status **NOT_TESTED**, a testy systemowe
Windows/macOS oraz instalatora wymagają odpowiedniego środowiska.

Istniejące `npm run lint` nadal odmawia, ponieważ repozytorium nie zawiera
konfiguracji ESLint dla tego polecenia. Nie zaliczono tego wyniku jako PASS;
migracja konfiguracji lint nie jest częścią naprawy układu buildu.

Kod main zachowuje `contextIsolation: true` i `nodeIntegration: false`.
IPC, nawigacja okna i natywne zachowanie wymagają osobnej kwalifikacji.
Przejście renderera w Chromium nie jest dowodem **Electron native PASS**.
Zmiana głównej wersji Electron wymaga natywnych regresji przed wdrożeniem.

Nie zmieniono risk gate, handlu live, podpisów, licencji ani bramek Stage 9/10.
`STAGE_9=IN_PROGRESS`, `OBSERVER_3A=BLOCKED`, `STAGE_3B=NOT_AUTHORIZED`,
`PRODUCTION=NO-GO` pozostają wymaganym stanem.
