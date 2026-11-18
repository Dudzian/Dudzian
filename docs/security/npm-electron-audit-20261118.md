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

## Pozostałe ryzyko i ograniczenia

Pełny audyt z 2026-11-18 wskazuje 5 pakietów narzędziowych z poziomem high,
związanych z publicznym [GHSA-fc5q-39xc-vjpf](https://github.com/advisories/GHSA-fc5q-39xc-vjpf).
Zależność `picomatch` w dostępnej wersji 4.0.7 pozostaje objęta advisory.
Nie zastosowano wyłączenia, obniżenia progu, nieudokumentowanego override
ani własnej kopii biblioteki. Pełne `npm audit` pozostaje **FAIL**.
Wynik `--omit=dev` dotyczy wyłącznie zależności produkcyjnych i nie zastępuje
kontroli narzędzi budowania ani testów bezpieczeństwa wykonania aplikacji.

Kod main ustawia `contextIsolation: true` i `nodeIntegration: false`.
Nie uruchamiano natywnego Electron, systemowego sandboxu ani instalatora.
IPC, nawigacja okna i natywne zachowanie wymagają osobnej kwalifikacji.
Przejście renderera w Chromium nie jest dowodem **Electron native PASS**.
Zmiana głównej wersji Electron wymaga natywnych regresji przed wdrożeniem.

Nie zmieniono risk gate, handlu live, podpisów, licencji ani bramek Stage 9/10.
`STAGE_9=IN_PROGRESS`, `OBSERVER_3A=BLOCKED`, `STAGE_3B=NOT_AUTHORIZED`,
`PRODUCTION=NO-GO` pozostają wymaganym stanem.
