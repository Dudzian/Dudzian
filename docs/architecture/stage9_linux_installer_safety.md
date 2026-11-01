# Stage 9 — bezpieczeństwo instalatora Linux

Baza analizy i gałęzi: `main` na
`2ed7051c3dd92ec7b7408795b42eb1434fac6c19`. Przegląd źródeł, testy syntetyczne
i odizolowany PostgreSQL nie stanowią kwalifikacji fizycznej.

## Integracja po merge

Odczyt GitHub 2026-11-01 o 23:13:40 UTC: Style preflight zakończył się sukcesem.
CI [41359369493](https://github.com/Dudzian/Dudzian/actions/runs/41359369493),
Quality and Security [41359369406](https://github.com/Dudzian/Dudzian/actions/runs/41359369406),
CodeQL [41359369340](https://github.com/Dudzian/Dudzian/actions/runs/41359369340)
i Windows Build [41359369341](https://github.com/Dudzian/Dudzian/actions/runs/41359369341)
pozostawały `queued`, bez wyniku. Testów Windows/Ubuntu/macOS tego SHA nie
można oznaczyć PASS. Nie wykonano merge ani zmian GitHub Environments.

Pięć przypiętych modułów PR #3234 ma poprawne SHA-256. Testy rzeczywistych
checkoutów z `core.autocrlf=true/false` potwierdzają ochronę LF/CRLF; pliki
nowego instalatora otrzymały jawne `eol=lf`. Strukturalne testy zależności
Python/desktop i zgodność bezpośrednich pinów npm z lockiem przeszły.
Nie zastępuje to natywnej instalacji i audytu zależności wszystkich platform.

Nie wykryto nowej regresji tych merge. Wcześniejszy drift
`account_genesis_freshness_authority.py` nadal blokuje kwalifikację: actual
`81cd0df375fe1a48ba943c7645376b4d97350288c0ebe6523207ff90ed22572b`, reviewed
`cd508809f4b4f0663610f592b77695784d74eb4c7e2fba50c42d7a4dbf252f7b`.
Stan istnieje także przed PR #3234 i jest opisany w
`stage9_linux_functional_adapters.md`. Nie zmieniono pinu w celu zaliczenia testu.

Aktualny stan alertów CodeQL #54/#58 jest niedostępny: API zwraca Forbidden,
a nowa analiza jeszcze się nie zakończyła. W PR #3234 pozostaje jeden
nierozwiązany, nieaktualny wątek
[#58](https://github.com/Dudzian/Dudzian/pull/3234#discussion_r4437189441).
Nie deklarujemy zamknięcia alertów na podstawie samego przeglądu źródeł.

## Zagrożenia bazowego skryptu i zmiana

Poprzedni skrypt tworzył i modyfikował konta przed sprawdzeniem klastra,
usuwał kod przez `rm -rf`, kopiował bez stagingu i rekurencyjnie zmieniał
właścicieli/tryby, tracąc executable bits. Nadpisywał konfigurację i nie
weryfikował zatwierdzenia artefaktów, symlinków, kolizji NSS ani danych
zastanej instalacji. Błąd kopiowania lub brak `conf.d` pozostawiał częściowy
stan. Systemd `-I -m` nie udostępniał kodu skopiowanego wyłącznie do cwd.

Nowy wrapper wymaga jawnego `--preflight` albo `--install`. Preflight jest
odczytowy, wymaga niezależnie zaufanego klucza Ed25519 poza bundle i podpisu
dokładnych bajtów manifestu wiążącego pełne SHA, inwentarz, hashe i tryby.
Nie importuje bundle i odrzuca pliki specjalne, symlinki, hardlinki,
dodatkowy kod/cache, niepoprawne ścieżki, konta i uprawnienia.

Kontroluje pakiety, zależności systemowego interpretera, systemd PID 1,
istniejący PostgreSQL 16/main, NSS/UID/GID i separację usług. Sprawdza obce
jednostki, fragmenty oraz drop-iny, odmawiając przy niepełnej obserwacji.
Operacje plikowe korzystają z chronionych dirfd, O_NOFOLLOW/O_EXCL i limitów
inwentarza. Nowy kod jest weryfikowany w stagingu i publikowany przez
`renameat2(RENAME_NOREPLACE)`. Zachowuje podpisane tryby 0644/0755.

Instalacja nie nadpisuje plików ani danych. Idempotencja dotyczy wyłącznie
identycznego, kompletnego i ponownie zweryfikowanego wdrożenia. Legacy,
inne wydanie, drift i częściowe wdrożenie wymagają ręcznego planu recovery.
Dziennik zapisuje kolejne fazy; nie zakłada atomowości konfiguracji całego
hosta ani automatycznego rollbacku. Konta, grupy, klaster, schemat i klucze
są osobnymi ceremoniami; instalator ich nie tworzy ani nie usuwa. Nie steruje
usługami i nie restartuje hosta.

Stały launcher `freshness_verifier_entrypoint.py` rozwiązuje import pod `-I`,
bez wykonywania szerokiego `bot_core/__init__.py`. Nie korzysta z cwd ani
PYTHONPATH i pomija zastany lokalny bytecode. Zachowano wszystkie właściwości
sandboxu jednostki i dokładne źródłowe piny kwalifikacji.

## Faktyczny zakres produktu

| Komponent | Stan i zakres instalatora |
| --- | --- |
| FreshnessVerifier | Rzeczywisty entrypoint i ograniczone adaptery; instalowany kod/jednostka, bez startu ani kwalifikacji |
| AccountGenesis runtime | Biblioteka autoryzacji/CAS; brak wdrażanego produkcyjnego procesu i jednostki |
| RootProofSigner | Istniejący serwer pozostaje diagnostyczny; poza instalacją |
| HistoryAttestationSigner | Istniejący serwer pozostaje diagnostyczny; poza instalacją |
| RootProofIssuer | Entry point `--check`, emisja produkcyjna BLOCKED; poza instalacją |
| Source/CCA | Dostępne fixture TEST ONLY nie są produkcyjnymi usługami; poza instalacją |
| PostgreSQL | Wymagany gotowy klaster i osobny provisioning; kopiowane trzy zatwierdzone szablony, bez ich zastosowania do działającego serwera |
| systemd | Publikowany plik verifiera, bez reload/enable/start; sprawdzany obcy stan jednostki |
| IPC | Zachowane kontrakty verifiera i separacja UID; brak pełnego produkcyjnego klienta runtime i pełnej kwalifikacji obu procesów |

Przejrzane ścieżki pakowania desktop/OEM tworzą ZIP/tar.gz; nie stanowią
produkcyjnego buildera DEB runtime+verifier. Ten PR nie udaje pełnego
certyfikatu `linux_release_identity.py` ani dowodów fizycznych.

Osobny PR musi ustalić produkcyjny entrypoint/lifecycle AccountGenesis,
kompozycję signerów/issuera/Source/CCA, zaufany interpreter i pakiet,
rzeczywisty klient IPC z pełnymi odmowami/replay, obie funkcjonalne macierze
peer PostgreSQL oraz DEB baseline/update/failed-update. Następnie wymagane
są niezależne zatwierdzenie wydania i odrębna kwalifikacja fizyczna.

## Wyniki walidacji

| Zestaw | Wynik |
| --- | --- |
| Nowe testy instalatora, hosta i launchera | 128 PASS; Linux, bez uprawnień administratora |
| Systemd, IPC, adaptery PostgreSQL, verifier i powiązane kontrakty | 341 PASS, 2 SKIP zgodnie z warunkami platformy |
| Stage 9, physical/trusted-installation, platform evidence i topologia | 984 PASS |
| Kontrakty workflow jakości i fizycznych bramek | 144 PASS |
| LF/CRLF, piny i zależności Python/desktop | 16 PASS |
| Rzeczywisty PostgreSQL 16.15: adapter | 8 PASS |
| Rzeczywisty PostgreSQL: peer-auth/runtime/semantic | 23 PASS w końcowym przebiegu |

Zestawy częściowo się pokrywają; liczby nie są sumą unikalnych testów.
Pierwszy kontener z limitem CPU miał 22 PASS/1 FAIL przez dwusekundowe
oczekiwanie fixture na socket. Ten test oraz cały zestaw ponownie przeszły
bez zmiany kodu w świeżym kontenerze bez tego limitu. Kontenery nie były
uprzywilejowane, nie miały sieci, host mounts ani opublikowanych portów i
zostały usunięte. Nie mutowano PostgreSQL/NSS hosta ani Della.

Pokrycie nowych modułów z gałęziami: 94% łącznie; line diff coverage: 95%
przy niezmienionym progu 70%. Nowe moduły i testy dodano do istniejących
bramek jakości i kontraktów Linux; nie zmieniono progów, timeoutów,
historycznych TEST_ONLY ani inventory atomic/history.

Ruff (baseline oraz rozszerzone E/F/B/I nowych plików), dokładny hook
`python -m pre_commit run ruff-format --all-files`, jawny format nowych
modułów i configured mypy Linux/Win32 przeszły. Mypy obejmował 381 plików.
Pięć blokujących reguł Semgrep dało zero znalezisk w czterech zmienionych
modułach produkcyjnych. Betterleaks 1.9.0, zweryfikowany po SHA-256, nie
znalazł sekretów w czystym snapshotcie źródeł z indeksu Git; zachowano
istniejący baseline. Pierwotny skan całego dostarczonego workspace zgłosił
dwa pliki ignorowanych cache testowych, których snapshot źródeł nie zawiera.
Nie dodano wyjątków ani nie zmieniono baseline.

## Procedura i decyzja o merge

[Procedura operatora](../runbooks/STAGE9_LINUX_SAFE_INSTALLER.md) opisuje
niezależne zatwierdzenie materiałów, zaufany bootstrap, odczytowy preflight,
kody odmów, instalację, ochronę danych i recovery. Dell w opisanym stanie
ma otrzymać odmowę z powodu braku klastra i przygotowanych tożsamości;
instalator nie tworzy brakującego klastra. Uprzywilejowana próba pełnej
instalacji wymaga osobnej jednorazowej Ubuntu VM i pozostaje poza tym PR.

Rekomendacja: pozostawić PR jako draft do zakończenia wymaganych kontroli
integracji i nowej gałęzi oraz niezależnego przeglądu instalacji. Ewentualny
merge obejmuje wyłącznie opisany substrat; nie przyznaje gotowości produkcji.

```text
MAIN_INTEGRATION_CI_READY=PENDING
LINUX_INSTALLER_PREFLIGHT_READY=YES
LINUX_INSTALLER_SAFETY_READY=YES
LINUX_FUNCTIONAL_RUNTIME_COMPLETE=NO
LINUX_PHYSICAL_TESTS_READY=NO
STAGE9_PRODUCTION_READY=NO
```

Oba YES dotyczą zakresu kodu instalatora zweryfikowanego syntetycznie,
nie gotowości opisanego Della ani pełnej uprzywilejowanej instalacji.
