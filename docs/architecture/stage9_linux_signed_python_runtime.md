# Stage 9 — podpisany runtime Python dla FreshnessVerifier

Zakres: Ubuntu 24.04 amd64, systemowy CPython 3.12 oraz osobne, wersjonowane
środowisko. To kontynuacja kontraktów instalatora z PR #3235. Sam podpis
środowiska nie zatwierdza produktu, hosta ani brakujących dowodów Stage 9.

## Łańcuch uruchamiania i zaufania

1. Uprzednio zaufany system Ubuntu dostarcza bootstrap: systemowy Python,
   jego stdlib i `/usr/bin/openssl`. Administrator niezależnie zatwierdza
   kopię instalatora oraz bootstrapu; podpis nie uwierzytelnia własnego
   weryfikatora. Nie zakładamy odporności na złośliwego administratora root.
2. `freshness_runtime_bootstrap.py` sprawdza Ed25519 przez systemowy OpenSSL,
   używając anonimowych deskryptorów memfd. Nie importuje `cryptography`
   z ocenianego środowiska i nie uruchamia jego interpretera. Dane podpisu
   nie wymagają zapisu plików tymczasowych ani mutacji hosta.
3. `freshness_runtime.py` odczytuje chroniony, zewnętrznie dostarczony klucz
   `/etc/cryptohunter/freshness-runtime-authority.pub`. Klucz ma dokładnie
   32 surowe bajty Ed25519. Jego zaufanie pochodzi z osobnej ceremonii
   administratora; klucz dołączony przez producenta nie ustanawia zaufania.
   Podpis obejmuje dokładne bajty manifestu, przed interpretacją jego schematu.
4. Weryfikacja wiąże environment-id, rewizję, ABI, bazowy interpreter Ubuntu,
   konfigurację venv, pliki, symlinki, uprawnienia, wheelhouse oraz zewnętrzne
   pliki interpretera i bibliotek natywnych. Zmiana systemowego Pythona wymaga
   nowego przeglądu i zatwierdzenia; hash nie jest automatycznie aktualizowany.
5. Podpisany manifest aplikacji `cryptohunter.freshness.substrate.v2` wiąże
   dokładny runtime. Instalator porównuje go z chronioną konfiguracją
   `/etc/cryptohunter/freshness-runtime.json`. Następnie sonduje importy
   dokładnie tego interpretera, w izolacji `-I -s -B`.
6. Jednostka systemd ma dokładny `ExecStartPre` uruchamiający niezależny
   bootstrap przed każdym startem. `ExecStart` wskazuje dokładny interpreter
   `/opt/cryptohunter/freshness-runtime/releases/<environment-id>/bin/python3.12`.
   Punkt wejścia tworzy wyłącznie stałą przestrzeń `bot_core` z chronionego
   katalogu aplikacji i importuje rzeczywisty `freshness_verifier_service`.
7. Kwalifikacja odczytuje ponownie runtime i konfigurację systemd, dokładne
   typowane ExecStart/ExecStartPre, MainPID/ExecMainPID, pidfd, argv, UID/GID,
   cgroup oraz `/proc/PID/exe`. Rozwiązuje zatwierdzony symlink venv do bazowej
   binarki i porównuje inode, metadane oraz SHA-256. Ponowna obserwacja
   odmawia przy zmianie procesu lub konfiguracji.

Usługa zachowuje UID/GID, IPC, PostgreSQL i wszystkie dotychczasowe ustawienia
sandboxu, w tym NoNewPrivileges, puste capabilities oraz ProtectSystem=strict.
Szablon jednostki zawiera `@FRESHNESS_RUNTIME_PYTHON@`; materiał wydania musi
mieć to miejsce zastąpione zatwierdzoną ścieżką przed podpisaniem inwentarza.
Instalator nie przyjmuje nierozwiniętego szablonu ani alternatywnej komendy.

## Zamrożone wejścia buildera

[Lock](../../deployment/requirements-freshness-runtime.lock) zawiera pełny
zestaw runtime dla CPython 3.12. [Katalog wheelhouse](../../deployment/freshness-runtime-wheels.json)
wiąże dokładne nazwy plików, rozmiary, URL źródłowe i SHA-256:

| Dystrybucja | Wersja | Rola |
| --- | --- | --- |
| cryptography | 50.0.2 | runtime, backend OpenSSL w wheel |
| psycopg | 3.3.6 | runtime |
| psycopg-binary | 3.3.6 | runtime, binarne libpq i jego zależności |
| cffi | 2.1.1 | zależność cryptography |
| pycparser | 3.12 | zależność cffi |
| typing-extensions | 4.16.0 | zależność psycopg na CPython 3.12 |
| pip | 26.2.1 | wyłącznie narzędzie budowania, poza site-packages |

Źródłem są dokładne wheele z `files.pythonhosted.org`; pobrane bajty muszą
odpowiadać katalogowi. Katalog jest recenzowanym wejściem kodu, nie kluczem
zatwierdzającym wynik. Aktualizacja wersji lub hashy wymaga ponownego przeglądu.

Builder wymaga systemowego CPython 3.12 na Noble. Tworzy świeży venv bez pip,
nie kopiuje istniejącego venv. Zahashowany wheel pip jest osobnym narzędziem;
instaluje biblioteki wyłącznie do nowego środowiska z:

```text
pip --no-index --require-hashes --only-binary=:all:
```

Nie używa systemowego pip, user-site, PYTHONPATH, źródłowych pakietów ani
`--break-system-packages`. Usuwa skrypty aktywacji ze ścieżkami stagingu;
wynik zawiera stabilny pyvenv.cfg, wheelhouse, lock i zapis rzeczywistych
sond importów/Ed25519/ABI/OpenSSL/libpq. Pip nie trafia do site-packages.
Kontrola native obejmuje pliki `.so`, rozwiązaną listę zależności oraz systemowe
materiały potrzebne do uruchomienia zatwierdzonego interpretera.

Pobieranie jest oddzielną operacją online; budowanie i instalacja korzystają
z kompletnego wheelhouse offline. Na hoście docelowym publikowany jest gotowy,
niezależnie podpisany kandydat, bez pobierania pakietów i bez wykonywania pip.

## Publikacja i recovery

Stała struktura:

```text
/opt/cryptohunter/freshness-runtime/
  releases/<environment-id>/
  staging/<environment-id>/
  <environment-id>.journal
/etc/cryptohunter/freshness-runtime-authority.pub
/etc/cryptohunter/freshness-runtime.json
```

Katalog bazowy przygotowuje administrator. Publikacja najpierw weryfikuje
źródło, tworzy journal z wyłącznym tworzeniem pliku, kopiuje do osobnego stagingu,
sprawdza całość, synchronizuje pliki/katalogi i publikuje przez
`renameat2(RENAME_NOREPLACE)`. Nie nadpisuje istniejących wydań ani nie kasuje
poprzedniego runtime. Ponowna instalacja identycznego wydania sprawdza jego
integralność; niezgodność lub stan częściowy powoduje odmowę.

Journal i staging po przerwaniu wymagają ręcznej oceny. Nie usuwać ich
na podstawie samego komunikatu błędu. Operator zachowuje inwentarze, podpisy,
logi i snapshot VM, sprawdza czy rename nastąpił oraz niezależnie weryfikuje
wydanie. Dopiero po udokumentowanej decyzji może uporządkować konkretny,
nieaktywny staging/journal albo przywrócić snapshot. Recovery nie usuwa
poprzednich wydań, danych aplikacji, kont ani klastra PostgreSQL.

## Wersjonowanie i granice dowodów

Manifest instalatora v2 i jego receipt obejmują dokładne environment-id,
SHA-256 manifestu runtime, rewizję oraz bazowy interpreter. Historyczny v1
nie uzyskuje opcjonalnych pól i nie wystarcza do nowej instalacji.
`linux_release_identity.py` ma osobny ścisły certyfikat wydania v2; podpis
runtime nadal nie zastępuje autorytetu wydania produktu.

Fizyczna ścieżka dowodów ma własny ścisły profil. Jej migracja, pełny produkt,
AccountGenesis/Source/CCA oraz brakujące dowody systemd, clean-install,
reboot/lifecycle/update pozostają jawnymi blokerami. Nie zmieniono historycznych
TEST_ONLY, nie awansowano Stage 9 i nie ustawiono LINUX_PRODUCTION_READY=PASS.

Rzeczywisty builder jest osobno uruchamiany na Ubuntu 24.04 w workflow
[Freshness Python runtime](../../.github/workflows/freshness-runtime.yml).
Testy lokalne rozróżniają rzeczywiste operacje kryptograficzne/importy
od modeli granicy systemd i własności plików. Kontener nie zastępuje próby
natywnej VM z systemd PID 1. Procedura operatora:
[Ubuntu 24.04 — podpisany runtime](../runbooks/STAGE9_SIGNED_RUNTIME_VM_TRIAL.md).
