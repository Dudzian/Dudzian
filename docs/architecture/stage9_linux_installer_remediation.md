# PR #3235 — remediacja review instalatora Linux

Raport PR #3235 poniżej zachowuje historyczny kontekst. Aktualną poprawkę
odczytowej kontroli PostgreSQL opisuje
[dodatek host-probe z 2026-11-11](#remediacja-host-probe-postgresql--2026-11-11).

Data: 2026-11-02 (UTC / Atlantic/Reykjavik). Kontynuacja istniejącej gałęzi
`codex/stage9-linux-installer-safety`, baza review
`a5c4e882091a2ce3daaed0943ec1177fd88c17ff`. Bez nowego PR i bez merge.

## 1. Dokładna tożsamość uruchomionej usługi

Kwalifikacja nie wyszukuje już ścieżki jako fragmentu tekstu ExecStart.
`freshness_deployment_qualification.py` deleguje do nowego, odczytowego
`freshness_service_identity.py`. Wymagany jest jeden rekord ExecStart
odczytany jako typowana tablica `a(sasbttttuii)` z D-Bus systemd przez
`busctl --system --json=short get-property`. Porównywane są osobno program,
pełne argv, ignore-failure, metadane życia komendy i jej PID. Oczekiwane argv:

```text
/usr/bin/python3 -I -s -B /usr/lib/cryptohunter/deployment/freshness_verifier_entrypoint.py
```

Systemd musi wskazać załadowaną, aktywną jednostkę o właściwym fragmencie,
User/Group, MainPID=ExecMainPID, cgroup, środowisku i efektywnym sandboxie.
DropInPaths musi być puste, NeedDaemonReload=no. Nie przyjmujemy alternatywnych
źródeł environment, root image, supplementary groups ani podmienionego fragmentu.
Dwa pełne snapshoty konfiguracji i ExecStart muszą być zgodne.

MainPID jest przypinany przez pidfd, sprawdzane są jego kernelowe fdinfo
oraz brak sygnału zakończenia. Istniejący, niezmieniony
`capture_process_identity` sprawdza wszystkie UID/GID, grupy, start ticks,
hash obrazu i cgroup. Nowy helper niezależnie czyta NUL-separated cmdline
oraz `/proc/PID/exe`: rzeczywisty obraz musi odpowiadać rozwiązanemu systemowemu
Pythonowi przez ścieżkę, device/inode, metadane i SHA-256. Referencja musi być
root:root, wykonywalna i bez zapisu przez grupę/innych. Podmiana w trakcie
obserwacji, zgon/reuse PID, inny argv lub program kończą się odmową.

Oczekiwana ścieżka jako argument `echo`, shellowego `-c` albo innego programu
nie wystarcza. Zachowano dokładne hashe wcześniej przejrzanych modułów oraz
sprawdzenia plików, socketu, PostgreSQL i fizycznych kontraktów. Nie zmieniono
unit ani interpretera na podstawie domniemania kompatybilności. Windows ma
jawną odmowę przed użyciem natywnych kont; testy importów pozostają przenośne.

Rzeczywisty rootless D-Bus potwierdził różnicę formatu: `get-property` wypisuje
array bez dodatkowego wrappera metody `call`. Fixture ExecStart odwzorowuje
format właściwej operacji, a błędne opakowanie jest testem negatywnym.
To obserwacja lokalna, nie zdalna atestacja hosta ani pełny proof runtime.

## 2. Zgodność Ubuntu 24.04 i projekt runtime

Wykazano rzeczywistą niezgodność standardowych bibliotek Noble. Szczegółowe
źródła i projekt są w dodatku poniżej. Python 3.12 spełnia warunek języka,
ale APT dostarcza `cryptography 41.0.7` i `psycopg 3.1.17`, poniżej wymaganych
50.0.1 i 3.2. Obecny preflight ma odmówić; nie obniżono wymagań i nie użyto
systemowego pip. Dodatnia próba obecnego profilu na standardowym Noble jest
BLOCKED, nie PASS. Sam prywatny venv dewelopera nie naprawia hardcoded
/usr/bin/python3 użytego przez wrapper, preflight i jednostkę.

Przedstawiono projekt osobnego, podpisanego i chronionego CPython 3.12 runtime
z pełnym wheel/native lockiem i wiązaniem environment-id z manifestem aplikacji.
Nie wdrożono go bez kompletnego zatwierdzonego kontraktu; wymaga osobnego PR.

## 3. Pięć zgłoszeń GitGuardian

Bazowy check [125012223178](https://github.com/Dudzian/Dudzian/runs/125012223178)
ma failure, 5 Generic Password occurrences i 3 incydenty, potwierdzone w
[komentarzu bota](https://github.com/Dudzian/Dudzian/pull/3235#issuecomment-6565374973).
W bazowym `tests/deployment/test_production_local_installer.py` są to linie
555/557/560/976/978: monkeypatch `pwd.getpwall/getpwnam/getpwuid`.

Przejrzano callpath: `pwd` jest tu modułem POSIX, a dane to syntetyczne rekordy
NSS (UID 901/902, GID 801/802, home/nonexistent, nologin). Żadne z tych pięciu
miejsc nie zawiera wartości hasła, realnego sekretu ani użycia uwierzytelnienia.
Lokalny alias nazywa się teraz `account_database`; fixture i diagnostyka są
identyczne. Nie zmieniono skanowania, baseline, progów ani wyjątków. Nie
zamykano administracyjnie incydentów jako false positive. Wynik GitGuardian
nowego SHA musi zostać odczytany osobno; stary failure nie zmienia się w PASS.

## 4. Walidacja lokalna i jej granice

Nowe testy tożsamości obejmują dokładne argv, alternatywny program, script tylko
w argumentach, efektywny sandbox/drop-iny, MainPID, UID/GID/grupy/cgroup, pidfd,
inode/hash, odmowę dostępu i zmiany między snapshotami. Focused negative test
kompozycji potwierdza wywołanie helpera z właściwymi UID/GID oraz odmowę
przed `psycopg.connect`; wcześniejsze granice są modelowane tylko w tym teście.
Nie przypisuje on PASS całej kwalifikacji przy pominiętych pinach.

Natywne odczyty ograniczają się do własnego procesu potomnego i pidfd.
Executor ma interpreter należący do nobody:nogroup, więc produkcyjna kontrola
prawidłowo odmawia zaufania temu plikowi. Test pozytywnego odczytu cmdline,
inode i hash jawnie modeluje tylko root ownership; osobno sprawdza odmowę
niewłaściwego właściciela. Nie zmieniono systemowych metadanych. Nie jest to
dowód kwalifikacji docelowego Ubuntu. Rootless import sześciu rzeczywistych
modułów na CPython 3.12 opisano w dodatku; nie wywołano service.main.

Wyniki lokalne tej remediacji:

- nowy plik testów tożsamości: 112 PASS;
- łączny zestaw tożsamości, kwalifikacji, pinów i instalatora: 254 PASS;
- regresje verifiera, systemd, IPC, PostgreSQL, architektury Stage 9 oraz
  workflow: 576 PASS, 2 SKIP zgodnie z warunkami platformy;
- Ruff baseline, rozszerzone E/F/B/I nowych testów/helpera i jawny format: PASS;
- dokładny hook `python -m pre_commit run ruff-format --all-files`: PASS;
- configured mypy Linux/Win32: PASS, 383 pliki;
- blokujące reguły Semgrep dla dwóch zmienionych modułów produkcyjnych:
  zero znalezisk; Betterleaks staged: zero znalezisk, bez zmian baseline;
- helper ma 97% pokrycia z gałęziami; line diff coverage remediacji względem
  `a5c4e882` wynosi 98% przy niezmienionym progu 70%.

Zestaw 254 zawiera 112 nowych przypadków — nie sumować ich jako rozłącznych.
Pełny kwalifikator ma w tym pomiarze tylko 60% pokrycia: istniejące ścieżki
żywego PostgreSQL nie były uruchamiane. Nie deklarujemy ich kwalifikacji.
Próg, timeouty oraz manualne i fizyczne bramki pozostają bez zmian.
Wyniki GitHub nowego HEAD są podawane osobno w opisie PR i raporcie końcowym;
nie przenosimy wcześniejszego wyniku na późniejsze SHA.

## 5. GitHub Actions i nierozwiązane bezpieczeństwo

Dla starszego SHA z przeglądu `a5c4e882` odczytano obie strony check-runs: 111 kontroli,
94 success, 13 skipped, 4 failure. Quality and Security, CodeQL i Style były success;
Python testy Ubuntu/Windows/macOS także. Cztery niepowodzenia:

- GitGuardian: pięć opisanych powyżej miejsc fixture; oczekuje wyniku nowego SHA.
- [UI/QML w CI PR](https://github.com/Dudzian/Dudzian/actions/runs/41361362324/job/125028490003):
  16 374 testy PASS i 7 SKIP, a następnie SIGSEGV podczas shutdown z wątkami gRPC.
  Odpowiednik push przeszedł; przyczyny nie rozstrzygnięto. Cały job jest failure.
- [Linux deployment](https://github.com/Dudzian/Dudzian/actions/runs/41361362273/job/125025351057):
  brak CURRENT_EVIDENCE dla systemd, clean-install i reboot/lifecycle/update.
  Prawidłowa odmowa LINUX_PRODUCTION_READY=NOT_READY, nie regresja do zamaskowania.
- [Windows release](https://github.com/Dudzian/Dudzian/actions/runs/41361362273/job/125033852850):
  brak reboot recovery, long-running lifecycle, update/restart i failed-update
  safe behavior. Bramki pozostają bez zmian.

Aktualne stany alertów CodeQL #54/#58 pozostają UNVERIFIED: dostępne `gh api`
zwraca Forbidden. Sukces analizy nie dowodzi zamknięcia alertów. Nie szukano
innych tokenów ani nie zmieniano uprawnień. Wyniki nowego HEAD wymagają osobnego
odczytu, bez deklarowania PASS dla zadań queued/in_progress.

## 6. Próba VM, pozostałe blokery i merge

[Manualna procedura Ubuntu 24.04 VM](../runbooks/STAGE9_LINUX_UBUNTU2404_VM_TRIAL.md)
obejmuje native systemd, PostgreSQL 16, jawne konta/grupy, niezależne zatwierdzenie
podpisanego manifestu, rzeczywisty preflight, inwentarz, dane guard i backup,
instalację/idempotencję, odmowy, przerwanie oraz ręczny recovery. Jej faza
z bieżącym standardowym Ubuntu kończy się na poprawnej odmowie zależności. Dodatnia
część jest jawnie wstrzymana do osobnego wdrożenia kontraktu runtime; nie
używamy obejścia ani atrapy, żeby ją zaliczyć.

Nie uruchomiono testu uprzywilejowanego automatycznie i nie modyfikowano Della.
Pozostają blokery zależności/runtime Ubuntu, wcześniejszy drift pinu
AccountGenesis, pełna kompozycja produktu, dowody fizyczne/lifecycle i wynik
CI nowego SHA. UI shutdown oraz aktualny stan GitGuardian wymagają osobnego
potwierdzenia. Rekomendacja: pozostawić PR jako draft i nie wykonywać merge
na podstawie lokalnych testów. Nie ustawiać Stage 9 COMPLETE ani
LINUX_PRODUCTION_READY=PASS; STAGE9_PRODUCTION_READY pozostaje NO.

## Dodatek: źródła Ubuntu, sondy i projekt chronionego środowiska

Data weryfikacji: 2026-11-02 (UTC / Atlantic-Reykjavik). Analizowany bazowy HEAD PR: a5c4e882091a2ce3daaed0943ec1177fd88c17ff. Nie edytowano implementacji, jednostki ani konfiguracji hosta.

## Wniosek

Python 3.12 jest zgodny z bieżącym warunkiem wersji (>=3.11). Standardowy zestaw bibliotek z Ubuntu 24.04 Noble NIE spełnia jednak bieżącego kontraktu `cryptography>=50.0.1,<51` i `psycopg>=3.2,<4`. Preflight powinien odmówić, zamiast obniżać wymagania lub korzystać z systemowego pip. Nie potwierdzono dodatniej instalacji na natywnej Ubuntu 24.04 z systemd PID1.

Nie zmieniono wymagań ani wykonywalnego programu jednostki. Potrzebne jest odrębne, chronione środowisko wraz z pełnym podpisanym kontraktem, a nie samo podstawienie venv do ExecStart.

## Źródła Ubuntu

Odczytano oficjalne strony Canonical/Ubuntu Packages:

| Składnik | Opublikowany pakiet Noble | Bieżący warunek projektu | Wynik porównania |
| --- | --- | --- | --- |
| Python | python3.12 3.12.3-1ubuntu0.17, amd64/security | >=3.11 | wersja języka spełnia warunek |
| cryptography | python3-cryptography 41.0.7-4ubuntu0.4, amd64/security | >=50.0.1,<51 | poniżej minimum |
| psycopg | python3-psycopg 3.1.17-2, all/universe | >=3.2,<4 | poniżej minimum |

Źródła:
- https://packages.ubuntu.com/noble/python3.12
- https://packages.ubuntu.com/noble/python3-cryptography
- https://packages.ubuntu.com/python3-psycopg
- https://packages.ubuntu.com/search?keywords=psycopg&suite=noble
- pomocniczy oficjalny filelist potwierdzający dist-info psycopg 3.1.17: https://packages.ubuntu.com/noble/all/python3-psycopg/filelist

Strona python3-psycopg bezpośrednio pod /noble miała błąd pobrania; użyto oficjalnej wyszukiwarki suite Noble i jej filelist, nie domniemania z wersji innego Ubuntu. Są to opublikowane indeksy WWW, nie wynik `apt-cache policy` z docelowej VM ani dowód stanu konkretnego hosta. Wersje kandydatów operator musi zachować ponownie w dniu próby, razem ze źródłami APT.

## Bezpośrednia kontrola kodu

`deployment/production_local_installer_host.py:inspect_host` uruchamia dokładny /usr/bin/python3 z -I -s -B, importuje cryptography/psycopg, sprawdza ich wersje przez importlib.metadata i dopiero wtedy ustala python_ready. Brak lub niewłaściwe wersje są odmową. Wrapper `deployment/install-production-local.sh` również używa /usr/bin/python3. Sam launcher nie instaluje bibliotek i nie wybiera venv z PATH.

`deployment/freshness_verifier_entrypoint.py` tworzy stałą przestrzeń pakietu bot_core i importuje rzeczywisty `bot_core.freshness_verifier_service`, pomijając szeroki initializer. Lokalny łańcuch obejmuje dokładnie:
- freshness_verifier_service;
- freshness_semantic_verifier;
- postgresql_freshness_authority;
- linux_service_ipc;
- linux_service_postgresql;
- production_signer_ipc.

Te moduły importują zewnętrzne cryptography i psycopg. Do kompletnego środowiska trzeba ponadto włączyć transytywne oraz natywne zależności konkretnego wybranego wariantu dystrybucji, a nie tylko dwa nazwy pakietów.

## Wykonane sondy bez uprawnień administratora

W istniejącym prywatnym venv w /workspace/work, bez modyfikacji systemowego Pythona, uruchomiono świeży podproces `-I -s -B`:

- CPython 3.12.14;
- cryptography 50.0.2;
- psycopg 3.3.6;
- psycopg-binary 3.3.6;
- cffi 2.1.1;
- pycparser 3.12.

Dokładne wyrażenie wersji ze źródła preflight dało True. Rzeczywista operacja Ed25519 sign/verify dała OK, używając wyłącznie nowego syntetycznego klucza w pamięci. Psycopg załadował implementację binary i zgłosił libpq 180006. Nie otwarto połączenia PostgreSQL.

Drugi świeży podproces odtworzył namespace launchera i zaimportował rzeczywisty verifier z sześcioma wskazanymi modułami. Import przeszedł; nie wykonano bot_core/__init__.py ani service.main, sond NSS, bind, systemd lub operacji bazy. Nie obchodzono ochrony instalacji i nie uruchamiano usługi — była to sonda import closure, a nie kwalifikacja launchera na zainstalowanym hoście.

To potwierdza wykonalność łańcucha na odizolowanym Linux CPython 3.12, lecz nie natywną zgodność konkretnego obrazu Ubuntu, pakietów ABI, systemd czy przygotowanego wdrożenia. Nie uruchamiano Dockera, VM ani testu uprzywilejowanego.

W lokalnych metadanych:
- cryptography 50.0.2 na CPython wymaga cffi>=2.0.0;
- cffi wymaga pycparser;
- psycopg 3.3.6 z extra binary wymaga dokładnie psycopg-binary==3.3.6;
- psycopg na Python<3.13 wymaga typing-extensions>=4.6;
- konkretne wheels zawierają lub wiążą natywne biblioteki, których pełny inwentarz/proweniencja również wymaga zatwierdzenia.

Wersje lokalnego venv nie są proponowanym produkcyjnym lockiem i nie wolno przenosić go na hosta.

## Oczekiwany wynik realnego preflight na domyślnym Noble

Po instalacji wyłącznie oficjalnymi narzędziami Ubuntu pakietów python3-cryptography/python3-psycopg, z prawidłowym zewnętrznym kluczem, podpisanym manifestem i bundle, kontrola python_ready powinna zakończyć się odmową 20. To właściwy wynik wymagający zachowania raportu i braku mutacji.

Pierwsza niespełniona bramka kończy przebieg. Bez cryptography podpis może być nieweryfikowalny i dać wcześniejszą odmowę 10; uszkodzony manifest, uprawnienia lub tożsamości też mogą dać inne wcześniejsze kody. Nie wolno przedstawiać błędu 20 jako dowodu, że sprawdzono już brak klastra, ani oczekiwać koniecznie kodu 21 na tym samym nieprzygotowanym hoście.

Dodatnia próba instalacji obecnego profilu na standardowym Noble pozostaje BLOCKED. Nie obchodzić tego przez `sudo pip install`, `--break-system-packages`, user-site, PYTHONPATH, .pth, podmianę /usr/bin/python3 albo obniżenie wymagań.

## Projekt oddzielnego, chronionego runtime — do osobnego zatwierdzenia

Proponowana stała ścieżka to /opt/cryptohunter/freshness-runtime/releases/<environment-id>/bin/python3.12. Wskazanie konkretnej wersji/ID musi pochodzić z zatwierdzonego profilu; proces i manifest nie mogą swobodnie wybierać interpretera przez argumenty lub środowisko. Ścieżka jest propozycją, nie zaimplementowanym wyjątkiem obecnej kwalifikacji.

1. Osobny builder bez root przygotowuje wheelhouse dla CPython 3.12/Ubuntu24.04/amd64, pełny transytywny lock exact-version + SHA-256 i zgodne wheels bez instalacji z sieci na hoście. Odrębnie audytuje źródła, podatności, licencje i natywny OpenSSL/libpq/ABI. Nie generować locka z przypadkowego venv developera.
2. Manifest środowiska ma niezależny podpis i identyfikator. Wiąże OS/architekturę/ABI, bazowy pakiet Ubuntu interpretera, hash wykonywalnego pliku, pyvenv.cfg, include-system-site-packages=false, wszystkie zainstalowane pliki/tryby, wheels i ich hashe oraz cały łańcuch bibliotek natywnych. Żadne niezatwierdzone .pth/import hooks, user-site i lokalny bytecode nie mogą wejść do grafu.
3. Uprzednio zaufany bootstrap administratora waliduje niezależne materiały PRZED uruchomieniem interpretera nowego środowiska. W przeciwnym razie weryfikator importowałby niezweryfikowane biblioteki. Samocertyfikacja już uruchomionego procesu nie ustanawia zaufania.
4. Dopiero zatwierdzona, osobna ceremonia tworzy venv w chronionym stagingu z bazowego python3.12-venv zarządzanego APT. Instalacja pip wyłącznie wewnątrz venv: --no-index --require-hashes --only-binary=:all: z pełnym lockiem. Systemowy Python i dist-packages pozostają własnością APT. Po instalacji następują inwentarz/integralność, root:root, katalogi0755 i brak zapisu przez konta usług. Bez rekurencyjnych napraw przypadkowego zastanego środowiska.
5. Publikacja jest wersjonowana i bez nadpisywania. Dane PostgreSQL, kod aplikacji i runtime mają odrębne obszary/receipt. Bazowy interpreter Ubuntu bywa symlinkiem; kontrakt musi jawnie dopuścić tylko dokładne zatwierdzone wiązanie do pakietu Ubuntu, sprawdzać przodków, inode/hash i odmówić innych linków. Nie wolno uznać dowolnego symlinka za bezpieczny.
6. Obsługa uaktualnień Ubuntu bazowego interpretera wymaga ponownej oceny i odtworzenia środowiska, albo świadomie określonego zgodnego mechanizmu. Hash nie może być automatycznie aktualizowany po zaobserwowaniu zmiany.
7. Ewentualna wspólna read-only baza interpretera nie zmienia odrębnych UID/GID, IPC, sandboxu i separacji usług. Konto usługi nie może mieć prawa zapisu do venv, wheels, manifestów, pyvenv.cfg, unit ani kodu.

## Wpływ na istniejący kontrakt instalacji

Nie wystarczy zmienić jednego ExecStart:

- jednostka systemd musi wskazywać dokładny zatwierdzony interpreter środowiska z niezmienionymi -I -s -B i pozostałym sandboxem;
- wrapper instalatora i jego własny bootstrap kryptograficzny także muszą korzystać z uprzednio zaufanego środowiska albo niezależnego weryfikatora podpisów; obecny wrapper na sztywno wywołuje /usr/bin/python3;
- preflight musi sprawdzać środowisko używane przez jednostkę, weryfikować jego katalogi/pyvenv.cfg/inwentarz i import provenance, a nie wersje innego systemowego interpretera;
- tożsamość uruchomionej usługi wymaga nowej dokładnej listy argv oraz wiązania /proc/MainPID/exe z faktycznym bazowym interpreterem, inode/hash, UID/GID, cgroup i PIDFD. String symlinka venv i /proc/exe zwykle różnią się; nie wolno akceptować tego przez substring;
- manifest kodu aplikacji musi związać zatwierdzony environment-id i hash/kontrakt podpisanego manifestu środowiska. To wymaga wersjonowanego rozszerzenia, a nie arbitralnego pola w bieżącym ścisłym schema;
- recovery, idempotencja i aktualizacja muszą rozdzielać środowisko i kod bez kasowania poprzedniego wydania ani danych. Każda niezgodność daje odmowę;
- pełny linux_release_identity/physical profile również wymaga osobnej aktualizacji zatwierdzenia materiałów; scoped podpis instalatora sam nie kwalifikuje nowego środowiska;
- testy obejmują Ubuntu24.04, import closure/native ABI, odseparowanie system-site/user-site, .pth/pyc, drift bazowego interpretera, niechronione ścieżki, symlinki, niedozwolone członkostwa i awarie publikacji.

## Zakres pozostały i decyzja

W tym zadaniu wykazano rzeczywistą niezgodność, potwierdzono izolowany import na CPython 3.12 i przedstawiono projekt jej rozwiązania. Nie implementowano runtime lub zmiany jednostki bez kompletnego podpisanego kontraktu. Właściwa dodatnia próba systemd/instalacji na Ubuntu pozostaje do odrębnego przygotowania i wykonania operatora, bez fizycznego Della. Nie deklarować PASS dla instalacji Ubuntu, nie ustawiać LINUX_PRODUCTION_READY=PASS ani Stage 9 COMPLETE.

## Remediacja host-probe PostgreSQL — 2026-11-11

`include_dir` jest dyrektywą parsera PostgreSQL, nie parametrem GUC.
W pakiecie Ubuntu Noble `postgresql-common 257build1.1` funkcja
`PgCommon::read_conf_file()` rozwija zawartość katalogu i scala parametry
plików, ale nie zapisuje klucza `include_dir`. `pg_conftool … show include_dir`
odczytuje ten wynik; przy nieobecnym kluczu zwraca 1 bez stdout/stderr.
Taki wynik nie dowodzi braku aktywnej dyrektywy ani awarii klastra.
Odtworzono go rzeczywistym narzędziem z rozpakowanych pakietów Ubuntu,
na plikach tymczasowych, bez instalacji pakietów i zmian konfiguracji hosta.
Odczyty `port` i parametru z pliku `conf.d` równocześnie kończyły się kodem 0.

Źródła sprawdzone bezpośrednio:

- [Pakiet postgresql-common 257build1.1](https://archive.ubuntu.com/ubuntu/pool/main/p/postgresql-common/postgresql-common_257build1.1_all.deb):
  `PgCommon.pm`, linie 197–212; `pg_conftool`, linie 125–134.
- [Pakiet postgresql-client-common 257build1.1](https://archive.ubuntu.com/ubuntu/pool/main/p/postgresql-common/postgresql-client-common_257build1.1_all.deb).
- [Parser PostgreSQL 16](https://github.com/postgres/postgres/blob/REL_16_STABLE/src/backend/utils/misc/guc-file.l#L430):
  dyrektywa include jest przetwarzana od razu, bez tworzenia zmiennej.

`inspect_host()` zachowuje wymaganie klastra `16/main`, właściciela `postgres`,
portu 5432 i stanu `online`, a jednostka `postgresql@16-main.service` musi mieć
odpowiedź z dokładnie dwóch pól: `LoadState=loaded` i `ActiveState=active`.
Review PR #3264 ujawnił dwie dalsze granice wymagające jawnej polityki.

### P1: Ubuntu i root-owned przestrzeń nazw instalatora

`pg_createcluster` z pakietu 257build1.1 przypisuje katalog konfiguracji do
właściciela klastra (linie 384–385), a `conf.d` tworzy przez `mkdir(..., 0755)`
i przypisuje do tego samego właściciela (linie 583–584). Umask może ograniczyć
tryb tworzonego katalogu. Zgłoszony na `CryptoHunter-Stage9-ServiceTrial`
`postgres:postgres 0700` nie jest dowodem awarii PostgreSQL.

Jednak wcześniejszy, niezależny od sondy kontrakt
`production_local_installer.py::_Tree._directory()` i `_preflight()` wymaga
root-owned `0755` dla `conf.d` i jego przodków. Pozostaje on bez zmian;
sonda stosuje tę samą politykę. Nie dodano wyjątku dla `postgres` jako
właściciela katalogu. `0700` wyklucza obce konta, lecz właściciel `postgres`
nadal może dodawać i zastępować nazwy plików, w tym root-owned drop-in
instalatora. Taki katalog nie zapewnia jego wymaganej ochrony publikacji.
Akceptacja postgres-owned pliku głównego nie uzasadnia rozszerzenia prawa
do zmiany całej przestrzeni nazw.

Zgłoszona VM pozostaje zatem **BLOCKED przed S3**. Osobny
[plan weryfikacji i propozycja kontrolowanej remediacji](../runbooks/STAGE9_HOST_PROBE_REMEDIATION_VM_PLAN.md)
wymagają inwentaryzacji wszystkich rodziców i zatwierdzenia konkretnych
ścieżek przez operatora. Nie wykonano ani nie dodano automatycznego
`chown`, `chmod`, reload czy restartu VM.

### P2: dwa zamknięte profile konfiguracji i uwierzytelniania

Stały lokalny SELECT przez `/usr/lib/postgresql/16/bin/psql` używa roli SQL
`postgres`, socketu `/var/run/postgresql`, portu 5432 i bazy `postgres`.
Najpierw próbuje peer root→postgres; tylko niedostępne polecenie może
spowodować próbę przez `runuser -u postgres` dla peer postgres→postgres.
Odpowiedź dostępna, ale niezgodna z kontraktem, powoduje odmowę bez fallbacku.
Opcje `-X -w`, `passfile=/dev/null`, `ON_ERROR_STOP=1`, timeout połączenia 5 s,
timeout zapytania 5000 ms, `default_transaction_read_only=on` i
`search_path=pg_catalog` są stałe.
Podproces ma ograniczony czas, stałe środowisko i nie używa shella.

Wymagane są zgodne `config_file`, `data_directory`, port i wersja 16.x,
`system_user` równe `peer:root` albo `peer:postgres`, obie role SQL równe
`postgres` i rzeczywista transakcja read-only. Wiersze `pg_file_settings`
mogą pochodzić tylko z głównego pliku lub płaskiego `conf.d`.
`/var/lib/postgresql/16/main/postgresql.auto.conf` jest dodatkowo czytany
bezpośrednio: dopuszczalne są wyłącznie puste wiersze i komentarze UTF-8
bez NUL. Każda dyrektywa, także include pustego pliku niewidocznego w widoku
SQL, powoduje odmowę. Nie wystarcza brak wierszy GUC z tego źródła.
Brak widoków HBA/ident/GUC, błędy parsowania lub `pending_restart` oznaczają
odmowę. Dla zastosowanych wpisów krytycznych GUC porównuje się wartość
z dysku i z `pg_settings`: listen, socket, port, katalog danych, ścieżki
HBA/ident, `fsync` oraz `synchronous_commit`. Obie ostatnie wartości muszą
być efektywnie dokładnie `on`.

| Profil | Efektywny listen | Efektywny katalog socketu | HBA i ident |
| --- | --- | --- | --- |
| Przygotowany Ubuntu | dokładnie `localhost` | dokładnie `/var/run/postgresql` | domyślne pliki w `/etc/postgresql/16/main`, siedem reguł HBA i pusty ident |
| Wdrożony profil projektu | dokładnie pusty | dokładnie `/run/postgresql` | pliki w `/etc/cryptohunter/postgresql`, pięć reguł HBA i trzy mapowania ident z payloadu |

Dopuszczenie `/run/postgresql` wynika wyłącznie z dostarczanego
`production-local.conf`; nie jest ogólnym wyjątkiem dla alternatywnych socketów.
`*`, inne adresy, listy adresów/socketów i mieszanie obu profili powodują odmowę.
Reguły HBA są porównywane w całości, w kolejności, z tablicami baz/użytkowników,
adresem, maską, metodą i opcjami. Ubuntu wymaga kolejno: lokalnego peer dla
`all/postgres`, lokalnego peer dla `all/all`, loopback SCRAM dla `all/all`
(IPv4 i IPv6), lokalnego peer dla `replication/all` i loopback SCRAM dla
`replication/all` (IPv4 i IPv6). Wdrożony profil wymaga dokładnych pięciu
wierszy `deployment/postgresql/pg_hba.conf` i trzech mapowań
`deployment/postgresql/pg_ident.conf`. Dodatkowy `trust`, nawet za poprawnym
peer lub reject, zmieniona kolejność/opcja/mapowanie i reguła z innego pliku
powodują odmowę. Statyczny odczyt HBA/ident dodatkowo odrzuca include,
również pusty, listy `@plik` i kontynuacje wierszy. Uwierzytelnienie jednej
sesji nie zastępuje tej kontroli.

### Ochrona odczytu i zgodność z pełnym startem serwera

Snapshot obejmuje główny plik, aktywne nieukryte `conf.d/*.conf`, domyślną
parę HBA/ident, `postgresql.auto.conf` i, jeżeli istnieje, parę HBA/ident
wdrożonego profilu.
W głównym pliku wymagana jest jedna dyrektywa include wskazująca `conf.d`.
Brak, zła wartość, duplikat, inne/zagnieżdżone include lub wielowierszowa
i niejednoznaczna składnia powodują odmowę. To ograniczony profil instalatora.

Odczyt przez przypięte deskryptory używa `O_NOFOLLOW`. Pliki są zwykłe,
z jednym hardlinkiem, należą do root albo rzeczywistego UID `postgres`,
bez zapisu grupy/innych. Katalogi głównej konfiguracji i przodkowie są
root-owned `0755`; wyłącznie `/etc/cryptohunter/postgresql` zachowuje odrębny
kontrakt root:postgres `0750` z oczekiwanym GID, a jego rodzice nadal root `0755`.
Dojście do stałego `postgresql.auto.conf` jest odrębnym odczytem: wyłącznie
`/var/lib/postgresql`, `/var/lib/postgresql/16` i `/var/lib/postgresql/16/main`
mogą należeć do root lub rzeczywistego `postgres`, z trybem `0700`, `0750`
albo `0755`. `/var` i `/var/lib` pozostają root `0755`. Ten wąski wyjątek
obserwacji katalogów danych nie dotyczy `/etc`, nie daje prawa zapisu
i nie zmienia polityki P1. Sam `postgresql.auto.conf` podlega tym samym
kontrolom nofollow, hardlinków, właściciela, no022, rozmiaru i stabilności
oraz porównaniu mtime/ctime ze startem serwera co pozostałe chronione pliki.
Bajty i pełne kontrolowane metadane muszą być stabilne podczas
odczytu i zgodne w snapshotach przed/po SQL. Limity wynoszą 1 MiB na plik,
8 MiB łącznie i 1024 aktywne pliki w `conf.d`; niepoprawne UTF-8/NUL
w konfiguracji GUC i błędy odczytu powodują odmowę.

`pg_hba_file_rules` i `pg_ident_file_mappings` opisują pliki na dysku.
`pg_conf_load_time()` nie dowodzi udanego przeładowania wszystkich reguł HBA.
Dlatego odpowiedź SQL to dokładny obiekt JSON z `ready=true` i dodatnim,
nieprzyszłym `postmaster_start_ns`. Powtórzone klucze, niewłaściwe typy i
dodatkowe pola są odrzucane. Każdy chroniony plik oraz katalog `conf.d`
muszą mieć mtime i ctime nie późniejsze niż **pełny start postmastera**.
Zmiana po starcie powoduje odmowę, także po poprawnym reload. Ewentualny
pełny restart wymaga osobnego zatwierdzenia operatora; sonda go nie wykonuje.
Nie można zatem uznać nowo zapisanej konfiguracji instalacji za aktywną
przez sam ponowny preflight. Granicę bieżącego harnessu opisuje plan VM.

Ochrona przestrzeni nazw i stabilny odczyt nie są atestacją przeciw
złośliwemu procesowi PostgreSQL ani administratorowi root. Dopuszczone
postgres-owned pliki i odpowiedzi serwera zachowują tę granicę zaufania;
nie deklaruje się ochrony metadanych całego PGDATA ani wszystkich plików
PostgreSQL. Testy jednostkowe i jednorazowe klastry potwierdzają tylko
obserwowane granice implementacji, a nie zaliczenie ServiceTrial.

Wykonanie odbywa się w ramach istniejącej
[procedury v2](../runbooks/STAGE9_LINUX_VM_HARNESS.md), z jej niezależnym
zatwierdzeniem i podpisanym runtime. Zwykłe testy PR nie tworzą approval,
nie modyfikują VM i nie uruchamiają destrukcyjnej kwalifikacji. Progi, UID/GID,
IPC oraz wymagania podpisanego runtime pozostają bez zmian.
Stage 9 pozostaje **IN_PROGRESS / PRODUCTION_NO_GO**.
