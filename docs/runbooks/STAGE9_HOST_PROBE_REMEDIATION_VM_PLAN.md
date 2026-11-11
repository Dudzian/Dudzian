# Stage 9 — plan weryfikacji remediacji host-probe PR #3264

Plan dotyczy Ubuntu Server 24.04.5 LTS amd64, PostgreSQL 16.15 i
`postgresql-common 257build1.1`. Nie jest zapisem wykonanej próby ani zgodą
na zmianę `CryptoHunter-Stage9-ServiceTrial`. W tym PR nie zmieniano jej
konfiguracji, właścicieli, usług ani danych. Stage 9 pozostaje
**IN_PROGRESS / PRODUCTION_NO_GO**.

## Stan wejściowy i punkt zatrzymania przed S3

Operator zgłosił S0-clean, S1-packages i S2-substrate jako VERIFIED,
16/main ONLINE, guard database i system identities VERIFIED, sieć ISOLATED
oraz Production NO_GO. Nie zastępuje to ponownej obserwacji badanego SHA.
Zgłoszony `/etc/postgresql/16/main/conf.d` postgres:postgres `0700` powoduje
odmowę zgodnie z zachowanym kontraktem instalatora: katalog i jego przodkowie
muszą należeć do root i mieć dokładnie `0755`. Nie tworzono wyjątku
własności dla konta `postgres`.

Do czasu osobnej decyzji operatora **S3 i dodatni preflight pozostają BLOCKED**.
Stan działającego PostgreSQL i zgodność polityki instalacji są odrębnymi
faktami. Nie wykonywać na ServiceTrial automatycznego chmod/chown, reload,
restartu, edycji konfiguracji ani destrukcyjnych testów.

## 1. Odczytowe dowody przed próbą

Uprawniony operator zachowuje poza wejściami instalatora: nazwę VM,
identyfikatory obrazu/snapshotów, boot-id, zatwierdzony commit i pełne wersje
pakietów. Nie generuje approval z wyniku sondy. Przygotowanie chronionej kopii
przejrzanego kodu i wejść podlega istniejącej
[procedurze v2](STAGE9_LINUX_VM_HARNESS.md); nie uruchamiać edytowalnego
checkoutu ani niezweryfikowanego bundle jako root.

Następujące obserwacje są odczytowe; zapisywanie stdout jako dowodu jest
osobną czynnością operatora:

```sh
cat /etc/os-release
cat /proc/1/comm
cat /proc/sys/kernel/random/boot_id
/usr/bin/pg_lsclusters --no-header 16 main
/usr/bin/systemctl show postgresql@16-main.service \
  --property=LoadState,ActiveState,MainPID,ActiveEnterTimestamp
/usr/bin/systemctl show cryptohunter-freshness-verifier.service \
  --property=LoadState,ActiveState,FragmentPath,DropInPaths,MainPID
sudo /usr/bin/stat -c '%n %F %u:%g %a %h %s %y %z' \
  / /etc /etc/postgresql /etc/postgresql/16 /etc/postgresql/16/main \
  /etc/postgresql/16/main/conf.d
```

Nie podążać za wykrytym symlinkiem w celu uznania go za zgodny katalog.
Osobno zachować UID/GID i członkostwa kont usługowych oraz `postgres`,
spójność NSS name→id→name, interfejsy i adresy sieciowe, nasłuch TCP/Unix,
inwentarz aktywnych `conf.d/*.conf` i chronionych plików HBA/ident.
Nie odczytywać `shadow`, haseł ani sekretów do raportu.

Chroniony snapshot sondy obejmuje bajty i metadane: device/inode, tryb,
UID/GID, liczbę hardlinków, rozmiar, mtime/ctime. Dotyczy `postgresql.conf`,
nieukrytych `conf.d/*.conf`, domyślnych `pg_hba.conf`/`pg_ident.conf`,
stałego `/var/lib/postgresql/16/main/postgresql.auto.conf`, katalogu `conf.d`
i ewentualnej wdrożonej pary w
`/etc/cryptohunter/postgresql`. Dla tej ostatniej ścieżki obowiązuje
root:postgres `0750` z oczekiwanym GID; jej rodzice pozostają root `0755`.
Nie rozszerzać tego wyjątku na `conf.d` ani pozostałych przodków.
Odrębny odczyt `postgresql.auto.conf` dopuszcza root albo rzeczywistego
`postgres` z trybem `0700`/`0750`/`0755` tylko na trzech stałych katalogach:
`/var/lib/postgresql`, `/var/lib/postgresql/16`, `/var/lib/postgresql/16/main`.
`/var` i `/var/lib` nadal wymagają root `0755`; wyjątek nie obejmuje `/etc`.
Plik auto.conf musi być zwykły, pojedynczo dowiązany, root/postgres, no022,
ograniczony rozmiarem i stabilny. Jego bajty oraz mtime/ctime wchodzą do
porównania przed/po i do granicy pełnego startu; nie deklaruje to ochrony
pozostałego PGDATA ani nie wykonuje w nim zapisów.

## 2. HBA, listen i aktywna konfiguracja

Zachować wynik stałego zapytania z badanego, przejrzanego
`production_local_installer_host.py`, bez przepisywania jego warunków
na potrzeby VM. SQL musi działać przez istniejący peer, rolę `postgres`,
stały lokalny socket/port i read-only, z timeoutami i wyłączonym passfile.
Nie zmieniać HBA ani mapowań, by udostępnić sondzie połączenie.

Porównać cały profil, nie tylko metodę bieżącej sesji:

| Obserwacja | Przygotowany Ubuntu | Wdrożony profil projektu |
| --- | --- | --- |
| `listen_addresses` | dokładnie `localhost` | dokładnie pusty |
| `unix_socket_directories` | dokładnie `/var/run/postgresql` | dokładnie `/run/postgresql` |
| HBA/ident | domyślna para w `/etc/postgresql/16/main` | para w `/etc/cryptohunter/postgresql` |
| HBA | dokładne siedem reguł: lokalne peer i loopback SCRAM | dokładne pięć reguł z payloadu: peer z mapami i reject |
| ident | bez mapowań | dokładne trzy mapowania z payloadu |

Dokładne wiersze, kolejność, adresy/maski i opcje definiują stałe sondy,
opisane w [analizie remediacji](../architecture/stage9_linux_installer_remediation.md#p2-dwa-zamknięte-profile-konfiguracji-i-uwierzytelniania).
Każda dodatkowa lub zmieniona reguła, w szczególności `trust` obok poprawnego
peer, jest odmową. Niedostępny widok, błędy HBA/ident, inna tożsamość sesji
i mieszanie profili również powodują odmowę. Statyczny odczyt HBA/ident
odrzuca include (także pusty), listy `@plik` i kontynuacje wierszy, aby
niewidoczna w wyniku widoku zależność nie omijała ochrony źródeł.

Oczekiwany serwer wskazuje właściwe config_file/data_directory/port i wersję,
brak `pending_restart` oraz zgodne wartości krytycznych GUC na dysku i aktywnie,
z efektywnymi `fsync=on` i `synchronous_commit=on`.
Stały search_path sondy to `pg_catalog`. Bezpośredni odczyt auto.conf dopuszcza
tylko puste wiersze i komentarze UTF-8 bez NUL; każda dyrektywa, również
include pustego pliku niewidoczny w SQL, powoduje odmowę.
Odpowiedź SQL musi mieć dokładnie `ready=true` i `postmaster_start_ns`.
Snapshot musi być stabilny i pochodzić sprzed tego pełnego startu serwera.
Każda chroniona zmiana po starcie, nawet po poprawnym reload, powoduje odmowę.
Nie zastępować tej granicy `pg_conf_load_time()`: nie dowodzi ono udanego
przeładowania całego HBA. Sonda nie wykonuje naprawy ani aktywacji.

## 3. Niezmienność guard i porównanie po inspekcji

Przed i po pojedynczej inspekcji uprawniony operator wykonuje w istniejącej
bazie `stage9_vm_guard`, przez zgodny peer i z ograniczeniami połączenia sondy:

```sql
BEGIN READ ONLY;
SELECT COALESCE(jsonb_agg(jsonb_build_array(id, marker) ORDER BY id), '[]'::jsonb)
FROM public.installation_guard;
ROLLBACK;
```

Zachować i porównać dokładny wynik oraz hash istniejącego sprawdzonego backupu.
Nie tworzyć, nie wstawiać, nie aktualizować ani nie odtwarzać guard na aktualnej
ServiceTrial. Jeżeli jej zweryfikowany guard ma inny schemat, najpierw uzgodnić
odpowiedni odczyt; nie przebudowywać bazy do przykładu.

Porównać także chronione snapshoty, NSS, listę klastrów, PID i stan usług,
nasłuch oraz stdout/kody zakończenia. Oczekiwana odmowa na `conf.d` nie może
zmienić tych danych. Wykonanie SELECT może aktualizować statystyki i logi
serwera; nie są one dowodem modyfikacji konfiguracji przez sondę. Nie haszować
aktywnego PGDATA jako dowodu niezmienności całej bazy. Równość wybranych wierszy
guard nie dowodzi równości wszystkich danych PostgreSQL.

`checked_preflight()` porównuje tylko zakres plików opisany w procedurze v2,
więc nie zastępuje tych niezależnych obserwacji usług i guard.

## 4. Osobna propozycja remediacji wymagająca decyzji operatora

To propozycja do przeglądu, bez zgody na wykonanie i bez poleceń zmieniających
aktualną ServiceTrial. Preferowanym miejscem pierwszej próby jest osobny
zatwierdzony klon S2, z możliwością odtworzenia całego obrazu.

| Potwierdzona ścieżka | Zgłoszony stan | Proponowany stan zgodny z istniejącym instalatorem |
| --- | --- | --- |
| `/etc/postgresql/16/main/conf.d` | postgres:postgres `0700` | UID root, zweryfikowany dotychczasowy GID bez rozszerzania członkostw, dokładnie `0755` |

Przed zatwierdzeniem operator sporządza listę rzeczywistych metadanych
każdego rodzica. Ubuntu może również przypisywać katalog wersji lub klastra
do `postgres`. Każda dodatkowa niezgodna ścieżka wymaga własnej pozycji
z uzasadnieniem i zgodą; nie zgadywać ich stanu i nie stosować rekursywnego
chown/chmod ani szerokiego wyjątku własności. Nie zmieniać PGDATA, kont,
plików konfiguracji, danych ani podpisanego runtime w ramach tej propozycji.

Po zaakceptowanej zmianie metadanych czas ctime będzie późniejszy niż start
postmastera. Dodatnia próba wymagałaby zatem również osobno zatwierdzonego
pełnego restartu PostgreSQL na tym klonie, oceny ryzyka/dostępności i
ponownych dowodów przed/po. Nie wykonywać go w sondzie ani automatycznie po
odmowie. Restart lub restore zmieniający boot-id wymaga niezależnej oceny
istniejącego approval zgodnie z procedurą v2; nie aktualizować go automatycznie.

Po instalacji nowego payloadu zachodzi dodatkowa granica: pliki zapisane
po starcie oraz nieaktywne listen/HBA powodują odmowę. Jeśli postflight zwróci
dokładnie kod instalatora 21, bieżąca faza `install` zwraca `BLOCKED`,
`reason=POST_INSTALL_PREFLIGHT_REFUSED`, pełny wynik postflight i
`identical_reinstall=NOT_EXECUTED`. Inne kody pozostają dotychczasową odmową.
Nie omijać postflight ani nie usuwać receipt/journala. Operator musi osobno
ocenić opublikowany stan i propozycję aktywacji klona; sam restart nie musi
usunąć przyczyny odmowy. Po zatwierdzonej aktywacji można ponowić tę samą
fazę `install`: wykonuje pełny preflight już zainstalowanego profilu oraz
kontrolę identycznej reinstalacji. Dopiero ich rzeczywiste wyniki stanowią
dowód, a nie sam restart lub modele jednostkowe. Natywna pełna sekwencja
pozostaje do odrębnego potwierdzenia; nie dodano nowej fazy, pola approval
ani automatycznego restartu.

## 5. Dodatnie i negatywne przypadki wyłącznie na osobnych klonach

Każda próba wymaga zdrowej kontroli bazowej, jednej udokumentowanej zmiany,
właściwej granicy odmowy i porównania stanu. Niezerowy kod dowolnego wcześniejszego
warunku nie zalicza oczekiwanego przypadku. Przygotowanie konfiguracji przed
pełnym startem klastra jest osobną czynnością operatora klona.

| Przypadek | Oczekiwany wynik |
| --- | --- |
| Root-owned `0755`, pojedynczy include conf.d, pełny profil Ubuntu | Dodatnia inspekcja PostgreSQL po pełnym starcie; cały preflight wymaga też podpisanego runtime i pozostałych bramek |
| To samo i rzeczywisty `pg_conftool … show include_dir` 1/pusty | Nowa inspekcja przechodzi niezależnie od starego polecenia |
| Wcześniej ukończony, aktywny profil projektu | Dokładny wdrożony profil i pełny preflight bez zmian |
| `conf.d` postgres `0700`, obcy UID, zapis grupy/innych | Odmowa polityki własności/uprawnień |
| Symlink conf.d/rodzica/pliku, hardlink, podmiana UID/metadata podczas inspekcji | Odmowa chronionego odczytu lub stabilności |
| Brak/zła wartość/duplikat include, nested include, multiline, zła składnia, nieczytelność | Odmowa konfiguracji |
| `listen_addresses='*'` lub obcy adres; obcy/mieszany katalog socketu | Odmowa profilu |
| Dodatkowy trust przed/po peer lub reject; inna metoda/opcja/mapa; błąd HBA lub niedostępny widok | Odmowa pełnej polityki HBA/ident |
| Błędna tożsamość sesji, port/data directory/config_file; brak klastra lub zatrzymana usługa | Odmowa tożsamości/readiness |
| Dowolna dyrektywa auto.conf, także pusty include; GUC z ALTER SYSTEM, pending_restart, dysk różny od aktywnego stanu lub durability off | Odmowa zgodności |
| Zmiana HBA/ident po starcie, także po poprawnym reload | Odmowa do niezależnie zatwierdzonego pełnego startu |
| Dodatkowe członkostwo kont usługowych w grupie uprzywilejowanej | Odmowa NSS/kont bez naprawy UID/GID/IPC |

Wszystkie te przypadki VM pozostają **NOT_EXECUTED**, dopóki operator nie
zachowa dowodów danego SHA. Rzeczywiste jednorazowe klastry Linux testują SQL
i parser, ale nie Ubuntu VM z systemd ani jej aktualnego stanu.

Osobny opt-in `tests/deployment/test_production_local_installer_postgresql.py`
uruchamia jednorazowy klaster z konta nie-root, z już przygotowanymi binariami
PostgreSQL 16. Nie instaluje pakietów ani nie modyfikuje systemowego klastra,
kont i usług. Uruchomienie poniżej dotyczy wyłącznie osobnego zatwierdzonego
klona/testowego Linuxa i przejrzanego środowiska narzędzi; nie ServiceTrial:

```sh
CRYPTOHUNTER_RUN_STAGE9_HOST_PG_INTEGRATION=1 \
CRYPTOHUNTER_TEST_PG_BINDIR=/usr/lib/postgresql/16/bin \
  python -m pytest -q tests/deployment/test_production_local_installer_postgresql.py
```

Bez opt-in testy są SKIP. Fixture jawnie podstawia własne ścieżki/port
i tożsamość konta testowego. Sprawdza oba profile i odmowy rzeczywistym SQL,
snapshoty oraz granicę reload/pełnego startu; nie dowodzi natywnej tożsamości
root/postgres, systemd ani zaliczenia właściwej VM. Liczby wyników dotyczą
wyłącznie konkretnego SHA i są raportowane osobno w PR.

## 6. Warunki przed S3 i raport

Przed S3 potrzebne są: zakończony przegląd PR #3264 bez merge automatycznego,
decyzja operatora o niezgodności własności, zatwierdzone wejścia i chroniony
bootstrap z tego samego SHA, niezależnie podpisany runtime oraz wymagane
approval istniejącej procedury, właściwe autorytety i dowody izolacji.
Do tego dochodzą wykonane odczyty HBA/listen/socket/guard i plan odtworzenia.
Brak dowolnego warunku pozostaje BLOCKED; nie tworzyć syntetycznego approval.

Raport oddziela wykonane PASS/FAIL od SKIP/NOT_EXECUTED/BLOCKED, podaje commit,
wersje pakietów, snapshot/boot-id, dokładne zakresy porównań i pozostałe ryzyka.
Nie przedstawia jednostkowych testów ani poprawnego SELECT jako kwalifikacji
produkcji. Stage 9 pozostaje **IN_PROGRESS / PRODUCTION_NO_GO**.
