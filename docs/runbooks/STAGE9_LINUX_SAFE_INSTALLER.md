# Stage 9: bezpieczna instalacja substratu FreshnessVerifier

Ta procedura dotyczy wyłącznie instalatora
`deployment/install-production-local.sh`. Nie wdraża pełnego CryptoHunter,
nie przyznaje kwalifikacji fizycznej i nie zmienia statusu Stage 9.
W tym PR wykonujemy wyłącznie testy syntetyczne w katalogach tymczasowych.
Nie wykonujemy instalacji, restartów, tworzenia kont ani klastrów na Dellu.

**Weryfikacja Ubuntu 24.04 po review PR #3235:** domyślne pakiety Noble
`python3-cryptography` 41.0.7 i `python3-psycopg` 3.1.17 nie spełniają
wymagań projektu. Python 3.12 spełnia warunek wersji języka, lecz dodatnia
instalacja tego profilu na standardowym Noble pozostaje BLOCKED. Nie
instalować bibliotek przez systemowy pip i nie obniżać wymagań. Projekt
odrębnego chronionego środowiska opisuje
[raport remediacji](../architecture/stage9_linux_installer_remediation.md).
Pełna instrukcja manualnej próby, z jawną bramką zatrzymania, znajduje się
w [procedurze jednorazowej VM Ubuntu 24.04](STAGE9_LINUX_UBUNTU2404_VM_TRIAL.md).

## Zakres i granica zaufania

Instalator publikuje kod verifiera, jego jednostkę systemd oraz trzy szablony
PostgreSQL. Nie tworzy kont, grup, klastra, bazy, schematu ani kluczy. Nie
wykonuje `daemon-reload`, `enable`, `start`, `restart` ani reboot. Te działania
wymagają osobnej, zatwierdzonej procedury operatora po przeglądzie stanu hosta.

Kod instalatora, jego helper `production_local_installer_host.py`, wrapper,
interpreter `/usr/bin/python3`, zależności i całe środowisko importów muszą być
niezależnie zweryfikowane **przed** uruchomieniem z uprawnieniami administratora.
Należy użyć odrębnie przygotowanej, root-owned kopii bez zapisu przez grupę lub
innych użytkowników i bez symlinków w jej ścieżce. Dotyczy to również
systemowych pakietów, `.pth`, import hooks i zastanego bytecode. Sprawdzenie
podpisu wykonywane przez niezaufany instalator nie ustanawia zaufania do niego.
Nie uruchamiać `sudo python` z edytowalnego checkoutu ani z venv developera.

Manifest używa osobnego zakresu `cryptohunter.freshness.substrate.v1`.
Nie jest certyfikatem `linux_release_identity.py`: tamten kontrakt nadal wymaga
pełnych ról runtime i verifier, rzeczywistych pakietów DEB baseline/update/fault
oraz niezależnego autorytetu. Nie zastępujemy go manifestem tego instalatora.

## Materiały do niezależnego zatwierdzenia

Przygotowanie materiałów wykonuje się bez uprawnień administratora. Manifest
UTF-8 JSON zawiera dokładnie `schema_version: 1`, powyższy `scope`,
`source_revision` jako pełne SHA Git oraz `files`. Każda pozycja `files` zawiera
`path`, `sha256`, `size` i całkowity `mode` (`420` dla 0644, `493` dla 0755).
Ścieżki odnoszą się do katalogu bundle; nie mogą uciekać poza niego. Inwentarz
musi obejmować wymagane pliki wskazane przez `REQUIRED_FILES` instalatora,
w tym rzeczywisty verifier i jego lokalne importy, launcher, jednostkę i
szablony PostgreSQL. Nie dodawać plików `.pyc`, symlinków ani hardlinków.

Autorytet zatwierdza pochodzenie źródeł, wszystkie bajty, tryby i pełność
lokalnego łańcucha importów oraz użycie nowych ustawień PostgreSQL na konkretnym
hoście. Przechowywane osobno materiały to manifest, surowy 64-bajtowy podpis
Ed25519 **dokładnych bajtów manifestu** i surowy 32-bajtowy klucz publiczny.
Nie przepisywać ani nie normalizować manifestu po podpisaniu. Klucz publiczny
musi pochodzić z niezależnej ceremonii i leżeć w chronionej ścieżce poza bundle;
nie wolno przyjąć klucza dostarczonego razem z nieznanym pakietem jako autorytetu.
Ten PR nie generuje ani nie zmienia kluczy produkcyjnych.

## Odczytowy preflight

Wszystkie argumenty są obowiązkowe; samo wywołanie wrappera nie instaluje niczego.
Poniższe polecenia są wzorcem dla operatora po niezależnym zatwierdzeniu kopii
instalatora i materiałów. Pełne SHA trzeba uzyskać z zatwierdzonego wydania.

```sh
/opt/approved-installer/deployment/install-production-local.sh \
  --preflight \
  --bundle /opt/approved-release/payload \
  --manifest /opt/approved-release/release.json \
  --signature /opt/approved-release/release.sig \
  --trusted-key /etc/cryptohunter-release-authority.pub \
  --expected-revision FULL_APPROVED_GIT_SHA
```

Diagnostyka nie tworzy nawet pliku blokady ani dziennika. Brak możliwości
odczytu warunku daje odmowę, nigdy domyślny sukces. Raport JSON zawiera stały
kod i bezpieczny opis; nie publikuje stderr poleceń, rekordów NSS, sekretów
ani zawartości konfiguracji. Kod wyjścia 0 oznacza wyłącznie spełnienie
warunków tego ograniczonego instalatora, nie gotowość uruchomionego produktu.

| Kod | Przyczyna odmowy |
| --- | --- |
| 10 | Kontrakt, format lub rewizja manifestu |
| 11 | Niezależny autorytet lub podpis |
| 12 | Inwentarz, hash, rozmiar lub tryb artefaktu |
| 20 | Pakiety, systemd, interpreter lub zależności |
| 21 | Brak gotowego PostgreSQL 16/main |
| 22 | Konta, grupy, NSS lub separacja UID/GID |
| 23 | Chroniona ścieżka, własność lub uprawnienia |
| 24 | Obce/częściowe wdrożenie, drift lub konfiguracja jednostki |
| 30 | Przerwanie lub błąd publikacji wymagający inspekcji |
| 31 | Brak wymaganych uprawnień instalacji |

Błędy składni CLI mają standardowy kod argparse 2. Wcześniejsza odmowa
bootstrapu oznacza niespełnienie granicy zaufania instalatora, zanim działa
właściwy raport JSON. Kilka niespełnionych warunków nie daje obietnicy ich
jednoczesnego raportowania: pierwsza odmowa kończy przebieg bez zmian.
Na standardowym Noble poprawne materiały i zainstalowane oficjalne biblioteki
prowadzą do odmowy 20 z powodu wersji zależności. Bez `cryptography` może
wcześniej wystąpić 10 przy sprawdzaniu podpisu. Nie wolno przedstawiać takiego
wyniku jako potwierdzenia, że wykonano już kontrolę klastra z kodem 21.

Preflight sprawdza pakiety, PID 1/systemd, PostgreSQL 16/main, przygotowane
konta i grupy, kolizje UID/GID, interpreter i zależności, chronione ścieżki,
podpis oraz zawartość bundle i stan zastanej instalacji. Nie importuje kodu
bundle. Do chronionych ścieżek docelowych dopuszcza wyłącznie root-owned
katalogi bez zapisu przez grupę/innych. Standardowe katalogi konfiguracyjne
PostgreSQL będące własnością `postgres` mogą więc spowodować odmowę. Instalator
nie naprawia ich automatycznie. Operator musi osobno ocenić model własności
i konfigurację pakietu; nie należy wykonywać rekurencyjnego `chown`/`chmod`.

## Potwierdzony stan Della i wymagane przygotowanie

Na podstawie dostarczonego opisu: Ubuntu 24.04.5, kernel 7.0.0-38,
Python 3.11.17 w venv, PostgreSQL 16.15 bez zarejestrowanych klastrów i bez
`/etc/postgresql/16/main`, brak kont/usług CryptoHunter, systemd jako PID 1.
Taki host ma otrzymać **odmowę**. Sam pakiet PostgreSQL nie wystarcza.
Instalator nie tworzy klastra. Venv 3.11 nie dowodzi dostępności zależności
dla `/usr/bin/python3`, którego używa jednostka systemd.

Przygotowanie hosta pozostaje poza wykonaniem tego PR. Osobna procedura
administratora musi najpierw ustalić, czy istnieją dane lub klaster w innej
lokalizacji, wykonać i sprawdzić kopie zapasowe oraz zatwierdzić właściwy
klaster 16/main i konfigurację peer. Trzeba przygotować dwa odrębne konta
systemowe `os_freshness_crypto_verifier`, `os_freshness_runtime`, ich odrębne
grupy podstawowe i grupę `freshness_verifier_ipc` z dokładnie tymi członkami.
Konta mają mieć home `/nonexistent`, shell `/usr/sbin/nologin`, odrębne grupy
podstawowe o nazwach kont i systemowe UID/GID z zakresu 1–999. Mają być
nieinteraktywne, bez uprawnień administracyjnych. UID/GID nie
mogą kolidować z innymi tożsamościami ani istniejącymi plikami wdrożenia.
Nie używać zbiorczej automatycznej naprawy NSS.

Utworzenie kont/grup, klastra, ról i schematu nie jest atomowo odwracalne.
Usunięcie konta nie cofa skutków jego użycia ani własności plików po numerze
UID. Usunięcie klastra niszczy dane. Dziennik instalatora nie uprawnia do
cofania tych osobnych ceremonii. Nie podajemy polecenia automatycznego
tworzenia klastra na Dellu.

## Kontrolowana instalacja i ponowienie

Dopiero po spełnieniu wszystkich warunków w odrębnej, zatwierdzonej próbie
operator może użyć identycznego polecenia z `--install` zamiast `--preflight`,
jako root i z niezależnie zweryfikowanej kopii instalatora. Instalator ponawia
kontrole, blokuje równoległy przebieg, tworzy staging i sprawdza skopiowane
bajty oraz tryby. Nowy kod publikuje jako całość; konfiguracja i jednostka są
odrębnymi operacjami. Nie zakłada atomowości całego hosta.

Nie nadpisuje obcych plików, danych, istniejącego legacy deploymentu ani
różniącej się konfiguracji. Identyczna kompletna instalacja może zostać
zweryfikowana ponownie. Zmiana wydania, migracja starego układu, częściowy
stan lub drift wymagają osobnego planu ochrony danych i recovery. Odmowa
aktualizacji jest zamierzona; `--install` nie jest poleceniem migracji.

Po instalacji nie uruchamiać automatycznie verifiera ani nie restartować
PostgreSQL. Szablon zmienia globalny `hba_file`/`ident_file`, lokalny transport
i trwałość; operator musi ocenić wpływ na inne bazy oraz zaplanować osobne
zastosowanie konfiguracji. Zachować poprzednią konfigurację i sprawdzony
backup. Instalacja bajtów nie potwierdza efektywnych ustawień działającego serwera.

## Awaria, przerwanie i odzyskiwanie

Zatrzymać kolejne próby po błędzie. Zachować
`/var/lib/cryptohunter-installer/journal.json`, ewentualny `receipt.json`
oraz `/usr/lib/.cryptohunter-stage-<SHA256_manifestu>`, pełny podpisany
inwentarz, wynik preflight, kopię
dotychczasowej konfiguracji i listę opublikowanych ścieżek. Nie czyścić
automatycznie katalogów i nie uruchamiać usług. Błąd, SIGTERM, SIGKILL lub
utrata zasilania mogą pozostawić pliki; żadna deklaracja transakcji Pythona
nie usuwa tego ryzyka. Kolejne fazy dziennika to `preparing`, `staged`,
`code-published`, `configuration-1` do `configuration-4` i `complete`.
Dziennik jest wskazówką do inspekcji: po nagłej awarii ostatnia operacja
może wyprzedzać ostatni utrwalony wpis. Receipt i pełna zgodność plików
są wymagane do uznania zakończonej identycznej instalacji.

Administrator porównuje każdy pozostały plik z podpisanym inwentarzem,
własnością, trybem i dziennikiem. Przed jakimkolwiek usunięciem lub ręcznym
dokończeniem musi osobno zatwierdzić dokładne ścieżki i plan danych. Nie
usuwać zastanych danych PostgreSQL, katalogów runtime, kont, kluczy ani
nieznanych plików. Przy podejrzeniu ingerencji zachować materiał do analizy.
Identyczny kompletny stan można ponownie sprawdzić; częściowy stan nie jest
automatycznie uznawany za zakończony.

## Testy VM i brak kwalifikacji fizycznej

Zwykłe testy używają `tmp_path`, syntetycznych podpisów i zastąpionych
granic hosta. Nie instalują usług ani kont. Osobną próbę uprzywilejowaną
operator może przygotować wyłącznie w jednorazowej Ubuntu 24.04 VM z
systemd PID 1, PostgreSQL 16 i snapshotem sprzed próby. Bez montowania
katalogów hosta lub dostępu do produkcyjnych kluczy.

W takiej VM należy zachować hash manifestu i obraz systemu, sprawdzić
pozytywny preflight oraz odmowy: brak klastra, kolizje NSS, złe prawa,
symlink/zmiana ścieżki i artefaktu, istniejące obce pliki. Następnie sprawdzić
instalację, identyczne ponowienie, przerwanie na poszczególnych checkpointach,
utratę procesu/zasilania, integralność i executable bits. Zapis dziennika
i stan plików po awarii mają być zachowane do recovery. Przywrócenie snapshotu
dotyczy wyłącznie jednorazowej VM. Wykonanie tej próby pozostaje poza PR.

Pełne uruchomienie wymaga odrębnej kompozycji AccountGenesis runtime,
produkcyjnych signerów/issuera i Source/CCA, kwalifikacji peer PostgreSQL,
funkcjonalnych odmów IPC, rzeczywistych DEB i niezależnych dowodów fizycznych.
Znany drift pinu `account_genesis_freshness_authority.py` pozostaje blokadą
kwalifikacji; ten PR go nie aktualizuje. Wynik syntetyczny nie zmienia
`LINUX_PRODUCTION_READY` ani `STAGE9_PRODUCTION_READY=NO`.
