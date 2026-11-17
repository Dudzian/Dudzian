# Stage 9 / Observer 3A — przygotowanie wydania V5d3

Baza prac: `17e03a9cb42a84302401415f96ac90e756926915` na `main`
repozytorium `Dudzian/Dudzian`, po scaleniu PR #3278. V5d3 przygotowuje
materiał do niezależnego przeglądu. Nie kwalifikuje instalacji ani hosta.
Wskazane przez operatora raporty `/home/dudzian/stage9-v5d2-*` i archiwum
`/home/dudzian/stage9-v5d1-review-6oBYU1sl` nie są dostępne w środowisku
przygotowania; opisane wyniki są oświadczeniem operatora, a nie ponowionym
pomiarem. Raport V5d3 nie kopiuje ich do własnych wyników testów.

```text
Stage 9 = IN_PROGRESS
Observer 3A = BLOCKED
Stage 3B = NOT AUTHORIZED
Production = NO-GO
HOST_PREFLIGHT_VERIFIED = NOT_CLAIMED
EXECUTION_AUTHORIZATION = NOT_AUTHORIZED
PYC_RECONSTRUCTION = UNVERIFIED
Exact CPython execution = NOT_TESTED
```

## Kontrakt i zakres dowodów

Źródłem wymagań są [plan V5d2](stage9_operator_v5d2_deployment_plan.md),
[kontrakt V5d1](stage9_observer_kvm_preflight_v5d1.md),
[kontrakt V5c3](stage9_observer_pyc_execution_v5c3.md) oraz
[macierz domknięcia](stage9_closure_matrix_v5d1.json).
Wymagania bezpieczeństwa sprawdzono bezpośrednio w
[`stage9_observer_collector_identity_v5d1.py`](../../deployment/stage9_observer_collector_identity_v5d1.py),
[`stage9_observer_kvm_preflight_v5d1.py`](../../deployment/stage9_observer_kvm_preflight_v5d1.py),
[`stage9_observer_kvm_domain_v5d1.py`](../../deployment/stage9_observer_kvm_domain_v5d1.py),
[`stage9_observer_kvm_image_v5d1.py`](../../deployment/stage9_observer_kvm_image_v5d1.py)
i walidatorach V5c3. Narzędzie oraz CLI `codebase-memory` były niedostępne;
nie wykonano sprawdzenia pokrycia indeksu. Odczyt źródeł i testy nie stanowią
niezależnej aprobaty ani dowodu zupełności dynamicznego środowiska.

Zachowany profil wymaga osobnego hosta Ubuntu z lokalnym KVM/libvirt i
niezależnie przyjętą polityką. Nie istnieje zatwierdzona w kontrakcie macierz
wersji Ubuntu, jądra, QEMU ani libvirt, którą można wywnioskować z konfiguracji
laptopa Dell Latitude 5490. Sam dostęp do KVM nie kwalifikuje tego laptopa.
Przygotowanie nie otwiera połączenia libvirt, nie odczytuje obrazów, nie
uruchamia kontrolera i nie wykonuje badanego interpretera ani bytecode.

## Architektura pakietu

Głównym wejściem operatora jest
[`stage9_v5d3_prepare_release.sh`](../../scripts/stage9_v5d3_prepare_release.sh).
Odrębny
[`stage9_v5d3_verify_bundle.sh`](../../scripts/stage9_v5d3_verify_bundle.sh)
sprawdza pakiet jako dane przy użyciu zaufanego Pythona hosta. Weryfikator
pochodzi z niezależnie przejrzanej kopii narzędzi, nie z badanego pakietu.
Żaden plik `.py`, `.pyc`, rozszerzenie natywne ani interpreter z pakietu
nie jest uruchamiany w celu jego weryfikacji.

Pakiet ma następującą strukturę:

```text
stage9-v5d3-<commit12>-<losowy-identyfikator>/bundle/
  sources/deployment/...                  źródła dokładnej rewizji
  sources/scripts/...
  sources/pyproject.toml
  review/stage9_operator_v5d3_release_preparation.md
  logs/A.log ... logs/I.log
  RELEASE.json                           rewizja, baza i stan repozytorium
  IMPORTS.json                           statyczne importy i braki dynamiczne
  DEPENDENCIES.json                       pełne deklaracje projektu i extra
  INTERPRETER.json                        interpreter przygotowujący
  NATIVE.json                            komponenty natywne i braki dowodów
  MISSING.json                           blokady i brakujące decyzje
  SUMMARY.json
  SUMMARY.tsv
  COMPLETE.json                          znacznik ukończenia przygotowania
  MANIFEST.json                          ścieżki, rozmiary, tryby, SHA-256
  SHA256SUMS
```

Katalog próby leży poza checkoutem; `bundle/` jest właściwym pakietem,
a `verification.json` obok niego zachowuje wynik odrębnego weryfikatora.
Nie zawiera skopiowanego CPython,
pakietów Python ani `.so`: nie dostarczono zatwierdzonego zestawu tych
bajtów. Powtórne wykonanie tworzy odrębny wynik i zachowuje wcześniejsze
raporty; przerwany wynik nie staje się ukończonym wydaniem. Nie wolno
instalować katalogu roboczego jako chronionego środowiska.

Jeżeli końcowy weryfikator odmawia albo procedura obsługuje przerwanie
po zamknięciu manifestu, wcześniejsze `COMPLETE.json`, `MANIFEST.json`,
`SHA256SUMS`, `SUMMARY.json` i `SUMMARY.tsv` są zachowywane w osobnym
`attempt/rejected-seal-*/`. Katalog `bundle/` otrzymuje nowe podsumowanie
z `package=FAIL`, bez `COMPLETE.json`. Odrzucony zestaw pozostaje materiałem
diagnostycznym, a nie ważnym zakończonym pakietem. Po nieobsługiwalnym
przerwaniu procesu trzeba ponownie wykonać odrębną weryfikację; sama
obecność starszego znacznika nigdy nie stanowi aprobaty.

Minimalna jawna lista kodu wymagana przez tożsamość V5d1 obejmuje sześć plików:

- `deployment/canonical_json.py`;
- `deployment/stage9_observer_collector_identity_v5d1.py`;
- `deployment/stage9_observer_kvm_domain_v5d1.py`;
- `deployment/stage9_observer_kvm_image_v5d1.py`;
- `deployment/stage9_observer_kvm_preflight_v5d1.py`;
- `scripts/stage9_observer_kvm_preflight_v5d1.py`.

To lista minimalna, nie pełne domknięcie. Należy zachować inicjalizatory
pakietów i wszystkie osiągalne moduły repozytorium, w tym powiązane kontrakty
V5c3. V5d1 sprawdza również rzeczywiście zaimportowane moduły `deployment`
i `scripts`. Analiza AST nie ustala dynamicznych importów, konfiguracji
`site`, `.pth`, providerów ani bibliotek ładowanych natywnie. `libvirt`
i `libvirtmod` są wybierane podczas działania; `fcntl` jest importowany
warunkowo na Linux. Osiągalny kod obsługuje również dynamiczny, opcjonalny
import `zstandard`; osobne extra `offline-verifier` deklaruje
`zstandard==0.25.0`. Nie dodaje się go automatycznie do instalacji
kontrolera: wybór i domknięcie tej opcjonalnej ścieżki pozostają
`UNVERIFIED` do niezależnego przeglądu. Brak dowodu kompletnego rzeczywistego wyboru oznacza
`UNVERIFIED`, nawet gdy wszystkie statyczne krawędzie mają źródła.

Lokalny manifest opisuje przygotowane bajty. Nie jest zewnętrznym wykazem
zatwierdzonych oczekiwań. Podmiana plików wraz z ponownym wygenerowaniem
manifestu może zachować jego wewnętrzną zgodność; recenzent musi porównać
pochodzenie i bajty z niezależnie pozyskanym wydaniem.

## Procedura przygotowania i weryfikacji

Uruchomienie wykonuje zwykły użytkownik Linux (`UID != 0`) z niezależnie
przejrzanej, czystej kopii wydania. Narzędziem jest stały zaufany
`/usr/bin/python3` w wersji `>=3.11`, uruchamiany z `-I -S -B` i
oczyszczonym środowiskiem. Nie wybiera się interpretera z pakietu ani
z badanego obrazu. Skrypt nie pobiera aprobaty z SHA, nie instaluje pakietów do
systemowego Pythona, nie dodaje `--system-site-packages` i nie zmienia
automatycznie `sys.path`. Wymagane narzędzia oraz odmowy są zapisywane
w raporcie. Brak narzędzia pozostaje błędem środowiska; brak deklarowanego
źródła albo uszkodzony manifest pozostaje błędem pakietu.

Z katalogu przejrzanej finalnej rewizji operator uruchamia jedną komendę:

```bash
bash scripts/stage9_v5d3_prepare_release.sh \
  --repository /home/dudzian/CryptoHunter --output-parent /home/dudzian
```

`--output-parent` wskazuje istniejący zwykły katalog poza repozytorium;
skrypt tworzy w nim nowy `stage9-v5d3-<commit12>-<losowy-identyfikator>`.
Nie należy podawać chronionej ścieżki systemowej ani katalogu poprzedniego
wydania jako miejsca do nadpisania.

Po przerwaniu można zachować powiązanie z poprzednią próbą:

```bash
bash scripts/stage9_v5d3_prepare_release.sh \
  --repository /home/dudzian/CryptoHunter --output-parent /home/dudzian \
  --resume /home/dudzian/stage9-v5d3-<poprzedni-identyfikator>/bundle
```

`--resume` wskazuje poprzedni katalog `bundle`, nie katalog całej próby,
wyłącznie do identyfikacji i kontroli; ukończony wynik jest weryfikowany.
Powstaje nowy katalog, poprzedni nie
jest naprawiany ani nadpisywany. Skrypt ponawia odczytowe kontrole
przygotowania; nie uruchamia pełnej regresji Observera.

Odrębna komenda weryfikacyjna, z przejrzanej kopii narzędzi:

```bash
bash scripts/stage9_v5d3_verify_bundle.sh \
  /home/dudzian/stage9-v5d3-<identyfikator-z-raportu>/bundle
```

Symbole w nawiasach kątowych zastępuje się rzeczywistą ścieżką wyniku.
Kod 0 oznacza lokalną spójność przygotowanego pakietu, mimo jawnych blokad
zależności i wdrożenia. Kod 2 oznacza odmowę kontroli pakietu.
Nieprawidłowe argumenty skryptu przygotowującego dają kod 64, a
nieprawidłowe argumenty weryfikatora są odrzucane przez `argparse`
z kodem 2. Trzeba zachować dokładny exit status,
stdout i stderr; pomyślne przygotowanie nie jest zgodą na instalację.

Przygotowanie obejmuje etapy A–I: identyfikację i czystość rewizji,
zebranie źródeł, analizę zależności, manifesty, kontrole bezpieczeństwa,
plan instalacji, zestaw brakujących danych polityki, specyfikację VM i
`SUMMARY.tsv`/`SUMMARY.json`. Niezależne kontrole są kontynuowane po
częściowej odmowie, o ile można bezpiecznie zachować wyniki. Awaria lub
przerwanie nie uprawniają do nadpisania poprzednich raportów.

Weryfikator kontroluje zgodność struktury i wykazu plików, wymagane źródła,
deklaracje zależności oraz SHA-256. Odrzuca niebezpieczne ścieżki,
dowiązania, pliki specjalne, duplikaty, nieoczekiwane pliki i niespójny
manifest. Potencjalne sekrety są powodem odmowy; nie wolno przenosić ich
do raportu jako treści. Kontrole nazw i wybranych wzorców nie dowodzą
nieobecności wszystkich sekretów: kompletność skanu pozostaje
`UNVERIFIED` i wymaga odrębnego narzędzia/przeglądu. Weryfikacja nie jest
wykonaniem testów fizycznych.
Brak konkurujących procesów zapisujących do pakietu pozostaje warunkiem
całej procedury, w tym odczytu, przekazania i późniejszej instalacji.
Wynik weryfikatora opisuje zaobserwowany stan; taki snapshot nie jest
lifecycle fence i nie zapobiega późniejszym zmianom. Zaufanie do hosta
i narzędzi odczytu musi pochodzić spoza pakietu.

Statusy raportu mają następujące znaczenie:

| Status | Znaczenie |
| --- | --- |
| `PASS` | Określona kontrola przygotowania lub zgodności zakończyła się powodzeniem w zapisanym zakresie. |
| `FAIL` | Wykryto naruszenie sprawdzanego wymagania lub uszkodzony materiał. |
| `BLOCKED` | Brakuje obowiązkowej decyzji lub materiału pozwalającego przejść dalej. |
| `UNVERIFIED` | Dostępne obserwacje nie potwierdzają wymaganej własności lub niezależnego pochodzenia. |
| `NOT_TESTED` | Pomiaru lub próby nie wykonano. |

Gotowość pakietu, środowiska instalacyjnego, hosta, polityki, VM,
kwalifikacji i autoryzacja wykonania są odrębnymi pozycjami. `PASS`
integralności nie przyznaje `HOST_PREFLIGHT_VERIFIED` ani zgody na
instalację lub eksperyment.

## Zależności i interpreter

[`pyproject.toml`](../../pyproject.toml) wymaga hostowego Pythona `>=3.11`,
pełnego `project.dependencies` i extra `observer-kvm-preflight`.
W szczególności obowiązują `cryptography>=50.0.1,<51` i
`defusedxml==0.7.1`. Nie należy zastępować tej deklaracji wymyślonym
minimalnym środowiskiem. `types-defusedxml==0.7.0.20240218` służy
testom i kontroli typów, nie zastępuje parsera.

`libvirt-python` i `libvirtmod` pochodzą z osobno przejrzanego systemowego
bindingu. Obecność `python3-libvirt` w bazie pakietów lub `virsh` w PATH
nie dowodzi, że binding jest osiągalny z chronionego interpretera. Nie
wolno obejść niezgodnego ABI przez wybór innego interpretera, automatyczną
zmianę ścieżki importów lub włączenie systemowych site-packages.
Przyszłe środowisko wymaga niezależnie zatwierdzonego sposobu udostępnienia
zgodnych bajtów bindingu.

Recenzent musi otrzymać dokładny resolver/lock i hashe wszystkich pakietów
bezpośrednich oraz przechodnich, artefakty pobrania, platformę, ABI,
architekturę i wersję interpretera, pełny wykaz plików środowiska, `.pth`,
`sitecustomize`, `usercustomize`, `pyvenv.cfg`, launcherów i istniejącego
bytecode. Wykaz natywny obejmuje interpreter, loader, rozszerzenia,
libvirt, QEMU oraz zależne biblioteki i konfigurację providerów.
Statyczny wykaz bibliotek nie dowodzi kompletnego rzeczywistego ładowania.
Brak zatwierdzonego zestawu pakietów jest jawną blokadą; skrypt nie
rozwiązuje jej przez samodzielne pobranie i uznanie najnowszych pakietów.

Interpreter przygotowujący i przyszły hostowy verifier są oddzielone od
badanego Ubuntu CPython. Danych o interpreterze przygotowującym nie wolno
przedstawiać jako manifestu zainstalowanego `/opt/.../venv` ani jako dowodu
wykonania dokładnego interpretera V5c3.

## Przekazanie do niezależnego przeglądu

1. Zachowaj cały zakończony katalog, oryginalne logi i exit status oraz
   dokładną komendę, rewizję i tożsamość narzędzi. Przerwany wynik przekaż
   oddzielnie z informacją o nieukończeniu.
2. Recenzent pozyskuje źródła i oczekiwania inną zatwierdzoną drogą,
   porównuje commit, źródła, deklaracje, kompletność, manifesty i wyniki.
   Uruchamia własną przejrzaną kopię weryfikatora ponownie na odebranym
   pakiecie, przy wykluczonych konkurujących zapisach. Nie wykonuje kodu
   pakietu, aby udowodnić jego zaufanie. Wcześniejszy raport z maszyny
   przygotowującej nie zastępuje tej kontroli po przekazaniu.
3. Osobno recenzuje launcher, Python/native closure, konfigurację startową,
   plan przejścia własności, zewnętrzne oczekiwania i pochodzenie obrazów.
4. Zachowuje tożsamość recenzenta, zakres, rewizję kontraktu, decyzję,
   niezależne źródło oczekiwań i dokładne hashe przyjętych bajtów.
   Akceptacja materiału do przeglądu nie jest autoryzacją eksperymentu.

## Projekt późniejszej chronionej instalacji

Jest to plan do oddzielnego przeglądu administratora. V5d3 nie tworzy
`/opt`, `/etc`, kont, venv, polityki, flag immutable ani usług i nie
zawiera wykonania uprzywilejowanego instalatora.

| Miejsce | Docelowa zawartość i kontrola |
| --- | --- |
| `/opt/cryptohunter/observer-v5d1/deployment/` | Pełne zatwierdzone źródła importowane przez kontroler, z zachowaną strukturą. |
| `/opt/cryptohunter/observer-v5d1/scripts/stage9_observer_kvm_preflight_v5d1.py` | Dokładny zatwierdzony CLI. |
| `/opt/cryptohunter/observer-v5d1/venv/` | Osobno odtworzone i przejrzane środowisko Python/native, runtime i startup closure. |
| `/etc/cryptohunter/observer-v5d1/` | Tylko niezależnie przyjęte publiczne anchors, podpisana polityka i wymagania. |

Kolejność późniejszej procedury:

1. Administrator najpierw otrzymuje niezależną decyzję dotyczącą dokładnych
   źródeł, zależności, instalatora i metody przekazania. Nie wykonuje kodu
   z zapisywalnego checkoutu jako root, aby ustanowić jego zaufanie.
   Ponawia weryfikację odebranego pakietu przed użyciem, porównuje przyjęte
   niezależne oczekiwania i wyklucza konkurujące procesy zapisujące
   przez cały transfer bajtów. Sam wcześniejszy snapshot nie zapewnia
   takiej ochrony.
2. Sprawdza każdy komponent docelowej ścieżki od `/`: typ katalogu,
   UID 0, brak `mode & 022` i dowiązań. Dla plików obowiązują typ regularny,
   UID 0, jeden hardlink i brak group/other write. Kontrakt nie narzuca
   jednej grupy ani dokładnie `0600`/`0700`; wybrana grupa i prawa odczytu
   muszą zapewnić nieuprzywilejowanemu operatorowi niezbędny odczyt/traversal
   bez przyznania zapisu. Wybrana grupa wymaga udokumentowania.
3. Przygotowuje nowy obszar instalacji pod chronionym rodzicem i przenosi
   wyłącznie zatwierdzone bajty z kontrolą pełnego inwentarza. Nie scala
   ich z zastanym, nieznanym drzewem. Istniejąca instalacja lub obiekt pod
   ścieżką oznacza przerwanie i odrębny plan migracji.
4. Odtwarza środowisko wyłącznie z przyjętych artefaktów/wersji/hashy,
   zapisuje zależności natywne i startowe. Zwykły venv może zawierać
   symlink `bin/python`; symlink w mierzonej ścieżce nie spełnia V5d1.
   Trzeba niezależnie przyjąć launcher i rzeczywisty cel `/proc/self/exe`,
   bez osłabienia walidatora.
5. Przed publikacją administrator weryfikuje prawa każdego pliku i
   przodka, brak dowiązań/obcych plików, finalne hashe wszystkich bajtów,
   zgodność ABI i zatwierdzone rozwiązywanie importów. Nie uruchamia
   nieprzejrzanego interpretera, żeby go uwierzytelnić. Dołącza oryginalne
   protokoły pochodzenia i przejścia własności.
6. Zewnętrzny opiekun polityki przygotowuje ją dla finalnych rzeczywistych
   ścieżek i środowiska, a nie dla stagingu. Dokumentuje niezależne
   oczekiwania i podpisanie. W instalacji pozostają tylko publiczne dane.
7. Każda niezgodność przerywa publikację. Zachowuje się staging i raporty
   do analizy bez nadpisania istniejącej instalacji/polityki; nie dokonuje
   się automatycznej naprawy praw, zmiany zaufania ani rollbacku cudzych
   plików. Sposób późniejszego usunięcia nieopublikowanego stagingu wymaga
   zakresu zatwierdzonego administratora.
8. Niezależny świadek przegląda załadowany kod, interpreter, zależności,
   bootstrap i środowisko. Dopiero osobna decyzja może dopuścić późniejszą
   diagnostykę odczytową opisaną w V5d1. Jej standardowy exit 2 i odmowy
   pozostają niezmienione.

Chronione pliki i hashe nie rozwiązują kompromitacji root. V5d1 zawsze
zachowuje `COLLECTOR_IDENTITY_UNVERIFIED`,
`INDEPENDENT_LAUNCH_ATTESTATION_MISSING` i
`LOADED_EXECUTION_CLOSURE_UNVERIFIED`; standardowy launch `-I -B`
pozostawia również `SYSTEM_SITE_BOOTSTRAP_UNVERIFIED`. Nie istnieje
flaga aprobaty ani nieudokumentowany fallback `no-site`, który zamyka te
braki. V5d3 nie kopiuje historycznego polecenia zmieniającego `sys.path`
do automatycznej procedury przygotowania.

## Dane do niezależnego przygotowania polityki

Poniższe pozycje są dokumentacją pól, nie plikami polityki. Brakujących
wartości nie zastępuje się UUID przykładowej VM, zerowymi hashami,
syntetycznymi podpisami ani kluczem testowym. Wszystkie trzy pliki mają
stałe ścieżki pod `/etc/cryptohunter/observer-v5d1`. Nieznane i powtórzone
klucze są odrzucane. Hex używa małych liter. Limity rozmiaru wymagają
sprawdzenia dokładnych finalnych bajtów, nie tylko obiektu po parsowaniu.

| Obiekt | Dokładne pola i warunki |
| --- | --- |
| `operator-anchors.json` | `profile="stage9-observer-controller-anchors-v5d1"`, `authority_id`, `public_key_hex` (64 znaki hex, 32 bajty), `approved_revision`, `approved_policy_sha256` (64 hex). `authority_id` odpowiada `[A-Za-z0-9][A-Za-z0-9._:-]{0,127}`; rewizja jest int 1–9007199254740991, nie bool. |
| `approved-policy.json` | Dokładnie `policy`, `signature_hex` (128 hex, 64 bajty Ed25519). |
| `policy` | Dokładnie `profile="stage9-observer-controller-policy-v5d1"`, `authority_id`, `revision`, `artifacts`, `environment_sha256`. Tożsamość i rewizja muszą zgadzać się z anchors. |
| `artifacts` | 6–1024 unikalnych rekordów dokładnie `{path,role,sha256}`. Kanoniczne absolutne ścieżki Linux; pełne SHA-256. Obecne wszystkie role: `controller-code`, `interpreter`, `libvirt-code`, `libvirt-extension`, `qemu-tool`, `domain-requirements`. Jedyny `domain-requirements` wskazuje stałe `/etc/.../domain-requirements.json`. |
| `domain-requirements.json` | Dokładnie `profile="stage9-observer-kvm-requirements-v5d1"`, `domain`, `disk_chains`, `runtime_inventory`; maksymalnie 262144 bajty. |
| `domain` | Dokładnie `domain_uuid`, `memory_bytes`, `vcpu_count`, `cpu_period_us`, `cpu_quota_us`. UUID kanoniczny małymi literami; dodatnie int, nie bool. Maksima: 1 TiB, 64 vCPU, okres 1000000 μs, quota 64000000 μs; pamięć podzielna przez 1024, quota ≤ okres × vCPU. |
| `disk_chains` | 1–8 tablic od top do base; każda 1–16 rekordów dokładnie `{path,sha256,format}`. Ścieżki kanoniczne absolutne i globalnie unikalne, SHA-256 64 hex, `format` tylko `raw`/`qcow2`. |
| `runtime_inventory` | Dokładnie `files`, `needed_resolutions`, `loader_configuration_paths`, `import_paths`, `provider_configuration_paths`. |
| `files` | 5–512 rekordów dokładnie `{path,sha256,build_id,role,aliases}`. Role: `interpreter`, `library`, `loader`, `stdlib`, `tool`, `configuration`. Ścieżki i aliasy nie kolidują. `sha256=null` dopuszczalne wyłącznie jako jawna oczekiwana nieobecność `configuration`. `build_id` może być null albo 8–128 hex parzystej długości. Aliasów maksymalnie 32, wyłącznie dla `loader`. |
| `needed_resolutions` | Maksymalnie 4096 trójek `[consumer_path,needed_name,resolved_path]`, obie ścieżki obecne w `files`, bez powtórzeń pary consumer/name. Nazwa jest ograniczonym ASCII bez `/` i `:`. |
| Ścieżki konfiguracji/importów | `loader_configuration_paths` i `provider_configuration_paths`: do 256 każda i łącznie do 256 unikalnych wskazanych plików; `import_paths`: 1–64 unikalnych kanonicznych ścieżek. Obowiązują dalsze semantyczne wymagania V5c3 dotyczące loadera, providerów, bibliotek i wspólnej proweniencji snapshotu. |

Każdy artefakt polityki ma limit 256 MiB. Anchors i envelope również
podlegają limitowi 256 KiB. Polityka obejmuje faktyczny `/proc/self/exe`,
pełne importowane źródła, rzeczywiście wybrane `libvirt` i `libvirtmod`,
`/usr/bin/qemu-img` oraz hash surowych bajtów wymagań. Zmiana formatowania
wymagań zmienia ich pin. `environment_sha256` wiąże implementację i wersję
Python, architekturę, UID/GID/grupy, `sys.path`, flagi izolacji/site oraz
wybrane zmienne startowe. Wartości bazowe trzeba niezależnie przejrzeć;
sam fingerprint podany przez kontroler nie jest aprobatą.

Opiekun anchors wybiera `SHA256(canonical_json_bytes(policy))` i zgodną
rewizję. Podpis Ed25519 obejmuje dokładnie kanoniczny obiekt:

```text
{"domain":"stage9-observer-controller-policy-signature-v5d1","policy":policy}
```

Stosuje przejrzany `deployment.canonical_json.canonical_json_bytes`.
Podpisanie samego hasha lub innego serializera nie spełnia kontraktu.
Klucz prywatny pozostaje u niezależnej władzy poza repozytorium i badaną
VM; przygotowanie nie tworzy go ani nie odczytuje. Brakują: authority,
public key, decyzja i rewizja, pełne zatwierdzone piny, środowisko,
pochodzenie oczekiwań oraz podpis. V5d3 nie przyznaje im aprobaty.
Rewizja i lokalny czas nie ustanawiają authenticated freshness ani
durable anti-rollback.

## Specyfikacja osobnej jednorazowej VM

VM musi być osobną domeną persistent KVM, zatrzymaną (`SHUTOFF=5`), bez
managed-save i snapshotów libvirt. Jej kanoniczny UUID ma zostać dostarczony
i zaakceptowany przez niezależnego opiekuna rejestru; **wartość nie została
podana**. Żadna z sześciu istniejących VM CryptoHunter nie jest kandydatem.
Nie tworzy się, nie definiuje i nie uruchamia VM w V5d3.

Poniższy XML jest specyfikacją tekstową z nierozwiązanymi symbolami.
Nie jest gotowym plikiem do `virsh define`. Parametry pochodzą z
niezależnie przyjętego `domain-requirements.json`. Wartości domyślne
kontraktu to 536870912 bajtów RAM, 1 vCPU i period/quota 100000/100000 μs;
nie oznaczają zatwierdzenia tych wartości dla eksperymentu.

```xml
<domain type="kvm">
  <name>${APPROVED_DISPOSABLE_NAME}</name>
  <uuid>${APPROVED_CANONICAL_UUID}</uuid>
  <memory unit="bytes">${APPROVED_MEMORY_BYTES}</memory>
  <currentMemory unit="bytes">${APPROVED_MEMORY_BYTES}</currentMemory>
  <vcpu placement="static">${APPROVED_VCPU_COUNT}</vcpu>
  <cputune>
    <period>${APPROVED_CPU_PERIOD_US}</period>
    <quota>${APPROVED_CPU_QUOTA_US}</quota>
  </cputune>
  <os>
    <type arch="x86_64" machine="pc-q35-8.2">hvm</type>
    <boot dev="hd"/>
  </os>
  <cpu mode="host-model" check="full">
    <topology sockets="1" dies="1" clusters="1"
              cores="${APPROVED_VCPU_COUNT}" threads="1"/>
  </cpu>
  <clock offset="utc"/>
  <on_poweroff>destroy</on_poweroff>
  <on_reboot>destroy</on_reboot>
  <on_crash>destroy</on_crash>
  <devices>
    <emulator>/usr/bin/qemu-system-x86_64</emulator>
    <disk type="file" device="disk" snapshot="no">
      <driver name="qemu" type="${APPROVED_FORMAT}"
              cache="none" io="native" discard="ignore" detect_zeroes="off"/>
      <source file="${APPROVED_ABSOLUTE_TOP_IMAGE_PATH}"/>
      <target dev="vda" bus="virtio"/>
      <readonly/>
    </disk>
    <controller type="usb" model="none"/>
    <memballoon model="none"/>
  </devices>
</domain>
```

Dla wielu dysków powtarza się wyłącznie zatwierdzone rekordy w kolejności
`vda`, `vdb`, …; typy i ścieżki muszą odpowiadać wymaganiom. Dla obrazów
z backing chain należy dodatkowo opisać pełny zgodny z pinami łańcuch
`backingStore` z formatami i źródłami. Nie wolno zgadywać brakującej bazy.
Libvirt może dopisać nieobsługiwane elementy XML; odmowa oznacza powrót do
niezależnego przeglądu provisioningu, nie automatyczną redefinicję domeny.
Profil odrzuca nieznane węzły/atrybuty, duplikaty, nieobsługiwane adresy,
seclabel/resource semantics, NVRAM i rozszerzenia QEMU.

Nie są dozwolone interfejsy sieciowe, passthrough, urządzenia hosta,
filesystem shares, kanały, sockety, konsole/serial/graphics, RNG, TPM,
vsock ani shared memory. Żadna niezatwierdzona droga sieciowa nie może
zostać dodana jako kanał pomiaru. Dostęp diagnostyczny pozostaje wyłącznie
odczytowym lokalnym `qemu:///system`, bez alternatywnego URI i eskalacji.

Każdy obraz/element backing chain ma być regularnym plikiem root,
z jednym hardlinkiem, chronionymi przodkami, bez symlinków, z faktycznie
ustawionym Linux `FS_IMMUTABLE_FL`. Obsługiwane są raw i ograniczone
QCOW2 v2/v3: bez szyfrowania, internal snapshots, external data files,
nieznanych feature bits i niestandardowego nagłówka v3. Limit mierzonego
pliku to 128 GiB; łańcuch ma maksymalnie 16 pozycji. Pełny pin pochodzi
z niezależnie przyjętego materiału źródłowego. Zatrzymana VM, read-only
deskryptor lub dwa identyczne stat nie zastępują immutable view,
wykluczenia uprzywilejowanych writerów i lifecycle fence. V5d3 nie zmienia
obrazów ani flag, nie konwertuje, nie montuje i nie naprawia dysków.

Konfiguracja XML potwierdza tylko żądane ustawienia. Potrzebne są osobne
dowody faktycznego QEMU executable/argv/modules, socketów i urządzeń,
network namespace/routes/filtering, UID/GID, namespaces/cgroups, LSM,
seccomp, capabilities i device access. V5c3 wymaga zewnętrznego egzekwowania
CPU/RAM/wall time/process count przez cały eksperyment i zbieranie wyników.
Domyślne limity V5c3 to 1 CPU, 512 MiB, 30 sekund i 64 procesy; konkretne
wartości wymagają decyzji. Zakres walidatora wykonania V5c3 jest węższy
od dopuszczalnych liczb w wymaganiach XML: 1–2 CPU, 64 MiB–1 GiB,
1–60 sekund i 1–128 procesów. Późniejszy plan musi być zgodny z oboma
kontraktami. Hostowe `pids.max` nie dowodzi limitu procesów
wewnątrz guest. Tych braków nie zamyka poprawny XML zatrzymanej VM.

Niezależny kanał pomiarowy musi przechwycić komplet rzeczywistych bajtów
runtime z tego samego immutable view, pełne dostępy do plików i mapowania
ELF z wykrywaniem utraty zdarzeń, dokładne argumenty/środowisko/status,
ograniczone stdout/stderr i niezależnie wyekstrahowane bajty PYC. Same
`DT_NEEDED` i dane wydrukowane przez guest nie są takim kanałem.
Pełne runtime tree, loader/providers/imports i snapshot/package provenance
pozostają obowiązkowe.

Mechanizm zatrzymania ma działać poza guest, zakończyć całą jednorazową VM
przy timeout, przerwaniu lub błędzie pomiaru, potwierdzić zatrzymanie i
cleanup oraz usunąć pozytywne wyniki przy niepotwierdzonym stop. Odczytowy
kontroler V5d1 nie ma takiej operacji. Trzeba osobno przejrzeć i zatwierdzić
kontroler lifecycle oraz dowód skuteczności; tekst `destroy` w XML nie
jest dowodem działania awaryjnego stop. Nie wolno nadawać V5d1 uprawnień
mutujących ani wykonywać próby na istniejących sześciu domenach.

## Blokady i następna czynność operatora

| Obszar | Pozostający brak |
| --- | --- |
| Pochodzenie wydania | Niezależna akceptacja finalnej rewizji, źródeł, narzędzi i manifestu. |
| Środowisko instalacyjne | Zatwierdzone kompletne pakiety przechodnie/native/startup, interpreter/ABI i launcher. |
| Chroniona instalacja | Weryfikowane finalne bajty, właściciele/prawa i odrębnie przejrzany administrator/instalator; według operatora instalacja nie istnieje. |
| Polityka | Authority, public key, oczekiwania, revision/digests, niezależna decyzja i podpis; według operatora katalog nie istnieje. |
| Tożsamość uruchomienia | Niezależny launch witness i loaded execution closure; obecny V5d1 nie ma ścieżki ich przyjęcia. |
| Host i VM | Niezależna kwalifikacja hosta, osobny UUID/domain/images i skuteczna izolacja. |
| Eksperyment | Pełny runtime/trace/channel, whole-VM stop, zatwierdzony request/output i osobna autoryzacja. |
| Przyjęcie dowodów | Zatwierdzona ścieżka real-observation admission, registry, authority, freshness i durable anti-rollback. |

Najbliższa czynność operatora to uruchomienie przygotowania z czystej,
przejrzanej finalnej rewizji, zachowanie całego wyniku i przekazanie go wraz
z osobną kopią weryfikatora niezależnemu recenzentowi. Następnie recenzent
wskazuje brakujące niezależne źródła pakietów i decyzje instalacyjne.
Nie należy jeszcze instalować do `/opt`/`/etc`, podpisywać polityki,
uruchamiać VM ani wykonywać dokładnego CPython. Draft PR kończy tę fazę;
merge i fizyczne wdrożenie pozostają poza jej zakresem.
