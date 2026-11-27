# Stage 9 V5d11 — kwalifikacja libvirt i rzeczywistego loadera

`NATIVE_LIBVIRT=NOT_TESTED` i `NATIVE_LOADER=NOT_TESTED`. Kwalifikator ocenia
spójność zapisanych dowodów offline. Kwalifikacja wymaga oddzielnej zgody na
uruchomienie diagnostyki profilu A i osobnej zgody na przyszły preflight.
Zachowaj fizyczne dane, ścieżki, identyfikatory i wszystkie ślady w prywatnym
katalogu operatora.

## Raport statyczny

Użyj zachowanego, dokładnego raportu ABI V5d10. Gdy raport nie jest dostępny,
odtwórz go z przypiętego receipt, request i expectations. Wykonaj polecenie z
zaufanej kopii repozytorium jako zwykły użytkownik. Adapter V5d10 korzysta z
`readelf` i czyta przypięte pliki A. Profil A zachowuje swoje bajty i tryby.

```sh
umask 077
python -m scripts.stage9_v5d11_qualify \
  --receipt "$A_RECEIPT" --receipt-sha256 "$A_RECEIPT_SHA256" \
  --abi-request "$A_ABI_REQUEST" --abi-expectations "$A_ABI_EXPECTATIONS" \
  --emit-static-report > "$PRIVATE_REVIEW_DIR/static-abi.json"
```

Recenzent porównuje wejścia z V5d10 i przypina dokładne bajty odtworzonego
raportu. Skrót dokumentu dowodowego również przychodzi z niezależnego kanału.
Sprawdzenie kompletnego materiału odbywa się jednym poleceniem.

```sh
python -m scripts.stage9_v5d11_qualify \
  --receipt "$A_RECEIPT" --receipt-sha256 "$A_RECEIPT_SHA256" \
  --abi-request "$A_ABI_REQUEST" \
  --static-report "$PRIVATE_REVIEW_DIR/static-abi.json" \
  --static-report-sha256 "$APPROVED_STATIC_REPORT_SHA256" \
  --evidence "$PRIVATE_REVIEW_DIR/native-evidence.json" \
  --evidence-sha256 "$APPROVED_NATIVE_EVIDENCE_SHA256" \
  --artifacts "$PRIVATE_REVIEW_DIR"
```

Pomiń trzy ostatnie opcje, gdy nie ma fizycznego materiału. Wynik `NOT_TESTED`
zachowuje stan bramki. `EVIDENCE_CONSISTENT_REVIEW_REQUIRED` wymaga przeglądu
pochodzenia i kompletności pomiarów. Każdy wynik zawiera
`authorization=NOT_GRANTED`. Kod wyjścia 2 kwalifikatora oznacza odmowę wejścia.
CLI Observera V5d1 ma osobny kontrakt kodu 2.

## Oddzielnie autoryzowana replika diagnostyczna

Tryb `diagnose` tworzy nową replikę, sprawdza jej pomiar i zwalnia bramkę
dopiero po podpisie recenzenta. Wszystkie pięć sond dzieli ten widok.
Kolektor potwierdza PID i przodka, tożsamość przestrzeni, plik wykonywalny,
AppArmor, seccomp, NNP oraz capabilities. Proces lub dowód poza ustaloną
granicą blokuje ocenę. Szybki potomek qemu-img może zakończyć się przed
uzyskaniem mapy. Taką próbę kwalifikator odrzuca. [Polecenie operatora](stage9_v5d11_observer_isolation.md)

| Przypadek | Oczekiwana czynność |
| --- | --- |
| `python-startup` | Przypięty interpreter `-I -S -B`, pusta czynność diagnostyczna. |
| `python-imports` | Izolowany interpreter, stały katalog przypiętych pakietów i `exercise_imports()` z [payloadu diagnostycznego](../../scripts/stage9_v5d11_diagnostic.py). |
| `python-libvirt` | Ten sam przypięty katalog i `exercise_libvirt()` z zatwierdzonym, istniejącym testowym UUID i siecią. |
| `qemu-raw`, `qemu-qcow2` | Przypięty qemu-img wywołany przez deskryptor, z `info --output=json -f raw` albo `info --output=json -f qcow2` na osobnym deskryptorze niezmiennej kopii fixture. |

Kolektor zapisuje stderr glibc `LD_DEBUG=files,bindings,versions`,
surowe i znormalizowane mapy, status, etykiety i tablice symboli. Osobny
profil diagnostyczny qemu-img używa przejścia `px`, ponieważ `Px` czyści
`LD_DEBUG`. Wariant diagnostyczny ma odrębne piny do przeglądu i zachowuje
oba składniki etykiety potomka. Wynik dotyczy tego wariantu i zmierzonych
przypadków. Widok oraz loader przyszłego preflight Observera mają nadal
status `NOT_TESTED`.

Powiąż `st_dev` i `st_ino` każdego odczytanego ELF z mapą procesu,
zweryfikuj hash tych samych bajtów i zapisz `LC_ALL=C readelf --dyn-syms -W`
z tego obiektu. Kolektor qemu-img zachowuje wcześniejsze migawki, które
wnoszą nową zaobserwowaną tożsamość. Każda migawka pochodzi od tego samego
procesu, po sprawdzeniu PID, czasu startu, przestrzeni nazw i tożsamości
pliku wykonywalnego. Biblioteka zwolniona przez `dlclose` wymaga zgodnej
tożsamości z wcześniejszej migawki. Gdy krótkiego mapowania nie uchwycono,
kwalifikator odmawia oceny pakietu. Porównuje też mapy znormalizowane z
oryginalnymi i odrzuca zmianę tożsamości między migawkami. Zestawienia wersji
porównuj z krawędziami
`DT_NEEDED` i przypiętymi providers. Pozytywne wiązanie słabego symbolu można
obserwować. Brak zdarzenia może oznaczać leniwe wiązanie, nieużyty kod albo
brak definicji. `weak_references_without_observed_binding` zachowuje tę
niepewność. Przypadki diagnostyczne nie rozszerzają listy bibliotek A.

## Granica protokołu libvirt

W tej samej replice udostępnij jeden root-owned socket `libvirt-sock-ro` albo
`virtqemud-sock-ro`. Zmierz `device`, `inode`, `uid`, `gid` i tryb po obu
stronach bind mount. Konfiguracja klienta ma dokładne bajty
`uri_default = "qemu:///system"` z końcowym LF. Kolektor ustala
`SO_PEERCRED` po obu stronach widoku. Dla `SOCK_DIAG` oddzielny wątek
kolektora używa przypiętego deskryptora przestrzeni sieciowej żywego klienta.
Linux tworzy oba końce ustanowionego połączenia UNIX w przestrzeni klienta.
Kolektor wymaga jednej pary wzajemnych peer, zgodnego inode socketu w VFS
oraz deskryptorów przyjętego połączenia i listenera w zmierzonym demonie.
PID demona spoza prywatnego PID namespace może przyjąć 0 wewnątrz. Na hoście przypnij PID,
start ticks, `/proc/PID/exe`, cmdline, pliki konfiguracji, jednostki i wszystkie
effective drop-ins oraz przyporządkuj inode listenera do aktywnej jednostki i
procesu. Przy socket activation uwzględnij pochodzenie deskryptora z PID 1.

Zachowanie z końcami połączenia w przestrzeni klienta występuje w źródłach
[Linux 6.8](https://github.com/torvalds/linux/blob/v6.8/net/unix/af_unix.c#L1465-L1501)
i [Ubuntu `linux-hwe-7.0` `7.0.0-43.43~24.04.1`](https://git.launchpad.net/~ubuntu-kernel/ubuntu/+source/linux/+git/noble/tree/net/unix/af_unix.c?id=60e65196d87a3b82950ceb158a08953b6468c785#n1625).
[Zrzut SOCK_DIAG w tym wydaniu Ubuntu](https://git.launchpad.net/~ubuntu-kernel/ubuntu/+source/linux/+git/noble/tree/net/unix/diag.c?id=60e65196d87a3b82950ceb158a08953b6468c785#n177)
pozostaje ograniczony do przestrzeni monitora. Podczas inwentaryzacji odczytaj
dokładną wersję źródłową pakietu jądra, `CONFIG_UNIX_DIAG` i stan pary jednostek.
Poniższe polecenia odczytują lokalne metadane.

```sh
release=$(uname -r)
cat /proc/version_signature
dpkg-query -W -f='${binary:Package}\t${Version}\t${source:Package}\t${source:Version}\n' \
  "linux-image-$release" "linux-modules-$release"
grep -E '^(CONFIG_UNIX_DIAG=|# CONFIG_UNIX_DIAG is not set)' "/boot/config-$release"
systemctl --system --no-pager show -p Id,ActiveState,MainPID,FragmentPath,DropInPaths \
  libvirtd.service libvirtd-ro.socket
```

Aktywny `libvirtd-ro.socket` z `MainPID=0` usługi pozostawia pomiar tożsamości
demona `BLOCKED`. Inwentaryzacja nie otwiera połączenia, które mogłoby aktywować
usługę. Ocena wymaga później działającego, zweryfikowanego demona i obsługi
AF_UNIX `SOCK_DIAG` w jądrze stanowiska.

Uruchamiaj wyłącznie odczyty na wcześniej przygotowanej domenie testowej.
Payload wymaga, aby aktywne XML faktycznie odwoływało się do testowej sieci.
Brak tego RPC po przejściu na modularny `virtqemud` blokuje kwalifikację i
wymaga osobnego przeglądu. Dodatkowych socketów nie należy dołączać do A.

Oddzielnie zachowaj ślad protokołu z dyspozycyjnej instancji demona o tym
samym kliencie, binariach i istotnej konfiguracji. Użyj połączenia na jego
sockecie odczytowym z flagami otwarcia `0`; za pośrednictwem zrecenzowanego
kolektora protokołu wykonaj wyłącznie żądanie odczytowe XML z
`VIR_DOMAIN_XML_SECURE=1`. Sprawdź rzeczywistą odpowiedź serwera
`VIR_ERR_OPERATION_DENIED` i przypnij zrecenzowany kod obsługi trybu RO dla
wersji demona. Odczyt ten odróżnia decyzję serwera od kontroli flagi RO po
stronie `libvirt-python` bez wykonywania operacji zmieniającej. Zachowaj
oryginalne ramki w prywatnym pliku `fixture-readonly.rpc` i sprawdź zgodność
pozostałych reguł protokołu z implementacją wersji demona. Ślad wiąże sesję,
tożsamość połączenia, endpoint i zweryfikowaną tożsamość fixture. Kwalifikator
odczytuje ramki ponownie i sprawdza pełną korelację żądań i odpowiedzi,
w tym status i kod odmowy serwera. Kolektor zapisuje plik z trybem `0600`.
Po znaczniku formatu i kanonicznych metadanych plik zawiera dokładne bajty
ośmiu ramek wraz z ich oryginalnymi prefiksami długości. Każde żądanie i
odpowiedź należą do tej samej sesji i połączenia. Sam mount ro pozwala wysłać RPC i
odczyt `openReadOnly()` nie wystarcza do oceny drogi dla zwykłego klienta.

## Format prywatnych dowodów

Plik `native-evidence.json` ma profil `stage9-v5d11-profile-a-native-evidence-v2`
i klucze `profile`, `receipt_sha256`, `abi_request_sha256`,
`static_report_sha256`, `session`, `view`, `libvirt`, `loader`. `session`
wiąże `spec_sha256`, `authorization_sha256` oraz `nonce`. Każda referencja do pliku
ma dokładnie `path` (względem `--artifacts`) i `sha256` (dokładne bajty).
Pakiet v1 bez wymaganych dowodów źródłowych otrzymuje odmowę kwalifikacji.

- `view` zawiera referencję `mountinfo`, `namespaces` z `mnt`, `pid`, `net`,
  oraz `apparmor_sha256`, `seccomp_sha256` i `purpose=diagnostic`. Mountinfo ma obejmować dokładnie
  `/`, prywatny `/proc`, `/dev/null` i pojedynczy socket odczytowy.
- `libvirt` zawiera `socket`, `socket_identity`, `host_socket_identity`,
  `peer`, `host_peer`, `daemon`, `client_config_sha256`, `uri`, `connect_paths`,
  `read_results` i `fixture_denial`. `read_results` odwzorowuje każdy klucz
  zwracany przez payload na `PASS`. `fixture_denial` zawiera metadane
  tożsamości oraz referencję do surowego śladu. Struktury demona i dowodu odmowy stosują
  pola i warunki walidacji w [`_libvirt`](../../scripts/stage9_v5d11_qualify.py).
- `loader` zawiera wszystkie pięć przypadków z tabeli. Każdy ma `pid` (PID
  widziany wewnątrz), `argv0`, referencje do `trace`, `status`, `label` i
  `maps_root` z dokładną ścieżką katalogu przygotowanego widoku. `maps`
  zawiera od 1 do 16 migawek `/proc/PID/maps`, każda z referencjami `raw`
  i `normalized`. `symbols` mapuje rzeczywistą ścieżkę ELF do referencji wyjścia
  `readelf`. `objects` mapuje każdą zmierzoną ścieżkę ELF do słownika
  `sha256` i `identity=[major, minor, inode]`. `outcome` programu Python to
  `PASS`. `outcome` qemu-img zawiera `argv`, `fixture_sha256`, `exit` (0) i
  `json_format` (`raw` albo `qcow2`). Granica formatu jest testowana na
  syntetycznym pakiecie w [testach kwalifikatora](../../tests/deployment/test_stage9_v5d11_qualify.py).

Zapisz w prywatnym pakiecie również surowy dowód z systemu, dokładną wersję
kolektora, przepływ uprawnień, wyniki negatywne, i sposób uzyskania pinów.
Kwalifikator porównuje wybrane pola i pliki. Niezależne ustalenie ich
pochodzenia oraz pełnego pokrycia procesów jest osobnym wymaganiem.
