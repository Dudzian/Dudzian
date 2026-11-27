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
  --artifacts "$PRIVATE_REVIEW_DIR/artifacts"
```

Pomiń trzy ostatnie opcje, gdy nie ma fizycznego materiału. Wynik `NOT_TESTED`
zachowuje stan bramki. `EVIDENCE_CONSISTENT_REVIEW_REQUIRED` wymaga przeglądu
pochodzenia i kompletności pomiarów. Każdy wynik zawiera
`authorization=NOT_GRANTED`. Kod wyjścia 2 kwalifikatora oznacza odmowę wejścia.
CLI Observera V5d1 ma osobny kontrakt kodu 2.

## Oddzielnie autoryzowana replika diagnostyczna

Przygotuj replikę według tego samego planu plików, aliasów i montowań oraz
przypiętych polityk. Zbierz własne identyfikatory przestrzeni i mountinfo
repliki. Utrwal AppArmor i seccomp załadowane przed exec. Sprawdź brak
odziedziczonych FD, obcych bibliotek, modułów Pythona oraz zapisu. W przypadku
qemu-img sprawdź również docelową etykietę AppArmor i zachowane ograniczenia
seccomp, NNP i capabilities potomka. Dozwolony sampler musi działać przed
pierwszym exec i obejmować wszystkie procesy oraz ich zakończenie. Utrata
zdarzeń, ucięty ślad lub wyjście poza zakres blokują kwalifikację. Zachowaj
istniejące zabezpieczenia hosta podczas przechwytywania.

| Przypadek | Oczekiwana czynność |
| --- | --- |
| `python-startup` | Przypięty interpreter `-I -S -B`, pusta czynność diagnostyczna. |
| `python-imports` | Przypięty finder V5d10 i `exercise_imports()` z [payloadu diagnostycznego](../../scripts/stage9_v5d11_diagnostic.py). |
| `python-libvirt` | Ten sam finder i `exercise_libvirt()` z zatwierdzonym, istniejącym testowym UUID i siecią. |
| `qemu-raw`, `qemu-qcow2` | Przypięty qemu-img wywołany przez deskryptor, z `info --output=json -f raw` albo `info --output=json -f qcow2` na osobnym deskryptorze niezmiennej kopii fixture. |

Dla diagnostyki glibc zbierz stderr `LD_DEBUG=files,bindings,versions` bez
`LD_DEBUG_OUTPUT`. Jeżeli potrzebny jest dodatkowy przypadek z `LD_BIND_NOW=1`,
zapisz go oddzielnie, ponieważ zmienia moment rozwiązywania symboli. Powiąż
normalną i diagnostyczną ścieżkę przez zapisany exec, piny obiektów i odczyty
map z jądra. `LD_DEBUG` zmienia środowisko względem launchera. Przejście
AppArmor `Px` qemu-img może usunąć `LD_*`. Jeżeli standardowy ślad glibc jest
dostępny wyłącznie w replice ze zmienionymi ustawieniami, osobna kwalifikacja
dokładnego potomka w niezmienionym widoku pozostaje `NOT_TESTED`. Do tego kroku
potrzebny jest zatwierdzony kolektor zewnętrzny działający przy zachowanych
politykach i dostarczający porównywalne zdarzenia. Aktualny bootstrap służy do
pomiaru widoku i nie ma trybu wykonania tego payloadu.

Zachowaj oryginalne mapy, status, AppArmor oraz zdarzenia `open`, `mmap`, exec
i zamknięcia procesu. Powiąż `st_dev` i `st_ino` odczytanego pliku z mapą,
zweryfikuj hash tych samych bajtów i zapisz `LC_ALL=C readelf --dyn-syms -W`
z tego obiektu. Biblioteka zwolniona przez `dlclose` wymaga powiązania z jej
wcześniejszym mapowaniem. Ślad i nazwa SONAME bez potwierdzenia rzeczywistej
ścieżki są niewystarczające. Zestawienia wersji porównuj z krawędziami
`DT_NEEDED` i przypiętymi providers. Pozytywne wiązanie słabego symbolu można
obserwować. Brak zdarzenia może oznaczać leniwe wiązanie, nieużyty kod albo
brak definicji. `weak_references_without_observed_binding` zachowuje tę
niepewność. Przypadki diagnostyczne nie rozszerzają listy bibliotek A.

## Granica protokołu libvirt

W tej samej replice udostępnij jeden root-owned socket `libvirt-sock-ro` albo
`virtqemud-sock-ro`. Zmierz `device`, `inode`, `uid`, `gid` i tryb po obu
stronach bind mount. Konfiguracja klienta ma dokładne bajty
`uri_default = "qemu:///system"` z końcowym LF. Zewnętrznie zarejestruj wybrane
ścieżki `connect(2)` i ustal `SO_PEERCRED` po obu stronach widoku. PID demona
spoza prywatnego PID namespace może przyjąć 0 wewnątrz. Na hoście przypnij PID,
start ticks, `/proc/PID/exe`, cmdline, pliki konfiguracji, jednostki i wszystkie
effective drop-ins oraz przyporządkuj inode listenera do aktywnej jednostki i
procesu. Przy socket activation uwzględnij pochodzenie deskryptora z PID 1.

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
oryginalny ślad źródłowy i sprawdź zgodność wszystkich pozostałych reguł
protokołu z implementacją wersji demona. Sam mount ro pozwala wysłać RPC i
odczyt `openReadOnly()` nie wystarcza do oceny drogi dla zwykłego klienta.

## Format prywatnych dowodów

Plik `native-evidence.json` ma profil `stage9-v5d11-profile-a-native-evidence-v1`
i klucze `profile`, `receipt_sha256`, `abi_request_sha256`,
`static_report_sha256`, `view`, `libvirt`, `loader`. Każda referencja do pliku
ma dokładnie `path` (względem `--artifacts`) i `sha256` (dokładne bajty).

- `view` zawiera referencję `mountinfo`, `namespaces` z `mnt`, `pid`, `net`,
  oraz `apparmor_sha256` i `seccomp_sha256`. Mountinfo ma obejmować dokładnie
  `/`, prywatny `/proc`, `/dev/null` i pojedynczy socket odczytowy.
- `libvirt` zawiera `socket`, `socket_identity`, `host_socket_identity`,
  `peer`, `host_peer`, `daemon`, `client_config_sha256`, `uri`, `connect_paths`,
  `read_results` i `fixture_denial`. `read_results` odwzorowuje każdy klucz
  zwracany przez payload na `PASS`. Struktury demona i dowodu odmowy stosują
  pola i warunki walidacji w [`_libvirt`](../../scripts/stage9_v5d11_qualify.py).
- `loader` zawiera wszystkie pięć przypadków z tabeli. Każdy ma `pid` (PID
  widziany wewnątrz), `argv0`, referencje do `trace`, `maps`, `status`,
  `label` oraz `symbols` mapujące rzeczywistą ścieżkę ELF do referencji wyjścia
  `readelf`. `objects` mapuje każdą zmierzoną ścieżkę ELF do słownika
  `sha256` i `identity=[major, minor, inode]`. `outcome` programu Python to
  `PASS`. `outcome` qemu-img zawiera `argv`, `fixture_sha256`, `exit` (0) i
  `json_format` (`raw` albo `qcow2`). Granica formatu jest testowana na
  syntetycznym pakiecie w [testach kwalifikatora](../../tests/deployment/test_stage9_v5d11_qualify.py).

Zapisz w prywatnym pakiecie również surowy dowód z systemu, dokładną wersję
kolektora, przepływ uprawnień, wyniki negatywne, i sposób uzyskania pinów.
Kwalifikator porównuje wybrane pola i pliki. Niezależne ustalenie ich
pochodzenia oraz pełnego pokrycia procesów jest osobnym wymaganiem.
