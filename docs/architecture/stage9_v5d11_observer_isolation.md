# Stage 9 V5d11 — pomiar izolacji Observera

V5d11 przygotowuje prywatny widok profilu A dla launchera V5d10 i wystawia
kandydaturę w jego istniejącym formacie. Bootstrap zatrzymuje się po pomiarze.
Bieżąca rewizja odrzuca każdą próbę przejścia do preflightu. Wersja V5d1
zachowuje kontrakt wyjścia `2`, również w przyszłym, odrębnie dopuszczonym
preflighcie, w którym rzeczywiście przejdą odczyty lokalne.

| Granica | Bieżący stan |
| --- | --- |
| Zgodność archiwów i danych materializacji V5d10 | `PASS`, sprawdzona statycznie |
| Prywatny widok, AppArmor, seccomp, libvirt i loader na laptopie | `NOT_TESTED` |
| Niezależna ocena rzeczywistej sesji | `NOT_AUTHORIZED` |
| Diagnostyka V5d1 przez launcher V5d10 | `BLOCKED` |

Źródłem wymagań są [launcher V5d10](../../scripts/stage9_v5d10_launch.py),
[kontrakt profilu A](../../scripts/stage9_v5d10_contract.py) i
[kwalifikacja V5d10](stage9_v5d10_observer_preparation.md).

## Przygotowanie operatora

Pakiet źródłowy odpowiada rewizji draft PR. Zainstaluj przejrzane źródła w
chronionym katalogu na laptopie. Wszystkie elementy ścieżki katalogu i jego
zawartość dostępna dla uprzywilejowanego narzędzia mają należeć do roota i
zezwalać na zapis wyłącznie właścicielowi. Polecenia niżej przyjmują checkout w
`/opt/stage9-reviewed` oraz osobny prywatny katalog wejść `/root/stage9-private`
o trybie `0700`. Wszystkie dowody z laptopa zachowaj w obiegu prywatnym.

Utwórz `inputs.json` na podstawie
[przykładu](../../deployment/stage9_v5d11/operator-inputs.example.json).
Ustaw tryb `0600`. Wprowadź niezależne piny istniejących receipt, aprobaty V2,
żądania ABI oraz klucza recenzenta. Umieść oryginalną aprobatę i żądanie ABI
poza drzewem A. Pole `root` ma dokładnie odpowiadać ścieżce z receipt, a UID i
GID są dodatnimi ID dedykowanego konta roboczego. Wybierz istniejący lokalny
socket RO. Puste `data_files` wystarcza do pierwszego pomiaru. Opcjonalny wpis
podaje ścieżkę docelową z dozwolonego zakresu, źródło `source` i jego oddzielnie
przypięty `sha256`. Kopie obrazów w tym pakiecie nie zapewniają semantyki
niezmiennego inode wymaganej przez V5d1.

Składanie statycznego bootstrapu i odczyt jego pinów nie wykonują kodu A.
Operator buduje pod zaufanym kompilatorem Ubuntu z biblioteką statyczną libc.
Przekaż recenzentowi dokładny plik wynikowy i skróty do niezależnego przeglądu.
Dla zwykłego użytkownika wybierz jego prywatny katalog roboczy zamiast podanej
ścieżki w `/root`, po czym zainstaluj przejrzany plik pod ścieżką z manifestu.

```sh
/usr/bin/python3.12 -I -S -B /opt/stage9-reviewed/scripts/stage9_v5d11_operator.py \
  build /root/stage9-private/bootstrap
```

Z wersjonowanego szablonu polityki wyrenderuj dokładne źródło i skompiluj je
według instrukcji [polityk](../../deployment/stage9_v5d11/README.md). Recenzent
porównuje piny kodu, renderu, wyniku parsera i bajtów BPF. Po przeglądzie zapisz
piny w prywatnym manifeście. Oddzielnie zatwierdzone dodanie polityki testowej
na stanowisku operatora używa trybu `add`. Istniejące polityki hosta zachowują
swoje ustawienia i zakres.

```sh
sudo /usr/bin/env -i PATH=/usr/bin:/bin LANG=C LC_ALL=C \
  /usr/bin/python3.12 -I -S -B /opt/stage9-reviewed/scripts/stage9_v5d11_operator.py \
  load-policy /root/stage9-private/inputs.json
```

## 1. Pomiar świeżego widoku

Na fizycznym stanowisku, po akceptacji wykonania pomiaru, uruchom jedno
polecenie. Katalog `measurement` przed startem musi być wolny. Zwykły tryb
kończy pomiar i zamyka widok. Wariant `--wait-for-review` utrzymuje widok do
godziny i czeka na pojedynczy wiersz z absolutną ścieżką podpisanej odpowiedzi.
EOF zamyka pomiar bez odpowiedzi.

```sh
sudo /usr/bin/env -i PATH=/usr/bin:/bin LANG=C LC_ALL=C \
  /usr/bin/python3.12 -I -S -B /opt/stage9-reviewed/scripts/stage9_v5d11_operator.py \
  measure /root/stage9-private/inputs.json /root/stage9-private/measurement --wait-for-review
```

Supervisor sprawdza przypięty receipt i aprobatę oraz kopiuje wyłącznie
przypięte bajty do osobnego katalogu roboczego. Worker odcina stary root,
tworzy prywatne przestrzenie mount, PID i network, prywatny procfs i ogranicza
montowania do root, procfs, urządzenia null i dokładnie jednego socketu RO.
Kontroluje wyczyszczenie odziedziczonych deskryptorów, GID/UID, capabilities,
`NoNewPrivs` i etykietę AppArmor. Sprawdza odmowę odczytu `environ` i
zachowanie filtra seccomp na próbach bez skutków ubocznych. Python operatora
porównuje eksport faktycznej polityki jądra oraz zmierzoną tożsamość socketu i
przestrzeni nazw. Polityka dla potomka qemu-img wymaga osobnej próby fizycznej.

Po odebraniu bajtów `mountinfo` supervisor zapisuje dokładną kandydaturę raz,
do istniejącego inode z zaufanej strony zaplecza. Root widoczny dla workera
pozostaje RO. Zapis nie dodaje montowania i nie przekazuje deskryptora po
stronie workera. Do katalogu wynikowego trafiają prywatne pliki `mountinfo`,
`measurement.json` i `launch-authorization.candidate.json`. Wyjście
`CANDIDATE_ONLY` oznacza ukończoną procedurę pomiaru i wciąż otwarty przegląd.

## 2. Niezależny przegląd

Recenzent pobiera prywatny pakiet pomiaru, oryginalne piny wejść i dokładne
polityki. Sprawdza `mountinfo`, tożsamości przestrzeni nazw, pliki A,
rzeczywisty stan polityk, przypadki odmowy, tożsamość socketu i konfigurację
demona. Przegląd zachowania libvirt oraz wyboru bibliotek prowadzi oddzielnie
według [kwalifikacji libvirt i loadera](stage9_v5d11_libvirt_loader_qualification.md).

Jeżeli recenzent akceptuje **pomiar** żywej sesji, sam przygotowuje kanoniczny
obiekt `review` z dokładnie pięcioma polami. `profile` ma wartość
`stage9-v5d11-isolation-review-v1`, `decision` ma wartość
`ACCEPT_MEASURED_VIEW`, a `session`, `candidate_sha256` i `measurement_sha256`
wiążą identyfikator z pomiaru i skróty dokładnych bajtów dwóch plików JSON.
Podpis Ed25519 obejmuje kanoniczne bajty tego obiektu według serializatora
V5d10 (`sort_keys=True`, `separators=(",", ":")`, ASCII, bez końcowego znaku
nowej linii). Prywatny klucz pozostaje u recenzenta. Odpowiedź ma wyłącznie
pola `review` i `signature_hex` w małych znakach hex. Wcześniej niezależnie
przypięty klucz publiczny jest 32-bajtowym Ed25519 w DER SubjectPublicKeyInfo.

Operator odkłada odpowiedź jako chroniony plik `0600` i podaje jedną ścieżkę w
już oczekującym terminalu. Supervisor sprawdza podpis, aktualny hash
kandydatury, politykę i istnienie tego samego workera. Zapisuje `review.json`
i kończy z `REVIEW_VERIFIED_EXECUTION_BLOCKED`. Wygaśnięcie lub utrata workera
wymaga nowego pomiaru i przeglądu nowej sesji.

## 3. Osobna autoryzacja uruchomienia

Ten etap pozostaje zablokowany. [Analiza V5d1](stage9_v5d11_v5d1_compatibility.md)
wiąże blokadę własności z konkretnymi trybami i wymaganiami kodu. Po przeglądzie
rozwiązania tej niezgodności będą potrzebne kwalifikacja jądra i potomka
qemu-img, udokumentowana granica protokołu libvirt oraz pomiar faktycznego
loadera. Przyszła, jawna zgoda operatora będzie musiała wskazać dokładny widok,
zaakceptowane materiały i jednorazowe wykonanie istniejącego launchera dla
profilu A. Sam plik kandydacki nie stanowi zgody.

`STAGE_9=IN_PROGRESS`, `OBSERVER_3A=BLOCKED`, `STAGE_3B=NOT_AUTHORIZED`,
`PRODUCTION=NO-GO`.
