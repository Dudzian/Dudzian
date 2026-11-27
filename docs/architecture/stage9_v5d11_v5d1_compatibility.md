# Stage 9 V5d11 — zgodność z preflight V5d1

## Własność plików i uprawnienia profilu A

Launcher V5d10 wymaga niezerowego efektywnego UID i sprawdza dokładne tryby
plików oraz katalogów względem receipt. W przypiętym profilu A katalog główny
ma tryb `0700`, plik startowy i źródła kontrolera mają tryb `0600`, a interpreter
ma tryb `0755`. Bootstrap może udostępnić kopię w tych trybach procesowi
launchera, przypisując ją jego UID i zamykając zapis przez montowanie `ro`.

Weryfikator tożsamości V5d1 wymaga `st_uid == 0` dla każdego przypiętego
artefaktu i wszystkich jego katalogów nadrzędnych. Wymaga także wyłączenia
zapisu dla grupy oraz pozostałych użytkowników. Dotyczy to źródeł kontrolera
i rzeczywistej ścieżki interpretera odczytanej z `/proc/self/exe`. Dla kopii
profilu A należącej do UID launchera kończy się to `IDENTITY_PATH_NOT_PROTECTED`.
Agregator V5d1 traktuje ten powód jako blokadę pomiarów kontrolera przed
otwarciem połączenia libvirt. Gdy A należy do UID `0`, tryby `0700` i `0600`
blokują dostęp launchera po zrzuceniu uprawnień. Nazwana reguła POSIX ACL
dająca innemu UID odczyt lub przejście wymaga maski odzwierciedlonej w bitach
grupy, więc zmieniłaby tryby wiązane przez receipt.

[Launcher V5d10](../../scripts/stage9_v5d10_launch.py),
[kontrakt receipt](../../scripts/stage9_v5d10_contract.py),
[weryfikator tożsamości V5d1](../../deployment/stage9_observer_collector_identity_v5d1.py),
[agregator V5d1](../../deployment/stage9_observer_kvm_preflight_v5d1.py).

### Odrębny receipt uruchomieniowy

V5d11 wyprowadza osobną kandydaturę receipt z dokładnych bajtów receipt
materializacji i jego niezależnej aprobaty. Wiąże ich SHA-256, hash starego
planu, skrót zatwierdzonych źródeł, UID/GID oraz nowy plan. Weryfikuje też
piny kodu, natywnych składników wheel, wiązań libvirt oraz interpretera.
Oryginalny `ENVIRONMENT.json` trafia do kopii jako przypięty plik o tych samych
bajtach. Dozwolona transformacja obejmuje wyłącznie właściciela i tryby
prywatnej kopii. Oryginalny plan i weryfikator V5d10 zachowują swój kontrakt.

Kopia używa właściciela UID 0 i dedykowanego GID. Katalogi mają tryb `0550`,
pliki danych i źródeł `0440`, a pliki wcześniej wykonywalne `0550`.
Katalogi nadrzędne udostępniają odczyt i przejście, których wymaga używane
przez V5d1 `O_RDONLY | O_DIRECTORY`. Weryfikator kopii sprawdza dokładne
drzewo, bajty, tryby i tożsamości inode. Odrzuca dowiązania, kolizje inode,
zmiany ścieżek oraz dostęp grupy do zapisu. Bootstrap zrzuca dodatkowe grupy.
Dedykowany proces ma odczyt i przejście dzięki swojemu GID. Po weryfikacji
kopii pomiar zapisuje `runtime-receipt.candidate.json`. Osobna aprobata
diagnostyki wiąże SHA-256 dokładnych bajtów tego pliku.

Testy syntetyczne obejmują rzeczywisty odczyt i odmowę zapisu przez kernel,
rozbieżność właściciela lub GID, nowy plik, dowiązania i odrębność inode.
Test z rzeczywistą zmianą tożsamości procesu sprawdza odczyt członka grupy
i odmowę dla obcego GID przez niezmieniony mechanizm V5d1. Na runnerze bez
uprawnień `SETUID`, `SETGID` i `SYS_CHROOT` ten przypadek ma stan `NOT_TESTED`.

[Reguły projekcji runtime](../../scripts/stage9_v5d11_runtime.py),
[testy DAC i tożsamości](../../tests/deployment/test_stage9_v5d11_runtime.py).

## Niezmienność obrazów

V5d1 wymaga dla każdego obrazu zwykłego pliku o jednym dowiązaniu, chronionych
katalogów nadrzędnych i flagi inode `FS_IMMUTABLE_FL` odczytanej przez
`FS_IOC_GETFLAGS`. Sprawdza flagę przed uruchomieniem `qemu-img` i po ponownym
otwarciu ścieżki. Kwalifikacja obrazu wymaga rzeczywistej flagi na obiekcie
widocznym w izolacji. Montowanie `ro` egzekwuje odrębne ograniczenie zapisu
na poziomie montowania. Utworzenie kopii obrazu ustanawia nowy inode, którego
flagi i tożsamość należy zmierzyć osobno.

V5d11 tworzy wyłącznie nowe, przypięte kopie w chronionym obszarze roboczym
na ext4. Ustawia i sprawdza `FS_IMMUTABLE_FL` na otwartym deskryptorze kopii.
Montowanie zachowuje ten sam inode. Bootstrap ponownie sprawdza urządzenie,
inode, rozmiar i flagę wewnątrz właściwego widoku. Źródłowy obraz operatora
jest otwierany do odczytu. Sprzątanie dotyczy inode utworzonych dla tej próby.

[Pomiar obrazów V5d1](../../deployment/stage9_observer_kvm_image_v5d1.py),
[przygotowanie kopii V5d11](../../scripts/stage9_v5d11_images.py).

## Punkt wznowienia

Fizyczne próby loadera, libvirt, potomnego `qemu-img`, DAC i obrazu mają stan
`NOT_TESTED`. Dla kwalifikacji tych granic potrzebny jest osobno zaakceptowany
przebieg diagnostyczny na laptopie. Preflight V5d1 wymaga jeszcze oddzielnego
przeglądu i dopuszczenia. CLI V5d1 zachowuje wyjście `2` również wtedy, gdy
wykona lokalne odczyty.

[CLI V5d1](../../scripts/stage9_observer_kvm_preflight_v5d1.py).
