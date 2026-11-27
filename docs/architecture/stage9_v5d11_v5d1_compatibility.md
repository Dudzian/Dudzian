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

## Niezmienność obrazów

V5d1 wymaga dla każdego obrazu zwykłego pliku o jednym dowiązaniu, chronionych
katalogów nadrzędnych i flagi inode `FS_IMMUTABLE_FL` odczytanej przez
`FS_IOC_GETFLAGS`. Sprawdza flagę przed uruchomieniem `qemu-img` i po ponownym
otwarciu ścieżki. Kwalifikacja obrazu wymaga rzeczywistej flagi na obiekcie
widocznym w izolacji. Montowanie `ro` egzekwuje odrębne ograniczenie zapisu
na poziomie montowania. Utworzenie kopii obrazu ustanawia nowy inode, którego
flagi i tożsamość należy zmierzyć osobno.

[Pomiar obrazów V5d1](../../deployment/stage9_observer_kvm_image_v5d1.py).

## Punkt wznowienia

Fizyczne próby loadera, libvirt i potomnego `qemu-img` mają stan `NOT_TESTED`.
Dla kwalifikacji tych granic należy użyć osobno zaakceptowanego przebiegu
diagnostycznego. Przed odrębnym dopuszczeniem preflight V5d1 konieczny jest
przegląd sposobu pogodzenia uprawnień A z
weryfikatorem V5d1 i zapewnienia flagi niezmienności obrazów w dokładnym widoku.
CLI V5d1 zachowuje wyjście `2` również wtedy, gdy wykona lokalne odczyty.

[CLI V5d1](../../scripts/stage9_observer_kvm_preflight_v5d1.py).
