# Stage 9 — S9-RPI-17: współpracujący TEST lease źródła

**Status:** `TEST_ONLY_NON_AUTHORIZING`. **Produkcja:** `PRODUCTION_NO_GO`.

## Problem zamykany w eksperymencie

S9-RPI-14/15/16 sprawdzają oryginalne bajty, politykę Ed25519,
tożsamość plików i zewnętrzny, testowy `floor` po restarcie.
Nie serializują jednak odczytu względem **współpracującego**
procesu zmieniającego stan źródła. Snapshot plików nie stanowi
source custody lease.

S9-RPI-17 wykorzystuje **istniejący** `flock` na
`root/lock` w [TEST Linux source worker](../../tests/security/_custody_source_demo.py).
W osobnym [teście integracyjnym](../../tests/security/test_source_custody_shared_lease_linux.py)
trzymamy `LOCK_SH` przez całe `inspect_synthetic_retained_source_original()`
wraz z trzema odczytami źródła i kontrolą floor/policy z S9-RPI-16.
Istniejący worker potrzebuje na tym samym inode `LOCK_EX`.
Nie wprowadzamy nowego pliku lock, migracji, produkcyjnego lease
ani API do wywołania mutation.

### Kontrakt TEST read lease

- Otwórz istniejący katalog `root` i plik `lock` przez Linux
  `O_DIRECTORY/O_NOFOLLOW/O_CLOEXEC` oraz katalogowy `dirfd`.
- Porównaj `st_dev/st_ino` z oryginalnym pinem fixture i sprawdź
  faktyczne UID/GID, `0600`, regular file, `nlink=1` oraz
  katalog `0700`.
- Zdobądź współdzielony `flock` i ponów weryfikację lock inode.
- Trzymaj lock **aż do zakończenia** końcowej kontroli oryginalnego
  eventu, podpisu i testowego floor, a także przez consumer callback.
- Przed zwolnieniem sprawdź ponownie ten sam plik i ścieżkę.
  Zmiana inode/ACL/mode, podmiana katalogu lub brak pinu odmawia.
- Zwracaj wyłącznie `SyntheticSourceReadLease`:
  `NOT_QUALIFIED`, `BLOCKED`, `source_authority_qualified=False`.

### Testy

- Pełny oryginalny odczyt S9-RPI-16 przy współdzielonym
  `flock`; wszystkie wyniki `NOT_QUALIFIED`.
- Prawdziwy oddzielny proces `spawn`, sygnalizujący **gotowość
  do próby** `LOCK_EX`; nie przechodzi jej przed zwolnieniem
  `LOCK_SH`, a następnie uzyskuje blokadę. `spawn` jest ważny,
  bo zwykłe `fork` odziedziczyłoby zablokowany deskryptor
  po rodzicu i wypaczyłoby wynik.
- Istniejący worker custody source uruchomiony pod `spawn`
  czeka na wspólną blokadę, po zwolnieniu odzyskuje identyczne
  oryginalne bajty bez nowego podpisu.
- Odmowy: lock o innym inode, symlink, zmienione uprawnienia,
  podmiana inode w trakcie callbacku.
- Celowo **niekooperujący** zapis `events/original` potrafi
  ominąć `flock`, ale istniejące rechecks odrzucają zmienione
  bytes. To dowodzi ograniczenia, nie jego usunięcia.

Test jest Linux-only, lecz importy na poziomie modułu pozostają
przenośne; `fcntl` ładowany jest dopiero po sprawdzeniu platformy.
Włączony do istniejącego `PostgreSQL atomic Root Proof commit`,
Ruff Format i strict CI inventory. Nie ma nowego ciężkiego
workflow ani produkcyjnej zależności.

## Zakazy i dalsze zależności

Ta blokada jest tylko **współpracująca** i działa w testowej,
same-UID domenie. Nie dowodzi bieżącej/historical eligibility
credentials, niezależnej source authority, rzeczywistego
source custody lock, odrębnego CCA principal ani jednolitego
globalnego lock order przez trwałą decyzję CCA. Nie zapobiega
mutation poza `flock`, ABA, skoordynowanemu rollbackowi całego
hosta i ingerencji w policy/pin. Nie wykazuje też trwałości
w razie utraty zasilania.

Brak zmian w production SQL, `complete_close`, grants,
`SIGN/BIND`, produkcyjnych signerach, enrollment, frozen
kontraktach, Stage 9/10 statusach lub completion publication.

**Następne materialne blokery:** realne rozłączne OS principals,
protected installation pins, kwalifikowane ACL i source custody
lease obejmujący exact generation/descendant, zdefiniowany
lock order source/CCA, independent rollback domain i end-to-end
crash/restart qualification.

**Sprzęt:** S9-RPI-17 nadal działa na GitHub Linux runner
bez dodatkowego komputera. Fizyczna kwalifikacja instalacji
wymaga natywnego Linux/systemd z prawdziwą separacją UID/GID i
ACL. Oddzielny host/nośnik dla niezależnej rollback domain
wymaga osobnej decyzji.
