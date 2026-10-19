# Stage 9 — S9-RPI-15: real Linux TEST source original disk bridge

**Status:** `TEST_ONLY_NON_AUTHORIZING`. **Production:** `PRODUCTION_NO_GO`.
Continuation of S9-RPI-14 (PR #3169) and its policy snapshot repair
(PR #3170).

## Zakres

S9-RPI-14 porównywał source original ze wstrzykniętym obiektem mock.
Tym razem istniejący [isolated Linux source worker](../../tests/security/_custody_source_demo.py) tworzy w jednorazowym
katalogu realny plik `events/original`, osobny `public-policy`
oraz `journal/operation` i `metadata`. Worker sam wyprowadza
payload, wykonuje testową zmianę metadata, deklarowane fsync
plików i katalogów oraz zapisuje testowy event bez przekazywania
klucza podpisującego do procesu nadrzędnego.

[Test weryfikacji fizycznego pliku](../../tests/security/test_retained_source_real_linux_disk_bridge.py)
dodaje wyłącznie **nieprodukcyjny test-adapter** do istniejącego
`RetainedSourceEventReader`: odczytuje oryginał przez `dirfd`,
`O_NOFOLLOW`, `fstat`, exact ownership/modes, oczekiwane
inode i metadane oraz sprawdza ścieżkę przed/po odczycie.
Oryginalny Ed25519 event przechodzi następnie istniejącą
`inspect_synthetic_retained_source_original()`, trzy lookupy,
niezależny (ale tylko fixture-level) policy loader i ścisły
source log position binding.

Testy sprawdzają restart z **identycznymi original bytes** oraz
odmowy po:
- `chmod` oryginalnego pliku;
- podmianie na inny inode z identycznymi bytes;
- symlinkowaniu retained source file;
- podmianie katalogu `events`;
- obcięciu oryginału;
- mutacji oryginału w czasie callbacku consumer;
- podmianie pliku policy na inną inode podczas callbacku.

Test włącza się tylko na Linux w istniejącym `atomic-root-proof-commit`
i jego Ruff/strict CI inventory. **Brak dodatkowego produkcyjnego
modułu**, SQL/migracji, grantów, kluczy, service unit czy listenerów.

## Co jest dowodem, a co nim nie jest

Jest to materialny krok ponad in-memory mock: test czyta **istniejące
bajty zapisane przez osobny proces** i sprawdza obecny inode,
uprawnienia oraz regresje przejść. Fizyczny Linux source worker
wykonuje `fsync`, lecz nie jest potwierdzona trwałość wobec
odcięcia zasilania ani niezależna instalacyjna kwalifikacja
wykonania custody mutation.

Reader i worker nadal działają pod **tym samym UID**. Pin i
expected digest pochodzą z fixture (TOFU wyłącznie na potrzeby
testu), a nie niezależnej chronionej instalacji. `stat`/path
checks są snapshotami, nie kernelowym permanentnym lease;
nie uniemożliwiają zmian i przywrócenia stanu ABA ani ingerencji
tego samego UID/administratora, skoordynowanego rollbacku, replay
czy ataku na zaufany loader. Nie ma trwałego zewnętrznego floor,
proof of prior fsync, uprawnionej polityki source, fizycznego
CCA reader principal, source custody lease i lifecycle rotation
interlock.

**To nie jest gotowy produkcyjny retained source reader**.
Wynik `SyntheticRetainedSourceEvidence` pozostaje `NOT_QUALIFIED`,
`BLOCKED`, `provenance_qualified=False`,
`durability_attested=False`. Stage 9 pozostaje
`IN_PROGRESS / PRODUCTION_NO_GO`.

## Dalsze wymagania

Następna kwalifikacja wymaga niezależnego źródła installation
identity/ACL/protected pins, fizycznego source logu z trwałym
checkpoint/floor i blokowaniem rollback, zdefiniowanego lock order
source/CCA oraz udokumentowanego fail-closed crash recovery
wobec unknown outcome. Dopiero wtedy ma sens wiązać to
z source custody lease i credential eligibility.

**Dodatkowy komputer: NIE POTRZEBNY teraz.**
GitHub Linux CI uruchamia obecny proces. Przy produkcyjnej
kwalifikacji potrzebny będzie native Linux/systemd ze
zdefiniowanymi osobnymi UID/GID i realnymi ACL. Fizyczna separacja
rollback domain na odrębnym hoście/nośniku to osobny wybór
architektoniczny, nie efekt tego testu.
