# Stage 9 — S9-RPI-19: pełny kryptograficzny łańcuch source → CCA (TEST)

**Status:** `TEST_ONLY_NON_AUTHORIZING`, `NOT_QUALIFIED`,
`BLOCKED`. **Produkcja:** `PRODUCTION_NO_GO`.

## Dlaczego S9-RPI-18 nie był wystarczający

S9-RPI-18 zachował oryginalny source envelope z realnego
Linux TEST workera w niezmiennej decyzji logu CCA, lecz używał
`parse_independent_cca_decision()`, który waliduje strukturę,
**nie** podpis CCA ani podpis zawartego eventu względem policy.

W oryginalnym workerze TEST policy zawierała losowy publiczny
klucz CCA, do którego nie zachowano klucza prywatnego. Podmienienie
samej policy podczas testu byłoby niezgodne z retained original
i osłabiłoby dowód. Dlatego S9-RPI-19 rozszerza wyłącznie
[TEST source fixture](../../tests/security/_authenticated_custody_source_fixtures.py)
oraz [izolowany Linux worker](../../tests/security/_custody_source_demo.py):
tylko przy jawnej inicjalizacji przyjmują **32-bajtowy publiczny
klucz CCA**, generowany w osobnym procesie testowym.
Nigdy nie przekazują workerowi prywatnego klucza CCA;
prywatny klucz source nadal powstaje i pozostaje u workera.

Domyślna ścieżka bez podania publicznego CCA zachowuje
istniejący proces tworzenia syntetycznej polityki.

## Co test dowodzi

[Test integracyjny S9-RPI-19](../../tests/security/test_source_cca_cryptographic_chain_linux.py):

1. Osobny TEST orkiestrator generuje CCA Ed25519 keypair, lecz
   przekazuje workerowi wyłącznie `cca_public` **przed**
   utworzeniem source eventu i `public-policy`.
2. Linux source worker sam wykonuje syntetyczną zmianę custody,
   zachowuje original i policy z fsync, a policy z wyprzedzeniem
   zawiera ten sam CCA public key, którego prywatną część
   zatrzymał inny testowy proces.
3. S9-RPI-15/16 ponownie odczytuje rzeczywisty plik
   `events/original`, sprawdza source signature Ed25519,
   tożsamość pliku i niezmieniony TEST witness floor.
4. Synthetic CCA payload wiąże exact bajty **tego**
   original, a jego explicit CCA policy zawiera **niezmienione
   bajty faktycznej `events/public-policy`**, a nie policy
   nowego, niezwiązanego source fixture. Principal inventory
   jest wiązany z realnym TEST public-material identity źródła.
5. Pod utrzymanym S9-RPI-17 cooperative `LOCK_SH` istniejący
   testowy CCA log zapisuje immutable decision, reservation floor
   i oryginalny signed envelope bez zmiany produkcji.
6. `verify_independent_cca_decision()` bada **oba podpisy
   Ed25519**, source i CCA, stałość CCA policy przed/po
   weryfikacji oraz ich exact binding.
7. Sprawdzany jest ponowny odczyt retained CCA original,
   recovery source worker i exact retry CCA **bez re-sign**.

### Negatywne testy

- Wstrzyknięty fałszywy podpis CCA (`64*00`) nie jest
  akceptowany mimo poprawnego source original i policy.
- Poprawnie ustrukturyzowana policy CCA zmieniona pomiędzy
  początkowym i końcowym odczytem powoduje odmowę.
- Zmiana realnego pliku `events/public-policy` odmawia przez
  istniejące inode/metadata checks pod trzymanym source lease,
  niezależnie od tego, że wcześniej pobrany i jawnie przypięty
  snapshot historycznie da się zweryfikować offline.

## Ścisłe ograniczenia

Wartości `QUALIFIED_INSTALLED_SOURCE`, `SOURCE_AND_CCA_RECORDS_SYNCED`
itd. są tylko składnią **syntetycznego DTO**. Sukces dwóch
podpisów nie stanowi potwierdzenia legalnego enrollment,
`PRODUCTION_LOCAL` installation authority, niezależnych
UID/GID/ACL ani niezależnego DB committed-close reader.
Publiczny klucz CCA jest dostarczony **przez fixture**,
nie z protected installation manifest, a CCA private key
pozostaje w tym samym kontenerze/UID testu. To nie jest
independent authenticated CCA credential lifecycle ani
kwalifikowana separacja source/CCA principal. `flock` jest
advisory i nie blokuje niekooperującego same-UID writer.
`fsync` nie dowodzi power-loss durability, ani nie chroni
przed skoordynowanym rollbackiem obu floorów.

Cała implementacja pozostaje w `tests/security` i
`docs/architecture`; produkcyjne moduły, kontrakty, SQL,
granty, `SIGN/BIND`, uprawnienia podpisywania, runtime CCA,
root-proof issuer i completion publication nie są zmienione.

Włączony do istniejących Linux Stage 9 atomic root-proof
pytest, Ruff format i strict inventory. Brak dodatkowego
workflow i nowych produkcyjnych zależności.

**Sprzęt TERAZ: dodatkowy komputer NIE jest potrzebny.**
Do celowanego TEST Linux wystarczy GitHub runner.
Do fizycznej kwalifikacji potrzeba natywnego Linux/systemd,
odrębnych principalów, chronionych instalacyjnych pinów
oraz osobnej decyzji o niezależnej domenie rollback.
