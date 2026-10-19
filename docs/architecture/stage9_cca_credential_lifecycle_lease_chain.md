# Stage 9 — S9-RPI-20: credential-lifecycle fence podczas source → CCA (TEST)

**Stan:** `TEST_ONLY_NON_AUTHORIZING`, `NOT_QUALIFIED`,
`BLOCKED` i `PRODUCTION_NO_GO`.

## Problem

S9-RPI-19 sprawdza podpisy oryginalnego source eventu i decyzji CCA
wobec niezmienionych testowych polityk. Same poprawne podpisy i
ponowne odczytanie `ACTIVE` **nie zabezpieczają** decyzji przed
równoległą rotacją/revokacją poświadczenia pomiędzy weryfikacją a
trwałym zapisem. S9-RPI-07 testował blokadę credential lifecycle w
izolowanym demonstratorze, lecz nie w pełnym łańcuchu kryptograficznym.

[Nowy test S9-RPI-20](../../tests/security/test_cca_credential_lifecycle_chain_linux.py)
wykorzystuje ten sam `start_rotation` z
[`_cca_log_process.py`](../../tests/security/_cca_log_process.py),
a także zapis oryginału i sprawdzanie Ed25519 z S9-RPI-19.

### Testowa kolejność blokad

1. Przy inicjalizacji TEST fixture zapisz stabilne piny katalogu
   `credential-lease`, pliku blokady i pliku `credential-state`.
2. Uzyskaj istniejący `source LOCK_SH` z S9-RPI-17.
3. Uzyskaj `credential LOCK_SH` z bounded `LOCK_NB` + deadline;
   sprawdź identyczny inode, UID/GID, uprawnienia, liczbę linków
   i dokładny `ACTIVE:1:1`. Brak/zmiana/timeout kończy się
   `TestOnlyCredentialLeaseError`, bez awaryjnego utworzenia lock.
4. Wykonaj exact source-original/floor/policy read oraz
   `TestOnlyCCADecisionLog.decide`, które zdobywa własny
   `CCA log LOCK_EX`, zapisuje intent, checkpoint/floor i
   oryginalną decyzję z `fsync`.
5. Pod wciąż trzymanymi source/credential lease zweryfikuj
   **oba podpisy Ed25519**, source i CCA, i końcowe odczyty.
6. Przed zwolnieniem credential lease ponów sprawdzenie
   aktywnego stanu poświadczenia, inode i ścieżek (w tym
   samego katalogu). Dopiero wtedy zwolnij credential
   `LOCK_SH`, a następnie source `LOCK_SH`.

### Scenariusze

- W pełni syntetyczny pozytywny łańcuch pozostawia dokładnie
  jedną retained CCA decision, `NOT_QUALIFIED/BLOCKED`.
- Dwa rzeczywiste procesy `spawn` rotacji
  `ACTIVE → VERIFY_ONLY:2:1` oraz `ACTIVE → REVOKED:2:1`
  próbują zdobyć `credential LOCK_EX` podczas
  `after_floor_reservation`; kernel zgłasza rzeczywisty
  konflikt. Rotacja przechodzi dopiero po finalnym
  `after_durable_decision` i zwolnieniu credential lease.
- Po rotacji nowa próba decyzyjna odmawia, natomiast
  `exact_lookup` dawnej decyzji w logu zwraca jedynie oryginalne
  bajty — **nie** dowód jej bieżącej lub historycznej eligibility.
- Już cofnięte poświadczenie odmawia przed zapisaniem CCA intent.
- Bezpośrednia, niekooperująca zmiana pliku stanu pod advisory flock
  powoduje odmowę przy końcowym rechecku. Zapisane bajty
  testowej decyzji nie mogą zostać uznane za ACK produkcyjne.
- Niebezpieczne pliki lock (missing/symlink/mode/inode) oraz
  podmiana katalogu podczas lease odmawiają.

## Granice dowodowe

To jest zamknięta symulacja `ACTIVE:1:1` i dwóch docelowych
stanów poświadczenia, **nie kwalifikowany historyczny
credential eligibility protocol**. Oryginalny signer CCA ma
nadal testowy klucz prywatny, oba procesy i pliki działają pod
tym samym UID; `flock` jest współpracujące (advisory).
Zewnętrzny `fixture_pin` pochodzi z tej samej domeny, a nie
z chronionego root-owned manifestu. Nie ma realnej niezależnej
CCA authority, systemd, różnic UID/GID, żywego PostgreSQL close
reader, zewnętrznego lifecycle journal/floor, gwarancji
power-loss ani ochrony przed skoordynowanym rollbackiem.
Nie ma też wiarygodnego potwierdzenia, że stale podpisane
operacje były zgodne z historycznymi prawami klucza.

Istniejące pola `QUALIFIED_INSTALLED_SOURCE` i
`SOURCE_AND_CCA_RECORDS_SYNCED` w testowej strukturze CCA
pozostają tylko słownikiem fixture; nie są prawdziwymi dowodami.
Nie ma zmian w production bot_core, SQL, grants, frozen contracts,
production signing, `SIGN/BIND`, publication i statusie Stage 9/10.

Włączony wyłącznie do istniejącego Linux atomic root-proof
pytest, Ruff i strict CI inventory. Nie dodawano kolejnego workflow.

**Dodatkowy komputer TERAZ: NIEPOTRZEBNY.** GitHub Linux runner
wystarcza do tego testu. Fizyczna kwalifikacja produkcyjna
wymaga natywnego Linux/systemd, rzeczywiście rozdzielonych UID/GID,
chronionego instalacyjnego manifestu i źródła historii credentiali.
Niezależna domena rollback wymaga osobnej decyzji o nośniku/hoście.
