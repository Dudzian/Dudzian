# Stage 9 — S9-RPI-18: source lease → CCA durable decision (TEST)

**Zakres:** `TEST_ONLY_NON_AUTHORIZING`.
**Produkcja:** `IN_PROGRESS / PRODUCTION_NO_GO`.

## Zaobserwowany problem

S9-RPI-17 dowodził, że współpracujący worker nie uzyskuje
`LOCK_EX`, gdy syntetyczny CCA-style reader trzyma
`LOCK_SH`. Osobny S9-RPI-07 realizował trwały (testowy)
rejestr CCA i kontrolowane recovery przy interrupted intent,
a S9-RPI-16 zachowywał odrębny testowy floor source original.
Brakowało konkretnego testu, w którym **ta sama blokada źródła**
jest utrzymana podczas zapisu decyzji do **istniejącego** logu CCA.

[Test S9-RPI-18](../../tests/security/test_source_lease_cca_retained_decision_order.py)
wspina razem istniejące komponenty:
- rzeczywisty Linux TEST custody worker z S9-RPI-15,
  `events/original` i `public-policy`;
- niezależnie ponawiany odczyt oryginalnego source eventu,
  weryfikację Ed25519 i TEST witness z S9-RPI-16;
- współpracującą blokadę source `LOCK_SH` z S9-RPI-17;
- TEST-only `TestOnlyCCADecisionLog` z S9-RPI-07,
  mający własny log `flock`, `intent`, rezerwację
  `floor`, immutable `decision`, `fsync` i exact lookup.

Kolejność w **tym eksperymencie**:

1. `source LOCK_SH` i przypięcie istniejącego inode.
2. Początkowy exact source original/policy/floor lookup.
3. Odrębny testowy podpisujący CCA tworzy *syntetyczne*
   `IndependentCustodyCompletionV1`, odnoszące się do
   **dokładnie tego samego** oryginalnego source envelope
   zapisanego przez workera Linux.
4. `CCA log LOCK_EX`, trwały intent, osobna rezerwacja
   CCA floor, immutable decision i sync przed ACK z logu.
5. Exact CCA lookup oraz końcowe rechecki source original,
   policy i source floor **z nadal trzymanym** `source LOCK_SH`.
6. Zwolnienie `source LOCK_SH` dopiero po ukończeniu
   odpowiedniej sekwencji.

### Ograniczenia tego kontraktu

Wewnętrzne pola CCA fixture deklarujące
`QUALIFIED_INSTALLED_SOURCE` i `SOURCE_AND_CCA_RECORDS_SYNCED`
to **wyłącznie syntetyczne wartości schematu**, nie realna
kwalifikacja! CCA envelope tworzy pomocniczy klucz TEST.
Nie ma zewnętrznie kwalifikowanego close readera PostgreSQL
ani źródła current/historical credential eligibility. Prawdziwy
CCA nie jest zainstalowany i nie istnieje usługa produkcyjna,
która atomowo utrzymuje lock order między dwoma principalami.
Przekazanie oryginalnego source envelope do TEST case nie
zmienia statusu jego pochodzenia z fixture.

### Weryfikowane ścieżki

- Zapis jednej decyzji i dokładnie jedna decyzja w rejestrze;
  ponowny exact lookup po zwolnieniu lease zwraca te same bajty.
  Retry nie uruchamia kolejnego callbacku podpisującego.
- Rzeczywisty `spawn` contender raportuje próbę
  `LOCK_EX` i nie przekracza źródłowej blokady ani przy
  rezerwacji CCA floor, ani po `after_durable_decision`,
  ani przed ostatnim recheckiem source.
- Przerwanie po `after_floor_reservation` pozostawia
  dokładny pending intent i prowadzi do
  `TestOnlyCCALogIndeterminate` bez ponownego podpisania.
- Podmiana source original przed pozyskaniem pozytywnego
  testowego dowodu musi zostać odrzucona **przed** zapisem
  jakiegokolwiek CCA intent/decision.

## Bezpieczeństwo i przyszła kwalifikacja

Wynik to współpracująca serializacja jednego testowego
source worker i jednego TEST CCA logu, pod tym samym UID.
Nie chroni przed pominięciem `flock`, skoordynowanym
rollbackiem, podmianą pinu i floor, niekooperującym writerem,
brakiem źródła historycznych credential generations,
fizycznym power loss ani crash processów w różnych punktach
transakcji CCA. Samo istnienie retained CCA record nie jest
poświadczeniem execution source; unknown outcome pozostaje
unknown.

Nie ma nowego produkcyjnego source lease, uprawnień,
key provisioning, SQL migrations, `SIGN/BIND`, akceptacji
completion publication, modyfikacji frozen kontraktów ani
zmiany decyzji Stage 9/10.

**Materialne kolejne blokery:** niezależne native Linux OS
principals i authenticated peer/pin, verified source generation
oraz lock order source/CCA/credential lifecycle, zaufany
floor poza domeną rollback i crash-powerloss qualification.

**Dodatkowy komputer TERAZ: NIE.** Test przechodzi przez GitHub
Linux CI. Fizyczną kwalifikację przygotować na natywnym
Linux/systemd z co najmniej rozdzielonymi principalami i
chronioną instalacją. Oddzielny host lub nośnik do niezależnej
domeny rollbacku to odrębna decyzja.
