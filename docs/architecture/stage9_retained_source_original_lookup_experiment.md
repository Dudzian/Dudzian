# Stage 9 — S9-RPI-14: retained source event, exact original lookup

**Status:** `TEST_ONLY_NON_AUTHORIZING`; **production:** `NO-GO`.

S9-RPI-13 (PR #3168) został scalony. Spójność transakcji czytnika CCA
i zgodny podpis źródła **nie dowodzą**, że oryginalny source event
został zachowany w instalacyjnym, odpornym na rollback rejestrze.

## Zakres tego eksperymentu

Nowy moduł
[`test_only_retained_source_original.py`](../../bot_core/test_only_retained_source_original.py)
demonstruje protokół *exact lookup* oryginalnego envelope
`CustodyMutationExecutionV1` wobec **zewnętrznie podanych
syntetycznych oczekiwań**. Przyjmuje istniejący, produkcyjnie
niezaimplementowany port `RetainedSourceEventReader` z
[`independent_cca_decision.py`](../../bot_core/independent_cca_decision.py).
Nie implementuje trwałego rejestru, instalatora ani peer-auth transportu.

Kroki:
1. Pobiera `ExplicitSourceVerificationPolicy` z osobnego bezargumentowego
   loadera. Loader **nie otrzymuje** od callera envelope, key ani ścieżki.
2. Wywołuje `lookup_original_event(installation_id, operation_id, execution_id)`
   z identyfikatorami ze stałego syntetycznego `SyntheticRetainedSourcePin`,
   a nie z pobranych z nieufnego envelope.
3. Wymaga identycznych zachowanych oryginalnych bytes,
   ich SHA256, log namespace/sequence/previous digest, operation/execution
   i installation ID. Weryfikuje istniejący, niezależnie
   przypięty Ed25519 source signature i zamknięty protokół S9-RPI-06.
4. Powtarza lookup i policy snapshot przed zwróceniem wyniku
   oraz ponownie po consumer callback, przed pomyślnym zakończeniem `with`.
5. Nie daje żadnej pozytywnej kwalifikacji: wynik ma zawsze
   `status=NOT_QUALIFIED`, `production_authorization=BLOCKED`,
   `provenance_qualified=false`, `durability_attested=false`.

## Testy i odmowy

[`test_retained_source_original_lookup.py`](../../tests/security/test_retained_source_original_lookup.py)
używa wyłącznie syntetycznego `source_case`, klucza testowego i
wstrzykniętego `_SyntheticReader`; **nie instaluje źródła**.
Negatywy: obce installation/operation/execution, log namespace/position,
digest, caller-provided envelope, hash zgodny z podmienionym source
envelope lecz niepoprawny podpis, podmiana zwracanego original między
odczytami i podczas callbacku, utrata niezależnej policy oraz pusty/
niepoprawny output portu. Dodatkowy test sprawdza brak bezpośredniego
importu modułu TEST-only w grafie produkcyjnego CCA.

Test jest jawnie podłączony do istniejącego Stage 9 atomic security
pytest, pokrycia `--cov`, Ruff Format i ścisłego inwentarza CI.
Nie deklarujemy PASS dopóki workflow nie zwróci wyniku.

## Granice bezpieczeństwa

Ten etap wykazuje tylko poprawność *procedury porównania danych*
dla mockowanego źródła. Nie kwalifikuje samodzielnego pochodzenia
pinu, policy czy rejestru. Mock może zwrócić identyczne bytes
bez jakiegokolwiek utrwalenia, niezależności, poprawnego source
process UID/GID, rzeczywistego custody mutation lub wcześniejszego
`fsync`. Pośredni `ABA` (podmiana i przywrócenie), koordynowany
rollback wielu katalogów oraz replay nie zostają wykluczone przez
powtarzany odczyt. W kontekście nie jest trzymany **source custody
lease** ani trwały checkpoint/floor i nie ma blokowania rotacji/
revoke credential. Test-only source key to nie production key.

Przyszły `PRODUCTION_GO` będzie wymagał rzeczywistego,
instalacyjnie chronionego retained source logu, niezależnych
principalów i peer auth, end-to-end durable source mutation,
trusted floor / antirollback, wspólnego lock order source/CCA,
historical eligibility i zweryfikowanego outcome po crash/restart.
Przed produkcyjną autoryzacją należy osobno potwierdzić
niezależność cryptographic source/CCA authority i efektywne ACL.

Brak modyfikacji SQL, funkcji `complete_close`, signerów,
statusów `deployment/stage9_current_status.json`, frozen
kontraktów, prywatnych kluczy, SIGN/BIND, enrollment czy Stage 10.

## Wymagania sprzętowe

**Nie jest potrzebny dodatkowy komputer teraz.**
Eksperyment jest w pełni niezależny od PostgreSQL i działa jako
pytest w obecnym GitHub CI.

**Późniejszy krok z fizycznym źródłem i CCA wymaga natywnego
Linux/systemd oraz rzeczywistych principalów/ACL/protected pins**.
Jeden host może wystarczyć w ograniczonym modelu
`PRODUCTION_LOCAL`; nie zapewnia fizycznej niezależności rollback.
