# Stage 9 — S9-RPI-16: testowy source-original floor witness

**Status:** `TEST_ONLY_NON_AUTHORIZING`; **produkcja:**
`PRODUCTION_NO_GO`. Kontynuacja S9-RPI-15 (PR #3171),
po poprawce source-policy snapshot PR #3170.

## Cel i rzeczywisty zakres

S9-RPI-15 wiązał retained source event z oryginalnymi bajtami,
plikami i uprawnieniami istniejącego izolowanego workera Linux.
Nie dostarczał jednak osobnego checkpointu, wobec którego dałoby
się porównać te pliki **po odtworzeniu procesu**.

[Test S9-RPI-16](../../tests/security/test_retained_source_test_floor_witness.py)
wykonuje kwalifikacyjny, wyłącznie syntetyczny eksperyment:

1. W istniejącym jednorazowym workerze source wykonuje realną
   TEST custody mutation i zapis `events/original` przez istniejący
   `fsync` + atomowy rename + directory `fsync`.
2. Zanim utrwali witness, wymaga pomyślnego, **nieautoryzującego**
   `inspect_synthetic_retained_source_original()` z S9-RPI-14/15.
3. W istniejącym, pustym, prywatnym katalogu **poza katalogiem
   źródła** zapisuje dokładnie jeden niezmienny plik
   `source-original-floor`, zawierający installation/operation/
   execution ID, `source_log_position` (namespace, sequence,
   predecessor digest) oraz digest oryginalnego envelope i policy.
4. Reużywa niskopoziomowy TEST-only Linux `_Directory` z
   [logu CCA S9-RPI-07](../../bot_core/test_only_cca_decision_log.py):
   `O_NOFOLLOW`, przypięte inode katalogów i pliku, 0700/0400,
   closed names, ograniczony odczyt, `fsync` pliku i katalogu.
   Nie kopiuje ani nie dodaje nowego produkcyjnego logu.
5. `_FloorGuardedReader` jest **lokalnym adapterem w testach**
   i sprawdza utrwalony witness podczas każdego exact source lookup
   i przy każdym odczycie policy. Dopiero później istniejący
   S9-RPI-14 wykonuje sprawdzenia retained bytes, binding i Ed25519.
6. Restart istniejącego workera zwraca identyczny oryginał bez
   ponownego podpisania, a witness nie zostaje nadpisany.

Negatywne testy wymagają odmowy po utracie, ucięciu, zmianie
inode lub symlinkowaniu floor; po osieroconym `.stage`;
przy ponownym, nieuprawnionym provisioningu floor oraz przy
zmianie source envelope/policy, **nawet gdy nowy adapter zostanie
dopiero później przypięty do zmodyfikowanego inode**.

Zapis floor jest jawnym jednorazowym krokiem fixture;
runtime nie może tworzyć brakującego lub zastępczego floor.
Niezależny odczyt nie traktuje `NOT_FOUND` jako dowodu, że
można podpisać nowe zdarzenie.

## Dokładne ograniczenia bezpieczeństwa

To nadal **nie jest trusted antirollback checkpoint**. Źródło,
testowy witness oraz czytnik mają ten sam UID i ten sam host/rollback
domain. Ścieżka i pin floor oraz policy powstają z fixture i nie są
zewnętrznie chronionym root-owned installation manifestem.
Posiadacz tego UID może ingerować we wszystkie dane, a administrator
może spójnie wycofać obie strony i pin. Prosty, jednorazowy floor
nie dowodzi monotoniczności całej historii ani odporności na
`ABA`, zanik zasilania, storage lying about `fsync`,
replay historyczny, kompromitację source key ani fizyczną
autentyczność operacji custody.

Nie ma production source ledger, niezależnego live trusted floor,
zaufanego recovery journal, cross-principal OS access audit,
kwalifikowanego source custody lease, credential eligibility lease,
end-to-end lock order source/CCA ani niezależnej trwałości decyzji.
W szczególności test nie potwierdza, że wykonane wcześniej
`fsync` zostały faktycznie utrwalone na nieulotnym nośniku.

Wszystkie pozytywne przypadki pozostają
`NOT_QUALIFIED`, `BLOCKED`, `provenance_qualified=False`,
`durability_attested=False`. Produkcyjny Root Proof issuer
i completion publication nie otrzymują żadnego uprawnienia.
Bez zmian pozostają SQL, grants, frozen contracts, podpisy
produkcyjne, manifesty, status Stage 9 oraz Stage 10.

## Walidacja i sprzęt

Test Linux jest jawnie włączony do istniejącego
`PostgreSQL atomic Root Proof commit`, Ruff Format i strict
CI inventory. Testy Python na tym etapie mają potwierdzić
integralność witness i reguły fail-closed; nie stanowią
kwalifikacji instalacji lub awarii zasilania.

**Dodatkowy komputer teraz: NIE.** GitHub Linux runner
wystarczy do S9-RPI-16. W późniejszym kroku native Linux/systemd
i prawdziwe UID/GID/ACL oraz chroniony pin są konieczne
dla fizycznej kwalifikacji. Odseparowany rollback domain na
innym nośniku lub hoście wymaga osobnej decyzji i wdrożenia.
