# Stage 9: trwałe przechowywanie dokładnego Root Proof

## Zakres i wynik audytu

`postgresql_root_proof_retention.py` oraz
`postgresql_root_proof_retention_schema.py` tworzą osobny magazyn przygotowanych
podpisanych Root Proof. Każdy rekord ma stan `STAGED`: jest trwały i niezmienny,
ale nie oznacza zatwierdzonego wydania, `BOUND`, przyjęcia przez CHA ani
uprawnienia do wykonania Account Genesis.

`retain_prepared()` i `read_prepared()` zwracają `PreparedRootProof` ze stanem
`STAGED` i `authorizes_operations=False`. `require_issued()` zawsze odmawia;
także poprawny historyczny podpis i exact retry nie odblokowują tej operacji.

Audyt bazy `1b4eeef49fd8095010534305dec23c73b9d21d06` po PR #3107 wykazał:

- `PostgreSQLEntitlementRegistryProvider.compare_and_swap_bind()` wykonuje własną
  transakcję `SERIALIZABLE`; SQL `append_bind` dopisuje registry history i
  aktualizuje wskaźnik lineage.
- `PostgreSQLAuthenticatedIssuerHistory.append()` i `retain_attested_head()`
  otwierają własne połączenia i transakcje. Podpisany head nie jest zachowywany
  w transakcji BIND.
- `PostgreSQLRootProofIssuanceAuthority` agreguje odczytowe autorytety entitlement,
  requester i claimant; nie zawiera transakcyjnego magazynu proof ani historii.
- `PostgreSQLCheckpointAuthority` zatwierdza własny checkpoint i acceptance;
  nie łączy transakcji registry i history.
- Orkiestracja i recovery pozostają odczytowe i blokują mutacje. Wynik
  `EXACT_COMMITTED` classifiera nie dowodzi zachowania podpisanych bajtów.

Niezależne commity PostgreSQL nie spełniają
`8_ATOMICALLY_BIND_AND_PERSIST_IDENTICAL_SIGNED_PROOF_AND_HISTORY`. Wspólny commit
wymaga audytowanego właściciela transakcji, zgodnej topologii jednej bazy,
mechanizmu SQL obejmującego proof, BOUND i historię oraz linearizacji uprawnień
lifecycle. Obecne kwalifikacje odrębnych direct-login ról i ACL nie przyznają
takiej możliwości. Kompensacje i blokady procesu jej nie zastępują.

## Podpisane bajty i exact retry

Retencja zachowuje oryginalne kanoniczne bajty `signed_payload`, oryginalną
sygnaturę Ed25519 oraz dokładny kanoniczny envelope. Odczyt korzysta z zachowanych
wartości `BYTEA`; ponowna serializacja nie zastępuje oryginału. Digest służy
kontroli integralności i powiązaniu, a nie odtwarzaniu treści.

Weryfikacja kryptograficzna stosuje domenę zamrożonego kontraktu:

```text
ASCII("CRYPTOHUNTER_ACCOUNT_GENESIS_ROOT_PROOF_V1") || 0x00 || UTF-8 JCS(signed_payload)
```

Payload wiąże zamknięty zestaw pól kontraktu, w tym środowisko, trust domain,
product, action, account, operation, entitlement/generation, request fingerprint,
`issuance_attempt_id`, `root_proof_id` i issuer/key/version. Rekord zachowuje także
rolę semantyczną signera, namespace, credential identity, key version oraz
fingerprint oryginalnego materiału publicznego.

Exact retry rozwiązuje zachowany rekord dla tego samego trwałego attempt i
porównuje pełną niezmienną tożsamość oraz bajty. Obejmuje powiązanie
`INITIAL_BINDING` reference/digest, tożsamość podpisanego requestu oraz
requester/claimant identities i key versions. Konflikt payloadu, podpisu, klucza
lub identyfikatora jest odmową; retry nie tworzy nowego `root_proof_id`.
Unikalność i idempotencja należą do PostgreSQL, również między procesami.

Nieznany wynik zapisu po utracie odpowiedzi wymaga nowego kwalifikowanego odczytu
lub exact retry. Nie jest zwykłym brakiem rekordu. Sam brak retention nie dowodzi
`AUTHORITATIVELY_UNBOUND` i nie pozwala zastąpić attempt ani rozpocząć nowej emisji.

Backend nie tworzy `root_proof_id`, nie podpisuje i nie zachowuje zamiaru podpisania
ani trwałej rezerwacji `rpf_` przed pierwszym commitem `STAGED`. Po nieznanym
wyniku zapisu retry musi użyć tego samego oryginalnego kandydata. Utrata kandydata
lub wynik `read_prepared() == None` nie uprawniają do utworzenia nowego ID ani
ponownego podpisania. Brakująca granica podpisywania i rezerwacji jest częścią
blokady pełnego issuera.

## Granice zaufania i lifecycle

To wewnętrzny backend zaufanej kompozycji. Konstruktor przypina dokładny provider
weryfikacji i konfigurację magazynu; `retain_prepared()` i `read_prepared()` nie
wybierają providera. Backend nie jest jeszcze podłączony przez kompozycję
wdrożenia do `orchestrate_root_proof_issuance()`. Caller nie może traktować alternatywnego
providera ani historii jako capability. Retention nie generuje kluczy, nie
podpisuje ponownie i nie zapisuje do registry, authenticated history, checkpoint
ani CHA.

`ACTIVE` może być koniecznym warunkiem przygotowania nowego proof, ale zapis
`STAGED` nie ustanawia prawa do emisji. `VERIFY_ONLY` pozwala sprawdzać zachowane
historyczne podpisy bez uprawnienia do nowych podpisów. `REVOKED` nie może samym
poprawnym podpisem ustanowić wcześniejszego zatwierdzenia: potrzeba niezależnie
zachowanego issuance record, revocation boundary/reason, lifecycle history
i checkpointu uwierzytelnionego niezagrożonym trust root. Braki oznaczają odmowę.

Trwałość PostgreSQL i sprawdzanie podpisów nie wykrywają przywrócenia
wcześniejszego poprawnego stanu całego magazynu. Retention nie ma własnego
niezależnego checkpointu dowodzącego kierunku rollback; zatwierdzone recovery
wymaga powiązania z taką niezależną historią i acceptance. Techniczny odczyt
`STAGED` nigdy nie jest odzyskaniem zatwierdzonego proof.

Odczyt i exact retry najpierw sprawdzają integralność zachowanych bajtów. Negatywna
kontrola Ed25519 względem zachowanego klucza odrzuca nieważny podpis również podczas
awarii custody; pozytywny wynik nadal wymaga przypiętego `verify_historical()`.
Jednoznaczna blokada custody, oczekująca rotacja i rozpoznana niedostępność
infrastruktury oznaczają odmowę `UNAVAILABLE`. Niespójność bajtów, niezgodna
tożsamość, niebezpieczne ACL, `REVOKED` i niejednoznaczna odmowa pozostają
`CORRUPT`. Klasyfikacja dotyczy tylko dostępności weryfikacji; żaden z tych
stanów nie nadaje uprawnień, nie naprawia rekordu ani nie zastępuje exact retry.

## Bezpośredni SQL i zatrucie STAGED: otwarte wymaganie architektoniczne

Runtime principal ma `EXECUTE` na `retain_prepared(bytea,bytea,bytea,bytea,bytea,
bytea,bytea,bytea)` jako funkcji `SECURITY DEFINER`. Funkcja sprawdza kanoniczność,
zgodność przechowywanych tożsamości, stan `STAGED`, długość podpisu 64 bajtów oraz
trwałość PostgreSQL. Nie weryfikuje Ed25519; ta weryfikacja należy do przypiętego
`LocalRootProofSigningProvider` w backendzie Python. Kwalifikacja definicji SQL
i ACL nie dowodzi, że bezpośredni caller wykonał wcześniej tę weryfikację.

Posiadacz poświadczeń runtime, również kod wykonujący bezpośrednie SQL w tej
sesji, może zapisać spójny kanoniczny envelope i payload z niepoprawnym podpisem.
Commit zachowuje taki rekord trwale. Ponieważ bajty są niezmienne i unikalność
fencingu nie zależy od poprawności podpisu, rekord może zablokować późniejszy
poprawny kandydat: globalnie dla attempt i Root Proof ID, a w zakresie scope
także dla operation, entitlement/generation i account. Jest to ryzyko utraty
dostępności oraz zanieczyszczenia magazynu przez runtime spoza zaufanej ścieżki
Python. Samo ograniczenie dostępu do tabel nie eliminuje tej możliwości.

Test `test_direct_sql_invalid_signature_durably_fences_retry_but_never_becomes_proof`
wywołuje SQL jako rzeczywisty direct-login runtime. Potwierdza trwały zapis i
exact SQL retry niepoprawnego podpisu, konflikt poprawnego retry oraz odmowę
naprawy przez `UPDATE`, `DELETE` i `TRUNCATE`. Nowe połączenie backendu Python
odrzuca odczyt i retry jako `CORRUPT`; `require_issued()` również odmawia i nie
zwraca proof. Testy struktury SQL potwierdzają odmowę niekanonicznych bajtów,
niespójnej tożsamości, niewłaściwej długości podpisu i stanu `ISSUED`.
Wynik funkcji SQL nie jest `PreparedRootProof`, poprawnym proof ani autoryzacją.
Dotychczasowe kontrakty `authorizes_operations=False` oraz odmowa emisji dla
poprawnego `STAGED` nadal obowiązują.

W aktualnej granicy ACL brak niewielkiego zabezpieczenia dowodzącego poprawności
podpisu przed commitem: ta sama rola wywołuje SQL po weryfikacji Python i może
wywołać go bez niej. Dodatkowy digest, flaga „zweryfikowano” lub kontrola kształtu
nie zapewnią pochodzenia ani weryfikacji kryptograficznej. Usuwanie lub zastępowanie
niepoprawnego rekordu przez runtime naruszyłoby niezmienność i exact retry;
nie jest mechanizmem naprawy tej granicy zaufania.

Osobne wymaganie architektoniczne: przed uznaniem retencji za odporną na
bezpośrednie wywołania przez niezaufany runtime należy ustanowić audytowaną
granicę, która przed trwałym zajęciem tożsamości dowodzi podpisu, przypiętej
tożsamości custody i zgodnego lifecycle. Wymaga to przejrzanego autorytetu zapisu
oddzielonego od callerów SQL albo rzeczywistej kwalifikowanej weryfikacji
kryptograficznej po stronie PostgreSQL wraz z modelem zaufania do klucza i
lifecycle. Żadna z tych możliwości nie istnieje w obecnym schemacie i ACL;
ten PR nie dodaje nowego autorytetu, rozszerzenia PostgreSQL ani obejścia
transakcyjnego i nie deklaruje eliminacji ryzyka.

## Kwalifikacja i operacje zablokowane

Instalacja i runtime muszą kwalifikować rzeczywisty PostgreSQL 16, trwałość,
tożsamości ról, ACL, relacje, constraints oraz przejrzane definicje SQL.
Niedostępność bazy, błędny zakres, uszkodzenie bajtów lub historii i niepewny
wynik operacji zamykają ścieżkę. Raporty nie zawierają prywatnych kluczy,
DSN ani sekretów.

Negatywny test trwałości korzysta z drugiego, odizolowanego, jednorazowego
klastra PostgreSQL 16 z `fsync=off`; instalator i runtime muszą go odrzucić.
Pierwszy klaster testowy zachowuje trwałe ustawienia. Ten test nie zmienia
wdrożenia ani bramek sprzętowych.

Nie są udostępnione: atomic BIND + proof + history, autoryzowane proof recovery,
produkcja Root Proof, Account Genesis COMMITTED, transport i Stage 10.
`deployment/stage9_current_status.json` pozostaje bez zmian. Wyniki testów
i statusy dostępności należy wyprowadzić z celowanych testów na PostgreSQL 16
oraz istniejących bramek jakości; samo istnienie tego dokumentu ich nie ustanawia.
