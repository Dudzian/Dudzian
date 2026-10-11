# Stage 9: trwały lokalny autorytet checkpointu i potwierdzenia V2

Ten moduł uzupełnia istniejącą historię PostgreSQL o trwały
`CheckpointAuthorityProvider`. Nie implementuje Semantic RootProofIssuer,
emisji Root Proof, Entitlement BIND, transportu produkcyjnego ani Stage 10.
Zamrożone modele, domeny kryptograficzne i kontrakty M0.5 pozostają podstawą
weryfikacji. Windows Stage 9 pozostaje `10/15 DONE`.

## Źródła kontraktu i brakujące elementy

Analiza obejmuje:

- `bot_core/authenticated_issuer_history.py`;
- `bot_core/postgresql_authenticated_issuer_history.py`;
- [wykonywalny kontrakt historii i checkpointu](authenticated_issuer_history_checkpoint_contract.md);
- [trwałą historię PostgreSQL](issuer_history_production_local.md);
- [niezależny RootProofIssuer M0.5](cryptohunter_product_architecture/m05_account_genesis_independent_root_proof_issuer_contract.json);
- [gotowość implementacji M0.5](cryptohunter_product_architecture/m05_account_genesis_root_proof_issuer_implementation_readiness_contract.json);
- [wybór podłoża produkcyjnego M0.5](cryptohunter_product_architecture/m05_account_genesis_root_proof_issuer_production_substrate_selection_contract.json);
- [trwałość fizyczną i atomowość awarii M0.5](cryptohunter_product_architecture/m05_account_genesis_physical_persistence_crash_atomicity_contract.json).

`LocalCheckpointProvider` jest referencyjnym autorytetem CAS w pamięci procesu.
Jego tożsamość instancji, checkpoint i łańcuch
`HistoricalHeadAcceptanceEvidence` nie przeżywają restartu. Istniejąca historia
PostgreSQL zapewnia już trwałość rekordów i podpisanych nagłówków; nie wymaga
przebudowy w celu dodania checkpointu.

Sam rekord V2 acceptance wiąże pełną tożsamość streamu i signera, rewizję,
sekwencję historii, digest, lifecycle oraz predecessor. Nie zawiera podpisu.
Dlatego trwała koperta zaakceptowanej rewizji zachowuje również dokładny
`AttestedHistoryHead`, jego canonical attestation bytes i podpis. Dodatkowa
koperta fizyczna nie zmienia zamrożonego digestu ani interpretacji V2.

## Granica autorytetu

`PostgreSQLCheckpointAuthority` jest osobnym autorytetem lokalnej trwałości.
Zaufany composition root wiąże go z kanoniczną instancją historii i wybranymi
portami podpisu historii. Zgodność strukturalnego Python Protocol, nazwa
providera oraz caller-selected capability report nie ustanawiają pochodzenia
autorytetu. Również dokładny typ implementacji nie zastępuje zaufanego
composition root.

Nagłówek przekazany do operacji jest danymi do ponownej weryfikacji.
`VerifiedHistoryHead`, `LocalCheckpoint`, publiczny acceptance report, token
lub kopia obiektu nie nadają uprawnień. Autorytet sam sprawdza historię,
podpis, retained credential identity i lifecycle. Integracja dopuszcza jawnie
dokładne implementacje `LocalCheckpointProvider` oraz
`PostgreSQLCheckpointAuthority`; nie dopuszcza subclass ani dowolnego obiektu
z podobnymi metodami.

Zakres checkpointu obejmuje exact stream tuple: `stream_id`,
`issuer_authority_identity`, `security_profile`, `environment`, `trust_domain`,
`product_scope`, `security_epoch`. Signer jest identyfikowany przez pełny
`CredentialRoleIdentity`, w tym role, credential identity, provider namespace,
key handle/version, custody lifecycle namespace i key material identity.
Public key pochodzi z zaufanego portu podpisu historii, a jego fingerprint
musi odpowiadać zadeklarowanemu key material identity. Baza checkpointu nie
przechowuje prywatnych kluczy.

Runtime LOGIN i jego `EXECUTE` należą do zaufanego checkpoint providera.
Nie wolno udostępniać runtime DSN ani raw SQL `commit_checkpoint` użytkownikowi
wywołującemu lub signerowi. SQL sprawdza strukturę i integralność koperty, lecz
nie odpytuje custody i nie ustanawia samodzielnie dowodu ACTIVE lifecycle.
Posiadacz skompromitowanych runtime DB credentials może przesłać strukturalnie
poprawną ACTIVE acceptance ze starym, prawdziwie podpisanym nagłówkiem klucza
obecnie REVOKED. Ochrona API przed caller-created obiektem nie zapobiega temu
obejściu zaufanego runtime principal. W takim scenariuszu historyczna akceptacja
REVOKED nie ma gwarantowanej niezależnej podstawy. Silniejsza integralność po
kompromitacji runtime credentials wymaga dodatkowego komponentu i pozostaje
`BLOCKED`.

## Stan, CAS i transakcje

Offline provisioning ustanawia trwałą tożsamość instancji checkpointu i exact
stream. Constructor ani restart nie tworzą zastępczej instancji, brakującej
historii lub brakującego evidence. Odczyt po restarcie waliduje odzyskany stan.

Checkpoint wskazuje ostatnią zatwierdzoną rewizję. Każda nowa rewizja atomowo
publikuje successor checkpoint i immutable acceptance wraz z exact podpisanym
nagłówkiem. Łańcuch acceptance zachowuje contiguous checkpoint revisions,
predecessor acceptance digest oraz wszystkie wcześniejsze zaakceptowane
tożsamości signerów. Historia może zostać zaakceptowana po kilku appendach;
sekwencje zaakceptowanych nagłówków muszą rosnąć, lecz nie muszą być sąsiednie.

Przed CAS oraz przed authoritative read działa pełna walidacja retained state:
canonical bytes i digests, stream i authority identity, continuity revisions,
lifecycle przy akceptacji, powiązania nagłówków, cardinality oraz zgodność
ostatniego acceptance z bieżącym checkpointem. Uszkodzony środkowy rekord,
brakujące evidence i checkpoint-only rollback blokują również exact replay.

Weryfikacja historii i signera oraz stabilna próbka lifecycle poprzedzają
checkpoint row lock. Nie należy trzymać blokad historii, checkpointu i custody
jednocześnie. Próbka ACTIVE wyznacza eligibility zgodnie z istniejącym
kontraktem; późniejszy append może pozostawić poprawnie zatwierdzony checkpoint
w stanie `STALE`. Późniejsza revocation nie usuwa wcześniej zatwierdzonego
exact-head acceptance. Checkpoint row lock i expected-revision CAS wybierają
jeden zwycięski commit między procesami.

Ta kolejność opisuje nową ścieżkę produkcyjną, której źródłem jest dokładna
instancja `PostgreSQLAuthenticatedIssuerHistory`. Reference oracle zachowuje
dotychczasowy process-local `RLock` podczas weryfikacji podpisu i odczytu
historycznego potwierdzenia; jego istniejące regresje wymagają coherent
snapshot. Nie jest to deklaracja usunięcia wszystkich zagnieżdżeń w oracle ani
kwalifikacja reference implementacji jako trwałego autorytetu. Produkcyjne
odczyty historii używają read-only snapshot bez history row lock; zapis
checkpointu nie wywołuje custody ani historii pod checkpoint row lock.

Niższa sekwencja jest rollbackiem. Ta sama sekwencja z innym digestem lub
signerem jest split-brain. Same-head replay nie tworzy drugiego acceptance.
Reference oracle sprawdza expected revision przed decyzją replay; przy utracie
odpowiedzi rozstrzygnięcie opiera się na authoritative reread i retained
exact-operation state. Błąd połączenia lub brak odpowiedzi nie dowodzą braku
commitu. Nieznany wynik zapisu pozostaje fail-closed do dokładnego
rozstrzygnięcia, bez automatycznej emisji ani blind retry.
Reconciliation zachowuje `UNAVAILABLE` dla niedostępnego checkpoint authority
i `INDETERMINATE` dla nieznanego wyniku, zamiast traktować je jako brak rekordu
lub dowód uszkodzenia danych.

`resolve_advance(expected_revision=..., head=...)` odczytuje dokładną retained
rewizję powstałą z pierwotnej operacji. Zwraca wcześniej zatwierdzony checkpoint
bez nowego CAS, także gdy bieżący checkpoint zdążył przejść dalej.
`retry_advance` wykonuje nowe `advance` tylko wtedy, gdy authoritative read nie
odnajdzie pierwotnej rewizji. Stare expected revision nie nadaje uprawnienia do
innego successor. Conflicting head dla tej rewizji blokuje recovery.

Resolver rozstrzyga operacje, które zapisały nową rewizję. No-op replay nie
tworzy osobnego journal entry. Jeżeli późniejszy successor wykorzysta jego
expected revision, wcześniejszy no-op nie jest reinterpretowany jako tamten
commit: recovery kończy się konfliktem. Zachowuje to referencyjne reguły CAS;
nie stanowi kompletnego dziennika wyników wszystkich wywołań API.

`authenticated_checkpoint_at(sequence)` odczytuje historyczny checkpoint po
weryfikacji całego retained authority state. Port `advance_exact_successor`
traktuje przekazany `LocalCheckpoint` jako żądanie: sam odczytuje retained
attestation z kanonicznej historii, porównuje exact successor i przeprowadza
normalną weryfikację oraz CAS. Żaden z tych portów nie emituje Root Proof.

## Lifecycle i podpisy historyczne

Nowe acceptance wymaga ACTIVE credential i stabilnego lifecycle generation.
`VERIFY_ONLY` pozwala weryfikować zgodne nagłówki, lecz nie tworzyć nowego
acceptance. `REVOKED` nie tworzy nowego zaufania podpisem.

Historyczne REVOKED verification wymaga poprawnego podpisu, kompletnej
zweryfikowanej retained history, dokładnego rekordu oraz matching acceptance
z kanonicznego checkpoint authority. Acceptance musi pochodzić z ACTIVE
generation wcześniejszej niż aktualna REVOKED generation. Potwierdzenie
sekwencji N nie potwierdza N−1 ani N+1. Brak wymaganej historycznej podstawy
oznacza `HistoricalRevokedSignatureVerificationUnavailable`, a nie zgodę na
odbudowanie evidence z caller data lub bieżącego credential stanu.

## SQL, role i trwałość

Instalacja jest odrębną uprzywilejowaną operacją offline. Owner, administrator
i runtime mają oddzielne role; runtime nie otrzymuje bezpośredniego DML ani
uprawnień do provisioningu. PUBLIC nie otrzymuje dostępu do autorytetu.
Przeglądane `SECURITY DEFINER` functions wiążą direct session principal i
`search_path=pg_catalog`.

Kwalifikacja sprawdza schema identity i wersję, zapisane role OIDs i atrybuty,
memberships, ownership, exact ACL allowlists, źródła i security metadata SQL,
inventory functions oraz fizyczne columns, constraints i indexes. Zmieniony
SQL, ACL, schema lub nieznana instalacja powodują fail-closed. Sam zapisany
fingerprint instalacji nie zastępuje porównania z przeglądaną lokalną definicją.

Trwałość wymaga PostgreSQL co najmniej 16, `fsync=on` i
`synchronous_commit=on`. Atomowość checkpointu i evidence wynika z jednego
commitu PostgreSQL. Rzeczywista trwałość zależy również od serwera i nośnika;
kwalifikacja ustawień nie dowodzi poprawności infrastruktury przechowywania.
Recovery zachowuje cały autorytet wraz z jego tożsamością i historią rewizji.
Nie ma implicit migration ani odbudowy brakujących rekordów z cache.

Instalator `bot_core.postgresql_checkpoint_schema` ustanawia schema identity
`cryptohunter.issuer_history_checkpoint.postgresql`, wersję 1 i storage family
`POSTGRESQL_ISSUER_HISTORY_CHECKPOINT_V2`. Tabele `metadata`, `checkpoints` oraz
`entries` zachowują koperty w exact canonical bytes. Operator tworzy nową
instalację przez `provision_postgresql_checkpoint`, a osobny authenticated
administrator nadaje checkpoint identity i stream przez
`provision_postgresql_checkpoint_stream`. Istniejąca instalacja nie jest
nadpisywana.

Composition root tworzy `PostgreSQLCheckpointAuthority` z runtime connection,
`schema`, `checkpoint_id`, exact `stream`, kanonicznym `history` i niepustym
tuple `verification_authorities`. Tuple zachowuje także wcześniej użyte
signing ports, potrzebne do uwierzytelnienia starszych rewizji. Po restarcie
te same zaufane źródła i retained credential identities pozostają wymagane;
usunięty historyczny port lub public key oznacza fail-closed.

## Granice i kwalifikacje pozostające BLOCKED

Lokalna trwałość i łańcuch digestów wykrywają niespójności, cofnięcie checkpoint
pointer względem retained revisions oraz cofnięcie historii poniżej zachowanego
zaakceptowanego nagłówka. Poprawny starszy checkpoint przy nowszej historii
może oznaczać dozwolony stan `STALE`; sama różnica sekwencji nie dowodzi
kierunku restore. Moduł nie wykrywa skoordynowanego restore całej maszyny lub
wszystkich lokalnych authority stores do starszego, wewnętrznie poprawnego
stanu. PostgreSQL i ten sam host nie stanowią niezależnego rollback domain.

Instancja działająca w procesie zachowuje zaobserwowany high-water mark i wykrywa
cofnięcie całego checkpoint chain do wcześniej obserwowanego valid prefix.
High-water mark procesu nie jest trwałym niezależnym witness. Nowa instancja
po przywróceniu starszego, wewnętrznie poprawnego checkpoint prefix nie może
ustalić tego cofnięcia bez witness poza przywróconym stanem.

Kompromitacja runtime DB credentials, uprzywilejowana zmiana wszystkich
retained danych, obejście runtime ACL przez owner/superuser oraz host-admin key
extraction pozostają poza lokalną gwarancją. Kwalifikacja nie jest ciągłym
dowodem braku uprzywilejowanych zmian
między sprawdzeniem katalogu a użyciem. Po niejednoznacznym restore kierunek
wymaga zaufanego reconciliation; timestamp ani wybór większej rewizji nie są
arbitrażem.

Niezależny anti-rollback i SERVER_READY pozostają `BLOCKED`; wymagają odrębnego
rollback oraz administration/security domain i silniejszego custody.
`ROOT_PROOF_ISSUER_IMPLEMENTED`, `PRODUCTION_LOCAL_RUNTIME_AVAILABLE` i
`STAGE9_READY` nie są promowane. Trwały checkpoint nie ustanawia dodatniego
`AUTHORITATIVELY_UNBOUND`, nie zastępuje frozen reconciliation evidence i nie
tworzy distributed ACID między registry, history, checkpoint i CHA.
