# Stage 9 — S9-RPI-08: wersjonowane addendum zakresu transakcji A+C (kandydat do przeglądu)

**Status:** `SEPARATE_CONTRACT_REVIEW_PENDING` · **Produkt:** `PRODUCTION_NO_GO`  
**Źródło decyzji:** [zatwierdzony kierunek A+C](../decisions/stage9_root_proof_transaction_authority_owner_approval_v1.md)  
**Kontrakt maszynowy:** [stage9_root_proof_transaction_scope_addendum_v1.json](stage9_root_proof_transaction_scope_addendum_v1.json)

## Precedence i formalne ograniczenie

Zgoda właściciela projektu z 2026-10-18 obejmuje **kierunek A+C**, ale nie kwalifikuje wdrożenia. Niniejszy oddzielny, wersjonowany dokument przygotowuje jego formalną integrację. Dopóki nie zostanie niezależnie przejrzany i zaakceptowany jako contract addendum, zachowuje status **review candidate**; nie zastępuje samodzielnie frozen kontraktów i nie przyznaje runtime authority.

Jedyny rozstrzygany konflikt interpretacyjny dotyczy wymogu separacji **własności transakcji**: obowiązuje on między niezależnymi autorytetami, a **nie** pomiędzy podrzędnymi strukturami przechowywania jednej decyzji issuera. Historyczne kontrakty pozostają normatywne we wszystkich pozostałych aspektach, w tym kolejności BIND, provenance, ownerach, retry, bramkach gotowości i fizycznej semantyce CHA/freshness.

## A — atomowy finalny write set

Jeden issuer-owned coordinator posiada od początku do `COMMIT/ROLLBACK` **jedną fizyczną transakcję PostgreSQL 16 SERIALIZABLE**. Jej niepodzielny finalny zapis obejmuje jednocześnie:

1. Dokładny entitlement CAS BIND z historią rejestru.
2. Powiązanie **oryginalnych** signed payload/signature/envelope bytes z wygrywającą próbą przez chronioną immutable retencję/referencję w tej samej bazie.
3. Uwierzytelnioną historię issuera, exact predecessor CAS i signed head.
4. Immutable marker decyzji z dokładnym winner/attempt/operation oraz byte-identical proof/history linkage.

Requester/claimant, Root signer, History signer i trust activation wymagają weryfikacji oraz utrzymania lifecycle/trust fences do finalnego `COMMIT`. `RESERVED` i `STAGED` są wyłącznie przygotowaniem, nie wydaniem Root Proof. Żadnego resigne/remint po nieznanym wyniku COMMIT.

## C — separacja kompetencji

Schema/role/credential/namespace/lifecycle pozostają niezależne. Podległe owner-owned SQL helpers mogą uczestniczyć w jednym issuer-owned finalnym COMMIT bez transferu semantic ownership, uprawnień admin, arbitralnych DML/DDL czy bezpośrednich grantów EXECUTE dla issuera.

Poza tą transakcją pozostają CHA AttemptStore, FreshnessAuthority, CheckpointAuthority, AccountGenesis account commit, credential lifecycle administration i custody mutation. Issuer nie może pisać do CHA AttemptStore, samowolnie przesuwać freshness ani publikować konta; nie używa 2PC lub kompensacji do symulowania atomowości swojego write set.

## Machine parity i niezmienione bezpieczeństwo

Testy [test_stage9_root_proof_transaction_scope_addendum_v1.py](../../../tests/architecture/test_stage9_root_proof_transaction_scope_addendum_v1.py):

- porównują wersjonowany dokument z zamkniętym zestawem dozwolonych pól, dokładnym write set i zakazami poszerzenia scope;
- weryfikują 4 historyczne canonical JSON poprzez przypięte Git blob IDs oraz rzeczywiste JSON Pointer; brak automatycznego repinningu;
- wykrywają próbę otwarcia SIGN/BIND, zmiany ownera, dopisania drugiego COMMIT, 2PC, direct grants lub obejścia freshness;
- potwierdzają niezmienione `deployment/stage9_current_status.json`, w którym Stage 9 nadal jest `IN_PROGRESS`, issuance nie jest uruchomione, a produkcyjna gotowość pozostaje `false`.

Żaden frozen JSON ani jego historyczna Markdown projection nie jest edytowany; nie ma SQL migration, nowych grants, production ingress, key provisioning, runtime policy loader, enrollment ani uruchomienia Stage 10.

## Kolejna zależność i sprzęt

Po niezależnym review kontraktu pozostają rzeczywiste production source/CCA, qualified committed-close reader, historyczne credential eligibility, independent lock order, kontrolowana migracja SQL i fizyczna kwalifikacja. **Nie wolno** uznać merge addendum za `PRODUCTION_GO`.

**HARDWARE_DEPENDENCY:** do tej zmiany wystarczy Windows + Python/Git i GitHub-hosted CI; drugi komputer nie jest wymagany. Fizyczne wdrożenie signer/source/CCA będzie wymagało natywnego hosta Linux z systemd, rzeczywistymi UID/GID/ACL, chronionymi kluczami i restart/power-loss evidence. Jeden właściwie przygotowany host może wystarczyć dla profilu lokalnego; niezależny rollback/security domain będzie osobną decyzją dla silniejszego profilu.
