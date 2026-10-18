# S9-RPI-04 — decyzja właściciela projektu: A + C (v1)

**Data decyzji:** 2026-10-18  
**Status:** `OWNER_ARCHITECTURE_DIRECTION_APPROVED`  
**Status wdrożenia:** `CONTRACT_ADDENDUM_INTEGRATION_PENDING` / `PRODUCTION_NO_GO`

## Pochodzenie i zakres zatwierdzenia

Właściciel projektu jednoznacznie potwierdził w rozmowie 2026-10-18 zgodę na rekomendację **A + C** w kształcie wyjaśnionym przed potwierdzeniem: jeden fizyczny, atomowy COMMIT dla podległego persistence jednej decyzji Independent RootProofIssuer oraz wąskie, wersjonowane addendum doprecyzowujące separację transakcji między niezależnymi autorytetami.

To zapis otrzymanej decyzji architektonicznej, **nie** kryptograficznie podpisana zewnętrzna atestacja właściciela, token uprawniający runtime, zgoda na produkcyjną migrację ani kwalifikacja wdrożenia. Ślad przeglądu oraz przyjęcia tego dokumentu pozostaje w historii repozytorium/PR. Zmiana tego dokumentu nie może automatycznie zmienić stanu gotowości.

Podstawa rozstrzygnięcia:
- [ADR S9-RPI-04](../stage9_root_proof_transaction_authority_reconciliation.md);
- [propozycja addendum C z PR #3155](../proposals/stage9_root_proof_transaction_authority_addendum.md);
- [independent issuer — zamrożony kontrakt](../cryptohunter_product_architecture/m05_account_genesis_independent_root_proof_issuer_contract.json);
- [production substrate — zamrożony kontrakt](../cryptohunter_product_architecture/m05_account_genesis_root_proof_issuer_production_substrate_selection_contract.json);
- [implementation readiness — zamrożony kontrakt](../cryptohunter_product_architecture/m05_account_genesis_root_proof_issuer_implementation_readiness_contract.json).

## Zatwierdzona zasada A + C

**A — jedna finalna transakcja:** zaufany, issuer-owned coordinator posiada od początku do końca jedną fizyczną, zapisywalną transakcję PostgreSQL 16 `SERIALIZABLE` i jej `COMMIT`. Jej **niepodzielny** write set obejmuje:

1. Entitlement CAS, dokładny BIND i odpowiednią historię rejestru.
2. Trwałe powiązanie identycznych oryginalnych, podpisanych bajtów Root Proof — payload, signature i envelope — z wygrywającą próbą; immutable `RESERVED` i `STAGED` są wcześniejszym przygotowaniem, nie decyzją emisji.
3. Uwierzytelniony issuer history record, exact predecessor CAS i podpisany head.
4. Niezmienny decision marker, exact winner/attempt/operation identity oraz byte-identical retained proof/history linkage.

Przed finalnym COMMIT ta sama transakcja kwalifikuje i utrzymuje odpowiednie blokady / fences `ACTIVE` dla requester/claimant, root/history signer i trust activation, według zatwierdzonego lock order. Sama lokalna próbka stanu klucza bez fence utrzymanego do COMMIT nie wystarcza.

**C — ograniczenie zakresu separacji transakcji:** odrębność właścicieli semantycznych, schematów, principals, credentials, namespaces, API, uprawnień i lifecycle obowiązuje nadal. Wewnętrzne owner-owned helpers entitlement, retention i history mogą uczestniczyć w **tej samej** fizycznej transakcji finalnej decyzji issuera, nie stając się jego własnością semantyczną. Issuer writer nie otrzymuje dowolnego DML/DDL, owner membership, uprawnień administracyjnych ani bezpośredniego EXECUTE na podległych helperach.

**Odrębne transakcje i kompetencje** zachowują CHA / CHA AttemptStore, FreshnessAuthority, CheckpointAuthority, AccountGenesis oraz administracja credential lifecycle i mutacje custody. Nie istnieje jeden globalny COMMIT dla issuer + CHA + freshness + checkpoint + account. Issuer nie otrzymuje prawa do ich mutacji.

## Rozstrzygnięcia D1–D5

| Decyzja | Wynik właściciela projektu |
| --- | --- |
| D1 | **TAK** — osobna własność transakcyjna między niezależnymi autorytetami; ograniczony wyjątek dla podległego persistence *jednej* decyzji issuera. |
| D2 | **TAK** — jeden issuer-owned coordinator i ściśle ograniczone owner-owned helpers jako realizacja atomowego kroku BIND + exact proof + history. |
| D3 | **TAK WARUNKOWO** — oryginalne immutable STAGED bytes i chroniony atomowy marker/linkage muszą rzeczywiście zachowywać oraz ponownie weryfikować identyczne bajty; nie wolno użyć samego hasha lub ponownie podpisywać. |
| D4 | **TAK** — oddzielne wersjonowane addendum, jawne parity / kontraktowe testy i przyszła kontrolowana migracja; bez modyfikacji historycznych frozen JSON lub ich projekcji. |
| D5 | **TAK** — nie implementować niezależnych COMMIT ani 2PC dla jednego issuer write set; wszystkie inne bramki produkcyjne pozostają w mocy. |

To rozstrzygnięcie zastępuje **oczekiwanie na decyzję właściciela o A + C**, ale nie znosi innych przesłanek `NOT_READY`. Nie traktować automatycznie tego dokumentu ani późniejszego merge jako dowodu `PRODUCTION_QUALIFIED`.

## Obowiązkowe działania implementacyjne po ratyfikacji

1. Wprowadzić osobno reviewed, maszynowo weryfikowane **versioned contract addendum** z zachowaniem historical JSON freeze i jawnego rozstrzygnięcia jego precedence/parity. Nie wystarczy zmienić statusu w ADR.
2. Dopiero w wyraźnie autoryzowanym zakresie przygotować niezależny production semantic issuer ingress, zamknięte uprawnienia SQL, finalne fenced `SERIALIZABLE` i kwalifikację konkretnych principal/instance/ACL/source.
3. Domknąć produkcyjną niezależną custody completion authority i rzeczywisty private-control proof, cross-role isolation, lifecycle oraz protected trust pin. Odczyt journalu nie jest sam w sobie autoryzacją completion.
4. Przetestować end-to-end crash/restart/rollback/COMMIT-response-loss, dwa konkurujące legalne attempts, exact retry, utratę odpowiedzi, rollback/restore, history/checkpoint verification i byte-identical recovery.
5. Przeprowadzić oddzielny przegląd GO/NO-GO na rzeczywistym deployment — samo zatwierdzenie A + C nie daje tego wyniku.

## Twarde zakazy

Nie zmieniać w tym kroku canonical frozen JSON, `deployment/stage9_current_status.json`, workflow, progów testów, ani produkcyjnych gate. Nie odblokowywać `ProductionRootLifecycleBoundary.commit()`, `publish_administrator_completion()`, SQL `complete_close`, produkcyjnego SIGN/BIND, enrollment, activation, nowego issuer writer, account commit ani Stage 10. Nie uznawać żadnych flag `approved`, testowych tokenów, DTO i samych dokumentów za runtime capability.

**Na dzień decyzji:** `stage_9=IN_PROGRESS`; `semantic_root_proof_issuer_runtime=NOT_IMPLEMENTED / BLOCKED`; `root_proof=NOT_ISSUED`; `production_provisioning_ready=false`. `PRODUCTION_NO_GO` obowiązuje do osobnej kwalifikacji.
