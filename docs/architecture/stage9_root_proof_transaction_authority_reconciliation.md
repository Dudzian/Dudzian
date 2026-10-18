# ADR — S9-RPI-04: Root Proof Transaction Authority Reconciliation

> **Aktualizacja 2026-10-18:** właściciel projektu zatwierdził kierunek **A + C** (D1–D5, z warunkami D3). Zob. [decyzja właściciela v1](decisions/stage9_root_proof_transaction_authority_owner_approval_v1.md). Poniższy status `CONTRACT_RECONCILIATION_PENDING_APPROVAL` dokumentuje stan z chwili opracowania ADR; decyzja architektoniczna została podjęta, lecz versioned contract integration, produkcyjna kwalifikacja i wszystkie blokady `NO-GO` pozostają otwarte.

## Decyzja i status

**CONTRACT_RECONCILIATION_PENDING_APPROVAL; NO-GO.** Rekomendujemy wariant
**A**, warunkowo po osobnym zatwierdzeniu propozycji **C**: jeden fizyczny
COMMIT finalnej decyzji issuera, przy rozłącznych właścicielach semantycznych,
schematach, principalach, credentialach i lifecycle. Literalnego wymagania
odrębnych transakcji nie uznajemy za spełnione przez samą separację SQL ownerów.
To rekomendacja do zatwierdzenia, nie nowa obowiązująca interpretacja freeze.

| Wynik | Znaczenie | Wynik tego ADR |
| --- | --- | --- |
| `TECHNICALLY_FEASIBLE` | Istnieje techniczny mechanizm o określonych, sprawdzalnych gwarancjach. | A: wykonalny atomowy persistence write set, wykonywalny w izolowanym teście; nie cały produkcyjny pipeline. B: sekwencja lokalnych commitów nie zapewnia wymaganych gwarancji; 2PC wymaga nowego protokołu. |
| `CONTRACTUALLY_APPROVED` | Właściciel kontraktu niezależnie zatwierdził dokładny zakres i wersję zmiany. | **Nie**; propozycja C oczekuje decyzji. ADR, test, DTO ani flaga nie udzielają zgody. |
| `PRODUCTION_QUALIFIED` | Rzeczywista instalacja przeszła niezależne bramki wdrożenia, custody, recovery i bezpieczeństwa. | **Nie**; brak dowodów i produkcyjnego semantic issuer. |

Żaden z tych stanów nie implikuje pozostałych. Zachowujemy
`ROOT_PROOF_ISSUANCE_AVAILABLE=False`, `PRODUCTION_READY=False`,
`ATOMIC_ISSUER_READY=NOT_READY` oraz
`semantic_root_proof_issuer_runtime=NOT_IMPLEMENTED / BLOCKED`.
Root Proof issuance, produkcyjne SIGN/BIND, enrollment, aktywacja kluczy
i Stage 10 pozostają zablokowane. Nie dodajemy publicznego write entrypointu
ani modelu wydającego capability na podstawie klasyfikacji.

## Podstawa i granice audytu

Audyt źródeł wykonano na `main` po merge #3151–#3154:
`7ed0e90321c9e24d6770f968e5bd7026bfa73cb7`. Obejmuje osiem modułów
wskazanych w zadaniu oraz powiązane registry, intent, retention, history,
custody, credentials, CHA, freshness i checkpoint. Źródła funkcji SQL,
instalację ról i grantów, kwalifikatory oraz rzeczywiste punkty COMMIT
sprawdzono bezpośrednio; wcześniejszy opis architektury nie jest dowodem
bieżącego zachowania.

Codebase Memory MCP i jego CLI nie są dostępne w tej sesji. Użyto `rg`,
bezpośredniego odczytu źródeł i celowanych testów; nie deklarujemy kompletności
nieobecnego indeksu. Wyniki PostgreSQL dotyczą jednorazowych izolowanych
klastrów, a nie wdrożenia produkcyjnego.

Dokumenty porównawcze:
[Atomic Commit V1](stage9_atomic_root_proof_commit_authority_v1.md),
[qualified atomic boundary](stage9_qualified_atomic_root_proof_boundary.md),
[production lifecycle isolation](stage9_production_root_proof_lifecycle_isolation.md)
i [custody completion verifier](stage9_production_custody_completion_verifier.md).
Pierwsza kwalifikowana granica opisywała stan po #3110, przed instalacją
testowego coordinatora V1/V2; jej historyczny brak coordinatora nie jest
bieżącym stanem kodu. #3154 dodał odczyt durable completion, nie niezależną
authority publikacji completion ani produkcyjny COMMIT.

## Dokładne pola zamrożonych kontraktów

Poniższe skróty wskazują canonical JSON; ścieżki są JSON Pointer, indeksy
tablic zaczynają się od zera. Sąsiadujące pliki `.md` deklarują, że są
deterministycznymi projekcjami JSON; nie są dodatkową zgodą na implementację.

- **I** — [independent issuer](cryptohunter_product_architecture/m05_account_genesis_independent_root_proof_issuer_contract.json).
- **S** — [production substrate](cryptohunter_product_architecture/m05_account_genesis_root_proof_issuer_production_substrate_selection_contract.json).
- **R** — [implementation readiness](cryptohunter_product_architecture/m05_account_genesis_root_proof_issuer_implementation_readiness_contract.json).
- **P** — [physical persistence](cryptohunter_product_architecture/m05_account_genesis_physical_persistence_crash_atomicity_contract.json).

| Źródło / JSON Pointer | Dokładna wartość |
| --- | --- |
| I `/bind_authorization_order/7` | `"8_ATOMICALLY_BIND_AND_PERSIST_IDENTICAL_SIGNED_PROOF_AND_HISTORY"` |
| I `/authority_model/semantic_authority_owner` | `"IndependentAccountGenesisRootProofIssuer"` |
| I `/authority_model/retained_issuance_history_owner` | `"independent root-proof issuer globally serialized registry"` |
| I `/cross_authority_atomicity/model` | `"NO_DISTRIBUTED_TRANSACTION; immutable exact-operation proof plus recoverable local phase machine and independent freshness CAS"` |
| I `/replay_and_consumption/consumption_owner` | `"NONE; CHA does not call issuer after issuance to consume"` |
| I `/lost_response/same_attempt_id_success` | `"return byte-identical root_proof_id, signed payload, signature and retained history for that exact attempt"` |
| I `/lost_response/requested_attempt_id_must_equal` | `"issuer_history.bound_issuance_attempt_id"` |
| S `/authority_separation/rule` | `"PHYSICAL COLOCATION != AUTHORITY REUSE"` |
| S `/authority_separation/separate_semantic_owners_schemas_roles_credentials_namespaces_transactions_and_lifecycles` | `true` |
| S `/authority_separation/required_separation/4` | `"transaction ownership"` |
| S `/authority_separation/CHA_direct_registry_write` | `false` |
| S `/authority_separation/issuer_direct_account_commit` | `false` |
| S `/authority_separation/Issuer_to_CHA_AttemptStore_write` | `false` |
| S `/authority_separation/CHA_to_Issuer_Registry_semantic_BIND_write` | `false` |
| R `/frozen_semantics_change_rule` | `"DO_NOT_REOPEN_WITHOUT_CONCRETE_EXPLOIT_OR_CONTRADICTION"` |
| R `/status/semantic_contract_frozen` | `true` |
| R `/status/implementation_readiness_frozen` | `true` |
| R `/implementation_gate/readiness_freeze_does_not_allow_implementation` | `true` |
| R `/implementation_gate/non_production_implementation_mode` | `"NONE_FROZEN"` |
| R `/status/implementation_allowed` | `false` |
| R `/production_M05_gate/issuer_implementation_allowed_alone_is_sufficient` | `false` |
| P `/authority_vs_storage_roles/transaction_coordinator` | `"CryptoHunterAccountAuthority"` |
| P `/selected_or_blocked_protocol/logical_authority_atomicity_equals_single_physical_transaction` | `false` |
| P `/result/selected_protocol` | `"INITIAL_BINDING -> PREPARED -> EXTERNAL_FRESHNESS_CAS -> LOCAL_FINAL_COMMIT"` |
| P `/selected_or_blocked_protocol/point_of_no_return` | `"exact authoritative freshness acceptance N -> N+1"` |
| P `/freshness_ownership_boundary/coordinator_may_unilaterally_advance_authoritative_generation` | `false` |
| P `/publishability_rule/authority_acceptance_before_local_final_commit` | `"NOT_PUBLISHABLE"` |

Konkretną podstawą rozpatrzenia R `/frozen_semantics_change_rule` jest kolizja
zakresu: jeśli „transactions” w S rozciąga się na osobne entitlement,
retention i history persistence, żadna sekwencja ich niezależnych COMMIT nie
wykonuje wymaganego atomowego kroku I. Jeżeli zakres dotyczył wyłącznie
niezależnych autorytetów protokołu, trzeba tę granicę jawnie zatwierdzić.
Wartość `true` nie definiuje takiego wyjątku, a owner-owned helper nie tworzy
nowej transakcji. Nie stwierdzamy, że wszelkie możliwe interpretacje freeze są
logicznie sprzeczne; stwierdzamy, że A nie dowodzi literalnej separacji
fizycznych transakcji podległych komponentów. R dopuszcza przedstawienie
konkretnego konfliktu do review; nie zatwierdza proponowanego rozwiązania.

P nie jest wyjątkiem uprawniającym do rozdzielenia write set issuera.
Jego wielofazowy protokół dotyczy utworzenia konta przez CHA, niezależnej
freshness CAS oraz własnego lokalnego closure. Issuer decision COMMITTED
nie oznacza AccountGenesis COMMITTED.


## Macierz własności i granic transakcji

Klucze A01–A13 łączą dwie tabele; każda para obejmuje dziesięć wymaganych pól. Requester i claimant są pokazani osobno. A13 oddziela istniejący testowy coordinator od docelowego produkcyjnego issuera.

### Macierz 1/2 — ownership, principals, credential i dostęp

| Klucz / komponent | Semantic owner | SQL/schema owner | Runtime/admin principal | Credentials i trust boundary | Dozwolone read/write |
| --- | --- | --- | --- | --- | --- |
| A01 RootProofIssuer | Niezależny issuer; odrębny od CHA genesis admission | Brak aktywnego production final write API; odrębni ownerzy persistence; test coordinator ma NOLOGIN owner | Production issuer i administrator są rozdzieleni od CHA/requester/claimant/dwóch signerów | Genuine InitialBinding i signed immutable attempt, live public registries, issuer/history custody i trust activation | Production: tylko preflight/reconciliation; SIGN/BIND/history/checkpoint/CHA finalize/replacement/transport zablokowane |
| A02 Entitlement registry | Authority entitlement/provenance/lifecycle/BIND | V1 configurable owner (instalator tworzy LOGIN); migrated V2 zachowuje odrębnego ownera NOLOGIN | V1 runtime `append_bind`, admin `provision/revoke/supersede`; V2 public runtime wyłącznie reader | Direct login, exact live roles/ACL/source/durability; claimant anchor i entitlement generation | V1 read + standalone CAS BIND/admin lifecycle; V2 read oraz zamknięte test coordinator helper |
| A03 Signing intent | Issuer: immutable `rpf_` i unsigned payload | Dedicated NOLOGIN owner; V1 intents/metadata; jawny V2 catalog migration | V1 issuer_writer direct login; V2 preparation writer distinct from final coordinator | Genuine binding/attempt, pinned root signer snapshot i public authority; writer isolation nie wynika z DTO | Read/reserve immutable RESERVED; bez SIGN/BIND/issuance |
| A04 Retention | Issuer: exact signed candidate STAGED | Dedicated NOLOGIN owner, prepared/metadata; V2 migrated helper | V1 dedicated runtime; V2 preparation writer; offline provisioning | Pinned root verification authority i original ACTIVE snapshot, exact request/payload/signature | Read/retain exact bytes; bez SIGN, BIND albo issuance |
| A05 Issuer history | Issuer history/attestation authority | Dedicated NOLOGIN history owner | V1 runtime read/append/retain_head, admin provision_stream; V2 reader i closed atomic helper | DB credential odrębne od history signing key; configured verification authorities/checkpoint | V1 standalone record append i osobne head retention; V2 atomic helper/test reader |
| A06 Root/history signing custody i completion | Dwa odrębne signery + lifecycle admin; completion authentication pozostaje odrębne | Private key/journal poza SQL; public production lifecycle ma własny NOLOGIN owner | Installed separate Linux UID/GID signerów; peer-auth issuer/admin; admin może close, nie completion publication | Native keyring i protected metadata; seed/public match i journal hash nie dowodzą independent private control/authentication | Local substrate signing/lifecycle; production IPC tylko DIAGNOSTIC/QUALIFY; #3154 tylko read existing COMPLETE |
| A07 Requester credentials/custody | CHA requester `CryptoHunterAccountAuthority` | `cryptohunter_requester_owner` NOLOGIN; private custody poza SQL | `cryptohunter_requester_runtime` read; requester_admin apply; osobny offline custody admin | Requester semantic role, scoped material/version/revision/lifecycle; private key nie trafia do SQL | Public lookup/admin PROVISION/ROTATE/TRANSITION; private runtime tylko domain-bound request signature |
| A08 Claimant credentials/custody | Pre-account provisioning principal z immutable entitlement provenance | `cryptohunter_claimant_owner` NOLOGIN; private custody poza SQL | `cryptohunter_claimant_runtime` read; claimant_admin apply; osobny offline custody admin | Claimant role/principal/key/version/material, exact provenance; zakaz aliasu z requesterem | Public lookup/admin lifecycle; private runtime tylko claimant authorization |
| A09 CHA AttemptStore | CHA: identity/state/recovery lokalnego attempt | Dedykowany SQLite plik V6, filesystem owner; bez PG role | CHA writable; issuer `open_readonly` bez migracji/tworzenia | Store bez signing credential; zachowuje immutable authorization/request bytes i dwa podpisy | CHA reservation/signature checkpoint/finalize/recovery/replacement wg fence; issuer tylko read |
| A10 FreshnessAuthority | Odrębna anti-rollback/freshness authority; CHA jest proposer | `freshness_schema_owner` i odrębny `freshness_function_owner` | runtime CAS; crypto_verifier prepare; admin provision/transition; reader SELECT | Odrębne proposer/finalization credentials; fixed AF_UNIX verifier i peer PostgreSQL login | Verify→durable prepare→CAS; bez genesis admission, Root Proof SIGN/BIND albo remint |
| A11 CheckpointAuthority | Osobna acceptance authority exact signed history head | Dedicated NOLOGIN checkpoint owner | Direct runtime read/commit_checkpoint; admin provision_stream | Exact history/signer composition, acceptance lineage i retained signature | Authenticated read, own CAS advance/resolve; bez issuer BIND |
| A12 AccountGenesis | Jeden `CryptoHunterAccountAuthority` — finalne admission/genesis | Docelowy dedicated authority SQLite adapter niedostępny; projection/AttemptStore nie są nim | Obecnie installed reservation verifier/CHA; brak pełnego production coordinator | Exact operation/request/reservation/account, accepted independent Root Proof i fresh authenticated closure | Wąski INITIAL_BINDING candidate reservation; bez production PREPARED→COMMITTED |
| A13 Atomic commit coordinator | Issuer persistence decision, bez przejęcia semantycznej własności component authority | Dedicated coordinator NOLOGIN owner i sześciu odrębnych component owners | Direct issuer_writer; osobny test_admin/verifier, bez wspólnych credentials i memberships | Genuine PostgreSQLAtomicRootProofTestDeployment, external catalog/physical pin, trusted exact preparation/approval; test fences nie production custody | Writer: fixed commit/read/qualify API; owner: wyłącznie zamknięte helper EXECUTE, bez cross-owner DML |

### Macierz 2/2 — transakcje, linearizacja, zależności, recovery i status

| Klucz | Transaction owner | Linearization / COMMIT | Dependencies | Atomicity / CAS / recovery | Production / test / unimplemented status |
| --- | --- | --- | --- | --- | --- |
| A01 | Production final owner niewykonawczy; V1 adaptery mają własne connection. Testowy `PostgreSQLAtomicRootProofCommit` posiada jedno connection/xid | Docelowo issuer COMMIT pełnego write set; tylko test wykonuje taki COMMIT; preflight ma `atomic_global_snapshot=False` | A02–A09, trust activation; późniejsze A11; późniejsze CHA admission | Unknown wymaga positive exact resolution, bez remint/re-sign/UNBIND; connection token dowodzi continuity, nie production authority | Read-only runtime implemented; production issuance `NOT_IMPLEMENTED/BLOCKED`; atomic commit `ISOLATED_TEST_ONLY` |
| A02 | V1 `_serializable` własnego connection; V2 test helper uczestniczy w coordinator connection | V1 standalone BIND COMMIT; V2 test BIND w jednym issuer write set | Entitlement contract/provenance; A08 anchor | Expected revision/lineage lock, append-only history, exact replay/CAS; standalone sukces nie dowodzi proof/history closure | Production-local V1 substrate implemented; V2 tylko jawnie migrated test deployment; production issuer flow blocked |
| A03 | V1/V2 preparation własna transakcja reserve | Osobny COMMIT RESERVED przed signing i finalnym COMMIT | A07/A08/A09 + pinned issuer signer | Attempt lock i immutable unique ID fences; exact retry zwraca oryginał; unknown→indeterminate; NOT_FOUND nie dowodzi unbound | V1 substrate/V2 test preparation implemented; brak production composition; RESERVED nie issued |
| A04 | V1/V2 preparation własna retention transaction | Osobny STAGED COMMIT przed finalną decyzją | A03/A09 + root signer historical verify | Exact immutable bytes i unique fences, no re-sign; unknown→indeterminate; stage bez marker nie issued | Substrate/V2 test implemented; final production issuance unavailable |
| A05 | V1 append/head retention osobne transactions; V2 closed helper w test coordinator | Każdy standalone append/head ma własny COMMIT; test atomowo zapisuje record/head z BIND | Exact stream/event predecessor, A06 history signer, A11 | Stream lock + digest CAS + exact replay/full chain; unknown bez automatycznego tworzenia nowego eventu; valid-prefix rollback wymaga checkpoint | Production-local standalone history implemented; V2 atomic tylko test; production closure blocked |
| A06 | Local custody OS lock/files/keyring; admin close posiada osobny SERIALIZABLE SQL COMMIT | DB `PENDING_CLOSED` musi być pozytywnie committed przed custody mutation; niezależny completion publication COMMIT niedostępny | Installation/service/role/OID/catalog/physical pins, peer IPC, exact before/after journal | Lock-synchronized READ COMMITTED close resolution; brak=unknown. #3154 shared flock/read-only COMPLETE nie otwiera fence; V1 receipt legacy-unverified | Production closed V2 i read-only verifier implemented; independent completion/publication/activation/production SIGN BLOCKED; durable coordinator test-only |
| A07 | Registry admin własny SQL transaction; private custody odrębny lock/keyring | Public lifecycle admin COMMIT; podpis to odrębna lokalna operacja; final issuer wymaga nowego locked read | A08 peer anti-alias registry, A09 signed attempt | SERIALIZABLE + wspólny preaccount alias advisory lock + row/revision CAS; brak atomic keyring–metadata–PG | Public/custody substrate implemented; V2 preparation reader/test helper nie nadają production issuance |
| A08 | Jak A07, własny claimant authority transaction | Claimant lifecycle COMMIT; nie issuer BIND COMMIT | A02 immutable provenance; A07 material anti-alias; A09 | Forward VERIFY_ONLY/REVOKED, exact public history; final issuer musi blokować ACTIVE stan w swoim COMMIT | Public/custody substrate implemented; nie issuer/root trust ani completion authority |
| A09 | CHA SQLite `_write`: BEGIN IMMEDIATE/COMMIT, WAL/FULL/FK | Oddzielne reservation/checkpoint/finalize/recovery COMMITy; nie issuer COMMIT | Genuine InitialBinding A12, A07/A08, authoritative unbound evidence dla replacement | expected_fence CAS, immutable triggers/unique keys i signature retention; read-only v6 bez upgrade; brak response nie zezwala na replacement | Dedicated PRODUCTION_LOCAL store implemented; hosted tests nie fizyczna qualification; brak uczestnictwa w PG issuer commit |
| A10 | Verifier prepare własny COMMIT; runtime nowe SERIALIZABLE connection/COMMIT | CAS COMMIT authoritative generation/document/heads/decision/receipt; prepare nie jest acceptance | Fixed verifier, exact predecessor/full head set, two public lifecycle credentials | Jedna CAS transakcja blokuje prep/lineage/credentials/lifecycle i atomowo consumes prep; exact replay, oddzielne unknown prepare/CAS; exact serialization retry | PRODUCTION_LOCAL substrate/high-level adapter implemented; nie SERVER_READY/whole-host rollback independence/genesis authority |
| A11 | Osobny checkpoint connection/transaction; wcześniejsze history/custody reads są odrębne | Own acceptance COMMIT po issuer COMMIT; standalone ACTIVE/generation sample nie jest DB fence aż do COMMIT | A05 history i A06 signer lifecycle; docelowo production DB lifecycle fence | Locked revision/head/acceptance CAS, exact replay/resolve_advance, reject rewind/fork; local floor nie niezależny durable witness | Standalone local checkpoint implemented; production acceptance lifecycle coupling i pełny issuance flow blocked; independent rollback/admin domain=false |
| A12 | Obecny InitialBinding OS lock + durable file publisher; docelowy genesis coordinator posiada własną authority persistence | INITIAL_BINDING_COMMITTED tylko candidate; genuine COMMITTED wymaga pełnej operation/reservation/root-validation/freshness closure | A01 independent root evidence, own request/reservation/account histories i A10 | Exact retained winner retry, no remint po durable reservation; ambiguous publication reread/reassert fences; brak authority fail closed | Owner topology FROZEN; full production genesis/admission BLOCKED. Legacy Stage9AccountGenesisAuthority i LPPI membership TEST_ONLY |
| A13 | Trusted PostgreSQLAtomicRootProofCommit posiada connection i finalne SERIALIZABLE; SQL helper nie posiada COMMIT | Jedno conn.commit po pełnym write set/continuity check; unknown response nie jest sukcesem ani rollbackiem | Jedna kwalifikowana DB16, migrated schemas, protected approval, original bytes, ACTIVE lifecycle/credentials i history predecessor | Deferred obustronne linkage constraints, immutable/unique winner marker, exact CAS i positive read-only byte-identical recovery; brak nowego SIGN/attempt | ISOLATED_TEST_ONLY; brak produkcyjnego adaptera/ingress/approval; finalna produkcyjna transakcja niewdrożona |

### Źródła macierzy i granice dowodu

`PRODUCTION_LOCAL substrate implemented` w tabelach oznacza istniejący adapter
tego profilu, nie kwalifikację rzeczywistej instalacji ani dostępną emisję.
Parametry owner/principal są porównywane z żywymi OID; równe nazwy/DSN
nie ustanawiają ani authority, ani wspólnego COMMIT. Poniższe dowody kodu
uzupełniają macierz. Zakres **wykonanych** testów podano oddzielnie w walidacji;
istnienie innych testów w repozytorium nie jest raportem ich wykonania.

| Klucz | Zweryfikowane źródła i konkretna granica dowodu |
| --- | --- |
| A01 | [runtime](../../bot_core/root_proof_issuer_runtime.py) `preflight_root_proof_issuance_attempt`, [orchestration](../../bot_core/root_proof_issuer_orchestration.py) `_orchestrate_selected_authorities`, [substrate](../../bot_core/root_proof_issuer_substrate.py) `RootProofIssuerCompositionGate`: genuine input/live reads, ale literalne false dla issuance/authorization i brak globalnego snapshotu. [Production boundary](../../bot_core/production_root_lifecycle.py) `commit` ma bezwarunkową odmowę. |
| A02 | [registry](../../bot_core/postgresql_entitlement_registry.py) `_provisioning_sql`, `_serializable`, `compare_and_swap_bind`: owner V1 LOGIN, lineage FOR UPDATE, revision CAS i własny standalone COMMIT; brak signed proof/history closure. [Migracja](../../bot_core/postgresql_atomic_root_proof_schema.py) `install_atomic_root_proof_authority` odbiera stare granty i zmienia ownera na NOLOGIN wyłącznie w wybranym isolated deployment. |
| A03 | [intent runtime](../../bot_core/postgresql_root_proof_signing_intent.py) `reserve`, `_restore` i [schema](../../bot_core/postgresql_root_proof_signing_intent_schema.py): immutable RESERVED/unique identity i own transaction. [V2 preparation](../../bot_core/postgresql_root_proof_preparation_v2.py) `PostgreSQLRootProofSigningIntentV2.reserve`: oddzielny commit, qualification i testowy signer fence. |
| A04 | [retention](../../bot_core/postgresql_root_proof_retention.py) `retain_prepared`, `_restore`, `PostgreSQLRootProofRetention.require_issued` i [schema](../../bot_core/postgresql_root_proof_retention_schema.py): canonical BYTEA/signature oraz historical verification, STAGED nie issued. [V2](../../bot_core/postgresql_root_proof_preparation_v2.py) `PostgreSQLRootProofRetentionV2`: own transaction i exact intent linkage. |
| A05 | [history](../../bot_core/postgresql_authenticated_issuer_history.py) `append`, `retain_attested_head`, `authenticated_read_snapshot`; [schema](../../bot_core/postgresql_issuer_history_schema.py) `_function_sources`: stream lock, predecessor CAS, record i head w osobnych standalone transakcjach. V2 atomic helpery należą do [atomic schema](../../bot_core/postgresql_atomic_root_proof_schema.py) `_component_inventory`, nie produkcyjnej orchestration. |
| A06 | [local custody](../../bot_core/local_signing_custody.py) `_custody_lock`, `_sign_result`; [production custody](../../bot_core/production_signer_custody.py) `qualify_installed_signer_custody`; [IPC](../../bot_core/production_signer_ipc.py) `qualify_signer_ipc`; [completion observation](../../bot_core/local_signing_lifecycle_evidence.py) `observe_durable_signing_lifecycle_completion`: real locks i exact metadata, ale brak independent private control/authenticated publication. [Production lifecycle](../../bot_core/production_root_lifecycle.py) `_administrative_operation`, `resolve_close`, `publish_administrator_completion`; [SQL V2](../../bot_core/postgresql_production_root_lifecycle_schema.py) reviewed functions: durable PENDING_CLOSED i completion tombstone, bez otwarcia fence. |
| A07/A08 | [public credentials](../../bot_core/postgresql_preaccount_credentials.py) `_ddl`, `_apply_source`, `_qualify`, requester/claimant providers: sześć distinct roles, alias advisory lock, current/history i own admin SERIALIZABLE CAS. [Private custody](../../bot_core/cha_root_proof_signing_custody.py) `_RuntimeCustody`, requester/claimant classes: role-bound signature i local lock, bez atomic keyring–metadata–PostgreSQL. [Atomic credential helper](../../bot_core/postgresql_atomic_root_proof_preaccount_sql.py) `reviewed_preaccount_helper`: exact ACTIVE generation/revision i row locks w finalnej testowej transakcji. |
| A09 | [CHA store](../../bot_core/cha_attempt_store.py) `SQLiteCHAAttemptStore`, `_write`, `_read`, `_cas`, `open_readonly`: dedicated schema V6/WAL/FULL, own BEGIN IMMEDIATE, expected fence i issuer read-only ingress. [Signed attempt](../../bot_core/licensing/cha_root_proof_signed_attempt.py) utrwala original request/signature checkpoints; nie ma prawa issuer BIND. |
| A10 | [freshness adapter](../../bot_core/account_genesis_freshness_authority.py) `ProductionLocalFreshnessAuthority`, `_compare_and_advance_once`; [SQL](../../bot_core/postgresql_freshness_authority.py) `PostgreSQLFreshnessAuthorityProvisioning`, `_sources`; [verifier](../../bot_core/freshness_semantic_verifier.py): separate prepare, live authenticated CAS, locked lineage/credentials i original acceptance bytes. Nie ustanawia whole-host rollback independence ani AccountGenesis admission. |
| A11 | [checkpoint](../../bot_core/postgresql_checkpoint_authority.py) `advance`, `_read_authenticated`, `authenticated_read_snapshot`, `capabilities`; [schema](../../bot_core/postgresql_checkpoint_schema.py): own revision CAS/acceptance transaction. ACTIVE/generation sprawdzane wcześniej nie utrzymują DB lifecycle lock do acceptance COMMIT; process `_highest_revision` nie jest niezależnym trwałym witness. `independent_rollback_domain` i `independent_admin_or_security_domain` pozostają false. |
| A12 | [Production reservation](../../deployment/windows_production_cha_account_reservation.py) `establish_installed_account_initial_binding`, `_reserve`; [InitialBinding capability](../../bot_core/licensing/cha_account_reservation.py) `VerifiedAccountGenesisInitialBinding`: źródła rozróżniają durable candidate reservation od pełnej authority closure. [Legacy genesis](../../bot_core/licensing/external_provisioning.py) `Stage9AccountGenesisAuthority`, `_membership_payload`: TEST_ONLY/LPPI_TEST_ONLY, bez awansu do produkcji. [Physical persistence contract](cryptohunter_product_architecture/m05_account_genesis_physical_persistence_crash_atomicity_contract.json) `authority_vs_storage_roles` i [topology](cryptohunter_product_architecture/m05_account_genesis_authority_topology_resolution_after_binding_freeze.json) `decision/role_matrix`: pełny genesis wymaga genuine operation COMMITTED + reservation CONSUMED_COMMITTED + accepted Root Proof i authenticated freshness, nie samego receipt. |
| A13 | [commit](../../bot_core/postgresql_atomic_root_proof_commit.py) `PostgreSQLAtomicRootProofCommit.commit`, `reconcile`; [schema](../../bot_core/postgresql_atomic_root_proof_schema.py) `_function_sources`, `install_atomic_root_proof_authority`; [qualification](../../bot_core/postgresql_atomic_root_proof_qualification.py) `_catalog_fingerprint`, `qualify_atomic_root_proof_authority`; [transaction](../../bot_core/postgresql_atomic_root_proof_transaction.py) `_observe`, `validate_atomic_root_proof_transaction_owner`: wykonywalny closed write set i continuity, bez produkcyjnej authority. |

Istotne ograniczenia: checkpoint nie ma jeszcze udowodnionej produkcyjnej
linearizacji lifecycle z acceptance; public credential preflight nie zastępuje
finalnych locked reads; #3154 nie ustanawia niezależnej completion authority.
Równe `system_identifier`/database OID mogą przetrwać clone/restore i nie
dowodzą wspólnego live primary. Żaden local substrate sam nie dowodzi
odporności na skoordynowany whole-host rollback, HSM/non-exportability ani
`PRODUCTION_SERVER_READY`.


## Rzeczywista granica SQL i finalny write set

`postgresql_atomic_root_proof_schema.py::install_atomic_root_proof_authority`
instaluje funkcje PL/pgSQL `SECURITY DEFINER SET search_path=pg_catalog`.
`commit_root_proof(bytea)` jest **funkcją**, nie procedurą posiadającą
transakcję. Nested helpers wykonują się w tej samej sesji, backendzie i xid;
zmiana `current_user` na ownera helpera nie tworzy nowej transakcji.

Istniejący testowy coordinator
`PostgreSQLAtomicRootProofCommit.commit()` sam otwiera połączenie, wykonuje
`BEGIN ISOLATION LEVEL SERIALIZABLE`, kwalifikację i capture ownera, odczytuje
chronione approval, wywołuje SQL, sprawdza exact bytes i continuity, a następnie
**raz** wywołuje `conn.commit()`. Dopiero pomyślna odpowiedź COMMIT pozwala
zwrócić decyzję. Wynik funkcji przed tym punktem jest jeszcze wynikiem
transakcji otwartej. Błąd po wysłaniu COMMIT oznacza wynik nieznany.

### Przygotowanie i finalizacja

`RESERVED` intent, `STAGED` proof i immutable test-verifier approval mają
wcześniejsze, własne COMMIT. To legalne przygotowanie, bez emisji.
`require_issued()` retention odmawia. `environment=PRODUCTION` wewnątrz
testowego pakietu jest polem danych, nie świadectwem wdrożenia produkcyjnego.
Finalny write set obejmuje jednocześnie:

1. CAS entitlement `ACTIVE/UNBOUND → BOUND`, zmianę lineage revision
   i odpowiadający registry history event;
2. authenticated issuer record i przesunięcie stream head przez predecessor CAS;
3. dokładny signed history head wraz z podpisem;
4. immutable decision marker z winner attempt, operation, entitlement/root/account
   identity oraz **kopią oryginalnych BYTEA**: reserved envelope, staged envelope,
   canonical proof payload, signature, canonical issuer record, canonical signed
   head i history signature, powiązaną z BOUND i chronionym approval.

Sam intent i stage nie są aktualizowane do nowego statusu; marker ustanawia
decyzję i zachowuje exact bytes. Digest lub niechroniony zewnętrzny odsyłacz
nie zastępują tej retencji. `validate_decision()` ponownie zestawia marker
z oryginalnymi intent/stage i rzeczywistym BOUND/history. Deferred constraint
triggers wymagają odwrotnego powiązania nowych BOUND/records/heads z decyzją.
Nie można zatwierdzić samego BIND, a później uzupełnić jego historii.

Przed tymi zapisami ta sama transakcja utrzymuje uporządkowane blokady
root/history/trust fences, wspólny credential alias advisory lock oraz
requester/claimant row locks, sprawdza ACTIVE/generation i exact signed request
identity. Credential helpers wyłącznie odczytują/blokują; nie przejmują
credential administration ani key custody. Entitlement/history locks i unique
constraints serializują CAS. Widoczność całej decyzji ustanawia trwały fizyczny
COMMIT przy `fsync=on`, `synchronous_commit=on`, `full_page_writes=on`.

Nie obejmuje to key-store mutation, checkpoint advance, CHA AttemptStore,
freshness CAS ani account commit. Podpis przygotowany wcześniej nie jest
automatycznie dowodem aktualnej eligibility w finalnej transakcji.

### ACL, role i zależności

Coordinator owner jest `NOLOGIN`; ma w cudzych schematach tylko określone
`USAGE`/helper `EXECUTE`. Component owners pozostają osobni i po migracji V2
są `NOLOGIN`. Issuer writer ma coordinator runtime API i odczyt metadata,
bez bezpośredniego DML/DDL, helper EXECUTE, membership ani administracji.
Verifier/admin ma osobne testowe `approve_commit`, niedostępne writerowi.
`_guard()` porównuje direct `session_user` z writerem oraz `current_user`
z ownerem funkcji. Helper ACL jest konieczny: wewnątrz nested definer
`current_user` identyfikuje helper ownera, nie jego bezpośredniego callera.

`SET ROLE` nie zmienia `session_user`; nadanie membership, szerokiego grantu
lub dodatkowego pośredniego definer nie jest dozwolonym sposobem integracji.
`postgresql_atomic_root_proof_qualification.py::_catalog_fingerprint`
przypina role/OID/atrybuty/memberships, schemas, relations/columns, constraints,
indexes, functions (w tym `prosecdef`, `proconfig`, `prosrc`, `proacl`, `prokind`
i pełną definicję), triggers, rules, policies, defaults i external ACL/definers.
Reviewed sources i manifest nie są wybierane przez caller runtime.
Zależności wywołań PL/pgSQL weryfikujemy również w treści funkcji. Kwalifikatory
nie wykonują ogólnego rekurencyjnego domknięcia `pg_depend`; `pg_shdepend`
w SQL snapshot sprawdza wskazane owned objects writera/admin. Przypięcie
helperów nie atestuje całego serwera, builtinów, operatorów/castów ani hosta.
Nowa extension lub zewnętrzna zależność wymaga osobnego audytu i inventory.

Migracja jawnie odbiera standalone `append_bind`, `append_record` i
`retain_head` wcześniejszym runtime principalom; deferred linkage domyka SQL
write set. Dotyczy to **migrowanego izolowanego inventory V2**. Nie oznacza
automatycznego odebrania praw w dowolnej starej, niemigrowanej bazie.
Legacy `compare_and_swap_bind()` nadal jest realnym niższym API z własnym
SERIALIZABLE i grantem runtime w niemigrowanym registry. Production selector
komponuje V1 do odczytu/autoryzacji; nie odbiera tego niższego capability.
Publiczny semantic issuer nie wywołuje BIND i nie ma do niego fallthrough.
To ograniczenie zakresu dzisiejszych blokad, nie globalny zakaz SQL we wszystkich
historycznych bazach. Przyszły produkcyjny issuer wymaga osobno przejrzanej
migracji zamykającej te porty. Trusted DBA/superuser i administrator hosta pozostają w jawnej
granicy zaufania: mogą zmienić instalację, ale nie stanowią runtime authority
ani dowodu kwalifikacji. Catalog drift odmawia kwalifikacji zamiast naprawiać
go w locie.

`postgresql_atomic_root_proof_transaction.py` obserwuje exact connection,
backend PID, xid, database/OID, login/OID oraz writable SERIALIZABLE.
Wcześniejszy COMMIT, inne połączenie lub zmiana roli odmawiają walidacji.
Token nie udziela uprawnień i nie otwiera transakcji. Nie dowodzi zachowania
row locks przez rollback caller-owned savepointu; dlatego finalny trusted
coordinator musi posiadać cały przebieg transakcji i nie przyjmować callbacków.
Nie rozstrzyga też utraty odpowiedzi po COMMIT.

Nie ma serwerowego wyłącznego ownership końcowego COMMIT: trusted Python
coordinator posiada połączenie i sekwencję poleceń. Zamiana funkcji na procedurę
nie dowodzi takiej wyłączności. W PostgreSQL procedura `SECURITY DEFINER`
lub z klauzulą `SET` nie może wykonywać transaction control. Invoker procedure
bez `SET` wymaga praw callera do helperów; taki grant umożliwia też ich
bezpośrednie wywołanie poza wrapperem. Membership, GUC lub token caller-owned
tego nie rozwiązują. Obecny SQL nadal wymusza kompletny write set, lecz nie
odbiera loginowi możliwości wysłania COMMIT/ROLLBACK. Docelowy trusted service
musi zachować wyłączne posiadanie połączenia i nie udostępniać go callerowi.

## Wariant A — wspólny fizyczny COMMIT, osobni właściciele semantyczni

Technicznie wykonalny w jednej bazie PostgreSQL 16. Pozwala zachować jeden
atomowy write set, jednego winnera CAS oraz byte-identical recovery po
utracie odpowiedzi. Uporządkowane owner-owned helpers zapewniają minimalne
uprawnienia do konkretnych operacji bez przekazania writerowi owner membership
lub bezpośredniego DML. Własność semantyczna nie wymaga osobnego COMMIT
każdej tabeli, ale **freeze obecnie literalnie wymaga także separacji
transaction ownership**. Wariant A sam tego wymagania nie spełnia.

Zalety: lokalne constraint enforcement i WAL, jeden jednoznaczny commit point,
brak distributed recovery coordinatora, zachowanie oryginalnych bajtów i
obecnych CAS. Koszty: wspólna domena awarii finalnego persistence, cross-schema
helpers/constraints i ich wersjonowanie, wspólny lock order, ryzyko deadlocku
i serialization failure, wymagane zamknięcie historycznych write API.
Nie dowodzi osobnego rollback domain checkpointa, niezależnej custody
ani bezpieczeństwa produkcyjnego ingress.

Rekomendujemy A dopiero po C, bez zmiany payloadów, identity, cardinality,
crypto domains lub historycznych bytes. CHA, FreshnessAuthority, CheckpointAuthority
i AccountGenesis zachowują własne transakcje; nie ma globalnego issuer/CHA/account
COMMIT. Credential lifecycle admin i custody mutation także nie stają się
zapisami coordinatora.

## Wariant B — odrębne transakcje fizyczne

### Zwykłe niezależne COMMIT

Nie istnieje bezpieczna kolejność dwóch niezależnych commitów, która zapewnia
oba niezmienniki. Jeżeli BIND commit jest pierwszy, crash przed proof/history
pozostawia BOUND bez wymaganej retencji. Jeżeli proof/history **decyzji** commit
jest pierwszy, crash przed BIND pozostawia COMMITTED proof/history bez winnera.
STAGED może być zachowane wcześniej, lecz nie staje się przez to COMMITTED.
Trzeci immutable marker nie usuwa już widocznego prefiksu zapisów; czytelnicy
musieliby dostać nowy protokół widoczności, a obecne niezmienniki i SQL guards
uległyby zmianie. Kompensacja nie cofa wyemitowanego podpisu/obserwacji, narusza
append-only historię i nie zastępuje atomowego write set.

Oddzielne CAS mogą ustanowić jednego lokalnego winnera, ale nie atomowość
całej decyzji. Exact retry i odczyt oryginalnych bajtów rozwiązują duplikację
po znanym commicie, nie brak drugiego commitu po crash. Timeout albo brak
rekordu po utracie odpowiedzi pozostaje `OUTCOME_UNKNOWN`/INDETERMINATE;
nie jest dowodem UNBOUND ani zgodą na nowy attempt lub resigning.

### Distributed transaction / PostgreSQL two-phase commit

2PC wymaga durably recorded global decision, jednoznacznych identyfikatorów
uczestników i prepared transactions, autoryzowanego recovery coordinatora,
odbudowy po restarcie, monitoringu `pg_prepared_xacts` i rozstrzygania dokładnie
tych samych bytes. `PREPARE TRANSACTION` utrwala przygotowany stan i utrzymuje
blokady; po crash koordynatora uczestnicy mogą pozostać zablokowani bez limitu.
Nie wolno heurystycznie abortować uczestnika po trwałej globalnej decyzji COMMIT.

PostgreSQL wymaga jawnego `max_prepared_transactions > 0`, odpowiedniej pojemności
lock/WAL i utrzymania recovery credentials. `COMMIT PREPARED` nie jest
transakcją obejmującą inne `COMMIT PREPARED`: pomiędzy ich wykonaniami istnieje
okno różnej widoczności dla zwykłych czytelników. 2PC może zapewnić wspólny
nieodwracalny outcome pod założeniami poprawnego recovery, ale nie samoczynnie
jednoczesną widoczność wszystkich zapisów w osobnych bazach/transakcjach.
Ścisła widoczność wymagałaby dodatkowych participant/reader fences i wspólnego
autorytetu decyzji. To nowy protokół i rozszerzenie trust boundary, wymagające
własnego rozstrzygnięcia kontraktowego i kwalifikacji.

Prepared transactions zatrzymują zasoby, mogą blokować admin lifecycle,
revoke i postęp vacuum; deadlocki między uczestnikami wymagają globalnego
porządku blokad, limitów i recovery. Utrata odpowiedzi po PREPARE albo COMMIT
PREPARED wymaga pozytywnego powiązania statusu uczestnika z trwałą globalną
decyzją; samo zniknięcie GID nie rozróżnia COMMIT od ROLLBACK. Unikalność
winnera i byte-identical retry nadal muszą być egzekwowane we wszystkich
uczestnikach, nie wynikają z włączenia PostgreSQL 2PC.

I `/cross_authority_atomicity/model` dodatkowo zachowuje `NO_DISTRIBUTED_TRANSACTION`
w granicy niezależnych autorytetów; 2PC nie daje prawa do wspólnej decyzji
issuer/CHA/freshness/checkpoint. Samo fizyczne rozdzielenie uczestników
nie rozstrzyga kontraktowej własności globalnego coordinatora.

Nie ma takiego protokołu w audytowanym issuerze. Nie uruchamiamy 2PC ani
nie włączamy prepared transactions jako pozornej naprawy kontraktu. B ma
wyższy koszt dostępności, administracji, autoryzacji i testów niż A, bez
wykazanej zgodności z literalnym jednym atomowym persistence.

## Wariant C — osobno zatwierdzane addendum

Minimalna proponowana zmiana dotyczy **zakresu separacji transakcji**:
utrzymać niezależną własność semantyczną wszystkich authority, a jawnie
dopuścić jedną issuer-owned fizyczną transakcję wyłącznie dla finalnego
write set BIND + identical signed proof retention + issuer history/head
+ decision marker. Podległe persistence APIs nie stają się ownerami emisji
ani nie otrzymują wspólnych credentials. Credential reads/fences w tej
transakcji nie przenoszą ich administracji do issuera.

Pełny tekst, zakres i warunki ratyfikacji są wyłącznie w
[propozycji addendum](proposals/stage9_root_proof_transaction_authority_addendum.md).
Canonical JSON i historyczny freeze pozostają bez zmian. Zatwierdzenie
propozycji musi skutkować osobną wersją/addendum, parity checks i jawną migracją;
nie wystarczy zmienić odczytu pola true w kodzie ani dopisać `approved=true`.

## Egzekwowanie blokad i recovery

`ProductionRootLifecycleBoundary.commit()` bezwarunkowo podnosi
`PRODUCTION_BIND_BLOCKED_FROZEN_TRANSACTION_CONFLICT_AND_UNQUALIFIED_ATTEMPT_INGRESS`.
Nie ma produkcyjnej gałęzi sukcesu zależnej od DTO, konfiguracji lub test tokena.
Publiczny selector nie importuje testowego atomic write authority; publiczna
orchestration kończy się `EXECUTION_BLOCKED` także po pozytywnych odczytach.
Jej `EXACT_COMMITTED` to klasyfikacja wcześniejszych obserwacji, nie capability
do emisji ani dowód zachowanych proof bytes.

Isolated coordinator wymaga genuine test deployment, preparation i chronionego
oddzielnego test approval; nie przyjmujemy go jako production adaptera.
Zainstalowana production boundary osobno wymaga protected installation,
process UID/GID, exact PostgreSQL peer `system_user`, bez memberships i z
przypiętym inventory. `assert_active_fences()` nie jest write entrypointem
i nie zastępuje finalnej authority ani zatwierdzenia kontraktu.

Po #3154 `publish_administrator_completion()` nadal zawsze odmawia, a SQL
`complete_close` jest tombstone bez prawa EXECUTE. Odczyt `COMPLETE` journal
nie otwiera SQL fence. `NOT_FOUND` close/decision, timeout, przerwany backend
lub brak odpowiedzi nie stanowią pozytywnego dowodu rollbacku. Recovery
decyzji odczytuje istniejący marker i oryginalne payload/signature/envelope,
bez drugiej historii, nowego proof ID, nowego attempt albo ponownego podpisu.

Dodane regresje w
`tests/security/test_root_proof_transaction_authority_reconciliation.py`
sprawdzają actual refusal guards: podmieniony raport z sukcesem, detached
COMMITTED DTO, obiekt isolated coordinatora, fabricated transaction owner,
readiness flags i `approved`/`CONTRACTUALLY_APPROVED` nie dochodzą do SQL
ani composition. Nie znaleziono luki uzasadniającej nową abstrakcję runtime.

## Walidacja wykonana w tej zmianie

2026-10-18, Linux, Python **3.12.14** (CI wybiera 3.11). PostgreSQL **16.15**
w dwóch jednorazowych klastrach `postgres:16`, porty lokalne 55432/55434,
różne system identifiers; `fsync=on`, `synchronous_commit=on`,
`full_page_writes=on`, primary poza recovery. Testowe `trust` nie dowodzi
produkcyjnego peer authentication. Genuine lineage używa testowego keyring,
nie produkcyjnych kluczy/enrollmentu.

| Wykonany zakres | Wynik | Co ten wynik dowodzi |
| --- | --- | --- |
| 8 suites architecture: independent issuer, production substrate, implementation readiness, physical persistence, atomic import boundary, attempt reservation freeze, preaccount credentials freeze, signed immutable attempt freeze | **845 PASS**, 22.58 s | Walidatory frozen semantyki, projekcji i historycznych bytes; readiness/NO-GO i publiczny import graph pozostają zamknięte. |
| Nowe `test_root_proof_transaction_authority_reconciliation.py` | **23 PASS**, 0.34 s | Cross-layer DTO/flag/report/test-coordinator/transaction-owner claims nie otwierają production COMMIT ani publicznej composition. |
| `test_postgresql_atomic_root_proof_transaction.py`, `test_postgresql_production_root_lifecycle_qualification.py`, `test_production_root_lifecycle_schema.py`, `test_production_signer_lifecycle.py` | **108 PASS**, 13.16 s, zero skip | Rzeczywiste early COMMIT/new xid i role-switch odmowy, kwalifikacja owner/principal/ACL/physical identity, catalog drift, completion blocker i odczytowe signer facts. |
| Protected catalog PUBLIC EXECUTE probe + direct production cross-role port denial + closed bootstrap/no raw writer (`test_root_proof_atomic_composition_integration.py`, `test_production_root_lifecycle_races.py`) | **4 PASS**, 0.77 s, zero skip | Unauthorized helper/port i bootstrap nie dają ogólnego prawa zapisu. Osobny klaster zapobiega interferencji PUBLIC grant testu z atomic qualification. |
| Wybrane `test_postgresql_atomic_root_proof_commit.py`, wspólna genuine lineage i dwa oryginalnie przygotowane kandydaty | **19 PASS**, 188.04 s, zero skip | Exact retry/original bytes, rollback i NOT_FOUND→indeterminate, backend termination, COMMIT response loss i read-only byte-identical recovery, actual receive disconnect, ACL/membership/source drift, odmowa direct/legacy standalone BIND/history i brakującego STAGED oraz zmienionych recovery identities. |
| Ruff 0.14.11, pełny scope CI; nowy test także bezpośrednio | **PASS** | Lint bez zmiany reguł. Trzy istniejące ostrzeżenia legacy noqa bez nowych suppressions. |
| Dokładny `python -m pre_commit run ruff-format --all-files` | **PASS**, bez churn | Niezależny format gate; nowy test objęty także targeted format check i końcowym hookiem po staging. |
| `python -m mypy`; `python -m mypy --platform win32` | **PASS**, po 338 plików | Niezmieniony configured scope; Win32 to analiza typów na Linux, nie wykonanie Windows. |
| Semgrep CE 1.179.0 `.semgrep/cryptohunter.yml` + rule fixtures | **PASS**, 5/5 fixtures | Istniejące blokujące custom security rules bez wyłączeń. |
| Betterleaks 1.9.0, offline dir/baseline, binary zweryfikowane pinned CI SHA256 i upstream checksum | **PASS**, zero nowych findings | Pełny czysty snapshot plików repozytorium z finalnymi nowymi plikami; bez lokalnych venv/cache i bez dodanych wyłączeń. |
| Diff canonical JSON/Markdown/historycznych freeze względem bazowego `main`, `git diff --check` | **PASS** | Żaden canonical/frozen artefakt nie został nadpisany; zmiany ograniczone do ADR, osobnej propozycji i regresji. |

Testy wykonano z `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 DUDZIAN_TEST_MODE=1
DUDZIAN_ALLOW_LONG_POLL=0`, jawnie `-p asyncio -p pytest_cov.plugin` oraz
markerami `not hardware and not destructive and not network`; architecture
run dodatkowo wykluczał `external_postgresql`. DSN-y testów kierowały wyłącznie
do lokalnych klastrów. Nie zmieniono timeoutów, coverage, markerów ani scope
istniejącego CI. Pierwszą próbę atomic zatrzymał brak `keyring` w minimalnym
venv; po instalacji deklarowanych zależności wykonano te same niezmienione
19 przypadków. Do mypy użyto NumPy 1.26.4 zgodnie z kanonicznym desktop lock,
bez zmiany konfiguracji lub źródeł.

**FULL_CI_PENDING.** Nie uruchomiono pełnego wielogodzinnego CI, wszystkich
58 atomic cases, pełnego hardware/native-keyring/Windows qualification,
produkcyjnej instalacji ani nowego protokołu 2PC. W lokalnym wybranym atomic
scope nie wykonano kontrolowanego restartu serwera; crash/response-loss
wyniki nie zastępują tego osobnego scenariusza. Nie są to twierdzenia PASS.
Local tests nie dowodzą independent custody, rzeczywistego ingress,
whole-host rollback resistance ani zgody kontraktowej.


## Brakujące dowody i decyzje do zatwierdzenia

Właściciel projektu/kontraktu musi osobno:

1. Przyjąć albo odrzucić dokładny zakres addendum C i wariant A; do tej decyzji
   status pozostaje `CONTRACT_RECONCILIATION_PENDING_APPROVAL`.
2. Ustalić wersję i authority zatwierdzającą kontrakt, historyczne źródła,
   nowe parity/freeze oraz procedurę migracji; nie nadpisywać dawnych bajtów.
3. Zatwierdzić production semantic issuer ingress, accountable transaction
   owner i minimalne owner-owned helper API z zamknięciem dawnych write ports.
4. Rozstrzygnąć niezależną custody completion authority/trust anchor,
   produkcyjne root/history custody, cross-role keyring i namespace isolation,
   activation/succession oraz credential lifecycle fencing. Sam ADR i lokalny
   journal verifier tych braków nie usuwają.
5. Zatwierdzić odrębne domeny checkpoint/rollback i admin, recovery runbook,
   migrację istniejących danych, backup/restore/failover oraz zasady rozstrzygania
   nieznanego outcome bez nowej emisji. Wariant B wymagałby dodatkowo zatwierdzenia
   całego distributed recovery/visibility protocol.

Przyszłe zniesienie blokady wymaga **łącznie**:

- osobno zatwierdzonego kontraktu/versioned addendum z kontrolami parity;
- zaimplementowanego i niezależnie przejrzanego produkcyjnego ingress oraz
  coordinatora, bez caller-owned providers, DSN, SQL identifiers/callbacków,
  flags i test credentials;
- kwalifikacji rzeczywistej instalacji: direct principals/OID, OS i peer auth,
  katalog/grants/defaults/pośrednie definers, immutable byte retention,
  schema versions i zamknięcie standalone BIND/history;
- linearizacji produkcyjnych requester/claimant i root/history/trust lifecycle
  fences w finalnej transakcji, z niezależnie uwierzytelnioną custody completion;
- rzeczywistych fault/race/restart/response-loss tests z dowodem jednego
  winnera, całego write set i exact recovery; scenariusz różnych legalnych
  attempts tego samego entitlement musi odpowiadać zatwierdzonej kardynalności,
  a test dwóch exact deliveries lub różnych entitlement go nie zastępuje;
- niezależnego checkpoint anti-rollback, monotonic history recovery i rozłącznych
  transakcji CHA/freshness/account, bez utożsamiania issuer COMMITTED z genesis;
- osobnej oceny i jawnej decyzji o produkcyjnej kwalifikacji. Zakończenie CI,
  sukces isolated testu albo aprobata addendum same nie dają GO.

Następny krok implementacyjny po decyzji kontraktowej: zaprojektować minimalny
produkcyjny ingress i finalną kompozycję nad niezależną installed lifecycle
authority, z wyłącznym posiadaniem connection/transaction przez trusted
coordinator i testami jej pełnego write set. Produkcyjna emisja wymaga także
usunięcia pozostałych niezależnych blokad; S9-RPI-04 ich nie omija.
