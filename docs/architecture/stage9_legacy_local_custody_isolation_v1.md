# Stage 9: Legacy Local Custody Isolation V1

Ten kontrakt ogranicza administracyjne API lokalnego custody dla
`ROOT_PROOF_SIGNING` i `HISTORY_ATTESTATION_SIGNING`. Obie role mogą zmieniać
lifecycle wyłącznie przez kwalifikowany protokół Durable Custody Lifecycle V2,
z enrollment SQL Contract V3. Nie implementuje planned rotation ani
aktywacji następcy tych signerów.

`PRODUCTION_LIFECYCLE_ISOLATION=BLOCKED` i
`ATOMIC_ISSUER_READY=NOT_READY` pozostają obowiązkowe. Admission w Pythonie
nie stanowi niezależnej granicy wobec procesu mogącego zmienić kod, pliki
custody lub zawartość native keyring.

## Źródła i wersja kontraktu

Audyt oparto na źródłach repozytorium oraz celowanych wyszukiwaniach `rg`.
Codebase Memory nie było dostępne; nie deklarujemy kompletności indeksu
strukturalnego ani pokrycia dowolnych zewnętrznych skryptów administratora.
Sprawdzono:

- [local custody](../../bot_core/local_signing_custody.py), oba publiczne
  lifecycle wrappers, shared mutator, generic writers, provisioning oraz
  wszystkie helpery freshness rotation/recovery;
- [durable local evidence](../../bot_core/local_signing_lifecycle_evidence.py),
  INTENT/COMPLETE, odtworzenie po przerwaniu oraz verification receipt;
- [PostgreSQL lifecycle coordinator](../../bot_core/postgresql_custody_lifecycle.py),
  [schema](../../bot_core/postgresql_custody_lifecycle_schema.py) i
  [qualification](../../bot_core/postgresql_custody_lifecycle_qualification.py);
- [atomic commit](../../bot_core/postgresql_atomic_root_proof_commit.py),
  SQL enrollment guards, odrębne
  [CHA issuance custody](../../bot_core/cha_root_proof_signing_custody.py)
  i [Windows filesystem qualification](../../bot_core/windows_custody_filesystem.py).

Local admission contract ma wersję 1. Nie zmienia on
`cryptohunter.production-local-signing-custody` schema version 1,
signed credential identities, istniejących operation descriptors ani
journal schema version 2. Nie dodaje lokalnego znacznika enrollment.
Autorytatywnym enrollment pozostaje kwalifikowane PostgreSQL
`lifecycle_credentials` wraz z immutable SQL Contract V3.

## Znalezione obejścia

Przed tą zmianą oba publiczne wrappers mogły wykonać dowolne legalne one-way
przejście lokalnego Root Proof/history credential bez zamknięcia DB fence.
Exact wrapper obejmował również retained credential. Wspólny
`_transition_exact_locked()` sprawdzał generation i graf przejść, lecz nie
sprawdzał autoryzacji. Jego bezpośrednie wywołanie miało ten sam skutek.

`_replace_record()` pozwalał zastąpić metadata bez jakiejkolwiek admission,
a `_write_new_record()` mógł opublikować nowe current/retained metadata bez
kwalifikowanego administratora. Bezpośrednie `_recover_rotation_locked()`,
`_retained_rotation_state()` i `_publish_or_preserve_lifecycle_state()` nie
miały własnej kontroli freshness role: ograniczenie było tylko w publicznym
wrapperze. W konsekwencji sam prefix `_` nie izolował tych ścieżek.

Ponadto `_authorize()` koordynatora PostgreSQL wydawał registry-authentic
`DurableCloseAuthorization` na podstawie descriptor, bez własnego pozytywnego
odczytu committed operation. Membership sprawdzane przez durable helper
nie wystarczało, jeśli taki issuer capability można było wywołać bez Phase A.

Celowane wyszukiwanie rzeczywistych wywołań poza testami znalazło
`PostgreSQLCustodyLifecycleAdministrator.mutate_custody()` → durable helper
→ shared mutator → `_replace_record()`. Dwa legacy wrappers nie mają
produkcyjnego callera w sprawdzonych źródłach, ale pozostają importowalnymi
API. CHA issuance administrator używa tego samego generic writer dla
odrębnego `custody.json`; jego lifecycle nie jest lifecycle Root Proof/history
issuer credential.

## Role i punkty wejścia

Skróty: RP = Root Proof Signing, HA = History Attestation Signing,
FA = Freshness Authority Finalization Signing, CP = CHA Freshness Proposer
Signing. Zapis „custody EX” oznacza istniejący wyłączny process lock:
POSIX `flock` lub Windows byte-range lock. Lock serializuje współpracujące
procesy; nie nadaje uprawnień OS ani DB.

| Punkt wejścia | Role i rzeczywisty caller | Lock | Wymagana capability / admission | Rzeczywiste uprawnienia i zakres |
| --- | --- | --- | --- | --- |
| `transition_local_signing_lifecycle()` | RP, HA, FA, CP; importowalne legacy API | custody EX | RP/HA: odmowa bez durable admission; FA/CP: dotychczasowe one-way reguły | Nie wymaga dostępu do klucza. Wcześniej sam dostęp do zapisu metadata wystarczał do zmiany state/generation. |
| `transition_local_signing_credential_lifecycle()` | Te same role; exact current lub retained identity | custody EX | Ten sam admission co dla current wrapper | Ochrona retained credential nie zależy od tego, czy jest bieżącym signerem. |
| `_transition_exact_locked()` | Oba wrappers i durable helper | Lock posiadany przez callera | RP/HA: sealed exact mutation admission z genuine close, INTENT i aktywną custody EX lease; FA/CP: dotychczasowy CAS | Jeden mechanizm dopuszcza jednocześnie state i generation; zachowuje immutable identity, key version, public material i secret reference. |
| `_replace_record()` | Shared mutator; journal COMPLETE; freshness rotation/recovery; CHA issuance custody | RP/HA: wymaga genuine active custody EX lease; helper nie nabywa zagnieżdżonego lock | Zapis RP/HA rozpoznany przez destination i existing/candidate metadata wymaga sealed exact mutation admission; journal i inne role zachowują własne protokoły | Kontrola poniżej wrappers zapobiega obchodzeniu admission przez drugi wrapper, direct generic writer albo pominięcie lock. |
| `_write_new_record()` | Initial provisioning; INTENT; freshness retained/rotation; CHA staged custody | Initial RP/HA current wymaga genuine active custody EX lease; helper nie nabywa zagnieżdżonego lock | Initial RP/HA current wymaga exact native administratora; nie nadaje enrollment ani successor rights; RP/HA retained/rotation publication nie jest legacy administracją | New-file publication nie może obejść ochrony przez pominięcie `_replace_record()`. |
| `provision_local_signing_authority()` | RP, HA, FA, CP; offline administrator | custody EX | Exact `NativeKeyringSigningSecretAdministrator`, authority scope; bez zmian dla FA/CP | Istniejący credential może być idempotentnie odczytany. Nowy local ACTIVE/generation 1 nie staje się enrolled V2 successor. |
| `rotate_local_freshness_signing_authority()` | FA, CP | custody EX | Exact scoped native administrator, operation-bound R0–R7 protocol | Planned rotation jest odrębną authority. RP/HA odrzucone; nie implementujemy ich rotacji. |
| `_recover_rotation_locked()` oraz retained publication helpers | Tylko FA, CP po własnej kontroli role | Lock posiadany przez freshness wrapper | Freshness role i existing exact rotation records/secret | Odrzucają RP/HA także przy bezpośrednim wywołaniu. |
| `durable_transition_local_signing_credential_lifecycle()` | RP, HA; Phase B koordynatora | custody EX | Genuine exact `DurableCloseAuthorization`, operation descriptor, before/after INTENT | CAS i fsync pozostają lokalne; DTO, string, hash albo samo operation ID nie dają prawa do zapisu. |
| `begin_close()` / `resolve_close()` | RP, HA; lifecycle administrator | DB fence `FOR UPDATE`; bez custody lock | Qualified direct-login admin, physical DB authority, immutable enrollment, exact request | Capability wynika z pozytywnej committed obserwacji. Nie może powstać przez sam constructor DTO lub niekwalifikowany issuer helper. |
| `mutate_custody()` | RP, HA; Phase B | DB resolution kończy się przed custody EX | Qualified deployment oraz ponowna positive close resolution | Brak DB lub authority drift blokują rozpoczęcie lokalnej nieodwracalnej zmiany. |
| `publish_completion()` | RP, HA; Phase C | custody EX → DB fence | Genuine reread COMPLETE receipt i osobny administrator HMAC key | COMMIT publikuje VERIFY_ONLY/REVOKED, nigdy ACTIVE. Writer nie ma tego credential. |
| `migrate_custody_lifecycle_v2()` / SQL V3 upgrade | RP, HA; osobny bootstrap | DB decisions → fences locks | Qualified bootstrap principal, exact physical DB identity, reviewed predecessor catalog | Tworzą/potwierdzają enrollment, nie wykonują automatic local startup migration. |
| `OfflineIssuanceCustodyAdministrator` | CHA requester/claimant, poza RP/HA/FA/CP | custody EX | Własny staged/registry activation contract | Odrębny semantic signing role i format metadata; shared filesystem helper nie łączy jego uprawnień z issuer lifecycle. |

## Admission i downgrade protection

Nie da się bezpiecznie wybrać trybu legacy na podstawie braku usuwalnego
lokalnego pliku. Admission V1 stosuje zatem konserwatywną regułę: każde
lokalne RP/HA lifecycle mutation wymaga DB-first authorization, również
gdy legacy credential nie był jeszcze enrolled. Runtime nie wybiera
„legacy vs enrolled” przez argument, środowisko, thread-local ani nazwę
procesu. Dla FA/CP zachowano poprzedni kontrakt administracyjny.

Koszt kompatybilności jest jawny: starsza, offline administracja lifecycle
RP/HA bez V2 enrollment już nie wykonuje mutacji. Może odczytać existing
credential. W celu dalszej administracji operator przeprowadza jawne
kwalifikowane enrollment istniejącego credential; nie odtwarza klucza i
nie wymienia podpisanych identities.

Brak, uszkodzenie lub rollback lokalnego journal nie może przywrócić legacy
uprawnień: odmowa nie jest uzależniona od obecności journal. Existing INTENT
lub COMPLETE jest evidence konkretnej operacji, a nie przełącznikiem trybu.
Niepoprawna evidence blokuje durable recovery i publication. Reserved
destination RP/HA musi być sprawdzona także wtedy, gdy candidate metadata
jest uszkodzone albo próbuje zmienić rolę/format; decoder failure nie jest
przepustką do niezabezpieczonego generic writer.

Generic writer nie importuje modułu durable evidence ani administratora
PostgreSQL. Zachowuje to frozen import isolation lokalnego custody.
Custody definiuje sealed `_LocalSigningMutationAdmission` oraz słaby
process-local membership registry. Jego binding obejmuje exact destination,
exact before/after metadata, genuine aktywną exclusive-lock lease oraz live
validator konkretnego durable close. Sam constructor DTO ani caller-supplied
callable nie może wpisać membership przez dostępne API mintowania.

Durable evidence wystawia admission dopiero po własnej weryfikacji genuine
authorization i exact INTENT, wewnątrz rzeczywiście nabytego custody EX lock.
Najniższy writer sprawdza membership, destination, metadata predecessor,
canonical successor bytes, aktywną
lease i positive live DB validation przed zapisem. Lease pochodzi z
`_custody_lock()`, jest związana z procesem i wątkiem wykonującym operację
i traci ważność przy wyjściu z context, przed zwolnieniem lock. Weryfikacja
właściciela lease nie jest thread-local override ani flagą „authorized”.
Otrzymany token nie nadaje prawa do innego credential, późniejszej generation,
innej ścieżki ani mutacji bez lock. Lease po fork/restart nie jest autoryzacją.

Registry jest zaufanym stanem implementacji w tym samym interpreterze,
nie granicą izolacji procesowej. Proces mogący bezpośrednio modyfikować
module globals, registry, funkcje lub kod może podrobić ten stan. Takie
przejęcie należy do modelu B i pozostawia produkcyjny blocker.

Legalne admission łączy exact authorized descriptor z kwalifikowanym
`lifecycle_credentials`: custody directory, security profile/trust domain,
credential identity, signing role, key version, public key material,
predecessor generation i physical PostgreSQL authority. SQL Contract V3
jest weryfikowany przez deployment qualification i pinned immutable catalog.
Nie zmieniamy istniejącego descriptor ani SQL contract, by dodać
caller-controlled pole „authorized”. Fizyczna stabilność ścieżki custody
podlega istniejącym kontroli ownership, bezpiecznym descriptor i journal
directory identity; sama ścieżka w enrollment nie dowodzi niezależnej
ochrony przed full-host rollback.

## Zachowany protokół A/B/C

1. A: kwalifikowane PostgreSQL zamyka exact enrolled fence i kończy
   `COMMIT PENDING_CLOSED` przed wydaniem positive close capability.
2. B: lokalny administrator zapisuje/fsyncuje exact INTENT, wykonuje
   jednorazowy CAS, fsyncuje metadata i zapisuje/fsyncuje COMPLETE.
   Shared mutation boundary wymaga autentycznego close oraz zgodności
   INTENT before/after. Obie zmiany, state i generation, mają ten admission.
3. C: administrator ponownie odczytuje COMPLETE i exact metadata pod custody
   lock, uwierzytelnia evidence i kończy DB COMMIT VERIFY_ONLY/REVOKED.

Przerwanie po A zostawia zamknięty fence i nie daje legacy prawa do B.
Przerwanie po local CAS zostawia INTENT, którego exact recovery uznaje
wyłącznie wskazany before albo after. COMPLETE po restarcie wymaga
ponownej kwalifikacji DB i odczytu evidence; nowe process membership
capabilities nie powstają z zapamiętanego hasha. Retry nie zmienia
operation bytes, public material ani credential identity. Późniejszy
legalny descendant nie jest completion poprzedniej operacji.

Phase A i DB resolution nie trzymają custody lock podczas rozpoczęcia
DB transaction; kończą się przed B. Phase C zachowuje istniejącą kolejność
custody lock → DB fence lock. Nie dodajemy odwrotnej kolejności ani DB
transaction podczas podpisywania prywatnym kluczem. Concurrent legacy RP/HA
transition przegrywa admission; FA/CP korzystają z istniejących locks.

Committed Root Proof bytes, history/BOUND linkage i SQL V3 enrollment
guards pozostają bez zmian. PENDING_CLOSED blokuje nowe BIND, natomiast
wcześniejszy committed Root Proof zachowuje autorytatywność. Nowy local
ACTIVE credential nie może odziedziczyć istniejącego fence/enrollment:
SQL V3 porównuje faktyczne signer identities i public material, a V2
blokuje legacy fence provisioning/reaktywację.

## Migracja istniejących danych

Nie ma nowej lokalnej migracji danych ani nowego enrollment format.
Local admission V1 jest zmianą polityki kodu. Zwykły start aplikacji nie
zmienia custody metadata, SQL enrollment ani journal. Istniejące
credentials, seed references, generation i signed bytes są zachowane.

Dla już kwalifikowanego SQL Contract V3 nie jest potrzebne nowe DDL.
Dla starszego V2 pozostaje jawny
`upgrade_custody_lifecycle_signer_enrollment_v3()` oraz migration ID
`DURABLE_CUSTODY_V2_SIGNER_ENROLLMENT_V3`. Funkcja kwalifikuje autentyczny
predecessor, konserwuje application/lifecycle rows i MAC secret, a dla
już V3 wykonuje read-only qualification. Utrata odpowiedzi po COMMIT
wymaga istniejącej bootstrap requalification, nie automatycznej naprawy.

Dla V1 operator używa istniejącego `migrate_custody_lifecycle_v2()` z
exact istniejącym custody binding i SQL Contract V3. Bez tego enrollment
RP/HA administracja pozostaje zamknięta. Nie wymaga to reinstalacji,
destrukcyjnego odtworzenia custody ani planned rotation.

## Modele przeciwnika i pozostała granica

| Model | Co rzeczywiście egzekwuje implementacja | Czego nie gwarantuje |
| --- | --- | --- |
| A: zwykły caller starszego API | Odmowa RP/HA na wrappers i niższych mutation/publication boundaries; odmowa fake/stale capability i exact-request mismatch; brak fallback po utracie journal; SQL enrollment uniemożliwia podmianę successor identity. | Nie twierdzimy, że sama nazwa prywatnego helpera ogranicza dostęp. Wszystkie wskazane mutation entrypoints wymagają admission. |
| B: skompromitowany proces z dostępem do tych samych plików/native keyring | Qualified DB writer nadal nie uzyskuje admin EXECUTE/DML ani owner principal przez local API. POSIX/NTFS qualification i native locks wykrywają część unsafe substitutions i serializują współpracujące procesy. | Ten proces może ominąć Python, nadpisać/delete/rollback pliki, użyć klucza lub zmienić membership registry/kod. Nie ma oddzielnego OS principal custody-admin, obowiązkowego broker ACL ani independently enforced monotonic state. |
| C: bootstrap administrator, PostgreSQL superuser lub full-host rollback | Standardowe qualified principals, catalog pins i zachowanie immutable rows opisują nieprzejętą granicę; zwykłe drift/unavailability daje fail-closed. | Compromised bootstrap/superuser może zmienić trusted enrollment/catalog/secret. Full-host rollback nie jest wykrywany bez niezależnej domeny anti-rollback. |

Na POSIX metadata należą do bieżącego effective UID, a newly-created records
mają mode 0600. Ta kontrola nie odróżnia administratora od przejętego
runtime pod tym samym UID. Native keyring także dziedziczy uprawnienia
principal; read-only facade nie odbiera innemu kodowi uprawnień OS.
Na Windows trusted SID set obejmuje bieżącego użytkownika, SYSTEM,
Administrators i istniejące service principals. ACL qualification odrzuca
niezaufanych writers, lecz nie dowodzi read-only access runtime względem
administratorów tej samej custody authority.

Produkcyjny blocker wymaga osobnej, niezależnie egzekwowanej granicy:
wydzielonego custody administrator/brokera, runtime bez write/delete/replace
ani key-management rights, weryfikowanego exact DB-first admission w tej
granicy i odpowiedniej independent anti-rollback authority, jeśli taka
odporność ma być deklarowana. Ten PR nie ustanawia tej granicy.

## Wymagane dowody regresyjne

Testy integracyjne mają używać rzeczywistego PostgreSQL 16, niezależnych
procesów i native local locks. Każda odmowa ma zachować fizyczne bytes
metadata oraz rzeczywisty DB fence; same mock assertions nie są dowodem
DB-first bezpieczeństwa. Należy sprawdzić oba wrappers, shared mutator,
direct writers, retained credential, counterfeit capability, stale
operation/generation, journal utratę/uszkodzenie i administrator restart.

Legalny protokół ma pokrywać ACTIVE → VERIFY_ONLY, ACTIVE → REVOKED,
VERIFY_ONLY → REVOKED, przerwanie przed/po local CAS i COMPLETE, exact
retry, nieaktywującą Phase C, retained old Root Proof/history oraz
odrzucenie nowego BIND po PENDING_CLOSED. Osobne FA/CP regression zachowują
provisioning, transitions, planned rotation i recovery bez zmiany semantyki.
Raport wykonania testów i finalne readiness wartości należą do raportu PR;
ten dokument nie zastępuje wyników wykonanych testów.
