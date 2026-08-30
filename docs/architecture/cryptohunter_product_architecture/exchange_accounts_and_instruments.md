# CryptoHunter M0.5 — Exchange Accounts and Instruments Contract

Status: `closed`

Ten dokument opisuje maszynowy kontrakt `exchange_accounts_and_instruments.json`. JSON jest źródłem prawdy. M0.5 jest kontraktem architektonicznym: nie wdraża runtime, adapterów giełdowych, endpointów, sekretów, routingu strategii ani order execution.

## Identity layers

`exchange_id`, `exchange_account_id` i `instrument_id` są osobnymi poziomami identity. `exchange_id` pochodzi z zamkniętego build-time registry, nie z configu, CLI, environment variable ani IPC. `exchange_account_id` (`xacc`) identyfikuje lokalne konto należące do jednego Portfolio i jednego tuple exchange/environment/market type/subaccount. `instrument_id` (`instr`) identyfikuje venue-specific instrument w Workspace. Display name, adapter class name, alias i `primary` nie są trwałą identity.

Symbol nie jest globalną identity. Instrument identity tuple to dokładnie `exchange_id`, `environment`, `market_type`, `venue_symbol`. `venue_symbol` jest exact venue identifier: bez case-fold, trimowania prowadzącego do kolizji i Unicode normalization. BTCUSDT na różnych giełdach, w TESTNET/LIVE albo SPOT/PERPETUAL oznacza różne instrumenty.

## Rejestry

Market types: `SPOT`, `MARGIN`, `PERPETUAL`, `DELIVERY_FUTURES`, `OPTIONS`. Instrument types: `SPOT_PAIR`, `MARGIN_PAIR`, `PERPETUAL_CONTRACT`, `DELIVERY_FUTURE`, `OPTION`. Dozwolone pary: SPOT→SPOT_PAIR, MARGIN→MARGIN_PAIR, PERPETUAL→PERPETUAL_CONTRACT, DELIVERY_FUTURES→DELIVERY_FUTURE, OPTIONS→OPTION. Nieznane typy fail-closed i wymagają nowej wersji kontraktu lub kontrolowanej migracji registry.

Exchange Registry nie ma mandatory exchange (`null`), primary exchange ani implicit default exchange. Runtime config nie rozszerza registry. Unknown exchange_id kończy się `UNKNOWN_EXCHANGE_ID`. Capabilities adaptera i ExchangeCapabilities nie rozszerzają ProductCapabilities M0.4.

## ExchangeAccount

ExchangeAccount ma ID `exchange_account_id` z prefiksem `xacc`, parent `Portfolio` i identity fields: `exchange_account_id`, `portfolio_id`, `exchange_id`, `environment`, `market_type`. Zmiana Portfolio, exchange_id, environment albo market_type wymaga nowego konta albo jawnej migracji; Testnet i Live nigdy nie dzielą konta. Usunięcie jest logicznym `RETIRED`, nie hard delete.

Lifecycle (`DRAFT`, `ACTIVE`, `DISABLED`, `RETIRED`), connection state (`DISCONNECTED`, `CONNECTING`, `SYNCHRONIZING`, `ONLINE`, `DEGRADED`, `BLOCKED`) i execution authorization (`READ_ONLY`, `ORDER_ENTRY_ALLOWED`, `BLOCKED_BY_POLICY`, `BLOCKED_BY_KILL_SWITCH`, `BLOCKED_BY_LEASE`, `BLOCKED_BY_RECONCILIATION`) są trzema niezależnymi osiami. External identity mismatch blokuje konto przez `BLOCKED` i `BLOCKED_BY_RECONCILIATION`.

## CredentialProfile i sekrety

CredentialProfile ma ID `credential_profile_id` z prefiksem `cred`, parent ExchangeAccount, `saas_sync_candidate=false` i zawiera tylko metadata oraz secure-store reference. Kontrakt zabrania API secret, secret key, password, PIN, biometric template, private key, access token, session token, odszyfrowanej wartości credential oraz zaszyfrowanego sekretu transportowanego w tym kontrakcie.

Machine-readable lifecycle registry CredentialProfile to dokładnie `ACTIVE`, `RETIRED` (nie dziedziczy stanów ExchangeAccount). `ACTIVE` wymaga `retired_at_utc=null`; `RETIRED` wymaga canonical `retired_at_utc >= created_at_utc`. Nullable fields to dokładnie `public_key_identifier`, `rotated_from_credential_profile_id`, `retired_at_utc`; rotation ID, gdy obecne, jest canonical ID z prefiksem `cred`. Jest to intrinsic single-record consistency bez transition graphu i bez wall-clock authority; istniejący contextual lineage timestamp algorithm pozostaje odrębny i nieosłabiony.

Profile scope musi równać się environment konta, exchange_id musi pasować do konta, TESTNET nie przyjmuje LIVE credentials, LIVE nie wiąże się z TESTNET, PAPER nie wymaga credentials. `WITHDRAW` jest zabronione i blokuje readiness; `INTERNAL_TRANSFER` nie jest używane przez M0.5; brak `PLACE_ORDERS` wymusza `READ_ONLY`. Same credentials nie tworzą execution authorization ani ProductCapabilities.

## External identity i account capabilities

External account identity ma stany `UNVERIFIED`, `VERIFYING`, `VERIFIED`, `MISMATCH`, `UNAVAILABLE`. Verified identity obejmuje exchange, environment, market type, venue account identifier, subaccount, account type, observed permissions, timestamp i adapter/version source. Account capability snapshot jest immutable, wersjonowany i może tylko dodatkowo ograniczać ProductCapabilities M0.4.

## Instrument metadata i asset references

Nie powstaje trwały byt Asset. M0.5 używa value objects base/quote/settlement z polami `venue_asset_code`, `canonical_display_code`, `asset_namespace`, `mapping_status`. `EXACT` i `EXPLICIT_ALIAS` są jawne; `AMBIGUOUS` i `UNKNOWN` fail-closed. Parsing symbolu typu BTCUSDT jest zabroniony; base/quote/settlement pochodzą z catalog metadata.

Instrument record zawiera identity, typ, exact symbol, display symbol, asset references, trading status, tick/step/limity, pola pochodnych, catalog snapshot, metadata version, czasy ważności i adapter family source. Wartości finansowe są exact decimal strings: bez float, scientific notation, NaN i Infinity; tick i step są dodatnie; min nie przekracza max; null limit nie oznacza zera.

## Derivatives i katalog

SPOT_PAIR/MARGIN_PAIR nie mają contract_size, expiry, strike ani option_side. PERPETUAL wymaga contract_size i settlement asset, nie ma expiry/strike/option_side i używa LINEAR albo INVERSE. DELIVERY_FUTURE wymaga contract_size, settlement i expiry. OPTION wymaga contract_size, settlement, expiry, strike oraz CALL albo PUT. Nieprawidłowe kombinacje fail-closed.

Instrument catalog snapshot jest immutable i versioned: ma catalog_snapshot_id, exchange/environment/market, adapter family/version, observed/effective/stale timestamps, instrument_ids, content_hash, previous_snapshot_id i status `VALID`, `PARTIAL`, `STALE` albo `REJECTED`. STALE blokuje wykonanie fail-closed; PARTIAL nie usuwa automatycznie brakujących instrumentów; refresh nie nadpisuje historii.

## Alias migration i TradingUniverse

Symbol alias jest scoped przez exchange_id, environment i market_type, nie jest durable identity i nie zastępuje instrument_id. AMBIGUOUS alias jest odrzucany; legacy symbol string wymaga jawnej migracji i nie może być mapowany po podobieństwie tekstu.

TradingUniverse ma ID `trading_universe_id` z prefiksem `univ`, parent ExchangeAccount i versioned immutable membership. ACTIVE universe nie może być puste, instrumenty muszą być unikalne, istnieć w zwalidowanym katalogu i pasować do exchange/environment/market type konta. DELISTED i UNKNOWN nie wchodzą do nowej ACTIVE wersji. Universe nie definiuje strategii ani routingu; przypisanie strategii jest odłożone do M0.6.

## Current edition, denial i audit

M0.5 zachowuje M0.4: dokładnie PAPER, TESTNET, LIVE; SAFE_LOCAL_ONLY fail-closed; brak Testnet-to-Live fallback; brak odczytu Live secretów. Current edition blokuje wszystkie operacje LIVE kodem `LIVE_BLOCKED_BY_EDITION`; legacy Live records mogą być tylko wykryte i quarantined bez użycia sekretów.

Denial codes i audit events są zamkniętymi registry w JSON. CoreHost jest jedynym authority stosującym trwałe/runtime mutations; DesktopShell i TrayAgent tylko żądają, Bootstrapper nie zarządza kontami ani instrumentami, a GUI nie zawiera logiki giełdowej. Odłożone do M0.6+ pozostają strategie, routing market data i execution, order lifecycle, idempotency, ledger, risk hierarchy, kill switch implementation, ExecutionLease implementation, persistence, SaaS sync i GUI.

## M0.5 FIX — fail-closed validators

Current-edition Live operability denial jest osobną polityką: istniejący legacy Live record może być wyłącznie quarantined/blocked, natomiast operacyjne Live account, Live ONLINE, Live ORDER_ENTRY_ALLOWED, aktywny Live CredentialProfile, secret read, endpoint resolution, adapter creation, private connection, Live catalog refresh, TradingUniverse activation for execution i fallback TESTNET→LIVE kończą się `LIVE_BLOCKED_BY_EDITION`.

Operation-specific matrix w JSON zastępuje placeholdery: każda operacja ma exact required inputs, allowed lifecycle/connection/authorization states, preconditions, complete denial codes, success audit event, denial event by code i forbidden side effects on denial. Success vs denial audit events są rozdzielone, a per-event audit payload schemas wymagają identifier fields, context fields, result enum, nullable albo required denial_code i `secret_fields_forbidden=true`.

Canonical decimal grammar to regex `^(0|[1-9][0-9]*)(\.[0-9]*[1-9])?$`: bez plusa, minusa, leading zeros, `.5`, `1.`, trailing fractional zeros, underscore, whitespace, scientific notation, NaN, Infinity ani typów float/int. `0` jest kanonicznym zerem tylko tam, gdzie zero ma sens; positive fields muszą być większe od zera.

Strict timestamps są parsowane semantycznie jako UTC `Z`, z opcjonalną frakcją 1–9 cyfr; invalid dates fail closed, `observed_at_utc <= effective_at_utc`, `effective_at_utc < stale_after_utc`, `now < stale_after_utc` oznacza fresh, a equality oznacza stale.

Deterministic content hashes używają SHA-256, domain separatorów, JSON canonicalization z sortowanymi kluczami, UTF-8, lowercase hex oraz canonical ordering instrument IDs/source catalog IDs. Identity collision policy wymaga jeden identity tuple → dokładnie jeden instrument_id i jeden instrument_id → dokładnie jeden identity tuple.

Capability snapshot versioning dodaje version, previous_snapshot_id, status, observed/effective/stale timestamps i deterministic content hash. Catalog i universe lineage są jawne: katalog waliduje previous snapshot lineage, a TradingUniverse używa wyłącznie własnych `source_catalog_snapshot_ids`, monotonic version i `previous_version_id`.

## M0.5 FIX 2 — operation enforcement and audit schemas

M0.5 nie posiada caller-controlled operation result: żądanie nie może ustawić `result`, `denial_code`, `success`, `allowed`, `force_denial` ani `authority`, aby zmienić decyzję walidatora. Operation dispatcher działa po zamkniętym registry operacji, sprawdza exact request schema, wybiera operation-specific pure validator, ocenia rzeczywiste rekordy i dopiero z obliczonego denial code wybiera success albo denial audit event. Nieznana operacja fail-closed do `UNKNOWN_OPERATION` i `UNKNOWN_OPERATION_REJECTED`, bez KeyError.

Live update denial obowiązuje dla każdej normalnej operacji konta. `UPDATE_ACCOUNT` dla Live nie może zachować ani ustawić `ACTIVE`, `CONNECTING`, `SYNCHRONIZING`, `ONLINE`, połączeniowego `DEGRADED`, `ORDER_ENTRY_ALLOWED`, aktywnego CredentialProfile, secret read, endpoint resolution, adapter creation, private connection, aktywacji katalogu ani aktywacji TradingUniverse. Legacy Live record jest obsługiwany wyłącznie przez maintenance quarantine path `QUARANTINE_LEGACY_LIVE_RECORD` albo retirement: tylko DISABLED/RETIRED, DISCONNECTED/BLOCKED, BLOCKED_BY_POLICY i bez sekretów, endpointów, adapterów ani connection.

TradingUniverse activation wymaga account readiness: konto musi mieć zamknięty schema, istnieć, pasować `exchange_account_id`, być `ACTIVE`, nie być DISABLED/RETIRED, mieć TESTNET identity `VERIFIED`, dozwolony connection state, dozwoloną authorization, potwierdzone readiness, spełnione credential/capability requirements i wszystkie wskazane source catalogs `VALID` oraz fresh. Każdy wynik `ok=false` z account operability jest propagowany fail-closed.

Audit schemas mają dokładnie jeden record per event name. `set(schema event names) == set(audit_event_registry)`, required identifier/context fields są unikalne, success event wymaga `denial_code=null`, denial event wymaga `denial_code` należącego do `allowed_denial_codes`, a wszystkie schematy mają `secret_fields_forbidden=true`.

Nested-type fail-closed obejmuje listy unikalnych stringów w registry, credential permissions i external identity permissions. Venue symbols są opaque exact identifiers: `venue_symbol` jest jedynym polem identity, nie ma `venue_symbol_exact`; validator nie wykonuje trim, case-fold ani Unicode normalization i akceptuje lowercase/mixed-case, jeśli katalog venue dostarczył taki exact symbol.

Canonical decimal validation używa pełnego dopasowania `re.fullmatch()` do regexu `^(0|[1-9][0-9]*)(\.[0-9]*[1-9])?$`, więc trailing newline, carriage return, tab oraz każdy leading/trailing whitespace są odrzucane. Credential rotation lineage wymaga mapy poprzedników: poprzedni profil musi istnieć, należeć do tego samego konta/exchange/environment, być RETIRED, nie tworzyć cyklu, a tylko jeden nowy profil może być aktywnie związany z kontem. Catalog binding wymaga zgodności adapter family z Exchange Registry, niepustej wersji adaptera, zgodności instrument.catalog_snapshot_id i source_adapter_family_id oraz poprawnego previous-snapshot binding bez self-reference/cykli.


## M0.5 FINAL FIX — complete dispatch and lineage

Request używa wyłącznie transportowe IDs, hash i listy: pola `_id` są stringami, `source_catalog_snapshot_ids` jest listą stringów, `content_hash` jest canonical lowercase SHA-256, a patch jest zamkniętym obiektem mutable fields. Trusted validation context jest oddzielną granicą i zawiera mapy `accounts_by_id`, `credential_profiles_by_id`, `external_identity_snapshots_by_account_id`, `universes_by_id`, `catalogs_by_id`, `instruments_by_id`, `previous_profiles_by_id`, `previous_catalogs_by_id`, `active_universes`, `active_profile_ids` i readiness/capability context.

Każda operacja ma pełny handler: CREATE_ACCOUNT, UPDATE_ACCOUNT, BIND_CREDENTIAL_PROFILE, VERIFY_EXTERNAL_IDENTITY, REFRESH_INSTRUMENT_CATALOG, ACTIVATE_TRADING_UNIVERSE, RETIRE_ACCOUNT, RETIRE_INSTRUMENT i QUARANTINE_LEGACY_LIVE_RECORD. Obowiązuje brak default-success: operacja bez handlera jest błędem kontraktu i fail-closed, a `UNKNOWN_OPERATION_REJECTED` jest zarezerwowane wyłącznie dla nieznanego operation ID.

Maintenance quarantine waliduje wszystkie target states, legacy flag i reason; nie uznaje samego obecnego bezpiecznego stanu za dowód bezpieczeństwa żądanych stanów. Audit mapping nie używa unknown-operation fallback dla znanych deniali, a denial schemas mają niepuste `allowed_denial_codes`. Credential lineage i catalog lineage przechodzą cały łańcuch przez visited set, wykrywają self-cycle, 2-node cycle i multi-node cycle. `readiness_confirmed` musi być literal bool readiness `true`; wszystkie nested request/context/record typy są nested fail-closed przed `.get()`, `set()`, `.startswith()`, indeksowaniem, iterowaniem i Decimal parsing.

Fraza kontrolna: trusted validation context pozostaje oddzielony od request i jest jedynym źródłem zaufanych rekordów.

## Final hardening: nested context, catalog scope i runtime audit

Każdy handler najpierw waliduje zamknięty `validation_context`: wszystkie wymagane mapy, w tym scoped `active_profile_ids_by_account_id`, muszą mieć właściwe typy, lista `active_universes` zawiera pełne rekordy, a rekordy zagnieżdżone są sprawdzane przed dostępem. Malformed request, context lub record kończy się kontrolowaną odmową bez wyjątku. Requesty mają operation-specific exact types: `reconnect` jest literalnym boolean, a zewnętrzne referencje konta są `null` albo niepustym stringiem.

Snapshot katalogu przechodzi walidację registry scope (enabled exchange, wspierane environment i market type), adapter binding, timestamp/hash/lineage oraz pełny validator każdego Instrument. Aktywacja TradingUniverse przekazuje pełne `previous_catalogs_by_id`; członkostwo wymaga obecności instrumentu w jednym z jawnie wskazanych i zaakceptowanych source catalogs oraz zgodnego `catalog_snapshot_id`. Sama obecność w globalnej mapie instrumentów nie wystarcza.

Runtime audit mapping nie ma fallbacku do niepowiązanego eventu. Każdy denial znanej operacji musi być zadeklarowany w matrix, mieć własne mapowanie i schema dopuszczający dokładnie ten kod. Niespójność samego kontraktu jest raportowana jako `CONTRACT_VALIDATION_MAPPING_ERROR` przez dedykowane `CONTRACT_VALIDATION_MAPPING_REJECTED`; `UNKNOWN_OPERATION_REJECTED` pozostaje wyłącznie dla nieznanego operation ID. M0.5 jest zamkniętym kontraktem.

## Final invariants: TradingUniverse, state matrix i identity lineage

TradingUniverse przechodzi pełną walidację zamkniętego rekordu przed katalogami i sukcesem: identyfikatory są niepustymi stringami, `version` jest dodatnim `int` z wykluczeniem `bool`, membership i source catalogs są niepustymi unikalnymi listami, timestampy są ścisłe, a hash jest kanonicznym SHA-256. `RETIRED` i `REJECTED` nie mogą być aktywowane. `previous_universes_by_id` przechowuje pełne rekordy, a cały łańcuch wersji jest przechodzony z `visited set`, ciągłą numeracją i scope konta. Konflikt ACTIVE jest liczony wyłącznie per `exchange_account_id`.

Jeden wspólny fail-closed helper egzekwuje niepuste osie `allowed_lifecycle_states`, `allowed_connection_states` i `allowed_authorization_states` przed sukcesem operacji. Pusta lista oznacza, że operacja nie korzysta z osi. Każda mapa context ma własny kontrakt `key == record ID`; lookup jest ponownie wiązany z ID requestu.

Credential lineage i catalog lineage wymagają pełnego predecessor record, zgodności map key z rekordem, poprawnych timestampów/hashów/adaptera oraz pełnego cycle detection. Aktywne credentials są scoped przez `active_profile_ids_by_account_id`; dokładnie jeden profil docelowego konta jest wymagany również bez rotacji, a profile innych kont nie powodują konfliktu. M0.5 jest zamkniętym kontraktem.

## Trust boundary i finalne invariants

`VERIFY_EXTERNAL_IDENTITY` przenosi wyłącznie transportowe ID, literalny `reconnect` i oczekiwane źródło używane tylko do porównania. Identity snapshot nigdy nie pochodzi z requestu: CoreHost rozwiązuje go wyłącznie z zamkniętego `validation_context.external_identity_snapshots_by_account_id`; caller nie może ustawić `VERIFIED`, permissions, venue account ID ani źródła authority.

Katalog `PARTIAL` może zostać zachowany lub obsłużony przez refresh, ale aktywacja TradingUniverse wymaga dokładnie `VALID`. UPDATE egzekwuje state matrix na wynikowym rekordzie. Instrument, CredentialProfile oraz wszystkie historyczne predecessors przechodzą pełne, typowane, fail-closed validatory; lineage universe przelicza canonical hash, a ten sam aktywny durable universe ID jest idempotentny wyłącznie dla dokładnie identycznego rekordu.

`validation_context` ma dokładnie zamknięty zestaw pól bez caller authority i secret payloads. Multi-account policy nie utożsamia kont po samym portfolio/exchange/environment/market: wiele DRAFT i różne subkonta są dozwolone, natomiast zweryfikowana identyczna venue account + subaccount identity nie może należeć do dwóch ACTIVE ExchangeAccount. M0.5 jest zamkniętym kontraktem.

## Authority i fail-closed final gaps

`UPDATE_ACCOUNT.mutable_patch` nie zawiera `external_account_identity_state`; tylko Core-owned VERIFY może zmieniać tę oś. Trusted venue identity jest exact tuple `(exchange_id, environment, market_type, venue_account_identifier, subaccount_identifier)` pobieranym wyłącznie z context. Caller `external_account_reference` i `external_subaccount_reference` nie są authority. Ten sam tuple nie może należeć do dwóch ACTIVE accounts; różne niepuste subkonta są odrębne, a root używa `null`.

Refresh akceptuje pełnie zwalidowane `VALID` i `PARTIAL`, natomiast aktywacja universe nadal wymaga `VALID`. Każdy request przechodzi operation-specific exact type validation przed handlerem. Secure-store reference ma gramatykę `secure-store://` + niepusty opaque locator, bez whitespace, query, fragmentu, `=` i markerów payloadu. Lifecycle timestamps kont i profili są monotoniczne i zależne od stanu. M0.5 jest zamkniętym kontraktem.

## Trusted bindings i metadata lifecycle

Generic UPDATE zmienia wyłącznie `display_name`, lifecycle, connection, authorization i spójny retirement timestamp. Nie może ustawić ani wyczyścić `active_credential_profile_id` ani `account_capability_snapshot_id`: aktywny profil zmienia wyłącznie `BIND_CREDENTIAL_PROFILE`, a capability binding jest Core-owned po pełnej walidacji snapshotu. Ten sam helper trusted identity uniqueness jest używany przez VERIFY i przez aktywację konta w UPDATE.

AccountCapabilitySnapshot ma pełny typed validator, canonical hash, strict scope/timestamps, ciągłe immutable lineage z `visited set` oraz map-key binding. Mapy `account_capability_snapshots_by_id` i `previous_account_capability_snapshots_by_id` należą do exact context. STALE/REJECTED tylko ograniczają, nigdy nie rozszerzają ProductCapabilities ani Live.

Poprawność metadata Instrument jest oddzielona od tradability: katalog zachowuje poprawne TRADING/HALTED/SUSPENDED/DELISTED/UNKNOWN, ale aktywacja universe wymaga TRADING. Credential lineage porównuje każdą bezpośrednią krawędź przez jawny `successor`: `predecessor.created <= predecessor.retired <= successor.created`. M0.5 jest zamkniętym kontraktem.

## Trusted readiness, capability operability i Instrument identity

Aktywacja TradingUniverse nie ufa stringowi `account.external_account_identity_state`: rozwiązuje pełny snapshot z exact context key, uruchamia external identity validator oraz wspólną globalną uniqueness trusted tuple. Caller external references pozostają bez authority.

Dla giełdy `ADAPTER_SNAPSHOT_REQUIRED` aktywacja wymaga przypiętego AccountCapabilitySnapshot o statusie `VALID`, pełnym lineage i `now < stale_after_utc`. `STALE` i `REJECTED` są historycznie zachowywalne, lecz dają `ACCOUNT_READINESS_BLOCKED`. `STATIC_BUILD_TIME` PAPER nie wymaga prywatnego snapshotu; jawnie przypięty snapshot nadal musi być prawidłowy. Capability zawsze tylko ogranicza i nigdy nie tworzy Live.

Katalog buduje dwukierunkowy exact index Instrument identity `(exchange_id, environment, market_type, venue_symbol)`: jeden tuple → jedno `instrument_id` i jedno ID → jeden tuple bez jawnej migracji aliasu. Nie ma trim, case-fold ani Unicode normalization; `display_symbol` nie uczestniczy w identity. Metadata/tradability split pozostaje zachowany. Status M0.5: `closed`; M0.5 jest zamkniętym kontraktem.

## Jedna bramka aktywacji i zaufana historia Instrument

Każda ścieżka sukcesu `ACTIVATE_TRADING_UNIVERSE`, również bezpośrednie wywołanie validatora wersji universe, przechodzi deterministycznie przez jedną bramkę `validate_account_activation_readiness` z pełnym `validation_context`. Brak contextu jest odmową. Sam string `VERIFIED` w koncie nie jest authority: wymagany jest pełny trusted identity snapshot, globalna kontrola exact identity tuple oraz literalne trusted readiness.

Dla `TESTNET` bramka wymaga dokładnie przypiętego, aktywnego CredentialProfile tego konta, zgodnego mapowania active-profile i pełnego lineage. Effective permissions są przecięciem permission snapshot profilu, AccountCapabilitySnapshot i trusted external identity. `READ_ACCOUNT` jest obowiązkowe, `ORDER_ENTRY_ALLOWED` wymaga `PLACE_ORDERS` w każdym źródle, a `WITHDRAW` w dowolnym źródle blokuje aktywację. `PAPER` z `STATIC_BUILD_TIME` może nie mieć profilu prywatnego; `LIVE` pozostaje zablokowane przed odczytem sekretu.

Przypięty AccountCapabilitySnapshot musi być `VALID`, świeży i ograniczający. Jego `supported_instrument_types` ogranicza każdy instrument universe; lista pusta blokuje niepusty universe i nigdy nie rozszerza Exchange Registry, par market/type ani ProductCapabilities.

Exact context zawiera Core-owned `instrument_history_by_id`: mapę `instrument_id` do niepustej, rosnącej po unikalnym `metadata_version` listy pełnych rekordów Instrument. Każdy rekord wiąże key z `record.instrument_id`; historyczne statusy non-TRADING pozostają rozwiązywalne. Tuple `(exchange_id, environment, market_type, venue_symbol)` jest porównywane exact i niezmienne bez jawnej migracji; `display_symbol` oraz `trading_status` mogą ewoluować. Caller request nie może dostarczyć ani nadpisać tej historii.

## Trusted argument binding i kompletna pojedyncza bramka readiness

`validation_context` jest jedyną authority. Bezpośredni validator aktywacji najpierw waliduje pełny, zamknięty context, a następnie wymaga exact equality argumentów `account`, `universe`, `instruments`, `catalogs`, map lineage i listy aktywnych universe z odpowiadającymi im trusted rekordami lub kolekcjami contextu. Osobne argumenty są wyłącznie nieufnymi wartościami porównawczymi; mismatch, brak albo malformed context daje kontrolowaną odmowę bez wyjątku. Globalna i ukryta authority są zabronione.

Pojedyncza `validate_account_activation_readiness` obejmuje exact trusted account binding, pełny `validate_account`, trusted identity i uniqueness, `ACTIVE`, state matrix, literalne `readiness_confirmed` oraz trusted readiness map, nieblokujące connection/authorization, `VALID` i świeży capability snapshot, dokładnie przypięty aktywny CredentialProfile z lineage, przecięcie effective permissions oraz blokadę Live. `WITHDRAWAL_PERMISSION_FORBIDDEN` jest zachowywany dokładnie dla profilu, capability i identity oraz mapowany na `TRADING_UNIVERSE_REJECTED`.

Trusted `instrument_history_by_id` jest obowiązkowym jawnym argumentem direct catalog validation; pusta mapa oznacza system bez historii, lecz `None` lub malformed mapa są odmową. Identity tuple jest sprawdzany pomiędzy wszystkimi historycznymi rekordami nawet bez rekordu bieżącego, a bieżący `metadata_version` musi być ściśle większy od historycznego maksimum. M0.5 odrzuca każdy rewrite tuple fail-closed; alias/migration zostaje odłożony do przyszłego jawnego kontraktu. Historyczna poprawność strukturalna sprawdza schema, registry, typy, timestamp ordering, decimal, asset i derivative rules, ale nie wymaga bieżącej execution freshness; rekord bieżący nadal musi być świeży.

## Multi-catalog binding, wspólna historia i pełny trusted context

Aktywacja przekazuje pełną trusted mapę `validation_context.catalogs_by_id`; `source_catalog_snapshot_ids` wybiera z niej wyłącznie katalogi używane przez universe. Dodatkowy poprawny katalog nie blokuje aktywacji i nie rozszerza membership. Dispatcher i direct validator używają identycznych trusted records, a caller-supplied, obcy lub zmieniony katalog jest odrzucany.

Jeden `validate_instrument_history_map` obsługuje `validate_context`, direct catalog validation i ścieżkę aktywacji. Exact identity rewrite daje `INSTRUMENT_IDENTITY_COLLISION`; malformed, duplicate, out-of-order, rollback albo `current.metadata_version <= historical max` daje `INSTRUMENT_METADATA_INVALID`. Historyczne rekordy używają `require_fresh=False`, zachowując pełne invariants strukturalne bez przyznawania execution eligibility.

Full closed `validation_context` waliduje strukturalnie każdy rekord każdej trusted mapy, także niezwiązany z bieżącym targetem. Structural validity pozostaje oddzielona od operability: `UNVERIFIED`, `DISABLED`, historyczny `STALE` katalog lub stale historyczny Instrument mogą być przechowywalne, lecz nie przyznają readiness. Malformed unrelated record kończy cały request fail-closed bez wyjątku. M0.5 jest zamkniętym kontraktem.

## Active indexes, durable ID spaces i trusted references

`credential_profiles_by_id` jest grupowane według konta: najwyżej jeden profil może być `ACTIVE`, musi być dokładnie wskazany przez `account.active_credential_profile_id` i jednoelementowy `active_profile_ids_by_account_id`; ukryty drugi profil, profil `RETIRED` w indeksie albo `ACTIVE` w `previous_profiles_by_id` są odrzucane. Konflikt aktywnych universe jest liczony z unii exact-bound `active_universes` oraz wszystkich rekordów `ACTIVE` w `universes_by_id`, z wyłączeniem targetu przez exact durable ID.

Current i previous mapy CredentialProfile, Catalog, TradingUniverse oraz AccountCapabilitySnapshot mają rozłączne przestrzenie durable ID. Trusted context egzekwuje referencje profile→account, instrument→catalog, catalog→current/history member, universe→account/source catalogs/instruments, capability→account oraz exact active indexes. Historyczne rekordy zachowują structural validity bez wymagania execution freshness.

Każda obowiązkowa lineage map wymaga exact `dict`; falsey coercion jest zabronione. `False`, `0`, pusty string, list, tuple i set są odmową, natomiast jawne `{}` jest poprawne wyłącznie dla pierwszej wersji bez poprzednika. M0.5 jest zamkniętym kontraktem.

## Pełny trusted reference graph i exact membership

Każde konto przechodzi `validate_account_owned_bindings`: null profile pointer zabrania bieżącego `ACTIVE` profilu i active-index entry, a non-null pointer exact-binduje jeden current `ACTIVE` CredentialProfile oraz jednoelementowy indeks z tym samym account/exchange/environment scope. Non-null capability pointer rozwiązuje wyłącznie current capability mapę i exact-binduje account/exchange/environment/market wraz z pełnym lineage; dangling reference na niezwiązanym koncie unieważnia cały context.

Wspólny `resolve_catalog_member` potwierdza membership wyłącznie pełnym rekordem przypisanym do exact `catalog_snapshot_id`, scope i adapter family. Sama obecność ID w current/history nie wystarcza; history wybiera deterministycznie najwyższą pasującą `metadata_version`, zachowując structural stale metadata bez przyznawania execution eligibility.

`validate_universe_source_membership` stosuje tę samą politykę do current i previous universe: konto i source catalogs muszą istnieć, a każdy instrument musi być członkiem co najmniej jednego zadeklarowanego source catalog. Globalny lub niezadeklarowany katalog nie rozszerza membership. Wspólne graph helpery są używane przez full `validate_context` i wszystkie direct/dispatcher paths, więc unrelated dangling record blokuje sukces fail-closed. M0.5 jest zamkniętym kontraktem.

## Exact `catalog.instrument_ids` i totalny resolver

Exact Catalog membership wymaga również, aby sprawdzany `instrument_id` występował w unikalnej liście niepustych ID `catalog.instrument_ids`. Sama zgodność `record.catalog_snapshot_id == catalog.catalog_snapshot_id`, scope i adapter family nie potwierdza członkostwa. Reguła obowiązuje current/history Instrument, current/previous TradingUniverse, full context, direct validators i dispatcher paths.

`resolve_catalog_member` jest totalny: malformed katalog, mapa, lista historii, rekord lub `metadata_version` zwraca `None` bez wyjątku. Deterministyczne wybranie najwyższej wersji następuje dopiero po potwierdzeniu dodatnich integerów z wykluczeniem bool. Full `validate_context` uruchamia `validate_instrument_history_map` przed jakimkolwiek history member resolution; malformed trusted history unieważnia cały context i nie może zostać cicho pominięta. M0.5 jest zamkniętym kontraktem.

## Dwukierunkowy Catalog graph i pełna totalność resolvera

Każdy current Instrument z `instruments_by_id` musi być listed memberem katalogu wskazanego przez własny `catalog_snapshot_id`; graph validation jest obowiązkowo dwukierunkowa: Catalog → Instrument oraz Instrument → Catalog. Resolver musi zwrócić dokładnie current record, więc rekord historyczny nie może maskować brakującego current membership. Malformed unrelated current Instrument unieważnia cały trusted context.

Przed zwróceniem current lub historical membera resolver sprawdza closed strukturę katalogu oraz pełny `validate_instrument_record(require_fresh=False)`. Historia wymaga dodatnich, niebooleanowych i ściśle rosnących `metadata_version` również w direct resolver path; out-of-order, duplicate, mixed types, malformed schema, obce ID lub identity rewrite zwracają `None` bez wyjątku. M0.5 jest zamkniętym kontraktem.

## Historyczny Instrument graph i parytet direct resolvera

Odwrotny binding `Instrument → Catalog` obejmuje każdy current rekord oraz każdy pełny rekord z `instrument_history_by_id`. Każdy historyczny `catalog_snapshot_id` jest obowiązkową trusted reference do current albo previous Catalog; historyczny rekord musi występować w `catalog.instrument_ids` i exact-matchować exchange, environment, market oraz adapter family. Dangling, unlisted lub scope-mismatched historia — także niezwiązana z targetem operacji — unieważnia cały trusted context.

Direct resolver przed zwróceniem membera waliduje semantycznie pełny source Catalog: closed schema, enabled Exchange Registry scope, status, adapter/version, timestamp ordering, canonical content hash, unikalną listę członków oraz lineage, gdy zadeklarowano predecessor. Dla current Instrument wymaga również `current.metadata_version > max(all historical metadata_version)`; equality, rollback i malformed version zwracają `None` bez historycznego fallbacku. Direct i dispatcher paths wydają identyczną fail-closed decyzję dla tego samego trusted graph. M0.5 jest zamkniętym kontraktem.

## Totalny Catalog node i kompletne direct graph closure

Jeden totalny Catalog node preflight sprawdza closed schema i bezpieczne typy wszystkich pól, registry scope, adapter/version, status, timestamp ordering, unikalne `instrument_ids` oraz canonical SHA-256 przed hashowaniem, lookupem, użyciem visited set albo traversal lineage. Wartości malformed i non-JSON-serializable zwracają kontrolowane `False`/`None` bez wyjątku. Każdy predecessor przechodzi identyczny node validator, a lineage zachowuje cycle detection.

Direct resolution waliduje pełny binding historii target Instrument do trusted current/previous Catalog graph przed zwróceniem current candidate. Direct universe membership dodatkowo domyka każdy source Catalog przez rozwiązanie wszystkich jego `instrument_ids`; dodatkowy missing, malformed, dangling lub scope-mismatched member blokuje sukces. Direct, full-context i dispatcher mają tę samą decyzję dla dangling/unlisted historii, niepełnego member closure i malformed unrelated trusted records. M0.5 jest zamkniętym kontraktem.

## Jeden complete Catalog/Instrument graph validator

`validate_catalog_instrument_graph` jest wspólną bramką dla full context i direct TradingUniverse membership. Totalnie preflightuje wszystkie current/previous Catalog maps, current Instrument oraz pełne target i unrelated `instrument_history_by_id`; egzekwuje rozłączne Catalog ID spaces, map-key binding, pełny history validator, odwrotne Instrument→Catalog binding i closure każdego członka każdego katalogu. Orphan lub malformed unrelated current Instrument, malformed unrelated Catalog, unresolved member i pusta, out-of-order, rewritten albo rollback history blokują direct membership bez wyjątku.

Historia może exact-bindować się do katalogu z trusted current albo previous mapy. Wszystkie complete-graph resolver calls otrzymują obie mapy, dzięki czemu drugi poprawny current Catalog jest widoczny identycznie dla direct, full-context i dispatcher paths. Parytet obejmuje zarówno kontrolowane odmowy, jak i prawidłowe sukcesy. M0.5 jest zamkniętym kontraktem.
