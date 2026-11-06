# S9-FV-01 — AccountGenesis przez zweryfikowany transport, TEST_ONLY

6 listopada 2026 r. Baza: `main` po merge PR #3257,
`6b426c9f42aedf54805f32d643d6d11ac1c9f947`.
**Materiał migracyjny do niezależnego przeglądu; wdrożenie BLOCKED.**
Etap 1 K4-v2: **PENDING_OWNER_ACCEPT**, K4-v2 nieaktywny;
natywny dowód dwóch UID: **NOT_RUN**.
Stage 9: **IN_PROGRESS / PRODUCTION_NO_GO**.

Poprawka P2 do tego samego PR zamyka możliwość zielonego kroku CI po pominięciu
obowiązkowych prób AF_UNIX. [Dokładny wykaz 83 przypadków](s9_fv_01_required_cases.md)
oraz [raport bramki JUnit](s9_fv_01_p2_junit_gate.md) opisują egzekwowanie wykonania.
Dotychczasowe propozycje adaptera/serwera i zakres produkcyjny pozostają bez zmian.

## Inwentaryzacja i granice ustaleń

Przeczytano `AGENTS.md`, źródła AccountGenesis, klienta runtime i verifiera,
[patch czterech poprawek](../../tests/fixtures/freshness_verifier_remediation_review.patch),
[jego analizę](freshness_verifier_remediation_review.md),
`freshness_application.py`, `freshness_deployment_qualification.py`, instalator
oraz [kontrakt K4-v2](../architecture/stage9_verifier_identity_acceptance_contract_v1.md).
Codebase Memory MCP ani jego lokalne CLI nie są dostępne. Nie uzyskano grafu,
odświeżenia indeksu ani `check_index_coverage`; ustalenia wynikają z bezpośrednich
odczytów źródeł i wyszukiwania `rg`, bez deklaracji kompletności indeksu.

## Dokładne bajty podpisanego closure i zachowane piny

[Weryfikator aplikacji](../../deployment/freshness_application.py)
sprawdza podpis Ed25519 **dokładnych surowych bajtów `manifest.json` przed
interpretacją JSON**. Manifest v2 zawiera `source_revision`, `runtime` oraz
pełną listę rekordów `path`, `sha256`, `size`, `mode`. Podpis wiąże przez te
rekordy dokładne bajty wszystkich plików payloadu. Instalacja nie może zawierać
dodatkowego niepodpisanego pliku ani pomijać podpisanego. Dla każdego pliku
weryfikowane są ponadto chronieni przodkowie, owner, tryb, brak symlinku,
pojedynczy link i niezmienność metadanych podczas odczytu. `binding.json`
wiąże SHA-256 manifestu, rewizję i dokładny runtime; nie jest alternatywnym
źródłem zatwierdzenia nowych bajtów.

Wspólny wymagany inwentarz `REQUIRED_FILES` w weryfikatorze aplikacji oraz
[instalatorze](../../deployment/production_local_installer.py) obejmuje 16 plików:

```text
bot_core/__init__.py
deployment/__init__.py
bot_core/freshness_verifier_service.py
bot_core/freshness_semantic_verifier.py
bot_core/postgresql_freshness_authority.py
bot_core/linux_service_ipc.py
bot_core/linux_service_postgresql.py
bot_core/production_signer_ipc.py
deployment/freshness_verifier_entrypoint.py
deployment/freshness_application.py
deployment/freshness_runtime.py
deployment/freshness_runtime_bootstrap.py
deployment/systemd/cryptohunter-freshness-verifier.service
deployment/postgresql/pg_hba.conf
deployment/postgresql/pg_ident.conf
deployment/postgresql/production-local.conf
```

To wymagane minimum profilu `SIGNED_FRESHNESS_SUBSTRATE`, a nie pełny inwentarz
produktu. Manifest musi obejmować również każdy dodatkowy plik rzeczywiście
umieszczony w payloadzie. Template unitu jest przed podpisaniem materializowany
do konkretnego zatwierdzonego interpretera. Runtime ma osobny podpisany manifest
obejmujący interpreter, stdlib, biblioteki, wheels, symlinki i zewnętrzne
zależności natywne, zgodnie z [kontraktem runtime](../architecture/stage9_linux_signed_python_runtime.md).
Podpis aplikacji wiąże dokładne środowisko przez `runtime`.

Obecne zatwierdzenia, bez ich aktualizacji w tym zadaniu:

| Plik w `bot_core/` | SHA-256 zatwierdzonego pinu | Profil |
| --- | --- | --- |
| `linux_service_ipc.py` | `099e6a3a08717b48e7a01606b9908f56f0eec40aced260d554d7dfc4562b451e` | Oba |
| `linux_service_postgresql.py` | `d09d359535e18408d8cee282cb757c4eca8a4e8c422bfce8fa804860ad59f0d1` | Oba |
| `production_signer_ipc.py` | `be755e67cb1ea4466c074e4c2c0dbe8e6528552f4a70fd7769a5e876c878f552` | Oba |
| `freshness_verifier_service.py` | `df23a3c2fcd1e04877192fe2fe376909a29aa5183a516485fbfdad5a72f9f057` | Oba |
| `freshness_semantic_verifier.py` | `64dd105d54a2eabd9be9164846fe642ca0a274b30ca5d0a83735a4436d9f39e1` | Oba |
| `account_genesis_freshness_authority.py` | `cd508809f4b4f0663610f592b77695784d74eb4c7e2fba50c42d7a4dbf252f7b` | Wyłącznie pełny `PRODUCTION_LOCAL` |

Pięć pierwszych pinów występuje w `REVIEWED_SUBSTRATE_DIGESTS` i
`REVIEWED_CODE_DIGESTS`; odpowiadają rzeczywistym bajtom bazy.
**Zastany drift AccountGenesis pozostaje blokadą:** plik w bazie ma SHA-256
`81cd0df375fe1a48ba943c7645376b4d97350288c0ebe6523207ff90ed22572b`, różny od
zatwierdzonego szóstego pinu. Zachowanie pliku bez zmian względem `main` nie
oznacza zgodności z tym zatwierdzeniem. Nie koryguje się tego przez rehash.

`verify_installed_application()` dodatkowo wymaga zatwierdzonych pięciu digestów
wewnątrz podpisanego manifestu; sam nowy podpis nie znosi starego pinu.
Pełny `_qualify_reviewed_sources()` sprawdza sześć pinów bezpośrednio z plików,
przed dalszą kwalifikacją systemd/PG. Sukces minimalnego profilu zwraca
`production_qualification=WITHHELD`, nigdy pełną kwalifikację produktu.

## Kto może zatwierdzić przyszłą zmianę

Potrzebne są niezależny przegląd dokładnych nowych bajtów i jawna decyzja wydania.
Zaufanie klucza pochodzi z osobnej ceremonii administratora. Instalator wymaga
chronionego zewnętrznego autorytetu, poza ocenianym bundle; zaufany bootstrap
weryfikuje podpis bez uruchomienia ocenianego środowiska. Podpis aplikacji i
runtime nie zatwierdza własnego weryfikatora: administrator niezależnie
zatwierdza instalator i bootstrap. Źródła nie wskazują imiennego posiadacza
uprawnień do podpisu ani osoby, którą ten raport mógłby zastąpić.

Zmiana kodowych pinów musi być osobno zatwierdzoną zmianą obu właściwych
bramek, powiązaną z przejrzanym wydaniem i materiałami podpisu. Autor patcha,
zielony test, merge dokumentacji ani odpowiedź verifiera nie autoryzują
odświeżenia pinów. Dla K4-v2 osobny rekord **właściciela i niezależnego
recenzenta** musi wskazywać commit kontraktu i rozstrzygać D01–D08; obecnie
wszystkie pozostają PENDING. Niniejszy zakres TEST_ONLY nie udziela ACCEPT
etapu 1 ani zgody na eksperyment nowej podstawy K4-v2.

## Docelowy model transportu i blokady K4

Jedna operacja ma przekazać dokładnie trzy niezmienne obiekty:
AccountGenesis → `RuntimeFreshnessVerifierClient` → uwierzytelniony
FreshnessVerifier → `PREPARED` → odrębna transakcja CAS wykonywana wyłącznie
przez uprawniony runtime. Verifier zachowuje własny principal przygotowania;
runtime nie otrzymuje prawa prepare, a verifier prawa CAS. Granice transakcji,
trwałość i klasyfikacja niejednoznacznego COMMIT pozostają dotychczasowe.

`ExpectedPeerProcess` jest konstrukcyjnym wejściem klienta z niezależnie
zatwierdzonej obserwacji usługi/wydania: PID, UID, GID, ticks i hash executable.
Sama poprawność typu tej krotki nie ustanawia jej pochodzenia. W testach źródło
expected jest modelem kontrolera, a nie produkcyjnym loaderem zatwierdzenia.
Przyszły właściciel procesu runtime musi niezależnie powiązać te wartości z
efektywnym systemd, zatwierdzonym wydaniem i podpisanym closure. Dane z odpowiedzi
samego verifiera nie mogą wypełniać expected ani observed.

Istniejący klient sprawdza tożsamość runtime, chronioną stałą ścieżkę socketu,
`SO_PEERCRED`, socket-bound `SO_PEERPIDFD`, bieżący proces, cgroup, ticks,
grupy i hash; ponawia obserwacje przed send i po odpowiedzi. Wysyła jedną
ramkę `!I` + kanoniczny JSON/base64url, robi `shutdown(SHUT_WR)`, odbiera jedną
ograniczoną ramkę i wymaga EOF odpowiedzi. Weryfikuje strukturę i tożsamość
przygotowania oraz binding do dokładnie przekazanych trzech obiektów. Limit
ramki pozostaje 4 MiB, budżet 3,0 s dla wymiany; nie dodaje się reconnect/retry.

Obecny K4 nadal wymaga rzeczywiście obserwowanego hasha. Gdy
`capture_process_identity(..., allow_missing_executable=True)` napotyka
`PermissionError` dla `/proc/PID/exe`, zachowuje `exe_sha256=None`.
`_check_server()` porównuje je z wymaganym hashem i odmawia **przed pierwszym
payload**: `UNAVAILABLE / TRANSPORT_OR_IDENTITY_UNAVAILABLE`. Nie wstawia się
expected w miejsce obserwacji ani hasha deklarowanego przez serwer. Po
rozpoczęciu wysyłania niepełny wynik daje `OUTCOME_UNKNOWN`; timeout nie dowodzi
rollbacku ani anulowania przygotowania.

Na rzeczywiście odrębnych UID wymagane odczyty procfs pozostają nierozwiązane.
K4-v2 proponuje inną podstawę, E01–E09, lecz jej implementacja i natywne N01–N18
nie są zatwierdzone. Natywny dowód dwóch UID nadal **NOT_RUN**. Dodatkowo T3/PG
wymaga osobnego projektu: publiczny dla UID verifiera endpoint PG z samym peer
nie odróżnia obcego procesu tego samego UID i według kontraktu oznacza
`AUTHORITY_REJECT`. Testowe modele przygotowania/SQL nie kwalifikują tej granicy.
Istniejące `IPC_BLOCKERS`, w tym brak właściwej integracji runtime i trwałej
ochrony challenge/replay, pozostają aktywne.

## Dokładny zakres przyszłego ponownego zatwierdzenia

| Materiał | Wymagana przyszła decyzja |
| --- | --- |
| `bot_core/account_genesis_freshness_authority.py` | Nowe dokładne bajty migracji, usunięcie starego transportu/retry i doprowadzenie niezależnego expected; osobne rozstrzygnięcie istniejącego driftu i nowego pinu pełnego produktu. |
| `bot_core/freshness_semantic_verifier.py` | Dokładne bajty czterech poprawek z #3257, ponowny przegląd obu pinów i podpisanego payloadu. |
| `bot_core/freshness_runtime_ipc.py` | Jawne objęcie klienta zatwierdzonym closure aplikacji runtime. Modułu nie ma w wymaganym minimum 16 plików substrate. |
| `bot_core/freshness_service_identity.py`, `deployment/linux_physical_ipc.py` | Zatwierdzenie dokładnych zależności importowanych przez klienta, wraz z istniejącymi zależnościami już należącymi do substrate. Sam podpis minimalnego verifiera ich nie obejmuje. |
| Właściciel runtime dostarczający expected | Zatwierdzony loader/kompozycja i pochodzenie danych; nie powstaje tu produkcyjny plik takiego loadera. Jego nazwę i nowe zależności trzeba zamknąć przed zatwierdzeniem wydania. |
| `deployment/freshness_application.py`, `bot_core/freshness_deployment_qualification.py` | Osobno przejrzana zmiana właściwych pinów/inwentarza, dopiero po autoryzacji nowych źródeł. Nie aktualizować automatycznie. |
| `deployment/production_local_installer.py` oraz bootstrap | Ponowny przegląd, jeśli zmieni się wymagany inwentarz lub granica zaufania instalacji; każdy zmieniony helper musi odpowiadać niezależnie zatwierdzonym bajtom bootstrapu. |
| Manifest/podpis/binding aplikacji i odrębny runtime | Nowy pełny inwentarz, rewizja i zgodne powiązanie runtime, podpisane przez właściwy niezależny autorytet. Nowy runtime/manifest runtime tylko według odrębnego kontraktu, jeśli zmieni się środowisko. Brak samodzielnej rotacji kluczy. |

Lista rozróżnia pliki ze zmienianymi bajtami i dodatkowe bajty, które nowa ścieżka
zaczyna wykonywać. Pełny manifest przyszłego wydania musi ponownie obejmować
cały rzeczywisty payload; ta lista nie zastępuje jego zamkniętego inwentarza.
K4-v2 ewentualnie wymagałby także osobno zatwierdzonych startu/polityk i ich
closure — poza zakresem tej migracji.

## Materiał migracyjny i różnice względem starszej ścieżki

[Patch adaptera](../../tests/fixtures/account_genesis_verified_transport_review.patch)
jest wyłącznie danymi do przeglądu. [Loader testowy](../../tests/security/account_genesis_transport_review_support.py)
sprawdza dokładny SHA bazy AccountGenesis `81cd0df…22572b`, kopiuje ją do prywatnego
katalogu, sprawdza i stosuje diff, następnie importuje kopię pod odrębną nazwą.
Ten SHA identyfikuje bazę testu, **nie nowe zatwierdzenie wydania**. Odrębny
loader z #3257 stosuje wszystkie cztery poprawki do prywatnej kopii verifiera.
Żaden produkcyjny caller nie importuje tych helperów.

Proponowana istniejąca klasa wymaga
`ProductionLocalFreshnessAuthority(config, *, expected_verifier=...)`.
Nie powstaje konkurencyjna implementacja FreshnessAuthority. Konstruktor odmawia
alternatywnej ścieżki verifiera; obowiązuje stały `VERIFIER_SOCKET_PATH` klienta.
Usunięte są bezpośrednie `socket.socket`, własny framing, `_request_preparation_once`,
stan starego transportu oraz automatyczny retry przygotowania. Jedno wywołanie
`_request_preparation` wykonuje dokładnie jedno `verify_exact_candidate`.
Kompozycja runtime dostarcza expected przed tą operacją; brak wdrożonego źródła
zatwierdzenia pozostaje blokadą, a nie powodem użycia danych serwera.

`authenticate_and_advance`, `_connect_runtime`, `_compare_and_advance` i
`_compare_and_advance_once` zachowują **identyczne źródła** — dowodzi tego test
`test_migration_preserves_public_operation_and_exact_cas_methods`. CAS pozostaje
osobną transakcją `SERIALIZABLE` jako `freshness_runtime`. Zachowane są istniejące
wyniki, trwałość, sprawdzanie receipt, ograniczone ponowienia natywnego `40001`
i niejednoznaczność COMMIT. Retry transakcji CAS nie wykonuje ponownie prepare.
Odrębna jawna operacja caller po wyniku UNKNOWN nie jest tu rozstrzygana ani
automatycznie inicjowana przez adapter.

Binding zachowuje istniejący protokół: klient odtwarza preimages proposera i
dokumentu bez pól podpisu, a receipt hashuje w całości wraz z podpisem. Nie
dodaje nowego hasha pełnych bajtów wszystkich trzech obiektów. Podpisy i retained
credentials nadal sprawdza uwierzytelniony verifier; klient dodatkowo odrzuca
samospójny PREPARED dotyczący innych przekazanych obiektów. Szczegóły są w
[kontrakcie klienta](../architecture/stage9_linux_freshness_runtime_ipc.md).

Materializowane, **niezatwierdzone** propozycje mają SHA-256:

| Propozycja | SHA-256 |
| --- | --- |
| AccountGenesis | `17fb5d669d8e3a36efb4959d4d071ebee8e23ffdbad8da881fd8d20f0bc2f16b` |
| Verifier z czterema poprawkami | `43daa44b854883ddb80614031a4bbff10b299043cd362fd18fc8d2a6975d618d` |

### Mapowanie wyników

| Wynik zweryfikowanego klienta / etap | Wynik AccountGenesis | SQL CAS |
| --- | --- | --- |
| Wejście innego typu niż dokładne `bytes` | `INVALID_DOCUMENT` przed klientem | Nie |
| `INVALID`, reason `INVALID_LOCAL_INPUT` lub `REQUEST_TOO_LARGE` | `INVALID_DOCUMENT` | Nie |
| `INVALID` z poprawnej odpowiedzi verifiera | `INVALID_AUTHENTICATION` | Nie |
| `UNAVAILABLE`, w tym brak dowodu K4 przed send | `UNAVAILABLE` | Nie |
| `OUTCOME_UNKNOWN`, także jawny wynik serwera | `PREPARATION_OUTCOME_UNKNOWN` | Nie; jedna próba |
| Zły framing/JSON/EOF lub binding po rozpoczęciu send | Klient `OUTCOME_UNKNOWN` → `PREPARATION_OUTCOME_UNKNOWN` | Nie; bez reconnect |
| Niepełne wewnętrzne PREPARED z modelu klienta | `PREPARATION_OUTCOME_UNKNOWN` | Nie |
| Poprawne PREPARED związane z kandydatem | Istniejący CAS: m.in. `CAS_ACCEPTED`, `ALREADY_ACCEPTED_EXACT`, `CAS_CONFLICT` | Wyłącznie runtime |
| Niejednoznaczny COMMIT istniejącego CAS | `FRESHNESS_OUTCOME_UNKNOWN` | Bez ponowienia prepare |

Starszy transport traktował część błędnych odpowiedzi jako `CORRUPT` i ponawiał
przygotowanie po UNKNOWN. Nowy adapter zachowuje ostrożną klasyfikację istniejącego
klienta runtime: po rozpoczęciu send brak wiarygodnego pełnego wyniku nie dowodzi
braku przygotowania. `CORRUPT` pozostaje w niezmienionej semantyce warstwy CAS.

## Macierz prób i rodzaj dowodu

Wyniki dotyczą prywatnych propozycji, nie zatwierdzonej produkcji. Testy
[kontraktu](../../tests/security/test_account_genesis_verified_transport_contract.py),
[AF_UNIX](../../tests/security/test_account_genesis_verified_transport_unix.py) oraz
[zakresu wydania](../../tests/security/test_account_genesis_verified_transport_scope.py)
są rozdzielone. Każdy wariant parametrów jest oddzielnym przypadkiem pytest.

| Nr | Próba / test | Wynik i klasyfikacja | Rzeczywiste jądro / jawny model |
| --- | --- | --- | --- |
| 1 | `test_real_adapter_half_close_and_three_objects_reach_separate_cas` | PASS: jedno prepare, jedno CAS, dokładne document/receipt i binding | AF_UNIX, send/recv, half-close, EOF, credentials/PIDFD/procfs rzeczywiste; prepare i CAS liczone w modelach |
| 2 | `test_real_legacy_client_without_eof_is_refused_before_prepare` | PASS: pojedyncza stara próba bez EOF → odmowa, zero prepare | Prawdziwy stary `_request_preparation_once`, socket/deadline; writer model, celowo bez uruchamiania starego retry |
| 3 | `test_real_adapter_mismatched_expected_denies_before_payload`, `test_real_adapter_k4_missing_proof_denies_before_payload` | PASS: błędne PID/UID/GID/ticks/hash/cgroup/socket → UNAVAILABLE, zero payload/CAS | Kernelowy peer/proc/socket rzeczywiste; błędne expected i cgroup/grupy/instalacja jawnie modelowane; socket symlink rzeczywisty |
| 4 | Wariant `exe-unreadable` powyższego testu | PASS: brak obserwowanego exe → odmowa przed payload, bez zastępowania expected | PermissionError przy otwarciu exe jest wstrzykniętym modelem granicy `/proc`; brak natywnego dowodu dwóch UID |
| 5 | `test_real_partial_send_is_unknown_with_one_connection`, testy timeout/BrokenPipe | PASS: po rozpoczęciu send → PREPARATION_OUTCOME_UNKNOWN, jedno połączenie | Prefiks send rzeczywisty, wyjątki timeout/zerwania częściowego send wstrzyknięte; osobne timeout i BrokenPipe odpowiedzi rzeczywiste |
| 6 | `test_real_adapter_bad_response_is_unknown_without_cas_or_retry`, warianty bindingu proposer/document/receipt | PASS: samospójny obcy binding → UNKNOWN, zero CAS, bez retry | Ramka AF_UNIX rzeczywista; skryptowy peer/model odpowiedzi, kryptografia fixture wykonywana naprawdę |
| 7 | `test_client_results_map_once_without_sql_or_retry`, testy failure/invalid/bad_response | PASS: INVALID/UNAVAILABLE/OUTCOME_UNKNOWN oraz błędne length/JSON/trailing jak w tabeli | Modele wyników w kontrakcie oraz rzeczywiste ramki i poprawiony serwer w testach UNIX |
| 8 | Te same testy UNKNOWN i `test_real_partial_send_is_unknown_with_one_connection` | PASS: dokładnie jedna próba, bez reconnect lub automatycznego prepare retry | Licznik prawdziwych połączeń listenera oraz model wywołań klienta/writera |
| 9 | Testy non_bytes/local_length/encoded_request_limit i `test_real_adapter_invalid_signed_input_never_prepares_or_calls_cas` | PASS: złe typy/NaN/infinity/rozmiary/podpisane obiekty → zero prepare/CAS/nieuprawnionego SQL | Walidacja i kryptografia rzeczywiste; zakaz SQL i liczniki modelowane, brak prawdziwego PG |
| 10 | `test_real_timeout_keeps_late_effect_and_next_adapter_client_works` | PASS: po timeout efekt modelu pojawia się później, brak domniemania rollbacku; następna jawna operacja działa | Timeout/EOF i dalszy listener rzeczywiste; trwały efekt/COMMIT wyłącznie modelowany |
| 11 | `test_baseline_drift_does_not_refresh_frozen_product_pin` oraz istniejące testy pinów | PASS: oryginalne piny zachowane, zastany drift nie maskowany | Rzeczywiste bajty i SHA; bez modelowania digestów lub pinów |
| 12 | `test_existing_release_gate_rejects_each_proposed_composition` — AccountGenesis/verifier/oba | PASS: istniejący gate odrzuca nowe bajty w każdym wariancie | Prawdziwa bramka i pliki; modelowane wyłącznie chronione metadane instalacji |

Wszystkie cztery poprawki #3257 sprawdzono **w kompozycji z adapterem**:

| Poprawka | Próba łączona |
| --- | --- |
| Deadline od accept | Niepełny header/body zostaje zamknięty, potem nowy adapter osiąga PREPARED/CAS na tym samym listenerze (`test_real_deadline_frees_listener_for_adapter`). |
| Własność sesji po błędzie SELECT | Poprawiony `PostgreSQLVerifierConnection.connect` jest wykonywany w writerze poprawionego serwera: execute/fetchone failure → close=1, UNAVAILABLE do adaptera, zero prepare/CAS; następna jawna operacja działa. Granica psycopg jest modelem. |
| Błąd odpowiedzi nie kończy usługi | Prawdziwy BrokenPipe po modelowanym prepare, następnie udana operacja adaptera na tym samym listenerze (`test_real_response_broken_pipe_after_prepare_keeps_adapter_listener_usable`). |
| EOF przed przygotowaniem | Stary klient bez EOF nie przygotowuje nic; nowy adapter wykonuje half-close i uzyskuje PREPARED. |

W testach UNIX kontroler sam odczytuje procfs znanego PID utworzonego potomka
**przed** połączeniem. Pipe służy gotowości, odpowiedź protokołu nie ustanawia
expected. Odczyt jest rzeczywisty dla własnego UID; akceptacja właściciela,
chronione konto i instalacja pozostają modelami. Natywne dwa UID: **NOT_RUN**.
Budżety 0,1–1,0 s dotyczą tylko szybkich prób testowych. Deadline socketu
nie anuluje PG/COMMIT/CPU; nie ma zmiany produkcyjnych limitów.

## Walidacja

Końcowy wspólny przebieg: **631 PASS / 3 SKIP**. Nowe przypadki: **83 PASS** —
46 kontraktu/CAS, 33 rzeczywistego transportu, 4 zakresu/pinów. Istniejące
regresje: **548 PASS / 3 SKIP**. Pominięcia dotyczą
rzeczywistych dodatkowych grup konta testowego, natywnego Windows oraz osobnego
builda wymagającego `FRESHNESS_RUNTIME_WHEELHOUSE` na Ubuntu 24.04.

Pierwsza próba w ograniczonym sandboxie dała 31 FAIL / 470 PASS / 50 SKIP w
istniejących testach oraz 30 SKIP w pierwszej wersji nowych testów UNIX:
`bind`/`SO_PEERPIDFD` były odrzucane przez sandbox jako EPERM. Ponowienie wyłącznie
nieuprzywilejowanych prób lokalnych poza tym ograniczeniem dało powyższe wyniki.
Nie zmieniano UID, usług, ról, LSM ani konfiguracji PostgreSQL.

Szybkie odtworzenie nowych przypadków:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 DUDZIAN_TEST_MODE=1 DUDZIAN_ALLOW_LONG_POLL=0 \
python -m pytest -q -ra -p asyncio tests/security/test_account_genesis_verified_transport_*.py
```

Zwykły istniejący workflow `Quality and Security` otrzymuje krok uruchamiający
ten zestaw wraz z regresjami klienta runtime i czterech poprawek #3257. Nie
uruchamia instalacji ani natywnych kwalifikacji dwóch UID; nie zmienia progów,
zakresów wcześniejszych bramek ani workflow wymagających ręcznej zgody.

| Bramka | Wynik |
| --- | --- |
| Ruff 0.14.11, cały dotychczasowy zakres oraz E/F/B/I nowych plików | PASS |
| Ruff lint/format obu materializowanych kandydatów | PASS |
| Dokładne `python -m pre_commit run ruff-format --all-files` | PASS; pierwszy właściwy przebieg sformatował wyłącznie nowy test scope, po review i dodaniu zmiany do indeksu następne przebiegi exit 0, bez dalszych zmian |
| mypy 1.10.0, skonfigurowany Linux i `--platform win32` | PASS, po 392 pliki |
| Te same dwa przebiegi z `--shadow-file` obu propozycji | PASS, po 392 pliki; skonfigurowany zakres obejmuje verifier, nie AccountGenesis/klienta |
| Osobne jawne mypy AccountGenesis(kandydat) i `freshness_runtime_ipc.py`, Linux/Win32 | PASS, po 2 pliki; to kontrola typu platformy, nie wykonanie Windows |
| Semgrep CE 1.179.0, `.semgrep/cryptohunter.yml` w repo | PASS, 5 reguł, 1123 pliki, zero znalezisk |
| Semgrep obu prywatnych kandydatów | PASS, 4 właściwe reguły, 2 pliki, zero znalezisk |
| Testy fixture reguł Semgrep | PASS, 5/5 |
| Doradcze community Semgrep `p/python` + `p/security-audit` | BLOCKED, exit 2: proxy HTTP 403 przy pobieraniu `semgrep.dev/c/p/python`; nie zmieniano polityki ani nie obchodzono proxy |
| Betterleaks 1.9.0, pełne repo offline, istniejący baseline, redakcja 100% | PASS, zero znalezisk |
| `git diff --check` i `git diff --cached --check` | PASS |

Próby sprzed udostępnienia narzędzi zgłaszały brak `pre_commit`/Semgrep/Betterleaks
i niedostępne połączenie proxy w ograniczonym sandboxie. Po nadaniu uprawnienia
sieciowego korzystano z tego samego proxy i normalnie zainstalowano narzędzia;
nie były to ostateczne wyniki bramek. Semgrep ma oddzielny venv; zależności
repozytorium i locki nie zostały zmienione. Pierwszy Betterleaks wskazał kopię
istniejącego, zatwierdzonego fixture klucza w nieśledzonym wygenerowanym
`tests/__pycache__/test_runtime_bootstrap.cpython-311-pytest-9.1.1.pyc`.
Usunięto wyłącznie ten wygenerowany cache i ponowiono pełny skan; nie zmieniano
baseline, reguł ani źródłowego fixture. Prywatne klucze testowe nie są nowymi
poświadczeniami produkcji. Artefakt Betterleaks zweryfikowano przypiętym SHA
i oficjalnym checksum. Lokalny Semgrep fixture/kandydat wymagał wykonania poza
ograniczeniem sandboxa, podobnie jak AF_UNIX; bez zmian badanego kodu.

Patch adaptera ma zerowy kontekst, a helper używa `git apply --unidiff-zero`
(najpierw `--check`). Eliminuje to whitespace pustych linii kontekstu w samym
materiale patcha. Po tej zmianie ponownie odtworzono identyczny SHA kandydata
i uruchomiono **83 PASS**; nie zmieniła ona bajtów proponowanego adaptera.

Kontrole jakości dają się powtórzyć standardowymi poleceniami repozytorium:

```bash
python -m ruff check --no-cache bot_core core scripts tests deploy stage6_samples ui
python -m ruff check --select E,F,B,I tests/security/*account_genesis*transport*.py
python -m pre_commit run ruff-format --all-files
python -m mypy
python -m mypy --platform win32
semgrep scan --config .semgrep/cryptohunter.yml --metrics=off --disable-version-check --error bot_core core deployment scripts
semgrep scan --test --config .semgrep/cryptohunter.yml .semgrep/cryptohunter.py --metrics=off --disable-version-check --jobs 1
betterleaks dir --baseline-path .betterleaks-baseline.json --redact=100 --no-banner --no-color .
git diff --check
git diff --cached --check
```

Dla materializowanych kopii uruchomiono także `mypy --shadow-file` dla obu
ścieżek źródłowych oraz jawne `python -m mypy --config-file pyproject.toml
--follow-imports skip bot_core/account_genesis_freshness_authority.py
bot_core/freshness_runtime_ipc.py --shadow-file bot_core/account_genesis_freshness_authority.py
KOPIA_ADAPTERA`, osobno Linux i `--platform win32`. `KOPIA_ADAPTERA` oznacza
wyłącznie prywatny plik odtworzony loaderem, nie podmianę źródła produkcyjnego.
Semgrep sprawdził te same prywatne bajty w katalogu zachowującym prefiks `bot_core/`.

## Warunki dopuszczenia przyszłego wdrożenia

Przed wdrożeniem konieczne są: niezależna akceptacja dokładnego patcha adaptera
i serwera; zatwierdzony właściciel/loader expected; rozwiązanie obecnego K4
bez fallbacku albo odrębny ACCEPT i dowody nowej podstawy; zamknięte podpisane
closure i autoryzowane zmiany właściwych pinów; regresje całej kompozycji;
natywne dwa UID, rzeczywiste SQL/ACL i trwałość wraz z utratą odpowiedzi COMMIT;
odrębna kwalifikacja instalacji, startu, restartu/lifecycle oraz pozostałe
bramki Stage 9. Żaden test TEST_ONLY nie zastępuje tych dowodów.

Do spełnienia tych warunków wynik dopuszczenia pozostaje **BLOCKED**.
`deployment/stage9_current_status.json`, konfiguracja PG/role, produkcyjny
AccountGenesis i verifier, manifesty, podpisy, klucze oraz piny pozostają
poza zakresem zmian. Bez merge i bez restartowania istniejących workflow.
