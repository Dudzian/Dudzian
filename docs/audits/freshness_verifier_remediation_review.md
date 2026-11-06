# FreshnessVerifier — remediacja czterech usterek z S9-ID-02

6 listopada 2026 r. Baza: `0ff0f604fd5ceaf6909acdfdeaf16e710d66d445`,
`main` po merge [#3256](https://github.com/Dudzian/Dudzian/pull/3256).
Punkt odniesienia: [raport S9-ID-02](stage9_python_verifier_feasibility_s9_id02.md).
**Materiał do niezależnego przeglądu, bez nowego wydania produkcyjnego.**

## Granica podpisanego closure

`bot_core/freshness_semantic_verifier.py` jest w `REQUIRED_FILES` i ma zatwierdzony
SHA-256 `64dd105d54a2eabd9be9164846fe642ca0a274b30ca5d0a83735a4436d9f39e1`
w [freshness_application.py:24–51](../../deployment/freshness_application.py#L24)
oraz [freshness_deployment_qualification.py:61–68](../../bot_core/freshness_deployment_qualification.py#L61).
Zmiana tego pliku wymaga ponownego review dokładnych bajtów i niezależnego
zatwierdzenia wydania. Sam podpis nowego manifestu nie wystarczy przy starym pinie;
`verify_installed_application` sprawdza także `REVIEWED_SUBSTRATE_DIGESTS`.

Zgodnie z warunkiem zadania Draft zawiera testy, plan i dokładny
[proponowany diff](../../tests/fixtures/freshness_verifier_remediation_review.patch).
**Produkcyjny moduł, oba piny, podpisy i manifesty pozostają bez zmian.**
Diff jest danymi testowymi: helper stosuje go przez `git apply --unidiff-zero`
(najpierw `--check`) wyłącznie do prywatnej kopii w `tmp_path`, po sprawdzeniu
dokładnego SHA źródła. Nie jest launcherem ani alternatywą
produkcyjnej bramki. Nowe testy domyślnie badają tę kopię; wynik PASS oznacza
walidację propozycji, nie naprawienie zatwierdzonego artefaktu.
`test_proposed_source_is_rejected_by_existing_source_gate` dowodzi odmowy nowych
bajtów przez rzeczywistą bramkę, z modelowanymi wyłącznie metadanymi instalacji.
Piny nie są podmieniane w tej kontroli. Osobny test potwierdza niezmieniony SHA
oryginału i obu zatwierdzeń.

Przeczytano `AGENTS.md`; Codebase Memory i jego CLI pozostają niedostępne.
Wnioski sprawdzono bezpośrednio w źródłach. Nie uruchamiano hostowych usług,
administracyjnego PG, sysctl, LSM ani testów uprzywilejowanych.

## Odtworzenie i najmniejsza naprawa

Numery linii odnoszą się do bazy; diff podaje dokładne zmiany.

| Problem i dowód | Testy ujemne | Proponowana naprawa / wpływ |
| --- | --- | --- |
| Niekompletna ramka blokuje sekwencyjny handler: `serve_local_unix_socket:728–732`, `_receive_exact:779–786`. | `test_partial_frame_deadline_closes_without_preparation` (header/body), `test_slow_client_cannot_reset_absolute_read_deadline`. W bazie odczyt nie kończy się przed zewnętrznym watchdogiem; zero przygotowań. | Jeden monotoniczny deadline od accept, istniejące 3,0 s; przed każdym recv obliczanie remaining. Ten sam deadline obejmuje EOF i send, bez odnawiania per bajt/fazę. Brak nowego publicznego parametru. Usuwa odtworzoną blokadę transportową, nie dowodzi odporności na dowolny DoS. |
| Kontrolny SELECT może zgubić własność sesji: `PostgreSQLVerifierConnection.connect:153–172`. | `test_session_check_error_closes_once_and_preserves_primary` (execute/fetchone), także nieoczekiwany błąd SELECT, niezgodne facts, błędy connect i poprawny zwrot. Model psycopg w bazie obserwuje `close=0` po błędzie SELECT. | `connection=None`, flaga przekazania własności, `finally` zamyka uzyskane, nieprzekazane połączenie dokładnie raz. Poprawna sesja pozostaje otwarta dla caller. Nie otaczać zwracanego połączenia contextem, który zamknąłby je przed użyciem. Close nie wykonuje prepare ani commit. |
| Błąd odpowiedzi przerywa usługę: `serve_local_unix_socket:774`, poza istniejącym try. | `test_disconnect_during_response_does_not_stop_server`: rzeczywisty klient znika po wejściu writera, przed zwolnieniem odpowiedzi; w bazie `BrokenPipeError` kończy listener. Osobno utrata odpowiedzi po timeout klienta. | Wąskie `except OSError` przy remaining/sendall kończy tylko bieżące połączenie; licznik obsłużonych prób nadal rośnie. Następny klient trafia do tego samego listenera. Bez ponownego send, reconnect, prepare ani zapewnienia rollbacku. |
| Brak EOF przed przygotowaniem: `serve_local_unix_socket:728–752`. | `test_trailing_bytes_are_invalid_before_prepare` (bajt/druga ramka), opóźniony trailing byte, pełna ramka bez EOF; w bazie dochodzi do PREPARED. | Po pełnym bounded body `recv(1)` z remaining: tylko `b''` pozwala dalej przejść do parsowania i verifiera. Dodatkowy bajt → INVALID, zero prepare; brak EOF → deadline/close, zero prepare. Sprawdzić remaining ponownie po dekodowaniu, przed verifierem. |

Pierwsze cztery problemy odtworzono **przed utworzeniem diffu**. Początkowy zestaw
tych samych asercji dał 18 FAIL / 15 PASS. Po rozszerzeniu o opóźniony trailing
byte i nieoczekiwane błędy SELECT: **21 FAIL / 15 PASS w baseline**, zgodnie
z listą braków. Nie użyto xfail, wyłączania testów ani obniżania progów.
Domyślny wariant po poprawce oraz kontrole scope wchodzą do zielonego zestawu.

To potwierdzone błędy dostępności i ścisłości framingu. Testy nie wykazują
podszycia pod verifier, obejścia K4, błędnej kryptografii ani nieautoryzowanej
transakcji PG. Liczniki writera i zachowanego efektu są jawnym modelem,
nie dowodem rzeczywistego COMMIT.

## Wyniki i granice protokołu/transakcji

| Sytuacja | Zachowane zachowanie propozycji |
| --- | --- |
| Złe dane, nadmiarowy bajt/ramka, EOF w niepełnym header/body | INVALID; verifier nie przygotowuje kandydata. Wysyłka odpowiedzi zależy od pozostałego czasu i sprawnego kanału. |
| Błąd PG/transportu przed przygotowaniem | Dotychczasowe UNAVAILABLE, jeżeli odpowiedź można dostarczyć. Timeout nie staje się INVALID. |
| Wyczerpany wspólny deadline | Zamknięcie kanału; brak nowego budżetu na odpowiedź. Klient po rozpoczęciu send bez pełnego wyniku zachowuje OUTCOME_UNKNOWN, nawet jeśli kontroler testu wie, że prepare nie ruszyło. |
| Poprawne żądanie z EOF i znane przygotowanie | PREPARED z dotychczasowym bindingiem i identyfikatorem; brak zmiany SQL, podpisów lub formatu. |
| `PreparationOutcomeUnknown` | OUTCOME_UNKNOWN, jeżeli kanał/czas pozwalają przesłać odpowiedź; nigdy INVALID ani zapewnienie rollbacku. |
| Rozłączenie po przygotowaniu / opóźniony writer | Modelowany efekt pozostaje, usługa obsługuje następnego klienta, brak retry. Klient bez pełnego wyniku zwraca OUTCOME_UNKNOWN. |

Deadline ogranicza oczekiwanie na socket. **Nie anuluje connect/execute/COMMIT PG,
CPU ani FFI** i nie dowodzi twardego czasu całego handlera ≤3,0 s. Test opóźnionego
writera potwierdza, że klient może już zwrócić OUTCOME_UNKNOWN, a rozpoczęta
praca nadal trwa i później pozostawia efekt modelu. Nie zmieniano limitów SQL.
Przy cleanup po błędzie SELECT zachowanie pierwotnej odmowy obejmuje wtórne
`psycopg.Error` z `close`; nie wprowadzono ogólnego suppress(Exception/BaseException).

`test_existing_runtime_client_half_close_is_compatible` używa niezmienionego
`RuntimeFreshnessVerifierClient`, rzeczywistego AF_UNIX i obserwacji PIDFD/proc.
Konta i chroniona instalacja są jawnie modelowane, tak jak w istniejących testach;
nie jest to kwalifikacja K4. Klient wykonuje
[shutdown(SHUT_WR):351](../../bot_core/freshness_runtime_ipc.py#L351), więc EOF
jest zgodny z jego protokołem. Test otrzymuje PREPARED i dokładnie jedno prepare.

**Ograniczenie zgodności do rozstrzygnięcia przed zastosowaniem diffu:** starszy
`ProductionLocalFreshnessAuthority._request_preparation_once` w
[AccountGenesis:173–174](../../bot_core/account_genesis_freshness_authority.py#L173)
nie wykonuje half-close i ma odrębne zachowanie retry. Nie zmieniano tej klasy,
nie dopuszczono jej retry do klienta runtime i nie dodano obejścia EOF.
Ta ścieżka z proponowanym verifierem będzie odmawiać bez przygotowania;
wymaga osobnej decyzji o migracji albo wyłączeniu użycia przed zatwierdzeniem.
Także istniejący helper testowy `_ipc_exchange` nie robi half-close: przy
przeniesieniu diffu do produkcyjnego źródła trzeba go dostosować do rzeczywistego
klienta i ponownie uruchomić jego testy przeciwko zmienionemu modułowi.
Obecny zielony zestaw tych starszych testów nadal bada niezmieniony oryginał.

## Walidacja i plan zastosowania

Zestaw istniejących regresji oraz nowych testów propozycji: **920 PASS / 4 SKIP**.
W tym **38 nowych przypadków** (36 transport/PG + 2 scope). Pominięcia: konto
bez dodatkowych grup, test wymagający Windows i dwa jawne root harnessy.
Rzeczywisty transport działa w prywatnych procesach/katalogach AF_UNIX;
nie uruchomiono prawdziwego PG. Testy transportu skracają lokalny budżet do
0,2 s, aby odtworzyć slow client; nie jest to benchmark ani nowy limit produkcji.
Pełne polecenia i końcowe wyniki Ruff, mypy, Semgrep, skanu sekretów,
`ruff-format --all-files` oraz CI należą do opisu Draft PR.

Powtórzenie nowych asercji na oryginale (oczekiwany wynik FAIL dokumentujący usterki):

```bash
FRESHNESS_REMEDIATION_SOURCE=baseline python -m pytest -q -ra tests/security/test_freshness_verifier_transport_remediation.py tests/security/test_freshness_verifier_pg_remediation.py
```

Walidacja propozycji na prywatnej kopii, bez zmiany podpisanego źródła:

```bash
python -m pytest -q -ra tests/security/test_freshness_verifier_transport_remediation.py tests/security/test_freshness_verifier_pg_remediation.py tests/security/test_freshness_verifier_remediation_scope.py
```

Najmniejszy kolejny krok po niezależnym review: rozstrzygnąć zgodność starszej
ścieżki, przenieść przejrzany diff do źródła w osobno zatwierdzonym zakresie,
przenieść regresje na zwykły import zmienionego modułu, ponowić wszystkie bramki
i przeprowadzić osobną procedurę zatwierdzenia closure/wydania. Ten Draft nie
aktualizuje digestów i nie udziela zgody na publikację artefaktu.

**Wszystkie cztery usterki: odtworzone, propozycja naprawy zweryfikowana;
zatwierdzony artefakt produkcyjny pozostaje niezmieniony.** K4/K4-v2,
AccountGenesis, role PG i `deployment/stage9_current_status.json` bez zmian.
Bez restartów GitHub Actions i bez merge. K4-v2 nieaktywny i niezatwierdzony,
`PENDING_OWNER_ACCEPT`; Stage 9: **IN_PROGRESS / PRODUCTION_NO_GO**.
