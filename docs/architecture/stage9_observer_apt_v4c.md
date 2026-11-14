# Stage 9 / Observer 3A — podstawa akceptacji APT i checkpointów V4c

V4c dodaje osobną, nieprodukcyjną warstwę ponad
[V4b](stage9_observer_apt_v4b.md): zamknięte kontrakty podpisanej akceptacji,
checkpointu i dowodu czasu oraz czysty model proponowania następnego stanu.
Podstawą jest `main` po PR #3270:
`4d43ef1ae8687c86ec8a41d3659f3e3e6b0c22f6`.

Implementacja znajduje się w
[`stage9_observer_trust_v4c.py`](../../deployment/stage9_observer_trust_v4c.py),
[`stage9_observer_checkpoint_v4c.py`](../../deployment/stage9_observer_checkpoint_v4c.py)
i [`stage9_observer_apt_v4c.py`](../../deployment/stage9_observer_apt_v4c.py).
Dotychczasowe API, wyniki i statusy V1–V4b pozostają odrębnymi kontraktami.

## Granica zaufania

Żaden badany dokument nie ustanawia własnego autorytetu. Materiały nie mogą
dostarczyć klucza z uprawnieniami, obniżyć progu podpisów ani zatwierdzić
pochodzenia własnej polityki. Nie istnieje TOFU, pobieranie kluczy/polityk,
domyślny root ani automatyczna rotacja. Poprawność podpisu nie potwierdza
uprawnienia podpisującego do zatwierdzania APT.

Jedyny model kotwicy w tym PR to jawnie przekazany `TestAuthorityV4c` ze stałym
`scope=SYNTHETIC_TEST_ONLY`. Nie ma produkcyjnego odpowiednika ani adaptera
niezależnej akceptacji. Osobne argumenty `acceptance_authority`,
`checkpoint_authority` i `time_authority` wybierają kotwice trzech ról. Nie ma
fallbacku między rolami ani dziedziczenia ich z kontekstu V4b, pliku wejściowego
lub innego subsystemu Stage 9. API testowe nie wymusza niezależnej administracji
trzema rolami; jej ustanowienie pozostaje wymaganiem zewnętrznym.

Przejrzano istniejące kontrakty Observera Stage 9 oraz mechanizm
`windows_stage9_root_of_trust_freeze._verify_envelope`. Z tych innych kontraktów
współdzielona jest wyłącznie zgodna
kanonizacja `deployment.canonical_json.canonical_json_bytes`; weryfikację
Ed25519 wykonuje istniejąca zależność `cryptography`. Klucze, domeny i autorytet
innych mechanizmów Stage 9 nie są przenoszone do APT.

| Wynik techniczny | Co potwierdza | Pozostawiona granica |
| --- | --- | --- |
| Weryfikacja podpisów | Każdy podpis odpowiada wskazanemu kluczowi TEST i dokładnym podpisanym bajtom | Brak niezależnie zatwierdzonego autorytetu |
| Zgodność dokumentów | Dokładne powiązanie kontekstu, keyringu, repozytorium, snapshotu i checkpointu | Nazwy oraz źródła pozostają deklaracjami TEST |
| Porównanie czasu | Podane wartości spełniają polityki ważności | Podpis TEST nie ustanawia aktualnego, niezależnie uwierzytelnionego czasu |
| Poprawne przejście | Propozycja spełnia reguły następstwa wobec wskazanego poprzednika | Brak dowodu, że poprzednik jest aktualnym zaakceptowanym stanem, i brak trwałego zapisu |

Nawet kompletny samodzielnie podpisany zestaw materiałów nie daje niezależnej
akceptacji. Bez osobno przekazanych kotwic TEST odmawia już techniczna
weryfikacja podpisów. Z kotwicami TEST możliwy jest wyłącznie pozytywny wynik
techniczny; autoryzacja pozostaje zablokowana.

## Wspólny format podpisanych dokumentów

Istnieją trzy rodzaje dokumentu: `acceptance`, `checkpoint` i `time`.
Każda koperta ma dokładnie pola `payload` oraz `signatures`. Nie przyjmuje
`public_key`, `threshold`, statusu zatwierdzenia ani innych dodatkowych pól.
Każdy element `signatures` ma dokładnie `signer_id` i `signature_hex`.

Kanonizacja i ograniczenia są obowiązkowe przed weryfikacją kryptograficzną:

- Wejście jest niepustym `bytes` w poprawnym UTF-8, najwyżej 65 536 bajtów
  na całą kopertę; maksymalna głębokość zagnieżdżenia wynosi 8.
- Duplikaty kluczy JSON, liczby zmiennoprzecinkowe, NaN, Infinity, niesparowane
  surogaty, nieznane i brakujące pola są odrzucane. Wszystkie obiekty, również
  zagnieżdżone, mają zamknięte zbiory pól.
- Całe wejście musi być identyczne z wynikiem wspólnego kanonizatora: klucze
  obiektów sortowane według bajtów UTF-16BE, zwarte separatory `,` i `:`,
  ciągi z `ensure_ascii=False`, brak dodatkowego whitespace, BOM i końcowego LF.
  Kolejność tablic zostaje zachowana. Odmowa nie normalizuje wejścia.
- Liczby są dokładnie typem całkowitym, bez booli, od 0 do `2**53-1`.
  Tekstowy token liczby ma najwyżej 16 znaków. Węższe ograniczenia pól obowiązują
  dodatkowo. Znaczniki czasu to całkowite sekundy Unix od 0 do 253402300799.
- Identyfikator ma 1–256 znaków ASCII i pasuje do
  `[A-Za-z0-9][A-Za-z0-9._:/+#=-]{0,255}`.
- SHA-256 to dokładnie 64 małe cyfry szesnastkowe; samych zer nie dopuszcza
  kontrakt podpisanych dokumentów. Fingerprint APT ma dokładnie 40 wielkich
  cyfr szesnastkowych i nie może składać się z samych zer.
- Koperta zawiera 1–8 podpisów, posortowanych rosnąco po `signer_id`, bez
  duplikatów. Podpis Ed25519 ma dokładnie 128 małych cyfr szesnastkowych,
  reprezentujących 64 bajty. Niepoprawna kolejność także oznacza odmowę.

`algorithm` jest dokładnie `ed25519-sha256`. Dla rodzaju `kind` wiadomość ma
postać:

```text
ASCII(domain[kind]) || 0x00 || SHA256(canonical_json_bytes(payload)).digest()
```

Następnie zwykły Ed25519 podpisuje te bajty. Jest jeden NUL; na końcu znajdują
się 32 surowe bajty SHA-256, a nie 64 znaki hex. Nie jest to Ed25519ph.
SHA-256 całej koperty, używany do wiązania checkpointu, jest osobnym hashem
obejmującym również inwentarz podpisów.

| Rodzaj | Dokładna wartość `schema` i `domain` |
| --- | --- |
| `acceptance` | `stage9-apt-context-acceptance-v1` |
| `checkpoint` | `stage9-apt-checkpoint-v1` |
| `time` | `stage9-apt-time-evidence-v1` |

Każdy `payload` zawiera dokładnie pola wspólne oraz pola swojego rodzaju.

| Pole wspólne | Ograniczenie |
| --- | --- |
| `schema`, `domain` | Dokładne wartości z tabeli powyżej dla danego rodzaju |
| `algorithm` | `ed25519-sha256` |
| `authority_id` | Identyfikator, zgodny z osobno przekazaną kotwicą danej roli |
| `policy_version` | Dokładna wersja polityki autorytetu, liczba od 1 do 2147483647; równość z kotwicą, bez przyjmowania nowszej wersji automatycznie |
| `valid_from`, `valid_until` | Znaczniki czasu, `valid_from < valid_until`; wszystkie przedziały ważności są lewostronnie domknięte i prawostronnie otwarte |

`TestAuthorityV4c` ma `authority_id`, `policy_version`, `keys`, `threshold`,
`valid_from`, `valid_until`, `revoked` i stały zakres TEST. `keys` jest krotką
1–8 `TestAuthorityKeyV4c`; klucz zawiera `signer_id`, 32-bajtowy
`public_key_hex`, `valid_from`, `valid_until`, `revoked`. Identyfikatory i bajty
kluczy są unikalne; klucz zerowy jest odrzucany. Próg jest dokładnie liczbą
całkowitą od 1 do liczby kluczy i pochodzi wyłącznie z osobnego argumentu API.

Przedział dokumentu musi zawierać czas oceny i mieścić się w przedziale
kotwicy. Kotwica, każdy użyty klucz i dokument muszą być ważne przy czasie
oceny; cofnięta kotwica lub użyty cofnięty klucz powodują odmowę. Każdy podpis,
w tym każdy ponad próg, musi pochodzić od znanego, dozwolonego klucza i być
poprawny. Osiągnięcie progu nie pozwala zignorować dodatkowego błędnego podpisu.

## Zamknięty payload akceptacji

Do pól wspólnych dochodzi dokładnie następujący zbiór:

| Pole | Znaczenie i reguła |
| --- | --- |
| `context_sha256` | SHA-256 pełnego kontekstu V4b według reprezentacji opisanej niżej |
| `keyring_sha256` | Dokładny SHA-256 keyringu z kontekstu V4b |
| `allowed_signers` | Tablica 1–32 obiektów z dokładnie `signing_fingerprint`, `primary_fingerprint`; unikalne pary uporządkowane leksykograficznie według obu wartości |
| `repository_id`, `snapshot_id` | Identyfikatory równe kontekstowi i obu pozostałym dokumentom |
| `snapshot_sha256` | SHA-256 dokładnych bajtów Release; równy kontekstowi i wynikowi V4b |
| `sequence` | Sekwencja od 0 do `2**53-1`, identyczna z kandydatem checkpointu |
| `issued_at` | Deklarowany czas podpisania akceptacji; wewnątrz jej przedziału i przedziałów użytych kluczy, nie późniejszy niż czas oceny |
| `checkpoint_id` | Identyfikator podpisanego checkpointu kandydata |
| `checkpoint_sha256` | SHA-256 całej kanonicznej koperty tego checkpointu |
| `rotation_policy` | Dokładnie `EXPLICIT_CONTEXT_LINK` |
| `revocation_policy` | Dokładnie `DENY_REVOKED` |
| `previous_context_sha256` | SHA-256 poprzedniego kontekstu albo `null`; zgodny z polem checkpointu |

Wersja polityki autorytetu jest osobna od `context_version` V4b. Akceptacja
wiąże dokładną wersję kontekstu przez jego pełny hash. Zestaw signerów w
akceptacji jest porównywany z posortowanymi dokładnymi parami kontekstu;
ważność, revocation i pozostałe reguły signerów są dodatkowo związane pełnym
hashem kontekstu i sprawdzane według V4b.

`context_sha256_v4c(context)` sprawdza istniejący kontrakt V4b i oblicza:

```text
SHA256(canonical_json_bytes(recursively_convert_tuples_to_lists(asdict(context)))).hexdigest()
```

Hash obejmuje wszystkie pola dataclass: `schema`, `context_id`,
`context_version`, `allowed_signers`, `approved_algorithms`, `subkey_policy`,
`repository_id`, `snapshot_id`, `release_sha256`, `keyring_sha256`,
`valid_from`, `valid_until`, `revoked`, `time_rules`, `external_authority`.
Zagnieżdżone rekordy są pełnymi obiektami, krotki stają się tablicami przy
zachowaniu kolejności, `None` staje się `null`. Nic nie jest pomijane ani
podmieniane przed hashowaniem. Kanoniczna reprezentacja ma najwyżej 65 536
bajtów. Lokalny `evaluation_time` i referencje V4b są częścią hasha; nie stają
się przez to uwierzytelnionym czasem lub niezależną akceptacją.

## Zamknięty payload checkpointu

Do pól wspólnych dochodzi dokładnie:

| Pole | Znaczenie i reguła |
| --- | --- |
| `checkpoint_id`, `checkpoint_source_id` | Identyfikatory checkpointu i deklarowanego źródła |
| `repository_id`, `snapshot_id` | Dokładne identyfikatory repozytorium i snapshotu |
| `context_sha256`, `snapshot_sha256` | Dokładne SHA-256 kontekstu i Release kandydata |
| `sequence` | Sekwencja od 0 do `2**53-1` |
| `successor_policy` | Dokładnie `EXACT_NEXT_SEQUENCE` |
| `rotation_policy` | Dokładnie `EXPLICIT_CONTEXT_LINK` |
| `revoked` | Dokładny bool; `true` blokuje przejście |
| `previous_checkpoint_id` | Identyfikator poprzednika albo `null` |
| `previous_snapshot_sha256`, `previous_context_sha256` | Hashe poprzednika albo `null`; wszystkie trzy referencje poprzednika są obecne razem lub wszystkie są `null` |
| `permitted_context_successors` | Posortowana, unikalna tablica 0–32 SHA-256 kontekstów dopuszczonych jako następcy |

Checkpoint wejściowy opisuje kandydata, tak samo jak akceptacja. Referencje
`previous_*` łączą go ze stanem poprzednika. Nie wolno użyć checkpointu innego
repozytorium ani odziedziczyć stanu po zmianie źródła. Parser nie tworzy stanu
początkowego: dokument bez referencji poprzednika nie jest samodzielnym
uprawnieniem do bootstrapu.

## Zamknięty payload dowodu czasu

Do pól wspólnych dochodzi dokładnie:

| Pole | Znaczenie i reguła |
| --- | --- |
| `proof_id`, `time_source_id` | Jawne identyfikatory dowodu i deklarowanego źródła |
| `repository_id`, `snapshot_id` | Dokładne identyfikatory badanego repozytorium i snapshotu |
| `snapshot_sha256`, `context_sha256` | Powiązanie czasu z dokładnym Release i kontekstem |
| `checkpoint_id` | Identyfikator checkpointu kandydata |
| `evaluation_time` | Czas oceny w sekundach Unix, wewnątrz przedziału ważności dowodu |

To interfejs materiału zewnętrznego, a nie implementacja źródła czasu.
Weryfikator nie pyta zegara systemowego ani `datetime.now()` i nie generuje
dowodów. Podpis TEST pozwala zbadać integralność i powiązanie kontraktu;
nie dowodzi, że podana chwila jest aktualna. Nie ma produkcyjnego mechanizmu
challenge, ochrony przed odtworzeniem dowodu ani akceptacji `time_source_id`.

Rozdzielone są: timestamp podpisu APT z V4b, `issued_at` akceptacji, Release
`Date`, Release `Valid-Until`, przedziały signerów APT, kluczy autorytetów,
polityki/kontekstu i dowodu czasu oraz osobny `evaluation_time`. Podpisana data
Release nie staje się dowodem aktualności. Brak dowodu czasu albo kotwicy TEST
powoduje odmowę techniczną; nawet poprawne materiały TEST pozostawiają
`authenticated_time=UNVERIFIED` i `freshness=BLOCKED`.

Koperty checkpointu i czasu nie zawierają czasu utworzenia podpisu; ich
przedziały ważności i `evaluation_time` nie są takim timestampem. `issued_at`
akceptacji jest podpisaną deklaracją, a nie zewnętrznym dowodem czasu podpisania.

Dla technicznego porównania Release musi mieć poprawne `Date < Valid-Until`,
nie może być przeterminowany przy `evaluation_time`, a jego data nie może
przekraczać czasu oceny powiększonego o jawny limit przyszłego odchylenia V4b.
Ponownie sprawdzane są reguły podpisu i signerów V4b przy podanym czasie oceny.
Release ma najwyżej 1 MiB, 4096 LF i 4096 znaków na linię; przyjmowany jest
ASCII, LF z końcowym LF, bez CR, z jednoznacznymi polami dat i strefą UTC.
Daty mają dokładny format `Sat, 14 Nov 2026 12:00:00 UTC`, z angielskimi
skrótami dni/miesięcy, dwucyfrowymi dniami i poprawnym dniem tygodnia;
dozwolone równoważne oznaczenia strefy to `GMT` i `+0000`.

## Czysty model następstwa

`SnapshotStateV4c` zawiera `repository_id`, `checkpoint_source_id`,
`checkpoint_id`, `sequence`, `snapshot_sha256`, `context_sha256`,
`checkpoint_sha256`, `permitted_context_successors`, `revoked` oraz stałe
polityki `EXACT_NEXT_SEQUENCE` i `EXPLICIT_CONTEXT_LINK`.
`SnapshotCandidateV4c` dodatkowo zawiera trzy referencje `previous_*`.

`propose_transition_v4c` nie czyta plików i niczego nie zapisuje. Przyjmuje
niemutowalny poprzedni stan, kandydata i dwie dostarczone obserwacje SHA-256
poprzedniego checkpointu. Obie obserwacje muszą być sobie równe oraz równe
`previous_state.checkpoint_sha256`. Równość dowolnych dwóch tokenów nie jest
wystarczająca i nie dowodzi autentyczności źródła.

Obowiązują wszystkie reguły:

1. Brak znanego stanu, uszkodzona reprezentacja, brak obserwacji lub zmiana
   obserwacji oznaczają odmowę; nie ma TOFU.
2. Sekwencja niższa oznacza rollback. Ta sama sekwencja z innym snapshotem
   oznacza equivocation. Ta sama sekwencja wymaga ponadto identycznego kontekstu,
   źródła, identyfikatora i hasha całej koperty checkpointu oraz listy rotacji;
   tylko taki dokładny replay może być technicznie idempotentny.
3. Nowa sekwencja musi być dokładnie o jeden wyższa i wskazywać poprzedni
   checkpoint, snapshot i kontekst. Wymaga nowego identyfikatora i hasha
   checkpointu. Pominięcie sekwencji jest odmową.
4. Kontekst pozostaje ten sam albo nowy hash był jawnie dozwolony w podpisanym
   poprzedniku. Reguły dalszych rotacji pochodzą wyłącznie z nowego checkpointu;
   wcześniejsze granty nie są automatycznie dziedziczone.
5. Cofnięty poprzednik lub kandydat oznacza odmowę. Zmiana repozytorium lub
   źródła także oznacza odmowę.
6. Odmowa i przerwanie operacji nie zwracają proponowanego stanu ani częściowego
   sukcesu. Poprzednik pozostaje niezmieniony.

Model niezależnie od wyniku zachowuje `source_acceptance=UNVERIFIED`,
`durable_state=BLOCKED`, `authorization=BLOCKED`. Samodzielne użycie modelu
zakłada uprzednią osobną weryfikację podpisanych bajtów obu checkpointów;
nie ustanawia jej przez konstruktor dataclass.

Warstwa integracji przyjmuje również `previous_checkpoint_raw`, weryfikuje
jego podpis pod jawną kotwicą roli checkpointu i porównuje hash oraz pełną
projekcję pól stanu z `previous_state`, łącznie z regułami rotacji, revocation
i politykami następstwa. Zgodność podpisanego poprzednika nie dowodzi, że jest
on najnowszym niezależnie zaakceptowanym checkpointem. Przekazane obserwacje
nie są odczytywane przez ten interfejs ani uwierzytelniane jako trwały magazyn.

Nie istnieje produkcyjny writer ani funkcja zastosowania propozycji. Zwykły
plik JSON, jego obecność, podpis TEST, zgodne hashe lub udany lokalny zapis nie
stanowią trwałej ochrony przed rollbackiem. Zastosowanie propozycji wymaga
późniejszego, zatwierdzonego, atomowego i trwałego mechanizmu zapisu.

## Integracja i atomowy wynik V4c

`verify_apt_acceptance_v4c` przyjmuje wynik V4b, dokładny kontekst i bajty
Release, trzy podpisane dokumenty kandydata, podpisany checkpoint poprzednika,
trzy osobno wybrane kotwice TEST, poprzedni stan i obserwacje jego hasha.
Wynik V4b jest wyłącznie czytany. Wynik nie zastępuje uruchomienia właściwej
weryfikacji V4b w zaufanym, później kwalifikowanym procesie: przekazany obiekt
Python także nie uwierzytelnia własnego pochodzenia.

V4c wymaga technicznie poprawnego wyniku V4b, ścisłego powiązania wszystkich
dokumentów i pełnego poprawnego przejścia. Zwraca oddzielny `AptResultV4c`.
Przy `technical_validation=REFUSED` nie zwraca proponowanego stanu. Przy
`technical_validation=VALID` zwraca wyłącznie niemutowalną propozycję oraz
jawne powody nadal obowiązujących blokad. `refused=True` i `acceptance=BLOCKED`
są stałe także po pozytywnej walidacji technicznej. Nie ma częściowej autoryzacji
ani opcji ustawienia `VERIFIED` przez argument konstruktora.

Wyjątki formatu, brak materiałów, nieznany signer, podstawiony klucz, błędny
podpis/próg, duplikaty, revocation, nieważność czasowa, niezgodność powiązań,
przekroczenie limitów, błąd I/O i przerwanie operacji prowadzą do odmowy.
Kody przyczyn są stałymi diagnostykami; treść nieufnych dokumentów nie jest
w nich odzwierciedlana.

## Otwarte wymagania niezależnej akceptacji

| ID | Wymagany dowód lub mechanizm | Status po V4c |
| --- | --- | --- |
| V4C-A01 | Niezależne zatwierdzenie autorytetów APT, ich zakresu, dokładnych kluczy i wersji polityk poza badanym zbiorem materiałów | UNVERIFIED |
| V4C-A02 | Zatwierdzone custody i dostarczanie roots, separacja ról, podpisane i niezależnie zatwierdzone rotacje/revocation oraz obsługa kompromitacji | UNVERIFIED |
| V4C-A03 | Niezależnie uwierzytelnione źródło aktualnego czasu, powiązanie żądania i odpowiedzi oraz ochrona przed odtworzeniem dowodu czasu | UNVERIFIED |
| V4C-A04 | Uwierzytelnione, aktualne źródło checkpointów i niezależny wybór właściwego poprzednika; sam podpis starego checkpointu nie wystarcza | UNVERIFIED |
| V4C-A05 | Chroniony trwały stan: atomowy compare-and-swap, synchronizacja trwałego zapisu, odporność na crash/przerwanie, recovery, równoległość i odtworzenie starego magazynu | BLOCKED |
| V4C-A06 | Akceptacja binariów/narzędzi, pełnego runtime, środowiska importów i izolacji OS; dowody wykonania w docelowym profilu | BLOCKED |
| V4C-A07 | Niezależna kwalifikacja Observera 3A, osobna autoryzacja 3B, instalatora, launchera i dostępu uprzywilejowanego | BLOCKED / NOT AUTHORIZED |

## Dokładny zakres pozostawiony do V5 i późniejszej kwalifikacji

V5 nie jest implementowane w tym PR. Kolejny etap musi najpierw rozstrzygnąć
V4C-A01–A04 i dostarczyć produkcyjne adaptery źródeł, bez przekształcania
kotwic TEST w roots. Musi ustalić zatwierdzone reguły aktualności,
separacji/dostarczenia polityk i rotacji, challenge/antireplay czasu oraz
aktualności i unieważniania checkpointów.

Osobno pozostaje V4C-A05: integracja proponowanego stanu z chronionym,
atomowym, trwałym magazynem, z testami utraty zasilania, crash/recovery,
równoczesnych prób i odtworzenia starego stanu. Zwykły odczyt/zapis JSON nie
realizuje tego zakresu. Wymagane będą ponadto V4C-A06–A07: akceptacja
rzeczywistego runtime i narzędzi, izolacji, dowodów funkcjonalnych i właściwych
bramek Linux/Windows. Nie wolno wywodzić tych uprawnień z samych testów V4c.

W V4c nie uruchamia się badanych pakietów ani instalatora, nie wykonuje prób
jako root, nie aktywuje standardowych parserów Observera lub Stage 3B i nie
zmienia bramek gotowości Linux/Windows. Syntetyczne testy weryfikują kontrakt
i odmowy; nie zastępują niezależnego zatwierdzenia ani kwalifikacji fizycznej.

```text
Stage 9 = IN_PROGRESS
Observer 3A qualification = BLOCKED
Stage 3B = NOT AUTHORIZED
Production = NO-GO

External authority acceptance = UNVERIFIED
Authenticated time = UNVERIFIED
Durable anti-rollback = BLOCKED
Runtime qualification = BLOCKED
```
