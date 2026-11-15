# V5c2: przyszły kontrakt rekonstrukcji PYC

`deployment/stage9_observer_pyc_reconstruction_v5c2.py` udostępnia wyłącznie
niemutowalny kontrakt danych i ograniczoną walidację jego struktury. Nie uruchamia
interpretera ani źródła, nie tworzy PYC i nie analizuje obiektów `marshal`.
`STRUCTURE_VALID` oznacza spójność deklaracji z przekazanym osobno oczekiwaniem.
Nie oznacza wykonania, akceptacji oczekiwania ani autentyczności deklaracji.

## Dane i powiązania

`PycReconstructionPolicyV5c2` ma profil
`stage9-observer-pyc-reconstruction-policy-v5c2`. Wszystkie pola są wymagane;
walidator nie uzupełnia ich pomiarami badanego artefaktu.

| Pole | Wymaganie strukturalne |
| --- | --- |
| `interpreter_sha256`, `interpreter_build_id` | Pełny SHA-256 oraz pełny 20-bajtowy build ID profilu CPython 3.12.3. Zgodność z osobno przekazaną polityką jest obowiązkowa. |
| `interpreter_package` | `ReconstructionPackageV5c2`: nazwa `python3.12-minimal`, wersja `3.12.3-1ubuntu0.17`, `amd64`, SHA-256 DEB i snapshotu. |
| `required_libraries` | Niepusty, ograniczony wykaz `ReconstructionLibraryV5c2`: literalna ścieżka, SHA-256, pełny build ID i tożsamość pakietu każdej biblioteki. Ścieżki są unikalne, snapshot zgodny z interpreterem. |
| `source` | Pełny `SourceBindingV5c1`: source ID, pakiet, wersja, architektura, snapshot, SHA-256 i rozmiar DEB, literalny członek TAR, record ID, SHA-256 i rozmiar źródła. |
| `parser_version` | Jawna wersja `CPython-3.12.3`. |
| `encoding_configuration` | Jawna konfiguracja `pep263-utf8-default-v1`: przyszły parser ma interpretować surowe bajty zgodnie z PEP 263 i domyślnym UTF-8; obecny walidator ich nie dekoduje. |
| `co_filename`, `optimization`, `invalidation` | Bezwzględna nazwa pliku, dokładnie całkowity poziom 0–2, wyłącznie `checked_hash`. |
| `arguments`, `environment` | Pełne literalne argumenty i uporządkowane pary zmiennych środowiska. Nazwy zmiennych muszą być unikalne. Nie są przekazywane procesowi. |
| `isolation_requirements` | Niepusty, unikalny wykaz literalnych ograniczeń izolacji; ich egzekwowanie pozostaje niezweryfikowane. |
| `tool_image_id`, `tool_image_sha256` | Identyfikator i pełny SHA-256 obrazu narzędziowego. |
| `expected_image_sha256`, `expected_pyc_sha256` | Osobno przypięte oczekiwanie obrazu i pełnego wyniku PYC. Nie powstają z deklarowanego wyniku. |

`PycReconstructionEvidenceV5c2` ma profil
`stage9-observer-pyc-reconstruction-evidence-v5c2`. Obejmuje pełną politykę,
deklarowany `output_pyc_sha256`, `run_id`, `evidence_id`, opis pochodzenia
`evidence_origin`, `interrupted`, `interruption_reason` i
`verification_status="RECONSTRUCTION_UNVERIFIED"`. Każdy status sugerujący
zweryfikowaną rekonstrukcję jest odrzucany. Brak przerwania wymaga `None` jako
powodu; zadeklarowane przerwanie wymaga niepustego opisu i zwraca `INCOMPLETE`.
Kontrakt nie zawiera podpisu, zaufanego zegara ani pola nadającego uprawnienia
kolektorowi.

## Wywołanie i znaczenie wyniku

`validate_reconstruction_evidence_v5c2(evidence, *, expected_policy,
source_bytes, limits=Limits())` wymaga osobnego argumentu `expected_policy`.
Porównuje całą politykę, a następnie deklarowany SHA-256 wyniku z oczekiwaniem.
Weryfikuje także dokładny rozmiar i SHA-256 surowych `source_bytes`, bez
normalizacji końców linii, dekodowania, importu czy kompilacji. Integrujący
wywołujący musi uzyskać te bajty z niezmienionego adaptera V5b. Ta funkcja nie
zastępuje V3/V5b i nie nadaje dostarczonym bajtom zweryfikowanego pochodzenia.

Oczekiwanie przekazane przez wywołującego pozostaje niezaakceptowane. Przekazanie
tego samego obiektu jako polityki deklaracji i `expected_policy` może potwierdzić
tylko strukturę; nie dowodzi niezależności oczekiwania. Do V5c2 nie wprowadzono
autorytetu, który mógłby tę niezależność zaakceptować. Analogicznie dowolny tekst
w `evidence_origin`, w tym twierdzenie o zaufanym podpisie, pozostaje opisem
dostarczonym przez wywołującego.

Wynik `PycReconstructionContractResultV5c2` ma osobny profil
`stage9-observer-pyc-reconstruction-result-v5c2` i jeden ze statusów:

- `STRUCTURE_VALID`: wszystkie wymagane pola i ich powiązania są spójne;
- `NOT_DECLARED`: brak deklaracji;
- `MISMATCH`: podmieniona tożsamość, źródło lub oczekiwany wynik;
- `INCOMPLETE`: wadliwy format, brak oczekiwania/bajtów, przerwanie albo limit.

Każda odmowa usuwa pole `declaration`; nie pozostają częściowe wyniki pozytywne.
Limity czasu, danych i konserwatywnie liczonej pamięci pochodzą z istniejącego
`Limits`; można je zaostrzyć, nie zwiększyć. Nie są limitem RSS systemu
operacyjnego. Ograniczone operacje kontrolują wspólny termin.

Nawet dla `STRUCTURE_VALID` wynik zawsze zachowuje:

```text
reconstruction = RECONSTRUCTION_UNVERIFIED
PYC_RECONSTRUCTION = UNVERIFIED
execution = MISSING
evidence_origin = UNVERIFIED
External authority acceptance = UNVERIFIED
Authenticated time = UNVERIFIED
Durable anti-rollback = BLOCKED
Runtime qualification = BLOCKED
Stage 9 = IN_PROGRESS
Observer 3A qualification = BLOCKED
Stage 3B = NOT AUTHORIZED
Production = NO-GO
```

## Brakujące dowody i następny podetap

Obowiązkowe braki obejmują niezależne świadectwo wykonania, zaakceptowane
pochodzenie kolektora, akceptację polityki oczekiwań, pomiar rzeczywistych bajtów
interpretera i bibliotek oraz dowód egzekwowania izolacji. Weryfikacja SHA-256
zadeklarowanego wyniku nie stanowi pomiaru wygenerowanego pliku. Zielone testy
syntetycznych deklaracji nie usuwają żadnego z tych braków.

Następny, osobny podetap wymaga uprzedniego zatwierdzenia granic izolacji i
zasad niezależnego dowodu wykonania. Musi identyfikować i mierzyć rzeczywisty
interpreter, biblioteki, obraz i wejściowe bajty, zebrać niezależny dowód
konkretnego uruchomienia, przerwań oraz ustawień parsera/kompilatora, zmierzyć
rzeczywisty wynik i porównać go z niezależnie zaakceptowanym oczekiwaniem.
V5c2 nie implementuje ani nie autoryzuje tego uruchomienia. Poprawny hash
źródła i zgodny SHA-256 gotowego PYC nadal nie dowodzą poprawności code object.
