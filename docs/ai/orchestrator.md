# Decision Orchestrator – adaptacyjne sterowanie strategiami

Niniejszy dokument opisuje rozszerzenia modułu `DecisionOrchestrator` odpowiedzialne za
adaptacyjny wybór strategii oraz rekomendacje ryzyka. Aktualizacja łączy kontekstową
bandytę LinUCB z próbkowaniem Thompsona tak, aby przy każdej ewaluacji kandydata:

* wykorzystywać metadane modelu, reżim rynku oraz sygnały meta-labelingu,
* wyznaczać rekomendowane tryby wykonania strategii (np. `live`, `shadow`),
* sugerować dopasowaną wielkość pozycji dla managera ryzyka.

## Kontekstowa eksploracja strategii (LinUCB)

Nowa klasa pomocnicza `_StrategyBanditAdvisor` agreguje historię ewaluacji
wielowymiarowych kontekstów. Dla każdej pary `(strategia, reżim)` utrzymywany jest stan
ramienia LinUCB (`_LinUCBArm`). Wektor cech powstaje z:

* prawdopodobieństwa sukcesu i oczekiwanego zwrotu modelu,
* netto edge po kosztach,
* wagi/wyniku wybranego modelu inference,
* zaufania do metadanych (`model_confidence`, `model_metadata.confidence`, …),
* oceny meta-labelingu,
* zakodowanego reżimu rynku (`trend`, `mean_reversion`, `daily`).

Rezultatem jest wartość eksploracyjna, która determinuję rekomendację trybu pracy:
`("live", "aggressive")`, `("live", "balanced")`, `("shadow", "defensive")` lub
`("disabled", "monitor")`.

## Sugestie ryzyka (Thompson sampling)

Równolegle utrzymywany jest prosty model Beta-Bernoulli (`_ThompsonArm`). Każda
obserwacja wykorzystuje meta-label (jeżeli dostępny) oraz wynik ewaluacji (akceptacja,
netto edge) do aktualizacji parametrów. Średnia a posteriori stanowi komponent wskaźnika
ryzyka, który dodatkowo uwzględnia prawdopodobieństwo sukcesu modelu oraz bias zależny od
regime’u.

Na podstawie ryzyka wyliczana jest sugerowana wielkość pozycji – domyślnie zakres od 40%
notional do maksymalnie 160%, co umożliwia zarówno redukcję, jak i skalowanie ekspozycji.

## Nowe pola w odpowiedziach

* `DecisionEvaluation.recommended_modes` – krotka rekomendowanych trybów pracy.
* `DecisionEvaluation.recommended_position_size` – sugerowana wartość notional (float).
* `DecisionEvaluation.recommended_risk_score` – skalowany (0–1) wskaźnik ryzyka użyty przy
  kalkulacji rekomendacji.
* `ModelSelectionMetadata.recommended_modes` – mirror rekomendacji przekazywany do
  komponentów korzystających z metadanych selekcji.
* `ModelSelectionMetadata.recommended_position_size` – rekomendowana wielkość pozycji
  w kontekście konkretnego wyboru modelu.
* `ModelSelectionMetadata.recommended_risk_score` – współdzielony wskaźnik ryzyka do
  wykorzystania w pipeline’ach downstream.

Serializacja (`to_mapping`) zwraca nowe pola jako listę trybów oraz wartość liczbową, co
zapewnia kompatybilność z istniejącymi konsumentami API.

## Obsługa reżimów i meta-labelingu

`DecisionOrchestrator` udostępnia metodę `_resolve_regime`, która odczytuje reżim z
metadanych kandydata (`market_regime`, `regime`, zagnieżdżone mapowania). W przypadku
niepoprawnych wartości przyjmowany jest domyślny `trend`.

Meta-label jest pozyskiwany z kluczy `meta_label`, `meta_label_score`, `meta_probability`
oraz bloku `meta_labeling`. Wartości są normalizowane do przedziału `[0, 1]`.

## Integracja z cyklem ewaluacji

1. Przed utworzeniem `DecisionEvaluation` orkiestrator wywołuje `recommend(...)`,
   otrzymując tryby i wielkość pozycji.
2. Rekomendacje są doklejane do `ModelSelectionMetadata` (jeżeli istnieje) oraz do
   końcowego wyniku ewaluacji.
3. Po zakończeniu ewaluacji wywoływane jest `observe(...)`, które aktualizuje zarówno
   ramię LinUCB, jak i rozkład Thompsona bazując na faktycznym net edge oraz statusie
   akceptacji.

W razie potrzeby orchestrator umożliwia wstrzyknięcie alternatywnego doradcy poprzez
parametr `strategy_advisor`. Testy mogą w ten sposób podmienić logikę eksploracji na
deterministyczną implementację, zachowując interfejs `recommend/observe`.

## Testy

Plik `tests/test_decision_orchestrator.py` został rozbudowany o scenariusze walidujące:

* obecność rekomendowanych trybów i pozycji w wynikach,
* propagację rekomendacji do metadanych selekcji modelu,
* poprawność integracji ze stubem inference (wymuszenie wykorzystania cech).
* deterministyczne scenariusze z wstrzykniętym doradcą w `tests/decision/test_orchestrator.py`.

Dzięki temu regresje w logice adaptacyjnej będą wychwytywane w trakcie CI.

## Podsumowania Decision Engine

`DecisionSummaryAggregator` uwzględnia nowe pola rekomendacji. Najnowsze podsumowanie
(`latest_recommended_modes`, `latest_recommended_position_size`,
`latest_recommended_risk_score`) oraz histogramy (`avg_recommended_position_size`,
`avg_recommended_risk_score`, itp.) bazują na statystykach `_StatAccumulator`. Dzięki temu
dashboardy korzystające z agregatora otrzymują zarówno bieżące rekomendacje, jak i ich
skumulowane rozkłady.
