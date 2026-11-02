# Roadmap — Stage 11: Trading Intelligence Hardening

## Status

**PLANNED — Stage 11; required prerequisite for Stage 12 (Autonomous Strategy Discovery & Promotion)**

**Stage 11 (TIH-1–TIH-7)** jest planowany **po zamknięciu Stage 10 i przed rozpoczęciem produkcyjnego Stage 12 — Autonomous Strategy Discovery & Promotion (ASD)**. Numeracja dotyczy etapów rozwoju produktu: nie zmienia zamrożonych kontraktów M0–M6 ani Windows Stage 0–14. Jego zadaniem jest wzmocnienie jakości danych, rozpoznawania rynku, walidacji, execution i alokacji kapitału tak, aby późniejszy Strategy Discovery nie optymalizował strategii na słabym modelu rynku lub nierealistycznym execution.

## Cel nadrzędny

Przekształcić istniejący pipeline tradingowy CryptoHuntera z dobrego fundamentu decyzyjnego w **production-grade intelligence layer**, który:

- lepiej rozpoznaje aktualny reżim rynku i prawdopodobieństwo jego zmiany;
- potrafi wykrywać warunki sprzyjające lokalnym reversalom;
- podejmuje decyzje wykonawcze z uwzględnieniem realnych kosztów, płynności i pilności;
- odrzuca przewagi powstałe wskutek overfittingu, leakage lub selection bias;
- wykrywa degradację edge'u już po wdrożeniu;
- dynamicznie alokuje kapitał między strategie na poziomie całego portfela;
- potrafi odtwarzać mikrostrukturę rynku tam, gdzie strategia tego wymaga.

Docelowy przepływ:

```text
Market data / L2 / trades / funding / OI
        ↓
Regime Intelligence v2
        ↓
Microstructure & Reversal Intelligence
        ↓
Strategy / AI Governor / DecisionOrchestrator
        ↓
Dynamic Capital Allocation v2
        ↓
Execution Optimizer
        ↓
Risk Engine + ExecutionLease
        ↓
Exchange
        ↓
TCA + Live Edge Decay Monitor
        ↓
feedback / degrade / abstain / recalibrate
```

## Zasady projektowe

1. **Rozbudowujemy istniejące komponenty, zamiast budować równoległe systemy.**
2. AI Governor, DecisionOrchestrator, bandity, PortfolioGovernor, TCO, Risk Engine i ExecutionLease pozostają podstawą orkiestracji.
3. Lepszy sygnał nie może omijać limitów ryzyka ani zasad środowiska.
4. Każdy model kosztów i execution musi być kalibrowany na danych rzeczywistych, a nie tylko na stałych założeniach z backtestu.
5. System musi mieć pełnoprawny wynik `ABSTAIN/SKIP`, jeżeli edge po kosztach jest zbyt mały lub niepewny.
6. Wyniki research muszą być odtwarzalne i point-in-time correct.
7. Rozbudowa mikrostruktury nie oznacza automatycznego przejścia w HFT; zakres ma odpowiadać realnym możliwościom infrastruktury i giełdy.

# TIH-1 — Regime Intelligence v2

## Cel

Zastąpić zbyt grubą klasyfikację rynku warstwą probabilistyczną, multi-timeframe i change-aware.

## Zakres

- klasyfikacja wielowymiarowa zamiast wyłącznie pojedynczej etykiety;
- multi-timeframe regime state;
- confidence i uncertainty dla każdego reżimu;
- transition probability między reżimami;
- change-point detection;
- volatility state i liquidity state jako osobne osie;
- trend strength / trend exhaustion;
- correlation regime;
- risk-on / risk-off proxy na poziomie rynku crypto;
- regime persistence i hysteresis, aby uniknąć przełączania przy szumie;
- kalibracja probability outputs;
- walidacja walk-forward i OOS klasyfikatora.

Przykładowy kontrakt logiczny:

```text
TREND_UP                 0.58
TREND_EXHAUSTION         0.24
MEAN_REVERSION           0.11
HIGH_VOL_TRANSITION      0.07

P(regime_change, 30m)    0.41
confidence               0.78
```

## Kryteria akceptacji

- brak wymuszonej pewnej etykiety przy wysokiej niepewności;
- mierzalna kalibracja confidence;
- stabilność OOS;
- brak look-ahead w konstrukcji cech;
- zachowanie kompatybilności z istniejącym AI Governor / DecisionOrchestrator.

# TIH-2 — Market Microstructure & Reversal Intelligence

## Cel

Wykorzystać istniejące i nowe dane mikrostrukturalne do oceny krótkoterminowej presji kupna/sprzedaży, płynności i prawdopodobieństwa lokalnego zwrotu.

## Zakres

- bid/ask imbalance;
- weighted depth imbalance;
- spread dynamics;
- order-book slope / convexity;
- depth depletion / replenishment;
- aggressive buy/sell trade imbalance;
- cumulative volume delta lub równoważny flow metric;
- absorption heuristics;
- liquidity sweep detection;
- local liquidity vacuum detection;
- short-horizon realized volatility;
- funding deviation;
- basis deviation;
- open-interest impulse;
- liquidation impulse, jeżeli wiarygodne źródło danych jest dostępne;
- divergence między price / flow / OI / funding;
- microstructure feature quality i stale-data detection.

### TIH-2.C — Candlestick / Price Action Feature Research (PLANNED)

**Cel:** zbadać, czy geometria świec i formacje jedno-/wieloświecowe poprawiają jakość decyzji AI ponad istniejące cechy OHLCV, reżimy rynku i koszty egzekucji. To **hipoteza badawcza**, a nie gotowy sygnał kupna/sprzedaży ani obietnica dodatniego edge'u.

- **Reuse-first / audyt:** zinwentaryzować istniejące cechy z `bot_core/ai/feature_engineering.py` (`candle_body_ratio`, `candle_upper_shadow_ratio`, `candle_lower_shadow_ratio`, `close_location_value`, relacje do ATR, Heikin-Ashi); nie liczyć ich drugi raz. Oddzielnie ustalić, które z nich faktycznie docierają do `TradingOpportunityAI` w `bot_core/ai/trading_engine.py`, zamiast zakładać automatyczną integrację.
- **Kandydaci do badań:** Hammer / Hanging Man, Shooting Star, Doji, bullish/bearish Engulfing, Inside Bar, Hikkake, breakout / retest oraz relacje między świecami (kontekst trendu, knotów, wolumenu, wsparć/oporów i reżimu). Traktować nazwy formacji jako jawne, wersjonowane definicje algorytmiczne, nie subiektywną interpretację wykresu.
- **Cechy i dane:** najpierw wykorzystać ciągłe, znormalizowane cechy geometrii OHLCV oraz cechy kontekstowe; binarne etykiety formacji traktować jako kandydatów, nie wymuszoną przewagę. Analizować wiele interwałów tylko przy point-in-time alignment. Do decyzji używać wyłącznie świec zamkniętych i danych dostępnych przed decyzją; brak future-bar confirmation, leakage i repaintingu.
- **Kontrakt:** utrzymać identyczną definicję i kolejność cech w treningu oraz inference; wersjonować `feature_spec`, walidować dostępność/jakość danych, provenance i freshness, a zmiany istniejącego modelu opportunity wdrażać jako nową wersję modelu/kontraktu bez cichego rozszerzania `opportunity_features_v1`.
- **Eksperyment ablation:** porównać bazowy model bez nowych formacji z modelem z formacjami, przy tych samych danych, horyzoncie, universe, reżimach, kosztach i seedach. Ocenić OOS / walk-forward, prawdopodobieństwa i kalibrację, oczekiwany wynik netto po fees/spread/slippage, drawdown, stabilność między parami/interwałami i degradację po perturbacji danych. Uwzględnić multiple testing i zależność prób według TIH-4.
- **Wdrażanie wyłącznie po dowodach:** najpierw research → shadow → paper; bez domyślnych zleceń LIVE, zmian sizingu ani omijania `Risk Engine`, `ExecutionLease`, AI Governor i `DecisionOrchestrator`. Brak spełnienia kryteriów w kompletnym badaniu oznacza **REJECT / no added features**; niepełne dowody oznaczają **INCOMPLETE**, nie domniemanie braku przewagi.

#### Protokół badania i rozstrzygnięcie (TIH-2.C / TIH-4)

Przed eksperymentem i ujawnieniem wyników OOS **zapisać, uzasadnić i zamrozić wersjonowany protokół** określający:

- model bazowy i rozszerzenie oraz porównywalne dane, horyzont, universe, koszty i budżet strojenia;
- główną metrykę korzyści **netto po kosztach** i minimalną praktycznie istotną poprawę wobec baseline;
- sposób oszacowania niepewności (zależność obserwacji, nakładające się horyzonty) i z góry ustaloną regułę wystarczających dowodów;
- maksymalne dopuszczalne pogorszenie drawdownu i kalibracji oraz metody ich pomiaru;
- minimalną **efektywnie wystarczającą** próbę OOS i pokrycie reżimów rynku, z uwzględnieniem zależnych obserwacji.

Przed badaniem określić rodzinę hipotez i warianty uczestniczące w selekcji (definicje formacji, parametry, interwały, universe), sposób ewidencji wszystkich prób w **trial-budget registry przewidzianym w TIH-4** oraz uzasadnioną metodę kontroli multiple testing i zależności prób. Rejestrować również wyniki nieudane i odrzucone. **Nie wolno dobierać progów ACCEPT do obejrzanych wyników OOS**; każda późniejsza zmiana kryteriów wymaga nowej wersji protokołu i świeżej, odseparowanej oceny OOS. Roadmapa nie narzuca uniwersalnych wartości progów: trzeba je uzasadnić i ustalić dla konkretnego eksperymentu przed oceną.

- **ACCEPT:** kompletne, poprawne badanie spełnia wszystkie uprzednio ustalone kryteria korzyści, niepewności, stabilności i ryzyka; dopuszcza wyłącznie dalszą kwalifikację integracyjną, nie LIVE.
- **REJECT:** kompletne, poprawne badanie nie spełnia kryteriów; zamyka research bez dodawania nowych cech.
- **INCOMPLETE:** brak danych, wystarczającej próby, pokrycia reżimów lub wiarygodnych dowodów; nie pozwala zaakceptować cechy ani stwierdzić, że wykazano brak przewagi.

**Kryteria akceptacji TIH-2.C:** deterministyczne testy formacji (w tym brakujące dane), brak look-ahead/repaintingu, zgodność train/inference oraz odtwarzalny raport ablation z wersją protokołu i kryteriami ustalonymi przed badaniem. Raport dokumentuje porównanie z baseline, niepewność, ograniczenia ryzyka, ewidencję prób, kontrolę multiple testing i uzasadnienie **ACCEPT / REJECT / INCOMPLETE** na podstawie walk-forward/OOS. Udokumentowany REJECT zamyka badania bez nowych cech; INCOMPLETE nie przechodzi bramki. ACCEPT wymaga dalszej kwalifikacji integracyjnej przed wpływem na decyzje tradingowe; Stage 9/10 i LIVE pozostają bez zmian.

Warstwa powinna produkować m.in.:

```text
buy_pressure_score
sell_pressure_score
liquidity_score
reversal_probability_up
reversal_probability_down
expected_short_horizon_move
microstructure_confidence
```

## Ograniczenia

- brak traktowania pojedynczego order-book snapshotu jako sygnału o wysokiej wiarygodności;
- ochrona przed stale book, gaps, reconnect i crossed/inconsistent book;
- model musi uwzględniać specyfikę giełdy i pary;
- spoofing-sensitive features nie mogą być używane bez filtrów trwałości i wykonanych transakcji.

# TIH-3 — Execution Optimizer & TCA v2

## Cel

Zamienić sygnał tradingowy w decyzję **jak wykonać zlecenie**, a nie tylko czy je wykonać.

## Zakres

- wybór MARKET / LIMIT / POST_ONLY tam, gdzie giełda wspiera;
- cena limit na podstawie booka i urgency;
- maker/taker expected cost;
- expected fill probability;
- timeout i cancel/replace;
- partial-fill handling;
- stale-order protection;
- spread-aware entry/exit;
- liquidity-aware sizing;
- slippage estimator zależny od wielkości zlecenia i głębokości rynku;
- adverse-selection estimator;
- expected implementation shortfall;
- decyzja `SKIP`, jeżeli execution niszczy edge;
- kalibracja modelu kosztów na podstawie rzeczywistych filli;
- TCA per strategy / symbol / exchange / order type / regime.

Przykład decyzji:

```text
raw_edge_bps             +34
expected_fee_bps          -8
expected_slippage_bps     -5
adverse_selection_bps     -4
net_edge_bps             +17
recommended_order         POST_ONLY_LIMIT
fill_probability          0.71
max_wait_ms               4200
```

## Kryteria akceptacji

- decyzje execution są deterministycznie audytowane;
- model nie zakłada pełnego fillu bez podstawy;
- realized vs expected slippage jest monitorowany;
- system potrafi odrzucić opłacalny „na wykresie” trade, jeżeli po kosztach staje się nieopłacalny.

# TIH-4 — Anti-Overfitting & Research Validation Guard

## Cel

Uniemożliwić promocję strategii, której przewaga wynika z błędów badawczych, data snooping lub nadmiernego strojenia.

## Zakres obowiązkowy

- strict train/validation/OOS separation;
- walk-forward;
- purging / embargo tam, gdzie charakter targetu i overlap tego wymagają;
- point-in-time feature validation;
- dla TIH-2.C: ablation vs baseline według uprzednio zamrożonego protokołu, z trial-budget registry, kontrolą leakage / multiple testing i separacją OOS;
- leakage detection;
- survivorship-bias controls dla universe;
- parameter stability tests;
- perturbation / noise sensitivity;
- transaction-cost stress;
- latency stress;
- bootstrap / Monte Carlo sekwencji wyników;
- multiple-hypothesis / multiple-testing accounting;
- trial-budget registry dla Strategy Discovery;
- selection-bias controls;
- Deflated Sharpe Ratio lub równoważna korekta performance selection;
- Probability of Backtest Overfitting / CSCV lub równoważny test tam, gdzie zastosowalny;
- minimalna liczba transakcji i minimalny zakres reżimów;
- reprodukowalny promotion report.

## Zasada fail-closed

Brak wymaganych dowodów walidacyjnych oznacza `REJECT/INCOMPLETE`, a nie ostrzeżenie pozwalające przejść dalej.

# TIH-5 — Live Edge Decay Monitor

## Cel

Wykrywać utratę przewagi strategii wcześniej niż klasyczny limit drawdownu.

## Zakres

Monitorowanie rolling expected vs realized:

- expectancy;
- hit rate;
- payoff ratio;
- profit factor;
- Sharpe/Sortino proxy;
- implementation shortfall;
- fees/slippage;
- fill rate;
- adverse selection;
- regime-conditioned performance;
- calibration drift;
- signal frequency drift;
- latency drift.

Stany zdrowia:

```text
HEALTHY
WATCH
DEGRADED
SUSPENDED
RETIRED
```

Automatyczne reakcje mogą obejmować:

- obniżenie allocation;
- przejście do SHADOW/PAPER;
- wymuszenie rewalidacji;
- blokadę nowych pozycji;
- retirement strategii.

**Utrata edge'u nie może prowadzić do zwiększania ryzyka w celu „odrobienia”.**

# TIH-6 — Dynamic Capital Allocation v2

## Cel

Rozbudować istniejący PortfolioGovernor tak, aby kapitał był rozdzielany na podstawie jakości przewagi i ryzyka całego portfela, a nie izolowanej oceny strategii.

## Zakres

Allocation score powinien uwzględniać co najmniej:

```text
validated_edge
× confidence
× regime_fit
× liquidity
× live_health
× diversification_value
÷ volatility
÷ tail_risk
```

Dodatkowo:

- correlation clustering strategii;
- exposure overlap;
- factor concentration;
- per-exchange concentration;
- per-asset concentration;
- marginal contribution to risk;
- drawdown-aware de-risking;
- capacity constraints;
- turnover-aware allocation;
- strategy-level min/max allocation;
- cash reserve floor;
- niezależne limity canary.

## Zasada

Allocator może zmniejszyć lub wyzerować pozycję strategii mimo dodatniego standalone edge'u, jeśli pogarsza ona profil całego portfela.

# TIH-7 — L2 / Trades Market Replay Engine

## Status

**CONDITIONAL REQUIRED** — obowiązkowy dla strategii zależnych od mikrostruktury, krótkiego horyzontu lub execution path; nie musi blokować strategii wyłącznie OHLCV o długim horyzoncie.

## Cel

Zapewnić deterministyczne odtwarzanie rynku wystarczające do testowania order-book-sensitive i execution-sensitive strategii.

## Zakres

- normalized trades;
- L2 snapshots/deltas w zakresie obsługiwanym przez źródła danych;
- sequence/checksum integrity;
- spread/depth reconstruction;
- exchange timestamps + local receive timestamps;
- replay clock;
- deterministic event ordering;
- reconnect/gap representation;
- funding/OI event timeline, jeżeli wymagane;
- execution simulation oparta o dostępny book/liquidity;
- zapis provenance datasetu.

## Kryteria akceptacji

- ten sam dataset + ten sam StrategySpec → deterministyczny replay;
- brak „magicznego” fillu po cenie niewystępującej w odtworzonym rynku;
- jawne ograniczenia fidelity zależne od źródła danych.

# Integracja z istniejącymi komponentami

TIH ma wykorzystywać istniejące elementy, m.in.:

- `bot_core.market_intel` i order book feeds;
- `bot_core.market_intel.regime`;
- `bot_core.ai`;
- `DecisionOrchestrator`;
- `AutoTrader` / AI Governor;
- contextual bandits / strategy advisors;
- `bot_core.tco`;
- backtest / walk-forward;
- `PortfolioGovernor` i risk correlation/stress;
- `Risk Engine`;
- `ExecutionLease`;
- execution adapters;
- audit / decision journal / observability.

Nie należy tworzyć równoległego „drugiego AI Governora” ani osobnego portfolio stacku, jeżeli istniejący komponent można bezpiecznie rozszerzyć.

# Kolejność realizacji

Rekomendowana zależność:

```text
TIH-1 Regime Intelligence v2
        ↓
TIH-2 Microstructure & Reversal Intelligence
        ↓
TIH-3 Execution Optimizer & TCA v2
        ↓
TIH-4 Anti-Overfitting Guard
        ↓
TIH-5 Live Edge Decay Monitor
        ↓
TIH-6 Dynamic Capital Allocation v2
        ↓
TIH-7 L2 Replay (obowiązkowo tam, gdzie wymaga tego typ strategii)
```

TIH-4 może być rozwijany równolegle z TIH-1–TIH-3, ale musi być zamknięty przed produkcyjnym Strategy Discovery.

# Kryteria wejścia

- Stage 10 zamknięty i zaakceptowany;
- stabilny runtime i persistence;
- działający audit/observability;
- stabilne kontrakty risk i execution;
- wiarygodny paper/testnet pipeline;
- historyczne i live dane dostępne z jednoznaczną provenance.

# Definition of Done

Trading Intelligence Hardening jest zamknięty dopiero, gdy:

1. Regime Intelligence v2 działa probabilistycznie, multi-timeframe i przechodzi OOS;
2. microstructure/reversal layer produkuje audytowalne feature'y i confidence;
3. Execution Optimizer potrafi odmówić trade'u po uwzględnieniu kosztów i płynności;
4. TCA porównuje expected vs realized execution;
5. Anti-Overfitting Guard działa fail-closed i kontroluje multiple testing;
6. Live Edge Decay Monitor może automatycznie zdegradować strategię;
7. allocation uwzględnia portfolio-level correlation/concentration/capacity;
8. strategie mikrostrukturalne mają wymagany replay/fidelity evidence;
9. wszystkie warstwy zachowują nadrzędność Risk Engine i ExecutionLease;
10. istnieje komplet testów jednostkowych, integracyjnych, adversarial i E2E dla nowych gate'ów.

W TIH-2.C poprawnie udokumentowany **REJECT** zamyka zadanie badawcze bez wdrażania cech, natomiast **INCOMPLETE** nie spełnia kryteriów zamknięcia; **ACCEPT** nie upoważnia do LIVE.

# Pozycja w roadmapie

```text
Current foundation
    ↓
Stage 9 closure
    ↓
S9-S10-PRE-01 — dedykowany self-hosted Windows runner
+ przekazanie i weryfikacja qualified MSI z bieżącego runu
    ↓
Stage 10 closure / platform stabilization
    ↓
Stage 11 — Trading Intelligence Hardening (TIH-1–TIH-7)
    ↓
Stage 12 — Autonomous Strategy Discovery & Promotion (ASD-1–ASD-6)
    ↓
Shadow / Paper / Canary Live
    ↓
continuous self-improving strategy portfolio
```
