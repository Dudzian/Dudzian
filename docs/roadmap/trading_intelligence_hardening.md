# Roadmap — Trading Intelligence Hardening

## Status

**PLANNED — required capability block before Autonomous Strategy Discovery & Promotion**

Ten blok jest planowany **po zamknięciu Stage 10 i przed rozpoczęciem produkcyjnego Autonomous Strategy Discovery & Promotion (ASD)**. Nie zmienia zamrożonych kontraktów M0 ani numeracji Windows Stage 0–14. Jego zadaniem jest wzmocnienie jakości danych, rozpoznawania rynku, walidacji, execution i alokacji kapitału tak, aby późniejszy Strategy Discovery nie optymalizował strategii na słabym modelu rynku lub nierealistycznym execution.

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

# Pozycja w roadmapie

```text
Current foundation
    ↓
Stage 9 closure
    ↓
Stage 10 closure / platform stabilization
    ↓
Trading Intelligence Hardening
    ↓
Autonomous Strategy Discovery & Promotion
    ↓
Shadow / Paper / Canary Live
    ↓
continuous self-improving strategy portfolio
```
