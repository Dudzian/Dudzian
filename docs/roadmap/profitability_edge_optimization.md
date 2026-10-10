# Roadmap — Profitability & Edge Optimization

## Status

**PLANNED — required post-ASD capability block**

Ten blok jest planowany po zamknięciu **Trading Intelligence Hardening (TIH)** oraz wdrożeniu podstawowego **Autonomous Strategy Discovery & Promotion (ASD)**. Nie zmienia zamrożonych kontraktów M0 ani numeracji Windows Stage 0–14.

Celem bloku nie jest zwiększanie liczby transakcji ani agresywności. Celem jest zwiększanie **netto expectancy na jednostkę ryzyka** przez lepszą filtrację okazji, kalibrację prawdopodobieństw, analizę zależności między rynkami, attribution i bezpieczne uczenie online.

## Główna zasada

```text
nie:
więcej danych → więcej modeli → więcej transakcji

tylko:
lepsze dane
    ↓
lepsza ocena prawdopodobieństwa
    ↓
lepszy SKIP
    ↓
lepszy sizing
    ↓
lepsze execution
    ↓
więcej NETTO na jednostkę ryzyka
```

`SKIP/ABSTAIN` jest pełnoprawną i potencjalnie najbardziej wartościową decyzją systemu.

# PEO-1 — Meta-Labeling & Opportunity Filter

## Cel

Oddzielić generowanie sygnału od decyzji, czy konkretny sygnał ma wystarczającą przewagę po kosztach, aby w ogóle wejść w pozycję.

## Zakres

- meta-label dla każdego sygnału wejściowego;
- `P(profitable_after_costs)`;
- expected net edge;
- expected adverse move / drawdown;
- confidence i uncertainty;
- regime fit;
- liquidity/execution feasibility;
- decyzja `TAKE / REDUCE / SKIP`;
- sizing zależny od jakości opportunity;
- walidacja OOS i kalibracja meta-modelu;
- attribution skuteczności filtra względem bazowej strategii.

## Zasada

Meta-model nie może „naprawiać” słabej strategii przez zwiększanie ryzyka. Jego podstawową funkcją jest filtrowanie i skalowanie.

# PEO-2 — Probability Calibration & Uncertainty Engine

## Cel

Sprawić, aby wartości confidence/probability miały wiarygodne znaczenie statystyczne i mogły być bezpiecznie używane do filtracji oraz sizingu.

## Zakres

- calibration curves;
- Brier score i log loss tam, gdzie adekwatne;
- isotonic / Platt / inne metody kalibracji dobierane walidacyjnie;
- uncertainty intervals;
- conformal prediction tam, gdzie ma zastosowanie;
- calibration per regime / symbol / horizon;
- drift kalibracji LIVE;
- fail-safe przy braku kalibracji lub rozszerzeniu uncertainty.

Przykład:

```text
raw confidence           0.91
calibrated probability   0.68
uncertainty               HIGH
result                    SKIP
```

# PEO-3 — Cross-Asset & Lead/Lag Intelligence

## Cel

Wykrywać zależności czasowe i informacyjne między aktywami, rynkami spot/perpetual/futures oraz szerokim rynkiem crypto.

## Zakres

- lead/lag discovery z kontrolą data snooping;
- rolling cross-correlation;
- partial correlation / conditional relationships;
- spot ↔ perpetual / futures basis relationships;
- BTC ↔ ETH ↔ alt basket relationships;
- market breadth;
- dominance / relative-strength proxies;
- stablecoin-flow proxies, jeżeli dane są wiarygodne;
- regime-conditioned dependency graph;
- decay monitor wykrytych relacji;
- zakaz użycia relacji bez stabilności OOS.

# PEO-4 — Smart Venue Selection & Best Execution Routing

## Cel

Jeżeli dostępnych jest kilka giełd, wybierać miejsce wykonania na podstawie oczekiwanego wyniku netto, a nie statycznego przypisania do venue.

## Zakres

- spread;
- maker/taker fee;
- expected slippage;
- depth/liquidity;
- latency;
- fill probability;
- venue health;
- rate limits;
- transfer/capital availability;
- failure/failover risk;
- expected implementation shortfall;
- best-execution audit trail.

Best Execution Router ma rozszerzać istniejące exchange abstractions i Execution Optimizer, a nie tworzyć osobny execution stack.

# PEO-5 — On-Chain Intelligence

## Status

**OPTIONAL / DATA-DEPENDENT** — wdrażany tylko tam, gdzie historyczne i live dane mają wystarczającą jakość, point-in-time provenance oraz wykazany OOS edge.

## Zakres potencjalny

- exchange inflow/outflow;
- whale / large-holder flows;
- stablecoin issuance / flows;
- realized-cap-derived metrics;
- SOPR/MVRV-like metrics;
- miner flows;
- long-term holder behavior;
- network activity/liquidity proxies.

On-chain jest kontekstem i źródłem feature'ów, nie prostym generatorem `BUY/SELL`.

# PEO-6 — Event Intelligence

## Cel

Rozszerzyć istniejący sentiment/news stack z prostego biasu na ustrukturyzowane rozumienie wydarzeń i ich potencjalnego wpływu na zmienność, aktywa oraz horyzont.

## Zakres

- event classification;
- affected assets;
- surprise magnitude;
- expected volatility horizon;
- novelty / duplicate detection;
- source quality / confidence;
- macro events (np. CPI/rates);
- regulatory events;
- exchange incidents;
- hacks/exploits;
- listings/delistings;
- ETF / institutional events;
- event-driven regime override z twardymi limitami ryzyka;
- event decay i post-event validation.

LLM może interpretować wydarzenie, ale decyzje tradingowe pozostają pod kontrolą istniejących gate'ów i Risk Engine.

# PEO-7 — Profit Attribution Engine

## Cel

Wyjaśniać, skąd faktycznie pochodzi PnL i gdzie tracony jest edge.

## Attribution co najmniej per:

- strategy;
- signal family;
- regime;
- symbol;
- exchange;
- horizon;
- model/version;
- execution type;
- fees;
- slippage;
- adverse selection;
- missed opportunity / SKIP quality, gdzie możliwe;
- portfolio interaction / diversification.

Przykład:

```text
total PnL              +8.2%
momentum               +5.1%
mean reversion          +2.7%
microstructure          +1.4%
fees                    -0.6%
slippage                -0.5%
regime mismatch         -1.1%
```

Attribution ma być wejściem dla Strategy Discovery, Edge Decay Monitor i Capital Allocation, ale nie może samodzielnie omijać promotion gates.

# PEO-8 — Adaptive Trading Horizon

## Cel

Pozwolić systemowi oceniać nie tylko kierunek, ale także horyzont, na którym istnieje przewaga.

## Zakres

- multi-horizon predictions;
- horizon-specific edge;
- horizon-specific transaction-cost model;
- horizon-specific confidence/uncertainty;
- expected holding time;
- time-stop optimization z walidacją OOS;
- dynamic exit horizon;
- porównanie edge po kosztach między horyzontami;
- możliwość `SKIP`, jeżeli krótkoterminowa przewaga zostaje zjedzona przez koszty.

# PEO-9 — Capacity & Market Impact Model

## Status

**SCALE-DEPENDENT** — może pozostać uśpiony przy małym kapitale, ale kontrakt powinien istnieć przed istotnym skalowaniem AUM.

## Zakres

- strategy capacity estimate;
- participation-rate limits;
- order-book depth consumption;
- expected market impact;
- nonlinear slippage vs notional;
- liquidity regime sensitivity;
- per-symbol capacity;
- per-venue capacity;
- portfolio crowding / overlap;
- allocation cap wynikający z capacity.

System nie może zakładać, że stopa zwrotu skaluje się liniowo z kapitałem.

# PEO-10 — Safe Online Learning

## Cel

Uczyć się z nowych danych LIVE bez pozwalania aktywnemu modelowi na samowolną zmianę polityki produkcyjnej.

## Dozwolony przepływ

```text
LIVE observations
    ↓
new candidate model / parameters
    ↓
validation
    ↓
challenger
    ↓
shadow / paper
    ↓
canary
    ↓
promotion
```

## Niedozwolony przepływ

```text
LIVE loss
    ↓
model rewrites itself
    ↓
next LIVE trade with unvalidated policy
```

## Wymagania

- immutable model/version lineage;
- no self-promotion;
- bounded update cadence;
- replayable training datasets;
- concept-drift triggers;
- rollback;
- promotion przez istniejący ASD lifecycle;
- Risk Engine / ExecutionLease zawsze nadrzędne.

# Priorytety realizacji

Pierwsza fala:

1. **PEO-1 Meta-Labeling & Opportunity Filter**
2. **PEO-2 Probability Calibration & Uncertainty**
3. **PEO-3 Cross-Asset & Lead/Lag Intelligence**
4. **PEO-7 Profit Attribution Engine**

Druga fala:

5. PEO-4 Smart Venue Selection
6. PEO-6 Event Intelligence
7. PEO-8 Adaptive Trading Horizon
8. PEO-10 Safe Online Learning

Warunkowo / wraz ze skalą i jakością danych:

9. PEO-5 On-Chain Intelligence
10. PEO-9 Capacity & Market Impact

# Kryteria wejścia

- TIH zamknięty;
- ASD core lifecycle działa end-to-end;
- istnieje stabilny Shadow/Paper/Canary pipeline;
- TCA i Live Edge Decay są wiarygodne;
- attribution danych i wersji modeli jest możliwe;
- brak krytycznych blockerów bezpieczeństwa i execution.

# Definition of Done

Blok jest zamknięty, gdy:

1. meta-labeling potrafi mierzalnie filtrować opportunity OOS bez wzrostu tail risk;
2. prawdopodobieństwa są kalibrowane i monitorowane LIVE;
3. cross-asset relacje mają stability/decay controls;
4. multi-venue execution potrafi wybierać venue na podstawie expected net execution, jeśli używamy wielu giełd;
5. attribution wyjaśnia PnL i koszty co najmniej per strategy/regime/symbol/execution;
6. adaptive horizon przechodzi OOS i cost-aware validation;
7. online learning nie posiada bezpośredniej ścieżki do LIVE;
8. on-chain/event features nie mogą wejść do produkcji bez point-in-time provenance i OOS evidence;
9. capacity model ogranicza skalowanie, gdy kapitał zaczyna wpływać na execution;
10. wszystkie nowe warstwy respektują `ABSTAIN/SKIP`, Risk Engine, kill switch i ExecutionLease.

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
Trading Intelligence Hardening
    ↓
Autonomous Strategy Discovery & Promotion
    ↓
Shadow / Paper / Canary Live
    ↓
Profitability & Edge Optimization
    ↓
mature continuous optimization
```

# Roadmap scope freeze

Po dodaniu tego bloku **nie planujemy kolejnych rozszerzeń funkcjonalnych roadmapy**, dopóki obecnie zapisane bloki nie zostaną zrealizowane i zweryfikowane w praktyce.

Wyjątki od scope freeze:

- krytyczna luka bezpieczeństwa;
- blocker uniemożliwiający realizację istniejącego etapu;
- wymaganie giełdy/API/regulacyjne, bez którego produkt przestaje działać;
- wynik testów LIVE/PAPER wskazujący, że założenie roadmapy jest błędne i wymaga korekty.

Nowe „ciekawe funkcje” bez takiego uzasadnienia trafiają do backlogu pomysłów, ale nie rozszerzają aktywnej roadmapy.
