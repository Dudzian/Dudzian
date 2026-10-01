# Roadmap — Autonomous Strategy Discovery & Promotion

## Status

**PLANNED — post-Trading-Intelligence-Hardening capability block**

Ten blok jest planowany po zamknięciu aktualnego fundamentu platformy oraz bloku **Trading Intelligence Hardening (TIH)** i **nie zmienia zamrożonych kontraktów architektury M0 ani numeracji Windows Stage 0–14**. Implementacja produkcyjna może rozpocząć się dopiero po zamknięciu Stage 10, zamknięciu TIH i potwierdzeniu stabilności runtime'u, updatera, lifecycle'u, persistence, risk i execution.

Bezpośredni prerequisite:
[`docs/roadmap/trading_intelligence_hardening.md`](trading_intelligence_hardening.md).

## Cel nadrzędny

Rozszerzyć CryptoHuntera z systemu, który wybiera i dostraja istniejące strategie, do systemu, który potrafi **samodzielnie tworzyć kandydatów strategii, walidować ich przewagę, testować ich odporność i bezpiecznie promować najlepsze warianty do kolejnych środowisk**, bez możliwości samowolnego pominięcia kontroli ryzyka.

Docelowy przepływ:

```text
AI/ML tworzy hipotezę strategii
        ↓
StrategySpec / Candidate
        ↓
Backtest
        ↓
Walk-forward
        ↓
Strict OOS
        ↓
Robustness / perturbation / cost stress
        ↓
Shadow
        ↓
Paper
        ↓
Canary Live
        ↓
Champion / Challenger / Retired
```

## Zasady bezpieczeństwa

1. **LLM/AI może generować hipotezy, ale nie może samodzielnie zatwierdzić własnej hipotezy.**
2. Żaden kandydat nie może trafić bezpośrednio do LIVE.
3. Każda promocja musi przechodzić deterministyczne gate'y jakości, ryzyka i kosztów.
4. Risk Engine, ExecutionLease, kill switch, limity ekspozycji i polityki środowiska pozostają niezależne od Strategy Discovery.
5. Strategia musi być opisana deklaratywnie przez wersjonowany `StrategySpec`; generowany arbitralny kod nie jest dopuszczony do bezpośredniego runtime'u produkcyjnego.
6. Walidacja musi uwzględniać maker/taker fees, spread, slippage, latency oraz ograniczenia płynności.
7. Każdy kandydat, test, awans, degradacja i odrzucenie muszą być audytowalne i reprodukowalne.
8. Brak wystarczającego edge'u oznacza `ABSTAIN/REJECT`, a nie wymuszone znalezienie strategii.
9. ASD korzysta z TIH Regime v2, Microstructure/Reversal Intelligence, Execution Optimizer, Anti-Overfitting Guard, Live Edge Decay Monitor i Dynamic Capital Allocation zamiast tworzyć ich równoległe odpowiedniki.

## Zakres funkcjonalny

### 1. StrategySpec i bezpieczna przestrzeń poszukiwań

- wersjonowany `StrategySpec` opisujący wejścia, wyjścia, filtry, horyzont, wymagane cechy i parametry;
- jawny katalog dozwolonych cech i operatorów;
- walidacja typów, zakresów i zależności parametrów;
- limity złożoności strategii i liczby stopni swobody;
- semantic fingerprint i pełna provenance kandydata;
- zakaz bezpośredniego wykonywania dowolnego kodu wygenerowanego przez model.

### 2. StrategyCandidateGenerator

- generowanie nowych hipotez przez AI/ML;
- mutacja istniejących strategii i ich parametrów;
- kompozycja zatwierdzonych sygnałów;
- generowanie strategii zależnych od reżimu rynku;
- eksploracja kombinacji price action, wolumenu, order book, volatility, funding, open interest, korelacji, sentymentu i innych zatwierdzonych źródeł;
- kontrola duplikatów i podobieństwa do istniejących strategii.

### 3. Research & validation pipeline

Każdy kandydat przechodzi co najmniej:

- backtest z realistycznymi kosztami;
- walk-forward validation;
- strict out-of-sample validation;
- test stabilności parametrów;
- perturbation / noise tests;
- Monte Carlo / bootstrap sekwencji transakcji;
- stress kosztów, spreadu, slippage i latency;
- testy na różnych reżimach rynku;
- kontrolę look-ahead bias, leakage i survivorship bias;
- kontrolę liczby prób / data snooping oraz korektę selection bias;
- test minimalnej liczby transakcji i czasu obserwacji;
- obowiązkowe przejście przez TIH Anti-Overfitting & Research Validation Guard.

### 4. Objective scoring

Promocja nie może bazować na opinii LLM. Kandydaci są oceniani na podstawie jawnych metryk, m.in.:

- expectancy netto po kosztach;
- profit factor;
- Sharpe / Sortino;
- max drawdown;
- tail loss / expected shortfall;
- win rate wraz z payoff ratio;
- stabilność między oknami walk-forward;
- stabilność OOS;
- turnover i koszt wykonania;
- capacity / liquidity limits;
- korelacja z aktywnym portfelem strategii;
- odporność na zmianę parametrów i pogorszenie kosztów.

Nie istnieje pojedyncza metryka typu „najwyższy PnL = wygrywa”.

### 5. Shadow → Paper → Canary Live

Promocja środowiskowa:

```text
RESEARCH
  ↓
VALIDATED_CANDIDATE
  ↓
SHADOW
  ↓
PAPER
  ↓
CANARY_LIVE
  ↓
CHALLENGER
  ↓
CHAMPION
```

Każde przejście wymaga osobnego gate'u. Kandydat może zostać cofnięty do wcześniejszego stanu lub odrzucony.

Canary Live musi posiadać:

- twardy limit kapitału;
- limit ryzyka per trade i per day;
- limit maksymalnego drawdown;
- limit liczby równoległych pozycji;
- niezależny kill switch;
- automatyczne zatrzymanie przy rozjeździe LIVE vs expected/paper;
- minimalny okres i minimalną liczbę obserwacji przed promocją;
- aktywny TIH Live Edge Decay Monitor;
- execution przez TIH Execution Optimizer/TCA v2, jeżeli strategia wymaga aktywnego execution optimization.

### 6. Champion / Challenger lifecycle

System utrzymuje wersjonowany rejestr strategii:

- `CANDIDATE`
- `VALIDATED`
- `SHADOW`
- `PAPER`
- `CANARY`
- `CHALLENGER`
- `CHAMPION`
- `DEGRADED`
- `RETIRED`
- `REJECTED`

Aktywny Champion może zostać zdegradowany po wykryciu dryfu, utracie edge'u, wzroście kosztów, przekroczeniu drawdownu lub pojawieniu się lepszego Challengera.

### 7. Continuous discovery

Po uruchomieniu blok może cyklicznie:

- szukać nowych kandydatów;
- retrenować modele;
- ponownie walidować aktywne strategie;
- wykrywać utratę edge'u;
- uruchamiać challenger tests;
- dostosowywać repertuar strategii do nowych reżimów rynku;
- wycofywać strategie, które przestały spełniać gate'y.

Ciągłe discovery nie oznacza ciągłego wdrażania do LIVE. Wdrożenie pozostaje kontrolowanym procesem promocji.

## Integracja z istniejącym CryptoHunterem

Blok ma wykorzystać istniejące komponenty zamiast je duplikować:

- `bot_core.strategies` / Strategy Registry;
- `bot_core.backtest` i walk-forward;
- `bot_core.ai` / retraining / model artifacts;
- `bot_core.market_intel.regime` oraz TIH Regime Intelligence v2;
- TIH Market Microstructure & Reversal Intelligence;
- `DecisionOrchestrator`;
- `AutoTrader` / AI Governor;
- contextual bandits / strategy advisors;
- `bot_core.risk`;
- `ExecutionLease` i execution layer;
- TIH Execution Optimizer / TCA v2;
- TIH Anti-Overfitting Guard;
- TIH Live Edge Decay Monitor;
- TIH Dynamic Capital Allocation v2;
- paper trading;
- decision journal / audit artifacts;
- monitoring driftu i jakości danych.

## Fazy realizacji

### ASD-1 — Candidate Contract

- `StrategySpec`;
- feature/operator registry;
- canonical serialization i fingerprint;
- lifecycle states;
- provenance i audit schema.

### ASD-2 — Candidate Generation

- `StrategyCandidateGenerator`;
- mutacje i kompozycje strategii;
- generation budget;
- similarity/duplicate detection;
- sandbox research environment.

### ASD-3 — Validation Factory

- automatyczny backtest;
- WFO/OOS;
- robustness, Monte Carlo i cost stress;
- TIH Anti-Overfitting Guard;
- deterministic promotion report.

### ASD-4 — Shadow & Paper Promotion

- shadow runtime;
- paper promotion gate;
- monitoring live market drift;
- porównanie expected vs realized behavior.

### ASD-5 — Canary Live

- mikroalokacja;
- niezależne limity canary;
- rollback/degrade;
- parity PAPER/LIVE;
- incident evidence;
- TIH Edge Decay Monitor;
- TIH Dynamic Capital Allocation v2.

### ASD-6 — Champion/Challenger Automation

- ranking kandydatów według jawnych metryk;
- portfolio-level strategy selection;
- automatyczna degradacja i retirement;
- okresowa rewalidacja Championów;
- pełny continuous discovery loop.

## Kryteria wejścia

Blok nie powinien wejść do implementacji produkcyjnej przed spełnieniem wszystkich warunków:

- Stage 10 zamknięty i zaakceptowany;
- **Trading Intelligence Hardening zamknięty i zaakceptowany**;
- TIH Regime Intelligence v2 gotowy;
- TIH Microstructure/Reversal Intelligence gotowy w zakresie wymaganym przez generowane strategie;
- TIH Execution Optimizer/TCA v2 gotowy;
- TIH Anti-Overfitting & Research Validation Guard działa fail-closed;
- TIH Live Edge Decay Monitor gotowy;
- TIH Dynamic Capital Allocation v2 gotowy;
- L2/Trades Replay gotowy dla klas strategii, które od mikrostruktury zależą;
- stabilne PAPER/TESTNET/LIVE execution contracts;
- działający i zweryfikowany Risk Engine + kill switch + ExecutionLease;
- stabilny persistence/audit trail;
- wiarygodny model kosztów wykonania;
- działający walk-forward/OOS pipeline;
- zamknięty monitoring data quality i drift;
- możliwość bezpiecznego ograniczenia kapitału dla canary.

## Kryteria wyjścia / Definition of Done

Blok jest zakończony dopiero, gdy:

1. system sam tworzy nowe `StrategySpec` bez arbitralnego kodu produkcyjnego;
2. każdy kandydat przechodzi automatyczny i reprodukowalny pipeline WFO/OOS/robustness/cost stress;
3. system potrafi odrzucić wszystkie kandydaty, gdy żaden nie ma wystarczającej przewagi;
4. shadow i paper są obowiązkowe przed canary;
5. canary nie może przekroczyć niezależnych limitów kapitału i ryzyka;
6. Strategy Registry zachowuje pełną historię wersji, testów, promocji i degradacji;
7. Champion/Challenger działa bez możliwości ominięcia Risk Engine;
8. utrata edge'u prowadzi do automatycznego `DEGRADED/RETIRED`, a nie do zwiększania ryzyka;
9. wszystkie decyzje są audytowalne i odtwarzalne;
10. testy adversarial potwierdzają brak bezpośredniej ścieżki `AI-generated candidate → LIVE`.

## Pozycja w roadmapie

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

To jest blok rozwojowy CryptoHuntera, a nie rozszerzenie Windows Stage 0–14. Nazwa i numeracja techniczna mogą zostać przypisane dopiero przy otwarciu implementacji, żeby nie naruszyć istniejących kontraktów etapów.
