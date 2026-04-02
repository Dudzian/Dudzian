# Audit kodowy (hard) — AI Governor / AI orchestration

Data audytu: 2026-04-02
Zakres: `bot_core/auto_trader/ai_governor.py` + najbliższa orkiestracja (`bot_core/decision/*`, `bot_core/ai/*`) i testy.

## 1) Czy w AI Governor jest realny model ML/inference?

**W samym AI Governorze: NIE.**

`AutoTraderAIGovernor` jest implementacją heurystyczną (if/else) opartą o:
- mapowanie reżimu rynku -> tryb (`scalping/grid/hedge`),
- progi kosztu transakcyjnego,
- sufit ryzyka (`risk_ceiling`),
- flagi guardrail/cooldown,
- prostą formułę confidence.

Nie ma tu ładowania modelu, inferencji, featuryzacji ani treningu.

## 2) Czy jest tylko rule/scoring/policy logic?

**W AI Governorze: tak, rule/scoring/policy.**

- Decyzje trybu to deterministyczne reguły.
- `confidence` to ręcznie zdefiniowana funkcja skoringowa, nie wynik modelu statystycznego.
- Runner konsumuje snapshoty wydajności strategii i telemetrię; nie uruchamia inferencji ML.

Jednocześnie **w sąsiedniej warstwie Decision Engine istnieje inference modelowe** (`DecisionModelInference`) podpinane do `DecisionOrchestrator`.

## 3) Czy są twarde dowody model lifecycle / evaluation hooks / out-of-sample workflow?

**Tak — ale w warstwie `bot_core.ai` / `Decision Engine`, nie w AI Governorze.**

W kodzie są:
- repozytorium wersji modeli + aliasy + aktywna wersja,
- trening modelu (wbudowany GBM + adaptery zewnętrzne),
- walidacje/cross-validation,
- raporty jakości i decyzje champion/degraded,
- monitoring driftu i quality gating na inferencji,
- hooki retrainingu i harmonogramowania.

To są realne elementy lifecycle, choć testy jednostkowe często stubują backend modeli.

## 4) Nazwy potencjalnie nadmuchane vs implementacja

1. **"AI Governor"** — w bieżącym kodzie to głównie silnik reguł/polityk przełączania trybów, nie governor oparty o ML policy learning.
2. **"Zaawansowany pipeline AI" (README)** — częściowo uzasadnione dla `bot_core.ai` (trening/walidacja/repozytorium), ale **nie** dla samego modułu AI Governor.
3. **"Adaptacyjne sterowanie" w dokumentacji orchestratora** — adaptacja jest, ale oparta o polityki/bandytę + progi; nie jest to pełny end-to-end system continuous online learning.

## Wniosek operacyjny

- Jeśli celem jest precyzyjny przekaz: warto rozdzielić nazewnictwo na
  - `AI Governor` (heurystyczny policy switcher),
  - `Decision Engine ML` (model inference + lifecycle).
- Decyzja „rebrand vs formalizacja”:
  - **rebrand** sensowny dla modułu Governor,
  - **formalizacja** sensowna dla warstwy modelowej (`bot_core.ai`).
