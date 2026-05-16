# Stage 8C: Canonical DecisionEnvelope — feasibility audit (spec-only)

## Cel i zakres etapu

Ten etap **nie wprowadza zmian runtime**. Celem jest ocena wykonalności i opis kontraktu docelowego `DecisionEnvelope` przy zachowaniu obecnego, rozproszonego kontraktu decyzji.

## Stan aktualny (rozproszony kontrakt)

Kontrakt decyzji jest dziś rozdzielony pomiędzy:

- `OpportunityDecision` (wynik inferencji/rankingu okazji),
- metadane i walidacje autonomy/handoff w `TradingController`,
- payload `performance_guard` (local guard),
- metadane `opportunity_shadow_record_key` / `opportunity_decision_timestamp`,
- provenance final-label/journal (environment/portfolio/lineage).

Repo ma już `bot_core/runtime/contracts.py`, ale moduł zawiera obecnie wyłącznie protokoły `Scheduler` i `PipelineBuilder`, więc **nie jest naturalnym miejscem na runtime-wymuszoną klasę envelope w tym etapie**.

## Canonical DecisionEnvelope (docelowy kontrakt)

Poniższa lista to minimalny zestaw pól do przyszłej konsolidacji:

- `decision_id` (lub `correlation_key`),
- `action` / `intent`,
- `symbol`,
- `side`,
- `quantity` (lub sizing input),
- `decision_source`,
- `effective_mode`,
- `model_version`,
- `inference_model`,
- `inference_model_version`,
- `confidence`,
- `score`,
- `rank`,
- `opportunity_shadow_record_key`,
- `opportunity_decision_timestamp`,
- `performance_guard` (payload),
- `risk_result` / `risk_budget`,
- `blocking_reason` / `blocking_reasons`,
- `environment_scope`,
- `portfolio_scope`,
- `provenance`.

## Mapping obecnych źródeł pól (audyt)

| Pole canonical | Aktualne źródło (obecnie) | Status 8C |
|---|---|---|
| `decision_id` / `correlation_key` | metadata `opportunity_shadow_record_key` + handoff tracker | istnieje, rozproszone |
| `action` / `intent` | żądania open/close + payload mode w controllerze | istnieje, rozproszone |
| `symbol`, `side`, `quantity` | signal/request/journal event | istnieje, testowane |
| `decision_source`, `model_version`, `confidence`, `rank`, `provenance` | `OpportunityDecision` + lineage/provenance final labels | istnieje, testowane/rozproszone |
| `effective_mode`, `inference_model*`, `blocking_reasons` | autonomy payload chain w controllerze | istnieje, rozproszone |
| `opportunity_decision_timestamp` | metadata handoff/replay | istnieje, luźne metadata |
| `performance_guard` | payload `performance_guard` + local diagnostics | istnieje, rozproszone |
| `risk_result` / `risk_budget` | risk engine/journal (`risk_budget_bucket`) | częściowo, rozproszone |
| `environment_scope`, `portfolio_scope` | provenance (`environment`, `portfolio`, `portfolio_id`) | istnieje, testowane |
| `score` | pola rankingowo-scoringowe w różnych komponentach | istnieje, rozproszone |

## Rekomendacja lokalizacji canonical envelope

- **Teraz (8C): docs-only spec** (ten dokument).
- **Przyszłość:** po etapie hardeningu można rozważyć dedykowany moduł typu `bot_core/runtime/decision_contract.py` lub `bot_core/runtime/types.py` z adapterami z istniejących payloadów.
- **Nie rekomenduje się** dodawania martwej klasy produkcyjnej w 8C, bo aktualny etap nie obejmuje migracji runtime ani mapowania event payloadów.

## Non-goals (8C)

- Brak zmiany flow decyzji.
- Brak zmiany `TradingController` behavior.
- Brak zmiany handoff/replay/failover.
- Brak zmiany reason mapping.
- Brak zmiany event payloadów.

## Ryzyka przyszłej migracji

1. **Reason mapping drift** między upstream/local guard/final provenance.
2. **Replay/final-label lineage regression** (utrata priorytetu źródła open lineage).
3. **Performance guard contract drift** (payload vs local diagnostics).
4. **CI matrix drift** — konieczność utrzymania zgodności z autonomy matrix i istniejącymi selectorami testów.

## Następny etap (proponowany)

1. Dodać test kontraktowy docs/spec (test-only), który pilnuje obecności pól minimalnych w tej specyfikacji.
2. Zdefiniować adaptery mapujące obecne źródła do envelope bez zmiany runtime flow.
3. Dopiero potem rozważyć produkcyjną klasę canonical envelope.
