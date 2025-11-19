# Stage 5 – Discovery i analiza zakresu

## 1. Cele discovery
- Zidentyfikować wszystkie komponenty kosztowe (prowizje, spready, funding, koszty transferów) w kontekście strategii Stage4.
- Określić KPI oraz SLO dla Etapu 5: `avg_cost_per_trade`, `slippage_bps`, `decision_latency_ms`, `compliance_audit_score`.
- Potwierdzić dostępność danych źródłowych (adaptery giełdowe, raporty PnL, logi execution) oraz wymagane rozszerzenia manifestów.
- Zweryfikować wymagania compliance i audytu: częstotliwość raportów, formaty PDF/CSV, podpisy HMAC, retention offline.

## 2. Pytania badawcze
| Obszar | Pytanie | Artefakt docelowy |
| --- | --- | --- |
| Dane | Czy każdy adapter giełdowy udostępnia historyczne prowizje i stawki funding? | Uaktualnione manifesty danych + raport braków |
| Ryzyko | Jakie limity kosztowe należy wprowadzić do `ThresholdRiskEngine` dla profili konserwatywny/zbalansowany/agresywny/manualny? | Propozycja progów w `config/core.yaml` |
| Egzekucja | Jak DecisionOrchestrator będzie wpływał na kolejkę zleceń i fallbacki? | Schemat przepływu + test plan regresji |
| Observability | Jakie źródła zasilą dashboard Stage5 i alerty SLO? | Lista metryk + reguły Prometheus |
| Compliance | Jak często audytorzy oczekują raportów i w jakim formacie? | Runbook audytu + plan archiwizacji |

## 3. Dane wejściowe do Etapu 5
- Lista datasetów backtestowych wymagających rozszerzenia o kolumny kosztowe (`fees`, `slippage`, `liquidity_gap`).
- Specyfikacja integracji z `run_risk_simulation_lab.py`, aby raportować koszty przy scenariuszach stress testowych.
- Wymagania dotyczące podpisów raportów (algorytm HMAC-SHA384, identyfikatory kluczy, rotacja).
- Aktualizacja decision logu: nowe pola `tco_kpi`, `decision_path`, `rotation_event_id`.

## 4. Artefakty discovery
- `docs/architecture/stage5_spec.md` – specyfikacja zakresu.
- `docs/runbooks/STAGE5_COMPLIANCE_CHECKLIST.md` – checklista wymagań audytowych.
- Raport luk danych (`audit/tco/data_gaps_<date>.json`).
- Notatki warsztatów z operatorami (L1/L2) i analitykami kosztów (`docs/training/stage5_workshop.md`).

## 5. Plan działania discovery
1. Warsztat z zespołem danych i ryzyka (1 dzień) – weryfikacja manifestów, braków i KPI.
2. Sesja z zespołem compliance (0,5 dnia) – wymagania raportowe, SLA, retention offline.
3. Prototyp raportu TCO (1 dzień) – generacja z istniejących logów paper tradingu.
4. Aktualizacja specyfikacji i runbooków (0,5 dnia) – włączenie ustaleń discovery.
5. Review architektoniczne (0,5 dnia) – potwierdzenie zakresu i kryteriów akceptacji Etapu 5.
