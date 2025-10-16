# Stage6 – Specyfikacja architektoniczna

## Cel etapu
Stage6 rozszerza bota Dudzian o orkiestrację portfela klasy CryptoHopper, lokalną
inteligencję rynkową oraz procedury odporności i obserwowalności umożliwiające
prowadzenie operacji w trybie hypercare. Wszystkie moduły działają lokalnie i
komunikują się poprzez istniejące interfejsy bot_core, z naciskiem na minimalną
złożoność infrastrukturalną.

## Zakres funkcjonalny
1. **PortfolioGovernor** – adaptacyjne zarządzanie alokacją, egzekucja tolerancji
   dryfu, budżetów ryzyka i override z SLO/Stress Lab. Wymaga konfiguracji w
   `config.core.portfolio_governor` oraz integracji z schedulerem runtime.
   Implementacja łączy rekomendacje Stress Lab z decyzją oraz logiem HMAC.
   Automatyczny cykl hypercare (`bot_core.portfolio.hypercare`,
   `scripts/run_stage6_portfolio_cycle.py`) generuje podpisane podsumowania
   JSON/CSV oraz wpisy decision logu na bazie raportów Market Intel, SLO i
   Stress Lab.
2. **Market Intelligence Aggregator** – konsolidacja metryk zmienności,
   płynności, momentum i spreadów z `data` i cache OHLCV. Udostępnia snapshoty
   dla PortfolioGovernora i raporty z CLI `scripts/build_market_intel_metrics.py`.
3. **Stress Lab** – analiza raportów Paper Labs, generowanie adaptacyjnych
   override i rekomendacji dla PortfolioGovernora z uwzględnieniem
   scenariuszy płynności i latencji egzekucji. Raporty JSON/CSV podpisywane HMAC
   poprzez `scripts/run_stress_lab.py`. Kalibracja progów odbywa się narzędziem
   `scripts/calibrate_stress_lab_thresholds.py`, które łączy Market Intel i
   raporty symulacyjne; narzędzie potrafi automatycznie budować segmenty
   wolumenowe na bazie metryk płynności i przypisywać im prefiksy budżetów
   ryzyka.
4. **Resilience Stage6** – bundlowanie artefaktów, audyt, polityki, failover
   drill oraz raporty podpisywane HMAC (`export_resilience_bundle.py`,
   `verify_resilience_bundle.py`, `audit_resilience_bundles.py`,
   `failover_drill.py`) wraz z automatycznym self-healingiem restartującym
   moduły runtime (`bot_core/resilience/self_healing.py`). Komponent
   `ResilienceHypercareCycle` (`bot_core/resilience/hypercare.py`) oraz skrypt
   `run_stage6_resilience_cycle.py` spinają powyższe kroki w jednym przebiegu
   hypercare.
5. **Observability++** – eksport paczek obserwowalności, monitor SLO (wraz z
   kompozytowymi SLO2) oraz integracja statusów SLO z PortfolioGovernorem
   (`scripts/slo_monitor.py`, generujący raport JSON i CSV, `scripts/export_observability_bundle.py`,
   `scripts/verify_observability_bundle.py`, `scripts/log_portfolio_decision.py`).
   Pełen cykl hypercare można uruchomić jedną komendą dzięki
   `scripts/run_stage6_observability_cycle.py`, która łączy monitor SLO, override'y,
   anotacje i budowę paczki w jednym przebiegu z opcjonalną weryfikacją HMAC.
6. **Stage6 Hypercare Orchestrator** – zautomatyzowane spięcie cykli
   Observability, Resilience i Portfolio w pojedynczym raporcie zbiorczym.
   `bot_core.runtime.stage6_hypercare.Stage6HypercareCycle` generuje podpisany
   JSON z podsumowaniem komponentów, statusami ostrzeżeń/błędów i listą
   artefaktów. CLI `scripts/run_stage6_hypercare_cycle.py` pobiera definicję w
   YAML/JSON i uruchamia wszystkie moduły w jednej komendzie, wymuszając HMAC
   raportu końcowego, a `scripts/verify_stage6_hypercare_summary.py` pozwala
   zweryfikować podpis i integralność zbiorczego raportu. Raporty Stage5 i
  Stage6 można scalić w finalny przegląd hypercare dzięki
  `bot_core.runtime.full_hypercare.FullHypercareSummaryBuilder` oraz CLI
  `scripts/run_full_hypercare_summary.py`, a poprawność i podpis raportu
  potwierdza `scripts/verify_full_hypercare_summary.py` z opcjonalną ponowną
  weryfikacją Stage5/Stage6.

## Wymagania niefunkcjonalne
- Wszystkie artefakty operacyjne trafiają do `var/audit/...` i są podpisywane
  HMAC z użyciem kluczy rotowanych w hypercare.
- Runbooki Stage6 muszą zawierać sekcję „Artefakty/Akceptacja” z listą plików do
  archiwizacji i podpisu decyzji.
- Komunikacja między modułami pozostaje w obrębie bot_core (gRPC/mTLS lub IPC
  zgodnie z istniejącą architekturą, brak WebSocketów).

## Integracje
- Scheduler runtime ładuje konfiguracje PortfolioGovernora i udostępnia snapshoty
  Market Intel oraz raporty Stress Lab w trakcie cykli decyzyjnych.
- Scheduler multi-strategy uruchamia PortfolioGovernora przez koordynator runtime,
  korzystając z lokalnego cache Market Intel oraz metadanych sygnałów strategii.
- Sekcja `runtime.multi_strategy_schedulers.*.portfolio_inputs` wskazuje ścieżki do
  raportów SLO i Stress Lab, które runtime ładuje automatycznie, odrzucając
  przestarzałe artefakty.
- Monitor SLO emituje statusy wykorzystywane przez PortfolioGovernora i moduły
  alertów. Alerty Stage6 muszą wspierać override priorytetów.
- Bundler resilience zbiera manifesty z modułów runtime, SLO oraz Market Intel w
  jednym pakiecie, a failover drill waliduje gotowość do przełączenia.

## Dostarczone artefakty
- Moduły Python w `bot_core` dla market_intel, portfolio, observability, risk,
  resilience.
- CLI w `scripts/` odpalane w hypercare: market intel, SLO, bundling, audyt,
  failover drill, Stress Lab, kalibracja progów Stress Lab, cykl odporności
  (`run_stage6_resilience_cycle.py`), cykl portfelowy
  (`run_stage6_portfolio_cycle.py`) oraz pełny przebieg Stage6
  (`run_stage6_hypercare_cycle.py`).
- Dashboards i reguły alertów Stage6 w `deploy/grafana` oraz
  `deploy/prometheus`.
- Testy jednostkowe i integracyjne w `tests/` pokrywające kluczowe scenariusze
  Stage6.

## Otwarte działania
- Zbieranie feedbacku operatorów co do rekomendowanych progów oraz kalibracja
  profili wykonania (np. narzut kosztów transakcyjnych).
- Integracja cyklu Stage6 z harmonogramem tygodniowym hypercare (np. crontab) i
  automatycznym przekazywaniem podpisanych raportów do repozytorium audytowego.
