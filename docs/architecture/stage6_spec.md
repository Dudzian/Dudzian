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
   `python scripts/run_stage6_portfolio_cycle.py`) generuje podpisane podsumowania
   JSON/CSV oraz wpisy decision logu na bazie raportów Market Intel, SLO i
   Stress Lab.
2. **Market Intelligence Aggregator** – konsolidacja metryk zmienności,
   płynności, momentum i spreadów z `data` i cache OHLCV. Udostępnia snapshoty
   dla PortfolioGovernora i raporty z CLI `python scripts/build_market_intel_metrics.py`.
3. **Stress Lab** – analiza raportów Paper Labs, generowanie adaptacyjnych
   override i rekomendacji dla PortfolioGovernora z uwzględnieniem
   scenariuszy płynności i latencji egzekucji. Raporty JSON/CSV podpisywane HMAC
   poprzez `python scripts/run_stress_lab.py`. Kalibracja progów odbywa się narzędziem
   `python scripts/calibrate_stress_lab_thresholds.py`, które łączy Market Intel i
   raporty symulacyjne; narzędzie potrafi automatycznie budować segmenty
   wolumenowe na bazie metryk płynności i przypisywać im prefiksy budżetów
   ryzyka.
4. **Resilience Stage6** – bundlowanie artefaktów, audyt, polityki, failover
   drill oraz raporty podpisywane HMAC (`python scripts/export_resilience_bundle.py`,
   `python scripts/verify_resilience_bundle.py`, `python scripts/audit_resilience_bundles.py`,
   `python scripts/failover_drill.py`) wraz z automatycznym self-healingiem restartującym
   moduły runtime (`bot_core/resilience/self_healing.py`). Komponent
   `ResilienceHypercareCycle` (`bot_core/resilience/hypercare.py`) oraz skrypt
   `python scripts/run_stage6_resilience_cycle.py` spinają powyższe kroki w jednym przebiegu
   hypercare.
5. **Observability++** – eksport paczek obserwowalności, monitor SLO (wraz z
   kompozytowymi SLO2) oraz integracja statusów SLO z PortfolioGovernorem
   (`python scripts/slo_monitor.py`, generujący raport JSON i CSV, `python scripts/export_observability_bundle.py`,
   `python scripts/verify_observability_bundle.py`, `python scripts/log_portfolio_decision.py`).
   Pełen cykl hypercare można uruchomić jedną komendą dzięki
   `python scripts/run_stage6_observability_cycle.py`, która łączy monitor SLO, override'y,
   anotacje i budowę paczki w jednym przebiegu z opcjonalną weryfikacją HMAC.
6. **Stage6 Hypercare Orchestrator** – zautomatyzowane spięcie cykli
   Observability, Resilience i Portfolio w pojedynczym raporcie zbiorczym.
   `bot_core.runtime.stage6_hypercare.Stage6HypercareCycle` generuje podpisany
   JSON z podsumowaniem komponentów, statusami ostrzeżeń/błędów i listą
   artefaktów. CLI `python scripts/run_stage6_hypercare_cycle.py` pobiera definicję w
   YAML/JSON i uruchamia wszystkie moduły w jednej komendzie, wymuszając HMAC
   raportu końcowego, a `python scripts/verify_stage6_hypercare_summary.py` pozwala
   zweryfikować podpis i integralność zbiorczego raportu. Raporty Stage5 i
  Stage6 można scalić w finalny przegląd hypercare dzięki
  `bot_core.runtime.full_hypercare.FullHypercareSummaryBuilder` oraz CLI
  `python scripts/run_full_hypercare_summary.py`, a poprawność i podpis raportu
  potwierdza `python scripts/verify_full_hypercare_summary.py` z opcjonalną ponowną
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
- CLI w `python scripts/` odpalane w hypercare: market intel, SLO, bundling, audyt,
  failover drill, Stress Lab, kalibracja progów Stress Lab, cykl odporności
  (`python scripts/run_stage6_resilience_cycle.py`), cykl portfelowy
  (`python scripts/run_stage6_portfolio_cycle.py`) oraz pełny przebieg Stage6
  (`python scripts/run_stage6_hypercare_cycle.py`).
- Dashboards i reguły alertów Stage6 w `deploy/grafana` oraz
  `deploy/prometheus`.
- Testy jednostkowe i integracyjne w `tests/` pokrywające kluczowe scenariusze
  Stage6.

## Otwarte działania
- Zbieranie feedbacku operatorów co do rekomendowanych progów oraz kalibracja
  profili wykonania (np. narzut kosztów transakcyjnych).
- Integracja cyklu Stage6 z harmonogramem tygodniowym hypercare (np. crontab) i
  automatycznym przekazywaniem podpisanych raportów do repozytorium audytowego.
# Specyfikacja Etapu 6 – Autonomiczna orkiestracja portfela i odporność operacyjna

## 1. Kontekst
Po Etapie 5 (TCO + rozszerzony decision engine) platforma OEM dysponuje kompletnym zestawem narzędzi kosztowo-decyzyjnych oraz
pełną obsługą compliance. Etap 6 ma przekształcić system w autonomiczny, odporny na awarie organizm, który potrafi dynamicznie
zarządzać kapitałem, reagować na zdarzenia rynkowe i utrzymać pełną ścieżkę audytową offline. Celem jest osiągnięcie poziomu
„self-driving” z kontrolą operatora, aby produkt konkurował z komercyjnymi botami klasy Cryptohopper Pro, zachowując architekturę
modułową (`bot_core`).

## 2. Zakres
- **Strumień Adaptive Portfolio Intelligence (API):**
  - Warstwa `PortfolioGovernor` balansująca kapitał między strategiami na podstawie scoringu TCO/SLO i sygnałów alfa.
  - Integracja z DecisionOrchestrator (konsolidacja metadanych decyzji, priorytety wykonania, automatyczne rollbacki).
  - Moduł auto-rebalancingu i korekty parametrów strategii (np. wagi, target volatility) z podpisanymi raportami zmian.
  - AutoTradeEngine dostarcza telemetrię blokad ryzyka (`auto_risk_freeze`, `auto_risk_freeze_extend`, `auto_risk_unfreeze`) z polami
    `reason`, `triggered_at`, `last_extension_at`, `released_at`, `frozen_for`, `risk_level`, `risk_score`, co umożliwia operatorom L1/L2 audyt automatycznych
    blokad i dokumentowanie decyzji hypercare. Powody obejmują `risk_score_threshold`, `risk_level_threshold`,
    `risk_level_and_score_threshold`, `risk_score_increase`, `risk_level_escalated`, `expiry_near`, `risk_recovered`,
    `risk_level_recovered`, `risk_score_recovered` oraz `expired`. Manualne alerty (`risk_freeze`, `risk_freeze_extend`,
    `risk_unfreeze`) raportują spójny zestaw pól (`reason`, `triggered_at`, `last_extension_at`, `released_at`, `frozen_for`,
    `source_reason`) i obsługują kody `risk_alert`, `risk_limit`, `risk_limit_escalated`, `manual_override` oraz `expired`,
    aby nadzór operacyjny mógł łatwo korelować blokady z decyzjami operatorów i alertami nadzorczymi. Wszystkie blokady są
    równolegle zapisywane w `TradingDecisionJournal` (status, tryb `auto`/`manual`, powód, metadane czasu), aby compliance
    miało pełną ścieżkę audytową niezależną od strumienia statusów runtime.
  - AutoTradeEngine przed publikacją sygnału odpytuje `DecisionModelInference`, aby uzyskać oczekiwany zwrot i prawdopodobieństwo sukcesu (`ai_inference`).
    Wynik skaluje aktywne wagi strategii oraz jest rejestrowany w `TradingDecisionJournal` razem ze zrzutem cech wejściowych i
    kontekstem `environment/portfolio`, co pozwala compliance prześledzić wpływ modeli AI na decyzje wykonawcze.
- **Strumień Market Intelligence & Stress Labs:**
  - Rozszerzenie danych o depth-of-book, wolumen w czasie rzeczywistym, wskaźniki funding/sentiment (manifesty Parquet/SQLite).
  - Symulator stresowy nowej generacji (`bot_core/risk/stress_lab.py`) obsługujący scenariusze multi-market oraz blackout
    infrastrukturalny.
  - Integracja z pipeline’em demo → paper → live (automatyczne generowanie raportów stresowych i gating release).
  - Narzędzie `python scripts/build_market_intel_metrics.py` generujące podpisane baseline’y JSON na podstawie bazy SQLite.
  - Runbook operacyjny: `docs/runbooks/STAGE6_STRESS_LAB_CHECKLIST.md`.
- **Strumień Resilience & Failover:**
  - Mechanizmy self-healing runtime (wykrywanie błędów adapterów, automatyczne przełączanie na zapasowe giełdy, rotacja kluczy).
  - Udoskonalony `live_router` z obsługą sekwencji failover oraz audytami latencji w decision logu.
  - Narzędzia `python scripts/failover_drill.py`, `python scripts/export_resilience_bundle.py`, `python scripts/verify_resilience_bundle.py` oraz checklisty DR (Disaster Recovery).
  - Automatyczny workflow `github_actions_stage6_resilience.yml` archiwizujący raporty failover i podpisy HMAC.
- **Strumień Observability++ & Reporting:**
  - Paczki obserwowalności Stage6 (dashboardy SLO2, alerty DR, raporty PDF/CSV dla audytu resilience) podpisane HMAC.
  - Automatyczne raporty miesięczne z działaniami PortfolioGovernora, failoverami i wynikami stres testów.
- **Enablement & Compliance:**
  - Playbooki operatorów Stage6 (L1/L2), warsztaty resilience, procedury override decyzji autonomicznych.
  - Rozszerzenie checklists demo → paper → live o progi API, failover i stres testy Stage6.

## 3. Zależności
- Dane: konieczne rozszerzenia pipeline’u danych (depth-of-book, funding, sentiment) oraz walidatory jakości.
- Runtime: integracja PortfolioGovernora z schedulerem multi-strategy, DecisionOrchestratora oraz silnikiem ryzyka.
- Bezpieczeństwo: dodatkowe podpisy i rotacje kluczy dla nowych paczek (resilience bundle, raporty rebalancingu).
- Observability: Prometheus/Grafana muszą obsłużyć nowe metryki SLO2 i alerty DR.
- Operacje: operatorzy wymagają nowych narzędzi override i playbooków awaryjnych.

## 4. Definition of Done
- **Kod:** moduły PortfolioGovernor, Stress Labs, rozszerzenia live_routera oraz narzędzia resilience posiadają testy jednostkowe,
  integracyjne i testy obciążeniowe; pokrycie ≥85% i integracja w CI.
- **Konfiguracja:** `config/core.yaml` zawiera sekcje `portfolio_governor`, `stress_lab`, `resilience`, `observability.stage6` z
  parametrami podpisanymi i walidowanymi bundlerem OEM.
- **Dokumentacja:** spec, discovery, runbooki Stage6 (`STAGE6_SUPPORT_PLAYBOOK.md`, `STAGE6_DRILL_PLAYBOOK.md`, aktualizacje
  checklist demo → paper → live) oraz raport architektoniczny w `docs/architecture`.
- **Testy:** workflow CI obejmuje smoke test PortfolioGovernora, stres test Stage6, failover drill, walidację alertów SLO2 oraz
  bundling resilience. Wyniki archiwizowane i podpisane HMAC.
- **Operacje & compliance:** decision log uwzględnia wpisy auto-rebalancing, failover i overrides operatora; warsztaty Stage6
  zarejestrowane i podpisane.

## 5. Kamienie milowe
1. **Discovery i plan danych:** zakończenie dokumentu discovery (artefakty w `var/audit/stage6_discovery/`).
2. **PortfolioGovernor + auto-rebalancing:** implementacja modułu, integracja z DecisionOrchestrator i schedulerem.
3. **Stress Labs + dane rozszerzone:** symulator, pipeline danych i raporty gatingowe.
4. **Resilience & failover:** narzędzia failover, rozszerzenie live_routera, bundling resilience, checklisty DR.
5. **Observability++ & Enablement:** dashboardy Stage6, alerty, raporty miesięczne, playbooki operatorów.
6. **Benchmark Stage6 vs CryptoHopper:** utrzymywanie porównania funkcji i luk (automatyzacja, marketplace, UI, compliance) w `docs/benchmark/cryptohopper_comparison.md`, z naciskiem na rozszerzenie pokrycia giełdowego, publikację presetów oraz finalizację integracji UI ↔ runtime.
   - Strategia: parzystość funkcjonalna – pilnować roadmapy presetów i komunikacji Stress Labs; utrzymywać status 🟡 do czasu publikacji katalogu.
   - Automatyzacja: przewaga Stage6 – utrzymać autonomiczny hypercare z podpisami HMAC (status 🟢 monitorowany miesięcznie w tablicy wyników).
   - UI: luka – domknąć feed gRPC „Decyzje AI” i telemetrykę runtime (status 🔴 wymaga dem komponents UI/Runtime w Q3 2024).
   - Compliance: przewaga – egzekwować offline-first journale i bundling artefaktów (status 🟢 potwierdzany audytem kwartalnym).
   - **Cadence:** aktualizacje tablicy wyników i harmonogramu działań korygujących są elementem bramki release (checklista wsparcia + `docs/runtime/status_review.md`).
   - **Kronika benchmarku:** sekcja „Historia aktualizacji benchmarku” musi zawierać wpis po każdej rewizji release’owej lub audycie, z linkiem do artefaktów i odnotowanym wpływem na roadmapę.
   - **Procedura metryk:** proces zbierania metryk (hypercare, adaptery giełdowe, marketplace, UI, compliance) jest wykonywany według kroków opisanych w benchmarku; odstępstwa logujemy w `docs/runtime/status_review.md`.

## 6. Ryzyka i mitigacje
- **Niedostępność danych L2/sentiment:** fallback do lokalnych kolejek i emulacji, walidacja jakości w pipeline’ach.
- **Złożoność autonomicznego zarządzania kapitałem:** jasne progi override, podpisane decyzje operatora, sandbox testów.
- **Ryzyko failoveru:** symulacje DR w trybie paper, wymuszone checklisty i podpisane raporty drill.
- **Obciążenie operacyjne:** warsztaty Stage6, automatyczne raporty i alerty, playbooki awaryjne.

## 7. Harmonogram wysokiego poziomu (proponowany)
| Tydzień | Aktywności |
| --- | --- |
| 1 | Finalizacja discovery, zatwierdzenie spec, przygotowanie manifestów danych. |
| 2 | Implementacja PortfolioGovernora, integracja z DecisionOrchestrator i schedulerem, testy jednostkowe. |
| 3 | Budowa Stress Labs, rozszerzenie pipeline’u danych, integracja z gatingiem demo → paper → live. |
| 4 | Implementacja narzędzi failover i resilience, bundling paczek, testy DR. |
| 5 | Observability++ (dashboardy, alerty), raporty miesięczne, enablement operacyjny i finalny audit Stage6. |
