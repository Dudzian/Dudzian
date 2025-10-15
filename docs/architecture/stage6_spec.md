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
- **Strumień Market Intelligence & Stress Labs:**
  - Rozszerzenie danych o depth-of-book, wolumen w czasie rzeczywistym, wskaźniki funding/sentiment (manifesty Parquet/SQLite).
  - Symulator stresowy nowej generacji (`bot_core/risk/stress_lab.py`) obsługujący scenariusze multi-market oraz blackout
    infrastrukturalny.
  - Integracja z pipeline’em demo → paper → live (automatyczne generowanie raportów stresowych i gating release).
  - Narzędzie `scripts/build_market_intel_metrics.py` generujące podpisane baseline’y JSON na podstawie bazy SQLite.
  - Runbook operacyjny: `docs/runbooks/STAGE6_STRESS_LAB_CHECKLIST.md`.
- **Strumień Resilience & Failover:**
  - Mechanizmy self-healing runtime (wykrywanie błędów adapterów, automatyczne przełączanie na zapasowe giełdy, rotacja kluczy).
  - Udoskonalony `live_router` z obsługą sekwencji failover oraz audytami latencji w decision logu.
  - Narzędzia `scripts/failover_drill.py`, `scripts/export_resilience_bundle.py`, `scripts/verify_resilience_bundle.py` oraz checklisty DR (Disaster Recovery).
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
